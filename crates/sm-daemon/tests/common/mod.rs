//! The harness for the daemon's integration tests (SPEC 14, stream E,
//! Tests).
//!
//! A daemon is started in the test process on a temporary profile and
//! reached over the real endpoint: a named pipe on Windows, a Unix socket
//! elsewhere. It uses the fake embedder and the whitespace chunker of
//! `sm_core::testing`.
//!
//! No test reads the real profile folder or the user's environment. The
//! endpoint is passed in `DaemonOptions`, so `XDG_RUNTIME_DIR` is never
//! read. Each project folder holds its own `.git` folder, so project
//! resolution stops there and never at whatever encloses the temporary
//! folder.

#![allow(dead_code)]

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use serde_json::Value;
use sm_core::config::{ProfileName, Settings};
use sm_core::error::{CoreError, ErrorCode, Result};
use sm_core::paths::{Endpoint, ProfilePaths};
use sm_core::protocol::{
    DaemonInfo, DaemonState, ExportParams, ExportReport, ImportParams, ImportReport, MigrateParams,
    Op, Reply, Request, Response, ShareParams, ShareReply, StatusParams, StatusReply, SyncParams,
    SyncReply, read_message, write_frame,
};
use sm_core::testing::{FakeEmbedder, FakeProvider, SimpleChunker, fake_info};
use sm_core::traits::{DocInput, Embedder, EmbedderInfo, EmbedderProvider, SessionKind};
use sm_core::types::{Memory, NewMemory, Provenance, ProvenanceSource};
use sm_daemon::{
    DaemonHandle, DaemonOptions, Exit, LoadedModel, ModelLoader, SharingContext, SharingOps,
    Started, UnwiredSharing,
};
use sm_store::{Access, MemoryInput, MemoryListFilter, NewChunk, Store};

static COUNTER: AtomicUsize = AtomicUsize::new(0);

/// A name no other test in any test process uses.
pub fn unique(prefix: &str) -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or_default();
    format!(
        "{prefix}-{}-{}-{nanos}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::SeqCst)
    )
}

/// Polls `cond` every 10 ms for up to `timeout`.
pub fn wait_until(timeout: Duration, mut cond: impl FnMut() -> bool) -> bool {
    let start = Instant::now();
    loop {
        if cond() {
            return true;
        }
        if start.elapsed() >= timeout {
            return false;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// Reads a fixture under the repository's `fixtures/`.
pub fn fixture_path(rel: &str) -> PathBuf {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures")).join(rel)
}

pub fn fixture(rel: &str) -> Value {
    let path = fixture_path(rel);
    let text =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

/// Daemon options for tests: the given endpoint and short waits.
pub fn test_options(endpoint: Endpoint) -> DaemonOptions {
    DaemonOptions {
        endpoint: Some(endpoint),
        lock_wait: Duration::from_millis(300),
        bind_wait: Duration::from_millis(300),
        retry_every: Duration::from_millis(10),
        bulk_grace: Duration::from_millis(200),
        bulk_error_backoff: Duration::from_millis(50),
        physical_cores: Some(8),
        shutdown_grace: Duration::from_secs(5),
        ..DaemonOptions::default()
    }
}

/// One temporary profile and its endpoint.
pub struct Rig {
    pub tmp: tempfile::TempDir,
    pub paths: ProfilePaths,
    pub endpoint: Endpoint,
    pub settings: Settings,
    pub opts: DaemonOptions,
}

impl Rig {
    pub fn new() -> Rig {
        Rig::with_profile("test")
    }

    pub fn with_profile(profile: &str) -> Rig {
        let tmp = tempfile::Builder::new()
            .prefix("smd")
            .tempdir()
            .expect("a temporary folder");
        let paths = ProfilePaths::new(
            tmp.path().join("root"),
            ProfileName::parse(profile).expect("a profile name"),
        );
        let endpoint = new_endpoint(tmp.path());
        let opts = test_options(endpoint.clone());
        Rig {
            tmp,
            paths,
            endpoint,
            settings: Settings::default(),
            opts,
        }
    }

    pub fn profile(&self) -> ProfileName {
        self.paths.profile.clone()
    }

    /// A project folder: a repository without commits or remote.
    pub fn project(&self, name: &str) -> String {
        let dir = self.tmp.path().join("work").join(name);
        std::fs::create_dir_all(dir.join(".git")).expect("create a project");
        std::fs::write(dir.join(".git").join("HEAD"), "ref: refs/heads/main\n")
            .expect("write HEAD");
        dir_string(&dir)
    }

    /// A folder outside any repository (nothing above the temporary folder
    /// is one on the test machines).
    pub fn plain_dir(&self, name: &str) -> String {
        let dir = self.tmp.path().join("plain").join(name);
        std::fs::create_dir_all(&dir).expect("create a folder");
        dir_string(&dir)
    }

    pub fn start(&self, loader: Arc<dyn ModelLoader>, sharing: Arc<dyn SharingOps>) -> Daemon {
        match self.try_start(loader, sharing).expect("the daemon starts") {
            Started::Running(h) => Daemon::new(h, self.profile()),
            Started::LockHeld => panic!("the lock was held"),
        }
    }

    pub fn try_start(
        &self,
        loader: Arc<dyn ModelLoader>,
        sharing: Arc<dyn SharingOps>,
    ) -> Result<Started> {
        sm_daemon::start(
            self.paths.clone(),
            self.settings.clone(),
            loader,
            sharing,
            self.opts.clone(),
        )
    }

    /// A daemon with the fake model and no sharing, ready.
    pub fn start_fake(&self) -> Daemon {
        let d = self.start(FakeLoader::new(), Arc::new(UnwiredSharing));
        d.wait_ready();
        d
    }

    /// A second handle on the profile's database (SQLite allows readers
    /// beside the daemon).
    pub fn store(&self) -> Store {
        Store::open(&self.paths).expect("open the store")
    }

    pub fn memories(&self) -> Vec<Memory> {
        self.store()
            .list_memories(&MemoryListFilter::default())
            .expect("list memories")
    }
}

impl Default for Rig {
    fn default() -> Self {
        Rig::new()
    }
}

fn dir_string(dir: &Path) -> String {
    dir.to_str().expect("a UTF-8 temporary path").to_owned()
}

#[cfg(windows)]
pub fn new_endpoint(_tmp: &Path) -> Endpoint {
    Endpoint::NamedPipe(unique("sm-test"))
}

#[cfg(not(windows))]
pub fn new_endpoint(tmp: &Path) -> Endpoint {
    Endpoint::UnixSocket(tmp.join("d.sock"))
}

/// A running daemon. Dropping it shuts it down and waits for its threads.
pub struct Daemon {
    handle: Option<DaemonHandle>,
    pub endpoint: Endpoint,
    pub profile: ProfileName,
}

impl Daemon {
    pub fn new(handle: DaemonHandle, profile: ProfileName) -> Daemon {
        Daemon {
            endpoint: handle.endpoint().clone(),
            handle: Some(handle),
            profile,
        }
    }

    pub fn client(&self) -> Client {
        Client::connect(&self.endpoint)
    }

    pub fn req(&self, cwd: &str, op: Op) -> Request {
        Request::new(cwd, self.profile.clone(), op)
    }

    pub fn send(&self, req: &Request) -> Response {
        self.client().call(req)
    }

    pub fn call(&self, cwd: &str, op: Op) -> Response {
        self.send(&self.req(cwd, op))
    }

    /// The reply of a request that must succeed.
    pub fn ok(&self, cwd: &str, op: Op) -> Reply {
        match self.call(cwd, op).into_result() {
            Ok(r) => r,
            Err(e) => panic!("{}: {}", e.code.as_str(), e.message),
        }
    }

    /// The error of a request that must fail.
    pub fn err(&self, cwd: &str, op: Op) -> CoreError {
        match self.call(cwd, op).into_result() {
            Ok(r) => panic!("expected an error, got {r:?}"),
            Err(e) => e,
        }
    }

    pub fn hello(&self) -> DaemonInfo {
        match self.ok("", Op::Hello) {
            Reply::Hello(i) => i,
            other => panic!("hello replied {other:?}"),
        }
    }

    pub fn wait_ready(&self) {
        assert!(
            wait_until(Duration::from_secs(10), || {
                self.hello().state == DaemonState::Ready
            }),
            "the daemon became ready"
        );
    }

    pub fn status(&self, cwd: &str) -> StatusReply {
        match self.ok(cwd, Op::Status(StatusParams::default())) {
            Reply::Status(s) => *s,
            other => panic!("status replied {other:?}"),
        }
    }

    /// Waits until no chunk waits for its embedding.
    pub fn wait_embedded(&self, cwd: &str) {
        assert!(
            wait_until(Duration::from_secs(20), || {
                self.status(cwd).index.pending_embeddings == 0
            }),
            "every chunk was embedded"
        );
    }

    pub fn handle(&self) -> &DaemonHandle {
        self.handle.as_ref().expect("a running daemon")
    }

    pub fn log_path(&self) -> PathBuf {
        self.handle().log_path().to_path_buf()
    }

    pub fn take(&mut self) -> DaemonHandle {
        self.handle.take().expect("a running daemon")
    }

    pub fn shutdown(mut self) -> Exit {
        self.take().shutdown("test finished")
    }
}

impl Drop for Daemon {
    fn drop(&mut self) {
        if let Some(h) = self.handle.take() {
            h.shutdown("test finished");
        }
    }
}

/// One connection to a daemon.
pub struct Client {
    pub stream: interprocess::local_socket::Stream,
}

impl Client {
    pub fn connect(endpoint: &Endpoint) -> Client {
        Client {
            stream: sm_daemon::connect(endpoint, Duration::from_secs(5)).expect("connect"),
        }
    }

    pub fn call(&mut self, req: &Request) -> Response {
        write_frame(&mut self.stream, req).expect("write a request");
        self.read().expect("a reply")
    }

    pub fn read(&mut self) -> Option<Response> {
        read_message(&mut self.stream).ok().flatten()
    }

    /// Writes `body` as one frame, whatever it holds.
    pub fn raw(&mut self, body: &[u8]) -> Option<Response> {
        let len = u32::try_from(body.len()).expect("a small body");
        self.stream
            .write_all(&len.to_le_bytes())
            .expect("write a prefix");
        self.stream.write_all(body).expect("write a body");
        self.stream.flush().expect("flush");
        self.read()
    }

    pub fn raw_value(&mut self, v: &Value) -> Response {
        self.raw(&serde_json::to_vec(v).expect("encode"))
            .expect("a reply")
    }

    /// Reads until the daemon closes the connection.
    pub fn closed(&mut self) -> bool {
        let mut buf = [0u8; 16];
        matches!(self.stream.read(&mut buf), Ok(0) | Err(_))
    }
}

/// The error of a response that must fail.
pub fn err_of(r: Response) -> CoreError {
    match r.into_result() {
        Ok(r) => panic!("expected an error, got {r:?}"),
        Err(e) => e,
    }
}

pub fn assert_code(e: &CoreError, code: ErrorCode) {
    assert_eq!(e.code, code, "{}: {}", e.code.as_str(), e.message);
}

// --- loaders -------------------------------------------------------------

/// Loads the fake model at once.
pub struct FakeLoader {
    pub provider: Arc<dyn EmbedderProvider>,
}

impl FakeLoader {
    pub fn new() -> Arc<FakeLoader> {
        Arc::new(FakeLoader {
            provider: Arc::new(FakeProvider::new()),
        })
    }

    pub fn with(provider: Arc<dyn EmbedderProvider>) -> Arc<FakeLoader> {
        Arc::new(FakeLoader { provider })
    }
}

impl ModelLoader for FakeLoader {
    fn info(&self) -> EmbedderInfo {
        fake_info()
    }
    fn load(&self) -> Result<LoadedModel> {
        Ok(LoadedModel {
            provider: Arc::clone(&self.provider),
            chunker: Arc::new(SimpleChunker),
        })
    }
}

/// Loads the fake model once [`GateLoader::open`] is called, so a test sees
/// the daemon in the `loading` state for as long as it needs.
pub struct GateLoader {
    gate: Mutex<bool>,
    cv: Condvar,
}

impl GateLoader {
    pub fn new() -> Arc<GateLoader> {
        Arc::new(GateLoader {
            gate: Mutex::new(false),
            cv: Condvar::new(),
        })
    }

    pub fn open(&self) {
        *self.gate.lock().expect("the gate") = true;
        self.cv.notify_all();
    }
}

impl ModelLoader for GateLoader {
    fn info(&self) -> EmbedderInfo {
        fake_info()
    }
    fn load(&self) -> Result<LoadedModel> {
        let mut open = self.gate.lock().expect("the gate");
        while !*open {
            open = self.cv.wait(open).expect("the gate");
        }
        Ok(LoadedModel {
            provider: Arc::new(FakeProvider::new()),
            chunker: Arc::new(SimpleChunker),
        })
    }
}

/// A loader whose `load` runs a closure (an injected error, or a real file
/// check before the fake model).
pub struct FnLoader<F>(pub F);

impl<F> ModelLoader for FnLoader<F>
where
    F: Fn() -> Result<LoadedModel> + Send + Sync,
{
    fn info(&self) -> EmbedderInfo {
        fake_info()
    }
    fn load(&self) -> Result<LoadedModel> {
        (self.0)()
    }
}

pub fn fake_model() -> LoadedModel {
    LoadedModel {
        provider: Arc::new(FakeProvider::new()),
        chunker: Arc::new(SimpleChunker),
    }
}

// --- a recording provider --------------------------------------------------

/// What a [`RecordingProvider`] saw.
#[derive(Debug, Default)]
pub struct Recorded {
    /// Every session opened: its kind and thread count.
    pub opened: Vec<(SessionKind, usize)>,
    /// Bulk threads alive at once, at most.
    pub max_live_bulk_threads: usize,
    live_bulk_threads: usize,
    pub documents: usize,
    pub queries: usize,
}

/// The fake provider, recording the sessions it opens and how many bulk
/// threads are alive at once; each document batch can be slowed down so a
/// bulk import lasts long enough to measure the hook during it.
pub struct RecordingProvider {
    info: EmbedderInfo,
    pub rec: Arc<Mutex<Recorded>>,
    pub per_document: Duration,
}

impl RecordingProvider {
    pub fn new(per_document: Duration) -> Arc<RecordingProvider> {
        Arc::new(RecordingProvider {
            info: fake_info(),
            rec: Arc::new(Mutex::new(Recorded::default())),
            per_document,
        })
    }

    pub fn recorded(&self) -> std::sync::MutexGuard<'_, Recorded> {
        self.rec.lock().expect("the record")
    }
}

impl EmbedderProvider for RecordingProvider {
    fn info(&self) -> &EmbedderInfo {
        &self.info
    }

    fn open(&self, kind: SessionKind, threads: usize) -> Result<Box<dyn Embedder>> {
        let mut r = self.rec.lock().expect("the record");
        r.opened.push((kind, threads));
        if kind == SessionKind::Bulk {
            r.live_bulk_threads += threads;
            r.max_live_bulk_threads = r.max_live_bulk_threads.max(r.live_bulk_threads);
        }
        Ok(Box::new(RecordingEmbedder {
            inner: FakeEmbedder::with_session(kind, threads),
            kind,
            threads,
            rec: Arc::clone(&self.rec),
            per_document: self.per_document,
        }))
    }
}

struct RecordingEmbedder {
    inner: FakeEmbedder,
    kind: SessionKind,
    threads: usize,
    rec: Arc<Mutex<Recorded>>,
    per_document: Duration,
}

impl Drop for RecordingEmbedder {
    fn drop(&mut self) {
        if self.kind == SessionKind::Bulk {
            let mut r = self.rec.lock().expect("the record");
            r.live_bulk_threads -= self.threads;
        }
    }
}

impl Embedder for RecordingEmbedder {
    fn info(&self) -> &EmbedderInfo {
        self.inner.info()
    }

    fn embed_query(&self, text: &str, max_tokens: usize) -> Result<Vec<f32>> {
        self.rec.lock().expect("the record").queries += 1;
        self.inner.embed_query(text, max_tokens)
    }

    fn embed_documents(&self, docs: &[DocInput<'_>]) -> Result<Vec<Vec<f32>>> {
        if !self.per_document.is_zero() {
            std::thread::sleep(self.per_document * docs.len() as u32);
        }
        self.rec.lock().expect("the record").documents += docs.len();
        self.inner.embed_documents(docs)
    }
}

// --- a recording sharing implementation ------------------------------------

/// One call a [`RecordingSharing`] received.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharingCall {
    pub op: &'static str,
    /// The resolved project's name.
    pub project: Option<String>,
    /// The resolved project's repository root, when it has one.
    pub repo_root: Option<PathBuf>,
    pub had_chunker: bool,
}

/// Records each call with the project the daemon resolved. `import` and
/// `migrate` store one memory with pending chunks through `ctx.store`, as
/// the real implementation does, so a test sees the queue and the maps
/// react.
#[derive(Default)]
pub struct RecordingSharing {
    pub calls: Mutex<Vec<SharingCall>>,
}

impl RecordingSharing {
    pub fn new() -> Arc<RecordingSharing> {
        Arc::new(RecordingSharing::default())
    }

    pub fn calls(&self) -> Vec<SharingCall> {
        self.calls.lock().expect("the calls").clone()
    }

    fn record(&self, op: &'static str, ctx: &SharingContext<'_>) {
        self.calls.lock().expect("the calls").push(SharingCall {
            op,
            project: ctx.project.map(|p| p.project.name.clone()),
            repo_root: ctx
                .project
                .and_then(|p| p.identity.repo.as_ref().map(|r| r.root.clone())),
            had_chunker: ctx.chunker.is_some(),
        });
    }

    fn store_one(ctx: &SharingContext<'_>, source: &str) -> Result<ImportReport> {
        let project = ctx
            .project
            .ok_or_else(|| CoreError::invalid_input("no project to import into"))?;
        let chunker = ctx
            .chunker
            .ok_or_else(|| CoreError::new(ErrorCode::DaemonLoading, "no chunker yet"))?;
        let content = format!("imported from {source}: the release checklist lives in docs");
        let plan = chunker.chunk(&content, &ctx.chunk_params)?;
        let memory = NewMemory::new(
            project.project.id,
            content,
            Provenance::new(ProvenanceSource::Import),
        );
        let chunks = plan.pieces.into_iter().map(NewChunk::pending).collect();
        ctx.store
            .insert_memory(MemoryInput { memory, chunks }, Access::Local)?;
        Ok(ImportReport {
            source: source.to_owned(),
            read: 1,
            inserted: 1,
            skipped_existing: 0,
            redacted: 0,
            projects_created: Vec::new(),
            issues: Vec::new(),
            pending_embeddings: 1,
        })
    }
}

impl SharingOps for RecordingSharing {
    fn share(&self, ctx: &SharingContext<'_>, _: &ShareParams) -> Result<ShareReply> {
        self.record("share", ctx);
        Ok(ShareReply {
            written: Vec::new(),
            refused: Vec::new(),
        })
    }

    fn sync(&self, ctx: &SharingContext<'_>, p: &SyncParams) -> Result<SyncReply> {
        self.record("sync", ctx);
        Ok(SyncReply {
            dry_run: p.dry_run,
            added: Vec::new(),
            updated: Vec::new(),
            removed: Vec::new(),
            unchanged: 0,
            errors: Vec::new(),
        })
    }

    fn migrate(&self, ctx: &SharingContext<'_>, p: &MigrateParams) -> Result<ImportReport> {
        self.record("migrate", ctx);
        RecordingSharing::store_one(ctx, &p.from)
    }

    fn import(&self, ctx: &SharingContext<'_>, p: &ImportParams) -> Result<ImportReport> {
        self.record("import", ctx);
        RecordingSharing::store_one(ctx, &p.path)
    }

    fn export(&self, ctx: &SharingContext<'_>, p: &ExportParams) -> Result<ExportReport> {
        self.record("export", ctx);
        Ok(ExportReport {
            path: p.path.clone(),
            projects: 1,
            memories: ctx.store.memory_count()?,
            links: 0,
        })
    }
}

// --- request helpers -------------------------------------------------------

/// Stores one memory and returns its id.
pub fn remember(d: &Daemon, cwd: &str, content: &str) -> sm_core::ids::MemoryId {
    remember_item(d, cwd, sm_core::protocol::RememberItem::new(content))
}

pub fn remember_item(
    d: &Daemon,
    cwd: &str,
    item: sm_core::protocol::RememberItem,
) -> sm_core::ids::MemoryId {
    match d.ok(
        cwd,
        Op::Remember(sm_core::protocol::RememberParams {
            memories: vec![item],
        }),
    ) {
        Reply::Remember(r) => r.memories[0].id,
        other => panic!("remember replied {other:?}"),
    }
}

pub fn recall(
    d: &Daemon,
    cwd: &str,
    p: sm_core::protocol::RecallParams,
) -> sm_core::protocol::RecallReply {
    match d.ok(cwd, Op::Recall(p)) {
        Reply::Recall(r) => r,
        other => panic!("recall replied {other:?}"),
    }
}

/// The value of a request with one extra key in `path` (`call.args`,
/// `call`, or the request itself when `path` is empty).
pub fn with_unknown_key(req: &Request, path: &[&str]) -> Value {
    let mut v = serde_json::to_value(req).expect("encode a request");
    let mut target = &mut v;
    for key in path {
        target = target
            .get_mut(*key)
            .unwrap_or_else(|| panic!("the request has no {key}"));
    }
    target
        .as_object_mut()
        .expect("an object")
        .insert("surprise".to_owned(), Value::from(1));
    v
}

// --- stream H helpers ----------------------------------------------------

impl Rig {
    /// A daemon with the given embedder provider, the whitespace chunker
    /// and no sharing, ready.
    pub fn start_with_provider(&self, provider: Arc<dyn EmbedderProvider>) -> Daemon {
        let d = self.start(FakeLoader::with(provider), Arc::new(UnwiredSharing));
        d.wait_ready();
        d
    }
}

/// The fake provider, except that every document batch fails (`internal`)
/// while [`DownProvider::down`] is set: an embedder-wide failure, not one
/// of any chunk.
pub struct DownProvider {
    inner: FakeProvider,
    pub down: Arc<std::sync::atomic::AtomicBool>,
}

impl DownProvider {
    pub fn new(down: bool) -> Arc<DownProvider> {
        Arc::new(DownProvider {
            inner: FakeProvider::new(),
            down: Arc::new(std::sync::atomic::AtomicBool::new(down)),
        })
    }
}

struct DownEmbedder {
    inner: Box<dyn Embedder>,
    down: Arc<std::sync::atomic::AtomicBool>,
}

impl EmbedderProvider for DownProvider {
    fn info(&self) -> &EmbedderInfo {
        self.inner.info()
    }
    fn open(&self, kind: SessionKind, threads: usize) -> Result<Box<dyn Embedder>> {
        Ok(Box::new(DownEmbedder {
            inner: self.inner.open(kind, threads)?,
            down: Arc::clone(&self.down),
        }))
    }
}

impl Embedder for DownEmbedder {
    fn info(&self) -> &EmbedderInfo {
        self.inner.info()
    }
    fn embed_query(&self, text: &str, max_tokens: usize) -> Result<Vec<f32>> {
        self.inner.embed_query(text, max_tokens)
    }
    fn embed_documents(&self, docs: &[DocInput<'_>]) -> Result<Vec<Vec<f32>>> {
        if self.down.load(Ordering::SeqCst) {
            return Err(CoreError::internal("injected: the embedder is down"));
        }
        self.inner.embed_documents(docs)
    }
}

impl Rig {
    /// A daemon with the given model loader and no sharing, not waited on:
    /// the test watches it load, or fail.
    pub fn start_loader(&self, loader: Arc<dyn ModelLoader>) -> Daemon {
        self.start(loader, Arc::new(UnwiredSharing))
    }
}

/// A loader that always fails with `internal`: the daemon stays up in the
/// `failed` state.
pub fn failing_loader() -> Arc<dyn ModelLoader> {
    Arc::new(FnLoader(|| -> Result<LoadedModel> {
        Err(CoreError::internal("injected: the model cannot load"))
    }))
}
