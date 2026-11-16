//! `sync`: applies the committed shared folder at `HEAD` to the store (SPEC
//! section 9, Team sharing rules 3 to 5, "Change detection before reading"
//! and "Re-embed only when the text changes"; decision B-78).
//!
//! # What is read
//!
//! Only what is committed. sync resolves `HEAD` to a commit, lists that
//! commit's tree under `.spatial-memory/shared/` with `git ls-tree`, and
//! reads blobs through one `git cat-file --batch` process (see the `git`
//! module). Edits in the working tree that are not committed are never
//! seen. A repository with no commits has nothing to sync: the reply is
//! empty and nothing is retracted.
//!
//! # Where each file lands
//!
//! Every listed entry lands in exactly one list of the reply, except the
//! folder's own `.gitattributes`, which share writes and which is not a
//! memory (it is skipped without a report):
//!
//! - `errors`: an entry that is not a regular `.md` blob, a file over
//!   [`MAX_SHARED_FILE_BYTES`], a file that is not UTF-8 or does not parse,
//!   a file with no body, an id in two files (both are reported and neither
//!   is applied), an id that belongs to another project, a local copy that
//!   differs from the file, text already stored under another id, a changed
//!   `supersedes`, and a file the store refused.
//! - `unchanged`: a file whose path and blob id equal the mark sync recorded
//!   on a shared memory of this project; it is not read again. A file that
//!   was read and needed no change to its memory is also counted here.
//! - `added`: a new id, inserted as a shared memory with pending chunks.
//! - `updated`: a shared memory revised from its file, or the author's own
//!   local copy adopted as shared.
//!
//! `removed` lists memories, not files: shared memories of this project
//! whose file is gone at `HEAD`. They are retracted, never deleted. A
//! memory whose recorded file is still listed but could not be read (it
//! does not parse, is too large, …) is not retracted: its id is unknown,
//! and the reported error is the thing to fix.
//!
//! # Order of work
//!
//! New files are inserted first, each superseded memory before the memory
//! that supersedes it, in batches of 128 in one transaction each. Known
//! files are revised or adopted next, then missing files are retracted.
//! Every id list in the reply is sorted, and the errors are sorted by path.
//!
//! # Supersedes
//!
//! A new file's `supersedes` is stored as `supersedes_id`, which marks the
//! target superseded, when the target is in this project and not retracted
//! (or was added earlier in the same sync). Otherwise the target is missing:
//! the file is still added, its `supersedes_id` is left empty and the
//! target's id is kept in provenance `extra.unresolved_supersedes`. When
//! that target arrives later it is stored as superseded.
//!
//! A file whose status is `active` is stored as `superseded` when a memory
//! of this project supersedes it, so a target's own unchanged file never
//! undoes the supersession. Retracting the superseding memory does not undo
//! it either.
//!
//! A stored memory's `supersedes` is fixed (the store has no way to change
//! it). A file that changes it is reported; the rest of the file is applied
//! and no mark is recorded, so the report repeats until the file is fixed.
//!
//! # Dry run
//!
//! `dry_run` computes the same reply and writes nothing. It chunks text as
//! the real run does, so a chunking failure is reported the same way. Only
//! a failure of the store itself on write cannot be foreseen.

use std::collections::{HashMap, HashSet};
use std::ffi::OsString;
use std::path::Path;

use serde_json::json;
use sm_core::hash::content_hash;
use sm_core::ids::MemoryId;
use sm_core::protocol::{FileIssue, RememberOutcome, SyncParams, SyncReply};
use sm_core::traits::{ChunkParams, Chunker};
use sm_core::types::{
    DEFAULT_IMPORTANCE, Memory, NewMemory, Origin, Project, Provenance, ProvenanceSource, Status,
};
use sm_core::{CoreError, Result};
use sm_store::{
    Access, ContentChange, Inserted, MemoryInput, MemoryListFilter, NewChunk, Revision, Store,
    SyncMark,
};

use crate::format::{SHARED_DIR, SharedDoc, memory_text_hash};
use crate::git::{Git, TreeEntry};
use crate::is_item_error;
use crate::share::require_git_entry;

/// The largest shared file sync reads: 1 MiB. A memory is a decision or a
/// pitfall, not a document; a larger file is reported and skipped, which
/// bounds what one sync holds in memory.
pub const MAX_SHARED_FILE_BYTES: u64 = 1024 * 1024;

/// New memories inserted per transaction.
const INSERT_BATCH: usize = 128;

/// The folder's own git attributes, written by share.
const GITATTRIBUTES_PATH: &str = ".spatial-memory/shared/.gitattributes";

/// Provenance `extra` key for a supersedes target that was not in the store
/// when the file was added.
pub(crate) const UNRESOLVED_SUPERSEDES: &str = "unresolved_supersedes";

/// Provenance `extra` key of the sync mark, in the layout
/// [`SyncMark::of`] reads: `{"blob": …, "text_hash": …}` (decision B-78).
const SYNC_MARK: &str = "sync";

/// Test hooks. Production passes the default.
#[derive(Debug, Clone, Default)]
pub(crate) struct SyncOptions {
    /// Extra environment for every git process.
    pub git_env: Vec<(OsString, OsString)>,
}

/// What a sync did that the reply does not show.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct SyncStats {
    /// Blobs read through `git cat-file --batch`.
    pub blobs_read: u64,
}

/// Applies the shared folder committed at `HEAD` of the repository at
/// `repo_root` to `project`; see the module documentation.
pub fn sync(
    store: &Store,
    chunker: &dyn Chunker,
    chunk_params: &ChunkParams,
    project: &Project,
    repo_root: &Path,
    params: &SyncParams,
) -> Result<SyncReply> {
    let opts = SyncOptions::default();
    sync_with(
        store,
        chunker,
        chunk_params,
        project,
        repo_root,
        params,
        &opts,
    )
    .map(|(r, _)| r)
}

/// [`sync`] with test hooks.
pub(crate) fn sync_with(
    store: &Store,
    chunker: &dyn Chunker,
    chunk_params: &ChunkParams,
    project: &Project,
    repo_root: &Path,
    params: &SyncParams,
    opts: &SyncOptions,
) -> Result<(SyncReply, SyncStats)> {
    require_git_entry(repo_root)?;
    let git = Git::new(repo_root, opts.git_env.clone());
    let Some(head) = git.head()? else {
        return Ok((empty_reply(params.dry_run), SyncStats::default()));
    };
    let entries = git.ls_tree(&head, &format!("{SHARED_DIR}/"))?;

    let mut run = Run::new(store, chunker, chunk_params, project, params.dry_run, head)?;

    // Sort the listing into files to skip, files unchanged since the last
    // sync, and files to read.
    let mut unread_paths: HashSet<String> = HashSet::new();
    let mut unchanged: Vec<(String, MemoryId)> = Vec::new();
    let mut to_read: Vec<(String, TreeEntry)> = Vec::new();
    for e in entries {
        let Ok(path) = String::from_utf8(e.path.clone()) else {
            let shown = String::from_utf8_lossy(&e.path).into_owned();
            run.issue(&shown, "skipped: the path is not valid UTF-8".to_owned());
            continue;
        };
        if path == GITATTRIBUTES_PATH {
            continue;
        }
        if let Some(reason) = skip_reason(&e, &path) {
            run.issue(&path, reason);
            unread_paths.insert(path);
            continue;
        }
        if let Some(id) = run.marks.get(&(path.clone(), e.oid.clone())) {
            unchanged.push((path, *id));
            continue;
        }
        to_read.push((path, e));
    }

    // Read and parse the rest through one cat-file process.
    let mut parsed: Vec<Parsed> = Vec::new();
    let mut stats = SyncStats::default();
    if !to_read.is_empty() {
        let mut reader = git.blob_reader()?;
        for (path, e) in to_read {
            let Some(bytes) = reader.read(&e.oid)? else {
                run.issue(
                    &path,
                    format!("skipped: git reports blob {} missing", e.oid),
                );
                unread_paths.insert(path);
                continue;
            };
            match read_doc(&bytes) {
                Ok(doc) => parsed.push(Parsed {
                    path,
                    blob: e.oid,
                    doc,
                }),
                Err(reason) => {
                    run.issue(&path, reason);
                    unread_paths.insert(path);
                }
            }
        }
        stats.blobs_read = reader.reads();
        reader.finish()?;
    }

    // An id in two files: report every file and apply none of them.
    let mut files_of: HashMap<MemoryId, Vec<String>> = HashMap::new();
    for p in &parsed {
        files_of.entry(p.doc.id).or_default().push(p.path.clone());
    }
    for (path, id) in &unchanged {
        files_of.entry(*id).or_default().push(path.clone());
    }
    let present: HashSet<MemoryId> = files_of.keys().copied().collect();
    let mut doubled: Vec<(MemoryId, Vec<String>)> = files_of
        .into_iter()
        .filter(|(_, paths)| paths.len() > 1)
        .collect();
    doubled.sort();
    let doubled_ids: HashSet<MemoryId> = doubled.iter().map(|(id, _)| *id).collect();
    for (id, mut paths) in doubled {
        paths.sort();
        let list = paths.join(", ");
        for p in &paths {
            run.issue(
                p,
                format!(
                    "memory id {id} is in {} files ({list}); none of them was applied. \
                     Keep one file per id.",
                    paths.len()
                ),
            );
        }
    }
    parsed.retain(|p| !doubled_ids.contains(&p.doc.id));
    unchanged.retain(|(_, id)| !doubled_ids.contains(id));
    run.unchanged += unchanged.len();

    // New ids, known ids, and ids of other projects.
    let mut news: Vec<Parsed> = Vec::new();
    let mut known: Vec<(Parsed, bool)> = Vec::new();
    for p in parsed {
        match run.by_id.get(&p.doc.id) {
            Some(m) => {
                let adopt = m.origin != Origin::Shared;
                known.push((p, adopt));
            }
            None => match store.find_memory(&p.doc.id)? {
                Some(other) => {
                    let name = store.project(other.project_id)?.name;
                    run.issue(
                        &p.path,
                        format!(
                            "memory id {} belongs to project {name:?} in this store, not {:?}; \
                             the file was not applied",
                            p.doc.id, project.name
                        ),
                    );
                }
                None => news.push(p),
            },
        }
    }

    run.insert_new(news)?;
    for (p, adopt) in known {
        run.apply_known(p, adopt)?;
    }
    run.retract_missing(&present, &unread_paths)?;
    Ok((run.into_reply(), stats))
}

fn empty_reply(dry_run: bool) -> SyncReply {
    SyncReply {
        dry_run,
        added: Vec::new(),
        updated: Vec::new(),
        removed: Vec::new(),
        unchanged: 0,
        errors: Vec::new(),
    }
}

/// Why a listed entry is skipped without being read, if it is.
fn skip_reason(e: &TreeEntry, path: &str) -> Option<String> {
    if e.kind != "blob" || !matches!(e.mode.as_str(), "100644" | "100755") {
        return Some(format!(
            "skipped: not a regular file (git mode {}, {})",
            e.mode, e.kind
        ));
    }
    if !path.ends_with(".md") {
        return Some("skipped: not a .md file".to_owned());
    }
    match e.size {
        Some(size) if size > MAX_SHARED_FILE_BYTES => Some(format!(
            "skipped: the file is {size} bytes, over the {MAX_SHARED_FILE_BYTES}-byte limit \
             for a shared file"
        )),
        _ => None,
    }
}

/// Parses one blob, or says why it cannot be applied.
fn read_doc(bytes: &[u8]) -> std::result::Result<SharedDoc, String> {
    if bytes.len() as u64 > MAX_SHARED_FILE_BYTES {
        return Err(format!(
            "skipped: the file is {} bytes, over the {MAX_SHARED_FILE_BYTES}-byte limit \
             for a shared file",
            bytes.len()
        ));
    }
    let text = std::str::from_utf8(bytes)
        .map_err(|e| format!("the file is not valid UTF-8 (byte {})", e.valid_up_to()))?;
    let doc = SharedDoc::parse(text).map_err(|e| format!("the file does not parse: {e}"))?;
    if doc.body.trim().is_empty() {
        return Err("the file has no body below its title; a memory needs content".to_owned());
    }
    Ok(doc)
}

/// The supersedes target a memory records: its `supersedes_id`, or the
/// unresolved target kept in provenance.
fn recorded_supersedes(m: &Memory) -> Option<MemoryId> {
    m.supersedes_id.or_else(|| {
        let v = m.provenance.extra.as_ref()?.get(UNRESOLVED_SUPERSEDES)?;
        MemoryId::parse(v.as_str()?).ok()
    })
}

/// The order to insert new files in: every superseded file before the file
/// that supersedes it. A cycle is cut where the walk meets itself.
fn insertion_order(news: &[Parsed]) -> Vec<usize> {
    let index: HashMap<MemoryId, usize> = news
        .iter()
        .enumerate()
        .map(|(i, p)| (p.doc.id, i))
        .collect();
    let mut visited = vec![false; news.len()];
    let mut order = Vec::with_capacity(news.len());
    for start in 0..news.len() {
        let mut chain = Vec::new();
        let mut cur = Some(start);
        while let Some(i) = cur {
            if visited[i] {
                break;
            }
            visited[i] = true;
            chain.push(i);
            cur = news[i].doc.supersedes.and_then(|t| index.get(&t).copied());
        }
        order.extend(chain.into_iter().rev());
    }
    order
}

/// A file that was read and parsed.
#[derive(Debug, Clone)]
struct Parsed {
    path: String,
    blob: String,
    doc: SharedDoc,
}

/// A new file ready to insert.
struct Prepared {
    parsed: Parsed,
    status: Status,
    chunks: Vec<NewChunk>,
    mark: SyncMark,
}

/// One sync's state and results.
struct Run<'a> {
    store: &'a Store,
    chunker: &'a dyn Chunker,
    /// Shared text is indexed in full (`index_cap: None`).
    chunk_params: ChunkParams,
    project: &'a Project,
    dry_run: bool,
    head: String,
    /// Every memory of the project as it was before this sync.
    by_id: HashMap<MemoryId, Memory>,
    /// `(path, blob id)` of each mark on a shared memory that is not
    /// retracted. A retracted memory's file is always read again, so a
    /// restored file is reapplied even with the same blob id.
    marks: HashMap<(String, String), MemoryId>,
    /// Content hashes the project holds, and those added in this sync.
    hashes: HashMap<String, MemoryId>,
    /// Ids some memory of the project supersedes.
    targets: HashSet<MemoryId>,
    /// Ids added in this sync.
    done: HashSet<MemoryId>,
    added: Vec<MemoryId>,
    updated: Vec<MemoryId>,
    removed: Vec<MemoryId>,
    unchanged: usize,
    errors: Vec<FileIssue>,
}

impl<'a> Run<'a> {
    fn new(
        store: &'a Store,
        chunker: &'a dyn Chunker,
        chunk_params: &ChunkParams,
        project: &'a Project,
        dry_run: bool,
        head: String,
    ) -> Result<Self> {
        let memories = store.list_memories(&MemoryListFilter {
            project: Some(project.id),
            ..MemoryListFilter::default()
        })?;
        let mut hashes = HashMap::new();
        let mut marks = HashMap::new();
        let mut targets = HashSet::new();
        let mut by_id = HashMap::with_capacity(memories.len());
        for m in memories {
            hashes.entry(m.content_hash.clone()).or_insert(m.id);
            if let Some(t) = recorded_supersedes(&m) {
                targets.insert(t);
            }
            if m.origin == Origin::Shared
                && m.status != Status::Retracted
                && let Some(mark) = SyncMark::of(&m)
            {
                marks.insert((mark.file_path, mark.blob_id), m.id);
            }
            by_id.insert(m.id, m);
        }
        Ok(Run {
            store,
            chunker,
            chunk_params: ChunkParams {
                index_cap: None,
                ..*chunk_params
            },
            project,
            dry_run,
            head,
            by_id,
            marks,
            hashes,
            targets,
            done: HashSet::new(),
            added: Vec::new(),
            updated: Vec::new(),
            removed: Vec::new(),
            unchanged: 0,
            errors: Vec::new(),
        })
    }

    fn issue(&mut self, path: &str, message: String) {
        self.errors.push(FileIssue {
            path: path.to_owned(),
            message,
        });
    }

    fn mark(&self, p: &Parsed) -> SyncMark {
        SyncMark {
            file_path: p.path.clone(),
            commit_sha: Some(p.doc.commit.clone().unwrap_or_else(|| self.head.clone())),
            blob_id: p.blob.clone(),
            text_hash: p.doc.text_hash(),
        }
    }

    /// The status to store: `superseded` for an active file that a memory
    /// of the project supersedes.
    fn effective_status(&self, doc: &SharedDoc) -> Status {
        if doc.status == Status::Active && self.targets.contains(&doc.id) {
            Status::Superseded
        } else {
            doc.status
        }
    }

    fn chunk(&self, body: &str) -> Result<Vec<NewChunk>> {
        let plan = self.chunker.chunk(body, &self.chunk_params)?;
        Ok(plan.pieces.into_iter().map(NewChunk::pending).collect())
    }

    /// The `supersedes_id` to store for a target: the target when it is
    /// added in this sync (`also` holds ids added earlier in the same
    /// batch) or is a memory of the project that is not retracted.
    fn resolve(&self, target: Option<MemoryId>, also: &HashSet<MemoryId>) -> Option<MemoryId> {
        let t = target?;
        if self.done.contains(&t) || also.contains(&t) {
            return Some(t);
        }
        match self.by_id.get(&t) {
            Some(m) if m.status != Status::Retracted => Some(t),
            _ => None,
        }
    }

    fn input(&self, prep: &Prepared, supersedes_id: Option<MemoryId>) -> MemoryInput {
        let doc = &prep.parsed.doc;
        let mut provenance = Provenance::new(ProvenanceSource::Sync);
        provenance.commit_sha = prep.mark.commit_sha.clone();
        provenance.file_path = Some(prep.parsed.path.clone());
        // The mark goes in with the row, inside the batch's one transaction,
        // so a first sync of n files commits once per batch rather than once
        // more per file (the store writes with synchronous=FULL).
        let mut extra = serde_json::Map::new();
        if let (Some(t), None) = (doc.supersedes, supersedes_id) {
            extra.insert(UNRESOLVED_SUPERSEDES.to_owned(), json!(t.to_string()));
        }
        extra.insert(
            SYNC_MARK.to_owned(),
            json!({ "blob": prep.mark.blob_id, "text_hash": prep.mark.text_hash }),
        );
        provenance.extra = Some(serde_json::Value::Object(extra));
        MemoryInput {
            memory: NewMemory {
                id: Some(doc.id),
                project_id: self.project.id,
                memory_type: doc.memory_type,
                status: prep.status,
                title: Some(doc.title.clone()),
                content: doc.body.clone(),
                content_hash: None,
                supersedes_id,
                origin: Origin::Shared,
                importance: DEFAULT_IMPORTANCE,
                tags: doc.tags.clone(),
                imported: None,
                created_at: Some(doc.date),
                updated_at: None,
                provenance,
                anchors: doc.anchors.clone(),
            },
            chunks: prep.chunks.clone(),
        }
    }

    /// Inserts the new files in supersedes order, in batches.
    fn insert_new(&mut self, news: Vec<Parsed>) -> Result<()> {
        let order = insertion_order(&news);
        let mut slots: Vec<Option<Parsed>> = news.into_iter().map(Some).collect();
        let mut batch: Vec<Prepared> = Vec::new();
        for i in order {
            let Some(p) = slots[i].take() else { continue };
            if let Some(prep) = self.prepare(p) {
                batch.push(prep);
            }
            if batch.len() >= INSERT_BATCH {
                self.flush(std::mem::take(&mut batch))?;
            }
        }
        self.flush(batch)
    }

    /// Chunks a new file and checks that its text is not already stored.
    fn prepare(&mut self, p: Parsed) -> Option<Prepared> {
        let chunks = match self.chunk(&p.doc.body) {
            Ok(c) => c,
            Err(e) => {
                self.issue(
                    &p.path,
                    format!("the file could not be chunked: {}", e.message),
                );
                return None;
            }
        };
        let hash = content_hash(&p.doc.body);
        if let Some(other) = self.hashes.get(&hash) {
            let message = format!(
                "its text is already stored as memory {other}; the file was not added. \
                 Remove one of the two copies."
            );
            self.issue(&p.path, message);
            return None;
        }
        self.hashes.insert(hash, p.doc.id);
        let status = self.effective_status(&p.doc);
        let mark = self.mark(&p);
        Some(Prepared {
            parsed: p,
            status,
            chunks,
            mark,
        })
    }

    /// Inserts one batch in one transaction. When the store refuses the
    /// batch for one bad item, the items are inserted one by one so the
    /// rest still land and the bad one is reported.
    fn flush(&mut self, batch: Vec<Prepared>) -> Result<()> {
        if batch.is_empty() {
            return Ok(());
        }
        if self.dry_run {
            for prep in batch {
                self.added_one(&prep)?;
            }
            return Ok(());
        }
        let mut earlier = HashSet::new();
        let inputs: Vec<MemoryInput> = batch
            .iter()
            .map(|prep| {
                let sid = self.resolve(prep.parsed.doc.supersedes, &earlier);
                earlier.insert(prep.parsed.doc.id);
                self.input(prep, sid)
            })
            .collect();
        match self.store.insert_memories(inputs, Access::Sync) {
            Ok(results) => {
                for (prep, r) in batch.iter().zip(results) {
                    self.after_insert(prep, r)?;
                }
            }
            Err(e) if is_item_error(&e) => {
                for prep in &batch {
                    let sid = self.resolve(prep.parsed.doc.supersedes, &HashSet::new());
                    match self
                        .store
                        .insert_memory(self.input(prep, sid), Access::Sync)
                    {
                        Ok(r) => self.after_insert(prep, r)?,
                        Err(e) if is_item_error(&e) => {
                            let message = format!("the file could not be added: {}", e.message);
                            self.issue(&prep.parsed.path, message);
                        }
                        Err(e) => return Err(e),
                    }
                }
            }
            Err(e) => return Err(e),
        }
        Ok(())
    }

    fn after_insert(&mut self, prep: &Prepared, r: Inserted) -> Result<()> {
        match r.outcome {
            RememberOutcome::Created => self.added_one(prep),
            RememberOutcome::Duplicate => {
                let message = format!(
                    "its text is already stored as memory {}; the file was not added. \
                     Remove one of the two copies.",
                    r.id
                );
                self.issue(&prep.parsed.path, message);
                Ok(())
            }
        }
    }

    /// Records a new file as added. Its mark was stored with the row
    /// ([`Run::input`]).
    fn added_one(&mut self, prep: &Prepared) -> Result<()> {
        let id = prep.parsed.doc.id;
        self.done.insert(id);
        if let Some(t) = prep.parsed.doc.supersedes {
            self.targets.insert(t);
        }
        self.added.push(id);
        Ok(())
    }

    /// Revises a shared memory from its file, or adopts the author's local
    /// copy (`adopt`) when the file holds the same text.
    fn apply_known(&mut self, p: Parsed, adopt: bool) -> Result<()> {
        let Some(m) = self.by_id.get(&p.doc.id).cloned() else {
            return Err(CoreError::internal(format!(
                "memory {} was listed but is not loaded",
                p.doc.id
            )));
        };
        let doc = &p.doc;
        let recorded = recorded_supersedes(&m);
        let text_changed = doc.text_hash() != memory_text_hash(&m);
        if adopt && text_changed {
            let message = format!(
                "memory {} is a local memory of this project whose title or text differs from \
                 this file (share may have redacted it); it was not adopted. Share it again, or \
                 forget the local copy and run sync to add the shared one.",
                doc.id
            );
            self.issue(&p.path, message);
            return Ok(());
        }
        if adopt && doc.supersedes != recorded {
            let message = format!(
                "memory {} is a local memory of this project that supersedes {}, but this file \
                 says {}; it was not adopted. Fix supersedes in the file, or forget the local \
                 copy and run sync to add the shared one.",
                doc.id,
                show(recorded),
                show(doc.supersedes)
            );
            self.issue(&p.path, message);
            return Ok(());
        }

        let mut rev = Revision::default();
        if doc.memory_type != m.memory_type {
            rev.memory_type = Some(doc.memory_type);
        }
        let status = self.effective_status(doc);
        if status != m.status {
            rev.status = Some(status);
        }
        if doc.tags != m.tags {
            rev.tags = Some(doc.tags.clone());
        }
        if doc.anchors != m.anchors {
            rev.anchors = Some(doc.anchors.clone());
        }
        if text_changed {
            match self.chunk(&doc.body) {
                Ok(chunks) => {
                    rev.title = Some(doc.title.clone());
                    rev.content = Some(ContentChange {
                        content: doc.body.clone(),
                        chunks,
                    });
                }
                Err(e) => {
                    let message = format!("the file could not be chunked: {}", e.message);
                    self.issue(&p.path, message);
                    return Ok(());
                }
            }
        }
        let changed = rev != Revision::default();
        if changed && !self.dry_run {
            match self.store.revise_memory(&m.id, rev, Access::Sync) {
                Ok(_) => {}
                Err(e) if is_item_error(&e) => {
                    let message = format!("the file could not be applied: {}", e.message);
                    self.issue(&p.path, message);
                    return Ok(());
                }
                Err(e) => return Err(e),
            }
        }
        if doc.supersedes != recorded {
            let message = format!(
                "its supersedes changed from {} to {}; a stored memory's supersedes is fixed, \
                 so that change was not applied. The rest of the file was applied.",
                show(recorded),
                show(doc.supersedes)
            );
            self.issue(&p.path, message);
            return Ok(());
        }
        let mark = self.mark(&p);
        let marked = SyncMark::of(&m).is_some_and(|old| old == mark) && m.origin == Origin::Shared;
        if !marked && !self.dry_run {
            self.store.record_sync(&m.id, &mark)?;
        }
        if changed || adopt {
            self.updated.push(m.id);
        } else {
            self.unchanged += 1;
        }
        Ok(())
    }

    /// Retracts the shared memories of the project whose file is gone.
    fn retract_missing(
        &mut self,
        present: &HashSet<MemoryId>,
        unread_paths: &HashSet<String>,
    ) -> Result<()> {
        let mut gone: Vec<MemoryId> = self
            .by_id
            .values()
            .filter(|m| m.origin == Origin::Shared && m.status != Status::Retracted)
            .filter(|m| !present.contains(&m.id))
            .filter(|m| {
                !m.provenance
                    .file_path
                    .as_ref()
                    .is_some_and(|f| unread_paths.contains(f))
            })
            .map(|m| m.id)
            .collect();
        gone.sort();
        for id in gone {
            if !self.dry_run {
                let rev = Revision {
                    status: Some(Status::Retracted),
                    ..Revision::default()
                };
                self.store.revise_memory(&id, rev, Access::Sync)?;
            }
            self.removed.push(id);
        }
        Ok(())
    }

    fn into_reply(mut self) -> SyncReply {
        self.added.sort();
        self.updated.sort();
        self.removed.sort();
        self.errors
            .sort_by(|a, b| a.path.cmp(&b.path).then_with(|| a.message.cmp(&b.message)));
        SyncReply {
            dry_run: self.dry_run,
            added: self.added,
            updated: self.updated,
            removed: self.removed,
            unchanged: u32::try_from(self.unchanged).unwrap_or(u32::MAX),
            errors: self.errors,
        }
    }
}

fn show(id: Option<MemoryId>) -> String {
    id.map_or_else(|| "nothing".to_owned(), |i| i.to_string())
}
