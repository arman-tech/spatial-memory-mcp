//! The command-line and sharing ops (CONTRACTS 5; SPEC 9, 10, 11; B-30,
//! B-71, B-79).
//!
//! - `share`, `sync`, `import`, `export` and `migrate` run through the
//!   injected [`SharingOps`]. `share` and `sync` need a repository and fail
//!   with `not_git_repository` before the trait is called. After every call,
//!   successful or not (a failed import may have stored part of a file), the
//!   embedding queue is woken and the cached maps are dropped.
//! - `reindex` needs the chunker. `Store::reindex` backs the database up,
//!   re-chunks every memory with the configured settings and writes the
//!   stamp last (B-71); the daemon then lets the bulk queue run again.
//! - `project_rename` and `project_delete` forget the cached project
//!   resolutions and maps. A delete without `confirm` deletes nothing and
//!   fails with `confirm_required`, listing what would go (B-30).
//! - `doctor_eval` runs each query through recall's vector path (the query
//!   cut to the chunk limit, exact search over active memories, the first
//!   `EVAL_LIMIT`) and scores the lists with `sm_core::eval::evaluate`. A
//!   set with its own memories runs against a scratch store built from
//!   them in the profile folder and removed afterwards; a set without runs
//!   against the live store, over the current project when the working
//!   directory resolves to one and over every project otherwise.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::Ordering;

use sm_core::error::{CoreError, ErrorCode, Result};
use sm_core::eval::{Baseline, EVAL_LIMIT, EvalReport, EvalSet, Retrieved, evaluate};
use sm_core::ids::MemoryId;
use sm_core::project::{ProjectIdentity, project_path_key};
use sm_core::protocol::{
    DoctorEvalParams, ExportParams, ExportReport, ImportParams, ImportReport, MigrateParams,
    ProjectDeleteParams, ProjectRenameParams, ProjectReply, ReindexReport, ShareParams, ShareReply,
    SyncParams, SyncReply,
};
use sm_core::traits::{DocInput, MemoryFilter, QueryScope, SessionKind, VectorQuery};
use sm_core::types::{IndexStamp, NewMemory, Provenance, ProvenanceSource};
use sm_store::{Access, MemoryInput, Store};

use crate::dispatch::Ctx;
use crate::project;
use crate::sharing::{ResolvedProject, SharingContext};
use crate::state::{ProjectEntry, Ready, Shared};
use crate::tools::remember::pending_chunks;

fn not_git_repository(cwd: &str) -> CoreError {
    CoreError::new(
        ErrorCode::NotGitRepository,
        format!("{cwd} is not inside a git repository; shared memories live in one"),
    )
    .with_next_action("Run the command from inside the project's repository.")
}

/// Calls `op` with a sharing context, then wakes the queue and drops the
/// maps whatever the outcome.
fn with_sharing<T>(
    ctx: &Ctx<'_>,
    entry: Option<&ProjectEntry>,
    op: impl FnOnce(&SharingContext<'_>) -> Result<T>,
) -> Result<T> {
    let sh = ctx.sh;
    let ready = sh.ready_opt();
    let sctx = SharingContext {
        store: &sh.store,
        chunker: ready.as_ref().map(|r| &*r.chunker),
        scanner: &sh.scanner,
        chunk_params: sh.chunk_params(None),
        project: entry.map(|e| ResolvedProject {
            project: &e.project,
            identity: &e.identity,
        }),
    };
    let out = op(&sctx);
    sh.after_write();
    out
}

/// The current project, which must be in a repository.
fn repo_project(ctx: &Ctx<'_>) -> Result<ProjectEntry> {
    let entry = ctx.project()?;
    if entry.identity.repo.is_none() {
        return Err(not_git_repository(&ctx.req.cwd));
    }
    Ok(entry)
}

pub(crate) fn share(ctx: &Ctx<'_>, p: &ShareParams) -> Result<ShareReply> {
    let entry = repo_project(ctx)?;
    with_sharing(ctx, Some(&entry), |s| ctx.sh.sharing.share(s, p))
}

pub(crate) fn sync(ctx: &Ctx<'_>, p: &SyncParams) -> Result<SyncReply> {
    let entry = repo_project(ctx)?;
    with_sharing(ctx, Some(&entry), |s| ctx.sh.sharing.sync(s, p))
}

pub(crate) fn import(ctx: &Ctx<'_>, p: &ImportParams) -> Result<ImportReport> {
    let entry = ctx.project().ok();
    with_sharing(ctx, entry.as_ref(), |s| ctx.sh.sharing.import(s, p))
}

pub(crate) fn export(ctx: &Ctx<'_>, p: &ExportParams) -> Result<ExportReport> {
    let entry = ctx.project().ok();
    with_sharing(ctx, entry.as_ref(), |s| ctx.sh.sharing.export(s, p))
}

pub(crate) fn migrate(ctx: &Ctx<'_>, p: &MigrateParams) -> Result<ImportReport> {
    let entry = ctx.project().ok();
    with_sharing(ctx, entry.as_ref(), |s| ctx.sh.sharing.migrate(s, p))
}

pub(crate) fn reindex(ctx: &Ctx<'_>) -> Result<ReindexReport> {
    let sh = ctx.sh;
    let ready = sh.ready()?;
    let params = sh.chunk_params(None);
    let report = sh.store.reindex(|m| {
        let plan = ready.chunker.chunk(&m.content, &params)?;
        if plan.pieces.is_empty() {
            return Err(CoreError::invalid_input(format!(
                "memory {} gave no chunks; a memory is never stored without chunks",
                m.id
            )));
        }
        Ok(plan.pieces)
    })?;
    sh.index_ok.store(true, Ordering::SeqCst);
    sh.after_write();
    sh.log.line(format!(
        "reindexed {} memories, {} chunks queued; backup {}",
        report.memories, report.chunks_queued, report.backup
    ));
    Ok(report)
}

pub(crate) fn project_rename(ctx: &Ctx<'_>, p: &ProjectRenameParams) -> Result<ProjectReply> {
    let sh = ctx.sh;
    let reply = sh.store.rename_project(&p.from, &p.to)?;
    project::forget_cache(sh);
    sh.drop_maps();
    Ok(reply)
}

pub(crate) fn project_delete(ctx: &Ctx<'_>, p: &ProjectDeleteParams) -> Result<ProjectReply> {
    let sh = ctx.sh;
    if !p.confirm {
        let found: Vec<_> = sh
            .store
            .project_statuses()?
            .into_iter()
            .filter(|s| s.name == p.name)
            .collect();
        let status = match found.as_slice() {
            [] => {
                return Err(CoreError::not_found(format!(
                    "no project is named {:?}",
                    p.name
                )));
            }
            [one] => one,
            many => {
                return Err(CoreError::invalid_input(format!(
                    "{} projects are named {:?}; rename one from inside its repository first",
                    many.len(),
                    p.name
                )));
            }
        };
        let types = status
            .by_type
            .iter()
            .map(|(t, n)| format!("{n} {t}"))
            .collect::<Vec<_>>()
            .join(", ");
        let detail = if types.is_empty() {
            String::new()
        } else {
            format!(" ({types})")
        };
        return Err(CoreError::new(
            ErrorCode::ConfirmRequired,
            format!(
                "deleting project {:?} would remove its {} memories{detail}, {} of them shared, with their chunks, anchors, links, map runs and injections; nothing was deleted",
                status.name, status.memories, status.shared
            ),
        )
        .with_next_action("Repeat with confirm: true (`--yes`) to delete them."));
    }
    let reply = sh.store.delete_project(&p.name)?;
    project::forget_cache(sh);
    sh.drop_maps();
    Ok(reply)
}

pub(crate) fn doctor_eval(ctx: &Ctx<'_>, p: &DoctorEvalParams) -> Result<EvalReport> {
    let sh = ctx.sh;
    let set = EvalSet::load(Path::new(&p.set_path))?;
    let baseline = match &p.baseline_path {
        Some(b) => Some(Baseline::load(Path::new(b))?),
        None => None,
    };
    let ready = sh.ready()?;
    let retrieved = if set.memories.is_empty() {
        let scope = match ctx.project() {
            Ok(e) => QueryScope::Project(e.project.id),
            Err(_) => QueryScope::All,
        };
        run_queries(ctx, &ready, &sh.store, &set, scope, &HashMap::new())?
    } else {
        Scratch::create(sh)?.run(ctx, &ready, &set)?
    };
    evaluate(&set, &retrieved, baseline.as_ref())
}

/// Recall's vector path for each query; `names` maps store ids back to the
/// set's ids (empty for the live store).
fn run_queries(
    ctx: &Ctx<'_>,
    ready: &Ready,
    store: &Store,
    set: &EvalSet,
    scope: QueryScope,
    names: &HashMap<MemoryId, String>,
) -> Result<Vec<Vec<Retrieved>>> {
    let sh = ctx.sh;
    let mut out = Vec::with_capacity(set.queries.len());
    for q in &set.queries {
        let vector = ready
            .query
            .embed_query(&q.text, sh.settings.chunk.max_tokens as usize)?;
        let hits = store.nearest(&VectorQuery {
            vector: &vector,
            scope,
            filter: MemoryFilter::default(),
            limit: EVAL_LIMIT,
        })?;
        ctx.check_deadline("an evaluation query")?;
        out.push(
            hits.into_iter()
                .map(|h| Retrieved {
                    id: names
                        .get(&h.memory_id)
                        .cloned()
                        .unwrap_or_else(|| h.memory_id.to_string()),
                    score: h.score,
                })
                .collect(),
        );
    }
    Ok(out)
}

/// A store for a self-contained set, removed when dropped.
struct Scratch {
    dir: PathBuf,
    store: Option<Store>,
}

impl Drop for Scratch {
    fn drop(&mut self) {
        // The connection closes before its folder is removed.
        self.store.take();
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

impl Scratch {
    fn create(sh: &Shared) -> Result<Scratch> {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let dir = sh
            .paths
            .dir
            .join("eval")
            .join(format!("{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&dir)
            .map_err(|e| CoreError::storage(format!("cannot create {}: {e}", dir.display())))?;
        let mut scratch = Scratch { dir, store: None };
        let store = Store::open_at(&scratch.dir.join("memory.db"), &scratch.dir.join("backups"))?;
        store.configure_index(IndexStamp {
            model_id: sh.info.model_id.clone(),
            revision: sh.info.revision.clone(),
            dim: sh.info.dim,
            chunk_max_tokens: sh.settings.chunk.max_tokens,
            chunk_overlap: sh.settings.chunk.overlap,
        })?;
        scratch.store = Some(store);
        Ok(scratch)
    }

    fn store(&self) -> Result<&Store> {
        self.store
            .as_ref()
            .ok_or_else(|| CoreError::internal("the scratch store is closed"))
    }

    fn run(&self, ctx: &Ctx<'_>, ready: &Ready, set: &EvalSet) -> Result<Vec<Vec<Retrieved>>> {
        let sh = ctx.sh;
        let store = self.store()?;
        let root = self.dir.clone();
        let identity = ProjectIdentity {
            root_path: root.to_string_lossy().replace('\\', "/"),
            root_key: project_path_key(&root),
            root,
            git_remote: None,
            repo: None,
            name: "eval".to_owned(),
            capture_enabled: false,
            warnings: Vec::new(),
        };
        let project = store.upsert_project(&identity)?;
        let mut inputs = Vec::with_capacity(set.memories.len());
        for m in &set.memories {
            let (chunks, _) = pending_chunks(sh, ready, &m.content, None)?;
            let memory = NewMemory::new(
                project.id,
                m.content.clone(),
                Provenance::new(ProvenanceSource::Import),
            );
            inputs.push(MemoryInput { memory, chunks });
        }
        let inserted = store.insert_memories(inputs, Access::Local)?;
        // Two set memories with the same content are one stored memory
        // (B-29); the first id names it.
        let mut names: HashMap<MemoryId, String> = HashMap::new();
        for (i, m) in inserted.iter().zip(&set.memories) {
            names.entry(i.id).or_insert_with(|| m.id.clone());
        }
        embed_all(sh, ready, store)?;
        ctx.check_deadline("indexing the evaluation set")?;
        run_queries(
            ctx,
            ready,
            store,
            set,
            QueryScope::Project(project.id),
            &names,
        )
    }
}

/// Embeds every pending chunk of `store` with a bulk session of its own,
/// counted in `bulk_open` while it is open.
fn embed_all(sh: &Arc<Shared>, ready: &Ready, store: &Store) -> Result<()> {
    struct Open<'a>(&'a Shared);
    impl Drop for Open<'_> {
        fn drop(&mut self) {
            self.0.bulk_open.fetch_sub(1, Ordering::SeqCst);
        }
    }
    let session = ready.provider.open(
        SessionKind::Bulk,
        sh.settings.bulk_threads(sh.physical_cores),
    )?;
    sh.bulk_open.fetch_add(1, Ordering::SeqCst);
    let _open = Open(sh);
    let batch_size = sh.settings.embed.batch_size.max(1) as usize;
    loop {
        let batch = store.pending_chunks(batch_size)?;
        if batch.is_empty() {
            return Ok(());
        }
        let docs: Vec<DocInput<'_>> = batch
            .iter()
            .map(|c| DocInput {
                title: c.title.as_deref(),
                text: &c.text,
            })
            .collect();
        let vectors = session.embed_documents(&docs)?;
        if vectors.len() != batch.len() {
            return Err(CoreError::internal(format!(
                "the embedder returned {} vectors for {} chunks",
                vectors.len(),
                batch.len()
            )));
        }
        let items: Vec<_> = batch.iter().map(|c| c.id).zip(vectors).collect();
        store.set_embeddings(&session.info().model_id, &items)?;
    }
}
