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
    DoctorEvalParams, DoctorNumbersReply, ExportParams, ExportReport, ImportParams, ImportReport,
    MigrateParams, ProjectDeleteParams, ProjectRenameParams, ProjectReply, ReindexReport,
    ShareParams, ShareReply, SyncParams, SyncReply,
};
use sm_core::traits::{MemoryFilter, QueryScope, VectorQuery};
use sm_core::types::{IndexStamp, NewMemory, Provenance, ProvenanceSource};
use sm_store::{Access, MemoryInput, Store};

use crate::dispatch::Ctx;
use crate::project;
use crate::queue;
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
        capture_index_cap: sh.stored_index_cap(ProvenanceSource::Capture),
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
    let report = sh.store.reindex(|m| {
        // A captured memory keeps its capture index cap (B-118).
        let params = sh.chunk_params(sh.stored_index_cap(m.provenance.source));
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

/// `doctor_numbers` (SPEC 5 rule 7; B-125): the embedder's numbers, timed
/// on the daemon's own sessions. Bulk throughput is timed on the one bulk
/// session through [`queue::with_bulk`], so the bulk queue waits meanwhile
/// and no third session is opened; the query median on the query session
/// kept for life (spike decision 5). While the model loads the reply is
/// `daemon_loading`, and after a failed load the model's error.
pub(crate) fn doctor_numbers(ctx: &Ctx<'_>) -> Result<DoctorNumbersReply> {
    let sh = ctx.sh;
    let ready = sh.ready()?;
    let n = queue::with_bulk(sh, &ready, |bulk| {
        sm_embed::doctor::time_sessions(
            bulk,
            ready.query.as_ref(),
            ready.chunker.as_ref(),
            &sh.settings,
            sh.physical_cores,
        )
    })?;
    let count = |v: usize| u32::try_from(v).unwrap_or(u32::MAX);
    Ok(DoctorNumbersReply {
        physical_cores: count(n.physical_cores),
        query_threads: count(n.query_threads),
        bulk_threads: count(n.bulk_threads),
        batch_size: count(n.batch_size),
        sample_chunks: count(n.sample_chunks),
        query_tokens: count(n.query_tokens),
        query_runs: count(n.query_runs),
        bulk_elapsed_ms: n.bulk_elapsed.as_secs_f64() * 1000.0,
        bulk_chunks_per_sec: n.bulk_chunks_per_sec,
        query_median_ms: n.query_latency_median.as_secs_f64() * 1000.0,
    })
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

/// Embeds every pending chunk of the scratch `store` on the daemon's one
/// bulk session, a batch at a time, taking turns with the queue, so the
/// bulk threads stay within the budget (SPEC 3 rule 5; AT-3; e-report
/// decision 35).
fn embed_all(sh: &Arc<Shared>, ready: &Ready, store: &Store) -> Result<()> {
    let batch_size = sh.settings.embed.batch_size.max(1) as usize;
    loop {
        let batch = store.pending_chunks(batch_size)?;
        if batch.is_empty() {
            return Ok(());
        }
        queue::embed_chunks(sh, ready, store, &batch)?;
    }
}
