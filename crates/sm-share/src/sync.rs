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
//! memory (it is skipped without a report), and a known file whose
//! `supersedes` changed, which is in `errors` and also in `updated` when the
//! rest of the file changed its memory:
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
//! `updated` also lists a memory whose status sync changed because a memory
//! that supersedes it was added, retracted or brought back (see
//! Supersedes), a local memory included; such a memory's file, if it has
//! one, is not counted in `unchanged` as well.
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
//! files are revised or adopted next, then missing files are retracted, and
//! last the supersessions this sync touched are settled (see Supersedes).
//! Every id list in the reply is sorted, and the errors are sorted by path.
//!
//! # Supersedes
//!
//! A memory is superseded while at least one memory of its project that is
//! not retracted supersedes it (the lead's liveness rule, which replaces the
//! stream's first rule that a supersession is never undone).
//!
//! A new file's `supersedes` is stored as `supersedes_id`, which marks the
//! target superseded, when the target is in this project and not retracted,
//! or was added earlier in the same sync with a status other than
//! `retracted`. Otherwise the target is missing: the file is still added,
//! its `supersedes_id` is left empty and the target's id is kept in
//! provenance `extra.unresolved_supersedes`. When that target arrives later
//! it is stored as superseded if its superseder is live.
//!
//! A file whose status is `active` is stored as `superseded` when a memory
//! of this project that is not retracted after this sync supersedes it, so
//! a target's own unchanged file never undoes a live supersession. Memories
//! that are retracted, or that this sync retracts (their file is gone or
//! says `retracted`), do not count.
//!
//! After every file is applied, each target this sync touched is settled:
//! a target whose own file was applied, or one that a memory added in this
//! sync supersedes, or one whose superseder was retracted or brought back
//! from retracted. Such a target that is active with a live superseder
//! becomes superseded. One that is superseded with no live superseder is
//! restored: a shared memory to the status its file gives (when its mark
//! had skipped the file, the file is read again for that), a local memory
//! to active. A shared target whose file could not be read keeps its status
//! and the file's error is reported. A retracted target is never changed.
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
    let mut unchanged: Vec<(String, String, MemoryId)> = Vec::new();
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
            unchanged.push((path, e.oid.clone(), *id));
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
    for (path, _, id) in &unchanged {
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
    unchanged.retain(|(_, _, id)| !doubled_ids.contains(id));
    run.unchanged += unchanged.len();
    for (path, blob, id) in unchanged {
        run.unchanged_ids.insert(id);
        run.marked_files.insert(id, (path, blob));
    }

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

    // Which supersessions stay live once every file is applied (the lead's
    // liveness rule), before any file is applied.
    let gone = run.missing(&present, &unread_paths);
    run.plan_targets(&known, &gone);

    run.insert_new(news)?;
    for (p, adopt) in known {
        run.apply_known(p, adopt)?;
    }
    run.retract_missing(gone)?;
    run.settle(&git, &mut stats)?;
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
    /// The supersedes target each memory of the project records, whatever
    /// its status, and each memory added in this sync: superseder to target.
    supersedes_of: HashMap<MemoryId, MemoryId>,
    /// Ids that a memory which is not retracted after this sync supersedes
    /// ([`Run::plan_targets`], then [`Run::added_one`]).
    targets: HashSet<MemoryId>,
    /// Ids this sync retracts: files that are gone, and known files whose
    /// status is `retracted`.
    retracting: HashSet<MemoryId>,
    /// The status of every memory of the project as this sync has left it
    /// so far (what a dry run would have written).
    status_now: HashMap<MemoryId, Status>,
    /// The status written in the file of each memory whose file this sync
    /// applied.
    file_status: HashMap<MemoryId, Status>,
    /// `(path, blob id)` of each file skipped because its mark matched.
    marked_files: HashMap<MemoryId, (String, String)>,
    /// Ids counted in `unchanged`.
    unchanged_ids: HashSet<MemoryId>,
    /// Ids added in this sync, with the status each was stored with.
    done: HashMap<MemoryId, Status>,
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
        let mut supersedes_of = HashMap::new();
        let mut status_now = HashMap::with_capacity(memories.len());
        let mut by_id = HashMap::with_capacity(memories.len());
        for m in memories {
            hashes.entry(m.content_hash.clone()).or_insert(m.id);
            if let Some(t) = recorded_supersedes(&m) {
                supersedes_of.insert(m.id, t);
            }
            status_now.insert(m.id, m.status);
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
            supersedes_of,
            targets: HashSet::new(),
            retracting: HashSet::new(),
            status_now,
            file_status: HashMap::new(),
            marked_files: HashMap::new(),
            unchanged_ids: HashSet::new(),
            done: HashMap::new(),
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

    /// Plans which ids are superseded after this sync, before anything is
    /// written: the targets of every memory of the project that is not
    /// retracted after it. A known file sets its memory's status (so a
    /// retracted memory whose file is active again counts), and a memory
    /// whose file is gone is retracted. New files add their targets as they
    /// are inserted ([`Run::added_one`]); [`Run::settle`] corrects whatever
    /// a file that could not be applied changed.
    fn plan_targets(&mut self, known: &[(Parsed, bool)], gone: &[MemoryId]) {
        let mut after = self.status_now.clone();
        for (p, _) in known {
            after.insert(p.doc.id, p.doc.status);
            if p.doc.status == Status::Retracted {
                self.retracting.insert(p.doc.id);
            }
        }
        for id in gone {
            after.insert(*id, Status::Retracted);
            self.retracting.insert(*id);
        }
        self.targets = self
            .supersedes_of
            .iter()
            .filter(|(s, _)| after.get(s).is_some_and(|st| *st != Status::Retracted))
            .map(|(_, t)| *t)
            .collect();
    }

    /// The `supersedes_id` to store for a target: the target when it is a
    /// memory of the project, or one added in this sync (`earlier` holds the
    /// ids added earlier in the same batch, with the status each is stored
    /// with), and is not retracted. A retracted target is left unresolved:
    /// storing it would mark it superseded and so undo its retraction.
    fn resolve(
        &self,
        target: Option<MemoryId>,
        earlier: &HashMap<MemoryId, Status>,
    ) -> Option<MemoryId> {
        let t = target?;
        let status = earlier
            .get(&t)
            .or_else(|| self.status_now.get(&t))
            .copied()?;
        (status != Status::Retracted).then_some(t)
    }

    /// The report for a new file whose text the project already holds as
    /// `other`.
    fn duplicate_message(&self, other: &MemoryId) -> String {
        let retracted_now = self.status_now.get(other) == Some(&Status::Retracted);
        if !retracted_now && !self.retracting.contains(other) {
            return format!(
                "its text is already stored as memory {other}; the file was not added. \
                 Remove one of the two copies."
            );
        }
        let when = if retracted_now {
            "which is retracted"
        } else {
            "which this sync retracts"
        };
        let action = match self.by_id.get(other) {
            Some(m) if m.origin == Origin::Local => format!(
                "Forget memory {other} and run sync again to add this file, or change this \
                 file's text."
            ),
            _ => format!(
                "To bring the text back, restore memory {other}'s own file, or change this \
                 file's text."
            ),
        };
        format!(
            "its text is already stored as memory {other}, {when}; the file was skipped and \
             the retracted copy is kept. {action}"
        )
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
            let message = self.duplicate_message(other);
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
        let mut earlier = HashMap::new();
        let sids: Vec<Option<MemoryId>> = batch
            .iter()
            .map(|prep| {
                let sid = self.resolve(prep.parsed.doc.supersedes, &earlier);
                earlier.insert(prep.parsed.doc.id, prep.status);
                sid
            })
            .collect();
        if self.dry_run {
            // The store would mark each resolved target superseded.
            for (prep, sid) in batch.iter().zip(sids) {
                self.added_one(prep, sid);
            }
            return Ok(());
        }
        let inputs: Vec<MemoryInput> = batch
            .iter()
            .zip(&sids)
            .map(|(prep, sid)| self.input(prep, *sid))
            .collect();
        match self.store.insert_memories(inputs, Access::Sync) {
            Ok(results) => {
                for (prep, r) in batch.iter().zip(results) {
                    self.after_insert(prep, r);
                }
            }
            Err(e) if is_item_error(&e) => {
                for prep in &batch {
                    let sid = self.resolve(prep.parsed.doc.supersedes, &HashMap::new());
                    match self
                        .store
                        .insert_memory(self.input(prep, sid), Access::Sync)
                    {
                        Ok(r) => self.after_insert(prep, r),
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

    fn after_insert(&mut self, prep: &Prepared, r: Inserted) {
        match r.outcome {
            RememberOutcome::Created => self.added_one(prep, r.superseded),
            RememberOutcome::Duplicate => {
                let message = self.duplicate_message(&r.id);
                self.issue(&prep.parsed.path, message);
            }
        }
    }

    /// Records a new file as added. Its mark was stored with the row
    /// ([`Run::input`]); `marked` is the target the store marked superseded
    /// with it.
    fn added_one(&mut self, prep: &Prepared, marked: Option<MemoryId>) {
        let id = prep.parsed.doc.id;
        self.done.insert(id, prep.status);
        self.status_now.insert(id, prep.status);
        self.file_status.insert(id, prep.parsed.doc.status);
        if let Some(t) = prep.parsed.doc.supersedes {
            self.supersedes_of.insert(id, t);
            if prep.status != Status::Retracted {
                self.targets.insert(t);
            }
        }
        if let Some(t) = marked {
            self.status_now.insert(t, Status::Superseded);
        }
        self.added.push(id);
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
        // Against the status as this sync has left it: a new file's insert
        // may have marked this memory superseded.
        let status = self.effective_status(doc);
        let current = self.status_now.get(&m.id).copied().unwrap_or(m.status);
        if status != current {
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
        let new_status = rev.status;
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
        self.file_status.insert(m.id, doc.status);
        if let Some(s) = new_status {
            self.status_now.insert(m.id, s);
        }
        if doc.supersedes != recorded {
            let message = format!(
                "its supersedes changed from {} to {}; a stored memory's supersedes is fixed, \
                 so that change was not applied. The rest of the file was applied.",
                show(recorded),
                show(doc.supersedes)
            );
            self.issue(&p.path, message);
            if changed {
                self.updated.push(m.id);
            }
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
            self.unchanged_ids.insert(m.id);
        }
        Ok(())
    }

    /// The shared memories of the project whose file is gone, sorted.
    fn missing(
        &self,
        present: &HashSet<MemoryId>,
        unread_paths: &HashSet<String>,
    ) -> Vec<MemoryId> {
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
        gone
    }

    /// Retracts the shared memories whose file is gone ([`Run::missing`]).
    fn retract_missing(&mut self, gone: Vec<MemoryId>) -> Result<()> {
        for id in gone {
            if !self.dry_run {
                let rev = Revision {
                    status: Some(Status::Retracted),
                    ..Revision::default()
                };
                self.store.revise_memory(&id, rev, Access::Sync)?;
            }
            self.status_now.insert(id, Status::Retracted);
            self.removed.push(id);
        }
        Ok(())
    }

    /// Settles the supersessions this sync touched, after every file is
    /// applied (the lead's liveness rule): a memory is superseded while at
    /// least one memory of its project that is not retracted supersedes it.
    ///
    /// A target is touched when its own file was applied in this sync, or
    /// when a memory that supersedes it was added, or retracted, or brought
    /// back from retracted. A touched target that is active and has a live
    /// superseder becomes superseded. A touched target that is superseded
    /// and has none is restored: a shared one to the status its file gives
    /// (the file is read again when its mark had skipped it), a local one to
    /// active. A shared one whose file could not be read is left as it is;
    /// that file's error is reported. Retracted memories are never changed.
    fn settle(&mut self, git: &Git, stats: &mut SyncStats) -> Result<()> {
        let mut live: HashSet<MemoryId> = HashSet::new();
        let mut scope: HashSet<MemoryId> = HashSet::new();
        for (s, t) in &self.supersedes_of {
            let now = self.status_now.get(s).copied();
            if now.is_some_and(|st| st != Status::Retracted) {
                live.insert(*t);
            }
            let before = self.by_id.get(s).map(|m| m.status == Status::Retracted);
            let after = now.map(|st| st == Status::Retracted);
            if self.done.contains_key(s) || before != after {
                scope.insert(*t);
            }
            if self.file_status.contains_key(t) {
                scope.insert(*t);
            }
        }
        let mut scope: Vec<MemoryId> = scope.into_iter().collect();
        scope.sort();

        let mut reread: Vec<(MemoryId, String, String)> = Vec::new();
        for t in scope {
            let Some(now) = self.status_now.get(&t).copied() else {
                continue;
            };
            let want = match (now, live.contains(&t)) {
                (Status::Active, true) => Some(Status::Superseded),
                (Status::Superseded, false) => {
                    if let Some(&s) = self.file_status.get(&t) {
                        Some(s)
                    } else if self
                        .by_id
                        .get(&t)
                        .is_some_and(|m| m.origin == Origin::Local)
                    {
                        Some(Status::Active)
                    } else {
                        if let Some((path, blob)) = self.marked_files.get(&t) {
                            reread.push((t, path.clone(), blob.clone()));
                        }
                        None
                    }
                }
                _ => None,
            };
            if let Some(s) = want.filter(|s| *s != now) {
                self.set_status(t, s)?;
            }
        }
        if reread.is_empty() {
            return Ok(());
        }

        // Files whose mark had skipped them: read them now for their status.
        let mut reader = git.blob_reader()?;
        for (t, path, blob) in reread {
            let read = reader.read(&blob)?;
            let doc = match read.as_deref().map(read_doc) {
                Some(Ok(doc)) if doc.id == t => doc,
                Some(Ok(doc)) => {
                    let reason = format!(
                        "the file holds memory {}, not memory {t} that its mark names; \
                         memory {t} was left superseded",
                        doc.id
                    );
                    self.unread_again(t, &path, reason);
                    continue;
                }
                Some(Err(reason)) => {
                    self.unread_again(t, &path, reason);
                    continue;
                }
                None => {
                    let reason = format!("skipped: git reports blob {blob} missing");
                    self.unread_again(t, &path, reason);
                    continue;
                }
            };
            self.file_status.insert(t, doc.status);
            if doc.status != Status::Superseded {
                self.set_status(t, doc.status)?;
            }
        }
        stats.blobs_read += reader.reads();
        reader.finish()
    }

    /// A file read again by [`Run::settle`] that could not be used: it is
    /// reported, and no longer counted as unchanged.
    fn unread_again(&mut self, id: MemoryId, path: &str, reason: String) {
        if self.unchanged_ids.remove(&id) {
            self.unchanged -= 1;
        }
        self.issue(path, reason);
    }

    /// Sets one memory's status (only in memory on a dry run) and reports
    /// it as updated unless it was added in this sync.
    fn set_status(&mut self, id: MemoryId, status: Status) -> Result<()> {
        if !self.dry_run {
            let rev = Revision {
                status: Some(status),
                ..Revision::default()
            };
            self.store.revise_memory(&id, rev, Access::Sync)?;
        }
        self.status_now.insert(id, status);
        if self.unchanged_ids.remove(&id) {
            self.unchanged -= 1;
        }
        if !self.done.contains_key(&id) {
            self.updated.push(id);
        }
        Ok(())
    }

    fn into_reply(mut self) -> SyncReply {
        self.added.sort();
        self.updated.sort();
        self.updated.dedup();
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
