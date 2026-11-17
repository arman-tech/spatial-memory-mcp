//! Whole-store `export` and `import` as JSON Lines (decision B-77).
//!
//! **File.** UTF-8, LF line endings, one JSON object per line:
//! 1. the header `{"format":"spatial-memory-export","version":1,
//!    "build":…,"exported_at":…,"schema":…}`;
//! 2. one `{"project":{…}}` line per project, in id order, with its
//!    identity key, name, remote and root path;
//! 3. one `{"memory":{…}}` line per memory, in id order, with every stored
//!    field; its project is written as the project's identity key;
//! 4. one `{"link":{…}}` line per stored link, project by project.
//!
//! The lines are this module's own structs, so a change to `sm-core`'s
//! serde forms cannot change the file silently. Nothing derived is written:
//! no chunks, vectors, keyword rows, map runs or injections.
//!
//! **Export** refuses a path that exists, writes a temporary file in the
//! same folder and renames it, and gives the same bytes for the same store
//! apart from the header's `exported_at` (and `build` across versions).
//!
//! **Import** reads the file twice. The first pass checks the header (an
//! unknown `format` or a newer `version` is refused before anything is
//! written) and collects the memory ids and every `supersedes_id`. The
//! second pass applies the lines:
//! - A project line finds or creates the project by identity key
//!   (`Store::ensure_project`); new ones are listed in `projects_created`.
//! - A memory line keeps the id, type, status, origin, provenance,
//!   anchors, tags, importance, `supersedes_id`, `imported` and both
//!   timestamps. Every string stored is scanned and redacted as `migrate`
//!   does: content, title, the provenance author, session, commit, file
//!   path and tool, and every string value in `provenance.extra` and
//!   `imported` (values only, never keys). A tag or anchor the scanner
//!   matches cannot hold a marker, so it is dropped. Each memory changed
//!   this way counts once in `redacted`; redacting text that was already
//!   redacted changes nothing (B-64). The content hash is recomputed from
//!   the content stored. The content is chunked again and its chunks wait
//!   for the embedding queue.
//! - A memory whose `supersedes_id` names a memory later in the file waits
//!   until that memory is queued, so targets are always inserted first.
//!   Inserting a superseder marks its target superseded (`sm-store`).
//! - **Status restore.** The status each target's line gives it is kept,
//!   whichever of the two lines comes first. After the superseder's batch
//!   is stored, a target the store has `superseded` while its line says
//!   another status gets the line's status back. The restore is its own
//!   write after the batch, so a run stopped between the two is put right
//!   by the next run over the same file: a target that is already stored
//!   is checked the same way. The restore only ever moves a target out of
//!   `superseded`, so it changes nothing on a repeat, and it leaves a target
//!   that a stored memory outside the file, not retracted, supersedes.
//! - A link line is stored with `Store::add_link`. A link whose memory end
//!   is missing is an issue. Its extractor and verified commit are redacted;
//!   a link with an anchor end the scanner matches is an issue and is not
//!   stored.
//! - Memories use `Access::Sync`: an import restores a store, including
//!   shared memories (and supersessions of them) that local access may not
//!   write.
//!
//! **Counting.** `read` counts memory lines, link lines, and lines that
//! could not be used (malformed lines, unknown kinds, project lines that
//! fail). Each ends in exactly one of `inserted`, `skipped_existing` (a
//! memory id or an identical link already stored, so an interrupted import
//! resumes) and `issues`. Valid project lines are not records: they are
//! counted only in `projects_created`. An issue's `index` is the 0-based
//! line number (the header is line 0).
//!
//! **Round trip.** Export, import into an empty store and export again
//! gives the same bytes apart from the header. Import keeps every line's
//! `updated_at`: a superseder's insert changes its target's status and not
//! its content, so the target's `updated_at` is not stamped then, nor when
//! its status is restored (decisions B-92 and B-93).

use std::collections::{HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sm_core::hash::content_hash;
use sm_core::ids::{MemoryId, ProjectId};
use sm_core::protocol::{
    ExportParams, ExportReport, ImportIssue, ImportParams, ImportReport, RememberOutcome,
};
use sm_core::time::Timestamp;
use sm_core::traits::{ChunkParams, Chunker, SecretScanner};
use sm_core::types::{
    AnchorRef, Link, LinkEnd, LinkSource, Memory, MemoryType, NewMemory, Origin, Project,
    Provenance, ProvenanceSource, Relation, Status,
};
use sm_core::{BUILD, CoreError, ErrorCode, Result};
use sm_store::{
    Access, InsertOptions, Inserted, MemoryInput, NewChunk, ProjectRecord, ReviseOptions, Revision,
    SCHEMA_VERSION, Store,
};

use crate::is_item_error;
use crate::redact::{holds_secret, redact_opt, redact_string, redact_value};

/// The header's `format`.
pub const EXPORT_FORMAT: &str = "spatial-memory-export";
/// The header's `version`: the newest file version this build reads.
pub const EXPORT_VERSION: u32 = 1;

/// Memories per transaction on import.
const BATCH: usize = 256;

/// Import stores each line's `updated_at` as it is: a superseder's insert
/// does not stamp its target (B-92, B-93).
const KEEP_UPDATED_AT: InsertOptions = InsertOptions {
    keep_target_updated_at: true,
};

// ---------------------------------------------------------------------------
// Line structs
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct Header<'a> {
    format: &'a str,
    version: u32,
    build: &'a str,
    exported_at: String,
    schema: u32,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase", deny_unknown_fields)]
enum Line {
    Project(ProjectLine),
    Memory(Box<MemoryLine>),
    Link(LinkLine),
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProjectLine {
    identity: String,
    name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    git_remote: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    root_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct MemoryLine {
    id: MemoryId,
    /// The project's identity key.
    project: String,
    #[serde(rename = "type")]
    memory_type: MemoryType,
    status: Status,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    title: Option<String>,
    content: String,
    content_hash: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    supersedes_id: Option<MemoryId>,
    origin: Origin,
    importance: f64,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    imported: Option<Value>,
    created_at: Timestamp,
    updated_at: Timestamp,
    provenance: ProvenanceLine,
    #[serde(default)]
    anchors: Vec<AnchorRef>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProvenanceLine {
    source: ProvenanceSource,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    author: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    commit_sha: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    file_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    extra: Option<Value>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LinkLine {
    /// The project's identity key.
    project: String,
    src: LinkEnd,
    dst: LinkEnd,
    relation: Relation,
    source: LinkSource,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    extractor: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    verified_commit: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    verified_at: Option<Timestamp>,
}

impl ProvenanceLine {
    fn of(p: &Provenance) -> Self {
        ProvenanceLine {
            source: p.source,
            author: p.author.clone(),
            session_id: p.session_id.clone(),
            commit_sha: p.commit_sha.clone(),
            file_path: p.file_path.clone(),
            tool: p.tool.clone(),
            extra: p.extra.clone(),
        }
    }

    fn into_provenance(self) -> Provenance {
        let mut p = Provenance::new(self.source);
        p.author = self.author;
        p.session_id = self.session_id;
        p.commit_sha = self.commit_sha;
        p.file_path = self.file_path;
        p.tool = self.tool;
        p.extra = self.extra;
        p
    }
}

impl MemoryLine {
    fn of(m: &Memory, project: &str) -> Self {
        MemoryLine {
            id: m.id,
            project: project.to_owned(),
            memory_type: m.memory_type,
            status: m.status,
            title: m.title.clone(),
            content: m.content.clone(),
            content_hash: m.content_hash.clone(),
            supersedes_id: m.supersedes_id,
            origin: m.origin,
            importance: m.importance,
            tags: m.tags.clone(),
            imported: m.imported.clone(),
            created_at: m.created_at,
            updated_at: m.updated_at,
            provenance: ProvenanceLine::of(&m.provenance),
            anchors: m.anchors.clone(),
        }
    }
}

impl LinkLine {
    fn of(l: &Link, project: &str) -> Self {
        LinkLine {
            project: project.to_owned(),
            src: l.src.clone(),
            dst: l.dst.clone(),
            relation: l.relation,
            source: l.source,
            extractor: l.extractor.clone(),
            verified_commit: l.verified_commit.clone(),
            verified_at: l.verified_at,
        }
    }

    fn into_link(self) -> Link {
        Link {
            src: self.src,
            dst: self.dst,
            relation: self.relation,
            source: self.source,
            extractor: self.extractor,
            verified_commit: self.verified_commit,
            verified_at: self.verified_at,
        }
    }
}

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------

/// Writes the store, or one project of it, to `params.path`; see the module
/// documentation.
pub fn export(store: &Store, params: &ExportParams) -> Result<ExportReport> {
    let target = Path::new(&params.path);
    refuse_existing(target)?;
    let records = store.project_records()?;
    let selected: Vec<(Project, String)> = match &params.project {
        None => records,
        Some(name) => {
            let matched: Vec<(Project, String)> = records
                .into_iter()
                .filter(|(p, _)| p.name == *name)
                .collect();
            match matched.len() {
                0 => {
                    return Err(
                        CoreError::not_found(format!("no project is named {name:?}"))
                            .with_next_action(
                                "Run `spatial-memory project list` to see the project names.",
                            ),
                    );
                }
                1 => matched,
                n => {
                    return Err(CoreError::invalid_input(format!(
                        "{n} projects are named {name:?}, so export cannot tell which one to write"
                    ))
                    .with_next_action(
                        "Rename all but one with `spatial-memory project rename`, or export \
                         every project.",
                    ));
                }
            }
        }
    };
    let keys: HashMap<ProjectId, String> =
        selected.iter().map(|(p, k)| (p.id, k.clone())).collect();

    let dir = match target.parent() {
        Some(d) if !d.as_os_str().is_empty() => d.to_path_buf(),
        _ => PathBuf::from("."),
    };
    let name = target
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .ok_or_else(|| {
            CoreError::invalid_input(format!("{} does not name a file", target.display()))
                .with_next_action("Pass a file path to export to.")
        })?;
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let tmp = dir.join(format!(".{name}.tmp-{}-{nanos}", std::process::id()));
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&tmp)
        .map_err(|e| write_error(target, &e.to_string()))?;
    let mut report = ExportReport {
        path: params.path.clone(),
        projects: 0,
        memories: 0,
        links: 0,
    };
    let written = write_lines(store, file, &selected, &keys, &mut report).and_then(|()| {
        refuse_existing(target)?;
        fs::rename(&tmp, target).map_err(|e| write_error(target, &e.to_string()))
    });
    if let Err(e) = written {
        let _ = fs::remove_file(&tmp);
        return Err(e);
    }
    Ok(report)
}

fn write_lines(
    store: &Store,
    file: File,
    selected: &[(Project, String)],
    keys: &HashMap<ProjectId, String>,
    report: &mut ExportReport,
) -> Result<()> {
    let mut w = BufWriter::new(file);
    let header = Header {
        format: EXPORT_FORMAT,
        version: EXPORT_VERSION,
        build: BUILD,
        exported_at: Timestamp::now().to_rfc3339(),
        schema: SCHEMA_VERSION,
    };
    write_line(&mut w, &header)?;
    for (p, key) in selected {
        write_line(
            &mut w,
            &Line::Project(ProjectLine {
                identity: key.clone(),
                name: p.name.clone(),
                git_remote: p.git_remote.clone(),
                root_path: p.root_path.clone(),
            }),
        )?;
        report.projects += 1;
    }
    for m in store.iter_memories() {
        let m = m?;
        let Some(key) = keys.get(&m.project_id) else {
            continue;
        };
        write_line(&mut w, &Line::Memory(Box::new(MemoryLine::of(&m, key))))?;
        report.memories += 1;
    }
    for (p, key) in selected {
        for l in store.project_links(p.id)? {
            write_line(&mut w, &Line::Link(LinkLine::of(&l, key)))?;
            report.links += 1;
        }
    }
    let file = w
        .into_inner()
        .map_err(|e| CoreError::storage(format!("could not write the export: {e}")))?;
    file.sync_all()
        .map_err(|e| CoreError::storage(format!("could not write the export: {e}")))?;
    Ok(())
}

fn write_line<T: Serialize>(w: &mut impl Write, value: &T) -> Result<()> {
    let mut bytes = serde_json::to_vec(value)
        .map_err(|e| CoreError::internal(format!("could not encode an export line: {e}")))?;
    bytes.push(b'\n');
    w.write_all(&bytes)
        .map_err(|e| CoreError::storage(format!("could not write the export: {e}")))
}

fn refuse_existing(target: &Path) -> Result<()> {
    match fs::symlink_metadata(target) {
        Ok(_) => Err(CoreError::invalid_input(format!(
            "{} already exists; export never replaces a file",
            target.display()
        ))
        .with_next_action("Choose a new file name, or move the old file away first.")),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(CoreError::storage(format!(
            "could not check {}: {e}",
            target.display()
        ))),
    }
}

fn write_error(target: &Path, e: &str) -> CoreError {
    CoreError::storage(format!("could not write {}: {e}", target.display()))
        .with_next_action("Check that the folder exists and can be written to.")
}

// ---------------------------------------------------------------------------
// Import
// ---------------------------------------------------------------------------

/// Reads an export written by [`export`] into the store; see the module
/// documentation.
pub fn import(
    store: &Store,
    chunker: &dyn Chunker,
    scanner: &dyn SecretScanner,
    chunk_params: &ChunkParams,
    params: &ImportParams,
) -> Result<ImportReport> {
    import_with(
        store,
        chunker,
        scanner,
        chunk_params,
        params,
        &ImportOptions::default(),
    )
}

/// Import options; the defaults are what [`import`] uses. Tests shrink the
/// batch and stop a run part way to check resumption.
#[derive(Debug, Clone)]
pub(crate) struct ImportOptions {
    pub batch_size: usize,
    /// Stop with an `internal` error once this many batches are stored.
    pub stop_after_batches: Option<usize>,
    /// Stop with an `internal` error once the first batch is stored and
    /// before any status is restored, as a crash between the two would.
    pub stop_before_restore: bool,
}

impl Default for ImportOptions {
    fn default() -> Self {
        ImportOptions {
            batch_size: BATCH,
            stop_after_batches: None,
            stop_before_restore: false,
        }
    }
}

pub(crate) fn import_with(
    store: &Store,
    chunker: &dyn Chunker,
    scanner: &dyn SecretScanner,
    chunk_params: &ChunkParams,
    params: &ImportParams,
    opts: &ImportOptions,
) -> Result<ImportReport> {
    let path = Path::new(&params.path);
    let Survey { ids, targets } = survey(path)?;
    let mut run = Import {
        store,
        chunker,
        scanner,
        chunk_params,
        opts,
        in_file: ids,
        targets,
        projects: HashMap::new(),
        queued: HashSet::new(),
        waiting: HashMap::new(),
        exported_status: HashMap::new(),
        created_targets: HashSet::new(),
        held_elsewhere: None,
        batch: Vec::new(),
        batches: 0,
        report: ImportReport {
            source: params.path.clone(),
            read: 0,
            inserted: 0,
            skipped_existing: 0,
            redacted: 0,
            projects_created: Vec::new(),
            issues: Vec::new(),
            pending_embeddings: 0,
        },
    };
    let mut lines = LineReader::open(path)?;
    lines.next_line()?; // the header, checked by `survey`
    while let Some((index, line)) = lines.next_line()? {
        run.line(index, line)?;
    }
    run.finish_memories()?;
    // A duplicate content hash is found when its batch is stored, after
    // later lines' issues; report every issue in file order.
    run.report.issues.sort_by_key(|i| i.index);
    Ok(run.report)
}

/// Lines of an export: `(0-based line number, text or a decoding error)`,
/// blank lines skipped.
struct LineReader {
    reader: BufReader<File>,
    index: u64,
    path: PathBuf,
}

impl LineReader {
    fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(|e| {
            CoreError::new(
                if e.kind() == std::io::ErrorKind::NotFound {
                    ErrorCode::NotFound
                } else {
                    ErrorCode::Storage
                },
                format!("could not open {}: {e}", path.display()),
            )
            .with_next_action("Check the path of the export file.")
        })?;
        Ok(LineReader {
            reader: BufReader::new(file),
            index: 0,
            path: path.to_path_buf(),
        })
    }

    fn next_line(&mut self) -> Result<Option<(u64, std::result::Result<String, String>)>> {
        loop {
            let mut buf = Vec::new();
            let n = self.reader.read_until(b'\n', &mut buf).map_err(|e| {
                CoreError::storage(format!("could not read {}: {e}", self.path.display()))
            })?;
            if n == 0 {
                return Ok(None);
            }
            let index = self.index;
            self.index += 1;
            if buf.last() == Some(&b'\n') {
                buf.pop();
            }
            if buf.last() == Some(&b'\r') {
                buf.pop();
            }
            if buf.iter().all(u8::is_ascii_whitespace) {
                continue;
            }
            let text = String::from_utf8(buf).map_err(|_| "the line is not UTF-8".to_owned());
            return Ok(Some((index, text)));
        }
    }
}

/// What the first pass finds in a file.
struct Survey {
    /// Every memory id with a line.
    ids: HashSet<MemoryId>,
    /// Every `supersedes_id` a memory line names.
    targets: HashSet<MemoryId>,
}

/// First pass: checks the header and collects the memory ids and the
/// supersession targets in the file.
fn survey(path: &Path) -> Result<Survey> {
    #[derive(Deserialize)]
    struct Peek {
        memory: Option<PeekMemory>,
    }
    #[derive(Deserialize)]
    struct PeekMemory {
        id: MemoryId,
        #[serde(default)]
        supersedes_id: Option<MemoryId>,
    }
    let mut lines = LineReader::open(path)?;
    let not_export = |why: &str| {
        CoreError::invalid_input(format!(
            "{} is not a Spatial Memory export: {why}; nothing was imported",
            path.display()
        ))
        .with_next_action(
            "Pass a file written by `spatial-memory export`. A version 1 or 2 JSON export \
             is read by `spatial-memory migrate`.",
        )
    };
    let header: Value = match lines.next_line()? {
        Some((_, Ok(text))) => {
            serde_json::from_str(&text).map_err(|e| not_export(&format!("line 1: {e}")))?
        }
        Some((_, Err(e))) => return Err(not_export(&format!("line 1: {e}"))),
        None => return Err(not_export("the file is empty")),
    };
    if header.get("format").and_then(Value::as_str) != Some(EXPORT_FORMAT) {
        return Err(not_export(&format!(
            "the first line's format is {}, not {EXPORT_FORMAT:?}",
            header
                .get("format")
                .map_or("missing".to_owned(), Value::to_string)
        )));
    }
    match header.get("version").and_then(Value::as_u64) {
        Some(v) if v <= u64::from(EXPORT_VERSION) && v >= 1 => {}
        Some(v) => {
            return Err(CoreError::invalid_input(format!(
                "{} is export version {v}; this build reads version {EXPORT_VERSION} and \
                 older; nothing was imported",
                path.display()
            ))
            .with_next_action("Import it with the newer Spatial Memory build that wrote it."));
        }
        None => return Err(not_export("the header has no version number")),
    }
    let mut found = Survey {
        ids: HashSet::new(),
        targets: HashSet::new(),
    };
    while let Some((_, line)) = lines.next_line()? {
        if let Ok(text) = line
            && let Ok(Peek {
                memory: Some(PeekMemory { id, supersedes_id }),
            }) = serde_json::from_str::<Peek>(&text)
        {
            found.ids.insert(id);
            found.targets.extend(supersedes_id);
        }
    }
    Ok(found)
}

struct Import<'a> {
    store: &'a Store,
    chunker: &'a dyn Chunker,
    scanner: &'a dyn SecretScanner,
    chunk_params: &'a ChunkParams,
    opts: &'a ImportOptions,
    /// Memory ids with a line in the file.
    in_file: HashSet<MemoryId>,
    /// Memory ids some line of the file supersedes.
    targets: HashSet<MemoryId>,
    /// Identity key → project id.
    projects: HashMap<String, ProjectId>,
    /// Memories queued for insertion (or already handled) in this run.
    queued: HashSet<MemoryId>,
    /// Memories waiting for the memory they supersede to be queued.
    waiting: HashMap<MemoryId, Vec<(u64, MemoryLine)>>,
    /// The status each supersession target's line gives it.
    exported_status: HashMap<MemoryId, Status>,
    /// Targets this run created (not found already stored).
    created_targets: HashSet<MemoryId>,
    /// Targets that a stored memory outside the file, not retracted,
    /// supersedes; read from the store the first time it is needed.
    held_elsewhere: Option<HashSet<MemoryId>>,
    batch: Vec<Queued>,
    batches: usize,
    report: ImportReport,
}

struct Queued {
    index: u64,
    id: MemoryId,
    redacted: bool,
    input: MemoryInput,
}

impl Import<'_> {
    fn issue(&mut self, index: u64, id: Option<String>, message: impl Into<String>) {
        self.report.issues.push(ImportIssue {
            index,
            id,
            message: message.into(),
        });
    }

    fn line(&mut self, index: u64, line: std::result::Result<String, String>) -> Result<()> {
        let text = match line {
            Ok(t) => t,
            Err(why) => {
                self.report.read += 1;
                self.issue(index, None, format!("{why}; the line was skipped"));
                return Ok(());
            }
        };
        let parsed = match serde_json::from_str::<Line>(&text) {
            Ok(l) => l,
            Err(e) => {
                self.report.read += 1;
                self.issue(
                    index,
                    line_id(&text),
                    format!("the line does not parse ({e}); it was skipped"),
                );
                return Ok(());
            }
        };
        match parsed {
            Line::Project(p) => self.project(index, p),
            Line::Memory(m) => {
                self.report.read += 1;
                self.memory(index, *m)
            }
            Line::Link(l) => {
                self.report.read += 1;
                self.link(index, l)
            }
        }
    }

    fn project(&mut self, index: u64, p: ProjectLine) -> Result<()> {
        let rec = ProjectRecord {
            identity: p.identity.clone(),
            name: p.name,
            git_remote: p.git_remote,
            root_path: p.root_path,
        };
        match self.store.ensure_project(&rec) {
            Ok((project, created)) => {
                if created {
                    self.report.projects_created.push(project.name.clone());
                }
                self.projects.insert(p.identity, project.id);
                Ok(())
            }
            Err(e) if is_item_error(&e) => {
                self.report.read += 1;
                self.issue(
                    index,
                    Some(p.identity),
                    format!("the project could not be recorded: {}", e.message),
                );
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    fn project_id(&mut self, key: &str) -> Result<Option<ProjectId>> {
        if let Some(id) = self.projects.get(key) {
            return Ok(Some(*id));
        }
        let found = self.store.project_by_identity(key)?.map(|p| p.id);
        if let Some(id) = found {
            self.projects.insert(key.to_owned(), id);
        }
        Ok(found)
    }

    fn memory(&mut self, index: u64, m: MemoryLine) -> Result<()> {
        if !self.queued.insert(m.id) {
            self.issue(
                index,
                Some(m.id.to_string()),
                "this memory id appears earlier in the file; only the first line with it was used",
            );
            return Ok(());
        }
        if let Some(target) = m.supersedes_id
            && self.in_file.contains(&target)
            && !self.queued.contains(&target)
        {
            self.queued.remove(&m.id);
            self.waiting.entry(target).or_default().push((index, m));
            return Ok(());
        }
        self.queue(index, m)
    }

    /// Maps one memory line into the batch, then the memories that were
    /// waiting for it.
    fn queue(&mut self, index: u64, m: MemoryLine) -> Result<()> {
        let mut ready = vec![(index, m)];
        while let Some((index, m)) = ready.pop() {
            self.queued.insert(m.id);
            let id = m.id;
            if self.targets.contains(&id) {
                self.exported_status.insert(id, m.status);
            }
            match self.map_memory(m)? {
                Ok(q) => {
                    let q = Queued { index, ..q };
                    self.batch.push(q);
                    if self.batch.len() >= self.opts.batch_size.max(1) {
                        self.flush()?;
                    }
                }
                Err(message) => self.issue(index, Some(id.to_string()), message),
            }
            if let Some(mut next) = self.waiting.remove(&id) {
                next.reverse();
                ready.extend(next);
            }
        }
        Ok(())
    }

    /// The memory as a store input, or why it cannot be stored.
    fn map_memory(&mut self, m: MemoryLine) -> Result<std::result::Result<Queued, String>> {
        let Some(project_id) = self.project_id(&m.project)? else {
            return Ok(Err(format!(
                "its project {:?} has no project line and is not in the store; the memory \
                 was not imported",
                m.project
            )));
        };
        let scanner = self.scanner;
        let mut content = m.content;
        let mut title = m.title;
        let mut provenance = m.provenance;
        let mut imported = m.imported;
        let mut redacted = redact_string(scanner, &mut content);
        redacted |= redact_opt(scanner, &mut title);
        for s in [
            &mut provenance.author,
            &mut provenance.session_id,
            &mut provenance.commit_sha,
            &mut provenance.file_path,
            &mut provenance.tool,
        ] {
            redacted |= redact_opt(scanner, s);
        }
        if let Some(extra) = provenance.extra.as_mut() {
            redacted |= redact_value(scanner, extra);
        }
        if let Some(v) = imported.as_mut() {
            redacted |= redact_value(scanner, v);
        }
        // A tag or anchor cannot hold a `[REDACTED:<kind>]` marker, so one
        // that holds a secret is dropped and the memory counts as redacted.
        let (tag_count, anchor_count) = (m.tags.len(), m.anchors.len());
        let tags: Vec<String> = m
            .tags
            .into_iter()
            .filter(|t| !holds_secret(scanner, t))
            .collect();
        let anchors: Vec<AnchorRef> = m
            .anchors
            .into_iter()
            .filter(|a| !anchor_holds_secret(scanner, a))
            .collect();
        redacted |= tags.len() != tag_count || anchors.len() != anchor_count;
        let plan = match self.chunker.chunk(&content, self.chunk_params) {
            Ok(p) => p,
            Err(e) => {
                return Ok(Err(format!(
                    "the content could not be chunked ({}); the memory was not imported",
                    e.message
                )));
            }
        };
        let chunks = plan.pieces.into_iter().map(NewChunk::pending).collect();
        let hash = content_hash(&content);
        let mut memory = NewMemory::new(project_id, content, provenance.into_provenance());
        memory.id = Some(m.id);
        memory.memory_type = m.memory_type;
        memory.status = m.status;
        memory.title = title;
        memory.content_hash = Some(hash);
        memory.supersedes_id = m.supersedes_id;
        memory.origin = m.origin;
        memory.importance = m.importance;
        memory.tags = tags;
        memory.imported = imported;
        memory.created_at = Some(m.created_at);
        memory.updated_at = Some(m.updated_at);
        memory.anchors = anchors;
        Ok(Ok(Queued {
            index: 0,
            id: m.id,
            redacted,
            input: MemoryInput { memory, chunks },
        }))
    }

    fn flush(&mut self) -> Result<()> {
        if self.batch.is_empty() {
            return Ok(());
        }
        let batch = std::mem::take(&mut self.batch);
        let inputs: Vec<MemoryInput> = batch.iter().map(|q| q.input.clone()).collect();
        match self
            .store
            .insert_memories_with(inputs, Access::Sync, KEEP_UPDATED_AT)
        {
            Ok(results) => {
                if self.opts.stop_before_restore {
                    return Err(CoreError::internal(
                        "import stopped after storing a batch and before restoring statuses, \
                         as asked",
                    ));
                }
                for (q, r) in batch.iter().zip(results) {
                    self.count(q, r)?;
                }
            }
            Err(e) if is_item_error(&e) => {
                for q in &batch {
                    match self.insert_one(q.input.clone()) {
                        Ok(r) => self.count(q, r)?,
                        Err(e) if is_item_error(&e) => self.issue(
                            q.index,
                            Some(q.id.to_string()),
                            format!("the memory could not be stored: {}", e.message),
                        ),
                        Err(e) => return Err(e),
                    }
                }
            }
            Err(e) => return Err(e),
        }
        self.batches += 1;
        if let Some(stop) = self.opts.stop_after_batches
            && self.batches >= stop
        {
            return Err(CoreError::internal(format!(
                "import stopped after {} batches as asked",
                self.batches
            )));
        }
        Ok(())
    }

    /// Stores one memory as [`Import::flush`] stores a batch.
    fn insert_one(&self, input: MemoryInput) -> Result<Inserted> {
        self.store
            .insert_memories_with(vec![input], Access::Sync, KEEP_UPDATED_AT)?
            .pop()
            .ok_or_else(|| CoreError::internal("the store returned no result for a memory"))
    }

    fn count(&mut self, q: &Queued, r: Inserted) -> Result<()> {
        match r.outcome {
            RememberOutcome::Created => {
                self.report.inserted += 1;
                self.report.pending_embeddings += u64::from(r.chunks);
                if q.redacted {
                    self.report.redacted += 1;
                }
                if self.targets.contains(&q.id) {
                    self.created_targets.insert(q.id);
                }
                if let Some(target) = r.superseded {
                    self.restore_status(q.index, target)?;
                }
            }
            RememberOutcome::Duplicate if r.id == q.id => {
                self.report.skipped_existing += 1;
                // A run stopped between storing a superseder and restoring
                // its target leaves the target superseded; this run puts
                // that right.
                if self.targets.contains(&q.id) {
                    self.restore_status(q.index, q.id)?;
                }
            }
            RememberOutcome::Duplicate => self.issue(
                q.index,
                Some(q.id.to_string()),
                format!(
                    "its content is already stored as memory {} in the same project; the \
                     memory was not imported",
                    r.id
                ),
            ),
        }
        Ok(())
    }

    /// Gives a supersession target back the status its line had, when the
    /// store has it `superseded` and the line says otherwise. Idempotent:
    /// a target already stored with another status is left alone, so a run
    /// repeated over the same file changes nothing. A target this run did
    /// not create keeps `superseded` while a stored memory outside the file
    /// that is not retracted supersedes it, and a target that consolidation
    /// merged into another ([`crate::is_merged`]) always keeps it.
    fn restore_status(&mut self, index: u64, target: MemoryId) -> Result<()> {
        let Some(status) = self.exported_status.get(&target).copied() else {
            return Ok(());
        };
        if status == Status::Superseded {
            return Ok(());
        }
        // A memory that consolidation merged into another keeps `superseded`
        // whatever its line says: no superseder holds it, the merge does.
        match self.store.find_memory(&target)? {
            Some(stored) if stored.status == Status::Superseded && !crate::is_merged(&stored) => {}
            _ => return Ok(()),
        }
        if !self.created_targets.contains(&target) && self.held_elsewhere()?.contains(&target) {
            return Ok(());
        }
        let rev = Revision {
            status: Some(status),
            ..Revision::default()
        };
        let keep = ReviseOptions {
            keep_updated_at: true,
            ..ReviseOptions::default()
        };
        match self
            .store
            .revise_memory_with(&target, rev, Access::Sync, &keep)
        {
            Ok(_) => Ok(()),
            Err(e) if is_item_error(&e) => {
                self.issue(
                    index,
                    Some(target.to_string()),
                    format!(
                        "memory {target} was marked superseded and its status {status} could \
                         not be restored: {}",
                        e.message
                    ),
                );
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    /// The file's targets that a stored memory without a line in the file,
    /// and not retracted, supersedes. Read once, the first time a target
    /// that was already stored needs its status back.
    fn held_elsewhere(&mut self) -> Result<&HashSet<MemoryId>> {
        if self.held_elsewhere.is_none() {
            let mut held = HashSet::new();
            for m in self.store.iter_memories() {
                let m = m?;
                if let Some(t) = m.supersedes_id
                    && m.status != Status::Retracted
                    && self.targets.contains(&t)
                    && !self.in_file.contains(&m.id)
                {
                    held.insert(t);
                }
            }
            self.held_elsewhere = Some(held);
        }
        Ok(self.held_elsewhere.get_or_insert_with(HashSet::new))
    }

    /// Queues the memories still waiting (their target's line was missing
    /// or failed; the store reports a target that does not exist) and
    /// stores the last batch.
    fn finish_memories(&mut self) -> Result<()> {
        let mut rest: Vec<(u64, MemoryLine)> = self.waiting.drain().flat_map(|(_, v)| v).collect();
        rest.sort_by_key(|(index, _)| *index);
        for (index, m) in rest {
            if self.queued.contains(&m.id) {
                continue;
            }
            self.queue(index, m)?;
        }
        self.flush()
    }

    fn link(&mut self, index: u64, mut l: LinkLine) -> Result<()> {
        self.finish_memories()?;
        let scanner = self.scanner;
        let shown = scanner.redact(&format!("{} -> {}", l.src, l.dst)).text;
        if [&l.src, &l.dst].into_iter().any(|end| match end {
            LinkEnd::Anchor(a) => anchor_holds_secret(scanner, a),
            LinkEnd::Memory(_) => false,
        }) {
            self.issue(
                index,
                Some(shown),
                "an anchor end of the link holds a secret, which an anchor cannot keep redacted; \
                 the link was not imported",
            );
            return Ok(());
        }
        let redacted =
            redact_opt(scanner, &mut l.extractor) | redact_opt(scanner, &mut l.verified_commit);
        let Some(project) = self.project_id(&l.project)? else {
            self.issue(
                index,
                Some(shown),
                format!(
                    "its project {:?} has no project line and is not in the store; the link \
                     was not imported",
                    l.project
                ),
            );
            return Ok(());
        };
        match self.store.add_link(project, l.into_link()) {
            Ok(added) if added.created => {
                self.report.inserted += 1;
                if redacted {
                    self.report.redacted += 1;
                }
            }
            Ok(_) => self.report.skipped_existing += 1,
            Err(e) if e.code == ErrorCode::NotFound => self.issue(
                index,
                Some(shown),
                format!(
                    "an end of the link is missing ({}); the link was skipped",
                    e.message
                ),
            ),
            Err(e) if e.code == ErrorCode::InvalidInput => self.issue(
                index,
                Some(shown),
                format!(
                    "the link could not be stored ({}); it was skipped",
                    e.message
                ),
            ),
            Err(e) => return Err(e),
        }
        Ok(())
    }
}

/// True when the scanner finds a secret in an anchor's name, alone or
/// written as `kind:name`.
fn anchor_holds_secret(scanner: &dyn SecretScanner, a: &AnchorRef) -> bool {
    holds_secret(scanner, &a.name) || holds_secret(scanner, &a.to_string())
}

/// The memory id a malformed line names, when it can be found.
fn line_id(text: &str) -> Option<String> {
    let v: Value = serde_json::from_str(text).ok()?;
    v.get("memory")?.get("id")?.as_str().map(str::to_owned)
}
