//! `migrate`: imports a version 1 or 2 JSON export (SPEC section 10; AT-8;
//! decisions B-12, B-14, B-18, B-35 and B-79).
//!
//! The export is one JSON array of records, as version 1's `export_memories`
//! writes it. The file is streamed twice. No record's `vector` is ever
//! built (serde skips it), and at most one batch of mapped records is held
//! at a time.
//!
//! 1. **Survey.** The first pass checks the JSON and collects each record's
//!    `project`. Every value is then matched to a project (B-79), and the
//!    fallback for records with no project is resolved (B-35). A syntax
//!    error, two projects with one name, or records with no project and no
//!    fallback fail here, before anything is written.
//! 2. **Import.** The second pass creates the projects that do not exist
//!    yet, maps each record, and inserts the records in batches. Each batch
//!    is one transaction, so a stopped run leaves whole batches behind and
//!    the next run resumes: every id already stored counts as
//!    `skipped_existing`.
//!
//! **Project matching** for a record's trimmed `project` value (B-79):
//! 1. the project with identity `git:<value>`;
//! 2. else the one project named `<value>` (two such projects fail the run
//!    with `invalid_input`);
//! 3. else the working directory's project, when its name is `<value>`
//!    (created by its own identity, so opening the repository finds it, as
//!    B-79's gap note expects);
//! 4. else a new project: `git:<value>` with that remote when the value is a
//!    normalised remote (`host/path`), named like a repository's project
//!    (the last path segment); otherwise `name:<value>` with no remote.
//!
//! A record with no `project` (missing, null or blank) goes to
//! `default_project`, matched the same way, else to the working
//! directory's project, else the run fails with `project_unresolved`.
//!
//! **Field mapping** (SPEC 10):
//! - `id`: kept when it is a UUID, else the B-12 version 8 id.
//! - `content`: kept, except that secrets are redacted (B-64; the capture
//!   skip rule does not apply). `content_hash`: the version 1 value is kept
//!   when it is the B-14 hash of the stored content; otherwise the hash is
//!   recomputed and the version 1 value goes to `imported.content_hash`.
//! - `created_at`, `updated_at`: kept to the microsecond; a time with no
//!   offset is UTC (B-18). A missing value is left to the store (now, and
//!   `updated_at` = `created_at`); one that does not parse goes to
//!   `imported` under its own name instead.
//! - `namespace`: a tag, unless it is empty or equals the project value or
//!   the project's name. `tags`: kept. A tag that breaks the tag rules is
//!   cleaned (`sanitize_tag`) or, when nothing is left, left out; the
//!   original list then goes to `imported.tags` (`imported.namespace` for
//!   the namespace). A tag (or the namespace) that holds a secret is left
//!   out the same way, since a tag cannot hold a redaction marker, and the
//!   record counts in `redacted`.
//! - `source`, `metadata`: `provenance.extra.source` and `.metadata`, in
//!   whatever JSON form they come (an object, a string, a string holding
//!   JSON).
//! - `importance`, `access_count`, `last_accessed`, `expires_at`, and any
//!   other field this module does not know: `imported`, under their own
//!   names. `vector`: dropped.
//! - Always: type `note`, status `active`, origin `local`, provenance source
//!   `import`, the default importance, no title (B-18) and no anchors.
//!
//! **Redaction** (SPEC 9 "Secret redaction on write"; the lead's decision
//! in the stream F report). Every string `migrate` stores is scanned:
//! content, and every string value of `provenance.extra` and `imported` at
//! any depth, unknown fields included (see `crate::redact`). Values are
//! redacted, keys are not.
//!
//! **Counting.** `read` is the number of array elements, and each ends in
//! exactly one of `inserted`, `skipped_existing` and `issues`, so `read =
//! inserted + skipped_existing + issues.len()` (AT-8). An element that
//! cannot be stored (not an object, no id, empty content, a content hash
//! another memory of the project holds, an id seen earlier in the file) is
//! an issue; no record is dropped silently. `redacted` counts inserted
//! memories that had a secret redacted, and `pending_embeddings` the chunks
//! they queued.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use serde::de::{self, DeserializeSeed, IgnoredAny, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::{Map, Value};
use sm_core::hash::content_hash;
use sm_core::ids::{MemoryId, ProjectId};
use sm_core::project::{ProjectIdentity, normalize_remote_url};
use sm_core::protocol::{ImportIssue, ImportReport, MigrateParams, RememberOutcome};
use sm_core::time::Timestamp;
use sm_core::traits::{ChunkParams, Chunker, SecretScanner};
use sm_core::types::{
    DEFAULT_IMPORTANCE, MemoryType, NewMemory, Origin, Project, Provenance, ProvenanceSource,
    Status, sanitize_tag,
};
use sm_core::{CoreError, ErrorCode, Result};
use sm_store::{Access, Inserted, MemoryInput, NewChunk, ProjectRecord, Store};

use crate::is_item_error;
use crate::redact::{holds_secret, redact_string, redact_value};

/// Records per transaction.
const BATCH: usize = 256;

/// Fields that are mapped; every other field except `vector` goes to
/// `imported`.
const MAPPED: [&str; 9] = [
    "id",
    "content",
    "content_hash",
    "created_at",
    "updated_at",
    "namespace",
    "tags",
    "source",
    "metadata",
];

/// Run options; the defaults are what [`migrate`] uses. Tests shrink the
/// batch and stop a run part way to check resumption.
#[derive(Debug, Clone)]
pub(crate) struct MigrateOptions {
    pub batch_size: usize,
    /// Stop with an `internal` error once this many batches are stored.
    pub stop_after_batches: Option<usize>,
}

impl Default for MigrateOptions {
    fn default() -> Self {
        MigrateOptions {
            batch_size: BATCH,
            stop_after_batches: None,
        }
    }
}

/// Imports the version 1 or 2 JSON export at `params.from`; see the module
/// documentation. `cwd_project` is the project of the request's working
/// directory, when there is one. Fails before writing anything when the
/// file is not a JSON array, when two projects share a name the file uses,
/// or when records have no project and there is no fallback.
pub fn migrate(
    store: &Store,
    chunker: &dyn Chunker,
    scanner: &dyn SecretScanner,
    chunk_params: &ChunkParams,
    params: &MigrateParams,
    cwd_project: Option<&ProjectIdentity>,
) -> Result<ImportReport> {
    migrate_with(
        store,
        chunker,
        scanner,
        chunk_params,
        params,
        cwd_project,
        &MigrateOptions::default(),
    )
}

pub(crate) fn migrate_with(
    store: &Store,
    chunker: &dyn Chunker,
    scanner: &dyn SecretScanner,
    chunk_params: &ChunkParams,
    params: &MigrateParams,
    cwd_project: Option<&ProjectIdentity>,
    opts: &MigrateOptions,
) -> Result<ImportReport> {
    let path = Path::new(&params.from);
    let mut survey = Survey::default();
    read_records(path, &mut |_, el| {
        survey.note(&el);
        Ok(())
    })?;
    let plan = Plan::resolve(
        store,
        &survey,
        params.default_project.as_deref(),
        cwd_project,
    )?;
    let mut run = Run {
        store,
        chunker,
        scanner,
        chunk_params,
        opts,
        projects: HashMap::new(),
        fallback: None,
        seen: HashSet::new(),
        batch: Vec::new(),
        batches: 0,
        report: ImportReport {
            source: params.from.clone(),
            read: 0,
            inserted: 0,
            skipped_existing: 0,
            redacted: 0,
            projects_created: Vec::new(),
            issues: Vec::new(),
            pending_embeddings: 0,
        },
    };
    run.create_projects(plan)?;
    let read = read_records(path, &mut |index, el| run.record(index, el))?;
    run.flush()?;
    run.report.read = read;
    // A duplicate content hash is found when its batch is stored, after
    // later records' issues; report every issue in file order.
    run.report.issues.sort_by_key(|i| i.index);
    Ok(run.report)
}

// ---------------------------------------------------------------------------
// Streaming the array
// ---------------------------------------------------------------------------

/// One array element: a record with every field but `vector`, or what the
/// element was instead of an object.
pub(crate) enum Element {
    Record(Map<String, Value>),
    Other(&'static str),
}

impl<'de> Deserialize<'de> for Element {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        d.deserialize_any(ElementVisitor)
    }
}

struct ElementVisitor;

impl<'de> Visitor<'de> for ElementVisitor {
    type Value = Element;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a memory record")
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> std::result::Result<Element, A::Error> {
        let mut out = Map::new();
        while let Some(key) = map.next_key::<String>()? {
            if key == "vector" {
                map.next_value::<IgnoredAny>()?;
            } else {
                let value = map.next_value::<Value>()?;
                out.insert(key, value);
            }
        }
        Ok(Element::Record(out))
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> std::result::Result<Element, A::Error> {
        while seq.next_element::<IgnoredAny>()?.is_some() {}
        Ok(Element::Other("an array"))
    }

    fn visit_bool<E: de::Error>(self, _: bool) -> std::result::Result<Element, E> {
        Ok(Element::Other("a boolean"))
    }

    fn visit_i64<E: de::Error>(self, _: i64) -> std::result::Result<Element, E> {
        Ok(Element::Other("a number"))
    }

    fn visit_u64<E: de::Error>(self, _: u64) -> std::result::Result<Element, E> {
        Ok(Element::Other("a number"))
    }

    fn visit_f64<E: de::Error>(self, _: f64) -> std::result::Result<Element, E> {
        Ok(Element::Other("a number"))
    }

    fn visit_str<E: de::Error>(self, _: &str) -> std::result::Result<Element, E> {
        Ok(Element::Other("a string"))
    }

    fn visit_unit<E: de::Error>(self) -> std::result::Result<Element, E> {
        Ok(Element::Other("null"))
    }
}

type OnElement<'f> = &'f mut dyn FnMut(u64, Element) -> Result<()>;

struct ArraySeed<'f, 'g> {
    on_element: OnElement<'f>,
    failure: &'g mut Option<CoreError>,
}

impl<'de> DeserializeSeed<'de> for ArraySeed<'_, '_> {
    type Value = u64;

    fn deserialize<D: Deserializer<'de>>(self, d: D) -> std::result::Result<u64, D::Error> {
        d.deserialize_seq(self)
    }
}

impl<'de> Visitor<'de> for ArraySeed<'_, '_> {
    type Value = u64;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a JSON array of memory records")
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> std::result::Result<u64, A::Error> {
        let ArraySeed {
            on_element,
            failure,
        } = self;
        let mut index = 0u64;
        while let Some(el) = seq.next_element::<Element>()? {
            if let Err(e) = on_element(index, el) {
                *failure = Some(e);
                return Err(de::Error::custom("stopped"));
            }
            index += 1;
        }
        Ok(index)
    }
}

/// Streams the array at `path`, calling `on_element` with each element's
/// 0-based position, and returns the element count. An error from
/// `on_element` stops the read and is returned as is.
fn read_records(path: &Path, on_element: OnElement<'_>) -> Result<u64> {
    let file = File::open(path).map_err(|e| {
        CoreError::new(
            if e.kind() == std::io::ErrorKind::NotFound {
                ErrorCode::NotFound
            } else {
                ErrorCode::Storage
            },
            format!("could not open {}: {e}", path.display()),
        )
        .with_next_action("Check the path passed to --from.")
    })?;
    let mut de = serde_json::Deserializer::from_reader(BufReader::new(file));
    let mut failure = None;
    let result = ArraySeed {
        on_element,
        failure: &mut failure,
    }
    .deserialize(&mut de);
    match result {
        Ok(n) => {
            de.end().map_err(|e| bad_json(path, &e))?;
            Ok(n)
        }
        Err(e) => Err(failure.unwrap_or_else(|| bad_json(path, &e))),
    }
}

fn bad_json(path: &Path, e: &serde_json::Error) -> CoreError {
    CoreError::invalid_input(format!(
        "{} is not a version 1 or 2 JSON export (a JSON array of memory records): {e}",
        path.display()
    ))
    .with_next_action(
        "Export again from the old install with export_memories, format json and \
         include_vectors off. Nothing was imported.",
    )
}

// ---------------------------------------------------------------------------
// Survey and project plan
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Survey {
    /// Distinct trimmed `project` values, first seen first.
    values: Vec<String>,
    known: HashSet<String>,
    /// Some record has no project.
    missing: bool,
}

impl Survey {
    fn note(&mut self, el: &Element) {
        let Element::Record(rec) = el else {
            return;
        };
        match project_value(rec) {
            Ok(Some(v)) => {
                if self.known.insert(v.to_owned()) {
                    self.values.push(v.to_owned());
                }
            }
            Ok(None) => self.missing = true,
            Err(_) => {}
        }
    }
}

/// The record's trimmed project value: `None` when it is missing, null or
/// blank, an error message when it is not a string.
fn project_value(rec: &Map<String, Value>) -> std::result::Result<Option<&str>, String> {
    match rec.get("project") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) if s.trim().is_empty() => Ok(None),
        Some(Value::String(s)) => Ok(Some(s.trim())),
        Some(other) => Err(format!(
            "project is {}, not a string; the record was not imported",
            kind_of(other)
        )),
    }
}

/// Where a project value lands.
#[derive(Debug, Clone)]
enum Target {
    Existing(Project),
    Create(ProjectRecord),
}

struct Plan {
    /// By project value, in the order the file first uses them.
    values: Vec<(String, Target)>,
    fallback: Option<Target>,
}

impl Plan {
    fn resolve(
        store: &Store,
        survey: &Survey,
        default_project: Option<&str>,
        cwd: Option<&ProjectIdentity>,
    ) -> Result<Plan> {
        let mut values = Vec::with_capacity(survey.values.len());
        for v in &survey.values {
            values.push((v.clone(), match_project(store, v, cwd)?));
        }
        let fallback = if survey.missing {
            let default = default_project.map(str::trim).filter(|d| !d.is_empty());
            Some(match (default, cwd) {
                (Some(d), _) => match_project(store, d, cwd)?,
                (None, Some(id)) => Target::Create(record_of_identity(id)),
                (None, None) => {
                    return Err(CoreError::new(
                        ErrorCode::ProjectUnresolved,
                        "some records have no project, no --default-project was given, \
                         and the working directory is not in a project; nothing was imported",
                    )
                    .with_next_action(
                        "Pass --default-project <name>, or run migrate inside the \
                         project's repository.",
                    ));
                }
            })
        } else {
            None
        };
        Ok(Plan { values, fallback })
    }
}

/// B-79 matching for one project value; see the module documentation.
fn match_project(store: &Store, value: &str, cwd: Option<&ProjectIdentity>) -> Result<Target> {
    if let Some(p) = store.project_by_identity(&format!("git:{value}"))? {
        return Ok(Target::Existing(p));
    }
    let mut named = store.projects_by_name(value)?;
    match named.len() {
        0 => {}
        1 => return Ok(Target::Existing(named.remove(0))),
        n => {
            return Err(CoreError::invalid_input(format!(
                "{n} projects are named {value:?}, so migrate cannot tell which one the \
                 records with that project belong to; nothing was imported"
            ))
            .with_next_action(
                "Rename all but one with `spatial-memory project rename`, then run migrate again.",
            ));
        }
    }
    if let Some(id) = cwd
        && id.name == value
    {
        return Ok(Target::Create(record_of_identity(id)));
    }
    if is_normalised_remote(value) {
        let name = value.rsplit('/').next().unwrap_or(value).to_owned();
        return Ok(Target::Create(ProjectRecord {
            identity: format!("git:{value}"),
            name,
            git_remote: Some(value.to_owned()),
            root_path: None,
        }));
    }
    Ok(Target::Create(ProjectRecord {
        identity: format!("name:{value}"),
        name: value.to_owned(),
        git_remote: None,
        root_path: None,
    }))
}

fn record_of_identity(id: &ProjectIdentity) -> ProjectRecord {
    ProjectRecord {
        identity: id.key(),
        name: id.name.clone(),
        git_remote: id.git_remote.clone(),
        root_path: Some(id.root_path.clone()),
    }
}

/// `host/path` as version 1 stored remotes (`normalize_remote_url`): a
/// lower-case host with a dot (or `localhost`), at least one path segment,
/// and nothing a normalisation would change.
fn is_normalised_remote(value: &str) -> bool {
    let Some((host, path)) = value.split_once('/') else {
        return false;
    };
    if path.is_empty()
        || value.contains(char::is_whitespace)
        || value.contains(':')
        || value.contains('@')
        || !(host.contains('.') || host == "localhost")
        || host.starts_with('.')
        || host.ends_with('.')
    {
        return false;
    }
    normalize_remote_url(&format!("https://{value}")).as_deref() == Some(value)
}

// ---------------------------------------------------------------------------
// The import pass
// ---------------------------------------------------------------------------

struct Run<'a> {
    store: &'a Store,
    chunker: &'a dyn Chunker,
    scanner: &'a dyn SecretScanner,
    chunk_params: &'a ChunkParams,
    opts: &'a MigrateOptions,
    /// Project value → (id, name).
    projects: HashMap<String, (ProjectId, String)>,
    fallback: Option<(ProjectId, String)>,
    /// Ids mapped so far in this run, to report a repeated id.
    seen: HashSet<MemoryId>,
    batch: Vec<Pending>,
    batches: usize,
    report: ImportReport,
}

/// A mapped record waiting for its batch.
struct Pending {
    index: u64,
    id: MemoryId,
    /// The id as the file writes it, for issues.
    raw_id: String,
    redacted: bool,
    input: MemoryInput,
}

impl Run<'_> {
    fn create_projects(&mut self, plan: Plan) -> Result<()> {
        for (value, target) in plan.values {
            let p = self.realise(target)?;
            self.projects.insert(value, (p.id, p.name));
        }
        if let Some(target) = plan.fallback {
            let p = self.realise(target)?;
            self.fallback = Some((p.id, p.name));
        }
        Ok(())
    }

    fn realise(&mut self, target: Target) -> Result<Project> {
        match target {
            Target::Existing(p) => Ok(p),
            Target::Create(rec) => {
                let (p, created) = self.store.ensure_project(&rec)?;
                if created && !self.report.projects_created.contains(&p.name) {
                    self.report.projects_created.push(p.name.clone());
                }
                Ok(p)
            }
        }
    }

    fn issue(&mut self, index: u64, id: Option<String>, message: impl Into<String>) {
        self.report.issues.push(ImportIssue {
            index,
            id,
            message: message.into(),
        });
    }

    fn record(&mut self, index: u64, el: Element) -> Result<()> {
        let rec = match el {
            Element::Record(rec) => rec,
            Element::Other(kind) => {
                self.issue(
                    index,
                    None,
                    format!("the record is {kind}, not an object; it was not imported"),
                );
                return Ok(());
            }
        };
        match self.map_record(rec) {
            Ok(pending) => {
                let pending = Pending { index, ..pending };
                if !self.seen.insert(pending.id) {
                    self.issue(
                        index,
                        Some(pending.raw_id.clone()),
                        "this id appears earlier in the file; only the first record with it \
                         was imported",
                    );
                    return Ok(());
                }
                self.batch.push(pending);
                if self.batch.len() >= self.opts.batch_size.max(1) {
                    self.flush()?;
                }
                Ok(())
            }
            Err((id, message)) => {
                self.issue(index, id, message);
                Ok(())
            }
        }
    }

    /// Maps one record, or says why it cannot be stored.
    fn map_record(
        &self,
        mut rec: Map<String, Value>,
    ) -> std::result::Result<Pending, (Option<String>, String)> {
        let raw_id = match rec.get("id") {
            Some(Value::String(s)) if !s.trim().is_empty() => s.clone(),
            Some(Value::Number(n)) => n.to_string(),
            Some(Value::String(_)) => {
                return Err((None, "id is empty; the record was not imported".into()));
            }
            Some(other) => {
                return Err((
                    None,
                    format!(
                        "id is {}, not a string; the record was not imported",
                        kind_of(other)
                    ),
                ));
            }
            None => return Err((None, "the record has no id; it was not imported".into())),
        };
        let (id, _) = MemoryId::from_v1(&raw_id);
        let show = Some(raw_id.clone());
        let (project_id, project_name, project_value) = match project_value(&rec) {
            Ok(Some(v)) => match self.projects.get(v) {
                Some((pid, name)) => (*pid, name.clone(), Some(v.to_owned())),
                None => {
                    return Err((
                        show,
                        "internal: the record's project was not resolved".into(),
                    ));
                }
            },
            Ok(None) => match &self.fallback {
                Some((pid, name)) => (*pid, name.clone(), None),
                None => {
                    return Err((show, "internal: no project for a record without one".into()));
                }
            },
            Err(message) => return Err((show, message)),
        };
        let content = match rec.get("content") {
            Some(Value::String(s)) if !s.trim().is_empty() => s.clone(),
            Some(Value::String(_)) => {
                return Err((show, "content is empty; the record was not imported".into()));
            }
            Some(other) => {
                return Err((
                    show,
                    format!(
                        "content is {}, not a string; the record was not imported",
                        kind_of(other)
                    ),
                ));
            }
            None => {
                return Err((
                    show,
                    "the record has no content; it was not imported".into(),
                ));
            }
        };

        let scanner = self.scanner;
        let mut imported = Map::new();
        let mut content = content;
        let mut redacted = redact_string(scanner, &mut content);
        let hash = content_hash(&content);
        match rec.get("content_hash") {
            None | Some(Value::Null) => {}
            Some(Value::String(v1)) if *v1 == hash => {}
            Some(v1) => {
                imported.insert("content_hash".into(), v1.clone());
            }
        }

        let created_at = timestamp_field(&rec, "created_at", &mut imported);
        let updated_at = timestamp_field(&rec, "updated_at", &mut imported);

        let mut tags: Vec<String> = Vec::new();
        if let Some(ns) = rec.get("namespace") {
            match ns {
                Value::Null => {}
                Value::String(s) => {
                    let t = s.trim();
                    let same_as_project = t == project_name || project_value.as_deref() == Some(t);
                    if !t.is_empty() && !same_as_project {
                        match sanitize_tag(t) {
                            // A tag cannot hold a redaction marker: one that
                            // holds a secret is dropped, and the original
                            // goes to `imported`, where it is redacted.
                            Some(clean)
                                if holds_secret(scanner, t) || holds_secret(scanner, &clean) =>
                            {
                                imported.insert("namespace".into(), ns.clone());
                                redacted = true;
                            }
                            Some(clean) => {
                                if clean != t {
                                    imported.insert("namespace".into(), ns.clone());
                                }
                                tags.push(clean);
                            }
                            None => {
                                imported.insert("namespace".into(), ns.clone());
                            }
                        }
                    }
                }
                other => {
                    imported.insert("namespace".into(), other.clone());
                }
            }
        }
        if let Some(raw) = rec.get("tags") {
            let list = tag_list(raw, scanner);
            if !list.clean {
                imported.insert("tags".into(), raw.clone());
            }
            redacted |= list.dropped_secret;
            tags.extend(list.tags);
        }
        let mut deduped: Vec<String> = Vec::with_capacity(tags.len());
        for t in tags {
            if !deduped.contains(&t) {
                deduped.push(t);
            }
        }

        let mut extra = Map::new();
        for key in ["source", "metadata"] {
            if let Some(v) = rec.get(key) {
                extra.insert(key.into(), v.clone());
            }
        }
        let mut provenance = Provenance::new(ProvenanceSource::Import);
        if !extra.is_empty() {
            let mut extra = Value::Object(extra);
            redacted |= redact_value(scanner, &mut extra);
            provenance.extra = Some(extra);
        }

        for key in MAPPED.iter().chain(["project"].iter()) {
            rec.remove(*key);
        }
        for (k, v) in rec {
            imported.entry(k).or_insert(v);
        }
        let mut imported = Value::Object(imported);
        redacted |= redact_value(scanner, &mut imported);

        let plan = self
            .chunker
            .chunk(&content, self.chunk_params)
            .map_err(|e| {
                (
                    show.clone(),
                    format!(
                        "the content could not be chunked ({}); it was not imported",
                        e.message
                    ),
                )
            })?;
        let chunks = plan.pieces.into_iter().map(NewChunk::pending).collect();

        let mut memory = NewMemory::new(project_id, content, provenance);
        memory.id = Some(id);
        memory.memory_type = MemoryType::Note;
        memory.status = Status::Active;
        memory.origin = Origin::Local;
        memory.importance = DEFAULT_IMPORTANCE;
        memory.title = None;
        memory.content_hash = Some(hash);
        memory.tags = deduped;
        memory.imported = imported
            .as_object()
            .is_some_and(|o| !o.is_empty())
            .then_some(imported);
        memory.created_at = created_at;
        memory.updated_at = updated_at;
        memory.anchors = Vec::new();
        Ok(Pending {
            index: 0,
            id,
            raw_id,
            redacted,
            input: MemoryInput { memory, chunks },
        })
    }

    /// Stores the waiting batch in one transaction. A batch that fails on
    /// one record's data is stored record by record, so that record becomes
    /// an issue and the rest are kept.
    fn flush(&mut self) -> Result<()> {
        if self.batch.is_empty() {
            return Ok(());
        }
        let batch = std::mem::take(&mut self.batch);
        let inputs: Vec<MemoryInput> = batch.iter().map(|p| p.input.clone()).collect();
        match self.store.insert_memories(inputs, Access::Local) {
            Ok(results) => {
                for (p, r) in batch.iter().zip(results) {
                    self.count(p, r);
                }
            }
            Err(e) if is_item_error(&e) => {
                for p in &batch {
                    match self.store.insert_memory(p.input.clone(), Access::Local) {
                        Ok(r) => self.count(p, r),
                        Err(e) if is_item_error(&e) => self.issue(
                            p.index,
                            Some(p.raw_id.clone()),
                            format!("the record could not be stored: {}", e.message),
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
                "migrate stopped after {} batches as asked",
                self.batches
            )));
        }
        Ok(())
    }

    fn count(&mut self, p: &Pending, r: Inserted) {
        match r.outcome {
            RememberOutcome::Created => {
                self.report.inserted += 1;
                self.report.pending_embeddings += u64::from(r.chunks);
                if p.redacted {
                    self.report.redacted += 1;
                }
            }
            RememberOutcome::Duplicate if r.id == p.id => self.report.skipped_existing += 1,
            RememberOutcome::Duplicate => self.issue(
                p.index,
                Some(p.raw_id.clone()),
                format!(
                    "its content is already stored as memory {} in the same project; the \
                     record was not imported",
                    r.id
                ),
            ),
        }
    }
}

/// A timestamp field: parsed, or `None` with the original moved to
/// `imported` when it does not parse.
fn timestamp_field(
    rec: &Map<String, Value>,
    key: &str,
    imported: &mut Map<String, Value>,
) -> Option<Timestamp> {
    match rec.get(key) {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) => match Timestamp::parse(s) {
            Ok(t) => Some(t),
            Err(_) => {
                imported.insert(key.into(), Value::String(s.clone()));
                None
            }
        },
        Some(other) => {
            imported.insert(key.into(), other.clone());
            None
        }
    }
}

/// The tags a `tags` value gives.
struct TagList {
    tags: Vec<String>,
    /// False when anything was cleaned or left out.
    clean: bool,
    /// True when a tag was left out because it holds a secret.
    dropped_secret: bool,
}

/// Tags from a `tags` value: a list of strings, or a string holding a JSON
/// list (version 1's CSV and Parquet exports), or one tag. A tag that holds
/// a secret, before or after cleaning, is left out: a tag cannot hold a
/// redaction marker.
fn tag_list(raw: &Value, scanner: &dyn SecretScanner) -> TagList {
    let items: Vec<Value> = match raw {
        Value::Null => {
            return TagList {
                tags: Vec::new(),
                clean: true,
                dropped_secret: false,
            };
        }
        Value::Array(a) => a.clone(),
        Value::String(s) => match serde_json::from_str::<Vec<Value>>(s) {
            Ok(a) if s.trim_start().starts_with('[') => {
                let list = tag_list(&Value::Array(a), scanner);
                return TagList {
                    clean: false,
                    ..list
                };
            }
            _ => vec![Value::String(s.clone())],
        },
        _ => {
            return TagList {
                tags: Vec::new(),
                clean: false,
                dropped_secret: false,
            };
        }
    };
    let mut list = TagList {
        tags: Vec::new(),
        clean: true,
        dropped_secret: false,
    };
    for item in items {
        match item {
            Value::String(s) => match sanitize_tag(&s) {
                Some(t) if holds_secret(scanner, &s) || holds_secret(scanner, &t) => {
                    list.clean = false;
                    list.dropped_secret = true;
                }
                Some(t) => {
                    if t != s {
                        list.clean = false;
                    }
                    list.tags.push(t);
                }
                None => list.clean = false,
            },
            _ => list.clean = false,
        }
    }
    list
}

fn kind_of(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}
