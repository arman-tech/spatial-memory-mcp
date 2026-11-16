//! Whole-store `export` and `import` as JSON Lines (SPEC section 15, "One
//! user, several machines"; SPEC section 8, "Moved to the command line";
//! decision B-77).

use std::collections::{BTreeSet, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};
use sm_core::error::ErrorCode;
use sm_core::hash::content_hash;
use sm_core::ids::MemoryId;
use sm_core::protocol::{ExportParams, ExportReport, ImportParams, ImportReport};
use sm_core::time::Timestamp;
use sm_core::traits::{MemoryFilter, QueryScope};
use sm_core::types::{
    AnchorKind, AnchorRef, Link, LinkEnd, LinkSource, Memory, MemoryType, Origin, Project,
    ProvenanceSource, Relation, Status,
};
use sm_store::{Access, Revision, SCHEMA_VERSION, Store, SyncMark};

use super::{
    Fixture, add_note, all_memories, assert_code, chunk_params, chunker, embed_all_pending,
    fixture, fixture_path, identity, insert, note_input, scanner,
};
use crate::export::{EXPORT_FORMAT, EXPORT_VERSION, ImportOptions, export, import, import_with};

/// An id that sorts before every version 7 id made today, so the memory
/// that holds it comes first in an export, before the memory it supersedes.
const EARLY_ID: &str = "00000000-0000-7000-8000-0000000000a1";

/// A store with two projects and every kind of memory and link an export
/// carries.
struct Rich {
    f: Fixture,
    alpha: Project,
    beta: Project,
    /// Every optional field set.
    decision: MemoryId,
    old_local: MemoryId,
    new_local: MemoryId,
    retracted: MemoryId,
    shared: MemoryId,
    old_shared: MemoryId,
    new_shared: MemoryId,
    /// Superseded by `early`, then retracted.
    late_target: MemoryId,
    /// Supersedes `late_target`; its id sorts first.
    early: MemoryId,
    /// The memory end of a link.
    linked: MemoryId,
    in_beta: MemoryId,
}

impl Rich {
    fn memories(&self) -> usize {
        11
    }

    fn links(&self) -> usize {
        3
    }

    /// The memories another memory supersedes.
    fn targets(&self) -> HashSet<MemoryId> {
        [self.old_local, self.old_shared, self.late_target]
            .into_iter()
            .collect()
    }
}

fn mark(name: &str) -> SyncMark {
    SyncMark {
        file_path: format!(".spatial-memory/shared/note/{name}.md"),
        commit_sha: Some("4f2a9c1".to_owned()),
        blob_id: "a".repeat(40),
        text_hash: "b".repeat(64),
    }
}

fn anchor(kind: AnchorKind, name: &str) -> AnchorRef {
    AnchorRef::new(kind, name).expect("anchor")
}

fn link(src: LinkEnd, dst: LinkEnd, relation: Relation) -> Link {
    Link {
        src,
        dst,
        relation,
        source: LinkSource::Manual,
        extractor: None,
        verified_commit: None,
        verified_at: None,
    }
}

fn retract(store: &Store, id: &MemoryId) {
    store
        .revise_memory(
            id,
            Revision {
                status: Some(Status::Retracted),
                ..Revision::default()
            },
            Access::Local,
        )
        .expect("retract");
}

fn rich() -> Rich {
    let f = fixture();
    let s = &f.store;
    let alpha = s.upsert_project(&identity("alpha")).expect("project");
    let mut beta_id = identity("beta");
    beta_id.git_remote = Some("github.com/acme/beta".to_owned());
    let beta = s.upsert_project(&beta_id).expect("project");

    let mut i = note_input(
        &alpha,
        "Use optimistic concurrency on dbo.Orders: a rowversion column rejects stale writes.",
    );
    i.memory.memory_type = MemoryType::Decision;
    i.memory.title = Some("Optimistic concurrency on dbo.Orders".to_owned());
    i.memory.tags = vec!["database".to_owned(), "concurrency".to_owned()];
    i.memory.anchors = vec![
        anchor(AnchorKind::Table, "dbo.Orders"),
        anchor(AnchorKind::Service, "order-api"),
    ];
    i.memory.importance = 0.8;
    i.memory.imported = Some(json!({"access_count": 3, "importance": 0.699999988079071}));
    i.memory.created_at = Some(Timestamp::parse("2025-08-14T09:21:07.512340Z").expect("time"));
    i.memory.updated_at = Some(Timestamp::parse("2025-09-01T00:00:00.000001Z").expect("time"));
    let p = &mut i.memory.provenance;
    p.author = Some("dev".to_owned());
    p.session_id = Some("session-1".to_owned());
    p.commit_sha = Some("0123456789abcdef0123456789abcdef01234567".to_owned());
    p.file_path = Some("src/orders.rs".to_owned());
    p.tool = Some("claude-code".to_owned());
    p.extra = Some(json!({"source": "manual", "metadata": {"review_state": "approved"}}));
    let decision = insert(s, i);

    let old_local = add_note(s, &alpha, "Deploys run from the main branch.");
    let mut i = note_input(&alpha, "Deploys run from release branches.");
    i.memory.memory_type = MemoryType::Convention;
    i.memory.supersedes_id = Some(old_local);
    let new_local = insert(s, i);

    let retracted = add_note(s, &alpha, "The nightly job owns the cache warm-up.");
    retract(s, &retracted);

    let shared = add_note(s, &alpha, "The API retries a stale write once.");
    s.record_sync(&shared, &mark("shared")).expect("mark");
    let old_shared = add_note(s, &alpha, "Reviews need one approval.");
    s.record_sync(&old_shared, &mark("old-shared"))
        .expect("mark");
    let mut i = note_input(&alpha, "Reviews need two approvals.");
    i.memory.origin = Origin::Shared;
    i.memory.provenance.source = ProvenanceSource::Sync;
    i.memory.supersedes_id = Some(old_shared);
    let new_shared = s.insert_memory(i, Access::Sync).expect("insert").id;
    s.record_sync(&new_shared, &mark("new-shared"))
        .expect("mark");

    let late_target = add_note(s, &alpha, "Cache entries expire after an hour.");
    let mut i = note_input(&alpha, "Cache entries expire after ten minutes.");
    i.memory.id = Some(MemoryId::parse(EARLY_ID).expect("id"));
    i.memory.supersedes_id = Some(late_target);
    let early = insert(s, i);
    retract(s, &late_target);

    let linked = add_note(s, &alpha, "The order API is the only writer of dbo.Orders.");
    let mut calls = link(
        LinkEnd::Memory(decision),
        LinkEnd::Anchor(anchor(AnchorKind::Service, "order-api")),
        Relation::Calls,
    );
    calls.verified_commit = Some("4f2a9c1".to_owned());
    calls.verified_at = Some(Timestamp::parse("2026-11-01T10:00:00Z").expect("time"));
    s.add_link(alpha.id, calls).expect("link");
    s.add_link(
        alpha.id,
        link(
            LinkEnd::Memory(decision),
            LinkEnd::Memory(linked),
            Relation::RelatesTo,
        ),
    )
    .expect("link");

    let in_beta = add_note(s, &beta, "Billing rounds half to even.");
    s.add_link(
        beta.id,
        link(
            LinkEnd::Memory(in_beta),
            LinkEnd::Anchor(anchor(AnchorKind::Service, "billing")),
            Relation::DependsOn,
        ),
    )
    .expect("link");

    let r = Rich {
        f,
        alpha,
        beta,
        decision,
        old_local,
        new_local,
        retracted,
        shared,
        old_shared,
        new_shared,
        late_target,
        early,
        linked,
        in_beta,
    };
    assert_eq!(all_memories(&r.f.store).len(), r.memories());
    r
}

fn export_to(store: &Store, path: &Path, project: Option<&str>) -> sm_core::Result<ExportReport> {
    export(
        store,
        &ExportParams {
            path: path.to_string_lossy().into_owned(),
            project: project.map(str::to_owned),
        },
    )
}

fn import_from(store: &Store, path: &Path) -> sm_core::Result<ImportReport> {
    import(
        store,
        &chunker(),
        &scanner(),
        &chunk_params(),
        &ImportParams {
            path: path.to_string_lossy().into_owned(),
        },
    )
}

/// The lines of an export: LF only, a final LF, no blank line.
fn lines(path: &Path) -> Vec<String> {
    let text = fs::read_to_string(path).expect("an export is UTF-8");
    assert!(!text.contains('\r'), "LF line endings");
    assert!(text.ends_with('\n'), "a final LF");
    let out: Vec<String> = text.lines().map(str::to_owned).collect();
    assert!(out.iter().all(|l| !l.is_empty()), "no blank line");
    out
}

fn parse(line: &str) -> Value {
    serde_json::from_str(line).expect("a JSON line")
}

fn kind(line: &str) -> String {
    let v = parse(line);
    let o = v.as_object().expect("an object");
    assert_eq!(o.len(), 1, "one key per line: {line}");
    o.keys().next().expect("a key").clone()
}

fn memory_line_id(line: &str) -> Option<MemoryId> {
    let v = parse(line);
    let id = v.get("memory")?.get("id")?.as_str()?;
    Some(MemoryId::parse(id).expect("an id"))
}

fn write_lines(path: &Path, lines: &[String]) {
    let mut text = lines.join("\n");
    text.push('\n');
    fs::write(path, text).expect("write");
}

#[track_caller]
fn assert_balanced(r: &ImportReport) {
    assert_eq!(
        r.read,
        r.inserted + r.skipped_existing + r.issues.len() as u64,
        "read = inserted + skipped_existing + issues: {r:?}"
    );
}

fn pending(store: &Store) -> u64 {
    store.index_status().expect("status").pending_embeddings
}

/// The lines after the header, with the `updated_at` of supersession
/// targets taken out (B-77's one known difference).
fn comparable(lines: &[String], targets: &HashSet<MemoryId>) -> Vec<Value> {
    lines[1..]
        .iter()
        .map(|l| {
            let mut v = parse(l);
            if let Some(id) = memory_line_id(l)
                && targets.contains(&id)
            {
                v["memory"]
                    .as_object_mut()
                    .expect("a memory")
                    .remove("updated_at")
                    .expect("updated_at");
            }
            v
        })
        .collect()
}

#[test]
fn s15_export_writes_a_header_then_projects_memories_and_links() {
    let r = rich();
    let path = r.f.path("store.jsonl");
    let report = export_to(&r.f.store, &path, None).expect("export");
    assert_eq!(report.path, path.to_string_lossy());
    assert_eq!(
        (report.projects, report.memories, report.links),
        (2, r.memories() as u64, r.links() as u64)
    );

    let lines = lines(&path);
    let header = parse(&lines[0]);
    let keys: BTreeSet<&str> = header
        .as_object()
        .expect("an object")
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(
        keys,
        BTreeSet::from(["build", "exported_at", "format", "schema", "version"])
    );
    assert_eq!(header["format"], json!(EXPORT_FORMAT));
    assert_eq!(header["format"], json!("spatial-memory-export"));
    assert_eq!(header["version"], json!(EXPORT_VERSION));
    assert_eq!(header["version"], json!(1));
    assert_eq!(header["build"], json!(sm_core::BUILD));
    assert_eq!(header["schema"], json!(SCHEMA_VERSION));
    Timestamp::parse(header["exported_at"].as_str().expect("a string")).expect("RFC 3339");

    let kinds: Vec<String> = lines[1..].iter().map(|l| kind(l)).collect();
    let mut expected = vec!["project".to_owned(); 2];
    expected.extend(vec!["memory".to_owned(); r.memories()]);
    expected.extend(vec!["link".to_owned(); r.links()]);
    assert_eq!(kinds, expected, "projects, then memories, then links");

    // Projects carry their identity key; memories and links name it.
    assert_eq!(
        parse(&lines[1])["project"],
        json!({"identity": "path:/work/alpha", "name": "alpha", "root_path": "/work/alpha"})
    );
    assert_eq!(
        parse(&lines[2])["project"],
        json!({
            "identity": "git:github.com/acme/beta",
            "name": "beta",
            "git_remote": "github.com/acme/beta",
            "root_path": "/work/beta"
        })
    );

    // Memories in id order, every stored field, nothing derived.
    let ids: Vec<MemoryId> = lines.iter().filter_map(|l| memory_line_id(l)).collect();
    let mut sorted = ids.clone();
    sorted.sort();
    assert_eq!(ids, sorted, "memories in id order");
    assert_eq!(ids[0], r.early);
    let d = lines
        .iter()
        .find(|l| memory_line_id(l) == Some(r.decision))
        .map(|l| parse(l)["memory"].clone())
        .expect("the decision's line");
    let stored = r.f.store.get_memory(&r.decision).expect("memory");
    assert_eq!(d["project"], json!("path:/work/alpha"));
    assert_eq!(d["type"], json!("decision"));
    assert_eq!(d["title"], json!("Optimistic concurrency on dbo.Orders"));
    assert_eq!(d["content_hash"], json!(stored.content_hash));
    assert_eq!(d["tags"], json!(["database", "concurrency"]));
    assert_eq!(
        d["anchors"],
        json!(["table:dbo.Orders", "service:order-api"])
    );
    assert_eq!(d["importance"], json!(0.8));
    assert_eq!(
        d["imported"],
        json!({"access_count": 3, "importance": 0.699999988079071})
    );
    assert_eq!(d["created_at"], json!("2025-08-14T09:21:07.512340Z"));
    assert_eq!(d["provenance"]["author"], json!("dev"));
    assert_eq!(
        d["provenance"]["extra"]["metadata"]["review_state"],
        json!("approved")
    );
    for derived in ["chunks", "vector", "embedding", "project_id"] {
        assert!(d.get(derived).is_none(), "{derived} is not exported");
    }

    // Links name their ends as `memory:<id>` or `kind:name`.
    let calls = parse(&lines[lines.len() - r.links()])["link"].clone();
    assert_eq!(calls["project"], json!("path:/work/alpha"));
    assert_eq!(calls["src"], json!(format!("memory:{}", r.decision)));
    assert_eq!(calls["dst"], json!("service:order-api"));
    assert_eq!(calls["relation"], json!("calls"));
    assert_eq!(calls["verified_commit"], json!("4f2a9c1"));

    // The same store gives the same bytes, apart from the header.
    let again = r.f.path("again.jsonl");
    export_to(&r.f.store, &again, None).expect("export");
    assert_eq!(self::lines(&again)[1..], lines[1..]);
}

#[test]
fn s15_export_import_export_round_trips() {
    let r = rich();
    let first = r.f.path("first.jsonl");
    export_to(&r.f.store, &first, None).expect("export");

    let b = fixture();
    let report = import_from(&b.store, &first).expect("import");
    let read = (r.memories() + r.links()) as u64;
    assert_eq!(report.read, read);
    assert_eq!(report.inserted, read, "{report:?}");
    assert_eq!(report.skipped_existing, 0);
    assert!(report.issues.is_empty(), "{:?}", report.issues);
    assert_eq!(report.projects_created, ["alpha", "beta"]);
    assert_eq!(report.redacted, 0, "redacting again changes nothing");
    assert!(report.pending_embeddings > 0);
    assert_eq!(report.pending_embeddings, pending(&b.store));
    assert_balanced(&report);

    let second = b.path("second.jsonl");
    export_to(&b.store, &second, None).expect("export");
    let (one, two) = (lines(&first), lines(&second));
    assert_eq!(one.len(), two.len());

    // Every line but the header is byte-equal, except the updated_at of a
    // superseded memory, which the store stamps when its superseder is
    // inserted (B-77).
    let targets = r.targets();
    for (a, b) in one[1..].iter().zip(&two[1..]) {
        match memory_line_id(a) {
            Some(id) if targets.contains(&id) => {}
            _ => assert_eq!(a, b),
        }
    }
    assert_eq!(comparable(&one, &targets), comparable(&two, &targets));
    for id in &targets {
        let (a, bm) = (
            r.f.store.get_memory(id).expect("memory"),
            b.store.get_memory(id).expect("memory"),
        );
        assert!(bm.updated_at >= a.updated_at, "stamped at the import");
        assert_eq!(
            Memory {
                updated_at: a.updated_at,
                project_id: a.project_id,
                ..bm
            },
            a
        );
    }

    // Every other memory is equal field by field.
    for a in all_memories(&r.f.store) {
        let bm = b.store.get_memory(&a.id).expect("imported");
        assert_eq!(
            b.store.project(bm.project_id).expect("project").name,
            r.f.store.project(a.project_id).expect("project").name
        );
        if !targets.contains(&a.id) {
            assert_eq!(
                Memory {
                    project_id: a.project_id,
                    ..bm
                },
                a
            );
        }
    }

    // Keyword search works while the chunks wait; then everything embeds.
    let hits: Vec<MemoryId> = b
        .store
        .keyword_search("rowversion", QueryScope::All, &MemoryFilter::default(), 10)
        .expect("search")
        .into_iter()
        .map(|h| h.memory_id)
        .collect();
    assert_eq!(hits, [r.decision]);
    embed_all_pending(&b.store);
    assert_eq!(pending(&b.store), 0);
    for m in all_memories(&b.store) {
        assert!(b.store.memory_vector(&m.id).expect("vector").is_some());
    }

    // Importing the same file again stores nothing new.
    let again = import_from(&b.store, &first).expect("import again");
    assert_eq!(again.read, read);
    assert_eq!(again.inserted, 0);
    assert_eq!(again.skipped_existing, read);
    assert!(again.issues.is_empty(), "{:?}", again.issues);
    assert!(again.projects_created.is_empty());
    assert_eq!(all_memories(&b.store).len(), r.memories());
}

#[test]
fn s15_import_keeps_shared_memories_and_supersessions() {
    let r = rich();
    let path = r.f.path("store.jsonl");
    export_to(&r.f.store, &path, None).expect("export");
    let b = fixture();
    import_from(&b.store, &path).expect("import");
    let get = |id: &MemoryId| b.store.get_memory(id).expect("memory");

    // Local supersession.
    assert_eq!(get(&r.old_local).status, Status::Superseded);
    assert_eq!(get(&r.new_local).supersedes_id, Some(r.old_local));
    assert_eq!(get(&r.new_local).status, Status::Active);
    assert_eq!(get(&r.new_local).memory_type, MemoryType::Convention);

    // Shared memories keep their origin and their sync mark, and a shared
    // supersession is kept.
    for id in [r.shared, r.old_shared, r.new_shared] {
        let a = r.f.store.get_memory(&id).expect("memory");
        let m = get(&id);
        assert_eq!(m.origin, Origin::Shared);
        assert_eq!(SyncMark::of(&m), SyncMark::of(&a));
        assert!(SyncMark::of(&m).is_some());
    }
    assert_eq!(get(&r.old_shared).status, Status::Superseded);
    assert_eq!(get(&r.new_shared).supersedes_id, Some(r.old_shared));
    assert_eq!(get(&r.new_shared).provenance.source, ProvenanceSource::Sync);

    // A superseder whose line comes first waits for its target, and a
    // target that was retracted after it was superseded stays retracted.
    assert_eq!(get(&r.early).supersedes_id, Some(r.late_target));
    assert_eq!(get(&r.late_target).status, Status::Retracted);
    assert_eq!(get(&r.retracted).status, Status::Retracted);
    assert_eq!(get(&r.in_beta).origin, Origin::Local);

    // Links, in both projects.
    let alpha = b
        .store
        .project_by_identity("path:/work/alpha")
        .expect("find")
        .expect("alpha");
    let beta = b
        .store
        .project_by_identity("git:github.com/acme/beta")
        .expect("find")
        .expect("beta");
    assert_eq!(beta.git_remote, r.beta.git_remote);
    assert_eq!(alpha.name, r.alpha.name);
    assert_eq!(
        b.store.project_links(alpha.id).expect("links"),
        r.f.store.project_links(r.alpha.id).expect("links")
    );
    assert_eq!(
        b.store.project_links(beta.id).expect("links"),
        r.f.store.project_links(r.beta.id).expect("links")
    );
    assert!(
        b.store
            .project_links(alpha.id)
            .expect("links")
            .iter()
            .any(|l| l.dst == LinkEnd::Memory(r.linked))
    );
}

#[test]
fn s15_export_refuses_an_existing_file() {
    let r = rich();
    let out = r.f.path("out");
    fs::create_dir(&out).expect("folder");
    let path = out.join("store.jsonl");
    fs::write(&path, "keep me\n").expect("write");
    let e = assert_code(export_to(&r.f.store, &path, None), ErrorCode::InvalidInput);
    assert!(e.message.contains("already exists"), "{}", e.message);
    assert!(!e.next_action.is_empty());
    assert_eq!(fs::read_to_string(&path).expect("read"), "keep me\n");
    let left: Vec<PathBuf> = fs::read_dir(&out)
        .expect("list")
        .map(|e| e.expect("entry").path())
        .collect();
    assert_eq!(left, [path], "no temporary file is left behind");

    // A folder that does not exist fails without leaving a file.
    let missing = r.f.path("missing").join("store.jsonl");
    assert_code(export_to(&r.f.store, &missing, None), ErrorCode::Storage);
    assert!(!r.f.path("missing").exists());
}

#[test]
fn s15_export_selects_one_project_by_name() {
    let r = rich();
    let path = r.f.path("beta.jsonl");
    let report = export_to(&r.f.store, &path, Some("beta")).expect("export");
    assert_eq!((report.projects, report.memories, report.links), (1, 1, 1));
    let lines = lines(&path);
    assert_eq!(lines.len(), 4);
    assert_eq!(parse(&lines[1])["project"]["name"], json!("beta"));
    assert_eq!(memory_line_id(&lines[2]), Some(r.in_beta));
    assert_eq!(
        parse(&lines[3])["link"]["project"],
        json!("git:github.com/acme/beta")
    );

    // The one-project file imports as that project alone.
    let b = fixture();
    let imported = import_from(&b.store, &path).expect("import");
    assert_eq!((imported.read, imported.inserted), (2, 2));
    assert_eq!(imported.projects_created, ["beta"]);
    assert_eq!(all_memories(&b.store).len(), 1);

    // No project, or two, by that name.
    let none = r.f.path("none.jsonl");
    let e = assert_code(
        export_to(&r.f.store, &none, Some("gamma")),
        ErrorCode::NotFound,
    );
    assert!(e.message.contains("gamma"), "{}", e.message);
    assert!(!none.exists());

    let mut twin = identity("beta");
    twin.root_key = "/elsewhere/beta".to_owned();
    twin.root_path = "/elsewhere/beta".to_owned();
    twin.root = PathBuf::from("/elsewhere/beta");
    r.f.store.upsert_project(&twin).expect("a second beta");
    let two = r.f.path("two.jsonl");
    let e = assert_code(
        export_to(&r.f.store, &two, Some("beta")),
        ErrorCode::InvalidInput,
    );
    assert!(e.message.contains("2 projects are named"), "{}", e.message);
    assert!(!two.exists());
}

#[test]
fn s15_import_refuses_an_unknown_format_or_a_newer_version_before_writing() {
    let r = rich();
    let path = r.f.path("store.jsonl");
    export_to(&r.f.store, &path, None).expect("export");
    let good = lines(&path);
    let header = parse(&good[0]);

    let with_header = |name: &str, h: Value| {
        let mut l = good.clone();
        l[0] = h.to_string();
        let p = r.f.path(name);
        write_lines(&p, &l);
        p
    };
    let mut other = header.clone();
    other["format"] = json!("someone-elses-export");
    let mut newer = header.clone();
    newer["version"] = json!(EXPORT_VERSION + 1);
    let mut no_version = header.clone();
    no_version
        .as_object_mut()
        .expect("object")
        .remove("version");
    let mut no_format = header.clone();
    no_format.as_object_mut().expect("object").remove("format");
    let empty = r.f.path("empty.jsonl");
    fs::write(&empty, "").expect("write");
    let not_json = r.f.path("not-json.jsonl");
    let mut l = good.clone();
    l[0] = "spatial-memory-export v1".to_owned();
    write_lines(&not_json, &l);

    let cases: Vec<(PathBuf, &str)> = vec![
        (with_header("other.jsonl", other), "someone-elses-export"),
        (with_header("newer.jsonl", newer), "export version 2"),
        (
            with_header("no-version.jsonl", no_version),
            "no version number",
        ),
        (
            with_header("no-format.jsonl", no_format),
            "format is missing",
        ),
        (empty, "the file is empty"),
        (not_json, "line 1"),
        (
            fixture_path("migration/v1_export_full.json"),
            "not a Spatial Memory export",
        ),
    ];
    for (p, part) in cases {
        let b = fixture();
        let e = assert_code(import_from(&b.store, &p), ErrorCode::InvalidInput);
        assert!(e.message.contains(part), "{}: {}", p.display(), e.message);
        assert!(e.message.contains("nothing was imported"), "{}", e.message);
        assert!(!e.next_action.is_empty());
        assert_eq!(b.store.memory_count().expect("count"), 0, "{}", p.display());
        assert!(
            b.store.projects().expect("projects").is_empty(),
            "{}",
            p.display()
        );
    }

    let b = fixture();
    assert_code(
        import_from(&b.store, &r.f.path("nowhere.jsonl")),
        ErrorCode::NotFound,
    );
}

#[test]
fn s15_an_interrupted_import_resumes() {
    let r = rich();
    let path = r.f.path("store.jsonl");
    export_to(&r.f.store, &path, None).expect("export");
    let b = fixture();
    let stopped = import_with(
        &b.store,
        &chunker(),
        &scanner(),
        &chunk_params(),
        &ImportParams {
            path: path.to_string_lossy().into_owned(),
        },
        &ImportOptions {
            batch_size: 2,
            stop_after_batches: Some(1),
        },
    );
    assert_code(stopped, ErrorCode::Internal);
    assert_eq!(b.store.memory_count().expect("count"), 2, "one whole batch");

    let resumed = import_from(&b.store, &path).expect("resume");
    let read = (r.memories() + r.links()) as u64;
    assert_eq!(resumed.read, read);
    assert_eq!(resumed.skipped_existing, 2);
    assert_eq!(resumed.inserted, read - 2);
    assert!(resumed.issues.is_empty(), "{:?}", resumed.issues);
    assert!(
        resumed.projects_created.is_empty(),
        "created by the first run"
    );
    assert_balanced(&resumed);

    let second = b.path("second.jsonl");
    export_to(&b.store, &second, None).expect("export");
    let targets = r.targets();
    assert_eq!(
        comparable(&lines(&path), &targets),
        comparable(&lines(&second), &targets)
    );
}

#[test]
fn s15_import_reports_unusable_lines_and_redacts_secrets() {
    let r = rich();
    let path = r.f.path("store.jsonl");
    export_to(&r.f.store, &path, None).expect("export");
    let good = lines(&path);

    // Keep every line but `linked`'s memory line, so the link to it has a
    // missing end; add a secret by hand to the decision; then add lines that
    // cannot be used, before the links.
    let mut l: Vec<String> = vec![good[0].clone()];
    let mut links = Vec::new();
    for line in &good[1..] {
        match memory_line_id(line) {
            Some(id) if id == r.linked => {}
            Some(id) if id == r.decision => {
                let mut v = parse(line);
                v["memory"]["content"] = json!(format!(
                    "{} Key AKIAIOSFODNN7EXAMPLE.",
                    v["memory"]["content"].as_str().expect("content")
                ));
                l.push(v.to_string());
            }
            _ if kind(line) == "link" => links.push(line.clone()),
            _ => l.push(line.clone()),
        }
    }
    let decision_line = good
        .iter()
        .find(|line| memory_line_id(line) == Some(r.decision))
        .expect("line");
    let mut unknown_field = parse(decision_line);
    unknown_field["memory"]["id"] = json!(MemoryId::new_v7().to_string());
    unknown_field["memory"]["colour"] = json!("blue");
    let mut no_project = parse(decision_line);
    let orphan = MemoryId::new_v7();
    no_project["memory"]["id"] = json!(orphan.to_string());
    no_project["memory"]["project"] = json!("path:/nowhere");
    no_project["memory"]["content"] = json!("A memory of a project the file does not list.");
    let bad = [
        "this is not JSON".to_owned(),
        json!({"widget": {}}).to_string(),
        unknown_field.to_string(),
        good.iter()
            .find(|line| memory_line_id(line) == Some(r.in_beta))
            .expect("line")
            .clone(),
        no_project.to_string(),
    ];
    let first_bad = l.len() as u64;
    l.extend(bad.iter().cloned());
    let first_link = l.len() as u64;
    l.extend(links);
    let edited = r.f.path("edited.jsonl");
    write_lines(&edited, &l);

    let b = fixture();
    let report = import_from(&b.store, &edited).expect("import");
    assert_balanced(&report);
    let memories = r.memories() as u64 - 1;
    let links = r.links() as u64;
    assert_eq!(report.read, memories + links + bad.len() as u64);
    assert_eq!(report.inserted, memories + links - 1, "{report:?}");
    assert_eq!(report.skipped_existing, 0);
    assert_eq!(report.redacted, 1);

    let at: Vec<u64> = report.issues.iter().map(|i| i.index).collect();
    let mut expected: Vec<u64> = (first_bad..first_bad + bad.len() as u64).collect();
    expected.push(first_link + 1); // decision -> linked
    assert_eq!(at, expected, "{:?}", report.issues);
    let message = |n: usize| report.issues[n].message.as_str();
    assert!(message(0).contains("does not parse"), "{}", message(0));
    assert!(message(1).contains("does not parse"), "{}", message(1));
    assert!(message(2).contains("colour"), "{}", message(2));
    assert_eq!(
        report.issues[2].id,
        unknown_field["memory"]["id"].as_str().map(str::to_owned)
    );
    assert!(message(3).contains("appears earlier"), "{}", message(3));
    assert!(message(4).contains("path:/nowhere"), "{}", message(4));
    assert_eq!(report.issues[4].id, Some(orphan.to_string()));
    assert!(
        message(5).contains("an end of the link is missing"),
        "{}",
        message(5)
    );
    assert_eq!(
        report.issues[5].id,
        Some(format!("memory:{} -> memory:{}", r.decision, r.linked))
    );

    // The secret is redacted, and the hash is of the stored text.
    let d = b.store.get_memory(&r.decision).expect("memory");
    assert!(!d.content.contains("AKIAIOSFODNN7EXAMPLE"));
    assert!(d.content.contains("[REDACTED:"), "{}", d.content);
    assert_eq!(d.content_hash, content_hash(&d.content));
    assert!(
        b.store
            .keyword_search(
                "AKIAIOSFODNN7EXAMPLE",
                QueryScope::All,
                &MemoryFilter::default(),
                10
            )
            .expect("search")
            .is_empty()
    );
    assert!(b.store.find_memory(&orphan).expect("find").is_none());
    assert!(b.store.find_memory(&r.linked).expect("find").is_none());
}
