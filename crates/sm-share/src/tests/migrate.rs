//! `migrate` (SPEC section 10; AT-8; decisions B-12, B-14, B-18, B-35 and
//! B-79).

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use serde_json::{Value, json};
use sm_core::error::ErrorCode;
use sm_core::hash::content_hash;
use sm_core::ids::MemoryId;
use sm_core::project::ProjectIdentity;
use sm_core::protocol::{ImportReport, MigrateParams};
use sm_core::time::Timestamp;
use sm_core::traits::{MemoryFilter, QueryScope, SecretScanner};
use sm_core::types::{DEFAULT_IMPORTANCE, Memory, MemoryType, Origin, ProvenanceSource, Status};
use sm_store::Store;

use super::{
    Fixture, all_memories, assert_code, chunk_params, chunker, embed_all_pending, fixture,
    fixture_path, identity, project, read_fixture, scanner,
};
use crate::migrate::{MigrateOptions, migrate_with};

const FULL: &str = "migration/v1_export_full.json";
const SAMPLE: &str = "migration/v1_export_sample.json";

/// Fields `migrate` maps; every other field but `vector` goes to `imported`.
const MAPPED: [&str; 10] = [
    "id",
    "content",
    "content_hash",
    "created_at",
    "updated_at",
    "namespace",
    "tags",
    "source",
    "metadata",
    "project",
];

fn params_for(path: PathBuf, default_project: Option<&str>) -> MigrateParams {
    MigrateParams {
        from: path.to_string_lossy().into_owned(),
        default_project: default_project.map(str::to_owned),
    }
}

fn run_with(
    store: &Store,
    params: &MigrateParams,
    cwd: Option<&ProjectIdentity>,
    opts: &MigrateOptions,
) -> sm_core::Result<ImportReport> {
    migrate_with(
        store,
        &chunker(),
        &scanner(),
        &chunk_params(),
        params,
        cwd,
        opts,
    )
}

fn run(
    store: &Store,
    params: &MigrateParams,
    cwd: Option<&ProjectIdentity>,
) -> sm_core::Result<ImportReport> {
    run_with(store, params, cwd, &MigrateOptions::default())
}

/// The records of a fixture, parsed independently of `migrate`.
fn records(rel: &str) -> Vec<Value> {
    let v: Value = serde_json::from_str(&read_fixture(rel)).expect("a fixture parses");
    v.as_array().expect("an array").clone()
}

fn write_json(f: &Fixture, name: &str, v: &Value) -> PathBuf {
    let path = f.path(name);
    fs::write(&path, serde_json::to_string_pretty(v).expect("json")).expect("write");
    path
}

#[track_caller]
fn assert_balanced(r: &ImportReport) {
    assert_eq!(
        r.read,
        r.inserted + r.skipped_existing + r.issues.len() as u64,
        "read = inserted + skipped_existing + issues (AT-8): {r:?}"
    );
}

fn naive_utc(s: &str) -> Timestamp {
    Timestamp::parse(&format!("{s}Z")).expect("a fixture timestamp")
}

fn imported(m: &Memory) -> serde_json::Map<String, Value> {
    m.imported
        .as_ref()
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default()
}

fn pending(store: &Store) -> u64 {
    store
        .index_status()
        .expect("index status")
        .pending_embeddings
}

fn keyword(store: &Store, text: &str) -> Vec<MemoryId> {
    store
        .keyword_search(text, QueryScope::All, &MemoryFilter::default(), 20)
        .expect("keyword search")
        .into_iter()
        .map(|h| h.memory_id)
        .collect()
}

/// The checks of AT-8 that hold for every record of either fixture, then
/// embedding and a second run. Returns the store for record checks.
fn at8_migrates(rel: &str, default_project: Option<&str>, word: &str, word_id: &str) -> Fixture {
    let f = fixture();
    let recs = records(rel);
    let params = params_for(fixture_path(rel), default_project);
    let report = run(&f.store, &params, None).expect("migrate");

    assert_eq!(report.read, recs.len() as u64);
    assert_eq!(
        report.inserted, report.read,
        "every record is stored: {report:?}"
    );
    assert_eq!(report.skipped_existing, 0);
    assert!(report.issues.is_empty(), "{:?}", report.issues);
    assert_balanced(&report);
    assert_eq!(report.source, params.from);
    assert_eq!(f.store.memory_count().expect("count"), report.read);
    assert!(report.pending_embeddings > 0);
    assert_eq!(
        report.pending_embeddings,
        pending(&f.store),
        "chunks queued"
    );

    let s = scanner();
    let projects: HashMap<_, _> = f
        .store
        .projects()
        .expect("projects")
        .into_iter()
        .map(|p| (p.id, p.name))
        .collect();
    for rec in &recs {
        let raw_id = rec["id"].as_str().expect("an id");
        let (id, kept) = MemoryId::from_v1(raw_id);
        let m = f.store.get_memory(&id).expect("every record is stored");
        if kept {
            assert_eq!(m.id.to_string(), raw_id, "a UUID id is kept");
        } else {
            assert_ne!(m.id.to_string(), raw_id);
            assert_eq!(MemoryId::from_v1(raw_id).0, m.id, "B-12 is stable");
        }

        // Timestamps: a naive time is UTC, kept to the microsecond.
        for (key, got) in [("created_at", m.created_at), ("updated_at", m.updated_at)] {
            if let Some(t) = rec.get(key).and_then(Value::as_str) {
                assert_eq!(got, naive_utc(t), "{raw_id} {key}");
            }
        }

        // Content and hash. A record with no secret keeps its text and its
        // version 1 hash.
        let content = rec["content"].as_str().expect("content");
        let has_secret = !s.scan(content).is_empty();
        if !has_secret {
            assert_eq!(m.content, content, "{raw_id}: content is kept");
            if let Some(v1) = rec.get("content_hash").and_then(Value::as_str) {
                assert_eq!(m.content_hash, v1, "{raw_id}: the version 1 hash is kept");
                assert!(!imported(&m).contains_key("content_hash"));
            }
        }
        assert_eq!(m.content_hash, content_hash(&m.content));

        // Tags: the namespace first unless it is the project, then the tags.
        let project_value = rec.get("project").and_then(Value::as_str);
        let mut tags: Vec<String> = Vec::new();
        if let Some(ns) = rec.get("namespace").and_then(Value::as_str)
            && !ns.is_empty()
            && Some(ns) != project_value
            && Some(ns) != projects.get(&m.project_id).map(String::as_str)
        {
            tags.push(ns.to_owned());
        }
        for t in rec["tags"].as_array().expect("tags") {
            let t = t.as_str().expect("a tag").to_owned();
            if !tags.contains(&t) {
                tags.push(t);
            }
        }
        assert_eq!(m.tags, tags, "{raw_id}: tags");

        // Provenance and imported fields.
        assert_eq!(m.provenance.source, ProvenanceSource::Import);
        let extra = m.provenance.extra.clone().expect("source and metadata");
        assert_eq!(extra["source"], rec["source"], "{raw_id}");
        assert_eq!(extra["metadata"], rec["metadata"], "{raw_id}");
        let imp = imported(&m);
        for (k, v) in rec.as_object().expect("an object") {
            if MAPPED.contains(&k.as_str()) || k == "vector" {
                continue;
            }
            assert_eq!(imp.get(k), Some(v), "{raw_id}: {k} goes to imported");
        }
        assert!(!imp.contains_key("vector"), "the vector is dropped");

        assert_eq!(m.memory_type, MemoryType::Note);
        assert_eq!(m.status, Status::Active);
        assert_eq!(m.origin, Origin::Local);
        assert_eq!(m.importance, DEFAULT_IMPORTANCE);
        assert_eq!(m.title, None);
        assert!(m.anchors.is_empty());
        assert_eq!(m.supersedes_id, None);
    }

    // Keyword search finds a migrated memory while its chunks are pending.
    let (word_memory, _) = MemoryId::from_v1(word_id);
    assert_eq!(f.store.memory_vector(&word_memory).expect("vector"), None);
    assert!(keyword(&f.store, word).contains(&word_memory), "{word}");

    // After embedding, every memory has embedded chunks.
    embed_all_pending(&f.store);
    assert_eq!(pending(&f.store), 0);
    for m in all_memories(&f.store) {
        assert!(
            f.store.memory_vector(&m.id).expect("vector").is_some(),
            "{} is embedded",
            m.id
        );
    }

    // A second run inserts nothing.
    let again = run(&f.store, &params, None).expect("migrate again");
    assert_eq!(again.read, report.read);
    assert_eq!(again.inserted, 0);
    assert_eq!(again.skipped_existing, report.read);
    assert!(again.issues.is_empty(), "{:?}", again.issues);
    assert!(again.projects_created.is_empty());
    assert_eq!((again.pending_embeddings, again.redacted), (0, 0));
    assert_balanced(&again);
    assert_eq!(f.store.memory_count().expect("count"), report.read);
    f
}

#[test]
fn at8_full_fixture_migrates() {
    let f = at8_migrates(
        FULL,
        Some("inbox"),
        "rowversion",
        "3f6c2a1e-8b7d-4c59-9e0a-5d1f2b3c4a6e",
    );
    let m = |raw: &str| {
        f.store
            .get_memory(&MemoryId::from_v1(raw).0)
            .expect("memory")
    };
    let name_of = |m: &Memory| f.store.project(m.project_id).expect("project").name;

    // Record 1: microseconds, and the namespace becomes a tag.
    let r1 = m("3f6c2a1e-8b7d-4c59-9e0a-5d1f2b3c4a6e");
    assert_eq!(r1.created_at.as_micros(), 1_755_163_267_512_340);
    assert_eq!(r1.created_at.to_rfc3339(), "2025-08-14T09:21:07.512340Z");
    assert_eq!(
        r1.tags,
        ["decisions", "database", "concurrency", "decision"]
    );
    assert_eq!(name_of(&r1), "spatial-memory-mcp");
    assert_eq!(imported(&r1)["importance"], json!(0.699999988079071));
    assert_eq!(imported(&r1)["access_count"], json!(3));
    assert_eq!(imported(&r1)["expires_at"], Value::Null);

    // Record 2: its namespace equals its project, so no tag is added.
    let r2 = m("a8e5b0c4-1d2f-4e6a-b7c8-9d0e1f2a3b4c");
    assert_eq!(r2.tags, ["i18n", "cache"]);
    assert_eq!(name_of(&r2), "billing-service");

    // Record 3: no project, so the default project; the edge of a year.
    let r3 = m("c1d2e3f4-a5b6-4c7d-8e9f-0a1b2c3d4e5f");
    assert_eq!(name_of(&r3), "inbox");
    assert_eq!(r3.created_at.as_micros(), 1_735_689_599_999_999);
    assert_eq!(r3.updated_at.as_micros(), 1_735_689_600_000_001);

    // Record 4: the access key is gone from the text and from keyword
    // search (the secret access key: `at8_full_fixture_record_4_secret_is_redacted`);
    // the hash is recomputed and the v1 value kept aside.
    let r4 = m(RECORD_4);
    assert!(
        !r4.content.contains("AKIAIOSFODNN7EXAMPLE"),
        "{}",
        r4.content
    );
    assert!(keyword(&f.store, "AKIAIOSFODNN7EXAMPLE").is_empty());
    assert!(r4.content.contains("[REDACTED:aws_key]"), "{}", r4.content);
    assert!(scanner().scan(&r4.content).is_empty());
    assert!(keyword(&f.store, "staging deploy").contains(&r4.id));
    let v1 = "500cc0595d7d7263d62616fc108f23850b7b6d1b6541ca835cd677c560daff05";
    assert_ne!(r4.content_hash, v1);
    assert_eq!(r4.content_hash, content_hash(&r4.content));
    assert_eq!(imported(&r4)["content_hash"], json!(v1));

    // Record 5: surrounding whitespace, and the version 1 hash is kept.
    let r5 = m("f0e1d2c3-b4a5-4697-8879-6a5b4c3d2e1f");
    assert_eq!(
        r5.content_hash,
        "7b6fe195bd9f754f249229372006d392e5b4ae491f026c54deade773cf9ba28c"
    );
    assert!(!imported(&r5).contains_key("content_hash"));
    assert_eq!(r5.created_at.as_micros(), 1_762_353_910_000_300);

    // Record 6: Markdown unchanged.
    let r6 = m("5e4d3c2b-1a09-4f8e-8d7c-6b5a49382716");
    assert!(
        r6.content
            .starts_with("# Release checklist\n\n## Before tagging\n")
    );
    assert!(
        r6.content
            .contains("```text\nspatial-memory doctor\nspatial-memory status\n```\n")
    );
}

/// Record 4 of the full fixture: Amazon's documented example credentials.
const RECORD_4: &str = "0b9a8c7d-6e5f-4a3b-9c2d-1e0f9a8b7c6d";

/// The fixture's README: record 4 holds the example access key "and secret,
/// which migration must redact (SPEC 10 step 2, AT-9)".
#[test]
fn at8_full_fixture_record_4_secret_is_redacted() {
    let f = fixture();
    let r = run(
        &f.store,
        &params_for(fixture_path(FULL), Some("inbox")),
        None,
    )
    .expect("migrate");
    assert_balanced(&r);
    let id = MemoryId::parse(RECORD_4).expect("id");
    let r4 = f.store.get_memory(&id).expect("record 4");
    for secret in [
        "AKIAIOSFODNN7EXAMPLE",
        "wJalrXUtnFEMI",
        "bPxRfiCYEXAMPLEKEY",
    ] {
        assert!(
            !r4.content.contains(secret),
            "{secret} is redacted: {}",
            r4.content
        );
        assert!(
            keyword(&f.store, secret).is_empty(),
            "{secret} is not indexed"
        );
    }
}

#[test]
fn at8_full_fixture_report_lists_projects_and_redactions() {
    let f = fixture();
    let report = run(
        &f.store,
        &params_for(fixture_path(FULL), Some("inbox")),
        None,
    )
    .expect("migrate");
    assert_eq!(
        report.projects_created,
        ["spatial-memory-mcp", "billing-service", "inbox"]
    );
    assert_eq!(report.redacted, 1, "record 4");
    for name in ["spatial-memory-mcp", "billing-service", "inbox"] {
        let p = f
            .store
            .project_by_identity(&format!("name:{name}"))
            .expect("find")
            .unwrap_or_else(|| panic!("a name: project for {name}"));
        assert_eq!((p.name.as_str(), p.git_remote.as_deref()), (name, None));
    }
}

#[test]
fn at8_sample_fixture_migrates() {
    let f = at8_migrates(SAMPLE, None, "LanceDB", "mem-0001");
    let (id, kept) = MemoryId::from_v1("mem-0001");
    assert!(!kept, "mem-0001 is not a UUID");
    assert_eq!(id.as_uuid().get_version_num(), 8, "B-12: a version 8 id");
    let m = f.store.get_memory(&id).expect("memory");
    assert_eq!(m.tags, ["decisions", "decision", "infra", "lancedb"]);
    // mem-0004 and mem-0006 share the namespace "notes" and keep it.
    let m6 = f
        .store
        .get_memory(&MemoryId::from_v1("mem-0006").0)
        .expect("memory");
    assert_eq!(m6.tags, ["notes", "todo", "perf"]);
}

/// A run stopped after its first batch resumes: the stored ids are skipped
/// and the rest are inserted.
fn at8_resumes(rel: &str, default_project: Option<&str>) {
    let f = fixture();
    let n = records(rel).len() as u64;
    let params = params_for(fixture_path(rel), default_project);
    let stopped = run_with(
        &f.store,
        &params,
        None,
        &MigrateOptions {
            batch_size: 2,
            stop_after_batches: Some(1),
        },
    );
    assert_code(stopped, ErrorCode::Internal);
    assert_eq!(f.store.memory_count().expect("count"), 2, "one whole batch");

    let resumed = run(&f.store, &params, None).expect("resume");
    assert_eq!(resumed.read, n);
    assert_eq!(resumed.skipped_existing, 2);
    assert_eq!(resumed.inserted, n - 2);
    assert!(resumed.issues.is_empty(), "{:?}", resumed.issues);
    assert_balanced(&resumed);
    assert_eq!(f.store.memory_count().expect("count"), n);
    for rec in records(rel) {
        let raw = rec["id"].as_str().expect("id");
        assert!(
            f.store
                .find_memory(&MemoryId::from_v1(raw).0)
                .expect("find")
                .is_some(),
            "{raw}"
        );
    }
}

#[test]
fn at8_full_fixture_resumes_after_a_stopped_batch() {
    at8_resumes(FULL, Some("inbox"));
}

#[test]
fn at8_sample_fixture_resumes_after_a_stopped_batch() {
    at8_resumes(SAMPLE, None);
}

#[test]
fn s10_records_without_project_and_no_fallback_are_project_unresolved() {
    let f = fixture();
    let e = assert_code(
        run(&f.store, &params_for(fixture_path(FULL), None), None),
        ErrorCode::ProjectUnresolved,
    );
    assert!(!e.next_action.is_empty());
    assert_eq!(f.store.memory_count().expect("count"), 0, "nothing written");
    assert!(
        f.store.projects().expect("projects").is_empty(),
        "no project created"
    );
}

#[test]
fn s10_records_without_project_go_to_the_working_directory_project() {
    let f = fixture();
    let here = identity("here");
    let report =
        run(&f.store, &params_for(fixture_path(FULL), None), Some(&here)).expect("migrate");
    assert!(report.projects_created.contains(&"here".to_owned()));
    let p = f
        .store
        .project_by_identity(&here.key())
        .expect("find")
        .expect("created by its identity");
    let r3 = f
        .store
        .get_memory(&MemoryId::from_v1("c1d2e3f4-a5b6-4c7d-8e9f-0a1b2c3d4e5f").0)
        .expect("memory");
    assert_eq!(r3.project_id, p.id);

    // --default-project wins over the working directory.
    let g = fixture();
    run(
        &g.store,
        &params_for(fixture_path(FULL), Some("inbox")),
        Some(&here),
    )
    .expect("migrate");
    let r3 = g
        .store
        .get_memory(&MemoryId::from_v1("c1d2e3f4-a5b6-4c7d-8e9f-0a1b2c3d4e5f").0)
        .expect("memory");
    assert_eq!(
        g.store.project(r3.project_id).expect("project").name,
        "inbox"
    );
}

#[test]
fn s10_two_projects_with_one_name_is_invalid_input() {
    let f = fixture();
    project(&f.store, "spatial-memory-mcp");
    let mut elsewhere = identity("spatial-memory-mcp");
    elsewhere.root_key = "/elsewhere/spatial-memory-mcp".to_owned();
    elsewhere.root_path = "/elsewhere/spatial-memory-mcp".to_owned();
    elsewhere.root = PathBuf::from("/elsewhere/spatial-memory-mcp");
    f.store
        .upsert_project(&elsewhere)
        .expect("a second project");
    assert_eq!(
        f.store
            .projects_by_name("spatial-memory-mcp")
            .expect("by name")
            .len(),
        2
    );

    let e = assert_code(
        run(&f.store, &params_for(fixture_path(SAMPLE), None), None),
        ErrorCode::InvalidInput,
    );
    assert!(e.message.contains("2 projects are named"), "{}", e.message);
    assert_eq!(f.store.memory_count().expect("count"), 0, "nothing written");
    assert_eq!(f.store.projects().expect("projects").len(), 2);
}

#[test]
fn s10_project_values_match_existing_projects_by_remote_then_name() {
    let f = fixture();
    let mut remote = identity("orders");
    remote.git_remote = Some("github.com/acme/orders".to_owned());
    let by_remote = f.store.upsert_project(&remote).expect("project");
    let by_name = project(&f.store, "billing-service");
    let path = write_json(
        &f,
        "export.json",
        &json!([
            {"id": "r-1", "content": "Matched by the remote.", "project": "github.com/acme/orders"},
            {"id": "r-2", "content": "Matched by the name.", "project": " billing-service "},
        ]),
    );
    let report = run(&f.store, &params_for(path, None), None).expect("migrate");
    assert!(
        report.projects_created.is_empty(),
        "{:?}",
        report.projects_created
    );
    assert_eq!(report.inserted, 2);
    let at = |raw: &str| {
        f.store
            .get_memory(&MemoryId::from_v1(raw).0)
            .expect("memory")
            .project_id
    };
    assert_eq!(at("r-1"), by_remote.id);
    assert_eq!(at("r-2"), by_name.id);
}

#[test]
fn s10_a_remote_value_creates_a_git_project_and_other_values_a_name_project() {
    let f = fixture();
    let here = identity("workdir");
    let path = write_json(
        &f,
        "export.json",
        &json!([
            {"id": "r-1", "content": "From a repository.", "project": "github.com/acme/billing"},
            {"id": "r-2", "content": "From a named project.", "project": "plain-name"},
            {"id": "r-3", "content": "Not a remote.", "project": "Some/Thing"},
            {"id": "r-4", "content": "The working directory's.", "project": "workdir"},
        ]),
    );
    let report = run(&f.store, &params_for(path, None), Some(&here)).expect("migrate");
    assert_eq!(
        report.projects_created,
        ["billing", "plain-name", "Some/Thing", "workdir"]
    );

    let git = f
        .store
        .project_by_identity("git:github.com/acme/billing")
        .expect("find")
        .expect("a git: project");
    assert_eq!(git.git_remote.as_deref(), Some("github.com/acme/billing"));
    assert_eq!(git.name, "billing");
    let named = f
        .store
        .project_by_identity("name:plain-name")
        .expect("find")
        .expect("a name: project");
    assert_eq!(named.git_remote, None);
    assert_eq!(named.name, "plain-name");
    assert!(
        f.store
            .project_by_identity("name:Some/Thing")
            .expect("find")
            .is_some()
    );
    let cwd = f
        .store
        .project_by_identity(&here.key())
        .expect("find")
        .expect("the working directory's project, by its identity");
    let r4 = f
        .store
        .get_memory(&MemoryId::from_v1("r-4").0)
        .expect("memory");
    assert_eq!(r4.project_id, cwd.id);
}

#[test]
fn s10_unstorable_records_are_issues_and_the_count_balances() {
    let f = fixture();
    let path = write_json(
        &f,
        "export.json",
        &json!([
            42,
            {"content": "No id.", "project": "p"},
            {"id": "r-2", "content": "   ", "project": "p"},
            {"id": "r-3", "content": "First copy of a text.", "project": "p"},
            {"id": "r-3", "content": "A repeated id.", "project": "p"},
            {"id": "r-5", "content": "  first copy of a TEXT.", "project": "p"},
            {
                "id": "r-6", "content": "Tags to clean.", "project": "p",
                "tags": ["ok", "a,b", "  "], "namespace": "name[space]",
                "created_at": "not a time", "weird": true
            },
            {"id": 17, "content": "A numeric id.", "project": "p"},
            {"id": "r-8", "content": "A project that is not a string.", "project": 5},
        ]),
    );
    let report = run(&f.store, &params_for(path, None), None).expect("migrate");
    assert_eq!(report.read, 9);
    assert_eq!(report.inserted, 3, "{report:?}");
    assert_balanced(&report);
    let at: Vec<u64> = report.issues.iter().map(|i| i.index).collect();
    assert_eq!(at, [0, 1, 2, 4, 5, 8]);
    let message = |index: u64| {
        report
            .issues
            .iter()
            .find(|i| i.index == index)
            .expect("an issue")
            .message
            .clone()
    };
    assert!(message(0).contains("not an object"));
    assert!(message(1).contains("no id"));
    assert!(message(2).contains("content is empty"));
    assert!(message(4).contains("appears earlier"));
    let first = MemoryId::from_v1("r-3").0;
    assert!(
        message(5).contains(&format!("already stored as memory {first}")),
        "{}",
        message(5)
    );
    assert!(message(8).contains("not a string"));
    let r3 = report.issues.iter().find(|i| i.index == 4).expect("issue");
    assert_eq!(r3.id.as_deref(), Some("r-3"));

    let r6 = f
        .store
        .get_memory(&MemoryId::from_v1("r-6").0)
        .expect("memory");
    assert_eq!(r6.tags, ["name-space-", "ok", "a-b"]);
    let imp = imported(&r6);
    assert_eq!(imp["namespace"], json!("name[space]"));
    assert_eq!(imp["tags"], json!(["ok", "a,b", "  "]));
    assert_eq!(imp["created_at"], json!("not a time"));
    assert_eq!(imp["weird"], json!(true));
    assert!(
        f.store
            .find_memory(&MemoryId::from_v1("17").0)
            .expect("find")
            .is_some()
    );
}

#[test]
fn s10_a_file_that_is_not_a_json_array_is_refused_before_writing() {
    let f = fixture();
    for (name, text) in [
        ("object.json", r#"{"not": "an array"}"#),
        (
            "cut.json",
            r#"[{"id": "a", "content": "Stored only if the file is whole.", "project": "p"},"#,
        ),
    ] {
        let path = f.path(name);
        fs::write(&path, text).expect("write");
        assert_code(
            run(&f.store, &params_for(path, None), None),
            ErrorCode::InvalidInput,
        );
    }
    assert_eq!(f.store.memory_count().expect("count"), 0);
    assert!(f.store.projects().expect("projects").is_empty());
}

const AWS_KEY: &str = "AKIAIOSFODNN7EXAMPLE";
const GITHUB_TOKEN: &str = "ghp_a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8";
const PASSWORD: &str = "hunter2xyz";

fn extra(m: &Memory) -> serde_json::Map<String, Value> {
    m.provenance
        .extra
        .as_ref()
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default()
}

#[track_caller]
fn assert_redacted(v: &Value) {
    let text = v.to_string();
    for secret in [AWS_KEY, GITHUB_TOKEN, PASSWORD] {
        assert!(!text.contains(secret), "{text}");
    }
    assert!(text.contains("[REDACTED:"), "{text}");
}

/// The lead's redaction decision: `migrate` redacts every string value it
/// stores in `provenance.extra` and `imported`, unknown fields included,
/// keys left as they are, and drops a tag that holds a secret (SPEC 9
/// "Secret redaction on write"; SPEC 10).
#[test]
fn s10_migrate_redacts_every_string_it_stores() {
    let f = fixture();
    let path = write_json(
        &f,
        "export.json",
        &json!([
            {
                "id": "m-1", "content": "Rotate the deploy keys monthly.", "project": "p",
                "source": "notes",
                "metadata": {"note": format!("rotated {AWS_KEY}"), "password": PASSWORD, "kind": "ops"}
            },
            {
                "id": "m-2", "content": "Cache entries expire hourly.", "project": "p",
                "metadata": json!({"password": PASSWORD, "kind": "cache"}).to_string()
            },
            {
                "id": "m-3", "content": "Builds run on every push.", "project": "p",
                "source": format!("pasted from {GITHUB_TOKEN}"),
                "metadata": format!("deployed with {AWS_KEY}")
            },
            {
                "id": "m-4", "content": "Queues drain at night.", "project": "p",
                "legacy": {"api": {"token": GITHUB_TOKEN}, "level": 3},
                "comment": format!("key {AWS_KEY}")
            },
            {
                "id": "m-5", "content": "Reports go out on Fridays.", "project": "p",
                "tags": ["reports", AWS_KEY, "weekly"]
            },
            {
                "id": "m-6", "content": "Alerts page the on-call engineer.", "project": "p",
                "tags": json!(["alerts", GITHUB_TOKEN]).to_string()
            },
            {
                "id": "m-7", "content": "Nothing secret is kept here.", "project": "p",
                "metadata": {"kind": "plain"}, "tags": ["plain"], "comment": "fine"
            },
            {
                "id": "m-8", "content": "Invoices are sent on the first.", "project": "p",
                "namespace": AWS_KEY, "tags": ["billing"]
            },
        ]),
    );
    let report = run(&f.store, &params_for(path, None), None).expect("migrate");
    assert_eq!(report.read, 8);
    assert_eq!(report.inserted, 8, "{report:?}");
    assert!(report.issues.is_empty(), "{report:?}");
    assert_balanced(&report);
    assert_eq!(
        report.redacted, 7,
        "m-1 to m-6 and m-8 had a secret redacted"
    );

    let get = |id: &str| {
        f.store
            .get_memory(&MemoryId::from_v1(id).0)
            .expect("memory")
    };

    // Metadata as an object: values redacted, keys kept.
    let m1 = extra(&get("m-1"));
    assert_eq!(m1["source"], json!("notes"));
    let meta = m1["metadata"].as_object().expect("metadata object");
    assert_eq!(meta["password"], json!("[REDACTED:password]"));
    assert_eq!(meta["kind"], json!("ops"));
    assert_redacted(&meta["note"]);

    // Metadata as a string holding JSON: redacted value by value.
    let m2 = extra(&get("m-2"));
    let text = m2["metadata"].as_str().expect("metadata string");
    let meta: Value = serde_json::from_str(text).expect("still JSON");
    assert_eq!(meta["password"], json!("[REDACTED:password]"));
    assert_eq!(meta["kind"], json!("cache"));

    // Metadata and source as plain text.
    let m3 = extra(&get("m-3"));
    assert_redacted(&m3["metadata"]);
    assert_redacted(&m3["source"]);
    assert!(
        m3["metadata"]
            .as_str()
            .expect("text")
            .starts_with("deployed with ")
    );

    // Unknown fields, at any depth, in `imported`.
    let m4 = imported(&get("m-4"));
    assert_redacted(&m4["legacy"]["api"]["token"]);
    assert_eq!(m4["legacy"]["level"], json!(3));
    assert_redacted(&m4["comment"]);

    // A tag that holds a secret is dropped; the original list goes to
    // `imported.tags`, redacted.
    let m5 = get("m-5");
    assert_eq!(m5.tags, ["reports", "weekly"]);
    let raw = imported(&m5)["tags"].clone();
    assert_eq!(raw.as_array().map(Vec::len), Some(3), "{raw}");
    assert_eq!(raw[0], json!("reports"));
    assert_eq!(raw[2], json!("weekly"));
    assert_redacted(&raw);
    let m6 = get("m-6");
    assert_eq!(m6.tags, ["alerts"]);
    assert_redacted(&imported(&m6)["tags"]);

    // A namespace that holds a secret is not made a tag; the original goes
    // to `imported.namespace`, redacted.
    let m8 = get("m-8");
    assert_eq!(m8.tags, ["billing"]);
    assert_redacted(&imported(&m8)["namespace"]);

    // Nothing to redact: stored as it came.
    let m7 = get("m-7");
    assert_eq!(m7.tags, ["plain"]);
    assert_eq!(extra(&m7)["metadata"], json!({"kind": "plain"}));
    assert_eq!(imported(&m7)["comment"], json!("fine"));

    // No secret is stored anywhere.
    let stored = format!("{:?}", all_memories(&f.store));
    for secret in [AWS_KEY, GITHUB_TOKEN, PASSWORD] {
        assert!(!stored.contains(secret), "{secret} is stored");
    }
}

/// B-181 (SPEC 10, 16; CAPTURE-RCA C2): `migrate` stores every record it
/// keeps as an active memory whose provenance source is `import`, whatever
/// the record says of its status or source, so a version 1 export never
/// brings back a capture that waits for the judge, nor a memory marked as
/// captured. This held before B-181; the test states it directly.
#[test]
fn s16_b181_migrate_never_stores_a_pending_or_captured_memory() {
    let f = fixture();
    let path = write_json(
        &f,
        "export.json",
        &json!([
            {
                "id": "r-1", "content": "The uploader retries a failed part three times.",
                "project": "p", "status": "pending", "source": "capture"
            },
            {
                "id": "r-2", "content": "The uploader keeps its parts in a scratch folder.",
                "project": "p", "source": "capture",
                "metadata": {"status": "pending", "provenance": {"source": "capture"}}
            },
            {
                "id": "r-3", "content": "The uploader signs every part.",
                "project": "p", "status": "superseded", "source": "hook"
            },
        ]),
    );
    let report = run(&f.store, &params_for(path, None), None).expect("migrate");
    assert_balanced(&report);
    assert_eq!((report.read, report.inserted), (3, 3), "{report:?}");
    for raw in ["r-1", "r-2", "r-3"] {
        let m = f
            .store
            .get_memory(&MemoryId::from_v1(raw).0)
            .expect("memory");
        assert_eq!(m.status, Status::Active, "{raw}");
        assert_eq!(m.provenance.source, ProvenanceSource::Import, "{raw}");
    }
    assert!(all_memories(&f.store).iter().all(|m| m.status.is_served()));
}
