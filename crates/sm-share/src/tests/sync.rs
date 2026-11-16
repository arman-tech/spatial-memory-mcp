//! `sync` (SPEC section 9, Team sharing rules 3 to 5, "Change detection
//! before reading" and "Re-embed only when the text changes"; B-78).

use std::fs;

use sm_core::ids::MemoryId;
use sm_core::protocol::{ShareParams, SyncParams, SyncReply};
use sm_core::time::Timestamp;
use sm_core::traits::{MemoryFilter, QueryScope};
use sm_core::types::{
    AnchorKind, AnchorRef, MemoryType, NewMemory, Origin, Project, Provenance, ProvenanceSource,
    Status,
};
use sm_store::{MemoryInput, Store, SyncMark};

use super::{
    Fixture, Repo, add_note, all_memories, chunk_params, chunker, embed_all_pending, fixture,
    insert, pending_chunks, project, scanner,
};
use crate::format::{GITATTRIBUTES, SHARED_DIR, SharedDoc, date_of};
use crate::share;
use crate::sync::{
    MAX_SHARED_FILE_BYTES, SyncOptions, SyncStats, UNRESOLVED_SUPERSEDES, sync_with,
};

/// A shared file of a decision dated 2026-11-01.
fn doc(title: &str, body: &str) -> SharedDoc {
    SharedDoc {
        id: MemoryId::new_v7(),
        memory_type: MemoryType::Decision,
        status: Status::Active,
        date: Timestamp::parse_date("2026-11-01").expect("date"),
        supersedes: None,
        anchors: Vec::new(),
        tags: Vec::new(),
        commit: None,
        title: title.to_owned(),
        body: body.to_owned(),
    }
}

/// Writes a document at its own path in the work tree; returns the path.
fn put(repo: &Repo, d: &SharedDoc) -> String {
    let path = d.path();
    repo.write(&path, &d.render());
    path
}

fn run(store: &Store, project: &Project, repo: &Repo, dry_run: bool) -> (SyncReply, SyncStats) {
    sync_with(
        store,
        &chunker(),
        &chunk_params(),
        project,
        repo.root(),
        &SyncParams { dry_run },
        &SyncOptions {
            git_env: repo.git_env(),
        },
    )
    .expect("sync")
}

fn blob_id(repo: &Repo, path: &str) -> String {
    repo.git(&["rev-parse", &format!("HEAD:{path}")])
        .trim()
        .to_owned()
}

fn keyword_hits(store: &Store, project: &Project, text: &str) -> Vec<MemoryId> {
    store
        .keyword_search(
            text,
            QueryScope::Project(project.id),
            &MemoryFilter::default(),
            10,
        )
        .expect("keyword search")
        .into_iter()
        .map(|h| h.memory_id)
        .collect()
}

fn pending(store: &Store) -> u64 {
    store
        .index_status()
        .expect("index status")
        .pending_embeddings
}

fn chunk_total(store: &Store) -> u64 {
    store.index_status().expect("index status").chunks
}

/// A store with project `team` and an empty repository.
fn setup() -> (Fixture, Project, Repo) {
    let f = fixture();
    let p = project(&f.store, "team");
    (f, p, Repo::init())
}

#[test]
fn s9_sync_adds_a_new_file_as_shared_pending_and_marked() {
    let (f, p, repo) = setup();
    let mut d = doc(
        "Use optimistic concurrency on dbo.Orders",
        "Compare the rowversion before every update.\n\n## Why\n\nLost updates at peak.",
    );
    d.tags = vec!["concurrency".to_owned(), "sql".to_owned()];
    d.anchors = vec![AnchorRef::new(AnchorKind::Table, "dbo.Orders").expect("anchor")];
    let path = put(&repo, &d);
    let head = repo.commit_all("share a decision");

    let (reply, stats) = run(&f.store, &p, &repo, false);
    assert_eq!(reply.added, vec![d.id]);
    assert!(reply.updated.is_empty() && reply.removed.is_empty());
    assert_eq!(reply.unchanged, 0);
    assert!(reply.errors.is_empty(), "{:?}", reply.errors);
    assert!(!reply.dry_run);
    assert_eq!(stats.blobs_read, 1);

    let m = f.store.get_memory(&d.id).expect("the memory");
    assert_eq!(m.project_id, p.id);
    assert_eq!(m.origin, Origin::Shared);
    assert_eq!(m.status, Status::Active);
    assert_eq!(m.memory_type, MemoryType::Decision);
    assert_eq!(m.title.as_deref(), Some(d.title.as_str()));
    assert_eq!(m.content, d.body);
    assert_eq!(m.tags, d.tags);
    assert_eq!(m.anchors, d.anchors);
    assert_eq!(m.created_at, d.date);
    assert_eq!(m.provenance.source, ProvenanceSource::Sync);

    let mark = SyncMark::of(&m).expect("sync recorded a mark");
    assert_eq!(mark.file_path, path);
    assert_eq!(mark.commit_sha.as_deref(), Some(head.as_str()));
    assert_eq!(mark.blob_id, blob_id(&repo, &path));
    assert_eq!(mark.text_hash, d.text_hash());

    // Pending chunks: found by keyword search, not yet by vector.
    assert!(pending(&f.store) > 0);
    assert_eq!(pending(&f.store), chunk_total(&f.store));
    assert_eq!(f.store.memory_vector(&d.id).expect("vector"), None);
    assert_eq!(keyword_hits(&f.store, &p, "rowversion"), vec![d.id]);
    assert_eq!(embed_all_pending(&f.store), chunk_total(&f.store));
    assert!(f.store.memory_vector(&d.id).expect("vector").is_some());
}

#[test]
fn s9_sync_does_not_read_an_unchanged_file_again() {
    let (f, p, repo) = setup();
    let a = doc("First rule", "Always pin the toolchain.");
    let b = doc("Second rule", "Never tune a default to pass a test.");
    put(&repo, &a);
    put(&repo, &b);
    repo.commit_all("two files");
    let (reply, stats) = run(&f.store, &p, &repo, false);
    assert_eq!(reply.added.len(), 2);
    assert_eq!(stats.blobs_read, 2);

    // Nothing changed: no blob is read.
    let (reply, stats) = run(&f.store, &p, &repo, false);
    assert_eq!(reply.unchanged, 2);
    assert!(reply.added.is_empty() && reply.updated.is_empty() && reply.removed.is_empty());
    assert_eq!(stats.blobs_read, 0);

    // A later commit that touches nothing shared reads nothing either, and
    // one new file is the only blob read.
    repo.write("README.md", "not shared\n");
    repo.commit_all("unrelated");
    let (reply, stats) = run(&f.store, &p, &repo, false);
    assert_eq!(reply.unchanged, 2);
    assert_eq!(stats.blobs_read, 0);

    let c = doc("Third rule", "Write each log line in one call.");
    put(&repo, &c);
    repo.commit_all("a third file");
    let (reply, stats) = run(&f.store, &p, &repo, false);
    assert_eq!(reply.added, vec![c.id]);
    assert_eq!(reply.unchanged, 2);
    assert_eq!(stats.blobs_read, 1);
}

#[test]
fn s9_sync_metadata_change_revises_without_new_chunks() {
    let (f, p, repo) = setup();
    let mut d = doc("Retry policy", "Three attempts with jitter.");
    let path = put(&repo, &d);
    repo.commit_all("share");
    run(&f.store, &p, &repo, false);
    embed_all_pending(&f.store);
    let chunks = chunk_total(&f.store);
    let vector = f.store.memory_vector(&d.id).expect("vector");
    assert!(vector.is_some());

    d.tags = vec!["resilience".to_owned()];
    d.anchors = vec![AnchorRef::new(AnchorKind::Service, "order-api").expect("anchor")];
    d.memory_type = MemoryType::Convention;
    repo.write(&path, &d.render());
    repo.commit_all("tag it");

    let (reply, stats) = run(&f.store, &p, &repo, false);
    assert_eq!(reply.updated, vec![d.id]);
    assert_eq!(reply.unchanged, 0);
    assert!(reply.errors.is_empty(), "{:?}", reply.errors);
    assert_eq!(stats.blobs_read, 1);
    assert_eq!(pending(&f.store), 0, "no new chunks for a metadata change");
    assert_eq!(chunk_total(&f.store), chunks);
    assert_eq!(f.store.memory_vector(&d.id).expect("vector"), vector);

    let m = f.store.get_memory(&d.id).expect("memory");
    assert_eq!(m.tags, d.tags);
    assert_eq!(m.anchors, d.anchors);
    assert_eq!(m.memory_type, MemoryType::Convention);
    let mark = SyncMark::of(&m).expect("mark");
    assert_eq!(
        mark.blob_id,
        blob_id(&repo, &path),
        "the new blob is marked"
    );

    let (reply, stats) = run(&f.store, &p, &repo, false);
    assert_eq!(reply.unchanged, 1);
    assert_eq!(stats.blobs_read, 0);
}

#[test]
fn s9_sync_text_change_gives_new_pending_chunks() {
    let (f, p, repo) = setup();
    let mut d = doc("Cache policy", "Cache the price list for five minutes.");
    let path = put(&repo, &d);
    repo.commit_all("share");
    run(&f.store, &p, &repo, false);
    embed_all_pending(&f.store);
    assert_eq!(pending(&f.store), 0);

    d.title = "Cache policy, revised".to_owned();
    d.body = "Cache the catalogue for one minute.".to_owned();
    repo.write(&path, &d.render());
    repo.commit_all("revise the text");

    let (reply, _) = run(&f.store, &p, &repo, false);
    assert_eq!(reply.updated, vec![d.id]);
    assert!(reply.errors.is_empty(), "{:?}", reply.errors);
    assert!(pending(&f.store) > 0, "the new text waits for embedding");
    let m = f.store.get_memory(&d.id).expect("memory");
    assert_eq!(m.content, d.body);
    assert_eq!(m.title.as_deref(), Some("Cache policy, revised"));
    assert_eq!(SyncMark::of(&m).expect("mark").text_hash, d.text_hash());
    assert_eq!(keyword_hits(&f.store, &p, "catalogue"), vec![d.id]);
    assert!(keyword_hits(&f.store, &p, "price").is_empty());
}

#[test]
fn s9_sync_deleted_file_retracts_and_restored_file_reapplies() {
    let (f, p, repo) = setup();
    let d = doc(
        "Feature flags",
        "Flags are removed one release after launch.",
    );
    let path = put(&repo, &d);
    repo.commit_all("share");
    run(&f.store, &p, &repo, false);
    let text = repo.read(&path);

    repo.remove(&path);
    repo.commit_all("remove it");
    let (reply, _) = run(&f.store, &p, &repo, false);
    assert_eq!(reply.removed, vec![d.id]);
    assert!(reply.added.is_empty() && reply.updated.is_empty());
    let m = f.store.get_memory(&d.id).expect("retracted, not deleted");
    assert_eq!(m.status, Status::Retracted);

    // Retracting twice reports nothing new.
    let (reply, _) = run(&f.store, &p, &repo, false);
    assert!(reply.removed.is_empty());

    // The same bytes come back: same blob id, read again, reapplied.
    repo.write(&path, &text);
    repo.commit_all("restore it");
    let (reply, stats) = run(&f.store, &p, &repo, false);
    assert_eq!(reply.updated, vec![d.id]);
    assert!(reply.removed.is_empty() && reply.added.is_empty());
    assert_eq!(stats.blobs_read, 1);
    assert_eq!(
        f.store.get_memory(&d.id).expect("memory").status,
        Status::Active
    );

    let (reply, stats) = run(&f.store, &p, &repo, false);
    assert_eq!(reply.unchanged, 1);
    assert_eq!(stats.blobs_read, 0);
}

#[test]
fn s9_sync_adopts_the_authors_copy_when_hashes_match() {
    let (f, p, repo) = setup();
    let id = add_note(
        &f.store,
        &p,
        "Index foreign keys before a bulk load.\n\nOtherwise the load takes hours.",
    );
    let written = share(
        &f.store,
        &scanner(),
        &p,
        repo.root(),
        &ShareParams { ids: vec![id] },
    )
    .expect("share");
    assert_eq!(written.written.len(), 1);
    repo.commit_all("share my note");
    let before = f.store.memory_count().expect("count");

    let (reply, _) = run(&f.store, &p, &repo, false);
    assert_eq!(reply.updated, vec![id], "the author's copy is adopted");
    assert!(reply.added.is_empty(), "no second copy");
    assert!(reply.errors.is_empty(), "{:?}", reply.errors);
    assert_eq!(f.store.memory_count().expect("count"), before);
    let m = f.store.get_memory(&id).expect("memory");
    assert_eq!(m.origin, Origin::Shared);
    let mark = SyncMark::of(&m).expect("mark");
    assert_eq!(mark.file_path, written.written[0].path);

    let (reply, stats) = run(&f.store, &p, &repo, false);
    assert_eq!(reply.unchanged, 1);
    assert_eq!(stats.blobs_read, 0);
}

#[test]
fn s9_sync_reports_the_authors_copy_when_hashes_differ() {
    let (f, p, repo) = setup();
    let id = add_note(&f.store, &p, "Rotate the signing key every quarter.");
    let mut d = doc(
        "Rotate the signing key",
        "Rotate the signing key every month.",
    );
    d.id = id;
    let path = put(&repo, &d);
    repo.commit_all("a different text under my id");

    let (reply, _) = run(&f.store, &p, &repo, false);
    assert!(reply.updated.is_empty() && reply.added.is_empty());
    assert_eq!(reply.errors.len(), 1);
    assert_eq!(reply.errors[0].path, path);
    assert!(
        reply.errors[0].message.contains("it was not adopted"),
        "{}",
        reply.errors[0].message
    );
    let m = f.store.get_memory(&id).expect("memory");
    assert_eq!(m.origin, Origin::Local);
    assert_eq!(m.content, "Rotate the signing key every quarter.");
    assert!(SyncMark::of(&m).is_none());
}

#[test]
fn s9_sync_reports_an_id_in_two_files_and_applies_neither() {
    let (f, p, repo) = setup();
    let d = doc("Twice written", "The same id in two files.");
    let first = put(&repo, &d);
    let second = format!("{SHARED_DIR}/note/copy--{}.md", d.id.last8());
    repo.write(&second, &d.render());
    let other = doc("Fine", "A file that is applied.");
    put(&repo, &other);
    repo.commit_all("a doubled id");

    let (reply, _) = run(&f.store, &p, &repo, false);
    assert_eq!(reply.added, vec![other.id]);
    let mut paths: Vec<&str> = reply.errors.iter().map(|e| e.path.as_str()).collect();
    paths.sort_unstable();
    let mut expected = vec![first.as_str(), second.as_str()];
    expected.sort_unstable();
    assert_eq!(paths, expected);
    for e in &reply.errors {
        assert!(e.message.contains("is in 2 files"), "{}", e.message);
        assert!(e.message.contains(&d.id.to_string()));
    }
    assert!(f.store.find_memory(&d.id).expect("find").is_none());
}

#[test]
fn s9_sync_reports_an_id_of_another_project() {
    let (f, p, repo) = setup();
    let other = project(&f.store, "other");
    let id = add_note(&f.store, &other, "A memory of another project.");
    let mut d = doc("Borrowed id", "This file reuses another project's id.");
    d.id = id;
    let path = put(&repo, &d);
    repo.commit_all("a foreign id");

    let (reply, _) = run(&f.store, &p, &repo, false);
    assert!(reply.added.is_empty() && reply.updated.is_empty());
    assert_eq!(reply.errors.len(), 1);
    assert_eq!(reply.errors[0].path, path);
    assert!(
        reply.errors[0]
            .message
            .contains("belongs to project \"other\""),
        "{}",
        reply.errors[0].message
    );
    let m = f.store.get_memory(&id).expect("memory");
    assert_eq!(m.project_id, other.id);
    assert_eq!(m.content, "A memory of another project.");
}

#[test]
fn s9_sync_reports_text_already_stored_under_another_id() {
    let (f, p, repo) = setup();
    let local = add_note(&f.store, &p, "Deploy on Tuesdays only.");
    let d = doc("Deploy days", "Deploy on Tuesdays only.");
    let path = put(&repo, &d);
    repo.commit_all("same text, new id");

    let (reply, _) = run(&f.store, &p, &repo, false);
    assert!(reply.added.is_empty());
    assert_eq!(reply.errors.len(), 1);
    assert_eq!(reply.errors[0].path, path);
    assert!(
        reply.errors[0]
            .message
            .contains(&format!("already stored as memory {local}")),
        "{}",
        reply.errors[0].message
    );
    assert!(f.store.find_memory(&d.id).expect("find").is_none());
}

#[test]
fn s9_sync_reports_a_malformed_file_and_applies_the_rest() {
    let (f, p, repo) = setup();
    let good = doc("Good file", "This one is applied.");
    put(&repo, &good);
    let broken = format!("{SHARED_DIR}/decision/broken--00000001.md");
    repo.write(&broken, "no front matter here\n");
    let no_body = doc("Title only", "");
    let no_body_path = put(&repo, &no_body);
    let text_file = format!("{SHARED_DIR}/notes.txt");
    repo.write(&text_file, "not a memory\n");
    let not_utf8 = format!("{SHARED_DIR}/decision/latin1--00000002.md");
    fs::write(repo.root().join(&not_utf8), b"---\nschema: 1\n\xff\xfe\n").expect("write");
    let big = doc("Too large", &"x".repeat(MAX_SHARED_FILE_BYTES as usize + 1));
    let big_path = put(&repo, &big);
    repo.write(&format!("{SHARED_DIR}/.gitattributes"), GITATTRIBUTES);
    repo.commit_all("one good file and several bad ones");

    let (reply, stats) = run(&f.store, &p, &repo, false);
    assert_eq!(reply.added, vec![good.id]);
    let mut issues: Vec<(String, String)> = reply
        .errors
        .iter()
        .map(|e| (e.path.clone(), e.message.clone()))
        .collect();
    let sorted = {
        let mut s = issues.clone();
        s.sort();
        s
    };
    assert_eq!(issues, sorted, "errors are sorted by path");
    let expect = [
        (broken.as_str(), "the file does not parse: line 1"),
        (no_body_path.as_str(), "has no body"),
        (text_file.as_str(), "not a .md file"),
        (not_utf8.as_str(), "not valid UTF-8"),
        (big_path.as_str(), "byte limit"),
    ];
    assert_eq!(reply.errors.len(), expect.len(), "{:?}", reply.errors);
    for (path, part) in expect {
        let at = issues
            .iter()
            .position(|(p, _)| p == path)
            .unwrap_or_else(|| panic!("{path} is reported"));
        let (_, message) = issues.remove(at);
        assert!(message.contains(part), "{path}: {message}");
    }
    // The oversized file is skipped from its listed size, never read: the
    // good, broken, bodiless and non-UTF-8 files are.
    assert_eq!(stats.blobs_read, 4);
}

#[test]
fn s9_sync_ignores_uncommitted_edits() {
    let (f, p, repo) = setup();
    let mut d = doc("Committed rule", "The committed text.");
    let path = put(&repo, &d);
    repo.commit_all("share");
    run(&f.store, &p, &repo, false);

    d.body = "An edit that was never committed.".to_owned();
    repo.write(&path, &d.render());
    let staged = doc("Staged only", "Added to the index, not committed.");
    put(&repo, &staged);
    repo.git(&["add", "-A"]);
    let loose = doc("Untracked", "Never added.");
    put(&repo, &loose);

    let (reply, stats) = run(&f.store, &p, &repo, false);
    assert_eq!(reply.unchanged, 1);
    assert!(reply.added.is_empty() && reply.updated.is_empty() && reply.removed.is_empty());
    assert!(reply.errors.is_empty(), "{:?}", reply.errors);
    assert_eq!(stats.blobs_read, 0);
    assert_eq!(
        f.store.get_memory(&d.id).expect("memory").content,
        "The committed text."
    );
    assert!(f.store.find_memory(&staged.id).expect("find").is_none());
}

#[test]
fn s9_sync_of_a_repository_with_no_commits_syncs_nothing() {
    let (f, p, repo) = setup();
    // A shared memory of this project that a sync must not retract.
    let id = add_note(&f.store, &p, "Shared earlier from another clone.");
    let mark = SyncMark {
        file_path: format!("{SHARED_DIR}/note/shared-earlier--00000003.md"),
        commit_sha: None,
        blob_id: "0".repeat(40),
        text_hash: "0".repeat(64),
    };
    f.store.record_sync(&id, &mark).expect("mark it shared");
    put(
        &repo,
        &doc("Not committed", "The repository has no commit."),
    );
    let before = all_memories(&f.store);

    let (reply, stats) = run(&f.store, &p, &repo, false);
    assert_eq!(
        reply,
        SyncReply {
            dry_run: false,
            added: Vec::new(),
            updated: Vec::new(),
            removed: Vec::new(),
            unchanged: 0,
            errors: Vec::new(),
        }
    );
    assert_eq!(stats, SyncStats::default());
    assert_eq!(all_memories(&f.store), before);
}

#[test]
fn s9_sync_dry_run_writes_nothing() {
    let (f, p, repo) = setup();
    let mut b = doc("Kept and changed", "This file changes later.");
    let c = doc("Removed later", "This file is deleted later.");
    let b_path = put(&repo, &b);
    let c_path = put(&repo, &c);
    repo.commit_all("two files");
    run(&f.store, &p, &repo, false);

    b.tags = vec!["changed".to_owned()];
    repo.write(&b_path, &b.render());
    repo.remove(&c_path);
    let a = doc("New file", "This file is new.");
    put(&repo, &a);
    repo.commit_all("change, remove, add");

    let memories = all_memories(&f.store);
    let index = f.store.index_status().expect("index status");
    let (dry, _) = run(&f.store, &p, &repo, true);
    assert!(dry.dry_run);
    assert_eq!(dry.added, vec![a.id]);
    assert_eq!(dry.updated, vec![b.id]);
    assert_eq!(dry.removed, vec![c.id]);
    assert!(dry.errors.is_empty(), "{:?}", dry.errors);
    assert_eq!(all_memories(&f.store), memories, "a dry run writes nothing");
    assert_eq!(f.store.index_status().expect("index status"), index);

    let (real, _) = run(&f.store, &p, &repo, false);
    assert_eq!(
        real,
        SyncReply {
            dry_run: false,
            ..dry
        },
        "the dry run foretold the real run"
    );
    assert_eq!(
        f.store.get_memory(&c.id).expect("memory").status,
        Status::Retracted
    );
}

#[test]
fn s9_sync_inserts_a_supersedes_target_first_and_marks_it_superseded() {
    let (f, p, repo) = setup();
    // The superseding file sorts first in the listing ("alpha" < "zulu").
    let old = doc("Zulu older rule", "Retry five times.");
    let mut new = doc("Alpha newer rule", "Retry three times with jitter.");
    new.supersedes = Some(old.id);
    put(&repo, &new);
    let old_path = put(&repo, &old);
    repo.commit_all("a rule and the rule it replaces");

    let (reply, _) = run(&f.store, &p, &repo, false);
    let mut both = vec![old.id, new.id];
    both.sort();
    assert_eq!(reply.added, both);
    assert!(reply.errors.is_empty(), "{:?}", reply.errors);
    let n = f.store.get_memory(&new.id).expect("new");
    assert_eq!(n.supersedes_id, Some(old.id));
    assert!(
        n.provenance
            .extra
            .as_ref()
            .and_then(|e| e.get(UNRESOLVED_SUPERSEDES))
            .is_none()
    );
    let o = f.store.get_memory(&old.id).expect("old");
    assert_eq!(
        o.status,
        Status::Superseded,
        "its own active file does not undo it"
    );

    let (reply, _) = run(&f.store, &p, &repo, false);
    assert_eq!(reply.unchanged, 2);

    // A metadata change of the target's file keeps it superseded.
    let mut old2 = old.clone();
    old2.tags = vec!["retries".to_owned()];
    repo.write(&old_path, &old2.render());
    repo.commit_all("tag the old rule");
    let (reply, _) = run(&f.store, &p, &repo, false);
    assert_eq!(reply.updated, vec![old.id]);
    let o = f.store.get_memory(&old.id).expect("old");
    assert_eq!(o.status, Status::Superseded);
    assert_eq!(o.tags, old2.tags);
}

#[test]
fn s9_sync_keeps_an_unresolved_supersedes_until_the_target_arrives() {
    let (f, p, repo) = setup();
    let old = doc("Older rule", "Log at debug level.");
    let mut new = doc("Newer rule", "Log at info level.");
    new.supersedes = Some(old.id);
    put(&repo, &new);
    repo.commit_all("the newer rule only");

    let (reply, _) = run(&f.store, &p, &repo, false);
    assert_eq!(reply.added, vec![new.id]);
    let n = f.store.get_memory(&new.id).expect("new");
    assert_eq!(n.supersedes_id, None);
    let kept = n
        .provenance
        .extra
        .as_ref()
        .and_then(|e| e.get(UNRESOLVED_SUPERSEDES))
        .and_then(|v| v.as_str())
        .map(str::to_owned);
    assert_eq!(kept, Some(old.id.to_string()));

    put(&repo, &old);
    repo.commit_all("the older rule arrives");
    let (reply, _) = run(&f.store, &p, &repo, false);
    assert_eq!(reply.added, vec![old.id]);
    assert_eq!(reply.unchanged, 1);
    assert_eq!(
        f.store.get_memory(&old.id).expect("old").status,
        Status::Superseded
    );
}

/// Share from one store, commit, sync into another: the memory arrives
/// with the same title, body, type, tags and anchors.
#[test]
fn s9_share_commit_sync_carries_a_memory_between_stores() {
    let author = fixture();
    let reader = fixture();
    let repo = Repo::init();
    let p1 = project(&author.store, "team");
    let p2 = project(&reader.store, "team");

    let content = "Run schema migrations outside peak hours.\n\n```sql\nALTER TABLE dbo.Orders \
                   ADD Version rowversion;\n```";
    let mut memory = NewMemory::new(p1.id, content, Provenance::new(ProvenanceSource::Agent));
    memory.memory_type = MemoryType::Pitfall;
    memory.title = Some("Peak-hour migrations lock dbo.Orders".to_owned());
    memory.tags = vec!["migrations".to_owned(), "sql".to_owned()];
    memory.anchors = vec![
        AnchorRef::new(AnchorKind::Table, "dbo.Orders").expect("anchor"),
        AnchorRef::new(AnchorKind::Service, "order-api").expect("anchor"),
    ];
    let id = insert(
        &author.store,
        MemoryInput {
            memory,
            chunks: pending_chunks(content),
        },
    );
    let reply = share(
        &author.store,
        &scanner(),
        &p1,
        repo.root(),
        &ShareParams { ids: vec![id] },
    )
    .expect("share");
    assert!(reply.refused.is_empty(), "{:?}", reply.refused);
    repo.commit_all("share a pitfall");

    let (synced, _) = run(&reader.store, &p2, &repo, false);
    assert_eq!(synced.added, vec![id]);
    assert!(synced.errors.is_empty(), "{:?}", synced.errors);

    let a = author.store.get_memory(&id).expect("author's memory");
    let b = reader.store.get_memory(&id).expect("reader's memory");
    assert_eq!(b.title, a.title);
    assert_eq!(b.content, a.content);
    assert_eq!(b.memory_type, a.memory_type);
    assert_eq!(b.tags, a.tags);
    assert_eq!(b.anchors, a.anchors);
    assert_eq!(b.status, Status::Active);
    assert_eq!(b.origin, Origin::Shared);
    assert_eq!(b.project_id, p2.id);
    assert_eq!(b.created_at, date_of(a.created_at));
    assert_eq!(b.content_hash, a.content_hash);
    assert_eq!(keyword_hits(&reader.store, &p2, "rowversion"), vec![id]);
}
