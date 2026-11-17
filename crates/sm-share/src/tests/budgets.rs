//! Measured budgets at 10,000 items: sync change detection (SPEC section 9,
//! "Change detection before reading") and migration of a version 1 export
//! that still carries its vectors (SPEC section 10).
//!
//! The tests assert behaviour only: which files are read, what is stored and
//! what a second run skips. They print their timings to standard error, and
//! on Linux the process's peak resident memory; run them with
//! `cargo test -p sm-share --all-features --release budget -- --nocapture --test-threads=1`
//! to read the figures. No figure is asserted, so a slow machine cannot fail
//! them.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::json;
use sm_core::ids::MemoryId;
use sm_core::protocol::{MigrateParams, SyncParams, SyncReply};
use sm_core::time::Timestamp;
use sm_core::types::{MemoryType, Project, Status};
use sm_store::{MemoryListFilter, Store};

use super::{Repo, chunk_params, chunker, fixture, project, scanner};
use crate::format::{SHARED_DIR, SharedDoc};
use crate::git::Git;
use crate::migrate::{MigrateOptions, migrate_with};
use crate::sync::{SyncOptions, SyncStats, sync_with};

/// Items per measurement: the 10,000 of SPEC section 9 and section 10.
const BUDGET_ITEMS: usize = 10_000;

/// Dimensions of the version 1 vectors in a migrated export.
const V1_VECTOR_DIM: usize = 384;

fn timed<T>(what: &str, f: impl FnOnce() -> T) -> (T, Duration) {
    let start = Instant::now();
    let out = f();
    let took = start.elapsed();
    eprintln!("budget: {what}: {} ms", took.as_millis());
    (out, took)
}

/// The peak resident memory of this process so far, where the platform
/// reports it without extra code (`VmHWM` on Linux).
fn peak_memory() -> Option<String> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    status
        .lines()
        .find(|l| l.starts_with("VmHWM:"))
        .map(|l| l["VmHWM:".len()..].trim().to_owned())
}

fn report_peak(when: &str) {
    match peak_memory() {
        Some(peak) => eprintln!("budget: peak resident memory {when}: {peak}"),
        None => eprintln!("budget: peak resident memory {when}: not reported on this platform"),
    }
}

fn sync_run(
    store: &Store,
    project: &Project,
    repo: &Repo,
    dry_run: bool,
) -> (SyncReply, SyncStats) {
    sync_with(
        store,
        &chunker(),
        &chunk_params(),
        project,
        repo.root(),
        &SyncParams { dry_run },
        &SyncOptions {
            git_env: repo.git_env(),
            ..SyncOptions::default()
        },
    )
    .expect("sync")
}

/// Commits `files` (path, text) on `main` through one `git fast-import`
/// process, on top of the current `main` when `on_main` is set. The work
/// tree is not written: sync reads only the committed tree, and writing and
/// staging 10,000 files took 40 seconds on Windows.
fn fast_commit(repo: &Repo, message: &str, on_main: bool, files: &[(String, String)]) {
    fn data(out: &mut Vec<u8>, bytes: &[u8]) {
        out.extend_from_slice(format!("data {}\n", bytes.len()).as_bytes());
        out.extend_from_slice(bytes);
        out.push(b'\n');
    }
    let mut stream: Vec<u8> = Vec::new();
    stream.extend_from_slice(b"commit refs/heads/main\n");
    stream.extend_from_slice(b"committer Test Author <author@example.invalid> 1762000000 +0000\n");
    data(&mut stream, message.as_bytes());
    if on_main {
        stream.extend_from_slice(b"from refs/heads/main^0\n");
    }
    for (path, text) in files {
        stream.extend_from_slice(format!("M 100644 inline {path}\n").as_bytes());
        data(&mut stream, text.as_bytes());
    }

    let mut cmd = Command::new("git");
    cmd.args(["fast-import", "--quiet"])
        .current_dir(repo.root())
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    for var in [
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
    ] {
        cmd.env_remove(var);
    }
    for (k, v) in repo.git_env() {
        cmd.env(k, v);
    }
    let mut child = cmd.spawn().expect("start git fast-import");
    child
        .stdin
        .take()
        .expect("a pipe")
        .write_all(&stream)
        .expect("write the import stream");
    let out = child.wait_with_output().expect("wait for git fast-import");
    assert!(
        out.status.success(),
        "git fast-import failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// A one-chunk shared decision, about the size of a real one.
fn decision(i: usize) -> SharedDoc {
    SharedDoc {
        id: MemoryId::new_v7(),
        memory_type: MemoryType::Decision,
        status: Status::Active,
        date: Timestamp::parse_date("2026-11-01").expect("date"),
        supersedes: None,
        anchors: Vec::new(),
        tags: vec!["budget".to_owned()],
        commit: None,
        title: format!("Decision {i}: retry policy for service {}", i % 97),
        body: format!(
            "Service {} retries idempotent requests {} times with jittered backoff \
             starting at {} ms, because the upstream gateway drops about one request \
             in {} under load. Non-idempotent requests are never retried.",
            i % 97,
            2 + i % 4,
            50 + i % 200,
            500 + i
        ),
    }
}

#[test]
fn s9_budget_sync_reads_only_changed_files_among_10000() {
    let f = fixture();
    let p = project(&f.store, "team");
    let repo = Repo::init();

    let (docs, _) = timed("commit 10,000 shared files", || {
        let docs: Vec<SharedDoc> = (0..BUDGET_ITEMS).map(decision).collect();
        let files: Vec<(String, String)> = docs.iter().map(|d| (d.path(), d.render())).collect();
        fast_commit(&repo, "10,000 shared decisions", false, &files);
        docs
    });

    // A dry run lists, reads, parses and hashes every file and writes nothing.
    let ((dry, stats), _) = timed("dry-run sync of 10,000 new files", || {
        sync_run(&f.store, &p, &repo, true)
    });
    assert!(dry.dry_run);
    assert_eq!(stats.blobs_read, BUDGET_ITEMS as u64);
    assert_eq!(dry.added.len(), BUDGET_ITEMS);
    assert!(dry.errors.is_empty(), "{:?}", dry.errors);
    assert_eq!(f.store.memory_count().expect("count"), 0);

    let ((first, stats), _) = timed("first sync of 10,000 new files", || {
        sync_run(&f.store, &p, &repo, false)
    });
    assert_eq!(stats.blobs_read, BUDGET_ITEMS as u64);
    assert_eq!(first.added.len(), BUDGET_ITEMS);
    assert!(first.errors.is_empty(), "{:?}", first.errors);
    assert_eq!(f.store.memory_count().expect("count"), BUDGET_ITEMS as u64);

    // Nothing changed: the listing alone decides, and no blob is read.
    let ((second, stats), _) = timed("second sync, nothing changed", || {
        sync_run(&f.store, &p, &repo, false)
    });
    assert_eq!(stats.blobs_read, 0, "no file is read when nothing changed");
    assert_eq!(second.unchanged as usize, BUDGET_ITEMS);
    assert!(second.added.is_empty() && second.updated.is_empty() && second.removed.is_empty());
    assert!(second.errors.is_empty(), "{:?}", second.errors);

    // Where the time of a sync with nothing to read goes.
    let git = Git::new(repo.root(), repo.git_env());
    let head = git.head().expect("head").expect("a commit");
    let (entries, _) = timed("  part: git ls-tree -l of 10,000 files", || {
        git.ls_tree(&head, &format!("{SHARED_DIR}/"))
            .expect("ls-tree")
    });
    assert_eq!(entries.len(), BUDGET_ITEMS);
    let (listed, _) = timed("  part: loading the project's 10,000 memories", || {
        f.store
            .list_memories(&MemoryListFilter {
                project: Some(p.id),
                ..MemoryListFilter::default()
            })
            .expect("list")
    });
    assert_eq!(listed.len(), BUDGET_ITEMS);

    // One file changed: exactly that blob is read.
    let mut changed = docs[17].clone();
    changed.body.push_str(" Reviewed again in November.");
    fast_commit(
        &repo,
        "one decision changed",
        true,
        &[(changed.path(), changed.render())],
    );
    let ((third, stats), _) = timed("third sync, one file changed", || {
        sync_run(&f.store, &p, &repo, false)
    });
    assert_eq!(stats.blobs_read, 1);
    assert_eq!(third.updated, vec![changed.id]);
    assert_eq!(third.unchanged as usize, BUDGET_ITEMS - 1);
    assert!(third.errors.is_empty(), "{:?}", third.errors);
    report_peak("after the syncs");
}

/// Writes a version 1 export of `n` records, each with a vector of
/// [`V1_VECTOR_DIM`] floats, one record at a time; returns its size in bytes.
fn write_v1_export(path: &Path, n: usize) -> u64 {
    let mut w = BufWriter::new(File::create(path).expect("create the export"));
    w.write_all(b"[").expect("write");
    for i in 0..n {
        if i > 0 {
            w.write_all(b",\n").expect("write");
        }
        let vector: Vec<f64> = (0..V1_VECTOR_DIM)
            .map(|j| {
                let x = ((i * 7919 + j * 104_729) % 1_000_003) as f64;
                x / 1_000_003.0 - 0.5
            })
            .collect();
        let worker = ["ingest", "billing", "search"][i % 3];
        let namespace = ["decisions", "patterns", "notes"][i % 3];
        let project = ["alpha", "beta", "gamma"][i % 3];
        let content = format!(
            "Record {i}: the {worker} worker batches {} writes per transaction because \
             single-row commits saturated the disk at {} operations per second.",
            16 + i % 48,
            900 + i % 300
        );
        let record = json!({
            "id": format!("mem-{i:05}"),
            "content": content,
            "namespace": namespace,
            "tags": ["budget", format!("t{}", i % 11)],
            "importance": 0.5,
            "source": "budget",
            "created_at": "2025-06-01T12:00:00",
            "updated_at": "2025-06-02T08:30:00",
            "project": project,
            "vector": vector,
        });
        serde_json::to_writer(&mut w, &record).expect("write a record");
    }
    w.write_all(b"]\n").expect("write");
    w.flush().expect("flush");
    drop(w);
    fs::metadata(path).expect("the export").len()
}

#[test]
fn s10_budget_migrate_streams_a_10000_record_export_with_vectors() {
    let f = fixture();
    let path = f.path("v1_export_10000.json");
    let (bytes, _) = timed("write a 10,000-record export with vectors", || {
        write_v1_export(&path, BUDGET_ITEMS)
    });
    eprintln!(
        "budget: export size {:.1} MB ({bytes} bytes)",
        bytes as f64 / 1_000_000.0
    );
    report_peak("before migrate");

    let params = MigrateParams {
        from: path.to_string_lossy().into_owned(),
        default_project: None,
    };
    let migrate = || {
        migrate_with(
            &f.store,
            &chunker(),
            &scanner(),
            &chunk_params(),
            &params,
            None,
            &MigrateOptions::default(),
        )
        .expect("migrate")
    };

    let (first, _) = timed("migrate 10,000 records", migrate);
    assert_eq!(first.read, BUDGET_ITEMS as u64);
    assert_eq!(first.inserted, BUDGET_ITEMS as u64);
    assert_eq!(first.skipped_existing, 0);
    assert!(first.issues.is_empty(), "{:?}", first.issues);
    assert_eq!(first.projects_created.len(), 3);
    assert_eq!(first.pending_embeddings, BUDGET_ITEMS as u64);
    assert_eq!(f.store.memory_count().expect("count"), BUDGET_ITEMS as u64);
    report_peak("after migrate");

    // A second run of the same file stores nothing new.
    let (second, _) = timed("migrate the same 10,000 records again", migrate);
    assert_eq!(second.read, BUDGET_ITEMS as u64);
    assert_eq!(second.inserted, 0);
    assert_eq!(second.skipped_existing, BUDGET_ITEMS as u64);
    assert!(second.issues.is_empty(), "{:?}", second.issues);
    assert_eq!(f.store.memory_count().expect("count"), BUDGET_ITEMS as u64);
    report_peak("after the second migrate");
}
