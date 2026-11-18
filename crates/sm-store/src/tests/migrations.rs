//! SPEC section 4, Migrations: forward only, each in its own transaction, a
//! backup written before an existing database is migrated, newer schemas
//! refused. B-157: schema version 2 sets every automatic capture pending.
//! B-163: schema version 3 adds the capture state, empty. B-167: schema
//! version 4 adds the judge's queue and usage, empty. B-178: schema
//! version 5 records when the judge last failed on a capture.
//!
//! The framework tests add a probe migration one past [`SCHEMA_VERSION`] to
//! a database at the current version.

use rusqlite::Transaction;
use sm_core::error::Result;
use sm_core::types::Status;

use super::*;
use crate::SCHEMA_VERSION;
use crate::schema::{MIGRATIONS, Migration};

/// The probe's version: one past the current schema.
const NEXT: u32 = SCHEMA_VERSION + 1;

fn probe(tx: &Transaction<'_>) -> Result<()> {
    tx.execute_batch(
        "CREATE TABLE migration_probe (x INTEGER NOT NULL) STRICT;
         INSERT INTO migration_probe(x) SELECT count(*) FROM memories;",
    )
    .ctx("the probe migration")
}

fn probe_fails(tx: &Transaction<'_>) -> Result<()> {
    tx.execute_batch("CREATE TABLE migration_probe (x INTEGER NOT NULL) STRICT;")
        .ctx("the probe migration")?;
    Err(CoreError::storage("the probe migration failed on purpose"))
}

/// Every real migration, then the probe at [`NEXT`].
fn with_next(apply: fn(&Transaction<'_>) -> Result<()>) -> Vec<Migration> {
    MIGRATIONS
        .iter()
        .map(|m| Migration {
            version: m.version,
            apply: m.apply,
        })
        .chain(std::iter::once(Migration {
            version: NEXT,
            apply,
        }))
        .collect()
}

fn backup_files(dir: &Path) -> Vec<PathBuf> {
    match std::fs::read_dir(dir) {
        Ok(entries) => {
            let mut files: Vec<PathBuf> = entries
                .map(|e| e.expect("directory entry").path())
                .filter(|p| p.is_file())
                .collect();
            files.sort();
            files
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(e) => panic!("reading {}: {e}", dir.display()),
    }
}

fn has_table(store: &Store, name: &str) -> bool {
    store
        .read("reading the schema", |tx| {
            tx.query_row(
                "SELECT EXISTS (SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name = ?1)",
                [name],
                |r| r.get(0),
            )
            .ctx("reading the schema")
        })
        .expect("read")
}

/// A database at the current version holding one memory, closed, in its
/// own folder.
fn current_database() -> (TempDir, PathBuf, PathBuf, MemoryId) {
    let Fixture { dir, store } = fixture();
    let p = project(&store, "alpha");
    let id = add(&store, p, "written before the migration");
    drop(store);
    let db = dir.path().join("memory.db");
    let backups = dir.path().join("backups");
    (dir, db, backups, id)
}

#[test]
fn s4_new_database_is_created_without_a_backup() {
    let fx = fixture();
    assert_eq!(SCHEMA_VERSION, 5, "B-178");
    assert_eq!(MIGRATIONS.last().map(|m| m.version), Some(SCHEMA_VERSION));
    assert_eq!(fx.store.schema_version().expect("version"), SCHEMA_VERSION);
    assert!(backup_files(&fx.backups()).is_empty());
    // Opening a current database migrates nothing and backs nothing up.
    let again = fx.reopen(stamp());
    assert_eq!(again.schema_version().expect("version"), SCHEMA_VERSION);
    assert!(backup_files(&fx.backups()).is_empty());
}

#[test]
fn s4_migration_writes_a_backup_first() {
    let (_dir, db, backups, id) = current_database();
    let migrations = with_next(probe);
    let store = Store::open_with(&db, &backups, &migrations).expect("migrate to the probe");
    assert_eq!(store.schema_version().expect("version"), NEXT);
    assert!(has_table(&store, "migration_probe"));
    let probe: i64 = store
        .read("reading the probe", |tx| {
            tx.query_row("SELECT x FROM migration_probe", [], |r| r.get(0))
                .ctx("reading the probe")
        })
        .expect("read");
    assert_eq!(probe, 1, "the migration saw the existing data");
    store.configure_index(stamp()).expect("configure");
    assert_eq!(
        store.get_memory(&id).expect("get").content,
        "written before the migration"
    );

    let files = backup_files(&backups);
    assert_eq!(files.len(), 1, "{files:?}");
    let name = files[0]
        .file_name()
        .and_then(|n| n.to_str())
        .expect("file name");
    assert!(
        name.starts_with("memory-") && name.ends_with(&format!("-schema-v{SCHEMA_VERSION}.db")),
        "{name}"
    );

    // The backup is the current-version database as it was.
    let restore = tempfile::tempdir().expect("temporary folder");
    let copy = restore.path().join("memory.db");
    std::fs::copy(&files[0], &copy).expect("copy the backup");
    let old = Store::open_at(&copy, &restore.path().join("backups")).expect("open the backup");
    assert_eq!(old.schema_version().expect("version"), SCHEMA_VERSION);
    assert!(!has_table(&old, "migration_probe"));
    old.configure_index(stamp()).expect("configure");
    assert_eq!(old.memory_count().expect("count"), 1);
    assert!(old.find_memory(&id).expect("find").is_some());
    drop(old);

    // Opening the migrated database again backs nothing up.
    drop(store);
    let store = Store::open_with(&db, &backups, &migrations).expect("reopen");
    assert_eq!(store.schema_version().expect("version"), NEXT);
    assert_eq!(backup_files(&backups).len(), 1);
    drop(store);

    // A build that knows only the current version refuses the newer
    // database.
    match Store::open_at(&db, &backups) {
        Ok(_) => panic!("a newer schema was opened"),
        Err(e) => {
            assert_eq!(e.code, ErrorCode::Storage, "{e:?}");
            assert!(
                e.message.contains(&format!("schema version {NEXT}")),
                "{}",
                e.message
            );
            assert!(
                e.next_action.contains("restore a backup"),
                "{}",
                e.next_action
            );
        }
    }
}

#[test]
fn s4_failed_migration_leaves_the_previous_version() {
    let (_dir, db, backups, id) = current_database();
    match Store::open_with(&db, &backups, &with_next(probe_fails)) {
        Ok(_) => panic!("a failed migration was reported as done"),
        Err(e) => assert!(e.message.contains("failed on purpose"), "{e:?}"),
    }
    // The backup was written before the migration ran.
    assert_eq!(backup_files(&backups).len(), 1);

    let store = Store::open_at(&db, &backups).expect("open at the current version");
    assert_eq!(store.schema_version().expect("version"), SCHEMA_VERSION);
    assert!(!has_table(&store, "migration_probe"));
    store.configure_index(stamp()).expect("configure");
    assert!(store.find_memory(&id).expect("find").is_some());
    // Opening at the current version needs no second backup.
    assert_eq!(backup_files(&backups).len(), 1);
}

#[test]
fn s4_unreadable_schema_versions_are_refused() {
    for (value, needle) in [("99", "schema version 99"), ("one", "schema_version")] {
        let Fixture { dir, store } = fixture();
        store
            .write("damaging the schema version", |tx| {
                tx.execute(
                    "UPDATE meta SET value = ?1 WHERE key = 'schema_version'",
                    [value],
                )
                .ctx("damaging the schema version")
            })
            .expect("write");
        drop(store);
        let db = dir.path().join("memory.db");
        match Store::open_at(&db, &dir.path().join("backups")) {
            Ok(_) => panic!("schema version {value:?} was accepted"),
            Err(e) => {
                assert_eq!(e.code, ErrorCode::Storage, "{e:?}");
                assert!(e.message.contains(needle), "{value:?}: {}", e.message);
            }
        }
        assert!(backup_files(&dir.path().join("backups")).is_empty());
    }
}

fn status_of(store: &Store, id: &MemoryId) -> Status {
    store.get_memory(id).expect("get").status
}

fn set_status(store: &Store, id: &MemoryId, status: &str) -> Result<usize> {
    store.write("setting a status", |tx| {
        tx.execute(
            "UPDATE memories SET status = ?2 WHERE id = ?1",
            (id.to_string(), status),
        )
        .ctx("setting a status")
    })
}

#[test]
fn s16_v2_sets_every_active_capture_pending() {
    let dir = tempfile::tempdir().expect("temporary folder");
    let db = dir.path().join("memory.db");
    let backups = dir.path().join("backups");
    let capture = |p: ProjectId, text: &str, status: Status| {
        let mut input = note(p, text);
        input.memory.provenance = Provenance::new(ProvenanceSource::Capture);
        input.memory.status = status;
        input
    };

    // A version 1 database, as shipped: captures were stored active.
    let (saved, captured, retracted) = {
        let store = Store::open_with(&db, &backups, &MIGRATIONS[..1]).expect("open at version 1");
        store.configure_index(stamp()).expect("configure");
        assert_eq!(store.schema_version().expect("version"), 1);
        let p = project(&store, "alpha");
        let saved = add(&store, p, "saved with the remember tool");
        let captured = store
            .insert_memory(
                capture(p, "captured by a hook", Status::Active),
                Access::Local,
            )
            .expect("insert")
            .id;
        let retracted = store
            .insert_memory(
                capture(p, "captured, then retracted", Status::Retracted),
                Access::Local,
            )
            .expect("insert")
            .id;
        // Version 1 has no `pending`.
        assert!(set_status(&store, &saved, "pending").is_err());
        (saved, captured, retracted)
    };

    let store = Store::open_with(&db, &backups, &MIGRATIONS[..2]).expect("migrate to version 2");
    store.configure_index(stamp()).expect("configure");
    assert_eq!(store.schema_version().expect("version"), 2);
    assert_eq!(status_of(&store, &captured), Status::Pending);
    assert_eq!(status_of(&store, &saved), Status::Active);
    assert_eq!(status_of(&store, &retracted), Status::Retracted);

    // The migrated database is whole, kept its backup, and still refuses an
    // unknown status.
    let check: String = store
        .read("checking the database", |tx| {
            tx.query_row("PRAGMA integrity_check", [], |r| r.get(0))
                .ctx("checking the database")
        })
        .expect("check");
    assert_eq!(check, "ok");
    let files = backup_files(&backups);
    assert_eq!(files.len(), 1, "{files:?}");
    assert!(
        files[0].to_string_lossy().ends_with("-schema-v1.db"),
        "{files:?}"
    );
    assert!(set_status(&store, &saved, "archived").is_err());
    assert_eq!(set_status(&store, &saved, "pending").expect("pending"), 1);
    assert_eq!(set_status(&store, &saved, "active").expect("active"), 1);
    drop(store);

    // Another connection opened afterwards sees the new constraint too.
    let again = Store::open_with(&db, &backups, &MIGRATIONS[..2]).expect("reopen");
    again.configure_index(stamp()).expect("configure");
    let p = project(&again, "alpha");
    let late = again
        .insert_memory(
            capture(p, "captured after the migration", Status::Pending),
            Access::Local,
        )
        .expect("insert a pending capture")
        .id;
    assert_eq!(status_of(&again, &late), Status::Pending);
    assert_eq!(backup_files(&backups).len(), 1);
}

#[test]
fn s16_v3_adds_the_capture_state_empty() {
    let dir = tempfile::tempdir().expect("temporary folder");
    let db = dir.path().join("memory.db");
    let backups = dir.path().join("backups");

    // A version 2 database holding a pending capture.
    let (p, captured) = {
        let store = Store::open_with(&db, &backups, &MIGRATIONS[..2]).expect("open at version 2");
        store.configure_index(stamp()).expect("configure");
        assert_eq!(store.schema_version().expect("version"), 2);
        assert!(!has_table(&store, "capture_sessions"));
        assert!(!has_table(&store, "capture_seen"));
        let p = project(&store, "alpha");
        let mut input = note(p, "captured under version 2");
        input.memory.provenance = Provenance::new(ProvenanceSource::Capture);
        input.memory.status = Status::Pending;
        let captured = store
            .insert_memory(input, Access::Local)
            .expect("insert")
            .id;
        (p, captured)
    };

    // Version 3 alone (B-167 added version 4, tested below).
    let store = Store::open_with(&db, &backups, &MIGRATIONS[..3]).expect("migrate to version 3");
    store.configure_index(stamp()).expect("configure");
    assert_eq!(store.schema_version().expect("version"), 3);
    assert!(has_table(&store, "capture_sessions"));
    assert!(has_table(&store, "capture_seen"));
    assert_eq!(table_count(&store, "capture_sessions"), 0);
    // No hash is backfilled: a replay may capture the old text again.
    assert_eq!(table_count(&store, "capture_seen"), 0);
    let hash = sm_core::hash::content_hash("captured under version 2");
    assert!(store.captures_seen(p, &[hash]).expect("seen").is_empty());
    assert_eq!(status_of(&store, &captured), Status::Pending);
    let files = backup_files(&backups);
    assert_eq!(files.len(), 1, "{files:?}");
    assert!(
        files[0].to_string_lossy().ends_with("-schema-v2.db"),
        "{files:?}"
    );
    let check: String = store
        .read("checking the database", |tx| {
            tx.query_row("PRAGMA integrity_check", [], |r| r.get(0))
                .ctx("checking the database")
        })
        .expect("check");
    assert_eq!(check, "ok");
}

#[test]
fn s16_b167_v4_adds_the_judge_queue_empty() {
    let dir = tempfile::tempdir().expect("temporary folder");
    let db = dir.path().join("memory.db");
    let backups = dir.path().join("backups");

    // A version 3 database holding a pending capture, stored before the
    // judge existed.
    let captured = {
        let store = Store::open_with(&db, &backups, &MIGRATIONS[..3]).expect("open at version 3");
        store.configure_index(stamp()).expect("configure");
        assert_eq!(store.schema_version().expect("version"), 3);
        assert!(!has_table(&store, "capture_judge"));
        assert!(!has_table(&store, "judge_usage"));
        let p = project(&store, "alpha");
        let mut input = note(p, "captured under version 3");
        input.memory.provenance = Provenance::new(ProvenanceSource::Capture);
        input.memory.status = Status::Pending;
        store
            .insert_memory(input, Access::Local)
            .expect("insert")
            .id
    };

    let store = Store::open_with(&db, &backups, &MIGRATIONS[..4]).expect("migrate to version 4");
    store.configure_index(stamp()).expect("configure");
    assert_eq!(store.schema_version().expect("version"), 4);
    assert!(has_table(&store, "capture_judge"));
    assert!(has_table(&store, "judge_usage"));
    // No row is backfilled: the old capture has no context and the judge
    // never takes it; it waits, pending, until it is pruned.
    assert_eq!(table_count(&store, "capture_judge"), 0);
    assert_eq!(table_count(&store, "judge_usage"), 0);
    assert!(store.judge_queue(25, 100_000).expect("queue").is_empty());
    let stats = store.judge_queue_stats().expect("stats");
    assert_eq!((stats.pending, stats.queued, stats.oldest), (1, 0, None));
    assert_eq!(status_of(&store, &captured), Status::Pending);
    let files = backup_files(&backups);
    assert_eq!(files.len(), 1, "{files:?}");
    assert!(
        files[0].to_string_lossy().ends_with("-schema-v3.db"),
        "{files:?}"
    );
}

/// A capture pending from before the judge has no queue row; its wait for
/// `capture.judge.pending_max_days` starts at the upgrade, not when it was
/// captured, so the upgrade deletes nothing (B-167).
#[test]
fn s16_b167_a_capture_from_before_the_judge_waits_from_the_upgrade() {
    use sm_core::time::Timestamp;
    const DAY: i64 = 86_400 * 1_000_000;
    let dir = tempfile::tempdir().expect("temporary folder");
    let db = dir.path().join("memory.db");
    let backups = dir.path().join("backups");

    // Captured 30 days ago under version 3.
    let (old, young) = {
        let store = Store::open_with(&db, &backups, &MIGRATIONS[..3]).expect("open at version 3");
        store.configure_index(stamp()).expect("configure");
        let p = project(&store, "alpha");
        let ids: Vec<MemoryId> = ["captured a month ago", "captured just now"]
            .into_iter()
            .map(|text| {
                let mut input = note(p, text);
                input.memory.provenance = Provenance::new(ProvenanceSource::Capture);
                input.memory.status = Status::Pending;
                store
                    .insert_memory(input, Access::Local)
                    .expect("insert")
                    .id
            })
            .collect();
        let month_ago = Timestamp::now().as_micros() - 30 * DAY;
        store
            .write("backdating a capture", |tx| {
                tx.execute(
                    "UPDATE memories SET created_at = ?2 WHERE id = ?1",
                    (ids[0].to_string(), month_ago),
                )
                .ctx("backdating a capture")
            })
            .expect("backdate");
        (ids[0], ids[1])
    };

    let before = Timestamp::now();
    let store = Store::open_at(&db, &backups).expect("migrate to version 4");
    let after = Timestamp::now();
    let since: i64 = store
        .read("reading meta", |tx| {
            tx.query_row(
                "SELECT CAST(value AS INTEGER) FROM meta WHERE key = ?1",
                [crate::JUDGE_SINCE_KEY],
                |r| r.get(0),
            )
            .ctx("reading meta")
        })
        .expect("the upgrade time is recorded");
    assert!(
        (before.as_micros()..=after.as_micros()).contains(&since),
        "{since}"
    );

    // The first maintenance after the upgrade, at the default 14 days.
    assert_eq!(store.prune_pending(after, 14).expect("prune"), 0);
    assert_eq!(status_of(&store, &old), Status::Pending);
    // 13 days after the upgrade both still wait; 15 days after, both go.
    let day13 = Timestamp::from_micros(since + 13 * DAY);
    assert_eq!(store.prune_pending(day13, 14).expect("prune"), 0);
    let day15 = Timestamp::from_micros(since + 15 * DAY);
    assert_eq!(store.prune_pending(day15, 14).expect("prune"), 2);
    for id in [old, young] {
        assert!(store.get_memory(&id).is_err(), "{id} was pruned");
    }
}

/// Schema version 5 adds `capture_judge.failed_at`. A capture that had a bad
/// answer before the upgrade is dated to the upgrade, so the upgrade deletes
/// nothing, and a capture the judge has not given up on is never pruned,
/// however long it has been queued (B-178; owner step 20).
#[test]
fn s16_b178_v5_dates_earlier_failures_to_the_upgrade() {
    use sm_core::time::Timestamp;
    const DAY: i64 = 86_400 * 1_000_000;
    let dir = tempfile::tempdir().expect("temporary folder");
    let db = dir.path().join("memory.db");
    let backups = dir.path().join("backups");

    // Three captures queued a month ago under version 4: one never judged,
    // one with a bad answer, one the judge gave up on.
    let ids: Vec<MemoryId> = {
        let store = Store::open_with(&db, &backups, &MIGRATIONS[..4]).expect("open at version 4");
        store.configure_index(stamp()).expect("configure");
        let p = project(&store, "alpha");
        let ids: Vec<MemoryId> = store
            .insert_captures(
                ["never judged", "one bad answer", "given up"]
                    .into_iter()
                    .map(|text| {
                        let mut input = note(p, text);
                        input.memory.provenance = Provenance::new(ProvenanceSource::Capture);
                        input.memory.status = Status::Pending;
                        (input, String::new())
                    })
                    .collect(),
            )
            .expect("insert captures")
            .into_iter()
            .map(|i| i.id)
            .collect();
        let month_ago = Timestamp::now().as_micros() - 30 * DAY;
        store
            .write("backdating the queue", |tx| {
                tx.execute("UPDATE capture_judge SET queued_at = ?1", [month_ago])
                    .ctx("backdating the queue")?;
                tx.execute(
                    "UPDATE capture_judge SET attempts = 1 WHERE memory_id = ?1",
                    [ids[1].to_string()],
                )
                .ctx("a bad answer")?;
                tx.execute(
                    "UPDATE capture_judge SET attempts = ?2 WHERE memory_id = ?1",
                    (ids[2].to_string(), i64::from(crate::JUDGE_MAX_ATTEMPTS)),
                )
                .ctx("given up")
            })
            .expect("backdate");
        ids
    };

    let before = Timestamp::now();
    let store = Store::open_at(&db, &backups).expect("migrate to version 5");
    let after = Timestamp::now();
    assert_eq!(store.schema_version().expect("version"), 5);
    let failed_at: Vec<Option<i64>> = ids
        .iter()
        .map(|id| {
            store
                .read("reading the queue", |tx| {
                    tx.query_row(
                        "SELECT failed_at FROM capture_judge WHERE memory_id = ?1",
                        [id.to_string()],
                        |r| r.get(0),
                    )
                    .ctx("reading the queue")
                })
                .expect("queue row")
        })
        .collect();
    assert_eq!(failed_at[0], None);
    for at in &failed_at[1..] {
        let at = at.expect("an earlier failure is dated");
        assert!(
            (before.as_micros()..=after.as_micros()).contains(&at),
            "{at}"
        );
    }

    // The upgrade deletes nothing; 13 days on nothing goes; 15 days on only
    // the capture the judge gave up on goes; a year on the others remain.
    assert_eq!(store.prune_pending(after, 14).expect("prune"), 0);
    assert_eq!(
        store
            .prune_pending(after.add_micros(13 * DAY), 14)
            .expect("prune"),
        0
    );
    assert_eq!(
        store
            .prune_pending(after.add_micros(15 * DAY), 14)
            .expect("prune"),
        1
    );
    assert!(store.get_memory(&ids[2]).is_err(), "given up is pruned");
    assert_eq!(
        store
            .prune_pending(after.add_micros(365 * DAY), 14)
            .expect("prune"),
        0
    );
    for id in &ids[..2] {
        assert_eq!(status_of(&store, id), Status::Pending);
    }
}
