//! SPEC section 4, Migrations: forward only, each in its own transaction, a
//! backup written before an existing database is migrated, newer schemas
//! refused. B-157: schema version 2 sets every automatic capture pending.
//! B-163: schema version 3 adds the capture state, empty.
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
    assert_eq!(SCHEMA_VERSION, 3);
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

    let store = Store::open_at(&db, &backups).expect("migrate to version 3");
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
