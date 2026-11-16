//! SPEC section 4, Migrations: forward only, each in its own transaction, a
//! backup written before an existing database is migrated, newer schemas
//! refused.

use rusqlite::Transaction;
use sm_core::error::Result;

use super::*;
use crate::SCHEMA_VERSION;
use crate::schema::{MIGRATIONS, Migration};

fn v2(tx: &Transaction<'_>) -> Result<()> {
    tx.execute_batch(
        "CREATE TABLE migration_probe (x INTEGER NOT NULL) STRICT;
         INSERT INTO migration_probe(x) SELECT count(*) FROM memories;",
    )
    .ctx("migration 2")
}

fn v2_fails(tx: &Transaction<'_>) -> Result<()> {
    tx.execute_batch("CREATE TABLE migration_probe (x INTEGER NOT NULL) STRICT;")
        .ctx("migration 2")?;
    Err(CoreError::storage("migration 2 failed on purpose"))
}

fn with_v2(apply: fn(&Transaction<'_>) -> Result<()>) -> [Migration; 2] {
    [
        Migration {
            version: 1,
            apply: MIGRATIONS[0].apply,
        },
        Migration { version: 2, apply },
    ]
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

/// A version-1 database holding one memory, closed, in its own folder.
fn v1_database() -> (TempDir, PathBuf, PathBuf, MemoryId) {
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
    assert_eq!(SCHEMA_VERSION, 1);
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
    let (_dir, db, backups, id) = v1_database();
    let migrations = with_v2(v2);
    let store = Store::open_with(&db, &backups, &migrations).expect("migrate to v2");
    assert_eq!(store.schema_version().expect("version"), 2);
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
        name.starts_with("memory-") && name.ends_with("-schema-v1.db"),
        "{name}"
    );

    // The backup is the version-1 database as it was.
    let restore = tempfile::tempdir().expect("temporary folder");
    let copy = restore.path().join("memory.db");
    std::fs::copy(&files[0], &copy).expect("copy the backup");
    let old = Store::open_at(&copy, &restore.path().join("backups")).expect("open the backup");
    assert_eq!(old.schema_version().expect("version"), 1);
    assert!(!has_table(&old, "migration_probe"));
    old.configure_index(stamp()).expect("configure");
    assert_eq!(old.memory_count().expect("count"), 1);
    assert!(old.find_memory(&id).expect("find").is_some());
    drop(old);

    // Opening the migrated database again backs nothing up.
    drop(store);
    let store = Store::open_with(&db, &backups, &migrations).expect("reopen");
    assert_eq!(store.schema_version().expect("version"), 2);
    assert_eq!(backup_files(&backups).len(), 1);
    drop(store);

    // A build that knows only version 1 refuses the newer database.
    match Store::open_at(&db, &backups) {
        Ok(_) => panic!("a newer schema was opened"),
        Err(e) => {
            assert_eq!(e.code, ErrorCode::Storage, "{e:?}");
            assert!(e.message.contains("schema version 2"), "{}", e.message);
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
    let (_dir, db, backups, id) = v1_database();
    match Store::open_with(&db, &backups, &with_v2(v2_fails)) {
        Ok(_) => panic!("a failed migration was reported as done"),
        Err(e) => assert!(e.message.contains("failed on purpose"), "{e:?}"),
    }
    // The backup was written before the migration ran.
    assert_eq!(backup_files(&backups).len(), 1);

    let store = Store::open_at(&db, &backups).expect("open at version 1");
    assert_eq!(store.schema_version().expect("version"), 1);
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
