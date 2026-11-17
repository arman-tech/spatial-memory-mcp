//! [`Store`]: the open database, its connections and transactions.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError, RwLock};

use rusqlite::{Connection, Transaction, TransactionBehavior};
use sm_core::error::Result;
use sm_core::ids::MemoryId;
use sm_core::paths::ProfilePaths;
use sm_core::types::{IndexStamp, Memory};

use crate::error::Ctx;
use crate::{conn, memories, schema, vec1};

/// Idle reader connections kept for reuse. More readers than this may be
/// open at once; the extras are closed when they finish.
const MAX_IDLE_READERS: usize = 4;

/// Memories read per page by [`MemoryIter`].
const ITER_PAGE: usize = 256;

/// The store for one profile: one SQLite file (SPEC section 4).
///
/// `Store` is `Send + Sync`. Writes are serialised on one writer connection;
/// reads use pooled reader connections and never wait for the writer (WAL).
/// Two `Store`s on the same file (two processes, or a test) are safe: SQLite
/// serialises their writers, and each waits on the busy timeout.
pub struct Store {
    db_path: PathBuf,
    backups_dir: PathBuf,
    writer: Mutex<Connection>,
    readers: Mutex<Vec<Connection>>,
    /// The stamp of the configured embedder and chunk settings, set by
    /// [`Store::configure_index`]. Searches compare it with the stored one.
    pub(crate) configured: RwLock<Option<IndexStamp>>,
}

impl std::fmt::Debug for Store {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Store")
            .field("db_path", &self.db_path)
            .field("backups_dir", &self.backups_dir)
            .finish_non_exhaustive()
    }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    // A panic while holding a connection cannot leave a transaction open
    // (rusqlite rolls back on drop), so the connection is still usable.
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

impl Store {
    /// Opens (creating if needed) the profile's database and brings its
    /// schema up to date, backing it up to the profile's `backups` folder
    /// first when it holds an older schema.
    pub fn open(paths: &ProfilePaths) -> Result<Store> {
        Store::open_at(&paths.database(), &paths.backups_dir())
    }

    /// Opens the database at `db_path`, with backups written to
    /// `backups_dir`.
    pub fn open_at(db_path: &Path, backups_dir: &Path) -> Result<Store> {
        Store::open_with(db_path, backups_dir, schema::MIGRATIONS)
    }

    pub(crate) fn open_with(
        db_path: &Path,
        backups_dir: &Path,
        migrations: &[schema::Migration],
    ) -> Result<Store> {
        vec1::register()?;
        if let Some(parent) = db_path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }
        let mut writer = conn::open(db_path)?;
        schema::migrate(&mut writer, migrations, backups_dir)?;
        Ok(Store {
            db_path: db_path.to_path_buf(),
            backups_dir: backups_dir.to_path_buf(),
            writer: Mutex::new(writer),
            readers: Mutex::new(Vec::new()),
            configured: RwLock::new(None),
        })
    }

    /// The database file.
    pub fn path(&self) -> &Path {
        &self.db_path
    }

    /// Where backups are written.
    pub fn backups_dir(&self) -> &Path {
        &self.backups_dir
    }

    /// The schema version stored in the database.
    pub fn schema_version(&self) -> Result<u32> {
        let w = lock(&self.writer);
        schema::stored_version(&w)
    }

    /// The database file's size in bytes, including its WAL.
    pub fn database_bytes(&self) -> u64 {
        let main = fs::metadata(&self.db_path).map_or(0, |m| m.len());
        let mut wal = self.db_path.clone().into_os_string();
        wal.push("-wal");
        main + fs::metadata(PathBuf::from(wal)).map_or(0, |m| m.len())
    }

    /// Copies the whole database to the backups folder with `VACUUM INTO`
    /// and returns the copy's path. `reason` becomes part of the file name.
    pub fn backup(&self, reason: &str) -> Result<PathBuf> {
        let w = lock(&self.writer);
        schema::backup_to(&w, &self.backups_dir, reason)
    }

    /// Runs `f` in one `BEGIN IMMEDIATE` transaction on the writer and
    /// commits when it returns `Ok`. An `Err` rolls everything back.
    pub(crate) fn write<T>(
        &self,
        what: &str,
        f: impl FnOnce(&Transaction<'_>) -> Result<T>,
    ) -> Result<T> {
        let mut w = lock(&self.writer);
        let tx = w
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .ctx(&format!("starting to write ({what})"))?;
        let out = f(&tx)?;
        tx.commit().ctx(&format!("committing ({what})"))?;
        Ok(out)
    }

    /// Runs `f` in one read transaction, so everything it reads comes from
    /// one snapshot.
    pub(crate) fn read<T>(
        &self,
        what: &str,
        f: impl FnOnce(&Transaction<'_>) -> Result<T>,
    ) -> Result<T> {
        let mut c = self.take_reader()?;
        let out = (|| {
            let tx = c
                .transaction_with_behavior(TransactionBehavior::Deferred)
                .ctx(&format!("starting to read ({what})"))?;
            let out = f(&tx)?;
            // Ends the read snapshot; nothing was written.
            tx.commit().ctx(&format!("finishing a read ({what})"))?;
            Ok(out)
        })();
        self.return_reader(c);
        out
    }

    fn take_reader(&self) -> Result<Connection> {
        if let Some(c) = lock(&self.readers).pop() {
            return Ok(c);
        }
        conn::open_reader(&self.db_path)
    }

    fn return_reader(&self, c: Connection) {
        let mut pool = lock(&self.readers);
        if pool.len() < MAX_IDLE_READERS {
            pool.push(c);
        }
    }

    /// Every memory, in id order (UUID version 7 ids sort by creation
    /// time), read a page at a time so the whole store is never in memory.
    /// Each page is its own read; memories written meanwhile after the
    /// current position are included.
    pub fn iter_memories(&self) -> MemoryIter<'_> {
        MemoryIter {
            store: self,
            after: None,
            page: Vec::new().into_iter(),
            done: false,
        }
    }

    pub(crate) fn memories_page(
        &self,
        after: Option<&MemoryId>,
        limit: usize,
    ) -> Result<Vec<Memory>> {
        self.read("listing memories", |tx| {
            memories::page_after(tx, after, limit)
        })
    }
}

/// Streams memories in id order; see [`Store::iter_memories`].
pub struct MemoryIter<'a> {
    store: &'a Store,
    after: Option<MemoryId>,
    page: std::vec::IntoIter<Memory>,
    done: bool,
}

impl Iterator for MemoryIter<'_> {
    type Item = Result<Memory>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(m) = self.page.next() {
            self.after = Some(m.id);
            return Some(Ok(m));
        }
        if self.done {
            return None;
        }
        match self.store.memories_page(self.after.as_ref(), ITER_PAGE) {
            Ok(page) => {
                if page.len() < ITER_PAGE {
                    self.done = true;
                }
                self.page = page.into_iter();
                let m = self.page.next()?;
                self.after = Some(m.id);
                Some(Ok(m))
            }
            Err(e) => {
                self.done = true;
                Some(Err(e))
            }
        }
    }
}
