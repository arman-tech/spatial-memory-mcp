//! State shared by every daemon thread. One `Shared` per daemon; nothing is
//! process-global.

use std::collections::{HashMap, HashSet, VecDeque};
use std::ffi::OsString;
use std::hash::Hash;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, RwLock};
use std::time::{Duration, Instant};

use sm_capture::RegexSecretScanner;
use sm_core::config::Settings;
use sm_core::error::{CoreError, ErrorCode, Result};
use sm_core::ids::{MemoryId, RunId};
use sm_core::journal::Journal;
use sm_core::paths::{Endpoint, ProfilePaths};
use sm_core::project::ProjectIdentity;
use sm_core::protocol::DaemonState;
use sm_core::time::Timestamp;
use sm_core::traits::{ChunkParams, Chunker, Embedder, EmbedderInfo, EmbedderProvider};
use sm_core::types::Project;
use sm_store::Store;

use crate::log::DaemonLog;
use crate::options::DaemonOptions;
use crate::sharing::SharingOps;

/// Locks a mutex, recovering the data if a panicking thread poisoned it.
pub(crate) fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// The message of a caught panic, or an empty string.
pub(crate) fn panic_text(p: &(dyn std::any::Any + Send)) -> String {
    p.downcast_ref::<&str>()
        .map(|s| (*s).to_owned())
        .or_else(|| p.downcast_ref::<String>().cloned())
        .unwrap_or_default()
}

/// The model once loaded: the provider, the tokenizer chunker and the query
/// session, created first and kept for the daemon's life.
pub(crate) struct Ready {
    pub provider: Arc<dyn EmbedderProvider>,
    pub chunker: Arc<dyn Chunker>,
    pub query: Box<dyn Embedder>,
}

pub(crate) enum ModelState {
    Loading,
    Ready(Arc<Ready>),
    Failed(CoreError),
}

/// A small bounded map that forgets the least recently used entry.
pub(crate) struct Lru<K, V> {
    cap: usize,
    map: HashMap<K, V>,
    order: VecDeque<K>,
}

impl<K: Eq + Hash + Clone, V> Lru<K, V> {
    pub fn new(cap: usize) -> Self {
        Lru {
            cap: cap.max(1),
            map: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    fn touch(&mut self, k: &K) {
        if let Some(pos) = self.order.iter().position(|x| x == k) {
            let key = self.order.remove(pos).expect("position is in range");
            self.order.push_back(key);
        }
    }

    pub fn get(&mut self, k: &K) -> Option<&V> {
        if self.map.contains_key(k) {
            self.touch(k);
        }
        self.map.get(k)
    }

    pub fn insert(&mut self, k: K, v: V) {
        if self.map.insert(k.clone(), v).is_some() {
            self.touch(&k);
            return;
        }
        self.order.push_back(k);
        while self.order.len() > self.cap {
            if let Some(old) = self.order.pop_front() {
                self.map.remove(&old);
            }
        }
    }

    pub fn clear(&mut self) {
        self.map.clear();
        self.order.clear();
    }
}

/// The in-memory index of map runs (`regions`, `journey`). Every drop
/// moves the generation on, and a run is indexed only under the generation
/// read before its vectors were (e-report decision 34): a run computed from
/// vectors that a write replaced meanwhile is never kept.
#[derive(Default)]
pub(crate) struct MapIndex {
    generation: u64,
    runs: HashMap<String, RunId>,
}

impl MapIndex {
    pub fn get(&self, key: &str) -> Option<RunId> {
        self.runs.get(key).copied()
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn drop_all(&mut self) {
        self.generation += 1;
        self.runs.clear();
    }

    /// Indexes `run` if nothing was dropped since `generation`.
    pub fn insert(&mut self, key: String, run: RunId, generation: u64) -> bool {
        if generation != self.generation {
            return false;
        }
        self.runs.insert(key, run);
        true
    }
}

/// A wake-up signal: `notify` sets it, `wait` blocks until it is set or the
/// timeout passes, and clears it.
#[derive(Default)]
pub(crate) struct Signal {
    flag: Mutex<bool>,
    cv: Condvar,
}

impl Signal {
    pub fn notify(&self) {
        *lock(&self.flag) = true;
        self.cv.notify_all();
    }

    /// Returns whether the signal was set.
    pub fn wait(&self, timeout: Duration) -> bool {
        let guard = lock(&self.flag);
        let (mut guard, _) = self
            .cv
            .wait_timeout_while(guard, timeout, |set| !*set)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        std::mem::take(&mut *guard)
    }
}

/// A resolved project with the identity it came from.
#[derive(Clone)]
pub(crate) struct ProjectEntry {
    pub project: Project,
    pub identity: ProjectIdentity,
}

/// One queued capture (SPEC 16): the hook event as received, with the
/// request's session and working directory.
pub(crate) struct CaptureJob {
    pub request_id: String,
    pub cwd: String,
    pub session_id: Option<String>,
    pub client: Option<sm_core::types::Client>,
    pub params: sm_core::protocol::CaptureParams,
    /// Its file in `spool/new/`, named in [`Shared::held`]: a drained
    /// file, or the file the worker wrote when it took a socket capture.
    /// `None` until then, or when that write failed.
    pub file: Option<PathBuf>,
    /// Failed attempts so far in this daemon.
    pub attempts: u32,
    /// Not taken again before this (a retry's backoff).
    pub not_before: Option<Instant>,
}

pub(crate) struct Shared {
    pub paths: ProfilePaths,
    pub settings: Settings,
    pub opts: DaemonOptions,
    pub endpoint: Endpoint,
    pub store: Store,
    pub log: DaemonLog,
    pub journal: Journal,
    pub sharing: Arc<dyn SharingOps>,
    pub scanner: RegexSecretScanner,
    /// What the loader said the model will be; the index stamp uses it.
    pub info: EmbedderInfo,
    pub physical_cores: usize,
    pub started_at: Timestamp,
    pub model: RwLock<ModelState>,
    pub ready_seconds: Mutex<Option<f64>>,
    /// Whether the stored index stamp matches this daemon's; false until a
    /// reindex when it does not. The store itself refuses searches then.
    pub index_ok: AtomicBool,
    pub projects: Mutex<Lru<String, ProjectEntry>>,
    /// The last recall's result ids per session, for `reinforce`.
    pub ranks: Mutex<Lru<String, Vec<MemoryId>>>,
    /// Map runs computed by this daemon, by project and variant.
    pub maps: Mutex<MapIndex>,
    /// Wakes the bulk embedding queue.
    pub bulk_wake: Signal,
    /// The daemon's one bulk session, shared by the queue and `doctor_eval`
    /// (e-report decision 35).
    pub bulk: Mutex<Option<crate::queue::BulkSession>>,
    /// Bulk sessions currently open (0 or 1).
    pub bulk_open: AtomicUsize,
    pub captures: Mutex<VecDeque<CaptureJob>>,
    pub capture_wake: Signal,
    /// Captures queued, being processed or waiting for a retry.
    pub captures_pending: AtomicUsize,
    /// Names of the `spool/new/` files a capture job in this daemon holds:
    /// the drain skips them (e-report decision 28).
    pub held: Mutex<HashSet<OsString>>,
    /// Names of the `spool/new/` files of stored captures that could not be
    /// removed: the drain never processes them again, tries the removal
    /// again quietly, and forgets a name once its file is gone (stream H 6).
    pub unremovable: Mutex<HashSet<OsString>>,
    /// Held while a job moves from the queue to the worker, and while the
    /// queue is written to disk at shutdown, so no job is between the two.
    pub taking: Mutex<()>,
    pub stop: AtomicBool,
    /// Set at the end of shutdown, just before the lock is released: a
    /// request still arriving on an old connection is refused.
    pub closed: AtomicBool,
    pub stop_reason: Mutex<Option<String>>,
    pub last_activity: Mutex<Instant>,
    pub in_flight: AtomicUsize,
    /// Wakes the maintenance thread (shutdown).
    pub maintenance_wake: Signal,
}

impl Shared {
    pub fn chunk_params(&self, index_cap: Option<usize>) -> ChunkParams {
        ChunkParams {
            max_tokens: self.settings.chunk.max_tokens as usize,
            overlap: self.settings.chunk.overlap,
            index_cap,
        }
    }

    pub fn state(&self) -> (DaemonState, Option<CoreError>) {
        match &*self
            .model
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
        {
            ModelState::Loading => (DaemonState::Loading, None),
            ModelState::Ready(_) => (DaemonState::Ready, None),
            ModelState::Failed(e) => (DaemonState::Failed, Some(e.clone())),
        }
    }

    /// The loaded model, or `daemon_loading`, or the load error (B-36).
    pub fn ready(&self) -> Result<Arc<Ready>> {
        match &*self
            .model
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
        {
            ModelState::Ready(r) => Ok(Arc::clone(r)),
            ModelState::Loading => Err(loading()),
            ModelState::Failed(e) => Err(e.clone()),
        }
    }

    /// The model if it is loaded, without an error otherwise.
    pub fn ready_opt(&self) -> Option<Arc<Ready>> {
        self.ready().ok()
    }

    pub fn is_loading(&self) -> bool {
        matches!(
            &*self
                .model
                .read()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
            ModelState::Loading
        )
    }

    pub fn stopping(&self) -> bool {
        self.stop.load(Ordering::SeqCst)
    }

    pub fn touch(&self) {
        *lock(&self.last_activity) = Instant::now();
    }

    /// Drops the cached map runs: memories or vectors changed.
    pub fn drop_maps(&self) {
        lock(&self.maps).drop_all();
    }

    /// The map generation, read before a run's input.
    pub fn maps_generation(&self) -> u64 {
        lock(&self.maps).generation()
    }

    /// Indexes `run` under `key` unless the maps were dropped since
    /// `generation` was read. Returns whether it was indexed.
    pub fn index_map(&self, key: String, run: RunId, generation: u64) -> bool {
        lock(&self.maps).insert(key, run, generation)
    }

    /// After a write that stored chunks: wake the queue, drop the maps.
    pub fn after_write(&self) {
        self.bulk_wake.notify();
        self.drop_maps();
    }
}

pub(crate) fn loading() -> CoreError {
    CoreError::new(
        ErrorCode::DaemonLoading,
        "the embedding model is still loading",
    )
    .with_next_action("Retry in a few seconds; `spatial-memory doctor` shows progress.")
}

#[cfg(test)]
mod tests {
    use sm_core::ids::RunId;

    use super::MapIndex;

    #[test]
    fn s7_a_run_computed_before_a_drop_is_never_indexed() {
        let mut maps = MapIndex::default();
        // A request reads the generation, then the vectors; a batch is
        // stored and drops the maps before its run is indexed.
        let before = maps.generation();
        maps.drop_all();
        assert!(!maps.insert("1:regions".to_owned(), RunId(7), before));
        assert_eq!(maps.get("1:regions"), None);
        // A run computed after the drop is indexed, until the next drop.
        let after = maps.generation();
        assert!(maps.insert("1:regions".to_owned(), RunId(8), after));
        assert_eq!(maps.get("1:regions"), Some(RunId(8)));
        maps.drop_all();
        assert_eq!(maps.get("1:regions"), None);
    }
}
