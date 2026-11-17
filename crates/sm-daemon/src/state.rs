//! State shared by every daemon thread. One `Shared` per daemon; nothing is
//! process-global.

use std::collections::{HashMap, VecDeque};
use std::hash::Hash;
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

    pub fn get_mut(&mut self, k: &K) -> Option<&mut V> {
        if self.map.contains_key(k) {
            self.touch(k);
        }
        self.map.get_mut(k)
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
    /// Map runs computed by this daemon, by input hash.
    pub maps: Mutex<HashMap<String, RunId>>,
    /// Wakes the bulk embedding queue.
    pub bulk_wake: Signal,
    /// Bulk sessions currently open.
    pub bulk_open: AtomicUsize,
    pub captures: Mutex<VecDeque<CaptureJob>>,
    pub capture_wake: Signal,
    /// Captures queued or being processed.
    pub captures_pending: AtomicUsize,
    pub stop: AtomicBool,
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
        lock(&self.maps).clear();
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
