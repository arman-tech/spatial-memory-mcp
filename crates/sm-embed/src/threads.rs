//! The thread budget and below-normal priority for bulk work (SPEC 5,
//! "Using the machine's cores"; D-10; SPEC 13).
//!
//! A session never gets more intra-op threads than the machine has
//! physical cores. A bulk session's threads run below normal priority:
//! ONNX Runtime creates its pool threads through [`CountingThreads`], which
//! lowers each one as it starts and counts them, so tests and `doctor` can
//! read back how many run and at what priority. Where priority cannot be
//! set, the bulk budget is halved and the reason reported.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use sm_core::traits::SessionKind;

use crate::os;

/// Physical cores, at least 1, and never more than the processors this
/// process may use (B-133). `num_cpus::get_physical` counts the machine's
/// cores and ignores the affinity mask and, on Linux, the cgroup quota;
/// `available_parallelism` reads both, so a container given two processors
/// of a 16-core machine gets a budget for two, not sixteen (SPEC 5 rule 2).
pub fn physical_cores() -> usize {
    cap_by_available(
        num_cpus::get_physical(),
        std::thread::available_parallelism().ok().map(|n| n.get()),
    )
}

/// `physical`, at least 1, and at most `available` when that is known.
pub fn cap_by_available(physical: usize, available: Option<usize>) -> usize {
    let physical = physical.max(1);
    available.map_or(physical, |a| physical.min(a.max(1)))
}

/// `requested` threads, at least 1 and never more than `physical`.
pub fn clamp_threads(requested: usize, physical: usize) -> usize {
    requested.clamp(1, physical.max(1))
}

/// The SPEC 13 fallback when priority cannot be lowered: half the budget,
/// at least 1.
pub fn fallback_threads(threads: usize) -> usize {
    (threads / 2).max(1)
}

/// Whether this process can lower a thread's priority, found once by
/// lowering a short-lived thread and reading the result back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PriorityProbe {
    /// The priority a lowered thread read back as.
    pub lowered_to: Option<i32>,
    /// Why lowering failed; `None` when it worked.
    pub refused: Option<String>,
}

impl PriorityProbe {
    pub fn works(&self) -> bool {
        self.refused.is_none()
    }
}

/// The cached probe for this process.
pub fn priority_probe() -> &'static PriorityProbe {
    static PROBE: OnceLock<PriorityProbe> = OnceLock::new();
    PROBE.get_or_init(|| {
        let result = std::thread::Builder::new()
            .name("sm-embed-priority-probe".to_owned())
            .spawn(os::lower_current_thread)
            .map_err(|e| format!("cannot start a probe thread: {e}"))
            .and_then(|h| h.join().map_err(|_| "the probe thread panicked".to_owned()))
            .and_then(|r| r);
        match result {
            Ok(p) => PriorityProbe {
                lowered_to: Some(p),
                refused: None,
            },
            Err(e) => PriorityProbe {
                lowered_to: None,
                refused: Some(e),
            },
        }
    })
}

/// How a session's threads are set up.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThreadPlan {
    pub kind: SessionKind,
    /// What the caller asked for.
    pub requested: usize,
    /// Intra-op threads the session is built with.
    pub intra_threads: usize,
    /// Whether pool threads (and the bulk runner) are lowered.
    pub lower: bool,
    /// The SPEC 13 fallback applies: priority could not be lowered, so the
    /// bulk budget was halved.
    pub priority_fallback: bool,
    pub fallback_reason: Option<String>,
}

/// The thread plan for a session: clamped to the physical cores; a bulk
/// session lowered, or halved when the probe says lowering fails.
pub fn plan_threads(
    kind: SessionKind,
    requested: usize,
    physical: usize,
    probe: &PriorityProbe,
) -> ThreadPlan {
    let n = clamp_threads(requested, physical);
    match (kind, &probe.refused) {
        (SessionKind::Query, _) => ThreadPlan {
            kind,
            requested,
            intra_threads: n,
            lower: false,
            priority_fallback: false,
            fallback_reason: None,
        },
        (SessionKind::Bulk, None) => ThreadPlan {
            kind,
            requested,
            intra_threads: n,
            lower: true,
            priority_fallback: false,
            fallback_reason: None,
        },
        (SessionKind::Bulk, Some(reason)) => ThreadPlan {
            kind,
            requested,
            intra_threads: fallback_threads(n),
            lower: false,
            priority_fallback: true,
            fallback_reason: Some(reason.clone()),
        },
    }
}

#[derive(Debug, Default)]
struct PoolStats {
    live: AtomicUsize,
    peak: AtomicUsize,
    spawned: AtomicUsize,
    lowered: AtomicUsize,
    refused: AtomicUsize,
    ids: Mutex<Vec<u64>>,
    last_refusal: Mutex<Option<String>>,
}

impl PoolStats {
    fn enter(&self) {
        let live = self.live.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(live, Ordering::SeqCst);
        self.spawned.fetch_add(1, Ordering::SeqCst);
    }
}

/// Decrements the live count when a pool thread ends, even by panic.
struct Live(Arc<PoolStats>);

impl Drop for Live {
    fn drop(&mut self) {
        self.0.live.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Lowers the calling thread when asked, counting the outcome in `stats`.
fn lower_and_count(stats: &PoolStats) {
    match os::lower_current_thread() {
        Ok(_) => {
            stats.lowered.fetch_add(1, Ordering::SeqCst);
        }
        Err(e) => {
            stats.refused.fetch_add(1, Ordering::SeqCst);
            if let Ok(mut last) = stats.last_refusal.lock() {
                *last = Some(e);
            }
        }
    }
}

/// ONNX Runtime's thread manager for one session: starts pool threads as
/// named std threads, lowers them for bulk work, and counts them.
#[derive(Debug, Clone)]
pub struct CountingThreads {
    stats: Arc<PoolStats>,
    lower: bool,
}

impl CountingThreads {
    pub fn new(lower: bool) -> Self {
        CountingThreads {
            stats: Arc::new(PoolStats::default()),
            lower,
        }
    }

    /// Runs on a thread the session owns outside the pool (the bulk
    /// runner): records it and lowers it like a pool thread.
    pub fn adopt_current(&self) {
        if let Ok(mut ids) = self.stats.ids.lock() {
            ids.push(os::current_thread_id());
        }
        if self.lower {
            lower_and_count(&self.stats);
        }
    }

    /// What the pool did so far.
    pub fn snapshot(&self) -> PoolSnapshot {
        let s = &self.stats;
        PoolSnapshot {
            live: s.live.load(Ordering::SeqCst),
            peak: s.peak.load(Ordering::SeqCst),
            spawned: s.spawned.load(Ordering::SeqCst),
            lowered: s.lowered.load(Ordering::SeqCst),
            refused: s.refused.load(Ordering::SeqCst),
            thread_ids: s.ids.lock().map(|v| v.clone()).unwrap_or_default(),
            last_refusal: s.last_refusal.lock().ok().and_then(|v| v.clone()),
        }
    }
}

impl ort::environment::ThreadManager for CountingThreads {
    type Thread = std::thread::JoinHandle<()>;

    fn create(&self, work: impl FnOnce() + Send + 'static) -> ort::Result<Self::Thread> {
        let stats = Arc::clone(&self.stats);
        let lower = self.lower;
        std::thread::Builder::new()
            .name("sm-embed-pool".to_owned())
            .spawn(move || {
                stats.enter();
                let _live = Live(Arc::clone(&stats));
                if let Ok(mut ids) = stats.ids.lock() {
                    ids.push(os::current_thread_id());
                }
                if lower {
                    lower_and_count(&stats);
                }
                work();
            })
            .map_err(|e| ort::Error::new(format!("cannot start an inference thread: {e}")))
    }

    fn join(thread: Self::Thread) -> ort::Result<()> {
        thread
            .join()
            .map_err(|_| ort::Error::new("an inference thread panicked"))
    }
}

/// Pool counters at one moment.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PoolSnapshot {
    /// Pool threads alive now.
    pub live: usize,
    /// Most pool threads alive at once.
    pub peak: usize,
    /// Pool threads started in all.
    pub spawned: usize,
    /// Threads lowered below normal (pool threads and the bulk runner).
    pub lowered: usize,
    /// Threads whose lowering the OS refused.
    pub refused: usize,
    /// OS ids of the threads seen (the bulk runner first, if any).
    pub thread_ids: Vec<u64>,
    pub last_refusal: Option<String>,
}

/// A session's threads, as planned and as observed (AT-3, D-10).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThreadReport {
    pub plan: ThreadPlan,
    pub pool: PoolSnapshot,
}

impl ThreadReport {
    /// Threads running inference now: the pool plus the thread that calls
    /// `run`, which ONNX Runtime uses as one of the intra-op threads.
    pub fn running(&self) -> usize {
        self.pool.live + 1
    }

    /// Most threads that ran inference at once.
    pub fn max_running(&self) -> usize {
        self.pool.peak + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ort::environment::ThreadManager;

    fn works() -> PriorityProbe {
        PriorityProbe {
            lowered_to: Some(-1),
            refused: None,
        }
    }

    fn refused() -> PriorityProbe {
        PriorityProbe {
            lowered_to: None,
            refused: Some("not permitted".to_owned()),
        }
    }

    #[test]
    fn s5_threads_never_exceed_physical_cores() {
        assert_eq!(clamp_threads(4, 2), 2);
        assert_eq!(clamp_threads(4, 16), 4);
        assert_eq!(clamp_threads(0, 8), 1);
        assert_eq!(clamp_threads(3, 0), 1);
        for physical in 1..=64 {
            for requested in 0..=80 {
                let n = clamp_threads(requested, physical);
                assert!((1..=physical).contains(&n));
            }
        }
        assert!(physical_cores() >= 1);
    }

    #[test]
    fn s5_rule2_physical_cores_are_capped_by_the_processors_the_process_may_use() {
        // Two processors of a 16-core machine (`--cpuset-cpus 0-1`): two.
        assert_eq!(cap_by_available(16, Some(2)), 2);
        // 32 logical processors on 16 cores: the physical count stands.
        assert_eq!(cap_by_available(16, Some(32)), 16);
        assert_eq!(cap_by_available(16, Some(16)), 16);
        // Unknown parallelism leaves the physical count alone.
        assert_eq!(cap_by_available(16, None), 16);
        // Never below one.
        assert_eq!(cap_by_available(0, Some(4)), 1);
        assert_eq!(cap_by_available(4, Some(0)), 1);
        assert_eq!(cap_by_available(0, None), 1);
        // The live figure is within both.
        let live = physical_cores();
        assert!(live >= 1);
        assert!(live <= num_cpus::get_physical().max(1));
        if let Ok(available) = std::thread::available_parallelism() {
            assert!(live <= available.get());
        }
    }

    #[test]
    fn s5_query_sessions_run_at_normal_priority() {
        let plan = plan_threads(SessionKind::Query, 4, 2, &works());
        assert_eq!((plan.intra_threads, plan.lower), (2, false));
        assert!(!plan.priority_fallback);
        // A refused probe never touches query sessions.
        let plan = plan_threads(SessionKind::Query, 4, 8, &refused());
        assert_eq!((plan.intra_threads, plan.priority_fallback), (4, false));
    }

    #[test]
    fn d10_bulk_sessions_are_lowered_within_the_budget() {
        let plan = plan_threads(SessionKind::Bulk, 12, 16, &works());
        assert_eq!((plan.intra_threads, plan.lower), (12, true));
        let plan = plan_threads(SessionKind::Bulk, 24, 16, &works());
        assert_eq!(plan.intra_threads, 16);
    }

    #[test]
    fn s13_bulk_budget_halves_when_priority_is_refused() {
        let plan = plan_threads(SessionKind::Bulk, 12, 16, &refused());
        assert_eq!(plan.intra_threads, 6);
        assert!(plan.priority_fallback && !plan.lower);
        assert_eq!(plan.fallback_reason.as_deref(), Some("not permitted"));
        assert_eq!(fallback_threads(1), 1);
        assert_eq!(fallback_threads(3), 1);
        assert_eq!(fallback_threads(7), 3);
        let plan = plan_threads(SessionKind::Bulk, 30, 4, &refused());
        assert_eq!(plan.intra_threads, 2, "clamped first, then halved");
    }

    #[test]
    fn d10_counting_threads_count_and_lower_pool_threads() {
        let manager = CountingThreads::new(true);
        let (tx, rx) = std::sync::mpsc::channel();
        let handles: Vec<_> = (0..3)
            .map(|_| {
                let tx = tx.clone();
                manager
                    .create(move || {
                        let tid = os::current_thread_id();
                        tx.send((tid, os::thread_priority(tid))).unwrap();
                    })
                    .unwrap()
            })
            .collect();
        for h in handles {
            CountingThreads::join(h).unwrap();
        }
        drop(tx);
        let seen: Vec<_> = rx.iter().collect();
        let snap = manager.snapshot();
        assert_eq!(seen.len(), 3);
        assert_eq!(snap.spawned, 3);
        assert_eq!(snap.live, 0);
        assert!((1..=3).contains(&snap.peak));
        assert_eq!(snap.lowered + snap.refused, 3);
        assert_eq!(snap.thread_ids.len(), 3);
        if priority_probe().works() {
            assert_eq!(snap.lowered, 3);
            let normal = os::thread_priority(os::current_thread_id()).unwrap();
            for (_, p) in seen {
                assert!(os::is_below(p.unwrap(), normal) || p == Some(19));
            }
        }
        // Without lowering nothing is touched.
        let manager = CountingThreads::new(false);
        let h = manager.create(|| {}).unwrap();
        CountingThreads::join(h).unwrap();
        let snap = manager.snapshot();
        assert_eq!((snap.spawned, snap.lowered, snap.refused), (1, 0, 0));
    }

    #[test]
    fn s5_thread_report_counts_the_calling_thread() {
        let report = ThreadReport {
            plan: plan_threads(SessionKind::Bulk, 4, 8, &works()),
            pool: PoolSnapshot {
                live: 3,
                peak: 3,
                spawned: 3,
                ..PoolSnapshot::default()
            },
        };
        assert_eq!((report.running(), report.max_running()), (4, 4));
    }
}
