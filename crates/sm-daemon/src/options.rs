//! Tunable timings and bounds. The defaults are the production values; tests
//! shorten the intervals so the spool drain, idle exit and bulk release can
//! be observed in seconds (SPEC 14, stream E: "with an injectable interval").

use std::time::Duration;

use sm_core::paths::Endpoint;

/// Daemon options that are not user settings. `Default` gives the values
/// `spatial-memory daemon` runs with.
#[derive(Debug, Clone)]
pub struct DaemonOptions {
    /// Endpoint to bind instead of `ProfilePaths::endpoint()`. Tests use it so
    /// no test reads `XDG_RUNTIME_DIR` from the user's environment.
    pub endpoint: Option<Endpoint>,
    /// How long to try the lock before concluding another daemon owns it
    /// (CONTRACTS 4, Auto-start: 2 s).
    pub lock_wait: Duration,
    /// How long to retry the bind (CONTRACTS 4: 2 s).
    pub bind_wait: Duration,
    /// Retry period for the lock and the bind (CONTRACTS 4: 25 ms).
    pub retry_every: Duration,
    /// Spool drain period (SPEC 16, Spool: every 60 s).
    pub spool_interval: Duration,
    /// Period of the daily maintenance: injection and journal pruning.
    pub maintenance_interval: Duration,
    /// How long the bulk session stays open after the queue empties.
    pub bulk_grace: Duration,
    /// Length of one `daemon.idle_exit_minutes` unit; a minute in production.
    pub idle_unit: Duration,
    /// `daemon.log` size at which it is rotated to `daemon.log.1`.
    pub log_max_bytes: u64,
    /// Physical cores to plan threads for; `None` asks the operating system.
    pub physical_cores: Option<usize>,
    /// How long shutdown waits for in-flight requests and workers.
    pub shutdown_grace: Duration,
    /// Capture sessions whose dedupe state and transcript offset are kept.
    pub capture_sessions: usize,
    /// Sessions whose last recall ranks are kept for `reinforce`.
    pub rank_sessions: usize,
    /// Wait after a failed embedding batch before the queue retries.
    pub bulk_error_backoff: Duration,
    /// Wait before a capture that failed is tried again; doubled after each
    /// failure (e-report decision 27).
    pub capture_retry_backoff: Duration,
}

impl Default for DaemonOptions {
    fn default() -> Self {
        DaemonOptions {
            endpoint: None,
            lock_wait: Duration::from_secs(2),
            bind_wait: Duration::from_secs(2),
            retry_every: Duration::from_millis(25),
            spool_interval: Duration::from_secs(60),
            maintenance_interval: Duration::from_secs(24 * 60 * 60),
            bulk_grace: Duration::from_secs(10),
            idle_unit: Duration::from_secs(60),
            log_max_bytes: 1024 * 1024,
            physical_cores: None,
            shutdown_grace: Duration::from_secs(5),
            capture_sessions: 256,
            rank_sessions: 256,
            bulk_error_backoff: Duration::from_secs(5),
            capture_retry_backoff: Duration::from_secs(2),
        }
    }
}
