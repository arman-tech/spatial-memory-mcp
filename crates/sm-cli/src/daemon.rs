//! `spatial-memory daemon [--profile <name>]` (SPEC section 3; CONTRACTS
//! 3.1 and 4; decision D-1): the one process that holds the embedding model.
//!
//! The first client that finds no daemon spawns this subcommand detached,
//! with null standard streams and, on Windows, no inherited handle at all
//! (`client::daemon_command`, `client::spawn_detached`). It loads the
//! profile's settings, builds the EmbeddingGemma loader over the model
//! folder, the sharing adapter and the production [`DaemonOptions`], and
//! calls [`sm_daemon::run`], which blocks until a `shutdown` request or the
//! idle exit. The process exits as soon as `run` returns: whatever threads
//! are still winding down (a model load, a connection) go with it.
//!
//! Every way out writes one line to `daemon.log` in the profile folder and
//! to stderr, which a detached daemon does not have:
//!
//! | outcome                         | exit code |
//! |---------------------------------|-----------|
//! | `shutdown` request              | 0         |
//! | idle exit                       | 0         |
//! | another daemon holds the lock   | 0         |
//! | an error (settings, store, bind)| 1         |
//!
//! The lock case exits 0 because nothing went wrong: a client raced another
//! to start the daemon and the other won (CONTRACTS 4, Auto-start).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use sm_core::config::Settings;
use sm_core::error::{CoreError, Result};
use sm_core::traits::EmbedderInfo;
use sm_daemon::{DaemonLog, DaemonOptions, Exit, GemmaLoader, LoadedModel, ModelLoader};

use crate::cli::error_text;
use crate::profile::{self, GetEnv};

/// Runs the daemon for the profile `flag` names (or the one the usual rules
/// pick) and returns the process exit code. The caller exits with it at
/// once.
pub fn run(flag: Option<&str>, cwd: &Path, get_env: GetEnv<'_>) -> i32 {
    let resolved = match profile::resolve(flag, get_env, cwd) {
        Ok(r) => r,
        Err(e) => {
            // No profile folder to log into: stderr is all there is.
            let (code, line) = conclude(&Err(e));
            eprintln!("{line}");
            return code;
        }
    };
    let paths = resolved.paths.clone();
    let opts = DaemonOptions::default();
    let log = DaemonLog::new(paths.daemon_log(), opts.log_max_bytes);
    let result = resolved.settings().and_then(|settings| {
        settle_in(&paths.dir, &log);
        let loader = loader(&settings, get_env);
        sm_daemon::run(
            paths,
            settings,
            loader,
            Arc::new(sm_daemon::ShareAdapter),
            opts,
        )
    });
    let (code, line) = conclude(&result);
    log.line(&line);
    eprintln!("{line}");
    code
}

/// The daemon's working folder is its profile folder, never the folder of
/// the client that spawned it: on Windows a process's working folder cannot
/// be deleted or renamed while it runs, and the daemon runs for hours. No
/// request depends on it; clients send absolute paths (CONTRACTS 5).
fn settle_in(dir: &Path, log: &DaemonLog) {
    let moved = std::fs::create_dir_all(dir).and_then(|()| std::env::set_current_dir(dir));
    if let Err(e) = moved {
        log.line(format!(
            "pid {}: cannot make {} the working folder: {e}",
            std::process::id(),
            dir.display()
        ));
    }
}

/// The exit code and the log line for each way `run` ends.
pub fn conclude(result: &Result<Exit>) -> (i32, String) {
    let pid = std::process::id();
    match result {
        Ok(Exit::Shutdown { reason }) => (0, format!("pid {pid} exits 0: shut down ({reason})")),
        Ok(Exit::IdleExit) => (
            0,
            format!("pid {pid} exits 0: no request for daemon.idle_exit_minutes"),
        ),
        Ok(Exit::LockHeld) => (
            0,
            format!("pid {pid} exits 0: another daemon holds the lock, so this one did nothing"),
        ),
        Err(e) => (
            1,
            format!(
                "pid {pid} exits 1: {}",
                error_text(e).trim_end().replace('\n', "; ")
            ),
        ),
    }
}

/// EmbeddingGemma 2 from the model folder: `SPATIAL_MEMORY_MODEL_DIR`, or
/// the per-user cache. When the folder cannot be named at all (no cache
/// folder on this system), the daemon still starts and serves what needs no
/// model, and its load fails with that error and its next action (B-36).
fn loader(settings: &Settings, get_env: GetEnv<'_>) -> Arc<dyn ModelLoader> {
    match sm_core::paths::model_dir(
        get_env,
        sm_embed::model::MODEL_ID,
        sm_embed::model::REVISION,
    ) {
        Ok(dir) => Arc::new(GemmaLoader::new(dir, settings)),
        Err(error) => Arc::new(NoModelDir {
            inner: GemmaLoader::new(PathBuf::new(), settings),
            error,
        }),
    }
}

/// A loader whose model folder could not be named.
struct NoModelDir {
    /// Gives the identity the model would have; never loads.
    inner: GemmaLoader,
    error: CoreError,
}

impl ModelLoader for NoModelDir {
    fn info(&self) -> EmbedderInfo {
        self.inner.info()
    }

    fn load(&self) -> Result<LoadedModel> {
        Err(self.error.clone())
    }
}

#[cfg(test)]
mod tests {
    use sm_core::error::ErrorCode;

    use super::*;

    #[test]
    fn s3_each_way_out_of_the_daemon_has_an_exit_code_and_a_line() {
        let pid = std::process::id();
        let shutdown = conclude(&Ok(Exit::Shutdown {
            reason: "requested".into(),
        }));
        assert_eq!(
            shutdown,
            (0, format!("pid {pid} exits 0: shut down (requested)"))
        );
        let (code, line) = conclude(&Ok(Exit::IdleExit));
        assert_eq!(code, 0);
        assert!(line.contains("idle_exit_minutes"), "{line}");
        let (code, line) = conclude(&Ok(Exit::LockHeld));
        assert_eq!(code, 0);
        assert!(line.contains("another daemon holds the lock"), "{line}");
        let e = CoreError::new(ErrorCode::Storage, "the database is locked")
            .with_next_action("Close the other program.");
        let (code, line) = conclude(&Err(e));
        assert_eq!(code, 1);
        assert_eq!(
            line,
            format!(
                "pid {pid} exits 1: error: storage: the database is locked; next: Close the \
                 other program."
            )
        );
        assert!(!line.contains('\n'));
    }

    #[test]
    fn s5_a_model_folder_that_cannot_be_named_fails_the_load_not_the_daemon() {
        // `paths::model_dir` falls back to the real home folder, so the
        // missing cache folder is given here rather than arranged.
        let error = CoreError::new(ErrorCode::ModelMissing, "cannot find the cache folder")
            .with_next_action(
                "Set SPATIAL_MEMORY_MODEL_DIR to the folder holding the model files.",
            );
        let l = NoModelDir {
            inner: GemmaLoader::new(PathBuf::new(), &Settings::default()),
            error: error.clone(),
        };
        assert_eq!(l.info().model_id, sm_embed::model::MODEL_ID);
        assert_eq!(l.info().revision, sm_embed::model::REVISION);
        match l.load() {
            Err(e) => assert_eq!(e, error),
            Ok(_) => panic!("a load without a model folder succeeded"),
        }
    }

    #[test]
    fn s5_the_daemon_loads_from_the_model_folder_variable() {
        let settings = Settings::default();
        let dir = tempfile::tempdir().unwrap();
        let named = dir.path().join("gemma").to_string_lossy().into_owned();
        let env = move |k: &str| (k == sm_core::paths::MODEL_DIR_ENV).then(|| named.clone());
        match loader(&settings, &env).load() {
            Err(e) => {
                // The folder is empty, so the load says which file is
                // missing, in the folder the variable named.
                assert_eq!(e.code, ErrorCode::ModelMissing, "{e:?}");
                assert!(e.message.contains("gemma"), "{}", e.message);
            }
            Ok(_) => panic!("an empty model folder loaded"),
        }
    }
}
