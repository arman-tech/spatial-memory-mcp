//! `spatial-memory daemon` as a real child process (SPEC 3, CONTRACTS 3.1
//! and 4, decision D-1). The model folder of every sandbox is empty, so
//! the daemon's load fails at once with `model_missing`: these tests check
//! the process, its lock, its endpoint, its exit codes and its log lines,
//! never the model. Each daemon is shut down before its sandbox goes.

mod common;

use std::io::Read;
use std::process::{Child, Stdio};
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

use common::{Run, Sandbox, lock_is_held, shut_down_daemon};
use serde_json::json;

static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// How long a debug daemon may take to open its store and bind.
const START_WAIT: Duration = Duration::from_secs(30);

/// `spatial-memory daemon` in the sandbox, stderr kept.
fn spawn_daemon(sb: &Sandbox) -> Child {
    sb.command(&["daemon"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn spatial-memory daemon")
}

/// Waits for the daemon's endpoint, failing if the child exits first.
fn wait_listening(sb: &Sandbox, child: &mut Child) {
    let endpoint = sb.paths().endpoint();
    let start = Instant::now();
    while !sm_cli::client::is_listening(&endpoint) {
        if let Some(status) = child.try_wait().expect("poll the daemon") {
            let mut err = String::new();
            if let Some(mut pipe) = child.stderr.take() {
                let _ = pipe.read_to_string(&mut err);
            }
            panic!("the daemon exited with {status} before listening: {err}");
        }
        assert!(
            start.elapsed() < START_WAIT,
            "the daemon listened within {START_WAIT:?}"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// Runs `status` until the daemon no longer reports `loading`.
fn settled_status(sb: &Sandbox) -> Run {
    let start = Instant::now();
    loop {
        let run = sb.run(&["status"], &[], b"");
        if !run.stdout.starts_with("Daemon: loading ") {
            return run;
        }
        assert!(
            start.elapsed() < START_WAIT,
            "the model load ended within {START_WAIT:?}: {run:?}"
        );
        std::thread::sleep(Duration::from_millis(50));
    }
}

fn daemon_log(sb: &Sandbox) -> String {
    std::fs::read_to_string(sb.paths().daemon_log()).unwrap_or_default()
}

#[test]
fn s3_the_daemon_serves_until_shutdown_and_logs_each_way_out() {
    let _g = serial();
    let sb = Sandbox::with_daemon();
    let mut child = spawn_daemon(&sb);
    wait_listening(&sb, &mut child);
    assert!(lock_is_held(&sb.paths()), "the daemon holds the lock");

    // The model folder is empty: the daemon runs, its model failed, and
    // `status` says so with the error and next lines.
    let status = settled_status(&sb);
    assert_eq!(status.code, Some(1), "{status:?}");
    assert!(
        status.stdout.starts_with("Daemon: failed "),
        "{}",
        status.stdout
    );
    assert!(
        status.stderr.starts_with("error: model_missing: "),
        "{}",
        status.stderr
    );
    assert!(status.stderr.contains("\nnext: "), "{}", status.stderr);

    // A hook against that daemon still exits 0 and injects nothing.
    let prompt = serde_json::to_vec(&json!({
        "session_id": "s-daemon",
        "cwd": sb.tmp_token(),
        "hook_event_name": "UserPromptSubmit",
        "prompt": "How is the orders table partitioned?"
    }))
    .unwrap();
    let hook = sb.hook(
        &["user-prompt-submit", "--client", "claude-code"],
        &common::env_of(&[(sm_core::config::AMBIENT_ENV, "on")]),
        &prompt,
    );
    assert_eq!(hook.code, Some(0), "{hook:?}");
    assert_eq!(hook.stdout, "", "{hook:?}");

    // A second daemon finds the lock held and exits 0 having done nothing.
    let second = sb.run(&["daemon"], &[], b"");
    assert_eq!(second.code, Some(0), "{second:?}");
    assert!(
        second
            .stderr
            .contains("exits 0: another daemon holds the lock"),
        "{}",
        second.stderr
    );
    assert!(sm_cli::client::is_listening(&sb.paths().endpoint()));

    assert!(
        shut_down_daemon(&sb.paths(), "s3 test done"),
        "the lock came free"
    );
    let out = child.wait_with_output().expect("wait for the daemon");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "{stderr}");
    assert!(
        stderr.contains("exits 0: shut down (s3 test done)"),
        "{stderr}"
    );
    assert!(!lock_is_held(&sb.paths()));

    let log = daemon_log(&sb);
    for line in [
        " listening on ",
        "exits 0: another daemon holds the lock, so this one did nothing",
        "exits 0: shut down (s3 test done)",
    ] {
        assert!(log.contains(line), "{line}: {log}");
    }
}

#[test]
fn s3_a_daemon_that_cannot_open_its_store_exits_1_with_the_error_and_next_lines() {
    let _g = serial();
    // `Sandbox::new` puts a folder where the database goes.
    let sb = Sandbox::new();
    let run = sb.run(&["daemon"], &[], b"");
    assert_eq!(run.code, Some(1), "{run:?}");
    let line = run.stderr.trim_end();
    assert!(!line.contains('\n'), "one line: {line}");
    assert!(line.contains(" exits 1: error: "), "{line}");
    assert!(line.contains("; next: "), "{line}");
    assert!(daemon_log(&sb).contains(line), "{}", daemon_log(&sb));
    assert!(!lock_is_held(&sb.paths()), "the lock is released");
    assert!(!sm_cli::client::is_listening(&sb.paths().endpoint()));
}

#[test]
fn s3_a_client_that_finds_no_daemon_starts_the_real_one_detached() {
    let _g = serial();
    let sb = Sandbox::with_daemon();
    assert!(!sm_cli::client::is_listening(&sb.paths().endpoint()));
    // `status` is a waiting client: it spawns `spatial-memory daemon`
    // detached and asks once it answers.
    let first = sb.run(&["status"], &[], b"");
    assert!(
        first.stdout.starts_with("Daemon: loading ") || first.stdout.starts_with("Daemon: failed "),
        "{first:?}"
    );
    assert!(sm_cli::client::is_listening(&sb.paths().endpoint()));
    assert!(lock_is_held(&sb.paths()));
    let settled = settled_status(&sb);
    assert_eq!(settled.code, Some(1), "{settled:?}");
    assert!(
        settled.stderr.starts_with("error: model_missing: "),
        "{settled:?}"
    );
    assert!(
        daemon_log(&sb).contains(" listening on "),
        "{}",
        daemon_log(&sb)
    );

    assert!(shut_down_daemon(&sb.paths(), "s3 detached test done"));
    assert!(!sm_cli::client::is_listening(&sb.paths().endpoint()));
    let start = Instant::now();
    while !daemon_log(&sb).contains("exits 0: shut down (s3 detached test done)") {
        assert!(
            start.elapsed() < Duration::from_secs(10),
            "{}",
            daemon_log(&sb)
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// Windows: a handle the client inherited from its own parent does not
/// reach the daemon it starts. The test makes a pipe writer inheritable, so
/// `status` inherits it (as a hook inherits copies of the pipes of the
/// process that runs it) and spawns the daemon. Once the test and `status`
/// have closed their copies, the reader sees end of file while the daemon
/// still runs. A daemon that inherited the writer would hold the pipe open
/// until it exits, which is what kept `cargo test` waiting on leaked
/// daemons.
#[cfg(windows)]
#[test]
fn s3_a_detached_daemon_inherits_no_handle_of_its_parent() {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Foundation::{HANDLE_FLAG_INHERIT, SetHandleInformation};
    let _g = serial();
    let sb = Sandbox::with_daemon();
    let (mut reader, writer) = std::io::pipe().expect("a pipe");
    let inherit = |flag: u32| {
        // SAFETY: `writer` is a live pipe handle this test owns; only its
        // inherit flag changes.
        let ok = unsafe { SetHandleInformation(writer.as_raw_handle(), HANDLE_FLAG_INHERIT, flag) };
        assert_ne!(ok, 0, "set the inherit flag");
    };
    inherit(HANDLE_FLAG_INHERIT);
    let first = sb.run(&["status"], &[], b"");
    inherit(0);
    drop(writer);
    assert!(first.stdout.starts_with("Daemon: "), "{first:?}");
    assert!(lock_is_held(&sb.paths()), "status started the daemon");

    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut sink = Vec::new();
        let _ = tx.send(reader.read_to_end(&mut sink).map(|_| sink.len()));
    });
    let read = rx.recv_timeout(Duration::from_secs(10));
    let still_running = sm_cli::client::is_listening(&sb.paths().endpoint());
    assert!(shut_down_daemon(&sb.paths(), "s3 inherit test done"));
    assert!(still_running, "the daemon ran while the pipe was read");
    let eof = match &read {
        Ok(Ok(0)) => true,
        Ok(Err(e)) => e.kind() == std::io::ErrorKind::BrokenPipe,
        _ => false,
    };
    assert!(eof, "end of file while the daemon runs: {read:?}");
}

/// The daemon child has exited; its stderr.
fn exited(child: Child) -> (Option<i32>, String) {
    let out = child.wait_with_output().expect("wait for the daemon");
    (
        out.status.code(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

#[test]
fn s3_stop_ends_the_daemon_cleanly_and_never_starts_one() {
    // B-155: `stop` asks the profile's daemon to shut down and returns only
    // once its endpoint is closed and its lock is free.
    let _g = serial();
    let sb = Sandbox::with_daemon();
    let mut child = spawn_daemon(&sb);
    wait_listening(&sb, &mut child);
    assert!(lock_is_held(&sb.paths()), "the daemon holds the lock");

    let stop = sb.run(&["stop"], &[], b"");
    assert_eq!(stop.code, Some(0), "{stop:?}");
    assert!(stop.stdout.starts_with("Daemon: stopped"), "{stop:?}");
    assert!(
        !sm_cli::client::is_listening(&sb.paths().endpoint()),
        "the endpoint is closed when stop returns"
    );
    assert!(
        !lock_is_held(&sb.paths()),
        "the lock is free when stop returns"
    );
    let (code, stderr) = exited(child);
    assert_eq!(code, Some(0), "{stderr}");
    assert!(stderr.contains("exits 0: shut down (stop)"), "{stderr}");
    let log = daemon_log(&sb);
    assert!(log.contains("exits 0: shut down (stop)"), "{log}");

    // With no daemon, `stop` says so, exits 0 and starts none.
    let again = sb.run(&["stop"], &[], b"");
    assert_eq!(again.code, Some(0), "{again:?}");
    assert!(again.stdout.starts_with("Daemon: not running"), "{again:?}");
    std::thread::sleep(Duration::from_millis(500));
    assert!(!sm_cli::client::is_listening(&sb.paths().endpoint()));
    assert!(!lock_is_held(&sb.paths()));
    assert_eq!(
        daemon_log(&sb).matches(" listening on ").count(),
        1,
        "no second daemon: {}",
        daemon_log(&sb)
    );

    // A profile that was never used: nothing is created for it.
    let unused = sb.paths().dir.with_file_name("never-used");
    let other = sb.run(&["--profile", "never-used", "stop"], &[], b"");
    assert_eq!(other.code, Some(0), "{other:?}");
    assert!(other.stdout.starts_with("Daemon: not running"), "{other:?}");
    assert!(!unused.exists(), "{}", unused.display());
}

#[test]
fn s3_b180_stop_hold_keeps_the_daemon_stopped_until_release() {
    // B-180: `stop --hold` stops the daemon and keeps it stopped for an
    // upgrade. A hook, a command and a daemon started by hand all leave it
    // stopped until `release`; after it, the next hook starts one.
    let _g = serial();
    let sb = Sandbox::with_daemon();
    let mut child = spawn_daemon(&sb);
    wait_listening(&sb, &mut child);
    let hold = sb.paths().hold_file();

    let stop = sb.run(&["stop", "--hold"], &[], b"");
    assert_eq!(stop.code, Some(0), "{stop:?}");
    assert!(stop.stdout.starts_with("Daemon: stopped"), "{stop:?}");
    assert!(
        stop.stdout
            .contains("Held: no hook, session or command starts the daemon of profile default"),
        "{stop:?}"
    );
    assert!(hold.is_file(), "{}", hold.display());
    let (code, stderr) = exited(child);
    assert_eq!(code, Some(0), "{stderr}");
    assert!(
        stderr.contains("exits 0: shut down (stop --hold)"),
        "{stderr}"
    );

    // A prompt hook finds no daemon and starts none.
    let prompt = serde_json::to_vec(&json!({
        "session_id": "s-hold",
        "cwd": sb.tmp_token(),
        "hook_event_name": "UserPromptSubmit",
        "prompt": "Which cache did we choose for sessions?"
    }))
    .unwrap();
    let hook = sb.hook(
        &["user-prompt-submit", "--client", "claude-code"],
        &[],
        &prompt,
    );
    assert_eq!(hook.code, Some(0), "{hook:?}");
    // A command fails at once and names `release`.
    let status = sb.run(&["status"], &[], b"");
    assert_eq!(status.code, Some(1), "{status:?}");
    assert!(status.stderr.contains("is held stopped"), "{status:?}");
    assert!(
        status.stderr.contains("`spatial-memory release`"),
        "{status:?}"
    );
    // A daemon started by hand takes the lock, finds the hold and exits 0.
    let (code, stderr) = exited(spawn_daemon(&sb));
    assert_eq!(code, Some(0), "{stderr}");
    assert!(stderr.contains("the profile is held stopped"), "{stderr}");
    std::thread::sleep(Duration::from_millis(500));
    assert!(!sm_cli::client::is_listening(&sb.paths().endpoint()));
    assert!(!lock_is_held(&sb.paths()));
    assert_eq!(
        daemon_log(&sb).matches(" listening on ").count(),
        1,
        "no daemon while held: {}",
        daemon_log(&sb)
    );

    // `release` lifts the hold and starts nothing.
    let release = sb.run(&["release"], &[], b"");
    assert_eq!(release.code, Some(0), "{release:?}");
    assert!(
        release.stdout.starts_with("Released: profile default"),
        "{release:?}"
    );
    assert!(!hold.exists());
    assert!(!sm_cli::client::is_listening(&sb.paths().endpoint()));
    let again = sb.run(&["release"], &[], b"");
    assert_eq!(again.code, Some(0), "{again:?}");
    assert!(again.stdout.starts_with("Not held"), "{again:?}");
    // The same hook now starts the daemon.
    let hook = sb.hook(
        &["user-prompt-submit", "--client", "claude-code"],
        &[],
        &prompt,
    );
    assert_eq!(hook.code, Some(0), "{hook:?}");
    let start = Instant::now();
    while !sm_cli::client::is_listening(&sb.paths().endpoint()) {
        assert!(
            start.elapsed() < START_WAIT,
            "the hook started a daemon after release: {}",
            daemon_log(&sb)
        );
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(shut_down_daemon(&sb.paths(), "b180 test done"));
}

#[test]
fn s16_setup_remove_stops_the_daemon() {
    // B-155: `setup --remove` unregisters, then stops the profile's daemon,
    // so nothing of the install keeps running. Cursor, so that the sandbox
    // never runs the `claude` command.
    let _g = serial();
    let sb = Sandbox::with_daemon();
    let mut child = spawn_daemon(&sb);
    wait_listening(&sb, &mut child);

    let removed = sb.run(&["setup", "--client", "cursor", "--remove"], &[], b"");
    assert_eq!(removed.code, Some(0), "{removed:?}");
    assert!(
        removed
            .stdout
            .lines()
            .any(|l| l.starts_with("Daemon: stopped")),
        "{removed:?}"
    );
    assert!(!sm_cli::client::is_listening(&sb.paths().endpoint()));
    assert!(!lock_is_held(&sb.paths()));
    let (code, stderr) = exited(child);
    assert_eq!(code, Some(0), "{stderr}");
    assert!(
        stderr.contains("exits 0: shut down (setup --remove)"),
        "{stderr}"
    );

    // Run again with no daemon: still exit 0, and no daemon is started.
    let again = sb.run(&["setup", "--client", "cursor", "--remove"], &[], b"");
    assert_eq!(again.code, Some(0), "{again:?}");
    assert!(
        again
            .stdout
            .lines()
            .any(|l| l.starts_with("Daemon: not running")),
        "{again:?}"
    );
    std::thread::sleep(Duration::from_millis(500));
    assert!(!sm_cli::client::is_listening(&sb.paths().endpoint()));
}

#[test]
fn d1_one_binary_lists_every_subcommand() {
    let sb = Sandbox::new();
    let run = sb.run(&["--help"], &[], b"");
    assert_eq!(run.code, Some(0), "{run:?}");
    for name in [
        "daemon", "mcp", "hook", "setup", "doctor", "import", "export", "migrate", "reindex",
        "status", "stop",
    ] {
        let listed = run
            .stdout
            .lines()
            .any(|l| l.trim_start().starts_with(&format!("{name} ")) || l.trim() == name);
        assert!(listed, "{name} in --help: {}", run.stdout);
    }
}
