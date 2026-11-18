//! The daemon over its real endpoint: the lock and the endpoint (CONTRACTS
//! 3.1 and 4), frames and envelopes (SPEC 3, B-28), and every operation
//! with a success and its main error (SPEC 8 and the command-line ops).
//!
//! Each test starts its own daemon on a temporary profile with the fake
//! embedder; see `common/mod.rs`.

mod common;

use std::sync::Arc;
use std::time::{Duration, Instant};

use common::*;
use serde_json::json;
use sm_core::error::ErrorCode;
use sm_core::ids::MemoryId;
use sm_core::protocol::*;
use sm_core::time::Timestamp;
use sm_core::types::{AnchorKind, AnchorRef, LinkEnd, MemoryType, RecallMode, Relation, Scope};
use sm_daemon::{DaemonLock, Exit, Started};

// --- the lock and the endpoint (CONTRACTS 3.1, 4) -------------------------

#[test]
fn s3_two_daemons_on_one_profile_give_one_server_and_lock_held() {
    let rig = Rig::new();
    let d = rig.start_fake();
    match rig
        .try_start(FakeLoader::new(), Arc::new(UnwiredSharing))
        .expect("a second start")
    {
        Started::LockHeld => {}
        Started::Held { .. } => panic!("the profile is not held"),
        Started::Running(_) => panic!("two daemons hold one profile"),
    }
    let exit = sm_daemon::run(
        rig.paths.clone(),
        rig.settings.clone(),
        FakeLoader::new(),
        Arc::new(UnwiredSharing),
        rig.opts.clone(),
    )
    .expect("run");
    assert_eq!(exit, Exit::LockHeld);
    // The first daemon still serves.
    assert_eq!(d.hello().state, DaemonState::Ready);
}

#[test]
fn s3_shutdown_releases_the_lock_and_a_new_daemon_starts() {
    let rig = Rig::new();
    let d = rig.start_fake();
    let exit = d.shutdown();
    assert_eq!(
        exit,
        Exit::Shutdown {
            reason: "test finished".to_owned()
        }
    );
    let lock = DaemonLock::acquire(
        &rig.paths.lock_file(),
        Duration::ZERO,
        Duration::from_millis(10),
    )
    .expect("acquire")
    .expect("the lock is free after shutdown");
    lock.release();
    let again = rig.start_fake();
    assert_eq!(again.hello().state, DaemonState::Ready);
}

#[test]
fn s3_b180_a_held_profile_starts_no_daemon_even_one_waiting_for_the_lock() {
    // B-180: `daemon.hold` is read with the lock held. A daemon that waited
    // for the lock while `stop --hold` stopped the last one finds the hold
    // and exits without binding; with the hold gone, a daemon starts.
    let rig = Rig::new();
    let d = rig.start_fake();
    let mut opts = rig.opts.clone();
    opts.lock_wait = Duration::from_secs(20);
    let (paths, settings) = (rig.paths.clone(), rig.settings.clone());
    let waiting = std::thread::spawn(move || {
        sm_daemon::start(
            paths,
            settings,
            FakeLoader::new(),
            Arc::new(UnwiredSharing),
            opts,
        )
    });
    std::thread::sleep(Duration::from_millis(200));
    let line = "since 2026-11-18T00:00:00Z by process 7: stop --hold";
    std::fs::write(rig.paths.hold_file(), format!("{line}\n")).unwrap();
    assert_eq!(
        d.shutdown(),
        Exit::Shutdown {
            reason: "test finished".to_owned()
        }
    );
    match waiting.join().expect("the waiting start").expect("start") {
        Started::Held { hold } => assert_eq!(hold, line),
        Started::LockHeld => panic!("the lock was free after the shutdown"),
        Started::Running(h) => {
            h.shutdown("a held profile");
            panic!("a daemon started on a held profile");
        }
    }
    let lock = DaemonLock::acquire(
        &rig.paths.lock_file(),
        Duration::ZERO,
        Duration::from_millis(10),
    )
    .expect("acquire")
    .expect("the held start released the lock");
    lock.release();
    assert!(
        sm_daemon::connect(&rig.paths.endpoint(), Duration::from_millis(200)).is_err(),
        "nothing listens"
    );
    // A run on the held profile ends at once with the hold.
    let exit = sm_daemon::run(
        rig.paths.clone(),
        rig.settings.clone(),
        FakeLoader::new(),
        Arc::new(UnwiredSharing),
        rig.opts.clone(),
    )
    .expect("run");
    assert_eq!(
        exit,
        Exit::Held {
            hold: line.to_owned()
        }
    );
    // Released: a daemon starts again.
    std::fs::remove_file(rig.paths.hold_file()).unwrap();
    let again = rig.start_fake();
    assert_eq!(again.hello().state, DaemonState::Ready);
}

#[test]
fn s3_shutdown_over_the_socket_stops_the_daemon_with_its_reason() {
    let rig = Rig::new();
    let mut d = rig.start_fake();
    let reply = d.ok(
        "",
        Op::Shutdown(ShutdownParams {
            reason: "upgrade".to_owned(),
        }),
    );
    assert_eq!(reply, Reply::ShuttingDown);
    let exit = d.take().wait();
    assert_eq!(
        exit,
        Exit::Shutdown {
            reason: "upgrade".to_owned()
        }
    );
}

#[cfg(unix)]
#[test]
fn s3_a_leftover_socket_file_is_removed() {
    let rig = Rig::new();
    let sm_core::paths::Endpoint::UnixSocket(path) = &rig.endpoint else {
        unreachable!("a Unix socket endpoint on Unix")
    };
    std::fs::write(path, b"left over by a crashed daemon").expect("plant a file");
    let d = rig.start_fake();
    assert_eq!(d.hello().state, DaemonState::Ready);
}

#[cfg(unix)]
#[test]
fn s3_the_socket_is_user_only() {
    use std::os::unix::fs::PermissionsExt;
    let rig = Rig::new();
    let _d = rig.start_fake();
    let sm_core::paths::Endpoint::UnixSocket(path) = &rig.endpoint else {
        unreachable!("a Unix socket endpoint on Unix")
    };
    let mode = std::fs::metadata(path)
        .expect("the socket")
        .permissions()
        .mode();
    assert_eq!(mode & 0o077, 0, "mode {mode:o}");
}

#[cfg(windows)]
#[test]
fn spike1_a_second_first_instance_of_the_pipe_is_refused() {
    let rig = Rig::new();
    let _d = rig.start_fake();
    // Another profile (its own lock) on the same pipe name.
    let mut other = Rig::with_profile("other");
    other.opts.endpoint = Some(rig.endpoint.clone());
    other.endpoint = rig.endpoint.clone();
    let started = Instant::now();
    let e = match other.try_start(FakeLoader::new(), Arc::new(UnwiredSharing)) {
        Err(e) => e,
        Ok(_) => panic!("a second daemon bound the same pipe"),
    };
    assert_code(&e, ErrorCode::DaemonUnavailable);
    assert!(started.elapsed() >= other.opts.bind_wait);
}

#[cfg(windows)]
#[test]
fn spike1_the_pipe_dacl_admits_only_the_current_user() {
    let rig = Rig::new();
    let _d = rig.start_fake();
    let sm_core::paths::Endpoint::NamedPipe(name) = &rig.endpoint else {
        unreachable!("a named pipe on Windows")
    };
    let sid = sm_daemon::endpoint::current_user_sid().expect("the current SID");
    let sddl = sm_daemon::endpoint::pipe_dacl_sddl(name).expect("the pipe's DACL");
    let dacl = sddl
        .split_once("D:")
        .map(|(_, d)| d)
        .unwrap_or_else(|| panic!("no DACL in {sddl}"));
    // Protected: inherited entries are not added.
    assert!(dacl.starts_with('P'), "{sddl}");
    let aces: Vec<&str> = dacl
        .split('(')
        .skip(1)
        .map(|a| a.trim_end_matches(')'))
        .collect();
    assert_eq!(aces.len(), 1, "one entry: {sddl}");
    let fields: Vec<&str> = aces[0].split(';').collect();
    assert_eq!(fields[0], "A", "an allow entry: {sddl}");
    assert_eq!(fields.last().copied(), Some(sid.as_str()), "{sddl}");
}

// --- the profile folder is user-only (SPEC 3 rule 1, SPEC 9; stream H 17) --

/// Through a real start: a new profile folder is created 0700, and an
/// existing 0755 one is set to 0700 with one log line naming it.
#[cfg(unix)]
#[test]
fn s3_the_daemon_makes_the_profile_folder_user_only() {
    use std::os::unix::fs::PermissionsExt;
    let mode = |p: &std::path::Path| {
        std::fs::metadata(p)
            .expect("the folder")
            .permissions()
            .mode()
            & 0o777
    };

    let rig = Rig::new();
    assert!(!rig.paths.dir.exists(), "the profile folder is new");
    let d = rig.start_fake();
    assert_eq!(mode(&rig.paths.dir), 0o700);
    drop(d);

    let rig = Rig::new();
    std::fs::create_dir_all(&rig.paths.dir).expect("the profile folder");
    std::fs::set_permissions(&rig.paths.dir, std::fs::Permissions::from_mode(0o755))
        .expect("chmod");
    let d = rig.start_fake();
    assert_eq!(mode(&rig.paths.dir), 0o700);
    let log = std::fs::read_to_string(rig.paths.daemon_log()).expect("the log");
    let lines: Vec<&str> = log.lines().filter(|l| l.contains("mode 700")).collect();
    assert_eq!(lines.len(), 1, "{log}");
    assert!(
        lines[0].contains(&rig.paths.dir.display().to_string()),
        "{}",
        lines[0]
    );
    drop(d);
}

/// The Windows counterpart: the profile folder is left as it is (its
/// access list comes from `%LOCALAPPDATA%`), and nothing is logged.
#[cfg(windows)]
#[test]
fn s3_the_daemon_leaves_the_profile_folder_alone_on_windows() {
    let rig = Rig::new();
    std::fs::create_dir_all(&rig.paths.dir).expect("the profile folder");
    let before = std::fs::metadata(&rig.paths.dir)
        .expect("the folder")
        .permissions();
    let d = rig.start_fake();
    assert_eq!(d.hello().state, DaemonState::Ready);
    let after = std::fs::metadata(&rig.paths.dir)
        .expect("the folder")
        .permissions();
    assert_eq!(before, after);
    let log = std::fs::read_to_string(rig.paths.daemon_log()).expect("the log");
    assert!(!log.contains("mode 700"), "{log}");
    drop(d);
}

// --- envelopes and frames (SPEC 3) -----------------------------------------

#[test]
fn s3_another_build_gets_version_mismatch() {
    let rig = Rig::new();
    let d = rig.start_fake();
    let mut v = serde_json::to_value(d.req("", Op::Hello)).expect("encode");
    v["v"] = json!(PROTOCOL_VERSION + 1);
    v["build"] = json!("0.0.0-other");
    let e = err_of(d.client().raw_value(&v));
    assert_code(&e, ErrorCode::VersionMismatch);
    // An op this build does not know, from another build: still a
    // version mismatch, with the request id echoed.
    let foreign = json!({
        "v": PROTOCOL_VERSION + 1, "build": "9.9.9", "id": "aaaaaaaaaaaa",
        "cwd": "", "profile": "test", "call": {"op": "teleport", "args": {}}
    });
    let r = d.client().raw_value(&foreign);
    assert_eq!(r.id, "aaaaaaaaaaaa");
    assert_code(&err_of(r), ErrorCode::VersionMismatch);
}

#[test]
fn s3_shutdown_from_another_build_works_for_this_profile_only() {
    let rig = Rig::new();
    let mut d = rig.start_fake();
    let foreign = |profile: &str| {
        json!({
            "v": PROTOCOL_VERSION + 1, "build": "9.9.9", "id": "bbbbbbbbbbbb",
            "cwd": "", "profile": profile, "call": {"op": "shutdown", "args": {"reason": "upgrade"}}
        })
    };
    let e = err_of(d.client().raw_value(&foreign("other")));
    assert_code(&e, ErrorCode::InvalidInput);
    assert_eq!(d.hello().state, DaemonState::Ready);
    let r = d.client().raw_value(&foreign("test"));
    assert_eq!(r.into_result().expect("shutting down"), Reply::ShuttingDown);
    assert_eq!(
        d.take().wait(),
        Exit::Shutdown {
            reason: "upgrade".to_owned()
        }
    );
}

#[test]
fn s3_a_request_for_another_profile_is_refused() {
    let rig = Rig::new();
    let d = rig.start_fake();
    let mut req = d.req("", Op::Hello);
    req.profile = sm_core::config::ProfileName::parse("other").expect("a profile");
    let e = err_of(d.send(&req));
    assert_code(&e, ErrorCode::InvalidInput);
    assert!(e.message.contains("other"), "{}", e.message);
}

#[test]
fn s3_an_empty_frame_is_refused_and_the_connection_stays_usable() {
    let rig = Rig::new();
    let d = rig.start_fake();
    let mut c = d.client();
    let e = err_of(c.raw(b"").expect("a reply"));
    assert_code(&e, ErrorCode::InvalidInput);
    let r = c.call(&d.req("", Op::Hello));
    assert!(matches!(r.into_result(), Ok(Reply::Hello(_))));
}

#[test]
fn s3_a_malformed_body_is_refused() {
    let rig = Rig::new();
    let d = rig.start_fake();
    let mut c = d.client();
    let r = c.raw(b"{ this is not json").expect("a reply");
    assert_eq!(r.id, "");
    assert_code(&err_of(r), ErrorCode::InvalidInput);
    // An unknown op from this build.
    let mut v = serde_json::to_value(d.req("", Op::Hello)).expect("encode");
    v["call"] = json!({"op": "teleport"});
    assert_code(&err_of(c.raw_value(&v)), ErrorCode::InvalidInput);
    assert!(matches!(
        c.call(&d.req("", Op::Hello)).into_result(),
        Ok(Reply::Hello(_))
    ));
}

#[test]
fn s3_an_oversized_frame_closes_the_connection_and_the_daemon_still_serves() {
    let rig = Rig::new();
    let d = rig.start_fake();
    let mut c = d.client();
    use std::io::Write;
    c.stream
        .write_all(&u32::MAX.to_le_bytes())
        .expect("write a prefix");
    let r = c.read().expect("an error before closing");
    assert_code(&err_of(r), ErrorCode::InvalidInput);
    assert!(c.closed());
    assert_eq!(d.hello().state, DaemonState::Ready);
}

#[test]
fn s3_requests_on_many_connections_at_once_are_served() {
    let rig = Rig::new();
    let d = rig.start_fake();
    let cwd = rig.project("busy");
    remember(&d, &cwd, "the orders table is partitioned by month");
    std::thread::scope(|s| {
        for _ in 0..8 {
            s.spawn(|| {
                let mut c = d.client();
                for _ in 0..10 {
                    let mut p = RecallParams::new("orders");
                    p.mode = Some(RecallMode::Keyword);
                    let r = c.call(&d.req(&cwd, Op::Recall(p)));
                    assert!(matches!(r.into_result(), Ok(Reply::Recall(_))));
                }
            });
        }
    });
}

#[test]
fn s3_a_passed_deadline_gives_timeout() {
    let rig = Rig::new();
    let d = rig.start_fake();
    let cwd = rig.project("late");
    let req = d
        .req(&cwd, Op::Recall(RecallParams::new("anything")))
        .with_deadline_ms(0);
    let e = err_of(d.send(&req));
    assert_code(&e, ErrorCode::Timeout);
}

#[test]
fn s3_unknown_fields_are_refused_for_every_op() {
    let rig = Rig::new();
    let d = rig.start_fake();
    let cwd = rig.project("strict");
    let id = remember(&d, &cwd, "a memory that must survive every refused call");
    for op in sample_ops(id) {
        let name = op.name();
        let req = d.req(&cwd, op);
        let has_args = serde_json::to_value(&req).expect("encode")["call"]
            .get("args")
            .is_some();
        let paths: &[&[&str]] = if has_args {
            &[&[], &["call"], &["call", "args"]]
        } else {
            &[&[], &["call"]]
        };
        for path in paths {
            let v = with_unknown_key(&req, path);
            let e = err_of(d.client().raw_value(&v));
            assert_eq!(
                e.code,
                ErrorCode::InvalidInput,
                "{name} with an unknown key in {path:?}: {}",
                e.message
            );
            assert!(e.message.contains("surprise"), "{name}: {}", e.message);
        }
    }
    assert_eq!(d.hello().state, DaemonState::Ready, "shutdown was refused");
    assert_eq!(rig.memories().len(), 1, "forget was refused");
}

/// One valid instance of every op.
fn sample_ops(id: MemoryId) -> Vec<Op> {
    let mut revise = ReviseParams::new(id);
    revise.title = Some("a title".to_owned());
    vec![
        Op::Hello,
        Op::Shutdown(ShutdownParams {
            reason: "x".to_owned(),
        }),
        Op::Remember(RememberParams {
            memories: vec![RememberItem::new("another memory")],
        }),
        Op::Recall(RecallParams::new("memory")),
        Op::Related(RelatedParams {
            id: Some(id),
            anchor: None,
            k: None,
            scope: None,
            limit: None,
        }),
        Op::Revise(revise),
        Op::Forget(ForgetParams {
            ids: vec![id],
            filter: None,
            confirm: false,
        }),
        Op::Link(LinkParams {
            action: LinkAction::List,
            src: None,
            dst: None,
            relation: None,
            end: Some(LinkEnd::Memory(id)),
            limit: None,
        }),
        Op::Regions(RegionsParams {
            coords: false,
            central: None,
            label: None,
            limit: None,
        }),
        Op::Journey(JourneyParams {
            from: id,
            to: MemoryId::new_v7(),
            steps: None,
            scope: None,
            limit: None,
        }),
        Op::Consolidate(ConsolidateParams {
            threshold: None,
            memory_type: Vec::new(),
            apply: false,
            groups: Vec::new(),
            max_groups: None,
            limit: None,
        }),
        Op::Share(ShareParams { ids: vec![id] }),
        Op::Sync(SyncParams::default()),
        Op::Status(StatusParams::default()),
        Op::Reinforce(ReinforceParams { id, rank: None }),
        Op::AmbientRecall(AmbientParams {
            prompt: "memory".to_owned(),
            top_k: 3,
            scope: Scope::Project,
        }),
        Op::Capture(CaptureParams {
            event: HookEvent::Stop,
            payload: json!({}),
            received_at: Timestamp::now(),
        }),
        Op::Import(ImportParams {
            path: "in.jsonl".to_owned(),
        }),
        Op::Export(ExportParams {
            path: "out.jsonl".to_owned(),
            project: None,
        }),
        Op::Migrate(MigrateParams {
            from: "v1".to_owned(),
            default_project: None,
        }),
        Op::Reindex(ReindexParams {}),
        Op::ProjectRename(ProjectRenameParams {
            from: "strict".to_owned(),
            to: "loose".to_owned(),
        }),
        Op::ProjectDelete(ProjectDeleteParams {
            name: "strict".to_owned(),
            confirm: true,
        }),
        Op::DoctorEval(DoctorEvalParams {
            set_path: "set.json".to_owned(),
            baseline_path: None,
        }),
    ]
}

// --- the model loading (SPEC 3, Runtime) -----------------------------------

#[test]
fn s3_hello_says_loading_and_writes_wait_while_reads_are_served() {
    let rig = Rig::new();
    let cwd = rig.project("slow");
    // An earlier daemon on this profile left two memories behind.
    let earlier = rig.start_fake();
    let gone = remember(&earlier, &cwd, "the invoice exporter runs hourly");
    let kept = remember(
        &earlier,
        &cwd,
        "the invoice archive is kept for seven years",
    );
    earlier.wait_embedded(&cwd);
    earlier.shutdown();

    let gate = GateLoader::new();
    let d = rig.start(gate.clone(), Arc::new(UnwiredSharing));
    assert_eq!(d.hello().state, DaemonState::Loading);

    let e = d.err(
        &cwd,
        Op::Remember(RememberParams {
            memories: vec![RememberItem::new("written while loading")],
        }),
    );
    assert_code(&e, ErrorCode::DaemonLoading);
    assert_code(
        &d.err(&cwd, Op::Reindex(ReindexParams {})),
        ErrorCode::DaemonLoading,
    );
    let vector = d.err(&cwd, Op::Recall(RecallParams::new("anything")));
    assert_code(&vector, ErrorCode::DaemonLoading);

    // Keyword recall, forget and status need no model.
    let mut keyword = RecallParams::new("invoice");
    keyword.mode = Some(RecallMode::Keyword);
    let found = recall(&d, &cwd, keyword);
    let mut ids: Vec<MemoryId> = found.results.iter().map(|m| m.id).collect();
    ids.sort();
    let mut want = vec![gone, kept];
    want.sort();
    assert_eq!(ids, want);
    let Reply::Forget(forgot) = d.ok(
        &cwd,
        Op::Forget(ForgetParams {
            ids: vec![gone],
            filter: None,
            confirm: false,
        }),
    ) else {
        panic!("not a forget reply")
    };
    assert_eq!(forgot.deleted, vec![gone]);
    let status = d.status(&cwd);
    assert_eq!(status.daemon.state, DaemonState::Loading);
    assert!(!status.model.loaded);
    assert_eq!(status.projects[0].memories, 1);

    gate.open();
    d.wait_ready();
    let info = d.hello();
    assert!(info.ready_seconds.is_some());
    assert!(d.status(&cwd).model.loaded);
    remember(&d, &cwd, "written once the model is ready");
}

// --- idle exit and the log (CONTRACTS 4, 8) --------------------------------

#[test]
fn s3_the_daemon_exits_when_idle() {
    let mut rig = Rig::new();
    rig.settings.daemon.idle_exit_minutes = 1;
    rig.opts.idle_unit = Duration::from_millis(300);
    let mut d = rig.start_fake();
    let started = Instant::now();
    let exit = d.take().wait();
    assert_eq!(exit, Exit::IdleExit);
    assert!(started.elapsed() >= Duration::from_millis(250));
}

#[test]
fn s3_idle_exit_waits_while_requests_keep_coming() {
    let mut rig = Rig::new();
    rig.settings.daemon.idle_exit_minutes = 1;
    rig.opts.idle_unit = Duration::from_millis(400);
    let d = rig.start_fake();
    let until = Instant::now() + Duration::from_millis(1200);
    while Instant::now() < until {
        assert_eq!(d.hello().state, DaemonState::Ready);
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(!d.handle().stopping());
}

#[test]
fn s3_the_idle_exit_line_says_minute_for_one_and_minutes_otherwise() {
    for (minutes, expected) in [
        (1, "no request for 1 minute; exiting"),
        (2, "no request for 2 minutes; exiting"),
    ] {
        let mut rig = Rig::new();
        rig.settings.daemon.idle_exit_minutes = minutes;
        rig.opts.idle_unit = Duration::from_millis(150);
        let mut d = rig.start_fake();
        let log = d.log_path();
        assert_eq!(d.take().wait(), Exit::IdleExit);
        let text = std::fs::read_to_string(&log).expect("the log");
        let lines: Vec<&str> = text
            .lines()
            .filter(|l| l.contains("no request for"))
            .collect();
        assert_eq!(lines.len(), 1, "{text}");
        assert!(lines[0].ends_with(expected), "{}", lines[0]);
    }
}

#[test]
fn s3_the_daemon_log_stays_bounded() {
    let mut rig = Rig::new();
    rig.opts.log_max_bytes = 2_000;
    // Every bad spool file is logged when it is moved aside.
    std::fs::create_dir_all(rig.paths.spool_new()).expect("spool/new");
    for i in 0..60 {
        let name = format!("20260101T000000.{i:06}Z-{i:04}.json");
        std::fs::write(rig.paths.spool_new().join(name), b"not a request").expect("spool");
    }
    let d = rig.start_fake();
    assert!(wait_until(Duration::from_secs(10), || {
        std::fs::read_dir(rig.paths.spool_new())
            .map(|r| r.count() == 0)
            .unwrap_or(false)
    }));
    let log = d.log_path();
    // Stopped, so no line is half way through a rotation.
    d.shutdown();
    let size = std::fs::metadata(&log).expect("the log").len();
    assert!(size <= 2_000, "log is {size} bytes");
    let rotated = log.with_extension("log.1");
    let old = std::fs::metadata(&rotated)
        .unwrap_or_else(|e| panic!("the log rotated to {}: {e}", rotated.display()))
        .len();
    assert!(old <= 2_000, "log.1 is {old} bytes");
}

// --- every op: a success and its main error (SPEC 8) -----------------------

#[test]
fn s3_hello_reports_identity() {
    let rig = Rig::new();
    let d = rig.start_fake();
    let info = d.hello();
    assert_eq!(info.protocol, PROTOCOL_VERSION);
    assert_eq!(info.build, sm_core::BUILD);
    assert_eq!(info.pid, std::process::id());
    assert_eq!(info.profile, "test");
    assert_eq!(info.model_id, sm_core::testing::fake_info().model_id);
    assert_eq!(info.endpoint, rig.endpoint.display_path());
}

#[test]
fn s8_remember_creates_then_finds_the_duplicate() {
    let rig = Rig::new();
    let d = rig.start_fake();
    let cwd = rig.project("alpha");
    let op = || {
        Op::Remember(RememberParams {
            memories: vec![RememberItem::new("we deploy on Tuesdays after the freeze")],
        })
    };
    let Reply::Remember(first) = d.ok(&cwd, op()) else {
        panic!("not a remember reply")
    };
    assert_eq!(first.memories.len(), 1);
    assert_eq!(first.memories[0].outcome, RememberOutcome::Created);
    assert_eq!(first.memories[0].project, "alpha");
    assert!(first.memories[0].chunks >= 1);
    let Reply::Remember(second) = d.ok(&cwd, op()) else {
        panic!("not a remember reply")
    };
    assert_eq!(second.memories[0].outcome, RememberOutcome::Duplicate);
    assert_eq!(second.memories[0].id, first.memories[0].id);

    let e = d.err(&cwd, Op::Remember(RememberParams { memories: vec![] }));
    assert_code(&e, ErrorCode::InvalidInput);
    let e = d.err(
        &cwd,
        Op::Remember(RememberParams {
            memories: vec![RememberItem::new("   ")],
        }),
    );
    assert_code(&e, ErrorCode::InvalidInput);
}

#[test]
fn s8_remember_in_an_unresolvable_folder_is_refused() {
    let rig = Rig::new();
    let d = rig.start_fake();
    let e = d.err(
        "",
        Op::Remember(RememberParams {
            memories: vec![RememberItem::new("no working directory")],
        }),
    );
    assert_code(&e, ErrorCode::ProjectUnresolved);
}

#[test]
fn s8_recall_finds_by_vector_and_refuses_a_blank_query() {
    let rig = Rig::new();
    let d = rig.start_fake();
    let cwd = rig.project("alpha");
    let id = remember(&d, &cwd, "the billing service retries webhooks three times");
    remember(&d, &cwd, "lunch is at noon on Fridays");
    d.wait_embedded(&cwd);
    let r = recall(&d, &cwd, RecallParams::new("billing webhooks retries"));
    assert_eq!(r.results.first().map(|m| m.id), Some(id));
    assert!(r.results[0].score.is_some());
    assert_code(
        &d.err(&cwd, Op::Recall(RecallParams::new(" "))),
        ErrorCode::InvalidInput,
    );
}

/// SPEC 8 output and B-28: `limit` is a token budget. Results are kept in
/// order until the next one does not fit and the rest are counted in
/// `omitted`; a first result too large for the budget alone is kept with
/// its content cut (e-report decision 24).
#[test]
fn s8_a_reply_fits_its_limit_and_counts_what_it_left_out() {
    let rig = Rig::new();
    let d = rig.start_fake();
    let cwd = rig.project("budget");
    for i in 0..10 {
        remember(
            &d,
            &cwd,
            &format!("budget note {i}: the nightly build publishes the documentation site"),
        );
    }
    d.wait_embedded(&cwd);
    let ask = |limit: Option<u32>| {
        let mut p = RecallParams::new("nightly build documentation site");
        p.mode = Some(RecallMode::Vector);
        p.k = Some(10);
        p.limit = limit;
        recall(&d, &cwd, p)
    };

    let full = ask(None);
    assert_eq!(full.results.len(), 10, "the default budget holds all ten");
    assert!(!full.truncated && full.omitted == 0);
    // The daemon's estimate: the reply's skeleton (no results, `truncated`
    // set, `omitted` at its largest) plus 8 tokens of slack, then each
    // result's JSON at four characters a token plus one. Room for three and
    // not a fourth.
    let cost = |m: &MemoryView| tokens(m) + 1;
    let skeleton = RecallReply {
        results: Vec::new(),
        weak_only: true,
        truncated: true,
        omitted: u32::MAX,
    };
    let three: usize = full.results[..3].iter().map(cost).sum();
    let limit = tokens(&skeleton) + 8 + three + cost(&full.results[3]) - 1;
    let fitted = ask(Some(u32::try_from(limit).expect("small")));
    let ids = |r: &RecallReply| r.results.iter().map(|m| m.id).collect::<Vec<_>>();
    assert_eq!(ids(&fitted), ids(&full)[..3].to_vec());
    assert_eq!(fitted.omitted, 7);
    assert!(fitted.truncated);
    assert!(fitted.results.iter().all(|m| !m.content_truncated));

    let long_cwd = rig.project("budget-long");
    let long = format!(
        "the release checklist: {}",
        "verify the signed tag; ".repeat(100)
    );
    let id = remember(&d, &long_cwd, &long);
    d.wait_embedded(&long_cwd);
    let ask_long = |limit: u32| {
        let mut p = RecallParams::new("release checklist signed tag");
        p.limit = Some(limit);
        recall(&d, &long_cwd, p)
    };
    // At the smallest limit (B-112), a result's fixed fields (identifier,
    // project, type, status, origin, score, importance, provenance,
    // timestamps) fit beside the reply's own, so the one result is returned
    // with its content cut, not only counted (B-111).
    let floor = sm_core::config::MIN_OUTPUT_TOKEN_BUDGET;
    for limit in [floor, 250] {
        let cut = ask_long(limit);
        assert_eq!(ids(&cut), vec![id], "at {limit}");
        assert!(cut.truncated && cut.omitted == 0, "at {limit}");
        assert!(tokens(&cut) <= limit as usize, "at {limit}");
        let m = &cut.results[0];
        assert!(m.content_truncated, "at {limit}");
        assert!(m.content.len() < long.len() && long.starts_with(&m.content));
    }
}

fn tokens<T: serde::Serialize>(v: &T) -> usize {
    sm_core::ambient::estimate_tokens(&serde_json::to_string(v).expect("json"))
}

/// A memory as large as remember accepts in every field, with content that
/// JSON escapes to more than its length.
fn heavy_item(n: usize) -> RememberItem {
    let mut item = RememberItem::new(format!(
        "heavy memory {n}: {}",
        "the \"deploy\" step\n".repeat(150)
    ));
    item.title = Some(format!("title {n} {}", "t".repeat(1500)));
    item.tags = (0..30)
        .map(|i| format!("tag{n}-{i}-{}", "x".repeat(50)))
        .collect();
    item.anchors = (0..30)
        .map(|i| {
            AnchorRef::new(AnchorKind::File, format!("src/{n}/{i}/{}", "y".repeat(240)))
                .expect("an anchor")
        })
        .collect();
    item.author = Some("a".repeat(300));
    item.file_path = Some(format!("docs/{}", "f".repeat(300)));
    item
}

/// SPEC 8 Output, stream H 10: a memory whose content is too short to cut
/// keeps it whole and sheds its title, tags and anchors to fit the smallest
/// limit; the result still says it was cut.
#[test]
fn s8_a_short_memory_cut_to_fit_says_it_was_cut() {
    let rig = Rig::new();
    let d = rig.start_fake();
    let cwd = rig.project("short-cut");
    let mut item = heavy_item(0);
    item.content = "deploy on fridays".to_owned();
    let id = remember_item(&d, &cwd, item);
    d.wait_embedded(&cwd);
    let floor = sm_core::config::MIN_OUTPUT_TOKEN_BUDGET;
    let mut p = RecallParams::new("deploy on fridays");
    p.limit = Some(floor);
    let r = recall(&d, &cwd, p);
    assert!(tokens(&r) <= floor as usize);
    assert_eq!(r.results.iter().map(|m| m.id).collect::<Vec<_>>(), vec![id]);
    let m = &r.results[0];
    assert_eq!(m.content, "deploy on fridays", "the content is kept whole");
    assert!(m.tags.is_empty() && m.anchors.is_empty(), "{m:?}");
    assert!(m.content_truncated, "a caller can tell the view was cut");
    assert!(r.truncated);
}

/// SPEC 8 Output: no tool returns more than its token budget, in the worst
/// case: memories with every field long, at the smallest limit and above.
/// A first result that cannot fit whole sheds fields until it fits, or is
/// counted in `omitted`; `status`, whose fixed fields alone can pass a small
/// limit, is refused with the smallest limit that works, and fits that
/// (e-report decision 24).
#[test]
fn s8_no_reply_exceeds_its_limit_in_the_worst_case() {
    let rig = Rig::new();
    let d = rig.start_fake();
    let cwd = rig.project("worst-case");
    let ids: Vec<MemoryId> = (0..4)
        .map(|n| remember_item(&d, &cwd, heavy_item(n)))
        .collect();
    d.ok(
        &cwd,
        Op::Link(LinkParams {
            action: LinkAction::Add,
            src: Some(LinkEnd::Memory(ids[0])),
            dst: Some(LinkEnd::Memory(ids[1])),
            relation: Some(Relation::DependsOn),
            end: None,
            limit: None,
        }),
    );
    d.wait_embedded(&cwd);

    // Below the floor, the limit is refused before any tool runs (B-111).
    let floor = sm_core::config::MIN_OUTPUT_TOKEN_BUDGET;
    for limit in [100u32, 120, 150, floor - 1] {
        let mut p = RecallParams::new("the deploy step of a heavy memory");
        p.limit = Some(limit);
        let e = d
            .call(&cwd, Op::Recall(p))
            .into_result()
            .expect_err("a limit below the floor is refused");
        assert_code(&e, ErrorCode::InvalidInput);
        assert!(e.message.contains(&floor.to_string()), "{}", e.message);
        let e = d
            .call(
                &cwd,
                Op::Status(StatusParams {
                    metrics: true,
                    limit: Some(limit),
                }),
            )
            .into_result()
            .expect_err("a limit below the floor is refused");
        assert_code(&e, ErrorCode::InvalidInput);
    }

    let mut cut_kept = false;
    for limit in [floor, 250, 400, 1000, 4000] {
        let l = Some(limit);
        let within = |tool: &str, used: usize| {
            assert!(
                used <= limit as usize,
                "{tool} at limit {limit}: {used} tokens"
            );
        };

        let mut p = RecallParams::new("the deploy step of a heavy memory");
        p.mode = Some(RecallMode::Vector);
        p.k = Some(4);
        p.limit = l;
        let r = recall(&d, &cwd, p);
        within("recall", tokens(&r));
        assert_eq!(r.results.len() + r.omitted as usize, 4, "at {limit}");
        assert!(r.truncated, "recall at {limit} says it cut");
        if let Some(m) = r.results.first() {
            assert!(m.score.is_some(), "at {limit}");
            cut_kept |= m.content_truncated;
        }

        let Reply::Related(r) = d.ok(
            &cwd,
            Op::Related(RelatedParams {
                id: Some(ids[0]),
                anchor: None,
                k: None,
                scope: None,
                limit: l,
            }),
        ) else {
            panic!("not a related reply")
        };
        within("related", tokens(&r));
        assert!(r.truncated, "related at {limit} says it cut");

        let Reply::Regions(r) = d.ok(
            &cwd,
            Op::Regions(RegionsParams {
                coords: true,
                central: None,
                label: None,
                limit: l,
            }),
        ) else {
            panic!("not a regions reply")
        };
        within("regions", tokens(&r));

        let Reply::Journey(r) = d.ok(
            &cwd,
            Op::Journey(JourneyParams {
                from: ids[0],
                to: ids[3],
                steps: Some(4),
                scope: None,
                limit: l,
            }),
        ) else {
            panic!("not a journey reply")
        };
        within("journey", tokens(&r));

        let Reply::Consolidate(r) = d.ok(
            &cwd,
            Op::Consolidate(ConsolidateParams {
                threshold: Some(0.7),
                memory_type: Vec::new(),
                apply: false,
                groups: Vec::new(),
                max_groups: None,
                limit: l,
            }),
        ) else {
            panic!("not a consolidate reply")
        };
        within("consolidate", tokens(&r));

        let Reply::Link(r) = d.ok(
            &cwd,
            Op::Link(LinkParams {
                action: LinkAction::List,
                src: None,
                dst: None,
                relation: None,
                end: Some(LinkEnd::Memory(ids[0])),
                limit: l,
            }),
        ) else {
            panic!("not a link reply")
        };
        within("link", tokens(&r));

        let status = |limit: u32| {
            Op::Status(StatusParams {
                metrics: true,
                limit: Some(limit),
            })
        };
        match d.call(&cwd, status(limit)).into_result() {
            Ok(Reply::Status(s)) => within("status", tokens(&*s)),
            Ok(other) => panic!("status replied {other:?}"),
            Err(e) => {
                assert_code(&e, ErrorCode::InvalidInput);
                let least: u32 = e
                    .next_action
                    .split_whitespace()
                    .find_map(|w| w.trim_end_matches(',').parse().ok())
                    .expect("the smallest limit that works");
                assert!(least > limit, "{}", e.message);
                let Reply::Status(s) = d.ok(&cwd, status(least)) else {
                    panic!("not a status reply")
                };
                assert!(
                    tokens(&*s) <= least as usize,
                    "status at its smallest limit {least}"
                );
            }
        }
    }
    assert!(cut_kept, "a cut first result is kept at some limit");
}

#[test]
fn s8_related_by_id_and_its_errors() {
    let rig = Rig::new();
    let d = rig.start_fake();
    let cwd = rig.project("alpha");
    let a = remember(&d, &cwd, "the cache is warmed by the nightly job");
    let b = remember(&d, &cwd, "the nightly job also warms the search cache");
    d.wait_embedded(&cwd);
    let reply = d.ok(
        &cwd,
        Op::Related(RelatedParams {
            id: Some(a),
            anchor: None,
            k: None,
            scope: None,
            limit: None,
        }),
    );
    let Reply::Related(r) = reply else {
        panic!("not a related reply")
    };
    assert_eq!(r.from, LinkEnd::Memory(a));
    assert!(r.by_vector.iter().any(|m| m.id == b));
    assert!(r.by_vector.iter().all(|m| m.id != a));
    let none = RelatedParams {
        id: None,
        anchor: None,
        k: None,
        scope: None,
        limit: None,
    };
    assert_code(&d.err(&cwd, Op::Related(none)), ErrorCode::InvalidInput);
    let missing = RelatedParams {
        id: Some(MemoryId::new_v7()),
        anchor: None,
        k: None,
        scope: None,
        limit: None,
    };
    assert_code(&d.err(&cwd, Op::Related(missing)), ErrorCode::NotFound);
}

#[test]
fn s8_revise_changes_content_and_its_errors() {
    let rig = Rig::new();
    let d = rig.start_fake();
    let cwd = rig.project("alpha");
    let id = remember(&d, &cwd, "the queue holds 100 jobs");
    let mut p = ReviseParams::new(id);
    p.content = Some("the queue holds 500 jobs".to_owned());
    p.memory_type = Some(MemoryType::Convention);
    let Reply::Revise(r) = d.ok(&cwd, Op::Revise(p)) else {
        panic!("not a revise reply")
    };
    assert_eq!(r.memory.content, "the queue holds 500 jobs");
    assert_eq!(r.memory.memory_type, MemoryType::Convention);
    assert!(r.reembedded);
    assert_code(
        &d.err(&cwd, Op::Revise(ReviseParams::new(id))),
        ErrorCode::InvalidInput,
    );
    let mut missing = ReviseParams::new(MemoryId::new_v7());
    missing.title = Some("t".to_owned());
    assert_code(&d.err(&cwd, Op::Revise(missing)), ErrorCode::NotFound);
}

#[test]
fn s8_forget_needs_confirm_for_more_than_one_memory() {
    let rig = Rig::new();
    let d = rig.start_fake();
    let cwd = rig.project("alpha");
    let a = remember(&d, &cwd, "a memory to forget");
    let b = remember(&d, &cwd, "a second note");
    let c = remember(&d, &cwd, "a third note");
    let forget = |ids: Vec<MemoryId>, filter: Option<ForgetFilter>, confirm: bool| {
        Op::Forget(ForgetParams {
            ids,
            filter,
            confirm,
        })
    };
    let notes = || ForgetFilter {
        scope: None,
        memory_type: vec![MemoryType::Note],
        status: Vec::new(),
        anchor: None,
        tag: None,
        created_before: None,
    };
    // One memory needs no confirm (SPEC 8, `forget`).
    let Reply::Forget(r) = d.ok(&cwd, forget(vec![a], None, false)) else {
        panic!("not a forget reply")
    };
    assert_eq!(r.deleted, vec![a]);
    assert_eq!(rig.memories().len(), 2);
    // More than one, by id or by filter, does; nothing is deleted.
    let e = d.err(&cwd, forget(vec![b, c], None, false));
    assert_code(&e, ErrorCode::ConfirmRequired);
    let e = d.err(&cwd, forget(Vec::new(), Some(notes()), false));
    assert_code(&e, ErrorCode::ConfirmRequired);
    assert!(e.message.contains('2'), "{}", e.message);
    assert_eq!(rig.memories().len(), 2);
    let Reply::Forget(r) = d.ok(&cwd, forget(Vec::new(), Some(notes()), true)) else {
        panic!("not a forget reply")
    };
    let mut deleted = r.deleted;
    deleted.sort();
    let mut want = vec![b, c];
    want.sort();
    assert_eq!(deleted, want);
    assert!(rig.memories().is_empty());
    // An unknown id is not found.
    let e = d.err(&cwd, forget(vec![MemoryId::new_v7()], None, false));
    assert_code(&e, ErrorCode::NotFound);
    let e = d.err(
        &cwd,
        Op::Forget(ForgetParams {
            ids: Vec::new(),
            filter: None,
            confirm: false,
        }),
    );
    assert_code(&e, ErrorCode::InvalidInput);
}

#[test]
fn s8_link_add_list_remove_and_its_errors() {
    let rig = Rig::new();
    let d = rig.start_fake();
    let cwd = rig.project("alpha");
    let a = remember(&d, &cwd, "the api gateway calls the auth service");
    let b = remember(&d, &cwd, "the auth service owns the sessions table");
    let add = |action: LinkAction| LinkParams {
        action,
        src: Some(LinkEnd::Memory(a)),
        dst: Some(LinkEnd::Memory(b)),
        relation: Some(Relation::DependsOn),
        end: None,
        limit: None,
    };
    let Reply::Link(added) = d.ok(&cwd, Op::Link(add(LinkAction::Add))) else {
        panic!("not a link reply")
    };
    assert_eq!(added.action, LinkAction::Add);
    assert!(added.link.is_some());
    let list = LinkParams {
        action: LinkAction::List,
        src: None,
        dst: None,
        relation: None,
        end: Some(LinkEnd::Memory(b)),
        limit: None,
    };
    let Reply::Link(listed) = d.ok(&cwd, Op::Link(list.clone())) else {
        panic!("not a link reply")
    };
    assert_eq!(listed.links.len(), 1);
    let Reply::Link(removed) = d.ok(&cwd, Op::Link(add(LinkAction::Remove))) else {
        panic!("not a link reply")
    };
    assert_eq!(removed.removed, 1);
    let Reply::Link(empty) = d.ok(&cwd, Op::Link(list)) else {
        panic!("not a link reply")
    };
    assert!(empty.links.is_empty());

    let mut bad = add(LinkAction::List);
    bad.end = None;
    assert_code(&d.err(&cwd, Op::Link(bad)), ErrorCode::InvalidInput);
    let mut missing = add(LinkAction::Add);
    missing.dst = Some(LinkEnd::Memory(MemoryId::new_v7()));
    assert_code(&d.err(&cwd, Op::Link(missing)), ErrorCode::NotFound);
}

#[test]
fn s8_regions_and_its_error() {
    let rig = Rig::new();
    let d = rig.start_fake();
    let cwd = rig.project("alpha");
    for i in 0..6 {
        remember(
            &d,
            &cwd,
            &format!("memory number {i} about the payment flow"),
        );
    }
    d.wait_embedded(&cwd);
    let p = RegionsParams {
        coords: false,
        central: None,
        label: None,
        limit: None,
    };
    let Reply::Regions(r) = d.ok(&cwd, Op::Regions(p.clone())) else {
        panic!("not a regions reply")
    };
    assert_eq!(r.project, "alpha");
    assert_eq!(r.memories, 6);
    let mut bad = p;
    bad.central = Some(0);
    assert_code(&d.err(&cwd, Op::Regions(bad)), ErrorCode::InvalidInput);
}

#[test]
fn s8_journey_and_its_error() {
    let rig = Rig::new();
    let d = rig.start_fake();
    let cwd = rig.project("alpha");
    let a = remember(&d, &cwd, "the build starts with a lint step");
    let b = remember(&d, &cwd, "the release ends with a tag and a changelog");
    remember(&d, &cwd, "the lint step runs before the release tag");
    d.wait_embedded(&cwd);
    let p = JourneyParams {
        from: a,
        to: b,
        steps: Some(4),
        scope: None,
        limit: None,
    };
    let Reply::Journey(j) = d.ok(&cwd, Op::Journey(p.clone())) else {
        panic!("not a journey reply")
    };
    assert_eq!((j.from, j.to), (a, b));
    assert!(!j.steps.is_empty());
    let mut same = p;
    same.to = a;
    assert_code(&d.err(&cwd, Op::Journey(same)), ErrorCode::InvalidInput);
}

#[test]
fn s8_consolidate_proposes_then_applies_and_its_error() {
    let rig = Rig::new();
    let d = rig.start_fake();
    let cwd = rig.project("alpha");
    let keep = remember(
        &d,
        &cwd,
        "alpha beta gamma delta epsilon zeta eta theta iota kappa",
    );
    let dup = remember(
        &d,
        &cwd,
        "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda",
    );
    d.wait_embedded(&cwd);
    let propose = ConsolidateParams {
        threshold: None,
        memory_type: Vec::new(),
        apply: false,
        groups: Vec::new(),
        max_groups: None,
        limit: None,
    };
    let Reply::Consolidate(c) = d.ok(&cwd, Op::Consolidate(propose.clone())) else {
        panic!("not a consolidate reply")
    };
    assert!(!c.applied);
    assert_eq!(c.proposals.len(), 1, "{c:?}");
    let group = &c.proposals[0];
    let mut ids = vec![group.keep.id, group.duplicates[0].memory.id];
    ids.sort();
    let mut want = vec![keep, dup];
    want.sort();
    assert_eq!(ids, want);

    let mut apply = propose.clone();
    apply.apply = true;
    apply.groups = vec![MergeGroup {
        keep,
        merge: vec![dup],
    }];
    let Reply::Consolidate(applied) = d.ok(&cwd, Op::Consolidate(apply)) else {
        panic!("not a consolidate reply")
    };
    assert!(applied.applied);
    assert_eq!(applied.merged.len(), 1);

    let mut bad = propose;
    bad.apply = true;
    assert_code(&d.err(&cwd, Op::Consolidate(bad)), ErrorCode::InvalidInput);
}

#[test]
fn s8_status_reports_the_profile() {
    let rig = Rig::new();
    let d = rig.start_fake();
    let cwd = rig.project("alpha");
    remember(&d, &cwd, "one memory");
    d.wait_embedded(&cwd);
    let s = d.status(&cwd);
    assert_eq!(s.daemon.state, DaemonState::Ready);
    assert!(s.model.loaded);
    assert!(s.index.matches);
    assert_eq!(s.current_project.as_deref(), Some("alpha"));
    assert_eq!(s.projects.len(), 1);
    assert_eq!(s.projects[0].memories, 1);
    assert_eq!(s.spool_pending, 0);
    assert!(s.metrics.is_none());
    let Reply::Status(m) = d.ok(
        &cwd,
        Op::Status(StatusParams {
            metrics: true,
            limit: None,
        }),
    ) else {
        panic!("not a status reply")
    };
    assert!(m.metrics.is_some());
    // A folder that is no project is not an error for status.
    let home_like = d.status("");
    assert!(home_like.current_project.is_none());
    // Its main error is the shared envelope check: another profile.
    let mut req = d.req(&cwd, Op::Status(StatusParams::default()));
    req.profile = sm_core::config::ProfileName::parse("other").expect("a profile");
    assert_code(&err_of(d.send(&req)), ErrorCode::InvalidInput);
}

/// B-151 (D18): `status` reads. Asked from a project folder the store has
/// never seen, it reports no current project and creates none; `doctor`
/// sends it. A project that exists is still reported, under its stored
/// name.
#[test]
fn s8_status_never_creates_a_project() {
    let rig = Rig::new();
    let d = rig.start_fake();
    let fresh = rig.project("fresh");
    let s = d.status(&fresh);
    assert!(s.current_project.is_none(), "{:?}", s.current_project);
    assert!(s.projects.is_empty(), "status created {:?}", s.projects);
    let plain = rig.plain_dir("loose");
    assert!(d.status(&plain).projects.is_empty());
    remember(&d, &fresh, "one memory");
    let s = d.status(&fresh);
    assert_eq!(s.current_project.as_deref(), Some("fresh"));
    assert_eq!(s.projects.len(), 1);
}

/// B-151 (D18): `doctor --eval` on the live store reads. A set without
/// memories, run from a project folder the store has never seen, retrieves
/// nothing for each query and creates no project.
#[test]
fn s11_doctor_eval_on_the_live_store_never_creates_a_project() {
    let rig = Rig::new();
    let d = rig.start_fake();
    let fresh = rig.project("fresh");
    let set = rig.tmp.path().join("live-set.json");
    std::fs::write(
        &set,
        r#"{"queries": [{"text": "why the cache is keyed by folder", "ground_truth_ids": ["m1"]}]}"#,
    )
    .expect("write the set");
    let Reply::Eval(report) = d.ok(
        &fresh,
        Op::DoctorEval(DoctorEvalParams {
            set_path: set.to_string_lossy().into_owned(),
            baseline_path: None,
        }),
    ) else {
        panic!("not an eval reply")
    };
    assert_eq!(report.queries, 1);
    let s = d.status(&fresh);
    assert!(
        s.projects.is_empty(),
        "doctor --eval created {:?}",
        s.projects
    );
}

#[test]
fn s8_reinforce_after_recall_and_its_errors() {
    let rig = Rig::new();
    let d = rig.start_fake();
    let cwd = rig.project("alpha");
    let id = remember(&d, &cwd, "the feature flags live in the config service");
    d.wait_embedded(&cwd);
    let session = Some("s-reinforce".to_owned());
    let req = d
        .req(&cwd, Op::Recall(RecallParams::new("feature flags config")))
        .with_session(session.clone());
    assert!(matches!(d.send(&req).into_result(), Ok(Reply::Recall(_))));
    let req = d
        .req(&cwd, Op::Reinforce(ReinforceParams { id, rank: None }))
        .with_session(session);
    let Ok(Reply::Reinforce(r)) = d.send(&req).into_result() else {
        panic!("not a reinforce reply")
    };
    assert_eq!(r.id, id);
    assert_eq!(r.rank_source, RankSource::LastRecall);
    assert_eq!(r.rank, Some(1));
    let given = d.ok(&cwd, Op::Reinforce(ReinforceParams { id, rank: Some(2) }));
    let Reply::Reinforce(g) = given else {
        panic!("not a reinforce reply")
    };
    assert_eq!(g.rank_source, RankSource::Given);
    assert_code(
        &d.err(&cwd, Op::Reinforce(ReinforceParams { id, rank: Some(0) })),
        ErrorCode::InvalidInput,
    );
    assert_code(
        &d.err(
            &cwd,
            Op::Reinforce(ReinforceParams {
                id: MemoryId::new_v7(),
                rank: None,
            }),
        ),
        ErrorCode::NotFound,
    );
}

#[test]
fn s6_ambient_recall_answers_and_refuses_a_blank_prompt() {
    let rig = Rig::new();
    let d = rig.start_fake();
    let cwd = rig.project("alpha");
    remember(&d, &cwd, "the staging database is reset every Sunday night");
    d.wait_embedded(&cwd);
    let p = |prompt: &str| AmbientParams {
        prompt: prompt.to_owned(),
        top_k: 3,
        scope: Scope::Project,
    };
    let Reply::Ambient(a) = d.ok(
        &cwd,
        Op::AmbientRecall(p("when is the staging database reset every Sunday?")),
    ) else {
        panic!("not an ambient reply")
    };
    assert_eq!(a.project.as_deref(), Some("alpha"));
    assert!(!a.top_scores.is_empty());
    assert_code(
        &d.err(&cwd, Op::AmbientRecall(p("  "))),
        ErrorCode::InvalidInput,
    );
}

#[test]
fn s16_capture_accepts_capture_events_only() {
    let rig = Rig::new();
    let d = rig.start_fake();
    let cwd = rig.project("alpha");
    let capture = |event: HookEvent| {
        Op::Capture(CaptureParams {
            event,
            payload: json!({"session_id": "s1", "cwd": cwd}),
            received_at: Timestamp::now(),
        })
    };
    for event in [
        HookEvent::PostToolUse,
        HookEvent::PreCompact,
        HookEvent::Stop,
    ] {
        assert_eq!(
            d.ok(&cwd, capture(event)),
            Reply::Captured(CaptureReply { accepted: true }),
            "{event}"
        );
    }
    for event in [HookEvent::SessionStart, HookEvent::UserPromptSubmit] {
        assert_eq!(
            d.ok(&cwd, capture(event)),
            Reply::Captured(CaptureReply { accepted: false }),
            "{event}"
        );
    }
}

#[test]
fn s16_capture_is_refused_when_capture_is_disabled() {
    let mut rig = Rig::new();
    rig.settings.capture.enabled = false;
    let d = rig.start_fake();
    let cwd = rig.project("alpha");
    let r = d.ok(
        &cwd,
        Op::Capture(CaptureParams {
            event: HookEvent::Stop,
            payload: json!({"session_id": "s1"}),
            received_at: Timestamp::now(),
        }),
    );
    assert_eq!(r, Reply::Captured(CaptureReply { accepted: false }));
}

#[test]
fn s9_share_and_sync_need_a_repository_and_the_wired_implementation() {
    let rig = Rig::new();
    let d = rig.start_fake();
    let repo = rig.project("alpha");
    let plain = rig.plain_dir("loose");
    let id = remember(&d, &repo, "a memory to share");
    let share = || Op::Share(ShareParams { ids: vec![id] });
    let sync = || Op::Sync(SyncParams { dry_run: true });
    assert_code(&d.err(&plain, share()), ErrorCode::NotGitRepository);
    assert_code(&d.err(&plain, sync()), ErrorCode::NotGitRepository);
    // Without stream F's implementation the daemon says so.
    assert_code(&d.err(&repo, share()), ErrorCode::Internal);
    assert_code(&d.err(&repo, sync()), ErrorCode::Internal);
    assert_code(
        &d.err(&repo, Op::Share(ShareParams { ids: vec![] })),
        ErrorCode::InvalidInput,
    );
}

/// SPEC 9, e-report decision 30: a folder that becomes a repository after
/// the daemon resolved it is one from the next request, with no restart.
#[test]
fn s9_a_folder_that_becomes_a_repository_is_one_without_a_restart() {
    let rig = Rig::new();
    let sharing = RecordingSharing::new();
    let d = rig.start(FakeLoader::new(), sharing.clone());
    d.wait_ready();
    let dir = rig.plain_dir("later-a-repo");
    remember(&d, &dir, "a memory stored before git init");
    let sync = || Op::Sync(SyncParams { dry_run: true });
    assert_code(&d.err(&dir, sync()), ErrorCode::NotGitRepository);
    assert!(sharing.calls().is_empty());

    let git = std::path::Path::new(&dir).join(".git");
    std::fs::create_dir_all(&git).expect("git init");
    std::fs::write(git.join("HEAD"), "ref: refs/heads/main\n").expect("write HEAD");
    let Reply::Sync(s) = d.ok(&dir, sync()) else {
        panic!("not a sync reply")
    };
    assert!(s.dry_run);
    let calls = sharing.calls();
    assert_eq!(calls.len(), 1, "{calls:?}");
    let root = std::fs::canonicalize(&dir).expect("the folder");
    let got = std::fs::canonicalize(calls[0].repo_root.as_ref().expect("a repository root"))
        .expect("the root");
    assert_eq!(got, root);
}

#[test]
fn s9_sharing_ops_reach_the_implementation_with_the_resolved_project() {
    let mut rig = Rig::new();
    // Not the default, so the cap the context carries is seen to come from
    // the settings (B-134).
    rig.settings.capture.max_index_tokens = 4_000;
    let sharing = RecordingSharing::new();
    let d = rig.start(FakeLoader::new(), sharing.clone());
    d.wait_ready();
    let repo = rig.project("alpha");
    let id = remember(&d, &repo, "a memory to share");
    assert!(matches!(
        d.ok(&repo, Op::Share(ShareParams { ids: vec![id] })),
        Reply::Share(_)
    ));
    let Reply::Sync(s) = d.ok(&repo, Op::Sync(SyncParams { dry_run: true })) else {
        panic!("not a sync reply")
    };
    assert!(s.dry_run);
    let file = |name: &str| rig.tmp.path().join(name).display().to_string();
    let Reply::Import(i) = d.ok(
        &repo,
        Op::Import(ImportParams {
            path: file("in.jsonl"),
        }),
    ) else {
        panic!("not an import reply")
    };
    assert_eq!(i.inserted, 1);
    let Reply::Import(m) = d.ok(
        &repo,
        Op::Migrate(MigrateParams {
            from: file("v1-data"),
            default_project: None,
        }),
    ) else {
        panic!("a migrate reply is an import report")
    };
    assert_eq!(m.source, file("v1-data"));
    let Reply::Export(x) = d.ok(
        &repo,
        Op::Export(ExportParams {
            path: file("out.jsonl"),
            project: None,
        }),
    ) else {
        panic!("not an export reply")
    };
    assert_eq!(x.memories, 3);

    let calls = sharing.calls();
    let ops: Vec<&str> = calls.iter().map(|c| c.op).collect();
    assert_eq!(ops, ["share", "sync", "import", "migrate", "export"]);
    for c in &calls {
        assert_eq!(c.project.as_deref(), Some("alpha"), "{c:?}");
        assert!(c.had_chunker, "{c:?}");
        // A captured memory is chunked with its capture cap (SPEC 16; B-134).
        assert_eq!(c.capture_index_cap, Some(4_000), "{c:?}");
    }
    let root = std::fs::canonicalize(&repo).expect("the repository");
    for c in &calls[..2] {
        let got = std::fs::canonicalize(c.repo_root.as_ref().expect("a repository root"))
            .expect("the root");
        assert_eq!(got, root, "{c:?}");
    }
    // The imported memories were queued and embedded.
    d.wait_embedded(&repo);
}

#[test]
fn s9_import_export_and_migrate_without_the_implementation_say_so() {
    let rig = Rig::new();
    let d = rig.start_fake();
    let cwd = rig.project("alpha");
    let file = |name: &str| rig.tmp.path().join(name).display().to_string();
    let import = Op::Import(ImportParams {
        path: file("in.jsonl"),
    });
    let export = Op::Export(ExportParams {
        path: file("out.jsonl"),
        project: None,
    });
    let migrate = Op::Migrate(MigrateParams {
        from: file("v1"),
        default_project: None,
    });
    for op in [import, export, migrate] {
        let e = d.err(&cwd, op);
        assert_code(&e, ErrorCode::Internal);
        assert!(
            e.message.contains("sharing implementation"),
            "{}",
            e.message
        );
    }
}

/// CONTRACTS 5 (Requests): the file paths of import, export, migrate and
/// doctor_eval are absolute, resolved by the client. A relative path would
/// resolve against the daemon's working folder, so it is refused before
/// the op runs, with a next action (e-report decision 36).
#[test]
fn s9_a_relative_file_path_is_refused_before_the_op_runs() {
    let rig = Rig::new();
    let sharing = RecordingSharing::new();
    let d = rig.start(FakeLoader::new(), sharing.clone());
    d.wait_ready();
    let cwd = rig.project("alpha");
    let set = fixture_path("retrieval/retrieval_eval_set.json")
        .to_string_lossy()
        .into_owned();
    let cases = [
        (
            Op::Import(ImportParams {
                path: "in.jsonl".to_owned(),
            }),
            "`path`",
        ),
        (
            Op::Export(ExportParams {
                path: "out.jsonl".to_owned(),
                project: None,
            }),
            "`path`",
        ),
        (
            Op::Migrate(MigrateParams {
                from: "v1.json".to_owned(),
                default_project: None,
            }),
            "`from`",
        ),
        (
            Op::DoctorEval(DoctorEvalParams {
                set_path: "set.json".to_owned(),
                baseline_path: None,
            }),
            "`set_path`",
        ),
        (
            Op::DoctorEval(DoctorEvalParams {
                set_path: set,
                baseline_path: Some("baseline.json".to_owned()),
            }),
            "`baseline_path`",
        ),
    ];
    for (op, field) in cases {
        let name = op.name();
        let e = d.err(&cwd, op);
        assert_code(&e, ErrorCode::InvalidInput);
        assert!(e.message.contains(field), "{name}: {}", e.message);
        assert!(e.message.contains("absolute"), "{name}: {}", e.message);
        assert!(!e.next_action.is_empty(), "{name}: no next action");
    }
    assert!(
        sharing.calls().is_empty(),
        "no op reached the implementation"
    );
}

#[test]
fn s5_reindex_requeues_every_chunk() {
    let rig = Rig::new();
    let d = rig.start_fake();
    let cwd = rig.project("alpha");
    remember(&d, &cwd, "first memory");
    remember(&d, &cwd, "second memory");
    d.wait_embedded(&cwd);
    let Reply::Reindex(r) = d.ok(&cwd, Op::Reindex(ReindexParams {})) else {
        panic!("not a reindex reply")
    };
    assert_eq!(r.memories, 2);
    assert!(r.chunks_queued >= 2);
    assert!(std::path::Path::new(&r.backup).exists(), "{}", r.backup);
    d.wait_embedded(&cwd);
    assert!(d.status(&cwd).index.matches);
}

/// The fake provider, except that a document batch holding a chunk with
/// `PANICKY` panics and one holding `POISON` fails (`internal`).
struct PoisonProvider(sm_core::testing::FakeProvider);

struct PoisonEmbedder(Box<dyn sm_core::traits::Embedder>);

impl sm_core::traits::EmbedderProvider for PoisonProvider {
    fn info(&self) -> &sm_core::traits::EmbedderInfo {
        self.0.info()
    }
    fn open(
        &self,
        kind: sm_core::traits::SessionKind,
        threads: usize,
    ) -> sm_core::error::Result<Box<dyn sm_core::traits::Embedder>> {
        Ok(Box::new(PoisonEmbedder(self.0.open(kind, threads)?)))
    }
}

impl sm_core::traits::Embedder for PoisonEmbedder {
    fn info(&self) -> &sm_core::traits::EmbedderInfo {
        self.0.info()
    }
    fn embed_query(&self, text: &str, max_tokens: usize) -> sm_core::error::Result<Vec<f32>> {
        self.0.embed_query(text, max_tokens)
    }
    fn embed_documents(
        &self,
        docs: &[sm_core::traits::DocInput<'_>],
    ) -> sm_core::error::Result<Vec<Vec<f32>>> {
        if docs.iter().any(|d| d.text.contains("PANICKY")) {
            panic!("injected embedder panic");
        }
        if docs.iter().any(|d| d.text.contains("POISON")) {
            return Err(sm_core::error::CoreError::internal(
                "injected: this chunk never embeds",
            ));
        }
        self.0.embed_documents(docs)
    }
}

/// e-report decision 33: a batch that fails on a chunk, or panics in the
/// embedder, is retried one chunk at a time, so the other chunks are
/// stored; a chunk that fails alone in three rounds in which another chunk
/// was stored (stream H 8) is skipped with a log line and stays pending;
/// the thread survives the panic and embeds later chunks.
#[test]
fn s5_a_chunk_that_always_fails_is_skipped_and_the_queue_moves_on() {
    let rig = Rig::new();
    let d = rig.start(
        FakeLoader::with(Arc::new(PoisonProvider(
            sm_core::testing::FakeProvider::new(),
        ))),
        Arc::new(UnwiredSharing),
    );
    d.wait_ready();
    let cwd = rig.project("alpha");
    // One write, so the failing chunks come first in each batch (pending
    // chunks are taken in id order, `embed.batch_size` 4 at a time): three
    // rounds of the two bad chunks and two good ones, so each round stores
    // a chunk and strikes the bad ones.
    let contents = [
        "POISON a chunk the embedder always refuses",
        "PANICKY a chunk that makes the embedder panic",
        "the build cache lives in target",
        "the deploy script runs from the tools folder",
        "the test fixtures are data, not code",
        "the changelog is kept by hand",
        "the schema version lives in the meta table",
        "the socket folder is private to the user",
    ];
    match d.ok(
        &cwd,
        Op::Remember(RememberParams {
            memories: contents.iter().map(|c| RememberItem::new(*c)).collect(),
        }),
    ) {
        Reply::Remember(r) => assert_eq!(r.memories.len(), 8),
        other => panic!("remember replied {other:?}"),
    }
    let log = || std::fs::read_to_string(d.log_path()).unwrap_or_default();
    assert!(
        wait_until(Duration::from_secs(20), || {
            d.status(&cwd).index.pending_embeddings == 2
                && log().matches("skipped after failing alone 3 times").count() == 2
        }),
        "the six good chunks were stored and the two bad ones skipped: {} pending\n{}",
        d.status(&cwd).index.pending_embeddings,
        log()
    );
    let text = log();
    assert!(
        text.contains("a batch of 4 chunks failed; embedding them one at a time"),
        "{text}"
    );
    assert!(
        text.contains("internal: the embedder panicked: injected embedder panic"),
        "{text}"
    );
    assert!(
        text.contains("internal: injected: this chunk never embeds"),
        "{text}"
    );
    assert!(
        text.contains("it stays pending and a later daemon tries it again"),
        "{text}"
    );
    assert_eq!(
        text.matches("a batch of 4 chunks failed").count(),
        3,
        "three rounds, each storing two good chunks: {text}"
    );
    assert!(
        text.contains("failed alone (1 of 3)") && text.contains("failed alone (2 of 3)"),
        "{text}"
    );

    // The thread survived the panic: a later chunk is embedded, and the
    // skipped ones are not offered again.
    remember(&d, &cwd, "the release notes are written by hand");
    assert!(
        wait_until(Duration::from_secs(20), || {
            d.status(&cwd).index.pending_embeddings == 2
        }),
        "a chunk stored after the panic was embedded: {} pending",
        d.status(&cwd).index.pending_embeddings
    );
    assert_eq!(
        log().matches("skipped after failing alone").count(),
        2,
        "a skipped chunk is not retried by this daemon"
    );
    // Nothing was lost: nine memories, the two skipped still pending.
    assert_eq!(rig.memories().len(), 9);
}

/// The log of `d`, or nothing while it is not written.
fn log_of(d: &Daemon) -> String {
    std::fs::read_to_string(d.log_path()).unwrap_or_default()
}

/// Stream H 8: an embedder-wide `internal` error is not the chunks' fault.
/// While every batch and every chunk alone fails, no chunk is struck or
/// skipped, the wait between such rounds grows, and once the embedder
/// recovers every chunk is embedded by the same daemon.
#[test]
fn s5_an_embedder_wide_failure_strikes_no_chunk_and_every_chunk_embeds_after_it() {
    let rig = Rig::new();
    let provider = DownProvider::new(true);
    let d = rig.start_with_provider(provider.clone());
    let cwd = rig.project("alpha");
    let contents = [
        "the build cache lives in target",
        "the deploy script runs from the tools folder",
        "the test fixtures are data, not code",
        "the changelog is kept by hand",
        "the schema version lives in the meta table",
    ];
    match d.ok(
        &cwd,
        Op::Remember(RememberParams {
            memories: contents.iter().map(|c| RememberItem::new(*c)).collect(),
        }),
    ) {
        Reply::Remember(r) => assert_eq!(r.memories.len(), 5),
        other => panic!("remember replied {other:?}"),
    }
    let rounds = || {
        log_of(&d)
            .matches("no chunk of the 4 embedded alone; the failure is the embedder's")
            .count()
    };
    // Four rounds: one more than a chunk may be struck before it is skipped.
    assert!(
        wait_until(Duration::from_secs(20), || rounds() >= 4),
        "{}",
        log_of(&d)
    );
    // The wait doubles from `bulk_error_backoff` (50 ms here): after the
    // fourth round the next waits are 400 ms and 800 ms, so 600 ms hold at
    // most two more rounds (a fixed 50 ms wait would give about twelve).
    let after_four = rounds();
    std::thread::sleep(Duration::from_millis(600));
    assert!(
        rounds() <= after_four + 2,
        "the wait between rounds in which nothing is stored grows: {} rounds\n{}",
        rounds(),
        log_of(&d)
    );
    let text = log_of(&d);
    assert!(
        text.contains("internal: injected: the embedder is down"),
        "{text}"
    );
    assert!(
        !text.contains("failed alone ("),
        "no chunk is struck: {text}"
    );
    assert!(
        !text.contains("skipped after"),
        "no chunk is skipped: {text}"
    );
    assert_eq!(d.status(&cwd).index.pending_embeddings, 5);

    // The embedder recovers; a write wakes the queue at once.
    provider
        .down
        .store(false, std::sync::atomic::Ordering::SeqCst);
    remember(&d, &cwd, "the release notes are written by hand");
    assert!(
        wait_until(Duration::from_secs(20), || {
            d.status(&cwd).index.pending_embeddings == 0
        }),
        "every chunk is embedded once the embedder recovers: {} pending\n{}",
        d.status(&cwd).index.pending_embeddings,
        log_of(&d)
    );
    assert!(!log_of(&d).contains("skipped after"));
    assert_eq!(rig.memories().len(), 6);
}

/// Stream H 8: a batch of one chunk that fails has no other chunk to show
/// whether the fault is the chunk's or the embedder's, so it is never
/// struck, however often it fails; it is embedded once the embedder
/// recovers.
#[test]
fn s5_a_lone_chunk_that_fails_is_never_struck() {
    let rig = Rig::new();
    let provider = DownProvider::new(true);
    let d = rig.start_with_provider(provider.clone());
    let cwd = rig.project("alpha");
    remember(&d, &cwd, "the build cache lives in target");
    let rounds = || {
        log_of(&d)
            .matches("with no other chunk in its round; it is not struck")
            .count()
    };
    assert!(
        wait_until(Duration::from_secs(20), || rounds() >= 4),
        "{}",
        log_of(&d)
    );
    let text = log_of(&d);
    assert!(
        text.contains("internal: injected: the embedder is down"),
        "{text}"
    );
    assert!(!text.contains("failed alone ("), "{text}");
    assert!(!text.contains("skipped after"), "{text}");
    assert_eq!(d.status(&cwd).index.pending_embeddings, 1);

    // Recovery without a write: the queue tries again when its wait ends.
    provider
        .down
        .store(false, std::sync::atomic::Ordering::SeqCst);
    assert!(
        wait_until(Duration::from_secs(20), || {
            d.status(&cwd).index.pending_embeddings == 0
        }),
        "the lone chunk is embedded once the embedder recovers: {} pending\n{}",
        d.status(&cwd).index.pending_embeddings,
        log_of(&d)
    );
    assert!(!log_of(&d).contains("skipped after"));
}

#[test]
fn s8_project_rename_and_delete_and_their_errors() {
    let rig = Rig::new();
    let d = rig.start_fake();
    let cwd = rig.project("alpha");
    remember(&d, &cwd, "a decision about alpha");
    let Reply::Project(r) = d.ok(
        &cwd,
        Op::ProjectRename(ProjectRenameParams {
            from: "alpha".to_owned(),
            to: "omega".to_owned(),
        }),
    ) else {
        panic!("not a project reply")
    };
    assert_eq!(r.name, "omega");
    assert_eq!(r.renamed_from.as_deref(), Some("alpha"));
    assert_eq!(d.status(&cwd).current_project.as_deref(), Some("omega"));
    let missing = d.err(
        &cwd,
        Op::ProjectRename(ProjectRenameParams {
            from: "nothing-here".to_owned(),
            to: "x".to_owned(),
        }),
    );
    assert_code(&missing, ErrorCode::NotFound);

    let e = d.err(
        &cwd,
        Op::ProjectDelete(ProjectDeleteParams {
            name: "omega".to_owned(),
            confirm: false,
        }),
    );
    assert_code(&e, ErrorCode::ConfirmRequired);
    assert!(e.message.contains("1 memories"), "{}", e.message);
    assert_eq!(rig.memories().len(), 1);
    let Reply::Project(del) = d.ok(
        &cwd,
        Op::ProjectDelete(ProjectDeleteParams {
            name: "omega".to_owned(),
            confirm: true,
        }),
    ) else {
        panic!("not a project reply")
    };
    assert!(del.deleted);
    assert_eq!(del.memories, 1);
    assert!(rig.memories().is_empty());
    let e = d.err(
        &cwd,
        Op::ProjectDelete(ProjectDeleteParams {
            name: "omega".to_owned(),
            confirm: false,
        }),
    );
    assert_code(&e, ErrorCode::NotFound);
}

#[test]
fn s11_doctor_eval_runs_the_retrieval_set_and_its_error() {
    let rig = Rig::new();
    let d = rig.start_fake();
    let cwd = rig.project("alpha");
    let set = fixture_path("retrieval/retrieval_eval_set.json");
    let baseline = fixture_path("retrieval/retrieval_quality_baseline.json");
    let Reply::Eval(report) = d.ok(
        &cwd,
        Op::DoctorEval(DoctorEvalParams {
            set_path: set.to_string_lossy().into_owned(),
            baseline_path: Some(baseline.to_string_lossy().into_owned()),
        }),
    ) else {
        panic!("not an eval reply")
    };
    let expected = fixture("retrieval/retrieval_eval_set.json")["queries"]
        .as_array()
        .map(Vec::len)
        .expect("the set's queries");
    assert_eq!(report.queries, expected);
    assert_eq!(report.per_query.len(), expected);
    assert!(report.baseline.is_some());
    // The scratch store is gone and the live store untouched.
    assert!(rig.memories().is_empty());
    let eval_dir = rig.paths.dir.join("eval");
    assert!(
        !eval_dir.exists()
            || std::fs::read_dir(&eval_dir)
                .map(|r| r.count() == 0)
                .unwrap_or(true)
    );
    let missing = rig.tmp.path().join("no-such-set.json");
    let e = d.err(
        &cwd,
        Op::DoctorEval(DoctorEvalParams {
            set_path: missing.to_string_lossy().into_owned(),
            baseline_path: None,
        }),
    );
    assert!(
        matches!(
            e.code,
            ErrorCode::NotFound | ErrorCode::InvalidInput | ErrorCode::Storage
        ),
        "{}: {}",
        e.code.as_str(),
        e.message
    );
}

// --- spike decision 5: one query session for life (B-95) -------------------

/// What one session a [`TallyProvider`] opened did.
#[derive(Debug, Clone)]
struct Tally {
    kind: sm_core::traits::SessionKind,
    queries: usize,
    documents: usize,
    dropped: bool,
}

/// The fake provider, numbering each session it opens in order and
/// recording, per session, its kind, what it embedded and whether it was
/// dropped.
struct TallyProvider {
    inner: sm_core::testing::FakeProvider,
    sessions: Arc<std::sync::Mutex<Vec<Tally>>>,
}

struct TallyEmbedder {
    inner: Box<dyn sm_core::traits::Embedder>,
    n: usize,
    sessions: Arc<std::sync::Mutex<Vec<Tally>>>,
}

impl TallyProvider {
    fn sessions(&self) -> Vec<Tally> {
        self.sessions.lock().expect("the tally").clone()
    }
}

impl sm_core::traits::EmbedderProvider for TallyProvider {
    fn info(&self) -> &sm_core::traits::EmbedderInfo {
        self.inner.info()
    }
    fn open(
        &self,
        kind: sm_core::traits::SessionKind,
        threads: usize,
    ) -> sm_core::error::Result<Box<dyn sm_core::traits::Embedder>> {
        let inner = self.inner.open(kind, threads)?;
        let mut s = self.sessions.lock().expect("the tally");
        s.push(Tally {
            kind,
            queries: 0,
            documents: 0,
            dropped: false,
        });
        Ok(Box::new(TallyEmbedder {
            inner,
            n: s.len() - 1,
            sessions: Arc::clone(&self.sessions),
        }))
    }
}

impl Drop for TallyEmbedder {
    fn drop(&mut self) {
        if let Ok(mut s) = self.sessions.lock() {
            s[self.n].dropped = true;
        }
    }
}

impl sm_core::traits::Embedder for TallyEmbedder {
    fn info(&self) -> &sm_core::traits::EmbedderInfo {
        self.inner.info()
    }
    fn embed_query(&self, text: &str, max_tokens: usize) -> sm_core::error::Result<Vec<f32>> {
        self.sessions.lock().expect("the tally")[self.n].queries += 1;
        self.inner.embed_query(text, max_tokens)
    }
    fn embed_documents(
        &self,
        docs: &[sm_core::traits::DocInput<'_>],
    ) -> sm_core::error::Result<Vec<Vec<f32>>> {
        self.sessions.lock().expect("the tally")[self.n].documents += docs.len();
        self.inner.embed_documents(docs)
    }
}

/// Spike decision 5 (B-95): the daemon opens the query session first,
/// before it is ready, and keeps that one session for its life: the same
/// session serves requests before a bulk import, during and after it, and
/// after an idle period long enough for the bulk session to be dropped but
/// short of idle exit. No other query session is ever opened, and the
/// query session is never dropped while the daemon runs.
#[test]
fn spike5_the_query_session_is_opened_first_and_kept_for_life() {
    let mut rig = Rig::new();
    rig.settings.daemon.idle_exit_minutes = 1;
    rig.opts.idle_unit = Duration::from_secs(5);
    let provider = Arc::new(TallyProvider {
        inner: sm_core::testing::FakeProvider::new(),
        sessions: Arc::new(std::sync::Mutex::new(Vec::new())),
    });
    let d = rig.start_with_provider(provider.clone());
    let query_kind = sm_core::traits::SessionKind::Query;
    let first = provider.sessions();
    assert_eq!(first.len(), 1, "only the query session at ready: {first:?}");
    assert_eq!(first[0].kind, query_kind);
    assert!(!first[0].dropped);

    let cwd = rig.project("alpha");
    let ask = || {
        let mut p = RecallParams::new("the release checklist");
        p.mode = Some(RecallMode::Vector);
        recall(&d, &cwd, p)
    };
    let query_calls = || provider.sessions()[0].queries;
    remember(&d, &cwd, "the release checklist lives in docs");
    d.wait_embedded(&cwd);
    ask();
    ask();
    let before_import = query_calls();
    assert!(before_import >= 2, "{:?}", provider.sessions());

    // A bulk import of 300 memories (remember takes 100 at a time): the
    // bulk session embeds it while recall goes on.
    for b in 0..3 {
        let items: Vec<RememberItem> = (0..100)
            .map(|i| RememberItem::new(format!("bulk import record {b} {i} for the archive")))
            .collect();
        match d.ok(&cwd, Op::Remember(RememberParams { memories: items })) {
            Reply::Remember(r) => assert_eq!(r.memories.len(), 100),
            other => panic!("remember replied {other:?}"),
        }
        ask();
    }
    d.wait_embedded(&cwd);
    ask();
    let after_import = query_calls();
    assert!(after_import >= before_import + 2);

    // Idle past `bulk_grace` (200 ms here), short of idle exit (5 s): the
    // bulk session is dropped, the query session is not.
    assert!(
        wait_until(Duration::from_secs(5), || {
            provider
                .sessions()
                .iter()
                .any(|s| s.kind != query_kind && s.dropped)
        }),
        "the idle bulk session is dropped: {:?}",
        provider.sessions()
    );
    std::thread::sleep(Duration::from_millis(500));
    assert!(!d.handle().stopping(), "short of idle exit");
    ask();
    assert!(query_calls() > after_import);

    let all = provider.sessions();
    assert_eq!(all[0].kind, query_kind, "the query session came first");
    assert_eq!(
        all.iter().filter(|s| s.kind == query_kind).count(),
        1,
        "one query session for life: {all:?}"
    );
    assert!(!all[0].dropped, "the query session is never dropped");
    assert_eq!(all[0].documents, 0, "the query session embeds no chunks");
    let bulk: Vec<&Tally> = all.iter().filter(|s| s.kind != query_kind).collect();
    assert!(!bulk.is_empty(), "the import ran on a bulk session");
    assert!(
        bulk.iter().all(|s| s.queries == 0),
        "recall never ran on a bulk session: {all:?}"
    );
    assert!(bulk.iter().map(|s| s.documents).sum::<usize>() >= 301);
}

// --- SPEC 5 rule 7: doctor's numbers through the daemon (B-125) ------------

fn doctor_numbers(d: &Daemon, cwd: &str) -> DoctorNumbersReply {
    match d.ok(cwd, Op::DoctorNumbers) {
        Reply::DoctorNumbers(n) => n,
        other => panic!("doctor_numbers replied {other:?}"),
    }
}

/// SPEC 5 rule 7 (B-125): the reply's counts are the configured budget for
/// the daemon's core count (8 in the test options), the batch size and the
/// fixed sample of `sm-embed`'s doctor, and both timings are measured.
#[test]
fn s5_rule7_doctor_numbers_counts_match_the_settings() {
    for (budget, batch) in [(None, None), (Some(0.5), Some(8))] {
        let mut rig = Rig::new();
        if let Some(b) = budget {
            rig.settings.cpu.budget = b;
        }
        if let Some(b) = batch {
            rig.settings.embed.batch_size = b;
        }
        let d = rig.start_fake();
        let cwd = rig.project("alpha");
        let n = doctor_numbers(&d, &cwd);
        let cores = rig.opts.physical_cores.expect("the test core count");
        assert_eq!(n.physical_cores as usize, cores);
        assert_eq!(n.query_threads as usize, rig.settings.query_threads(cores));
        assert_eq!(n.bulk_threads as usize, rig.settings.bulk_threads(cores));
        assert_eq!(n.batch_size, rig.settings.embed.batch_size);
        assert_eq!(n.sample_chunks as usize, sm_embed::doctor::SAMPLE_CHUNKS);
        assert_eq!(n.query_tokens as usize, sm_embed::doctor::QUERY_TOKENS);
        assert_eq!(n.query_runs as usize, sm_embed::doctor::QUERY_RUNS);
        assert!(n.bulk_chunks_per_sec > 0.0, "{n:?}");
        assert!(
            n.bulk_elapsed_ms >= 0.0 && n.query_median_ms >= 0.0,
            "{n:?}"
        );
        if budget.is_some() {
            assert_eq!((n.query_threads, n.bulk_threads), (4, 4), "{n:?}");
        }
    }
}

/// SPEC 5 rule 7 and spike decision 5 (B-125): the timing runs on the
/// daemon's own sessions. The bulk session it opens is the one the queue
/// then embeds on, and a later timing uses it again; the query timing runs
/// on the one query session. No third session is opened, so AT-4's peak is
/// not raised.
#[test]
fn s5_rule7_doctor_numbers_uses_the_queues_bulk_session_and_opens_no_third() {
    let mut rig = Rig::new();
    // Long enough that the idle bulk session is not dropped mid-test.
    rig.opts.bulk_grace = Duration::from_secs(60);
    let provider = Arc::new(TallyProvider {
        inner: sm_core::testing::FakeProvider::new(),
        sessions: Arc::new(std::sync::Mutex::new(Vec::new())),
    });
    let d = rig.start_with_provider(provider.clone());
    let query_kind = sm_core::traits::SessionKind::Query;
    assert_eq!(provider.sessions().len(), 1, "the query session at ready");
    let cwd = rig.project("alpha");

    let first = doctor_numbers(&d, &cwd);
    let after_first = provider.sessions();
    assert_eq!(
        after_first.len(),
        2,
        "the timing opened the bulk session: {after_first:?}"
    );
    assert_eq!(after_first[1].kind, sm_core::traits::SessionKind::Bulk);
    let warm = (rig.settings.embed.batch_size as usize).min(first.sample_chunks as usize);
    let timed = warm + first.sample_chunks as usize;
    assert_eq!(after_first[1].documents, timed, "{after_first:?}");
    assert_eq!(after_first[1].queries, 0);
    let runs = 1 + first.query_runs as usize;
    assert_eq!(after_first[0].queries, runs, "{after_first:?}");
    assert_eq!(after_first[0].documents, 0);

    // The queue embeds on the session the timing opened.
    remember(&d, &cwd, "the release checklist lives in docs/release.md");
    d.wait_embedded(&cwd);
    let after_remember = provider.sessions();
    assert_eq!(after_remember.len(), 2, "{after_remember:?}");
    assert!(after_remember[1].documents > timed, "{after_remember:?}");
    let embedded = after_remember[1].documents;

    // A second timing reuses both.
    doctor_numbers(&d, &cwd);
    let all = provider.sessions();
    assert_eq!(all.len(), 2, "no third session: {all:?}");
    assert_eq!(all.iter().filter(|s| s.kind == query_kind).count(), 1);
    assert!(all.iter().all(|s| !s.dropped), "{all:?}");
    assert_eq!(all[1].documents, embedded + timed);
    assert_eq!(all[0].queries, 2 * runs);
    assert_eq!(all[0].documents, 0, "the query session embeds no chunks");
    assert_eq!(all[1].queries, 0, "no query runs on the bulk session");
}

/// SPEC 5 rule 7 (B-125): while the model loads, `doctor_numbers` is
/// `daemon_loading`; once it is ready, the numbers come back.
#[test]
fn s5_rule7_doctor_numbers_is_daemon_loading_while_the_model_loads() {
    let rig = Rig::new();
    let gate = GateLoader::new();
    let d = rig.start(gate.clone(), Arc::new(UnwiredSharing));
    let cwd = rig.project("alpha");
    assert_eq!(d.hello().state, DaemonState::Loading);
    let e = d.err(&cwd, Op::DoctorNumbers);
    assert_code(&e, ErrorCode::DaemonLoading);
    gate.open();
    d.wait_ready();
    let n = doctor_numbers(&d, &cwd);
    assert_eq!(n.query_tokens as usize, sm_embed::doctor::QUERY_TOKENS);
}

/// SPEC 5 rule 7 (B-125): after a failed load, `doctor_numbers` returns the
/// model's own error, the same one `hello` reports.
#[test]
fn s5_rule7_doctor_numbers_gives_the_models_error_after_a_failed_load() {
    let rig = Rig::new();
    let d = rig.start_loader(failing_loader());
    assert!(wait_until(Duration::from_secs(10), || d.hello().state
        == DaemonState::Failed));
    let cwd = rig.project("alpha");
    let e = d.err(&cwd, Op::DoctorNumbers);
    let model = d.hello().state_error.expect("the model's error in hello");
    assert_eq!(e.code, model.code);
    assert_code(&e, ErrorCode::Internal);
    assert_eq!(e.message, model.message);
    assert!(e.message.contains("the model cannot load"), "{}", e.message);
}

// --- stream H 15: a title-only revise re-embeds --------------------------------

fn revise_reply(d: &Daemon, cwd: &str, p: ReviseParams) -> ReviseReply {
    match d.ok(cwd, Op::Revise(p)) {
        Reply::Revise(r) => r,
        other => panic!("revise replied {other:?}"),
    }
}

fn titled(content: &str, title: &str) -> RememberItem {
    let mut item = RememberItem::new(content);
    item.title = Some(title.to_owned());
    item
}

fn doc(title: &str, text: &str) -> EmbeddedDoc {
    (Some(title.to_owned()), text.to_owned())
}

/// Stream H 15 (SPEC 5 "Bulk", SPEC 8 `revise`): the bulk queue embeds a
/// chunk with its memory's title, so a revise that changes only the title
/// rebuilds the chunks and they are embedded again with the new title; it
/// says `reembedded`. A revise to the title the memory has, or of the tags
/// alone, embeds nothing again.
#[test]
fn s8_a_title_only_revise_re_embeds_the_chunks_with_the_new_title() {
    let rig = Rig::new();
    let provider = DocsProvider::new();
    let d = rig.start_with_provider(provider.clone());
    let cwd = rig.project("titled");
    let text = "the cache holds 500 entries";
    let id = remember_item(&d, &cwd, titled(text, "cache size"));
    d.wait_embedded(&cwd);
    assert_eq!(provider.docs(), [doc("cache size", text)]);
    let chunks = d.status(&cwd).index.chunks;

    let mut p = ReviseParams::new(id);
    p.title = Some("the cache limit".to_owned());
    let r = revise_reply(&d, &cwd, p);
    assert!(r.reembedded, "a new title re-embeds");
    assert_eq!(r.memory.title.as_deref(), Some("the cache limit"));
    assert_eq!(r.memory.content, text, "the content is kept");
    d.wait_embedded(&cwd);
    assert_eq!(
        provider.docs(),
        [doc("cache size", text), doc("the cache limit", text)],
        "the chunk was embedded again, with the new title"
    );
    assert_eq!(d.status(&cwd).index.chunks, chunks, "the same chunks");
    let stored = rig.store().get_memory(&id).expect("the memory");
    assert_eq!(stored.title.as_deref(), Some("the cache limit"));
    assert_eq!(stored.content, text);

    let mut same = ReviseParams::new(id);
    same.title = Some("the cache limit".to_owned());
    assert!(!revise_reply(&d, &cwd, same).reembedded, "the same title");
    let mut tags = ReviseParams::new(id);
    tags.tags = Some(vec!["cache".to_owned()]);
    assert!(!revise_reply(&d, &cwd, tags).reembedded, "tags alone");
    // A later memory is embedded after anything those two queued.
    let later = "a later note about the queue";
    remember_item(&d, &cwd, titled(later, "later"));
    d.wait_embedded(&cwd);
    assert_eq!(
        provider.docs(),
        [
            doc("cache size", text),
            doc("the cache limit", text),
            doc("later", later)
        ],
        "neither the same title nor the tags embedded anything again"
    );
}

/// Stream H 15 and B-96: a new title needs the model's tokenizer to
/// rebuild the chunks, so while the model loads it is refused with
/// `daemon_loading` and the memory is unchanged; a revise to the same
/// title, or of other fields, is served. Once the model is ready the new
/// title is taken and embedded.
#[test]
fn s8_a_new_title_waits_for_the_model_and_other_fields_do_not() {
    let rig = Rig::new();
    let cwd = rig.project("loading");
    let text = "the invoice archive is kept for seven years";
    let earlier = rig.start_fake();
    let id = remember_item(&earlier, &cwd, titled(text, "invoice archive"));
    earlier.wait_embedded(&cwd);
    earlier.shutdown();

    let gate = GateLoader::new();
    let d = rig.start_loader(gate.clone());
    assert_eq!(d.hello().state, DaemonState::Loading);
    let mut p = ReviseParams::new(id);
    p.title = Some("invoice retention".to_owned());
    p.tags = Some(vec!["finance".to_owned()]);
    assert_code(
        &d.err(&cwd, Op::Revise(p.clone())),
        ErrorCode::DaemonLoading,
    );
    let unchanged = rig.store().get_memory(&id).expect("the memory");
    assert_eq!(unchanged.title.as_deref(), Some("invoice archive"));
    assert!(unchanged.tags.is_empty(), "nothing of the refused call");

    let mut same = ReviseParams::new(id);
    same.title = Some("invoice archive".to_owned());
    same.tags = Some(vec!["finance".to_owned()]);
    let r = revise_reply(&d, &cwd, same);
    assert!(!r.reembedded);
    assert_eq!(r.memory.tags, ["finance"]);
    assert_eq!(d.status(&cwd).index.pending_embeddings, 0);

    gate.open();
    d.wait_ready();
    let r = revise_reply(&d, &cwd, p);
    assert!(r.reembedded);
    assert_eq!(r.memory.title.as_deref(), Some("invoice retention"));
    d.wait_embedded(&cwd);
}
