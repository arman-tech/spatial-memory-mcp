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
use sm_core::types::{LinkEnd, MemoryType, RecallMode, Relation, Scope};
use sm_daemon::{DaemonLock, Exit, Started, UnwiredSharing};

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

#[test]
fn s9_sharing_ops_reach_the_implementation_with_the_resolved_project() {
    let rig = Rig::new();
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
    let Reply::Import(i) = d.ok(
        &repo,
        Op::Import(ImportParams {
            path: "in.jsonl".to_owned(),
        }),
    ) else {
        panic!("not an import reply")
    };
    assert_eq!(i.inserted, 1);
    let Reply::Import(m) = d.ok(
        &repo,
        Op::Migrate(MigrateParams {
            from: "v1-data".to_owned(),
            default_project: None,
        }),
    ) else {
        panic!("a migrate reply is an import report")
    };
    assert_eq!(m.source, "v1-data");
    let Reply::Export(x) = d.ok(
        &repo,
        Op::Export(ExportParams {
            path: "out.jsonl".to_owned(),
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
    let import = Op::Import(ImportParams {
        path: "in.jsonl".to_owned(),
    });
    let export = Op::Export(ExportParams {
        path: "out.jsonl".to_owned(),
        project: None,
    });
    let migrate = Op::Migrate(MigrateParams {
        from: "v1".to_owned(),
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
