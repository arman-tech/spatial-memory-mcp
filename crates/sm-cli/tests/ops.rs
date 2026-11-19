//! The command-line operations through the binary (SPEC 3, command table;
//! SPEC 8, errors): each sends one op through the waiting client to the
//! mock daemon with paths made absolute, prints the report, and prints a
//! daemon error with its code and next action. `status` exits 1 when the
//! daemon reports an error in its reply (SPEC 10, step 3).

mod common;

use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard};

use common::{MockDaemon, Mode, Sandbox};
use sm_core::error::{CoreError, ErrorCode};
use sm_core::protocol::{
    DaemonInfo, DaemonState, ExportReport, IndexStatus, ModelStatus, Op, ProjectReply,
    ProjectStatus, Reply, StatusReply,
};
use sm_core::time::Timestamp;

static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn same_folder(a: &Path, b: &Path) -> bool {
    std::fs::canonicalize(a).unwrap() == std::fs::canonicalize(b).unwrap()
}

#[test]
fn s3_export_sends_an_absolute_path_and_prints_the_report() {
    let _g = serial();
    let sb = Sandbox::new();
    let mock = MockDaemon::start(&sb.paths(), Mode::Answer);
    mock.set_responder(Arc::new(|request| match &request.call {
        Op::Export(p) => Some(Ok(Reply::Export(ExportReport {
            path: p.path.clone(),
            projects: 1,
            memories: 2,
            links: 0,
        }))),
        _ => None,
    }));
    let run = sb.run(&["export", "out.json", "--project", "app"], &[], b"");
    assert_eq!(run.code, Some(0), "{run:?}");
    let requests = mock.requests();
    let ops: Vec<&str> = requests.iter().map(|r| r.call.name()).collect();
    assert_eq!(ops, ["hello", "export"]);
    let Op::Export(p) = &requests[1].call else {
        unreachable!()
    };
    let path = Path::new(&p.path);
    assert!(path.is_absolute(), "{}", p.path);
    assert_eq!(path.file_name().unwrap(), "out.json");
    assert!(same_folder(path.parent().unwrap(), &sb.tmp), "{}", p.path);
    assert_eq!(p.project.as_deref(), Some("app"));
    let expected = sm_cli::ops::render(&Reply::Export(ExportReport {
        path: p.path.clone(),
        projects: 1,
        memories: 2,
        links: 0,
    }));
    assert_eq!(run.stdout, expected);
}

#[test]
fn s3_project_delete_without_confirm_prints_the_preview() {
    let _g = serial();
    let sb = Sandbox::new();
    let mock = MockDaemon::start(&sb.paths(), Mode::Answer);
    mock.set_responder(Arc::new(|request| match &request.call {
        Op::ProjectDelete(p) if !p.confirm => Some(Err(CoreError::new(
            ErrorCode::ConfirmRequired,
            "project app holds 3 memories",
        ))),
        Op::ProjectDelete(p) => Some(Ok(Reply::Project(ProjectReply {
            name: p.name.clone(),
            renamed_from: None,
            deleted: true,
            memories: 3,
        }))),
        _ => None,
    }));
    let preview = sb.run(&["project", "delete", "app"], &[], b"");
    assert_eq!(preview.code, Some(0), "{preview:?}");
    assert_eq!(
        preview.stdout,
        "Nothing was deleted. project app holds 3 memories\nRun again with --confirm to delete.\n"
    );
    let done = sb.run(&["project", "delete", "app", "--confirm"], &[], b"");
    assert_eq!(done.code, Some(0), "{done:?}");
    let ops: Vec<&str> = mock.requests().iter().map(|r| r.call.name()).collect();
    assert_eq!(ops, ["hello", "project_delete", "hello", "project_delete"]);
}

#[test]
fn s8_a_daemon_error_prints_its_code_message_and_next_action_and_exits_1() {
    let _g = serial();
    let sb = Sandbox::new();
    let mock = MockDaemon::start(&sb.paths(), Mode::Answer);
    let refusal = CoreError::new(ErrorCode::Storage, "the database is read-only")
        .with_next_action("Check the permissions of the profile folder.");
    let answer = refusal.clone();
    mock.set_responder(Arc::new(move |request| match &request.call {
        Op::Reindex(_) => Some(Err(answer.clone())),
        _ => None,
    }));
    let run = sb.run(&["reindex"], &[], b"");
    assert_eq!(run.code, Some(1), "{run:?}");
    assert_eq!(run.stdout, "");
    assert_eq!(run.stderr, sm_cli::cli::error_text(&refusal));
    assert!(run.stderr.contains("storage"), "{}", run.stderr);
    assert!(
        run.stderr
            .contains("Check the permissions of the profile folder."),
        "{}",
        run.stderr
    );
}

#[test]
fn s8_an_invalid_op_is_refused_before_the_daemon() {
    let _g = serial();
    let sb = Sandbox::new();
    let mock = MockDaemon::start(&sb.paths(), Mode::Answer);
    // An empty project name fails the op's own validation.
    let run = sb.run(&["project", "rename", "", "new"], &[], b"");
    assert_eq!(run.code, Some(1), "{run:?}");
    assert!(
        run.stderr.starts_with("error: invalid_input:"),
        "{}",
        run.stderr
    );
    assert!(run.stderr.contains("\nnext: "), "{}", run.stderr);
    assert_eq!(mock.connections(), 0);
}

// ---------------------------------------------------------------------------
// `spatial-memory status` (SPEC 10, step 3: "Verify with `spatial-memory
// status`")
// ---------------------------------------------------------------------------

/// A judge that can run, with nothing pending.
fn judge_ok() -> sm_core::protocol::JudgeStatus {
    sm_core::protocol::JudgeStatus {
        enabled: true,
        can_run: true,
        blocked: None,
        retry_at: None,
        model: "haiku".into(),
        command: Some("claude".into()),
        pending: 0,
        queued: 0,
        rejected: 0,
        calls_today: 0,
        max_calls_per_day: 200,
        judged_today: 0,
        cost_today_usd: 0.0,
        max_usd_per_day: Some(3.0),
        last_error: None,
        quality: None,
    }
}

fn status_reply(state: DaemonState, state_error: Option<CoreError>) -> StatusReply {
    let ready = state == DaemonState::Ready;
    StatusReply {
        daemon: DaemonInfo {
            protocol: sm_core::protocol::PROTOCOL_VERSION,
            build: sm_core::BUILD.to_owned(),
            pid: 4242,
            profile: "default".into(),
            state,
            state_error,
            started_at: Timestamp::now(),
            ready_seconds: ready.then_some(1.5),
            model_id: "embeddinggemma-2".into(),
            endpoint: "mock".into(),
        },
        model: ModelStatus {
            model_id: "embeddinggemma-2".into(),
            revision: sm_embed::model::REVISION.into(),
            dim: 768,
            loaded: ready,
        },
        index: IndexStatus {
            matches: true,
            differences: Vec::new(),
            chunks: 9,
            pending_embeddings: 0,
        },
        current_project: Some("app".into()),
        projects: vec![ProjectStatus {
            name: "app".into(),
            git_remote: None,
            memories: 4,
            by_type: [("decision".to_owned(), 4)].into(),
            by_status: [("active".to_owned(), 4)].into(),
            shared: 0,
        }],
        database: "memory.db".into(),
        database_bytes: 4096,
        spool_pending: 0,
        captures_queued: 0,
        judge: judge_ok(),
        metrics: None,
        omitted: 0,
    }
}

fn answer_status(mock: &MockDaemon, reply: StatusReply) {
    mock.set_responder(Arc::new(move |request| match &request.call {
        Op::Status(_) => Some(Ok(Reply::Status(Box::new(reply.clone())))),
        _ => None,
    }));
}

#[test]
fn s10_status_prints_the_daemon_model_index_and_projects_and_exits_0() {
    let _g = serial();
    let sb = Sandbox::new();
    let mock = MockDaemon::start(&sb.paths(), Mode::Answer);
    let reply = status_reply(DaemonState::Ready, None);
    answer_status(&mock, reply.clone());
    let run = sb.run(&["status"], &[], b"");
    assert_eq!(run.code, Some(0), "{run:?}");
    assert_eq!(run.stderr, "");
    assert_eq!(
        run.stdout,
        sm_cli::ops::render(&Reply::Status(Box::new(reply)))
    );
    for line in [
        "Daemon: ready (pid 4242, profile default, ",
        "Model: embeddinggemma-2 at revision ",
        "Index: matches the configuration; 9 chunks, 0 waiting for embedding",
        "Projects: 1 with 4 memories",
        "  app: 4 memories, 0 shared",
    ] {
        assert!(run.stdout.contains(line), "{line}: {}", run.stdout);
    }
    let requests = mock.requests();
    let ops: Vec<&str> = requests.iter().map(|r| r.call.name()).collect();
    assert_eq!(ops, ["hello", "status"]);
    let Op::Status(p) = &requests[1].call else {
        unreachable!()
    };
    assert!(!p.metrics);
}

#[test]
fn s10_status_of_a_daemon_whose_model_failed_exits_1_with_error_and_next() {
    let _g = serial();
    let sb = Sandbox::new();
    let mock = MockDaemon::start(&sb.paths(), Mode::Answer);
    let failure = CoreError::new(
        ErrorCode::ModelMissing,
        "model file onnx/model.onnx is missing",
    )
    .with_next_action("Run `spatial-memory setup` to download the model.");
    answer_status(
        &mock,
        status_reply(DaemonState::Failed, Some(failure.clone())),
    );
    let run = sb.run(&["status"], &[], b"");
    assert_eq!(run.code, Some(1), "{run:?}");
    assert_eq!(run.stderr, sm_cli::cli::error_text(&failure));
    assert!(
        run.stderr.starts_with("error: model_missing: "),
        "{}",
        run.stderr
    );
    assert!(
        run.stderr.contains("\nnext: Run `spatial-memory setup`"),
        "{}",
        run.stderr
    );
    // The report is still printed: the rest of the state is worth seeing.
    assert!(run.stdout.starts_with("Daemon: failed "), "{}", run.stdout);
    assert!(run.stdout.contains(", not loaded\n"), "{}", run.stdout);
}

#[test]
fn s10_status_with_a_daemon_error_reply_exits_1() {
    let _g = serial();
    let sb = Sandbox::new();
    let mock = MockDaemon::start(&sb.paths(), Mode::Answer);
    let refusal = CoreError::new(ErrorCode::Storage, "the database is locked")
        .with_next_action("Close the other program using memory.db.");
    let answer = refusal.clone();
    mock.set_responder(Arc::new(move |request| match &request.call {
        Op::Status(_) => Some(Err(answer.clone())),
        _ => None,
    }));
    let run = sb.run(&["status"], &[], b"");
    assert_eq!(run.code, Some(1), "{run:?}");
    assert_eq!(run.stdout, "");
    assert_eq!(run.stderr, sm_cli::cli::error_text(&refusal));
}
