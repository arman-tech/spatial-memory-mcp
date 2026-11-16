//! The command-line operations through the binary (SPEC 3, command table;
//! SPEC 8, errors): each sends one op through the waiting client to the
//! mock daemon with paths made absolute, prints the report, and prints a
//! daemon error with its code and next action.

mod common;

use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard};

use common::{MockDaemon, Mode, Sandbox};
use sm_core::error::{CoreError, ErrorCode};
use sm_core::protocol::{ExportReport, Op, ProjectReply, Reply};

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
