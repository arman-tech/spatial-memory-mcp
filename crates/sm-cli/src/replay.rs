//! `spatial-memory replay` (SPEC 16, Replay; B-172): runs the stored
//! transcripts of Claude Code sessions through the current capture
//! pipeline, on request only.
//!
//! Claude Code keeps a session's transcript at
//! `<config>/projects/<folder>/<session id>.jsonl`. `<config>` is
//! `CLAUDE_CONFIG_DIR` when set, otherwise `~/.claude`; `<folder>` is the
//! session's working folder with every character other than an ASCII letter
//! or digit replaced by `-`. The session's working folder, which names its
//! project, is the `cwd` the transcript records.
//!
//! The daemon queues each replay and replies at once. The candidates are
//! stored pending and the judge settles them, as for any capture.

use std::fmt::Write as _;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use serde_json::Value;
use sm_core::error::CoreError;
use sm_core::protocol::{Op, ReplayParams, Reply};
use sm_core::time::Timestamp;

use crate::ops::OpOutcome;

/// Lines read at most when looking for the working folder a transcript
/// records. Claude Code writes it on the first prompt.
const CWD_SCAN_LINES: usize = 1000;

/// One transcript to replay.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Transcript {
    /// Absolute path of the `.jsonl` file.
    pub path: PathBuf,
    /// The file's name without `.jsonl`.
    pub session_id: String,
    /// The working folder the transcript records.
    pub cwd: String,
}

/// What [`resolve`] found for one argument, or one file of `--all`.
#[derive(Debug)]
pub enum Item {
    Ready(Transcript),
    /// Not replayed: `name` is the argument or the file, with the reason.
    Skipped {
        name: String,
        error: CoreError,
    },
}

/// Claude Code's projects folder: `CLAUDE_CONFIG_DIR/projects`, or
/// `~/.claude/projects`.
pub fn projects_dir(home: &Path, env: &dyn Fn(&str) -> Option<String>) -> PathBuf {
    env("CLAUDE_CONFIG_DIR")
        .filter(|d| !d.trim().is_empty())
        .map_or_else(|| home.join(".claude"), PathBuf::from)
        .join("projects")
}

/// The name of the folder under `projects` where Claude Code keeps the
/// transcripts of sessions run in `cwd`.
pub fn folder_name(cwd: &Path) -> String {
    cwd.to_string_lossy()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect()
}

/// Resolves the command's arguments. Each argument is a transcript file, or
/// a session id of the current folder (`projects/<folder>/<id>.jsonl`).
/// With `all`, every `.jsonl` file directly in the current folder's
/// transcript folder, in name order. An argument or file that cannot be
/// replayed is an [`Item::Skipped`]; the error is only for a transcript
/// folder that `all` cannot list.
pub fn resolve(
    args: &[String],
    all: bool,
    cwd: &Path,
    projects: &Path,
) -> Result<Vec<Item>, CoreError> {
    let folder = projects.join(folder_name(cwd));
    let mut files: Vec<(String, Option<PathBuf>)> = Vec::new();
    if all {
        let listed = std::fs::read_dir(&folder).map_err(|e| {
            CoreError::invalid_input(format!(
                "cannot list the transcripts of this folder in {}: {e}",
                folder.display()
            ))
            .with_next_action(
                "Run replay from the folder the sessions ran in, or name the transcript files.",
            )
        })?;
        let mut paths: Vec<PathBuf> = listed
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.is_file() && p.extension().is_some_and(|x| x == "jsonl"))
            .collect();
        paths.sort();
        files.extend(
            paths
                .into_iter()
                .map(|p| (p.display().to_string(), Some(p))),
        );
    }
    for arg in args {
        let named = Path::new(arg);
        let path = if is_file_argument(arg) {
            crate::profile::absolute(named, cwd)
        } else {
            folder.join(format!("{arg}.jsonl"))
        };
        files.push((arg.clone(), path.is_file().then_some(path)));
    }
    Ok(files
        .into_iter()
        .map(|(name, path)| {
            match path {
            None => Item::Skipped {
                error: CoreError::invalid_input(format!("no transcript `{name}`")).with_next_action(
                    "Name a transcript file, or a session id of the folder you run replay from.",
                ),
                name,
            },
            Some(path) => match transcript(&path) {
                Ok(t) => Item::Ready(t),
                Err(error) => Item::Skipped { name, error },
            },
        }
        })
        .collect())
}

/// An argument names a file when it has a folder separator or ends in
/// `.jsonl`; otherwise it is a session id.
fn is_file_argument(arg: &str) -> bool {
    arg.contains('/') || arg.contains('\\') || arg.ends_with(".jsonl")
}

/// The transcript at `path`, with the session id from its name and the
/// working folder it records.
fn transcript(path: &Path) -> Result<Transcript, CoreError> {
    let session_id = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| CoreError::invalid_input("the transcript file has no name"))?;
    let cwd = recorded_cwd(path)?.ok_or_else(|| {
        CoreError::invalid_input(format!(
            "{} records no working folder, so its project is unknown",
            path.display()
        ))
        .with_next_action("Replay a Claude Code session transcript.")
    })?;
    Ok(Transcript {
        path: path.to_path_buf(),
        session_id,
        cwd,
    })
}

/// The first absolute `cwd` among the transcript's first
/// [`CWD_SCAN_LINES`] lines; lines that are not JSON objects are skipped.
fn recorded_cwd(path: &Path) -> Result<Option<String>, CoreError> {
    let file = std::fs::File::open(path)
        .map_err(|e| CoreError::invalid_input(format!("cannot read {}: {e}", path.display())))?;
    for line in BufReader::new(file).lines().take(CWD_SCAN_LINES) {
        let Ok(line) = line else {
            continue;
        };
        let Ok(entry) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        if let Some(cwd) = entry.get("cwd").and_then(Value::as_str)
            && Path::new(cwd).is_absolute()
        {
            return Ok(Some(cwd.to_owned()));
        }
    }
    Ok(None)
}

/// Sends one replay per ready item through `call` and reports each, one
/// line apiece. The exit code is 1 when any item was skipped, refused or
/// failed.
pub fn run(items: Vec<Item>, mut call: impl FnMut(Op) -> Result<Reply, CoreError>) -> OpOutcome {
    let mut out = OpOutcome {
        stdout: String::new(),
        stderr: String::new(),
        code: 0,
    };
    let mut queued = 0usize;
    for item in items {
        let t = match item {
            Item::Ready(t) => t,
            Item::Skipped { name, error } => {
                let _ = writeln!(out.stdout, "skipped {name}");
                out.stderr.push_str(&crate::cli::error_text(&error));
                out.code = 1;
                continue;
            }
        };
        let op = Op::Replay(ReplayParams {
            transcript_path: t.path.to_string_lossy().into_owned(),
            session_id: t.session_id.clone(),
            cwd: t.cwd.clone(),
            received_at: Timestamp::now(),
        });
        match call(op) {
            Ok(Reply::Captured(c)) if c.accepted => {
                queued += 1;
                let _ = writeln!(out.stdout, "queued {} ({})", t.session_id, t.cwd);
            }
            Ok(Reply::Captured(_)) => {
                let _ = writeln!(
                    out.stdout,
                    "not queued {}: automatic capture is off",
                    t.session_id
                );
                out.code = 1;
            }
            Ok(other) => {
                let e = CoreError::internal(format!(
                    "the daemon answered a replay with {}",
                    serde_json::to_string(&other).unwrap_or_default()
                ));
                let _ = writeln!(out.stdout, "failed {}", t.session_id);
                out.stderr.push_str(&crate::cli::error_text(&e));
                out.code = 1;
            }
            Err(e) => {
                let _ = writeln!(out.stdout, "failed {}", t.session_id);
                out.stderr.push_str(&crate::cli::error_text(&e));
                out.code = 1;
            }
        }
    }
    if queued > 0 {
        let _ = writeln!(
            out.stdout,
            "{queued} queued. Their candidates are stored pending and the judge settles them; \
             `spatial-memory status` shows the capture queue."
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use sm_core::protocol::CaptureReply;

    fn write(path: &Path, lines: &[Value]) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let text: String = lines.iter().map(|l| format!("{l}\n")).collect();
        std::fs::write(path, text).unwrap();
    }

    fn session(cwd: &Path) -> Vec<Value> {
        vec![
            serde_json::json!({"type": "summary", "summary": "earlier work"}),
            serde_json::json!({"type": "user", "cwd": cwd.to_string_lossy(), "message": {"role": "user", "content": "go"}}),
        ]
    }

    #[test]
    fn s16_b172_the_folder_name_replaces_every_character_but_letters_and_digits() {
        assert_eq!(
            folder_name(Path::new(r"C:\Users\me\code-repo\my_app.v2")),
            "C--Users-me-code-repo-my-app-v2"
        );
        assert_eq!(folder_name(Path::new("/home/me/app")), "-home-me-app");
    }

    #[test]
    fn s16_b172_the_projects_folder_follows_claude_config_dir() {
        let home = Path::new("/home/me");
        assert_eq!(
            projects_dir(home, &|_| None),
            home.join(".claude").join("projects")
        );
        assert_eq!(
            projects_dir(home, &|k| (k == "CLAUDE_CONFIG_DIR")
                .then(|| "/cfg".to_owned())),
            Path::new("/cfg").join("projects")
        );
        assert_eq!(
            projects_dir(home, &|_| Some("  ".to_owned())),
            home.join(".claude").join("projects")
        );
    }

    #[test]
    fn s16_b172_replay_resolves_ids_files_and_all_with_the_recorded_folder() {
        let tmp = tempfile::tempdir().unwrap();
        let projects = tmp.path().join("projects");
        let here = tmp.path().join("work").join("app");
        let other = tmp.path().join("work").join("other");
        let folder = projects.join(folder_name(&here));
        write(&folder.join("s-2.jsonl"), &session(&here));
        write(&folder.join("s-1.jsonl"), &session(&here));
        std::fs::write(folder.join("notes.txt"), "not a transcript").unwrap();
        let elsewhere = projects.join(folder_name(&other)).join("s-9.jsonl");
        write(&elsewhere, &session(&other));
        let nameless = folder.join("s-3.jsonl");
        write(&nameless, &[serde_json::json!({"type": "summary"})]);

        let ready = |items: Vec<Item>| -> Vec<(String, String)> {
            items
                .into_iter()
                .map(|i| match i {
                    Item::Ready(t) => (t.session_id, t.cwd),
                    Item::Skipped { name, error } => (format!("skipped {name}"), error.message),
                })
                .collect()
        };
        let h = here.to_string_lossy().into_owned();
        let o = other.to_string_lossy().into_owned();

        // A session id of the current folder, and a file of another
        // project, which keeps its own folder.
        let items = resolve(
            &["s-1".into(), elsewhere.to_string_lossy().into_owned()],
            false,
            &here,
            &projects,
        )
        .unwrap();
        assert_eq!(
            ready(items),
            vec![("s-1".to_owned(), h.clone()), ("s-9".to_owned(), o)]
        );

        // --all: every .jsonl in name order; one with no working folder is
        // skipped, not guessed.
        let items = ready(resolve(&[], true, &here, &projects).unwrap());
        assert_eq!(items.len(), 3, "{items:?}");
        assert_eq!(items[0], ("s-1".to_owned(), h.clone()));
        assert_eq!(items[1], ("s-2".to_owned(), h));
        assert!(items[2].0.starts_with("skipped "), "{items:?}");
        assert!(
            items[2].1.contains("records no working folder"),
            "{items:?}"
        );

        // An unknown id is skipped; --all in a folder with no transcripts
        // is an error.
        let items = ready(resolve(&["nope".into()], false, &here, &projects).unwrap());
        assert_eq!(items[0].0, "skipped nope");
        assert!(resolve(&[], true, &other.join("x"), &projects).is_err());
    }

    #[test]
    fn s16_b172_replay_sends_one_request_a_transcript_and_reports_each() {
        let t = |id: &str| {
            Item::Ready(Transcript {
                path: PathBuf::from(format!("/p/{id}.jsonl")),
                session_id: id.to_owned(),
                cwd: "/work/app".to_owned(),
            })
        };
        let mut sent = Vec::new();
        let out = run(
            vec![
                t("a"),
                Item::Skipped {
                    name: "b".into(),
                    error: CoreError::invalid_input("no transcript `b`"),
                },
                t("c"),
            ],
            |op| {
                sent.push(op);
                Ok(Reply::Captured(CaptureReply { accepted: true }))
            },
        );
        assert_eq!(sent.len(), 2);
        let Op::Replay(p) = &sent[0] else {
            panic!("{:?}", sent[0]);
        };
        assert_eq!((p.session_id.as_str(), p.cwd.as_str()), ("a", "/work/app"));
        assert!(p.transcript_path.ends_with("a.jsonl"), "{p:?}");
        assert!(
            out.stdout.contains("queued a (/work/app)\n"),
            "{}",
            out.stdout
        );
        assert!(out.stdout.contains("skipped b\n"), "{}", out.stdout);
        assert!(out.stdout.contains("2 queued."), "{}", out.stdout);
        assert!(out.stderr.contains("no transcript `b`"), "{}", out.stderr);
        assert_eq!(out.code, 1, "a skipped item fails the command");

        let off = run(vec![t("a")], |_| {
            Ok(Reply::Captured(CaptureReply { accepted: false }))
        });
        assert!(
            off.stdout
                .contains("not queued a: automatic capture is off"),
            "{}",
            off.stdout
        );
        assert_eq!(off.code, 1);
    }
}
