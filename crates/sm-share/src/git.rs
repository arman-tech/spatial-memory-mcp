//! Running git for sync (SPEC 9, Team sharing rules 3 to 5, "Change
//! detection before reading"; decision B-78).
//!
//! git is run directly, never through a shell, at the repository root, with
//! its arguments as a list. Every process gets `GIT_TERMINAL_PROMPT=0` so it
//! can never wait for a password, has no console window on Windows
//! (`CREATE_NO_WINDOW`), reads nothing from stdin unless it is the blob
//! reader, and ignores `GIT_DIR`, `GIT_WORK_TREE`, `GIT_COMMON_DIR` and
//! `GIT_INDEX_FILE` inherited from a caller such as a git hook, so the
//! repository is always the one at the root given.
//!
//! There is no git-specific error code (SPEC 8), so git missing from `PATH`
//! and git failing are both reported as `not_git_repository`, with git's own
//! message and a next action that fits the failure.

use std::ffi::OsString;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use sm_core::{CoreError, ErrorCode, Result};

/// Environment variables that would point git at another repository.
const REPO_OVERRIDES: [&str; 4] = [
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_COMMON_DIR",
    "GIT_INDEX_FILE",
];

/// Runs git at one repository root.
#[derive(Debug, Clone)]
pub(crate) struct Git {
    root: PathBuf,
    /// Extra environment for every git process. Empty in production; tests
    /// use it to isolate git from the user's configuration.
    env: Vec<(OsString, OsString)>,
}

/// One entry of `git ls-tree -r -l`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TreeEntry {
    pub mode: String,
    pub kind: String,
    pub oid: String,
    /// `None` for an entry with no size (a submodule).
    pub size: Option<u64>,
    /// The path from the repository root, as git stores it.
    pub path: Vec<u8>,
}

impl Git {
    pub(crate) fn new(root: &Path, env: Vec<(OsString, OsString)>) -> Self {
        Git {
            root: root.to_path_buf(),
            env,
        }
    }

    fn command(&self, args: &[&str]) -> Command {
        let mut c = Command::new("git");
        c.args(args)
            .current_dir(&self.root)
            .env("GIT_TERMINAL_PROMPT", "0")
            .stdin(Stdio::null());
        for k in REPO_OVERRIDES {
            c.env_remove(k);
        }
        for (k, v) in &self.env {
            c.env(k, v);
        }
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            c.creation_flags(windows_sys::Win32::System::Threading::CREATE_NO_WINDOW);
        }
        c
    }

    /// Runs git to completion and returns its exit code and output.
    fn run(&self, args: &[&str]) -> Result<(Option<i32>, Vec<u8>, String)> {
        let out = self
            .command(args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .map_err(|e| spawn_error(args, &e))?;
        let stderr = String::from_utf8_lossy(&out.stderr).trim().to_owned();
        Ok((out.status.code(), out.stdout, stderr))
    }

    /// The commit `HEAD` points at, or `None` when the repository has no
    /// commits yet.
    pub(crate) fn head(&self) -> Result<Option<String>> {
        let args = ["rev-parse", "--verify", "-q", "HEAD^{commit}"];
        let (code, stdout, stderr) = self.run(&args)?;
        let sha = String::from_utf8_lossy(&stdout).trim().to_owned();
        match code {
            Some(0) if !sha.is_empty() => Ok(Some(sha)),
            // `--verify -q` exits 1 quietly when HEAD names no commit.
            Some(1) if sha.is_empty() && stderr.is_empty() => Ok(None),
            _ => Err(failed(&args, code, &stderr)),
        }
    }

    /// Every entry under `prefix` in `commit`'s tree, recursively.
    pub(crate) fn ls_tree(&self, commit: &str, prefix: &str) -> Result<Vec<TreeEntry>> {
        let args = [
            "ls-tree",
            "-r",
            "-z",
            "-l",
            "--full-tree",
            commit,
            "--",
            prefix,
        ];
        let (code, stdout, stderr) = self.run(&args)?;
        if code != Some(0) {
            return Err(failed(&args, code, &stderr));
        }
        let mut out = Vec::new();
        for record in stdout.split(|b| *b == 0).filter(|r| !r.is_empty()) {
            out.push(parse_tree_record(record).ok_or_else(|| {
                CoreError::new(
                    ErrorCode::NotGitRepository,
                    format!(
                        "git ls-tree printed a line this build cannot read: {:?}",
                        String::from_utf8_lossy(record)
                    ),
                )
                .with_next_action(
                    "Check that `git --version` is 2.0 or later, then run sync again.",
                )
            })?);
        }
        Ok(out)
    }

    /// Starts one `git cat-file --batch` process for reading blobs.
    pub(crate) fn blob_reader(&self) -> Result<BlobReader> {
        let args = ["cat-file", "--batch"];
        let mut child = self
            .command(&args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| spawn_error(&args, &e))?;
        let stdin = child.stdin.take().map(BufWriter::new);
        let stdout = child.stdout.take().map(BufReader::new);
        match (stdin, stdout) {
            (Some(stdin), Some(stdout)) => Ok(BlobReader {
                child,
                stdin: Some(stdin),
                stdout,
                reads: 0,
            }),
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                Err(CoreError::internal("git cat-file started without pipes"))
            }
        }
    }
}

/// `<mode> SP <type> SP <object> SP <size> TAB <path>`; the size is padded
/// with spaces and is `-` for a submodule.
fn parse_tree_record(record: &[u8]) -> Option<TreeEntry> {
    let tab = record.iter().position(|b| *b == b'\t')?;
    let meta = std::str::from_utf8(&record[..tab]).ok()?;
    let mut parts = meta.split_whitespace();
    let mode = parts.next()?.to_owned();
    let kind = parts.next()?.to_owned();
    let oid = parts.next()?.to_owned();
    let size = match parts.next()? {
        "-" => None,
        s => Some(s.parse().ok()?),
    };
    if parts.next().is_some() {
        return None;
    }
    Some(TreeEntry {
        mode,
        kind,
        oid,
        size,
        path: record[tab + 1..].to_vec(),
    })
}

/// Reads blobs through one `git cat-file --batch` process.
pub(crate) struct BlobReader {
    child: Child,
    stdin: Option<BufWriter<ChildStdin>>,
    stdout: BufReader<ChildStdout>,
    reads: u64,
}

impl BlobReader {
    /// Blobs read so far.
    pub(crate) fn reads(&self) -> u64 {
        self.reads
    }

    /// The bytes of one blob, or `None` when git reports it missing.
    pub(crate) fn read(&mut self, oid: &str) -> Result<Option<Vec<u8>>> {
        match self.try_read(oid) {
            Ok(v) => Ok(v),
            Err(e) => Err(self.fail(e)),
        }
    }

    fn try_read(&mut self, oid: &str) -> std::io::Result<Option<Vec<u8>>> {
        let stdin = self
            .stdin
            .as_mut()
            .ok_or_else(|| std::io::Error::other("the blob reader is closed"))?;
        stdin.write_all(oid.as_bytes())?;
        stdin.write_all(b"\n")?;
        stdin.flush()?;
        let mut header = String::new();
        if self.stdout.read_line(&mut header)? == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "git cat-file closed its output",
            ));
        }
        let header = header.trim_end_matches('\n');
        let mut parts = header.split(' ');
        let (got, kind, size) = (parts.next(), parts.next(), parts.next());
        if kind == Some("missing") {
            return Ok(None);
        }
        let (Some(got), Some(kind), Some(size)) = (got, kind, size) else {
            return Err(std::io::Error::other(format!(
                "unexpected git cat-file header {header:?}"
            )));
        };
        let size: usize = size.parse().map_err(|_| {
            std::io::Error::other(format!("unexpected git cat-file header {header:?}"))
        })?;
        let mut data = vec![0u8; size];
        self.stdout.read_exact(&mut data)?;
        let mut lf = [0u8; 1];
        self.stdout.read_exact(&mut lf)?;
        if lf[0] != b'\n' {
            return Err(std::io::Error::other(
                "git cat-file output is not followed by a line feed",
            ));
        }
        if kind != "blob" || !got.eq_ignore_ascii_case(oid) {
            return Err(std::io::Error::other(format!(
                "git cat-file returned {kind} {got} for blob {oid}"
            )));
        }
        self.reads += 1;
        Ok(Some(data))
    }

    /// Stops the process and turns an I/O failure into an error that carries
    /// git's own message.
    fn fail(&mut self, e: std::io::Error) -> CoreError {
        self.stdin = None;
        let _ = self.child.kill();
        let mut stderr = String::new();
        if let Some(mut s) = self.child.stderr.take() {
            let _ = s.read_to_string(&mut stderr);
        }
        let _ = self.child.wait();
        let stderr = stderr.trim();
        let detail = if stderr.is_empty() {
            e.to_string()
        } else {
            format!("{e}: {stderr}")
        };
        CoreError::new(
            ErrorCode::NotGitRepository,
            format!("git cat-file --batch failed: {detail}"),
        )
        .with_next_action(next_action_for(stderr, &["cat-file", "--batch"]))
    }

    /// Closes the process's input and waits for it to exit.
    pub(crate) fn finish(mut self) -> Result<()> {
        self.stdin = None;
        let status = self.child.wait().map_err(|e| {
            CoreError::new(
                ErrorCode::NotGitRepository,
                format!("waiting for git cat-file failed: {e}"),
            )
        })?;
        if status.success() {
            return Ok(());
        }
        let mut stderr = String::new();
        if let Some(mut s) = self.child.stderr.take() {
            let _ = s.read_to_string(&mut stderr);
        }
        Err(failed(
            &["cat-file", "--batch"],
            status.code(),
            stderr.trim(),
        ))
    }
}

impl Drop for BlobReader {
    fn drop(&mut self) {
        // Closing stdin ends git; a reader dropped after an error has
        // already been stopped. `try_wait` keeps drop from blocking on a
        // process that did not exit.
        self.stdin = None;
        if matches!(self.child.try_wait(), Ok(None)) {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

fn display(args: &[&str]) -> String {
    format!("git {}", args.join(" "))
}

fn spawn_error(args: &[&str], e: &std::io::Error) -> CoreError {
    if e.kind() == std::io::ErrorKind::NotFound {
        CoreError::new(
            ErrorCode::NotGitRepository,
            format!("git is not installed or not on PATH ({e})"),
        )
        .with_next_action("Install git and make sure `git` is on PATH, then run sync again.")
    } else {
        CoreError::new(
            ErrorCode::NotGitRepository,
            format!("could not start `{}`: {e}", display(args)),
        )
        .with_next_action("Check that git runs from this folder, then run sync again.")
    }
}

fn failed(args: &[&str], code: Option<i32>, stderr: &str) -> CoreError {
    let status = code.map_or_else(|| "was stopped".to_owned(), |c| format!("exited with {c}"));
    let detail = if stderr.is_empty() {
        String::new()
    } else {
        format!(": {stderr}")
    };
    CoreError::new(
        ErrorCode::NotGitRepository,
        format!("`{}` {status}{detail}", display(args)),
    )
    .with_next_action(next_action_for(stderr, args))
}

fn next_action_for(stderr: &str, args: &[&str]) -> String {
    if stderr.contains("dubious ownership") {
        "git does not trust this folder's owner. Add it with `git config --global --add \
         safe.directory <path>` if you trust it, then run sync again."
            .to_owned()
    } else if stderr.contains("not a git repository") {
        "Run sync from inside a git repository.".to_owned()
    } else {
        format!(
            "Run `{}` in the repository to see the problem, fix it, then run sync again.",
            display(args)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s9_ls_tree_records_are_parsed_with_padded_sizes() {
        let e = parse_tree_record(
            b"100644 blob 0123456789abcdef0123456789abcdef01234567     120\t.spatial-memory/shared/note/a--12345678.md",
        )
        .unwrap();
        assert_eq!(e.mode, "100644");
        assert_eq!(e.kind, "blob");
        assert_eq!(e.size, Some(120));
        assert_eq!(e.path, b".spatial-memory/shared/note/a--12345678.md");
        let sub = parse_tree_record(
            b"160000 commit 0123456789abcdef0123456789abcdef01234567       -\t.spatial-memory/shared/x",
        )
        .unwrap();
        assert_eq!(sub.size, None);
        assert!(parse_tree_record(b"100644 blob abc").is_none());
    }
}
