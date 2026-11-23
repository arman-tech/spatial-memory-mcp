//! The commit in the program's build stamp (B-231), shared by `build.rs`,
//! which sets `SM_COMMIT` to [`Stamp::commit`] and tells cargo to watch
//! [`Stamp::watch`], and by the tests in `src/version.rs`, which run it on
//! repositories of their own.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The paths whose changes make a build `.dirty`, from the workspace root:
/// the source, the vendored C, the manifests, the toolchain, and each file
/// outside them that the program builds in with `include_str!` (doctor's
/// plugin manifest, the judge's prompts and schemas). The test
/// `s3_b231_every_file_built_into_the_program_is_stamped` fails when an
/// `include_str!` names a file outside this list.
pub const STAMPED: &[&str] = &[
    "crates",
    "vendor",
    "Cargo.toml",
    "Cargo.lock",
    "rust-toolchain.toml",
    "plugin/.claude-plugin/plugin.json",
    "docs/v3/spikes/judge/prompt.txt",
    "docs/v3/spikes/judge/schema.json",
    "docs/v3/spikes/judge/repeats-prompt.txt",
    "docs/v3/spikes/judge/repeats-schema.json",
    "docs/v3/spikes/judge/statements-prompt.txt",
    "docs/v3/spikes/judge/same-prompt.txt",
];

/// The commit of a build made where git could not say.
pub const UNKNOWN: &str = "unknown";

/// What follows the hash of a build made with uncommitted changes.
pub const DIRTY: &str = ".dirty";

/// The workspace root's files that decide which files git ignores and how
/// it compares them, so whether a build is `.dirty`.
const ROOT_RULES: &[&str] = &[".gitignore", ".gitattributes"];

/// Where `sm-cli`'s folder sits in its repository, as `git rev-parse
/// --show-prefix` prints it. git run in a source tree without its own
/// repository finds any repository above it, which is not this one.
const PREFIX: &str = "crates/sm-cli/";

/// The variables that point git at a repository, a work tree, an index or
/// objects: those `git rev-parse --local-env-vars` lists (git 2.50), less
/// `GIT_CONFIG_PARAMETERS` and `GIT_CONFIG_COUNT`. Those two carry the
/// user's own `-c` settings, such as `safe.directory`, and git keeps them
/// itself when it runs git for another repository. git sets some of the
/// others for a hook, so a `cargo` run from a hook would hand them to this
/// script's git, which would then read that hook's repository or index.
pub const LOCAL_ENV: &[&str] = &[
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_CONFIG",
    "GIT_OBJECT_DIRECTORY",
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_IMPLICIT_WORK_TREE",
    "GIT_GRAFT_FILE",
    "GIT_INDEX_FILE",
    "GIT_NO_REPLACE_OBJECTS",
    "GIT_REPLACE_REF_BASE",
    "GIT_PREFIX",
    "GIT_SHALLOW_FILE",
    "GIT_COMMON_DIR",
];

pub struct Stamp {
    /// git's short hash of `HEAD`, with `.dirty` after it when a tracked
    /// file under one of the [`STAMPED`] paths differs from that commit or
    /// an untracked one is there that git does not ignore; or `unknown`
    /// when git cannot say (no git, or a source tree without its own
    /// repository).
    pub commit: String,
    /// What cargo watches, so that it runs the script again when the
    /// commit or the mark may have changed, and only then: git's `HEAD`,
    /// branch refs, `packed-refs`, `reftable` (the worktree's own and the
    /// shared one) and `info/exclude` when git answered, and the stamped
    /// paths and the root's ignore and attribute files always, so that a
    /// build made without git asks git again after the next change. A path
    /// that does not exist is left out, since cargo runs a script whose
    /// watched path is missing on every build.
    pub watch: Vec<PathBuf>,
}

/// The stamp of the `sm-cli` folder `manifest`. `git` makes the command
/// that runs git: `build.rs` passes `|| Command::new("git")`.
pub fn stamp(manifest: &Path, git: &dyn Fn() -> Command) -> Stamp {
    // `crates/sm-cli` is two folders below the workspace root. The parent
    // is taken by name: a canonical Windows path is verbatim (`\\?\`), in
    // which the `/` of the paths below is not a separator.
    let root = manifest.parent().and_then(Path::parent).unwrap_or(manifest);
    let mut watch: Vec<PathBuf> = STAMPED
        .iter()
        .chain(ROOT_RULES)
        .map(|p| root.join(p))
        .collect();
    let commit = commit(manifest, root, git, &mut watch).unwrap_or_else(|| UNKNOWN.to_owned());
    watch.retain(|p| p.exists());
    Stamp { commit, watch }
}

/// The commit and its mark, or `None` when git cannot say. Adds git's own
/// files to `watch` when git answered for this repository.
fn commit(
    manifest: &Path,
    root: &Path,
    git: &dyn Fn() -> Command,
    watch: &mut Vec<PathBuf>,
) -> Option<String> {
    let found = run(
        git,
        manifest,
        &[
            "rev-parse",
            "--show-prefix",
            "--path-format=absolute",
            "--git-path",
            "HEAD",
            "--git-path",
            "packed-refs",
            "--git-path",
            "refs/heads",
            "--git-path",
            "info/exclude",
            // A repository made with `--ref-format=reftable` keeps its
            // branches here; its `HEAD` and `refs/heads` are placeholders.
            "--git-path",
            "reftable",
            // A linked worktree's `reftable` holds its own `HEAD`; its
            // branches are in the shared one, under the common folder.
            "--git-common-dir",
            "--short=7",
            "HEAD",
        ],
    )?;
    let lines: Vec<&str> = found.lines().collect();
    let [prefix, head, packed, heads, exclude, reftable, common, hash] = lines.as_slice() else {
        return None;
    };
    if *prefix != PREFIX {
        return None;
    }
    watch.extend([head, packed, heads, exclude, reftable].map(PathBuf::from));
    let shared = Path::new(common).join("reftable");
    if !watch.contains(&shared) {
        watch.push(shared);
    }
    // `--untracked-files=normal`: an untracked file counts even where the
    // user's git config hides them from `status`.
    let mut status = vec!["status", "--porcelain", "--untracked-files=normal", "--"];
    status.extend(STAMPED);
    let changes = run(git, root, &status)?;
    Some(if changes.is_empty() {
        (*hash).to_owned()
    } else {
        format!("{hash}{DIRTY}")
    })
}

/// Runs git in `dir` and returns its trimmed output, or `None` when it
/// could not run or failed.
fn run(git: &dyn Fn() -> Command, dir: &Path, args: &[&str]) -> Option<String> {
    let mut command = git();
    for name in LOCAL_ENV {
        command.env_remove(name);
    }
    // `--no-optional-locks`: `status` must not refresh git's index while
    // the user may be running git themselves.
    let out = command
        .arg("--no-optional-locks")
        .arg("-C")
        .arg(dir)
        .args(args)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8(out.stdout).ok()?.trim().to_owned())
}
