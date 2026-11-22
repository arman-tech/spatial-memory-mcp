//! The program's own build stamp (B-231). The release alone
//! ([`sm_core::BUILD`]) is what clients and the daemon compare; the commit
//! after `+` tells two builds of one release apart, for people and for
//! `doctor`, and is never part of that check.

/// The commit this program was built from: git's short hash, with `.dirty`
/// when the build had uncommitted changes, or `unknown` (see `build.rs`).
pub const COMMIT: &str = env!("SM_COMMIT");

/// `<release>+<commit>`, as `spatial-memory --version` prints it and the
/// MCP server reports it to Claude Code.
pub const STAMP: &str = concat!(env!("CARGO_PKG_VERSION"), "+", env!("SM_COMMIT"));

/// The commit of a build made where git could not say (`build.rs`).
pub const UNKNOWN: &str = "unknown";

/// What follows the hash of a build made with uncommitted changes
/// (`build.rs`).
pub const DIRTY: &str = ".dirty";

/// The release and the commit of a stamp: `3.0.0+1b5b54a` gives `3.0.0`
/// and `Some("1b5b54a")`, and a stamp without `+` (a build older than
/// B-231) gives no commit.
pub fn split(stamp: &str) -> (&str, Option<&str>) {
    match stamp.split_once('+') {
        Some((release, commit)) => (release, Some(commit)),
        None => (stamp, None),
    }
}

/// `build.rs`'s own code, for the tests below to run.
#[cfg(test)]
#[path = "../build/stamp.rs"]
mod stamp;

#[cfg(test)]
mod tests {
    use std::ffi::{OsStr, OsString};
    use std::path::{Path, PathBuf};
    use std::process::Command;

    use super::*;

    /// Lists untracked files, a new folder as one entry, whatever
    /// `status.showUntrackedFiles` says.
    const STATUS: [&str; 3] = ["status", "--porcelain", "--untracked-files=normal"];

    /// B-231: the stamp is the release clients and the daemon compare, then
    /// `+` and the commit. Without git it says `unknown`, never nothing.
    #[test]
    fn s3_b231_the_stamp_is_the_release_and_the_commit() {
        assert_eq!(split(STAMP), (sm_core::BUILD, Some(COMMIT)));
        assert!(!COMMIT.is_empty());
        assert!(!COMMIT.contains(char::is_whitespace), "{COMMIT}");
        // `build.rs` writes these two words itself; this ties them to the
        // constants doctor compares with.
        assert_eq!(stamp::UNKNOWN, UNKNOWN);
        assert_eq!(stamp::DIRTY, DIRTY);
        if COMMIT != UNKNOWN {
            let hash = COMMIT.strip_suffix(DIRTY).unwrap_or(COMMIT);
            assert!(hash.len() >= 7, "{COMMIT}");
            assert!(hash.chars().all(|c| c.is_ascii_hexdigit()), "{COMMIT}");
        }
        assert_eq!(split("3.0.0"), ("3.0.0", None));
        assert_eq!(
            split("3.0.0+abc1234.dirty"),
            ("3.0.0", Some("abc1234.dirty"))
        );
    }

    /// B-231: a file the program builds in changes the program, so an
    /// uncommitted edit to it must make the stamp `.dirty`. Every
    /// `include_str!` or `include_bytes!` in a crate's `src` must name a
    /// file under one of the paths `build.rs` stamps, and must name it with
    /// a literal this test can read.
    #[test]
    fn s3_b231_every_file_built_into_the_program_is_stamped() {
        // The reader finds a plain or raw literal after any white space,
        // and reports an argument that is not one.
        let sample = "INC_str!(\"a\") INC_str!(r\"b/c\") INC_bytes!(\n    \"d\") \
                      INC_str!(r#\"e\"#) INC_str!(concat!(\"f\"))"
            .replace("INC", "include");
        assert_eq!(
            included(&sample),
            [Ok("a"), Ok("b/c"), Ok("e"), Err("concat!(\"f\"))"), Ok("d")]
        );

        // Paths are joined before they are made canonical: a canonical
        // Windows path is verbatim (`\\?\`), where `/` and `..` are not
        // read as a separator and a parent.
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let stamped: Vec<PathBuf> = stamp::STAMPED
            .iter()
            .map(|p| {
                root.join(p)
                    .canonicalize()
                    .unwrap_or_else(|e| panic!("{p}: {e}"))
            })
            .collect();
        let mut sources = Vec::new();
        for krate in std::fs::read_dir(root.join("crates")).unwrap() {
            let src = krate.unwrap().path().join("src");
            if src.is_dir() {
                rust_files(&src, &mut sources);
            }
        }
        let mut count = 0;
        for file in sources {
            let text = std::fs::read_to_string(&file).unwrap();
            for named in included(&text) {
                let named = named.unwrap_or_else(|arg| {
                    panic!("{file:?} builds in `{arg}`, not a literal path: check it is stamped")
                });
                let target = file.parent().unwrap().join(named).canonicalize();
                let target = target.unwrap_or_else(|e| panic!("{named} in {file:?}: {e}"));
                assert!(
                    stamped.iter().any(|s| target.starts_with(s)),
                    "{file:?} builds in {target:?}, which build.rs does not stamp"
                );
                count += 1;
            }
        }
        // doctor's plugin manifest and the judge's four files at least.
        assert!(count >= 5, "{count}");
    }

    /// What each `include_str!` and `include_bytes!` in `text` names: the
    /// literal path, or `Err` with the start of an argument that is not a
    /// literal. The marks are built here so that this file's own text does
    /// not hold them.
    fn included(text: &str) -> Vec<Result<&str, &str>> {
        let mut out = Vec::new();
        for kind in ["_str", "_bytes"] {
            let mark = format!("include{kind}!(");
            for (at, _) in text.match_indices(&mark) {
                let arg = text[at + mark.len()..].trim_start();
                let (hashes, literal) = match arg.strip_prefix('r') {
                    Some(raw) => {
                        let body = raw.trim_start_matches('#');
                        (raw.len() - body.len(), body)
                    }
                    None => (0, arg),
                };
                let end = format!("\"{}", "#".repeat(hashes));
                let path = literal
                    .strip_prefix('"')
                    .and_then(|body| body.find(&end).map(|e| &body[..e]));
                out.push(path.ok_or_else(|| arg.lines().next().unwrap_or(arg)));
            }
        }
        out
    }

    fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                rust_files(&path, out);
            } else if path.extension().is_some_and(|e| e == "rs") {
                out.push(path);
            }
        }
    }

    /// The test's only git settings, written to `dir/gitconfig`. They name
    /// a missing file for the ignore rules and the attributes: otherwise git
    /// reads the user's `$XDG_CONFIG_HOME/git/ignore` and `attributes`, or
    /// those under `~/.config`, which `GIT_CONFIG_GLOBAL` does not replace.
    /// A `*.tmp` in the user's ignore file hid the test's untracked file.
    fn settings(dir: &Path) -> PathBuf {
        let none = quoted(&dir.join("none"));
        let path = dir.join("gitconfig");
        let text = format!("[core]\n\texcludesFile = {none}\n\tattributesFile = {none}\n");
        std::fs::write(&path, text).unwrap();
        path
    }

    /// `path` as a double-quoted value in a git settings file.
    fn quoted(path: &Path) -> String {
        let text = path.display().to_string();
        format!("\"{}\"", text.replace('\\', "\\\\").replace('"', "\\\""))
    }

    /// A template folder in `dir` whose `info/exclude` hides `*.tmp`.
    fn template(dir: &Path) -> PathBuf {
        let template = dir.join("template");
        let exclude = template.join("info").join("exclude");
        std::fs::create_dir_all(exclude.parent().unwrap()).unwrap();
        std::fs::write(exclude, "*.tmp\n").unwrap();
        template
    }

    /// A repository's config and index, and its `HEAD` commit.
    fn state(git: &dyn Fn() -> Command, repo: &Path) -> ([Vec<u8>; 2], String) {
        let dot_git = repo.join(".git");
        let files = ["config", "index"].map(|f| std::fs::read(dot_git.join(f)).unwrap());
        (files, git_in(git, repo, &["rev-parse", "HEAD"]))
    }

    /// Whether `name` is one of git's variables: it starts with `GIT_`, in
    /// any case, since Windows compares the names without case.
    fn is_gits(name: &OsStr) -> bool {
        name.to_string_lossy()
            .to_ascii_uppercase()
            .starts_with("GIT_")
    }

    /// `base` made into git for the test's own repositories, isolated from
    /// where the test runs. No variable of git's (any name that starts
    /// with `GIT_`), whether this process has it or `base` sets it: a hook
    /// or `git rebase -x` sets `GIT_DIR` and others, which would point the
    /// test's `git init`, `add` and `commit` at the repository being worked
    /// on (a hook's `GIT_DIR` made a decoy repository bare), and the user
    /// may set others, such as `GIT_TEMPLATE_DIR`. No system settings or
    /// attributes, no user settings but those of [`settings`], and no
    /// prompt on the terminal.
    fn isolated(mut c: Command, config: &Path) -> Command {
        let set = c.get_envs().map(|(name, _)| name.to_owned());
        let names: Vec<OsString> = std::env::vars_os()
            .map(|(name, _)| name)
            .chain(set)
            .filter(|name| is_gits(name))
            .collect();
        for name in names {
            c.env_remove(name);
        }
        c.env("GIT_CONFIG_GLOBAL", config)
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .env("GIT_ATTR_NOSYSTEM", "1")
            .env("GIT_TERMINAL_PROMPT", "0");
        c
    }

    /// Runs `git` in `dir` with a fixed user and no hooks, and returns its
    /// trimmed output.
    fn git_in(git: &dyn Fn() -> Command, dir: &Path, args: &[&str]) -> String {
        let no_hooks = format!("core.hooksPath={}", dir.join("no-hooks").display());
        let out = git()
            .args([
                "-c",
                "user.name=b231",
                "-c",
                "user.email=b231@example.invalid",
            ])
            .args(["-c", "commit.gpgsign=false", "-c", &no_hooks])
            .arg("-C")
            .arg(dir)
            .args(args)
            .output()
            .expect("this test needs git on PATH");
        assert!(
            out.status.success(),
            "git {args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8(out.stdout).unwrap().trim().to_owned()
    }

    /// A repository made by `git init` with the arguments `init`, holding
    /// each `(path, text)` of `files` in one commit; returns its short hash.
    fn repository(
        git: &dyn Fn() -> Command,
        dir: &Path,
        init: &[&str],
        files: &[(&str, &str)],
    ) -> String {
        for (f, text) in files {
            let path = dir.join(f);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, text).unwrap();
        }
        git_in(git, dir, &[&["init", "-q"], init].concat());
        git_in(git, dir, &["add", "-A"]);
        git_in(git, dir, &["commit", "-q", "-m", "one"]);
        git_in(git, dir, &["rev-parse", "--short=7", "HEAD"])
    }

    /// B-231: `build.rs` names the commit of the repository `sm-cli` is in,
    /// marks it `.dirty` for any change under a stamped path, and says
    /// `unknown` rather than another repository's commit. Cargo watches the
    /// stamped paths even without git, so a later build asks again. The
    /// test's own git is isolated from the repository it runs in. Its
    /// folder's name has an apostrophe, as a user's folder may.
    #[test]
    fn s3_b231_build_rs_names_this_repositorys_commit_and_every_change() {
        let tmp = tempfile::Builder::new()
            .prefix("b231-o'neil-")
            .tempdir()
            .unwrap();
        let config = settings(tmp.path());
        let git = || isolated(Command::new("git"), &config);
        let repo = tmp.path().join("repo");
        let manifest = repo.join("crates").join("sm-cli");
        let hash = repository(
            &git,
            &repo,
            &[],
            &[
                ("crates/sm-cli/Cargo.toml", "[package]\n"),
                ("Cargo.toml", "[workspace]\n"),
                (".gitignore", "target/\n"),
            ],
        );
        let dirty = format!("{hash}{DIRTY}");
        let commit = |git: &dyn Fn() -> Command| stamp::stamp(&manifest, git).commit;

        let s = stamp::stamp(&manifest, &git);
        assert_eq!(s.commit, hash);
        for watched in ["crates", "Cargo.toml", ".gitignore"] {
            assert!(
                s.watch.contains(&repo.join(watched)),
                "{watched}: {:?}",
                s.watch
            );
        }
        assert!(
            s.watch.iter().any(|p| p.ends_with(".git/HEAD")),
            "{:?}",
            s.watch
        );
        assert!(!s.watch.contains(&repo.join("vendor")), "{:?}", s.watch);

        // A tracked file that differs from the commit.
        std::fs::write(manifest.join("Cargo.toml"), "two\n").unwrap();
        assert_eq!(commit(&git), dirty);
        git_in(&git, &repo, &["checkout", "--", "."]);
        assert_eq!(commit(&git), hash);

        // An untracked file, even where the repository's config hides
        // untracked files from `status`.
        git_in(&git, &repo, &["config", "status.showUntrackedFiles", "no"]);
        std::fs::write(manifest.join("new.rs"), "").unwrap();
        assert_eq!(commit(&git), dirty);
        std::fs::remove_file(manifest.join("new.rs")).unwrap();

        // An ignored file, and a file outside the stamped paths, change
        // nothing.
        std::fs::create_dir_all(manifest.join("target")).unwrap();
        std::fs::write(manifest.join("target").join("out"), "").unwrap();
        std::fs::create_dir_all(repo.join("docs")).unwrap();
        std::fs::write(repo.join("docs").join("note.md"), "").unwrap();
        assert_eq!(commit(&git), hash);

        // The user's own `-c` settings reach the script's git, as git keeps
        // them for another repository: `safe.directory` among them, without
        // which git refuses a repository another user owns. Here one names
        // a file of ignore rules, through `GIT_CONFIG_COUNT` and through
        // `GIT_CONFIG_PARAMETERS`, which `git -c` sets.
        let exclude = tmp.path().join("exclude");
        std::fs::write(&exclude, "*.tmp\n").unwrap();
        let configured = || {
            let mut c = git();
            c.env("GIT_CONFIG_COUNT", "1")
                .env("GIT_CONFIG_KEY_0", "core.excludesFile")
                .env("GIT_CONFIG_VALUE_0", &exclude);
            c
        };
        std::fs::write(manifest.join("x.tmp"), "").unwrap();
        assert_eq!(commit(&git), dirty);
        assert_eq!(commit(&configured), hash);
        let parameters = || {
            let mut c = git();
            // Quoted so that git reads it back: each `'` becomes `'\''`.
            let value = exclude.display().to_string().replace('\'', "'\\''");
            c.env(
                "GIT_CONFIG_PARAMETERS",
                format!("'core.excludesFile'='{value}'"),
            );
            c
        };
        assert_eq!(commit(&parameters), hash);
        std::fs::remove_file(manifest.join("x.tmp")).unwrap();

        // The variables git sets for a hook in another repository do not
        // reach the script's git.
        let other = tmp.path().join("other");
        let other_hash = repository(&git, &other, &[], &[("f", "other\n")]);
        let other_git = other.join(".git");
        let hooked = || {
            let mut c = git();
            c.env("GIT_DIR", &other_git)
                .env("GIT_INDEX_FILE", other_git.join("index"));
            c
        };
        assert_ne!(other_hash, hash);
        assert_eq!(commit(&hooked), hash);

        // A source tree without its own repository, inside another one.
        let copy = other.join("copy");
        let copied = copy.join("crates").join("sm-cli");
        std::fs::create_dir_all(&copied).unwrap();
        std::fs::write(copied.join("Cargo.toml"), "[package]\n").unwrap();
        let s = stamp::stamp(&copied, &git);
        assert_eq!(s.commit, UNKNOWN);
        assert!(s.watch.contains(&copy.join("crates")), "{:?}", s.watch);
        assert!(
            !s.watch.iter().any(|p| p.ends_with("HEAD")),
            "{:?}",
            s.watch
        );

        // No git at all.
        let missing = || Command::new("git-b231-not-installed");
        let s = stamp::stamp(&manifest, &missing);
        assert_eq!(s.commit, UNKNOWN);
        assert!(s.watch.contains(&repo.join("crates")), "{:?}", s.watch);
    }

    /// B-231: in a repository that keeps its refs in a reftable, a commit
    /// changes `reftable` while `HEAD` and `refs/heads` stay placeholders,
    /// so cargo watches `reftable`. A linked worktree keeps its `HEAD` in
    /// its own reftable and its branches in the shared one, where another
    /// worktree may move them, so cargo watches both. `git init
    /// --ref-format` needs git 2.45 or newer.
    #[test]
    fn s3_b231_build_rs_watches_a_reftable_repositorys_branches() {
        let tmp = tempfile::tempdir().unwrap();
        let config = settings(tmp.path());
        let git = || isolated(Command::new("git"), &config);
        let version = git_in(&git, tmp.path(), &["version"]);
        let mut numbers = version
            .trim_start_matches("git version ")
            .split('.')
            .map(|n| n.parse::<u32>().unwrap_or(0));
        let found = (numbers.next().unwrap_or(0), numbers.next().unwrap_or(0));
        assert!(
            found >= (2, 45),
            "this test needs git 2.45 or newer, for `git init --ref-format=reftable`; \
             PATH has {version}"
        );
        let table = tmp.path().join("table");
        let hash = repository(
            &git,
            &table,
            &["--ref-format=reftable"],
            &[("crates/sm-cli/Cargo.toml", "[package]\n")],
        );
        let s = stamp::stamp(&table.join("crates").join("sm-cli"), &git);
        assert_eq!(s.commit, hash);
        assert!(
            s.watch.iter().any(|p| p.ends_with("table/.git/reftable")),
            "{:?}",
            s.watch
        );

        let linked = tmp.path().join("linked");
        let path = linked.to_str().unwrap();
        git_in(
            &git,
            &table,
            &["worktree", "add", "-q", "-b", "linked", path],
        );
        let s = stamp::stamp(&linked.join("crates").join("sm-cli"), &git);
        assert_eq!(s.commit, hash);
        for watched in [
            "table/.git/reftable",
            "table/.git/worktrees/linked/reftable",
        ] {
            assert!(
                s.watch.iter().any(|p| p.ends_with(watched)),
                "{watched}: {:?}",
                s.watch
            );
        }
    }

    /// B-231: the test's own git leaves alone the repository the test runs
    /// in, even when a hook or `git rebase -x` has set git's variables for
    /// it. Here they name a decoy repository.
    #[test]
    fn s3_b231_the_tests_git_leaves_the_repository_it_runs_in_alone() {
        let tmp = tempfile::tempdir().unwrap();
        let config = settings(tmp.path());
        let git = || isolated(Command::new("git"), &config);
        let decoy = tmp.path().join("decoy");
        repository(&git, &decoy, &[], &[("f", "decoy\n")]);
        let dot_git = decoy.join(".git");
        let before = state(&git, &decoy);
        let hooked = || {
            let mut c = Command::new("git");
            c.env("GIT_DIR", &dot_git)
                .env("GIT_INDEX_FILE", dot_git.join("index"));
            isolated(c, &config)
        };
        let own = tmp.path().join("own");
        repository(&hooked, &own, &[], &[("g", "own\n")]);
        assert!(own.join(".git").is_dir());
        assert_eq!(state(&git, &decoy), before);
    }

    /// B-231: the tests' own git reads none of the user's ignore rules or
    /// attributes, which git takes from `$XDG_CONFIG_HOME/git` (or
    /// `~/.config/git`) whatever `GIT_CONFIG_GLOBAL` names, and none of
    /// git's variables the user may have set: here a template whose
    /// `info/exclude` hides `*.tmp`, an attribute source that does not
    /// exist, and the ref format that is not git's default, which a control
    /// repository made without the variables shows. `--show-ref-format`
    /// needs git 2.45 or newer: an older git fails the test after its other
    /// checks.
    #[test]
    fn s3_b231_the_tests_git_reads_none_of_the_users_git_files_or_variables() {
        let tmp = tempfile::tempdir().unwrap();
        let config = settings(tmp.path());
        let user = tmp.path().join("user");
        for (path, text) in [
            (user.join("git").join("ignore"), "*.tmp\n"),
            (user.join("git").join("attributes"), "*.toml b231\n"),
        ] {
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, text).unwrap();
        }
        let hiding = template(tmp.path());
        let plain = || isolated(Command::new("git"), &config);
        let control = tmp.path().join("control");
        std::fs::create_dir_all(&control).unwrap();
        git_in(&plain, &control, &["init", "-q"]);
        let show = ["rev-parse", "--show-ref-format"];
        let default = git_in(&plain, &control, &show);
        let other = if default == "reftable" {
            "files"
        } else {
            "reftable"
        };
        let git = || {
            let mut c = Command::new("git");
            c.env("XDG_CONFIG_HOME", &user)
                .env("GIT_TEMPLATE_DIR", &hiding)
                .env("GIT_ATTR_SOURCE", "b231-no-such-tree")
                .env("GIT_DEFAULT_REF_FORMAT", other);
            isolated(c, &config)
        };
        let repo = tmp.path().join("repo");
        repository(&git, &repo, &[], &[("Cargo.toml", "[package]\n")]);
        std::fs::write(repo.join("x.tmp"), "").unwrap();
        assert_eq!(git_in(&git, &repo, &STATUS), "?? x.tmp");
        let attribute = ["check-attr", "b231", "--", "Cargo.toml"];
        assert_eq!(
            git_in(&git, &repo, &attribute),
            "Cargo.toml: b231: unspecified"
        );
        // Last, so that an older git still runs the checks above.
        assert!(
            default == "files" || default == "reftable",
            "{default}: --show-ref-format needs git 2.45 or newer"
        );
        assert_eq!(git_in(&git, &repo, &show), default);
    }

    /// B-231: the tests' own git reads no system settings, such as those
    /// Git for Windows keeps in its install folder. A test must not write
    /// there, so `GIT_CONFIG_SYSTEM`, set after [`isolated`], names a
    /// settings file in their place: its `init.templateDir` names a
    /// template whose `info/exclude` hides `*.tmp`.
    #[test]
    fn s3_b231_the_tests_git_reads_no_system_settings() {
        let tmp = tempfile::tempdir().unwrap();
        let config = settings(tmp.path());
        let system = tmp.path().join("system");
        let text = format!(
            "[init]\n\ttemplateDir = {}\n",
            quoted(&template(tmp.path()))
        );
        std::fs::write(&system, text).unwrap();
        let git = || {
            let mut c = isolated(Command::new("git"), &config);
            c.env("GIT_CONFIG_SYSTEM", &system);
            c
        };
        let repo = tmp.path().join("repo");
        repository(&git, &repo, &[], &[("Cargo.toml", "[package]\n")]);
        std::fs::write(repo.join("x.tmp"), "").unwrap();
        assert_eq!(git_in(&git, &repo, &STATUS), "?? x.tmp");
    }

    /// B-231: the tests' own git drops git's variables that this process
    /// has, as a test run by a hook or `git rebase -x` has them. A test
    /// cannot set its own process's variables without racing the tests
    /// beside it, so this one runs itself again, alone, as a child process
    /// that has them: `GIT_DIR` and `GIT_INDEX_FILE` naming a decoy
    /// repository, and `GIT_TEMPLATE_DIR` naming a template whose
    /// `info/exclude` hides `*.tmp`, and none of git's variables but these.
    /// The child makes a repository of its own with the tests' git, in the
    /// parent's folder; the decoy must be unchanged.
    #[test]
    fn s3_b231_the_tests_git_drops_the_variables_this_process_has() {
        const NAME: &str =
            "version::tests::s3_b231_the_tests_git_drops_the_variables_this_process_has";
        const CHILD: &str = "SM_B231_CHILD_DIR";
        let tmp = tempfile::tempdir().unwrap();
        let config = settings(tmp.path());
        let git = || isolated(Command::new("git"), &config);
        if let Some(dir) = std::env::var_os(CHILD) {
            let dir = PathBuf::from(dir);
            assert!(
                dir.is_absolute() && dir.is_dir(),
                "{CHILD} must name the parent's folder"
            );
            let own = dir.join("own");
            repository(&git, &own, &[], &[("g", "own\n")]);
            std::fs::write(own.join("x.tmp"), "").unwrap();
            assert_eq!(git_in(&git, &own, &STATUS), "?? x.tmp");
            return;
        }
        let decoy = tmp.path().join("decoy");
        repository(&git, &decoy, &[], &[("f", "decoy\n")]);
        let dot_git = decoy.join(".git");
        let before = state(&git, &decoy);
        let mut child = Command::new(std::env::current_exe().unwrap());
        for (name, _) in std::env::vars_os().filter(|(name, _)| is_gits(name)) {
            child.env_remove(name);
        }
        let out = child
            .args(["--exact", NAME])
            .env(CHILD, tmp.path())
            .env("GIT_DIR", &dot_git)
            .env("GIT_INDEX_FILE", dot_git.join("index"))
            .env("GIT_TEMPLATE_DIR", template(tmp.path()))
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(
            out.status.success() && stdout.contains("test result: ok. 1 passed"),
            "{stdout}{}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(tmp.path().join("own").join(".git").is_dir());
        assert_eq!(state(&git, &decoy), before);
    }

    /// B-231: the tests' own git reads no system attributes, such as those
    /// Git for Windows ships in `etc/gitattributes` in its install folder.
    /// No environment variable names another system attributes file, as
    /// `GIT_CONFIG_SYSTEM` does for the settings, and a test must not write
    /// under git's install folder, so this one reads git's own file in
    /// place. It fails when git names no file, as a git older than 2.42
    /// does, or one given `GIT_ATTR_NOSYSTEM`. It checks nothing when the
    /// file git names is missing, as on most Linux systems, or names no
    /// attribute for an extension of ASCII letters and digits.
    #[test]
    fn s3_b231_the_tests_git_reads_no_system_attributes() {
        let tmp = tempfile::tempdir().unwrap();
        let config = settings(tmp.path());
        let git = || isolated(Command::new("git"), &config);
        let plain = || {
            let mut c = git();
            c.env_remove("GIT_ATTR_NOSYSTEM");
            c
        };
        let repo = tmp.path().join("repo");
        repository(&git, &repo, &[], &[("Cargo.toml", "[package]\n")]);
        let out = plain()
            .arg("-C")
            .arg(&repo)
            .args(["var", "GIT_ATTR_SYSTEM"])
            .output()
            .expect("this test needs git on PATH");
        let system = String::from_utf8_lossy(&out.stdout).trim().to_owned();
        assert!(
            out.status.success() && !system.is_empty(),
            "git var GIT_ATTR_SYSTEM named no file ({}): {}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        );
        let text = match std::fs::read_to_string(&system) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                eprintln!("{system} does not exist; nothing to check");
                return;
            }
            Err(e) => panic!("{system}: {e}"),
        };
        // A file for each line that names an attribute for an extension of
        // ASCII letters and digits (`*.ext`).
        let files: Vec<String> = text
            .lines()
            .filter_map(|line| {
                let mut words = line.split_whitespace();
                let ext = words.next()?.strip_prefix("*.")?;
                let simple = !ext.is_empty() && ext.chars().all(|c| c.is_ascii_alphanumeric());
                (simple && words.next().is_some()).then(|| format!("a.{ext}"))
            })
            .collect();
        if files.is_empty() {
            eprintln!("{system} sets no attribute by extension; nothing to check");
            return;
        }
        let args: Vec<&str> = ["check-attr", "--all", "--"]
            .into_iter()
            .chain(files.iter().map(String::as_str))
            .collect();
        assert_ne!(git_in(&plain, &repo, &args), "", "{system} was not read");
        assert_eq!(git_in(&git, &repo, &args), "");
    }
}
