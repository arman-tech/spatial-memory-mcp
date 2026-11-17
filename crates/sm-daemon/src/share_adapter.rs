//! The production [`SharingOps`]: each operation calls the `sm-share`
//! function decision B-79 names, with the store, chunker, scanner, chunk
//! parameters and project the daemon resolved for the request.
//!
//! - `share` and `sync` work in the repository of the request's working
//!   directory. The daemon refuses a request outside one before the adapter
//!   is called; the adapter checks again and fails the same way.
//! - `sync`, `import` and `migrate` chunk what they store, so they need the
//!   model's tokenizer. While the model loads, or after it failed, they are
//!   refused with `daemon_loading` and nothing is written. `share` and
//!   `export` only read the store and work in either state.
//! - `migrate` passes the working directory's project identity, which
//!   `sm-share` uses for records without a project when no default project
//!   is given (SPEC 10).

use std::path::Path;

use sm_core::error::{CoreError, ErrorCode, Result};
use sm_core::protocol::{
    ExportParams, ExportReport, ImportParams, ImportReport, MigrateParams, ShareParams, ShareReply,
    SyncParams, SyncReply,
};
use sm_core::traits::Chunker;
use sm_core::types::Project;

use crate::sharing::{SharingContext, SharingOps};

/// Calls `sm-share`; see the module documentation.
#[derive(Debug, Default, Clone, Copy)]
pub struct ShareAdapter;

/// The chunker, or `daemon_loading` naming the operation.
fn chunker<'a>(ctx: &SharingContext<'a>, op: &str) -> Result<&'a dyn Chunker> {
    ctx.chunker.ok_or_else(|| {
        CoreError::new(
            ErrorCode::DaemonLoading,
            format!(
                "{op} chunks the memories it stores with the model's tokenizer, and the model is \
                 not loaded (it is still loading, or it failed to load); nothing was written"
            ),
        )
        .with_next_action(
            "Retry in a few seconds. If `spatial-memory status` shows the model failed, run \
             `spatial-memory doctor`.",
        )
    })
}

/// The resolved project and its repository root, or `not_git_repository`.
fn repo<'a>(ctx: &SharingContext<'a>, op: &str) -> Result<(&'a Project, &'a Path)> {
    let resolved = ctx.project.ok_or_else(|| {
        CoreError::new(
            ErrorCode::NotGitRepository,
            format!("{op} needs the project of the working directory, and none was resolved"),
        )
    })?;
    let repo = resolved.identity.repo.as_ref().ok_or_else(|| {
        CoreError::new(
            ErrorCode::NotGitRepository,
            format!(
                "{op} needs a git repository, and project {:?} is not in one",
                resolved.project.name
            ),
        )
    })?;
    Ok((resolved.project, repo.root.as_path()))
}

impl SharingOps for ShareAdapter {
    fn share(&self, ctx: &SharingContext<'_>, params: &ShareParams) -> Result<ShareReply> {
        let (project, root) = repo(ctx, "share")?;
        sm_share::share(ctx.store, ctx.scanner, project, root, params)
    }

    fn sync(&self, ctx: &SharingContext<'_>, params: &SyncParams) -> Result<SyncReply> {
        let (project, root) = repo(ctx, "sync")?;
        let chunker = chunker(ctx, "sync")?;
        sm_share::sync(ctx.store, chunker, &ctx.chunk_params, project, root, params)
    }

    fn migrate(&self, ctx: &SharingContext<'_>, params: &MigrateParams) -> Result<ImportReport> {
        let chunker = chunker(ctx, "migrate")?;
        sm_share::migrate(
            ctx.store,
            chunker,
            ctx.scanner,
            &ctx.chunk_params,
            params,
            ctx.project.map(|p| p.identity),
        )
    }

    fn import(&self, ctx: &SharingContext<'_>, params: &ImportParams) -> Result<ImportReport> {
        let chunker = chunker(ctx, "import")?;
        sm_share::import(ctx.store, chunker, ctx.scanner, &ctx.chunk_params, params)
    }

    fn export(&self, ctx: &SharingContext<'_>, params: &ExportParams) -> Result<ExportReport> {
        sm_share::export(ctx.store, params)
    }
}

#[cfg(test)]
mod tests {
    //! Each operation through a daemon started in the test process, with
    //! the fake embedder, over the real endpoint (a named pipe on Windows, a
    //! Unix socket elsewhere). Profiles and repositories are temporary
    //! folders; the git commands these tests run use an empty global
    //! configuration and no system one.

    use std::path::{Path, PathBuf};
    use std::process::Command;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    use sm_core::config::{ProfileName, Settings};
    use sm_core::error::{CoreError, ErrorCode, Result};
    use sm_core::ids::MemoryId;
    use sm_core::paths::{Endpoint, ProfilePaths};
    use sm_core::protocol::{
        DaemonState, ExportParams, ImportParams, MigrateParams, Op, RememberItem, RememberParams,
        Reply, Request, Response, ShareParams, SyncParams, read_message, write_frame,
    };
    use sm_core::testing::{FakeProvider, SimpleChunker, fake_info};
    use sm_core::traits::EmbedderInfo;

    use super::ShareAdapter;
    use crate::{DaemonHandle, DaemonOptions, LoadedModel, ModelLoader, Started};

    static COUNTER: AtomicUsize = AtomicUsize::new(0);

    fn unique(prefix: &str) -> String {
        format!(
            "{prefix}-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::SeqCst)
        )
    }

    /// The fake model, loaded at once, or a load that fails.
    struct Loader {
        fail: bool,
    }

    impl ModelLoader for Loader {
        fn info(&self) -> EmbedderInfo {
            fake_info()
        }
        fn load(&self) -> Result<LoadedModel> {
            if self.fail {
                return Err(CoreError::new(
                    ErrorCode::ModelMissing,
                    "the test model is missing",
                ));
            }
            Ok(LoadedModel {
                provider: Arc::new(FakeProvider::new()),
                chunker: Arc::new(SimpleChunker),
            })
        }
    }

    /// A temporary folder holding the profiles and the work folders.
    struct Space {
        tmp: tempfile::TempDir,
        git_config: PathBuf,
    }

    impl Space {
        fn new() -> Space {
            let tmp = tempfile::Builder::new()
                .prefix("smj")
                .tempdir()
                .expect("a temporary folder");
            let git_config = tmp.path().join("gitconfig");
            std::fs::write(&git_config, "").expect("an empty git config");
            Space { tmp, git_config }
        }

        fn path(&self) -> &Path {
            self.tmp.path()
        }

        /// A daemon on profile `name`, with the adapter.
        fn daemon(&self, name: &str, fail: bool) -> Daemon {
            let paths = ProfilePaths::new(
                self.path().join("root"),
                ProfileName::parse(name).expect("a profile name"),
            );
            let opts = DaemonOptions {
                endpoint: Some(self.endpoint(name)),
                lock_wait: Duration::from_millis(300),
                bind_wait: Duration::from_millis(300),
                retry_every: Duration::from_millis(10),
                bulk_grace: Duration::from_millis(200),
                bulk_error_backoff: Duration::from_millis(50),
                physical_cores: Some(8),
                shutdown_grace: Duration::from_secs(5),
                ..DaemonOptions::default()
            };
            let started = crate::start(
                paths.clone(),
                Settings::default(),
                Arc::new(Loader { fail }),
                Arc::new(ShareAdapter),
                opts,
            )
            .expect("the daemon starts");
            let Started::Running(handle) = started else {
                panic!("the lock was held");
            };
            let d = Daemon {
                endpoint: handle.endpoint().clone(),
                handle: Some(handle),
                profile: paths.profile.clone(),
            };
            let want = if fail {
                DaemonState::Failed
            } else {
                DaemonState::Ready
            };
            let start = Instant::now();
            while d.state() != want {
                assert!(
                    start.elapsed() < Duration::from_secs(10),
                    "the model reached {want:?}"
                );
                std::thread::sleep(Duration::from_millis(10));
            }
            d
        }

        #[cfg(windows)]
        fn endpoint(&self, _name: &str) -> Endpoint {
            Endpoint::NamedPipe(unique("sm-share-adapter"))
        }

        #[cfg(not(windows))]
        fn endpoint(&self, name: &str) -> Endpoint {
            Endpoint::UnixSocket(self.path().join(format!("{name}.sock")))
        }

        /// A git repository with no commits, on branch `main`.
        fn repo(&self, name: &str) -> String {
            let dir = self.path().join("work").join(name);
            std::fs::create_dir_all(&dir).expect("create a repository folder");
            self.git(&dir, &["init", "--quiet", "--initial-branch=main"]);
            dir.to_str().expect("a UTF-8 temporary path").to_owned()
        }

        /// Runs git isolated from the user's and the system's configuration.
        fn git(&self, dir: &Path, args: &[&str]) {
            let mut cmd = Command::new("git");
            cmd.args(args).current_dir(dir);
            for var in [
                "GIT_DIR",
                "GIT_WORK_TREE",
                "GIT_COMMON_DIR",
                "GIT_INDEX_FILE",
            ] {
                cmd.env_remove(var);
            }
            let out = cmd
                .env("GIT_CONFIG_GLOBAL", &self.git_config)
                .env("GIT_CONFIG_NOSYSTEM", "1")
                .env("GIT_TERMINAL_PROMPT", "0")
                .env("GIT_AUTHOR_NAME", "Test Author")
                .env("GIT_AUTHOR_EMAIL", "author@example.invalid")
                .env("GIT_COMMITTER_NAME", "Test Author")
                .env("GIT_COMMITTER_EMAIL", "author@example.invalid")
                .output()
                .expect("run git");
            assert!(
                out.status.success(),
                "git {args:?} failed: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        }

        fn commit_all(&self, repo: &str) {
            let dir = Path::new(repo);
            self.git(dir, &["add", "-A"]);
            self.git(dir, &["commit", "--quiet", "-m", "share a memory"]);
        }

        /// An absolute path in the temporary folder that does not exist yet.
        fn file(&self, name: &str) -> String {
            self.path()
                .join(name)
                .to_str()
                .expect("a UTF-8 temporary path")
                .to_owned()
        }
    }

    /// A running daemon; dropping it shuts it down.
    struct Daemon {
        handle: Option<DaemonHandle>,
        endpoint: Endpoint,
        profile: ProfileName,
    }

    impl Drop for Daemon {
        fn drop(&mut self) {
            if let Some(h) = self.handle.take() {
                h.shutdown("test finished");
            }
        }
    }

    impl Daemon {
        fn call(&self, cwd: &str, op: Op) -> Response {
            let mut stream =
                crate::connect(&self.endpoint, Duration::from_secs(5)).expect("connect");
            write_frame(&mut stream, &Request::new(cwd, self.profile.clone(), op))
                .expect("write a request");
            read_message(&mut stream)
                .expect("read a reply")
                .expect("a reply")
        }

        fn ok(&self, cwd: &str, op: Op) -> Reply {
            match self.call(cwd, op).into_result() {
                Ok(r) => r,
                Err(e) => panic!("{}: {} ({})", e.code.as_str(), e.message, e.next_action),
            }
        }

        fn err(&self, cwd: &str, op: Op) -> CoreError {
            match self.call(cwd, op).into_result() {
                Ok(r) => panic!("expected an error, got {r:?}"),
                Err(e) => e,
            }
        }

        fn state(&self) -> DaemonState {
            match self.ok("", Op::Hello) {
                Reply::Hello(i) => i.state,
                other => panic!("hello replied {other:?}"),
            }
        }

        fn remember(&self, cwd: &str, content: &str) -> MemoryId {
            match self.ok(
                cwd,
                Op::Remember(RememberParams {
                    memories: vec![RememberItem::new(content)],
                }),
            ) {
                Reply::Remember(r) => r.memories[0].id,
                other => panic!("remember replied {other:?}"),
            }
        }
    }

    #[test]
    fn s9_share_through_the_daemon_writes_the_memory_into_the_repository() {
        let space = Space::new();
        let d = space.daemon("a", false);
        let repo = space.repo("orders");
        let id = d.remember(&repo, "Deploys go out on Tuesdays after the change review.");

        let reply = match d.ok(&repo, Op::Share(ShareParams { ids: vec![id] })) {
            Reply::Share(r) => r,
            other => panic!("share replied {other:?}"),
        };

        assert!(reply.refused.is_empty(), "{:?}", reply.refused);
        assert_eq!(reply.written.len(), 1);
        assert_eq!(reply.written[0].id, id);
        let rel = &reply.written[0].path;
        assert!(rel.starts_with(".spatial-memory/shared/"), "{rel}");
        let text = std::fs::read_to_string(Path::new(&repo).join(rel)).expect("the shared file");
        assert!(text.contains("Deploys go out on Tuesdays"), "{text}");
        assert!(text.contains(&id.to_string()), "the file names the memory");
    }

    #[test]
    fn s9_sync_through_the_daemon_adds_a_committed_shared_memory() {
        let space = Space::new();
        let repo = space.repo("billing");
        let id = {
            let author = space.daemon("author", false);
            let id = author.remember(&repo, "Invoices are numbered per tenant, never globally.");
            match author.ok(&repo, Op::Share(ShareParams { ids: vec![id] })) {
                Reply::Share(r) => assert_eq!(r.written.len(), 1, "{r:?}"),
                other => panic!("share replied {other:?}"),
            }
            id
        };
        space.commit_all(&repo);

        let reader = space.daemon("reader", false);
        let dry = match reader.ok(&repo, Op::Sync(SyncParams { dry_run: true })) {
            Reply::Sync(r) => r,
            other => panic!("sync replied {other:?}"),
        };
        assert!(dry.dry_run);
        assert_eq!(dry.added, vec![id], "{dry:?}");

        let reply = match reader.ok(&repo, Op::Sync(SyncParams { dry_run: false })) {
            Reply::Sync(r) => r,
            other => panic!("sync replied {other:?}"),
        };
        assert!(!reply.dry_run);
        assert_eq!(reply.added, vec![id], "{reply:?}");
        assert!(reply.errors.is_empty(), "{:?}", reply.errors);

        let again = match reader.ok(&repo, Op::Sync(SyncParams { dry_run: false })) {
            Reply::Sync(r) => r,
            other => panic!("sync replied {other:?}"),
        };
        assert!(again.added.is_empty(), "{again:?}");
        assert_eq!(again.unchanged, 1, "{again:?}");
    }

    #[test]
    fn s9_export_then_import_through_two_daemons_carries_every_memory() {
        let space = Space::new();
        let repo = space.repo("search");
        let file = space.file("export.json");
        {
            let source = space.daemon("source", false);
            source.remember(&repo, "The search index is rebuilt nightly at two.");
            source.remember(&repo, "Synonyms live in the search team's wiki page.");
            let report = match source.ok(
                &repo,
                Op::Export(ExportParams {
                    path: file.clone(),
                    project: None,
                }),
            ) {
                Reply::Export(r) => r,
                other => panic!("export replied {other:?}"),
            };
            assert_eq!(report.memories, 2, "{report:?}");
            assert_eq!(report.projects, 1, "{report:?}");
            assert!(Path::new(&file).is_file(), "the export file was written");

            let e = source.err(
                &repo,
                Op::Export(ExportParams {
                    path: file.clone(),
                    project: None,
                }),
            );
            assert_eq!(e.code, ErrorCode::InvalidInput, "{}", e.message);
            assert!(e.message.contains("already exists"), "{}", e.message);
            assert!(Path::new(&file).is_file(), "the export file is kept");
        }

        let target = space.daemon("target", false);
        let report = match target.ok(&repo, Op::Import(ImportParams { path: file.clone() })) {
            Reply::Import(r) => r,
            other => panic!("import replied {other:?}"),
        };
        assert_eq!(report.read, 2, "{report:?}");
        assert_eq!(report.inserted, 2, "{report:?}");
        assert!(report.issues.is_empty(), "{:?}", report.issues);
        assert!(report.pending_embeddings > 0, "{report:?}");

        let again = match target.ok(&repo, Op::Import(ImportParams { path: file })) {
            Reply::Import(r) => r,
            other => panic!("import replied {other:?}"),
        };
        assert_eq!(again.inserted, 0, "{again:?}");
        assert_eq!(again.skipped_existing, 2, "{again:?}");
    }

    #[test]
    fn s10_migrate_through_the_daemon_stores_the_version_1_sample() {
        let space = Space::new();
        let d = space.daemon("a", false);
        let cwd = space.repo("here");
        let from = Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/migration/v1_export_sample.json"
        ))
        .to_str()
        .expect("a UTF-8 path")
        .to_owned();

        let report = match d.ok(
            &cwd,
            Op::Migrate(MigrateParams {
                from: from.clone(),
                default_project: None,
            }),
        ) {
            Reply::Import(r) => r,
            other => panic!("migrate replied {other:?}"),
        };

        assert_eq!(report.source, from);
        assert_eq!(report.read, 10, "{report:?}");
        assert_eq!(report.inserted, 10, "{report:?}");
        assert!(report.issues.is_empty(), "{:?}", report.issues);
        assert!(!report.projects_created.is_empty(), "{report:?}");
        assert!(report.pending_embeddings > 0, "{report:?}");
    }

    #[test]
    fn s9_without_the_model_the_chunking_ops_refuse_and_share_and_export_still_work() {
        let space = Space::new();
        let d = space.daemon("a", true);
        let repo = space.repo("orders");
        // `remember` needs the model too, so share is given an id the store
        // does not hold: sm-share refuses it in the reply, which is only
        // reached through the adapter.
        let id = MemoryId::new_v7();

        for (name, op) in [
            ("sync", Op::Sync(SyncParams { dry_run: false })),
            (
                "import",
                Op::Import(ImportParams {
                    path: space.file("missing.json"),
                }),
            ),
            (
                "migrate",
                Op::Migrate(MigrateParams {
                    from: space.file("missing-v1.json"),
                    default_project: None,
                }),
            ),
        ] {
            let e = d.err(&repo, op);
            assert_eq!(e.code, ErrorCode::DaemonLoading, "{name}: {}", e.message);
            assert!(e.message.starts_with(name), "{}", e.message);
            assert!(e.next_action.contains("spatial-memory status"), "{e:?}");
        }

        match d.ok(&repo, Op::Share(ShareParams { ids: vec![id] })) {
            Reply::Share(r) => {
                assert!(r.written.is_empty(), "{r:?}");
                assert_eq!(r.refused.len(), 1, "{r:?}");
                assert_eq!(r.refused[0].id, id);
                assert_eq!(r.refused[0].error.code, ErrorCode::NotFound, "{r:?}");
            }
            other => panic!("share replied {other:?}"),
        }
        let file = space.file("export.json");
        match d.ok(
            &repo,
            Op::Export(ExportParams {
                path: file.clone(),
                project: None,
            }),
        ) {
            Reply::Export(r) => assert_eq!(r.memories, 0, "{r:?}"),
            other => panic!("export replied {other:?}"),
        }
        assert!(Path::new(&file).is_file(), "the export file was written");
    }
}
