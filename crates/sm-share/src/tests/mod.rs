//! Sharing and migration tests (SPEC sections 9 and 10; AT-8, AT-9;
//! decisions B-77, B-78 and B-79).
//!
//! Every test opens its own store in a temporary folder, chunks with the
//! fake `SimpleChunker`, embeds with the fake embedder and scans with the
//! real `RegexSecretScanner` delivered by stream G (B-8). Repositories are
//! temporary folders driven by the git command line, isolated from the
//! user's configuration: `GIT_CONFIG_GLOBAL` names an empty file,
//! `GIT_CONFIG_NOSYSTEM=1`, and every commit sets its author and committer.

mod export_import;
mod format;
mod migrate;
mod share;
mod sync;

use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use sm_capture::RegexSecretScanner;
use sm_core::EMBED_DIM;
use sm_core::error::{CoreError, ErrorCode};
use sm_core::ids::MemoryId;
use sm_core::project::ProjectIdentity;
use sm_core::testing::{FAKE_MODEL_ID, FAKE_REVISION, FakeEmbedder, SimpleChunker};
use sm_core::traits::{ChunkParams, Chunker, DocInput, Embedder};
use sm_core::types::{IndexStamp, Memory, NewMemory, Project, Provenance, ProvenanceSource};
use sm_store::{Access, MemoryInput, MemoryListFilter, NewChunk, Store};
use tempfile::TempDir;

/// A store in its own temporary folder.
///
/// Fields drop in declaration order: the store closes its connections
/// before the folder is removed (Windows refuses to delete an open file).
pub(crate) struct Fixture {
    pub(crate) store: Store,
    pub(crate) dir: TempDir,
}

impl Fixture {
    /// A path in the fixture's folder.
    pub(crate) fn path(&self, name: &str) -> PathBuf {
        self.dir.path().join(name)
    }
}

/// The fake embedder's stamp with the SPEC 15 chunk defaults.
pub(crate) fn stamp() -> IndexStamp {
    IndexStamp {
        model_id: FAKE_MODEL_ID.to_owned(),
        revision: FAKE_REVISION.to_owned(),
        dim: EMBED_DIM,
        chunk_max_tokens: 256,
        chunk_overlap: 0.125,
    }
}

pub(crate) fn fixture() -> Fixture {
    let dir = tempfile::tempdir().expect("temporary folder");
    let store = Store::open_at(&dir.path().join("memory.db"), &dir.path().join("backups"))
        .expect("open the store");
    store.configure_index(stamp()).expect("configure the index");
    Fixture { store, dir }
}

/// The SPEC 15 chunk defaults.
pub(crate) fn chunk_params() -> ChunkParams {
    ChunkParams {
        max_tokens: 256,
        overlap: 0.125,
        index_cap: None,
    }
}

pub(crate) fn chunker() -> SimpleChunker {
    SimpleChunker
}

pub(crate) fn scanner() -> RegexSecretScanner {
    RegexSecretScanner::new()
}

/// A synthetic project identity (no repository, no remote).
pub(crate) fn identity(name: &str) -> ProjectIdentity {
    ProjectIdentity {
        root: PathBuf::from(format!("/work/{name}")),
        root_path: format!("/work/{name}"),
        root_key: format!("/work/{name}"),
        git_remote: None,
        repo: None,
        name: name.to_owned(),
        capture_enabled: true,
        warnings: Vec::new(),
    }
}

pub(crate) fn project(store: &Store, name: &str) -> Project {
    store.upsert_project(&identity(name)).expect("project")
}

/// A local note of `project` with pending chunks from `SimpleChunker`.
pub(crate) fn note_input(project: &Project, content: &str) -> MemoryInput {
    MemoryInput {
        memory: NewMemory::new(
            project.id,
            content,
            Provenance::new(ProvenanceSource::Agent),
        ),
        chunks: pending_chunks(content),
    }
}

pub(crate) fn pending_chunks(content: &str) -> Vec<NewChunk> {
    chunker()
        .chunk(content, &chunk_params())
        .expect("chunk")
        .pieces
        .into_iter()
        .map(NewChunk::pending)
        .collect()
}

pub(crate) fn insert(store: &Store, input: MemoryInput) -> MemoryId {
    store
        .insert_memory(input, Access::Local)
        .expect("insert a memory")
        .id
}

pub(crate) fn add_note(store: &Store, project: &Project, content: &str) -> MemoryId {
    insert(store, note_input(project, content))
}

/// Every memory, every project and status, by id.
pub(crate) fn all_memories(store: &Store) -> Vec<Memory> {
    store
        .list_memories(&MemoryListFilter::default())
        .expect("list memories")
}

/// Embeds every pending chunk with the fake embedder, as the daemon's queue
/// would, and returns how many chunks were embedded.
pub(crate) fn embed_all_pending(store: &Store) -> u64 {
    let embedder = FakeEmbedder::new();
    let mut total = 0;
    loop {
        let pending = store.pending_chunks(500).expect("pending chunks");
        if pending.is_empty() {
            return total;
        }
        let docs: Vec<DocInput<'_>> = pending
            .iter()
            .map(|c| DocInput {
                title: c.title.as_deref(),
                text: &c.text,
            })
            .collect();
        let vectors = embedder.embed_documents(&docs).expect("embed");
        let items: Vec<_> = pending.iter().map(|c| c.id).zip(vectors).collect();
        total += store
            .set_embeddings(FAKE_MODEL_ID, &items)
            .expect("set embeddings");
    }
}

#[track_caller]
pub(crate) fn assert_code<T: std::fmt::Debug>(
    r: Result<T, CoreError>,
    code: ErrorCode,
) -> CoreError {
    match r {
        Ok(v) => panic!("expected {code:?}, got Ok({v:?})"),
        Err(e) => {
            assert_eq!(e.code, code, "unexpected error: {e:?}");
            e
        }
    }
}

/// A fixture file under the repository's `fixtures/` folder.
pub(crate) fn fixture_path(rel: &str) -> PathBuf {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures")).join(rel)
}

pub(crate) fn read_fixture(rel: &str) -> String {
    fs::read_to_string(fixture_path(rel)).expect("read a fixture as UTF-8")
}

// ---------------------------------------------------------------------------
// Git repositories
// ---------------------------------------------------------------------------

/// A temporary git repository, isolated from the user's git configuration.
///
/// `config` holds the empty global configuration file, outside the work
/// tree so `git add -A` never sees it.
pub(crate) struct Repo {
    pub(crate) dir: TempDir,
    config: TempDir,
}

impl Repo {
    /// `git init` in a new temporary folder, on branch `main`.
    pub(crate) fn init() -> Repo {
        let dir = tempfile::tempdir().expect("temporary folder");
        let config = tempfile::tempdir().expect("temporary folder");
        fs::write(config.path().join("gitconfig"), "").expect("write an empty git config");
        let repo = Repo { dir, config };
        repo.git(&["init", "--quiet", "--initial-branch=main"]);
        repo
    }

    pub(crate) fn root(&self) -> &Path {
        self.dir.path()
    }

    fn config_file(&self) -> PathBuf {
        self.config.path().join("gitconfig")
    }

    /// The environment every git process of a test gets, sync's included.
    pub(crate) fn git_env(&self) -> Vec<(OsString, OsString)> {
        vec![
            (
                OsString::from("GIT_CONFIG_GLOBAL"),
                self.config_file().into_os_string(),
            ),
            (OsString::from("GIT_CONFIG_NOSYSTEM"), OsString::from("1")),
        ]
    }

    /// Runs git in the repository and returns its standard output; panics
    /// when git fails.
    pub(crate) fn git(&self, args: &[&str]) -> String {
        let mut cmd = Command::new("git");
        cmd.args(args).current_dir(self.root());
        for var in [
            "GIT_DIR",
            "GIT_WORK_TREE",
            "GIT_COMMON_DIR",
            "GIT_INDEX_FILE",
            "GIT_OBJECT_DIRECTORY",
            "GIT_ALTERNATE_OBJECT_DIRECTORIES",
            "GIT_NAMESPACE",
        ] {
            cmd.env_remove(var);
        }
        for (k, v) in self.git_env() {
            cmd.env(k, v);
        }
        cmd.env("GIT_TERMINAL_PROMPT", "0")
            .env("GIT_AUTHOR_NAME", "Test Author")
            .env("GIT_AUTHOR_EMAIL", "author@example.invalid")
            .env("GIT_COMMITTER_NAME", "Test Author")
            .env("GIT_COMMITTER_EMAIL", "author@example.invalid");
        let out = cmd.output().expect("run git");
        assert!(
            out.status.success(),
            "git {args:?} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8(out.stdout).expect("git output is UTF-8")
    }

    /// Writes a file in the work tree, creating its folders.
    pub(crate) fn write(&self, rel: &str, text: &str) {
        let path = self.root().join(rel);
        fs::create_dir_all(path.parent().expect("a folder")).expect("create folders");
        fs::write(path, text).expect("write a file");
    }

    pub(crate) fn read(&self, rel: &str) -> String {
        fs::read_to_string(self.root().join(rel)).expect("read a file")
    }

    pub(crate) fn exists(&self, rel: &str) -> bool {
        self.root().join(rel).exists()
    }

    pub(crate) fn remove(&self, rel: &str) {
        fs::remove_file(self.root().join(rel)).expect("remove a file this test wrote");
    }

    /// Stages everything and commits it with the test author; returns the
    /// commit id.
    pub(crate) fn commit_all(&self, message: &str) -> String {
        self.git(&["add", "-A"]);
        self.git(&["commit", "--quiet", "--allow-empty", "-m", message]);
        self.git(&["rev-parse", "HEAD"]).trim().to_owned()
    }

    /// The files under `prefix` in the work tree, relative to the root,
    /// with `/` separators, sorted.
    pub(crate) fn files_under(&self, prefix: &str) -> Vec<String> {
        let mut out = Vec::new();
        let mut stack = vec![self.root().join(prefix)];
        while let Some(dir) = stack.pop() {
            let Ok(entries) = fs::read_dir(&dir) else {
                continue;
            };
            for e in entries {
                let e = e.expect("a folder entry");
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                } else {
                    let rel = p
                        .strip_prefix(self.root())
                        .expect("under the root")
                        .to_string_lossy()
                        .replace('\\', "/");
                    out.push(rel);
                }
            }
        }
        out.sort();
        out
    }
}
