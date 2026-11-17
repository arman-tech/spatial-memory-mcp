//! `share` (SPEC section 9, Team sharing rules 1 and 2 and "Security";
//! AT-9; decision B-78).
//!
//! `share` writes memories of the current project into the repository's
//! working tree, one file each under `.spatial-memory/shared/`. It never
//! commits or pushes: the author reviews the files and opens a pull request.
//!
//! # Redaction
//!
//! The title and body are redacted with the secret scanner first and the
//! kinds replaced are listed in [`SharedFile::redactions`]. Then the whole
//! rendered file, front matter included, is scanned again. Anything still
//! found refuses the memory with `secret_detected` and no file is written:
//! a credential in a tag or an anchor cannot be redacted without changing
//! what the memory is attached to, so the author must fix it.
//!
//! # The memory is not changed
//!
//! `share` writes files only; the memory stays `local`. After the pull
//! request merges, `sync` adopts it (B-78): the stored memory becomes
//! `shared` when its text hash equals the file's. When redaction changed the
//! title or body, the hashes differ, and sync reports the file and leaves
//! the local copy alone; the author then forgets the local copy (sync adds
//! the shared one on the next run) or revises it to match.
//!
//! # Files
//!
//! - The path is [`crate::format::shared_path`]. Sharing a memory again
//!   replaces its earlier file, found by the `--<last 8 of id>.md` suffix
//!   and confirmed by the id inside it, even when the type or title (and so
//!   the folder or slug) changed. A file with the suffix that does not
//!   parse, or holds another id, is left alone.
//! - A file already at the target path that does not hold this memory is
//!   never overwritten: the memory is refused with `invalid_input`.
//! - `.spatial-memory/shared/.gitattributes` (`* text eol=lf`) is written
//!   when missing, before the first file, and an existing one is left alone.
//! - Each file is written to a temporary file in the same folder and renamed
//!   into place. Symbolic links in the shared folder are never followed.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use sm_core::ids::MemoryId;
use sm_core::protocol::{Refusal, ShareParams, ShareReply, SharedFile};
use sm_core::traits::SecretScanner;
use sm_core::types::{Memory, Origin, Project};
use sm_core::{CoreError, ErrorCode, Result};
use sm_store::Store;

use crate::format::{
    GITATTRIBUTES, SHARED_DIR, SharedDoc, clean_title, name_suffix, normalize_body,
};

/// Writes each listed memory of `project` into the working tree at
/// `repo_root`; see the module documentation. Per-memory problems are
/// refusals in the reply; a store failure, a `repo_root` that is not a git
/// repository, or a shared folder that is a link fails the call.
pub fn share(
    store: &Store,
    scanner: &dyn SecretScanner,
    project: &Project,
    repo_root: &Path,
    params: &ShareParams,
) -> Result<ShareReply> {
    params.validate()?;
    require_git_entry(repo_root)?;
    let shared_dir = repo_root.join(SHARED_DIR);
    check_not_link(&repo_root.join(".spatial-memory"))?;
    check_not_link(&shared_dir)?;
    let mut by_suffix = files_by_suffix(&shared_dir)?;

    let mut ids: Vec<MemoryId> = Vec::with_capacity(params.ids.len());
    for id in &params.ids {
        if !ids.contains(id) {
            ids.push(*id);
        }
    }

    let mut reply = ShareReply {
        written: Vec::new(),
        refused: Vec::new(),
    };
    for id in ids {
        let memory = match store.find_memory(&id)? {
            Some(m) if m.project_id == project.id => m,
            _ => {
                reply.refused.push(Refusal {
                    id,
                    error: CoreError::not_found(format!(
                        "memory {id} is not in project {:?}",
                        project.name
                    )),
                });
                continue;
            }
        };
        match share_one(scanner, repo_root, &shared_dir, &memory, &mut by_suffix) {
            Ok(file) => reply.written.push(file),
            Err(error) => reply.refused.push(Refusal { id, error }),
        }
    }
    Ok(reply)
}

/// `.git` must exist at the root: a folder, or a file in a linked worktree
/// or submodule.
pub(crate) fn require_git_entry(repo_root: &Path) -> Result<()> {
    match fs::symlink_metadata(repo_root.join(".git")) {
        Ok(_) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Err(CoreError::new(
            ErrorCode::NotGitRepository,
            format!(
                "{} is not the root of a git repository (it has no .git)",
                repo_root.display()
            ),
        )
        .with_next_action("Run this from inside a git repository.")),
        Err(e) => Err(CoreError::storage(format!(
            "could not check {} for a .git entry: {e}",
            repo_root.display()
        ))),
    }
}

/// Refuses a path that exists as a symbolic link or as something other
/// than a folder, so writes stay inside the repository.
fn check_not_link(dir: &Path) -> Result<()> {
    match fs::symlink_metadata(dir) {
        Ok(m) if m.file_type().is_symlink() || !m.is_dir() => {
            Err(CoreError::invalid_input(format!(
                "{} is a link or not a folder; share writes only into a plain folder",
                dir.display()
            ))
            .with_next_action("Replace it with a plain folder, then share again."))
        }
        Ok(_) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(CoreError::storage(format!(
            "could not read {}: {e}",
            dir.display()
        ))),
    }
}

/// Every regular file under the shared folder whose name ends in
/// `--<8 hex>.md`, by that suffix. Links are not followed.
fn files_by_suffix(shared_dir: &Path) -> Result<HashMap<String, Vec<PathBuf>>> {
    let mut out: HashMap<String, Vec<PathBuf>> = HashMap::new();
    let mut stack = vec![shared_dir.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = match fs::read_dir(&dir) {
            Ok(e) => e,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => {
                return Err(CoreError::storage(format!(
                    "could not list {}: {e}",
                    dir.display()
                )));
            }
        };
        for entry in entries {
            let entry = entry?;
            let ft = entry.file_type()?;
            let path = entry.path();
            if ft.is_dir() {
                stack.push(path);
            } else if ft.is_file()
                && let Some(suffix) = entry.file_name().to_str().and_then(name_suffix)
            {
                out.entry(suffix.to_owned()).or_default().push(path);
            }
        }
    }
    for paths in out.values_mut() {
        paths.sort();
    }
    Ok(out)
}

fn share_one(
    scanner: &dyn SecretScanner,
    repo_root: &Path,
    shared_dir: &Path,
    m: &Memory,
    by_suffix: &mut HashMap<String, Vec<PathBuf>>,
) -> Result<SharedFile> {
    if m.origin == Origin::Shared {
        let where_ = m
            .provenance
            .file_path
            .as_deref()
            .map_or_else(String::new, |p| format!(" in {p}"));
        return Err(CoreError::new(
            ErrorCode::SharedReadOnly,
            format!("memory {} is already shared{where_}", m.id),
        )
        .with_next_action(
            "Edit its file in the repository's .spatial-memory/shared folder and commit it; \
             sync applies the change.",
        ));
    }

    let content = scanner.redact(&m.content);
    let title = m.title.as_deref().map(|t| scanner.redact(t));
    let mut redactions: Vec<String> = Vec::new();
    for kind in title
        .iter()
        .flat_map(|r| r.kinds.iter())
        .chain(content.kinds.iter())
    {
        if !redactions.contains(kind) {
            redactions.push(kind.clone());
        }
    }
    let mut doc = SharedDoc::from_memory(m);
    doc.title = clean_title(title.as_ref().map(|r| r.text.as_str()), &content.text);
    doc.body = normalize_body(&content.text);
    let rendered = doc.render();

    let found = scanner.scan(&rendered);
    if !found.is_empty() {
        let front_end = front_matter_len(&rendered);
        let mut kinds: Vec<&str> = Vec::new();
        for f in &found {
            if !kinds.contains(&f.kind.as_str()) {
                kinds.push(&f.kind);
            }
        }
        let in_front = found.iter().any(|f| f.start < front_end);
        let place = if in_front {
            "in its front matter (tags, anchors or another key), which is not redacted"
        } else {
            "in its title or body after redaction"
        };
        return Err(CoreError::new(
            ErrorCode::SecretDetected,
            format!(
                "memory {}'s shared file still holds a possible secret ({}) {place}; \
                 no file was written",
                m.id,
                kinds.join(", ")
            ),
        )
        .with_next_action(
            "Remove the credential from the memory (revise its tags, anchors or text), \
             then share again.",
        ));
    }

    // The file must read back as the same document, or sync would see
    // something else.
    match SharedDoc::parse(&rendered) {
        Ok(back) if back == doc => {}
        Ok(_) => {
            return Err(CoreError::internal(format!(
                "memory {}'s shared file does not read back as written",
                m.id
            )));
        }
        Err(e) => {
            return Err(CoreError::internal(format!(
                "memory {}'s shared file does not parse back: {e}",
                m.id
            )));
        }
    }

    let rel = doc.path();
    let target = repo_root.join(&rel);
    check_target(&target, &rel, &m.id)?;

    ensure_dir(shared_dir)?;
    write_gitattributes(shared_dir)?;
    let type_dir = shared_dir.join(doc.memory_type.as_str());
    ensure_dir(&type_dir)?;
    write_atomically(&target, rendered.as_bytes())?;

    // Remove the memory's earlier file, when its type or title changed.
    let suffix = m.id.last8();
    let earlier = by_suffix.remove(&suffix).unwrap_or_default();
    let mut kept = Vec::new();
    for old in earlier {
        if same_rel_path(&old, &target) || !holds_id(&old, &m.id) {
            kept.push(old);
            continue;
        }
        if let Err(e) = fs::remove_file(&old) {
            return Err(CoreError::storage(format!(
                "wrote {rel} but could not remove memory {}'s earlier file {}: {e}",
                m.id,
                old.display()
            ))
            .with_next_action("Delete the earlier file by hand before committing."));
        }
    }
    kept.push(target);
    by_suffix.insert(suffix, kept);

    Ok(SharedFile {
        id: m.id,
        path: rel,
        redactions,
    })
}

/// The byte length of the front matter, fences included.
fn front_matter_len(rendered: &str) -> usize {
    rendered[4..]
        .find("\n---\n")
        .map_or(rendered.len(), |p| 4 + p + 5)
}

/// A file at the target path must hold this memory; anything else there is
/// refused rather than overwritten.
fn check_target(target: &Path, rel: &str, id: &MemoryId) -> Result<()> {
    let meta = match fs::symlink_metadata(target) {
        Ok(m) => m,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => {
            return Err(CoreError::storage(format!("could not read {rel}: {e}")));
        }
    };
    let refuse = |why: String| {
        Err(CoreError::invalid_input(format!(
            "{rel} already exists and {why}; it was not overwritten"
        ))
        .with_next_action("Move or delete that file, then share again."))
    };
    if !meta.is_file() || meta.file_type().is_symlink() {
        return refuse("is not a regular file".to_owned());
    }
    let text = fs::read(target)?;
    let text = String::from_utf8_lossy(&text);
    match SharedDoc::parse(&text) {
        Ok(doc) if doc.id == *id => Ok(()),
        Ok(doc) => refuse(format!("holds memory {}", doc.id)),
        Err(e) => refuse(format!("is not a shared file this build can read ({e})")),
    }
}

fn holds_id(path: &Path, id: &MemoryId) -> bool {
    fs::read(path)
        .ok()
        .and_then(|b| String::from_utf8(b).ok())
        .and_then(|t| SharedDoc::parse(&t).ok())
        .is_some_and(|d| d.id == *id)
}

/// Paths equal apart from ASCII case are the same file on Windows and
/// macOS; they are never removed as an "earlier" file.
fn same_rel_path(a: &Path, b: &Path) -> bool {
    a.to_string_lossy()
        .replace('\\', "/")
        .eq_ignore_ascii_case(&b.to_string_lossy().replace('\\', "/"))
}

fn ensure_dir(dir: &Path) -> Result<()> {
    fs::create_dir_all(dir)
        .map_err(|e| CoreError::storage(format!("could not create {}: {e}", dir.display())))?;
    check_not_link(dir)
}

/// Writes `.gitattributes` when the shared folder has none.
fn write_gitattributes(shared_dir: &Path) -> Result<()> {
    let path = shared_dir.join(".gitattributes");
    match fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
    {
        Ok(mut f) => {
            f.write_all(GITATTRIBUTES.as_bytes())?;
            f.sync_all()?;
            Ok(())
        }
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
        Err(e) => Err(CoreError::storage(format!(
            "could not write {}: {e}",
            path.display()
        ))),
    }
}

/// Writes a temporary file beside `target` and renames it into place.
pub(crate) fn write_atomically(target: &Path, bytes: &[u8]) -> Result<()> {
    let dir = target
        .parent()
        .ok_or_else(|| CoreError::internal("a file path has no folder"))?;
    let name = target
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let tmp = dir.join(format!(".{name}.tmp-{}-{nanos}", std::process::id()));
    let result = (|| -> std::io::Result<()> {
        let mut f = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
        drop(f);
        fs::rename(&tmp, target)
    })();
    if let Err(e) = result {
        let _ = fs::remove_file(&tmp);
        return Err(CoreError::storage(format!(
            "could not write {}: {e}",
            target.display()
        )));
    }
    Ok(())
}
