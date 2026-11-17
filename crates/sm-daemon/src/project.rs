//! Project resolution (SPEC 9, Projects): `resolve_project` on the
//! request's working directory, then upserted. Reads default to the
//! current project; `scope: all` is explicit.
//!
//! The identity is resolved afresh on every use (e-report decision 30): it
//! reads only a few files (the folder walk to `.git`, `project.toml`, the
//! git configuration) and never runs git. The cache, keyed by the working
//! directory, keeps the stored project for an identity, so the database is
//! written only when the identity changes. So a `capture.enabled = false`
//! added to `project.toml`, or a `git init` in a folder that was not a
//! repository, counts from the next request, with no restart.

use std::path::Path;

use sm_core::error::Result;
use sm_core::project::resolve_project;
use sm_core::traits::QueryScope;
use sm_core::types::Scope;

use crate::state::{ProjectEntry, Shared, lock};

pub(crate) fn resolve(sh: &Shared, cwd: &str) -> Result<ProjectEntry> {
    let identity = resolve_project(Path::new(cwd))?;
    let key = cwd.to_owned();
    if let Some(e) = lock(&sh.projects).get(&key)
        && e.identity == identity
    {
        return Ok(e.clone());
    }
    let project = sh.store.upsert_project(&identity)?;
    let entry = ProjectEntry { project, identity };
    lock(&sh.projects).insert(key, entry.clone());
    Ok(entry)
}

/// The stored project for `cwd`, or `None` when the store has none for its
/// identity. Nothing is written: this is for requests that only report the
/// project, such as `status`, which `doctor` sends (B-151). The cache is
/// neither read for a miss nor filled, so the next `resolve` still
/// refreshes a moved root path.
pub(crate) fn lookup(sh: &Shared, cwd: &str) -> Result<Option<ProjectEntry>> {
    let identity = resolve_project(Path::new(cwd))?;
    if let Some(e) = lock(&sh.projects).get(&cwd.to_owned())
        && e.identity == identity
    {
        return Ok(Some(e.clone()));
    }
    Ok(sh
        .store
        .project_by_identity(&identity.key())?
        .map(|project| ProjectEntry { project, identity }))
}

/// The search scope for a read. `scope: all` needs no project; the default
/// scope is the current project and fails when it cannot be resolved.
pub(crate) fn read_scope(
    sh: &Shared,
    cwd: &str,
    scope: Option<Scope>,
) -> Result<(QueryScope, Option<ProjectEntry>)> {
    match scope.unwrap_or_default() {
        Scope::All => Ok((QueryScope::All, resolve(sh, cwd).ok())),
        Scope::Project => {
            let p = resolve(sh, cwd)?;
            Ok((QueryScope::Project(p.project.id), Some(p)))
        }
    }
}

/// Drops every cached resolution (after a project rename or delete).
pub(crate) fn forget_cache(sh: &Shared) {
    lock(&sh.projects).clear();
}
