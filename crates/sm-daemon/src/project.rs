//! Project resolution (SPEC 9, Projects): `resolve_project` on the
//! request's working directory, cached per directory, then upserted. Reads
//! default to the current project; `scope: all` is explicit.

use std::path::Path;

use sm_core::error::Result;
use sm_core::project::resolve_project;
use sm_core::traits::QueryScope;
use sm_core::types::Scope;

use crate::state::{ProjectEntry, Shared, lock};

pub(crate) fn resolve(sh: &Shared, cwd: &str) -> Result<ProjectEntry> {
    let key = cwd.to_owned();
    if let Some(e) = lock(&sh.projects).get(&key) {
        return Ok(e.clone());
    }
    let identity = resolve_project(Path::new(cwd))?;
    let project = sh.store.upsert_project(&identity)?;
    let entry = ProjectEntry { project, identity };
    lock(&sh.projects).insert(key, entry.clone());
    Ok(entry)
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
