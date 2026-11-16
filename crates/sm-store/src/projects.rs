//! Projects: upsert by identity, rename, delete with cascade, counts (SPEC
//! section 4, `projects`; decision B-20).

use std::collections::BTreeMap;

use rusqlite::{Connection, OptionalExtension, Row};
use sm_core::error::{CoreError, Result};
use sm_core::ids::ProjectId;
use sm_core::project::ProjectIdentity;
use sm_core::protocol::{ProjectReply, ProjectStatus};
use sm_core::time::Timestamp;
use sm_core::types::Project;

use crate::Store;
use crate::codec::{count, ts};
use crate::error::Ctx;

const PROJECT_COLUMNS: &str = "id, name, git_remote, root_path";

fn project_from_row(r: &Row<'_>) -> rusqlite::Result<Project> {
    Ok(Project {
        id: ProjectId(r.get(0)?),
        name: r.get(1)?,
        git_remote: r.get(2)?,
        root_path: r.get(3)?,
    })
}

/// Fails with `not_found` unless the project exists.
pub(crate) fn require_project(c: &Connection, id: ProjectId) -> Result<()> {
    let found: bool = c
        .prepare_cached("SELECT EXISTS (SELECT 1 FROM projects WHERE id = ?1)")
        .and_then(|mut s| s.query_row([id.0], |r| r.get(0)))
        .ctx("looking up a project")?;
    if found {
        Ok(())
    } else {
        Err(CoreError::not_found(format!("project {id} does not exist")))
    }
}

fn by_name(c: &Connection, name: &str) -> Result<Vec<Project>> {
    let mut s = c
        .prepare_cached(&format!(
            "SELECT {PROJECT_COLUMNS} FROM projects WHERE name = ?1 ORDER BY id"
        ))
        .ctx("finding projects by name")?;
    let rows = s
        .query_map([name], project_from_row)
        .ctx("finding projects by name")?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .ctx("finding projects by name")
}

/// The one project named `name`; `not_found` when there is none and
/// `invalid_input` when the name is ambiguous.
fn one_by_name(c: &Connection, name: &str) -> Result<Project> {
    let mut found = by_name(c, name)?;
    match found.len() {
        0 => Err(CoreError::not_found(format!(
            "no project is named {name:?}"
        ))),
        1 => Ok(found.remove(0)),
        n => Err(CoreError::invalid_input(format!(
            "{n} projects are named {name:?}; rename one from inside its repository first"
        ))),
    }
}

fn memory_count(c: &Connection, id: ProjectId) -> Result<u64> {
    let n: i64 = c
        .prepare_cached("SELECT count(*) FROM memories WHERE project_id = ?1")
        .and_then(|mut s| s.query_row([id.0], |r| r.get(0)))
        .ctx("counting a project's memories")?;
    Ok(count(n))
}

impl Store {
    /// Finds the project with this identity (`ProjectIdentity::key`) or
    /// creates it (decision B-20). An existing project keeps its name, which
    /// may have been changed with `rename_project`; its root path is
    /// refreshed (a clone can move).
    pub fn upsert_project(&self, identity: &ProjectIdentity) -> Result<Project> {
        let key = identity.key();
        let now = ts(Timestamp::now());
        self.write("recording a project", |tx| {
            tx.prepare_cached(
                "INSERT INTO projects(identity, name, git_remote, root_path, created_at, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?5)
                 ON CONFLICT(identity) DO UPDATE SET
                     git_remote = excluded.git_remote,
                     root_path = excluded.root_path,
                     updated_at = CASE
                         WHEN projects.root_path IS excluded.root_path
                          AND projects.git_remote IS excluded.git_remote
                         THEN projects.updated_at ELSE excluded.updated_at END",
            )
            .and_then(|mut s| {
                s.execute((
                    &key,
                    &identity.name,
                    &identity.git_remote,
                    &identity.root_path,
                    now,
                ))
            })
            .ctx("recording a project")?;
            tx.prepare_cached(&format!(
                "SELECT {PROJECT_COLUMNS} FROM projects WHERE identity = ?1"
            ))
            .and_then(|mut s| s.query_row([&key], project_from_row))
            .ctx("reading a project back")
        })
    }

    /// The project with this identity key, if it has been recorded.
    pub fn project_by_identity(&self, key: &str) -> Result<Option<Project>> {
        self.read("finding a project", |tx| {
            tx.prepare_cached(&format!(
                "SELECT {PROJECT_COLUMNS} FROM projects WHERE identity = ?1"
            ))
            .and_then(|mut s| s.query_row([key], project_from_row).optional())
            .ctx("finding a project")
        })
    }

    /// The project with this id.
    pub fn project(&self, id: ProjectId) -> Result<Project> {
        self.read("reading a project", |tx| {
            tx.prepare_cached(&format!(
                "SELECT {PROJECT_COLUMNS} FROM projects WHERE id = ?1"
            ))
            .and_then(|mut s| s.query_row([id.0], project_from_row).optional())
            .ctx("reading a project")?
            .ok_or_else(|| CoreError::not_found(format!("project {id} does not exist")))
        })
    }

    /// Projects with this display name (names are not unique).
    pub fn projects_by_name(&self, name: &str) -> Result<Vec<Project>> {
        self.read("finding projects by name", |tx| by_name(tx, name))
    }

    /// Every project, by name then id.
    pub fn projects(&self) -> Result<Vec<Project>> {
        self.read("listing projects", |tx| {
            let mut s = tx
                .prepare_cached(&format!(
                    "SELECT {PROJECT_COLUMNS} FROM projects ORDER BY name, id"
                ))
                .ctx("listing projects")?;
            let rows = s.query_map([], project_from_row).ctx("listing projects")?;
            rows.collect::<rusqlite::Result<Vec<_>>>()
                .ctx("listing projects")
        })
    }

    /// Renames the one project named `from` to `to`. Fails when no project
    /// or more than one is named `from`, or when a project is already named
    /// `to`. Memories stay attached by project id; `memories` in the reply
    /// counts them.
    pub fn rename_project(&self, from: &str, to: &str) -> Result<ProjectReply> {
        if to.trim().is_empty() || to.trim() != to {
            return Err(CoreError::invalid_input(format!(
                "project name {to:?} is empty or has leading or trailing whitespace"
            )));
        }
        let now = ts(Timestamp::now());
        self.write("renaming a project", |tx| {
            let p = one_by_name(tx, from)?;
            if from != to && !by_name(tx, to)?.is_empty() {
                return Err(CoreError::invalid_input(format!(
                    "a project is already named {to:?}"
                )));
            }
            tx.execute(
                "UPDATE projects SET name = ?2, updated_at = ?3 WHERE id = ?1",
                (p.id.0, to, now),
            )
            .ctx("renaming a project")?;
            Ok(ProjectReply {
                name: to.to_owned(),
                renamed_from: Some(from.to_owned()),
                deleted: false,
                memories: memory_count(tx, p.id)?,
            })
        })
    }

    /// Deletes the one project named `name` and everything in it: memories,
    /// provenance, chunks and their index rows, anchors, links, map runs and
    /// injections (foreign keys cascade; triggers clear the indexes).
    pub fn delete_project(&self, name: &str) -> Result<ProjectReply> {
        self.write("deleting a project", |tx| {
            let p = one_by_name(tx, name)?;
            let memories = memory_count(tx, p.id)?;
            tx.execute("DELETE FROM projects WHERE id = ?1", [p.id.0])
                .ctx("deleting a project")?;
            Ok(ProjectReply {
                name: p.name,
                renamed_from: None,
                deleted: true,
                memories,
            })
        })
    }

    /// Memory counts for one project: total, by type, by status, and shared.
    /// Only types and statuses that occur are listed.
    pub fn project_status(&self, id: ProjectId) -> Result<ProjectStatus> {
        self.read("counting a project's memories", |tx| {
            let p = tx
                .prepare_cached(&format!(
                    "SELECT {PROJECT_COLUMNS} FROM projects WHERE id = ?1"
                ))
                .and_then(|mut s| s.query_row([id.0], project_from_row).optional())
                .ctx("reading a project")?
                .ok_or_else(|| CoreError::not_found(format!("project {id} does not exist")))?;
            status_of(tx, p)
        })
    }

    /// [`Store::project_status`] for every project, by name then id.
    pub fn project_statuses(&self) -> Result<Vec<ProjectStatus>> {
        self.read("counting memories", |tx| {
            let projects = {
                let mut s = tx
                    .prepare_cached(&format!(
                        "SELECT {PROJECT_COLUMNS} FROM projects ORDER BY name, id"
                    ))
                    .ctx("listing projects")?;
                let rows = s.query_map([], project_from_row).ctx("listing projects")?;
                rows.collect::<rusqlite::Result<Vec<_>>>()
                    .ctx("listing projects")?
            };
            projects.into_iter().map(|p| status_of(tx, p)).collect()
        })
    }
}

fn grouped(c: &Connection, sql: &str, id: ProjectId) -> Result<BTreeMap<String, u64>> {
    let mut s = c.prepare_cached(sql).ctx("counting memories")?;
    let rows = s
        .query_map([id.0], |r| Ok((r.get::<_, String>(0)?, count(r.get(1)?))))
        .ctx("counting memories")?;
    rows.collect::<rusqlite::Result<BTreeMap<_, _>>>()
        .ctx("counting memories")
}

fn status_of(c: &Connection, p: Project) -> Result<ProjectStatus> {
    let by_type = grouped(
        c,
        "SELECT type, count(*) FROM memories WHERE project_id = ?1 GROUP BY type",
        p.id,
    )?;
    let by_status = grouped(
        c,
        "SELECT status, count(*) FROM memories WHERE project_id = ?1 GROUP BY status",
        p.id,
    )?;
    let shared: i64 = c
        .prepare_cached("SELECT count(*) FROM memories WHERE project_id = ?1 AND origin = 'shared'")
        .and_then(|mut s| s.query_row([p.id.0], |r| r.get(0)))
        .ctx("counting shared memories")?;
    Ok(ProjectStatus {
        name: p.name,
        git_remote: p.git_remote,
        memories: by_status.values().sum(),
        by_type,
        by_status,
        shared: count(shared),
    })
}
