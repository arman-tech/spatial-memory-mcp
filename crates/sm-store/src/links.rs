//! Links between memories and anchors, and the walk behind `related` (SPEC
//! section 4, `links`; section 8, `link` and `related`; decision B-37).
//!
//! A link's memory end is stored as kind `memory` with the memory's id; an
//! anchor end as the anchor's kind and name. Links belong to one project, and
//! a link's memory ends must be memories of that project. Links are removed
//! with the memories they name (trigger `memories_ad`) and with their project.

use std::collections::{HashMap, VecDeque};

use rusqlite::{Connection, OptionalExtension, Row};
use sm_core::error::{CoreError, Result};
use sm_core::ids::{MemoryId, ProjectId};
use sm_core::protocol::{LinkView, RELATED_MAX_HOPS};
use sm_core::time::Timestamp;
use sm_core::types::{AnchorKind, AnchorRef, Link, LinkEnd, LinkSource, Relation, Status};

use crate::Store;
use crate::codec::{from_ts, parse_col, ts};
use crate::error::{Ctx, conversion};
use crate::memories::{anchors_of, memories_for_anchor, memory_not_found};
use crate::projects::require_project;

/// The `extractor` named on the synthetic edges [`Store::related`] reports
/// for anchors a memory is attached to (rather than stored links).
pub const ANCHOR_EDGE_EXTRACTOR: &str = "memory_anchors";

/// The result of [`Store::add_link`].
#[derive(Debug, Clone, PartialEq)]
pub struct LinkAdded {
    /// The stored link. When an identical link (same ends and relation)
    /// already existed, it is returned as stored and nothing is written.
    pub link: Link,
    pub created: bool,
}

/// One memory reached by [`Store::related`].
#[derive(Debug, Clone, PartialEq)]
pub struct WalkHit {
    pub memory_id: MemoryId,
    /// 1 or 2.
    pub hops: u8,
    /// The edges walked from the start, in order. Passing through an anchor
    /// the memories share shows as two `relates_to` edges with extractor
    /// [`ANCHOR_EDGE_EXTRACTOR`], memory to anchor and anchor to memory, and
    /// counts as one hop.
    pub via: Vec<Link>,
}

const LINK_COLUMNS: &str =
    "src_kind, src_id, dst_kind, dst_id, relation, source, extractor, verified_commit, verified_at";

fn end_cols(end: &LinkEnd) -> (&'static str, String) {
    match end {
        LinkEnd::Memory(id) => ("memory", id.to_string()),
        LinkEnd::Anchor(a) => (a.kind.as_str(), a.name.clone()),
    }
}

fn end_col(r: &Row<'_>, kind_idx: usize) -> rusqlite::Result<LinkEnd> {
    let kind: String = r.get(kind_idx)?;
    let id: String = r.get(kind_idx + 1)?;
    let end = if kind == "memory" {
        MemoryId::parse(&id).map(LinkEnd::Memory)
    } else {
        kind.parse::<AnchorKind>()
            .and_then(|k| AnchorRef::new(k, id))
            .map(LinkEnd::Anchor)
    };
    end.map_err(|e| conversion(kind_idx, e))
}

fn link_from_row(r: &Row<'_>) -> rusqlite::Result<Link> {
    Ok(Link {
        src: end_col(r, 0)?,
        dst: end_col(r, 2)?,
        relation: parse_col(r, 4)?,
        source: parse_col(r, 5)?,
        extractor: r.get(6)?,
        verified_commit: r.get(7)?,
        verified_at: r.get::<_, Option<i64>>(8)?.map(from_ts),
    })
}

/// A synthetic edge between a memory and an anchor it is attached to.
fn anchor_edge(memory: MemoryId, anchor: AnchorRef) -> Link {
    Link {
        src: LinkEnd::Memory(memory),
        dst: LinkEnd::Anchor(anchor),
        relation: Relation::RelatesTo,
        source: LinkSource::Extracted,
        extractor: Some(ANCHOR_EDGE_EXTRACTOR.to_owned()),
        verified_commit: None,
        verified_at: None,
    }
}

fn memory_project(c: &Connection, id: &MemoryId) -> Result<Option<ProjectId>> {
    c.prepare_cached("SELECT project_id FROM memories WHERE id = ?1")
        .and_then(|mut s| {
            s.query_row([id.to_string()], |r| r.get(0).map(ProjectId))
                .optional()
        })
        .ctx("reading a memory's project")
}

/// Every stored link with `end` at either side, oldest first.
fn links_at(c: &Connection, project: ProjectId, end: &LinkEnd) -> Result<Vec<Link>> {
    let (kind, id) = end_cols(end);
    let mut s = c
        .prepare_cached(&format!(
            "SELECT {LINK_COLUMNS} FROM links
             WHERE project_id = ?1
               AND ((src_kind = ?2 AND src_id = ?3) OR (dst_kind = ?2 AND dst_id = ?3))
             ORDER BY id"
        ))
        .ctx("reading links")?;
    let rows = s
        .query_map((project.0, kind, &id), link_from_row)
        .ctx("reading links")?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .ctx("reading links")
}

fn find_link(
    c: &Connection,
    project: ProjectId,
    src: &LinkEnd,
    dst: &LinkEnd,
    relation: Relation,
) -> Result<Option<Link>> {
    let (sk, si) = end_cols(src);
    let (dk, di) = end_cols(dst);
    c.prepare_cached(&format!(
        "SELECT {LINK_COLUMNS} FROM links
         WHERE project_id = ?1 AND src_kind = ?2 AND src_id = ?3
           AND dst_kind = ?4 AND dst_id = ?5 AND relation = ?6"
    ))
    .and_then(|mut s| {
        s.query_row(
            (project.0, sk, &si, dk, &di, relation.as_str()),
            link_from_row,
        )
        .optional()
    })
    .ctx("reading a link")
}

/// Checks an end before it is stored: anchors are revalidated, memories
/// must exist in `project`.
fn check_end(c: &Connection, project: ProjectId, end: &LinkEnd) -> Result<()> {
    match end {
        LinkEnd::Anchor(a) => AnchorRef::new(a.kind, a.name.clone()).map(drop),
        LinkEnd::Memory(id) => match memory_project(c, id)? {
            None => Err(memory_not_found(id)),
            Some(p) if p == project => Ok(()),
            Some(_) => Err(CoreError::invalid_input(format!(
                "memory {id} belongs to another project; a link joins ends in one project"
            ))),
        },
    }
}

/// Decides `LinkView.stale` from the project head alone (decision B-37).
///
/// The store cannot walk git ancestry without running git, so callers pass
/// an ancestry oracle to [`Store::links_of`]. This one answers only what the
/// head itself settles: a link verified at the head commit is not stale. Any
/// other commit, or an unknown head, is undecidable and `stale` stays absent.
/// A commit matches the head when it equals it ignoring case, or is a prefix
/// of it of at least 7 hexadecimal characters.
pub fn head_only(head: Option<&str>) -> impl Fn(&str) -> Option<bool> + '_ {
    move |commit: &str| {
        let head = head?.trim();
        let commit = commit.trim();
        let full = commit.eq_ignore_ascii_case(head);
        let prefix = commit.len() >= 7
            && commit.len() <= head.len()
            && commit.bytes().all(|b| b.is_ascii_hexdigit())
            && head.is_char_boundary(commit.len())
            && head[..commit.len()].eq_ignore_ascii_case(commit);
        (full || prefix).then_some(true)
    }
}

/// A node of the walk and how it was reached.
struct Reached {
    hops: u8,
    via: Vec<Link>,
}

impl Store {
    /// Stores a link in `project` and returns it. An identical link (same
    /// ends and relation) is returned as is, with `created` false. Fails
    /// with `invalid_input` when the ends are the same, an anchor is invalid
    /// or a memory end belongs to another project, and with `not_found` when
    /// a memory end or the project does not exist.
    pub fn add_link(&self, project: ProjectId, link: Link) -> Result<LinkAdded> {
        if link.src == link.dst {
            return Err(CoreError::invalid_input(
                "a link must join two different ends",
            ));
        }
        if let Some(c) = &link.verified_commit
            && c.trim().is_empty()
        {
            return Err(CoreError::invalid_input("verified_commit is empty"));
        }
        let now = ts(Timestamp::now());
        self.write("storing a link", |tx| {
            require_project(tx, project)?;
            check_end(tx, project, &link.src)?;
            check_end(tx, project, &link.dst)?;
            if let Some(existing) = find_link(tx, project, &link.src, &link.dst, link.relation)? {
                return Ok(LinkAdded {
                    link: existing,
                    created: false,
                });
            }
            let (sk, si) = end_cols(&link.src);
            let (dk, di) = end_cols(&link.dst);
            tx.prepare_cached(
                "INSERT INTO links(project_id, src_kind, src_id, dst_kind, dst_id, relation,
                                   source, extractor, verified_commit, verified_at, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            )
            .and_then(|mut s| {
                s.execute((
                    project.0,
                    sk,
                    &si,
                    dk,
                    &di,
                    link.relation.as_str(),
                    link.source.as_str(),
                    &link.extractor,
                    &link.verified_commit,
                    link.verified_at.map(ts),
                    now,
                ))
            })
            .ctx("storing a link")?;
            Ok(LinkAdded {
                link: link.clone(),
                created: true,
            })
        })
    }

    /// Removes the link from `src` to `dst` with `relation` in `project`.
    /// Returns 1 when it existed, 0 otherwise.
    pub fn remove_link(
        &self,
        project: ProjectId,
        src: &LinkEnd,
        dst: &LinkEnd,
        relation: Relation,
    ) -> Result<u32> {
        let (sk, si) = end_cols(src);
        let (dk, di) = end_cols(dst);
        self.write("removing a link", |tx| {
            let n = tx
                .execute(
                    "DELETE FROM links WHERE project_id = ?1 AND src_kind = ?2 AND src_id = ?3
                       AND dst_kind = ?4 AND dst_id = ?5 AND relation = ?6",
                    (project.0, sk, &si, dk, &di, relation.as_str()),
                )
                .ctx("removing a link")?;
            Ok(u32::from(n > 0))
        })
    }

    /// The links in `project` with `end` at either side, oldest first.
    /// `is_ancestor` answers whether a `verified_commit` is an ancestor of
    /// the project head (`None` when it cannot tell); see [`head_only`]. A
    /// link whose commit is not an ancestor is `stale`.
    pub fn links_of(
        &self,
        project: ProjectId,
        end: &LinkEnd,
        is_ancestor: impl Fn(&str) -> Option<bool>,
    ) -> Result<Vec<LinkView>> {
        let links = self.read("reading links", |tx| {
            if let LinkEnd::Memory(id) = end
                && memory_project(tx, id)?.is_none()
            {
                return Err(memory_not_found(id));
            }
            links_at(tx, project, end)
        })?;
        Ok(links
            .into_iter()
            .map(|link| {
                let stale = link
                    .verified_commit
                    .as_deref()
                    .and_then(&is_ancestor)
                    .map(|ancestor| !ancestor);
                LinkView { link, stale }
            })
            .collect())
    }

    /// Every stored link in `project`, oldest first, without the synthetic
    /// anchor edges (for `export`, decision B-77). Fails with `not_found`
    /// when the project does not exist.
    pub fn project_links(&self, project: ProjectId) -> Result<Vec<Link>> {
        self.read("reading links", |tx| {
            require_project(tx, project)?;
            let mut s = tx
                .prepare_cached(&format!(
                    "SELECT {LINK_COLUMNS} FROM links WHERE project_id = ?1 ORDER BY id"
                ))
                .ctx("reading links")?;
            let rows = s
                .query_map([project.0], link_from_row)
                .ctx("reading links")?;
            rows.collect::<rusqlite::Result<Vec<_>>>()
                .ctx("reading links")
        })
    }

    /// Memories within two hops of memory `id`, in its project (SPEC
    /// section 8, `related`). See [`Store::related_to_anchor`].
    pub fn related_to_memory(&self, id: &MemoryId, statuses: &[Status]) -> Result<Vec<WalkHit>> {
        self.read("walking links", |tx| {
            let project = memory_project(tx, id)?.ok_or_else(|| memory_not_found(id))?;
            walk(tx, project, &LinkEnd::Memory(*id), statuses)
        })
    }

    /// Memories within two hops of `anchor` in `project`.
    ///
    /// The walk follows stored links in either direction (one hop each) and
    /// anchors: two memories attached to the same anchor are one hop apart,
    /// and a memory attached to the starting anchor is one hop from it. Each
    /// memory is reported once, at its fewest hops, by hops then id; the
    /// start is not reported. The walk passes through memories of any
    /// status; only memories whose status is in `statuses` (empty means
    /// `active`) are reported, and never a `pending` one (B-157).
    pub fn related_to_anchor(
        &self,
        project: ProjectId,
        anchor: &AnchorRef,
        statuses: &[Status],
    ) -> Result<Vec<WalkHit>> {
        self.read("walking links", |tx| {
            require_project(tx, project)?;
            walk(tx, project, &LinkEnd::Anchor(anchor.clone()), statuses)
        })
    }
}

/// A 0-1 breadth-first walk: a memory reaches its anchors at no cost, every
/// other edge costs one hop. Nodes are settled the first time they leave
/// the queue, so each is reported at its fewest hops, and ties go to the
/// path found first (anchors by position, links by age, memories by id).
fn walk(
    c: &Connection,
    project: ProjectId,
    start: &LinkEnd,
    statuses: &[Status],
) -> Result<Vec<WalkHit>> {
    let mut settled: HashMap<LinkEnd, Reached> = HashMap::new();
    let mut queue: VecDeque<(LinkEnd, Reached)> = VecDeque::new();
    queue.push_back((
        start.clone(),
        Reached {
            hops: 0,
            via: Vec::new(),
        },
    ));
    while let Some((node, reached)) = queue.pop_front() {
        if settled.contains_key(&node) {
            continue;
        }
        if reached.hops < RELATED_MAX_HOPS {
            let mut next = |end: LinkEnd, edge: Link, cost: u8| {
                if settled.contains_key(&end) {
                    return;
                }
                let mut via = reached.via.clone();
                via.push(edge);
                let r = Reached {
                    hops: reached.hops + cost,
                    via,
                };
                if cost == 0 {
                    queue.push_front((end, r));
                } else {
                    queue.push_back((end, r));
                }
            };
            match &node {
                LinkEnd::Memory(m) => {
                    for a in anchors_of(c, m)? {
                        next(LinkEnd::Anchor(a.clone()), anchor_edge(*m, a), 0);
                    }
                }
                LinkEnd::Anchor(a) => {
                    for m in memories_for_anchor(c, project, a)? {
                        next(LinkEnd::Memory(m), anchor_edge(m, a.clone()), 1);
                    }
                }
            }
            for link in links_at(c, project, &node)? {
                let other = if link.src == node {
                    link.dst.clone()
                } else {
                    link.src.clone()
                };
                next(other, link, 1);
            }
        }
        settled.insert(node, reached);
    }

    // `pending` is never reported, whatever is asked for (B-157).
    let wanted: Vec<&str> = if statuses.is_empty() {
        vec![Status::Active.as_str()]
    } else {
        statuses
            .iter()
            .filter(|s| s.is_served())
            .map(|s| s.as_str())
            .collect()
    };
    let mut status_of = c
        .prepare_cached("SELECT status FROM memories WHERE id = ?1")
        .ctx("reading a memory's status")?;
    let mut hits = Vec::new();
    for (node, r) in settled {
        let LinkEnd::Memory(id) = node else { continue };
        if r.hops == 0 {
            continue;
        }
        let status: Option<String> = status_of
            .query_row([id.to_string()], |row| row.get(0))
            .optional()
            .ctx("reading a memory's status")?;
        if status.is_some_and(|s| wanted.contains(&s.as_str())) {
            hits.push(WalkHit {
                memory_id: id,
                hops: r.hops,
                via: r.via,
            });
        }
    }
    hits.sort_by(|a, b| a.hops.cmp(&b.hops).then(a.memory_id.cmp(&b.memory_id)));
    Ok(hits)
}
