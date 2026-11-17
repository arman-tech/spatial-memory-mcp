//! `consolidate` (SPEC 8; B-31, B-73): propose near-duplicate groups in the
//! current project, or apply exactly the groups the caller lists.
//!
//! Proposals compare every pair of the project's active memories that have
//! embedded chunks, by the cosine of their re-normalized mean vectors (the
//! store's per-memory vector). Memories are taken newest first; each one
//! not yet in a group gathers the older ungrouped memories at or above the
//! threshold, so the suggested survivor is the newest memory of its group.
//! This is O(n^2) in the project's memories: about 2 million dot products
//! at 2,000 memories. Nothing changes without `apply`.
//!
//! With `apply`, every listed id is checked first (it exists, is in the
//! current project, and each memory to merge is local), so a bad group
//! fails the call before any group is applied. Each group is then one
//! `merge_memories` transaction: the merged memories become `superseded`
//! and record `merged_into` (B-73); nothing is deleted.

use std::collections::HashMap;

use sm_core::error::{CoreError, Result};
use sm_core::ids::MemoryId;
use sm_core::protocol::{
    ConsolidateParams, ConsolidateReply, DEFAULT_CONSOLIDATE_THRESHOLD, DEFAULT_MAX_GROUPS,
    Duplicate, DuplicateGroup, MergeGroup,
};
use sm_core::traits::{MemoryFilter, QueryScope};
use sm_core::types::{Memory, Origin};
use sm_store::Access;

use crate::budget::{fit, fixed_cost};
use crate::dispatch::Ctx;
use crate::view::Names;

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Groups by index into `vectors`, which are ordered newest first: each
/// group's first index is its survivor, with the similarity of each other
/// member to it.
pub(crate) fn group(vectors: &[Vec<f32>], threshold: f32) -> Vec<(usize, Vec<(usize, f32)>)> {
    let mut taken = vec![false; vectors.len()];
    let mut groups = Vec::new();
    for i in 0..vectors.len() {
        if taken[i] {
            continue;
        }
        let mut dups = Vec::new();
        for j in i + 1..vectors.len() {
            if taken[j] {
                continue;
            }
            let s = dot(&vectors[i], &vectors[j]);
            if s >= threshold {
                dups.push((j, s));
            }
        }
        if !dups.is_empty() {
            taken[i] = true;
            for (j, _) in &dups {
                taken[*j] = true;
            }
            groups.push((i, dups));
        }
    }
    groups
}

fn propose(ctx: &Ctx<'_>, p: &ConsolidateParams) -> Result<ConsolidateReply> {
    let sh = ctx.sh;
    let project = ctx.project()?;
    let threshold = p.threshold.unwrap_or(DEFAULT_CONSOLIDATE_THRESHOLD) as f32;
    let max_groups = p.max_groups.unwrap_or(DEFAULT_MAX_GROUPS) as usize;
    let filter = MemoryFilter {
        types: p.memory_type.clone(),
        ..MemoryFilter::default()
    };
    let vectors = sh
        .store
        .memory_vectors(QueryScope::Project(project.project.id), &filter)?;
    let ids: Vec<MemoryId> = vectors.iter().map(|(id, _)| *id).collect();
    let mut memories: HashMap<MemoryId, Memory> = sh
        .store
        .get_memories(&ids)?
        .into_iter()
        .map(|m| (m.id, m))
        .collect();
    let mut order: Vec<(MemoryId, Vec<f32>)> = vectors
        .into_iter()
        .filter(|(id, _)| memories.contains_key(id))
        .collect();
    order.sort_by(|a, b| {
        let (ma, mb) = (&memories[&a.0], &memories[&b.0]);
        mb.created_at.cmp(&ma.created_at).then(a.0.cmp(&b.0))
    });
    let vecs: Vec<Vec<f32>> = order.iter().map(|(_, v)| v.clone()).collect();
    let groups = group(&vecs, threshold);
    ctx.check_deadline("comparing the memories")?;

    let found = groups.len();
    let names = Names::load(&sh.store)?;
    let mut proposals = Vec::with_capacity(found.min(max_groups));
    for (keep, dups) in groups.into_iter().take(max_groups) {
        let Some(k) = memories.remove(&order[keep].0) else {
            continue;
        };
        let duplicates = dups
            .into_iter()
            .filter_map(|(j, s)| {
                memories.remove(&order[j].0).map(|m| Duplicate {
                    memory: names.view(m),
                    similarity: s,
                })
            })
            .collect();
        proposals.push(DuplicateGroup {
            keep: names.view(k),
            duplicates,
        });
    }
    let over = (found - proposals.len()) as u32;
    let budget = ctx.budget(p.limit);
    let skeleton = ConsolidateReply {
        applied: false,
        proposals: Vec::new(),
        merged: Vec::new(),
        truncated: true,
        omitted: u32::MAX,
    };
    let fitted = fit(
        proposals,
        budget,
        fixed_cost("consolidate", &skeleton, budget)?,
    );
    Ok(ConsolidateReply {
        applied: false,
        proposals: fitted.items,
        merged: Vec::new(),
        truncated: fitted.truncated || over > 0,
        omitted: fitted.omitted + over,
    })
}

fn apply(ctx: &Ctx<'_>, p: &ConsolidateParams) -> Result<ConsolidateReply> {
    let sh = ctx.sh;
    let project = ctx.project()?;
    let pid = project.project.id;
    for g in &p.groups {
        for id in std::iter::once(&g.keep).chain(&g.merge) {
            let m = sh.store.get_memory(id)?;
            if m.project_id != pid {
                return Err(CoreError::invalid_input(format!(
                    "memory {id} is not in project {}",
                    project.project.name
                )));
            }
            if id != &g.keep && m.origin != Origin::Local {
                return Err(CoreError::new(
                    sm_core::error::ErrorCode::SharedReadOnly,
                    format!("memory {id} is shared; nothing was merged"),
                )
                .with_next_action(
                    "Shared memories change only through sync; leave it out of the group.",
                ));
            }
        }
    }
    let mut merged = Vec::with_capacity(p.groups.len());
    for g in &p.groups {
        let changed = sh.store.merge_memories(&g.keep, &g.merge, Access::Local)?;
        merged.push(MergeGroup {
            keep: g.keep,
            merge: changed,
        });
    }
    sh.drop_maps();
    Ok(ConsolidateReply {
        applied: true,
        proposals: Vec::new(),
        merged,
        truncated: false,
        omitted: 0,
    })
}

pub(crate) fn consolidate(ctx: &Ctx<'_>, p: &ConsolidateParams) -> Result<ConsolidateReply> {
    if p.apply {
        apply(ctx, p)
    } else {
        propose(ctx, p)
    }
}

#[cfg(test)]
mod tests {
    use super::group;

    #[test]
    fn s8_consolidate_groups_around_the_newest_memory() {
        // Ordered newest first: 0 and 2 are near-duplicates, 1 stands alone.
        let v = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.99, 0.141]];
        let g = group(&v, 0.85);
        assert_eq!(g.len(), 1);
        assert_eq!(g[0].0, 0);
        assert_eq!(g[0].1.len(), 1);
        assert_eq!(g[0].1[0].0, 2);
        assert!(group(&v, 0.995).is_empty());
    }
}
