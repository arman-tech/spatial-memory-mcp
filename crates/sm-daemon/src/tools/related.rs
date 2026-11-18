//! `related` (SPEC 6, Related): two lists from one start, a memory or an
//! anchor.
//!
//! - `by_vector`: the nearest active memories to the start's vector. A
//!   memory start uses its own mean chunk vector; an anchor start uses the
//!   re-normalized mean of the vectors of the active memories attached to
//!   it (no model is needed either way). The start, and for an anchor the
//!   memories attached to it, are left out: they are in `by_links` at one
//!   hop. A start with no embedded chunks yet has an empty vector list.
//! - `by_links`: memories within two hops over `links` and shared anchors,
//!   in the start's project (`related_to_memory`, `related_to_anchor`).
//!
//! A memory start that is an automatic capture waiting for the judge
//! (`pending`) is refused: its vector and its links would show what it
//! holds before the judge has kept it (B-181).
//!
//! Both lists share the reply budget, vector list first; when the vector
//! list does not fit whole, the link list gets none (e-report decision 24).

use std::collections::HashMap;

use sm_core::error::Result;
use sm_core::ids::MemoryId;
use sm_core::protocol::{LinkedMemory, RelatedParams, RelatedReply};
use sm_core::traits::{MemoryFilter, QueryScope, VectorQuery};
use sm_core::types::{LinkEnd, Memory};

use crate::budget::{fit, fixed_cost, tokens_of};
use crate::dispatch::Ctx;
use crate::view::Names;

/// The re-normalized mean of `vectors`, or `None` when there are none.
fn mean(vectors: &[Vec<f32>]) -> Option<Vec<f32>> {
    let first = vectors.first()?;
    let mut sum = vec![0.0f32; first.len()];
    for v in vectors {
        for (s, x) in sum.iter_mut().zip(v) {
            *s += x;
        }
    }
    let norm = sum.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm <= f32::EPSILON {
        return None;
    }
    Some(sum.into_iter().map(|x| x / norm).collect())
}

pub(crate) fn related(ctx: &Ctx<'_>, p: &RelatedParams) -> Result<RelatedReply> {
    let sh = ctx.sh;
    let k = p.k.unwrap_or(sh.settings.recall.k) as usize;
    let (scope, _) = ctx.scope(p.scope)?;
    let (from, start_vector, exclude, walk) = match (&p.id, &p.anchor) {
        (Some(id), _) => {
            // not_found for an unknown memory, and invalid_input for a
            // capture that waits for the judge, before any search (B-181).
            sh.store.get_memory(id)?;
            let walk = sh.store.related_to_memory(id, &[])?;
            let vector = sh.store.memory_vector(id)?;
            (LinkEnd::Memory(*id), vector, vec![*id], walk)
        }
        (None, Some(anchor)) => {
            let project = ctx.project()?;
            let pid = project.project.id;
            let filter = MemoryFilter {
                anchor: Some(anchor.clone()),
                ..MemoryFilter::default()
            };
            let attached = sh.store.memory_vectors(QueryScope::Project(pid), &filter)?;
            let vectors: Vec<Vec<f32>> = attached.iter().map(|(_, v)| v.clone()).collect();
            let mut exclude: Vec<MemoryId> = attached.into_iter().map(|(id, _)| id).collect();
            exclude.extend(sh.store.memories_for_anchor(pid, anchor)?);
            let walk = sh.store.related_to_anchor(pid, anchor, &[])?;
            (
                LinkEnd::Anchor(anchor.clone()),
                mean(&vectors),
                exclude,
                walk,
            )
        }
        // RelatedParams::validate requires exactly one of the two.
        (None, None) => {
            return Err(sm_core::error::CoreError::invalid_input(
                "give exactly one of id and anchor",
            ));
        }
    };

    let hits = match &start_vector {
        Some(vector) => sh.store.nearest(&VectorQuery {
            vector,
            scope,
            filter: MemoryFilter {
                exclude,
                ..MemoryFilter::default()
            },
            limit: k,
        })?,
        None => Vec::new(),
    };
    ctx.check_deadline("the vector search")?;

    let mut ids: Vec<MemoryId> = hits.iter().map(|h| h.memory_id).collect();
    ids.extend(walk.iter().map(|w| w.memory_id));
    let memories: HashMap<MemoryId, Memory> = sh
        .store
        .get_memories(&ids)?
        .into_iter()
        .map(|m| (m.id, m))
        .collect();
    let names = Names::load(&sh.store)?;
    // Ranks are numbered after a memory deleted since the search is
    // dropped, so they match the reply's order.
    let mut by_vector: Vec<_> = hits
        .iter()
        .filter_map(|h| {
            let mut v = names.view(memories.get(&h.memory_id)?.clone());
            v.score = Some(h.score);
            Some(v)
        })
        .collect();
    for (i, v) in by_vector.iter_mut().enumerate() {
        v.rank = Some(i as u32 + 1);
    }
    let by_links: Vec<LinkedMemory> = walk
        .into_iter()
        .filter_map(|w| {
            Some(LinkedMemory {
                memory: names.view(memories.get(&w.memory_id)?.clone()),
                hops: w.hops,
                via: w.via,
            })
        })
        .collect();

    let budget = ctx.budget(p.limit);
    let skeleton = RelatedReply {
        from: from.clone(),
        by_vector: Vec::new(),
        by_links: Vec::new(),
        truncated: true,
        omitted: u32::MAX,
    };
    let used = fixed_cost("related", &skeleton, budget)?;
    let vectors = fit(by_vector, budget, used);
    let used = used + tokens_of(&vectors.items);
    let links = if vectors.omitted > 0 {
        // The budget ran out in the first list; the second gets none.
        let n = by_links.len() as u32;
        crate::budget::Fitted {
            items: Vec::new(),
            truncated: n > 0,
            omitted: n,
        }
    } else {
        fit(by_links, budget, used)
    };
    Ok(RelatedReply {
        from,
        by_vector: vectors.items,
        by_links: links.items,
        truncated: vectors.truncated || links.truncated,
        omitted: vectors.omitted + links.omitted,
    })
}

#[cfg(test)]
mod tests {
    use super::mean;

    #[test]
    fn s6_anchor_start_vector_is_the_normalized_mean() {
        let m = mean(&[vec![1.0, 0.0], vec![0.0, 1.0]]).expect("a mean");
        let half = std::f32::consts::FRAC_1_SQRT_2;
        assert!((m[0] - half).abs() < 1e-6 && (m[1] - half).abs() < 1e-6);
        assert!(mean(&[]).is_none());
        assert!(mean(&[vec![1.0, 0.0], vec![-1.0, 0.0]]).is_none());
    }
}
