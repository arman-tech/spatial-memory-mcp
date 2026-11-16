//! `journey` (SPEC section 7, Journey): the great-circle path between two
//! memories and the nearest memory at each step.
//!
//! As in version 1, the path is spherical linear interpolation (slerp)
//! between the two memory vectors at `t = i / (steps - 1)`, falling back to
//! normalised linear interpolation when the two are almost parallel.
//!
//! Version 1 searched each step on its own, so one memory could fill several
//! steps. The version 3 contract (`JourneyStep.memory`: "nearest memory not
//! already on the path") removes repeats: step 0 is `from`, the last step is
//! `to`, and each step between is the nearest memory that is neither
//! endpoint nor chosen at an earlier step. Steps are filled in order from
//! `from`, so an earlier step wins a memory two steps would both pick. A step
//! with no memory left (every candidate excluded or filtered out) is empty.

use sm_core::ids::MemoryId;
use sm_core::protocol::{MAX_JOURNEY_STEPS, MIN_JOURNEY_STEPS};
use sm_core::traits::{MemoryFilter, QueryScope, VectorIndex, VectorQuery};
use sm_core::{CoreError, Result};

use crate::vector::{check_rows, cosine, unit};

/// Below this angle (radians) between the endpoints, slerp is replaced by
/// normalised linear interpolation (version 1's threshold).
const PARALLEL_ANGLE: f64 = 1e-6;

/// One end of a journey: a memory and its vector (the memory vector of SPEC
/// section 7, step 1, or any vector pointing the same way).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Endpoint<'a> {
    pub id: MemoryId,
    pub vector: &'a [f32],
}

/// A journey between two memories.
#[derive(Debug, Clone, PartialEq)]
pub struct JourneyRequest<'a> {
    pub from: Endpoint<'a>,
    pub to: Endpoint<'a>,
    /// Points on the path, both endpoints included:
    /// [`MIN_JOURNEY_STEPS`] to [`MAX_JOURNEY_STEPS`].
    pub steps: usize,
    /// Where intermediate memories are searched.
    pub scope: QueryScope,
    /// Filters for intermediate memories. Its `exclude` list is extended
    /// with the endpoints and the memories already on the path.
    pub filter: MemoryFilter,
}

/// One point of a journey.
#[derive(Debug, Clone, PartialEq)]
pub struct PathStep {
    /// 0-based.
    pub step: u32,
    /// Position on the path: 0 at `from`, 1 at `to`.
    pub t: f32,
    /// The unit vector on the path at `t`.
    pub point: Vec<f32>,
    /// The memory at this step: `from` at step 0, `to` at the last step,
    /// otherwise the nearest memory not already on the path.
    pub memory: Option<MemoryId>,
    /// The memory's cosine similarity to `point`: its best chunk's score
    /// from the index for intermediate steps, the endpoint vector's cosine
    /// for the endpoints.
    pub score: Option<f32>,
}

/// `steps` unit vectors evenly spaced in angle on the great circle from
/// `from` to `to`, each with its `t`. Both inputs are normalised first.
///
/// Fails with `invalid_input` when `steps` is below 2, the vectors are
/// empty, differ in length, hold a non-finite value or are zero, or point in
/// opposite directions (no single great circle joins them).
pub fn slerp_path(from: &[f32], to: &[f32], steps: usize) -> Result<Vec<(f32, Vec<f32>)>> {
    if steps < 2 {
        return Err(CoreError::invalid_input(format!(
            "a path needs at least 2 steps, got {steps}"
        )));
    }
    check_rows(&[from.to_vec(), to.to_vec()], "journey endpoints", true)?;
    let a = unit64(from);
    let b = unit64(to);
    let d = a
        .iter()
        .zip(&b)
        .map(|(x, y)| x * y)
        .sum::<f64>()
        .clamp(-1.0, 1.0);
    if 1.0 + d < 1e-6 {
        return Err(CoreError::invalid_input(
            "the two memories point in opposite directions, so no single path joins them",
        )
        .with_next_action("pick two memories that are not opposites"));
    }
    let omega = d.acos();
    let last = (steps - 1) as f64;
    let mut out = Vec::with_capacity(steps);
    for i in 0..steps {
        let t = i as f64 / last;
        let (wa, wb) = if omega < PARALLEL_ANGLE {
            (1.0 - t, t)
        } else {
            let s = omega.sin();
            (((1.0 - t) * omega).sin() / s, (t * omega).sin() / s)
        };
        let p: Vec<f64> = a.iter().zip(&b).map(|(x, y)| wa * x + wb * y).collect();
        let point = unit(&p).ok_or_else(|| {
            CoreError::internal(format!("the path point at t = {t} has zero length"))
        })?;
        out.push((t as f32, point));
    }
    Ok(out)
}

fn unit64(v: &[f32]) -> Vec<f64> {
    let norm = v
        .iter()
        .map(|&x| f64::from(x) * f64::from(x))
        .sum::<f64>()
        .sqrt();
    v.iter().map(|&x| f64::from(x) / norm).collect()
}

/// Walks the path from `req.from` to `req.to` and finds the memory at each
/// step (see the module documentation for the no-repeat rule).
///
/// Fails with `invalid_input` for a step count outside 2 to 20, the same
/// memory at both ends, or endpoint vectors [`slerp_path`] refuses; passes
/// on the index's errors.
pub fn journey(index: &dyn VectorIndex, req: &JourneyRequest<'_>) -> Result<Vec<PathStep>> {
    let (min, max) = (MIN_JOURNEY_STEPS as usize, MAX_JOURNEY_STEPS as usize);
    if !(min..=max).contains(&req.steps) {
        return Err(CoreError::invalid_input(format!(
            "steps must be {min} to {max}, got {}",
            req.steps
        )));
    }
    if req.from.id == req.to.id {
        return Err(CoreError::invalid_input(
            "a journey needs two different memories",
        ));
    }
    let path = slerp_path(req.from.vector, req.to.vector, req.steps)?;
    let last = path.len() - 1;
    let mut on_path = vec![req.from.id, req.to.id];
    let mut out = Vec::with_capacity(path.len());
    for (i, (t, point)) in path.into_iter().enumerate() {
        let (memory, score) = if i == 0 {
            (Some(req.from.id), Some(cosine(&point, req.from.vector)))
        } else if i == last {
            (Some(req.to.id), Some(cosine(&point, req.to.vector)))
        } else {
            let mut filter = req.filter.clone();
            filter.exclude.extend(on_path.iter().copied());
            let query = VectorQuery {
                vector: &point,
                scope: req.scope,
                filter,
                // One more than the memories on the path, so a hit is left
                // even if an index returned excluded memories anyway.
                limit: on_path.len() + 1,
            };
            let hit = index.nearest_memories(&query)?.into_iter().find(|h| {
                !on_path.contains(&h.memory_id) && !req.filter.exclude.contains(&h.memory_id)
            });
            match hit {
                Some(h) => {
                    on_path.push(h.memory_id);
                    (Some(h.memory_id), Some(h.score))
                }
                None => (None, None),
            }
        };
        out.push(PathStep {
            step: u32::try_from(i).unwrap_or(u32::MAX),
            t,
            point,
            memory,
            score,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sm_core::ErrorCode;

    #[test]
    fn s7_slerp_quarter_circle() {
        let path = slerp_path(&[1.0, 0.0], &[0.0, 1.0], 3).unwrap();
        let h = std::f32::consts::FRAC_1_SQRT_2;
        assert_eq!(path[0], (0.0, vec![1.0, 0.0]));
        assert!((path[1].1[0] - h).abs() < 1e-7 && (path[1].1[1] - h).abs() < 1e-7);
        assert_eq!(path[1].0, 0.5);
        assert!(path[2].1[0].abs() < 1e-7 && (path[2].1[1] - 1.0).abs() < 1e-7);
        assert_eq!(path[2].0, 1.0);
    }

    #[test]
    fn s7_slerp_parallel_and_opposite() {
        let same = slerp_path(&[1.0, 0.0], &[2.0, 0.0], 4).unwrap();
        assert!(same.iter().all(|(_, p)| p == &vec![1.0, 0.0]));
        let err = slerp_path(&[1.0, 0.0], &[-1.0, 0.0], 4).unwrap_err();
        assert_eq!(err.code, ErrorCode::InvalidInput);
        for (a, b, steps) in [
            (vec![1.0, 0.0], vec![0.0, 1.0], 1),
            (vec![], vec![], 3),
            (vec![1.0, 0.0], vec![1.0], 3),
            (vec![0.0, 0.0], vec![1.0, 0.0], 3),
            (vec![f32::NAN, 0.0], vec![1.0, 0.0], 3),
        ] {
            assert_eq!(
                slerp_path(&a, &b, steps).unwrap_err().code,
                ErrorCode::InvalidInput,
                "{a:?} {b:?} {steps}"
            );
        }
    }
}
