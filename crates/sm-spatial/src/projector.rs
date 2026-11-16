//! Projection with `holomap` 0.3.0 (SPEC section 7, step 2).

use holomap::{Holomap, Init, Metric};
use sm_core::traits::{MAX_PROJECTION_POINTS, ProjectParams, Projector};
use sm_core::{CoreError, Result};

use crate::vector::check_rows;

/// [`Projector`] backed by `holomap`: UMAP with the cosine metric, spectral
/// initialisation, `spread` 1 and the library's default epoch count (500 up
/// to 10,000 points), seeded from [`ProjectParams::seed`].
///
/// `holomap` is serial, uses exact nearest neighbours and draws every random
/// number from one seeded PCG64 stream in a fixed order, so the same input
/// and parameters give bit-identical output on every run and thread count.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HolomapProjector;

impl HolomapProjector {
    pub fn new() -> Self {
        HolomapProjector
    }
}

/// Checks [`ProjectParams`] in isolation (not against an input size).
pub(crate) fn check_project_params(params: &ProjectParams, name: &str) -> Result<()> {
    if params.dims == 0 {
        return Err(CoreError::invalid_input(format!(
            "{name}.dims must be at least 1"
        )));
    }
    if params.neighbors < 2 {
        return Err(CoreError::invalid_input(format!(
            "{name}.neighbors must be at least 2"
        )));
    }
    if !params.min_dist.is_finite() || params.min_dist < 0.0 {
        return Err(CoreError::invalid_input(format!(
            "{name}.min_dist must be a finite number of at least 0"
        )));
    }
    Ok(())
}

impl Projector for HolomapProjector {
    /// Projects unit vectors to `params.dims` dimensions, one output row per
    /// input row in the same order.
    ///
    /// Fails with `invalid_input` for more than [`MAX_PROJECTION_POINTS`]
    /// vectors (SPEC section 7: the projection is capped at 2,000), for no
    /// more vectors than `params.neighbors`, for rows of different lengths,
    /// non-finite values or zero vectors, and for invalid parameters.
    fn project(&self, vectors: &[Vec<f32>], params: &ProjectParams) -> Result<Vec<Vec<f32>>> {
        if vectors.len() > MAX_PROJECTION_POINTS {
            return Err(CoreError::invalid_input(format!(
                "projection takes at most {MAX_PROJECTION_POINTS} vectors, got {}",
                vectors.len()
            ))
            .with_next_action(
                "project a sample of at most 2,000 (sm_spatial::regions samples one itself)",
            ));
        }
        check_project_params(params, "project")?;
        let dims = check_rows(vectors, "projection vectors", true)?;
        if vectors.len() <= params.neighbors {
            return Err(CoreError::invalid_input(format!(
                "projection needs more vectors than its {} neighbours, got {}",
                params.neighbors,
                vectors.len()
            ))
            .with_next_action("pass more vectors or lower `neighbors`"));
        }
        let flat: Vec<f32> = vectors.iter().flatten().copied().collect();
        let out = Holomap::builder(params.seed)
            .n_components(params.dims)
            .n_neighbors(params.neighbors)
            .min_dist(params.min_dist)
            .metric(Metric::Cosine)
            .init(Init::Spectral)
            .fit_transform(&flat, dims)
            .map_err(|e| CoreError::invalid_input(format!("projection failed: {e}")))?;
        if out.len() != vectors.len() * params.dims || out.iter().any(|x| !x.is_finite()) {
            return Err(CoreError::internal(format!(
                "projection returned {} values for {} rows of {} dimensions, or a non-finite value",
                out.len(),
                vectors.len(),
                params.dims
            )));
        }
        Ok(out.chunks_exact(params.dims).map(<[f32]>::to_vec).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sm_core::ErrorCode;
    use sm_core::testing::fake_vector;

    fn texts(n: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| fake_vector(&format!("t{}w1 t{}w2 t{}w3 x{i}", i % 3, i % 3, i % 3)))
            .collect()
    }

    #[test]
    fn s7_projector_refuses_more_than_2000_inputs() {
        let v = vec![vec![1.0_f32, 0.0]; MAX_PROJECTION_POINTS + 1];
        let err = HolomapProjector
            .project(&v, &ProjectParams::default())
            .unwrap_err();
        assert_eq!(err.code, ErrorCode::InvalidInput);
        assert!(err.message.contains("2000"), "{}", err.message);
    }

    #[test]
    fn s7_projector_rejects_bad_input() {
        let p = ProjectParams::default();
        let good = texts(20);
        let cases: Vec<(Vec<Vec<f32>>, ProjectParams)> = vec![
            (vec![], p),
            (texts(15), p),
            (
                {
                    let mut v = good.clone();
                    v[3] = vec![1.0; 3];
                    v
                },
                p,
            ),
            (
                {
                    let mut v = good.clone();
                    v[3][0] = f32::INFINITY;
                    v
                },
                p,
            ),
            (
                {
                    let mut v = good.clone();
                    v[3] = vec![0.0; v[0].len()];
                    v
                },
                p,
            ),
            (good.clone(), ProjectParams { dims: 0, ..p }),
            (good.clone(), ProjectParams { neighbors: 1, ..p }),
            (
                good.clone(),
                ProjectParams {
                    min_dist: -0.5,
                    ..p
                },
            ),
            (
                good,
                ProjectParams {
                    min_dist: f32::NAN,
                    ..p
                },
            ),
        ];
        for (i, (v, params)) in cases.into_iter().enumerate() {
            let err = HolomapProjector.project(&v, &params).unwrap_err();
            assert_eq!(
                err.code,
                ErrorCode::InvalidInput,
                "case {i}: {}",
                err.message
            );
        }
    }

    #[test]
    fn at6_projection_is_bit_identical_across_runs() {
        let v = texts(60);
        let p = ProjectParams::default();
        let a = HolomapProjector.project(&v, &p).unwrap();
        let b = HolomapProjector.project(&v, &p).unwrap();
        assert_eq!(a.len(), 60);
        assert!(a.iter().all(|r| r.len() == 5));
        let bits =
            |x: &Vec<Vec<f32>>| -> Vec<u32> { x.iter().flatten().map(|f| f.to_bits()).collect() };
        assert_eq!(bits(&a), bits(&b));
        let other_seed = HolomapProjector
            .project(&v, &ProjectParams { seed: 7, ..p })
            .unwrap();
        assert_ne!(bits(&a), bits(&other_seed), "the seed reaches holomap");
    }

    #[test]
    fn s7_projector_handles_tiny_inputs() {
        for n in [3_usize, 5, 16] {
            let v = texts(n);
            let p = ProjectParams {
                dims: 2,
                neighbors: (n - 1).min(15),
                ..ProjectParams::default()
            };
            let out = HolomapProjector.project(&v, &p).unwrap();
            assert_eq!(out.len(), n);
            assert!(out.iter().flatten().all(|x| x.is_finite()));
        }
    }
}
