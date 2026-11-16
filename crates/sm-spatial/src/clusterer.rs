//! Clustering with `hdbscan` 0.12.0 (SPEC section 7, step 3).

use hdbscan::{DistanceMetric, Hdbscan, HdbscanHyperParams, NnAlgorithm};
use sm_core::traits::{ClusterParams, Clusterer};
use sm_core::{CoreError, Result};

use crate::vector::check_rows;

/// [`Clusterer`] backed by `hdbscan`: Euclidean distance on the projected
/// points, excess-of-mass selection, no single all-points cluster, no
/// epsilon merge, and `max_cluster_size` (the reason SPEC section 7 picked
/// this crate). `min_samples` is passed as
/// [`ClusterParams::rust_min_samples`], because the crate counts the point
/// itself.
///
/// The crate's default `serial` build is used; its `parallel` (rayon)
/// feature stays off, so a run is single-threaded and deterministic.
/// Parameters are checked here before they reach the crate, whose builder
/// would otherwise print a warning to standard output and clamp them (the
/// MCP server speaks on standard output).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HdbscanClusterer;

impl HdbscanClusterer {
    pub fn new() -> Self {
        HdbscanClusterer
    }
}

impl Clusterer for HdbscanClusterer {
    /// Labels each point with a cluster number, or `None` for noise. Cluster
    /// numbers are `0..k` and are only meaningful within this call.
    ///
    /// Fails with `invalid_input` when there are fewer points than
    /// `rust_min_samples()` (the core distance needs that many), the points
    /// differ in length or hold a non-finite value, `min_cluster_size` is
    /// below 2, or `max_cluster_size` is below `min_cluster_size`.
    fn cluster(&self, points: &[Vec<f32>], params: &ClusterParams) -> Result<Vec<Option<usize>>> {
        check_rows(points, "cluster points", false)?;
        if params.min_cluster_size < 2 {
            return Err(CoreError::invalid_input(
                "min_cluster_size must be at least 2",
            ));
        }
        if params.max_cluster_size < params.min_cluster_size {
            return Err(CoreError::invalid_input(format!(
                "max_cluster_size ({}) must be at least min_cluster_size ({})",
                params.max_cluster_size, params.min_cluster_size
            )));
        }
        let min_samples = params.rust_min_samples();
        if points.len() < min_samples {
            return Err(CoreError::invalid_input(format!(
                "clustering needs at least {min_samples} points (min_samples counting the point itself), got {}",
                points.len()
            )));
        }
        let data: Vec<Vec<f64>> = points
            .iter()
            .map(|p| p.iter().map(|&x| f64::from(x)).collect())
            .collect();
        let hp = HdbscanHyperParams::builder()
            .min_cluster_size(params.min_cluster_size)
            .max_cluster_size(params.max_cluster_size)
            .min_samples(min_samples)
            .allow_single_cluster(false)
            .epsilon(0.0)
            .dist_metric(DistanceMetric::Euclidean)
            .nn_algorithm(NnAlgorithm::Auto)
            .build();
        let labels = Hdbscan::new(&data, hp)
            .cluster()
            .map_err(|e| CoreError::invalid_input(format!("clustering failed: {e:?}")))?;
        if labels.len() != points.len() {
            return Err(CoreError::internal(format!(
                "clustering returned {} labels for {} points",
                labels.len(),
                points.len()
            )));
        }
        Ok(labels
            .into_iter()
            .map(|l| usize::try_from(l).ok())
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sm_core::ErrorCode;

    /// Three tight blobs of 20 on a line, plus one far outlier.
    fn blobs() -> Vec<Vec<f32>> {
        let mut pts = Vec::new();
        for c in 0..3 {
            for i in 0..20 {
                let j = i as f32 * 0.01;
                pts.push(vec![c as f32 * 10.0 + j, j * 0.5]);
            }
        }
        pts.push(vec![100.0, 100.0]);
        pts
    }

    #[test]
    fn s7_clusterer_finds_blobs_and_marks_noise() {
        let pts = blobs();
        let params = ClusterParams {
            min_cluster_size: 5,
            min_samples_python: 5,
            max_cluster_size: 30,
        };
        let labels = HdbscanClusterer.cluster(&pts, &params).unwrap();
        assert_eq!(labels.len(), 61);
        assert_eq!(labels[60], None, "the outlier is noise");
        for c in 0..3 {
            let first = labels[c * 20];
            assert!(first.is_some());
            assert!(labels[c * 20..(c + 1) * 20].iter().all(|l| *l == first));
        }
        let distinct: std::collections::BTreeSet<_> = labels.iter().flatten().collect();
        assert_eq!(distinct.len(), 3);
    }

    #[test]
    fn s7_clusterer_max_cluster_size_splits_a_merged_cluster() {
        // Two lines of 20 a little further apart than their own spacing, far
        // from a third: without a cap, excess of mass selects the two close
        // lines as one cluster of 40; capped at 20 it must take them apart.
        let mut pts = Vec::new();
        for i in 0..20 {
            let j = i as f32 * 0.01;
            pts.push(vec![j, 0.0]);
            pts.push(vec![0.25 + j, 0.0]);
            pts.push(vec![1000.0 + j, 0.0]);
        }
        let count = |max_cluster_size| {
            let params = ClusterParams {
                min_cluster_size: 5,
                min_samples_python: 5,
                max_cluster_size,
            };
            let labels = HdbscanClusterer.cluster(&pts, &params).unwrap();
            let distinct: std::collections::BTreeSet<_> =
                labels.iter().flatten().copied().collect();
            (distinct.len(), labels)
        };
        let (uncapped, labels) = count(60);
        assert_eq!(uncapped, 2, "{labels:?}");
        let (capped, labels) = count(20);
        assert_eq!(capped, 3, "{labels:?}");
    }

    #[test]
    fn s7_clusterer_rejects_bad_input() {
        let ok = ClusterParams::for_n(61);
        let cases: Vec<(Vec<Vec<f32>>, ClusterParams)> = vec![
            (vec![], ok),
            (vec![vec![0.0, 0.0]; 5], ok),
            (vec![vec![0.0, f32::NAN]; 30], ok),
            (
                {
                    let mut p = blobs();
                    p[2] = vec![1.0];
                    p
                },
                ok,
            ),
            (
                blobs(),
                ClusterParams {
                    min_cluster_size: 1,
                    ..ok
                },
            ),
            (
                blobs(),
                ClusterParams {
                    max_cluster_size: 3,
                    min_cluster_size: 5,
                    ..ok
                },
            ),
        ];
        for (i, (pts, params)) in cases.into_iter().enumerate() {
            let err = HdbscanClusterer.cluster(&pts, &params).unwrap_err();
            assert_eq!(
                err.code,
                ErrorCode::InvalidInput,
                "case {i}: {}",
                err.message
            );
        }
    }

    #[test]
    fn s7_clusterer_passes_min_samples_plus_one() {
        assert_eq!(ClusterParams::for_n(500).rust_min_samples(), 6);
        // With exactly rust_min_samples points the call succeeds (all noise,
        // since no cluster reaches min_cluster_size 5 twice over).
        let pts: Vec<Vec<f32>> = (0..6).map(|i| vec![i as f32, 0.0]).collect();
        let labels = HdbscanClusterer
            .cluster(&pts, &ClusterParams::for_n(6))
            .unwrap();
        assert_eq!(labels.len(), 6);
    }
}
