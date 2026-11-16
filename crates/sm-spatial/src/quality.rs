//! Clustering quality measures for the AT-7 report.

/// The adjusted Rand index between a ground truth and a clustering, as
/// scikit-learn's `adjusted_rand_score` computes it (the pair-confusion
/// form). Unclustered points (`None`) count as one more label, as the
/// SPEC's measurement did with HDBSCAN's `-1`.
///
/// 1.0 is a perfect match (up to renaming), about 0.0 is chance, and it can
/// be negative. Two empty or all-identical clusterings score 1.0.
///
/// # Panics
///
/// When the two slices differ in length.
pub fn adjusted_rand_index(truth: &[usize], predicted: &[Option<usize>]) -> f64 {
    assert_eq!(
        truth.len(),
        predicted.len(),
        "truth and predicted label one point each"
    );
    let n = truth.len() as i128;
    let mut t_labels: Vec<usize> = truth.to_vec();
    t_labels.sort_unstable();
    t_labels.dedup();
    let mut p_labels: Vec<Option<usize>> = predicted.to_vec();
    p_labels.sort_unstable();
    p_labels.dedup();
    let mut table = vec![vec![0_i128; p_labels.len()]; t_labels.len()];
    for (t, p) in truth.iter().zip(predicted) {
        let (Ok(i), Ok(j)) = (t_labels.binary_search(t), p_labels.binary_search(p)) else {
            continue;
        };
        table[i][j] += 1;
    }
    let sum_sq: i128 = table.iter().flatten().map(|&x| x * x).sum();
    let rows_sq: i128 = table
        .iter()
        .map(|row| row.iter().sum::<i128>().pow(2))
        .sum();
    let cols_sq: i128 = (0..p_labels.len())
        .map(|j| table.iter().map(|row| row[j]).sum::<i128>().pow(2))
        .sum();
    // Pair confusion matrix (ordered pairs, as scikit-learn counts them).
    let tp = sum_sq - n;
    let fp = cols_sq - sum_sq;
    let fn_ = rows_sq - sum_sq;
    let tn = n * n - fp - fn_ - sum_sq;
    if fn_ == 0 && fp == 0 {
        return 1.0;
    }
    let num = 2.0 * ((tp * tn - fn_ * fp) as f64);
    let den = ((tp + fn_) * (fn_ + tn) + (tp + fp) * (fp + tn)) as f64;
    num / den
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn at7_ari_matches_scikit_learn() {
        // Values from sklearn.metrics.adjusted_rand_score.
        assert_eq!(
            adjusted_rand_index(&[0, 0, 1, 1], &[Some(1), Some(1), Some(0), Some(0)]),
            1.0
        );
        let v = adjusted_rand_index(&[0, 0, 1, 2], &[Some(0), Some(0), Some(1), Some(1)]);
        assert!((v - 0.571_428_571_428_571_5).abs() < 1e-12, "{v}");
        assert_eq!(
            adjusted_rand_index(&[0, 0, 0, 0], &[Some(0), Some(1), Some(2), Some(3)]),
            0.0
        );
        let v = adjusted_rand_index(&[0, 0, 1, 1], &[Some(0), Some(1), Some(0), Some(1)]);
        assert!((v - -0.5).abs() < 1e-12, "{v}");
        // Noise is one more label.
        let v = adjusted_rand_index(&[0, 0, 1, 1], &[None, None, Some(0), Some(0)]);
        assert_eq!(v, 1.0);
        assert_eq!(adjusted_rand_index(&[], &[]), 1.0);
    }
}
