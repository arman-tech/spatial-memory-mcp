//! Vector helpers shared by the pipeline and journey.
//!
//! Sums run in `f64` in a fixed order, so every result is bit-identical on
//! every run and thread count; `f64::sqrt` is correctly rounded on every
//! platform.

use sm_core::{CoreError, Result};

/// One vector per memory (SPEC section 7, step 1): the mean of its chunk
/// vectors, re-normalised to unit length.
///
/// Fails with `invalid_input` when there are no chunks, the chunks differ in
/// length, a value is not finite, or the mean is the zero vector.
pub fn memory_vector(chunks: &[Vec<f32>]) -> Result<Vec<f32>> {
    let Some(first) = chunks.first() else {
        return Err(CoreError::invalid_input("a memory has no chunk vectors")
            .with_next_action("leave out memories whose embedding is still pending"));
    };
    let dims = first.len();
    if dims == 0 {
        return Err(CoreError::invalid_input("a chunk vector is empty"));
    }
    let mut sum = vec![0.0_f64; dims];
    for chunk in chunks {
        if chunk.len() != dims {
            return Err(CoreError::invalid_input(format!(
                "chunk vectors of one memory differ in length ({} and {dims})",
                chunk.len()
            )));
        }
        for (s, &x) in sum.iter_mut().zip(chunk) {
            if !x.is_finite() {
                return Err(
                    CoreError::invalid_input("a chunk vector holds a non-finite value")
                        .with_next_action("re-embed the memory with `spatial-memory reindex`"),
                );
            }
            *s += f64::from(x);
        }
    }
    unit(&sum).ok_or_else(|| {
        CoreError::invalid_input("a memory's chunk vectors sum to the zero vector")
            .with_next_action("re-embed the memory with `spatial-memory reindex`")
    })
}

/// Cosine similarity, accumulated in `f64`. Returns 0 when either vector has
/// zero length.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0.0_f64, 0.0_f64, 0.0_f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (f64::from(x), f64::from(y));
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

/// The dot product in `f32` over eight fixed lanes, for the bulk
/// comparisons of unit vectors (non-sample map positions). The order of
/// every addition is fixed by the code, and Rust never fuses or reorders
/// floating-point operations, so the result is the same on every platform;
/// the lanes only let the compiler use vector registers.
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut lanes = [0.0_f32; 8];
    let (mut ca, mut cb) = (a.chunks_exact(8), b.chunks_exact(8));
    for (x, y) in (&mut ca).zip(&mut cb) {
        for ((s, &p), &q) in lanes.iter_mut().zip(x).zip(y) {
            *s += p * q;
        }
    }
    let mut tail = 0.0_f32;
    for (&p, &q) in ca.remainder().iter().zip(cb.remainder()) {
        tail += p * q;
    }
    ((lanes[0] + lanes[4]) + (lanes[1] + lanes[5]))
        + ((lanes[2] + lanes[6]) + (lanes[3] + lanes[7]))
        + tail
}

/// `v` scaled to unit length, or `None` for the zero vector.
pub(crate) fn unit(v: &[f64]) -> Option<Vec<f32>> {
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm == 0.0 || !norm.is_finite() {
        return None;
    }
    Some(v.iter().map(|x| (x / norm) as f32).collect())
}

/// The re-normalised mean of the given rows of `vectors`, or the zero vector
/// when they cancel out (no real cluster does; the caller then ranks by id).
pub(crate) fn centre(vectors: &[Vec<f32>], rows: &[usize], dims: usize) -> Vec<f32> {
    let mut sum = vec![0.0_f64; dims];
    for &r in rows {
        for (s, &x) in sum.iter_mut().zip(&vectors[r]) {
            *s += f64::from(x);
        }
    }
    unit(&sum).unwrap_or_else(|| vec![0.0; dims])
}

/// Checks that `rows` is non-empty, every row has the same non-zero length,
/// every value is finite and (when `nonzero`) no row is the zero vector.
/// Returns the row length.
pub(crate) fn check_rows(rows: &[Vec<f32>], what: &str, nonzero: bool) -> Result<usize> {
    let Some(first) = rows.first() else {
        return Err(CoreError::invalid_input(format!("no {what} were given")));
    };
    let dims = first.len();
    if dims == 0 {
        return Err(CoreError::invalid_input(format!(
            "{what} have no dimensions"
        )));
    }
    for (i, row) in rows.iter().enumerate() {
        if row.len() != dims {
            return Err(CoreError::invalid_input(format!(
                "{what} differ in length: row 0 has {dims} values, row {i} has {}",
                row.len()
            )));
        }
        if row.iter().any(|x| !x.is_finite()) {
            return Err(CoreError::invalid_input(format!(
                "{what}: row {i} holds a non-finite value"
            )));
        }
        if nonzero && row.iter().all(|&x| x == 0.0) {
            return Err(CoreError::invalid_input(format!(
                "{what}: row {i} is the zero vector, which has no direction"
            )));
        }
    }
    Ok(dims)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sm_core::ErrorCode;

    #[test]
    fn s7_memory_vector_is_the_renormalised_mean_of_its_chunks() {
        let v = memory_vector(&[vec![1.0, 0.0], vec![0.0, 1.0]]).unwrap();
        let h = std::f32::consts::FRAC_1_SQRT_2;
        assert!((v[0] - h).abs() < 1e-7 && (v[1] - h).abs() < 1e-7);
        let one = memory_vector(&[vec![3.0, 4.0]]).unwrap();
        assert!((one[0] - 0.6).abs() < 1e-7 && (one[1] - 0.8).abs() < 1e-7);
    }

    #[test]
    fn s7_memory_vector_rejects_bad_chunks() {
        for bad in [
            vec![],
            vec![vec![]],
            vec![vec![1.0, 0.0], vec![1.0]],
            vec![vec![f32::NAN, 1.0]],
            vec![vec![1.0, 0.0], vec![-1.0, 0.0]],
        ] {
            let err = memory_vector(&bad).unwrap_err();
            assert_eq!(err.code, ErrorCode::InvalidInput, "{bad:?}");
        }
    }

    #[test]
    fn s7_dot_matches_cosine_for_unit_vectors() {
        // 11 values: one block of 8 lanes and a tail of 3.
        let a: Vec<f64> = (0..11).map(|i| f64::from(i) - 4.5).collect();
        let b: Vec<f64> = (0..11).map(|i| f64::from(i * i % 7) + 0.25).collect();
        let (a, b) = (unit(&a).unwrap(), unit(&b).unwrap());
        assert!((dot(&a, &b) - cosine(&a, &b)).abs() < 1e-6);
        assert!((dot(&a, &a) - 1.0).abs() < 1e-6);
        assert_eq!(dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0);
    }

    #[test]
    fn s7_cosine_handles_scale_and_zero() {
        assert!((cosine(&[2.0, 0.0], &[5.0, 0.0]) - 1.0).abs() < 1e-7);
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-7);
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 0.0]), 0.0);
    }
}
