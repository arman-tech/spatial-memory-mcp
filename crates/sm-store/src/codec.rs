//! Column encodings: vectors as little-endian `f32` blobs, timestamps as
//! microseconds, enums as their SPEC strings, tags as a JSON array.

use std::str::FromStr;

use rusqlite::Row;
use sm_core::EMBED_DIM;
use sm_core::error::{CoreError, Result};
use sm_core::ids::MemoryId;
use sm_core::time::Timestamp;

use crate::EMBEDDING_BYTES;
use crate::error::conversion;

/// How far a stored or query vector's squared length may be from 1. The
/// cross-project search relies on cosine and dot product ranking alike,
/// which holds only for unit vectors (the `Embedder` contract).
pub(crate) const UNIT_TOLERANCE: f32 = 1e-3;

/// Checks a vector before it is stored or searched with: 768 finite values
/// of unit length.
pub(crate) fn check_vector(what: &str, v: &[f32]) -> Result<()> {
    if v.len() != EMBED_DIM {
        return Err(CoreError::invalid_input(format!(
            "{what} has {} dimensions; the index holds {EMBED_DIM}",
            v.len()
        )));
    }
    if let Some(i) = v.iter().position(|x| !x.is_finite()) {
        return Err(CoreError::invalid_input(format!(
            "{what} has a non-finite value at position {i}"
        )));
    }
    let norm2: f32 = v.iter().map(|x| x * x).sum();
    if (norm2 - 1.0).abs() > UNIT_TOLERANCE {
        return Err(CoreError::invalid_input(format!(
            "{what} is not unit length (squared length {norm2})"
        )));
    }
    Ok(())
}

/// The stored form of a vector: little-endian `f32`, 3,072 bytes.
pub(crate) fn vector_to_blob(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// A stored vector read back.
pub(crate) fn blob_to_vector(blob: &[u8]) -> Result<Vec<f32>> {
    if blob.len() != EMBEDDING_BYTES {
        return Err(crate::error::corrupt(
            "chunk embedding",
            format!("{} bytes, expected {EMBEDDING_BYTES}", blob.len()),
        ));
    }
    Ok(blob
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

/// Dot product of a stored vector with `q`, read straight from the blob.
/// Eight running sums let the compiler vectorise the loop; the result is
/// deterministic for a given input. `blob` must hold `q.len()` values.
pub(crate) fn dot_blob(blob: &[u8], q: &[f32]) -> f32 {
    debug_assert_eq!(blob.len(), q.len() * 4);
    let mut acc = [0f32; 8];
    let blocks = blob.chunks_exact(32).zip(q.chunks_exact(8));
    for (b, qq) in blocks {
        for i in 0..8 {
            let x = f32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]]);
            acc[i] += x * qq[i];
        }
    }
    let mut sum = ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]));
    let done = q.len() / 8 * 8;
    for (b, x) in blob[done * 4..].chunks_exact(4).zip(&q[done..]) {
        sum += f32::from_le_bytes([b[0], b[1], b[2], b[3]]) * x;
    }
    sum
}

pub(crate) fn ts(t: Timestamp) -> i64 {
    t.as_micros()
}

pub(crate) fn from_ts(micros: i64) -> Timestamp {
    Timestamp::from_micros(micros)
}

/// Reads a column holding an enum's SPEC string or a memory id.
pub(crate) fn parse_col<T>(row: &Row<'_>, idx: usize) -> rusqlite::Result<T>
where
    T: FromStr<Err = CoreError>,
{
    let s: String = row.get(idx)?;
    s.parse().map_err(|e| conversion(idx, e))
}

/// Reads an optional column holding an enum's SPEC string or a memory id.
pub(crate) fn parse_opt_col<T>(row: &Row<'_>, idx: usize) -> rusqlite::Result<Option<T>>
where
    T: FromStr<Err = CoreError>,
{
    let s: Option<String> = row.get(idx)?;
    s.map(|s| s.parse().map_err(|e| conversion(idx, e)))
        .transpose()
}

pub(crate) fn memory_id_col(row: &Row<'_>, idx: usize) -> rusqlite::Result<MemoryId> {
    parse_col(row, idx)
}

/// Reads a JSON array of strings (the `tags` column).
pub(crate) fn tags_col(row: &Row<'_>, idx: usize) -> rusqlite::Result<Vec<String>> {
    let s: String = row.get(idx)?;
    serde_json::from_str(&s).map_err(|e| conversion(idx, CoreError::from(e)))
}

/// Reads a JSON column.
pub(crate) fn json_value_col(row: &Row<'_>, idx: usize) -> rusqlite::Result<serde_json::Value> {
    let s: String = row.get(idx)?;
    serde_json::from_str(&s).map_err(|e| conversion(idx, CoreError::from(e)))
}

/// Reads an optional JSON column.
pub(crate) fn opt_json_col(
    row: &Row<'_>,
    idx: usize,
) -> rusqlite::Result<Option<serde_json::Value>> {
    let s: Option<String> = row.get(idx)?;
    s.map(|s| serde_json::from_str(&s).map_err(|e| conversion(idx, CoreError::from(e))))
        .transpose()
}

pub(crate) fn tags_json(tags: &[String]) -> Result<String> {
    Ok(serde_json::to_string(tags)?)
}

/// Converts a count from SQLite (always non-negative here) to `u64`.
pub(crate) fn count(n: i64) -> u64 {
    u64::try_from(n).unwrap_or(0)
}

/// Converts a limit to an SQL parameter, saturating.
pub(crate) fn sql_limit(n: usize) -> i64 {
    i64::try_from(n).unwrap_or(i64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s4_vectors_are_768_little_endian_f32() {
        let mut v = vec![0f32; EMBED_DIM];
        v[0] = 0.6;
        v[767] = -0.8;
        let blob = vector_to_blob(&v);
        assert_eq!(blob.len(), 3072);
        assert_eq!(&blob[0..4], &0.6f32.to_le_bytes());
        assert_eq!(&blob[3068..3072], &(-0.8f32).to_le_bytes());
        assert_eq!(blob_to_vector(&blob).unwrap(), v);
        assert!(blob_to_vector(&blob[..3068]).is_err());
        assert!(check_vector("v", &v).is_ok());
        let simple: f32 = v.iter().zip(&v).map(|(a, b)| a * b).sum();
        assert!((dot_blob(&blob, &v) - simple).abs() < 1e-6);
    }

    #[test]
    fn s4_vectors_must_be_unit_length_and_finite() {
        assert!(check_vector("v", &[1.0, 0.0]).is_err());
        let mut v = vec![0f32; EMBED_DIM];
        assert!(check_vector("v", &v).is_err());
        v[3] = f32::NAN;
        assert!(check_vector("v", &v).is_err());
        v[3] = 2.0;
        assert!(check_vector("v", &v).is_err());
    }
}
