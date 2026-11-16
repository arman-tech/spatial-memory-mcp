//! Hashing rules several crates must agree on.
//!
//! `content_hash` reproduces version 1 exactly (`core/hashing.py`: SHA-256 hex
//! of `content.strip().lower()`), so migrated hashes are copied unchanged
//! (SPEC section 10) and duplicate detection works across the migration
//! boundary (SPEC section 16, pipeline step 4).

use sha2::{Digest, Sha256};

const HEX: &[u8; 16] = b"0123456789abcdef";

/// Lower-case hex of a byte slice.
pub fn to_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

/// SHA-256 of `bytes` as 64 lower-case hex characters.
pub fn sha256_hex(bytes: &[u8]) -> String {
    to_hex(Sha256::digest(bytes).as_slice())
}

/// The raw SHA-256 digest.
pub fn sha256(bytes: &[u8]) -> [u8; 32] {
    let digest = Sha256::digest(bytes);
    let mut out = [0u8; 32];
    out.copy_from_slice(digest.as_slice());
    out
}

/// Python's `str.isspace` for one character: Rust's Unicode `White_Space`
/// plus the four information separators U+001C to U+001F, which Python
/// counts as whitespace and Rust does not.
pub fn is_python_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// Python's `str.strip()` with no arguments.
pub fn python_strip(s: &str) -> &str {
    s.trim_matches(is_python_space)
}

/// Version 1's content hash: SHA-256 hex of the stripped, lower-cased text.
pub fn content_hash(content: &str) -> String {
    sha256_hex(python_strip(content).to_lowercase().as_bytes())
}

/// 64-bit FNV-1a. Not cryptographic; used where a stable, cheap,
/// dependency-free hash of a short string is enough.
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s10_sha256_hex_matches_known_vector() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(to_hex(&sha256(b"abc")), sha256_hex(b"abc"));
    }

    #[test]
    fn s10_python_strip_matches_python_isspace() {
        assert_eq!(python_strip("  \t\n x y \r\n"), "x y");
        assert_eq!(python_strip("\u{1c}\u{1f}a\u{1d}"), "a");
        assert_eq!(python_strip("\u{3000}a\u{a0}"), "a");
        // U+200B ZERO WIDTH SPACE is not whitespace for Python or Rust.
        assert_eq!(python_strip("\u{200b}a"), "\u{200b}a");
    }

    #[test]
    fn s10_content_hash_matches_v1_export_fixture() {
        // AT-8: version 1 content hashes are copied unchanged, so the rule
        // must reproduce the fixture's hashes byte for byte.
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/migration/v1_export_full.json"
        );
        let text = std::fs::read_to_string(path).unwrap();
        let records: Vec<serde_json::Value> = serde_json::from_str(&text).unwrap();
        assert_eq!(records.len(), 6);
        for rec in records {
            let content = rec["content"].as_str().unwrap();
            let expected = rec["content_hash"].as_str().unwrap();
            assert_eq!(content_hash(content), expected, "record {}", rec["id"]);
        }
    }

    #[test]
    fn s10_content_hash_ignores_case_and_surrounding_space() {
        assert_eq!(content_hash("  Hello World\n"), content_hash("hello world"));
        assert_ne!(content_hash("hello world"), content_hash("hello  world"));
    }

    #[test]
    fn s14_fake_embedder_hash_fnv1a64_matches_known_vectors() {
        assert_eq!(fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a64(b"a"), 0xaf63_dc4c_8601_ec8c);
    }
}
