//! Identifiers.
//!
//! Memories are identified by UUID (SPEC section 4: version 7 for new
//! memories). Identifiers are what persists: cluster numbers never are
//! (SPEC section 4, Cluster numbers are never keys). Migrated memories keep
//! their version 1 identifier when it is a UUID; any other identifier
//! becomes a deterministic version 8 UUID (decision B-12).

use std::fmt;
use std::str::FromStr;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{CoreError, Result};
use crate::hash::sha256;

/// Namespace prefix for deriving a UUID from a non-UUID version 1 id.
const V1_ID_DOMAIN: &str = "spatial-memory:v1-id:";

/// A memory identifier: any UUID, written in lower-case hyphenated form.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, JsonSchema,
)]
#[serde(transparent)]
pub struct MemoryId(pub Uuid);

impl MemoryId {
    /// A new time-ordered identifier (UUID version 7).
    pub fn new_v7() -> Self {
        MemoryId(Uuid::now_v7())
    }

    /// Parses any UUID text form `uuid` accepts; written back hyphenated.
    pub fn parse(text: &str) -> Result<Self> {
        Uuid::parse_str(text.trim())
            .map(MemoryId)
            .map_err(|e| CoreError::invalid_input(format!("invalid memory id {text:?}: {e}")))
    }

    /// The identifier for a migrated version 1 memory, and whether the
    /// original was kept (`true`) or derived (`false`). A derived identifier
    /// is a UUID version 8 built from the first 16 bytes of
    /// `sha256("spatial-memory:v1-id:" + id)`, so re-running a migration
    /// maps the same record to the same memory.
    pub fn from_v1(id: &str) -> (Self, bool) {
        if let Ok(u) = Uuid::parse_str(id) {
            return (MemoryId(u), true);
        }
        let digest = sha256(format!("{V1_ID_DOMAIN}{id}").as_bytes());
        let mut bytes = [0u8; 16];
        bytes.copy_from_slice(&digest[..16]);
        (MemoryId(Uuid::new_v8(bytes)), false)
    }

    /// The last 8 characters of the hyphenated form, used in shared file
    /// names (SPEC section 9, Path).
    pub fn last8(&self) -> String {
        let s = self.to_string();
        s[s.len() - 8..].to_owned()
    }

    pub fn as_uuid(&self) -> &Uuid {
        &self.0
    }
}

impl fmt::Display for MemoryId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0.hyphenated(), f)
    }
}

impl FromStr for MemoryId {
    type Err = CoreError;

    fn from_str(s: &str) -> Result<Self> {
        MemoryId::parse(s)
    }
}

macro_rules! row_id {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(
            Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize, JsonSchema,
        )]
        #[serde(transparent)]
        pub struct $name(pub i64);

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt::Display::fmt(&self.0, f)
            }
        }
    };
}

row_id!(
    /// Row id of a `projects` row. Local to one store; never written to a
    /// shared file or an export.
    ProjectId
);
row_id!(
    /// Row id of a `chunks` row.
    ChunkId
);
row_id!(
    /// Row id of an `anchors` row.
    AnchorId
);
row_id!(
    /// Row id of a `map_runs` row.
    RunId
);
row_id!(
    /// Row id of a `links` row.
    LinkId
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s4_new_memory_ids_are_uuid_v7() {
        let a = MemoryId::new_v7();
        let b = MemoryId::new_v7();
        assert_eq!(a.0.get_version_num(), 7);
        assert_ne!(a, b);
        let s = a.to_string();
        assert_eq!(s.len(), 36);
        assert_eq!(MemoryId::parse(&s).unwrap(), a);
        assert_eq!(serde_json::to_string(&a).unwrap(), format!("\"{s}\""));
    }

    #[test]
    fn s10_v1_uuid_ids_are_kept() {
        let (id, kept) = MemoryId::from_v1("3f6c2a1e-8b7d-4c59-9e0a-5d1f2b3c4a6e");
        assert!(kept);
        assert_eq!(id.to_string(), "3f6c2a1e-8b7d-4c59-9e0a-5d1f2b3c4a6e");
    }

    #[test]
    fn s10_non_uuid_v1_ids_map_to_stable_v8() {
        let (a, kept) = MemoryId::from_v1("mem-0001");
        assert!(!kept);
        assert_eq!(a.0.get_version_num(), 8);
        assert_eq!(MemoryId::from_v1("mem-0001").0, a);
        assert_ne!(MemoryId::from_v1("mem-0002").0, a);
    }

    #[test]
    fn s9_last8_is_the_id_suffix() {
        let id = MemoryId::parse("0193a8f0-5c1e-7000-8000-3f2a9c1e5b7d").unwrap();
        assert_eq!(id.last8(), "9c1e5b7d");
    }

    #[test]
    fn s4_malformed_memory_ids_are_rejected() {
        assert!(MemoryId::parse("not-a-uuid").is_err());
        assert!(MemoryId::parse("").is_err());
    }
}
