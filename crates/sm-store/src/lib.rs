//! The SQLite system of record (SPEC section 4).
//!
//! One database file per profile holds the memories, their provenance,
//! anchors and chunks, the FTS5 keyword index, the `vec1` vector table, links,
//! the map cache, the injection records and the capture state. [`Store`] is the only way in:
//! `sm-daemon` and `sm-share` call it directly (there is no store trait,
//! CONTRACTS.md section 2) and it implements
//! [`sm_core::traits::VectorIndex`].
//!
//! # Connections
//!
//! Every connection runs in WAL mode with a 30-second busy timeout, foreign
//! keys on and `synchronous=FULL`. A `Store` holds one writer connection
//! behind a mutex, so writes from one process are serialised before they
//! reach SQLite, and a small pool of reader connections. Every write is one
//! `BEGIN IMMEDIATE` transaction; every read is one deferred transaction, so a
//! search checks the index stamp and reads the vectors from one snapshot.
//! Readers never wait for the writer in WAL mode; a second process waits on
//! the busy timeout rather than failing. A lock still held after 30 seconds
//! is reported as `timeout`, never as a raw lock error.
//!
//! # Vectors
//!
//! Each chunk's embedding is 768 little-endian `f32` values in
//! `chunks.embedding`. Triggers mirror it into the `vec1` table
//! `chunk_vectors` (flat index, cosine distance), which serves cross-project
//! search; project-scoped search reads `chunks` through its `project_id` index
//! and scores in Rust. Both scopes return exactly the brute-force result.
#![deny(unsafe_code)]

mod capture_state;
mod codec;
mod conn;
mod error;
mod index;
mod injections;
mod links;
mod maps;
mod memories;
mod projects;
mod schema;
mod search;
mod store;
#[allow(unsafe_code)]
mod vec1;

#[cfg(test)]
mod tests;

pub use capture_state::{CAPTURE_STATE_RETENTION_DAYS, CaptureProgress, Nearest};
pub use index::PendingChunk;
pub use injections::INJECTION_RETENTION_DAYS;
pub use links::{ANCHOR_EDGE_EXTRACTOR, LinkAdded, WalkHit, head_only};
pub use maps::{MapCluster, MapPointRow, MapRun, NewMapRun};
pub use memories::{
    Access, ContentChange, ForgetCount, InsertOptions, Inserted, MarkRow, MemoryInput,
    MemoryListFilter, NewChunk, ReviseOptions, Revision, SyncMark, UNRESOLVED_SUPERSEDES,
};
pub use projects::ProjectRecord;
pub use schema::SCHEMA_VERSION;
pub use search::{KeywordHit, unit_mean};
pub use store::{MemoryIter, Store};

/// Bytes in one stored embedding: [`sm_core::EMBED_DIM`] little-endian `f32`.
pub const EMBEDDING_BYTES: usize = sm_core::EMBED_DIM * 4;
