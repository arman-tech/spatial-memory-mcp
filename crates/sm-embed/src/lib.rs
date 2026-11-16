//! Embedding for Spatial Memory 3.0 (SPEC section 5).
//!
//! - [`model`]: the pinned EmbeddingGemma 2 files and the integrity check
//!   the daemon runs before loading (AT-10).
//! - [`download`]: fetches the pinned files over HTTPS, hashing as it writes.
//! - [`gemma`]: [`GemmaProvider`] and its sessions behind
//!   `sm_core::traits::{EmbedderProvider, Embedder}`.
//! - [`chunker`]: the structure-aware [`StructuredChunker`] (SPEC 5,
//!   chunking rules 1 to 6).
//! - [`threads`]: the thread budget and below-normal priority for bulk work
//!   (SPEC 5, "Using the machine's cores").
//! - [`doctor`]: chunks per second and query latency at the configured
//!   budget (SPEC 5, rule 7).
//! - `arctic` (feature `arctic-embed-m`): the fallback model through
//!   fastembed (D-5, B-23).
//!
//! Only the daemon loads the model. Unit tests use the fakes from
//! `sm_core::testing` and the whitespace tokenizer in [`tokenize`].

pub mod chunker;
pub mod doctor;
pub mod download;
pub mod gemma;
pub mod input;
pub mod model;
pub mod os;
pub mod runtime;
pub mod threads;
pub mod tokenize;

#[cfg(feature = "arctic-embed-m")]
pub mod arctic;

pub use chunker::StructuredChunker;
pub use gemma::{EmbedConfig, GemmaEmbedder, GemmaProvider};
pub use model::{MODEL_FILES, ModelFile, VerifyReport, verify};
pub use threads::{ThreadReport, physical_cores};
pub use tokenize::{ModelTokenizer, TextTokenizer, WhitespaceTokenizer};
