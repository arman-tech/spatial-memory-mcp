//! Automatic capture for Spatial Memory 3.0 (SPEC section 16; CONTRACTS
//! section 7).
//!
//! The daemon hands this crate a hook payload already normalised to Claude
//! Code's field names (PostToolUse, PreCompact or Stop), the capture
//! settings and the project. [`capture`] returns candidate memories, each
//! with redacted content, a type, an importance, tags, provenance and
//! anchors, or the reason there are none. The daemon stores and embeds
//! them.
//!
//! The behaviour is version 1's hook pipeline with version 2's extraction:
//!
//! - [`signal`]: the 22 signal patterns, tiers and importance.
//! - [`secrets`]: the secret scanner ([`RegexSecretScanner`] implements
//!   `sm_core::traits::SecretScanner`) and version 1's skip rule.
//! - [`extract`]: per-tool extraction with one 200,000-character cap, and
//!   the tool skip rules.
//! - [`transcript`]: the stateless transcript delta reader, which offers
//!   the user's messages, compaction summaries, each turn's final text and
//!   the prose of tool text (B-179).
//! - [`select`]: the prose filter for tool text, and the units offered to
//!   the judge (B-179).
//! - [`session`]: per-session content-hash dedupe.
//! - [`candidate`]: the candidate type, tags and anchors.
//! - [`pipeline`]: the steps in order.
//!
//! Reading the transcript file a payload names is the only input or output
//! this crate does.

pub mod candidate;
pub mod extract;
pub mod pipeline;
pub mod secrets;
pub mod select;
pub mod session;
pub mod signal;
pub mod transcript;

pub use candidate::{Candidate, DroppedAnchor, DroppedTag};
pub use pipeline::{
    CaptureInput, CaptureOutcome, REPLAY_TRIGGER, Skip, TextDrop, TextSource, capture, replay,
};
pub use secrets::{RegexSecretScanner, SecretReport};
pub use session::{SESSION_HASH_CAP, SessionState};
pub use signal::{Signal, Tier, classify};

#[cfg(test)]
mod tests;
