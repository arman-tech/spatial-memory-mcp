//! Per-session dedupe (SPEC section 16, Pipeline in the daemon, step 4;
//! AT-14).
//!
//! PostToolUse captures a tool's output as it happens. PreCompact and Stop
//! later read the same session's transcript, which may repeat what was
//! already captured. [`SessionState`] holds the content hash (B-14) of each
//! candidate the session produced, so the transcript pass returns only what
//! was missed.
//!
//! Both sides of the comparison are the redacted content, the text a
//! candidate stores: the hash recorded for a candidate is its
//! `content_hash`, and a new text is hashed after redaction. Version 1
//! compared the unredacted transcript text against redacted queue files, so
//! a text holding a secret never matched its own capture.
//!
//! The dedupe against memories already stored is the daemon's job (B-29);
//! this type only remembers what this session produced.

use std::collections::{HashSet, VecDeque};

use sm_core::hash::{content_hash, python_strip};

/// The most hashes one session keeps. When full, the oldest is forgotten
/// first. At 64 hexadecimal characters a hash, the state stays under about
/// 600 KB a session even with the bookkeeping.
pub const SESSION_HASH_CAP: usize = 4096;

/// The content hashes one session has produced, oldest first, at most
/// [`SESSION_HASH_CAP`] of them.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SessionState {
    order: VecDeque<String>,
    set: HashSet<String>,
}

impl SessionState {
    /// An empty state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether `hash` (a [`content_hash`]) has been recorded.
    pub fn contains(&self, hash: &str) -> bool {
        self.set.contains(hash)
    }

    /// Records `hash`. Returns false when it was already recorded. When the
    /// state is full, the oldest hash is forgotten.
    pub fn record(&mut self, hash: impl Into<String>) -> bool {
        let hash = hash.into();
        if self.set.contains(&hash) {
            return false;
        }
        if self.order.len() >= SESSION_HASH_CAP
            && let Some(oldest) = self.order.pop_front()
        {
            self.set.remove(&oldest);
        }
        self.set.insert(hash.clone());
        self.order.push_back(hash);
        true
    }

    /// Records the content hash of `content`, unless it is blank. Returns
    /// whether it was newly recorded.
    pub fn record_content(&mut self, content: &str) -> bool {
        if python_strip(content).is_empty() {
            return false;
        }
        self.record(content_hash(content))
    }

    /// Version 1's `is_duplicate`: whether `content`'s hash has been
    /// recorded. Blank content is never a duplicate. Case and surrounding
    /// whitespace are ignored, as [`content_hash`] ignores them.
    pub fn is_duplicate(&self, content: &str) -> bool {
        if python_strip(content).is_empty() || self.set.is_empty() {
            return false;
        }
        self.set.contains(&content_hash(content))
    }

    /// How many hashes are recorded.
    pub fn len(&self) -> usize {
        self.order.len()
    }

    /// Whether no hash is recorded.
    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    /// The recorded hashes, oldest first.
    pub fn hashes(&self) -> impl Iterator<Item = &str> {
        self.order.iter().map(String::as_str)
    }
}
