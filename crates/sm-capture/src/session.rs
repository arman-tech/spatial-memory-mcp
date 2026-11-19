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
//!
//! It also remembers the sentences of the compaction summaries the session
//! offered (B-189): a later summary repeats much of an earlier one, and a
//! summary unit most of whose sentences an earlier summary unit of the
//! session already offered is not offered again
//! ([`SessionState::summary_repeats`]). The state is held in memory only, so
//! a restarted daemon offers a repeated summary unit again; the judge and
//! the duplicate check at activation then settle it.

use std::collections::{HashSet, VecDeque};
use std::hash::{DefaultHasher, Hash, Hasher};

use sm_core::hash::{content_hash, python_strip};

use crate::quote::sentence_keys;

/// The most hashes one session keeps. When full, the oldest is forgotten
/// first. At 64 hexadecimal characters a hash, the state stays under about
/// 600 KB a session even with the bookkeeping.
pub const SESSION_HASH_CAP: usize = 4096;

/// The most summary sentence keys one session keeps (B-189). When full,
/// the oldest is forgotten first. Each is held as a 64-bit hash, so the
/// keys stay under about 200 KB a session; a compaction summary has a few
/// hundred sentences.
pub const SUMMARY_KEY_CAP: usize = 8192;

/// A compaction summary unit is not offered when at least this share of its
/// sentence keys were in summary units the session offered before it
/// (B-189). Chosen on the design data (CAPTURE-RCA 8.9) as the lowest
/// share that loses no worth-keeping group no other candidate holds: at
/// 0.7 the filter drops 66 of 3,136 candidates and none; at 0.6 it drops
/// 97 and one group; at 0.5, 150 and nine, two of them must keep.
pub const SUMMARY_REPEAT: f64 = 0.7;

/// The content hashes one session has produced, oldest first, at most
/// [`SESSION_HASH_CAP`] of them, and the sentence keys of the compaction
/// summary units it offered, at most [`SUMMARY_KEY_CAP`] of them.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SessionState {
    order: VecDeque<String>,
    set: HashSet<String>,
    summary_order: VecDeque<u64>,
    summary_keys: HashSet<u64>,
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

    /// Whether a compaction summary unit repeats the summary units this
    /// session offered before it (B-189): whether at least
    /// [`SUMMARY_REPEAT`] of its sentence keys
    /// ([`crate::quote::sentence_keys`]) were recorded. Its keys are then
    /// recorded, whatever the answer. A unit without a sentence key repeats
    /// nothing.
    pub fn summary_repeats(&mut self, content: &str) -> bool {
        let keys: Vec<u64> = sentence_keys(content)
            .iter()
            .map(|k| {
                let mut h = DefaultHasher::new();
                k.hash(&mut h);
                h.finish()
            })
            .collect();
        if keys.is_empty() {
            return false;
        }
        let seen = keys
            .iter()
            .filter(|k| self.summary_keys.contains(k))
            .count();
        let repeats = seen as f64 / keys.len() as f64 >= SUMMARY_REPEAT;
        for k in keys {
            if !self.summary_keys.insert(k) {
                continue;
            }
            self.summary_order.push_back(k);
            if self.summary_order.len() > SUMMARY_KEY_CAP
                && let Some(oldest) = self.summary_order.pop_front()
            {
                self.summary_keys.remove(&oldest);
            }
        }
        repeats
    }
}
