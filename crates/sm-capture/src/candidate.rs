//! Candidate memories: what the pipeline returns for the daemon to store
//! (CONTRACTS section 7; SPEC section 16, Type and provenance).
//!
//! A [`Candidate`] carries the redacted content and its content hash, the
//! type from the pattern family, the importance from the tier, the tags,
//! the redaction kinds, the provenance and, for Edit and Write, a `file`
//! anchor. [`Candidate::to_new_memory`] turns it into the store's input
//! without filling anything in.

use serde_json::{Map, Value, json};
use sm_core::ids::ProjectId;
use sm_core::protocol::HookEvent;
use sm_core::types::{
    AnchorKind, AnchorRef, MemoryType, NewMemory, Origin, Provenance, ProvenanceSource, Status,
    normalize_tags,
};

use crate::signal::Tier;

/// The tag every transcript capture carries first (version 1's
/// `_derive_transcript_tags`).
pub const TRANSCRIPT_TAG: &str = "transcript";

/// A tag the pipeline derived but could not keep, because it breaks the
/// list-item rules of B-19.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DroppedTag {
    /// The tag as derived.
    pub tag: String,
    /// Why it was dropped.
    pub reason: String,
}

/// A `file` anchor the pipeline derived but could not keep.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DroppedAnchor {
    /// The anchor name as derived.
    pub name: String,
    /// Why it was dropped.
    pub reason: String,
}

/// One memory the pipeline found worth storing.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    /// The event that produced it.
    pub source_hook: HookEvent,
    /// The redacted text, stored whole.
    pub content: String,
    /// `sm_core::hash::content_hash` of `content` (B-14).
    pub content_hash: String,
    /// From the pattern family: decision, pitfall, convention or note.
    pub memory_type: MemoryType,
    /// Always pending: an automatic capture waits for the judge and is never
    /// served before (B-157; replaces SPEC 16 step 5, both tiers active).
    pub status: Status,
    /// From the tier: at least 0.7 for strong, at least 0.4 for medium.
    pub importance: f64,
    /// The classifier's score, the highest matching confidence.
    pub score: f64,
    /// Strong or medium; weak signals give no candidate.
    pub tier: Tier,
    /// The matching pattern families, first match first.
    pub patterns: Vec<String>,
    /// The kinds the scanner replaced, in order of appearance, repeats kept.
    pub redaction_kinds: Vec<String>,
    /// Valid tags (B-19), in derivation order without repeats.
    pub tags: Vec<String>,
    /// Derived tags that broke B-19 and were left out.
    pub dropped_tags: Vec<DroppedTag>,
    /// Capture as the source, with the session, tool, file path and commit
    /// as far as they are known, and the classification in `extra`.
    pub provenance: Provenance,
    /// A `file` anchor for Edit and Write; empty otherwise.
    pub anchors: Vec<AnchorRef>,
    /// The `file` anchor, when one was derived but broke B-19.
    pub dropped_anchor: Option<DroppedAnchor>,
    /// The namespace version 1 would have suggested (`derive_namespace`),
    /// for comparison and migration.
    pub v1_namespace: &'static str,
    /// Whether the safety cap cut this text at extraction (AT-17).
    pub cap_hit: bool,
}

impl Candidate {
    /// The store's input for this candidate, in project `project_id`: the
    /// content and its hash, type, pending status, importance, tags,
    /// provenance and anchors. The title, identifier and times are left
    /// for the store to fill.
    pub fn to_new_memory(&self, project_id: ProjectId) -> NewMemory {
        let mut memory = NewMemory::new(project_id, self.content.clone(), self.provenance.clone());
        memory.memory_type = self.memory_type;
        memory.status = self.status;
        memory.content_hash = Some(self.content_hash.clone());
        memory.origin = Origin::Local;
        memory.importance = self.importance;
        memory.tags = self.tags.clone();
        memory.anchors = self.anchors.clone();
        memory
    }
}

/// Keeps the tags that pass B-19, in order without repeats, and lists the
/// rest with the reason.
pub fn validate_tags<I, S>(tags: I) -> (Vec<String>, Vec<DroppedTag>)
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut kept: Vec<String> = Vec::new();
    let mut dropped = Vec::new();
    for tag in tags {
        let tag = tag.as_ref();
        match normalize_tags([tag]) {
            Ok(mut valid) => {
                if let Some(t) = valid.pop()
                    && !kept.contains(&t)
                {
                    kept.push(t);
                }
            }
            Err(e) => dropped.push(DroppedTag {
                tag: tag.to_owned(),
                reason: e.message,
            }),
        }
    }
    (kept, dropped)
}

/// Version 1's `_derive_tags` for PostToolUse: the tool name in lower
/// case, then each pattern family not already present.
pub fn derive_tags<S: AsRef<str>>(tool_name: &str, families: &[S]) -> Vec<String> {
    let mut tags: Vec<String> = Vec::new();
    if !tool_name.is_empty() {
        tags.push(tool_name.to_lowercase());
    }
    for family in families {
        let family = family.as_ref();
        if !tags.iter().any(|t| t == family) {
            tags.push(family.to_owned());
        }
    }
    tags
}

/// Version 1's `_derive_transcript_tags`: `transcript`, then each pattern
/// family not already present.
pub fn derive_transcript_tags<S: AsRef<str>>(families: &[S]) -> Vec<String> {
    let mut tags = vec![TRANSCRIPT_TAG.to_owned()];
    for family in families {
        let family = family.as_ref();
        if !tags.iter().any(|t| t == family) {
            tags.push(family.to_owned());
        }
    }
    tags
}

/// A `file` anchor named `name`, or why it cannot be one.
pub fn file_anchor(name: String) -> Result<AnchorRef, DroppedAnchor> {
    AnchorRef::new(AnchorKind::File, name.clone()).map_err(|e| DroppedAnchor {
        name,
        reason: e.message,
    })
}

/// What the pipeline knows about where a capture came from.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct Origins {
    pub session_id: Option<String>,
    pub tool: Option<String>,
    pub file_path: Option<String>,
    pub commit_sha: Option<String>,
    pub trigger: Option<String>,
}

/// The classification and redaction of one text.
pub(crate) struct Classified<'a> {
    pub event: HookEvent,
    pub content: String,
    pub content_hash: String,
    pub tier: Tier,
    pub score: f64,
    pub importance: f64,
    pub families: Vec<String>,
    pub redaction_kinds: Vec<String>,
    pub cap_hit: bool,
    pub tags: Vec<String>,
    pub anchor_name: Option<String>,
    pub origins: &'a Origins,
}

impl Classified<'_> {
    /// Builds the candidate: validated tags and anchor, type from the
    /// families, and provenance with the classification in `extra`.
    pub fn into_candidate(self) -> Candidate {
        let (tags, dropped_tags) = validate_tags(&self.tags);
        let (anchors, dropped_anchor) = match self.anchor_name {
            Some(name) => match file_anchor(name) {
                Ok(anchor) => (vec![anchor], None),
                Err(dropped) => (Vec::new(), Some(dropped)),
            },
            None => (Vec::new(), None),
        };
        let v1_namespace = crate::signal::v1_namespace(&self.families);
        let mut extra = Map::new();
        extra.insert("hook".to_owned(), json!(self.event.as_str()));
        extra.insert("tier".to_owned(), json!(self.tier.number()));
        extra.insert("score".to_owned(), json!(self.score));
        extra.insert("patterns".to_owned(), json!(self.families));
        extra.insert("v1_namespace".to_owned(), json!(v1_namespace));
        if !self.redaction_kinds.is_empty() {
            extra.insert("redactions".to_owned(), json!(self.redaction_kinds));
        }
        if self.cap_hit {
            extra.insert("truncated".to_owned(), Value::Bool(true));
        }
        if let Some(trigger) = &self.origins.trigger {
            extra.insert("trigger".to_owned(), json!(trigger));
        }
        let mut provenance = Provenance::new(ProvenanceSource::Capture);
        provenance.session_id = self.origins.session_id.clone();
        provenance.tool = self.origins.tool.clone();
        provenance.file_path = self.origins.file_path.clone();
        provenance.commit_sha = self.origins.commit_sha.clone();
        provenance.extra = Some(Value::Object(extra));
        Candidate {
            source_hook: self.event,
            memory_type: crate::signal::memory_type_for(&self.families),
            // Unjudged: stored, never served, until the judge keeps it
            // (B-157).
            status: Status::Pending,
            content: self.content,
            content_hash: self.content_hash,
            importance: self.importance,
            score: self.score,
            tier: self.tier,
            patterns: self.families,
            redaction_kinds: self.redaction_kinds,
            tags,
            dropped_tags,
            provenance,
            anchors,
            dropped_anchor,
            v1_namespace,
            cap_hit: self.cap_hit,
        }
    }
}
