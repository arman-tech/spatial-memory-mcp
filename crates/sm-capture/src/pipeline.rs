//! The capture pipeline the daemon runs for PostToolUse, PreCompact and
//! Stop (SPEC section 16, Pipeline in the daemon, steps 1 to 5;
//! CONTRACTS section 7).
//!
//! [`capture`] takes a payload already normalised to Claude Code's field
//! names, the capture settings and the project, and returns candidate
//! memories or the reason there are none. It stores nothing and embeds
//! nothing: the daemon does both, and also drops candidates whose content
//! hash a stored memory already has (B-29).
//!
//! The steps, per text:
//!
//! 1. Extract: the tool call (PostToolUse), the final assistant message
//!    (Stop), or the assistant text in the transcript since the offset
//!    (PreCompact, and Stop when the payload has no final message). One
//!    safety cap of 200,000 characters with a `...` marker.
//! 2. Classify with the 22 version 1 patterns. Weak signals are dropped.
//! 3. Redact. Version 1's skip rule drops a text holding a private key or
//!    mostly made of secrets.
//! 4. Drop a text whose redacted content hash this session already
//!    produced ([`SessionState`]).
//! 5. Build the candidate: type from the family, importance from the tier,
//!    tags, provenance, a `file` anchor for Edit and Write, and the context
//!    the judge reads with the text: the turn's user prompt, redacted and
//!    bounded to 2,000 characters, empty for PostToolUse (B-166).

use serde_json::Value;
use sm_core::config::CaptureSettings;
use sm_core::hash::{content_hash, python_strip};
use sm_core::project::{ProjectIdentity, normalize_file_anchor};
use sm_core::protocol::HookEvent;
use sm_core::traits::SecretScanner;

use crate::candidate::{Candidate, Classified, Origins, derive_tags, derive_transcript_tags};
use crate::extract::{
    Extracted, MAX_CAPTURE_CHARS, ToolSkip, extract_content, python_truthy, skip_rule,
    tool_response_text, truncate,
};
use crate::secrets::{RegexSecretScanner, SecretReport};
use crate::session::SessionState;
use crate::signal::{Signal, Tier, classify, importance};
use crate::transcript::{
    AssistantTexts, TranscriptEntry, TranscriptRead, extract_assistant_texts, last_prompt,
    read_turns, validate_transcript_path,
};

/// Stop's trigger when the payload gives none (version 2's dispatcher).
pub const DEFAULT_STOP_TRIGGER: &str = "session_end";

/// The provenance trigger of a candidate from [`replay`] (B-172).
pub const REPLAY_TRIGGER: &str = "replay";

/// What one capture request needs besides the session state.
#[derive(Debug, Clone, Copy)]
pub struct CaptureInput<'a> {
    /// The hook event.
    pub event: HookEvent,
    /// The hook's input, normalised to Claude Code's field names.
    pub payload: &'a Value,
    /// The profile's `[capture]` settings.
    pub settings: &'a CaptureSettings,
    /// The project the payload belongs to. Its `capture_enabled` carries
    /// the `project.toml` opt-out, and its `root` is what file anchors are
    /// made relative to.
    pub project: &'a ProjectIdentity,
    /// The project's current commit, for provenance, when the caller knows
    /// it. This crate reads no repository files.
    pub commit_sha: Option<&'a str>,
    /// The transcript byte offset this session reached last time; `None`
    /// for a session never read, which reads only its live segment (B-162).
    pub transcript_offset: Option<u64>,
}

/// Why a whole request gave nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Skip {
    /// `capture.enabled` is off.
    CaptureDisabled,
    /// The project's `project.toml` opts out of capture.
    ProjectOptedOut,
    /// The event is not one capture handles (SessionStart,
    /// UserPromptSubmit).
    NotACaptureEvent(HookEvent),
    /// PostToolUse without a tool name.
    NoToolName,
    /// The tool is not in `capture.tools`.
    ToolNotCaptured(String),
    /// Version 1's skip rules: a read-only tool or this server's own tool.
    ToolSkipped(ToolSkip),
    /// The file the tool touched matches a `capture.ignore` pattern.
    PathIgnored(String),
    /// Stop with `stop_hook_active` set: the loop guard.
    StopHookActive,
    /// No usable transcript path where the transcript had to be read.
    NoTranscript,
}

/// Why one text gave no candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextDrop {
    /// Empty or blank.
    Empty,
    /// The signal was weak (score below 0.5).
    Weak,
    /// The score was not a finite number.
    NonFiniteScore,
    /// A private key, or mostly secrets (version 1's `should_skip`).
    Secret,
    /// This session already produced the same content hash.
    Duplicate,
}

/// Where the texts came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextSource {
    /// The tool call (PostToolUse).
    ToolCall,
    /// The payload's `last_assistant_message` (Stop).
    LastAssistantMessage,
    /// The transcript since the offset (PreCompact; Stop without a final
    /// message).
    Transcript,
}

/// What one capture request produced.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CaptureOutcome {
    /// The candidates, in the order their texts appeared.
    pub candidates: Vec<Candidate>,
    /// Why the request gave nothing at all, when it stopped before any text.
    pub skipped: Option<Skip>,
    /// Where the texts came from, when any were looked at.
    pub source: Option<TextSource>,
    /// One entry per text that gave no candidate, in order.
    pub drops: Vec<TextDrop>,
    /// The transcript offset to keep for the next read, when the transcript
    /// was read. `None` means keep the previous one.
    pub new_offset: Option<u64>,
    /// Whether the safety cap cut any text.
    pub cap_hit: bool,
    /// The characters extracted before the safety cap, over all texts: the
    /// size the capture metrics record before the cap (SPEC section 16,
    /// Index cap).
    pub extracted_chars: usize,
}

impl CaptureOutcome {
    fn skip(reason: Skip) -> Self {
        CaptureOutcome {
            skipped: Some(reason),
            ..CaptureOutcome::default()
        }
    }
}

/// The pipeline's steps that do the work, apart from the order they run
/// in. [`capture`] uses [`RealStages`]; the fixture tests substitute version
/// 1's test fakes to check the order and the gating alone.
pub(crate) trait Stages {
    /// PostToolUse extraction.
    fn extract(&mut self, tool_name: &str, tool_input: &Value, response: &str) -> Extracted;
    /// The final text of each turn since `offset` (none for an unseen
    /// session), or `None` when `path` is not a transcript this crate may
    /// read.
    fn read(&mut self, path: &str, offset: Option<u64>) -> Option<TranscriptRead>;
    /// The assistant texts of the entries read, with the safety cap.
    fn assistant_texts(&mut self, entries: &[TranscriptEntry]) -> AssistantTexts;
    /// Classification.
    fn classify(&mut self, text: &str) -> Signal;
    /// Scanning and redaction, with the skip rule.
    fn redact(&mut self, text: &str) -> SecretReport;
    /// Whether this session already produced `content` (redacted).
    fn is_duplicate(&mut self, session: &SessionState, content: &str) -> bool;
}

/// The real steps.
pub(crate) struct RealStages;

impl Stages for RealStages {
    fn extract(&mut self, tool_name: &str, tool_input: &Value, response: &str) -> Extracted {
        extract_content(tool_name, tool_input, response)
    }

    fn read(&mut self, path: &str, offset: Option<u64>) -> Option<TranscriptRead> {
        validate_transcript_path(path)?;
        Some(read_turns(path, offset))
    }

    fn assistant_texts(&mut self, entries: &[TranscriptEntry]) -> AssistantTexts {
        extract_assistant_texts(entries)
    }

    fn classify(&mut self, text: &str) -> Signal {
        classify(text)
    }

    fn redact(&mut self, text: &str) -> SecretReport {
        RegexSecretScanner::new().report(text)
    }

    fn is_duplicate(&mut self, session: &SessionState, content: &str) -> bool {
        session.is_duplicate(content)
    }
}

/// Runs the pipeline for one hook event. Records each candidate's content
/// hash in `session`, so a later PreCompact or Stop in the same session
/// returns only what was missed (AT-14).
pub fn capture(input: &CaptureInput<'_>, session: &mut SessionState) -> CaptureOutcome {
    capture_with(input, session, &mut RealStages)
}

/// [`capture`] with the steps given.
pub(crate) fn capture_with(
    input: &CaptureInput<'_>,
    session: &mut SessionState,
    stages: &mut dyn Stages,
) -> CaptureOutcome {
    if !input.settings.enabled {
        return CaptureOutcome::skip(Skip::CaptureDisabled);
    }
    if !input.project.capture_enabled {
        return CaptureOutcome::skip(Skip::ProjectOptedOut);
    }
    match input.event {
        HookEvent::PostToolUse => post_tool_use(input, session, stages),
        HookEvent::PreCompact => transcript(input, session, stages, None),
        HookEvent::Stop => stop(input, session, stages),
        other => CaptureOutcome::skip(Skip::NotACaptureEvent(other)),
    }
}

/// Runs a session's stored transcript through the pipeline again, on
/// request only (SPEC 16, Replay; B-172). The transcript is read as for a
/// session seen from its start: every turn's final text, before and after
/// each compaction boundary, with the prompt that started its turn as its
/// context. Each text has the per-text cap of [`MAX_CAPTURE_CHARS`]; there
/// is no cap over all of them together, since nothing waits on a replay.
/// Steps 2 to 5 are those of a capture, with the trigger
/// [`REPLAY_TRIGGER`] and the event PreCompact (a transcript read).
///
/// A replay never moves the session's read position: `new_offset` is
/// always `None`, so live capture goes on from where it was.
pub fn replay(input: &CaptureInput<'_>, session: &mut SessionState) -> CaptureOutcome {
    replay_with(input, session, &mut RealStages)
}

/// [`replay`] with the steps given.
pub(crate) fn replay_with(
    input: &CaptureInput<'_>,
    session: &mut SessionState,
    stages: &mut dyn Stages,
) -> CaptureOutcome {
    if !input.settings.enabled {
        return CaptureOutcome::skip(Skip::CaptureDisabled);
    }
    if !input.project.capture_enabled {
        return CaptureOutcome::skip(Skip::ProjectOptedOut);
    }
    let Some(path) = str_field(input.payload, "transcript_path") else {
        return CaptureOutcome::skip(Skip::NoTranscript);
    };
    let Some(read) = stages.read(path, Some(0)) else {
        return CaptureOutcome::skip(Skip::NoTranscript);
    };
    let origins = transcript_origins(input, Some(REPLAY_TRIGGER.to_owned()));
    let mut outcome = CaptureOutcome {
        source: Some(TextSource::Transcript),
        ..CaptureOutcome::default()
    };
    let tags = |f: &[String]| derive_transcript_tags(f);
    for entry in &read.entries {
        if entry.role != "assistant" {
            continue;
        }
        let stripped = python_strip(&entry.text);
        if stripped.is_empty() {
            continue;
        }
        let original_chars = stripped.chars().count();
        outcome.extracted_chars += original_chars;
        let (text, cut) = truncate(stripped, MAX_CAPTURE_CHARS);
        outcome.cap_hit |= cut;
        process_text(
            HookEvent::PreCompact,
            Extracted {
                text,
                cap_hit: cut,
                original_chars,
            },
            &origins,
            None,
            &entry.prompt,
            &tags,
            session,
            stages,
            &mut outcome,
        );
    }
    outcome
}

/// A non-empty string field of `object`; `None` when absent, null, empty
/// or not a string.
fn str_field<'v>(object: &'v Value, key: &str) -> Option<&'v str> {
    object
        .get(key)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
}

fn post_tool_use(
    input: &CaptureInput<'_>,
    session: &mut SessionState,
    stages: &mut dyn Stages,
) -> CaptureOutcome {
    let payload = input.payload;
    let Some(tool_name) = str_field(payload, "tool_name") else {
        return CaptureOutcome::skip(Skip::NoToolName);
    };
    if !input.settings.captures_tool(tool_name) {
        return CaptureOutcome::skip(Skip::ToolNotCaptured(tool_name.to_owned()));
    }
    if let Some(rule) = skip_rule(tool_name) {
        return CaptureOutcome::skip(Skip::ToolSkipped(rule));
    }
    // Version 2: a tool_input that is not an object is read as `{}`.
    let empty = Value::Object(serde_json::Map::new());
    let tool_input = match payload.get("tool_input") {
        Some(v) if v.is_object() => v,
        _ => &empty,
    };
    let file =
        str_field(tool_input, "file_path").or_else(|| str_field(tool_input, "notebook_path"));
    let normalized_file = file.map(|f| normalize_file_anchor(&input.project.root, f));
    if let (Some(raw), Some(normalized)) = (file, normalized_file.as_deref())
        && (input.settings.is_ignored(normalized) || input.settings.is_ignored(raw))
    {
        return CaptureOutcome::skip(Skip::PathIgnored(normalized.to_owned()));
    }

    let response = tool_response_text(payload);
    let extracted = stages.extract(tool_name, tool_input, &response);
    // Edit and Write are anchored to their file (SPEC section 16, Type and
    // provenance). NotebookEdit records the notebook as the file path only.
    let anchor_name = match tool_name {
        "Edit" | "Write" => str_field(tool_input, "file_path")
            .map(|f| normalize_file_anchor(&input.project.root, f)),
        _ => None,
    };
    let origins = Origins {
        session_id: str_field(payload, "session_id").map(str::to_owned),
        tool: Some(tool_name.to_owned()),
        file_path: normalized_file,
        commit_sha: input.commit_sha.map(str::to_owned),
        trigger: None,
    };
    let tags = |families: &[String]| derive_tags(tool_name, families);
    let mut outcome = CaptureOutcome {
        source: Some(TextSource::ToolCall),
        cap_hit: extracted.cap_hit,
        extracted_chars: extracted.original_chars,
        ..CaptureOutcome::default()
    };
    process_text(
        input.event,
        extracted,
        &origins,
        anchor_name,
        "",
        &tags,
        session,
        stages,
        &mut outcome,
    );
    outcome
}

fn stop(
    input: &CaptureInput<'_>,
    session: &mut SessionState,
    stages: &mut dyn Stages,
) -> CaptureOutcome {
    if input
        .payload
        .get("stop_hook_active")
        .is_some_and(python_truthy)
    {
        return CaptureOutcome::skip(Skip::StopHookActive);
    }
    let trigger = str_field(input.payload, "trigger")
        .unwrap_or(DEFAULT_STOP_TRIGGER)
        .to_owned();
    // SPEC section 16 says Stop sends `last_assistant_message` and does not
    // read the transcript. Every fixture Stop payload lacks the field, so
    // when it is absent (or not a string) Stop falls back to the
    // transcript since the offset, as version 1 did. With the message, the
    // offset moves to the transcript's end (B-162): each turn's final text
    // was offered by its own Stop, so a later PreCompact reads only what
    // came after, and a session first seen at a Stop never reads its
    // history.
    match input
        .payload
        .get("last_assistant_message")
        .and_then(Value::as_str)
    {
        Some(message) => {
            let origins = transcript_origins(input, Some(trigger));
            let stripped = python_strip(message);
            let original_chars = stripped.chars().count();
            let (text, cap_hit) = truncate(stripped, MAX_CAPTURE_CHARS);
            let mut outcome = CaptureOutcome {
                source: Some(TextSource::LastAssistantMessage),
                new_offset: transcript_size(input.payload),
                cap_hit,
                extracted_chars: original_chars,
                ..CaptureOutcome::default()
            };
            let extracted = Extracted {
                text,
                cap_hit,
                original_chars,
            };
            let context = str_field(input.payload, "transcript_path")
                .map(last_prompt)
                .unwrap_or_default();
            process_text(
                input.event,
                extracted,
                &origins,
                None,
                &context,
                &|f: &[String]| derive_transcript_tags(f),
                session,
                stages,
                &mut outcome,
            );
            outcome
        }
        None => transcript(input, session, stages, Some(trigger)),
    }
}

/// The size of the payload's transcript, when it names one this crate may
/// read.
fn transcript_size(payload: &Value) -> Option<u64> {
    let path = validate_transcript_path(str_field(payload, "transcript_path")?)?;
    std::fs::metadata(path).ok().map(|m| m.len())
}

fn transcript_origins(input: &CaptureInput<'_>, trigger: Option<String>) -> Origins {
    Origins {
        session_id: str_field(input.payload, "session_id").map(str::to_owned),
        tool: None,
        file_path: None,
        commit_sha: input.commit_sha.map(str::to_owned),
        trigger: trigger.or_else(|| str_field(input.payload, "trigger").map(str::to_owned)),
    }
}

fn transcript(
    input: &CaptureInput<'_>,
    session: &mut SessionState,
    stages: &mut dyn Stages,
    trigger: Option<String>,
) -> CaptureOutcome {
    let Some(path) = str_field(input.payload, "transcript_path") else {
        return CaptureOutcome::skip(Skip::NoTranscript);
    };
    let Some(read) = stages.read(path, input.transcript_offset) else {
        return CaptureOutcome::skip(Skip::NoTranscript);
    };
    // Nothing new since the offset: stop here, as version 1 did, with the
    // offset the read reached.
    if read.entries.is_empty() {
        return CaptureOutcome {
            source: Some(TextSource::Transcript),
            new_offset: Some(read.new_offset),
            ..CaptureOutcome::default()
        };
    }
    let extracted_chars = read
        .entries
        .iter()
        .map(|e| python_strip(&e.text).chars().count())
        .sum();
    let texts = stages.assistant_texts(&read.entries);
    let origins = transcript_origins(input, trigger);
    let mut outcome = CaptureOutcome {
        source: Some(TextSource::Transcript),
        new_offset: Some(read.new_offset),
        cap_hit: texts.cap_hit,
        extracted_chars,
        ..CaptureOutcome::default()
    };
    let tags = |f: &[String]| derive_transcript_tags(f);
    for (i, text) in texts.texts.into_iter().enumerate() {
        let original_chars = text.chars().count();
        // The cap counts from the newest text back (B-161), so only the
        // first, oldest, text can be cut.
        let extracted = Extracted {
            text,
            cap_hit: texts.cap_hit && i == 0,
            original_chars,
        };
        let context = texts.prompts.get(i).map(String::as_str).unwrap_or("");
        process_text(
            input.event,
            extracted,
            &origins,
            None,
            context,
            &tags,
            session,
            stages,
            &mut outcome,
        );
    }
    outcome
}

/// Steps 2 to 5 for one text: classify, redact, dedupe, build. `context` is
/// the text's turn prompt, bounded; it is redacted here (B-166).
#[allow(clippy::too_many_arguments)]
fn process_text(
    event: HookEvent,
    extracted: Extracted,
    origins: &Origins,
    anchor_name: Option<String>,
    context: &str,
    tags: &dyn Fn(&[String]) -> Vec<String>,
    session: &mut SessionState,
    stages: &mut dyn Stages,
    outcome: &mut CaptureOutcome,
) {
    let text = extracted.text;
    if python_strip(&text).is_empty() {
        outcome.drops.push(TextDrop::Empty);
        return;
    }
    let signal = stages.classify(&text);
    if signal.tier == Tier::Weak {
        outcome.drops.push(TextDrop::Weak);
        return;
    }
    let Some(importance) = importance(signal.tier, signal.score) else {
        outcome.drops.push(TextDrop::NonFiniteScore);
        return;
    };
    let report = stages.redact(&text);
    if report.should_skip {
        outcome.drops.push(TextDrop::Secret);
        return;
    }
    let content = report.redaction.text;
    if stages.is_duplicate(session, &content) {
        outcome.drops.push(TextDrop::Duplicate);
        return;
    }
    let hash = content_hash(&content);
    session.record(hash.clone());
    let candidate = Classified {
        event,
        tags: tags(&signal.families),
        content,
        content_hash: hash,
        tier: signal.tier,
        score: signal.score,
        importance,
        families: signal.families,
        redaction_kinds: report.redaction.kinds,
        cap_hit: extracted.cap_hit,
        anchor_name,
        origins,
        context: if context.is_empty() {
            String::new()
        } else {
            RegexSecretScanner::new().redact(context).text
        },
    }
    .into_candidate();
    outcome.candidates.push(candidate);
}
