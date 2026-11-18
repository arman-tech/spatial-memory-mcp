//! The transcript delta reader and the assistant-text extractor for
//! PreCompact and Stop (SPEC section 16, Pipeline in the daemon, step 1;
//! AT-14, AT-17), ported from version 2's `transcript_reader.py` and
//! `transcript_extractor.py`.
//!
//! Capture reads with [`read_turns`]: the final text of each turn, and for
//! a session read for the first time only its live segment (B-162). The
//! reader is stateless: the caller passes the byte offset reached last time,
//! or none for an unseen session, and stores the new one. Reading the
//! transcript file is the only input or output this crate does.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use regex::Regex;
use serde_json::Value;
use sm_core::hash::python_strip;

use crate::extract::{MAX_CAPTURE_CHARS, python_str, python_truthy, truncate};
use crate::select::{TOOL_INPUT_PROSE_MAX, TOOL_RESULT_PROSE_MAX, prose, split_units};

/// Transcript lines longer than this, newline included, are skipped.
pub const MAX_LINE_BYTES: usize = 1_048_576;

/// The longest session identifier version 1 accepted.
pub const MAX_SESSION_ID_CHARS: usize = 128;

/// Only lines holding these bytes are parsed.
const ASSISTANT_MARKER: &str = "\"assistant\"";

/// The longest user prompt kept as a text's context for the judge, in
/// characters; a longer one is cut with the `...` marker (B-166).
pub const CONTEXT_CHARS: usize = 2_000;

/// How much of a transcript's end [`last_prompt`] reads, in bytes.
pub const LAST_PROMPT_TAIL_BYTES: u64 = 4 * 1_048_576;

/// Windows device names, with or without an extension (`CON`, `nul.txt`).
static WINDOWS_DEVICE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^(CON|NUL|PRN|AUX|COM[1-9]|LPT[1-9])(\..+)?$")
        .expect("the device name expression is valid")
});

/// Whether `name` is a Windows device name such as `CON` or `LPT1.txt`.
pub fn is_windows_device_name(name: &str) -> bool {
    WINDOWS_DEVICE.is_match(name)
}

/// Version 1's `sanitize_session_id`: the identifier when it is 1 to 128
/// characters of ASCII letters, digits, `_` and `-` and not a Windows
/// device name; otherwise `None`. Unlike Python, a trailing newline is
/// rejected (Python's `$` accepted `"abc\n"`).
pub fn sanitize_session_id(session_id: &str) -> Option<&str> {
    if session_id.is_empty() || session_id.chars().count() > MAX_SESSION_ID_CHARS {
        return None;
    }
    if !session_id
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
    {
        return None;
    }
    if is_windows_device_name(session_id) {
        return None;
    }
    Some(session_id)
}

/// Version 1's `validate_transcript_path`, a string check only: the path
/// when it is not empty, holds no `..` anywhere, and is absolute on this
/// platform; otherwise `None`.
pub fn check_transcript_path(path: &str) -> Option<&str> {
    if path.is_empty() || path.contains("..") || !Path::new(path).is_absolute() {
        return None;
    }
    Some(path)
}

/// The transcript path the reader will open: [`check_transcript_path`],
/// a file name that is not a Windows device name, and a regular file
/// (symbolic links followed). `None` otherwise.
pub fn validate_transcript_path(path: &str) -> Option<PathBuf> {
    let checked = Path::new(check_transcript_path(path)?);
    if let Some(name) = checked.file_name().and_then(|n| n.to_str())
        && is_windows_device_name(name)
    {
        return None;
    }
    let meta = std::fs::metadata(checked).ok()?;
    meta.is_file().then(|| checked.to_path_buf())
}

/// One text from the transcript: an assistant message or turn, or, from
/// [`read_texts`], any text offered to the judge (B-179).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscriptEntry {
    /// `assistant` for a reply or a tool's new text; `user` for a message
    /// the user wrote, a compaction summary or a tool's result.
    pub role: String,
    /// The text: for a reply, its text blocks, each stripped, joined by a
    /// blank line; for a tool's text, its prose.
    pub text: String,
    /// The entry's `timestamp`, as Python's `str()` writes it.
    pub timestamp: String,
    /// The entry's `uuid`, as Python's `str()` writes it.
    pub uuid: String,
    /// The entry's `type`: `assistant` or `user`.
    pub entry_type: String,
    /// The user prompt that started the entry's turn, bounded to
    /// [`CONTEXT_CHARS`]; empty when it is not known: version 1's reader, a
    /// turn whose prompt came before the offset, or a prompt with no text
    /// (B-166).
    pub prompt: String,
    /// What kind of text this is (B-179); [`TextKind::Reply`] for version
    /// 2's reader and [`read_turns`].
    pub kind: TextKind,
    /// For a text that is not a reply, the line that tells the judge where
    /// it came from (`Source: ...`); empty for a reply (B-179).
    pub source: String,
}

/// What kind of text a transcript entry holds (SPEC section 16, Candidate
/// selection; B-179).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TextKind {
    /// The final text of an assistant turn: its text blocks after the last
    /// tool call.
    #[default]
    Reply,
    /// A message the user wrote: not a command, a task notification, a
    /// tool result or an entry Claude Code wrote itself.
    Prompt,
    /// The summary Claude Code writes when it compacts the conversation.
    Summary,
    /// The prose of the new text of an Edit, MultiEdit or Write call.
    ToolInput,
    /// The prose of the result of a Bash, Read or Grep call.
    ToolResult,
}

impl TextKind {
    /// The name recorded in a candidate's provenance (`source`).
    pub fn as_str(self) -> &'static str {
        match self {
            TextKind::Reply => "reply",
            TextKind::Prompt => "prompt",
            TextKind::Summary => "summary",
            TextKind::ToolInput => "tool_input",
            TextKind::ToolResult => "tool_result",
        }
    }
}

fn str_field(data: &Value, key: &str) -> String {
    data.get(key).map(python_str).unwrap_or_default()
}

/// Version 1's `_parse_entry`: an assistant entry that is not a sidechain,
/// with at least one non-blank text block; `None` otherwise. Tool use and
/// tool result blocks are ignored.
pub fn parse_entry(data: &Value) -> Option<TranscriptEntry> {
    if data.get("type").and_then(Value::as_str) != Some("assistant") {
        return None;
    }
    if data.get("isSidechain").is_some_and(python_truthy) {
        return None;
    }
    let blocks = data
        .get("message")?
        .as_object()?
        .get("content")?
        .as_array()?;
    let parts: Vec<&str> = blocks
        .iter()
        .filter_map(|block| {
            let block = block.as_object()?;
            if block.get("type").and_then(Value::as_str) != Some("text") {
                return None;
            }
            let text = python_strip(block.get("text")?.as_str()?);
            (!text.is_empty()).then_some(text)
        })
        .collect();
    if parts.is_empty() {
        return None;
    }
    Some(TranscriptEntry {
        role: "assistant".to_owned(),
        text: parts.join("\n\n"),
        timestamp: str_field(data, "timestamp"),
        uuid: str_field(data, "uuid"),
        entry_type: "assistant".to_owned(),
        prompt: String::new(),
        kind: TextKind::Reply,
        source: String::new(),
    })
}

/// What one read of the transcript found.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TranscriptRead {
    /// The assistant entries after the offset, in file order.
    pub entries: Vec<TranscriptEntry>,
    /// Where the next read starts: the end of the file as read.
    pub new_offset: u64,
    /// Whether capture has now read every text of the session from its
    /// first line, so that nothing before `new_offset` was skipped (B-184).
    /// [`read_texts`] sets it; the older readers leave it `false`.
    pub from_start: bool,
}

/// Reads the assistant entries written since `last_offset`.
///
/// An invalid path, a missing or empty file gives no entries and offset 0.
/// An offset past the end of the file (the transcript was truncated or
/// replaced) starts again from 0. Each line, CRLF or LF ended, the last one
/// with or without a newline, is skipped when it is longer than
/// [`MAX_LINE_BYTES`], blank, without the `"assistant"` marker, not UTF-8,
/// not a JSON object or not an assistant entry ([`parse_entry`]). The new
/// offset is where reading stopped, the end of the file, even when the last
/// line was skipped. A read error gives no entries and the offset reading
/// started from.
///
/// Version 2's reader, kept for its fixture cases: every assistant text,
/// before a tool call or not, from the offset given. Capture reads with
/// [`read_turns`] (B-162), and the daemon keeps each session's offset in
/// the store (B-163).
pub fn read_transcript(path: &str, last_offset: u64) -> TranscriptRead {
    let Some(path) = validate_transcript_path(path) else {
        return TranscriptRead::default();
    };
    let Ok(meta) = std::fs::metadata(&path) else {
        return TranscriptRead::default();
    };
    let size = meta.len();
    if size == 0 {
        return TranscriptRead::default();
    }
    let offset = if last_offset > size { 0 } else { last_offset };
    match read_from(&path, offset) {
        Ok(read) => read,
        Err(_) => TranscriptRead {
            entries: Vec::new(),
            new_offset: offset,
            from_start: false,
        },
    }
}

fn read_from(path: &Path, offset: u64) -> std::io::Result<TranscriptRead> {
    let mut file = File::open(path)?;
    file.seek(SeekFrom::Start(offset))?;
    let mut reader = BufReader::with_capacity(64 * 1024, file);
    let mut entries = Vec::new();
    let mut position = offset;
    let mut line: Vec<u8> = Vec::new();
    loop {
        line.clear();
        let (consumed, oversized) = read_line_capped(&mut reader, &mut line)?;
        if consumed == 0 {
            break;
        }
        position += consumed as u64;
        if oversized {
            continue;
        }
        if let Some(entry) = parse_line(&line) {
            entries.push(entry);
        }
    }
    Ok(TranscriptRead {
        entries,
        new_offset: position,
        from_start: false,
    })
}

/// Reads the final text of each turn written since `last_offset` (RC1 and
/// RC5, decision B-162). This is the reader capture uses; [`read_transcript`]
/// is version 2's reader, kept for its fixture cases.
///
/// **Turns.** A user entry that is not a tool result (a prompt, a command,
/// the summary Claude Code writes after compacting) starts a turn. Within a
/// turn the assistant's text blocks are collected in order, and a tool call
/// discards what was collected before it: text written before a tool call
/// announces the next action and is not a finding. What is left when the
/// turn ends, its blocks joined by a blank line, is the turn's final text,
/// one entry with the timestamp and uuid of its last text. A turn that
/// ends in a tool call gives nothing. The last turn read may be incomplete;
/// it gives its text so far.
///
/// **An unseen session** (`last_offset` is `None`: the daemon has no offset
/// recorded for it, as for a session open before setup or a resumed one)
/// is read from the start, but only its live segment counts: a compaction
/// boundary (`type: system`, `subtype: compact_boundary`) discards every
/// turn before it. Text compacted out of the conversation before capture
/// saw the session is never captured. For a seen session a boundary only
/// ends the turn: the text before it, after the offset, is new.
///
/// **Lines.** Each line is skipped when it is longer than
/// [`MAX_LINE_BYTES`], blank, not UTF-8, not a JSON object, or a sidechain
/// entry. A last line without a newline that is not JSON is being written:
/// it is not read, and the new offset stays before it. An offset past the
/// end of the file (the transcript was replaced) reads it as unseen. An
/// invalid path, or a missing or empty file, gives no entries and offset 0;
/// a read error gives no entries and the offset reading started from.
pub fn read_turns(path: &str, last_offset: Option<u64>) -> TranscriptRead {
    let Some(path) = validate_transcript_path(path) else {
        return TranscriptRead::default();
    };
    let Ok(meta) = std::fs::metadata(&path) else {
        return TranscriptRead::default();
    };
    let size = meta.len();
    if size == 0 {
        return TranscriptRead::default();
    }
    let seen = last_offset.filter(|&o| o <= size);
    let offset = seen.unwrap_or(0);
    match turns_from(&path, offset, seen.is_some()) {
        Ok(read) => read,
        Err(_) => TranscriptRead {
            entries: Vec::new(),
            new_offset: offset,
            from_start: false,
        },
    }
}

/// One turn being read: its text blocks since the last tool call, the
/// timestamp and uuid of the last of them, and the prompt that started it.
#[derive(Default)]
struct Turn {
    texts: Vec<String>,
    timestamp: String,
    uuid: String,
    prompt: String,
}

impl Turn {
    /// Ends the turn, adding its final text to `entries` when it has one.
    fn end(&mut self, entries: &mut Vec<TranscriptEntry>) {
        let turn = std::mem::take(self);
        if turn.texts.is_empty() {
            return;
        }
        entries.push(TranscriptEntry {
            role: "assistant".to_owned(),
            text: turn.texts.join("\n\n"),
            timestamp: turn.timestamp,
            uuid: turn.uuid,
            entry_type: "assistant".to_owned(),
            prompt: turn.prompt,
            kind: TextKind::Reply,
            source: String::new(),
        });
    }
}

fn turns_from(path: &Path, offset: u64, seen: bool) -> std::io::Result<TranscriptRead> {
    let mut file = File::open(path)?;
    file.seek(SeekFrom::Start(offset))?;
    let mut reader = BufReader::with_capacity(64 * 1024, file);
    let mut entries = Vec::new();
    let mut turn = Turn::default();
    let mut position = offset;
    let mut line: Vec<u8> = Vec::new();
    loop {
        line.clear();
        let (consumed, oversized) = read_line_capped(&mut reader, &mut line)?;
        if consumed == 0 {
            break;
        }
        let data = if oversized { None } else { parse_object(&line) };
        if data.is_none() && !oversized && line.last() != Some(&b'\n') {
            // The last line, still being written: read it next time.
            break;
        }
        position += consumed as u64;
        let Some(data) = data else {
            continue;
        };
        if data.get("isSidechain").is_some_and(python_truthy) {
            continue;
        }
        match data.get("type").and_then(Value::as_str) {
            Some("system")
                if data.get("subtype").and_then(Value::as_str) == Some("compact_boundary") =>
            {
                turn.end(&mut entries);
                if !seen {
                    entries.clear();
                }
            }
            Some("user") if !is_tool_result(&data) => {
                // A meta entry or a compaction summary ends the turn but is
                // not a prompt: the next turn keeps the last real prompt.
                let prompt = user_prompt(&data).unwrap_or_else(|| turn.prompt.clone());
                turn.end(&mut entries);
                turn.prompt = prompt;
            }
            Some("assistant") => {
                let Some(blocks) = data
                    .get("message")
                    .and_then(|m| m.get("content"))
                    .and_then(Value::as_array)
                else {
                    continue;
                };
                for block in blocks {
                    match block.get("type").and_then(Value::as_str) {
                        Some("text") => {
                            let text = python_strip(
                                block.get("text").and_then(Value::as_str).unwrap_or(""),
                            );
                            if !text.is_empty() {
                                turn.texts.push(text.to_owned());
                                turn.timestamp = str_field(&data, "timestamp");
                                turn.uuid = str_field(&data, "uuid");
                            }
                        }
                        Some("tool_use") => turn.texts.clear(),
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
    turn.end(&mut entries);
    Ok(TranscriptRead {
        entries,
        new_offset: position,
        from_start: false,
    })
}

/// How [`read_texts`] treats a session it has not read before (B-179).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadMode {
    /// A Stop: an unseen session gives only its current turn, from the
    /// user's last message on.
    Turn,
    /// A PreCompact: an unseen session gives its live segment, after the
    /// last compaction boundary, without the summary that opens it.
    Segment,
    /// A replay (B-172): the whole transcript from its start, every
    /// compaction summary included, whatever offset is given.
    Whole,
}

/// The tools whose results' prose is offered (B-179).
pub const RESULT_TOOLS: [&str; 3] = ["Bash", "Read", "Grep"];

/// The longest command, file path or search pattern named in a source line.
pub const SOURCE_DETAIL_CHARS: usize = 300;

/// The source line of a message the user wrote.
pub const PROMPT_SOURCE: &str = "Source: a message the user wrote.";

/// The source line of a compaction summary.
pub const SUMMARY_SOURCE: &str =
    "Source: the summary Claude Code wrote when it compacted the conversation.";

/// A `<system-reminder>` block Claude Code adds to a user entry.
static SYSTEM_REMINDER: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)<system-reminder>.*?</system-reminder>")
        .expect("the system reminder expression is valid")
});

/// The starts of a user entry that is a command, its output or a task
/// notification, not a message the user wrote.
const NOT_A_MESSAGE: [&str; 8] = [
    "<task-notification>",
    "<command-name>",
    "<command-message>",
    "<local-command-stdout>",
    "<local-command-caveat>",
    "<bash-input>",
    "<bash-stdout>",
    "<bash-stderr>",
];

/// Reads the texts offered to the judge since `last_offset` (SPEC section
/// 16, Candidate selection; B-179). This is the reader capture uses;
/// [`read_turns`] reads the replies alone.
///
/// **Texts**, in file order, each an entry with its [`TextKind`], the
/// request that started its turn (`prompt`, bounded to [`CONTEXT_CHARS`])
/// and, unless it is a reply, the `Source:` line the judge reads with it:
///
/// - each turn's final text, as [`read_turns`] gives it: text written
///   before a tool call is not offered (RC5);
/// - each message the user wrote, without its `<system-reminder>` blocks.
///   Commands, their output, task notifications, tool results and entries
///   Claude Code wrote itself (`isMeta`) are not messages;
/// - each compaction summary (`isCompactSummary`);
/// - the prose ([`prose`]) of the new text of each Edit, MultiEdit and
///   Write call, at most [`TOOL_INPUT_PROSE_MAX`] characters a call;
/// - the prose of each Bash, Read and Grep result that is not an error, at
///   most [`TOOL_RESULT_PROSE_MAX`] characters a result. A result is matched
///   to its call by `tool_use_id`; a result whose call was before the offset
///   is not offered.
///
/// Raw command output and whole files are never offered.
///
/// **An unseen session** (`last_offset` is `None`, or past the end of the
/// file) is read from its start but never yields its history (B-162): a
/// compaction boundary discards every text before it, the summary after a
/// boundary is not offered, and with [`ReadMode::Turn`] each message the
/// user wrote discards every text before it too. [`ReadMode::Whole`] reads
/// every session as seen, from offset 0. A seen session offers every text
/// after its offset.
///
/// **Summaries and the session's start** (B-184). A compaction summary
/// restates the conversation before it, so it is offered only when capture
/// has read the session from its first line: `from_start` says whether the
/// reads up to `last_offset` did. An unseen read starts with it true and
/// turns it false when it skips anything: a compaction boundary or summary,
/// or, with [`ReadMode::Turn`], a message the user wrote that discards
/// earlier texts. A seen read keeps the `from_start` it was given, and
/// [`ReadMode::Whole`] reads with it true. The read returns the value for
/// the next read in [`TranscriptRead::from_start`].
///
/// **Lines** are read as [`read_turns`] reads them: an oversized, blank,
/// non-UTF-8, non-object or sidechain line is skipped, and a last line still
/// being written is left for the next read. An invalid path gives no
/// entries, offset 0 and `from_start` false; a missing or empty file gives
/// no entries and offset 0; a read error gives no entries and the offset
/// reading started from.
pub fn read_texts(
    path: &str,
    last_offset: Option<u64>,
    from_start: bool,
    mode: ReadMode,
) -> TranscriptRead {
    let Some(path) = validate_transcript_path(path) else {
        return TranscriptRead::default();
    };
    // A missing file reads as an empty one: no entries, offset 0.
    let size = std::fs::metadata(&path).map_or(0, |m| m.len());
    let (seen, from_start) = match mode {
        ReadMode::Whole => (Some(0), true),
        ReadMode::Turn | ReadMode::Segment => match last_offset.filter(|&o| o <= size) {
            Some(o) => (Some(o), from_start),
            None => (None, true),
        },
    };
    let offset = seen.unwrap_or(0);
    if size == 0 {
        return TranscriptRead {
            entries: Vec::new(),
            new_offset: 0,
            from_start,
        };
    }
    match texts_from(&path, offset, seen.is_some(), from_start, mode) {
        Ok(read) => read,
        Err(_) => TranscriptRead {
            entries: Vec::new(),
            new_offset: offset,
            from_start,
        },
    }
}

/// An entry for a text that is not a reply.
fn other_entry(
    kind: TextKind,
    text: String,
    source: String,
    prompt: &str,
    data: &Value,
) -> TranscriptEntry {
    let role = if kind == TextKind::ToolInput {
        "assistant"
    } else {
        "user"
    };
    TranscriptEntry {
        role: role.to_owned(),
        text,
        timestamp: str_field(data, "timestamp"),
        uuid: str_field(data, "uuid"),
        entry_type: str_field(data, "type"),
        prompt: prompt.to_owned(),
        kind,
        source,
    }
}

/// A command, path or pattern for a source line: its words joined by one
/// space, bounded to [`SOURCE_DETAIL_CHARS`].
fn detail(value: Option<&Value>) -> String {
    let text = value.and_then(Value::as_str).unwrap_or("");
    let one_line = text.split_whitespace().collect::<Vec<_>>().join(" ");
    truncate(&one_line, SOURCE_DETAIL_CHARS).0
}

/// The `Source:` line for the prose of a tool's result; `None` for a tool
/// whose results are not offered.
fn result_source(name: &str, input: &Value) -> Option<String> {
    Some(match name {
        "Bash" => format!(
            "Source: prose from the output of the command `{}`.",
            detail(input.get("command"))
        ),
        "Read" => format!(
            "Source: prose from the file {}, as read.",
            detail(input.get("file_path"))
        ),
        "Grep" => format!(
            "Source: prose from the results of a search for `{}`.",
            detail(input.get("pattern"))
        ),
        _ => return None,
    })
}

/// The new text of an Edit, MultiEdit or Write call; `None` for any other
/// tool.
fn new_text(name: &str, input: &Value) -> Option<String> {
    let field = |v: &Value, k: &str| v.get(k).and_then(Value::as_str).unwrap_or("").to_owned();
    match name {
        "Write" => Some(field(input, "content")),
        "Edit" => Some(field(input, "new_string")),
        "MultiEdit" => Some(
            input
                .get("edits")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .map(|e| field(e, "new_string"))
                .collect::<Vec<_>>()
                .join("\n\n"),
        ),
        _ => None,
    }
}

/// A message's or a result's text: a string, or its text blocks joined by
/// a newline.
fn content_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(blocks)) => blocks
            .iter()
            .filter(|b| b.get("type").and_then(Value::as_str) == Some("text"))
            .filter_map(|b| b.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

/// A text without its `<system-reminder>` blocks.
fn without_reminders(text: &str) -> std::borrow::Cow<'_, str> {
    SYSTEM_REMINDER.replace_all(text, "")
}

/// A user entry's whole text, without its `<system-reminder>` blocks,
/// stripped; `None` when the entry is not a message the user wrote: a meta
/// entry, a compaction summary, a command or its output, a task
/// notification, or nothing but white space.
fn user_message(data: &Value) -> Option<String> {
    if data.get("isMeta").is_some_and(python_truthy)
        || data.get("isCompactSummary").is_some_and(python_truthy)
    {
        return None;
    }
    let raw = content_text(data.get("message").and_then(|m| m.get("content")));
    let text = without_reminders(&raw);
    let text = python_strip(&text);
    if text.is_empty() || NOT_A_MESSAGE.iter().any(|m| text.starts_with(m)) {
        return None;
    }
    Some(text.to_owned())
}

fn texts_from(
    path: &Path,
    offset: u64,
    seen: bool,
    mut from_start: bool,
    mode: ReadMode,
) -> std::io::Result<TranscriptRead> {
    let mut file = File::open(path)?;
    file.seek(SeekFrom::Start(offset))?;
    let mut reader = BufReader::with_capacity(64 * 1024, file);
    let mut entries = Vec::new();
    let mut turn = Turn::default();
    // Each tool call read so far, by id: its name and its input.
    let mut calls: HashMap<String, (String, Value)> = HashMap::new();
    let mut position = offset;
    let mut line: Vec<u8> = Vec::new();
    loop {
        line.clear();
        let (consumed, oversized) = read_line_capped(&mut reader, &mut line)?;
        if consumed == 0 {
            break;
        }
        let data = if oversized { None } else { parse_object(&line) };
        if data.is_none() && !oversized && line.last() != Some(&b'\n') {
            // The last line, still being written: read it next time.
            break;
        }
        position += consumed as u64;
        let Some(data) = data else {
            continue;
        };
        if data.get("isSidechain").is_some_and(python_truthy) {
            continue;
        }
        match data.get("type").and_then(Value::as_str) {
            Some("system")
                if data.get("subtype").and_then(Value::as_str) == Some("compact_boundary") =>
            {
                turn.end(&mut entries);
                if !seen {
                    entries.clear();
                    from_start = false;
                }
            }
            Some("user") if is_tool_result(&data) => {
                let blocks = data
                    .get("message")
                    .and_then(|m| m.get("content"))
                    .and_then(Value::as_array);
                for block in blocks.into_iter().flatten() {
                    if block.get("type").and_then(Value::as_str) != Some("tool_result")
                        || block.get("is_error").is_some_and(python_truthy)
                    {
                        continue;
                    }
                    let Some(source) = block
                        .get("tool_use_id")
                        .and_then(Value::as_str)
                        .and_then(|id| calls.get(id))
                        .and_then(|(name, input)| result_source(name, input))
                    else {
                        continue;
                    };
                    let text = prose(&content_text(block.get("content")), TOOL_RESULT_PROSE_MAX);
                    if !text.is_empty() {
                        entries.push(other_entry(
                            TextKind::ToolResult,
                            text,
                            source,
                            &turn.prompt,
                            &data,
                        ));
                    }
                }
            }
            Some("user") => {
                // A meta entry or a compaction summary ends the turn but is
                // not a prompt: the next turn keeps the last real prompt.
                let prompt = user_prompt(&data).unwrap_or_else(|| turn.prompt.clone());
                turn.end(&mut entries);
                turn.prompt = prompt;
                let is_summary = data.get("isCompactSummary").is_some_and(python_truthy);
                if is_summary && !seen {
                    // It restates history this read did not offer.
                    from_start = false;
                }
                if is_summary && seen && from_start {
                    let summary = content_text(data.get("message").and_then(|m| m.get("content")));
                    let summary = python_strip(&summary);
                    if !summary.is_empty() {
                        entries.push(other_entry(
                            TextKind::Summary,
                            summary.to_owned(),
                            SUMMARY_SOURCE.to_owned(),
                            "",
                            &data,
                        ));
                    }
                }
                if let Some(message) = user_message(&data) {
                    if mode == ReadMode::Turn && !seen && !entries.is_empty() {
                        entries.clear();
                        from_start = false;
                    }
                    entries.push(other_entry(
                        TextKind::Prompt,
                        message,
                        PROMPT_SOURCE.to_owned(),
                        "",
                        &data,
                    ));
                }
            }
            Some("assistant") => {
                let Some(blocks) = data
                    .get("message")
                    .and_then(|m| m.get("content"))
                    .and_then(Value::as_array)
                else {
                    continue;
                };
                for block in blocks {
                    match block.get("type").and_then(Value::as_str) {
                        Some("text") => {
                            let text = python_strip(
                                block.get("text").and_then(Value::as_str).unwrap_or(""),
                            );
                            if !text.is_empty() {
                                turn.texts.push(text.to_owned());
                                turn.timestamp = str_field(&data, "timestamp");
                                turn.uuid = str_field(&data, "uuid");
                            }
                        }
                        Some("tool_use") => {
                            turn.texts.clear();
                            let name = block.get("name").and_then(Value::as_str).unwrap_or("");
                            let input = block.get("input").cloned().unwrap_or(Value::Null);
                            if let Some(body) = new_text(name, &input) {
                                let text = prose(&body, TOOL_INPUT_PROSE_MAX);
                                if !text.is_empty() {
                                    let source = format!(
                                        "Source: prose from the new text written to {}.",
                                        detail(input.get("file_path"))
                                    );
                                    entries.push(other_entry(
                                        TextKind::ToolInput,
                                        text,
                                        source,
                                        &turn.prompt,
                                        &data,
                                    ));
                                }
                            }
                            if let Some(id) = block.get("id").and_then(Value::as_str)
                                && RESULT_TOOLS.contains(&name)
                            {
                                calls.insert(id.to_owned(), (name.to_owned(), input));
                            }
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
    turn.end(&mut entries);
    Ok(TranscriptRead {
        entries,
        new_offset: position,
        from_start,
    })
}

/// A transcript line as a JSON object; `None` when it is blank, not UTF-8
/// or not a JSON object.
fn parse_object(raw: &[u8]) -> Option<Value> {
    if raw.trim_ascii().is_empty() {
        return None;
    }
    let data: Value = serde_json::from_str(std::str::from_utf8(raw).ok()?).ok()?;
    data.is_object().then_some(data)
}

/// The prompt a user entry gives, bounded to [`CONTEXT_CHARS`]: its text,
/// or its text blocks joined by a blank line, stripped, without the
/// `<system-reminder>` blocks Claude Code adds (B-179). `None` for an entry
/// Claude Code wrote itself (`isMeta`, `isCompactSummary`), which is not a
/// prompt.
fn user_prompt(data: &Value) -> Option<String> {
    if data.get("isMeta").is_some_and(python_truthy)
        || data.get("isCompactSummary").is_some_and(python_truthy)
    {
        return None;
    }
    let text = match data.get("message").and_then(|m| m.get("content")) {
        Some(Value::String(s)) => python_strip(&without_reminders(s)).to_owned(),
        Some(Value::Array(blocks)) => blocks
            .iter()
            .filter(|b| b.get("type").and_then(Value::as_str) == Some("text"))
            .filter_map(|b| b.get("text").and_then(Value::as_str))
            .map(|t| python_strip(&without_reminders(t)).to_owned())
            .filter(|t| !t.is_empty())
            .collect::<Vec<_>>()
            .join("\n\n"),
        _ => String::new(),
    };
    Some(truncate(&text, CONTEXT_CHARS).0)
}

/// The prompt of the transcript's last turn: the last user entry that is a
/// prompt (not a tool result, a sidechain, a meta entry or a compaction
/// summary) in the last [`LAST_PROMPT_TAIL_BYTES`] of the file, bounded to
/// [`CONTEXT_CHARS`]. Empty when there is none, or the path is not a
/// transcript this crate may read (B-166). Stop uses it as the context of
/// `last_assistant_message`.
pub fn last_prompt(path: &str) -> String {
    let Some(path) = validate_transcript_path(path) else {
        return String::new();
    };
    last_prompt_in(&path).unwrap_or_default()
}

fn last_prompt_in(path: &Path) -> std::io::Result<String> {
    let mut file = File::open(path)?;
    let size = file.metadata()?.len();
    let start = size.saturating_sub(LAST_PROMPT_TAIL_BYTES);
    file.seek(SeekFrom::Start(start))?;
    let mut reader = BufReader::with_capacity(64 * 1024, file);
    let mut line: Vec<u8> = Vec::new();
    let mut prompt = String::new();
    let mut first = start > 0;
    loop {
        line.clear();
        let (consumed, oversized) = read_line_capped(&mut reader, &mut line)?;
        if consumed == 0 {
            break;
        }
        if std::mem::take(&mut first) || oversized {
            // The first line of a tail starts mid-line.
            continue;
        }
        let Some(data) = parse_object(&line) else {
            continue;
        };
        if data.get("type").and_then(Value::as_str) != Some("user")
            || data.get("isSidechain").is_some_and(python_truthy)
            || is_tool_result(&data)
        {
            continue;
        }
        if let Some(p) = user_prompt(&data) {
            prompt = p;
        }
    }
    Ok(prompt)
}

/// Whether a user entry carries a tool's result, which continues the turn
/// rather than starting one.
fn is_tool_result(data: &Value) -> bool {
    data.get("message")
        .and_then(|m| m.get("content"))
        .and_then(Value::as_array)
        .is_some_and(|blocks| {
            blocks
                .iter()
                .any(|b| b.get("type").and_then(Value::as_str) == Some("tool_result"))
        })
}

/// Reads one line, newline included, into `line`, keeping at most
/// [`MAX_LINE_BYTES`] bytes. Returns the bytes consumed and whether the
/// line was longer than the limit (its bytes are then consumed but not
/// kept).
fn read_line_capped<R: BufRead>(
    reader: &mut R,
    line: &mut Vec<u8>,
) -> std::io::Result<(usize, bool)> {
    let mut consumed = 0usize;
    let mut oversized = false;
    loop {
        let buf = reader.fill_buf()?;
        if buf.is_empty() {
            return Ok((consumed, oversized));
        }
        let (chunk, done) = match buf.iter().position(|&b| b == b'\n') {
            Some(i) => (&buf[..=i], true),
            None => (buf, false),
        };
        let n = chunk.len();
        if !oversized {
            if line.len() + n > MAX_LINE_BYTES {
                oversized = true;
                line.clear();
            } else {
                line.extend_from_slice(chunk);
            }
        }
        reader.consume(n);
        consumed += n;
        if done {
            return Ok((consumed, oversized));
        }
    }
}

fn parse_line(raw: &[u8]) -> Option<TranscriptEntry> {
    if raw.trim_ascii().is_empty() {
        return None;
    }
    let text = std::str::from_utf8(raw).ok()?;
    if !text.contains(ASSISTANT_MARKER) {
        return None;
    }
    let data: Value = serde_json::from_str(text).ok()?;
    if !data.is_object() {
        return None;
    }
    parse_entry(&data)
}

/// Assistant texts ready to classify, and whether the safety cap cut them.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AssistantTexts {
    /// The texts, in transcript order.
    pub texts: Vec<String>,
    /// The prompt of each text's turn ([`TranscriptEntry::prompt`]), one per
    /// text; a shorter list means an empty prompt for the rest (B-166).
    pub prompts: Vec<String>,
    /// Whether the safety cap cut or dropped any text.
    pub cap_hit: bool,
    /// The kind of each text (B-179); a shorter list means a reply for the
    /// rest.
    pub kinds: Vec<TextKind>,
    /// Whether the safety cap cut each text's source text (B-179); a shorter
    /// list means not cut.
    pub cut: Vec<bool>,
}

/// Each assistant entry's stripped, non-blank text, in transcript order,
/// with one [`MAX_CAPTURE_CHARS`] cap over all of them together, counted
/// from the newest text back (RC3, decision B-161). The text that would
/// pass the cap is cut there and gets the `...` marker, and every older
/// text is dropped.
///
/// Version 2's `extract_assistant_text` counted from the oldest text, so a
/// long history kept its oldest text and dropped the newest, which is the
/// text a PreCompact or a Stop is there for; and the read offset moved past
/// the dropped text, so it was never read.
pub fn extract_assistant_texts(entries: &[TranscriptEntry]) -> AssistantTexts {
    let mut out = AssistantTexts::default();
    let mut total = 0usize;
    for entry in entries.iter().rev() {
        if entry.role != "assistant" {
            continue;
        }
        let stripped = python_strip(&entry.text);
        if stripped.is_empty() {
            continue;
        }
        let mut text = stripped.to_owned();
        let mut len = text.chars().count();
        if len > MAX_CAPTURE_CHARS {
            text = truncate(&text, MAX_CAPTURE_CHARS).0;
            len = text.chars().count();
            out.cap_hit = true;
        }
        if total + len > MAX_CAPTURE_CHARS {
            let remaining = MAX_CAPTURE_CHARS - total;
            if remaining > 0 {
                out.texts.push(truncate(&text, remaining).0);
                out.prompts.push(entry.prompt.clone());
            }
            out.cap_hit = true;
            break;
        }
        total += len;
        out.texts.push(text);
        out.prompts.push(entry.prompt.clone());
    }
    out.texts.reverse();
    out.prompts.reverse();
    out
}

/// The context the judge reads with a text of `entry` (B-166, B-179): for
/// a reply, the request that started its turn; for any other text, its
/// `Source:` line, then `Request: ` and the request when there is one.
pub fn text_context(entry: &TranscriptEntry) -> String {
    if entry.kind == TextKind::Reply || entry.source.is_empty() {
        return entry.prompt.clone();
    }
    if entry.prompt.is_empty() {
        entry.source.clone()
    } else {
        format!("{}\nRequest: {}", entry.source, entry.prompt)
    }
}

/// The units of the texts read, ready to classify (SPEC section 16,
/// Candidate selection; B-179): each entry's stripped, non-blank text, with
/// one [`MAX_CAPTURE_CHARS`] cap over all of them together counted from the
/// newest back (B-161), each split into units ([`split_units`]). Each unit
/// carries its entry's context ([`text_context`]) and kind; the units of
/// the text the cap cut are marked in `cut`.
pub fn capture_texts(entries: &[TranscriptEntry]) -> AssistantTexts {
    let mut kept: Vec<(String, &TranscriptEntry, bool)> = Vec::new();
    let mut cap_hit = false;
    let mut total = 0usize;
    for entry in entries.iter().rev() {
        let stripped = python_strip(&entry.text);
        if stripped.is_empty() {
            continue;
        }
        let len = stripped.chars().count();
        if total + len > MAX_CAPTURE_CHARS {
            let remaining = MAX_CAPTURE_CHARS - total;
            if remaining > 0 {
                kept.push((truncate(stripped, remaining).0, entry, true));
            }
            cap_hit = true;
            break;
        }
        total += len;
        kept.push((stripped.to_owned(), entry, false));
    }
    kept.reverse();
    let mut out = AssistantTexts {
        cap_hit,
        ..AssistantTexts::default()
    };
    for (text, entry, cut) in kept {
        let context = text_context(entry);
        for unit in split_units(&text) {
            out.texts.push(unit);
            out.prompts.push(context.clone());
            out.kinds.push(entry.kind);
            out.cut.push(cut);
        }
    }
    out
}
