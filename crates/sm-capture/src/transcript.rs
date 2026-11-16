//! The transcript delta reader and the assistant-text extractor for
//! PreCompact and Stop (SPEC section 16, Pipeline in the daemon, step 1;
//! AT-14, AT-17), ported from version 2's `transcript_reader.py` and
//! `transcript_extractor.py`.
//!
//! The reader is stateless: the caller passes the byte offset reached last
//! time and stores the new one. Reading the transcript file is the only
//! input or output this crate does.

use std::fs::File;
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use regex::Regex;
use serde_json::Value;
use sm_core::hash::python_strip;

use crate::extract::{MAX_CAPTURE_CHARS, python_str, python_truthy, truncate};

/// Transcript lines longer than this, newline included, are skipped.
pub const MAX_LINE_BYTES: usize = 1_048_576;

/// The longest session identifier version 1 accepted.
pub const MAX_SESSION_ID_CHARS: usize = 128;

/// Only lines holding these bytes are parsed.
const ASSISTANT_MARKER: &str = "\"assistant\"";

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

/// One assistant message from the transcript.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscriptEntry {
    /// Always `assistant`.
    pub role: String,
    /// The message's text blocks, each stripped, joined by a blank line.
    pub text: String,
    /// The entry's `timestamp`, as Python's `str()` writes it.
    pub timestamp: String,
    /// The entry's `uuid`, as Python's `str()` writes it.
    pub uuid: String,
    /// The entry's `type`; always `assistant`.
    pub entry_type: String,
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
    })
}

/// What one read of the transcript found.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TranscriptRead {
    /// The assistant entries after the offset, in file order.
    pub entries: Vec<TranscriptEntry>,
    /// Where the next read starts: the end of the file as read.
    pub new_offset: u64,
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
/// The reader keeps no state. If the daemon restarts and reads a transcript
/// from 0 again, the content hash dedupe against memories already stored
/// (SPEC section 16, step 4; B-29) catches what was captured before.
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
    /// Whether the safety cap cut or dropped any text.
    pub cap_hit: bool,
}

/// Version 2's `extract_assistant_text`: each assistant entry's stripped,
/// non-blank text, in order, with one [`MAX_CAPTURE_CHARS`] cap over all
/// of them together. A text that would pass the cap is cut there, gets the
/// `...` marker, and ends the list.
pub fn extract_assistant_texts(entries: &[TranscriptEntry]) -> AssistantTexts {
    let mut out = AssistantTexts::default();
    let mut total = 0usize;
    for entry in entries {
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
            }
            out.cap_hit = true;
            break;
        }
        total += len;
        out.texts.push(text);
    }
    out
}
