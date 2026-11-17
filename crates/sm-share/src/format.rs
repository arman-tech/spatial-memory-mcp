//! The shared-file format (SPEC section 9, "Shared file format"; decision
//! B-78).
//!
//! One Markdown file per memory, with a small front matter:
//!
//! ```text
//! ---
//! schema: 1
//! id: 0193a8f0-5c1e-7000-8000-3f2a9c1e5b7d
//! type: decision
//! status: active
//! date: 2026-11-15
//! supersedes:
//! anchors: [table:dbo.Orders, service:order-api]
//! tags: [concurrency, sql]
//! commit: 4f2a9c1
//! ---
//! # Use optimistic concurrency on dbo.Orders
//!
//! Body in Markdown.
//! ```
//!
//! # Writer
//!
//! [`SharedDoc::render`] writes the nine keys in SPEC 9's order, every time.
//! A key with no value is written as the key and a colon with nothing after
//! it (`supersedes:`), the form SPEC 9's own example uses; an empty list is
//! written the same way. Lists are `[a, b]` on one line. Then the title as a
//! level-1 heading, a blank line and the body. LF only, one trailing newline;
//! the same document always gives the same bytes.
//!
//! # Parser
//!
//! [`SharedDoc::parse`] is a strict reader for this subset of YAML, not a
//! YAML library. It accepts single-line `key: value` entries and one-line
//! lists only, the nine known keys only, and `schema: 1` only. Values are
//! literal: there is no quoting, escaping or comment syntax. Every failure
//! names the line and the reason ([`ParseError`]). The title is the first
//! line of the Markdown part, which must be a level-1 heading (`# Title`);
//! later headings, including other level-1 headings, stay in the body.
//!
//! The parser also reads a few non-canonical forms a person may write by
//! hand: keys in another order, an omitted optional key, `[]` for an empty
//! list, a byte-order mark, CRLF or CR line endings and blank lines between
//! the front matter and the title. Rendering such a file gives the
//! canonical bytes.
//!
//! # Body normalisation
//!
//! The body is stored and written in one normal form ([`normalize_body`]):
//! CRLF and lone CR become LF, blank lines before the first line with text
//! are dropped, and trailing whitespace at the end of the body is removed.
//! Whitespace inside the body, including indentation of the first line and
//! trailing spaces on inner lines, is kept. A title is one line: CR and LF
//! in a stored title become spaces and the result is trimmed
//! ([`clean_title`]); a memory with no title uses `derive_title` (B-18).
//!
//! # Text hash
//!
//! [`text_hash`] is the SHA-256 of the Markdown part exactly as render
//! writes it (`# <title>\n\n<body>\n`). It decides whether sync re-embeds a
//! memory (SPEC 9, "Re-embed only when the text changes"; B-78). The same
//! function serves a stored memory and a parsed file.
//!
//! # Path and slug
//!
//! `.spatial-memory/shared/<type>/<slug>--<last 8 of id>.md` ([`shared_path`]).
//! The slug ([`slug`]) is the title in lower-case ASCII: each run of
//! characters that are not ASCII letters or digits becomes one `-`, leading
//! and trailing `-` are removed, it is cut to 40 characters (and any `-`
//! the cut leaves at the end is removed), and an empty result is `memory`.

use std::fmt;
use std::str::FromStr;

use sm_core::hash::sha256_hex;
use sm_core::ids::MemoryId;
use sm_core::time::Timestamp;
use sm_core::types::{AnchorRef, Memory, MemoryType, Status, derive_title, normalize_tags};
use sm_core::{CoreError, ErrorCode};

/// The shared folder, from the repository root, with `/` separators.
pub const SHARED_DIR: &str = ".spatial-memory/shared";

/// The format version this build reads and writes.
pub const SCHEMA: u32 = 1;

/// The `.gitattributes` written into the shared folder when it has none, so
/// a Windows checkout keeps LF and the blob ids do not change.
pub const GITATTRIBUTES: &str = "* text eol=lf\n";

/// The longest slug, in characters.
pub const SLUG_MAX_CHARS: usize = 40;

/// The title used when a memory has neither a title nor any text to derive
/// one from. The store refuses empty content, so this is a last resort.
const FALLBACK_TITLE: &str = "Untitled";

const MICROS_PER_DAY: i64 = 86_400 * 1_000_000;

/// The front matter keys, in the order the writer uses.
const KEYS: [&str; 9] = [
    "schema",
    "id",
    "type",
    "status",
    "date",
    "supersedes",
    "anchors",
    "tags",
    "commit",
];

/// Keys a file must have.
const REQUIRED: [&str; 5] = ["schema", "id", "type", "status", "date"];

/// Keys whose value is a one-line list.
const LIST_KEYS: [&str; 2] = ["anchors", "tags"];

/// One shared memory as its file holds it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedDoc {
    pub id: MemoryId,
    pub memory_type: MemoryType,
    pub status: Status,
    /// The creation date at midnight UTC.
    pub date: Timestamp,
    pub supersedes: Option<MemoryId>,
    pub anchors: Vec<AnchorRef>,
    pub tags: Vec<String>,
    /// The commit the memory was last verified against: 4 to 64 hex digits.
    pub commit: Option<String>,
    /// One line, trimmed, never empty.
    pub title: String,
    /// In the normal form of [`normalize_body`].
    pub body: String,
}

/// Why a file did not parse, and on which line (1-based, counted after
/// CRLF and CR are read as line ends).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    pub line: usize,
    pub reason: String,
}

impl ParseError {
    fn new(line: usize, reason: impl Into<String>) -> Self {
        ParseError {
            line,
            reason: reason.into(),
        }
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}: {}", self.line, self.reason)
    }
}

impl std::error::Error for ParseError {}

impl From<ParseError> for CoreError {
    fn from(e: ParseError) -> Self {
        CoreError::new(
            ErrorCode::InvalidInput,
            format!("shared file does not parse: {e}"),
        )
        .with_next_action("Fix the file so it matches the shared-file format, then run sync.")
    }
}

impl SharedDoc {
    /// The file for a stored memory, before redaction. A `commit_sha` that
    /// is not 4 to 64 hex digits is left out, because the parser would
    /// refuse it.
    pub fn from_memory(m: &Memory) -> SharedDoc {
        SharedDoc {
            id: m.id,
            memory_type: m.memory_type,
            status: m.status,
            date: date_of(m.created_at),
            supersedes: m.supersedes_id,
            anchors: m.anchors.clone(),
            tags: m.tags.clone(),
            commit: m
                .provenance
                .commit_sha
                .as_deref()
                .filter(|c| is_commit(c))
                .map(str::to_owned),
            title: clean_title(m.title.as_deref(), &m.content),
            body: normalize_body(&m.content),
        }
    }

    /// The file's bytes.
    pub fn render(&self) -> String {
        let mut out = String::with_capacity(256 + self.title.len() + self.body.len());
        out.push_str("---\n");
        out.push_str(&format!("schema: {SCHEMA}\n"));
        out.push_str(&format!("id: {}\n", self.id));
        out.push_str(&format!("type: {}\n", self.memory_type));
        out.push_str(&format!("status: {}\n", self.status));
        out.push_str(&format!("date: {}\n", self.date.date_string()));
        push_scalar(
            &mut out,
            "supersedes",
            self.supersedes.map(|s| s.to_string()),
        );
        push_list(
            &mut out,
            "anchors",
            self.anchors.iter().map(|a| a.to_string()),
        );
        push_list(&mut out, "tags", self.tags.iter().cloned());
        push_scalar(&mut out, "commit", self.commit.clone());
        out.push_str("---\n");
        out.push_str(&render_markdown(&self.title, &self.body));
        out
    }

    /// The hash of the Markdown part (B-78).
    pub fn text_hash(&self) -> String {
        text_hash(Some(&self.title), &self.body)
    }

    /// The file's path from the repository root.
    pub fn path(&self) -> String {
        shared_path(self.memory_type, &self.title, &self.id)
    }

    /// Reads a file strictly; see the module documentation.
    pub fn parse(text: &str) -> Result<SharedDoc, ParseError> {
        let text = text.strip_prefix('\u{feff}').unwrap_or(text);
        let text = text.replace("\r\n", "\n").replace('\r', "\n");
        let lines: Vec<&str> = text.split('\n').collect();
        // `split` gives one empty piece after a final LF; it is not a line.
        let n_lines = if text.ends_with('\n') {
            lines.len() - 1
        } else {
            lines.len()
        };
        let lines = &lines[..n_lines];

        match lines.first() {
            Some(l) if l.trim_end() == "---" => {}
            Some(_) => {
                return Err(ParseError::new(
                    1,
                    "the file must start with a front matter line `---`",
                ));
            }
            None => return Err(ParseError::new(1, "the file is empty")),
        }
        let close = lines
            .iter()
            .skip(1)
            .position(|l| l.trim_end() == "---")
            .map(|p| p + 1)
            .ok_or_else(|| {
                ParseError::new(
                    lines.len(),
                    "the front matter is not closed with a `---` line",
                )
            })?;
        let close_line = close + 1;

        let mut entries: Vec<Entry<'_>> = Vec::new();
        for (i, raw) in lines[1..close].iter().enumerate() {
            entries.push(read_entry(i + 2, raw)?);
        }

        // The schema comes first, so a newer file is reported as newer
        // rather than as having unknown keys.
        let schema = entries
            .iter()
            .find(|e| e.key == "schema")
            .ok_or_else(|| ParseError::new(close_line, "missing required key `schema`"))?;
        match schema.value.parse::<u64>() {
            Ok(v) if v == u64::from(SCHEMA) => {}
            Ok(v) => {
                return Err(ParseError::new(
                    schema.line,
                    format!("schema {v} is not supported; this build reads schema {SCHEMA}"),
                ));
            }
            Err(_) => {
                return Err(ParseError::new(
                    schema.line,
                    format!("schema must be a number, found {:?}", schema.value),
                ));
            }
        }

        let mut seen: Vec<&str> = Vec::new();
        for e in &entries {
            if !KEYS.contains(&e.key) {
                return Err(ParseError::new(
                    e.line,
                    format!(
                        "unknown key `{}`; the known keys are {}",
                        e.key,
                        KEYS.join(", ")
                    ),
                ));
            }
            if seen.contains(&e.key) {
                return Err(ParseError::new(
                    e.line,
                    format!("duplicate key `{}`", e.key),
                ));
            }
            seen.push(e.key);
            check_shape(e)?;
        }
        for key in REQUIRED {
            if !seen.contains(&key) {
                return Err(ParseError::new(
                    close_line,
                    format!("missing required key `{key}`"),
                ));
            }
        }
        let get = |key: &str| entries.iter().find(|e| e.key == key);

        let id_entry = get("id").expect("required key checked above");
        let id = canonical_id(id_entry, "id")?;
        let ty = get("type").expect("required key checked above");
        let memory_type =
            MemoryType::from_str(ty.value).map_err(|e| ParseError::new(ty.line, e.message))?;
        let st = get("status").expect("required key checked above");
        let status = Status::from_str(st.value).map_err(|e| ParseError::new(st.line, e.message))?;
        // Only automatic capture sets `pending`, and `share` refuses a
        // pending memory, so no shared file holds one (B-157).
        if !status.is_served() {
            return Err(ParseError::new(
                st.line,
                format!(
                    "status `{status}` is not allowed in a shared file; use active, superseded or retracted"
                ),
            ));
        }
        let dt = get("date").expect("required key checked above");
        let date =
            Timestamp::parse_date(dt.value).map_err(|e| ParseError::new(dt.line, e.message))?;
        let supersedes = match get("supersedes").filter(|e| !e.value.is_empty()) {
            None => None,
            Some(e) => {
                let s = canonical_id(e, "supersedes")?;
                if s == id {
                    return Err(ParseError::new(e.line, "a memory cannot supersede itself"));
                }
                Some(s)
            }
        };
        let anchors = match get("anchors") {
            None => Vec::new(),
            Some(e) => {
                let mut out: Vec<AnchorRef> = Vec::new();
                for item in list_items(e)? {
                    let a = AnchorRef::from_str(item)
                        .map_err(|err| ParseError::new(e.line, err.message))?;
                    if out.contains(&a) {
                        return Err(ParseError::new(e.line, format!("duplicate anchor `{a}`")));
                    }
                    out.push(a);
                }
                out
            }
        };
        let tags = match get("tags") {
            None => Vec::new(),
            Some(e) => {
                let mut out: Vec<String> = Vec::new();
                for item in list_items(e)? {
                    normalize_tags([item]).map_err(|err| ParseError::new(e.line, err.message))?;
                    if out.iter().any(|t| t == item) {
                        return Err(ParseError::new(e.line, format!("duplicate tag `{item}`")));
                    }
                    out.push(item.to_owned());
                }
                out
            }
        };
        let commit = match get("commit").filter(|e| !e.value.is_empty()) {
            None => None,
            Some(e) => {
                if !is_commit(e.value) {
                    return Err(ParseError::new(
                        e.line,
                        format!("commit {:?} must be 4 to 64 hexadecimal digits", e.value),
                    ));
                }
                Some(e.value.to_owned())
            }
        };

        // The Markdown part: blank lines, then `# Title`, then the body.
        let mut i = close + 1;
        while i < lines.len() && lines[i].trim().is_empty() {
            i += 1;
        }
        let Some(first) = lines.get(i) else {
            return Err(ParseError::new(
                lines.len().max(close_line),
                "missing the `# Title` line after the front matter",
            ));
        };
        let title = match first.strip_prefix("# ") {
            Some(t) if !t.trim().is_empty() => t.trim().to_owned(),
            _ => {
                return Err(ParseError::new(
                    i + 1,
                    "the Markdown part must start with a level-1 heading `# Title`",
                ));
            }
        };
        let body = normalize_body(&lines[i + 1..].join("\n"));

        Ok(SharedDoc {
            id,
            memory_type,
            status,
            date,
            supersedes,
            anchors,
            tags,
            commit,
            title,
            body,
        })
    }
}

/// One `key: value` line of the front matter, checked for layout only.
struct Entry<'a> {
    line: usize,
    key: &'a str,
    value: &'a str,
}

fn read_entry(line: usize, raw: &str) -> Result<Entry<'_>, ParseError> {
    if raw.trim().is_empty() {
        return Err(ParseError::new(
            line,
            "blank lines are not allowed in the front matter",
        ));
    }
    if raw.starts_with(char::is_whitespace) {
        return Err(ParseError::new(
            line,
            "indented line: values must fit on one line, with no nested or continued values",
        ));
    }
    if raw.starts_with('#') {
        return Err(ParseError::new(
            line,
            "comments are not allowed in the front matter",
        ));
    }
    if raw == "-" || raw.starts_with("- ") {
        return Err(ParseError::new(
            line,
            "block list item: write lists on one line, as [a, b]",
        ));
    }
    let Some((key, rest)) = raw.split_once(':') else {
        return Err(ParseError::new(line, "expected `key: value`"));
    };
    if key.is_empty() || !key.bytes().all(|b| b.is_ascii_lowercase() || b == b'_') {
        return Err(ParseError::new(
            line,
            format!("invalid key {key:?}: expected `key: value` with a lower-case key"),
        ));
    }
    if !rest.is_empty() && !rest.starts_with(' ') {
        return Err(ParseError::new(
            line,
            format!("expected a space after `{key}:`"),
        ));
    }
    Ok(Entry {
        line,
        key,
        value: rest.trim(),
    })
}

/// Refuses the YAML forms this format does not have, and a list where a
/// single value belongs (or the reverse).
fn check_shape(e: &Entry<'_>) -> Result<(), ParseError> {
    let v = e.value;
    if v.starts_with('{') {
        return Err(ParseError::new(
            e.line,
            format!(
                "`{}` has a nested value; only single-line values and one-line lists are allowed",
                e.key
            ),
        ));
    }
    if v.starts_with('|') || v.starts_with('>') {
        return Err(ParseError::new(
            e.line,
            format!(
                "`{}` has a multi-line value; values must fit on one line",
                e.key
            ),
        ));
    }
    let is_list = LIST_KEYS.contains(&e.key);
    if v.starts_with('[') {
        if !is_list {
            return Err(ParseError::new(
                e.line,
                format!("`{}` takes a single value, not a list", e.key),
            ));
        }
        if !v.ends_with(']') || v.len() < 2 {
            return Err(ParseError::new(
                e.line,
                format!("the `{}` list is not closed on this line", e.key),
            ));
        }
        if v[1..v.len() - 1].contains(['[', ']']) {
            return Err(ParseError::new(
                e.line,
                format!(
                    "`{}` has a nested list; only one-line lists of plain items are allowed",
                    e.key
                ),
            ));
        }
    } else if is_list && !v.is_empty() {
        return Err(ParseError::new(
            e.line,
            format!("`{}` must be a one-line list, as [a, b]", e.key),
        ));
    }
    Ok(())
}

/// The items of a one-line list entry, trimmed. Shape is checked first.
fn list_items<'a>(e: &Entry<'a>) -> Result<Vec<&'a str>, ParseError> {
    let v = e.value;
    if v.is_empty() {
        return Ok(Vec::new());
    }
    let inner = &v[1..v.len() - 1];
    if inner.trim().is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for item in inner.split(',') {
        let item = item.trim();
        if item.is_empty() {
            return Err(ParseError::new(
                e.line,
                format!("`{}` has an empty list item", e.key),
            ));
        }
        out.push(item);
    }
    Ok(out)
}

/// A memory id written in its canonical (lower-case, hyphenated) form.
fn canonical_id(e: &Entry<'_>, what: &str) -> Result<MemoryId, ParseError> {
    let id = MemoryId::parse(e.value).map_err(|err| ParseError::new(e.line, err.message))?;
    if id.to_string() != e.value {
        return Err(ParseError::new(
            e.line,
            format!(
                "{what} {:?} must be written as a lower-case hyphenated UUID ({id})",
                e.value
            ),
        ));
    }
    Ok(id)
}

fn push_scalar(out: &mut String, key: &str, value: Option<String>) {
    match value {
        Some(v) => out.push_str(&format!("{key}: {v}\n")),
        None => out.push_str(&format!("{key}:\n")),
    }
}

fn push_list(out: &mut String, key: &str, items: impl Iterator<Item = String>) {
    let items: Vec<String> = items.collect();
    if items.is_empty() {
        out.push_str(&format!("{key}:\n"));
    } else {
        out.push_str(&format!("{key}: [{}]\n", items.join(", ")));
    }
}

/// 4 to 64 hexadecimal digits: an abbreviated or full commit id.
pub fn is_commit(s: &str) -> bool {
    (4..=64).contains(&s.len()) && s.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Midnight UTC of a time's date.
pub fn date_of(t: Timestamp) -> Timestamp {
    Timestamp::from_micros(t.as_micros().div_euclid(MICROS_PER_DAY) * MICROS_PER_DAY)
}

/// The normal form of a body; see the module documentation.
pub fn normalize_body(content: &str) -> String {
    let text = content.replace("\r\n", "\n").replace('\r', "\n");
    let mut lines = text.split('\n').skip_while(|l| l.trim().is_empty());
    let mut out = String::with_capacity(text.len());
    if let Some(first) = lines.next() {
        out.push_str(first);
        for l in lines {
            out.push('\n');
            out.push_str(l);
        }
    }
    out.truncate(out.trim_end().len());
    out
}

/// A memory's title as its file writes it: the stored title on one line,
/// trimmed, or the title derived from the content (B-18) when there is none.
pub fn clean_title(title: Option<&str>, content: &str) -> String {
    let stored = title
        .map(|t| t.replace(['\r', '\n'], " ").trim().to_owned())
        .filter(|t| !t.is_empty());
    stored
        .or_else(|| derive_title(content))
        .unwrap_or_else(|| FALLBACK_TITLE.to_owned())
}

/// The Markdown part exactly as [`SharedDoc::render`] writes it.
pub fn render_markdown(title: &str, body: &str) -> String {
    if body.is_empty() {
        format!("# {title}\n")
    } else {
        format!("# {title}\n\n{body}\n")
    }
}

/// The hash that decides re-embedding (B-78): SHA-256, in hex, of the
/// Markdown part as render writes it. For a stored memory pass its title and
/// content; for a parsed file, its title and body.
pub fn text_hash(title: Option<&str>, content: &str) -> String {
    let markdown = render_markdown(&clean_title(title, content), &normalize_body(content));
    sha256_hex(markdown.as_bytes())
}

/// The hash of a stored memory's Markdown part.
pub fn memory_text_hash(m: &Memory) -> String {
    text_hash(m.title.as_deref(), &m.content)
}

/// The file-name slug of a title; see the module documentation.
pub fn slug(title: &str) -> String {
    let mut out = String::new();
    let mut dash = false;
    for c in title.chars() {
        if c.is_ascii_alphanumeric() {
            if dash && !out.is_empty() {
                out.push('-');
            }
            dash = false;
            out.push(c.to_ascii_lowercase());
        } else {
            dash = true;
        }
    }
    let mut cut: String = out.chars().take(SLUG_MAX_CHARS).collect();
    while cut.ends_with('-') {
        cut.pop();
    }
    if cut.is_empty() {
        "memory".to_owned()
    } else {
        cut
    }
}

/// `.spatial-memory/shared/<type>/<slug>--<last 8 of id>.md`.
pub fn shared_path(memory_type: MemoryType, title: &str, id: &MemoryId) -> String {
    format!(
        "{SHARED_DIR}/{memory_type}/{}--{}.md",
        slug(title),
        id.last8()
    )
}

/// The last 8 characters of the id in a shared file name, if the name has
/// the `--<8 hex>.md` suffix.
pub fn name_suffix(file_name: &str) -> Option<&str> {
    let stem = file_name.strip_suffix(".md")?;
    let (_, tail) = stem.rsplit_once("--")?;
    (tail.len() == 8 && tail.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')))
        .then_some(tail)
}
