//! PostToolUse extraction (SPEC section 16, Pipeline in the daemon, step 1;
//! AT-17): version 2's per-tool extractors, one 200,000 character safety cap
//! with the `...` marker, and the tool skip rules.
//!
//! The text is stored whole after the cap; the index cap
//! (`capture.max_index_tokens`) belongs to the embedder, not to this crate.
//! Lengths are counted in Unicode scalar values, as Python counts code
//! points.

use serde_json::Value;
use sm_core::protocol::Tool;

/// The safety cap on one extracted capture, in characters.
pub const MAX_CAPTURE_CHARS: usize = 200_000;

/// Appended to a text cut at the cap.
pub const TRUNCATION_MARKER: &str = "...";

/// Version 2's `_truncate`: `text` unchanged when it has at most
/// `max_chars` characters, otherwise its first `max_chars` characters and
/// [`TRUNCATION_MARKER`]. The flag says whether the text was cut.
pub fn truncate(text: &str, max_chars: usize) -> (String, bool) {
    match text.char_indices().nth(max_chars) {
        None => (text.to_owned(), false),
        Some((cut, _)) => {
            let mut out = String::with_capacity(cut + TRUNCATION_MARKER.len());
            out.push_str(&text[..cut]);
            out.push_str(TRUNCATION_MARKER);
            (out, true)
        }
    }
}

/// Version 2's `_safe_get`: the field `key` of `object` as text. A missing
/// field, a JSON `null` or an `object` that is not a JSON object gives
/// `default`; a string is taken as it is; anything else is written as
/// Python's `str()` would write it ([`python_str`]).
pub fn safe_get(object: &Value, key: &str, default: &str) -> String {
    match object.get(key) {
        None | Some(Value::Null) => default.to_owned(),
        Some(Value::String(s)) => s.clone(),
        Some(other) => python_str(other),
    }
}

/// A JSON value written as Python's `str()` writes the value `json.loads`
/// makes of it: `True`, `None`, `42`, `1.5`, `[1, 2]`, `{'a': 'b'}`. Object
/// keys come in the order `serde_json` keeps them.
pub fn python_str(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => python_repr(other),
    }
}

/// Python's truthiness of the value `json.loads` makes: `null`, `false`,
/// zero, the empty string, the empty list and the empty object are false.
pub fn python_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(map) => !map.is_empty(),
    }
}

fn python_repr(value: &Value) -> String {
    match value {
        Value::Null => "None".to_owned(),
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        Value::Number(n) => python_number(n),
        Value::String(s) => python_string_repr(s),
        Value::Array(items) => {
            let parts: Vec<String> = items.iter().map(python_repr).collect();
            format!("[{}]", parts.join(", "))
        }
        Value::Object(map) => {
            let parts: Vec<String> = map
                .iter()
                .map(|(k, v)| format!("{}: {}", python_string_repr(k), python_repr(v)))
                .collect();
            format!("{{{}}}", parts.join(", "))
        }
    }
}

fn python_number(n: &serde_json::Number) -> String {
    if let Some(i) = n.as_i64() {
        i.to_string()
    } else if let Some(u) = n.as_u64() {
        u.to_string()
    } else {
        python_float(n.as_f64().unwrap_or(f64::NAN))
    }
}

/// A float as Python's `repr` writes it: the shortest digits that read
/// back to the same value, positional from 1e-4 up to 1e16 and with an
/// exponent of at least two digits outside that range.
pub fn python_float(f: f64) -> String {
    if f.is_nan() {
        return "nan".to_owned();
    }
    if f.is_infinite() {
        return if f > 0.0 { "inf" } else { "-inf" }.to_owned();
    }
    if f == 0.0 {
        return if f.is_sign_negative() { "-0.0" } else { "0.0" }.to_owned();
    }
    // `{:e}` gives the shortest round-trip digits: `-1.2345e-7`.
    let sci = format!("{f:e}");
    let (mantissa, exp) = sci.split_once('e').unwrap_or((sci.as_str(), "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let (sign, mantissa) = match mantissa.strip_prefix('-') {
        Some(rest) => ("-", rest),
        None => ("", mantissa),
    };
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    if (-4..16).contains(&exp) {
        let point = exp + 1;
        let body = if point <= 0 {
            format!("0.{}{}", "0".repeat(point.unsigned_abs() as usize), digits)
        } else {
            let point = point as usize;
            if digits.len() <= point {
                format!("{}{}.0", digits, "0".repeat(point - digits.len()))
            } else {
                format!("{}.{}", &digits[..point], &digits[point..])
            }
        };
        format!("{sign}{body}")
    } else {
        let mant = if digits.len() > 1 {
            format!("{}.{}", &digits[..1], &digits[1..])
        } else {
            digits
        };
        let esign = if exp < 0 { '-' } else { '+' };
        format!("{sign}{mant}e{esign}{:02}", exp.unsigned_abs())
    }
}

/// A string as Python's `repr` writes it: single quotes unless the string
/// holds a single quote and no double quote; backslash, the quote, `\n`,
/// `\r` and `\t` escaped; other control characters as `\xNN`. Python also
/// escapes some non-control characters it considers unprintable (format
/// characters, unassigned code points); those are written as they are.
pub fn python_string_repr(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || (0x7f..0xa0).contains(&(c as u32)) => {
                out.push_str(&format!("\\x{:02x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// A JSON value written as Python's `json.dumps(value, ensure_ascii=False)`
/// writes it: `, ` and `: ` separators, non-ASCII characters as they are,
/// floats as Python writes them. Object keys come in the order
/// `serde_json` keeps them.
pub fn python_json_dumps(value: &Value) -> String {
    let mut out = String::new();
    write_json(value, &mut out);
    out
}

fn write_json(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => out.push_str(&python_number(n)),
        Value::String(s) => write_json_string(s, out),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_json(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (i, (k, v)) in map.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_json_string(k, out);
                out.push_str(": ");
                write_json(v, out);
            }
            out.push('}');
        }
    }
}

fn write_json_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// The payload's `tool_response` as text, as version 2's dispatcher made
/// it: absent is empty, a string is taken as it is, and anything else
/// (`null` included) is JSON-encoded with [`python_json_dumps`].
pub fn tool_response_text(payload: &Value) -> String {
    match payload.get("tool_response") {
        None => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(other) => python_json_dumps(other),
    }
}

fn file_prefixed(label: &str, path: &str, body: &str) -> String {
    if body.is_empty() {
        return String::new();
    }
    if path.is_empty() {
        body.to_owned()
    } else {
        format!("{label}: {path}\n{body}")
    }
}

/// Version 2's Edit extractor: `File: <file_path>` on its own line when
/// there is a path, then `new_string`. Empty when `new_string` is empty.
pub fn extract_edit(tool_input: &Value) -> String {
    file_prefixed(
        "File",
        &safe_get(tool_input, "file_path", ""),
        &safe_get(tool_input, "new_string", ""),
    )
}

/// Version 2's Write extractor: `File: <file_path>` on its own line when
/// there is a path, then `content`. Empty when `content` is empty.
pub fn extract_write(tool_input: &Value) -> String {
    file_prefixed(
        "File",
        &safe_get(tool_input, "file_path", ""),
        &safe_get(tool_input, "content", ""),
    )
}

/// Version 2's Bash extractor: `Command: <command>`, then
/// `\nOutput: <tool_response>` when there is a response. Empty when the
/// command is empty.
pub fn extract_bash(tool_input: &Value, tool_response: &str) -> String {
    let command = safe_get(tool_input, "command", "");
    if command.is_empty() {
        return String::new();
    }
    if tool_response.is_empty() {
        format!("Command: {command}")
    } else {
        format!("Command: {command}\nOutput: {tool_response}")
    }
}

/// Version 2's NotebookEdit extractor: `Notebook: <notebook_path>` on its
/// own line when there is a path, then `new_source`. Empty when
/// `new_source` is empty.
pub fn extract_notebook_edit(tool_input: &Value) -> String {
    file_prefixed(
        "Notebook",
        &safe_get(tool_input, "notebook_path", ""),
        &safe_get(tool_input, "new_source", ""),
    )
}

/// Version 2's MCP extractor: `Tool: <tool_name>` and
/// `Response: <tool_response>`, each only when not empty, joined by a
/// newline. The tool input is not used.
pub fn extract_mcp_tool(_tool_input: &Value, tool_response: &str, tool_name: &str) -> String {
    let mut lines = Vec::with_capacity(2);
    if !tool_name.is_empty() {
        lines.push(format!("Tool: {tool_name}"));
    }
    if !tool_response.is_empty() {
        lines.push(format!("Response: {tool_response}"));
    }
    lines.join("\n")
}

/// Extracted text and whether the safety cap cut it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Extracted {
    /// The text, at most [`MAX_CAPTURE_CHARS`] characters plus the marker.
    pub text: String,
    /// Whether the text was cut at the safety cap.
    pub cap_hit: bool,
    /// The length of the extracted text before the cap, in characters.
    pub original_chars: usize,
}

/// Version 2's `extract_content`: the per-tool extractor for Edit, Write,
/// Bash, NotebookEdit and any `mcp__` tool, then the safety cap, once, on
/// the joined text. Any other tool gives empty text.
pub fn extract_content(tool_name: &str, tool_input: &Value, tool_response: &str) -> Extracted {
    let raw = match tool_name {
        "Edit" => extract_edit(tool_input),
        "Write" => extract_write(tool_input),
        "Bash" => extract_bash(tool_input, tool_response),
        "NotebookEdit" => extract_notebook_edit(tool_input),
        name if name.starts_with("mcp__") => extract_mcp_tool(tool_input, tool_response, name),
        _ => {
            return Extracted {
                text: String::new(),
                cap_hit: false,
                original_chars: 0,
            };
        }
    };
    let original_chars = raw.chars().count();
    let (text, cap_hit) = truncate(&raw, MAX_CAPTURE_CHARS);
    Extracted {
        text,
        cap_hit,
        original_chars,
    }
}

/// Tools version 1 never captured (`SKIP_TOOLS`).
pub const SKIP_TOOLS: [&str; 7] = [
    "Glob",
    "Grep",
    "LSP",
    "Read",
    "Task",
    "WebFetch",
    "WebSearch",
];

/// The Claude Code prefix of this server's own tools.
pub const SPATIAL_MEMORY_PREFIX: &str = "mcp__spatial-memory__";

/// Version 1's 23 tool names, recognised in Cursor's `MCP:<name>` form.
pub const V1_TOOL_NAMES: [&str; 23] = [
    "consolidate",
    "decay",
    "delete_namespace",
    "export_memories",
    "extract",
    "forget",
    "forget_batch",
    "health",
    "hybrid_recall",
    "import_memories",
    "journey",
    "namespaces",
    "nearby",
    "recall",
    "regions",
    "reinforce",
    "remember",
    "remember_batch",
    "rename_namespace",
    "setup_hooks",
    "stats",
    "visualize",
    "wander",
];

/// Why a tool is never captured.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolSkip {
    /// The payload names no tool.
    EmptyToolName,
    /// One of [`SKIP_TOOLS`].
    SkipTool(String),
    /// One of this server's own tools.
    SpatialMemoryTool,
}

impl ToolSkip {
    /// Version 1's reason string: `empty_tool_name`, `skip_tool:<name>` or
    /// `spatial_memory_tool`.
    pub fn reason(&self) -> String {
        match self {
            ToolSkip::EmptyToolName => "empty_tool_name".to_owned(),
            ToolSkip::SkipTool(name) => format!("skip_tool:{name}"),
            ToolSkip::SpatialMemoryTool => "spatial_memory_tool".to_owned(),
        }
    }
}

/// Whether `tool_name` is one of this server's own tools:
///
/// - Claude Code: `mcp__spatial-memory__<tool>`, or the plugin form
///   `mcp__plugin_<plugin>_spatial-memory__<tool>`, for any tool;
/// - Cursor: `MCP:<name>` where `<name>` is one of version 3's 13 tool
///   names or version 1's 23. Cursor's form does not name the server, so
///   another server's tool with one of these names is skipped too.
pub fn is_spatial_memory_tool(tool_name: &str) -> bool {
    if tool_name.starts_with(SPATIAL_MEMORY_PREFIX) {
        return true;
    }
    if let Some(rest) = tool_name.strip_prefix("mcp__")
        && let Some((server, _tool)) = rest.split_once("__")
        && server.starts_with("plugin_")
        && server.ends_with("_spatial-memory")
    {
        return true;
    }
    if let Some(name) = tool_name.strip_prefix("MCP:") {
        return V1_TOOL_NAMES.contains(&name) || Tool::ALL.iter().any(|t| t.as_str() == name);
    }
    false
}

/// Version 1's `should_skip_tool`, extended to version 3's tool names (see
/// [`is_spatial_memory_tool`]). `None` when the tool may be captured.
pub fn skip_rule(tool_name: &str) -> Option<ToolSkip> {
    if tool_name.is_empty() {
        Some(ToolSkip::EmptyToolName)
    } else if SKIP_TOOLS.contains(&tool_name) {
        Some(ToolSkip::SkipTool(tool_name.to_owned()))
    } else if is_spatial_memory_tool(tool_name) {
        Some(ToolSkip::SpatialMemoryTool)
    } else {
        None
    }
}
