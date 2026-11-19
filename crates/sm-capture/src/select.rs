//! Candidate selection from the transcript (SPEC section 16, Candidate
//! selection; decision B-179): the prose filter for tool text and the
//! splitting of a text into the units offered to the judge.
//!
//! B-179 measured where the statements worth keeping are in the owner's
//! sessions. Most are in the summaries Claude Code writes when it compacts a
//! conversation; some are in the user's messages, the assistant's final
//! replies, the new text of an edit, and repository files read with a tool.
//! Raw command output and whole files are never offered: from a tool's text
//! only its prose paragraphs are ([`prose`]), up to a cap.
//!
//! Every text offered is split into units of at most [`UNIT_MAX`]
//! characters ([`split_units`]); each unit is one pending capture the judge
//! settles. Lengths are counted in Unicode scalar values.

use std::sync::LazyLock;

use regex::Regex;

/// The longest unit offered to the judge, in characters.
pub const UNIT_MAX: usize = 1_200;

/// A unit shorter than this, stripped, is not offered.
pub const UNIT_MIN: usize = 20;

/// The most prose taken from one tool result (Bash, Read, Grep).
pub const TOOL_RESULT_PROSE_MAX: usize = 1_500;

/// The most prose taken from the new text of one Edit, MultiEdit or Write.
pub const TOOL_INPUT_PROSE_MAX: usize = 4_000;

/// A prose line has at least this many words of two or more ASCII letters,
/// and at least this many pieces separated by white space.
pub const MIN_WORDS: usize = 6;

/// In a prose line, letters and spaces make up at least this share of the
/// characters.
pub const MIN_LETTERS: f64 = 0.80;

/// A paragraph of prose lines is offered when its stripped lines hold at
/// least this many characters together.
pub const MIN_PARAGRAPH: usize = 120;

/// A line-number prefix as tools print it (`12→`, `src/a.rs:12:`, `12\t`,
/// `12  `). The second alternative needs a non-space after one or two
/// spaces; the group ends where the prefix does.
static LINE_NUMBER: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^\s*(?:[\w./\\:-]*?:)?\d+(?:([:\t\x{2192}-])|(\s{1,2})\S)")
        .expect("the line number expression is valid")
});

/// A comment, quote or list marker at the start of a line.
static MARKER: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^\s*(?://[!/]?|#+|--|;;|/\*+|\*/|\*|>|[-+]\s|\d+[.)]\s)\s*")
        .expect("the marker expression is valid")
});

/// A word: two or more ASCII letters.
static WORD: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[A-Za-z]{2,}").expect("the word expression is valid"));

/// The end of a sentence: punctuation followed by white space.
static SENTENCE_END: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[.!?;:]\s+").expect("the sentence end expression is valid"));

/// A heading line: a Markdown heading, or a short numbered title ending in
/// a colon (`2. Root causes:`).
static HEADING: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^\s*(?:#{1,6}\s|\d+\.\s+[A-Z][^.]{0,70}:\s*$)")
        .expect("the heading expression is valid")
});

/// Whether a line is a heading as a compaction summary writes one: `#`
/// marks, or a list number and a capitalised name ending in `:`.
pub(crate) fn is_heading(line: &str) -> bool {
    HEADING.is_match(line)
}

/// A heading's leading marks: `#`, a list number, or `**`.
static HEADING_MARKS: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^\s*(?:#{1,6}\s*|\d+\.\s+|\*\*)").expect("the heading marks expression is valid")
});

/// A blank line between paragraphs.
static PARAGRAPH_BREAK: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\n\s*\n").expect("the paragraph expression is valid"));

/// A line without its line-number prefix and up to two markers, trimmed.
fn strip_line(line: &str) -> &str {
    let mut rest = line;
    if let Some(caps) = LINE_NUMBER.captures(rest) {
        let end = caps.get(1).or_else(|| caps.get(2)).map_or(0, |m| m.end());
        rest = &rest[end..];
    }
    for _ in 0..2 {
        if let Some(m) = MARKER.find(rest) {
            rest = &rest[m.end()..];
        }
    }
    rest.trim()
}

/// Whether a line is prose: after [`strip_line`], at least [`MIN_WORDS`]
/// words and [`MIN_WORDS`] pieces separated by white space (a path or an
/// expression is one piece however many words it holds), and letters and
/// spaces at least [`MIN_LETTERS`] of it.
pub fn is_prose_line(line: &str) -> bool {
    let s = strip_line(line);
    if WORD.find_iter(s).count() < MIN_WORDS || s.split_whitespace().count() < MIN_WORDS {
        return false;
    }
    let total = s.chars().count().max(1);
    let letters = s.chars().filter(|c| c.is_alphabetic() || *c == ' ').count();
    letters as f64 / total as f64 >= MIN_LETTERS
}

/// The prose of a tool's text: each run of consecutive prose lines whose
/// stripped lines hold at least [`MIN_PARAGRAPH`] characters, its lines
/// trimmed, the runs joined by a blank line. Runs are taken in order until
/// the next would pass `cap` characters; it and every later run are left
/// out. Code, table rows of values, paths, numbers and short lines are not
/// prose.
pub fn prose(text: &str, cap: usize) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut run: Vec<&str> = Vec::new();
    let mut total = 0usize;
    for line in text.split('\n').chain(std::iter::once("")) {
        if is_prose_line(line) {
            run.push(line.trim());
            continue;
        }
        if run.is_empty() {
            continue;
        }
        let held: usize = run.iter().map(|l| strip_line(l).chars().count()).sum();
        if held >= MIN_PARAGRAPH {
            let paragraph = run.join("\n");
            let len = paragraph.chars().count();
            if total + len > cap {
                break;
            }
            total += len + 2;
            out.push(paragraph);
        }
        run.clear();
    }
    out.join("\n\n")
}

/// The first `n` characters of `s`, and the rest.
fn split_chars(s: &str, n: usize) -> (&str, &str) {
    match s.char_indices().nth(n) {
        Some((i, _)) => s.split_at(i),
        None => (s, ""),
    }
}

/// A line's sentences: split after `.`, `!`, `?`, `;` or `:` followed by
/// white space, the white space dropped.
fn sentences(line: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    for m in SENTENCE_END.find_iter(line) {
        // The punctuation is one ASCII byte and stays with its sentence.
        out.push(&line[start..m.start() + 1]);
        start = m.end();
    }
    out.push(&line[start..]);
    out
}

/// A text as the units offered to the judge, in order.
///
/// The text is cut into paragraphs at blank lines and the paragraphs into
/// lines; blank lines are dropped and each line loses its trailing white
/// space. A line longer than [`UNIT_MAX`] is cut into sentences, packed
/// together up to [`UNIT_MAX`], and a sentence longer than that is cut at
/// [`UNIT_MAX`] characters. The pieces are then packed into units of at
/// most [`UNIT_MAX`] characters, joined by a newline within a paragraph and
/// a blank line between paragraphs; a heading line always starts a new
/// unit. Units shorter than [`UNIT_MIN`] characters, stripped, are dropped.
pub fn split_units(text: &str) -> Vec<String> {
    // Each piece, and whether it starts a paragraph (or is a heading).
    let mut pieces: Vec<(String, bool)> = Vec::new();
    for paragraph in PARAGRAPH_BREAK.split(text) {
        let mut first = true;
        for line in paragraph.split('\n') {
            if line.trim().is_empty() {
                continue;
            }
            let line = line.trim_end();
            if line.chars().count() > UNIT_MAX {
                let mut current = String::new();
                for sentence in sentences(line) {
                    let mut s = sentence;
                    while s.chars().count() > UNIT_MAX {
                        if !current.is_empty() {
                            pieces.push((std::mem::take(&mut current), false));
                        }
                        let (head, tail) = split_chars(s, UNIT_MAX);
                        pieces.push((head.to_owned(), false));
                        s = tail;
                    }
                    if !current.is_empty()
                        && current.chars().count() + 1 + s.chars().count() > UNIT_MAX
                    {
                        pieces.push((std::mem::replace(&mut current, s.to_owned()), false));
                    } else if current.is_empty() {
                        current = s.to_owned();
                    } else {
                        current.push(' ');
                        current.push_str(s);
                    }
                }
                if !current.is_empty() {
                    pieces.push((current, false));
                }
            } else {
                pieces.push((line.to_owned(), first || HEADING.is_match(line)));
            }
            first = false;
        }
    }
    let mut units = Vec::new();
    let mut current = String::new();
    for (piece, paragraph_start) in pieces {
        let heading = HEADING.is_match(&piece);
        let sep = if paragraph_start { "\n\n" } else { "\n" };
        if !current.is_empty()
            && (heading || current.chars().count() + sep.len() + piece.chars().count() > UNIT_MAX)
        {
            units.push(std::mem::replace(&mut current, piece));
        } else if current.is_empty() {
            current = piece;
        } else {
            current.push_str(sep);
            current.push_str(&piece);
        }
    }
    if !current.is_empty() {
        units.push(current);
    }
    units.retain(|u| u.trim().chars().count() >= UNIT_MIN);
    units
}

/// The sections of a compaction summary as Claude Code writes them, each
/// by the start of its heading's name, and whether its units are offered
/// (B-189). The user's messages are offered where they were written, so
/// their verbatim copy in a summary is not; the pending tasks are a task
/// list. Measured on the design data, the design set and the first
/// hold-out (CAPTURE-RCA 8.9): the two sections
/// held 124 of 3,136 candidates and four worth-keeping groups that no
/// other candidate held, none of them must keep.
const SUMMARY_SECTIONS: &[(&str, bool)] = &[
    ("primary request", true),
    ("what the user asked", true),
    ("key technical", true),
    ("files and code", true),
    ("errors and fix", true),
    ("problem solv", true),
    ("all user messages", false),
    ("user messages", false),
    ("pending task", false),
    ("current work", true),
    ("optional next", true),
    ("next step", true),
];

/// A heading line's name as [`SUMMARY_SECTIONS`] matches it: leading marks
/// removed, trailing `:` and `*` removed, white space made single spaces,
/// lower-cased, at most 40 characters.
fn section_name(line: &str) -> String {
    let mut s = line;
    while let Some(m) = HEADING_MARKS.find(s)
        && m.end() > 0
    {
        s = &s[m.end()..];
    }
    let s = s.trim().trim_end_matches([':', '*']).trim();
    s.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
        .chars()
        .take(40)
        .collect()
}

/// The units of a compaction summary that are offered, in order, and how
/// many are not (B-189). Each unit is in the section of the last heading
/// named in [`SUMMARY_SECTIONS`] that opens it or a unit before it in the
/// same text; a heading the list does not name, such as a sub-heading,
/// leaves the unit in the section it is in. Every unit before the first
/// named heading, such as the summary's preamble, is offered.
pub fn summary_units(text: &str) -> (Vec<String>, usize) {
    let mut offered = true;
    let mut kept = Vec::new();
    let mut dropped = 0;
    for unit in split_units(text) {
        let first = unit.lines().next().unwrap_or("");
        if HEADING.is_match(first) {
            let name = section_name(first);
            if let Some((_, offer)) = SUMMARY_SECTIONS.iter().find(|(k, _)| name.starts_with(k)) {
                offered = *offer;
            }
        }
        if offered {
            kept.push(unit);
        } else {
            dropped += 1;
        }
    }
    (kept, dropped)
}
