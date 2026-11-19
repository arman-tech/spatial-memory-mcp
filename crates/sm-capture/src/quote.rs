//! A text's own sentences: the memory the judge's spans mark (B-173,
//! B-174, B-191), and the sentence keys the lean offering compares
//! (B-189).
//!
//! The judge never writes a memory. It marks each statement worth keeping
//! by where it starts and where it ends in the candidate's text: a few
//! words copied from the statement's start and a few from its end. The
//! memory is made of the sentences of the text from the start to the end,
//! copied from the text, in the text's order. A span whose start or end is
//! not in the text is dropped, so a memory can state nothing the text does
//! not state, and a span that starts or ends inside a sentence brings the
//! whole sentence, so no qualifier of that sentence is lost.
//!
//! - **Comparison.** Phrase and text are compared with `*` and `` ` ``
//!   removed, typographic quotation marks made plain and each run of
//!   whitespace made one space. A phrase's leading list marker, heading
//!   marks and quotation marks, and its closing punctuation, are ignored.
//!   A phrase must start and end at a word boundary of the text.
//! - **Span.** The start phrase is looked for first. The end phrase is the
//!   first one found at or after the start phrase's first character, so a
//!   span ends where its statement does even when the end words occur
//!   earlier in the text, and a span whose end is only before its start is
//!   dropped. A span whose start and end are the same phrase is that
//!   phrase.
//! - **Sentences.** The text is cut into lines, and a line after `.`, `!`
//!   or `?` (and any closing bracket, quotation mark or markdown mark)
//!   followed by whitespace and a character that is not a lower-case
//!   letter, so "e.g. the" stays one sentence. The `.` of a list number
//!   such as `1.` ends no sentence.
//! - **Parents** (B-174). A sentence that starts on an indented line, such
//!   as a nested list item or the continuation of a list item, is read
//!   under the nearest line above it that is less indented. Without that
//!   line it would read as part of whatever the memory puts before it. So
//!   it brings the opening sentence of that line, and of that line's own
//!   parent, up to a line that is not indented.
//! - **Memory.** The sentences found and their parents, merged where they
//!   touch, each trimmed, joined by line breaks.

use std::collections::HashSet;

/// What the spans of one verdict found in its text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Found {
    /// The sentences of the text that the spans found; empty when none was
    /// found.
    pub memory: String,
    /// Spans looked for (a span with a blank start or end is not counted).
    pub looked: usize,
    /// Spans not found in the text.
    pub dropped: usize,
}

/// Looks for each `(start, end)` span in `text` and returns the sentences
/// it covers.
pub fn spanned<'a, I>(text: &str, spans: I) -> Found
where
    I: IntoIterator<Item = (&'a str, &'a str)>,
{
    let canon = Canon::of(text);
    let sentences = sentences(text);
    let mut found: Vec<(usize, usize)> = Vec::new();
    let (mut looked, mut dropped) = (0, 0);
    for (start, end) in spans {
        let (start_key, end_key) = (key_of(start), key_of(end));
        if start_key.is_empty() || end_key.is_empty() {
            continue;
        }
        looked += 1;
        let Some(first) = canon.find_from(&start_key, 0) else {
            dropped += 1;
            continue;
        };
        let Some(last) = canon
            .find_from(&end_key, first)
            .map(|i| i + end_key.len() - 1)
        else {
            dropped += 1;
            continue;
        };
        let (from, to) = (canon.at[first].0, canon.at[last].1);
        let hit = sentences.iter().filter(|(a, b)| *a < to && *b > from).fold(
            None,
            |acc: Option<(usize, usize)>, &(a, b)| {
                Some(acc.map_or((a, b), |(x, y)| (x.min(a), y.max(b))))
            },
        );
        match hit {
            Some(span) => {
                found.push(span);
                found.extend(parents(text, &sentences, span.0));
            }
            None => dropped += 1,
        }
    }
    found.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (a, b) in found {
        match merged.last_mut() {
            Some((_, y)) if a <= *y || text[*y..a].trim().is_empty() => *y = (*y).max(b),
            _ => merged.push((a, b)),
        }
    }
    let memory = merged
        .iter()
        .map(|&(a, b)| text[a..b].trim())
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    Found {
        memory,
        looked,
        dropped,
    }
}

/// The shortest sentence key [`sentence_keys`] gives, in characters.
pub const SENTENCE_KEY_MIN: usize = 20;

/// The keys of the sentences of `text` (B-189): each sentence as phrases
/// are compared, lower-cased, without its list marker, heading marks and
/// closing punctuation; only keys of at least [`SENTENCE_KEY_MIN`]
/// characters.
pub fn sentence_keys(text: &str) -> HashSet<String> {
    sentences(text)
        .into_iter()
        .map(|(a, b)| {
            key_of(&text[a..b])
                .into_iter()
                .collect::<String>()
                .to_lowercase()
        })
        .filter(|k| k.chars().count() >= SENTENCE_KEY_MIN)
        .collect()
}

/// The opening sentence of each line that the line holding byte `at` is
/// read under (B-174): the nearest non-blank line above it that is less
/// indented, then that line's own parent, up to a line that is not
/// indented.
fn parents(text: &str, sentences: &[(usize, usize)], at: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut start = line_start(text, at);
    loop {
        let indent = indent_of(line_at(text, start));
        if indent == 0 {
            break;
        }
        let mut above = start;
        let parent = loop {
            if above == 0 {
                break None;
            }
            above = line_start(text, above - 1);
            let line = line_at(text, above);
            if !line.trim().is_empty() && indent_of(line) < indent {
                break Some(above);
            }
        };
        let Some(p) = parent else { break };
        let first = p + indent_of(line_at(text, p));
        if let Some(&s) = sentences.iter().find(|(a, _)| *a == first) {
            out.push(s);
        }
        start = p;
    }
    out
}

/// The byte at which the line holding byte `at` starts.
fn line_start(text: &str, at: usize) -> usize {
    text[..at].rfind('\n').map_or(0, |i| i + 1)
}

/// The line that starts at byte `start`, without its line break.
fn line_at(text: &str, start: usize) -> &str {
    let rest = &text[start..];
    &rest[..rest.find('\n').unwrap_or(rest.len())]
}

/// The bytes of leading spaces and tabs of `line`.
fn indent_of(line: &str) -> usize {
    line.len() - line.trim_start_matches([' ', '\t']).len()
}

/// One character as phrases and texts are compared: `None` for a
/// character that is left out.
fn plain(c: char) -> Option<char> {
    match c {
        '*' | '`' => None,
        '\u{2018}' | '\u{2019}' => Some('\''),
        '\u{201C}' | '\u{201D}' => Some('"'),
        c => Some(c),
    }
}

/// A text as it is compared, with the byte range in the original of each
/// of its characters.
struct Canon {
    chars: Vec<char>,
    at: Vec<(usize, usize)>,
}

impl Canon {
    fn of(text: &str) -> Canon {
        let mut chars = Vec::new();
        let mut at = Vec::new();
        let mut space: Option<(usize, usize)> = None;
        for (i, c) in text.char_indices() {
            let end = i + c.len_utf8();
            if c.is_whitespace() {
                space = Some(space.map_or((i, end), |(s, _)| (s, end)));
                continue;
            }
            let Some(p) = plain(c) else { continue };
            if let Some(range) = space.take()
                && !chars.is_empty()
            {
                chars.push(' ');
                at.push(range);
            }
            chars.push(p);
            at.push((i, end));
        }
        Canon { chars, at }
    }

    /// The index of the first compared character at which `key` is found at
    /// or after compared character `from`, starting and ending at a word
    /// boundary.
    fn find_from(&self, key: &[char], from: usize) -> Option<usize> {
        if key.is_empty() || key.len() > self.chars.len() || from > self.chars.len() - key.len() {
            return None;
        }
        let word = |c: char| c.is_alphanumeric() || c == '_';
        (from..=self.chars.len() - key.len()).find(|&i| {
            self.chars[i..i + key.len()] == *key
                && (i == 0 || !(word(self.chars[i - 1]) && word(key[0])))
                && self
                    .chars
                    .get(i + key.len())
                    .is_none_or(|&n| !(word(n) && word(key[key.len() - 1])))
        })
    }
}

/// A phrase as it is looked for: compared form, without a leading list
/// marker, heading marks or quotation marks, and without closing
/// punctuation or quotation marks.
fn key_of(phrase: &str) -> Vec<char> {
    let c = Canon::of(phrase).chars;
    let mut s: &[char] = &c;
    loop {
        let before = s.len();
        if let Some(rest) = strip_marker(s) {
            s = rest;
        }
        while let [first, rest @ ..] = s
            && (matches!(first, '"' | '\'' | '#' | '>') || first.is_whitespace())
        {
            s = rest;
        }
        if s.len() == before {
            break;
        }
    }
    while let [rest @ .., last] = s
        && (matches!(last, '.' | ',' | ';' | ':' | '!' | '?' | '"' | '\'') || last.is_whitespace())
    {
        s = rest;
    }
    s.to_vec()
}

/// The rest of `s` after a leading list marker and its space: `-`, `+`,
/// `•`, or a number followed by `.` or `)`.
fn strip_marker(s: &[char]) -> Option<&[char]> {
    match s {
        ['-' | '+' | '\u{2022}', ' ', rest @ ..] => Some(rest),
        _ => {
            let digits = s.iter().take_while(|c| c.is_ascii_digit()).count();
            match &s[digits..] {
                ['.' | ')', ' ', rest @ ..] if digits > 0 => Some(rest),
                _ => None,
            }
        }
    }
}

/// The byte ranges of the sentences of `text`, trimmed, in order.
fn sentences(text: &str) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut line_start = 0;
    for line in text.split_inclusive('\n') {
        let base = line_start;
        line_start += line.len();
        let mut start = 0;
        let chars: Vec<(usize, char)> = line.char_indices().collect();
        let mut i = 0;
        while i < chars.len() {
            let (at, c) = chars[i];
            let list_number = c == '.' && {
                let before = line[start..at].trim();
                !before.is_empty() && before.bytes().all(|b| b.is_ascii_digit())
            };
            if matches!(c, '.' | '!' | '?') && !list_number {
                let mut j = i + 1;
                while j < chars.len() && matches!(chars[j].1, ')' | ']' | '"' | '\'' | '*' | '`') {
                    j += 1;
                }
                let spaced = j < chars.len() && chars[j].1.is_whitespace();
                let mut k = j;
                while k < chars.len() && chars[k].1.is_whitespace() {
                    k += 1;
                }
                let next_lower = chars.get(k).is_some_and(|(_, n)| n.is_lowercase());
                if spaced && k < chars.len() && !next_lower {
                    let end = chars.get(j).map_or(line.len(), |(b, _)| *b);
                    push_trimmed(&mut out, text, base + start, base + end);
                    start = chars[k].0;
                    i = k;
                    continue;
                }
            }
            i += 1;
        }
        push_trimmed(&mut out, text, base + start, base + line.len());
    }
    out
}

fn push_trimmed(out: &mut Vec<(usize, usize)>, text: &str, a: usize, b: usize) {
    let s = &text[a..b];
    let lead = s.len() - s.trim_start().len();
    let trail = s.len() - s.trim_end().len();
    if lead + trail < s.len() {
        out.push((a + lead, b - trail));
    }
}
