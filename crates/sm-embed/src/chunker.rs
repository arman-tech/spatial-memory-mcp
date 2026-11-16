//! The structure-aware chunker (SPEC 5, chunking rules 1 to 4).
//!
//! Content at or under the limit is one chunk. Longer content is split on
//! ATX headings, then blank-line paragraphs, then sentences; fenced code
//! blocks stay whole. Adjacent pieces are packed up to the limit, never
//! across a heading. A piece still over the limit becomes token windows with
//! `overlap` overlap. Every chunk is a contiguous, trimmed part of the
//! content, and its token count is measured on its own text, so no chunk
//! is ever over the limit.

use std::sync::Arc;

use sm_core::error::{CoreError, Result};
use sm_core::traits::{ChunkParams, ChunkPiece, ChunkPlan, Chunker};

use crate::tokenize::{TextTokenizer, WhitespaceTokenizer};

/// [`Chunker`] over any [`TextTokenizer`]: the model's tokenizer in the
/// daemon, [`WhitespaceTokenizer`] in unit tests.
#[derive(Debug, Clone)]
pub struct StructuredChunker {
    tok: Arc<dyn TextTokenizer>,
}

impl StructuredChunker {
    pub fn new(tok: Arc<dyn TextTokenizer>) -> Self {
        StructuredChunker { tok }
    }

    /// A chunker that counts whitespace-separated words as tokens.
    pub fn whitespace() -> Self {
        Self::new(Arc::new(WhitespaceTokenizer))
    }

    pub fn tokenizer(&self) -> &Arc<dyn TextTokenizer> {
        &self.tok
    }
}

impl Chunker for StructuredChunker {
    fn count_tokens(&self, text: &str) -> Result<usize> {
        self.tok.count(text)
    }

    fn chunk(&self, content: &str, params: &ChunkParams) -> Result<ChunkPlan> {
        let max = params.max_tokens;
        if max == 0 {
            return Err(
                CoreError::invalid_input("chunk.max_tokens must be at least 1")
                    .with_next_action("Set chunk.max_tokens in the settings file (default 256)."),
            );
        }
        if !params.overlap.is_finite() || !(0.0..1.0).contains(&params.overlap) {
            return Err(CoreError::invalid_input(format!(
                "chunk.overlap must be at least 0 and below 1, not {}",
                params.overlap
            ))
            .with_next_action("Set chunk.overlap in the settings file (default 0.125)."));
        }
        // Overlap in tokens; a window always moves forward by at least one.
        let overlap = ((max as f64 * params.overlap).floor() as usize).min(max - 1);

        let spans = self.tok.spans(content)?;
        let total = spans.len();
        let cap = params.index_cap.map_or(total, |c| c.min(total));
        let capped = cap < total;
        if cap == 0 {
            return Ok(ChunkPlan {
                pieces: Vec::new(),
                total_tokens: total,
                indexed_tokens: 0,
                capped,
            });
        }
        let region_end = if capped {
            spans[..cap].iter().map(|s| s.1).max().unwrap_or(0)
        } else {
            content.len()
        };
        let region = &content[..region_end];

        let run = Run {
            tok: self.tok.as_ref(),
            text: region,
            max,
            overlap,
        };
        let mut pieces = Vec::new();
        if cap <= max {
            run.emit(0, region.len(), &mut pieces)?;
        } else {
            run.structured(&mut pieces)?;
        }
        let pieces = pieces
            .into_iter()
            .enumerate()
            .map(|(i, (text, n))| {
                Ok(ChunkPiece {
                    seq: u32::try_from(i)
                        .map_err(|_| CoreError::internal("more than u32::MAX chunks"))?,
                    text,
                    token_count: u32::try_from(n)
                        .map_err(|_| CoreError::internal("chunk token count over u32::MAX"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(ChunkPlan {
            pieces,
            total_tokens: total,
            indexed_tokens: cap,
            capped,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Heading,
    Text,
    Fence,
}

/// A heading line, a blank-line paragraph or a fenced code block, as a byte
/// range of the region. `section` counts the headings before it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Block {
    start: usize,
    end: usize,
    kind: Kind,
    section: usize,
}

/// What packing works on: a piece within the limit, or a window cut from an
/// oversized piece (windows are never packed with their neighbours).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Unit {
    Piece {
        start: usize,
        end: usize,
        section: usize,
    },
    Window {
        start: usize,
        end: usize,
    },
}

struct Run<'a> {
    tok: &'a dyn TextTokenizer,
    text: &'a str,
    max: usize,
    overlap: usize,
}

impl Run<'_> {
    fn count(&self, start: usize, end: usize) -> Result<usize> {
        self.tok.count(&self.text[start..end])
    }

    fn structured(&self, out: &mut Vec<(String, usize)>) -> Result<()> {
        let mut units = Vec::new();
        for b in blocks(self.text) {
            if self.count(b.start, b.end)? <= self.max {
                units.push(Unit::Piece {
                    start: b.start,
                    end: b.end,
                    section: b.section,
                });
                continue;
            }
            match b.kind {
                Kind::Text => {
                    for (s, e) in sentences(self.text, b.start, b.end) {
                        if self.count(s, e)? <= self.max {
                            units.push(Unit::Piece {
                                start: s,
                                end: e,
                                section: b.section,
                            });
                        } else {
                            self.windows(s, e, &mut units)?;
                        }
                    }
                }
                Kind::Heading | Kind::Fence => self.windows(b.start, b.end, &mut units)?,
            }
        }

        // Pack adjacent pieces of one section while the joined text stays
        // within the limit.
        let mut group: Option<(usize, usize, usize)> = None;
        for unit in units {
            match unit {
                Unit::Window { start, end } => {
                    if let Some((gs, ge, _)) = group.take() {
                        self.emit(gs, ge, out)?;
                    }
                    self.emit(start, end, out)?;
                }
                Unit::Piece {
                    start,
                    end,
                    section,
                } => {
                    if let Some((gs, ge, gsec)) = group {
                        if gsec == section && self.count(gs, end)? <= self.max {
                            group = Some((gs, end, gsec));
                            continue;
                        }
                        self.emit(gs, ge, out)?;
                    }
                    group = Some((start, end, section));
                }
            }
        }
        if let Some((gs, ge, _)) = group {
            self.emit(gs, ge, out)?;
        }
        Ok(())
    }

    /// Token windows over `start..end`, `overlap` tokens shared between
    /// neighbours. A window whose own text counts over the limit (a
    /// tokenizer can merge differently at the edges) is shortened.
    fn windows(&self, start: usize, end: usize, units: &mut Vec<Unit>) -> Result<()> {
        let sub = &self.text[start..end];
        let spans = self.tok.spans(sub)?;
        let n = spans.len();
        let mut i = 0;
        while i < n {
            let mut j = (i + self.max).min(n);
            loop {
                let s = spans[i].0;
                let e = spans[i..j].iter().map(|t| t.1).max().unwrap_or(s);
                let count = self.tok.count(&sub[s..e])?;
                if count <= self.max {
                    units.push(Unit::Window {
                        start: start + s,
                        end: start + e,
                    });
                    break;
                }
                if j == i + 1 {
                    return Err(CoreError::internal(format!(
                        "one token of text counts as {count} tokens, over the chunk limit of {}",
                        self.max
                    )));
                }
                j = j.saturating_sub((count - self.max).max(1)).max(i + 1);
            }
            if j == n {
                break;
            }
            i = j.saturating_sub(self.overlap).max(i + 1);
        }
        Ok(())
    }

    /// Adds the trimmed text of `start..end` as a chunk, measured on its own
    /// text; text that measures over the limit is cut into windows instead.
    fn emit(&self, start: usize, end: usize, out: &mut Vec<(String, usize)>) -> Result<()> {
        let raw = &self.text[start..end];
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Ok(());
        }
        let s = start + (raw.len() - raw.trim_start().len());
        let e = s + trimmed.len();
        let n = self.tok.count(trimmed)?;
        if n == 0 {
            return Ok(());
        }
        if n <= self.max {
            out.push((trimmed.to_owned(), n));
            return Ok(());
        }
        let mut windows = Vec::new();
        self.windows(s, e, &mut windows)?;
        for w in windows {
            if let Unit::Window { start, end } = w {
                let text = self.text[start..end].trim();
                let n = self.tok.count(text)?;
                if n > 0 {
                    out.push((text.to_owned(), n));
                }
            }
        }
        Ok(())
    }
}

/// One line: `start..end`, without its line break.
struct Line {
    start: usize,
    end: usize,
}

fn lines(text: &str) -> Vec<Line> {
    let mut out = Vec::new();
    let mut start = 0;
    while start < text.len() {
        let (end, next) = match text[start..].find('\n') {
            Some(i) => (start + i, start + i + 1),
            None => (text.len(), text.len()),
        };
        let end = if end > start && text.as_bytes()[end - 1] == b'\r' {
            end - 1
        } else {
            end
        };
        out.push(Line { start, end });
        start = next;
    }
    out
}

/// The line after up to three spaces of indent; `None` for an indented
/// code line (four spaces or a tab), which is never a heading or a fence.
fn strip_indent(line: &str) -> Option<&str> {
    let n = line.bytes().take_while(|&b| b == b' ').count();
    if n > 3 {
        return None;
    }
    let rest = &line[n..];
    if rest.starts_with('\t') {
        return None;
    }
    Some(rest)
}

/// An ATX heading: one to six `#`, then a space, a tab or the line end.
fn is_heading(line: &str) -> bool {
    let Some(rest) = strip_indent(line) else {
        return false;
    };
    let n = rest.bytes().take_while(|&b| b == b'#').count();
    (1..=6).contains(&n)
        && rest[n..]
            .chars()
            .next()
            .is_none_or(|c| c == ' ' || c == '\t')
}

/// A fence opener: three or more backticks or tildes.
fn fence_open(line: &str) -> Option<(u8, usize)> {
    let rest = strip_indent(line)?;
    let c = *rest.as_bytes().first()?;
    if c != b'`' && c != b'~' {
        return None;
    }
    let n = rest.bytes().take_while(|&b| b == c).count();
    if n < 3 || (c == b'`' && rest[n..].contains('`')) {
        return None;
    }
    Some((c, n))
}

/// A fence closer: the opener's character, at least as many times, then
/// only whitespace.
fn fence_close(line: &str, c: u8, n: usize) -> bool {
    let Some(rest) = strip_indent(line) else {
        return false;
    };
    let m = rest.bytes().take_while(|&b| b == c).count();
    m >= n && rest[m..].trim().is_empty()
}

/// Headings, paragraphs and fenced code blocks of `text`, in order. A fence
/// left open runs to the end of the text.
fn blocks(text: &str) -> Vec<Block> {
    let mut out = Vec::new();
    let mut section = 0;
    let mut para: Option<(usize, usize)> = None;
    let mut fence: Option<(usize, u8, usize)> = None;
    for line in lines(text) {
        let s = &text[line.start..line.end];
        if let Some((start, c, n)) = fence {
            if fence_close(s, c, n) {
                out.push(Block {
                    start,
                    end: line.end,
                    kind: Kind::Fence,
                    section,
                });
                fence = None;
            }
            continue;
        }
        let blank = s.trim().is_empty();
        let heading = !blank && is_heading(s);
        let opener = if blank || heading {
            None
        } else {
            fence_open(s)
        };
        if (blank || heading || opener.is_some())
            && let Some((start, end)) = para.take()
        {
            out.push(Block {
                start,
                end,
                kind: Kind::Text,
                section,
            });
        }
        if heading {
            section += 1;
            out.push(Block {
                start: line.start,
                end: line.end,
                kind: Kind::Heading,
                section,
            });
        } else if let Some((c, n)) = opener {
            fence = Some((line.start, c, n));
        } else if !blank {
            para = Some((para.map_or(line.start, |p| p.0), line.end));
        }
    }
    if let Some((start, ..)) = fence {
        out.push(Block {
            start,
            end: text.len(),
            kind: Kind::Fence,
            section,
        });
    }
    if let Some((start, end)) = para {
        out.push(Block {
            start,
            end,
            kind: Kind::Text,
            section,
        });
    }
    out
}

/// Sentences of the paragraph at `start..end`, as trimmed byte ranges. A
/// sentence ends after `.`, `!` or `?` (and any closing quotes or brackets)
/// followed by whitespace, or at a line break.
fn sentences(text: &str, start: usize, end: usize) -> Vec<(usize, usize)> {
    let s = &text[start..end];
    let mut out = Vec::new();
    let mut push = |a: usize, b: usize| {
        let raw = &s[a..b];
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            let from = a + (raw.len() - raw.trim_start().len());
            out.push((start + from, start + from + trimmed.len()));
        }
    };
    let mut cur = 0;
    let mut ended = false;
    for (i, c) in s.char_indices() {
        if c == '\n' {
            push(cur, i);
            cur = i + 1;
            ended = false;
        } else if c.is_whitespace() {
            if ended {
                push(cur, i);
                cur = i;
            }
            ended = false;
        } else if matches!(c, '.' | '!' | '?') {
            ended = true;
        } else if !(ended && matches!(c, '"' | '\'' | ')' | ']' | '\u{201d}' | '\u{2019}')) {
            ended = false;
        }
    }
    push(cur, s.len());
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use sm_core::error::ErrorCode;

    fn params(max: usize, overlap: f64, cap: Option<usize>) -> ChunkParams {
        ChunkParams {
            max_tokens: max,
            overlap,
            index_cap: cap,
        }
    }

    fn texts(plan: &ChunkPlan) -> Vec<&str> {
        plan.pieces.iter().map(|p| p.text.as_str()).collect()
    }

    fn words(prefix: &str, from: usize, to: usize) -> String {
        (from..to)
            .map(|i| format!("{prefix}{i}"))
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn check_invariants(content: &str, plan: &ChunkPlan, max: usize) {
        let tok = WhitespaceTokenizer;
        for (i, p) in plan.pieces.iter().enumerate() {
            assert_eq!(p.seq as usize, i);
            let n = tok.count(&p.text).unwrap();
            assert_eq!(p.token_count as usize, n, "token_count is measured");
            assert!(
                (1..=max).contains(&n),
                "piece {i} has {n} tokens, limit {max}"
            );
            assert_eq!(p.text, p.text.trim());
            assert!(
                content.contains(&p.text),
                "piece {i} is part of the content"
            );
        }
    }

    #[test]
    fn s5_rule1_counts_with_the_given_tokenizer() {
        let c = StructuredChunker::whitespace();
        assert_eq!(c.count_tokens("Mars is  red.\n\n# Next").unwrap(), 5);
        assert_eq!(c.count_tokens("").unwrap(), 0);
    }

    #[test]
    fn s5_rule2_content_at_or_under_the_limit_is_one_chunk() {
        let c = StructuredChunker::whitespace();
        let content = "\nalpha beta\n\n# Heading\n\n```\ngamma\n```\n";
        let plan = c.chunk(content, &params(7, 0.125, None)).unwrap();
        assert_eq!(texts(&plan), [content.trim()]);
        assert_eq!(plan.pieces[0].token_count, 7);
        assert_eq!((plan.total_tokens, plan.indexed_tokens), (7, 7));
        assert!(!plan.capped);
    }

    #[test]
    fn s5_rule3_never_packs_across_a_heading() {
        let c = StructuredChunker::whitespace();
        for nl in ["\n", "\r\n"] {
            let content = [
                "intro one",
                "",
                "# A",
                "",
                "para a1 a2",
                "",
                "## B",
                "para b1 b2",
            ]
            .join(nl);
            let plan = c.chunk(&content, &params(10, 0.0, None)).unwrap();
            assert_eq!(plan.total_tokens, 12);
            let a = ["# A", "", "para a1 a2"].join(nl);
            let b = ["## B", "para b1 b2"].join(nl);
            assert_eq!(texts(&plan), ["intro one", a.as_str(), b.as_str()]);
            check_invariants(&content, &plan, 10);
        }
        // `#tag` and a seven-hash line are not headings, so they pack.
        for content in [
            "intro\n\n#tag one\n\nx y z",
            "intro\n\n####### one\n\nx y z",
        ] {
            let plan = c.chunk(content, &params(4, 0.0, None)).unwrap();
            let packed = &content[..content.rfind("\n\n").unwrap()];
            assert_eq!(texts(&plan), [packed, "x y z"]);
        }
    }

    #[test]
    fn s5_rule3_packs_adjacent_paragraphs_up_to_the_limit() {
        let c = StructuredChunker::whitespace();
        let paras: Vec<String> = (0..4).map(|p| words(&format!("p{p}w"), 0, 4)).collect();
        let content = paras.join("\n\n");
        let plan = c.chunk(&content, &params(10, 0.125, None)).unwrap();
        assert_eq!(
            texts(&plan),
            [
                format!("{}\n\n{}", paras[0], paras[1]),
                format!("{}\n\n{}", paras[2], paras[3])
            ]
        );
        assert_eq!(plan.indexed_tokens, 16);
        check_invariants(&content, &plan, 10);
    }

    #[test]
    fn s5_rule3_an_oversized_paragraph_splits_into_sentences() {
        let c = StructuredChunker::whitespace();
        let content = "One two three. Four five six! Seven eight nine? Ten eleven.";
        let plan = c.chunk(content, &params(8, 0.125, None)).unwrap();
        assert_eq!(
            texts(&plan),
            [
                "One two three. Four five six!",
                "Seven eight nine? Ten eleven."
            ]
        );
        // Closing quotes stay with their sentence.
        let content = "He said \"stop.\" Then he left. And that was all.";
        let plan = c.chunk(content, &params(6, 0.0, None)).unwrap();
        assert_eq!(
            texts(&plan),
            ["He said \"stop.\" Then he left.", "And that was all."]
        );
        // Line breaks inside a paragraph end a sentence too.
        let content = "l1 a b c\nl2 d e f\nl3 g h i";
        let plan = c.chunk(content, &params(8, 0.125, None)).unwrap();
        assert_eq!(texts(&plan), ["l1 a b c\nl2 d e f", "l3 g h i"]);
    }

    #[test]
    fn s5_rule3_code_fences_stay_whole() {
        let c = StructuredChunker::whitespace();
        let content =
            "before text here\n\n```rust\nfn main() {\n\n# not a heading\n}\n```\n\nafter text";
        let plan = c.chunk(content, &params(12, 0.125, None)).unwrap();
        assert_eq!(plan.total_tokens, 15);
        assert_eq!(
            texts(&plan),
            [
                "before text here",
                "```rust\nfn main() {\n\n# not a heading\n}\n```\n\nafter text"
            ]
        );
        // Tildes, a longer closer, and a fence left open to the end.
        let content = "a b c d e f\n\n~~~\nx\n\ny\n~~~~\n\n```\nopen\n\nfence";
        let plan = c.chunk(content, &params(6, 0.0, None)).unwrap();
        assert_eq!(
            texts(&plan),
            ["a b c d e f", "~~~\nx\n\ny\n~~~~", "```\nopen\n\nfence"]
        );
    }

    #[test]
    fn s5_rule4_an_oversized_piece_becomes_overlapping_windows() {
        let c = StructuredChunker::whitespace();
        // A long log line: no sentence ends, no blank lines.
        let content = words("x", 0, 40);
        let plan = c.chunk(&content, &params(16, 0.125, None)).unwrap();
        assert_eq!(
            texts(&plan),
            [words("x", 0, 16), words("x", 14, 30), words("x", 28, 40)]
        );
        check_invariants(&content, &plan, 16);
        assert_eq!(plan.indexed_tokens, 40, "overlap is not counted twice");
        // A fence over the limit is windowed too, and windows are never
        // packed with their neighbours.
        let fence = format!("```\n{}\n```", words("f", 0, 18));
        let content = format!("head one\n\n{fence}\n\ntail one");
        let plan = c.chunk(&content, &params(8, 0.25, None)).unwrap();
        assert_eq!(plan.pieces.first().unwrap().text, "head one");
        assert_eq!(plan.pieces.last().unwrap().text, "tail one");
        let windows = &plan.pieces[1..plan.pieces.len() - 1];
        assert_eq!(windows.len(), 3);
        assert!(windows[0].text.starts_with("```\nf0"));
        assert!(windows[1].text.starts_with("f5 f6"), "{}", windows[1].text);
        assert!(windows[2].text.ends_with("f17\n```"));
        check_invariants(&content, &plan, 8);
        // No overlap: windows tile the text.
        let plan = c.chunk(&words("y", 0, 10), &params(4, 0.0, None)).unwrap();
        assert_eq!(
            texts(&plan),
            [words("y", 0, 4), words("y", 4, 8), words("y", 8, 10)]
        );
    }

    #[test]
    fn s5_index_cap_stops_chunking_and_reports_tokens() {
        let c = StructuredChunker::whitespace();
        let content = (0..6)
            .map(|p| words("w", p * 4, p * 4 + 4))
            .collect::<Vec<_>>()
            .join("\n\n");
        let plan = c.chunk(&content, &params(8, 0.125, Some(12))).unwrap();
        assert_eq!((plan.total_tokens, plan.indexed_tokens), (24, 12));
        assert!(plan.capped);
        assert_eq!(
            texts(&plan),
            [
                format!("{}\n\n{}", words("w", 0, 4), words("w", 4, 8)),
                words("w", 8, 12)
            ]
        );
        let plan = c.chunk(&content, &params(8, 0.125, Some(10))).unwrap();
        assert_eq!(plan.pieces.last().unwrap().text, "w8 w9");
        let plan = c.chunk(&content, &params(8, 0.125, Some(6))).unwrap();
        assert_eq!(texts(&plan), [format!("{}\n\nw4 w5", words("w", 0, 4))]);
        assert_eq!(plan.indexed_tokens, 6);
        let plan = c.chunk(&content, &params(8, 0.125, Some(24))).unwrap();
        assert!(!plan.capped);
        assert_eq!(plan.indexed_tokens, 24);
        let plan = c.chunk(&content, &params(8, 0.125, Some(0))).unwrap();
        assert!(plan.pieces.is_empty() && plan.capped);
    }

    #[test]
    fn s5_empty_and_whitespace_content_has_no_chunks() {
        let c = StructuredChunker::whitespace();
        for content in ["", "  \n\n\t "] {
            let plan = c.chunk(content, &params(8, 0.125, None)).unwrap();
            assert!(plan.pieces.is_empty());
            assert_eq!((plan.total_tokens, plan.indexed_tokens), (0, 0));
            assert!(!plan.capped);
        }
    }

    #[test]
    fn s5_invalid_parameters_are_invalid_input() {
        let c = StructuredChunker::whitespace();
        for p in [
            params(0, 0.125, None),
            params(8, -0.1, None),
            params(8, 1.0, None),
            params(8, f64::NAN, None),
        ] {
            let err = c.chunk("a b c", &p).unwrap_err();
            assert_eq!(err.code, ErrorCode::InvalidInput);
        }
    }

    /// A deterministic Markdown-like document: headings, paragraphs of
    /// sentences, long lines and fences, every word unique.
    fn generated(seed: u64) -> String {
        let mut state = seed;
        let mut next = |n: u64| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) % n
        };
        let mut word = 0usize;
        let mut take = |n: u64| {
            let w = words("k", word, word + n as usize + 1);
            word += n as usize + 1;
            w
        };
        let mut out = String::new();
        for _ in 0..40 {
            match next(6) {
                0 => out.push_str(&format!(
                    "{} {}\n\n",
                    "#".repeat(1 + next(3) as usize),
                    take(3)
                )),
                1 => out.push_str(&format!("```\n{}\n\n{}\n```\n\n", take(next(30)), take(4))),
                2 => out.push_str(&format!("{}\n\n", take(next(60)))),
                _ => {
                    for _ in 0..1 + next(5) {
                        let s = take(next(12));
                        out.push_str(&s);
                        out.push_str(if next(4) == 0 { "\n" } else { ". " });
                    }
                    out.push_str("\n\n");
                }
            }
        }
        out
    }

    #[test]
    fn s5_every_chunk_is_part_of_the_content_within_the_limit() {
        let c = StructuredChunker::whitespace();
        let tok = WhitespaceTokenizer;
        for seed in 1..=12u64 {
            let content = generated(seed);
            for max in [4, 8, 16, 32, 64] {
                for overlap in [0.0, 0.125, 0.25] {
                    let plan = c.chunk(&content, &params(max, overlap, None)).unwrap();
                    check_invariants(&content, &plan, max);
                    assert_eq!(plan.total_tokens, tok.count(&content).unwrap());
                    assert_eq!(plan.indexed_tokens, plan.total_tokens);
                    // Every word of the content is in some chunk.
                    let covered: std::collections::HashSet<&str> = plan
                        .pieces
                        .iter()
                        .flat_map(|p| p.text.split_whitespace())
                        .collect();
                    for w in content.split_whitespace() {
                        assert!(covered.contains(w), "{w} missing at max {max}");
                    }
                    // A heading only ever starts a chunk.
                    for p in &plan.pieces {
                        for line in p.text.lines().skip(1) {
                            assert!(
                                !is_heading(line) || p.text.starts_with("```"),
                                "heading inside a chunk: {:?}",
                                p.text
                            );
                        }
                    }
                    // Chunking is deterministic.
                    assert_eq!(
                        plan,
                        c.chunk(&content, &params(max, overlap, None)).unwrap()
                    );
                }
            }
        }
    }
}
