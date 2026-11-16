//! Model inputs: the prompts, the query and title cuts, the per-input cap,
//! batching and right padding (SPEC 5 table and chunking rules 1, 5 and 6).
//!
//! Everything here is plain data in and out, so the rules are unit-tested
//! with the whitespace tokenizer.

use sm_core::EMBED_DIM;
use sm_core::error::{CoreError, Result};
use sm_core::traits::{DocInput, MAX_TITLE_TOKENS};

use crate::tokenize::{TextTokenizer, cut_to_tokens};

/// The query prompt before the query text.
pub const QUERY_PREFIX: &str = "task: search result | query: ";
/// The title written when a memory has none.
pub const NO_TITLE: &str = "none";
/// Padding id (SPEC 5: pad on the right with token 0).
pub const PAD_ID: i64 = 0;
/// How far an output vector's length may be from 1.
pub const NORM_TOLERANCE: f32 = 1e-3;

/// `task: search result | query: {text}`.
pub fn query_prompt(text: &str) -> String {
    format!("{QUERY_PREFIX}{text}")
}

/// `title: {title} | text: {chunk}`.
pub fn document_prompt(title: &str, chunk: &str) -> String {
    format!("title: {title} | text: {chunk}")
}

/// The title as written in the prompt: `none` when absent or empty.
pub fn title_or_none(title: Option<&str>) -> &str {
    match title {
        Some(t) if !t.is_empty() => t,
        _ => NO_TITLE,
    }
}

/// One model input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Prepared {
    pub ids: Vec<i64>,
    /// Content tokens were dropped to respect the cap. The cut text is
    /// already within the limit, so this only happens if the tokenizer
    /// splits a word differently inside the prompt; it is counted and
    /// reported, never silent.
    pub truncated: bool,
}

/// Builds inputs that never exceed the chunk limit plus the prompt and the
/// title (SPEC 5, rule 6).
#[derive(Debug, Clone, Copy)]
pub struct InputBuilder<'a> {
    tok: &'a dyn TextTokenizer,
    chunk_limit: usize,
}

impl<'a> InputBuilder<'a> {
    pub fn new(tok: &'a dyn TextTokenizer, chunk_limit: usize) -> Self {
        InputBuilder {
            tok,
            chunk_limit: chunk_limit.max(1),
        }
    }

    pub fn chunk_limit(&self) -> usize {
        self.chunk_limit
    }

    /// The query cut to `max_tokens` tokens of its own text (and never more
    /// than the chunk limit), then put in the query prompt.
    pub fn query(&self, text: &str, max_tokens: usize) -> Result<Prepared> {
        let max = max_tokens.min(self.chunk_limit);
        let cut = cut_to_tokens(self.tok, text, max)?;
        let mut enc = self.tok.encode(&query_prompt(cut))?;
        let cap = self.tok.encode(QUERY_PREFIX)?.len() + max;
        let truncated = enc.truncate_content(cap);
        Ok(Prepared {
            ids: enc.ids,
            truncated,
        })
    }

    /// The document prompt for one chunk, its title cut to
    /// [`MAX_TITLE_TOKENS`]. A chunk over the limit is refused: the chunker
    /// produces chunks within it, and embedding a longer one would break
    /// rule 6.
    pub fn document(&self, doc: &DocInput<'_>) -> Result<Prepared> {
        let n = self.tok.count(doc.text)?;
        if n > self.chunk_limit {
            return Err(CoreError::invalid_input(format!(
                "a chunk of {n} tokens is over the chunk limit of {} tokens (SPEC 5, rule 6)",
                self.chunk_limit
            ))
            .with_next_action("Split the text with the chunker before embedding it."));
        }
        let title = cut_to_tokens(self.tok, title_or_none(doc.title), MAX_TITLE_TOKENS)?;
        let mut enc = self.tok.encode(&document_prompt(title, doc.text))?;
        let cap = self.tok.encode(&document_prompt(title, ""))?.len() + self.chunk_limit;
        let truncated = enc.truncate_content(cap);
        Ok(Prepared {
            ids: enc.ids,
            truncated,
        })
    }
}

/// Input indices grouped into batches of at most `batch_size`, shortest
/// first, so a batch carries little padding. Results are put back in input
/// order by the caller.
pub fn plan_batches(lengths: &[usize], batch_size: usize) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..lengths.len()).collect();
    order.sort_by_key(|&i| (lengths[i], i));
    order
        .chunks(batch_size.max(1))
        .map(<[usize]>::to_vec)
        .collect()
}

/// A batch padded on the right with [`PAD_ID`], row-major.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PaddedBatch {
    pub rows: usize,
    pub cols: usize,
    pub ids: Vec<i64>,
    pub mask: Vec<i64>,
}

pub fn pad_batch(inputs: &[&[i64]]) -> PaddedBatch {
    let rows = inputs.len();
    let cols = inputs.iter().map(|r| r.len()).max().unwrap_or(0);
    let mut ids = vec![PAD_ID; rows * cols];
    let mut mask = vec![0i64; rows * cols];
    for (r, row) in inputs.iter().enumerate() {
        ids[r * cols..r * cols + row.len()].copy_from_slice(row);
        mask[r * cols..r * cols + row.len()].fill(1);
    }
    PaddedBatch {
        rows,
        cols,
        ids,
        mask,
    }
}

/// Splits the model output into `rows` vectors of [`EMBED_DIM`] and checks
/// that each is unit length; the model normalises its output, so a vector
/// that is not is reported as an `internal` error rather than stored.
pub fn split_output(shape: &[i64], data: &[f32], rows: usize) -> Result<Vec<Vec<f32>>> {
    let expected = [rows as i64, EMBED_DIM as i64];
    if shape != expected || data.len() != rows * EMBED_DIM {
        return Err(CoreError::internal(format!(
            "the model returned sentence_embedding of shape {shape:?} ({} values); expected {expected:?}",
            data.len()
        )));
    }
    data.chunks(EMBED_DIM)
        .enumerate()
        .map(|(i, v)| {
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if !norm.is_finite() || (norm - 1.0).abs() > NORM_TOLERANCE {
                return Err(CoreError::internal(format!(
                    "the model returned a vector of length {norm} for input {i}; sentence_embedding should be unit length"
                )));
            }
            Ok(v.to_vec())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenize::{WS_BOS, WS_EOS, WhitespaceTokenizer};
    use sm_core::error::ErrorCode;

    fn words(n: usize) -> String {
        (0..n)
            .map(|i| format!("w{i}"))
            .collect::<Vec<_>>()
            .join(" ")
    }

    #[test]
    fn s5_prompts_match_the_spec_table() {
        assert_eq!(
            query_prompt("Which planet is known as the Red Planet?"),
            "task: search result | query: Which planet is known as the Red Planet?"
        );
        assert_eq!(
            document_prompt(title_or_none(None), "Mars is red."),
            "title: none | text: Mars is red."
        );
        assert_eq!(title_or_none(Some("")), "none");
        assert_eq!(title_or_none(Some("Orders")), "Orders");
    }

    #[test]
    fn s6_query_is_cut_before_the_prompt_is_added() {
        let tok = WhitespaceTokenizer;
        let b = InputBuilder::new(&tok, 256);
        // Prefix "task: search result | query:" is 5 whitespace tokens.
        let p = b.query(&words(200), 128).unwrap();
        assert_eq!(p.ids.len(), 1 + 5 + 128 + 1);
        assert_eq!(p.ids[0], WS_BOS);
        assert_eq!(*p.ids.last().unwrap(), WS_EOS);
        assert!(!p.truncated);
        let short = b.query("red planet", 128).unwrap();
        assert_eq!(short.ids.len(), 1 + 5 + 2 + 1);
        // Never beyond the chunk limit, whatever the caller asks.
        let b = InputBuilder::new(&tok, 16);
        assert_eq!(b.query(&words(200), 128).unwrap().ids.len(), 1 + 5 + 16 + 1);
    }

    #[test]
    fn s5_rule1_title_is_cut_to_32_tokens() {
        let tok = WhitespaceTokenizer;
        let b = InputBuilder::new(&tok, 256);
        let title = words(40);
        let doc = DocInput {
            title: Some(&title),
            text: "alpha beta gamma",
        };
        let p = b.document(&doc).unwrap();
        // BOS, "title:", 32 title words, "|", "text:", 3 words, EOS.
        assert_eq!(p.ids.len(), 1 + 1 + MAX_TITLE_TOKENS + 1 + 1 + 3 + 1);
        let none = b
            .document(&DocInput {
                title: None,
                text: "alpha",
            })
            .unwrap();
        let named_none = b
            .document(&DocInput {
                title: Some("none"),
                text: "alpha",
            })
            .unwrap();
        assert_eq!(none, named_none);
    }

    #[test]
    fn s5_rule6_no_input_exceeds_limit_plus_prompt_and_title() {
        let tok = WhitespaceTokenizer;
        let limit = 8;
        let b = InputBuilder::new(&tok, limit);
        let title = words(50);
        let at_limit = words(limit);
        let p = b
            .document(&DocInput {
                title: Some(&title),
                text: &at_limit,
            })
            .unwrap();
        let overhead = tok
            .encode(&document_prompt(&words(MAX_TITLE_TOKENS), ""))
            .unwrap()
            .len();
        assert!(p.ids.len() <= limit + overhead);
        let over = words(limit + 1);
        let err = b
            .document(&DocInput {
                title: None,
                text: &over,
            })
            .unwrap_err();
        assert_eq!(err.code, ErrorCode::InvalidInput);
        assert!(err.message.contains("rule 6"), "{}", err.message);
    }

    #[test]
    fn s5_batches_follow_batch_size_shortest_first() {
        let plan = plan_batches(&[5, 1, 3, 2, 4], 2);
        assert_eq!(plan, vec![vec![1, 3], vec![2, 4], vec![0]]);
        assert_eq!(plan_batches(&[1, 1, 1], 4), vec![vec![0, 1, 2]]);
        assert_eq!(plan_batches(&[3, 2], 0), vec![vec![1], vec![0]]);
        assert!(plan_batches(&[], 4).is_empty());
        for size in 1..=16 {
            let lengths: Vec<usize> = (0..37).map(|i| (i * 7919) % 101).collect();
            let plan = plan_batches(&lengths, size);
            assert!(plan.iter().all(|b| !b.is_empty() && b.len() <= size));
            let mut all: Vec<usize> = plan.concat();
            all.sort_unstable();
            assert_eq!(all, (0..37).collect::<Vec<_>>());
        }
    }

    #[test]
    fn s5_padding_is_on_the_right_with_zero() {
        let a = [2i64, 7, 8, 1];
        let b = [2i64, 9, 1];
        let batch = pad_batch(&[&a, &b]);
        assert_eq!((batch.rows, batch.cols), (2, 4));
        assert_eq!(batch.ids, [2, 7, 8, 1, 2, 9, 1, 0]);
        assert_eq!(batch.mask, [1, 1, 1, 1, 1, 1, 1, 0]);
    }

    #[test]
    fn s5_output_must_be_unit_length_and_shaped() {
        let mut data = vec![0f32; 2 * EMBED_DIM];
        data[0] = 1.0;
        data[EMBED_DIM + 5] = 1.0;
        let shape = [2i64, EMBED_DIM as i64];
        let rows = split_output(&shape, &data, 2).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[1][5], 1.0);
        data[EMBED_DIM + 5] = 0.5;
        let err = split_output(&shape, &data, 2).unwrap_err();
        assert_eq!(err.code, ErrorCode::Internal);
        assert!(err.message.contains("unit length"));
        let err = split_output(&[1, EMBED_DIM as i64], &data[..EMBED_DIM], 2).unwrap_err();
        assert_eq!(err.code, ErrorCode::Internal);
        data[0] = f32::NAN;
        assert!(split_output(&shape, &data, 2).is_err());
    }
}
