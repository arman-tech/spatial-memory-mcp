//! Tokenizers behind one small trait, so the chunker and the input builder
//! run on a whitespace tokenizer in unit tests and on the model's own
//! tokenizer in the daemon (SPEC 5, chunking rule 1).

use std::path::Path;

use sm_core::error::{CoreError, ErrorCode, Result};
use sm_core::hash::fnv1a64;

/// Model input ids for one text, with the special tokens the model expects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Encoding {
    pub ids: Vec<i64>,
    /// `true` where `ids[i]` is a special token (begin, end).
    pub special: Vec<bool>,
}

impl Encoding {
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Cuts the encoding to at most `cap` ids by dropping content tokens
    /// before the trailing special tokens, so the end token stays. Returns
    /// whether anything was dropped.
    pub fn truncate_content(&mut self, cap: usize) -> bool {
        if self.ids.len() <= cap {
            return false;
        }
        let trailing = self.special.iter().rev().take_while(|s| **s).count();
        let trailing = trailing.min(cap);
        let keep_head = cap - trailing;
        let tail_start = self.ids.len() - trailing;
        let mut ids = self.ids[..keep_head].to_vec();
        ids.extend_from_slice(&self.ids[tail_start..]);
        let mut special = self.special[..keep_head].to_vec();
        special.extend_from_slice(&self.special[tail_start..]);
        self.ids = ids;
        self.special = special;
        true
    }
}

/// What the chunker and the input builder need from a tokenizer.
pub trait TextTokenizer: Send + Sync + std::fmt::Debug {
    /// Byte ranges in `text` of its tokens, without special tokens, in
    /// order. Ranges fall on character boundaries.
    fn spans(&self, text: &str) -> Result<Vec<(usize, usize)>>;

    /// Tokens in `text`, without special tokens.
    fn count(&self, text: &str) -> Result<usize> {
        Ok(self.spans(text)?.len())
    }

    /// Model input ids for `text`, with special tokens.
    fn encode(&self, text: &str) -> Result<Encoding>;
}

/// The model's tokenizer (`tokenizer.json`), with truncation and padding
/// turned off: the input builder cuts and pads itself.
#[derive(Debug, Clone)]
pub struct ModelTokenizer {
    inner: tokenizers::Tokenizer,
}

impl ModelTokenizer {
    pub fn from_file(path: &Path) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path).map_err(|e| {
            CoreError::new(
                ErrorCode::ModelIntegrity,
                format!("cannot load the tokenizer {}: {e}", path.display()),
            )
        })?;
        Self::from_tokenizer(inner)
    }

    pub fn from_tokenizer(mut inner: tokenizers::Tokenizer) -> Result<Self> {
        inner
            .with_truncation(None)
            .map_err(|e| CoreError::internal(format!("cannot turn off truncation: {e}")))?;
        inner.with_padding(None);
        Ok(ModelTokenizer { inner })
    }

    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    pub fn inner(&self) -> &tokenizers::Tokenizer {
        &self.inner
    }
}

fn tokenizer_error(e: impl std::fmt::Display) -> CoreError {
    CoreError::internal(format!("tokenizer failed: {e}"))
}

impl TextTokenizer for ModelTokenizer {
    fn spans(&self, text: &str) -> Result<Vec<(usize, usize)>> {
        let enc = self.inner.encode(text, false).map_err(tokenizer_error)?;
        Ok(enc
            .get_offsets()
            .iter()
            .map(|&(s, e)| (floor_boundary(text, s), ceil_boundary(text, e)))
            .collect())
    }

    fn count(&self, text: &str) -> Result<usize> {
        Ok(self
            .inner
            .encode(text, false)
            .map_err(tokenizer_error)?
            .len())
    }

    fn encode(&self, text: &str) -> Result<Encoding> {
        let enc = self.inner.encode(text, true).map_err(tokenizer_error)?;
        Ok(Encoding {
            ids: enc.get_ids().iter().map(|&i| i64::from(i)).collect(),
            special: enc
                .get_special_tokens_mask()
                .iter()
                .map(|&m| m != 0)
                .collect(),
        })
    }
}

/// A token is a run of non-whitespace. Begin id 2, end id 1, pad 0; a
/// word's id is fixed by its hash. For unit tests.
#[derive(Debug, Default, Clone, Copy)]
pub struct WhitespaceTokenizer;

/// Begin and end ids of [`WhitespaceTokenizer`].
pub const WS_BOS: i64 = 2;
pub const WS_EOS: i64 = 1;

impl TextTokenizer for WhitespaceTokenizer {
    fn spans(&self, text: &str) -> Result<Vec<(usize, usize)>> {
        let mut out = Vec::new();
        let mut start = None;
        for (i, c) in text.char_indices() {
            match (c.is_whitespace(), start) {
                (true, Some(s)) => {
                    out.push((s, i));
                    start = None;
                }
                (false, None) => start = Some(i),
                _ => {}
            }
        }
        if let Some(s) = start {
            out.push((s, text.len()));
        }
        Ok(out)
    }

    fn encode(&self, text: &str) -> Result<Encoding> {
        let mut ids = vec![WS_BOS];
        for (s, e) in self.spans(text)? {
            ids.push(10 + (fnv1a64(&text.as_bytes()[s..e]) % 10_000) as i64);
        }
        ids.push(WS_EOS);
        let mut special = vec![false; ids.len()];
        special[0] = true;
        if let Some(last) = special.last_mut() {
            *last = true;
        }
        Ok(Encoding { ids, special })
    }
}

/// The largest character boundary at or below `i`.
pub fn floor_boundary(s: &str, i: usize) -> usize {
    let mut i = i.min(s.len());
    while !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// The smallest character boundary at or above `i`.
pub fn ceil_boundary(s: &str, i: usize) -> usize {
    let mut i = i.min(s.len());
    while !s.is_char_boundary(i) {
        i += 1;
    }
    i
}

/// `text` cut after its first `max_tokens` tokens (no special tokens).
/// Text with at most `max_tokens` tokens comes back whole.
pub fn cut_to_tokens<'t>(
    tok: &dyn TextTokenizer,
    text: &'t str,
    max_tokens: usize,
) -> Result<&'t str> {
    let spans = tok.spans(text)?;
    if spans.len() <= max_tokens {
        return Ok(text);
    }
    if max_tokens == 0 {
        return Ok("");
    }
    let end = spans[..max_tokens].iter().map(|s| s.1).max().unwrap_or(0);
    Ok(&text[..ceil_boundary(text, end)])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s5_whitespace_tokenizer_spans_and_encoding() {
        let t = WhitespaceTokenizer;
        let text = "  alpha béta\tgamma\n";
        let spans = t.spans(text).unwrap();
        let words: Vec<&str> = spans.iter().map(|&(s, e)| &text[s..e]).collect();
        assert_eq!(words, ["alpha", "béta", "gamma"]);
        assert_eq!(t.count(text).unwrap(), 3);
        let enc = t.encode(text).unwrap();
        assert_eq!(enc.len(), 5);
        assert_eq!(enc.ids[0], WS_BOS);
        assert_eq!(enc.ids[4], WS_EOS);
        assert_eq!(enc.special, [true, false, false, false, true]);
        assert!(t.spans("").unwrap().is_empty());
    }

    #[test]
    fn s5_truncate_content_keeps_the_end_token() {
        let mut enc = WhitespaceTokenizer.encode("a b c d e").unwrap();
        assert!(!enc.clone().truncate_content(7));
        assert!(enc.truncate_content(4));
        assert_eq!(enc.len(), 4);
        assert_eq!(enc.ids[0], WS_BOS);
        assert_eq!(*enc.ids.last().unwrap(), WS_EOS);
        assert_eq!(enc.special, [true, false, false, true]);
    }

    #[test]
    fn s6_cut_to_tokens_keeps_a_prefix() {
        let t = WhitespaceTokenizer;
        assert_eq!(
            cut_to_tokens(&t, "one two three four", 2).unwrap(),
            "one two"
        );
        assert_eq!(cut_to_tokens(&t, "one two", 2).unwrap(), "one two");
        assert_eq!(cut_to_tokens(&t, " one two ", 5).unwrap(), " one two ");
        assert_eq!(cut_to_tokens(&t, "one two", 0).unwrap(), "");
        assert_eq!(cut_to_tokens(&t, "é è ê", 2).unwrap(), "é è");
    }

    #[test]
    fn s5_boundaries_respect_characters() {
        let s = "aé";
        assert_eq!(floor_boundary(s, 2), 1);
        assert_eq!(ceil_boundary(s, 2), 3);
        assert_eq!(ceil_boundary(s, 9), 3);
    }
}
