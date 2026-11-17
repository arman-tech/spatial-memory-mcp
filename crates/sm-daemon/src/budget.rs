//! Output budgets (SPEC 8, Output): every reply fits the request's `limit`,
//! or `output.token_budget` when it gives none. Tokens are estimated as for
//! the hook block (`ambient::estimate_tokens`) over each item's JSON. Items
//! are kept in order until the next one does not fit; a first item that
//! does not fit alone is kept with its content cut. `omitted` counts the
//! items left out, and `truncated` says whether anything was cut or left
//! out.

use serde::Serialize;
use sm_core::ambient::estimate_tokens;
use sm_core::protocol::{LinkedMemory, MemoryView};

/// Room kept for the reply's own fields.
pub(crate) const REPLY_OVERHEAD_TOKENS: usize = 40;

pub(crate) fn tokens_of<T: Serialize>(item: &T) -> usize {
    serde_json::to_string(item).map_or(0, |s| estimate_tokens(&s))
}

pub(crate) struct Fitted<T> {
    pub items: Vec<T>,
    pub truncated: bool,
    pub omitted: u32,
}

/// Content that can be cut to fit.
pub(crate) trait Cut {
    /// Cuts the item's content so the item costs about `tokens`.
    fn cut_to(&mut self, tokens: usize);
}

fn cut_view(v: &mut MemoryView, tokens: usize) {
    let full = std::mem::take(&mut v.content);
    let other = tokens_of(v);
    let chars = tokens.saturating_sub(other).saturating_mul(4).max(16);
    let kept: String = full.chars().take(chars).collect();
    v.content_truncated = kept.len() < full.len();
    v.content = kept;
}

impl Cut for MemoryView {
    fn cut_to(&mut self, tokens: usize) {
        cut_view(self, tokens);
    }
}

impl Cut for LinkedMemory {
    fn cut_to(&mut self, tokens: usize) {
        let via = tokens_of(&self.via);
        cut_view(&mut self.memory, tokens.saturating_sub(via));
    }
}

/// Keeps items in order within `budget` tokens, of which `used` are taken.
pub(crate) fn fit<T: Serialize + Cut>(items: Vec<T>, budget: usize, used: usize) -> Fitted<T> {
    let mut used = used + REPLY_OVERHEAD_TOKENS;
    let total = items.len();
    let mut kept = Vec::new();
    let mut cut = false;
    for mut item in items {
        let cost = tokens_of(&item) + 1;
        if used + cost <= budget {
            used += cost;
            kept.push(item);
            continue;
        }
        if kept.is_empty() {
            item.cut_to(budget.saturating_sub(used));
            kept.push(item);
            cut = true;
        }
        break;
    }
    let omitted = (total - kept.len()) as u32;
    Fitted {
        truncated: cut || omitted > 0,
        omitted,
        items: kept,
    }
}

/// Keeps whole items that cannot be cut (links, issues) within the budget.
pub(crate) fn fit_whole<T: Serialize>(items: Vec<T>, budget: usize, used: usize) -> Fitted<T> {
    let mut used = used + REPLY_OVERHEAD_TOKENS;
    let total = items.len();
    let mut kept = Vec::new();
    for item in items {
        let cost = tokens_of(&item) + 1;
        if used + cost > budget && !kept.is_empty() {
            break;
        }
        used += cost;
        kept.push(item);
    }
    let omitted = (total - kept.len()) as u32;
    Fitted {
        truncated: omitted > 0,
        omitted,
        items: kept,
    }
}
