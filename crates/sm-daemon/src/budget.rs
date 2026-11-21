//! Output budgets (SPEC 8, Output; e-report decision 24): no reply is
//! longer than the request's `limit`, or `output.token_budget` when it
//! gives none.
//!
//! Tokens are estimated as for the hook block (`ambient::estimate_tokens`)
//! over JSON. A reply's own fields are measured on its skeleton: the reply
//! with every budgeted list empty, `truncated` set and `omitted` at its
//! largest, plus [`SKELETON_SLACK_TOKENS`] for the keys of lists an empty
//! list leaves out. When the skeleton alone is over the budget the request
//! is refused with `invalid_input`, naming the smallest limit that works
//! (only `status`, whose fixed fields are long, can reach this above the
//! minimum limit, `MIN_OUTPUT_TOKEN_BUDGET`, 200; B-112).
//!
//! Items are kept in order while they fit. The first item of a list that
//! does not fit whole is cut and kept only if it then fits; any later item
//! that does not fit ends the list. A memory is cut by its content first;
//! when that would leave fewer than [`MIN_CONTENT_CHARS`] characters, it
//! sheds, in order, its provenance `extra`, tags and anchors, then its
//! title, then its provenance beyond `source` and its `supersedes_id`, and
//! then keeps whatever content still fits, possibly none. The identifier,
//! project, type, status, score and provenance source SPEC 8 asks every
//! result to carry are never cut. A memory's `content_truncated` says the
//! view was cut: its content was shortened, or it shed a field it had
//! (stream H 10; the frozen view has no other flag, so a short memory that
//! lost only its tags, anchors or title is marked too). `omitted` counts
//! the items left out, and `truncated` says whether anything was cut or
//! left out.
//!
//! The whole reply never exceeds the budget: the estimate of a JSON text
//! is at most the sum of the estimates of its parts (`ceil(a + b) <=
//! ceil(a) + ceil(b)`), each item is charged one more token for its comma,
//! and every kept item is measured after it is cut.

use serde::Serialize;
use sm_core::ambient::estimate_tokens;
use sm_core::error::{CoreError, Result};
use sm_core::protocol::{Duplicate, DuplicateGroup, JourneyStep, LinkedMemory, MemoryView, Region};
use sm_core::types::Provenance;

/// Room for the keys of budgeted lists that an empty list leaves out of
/// the skeleton (`"links":[]` is 10 characters).
pub(crate) const SKELETON_SLACK_TOKENS: usize = 8;

/// A cut memory keeps at least this much content before it sheds other
/// fields.
pub(crate) const MIN_CONTENT_CHARS: usize = 16;

pub(crate) fn tokens_of<T: Serialize>(item: &T) -> usize {
    serde_json::to_string(item).map_or(0, |s| estimate_tokens(&s))
}

/// The tokens `skeleton` takes: the reply of `tool` with every budgeted
/// list empty, `truncated` set and `omitted` at `u32::MAX`. Refused when
/// that alone is over `budget`.
pub(crate) fn fixed_cost<T: Serialize>(tool: &str, skeleton: &T, budget: usize) -> Result<usize> {
    let used = tokens_of(skeleton) + SKELETON_SLACK_TOKENS;
    if used > budget {
        return Err(CoreError::invalid_input(format!(
            "limit {budget} is too small for {tool}: its fixed fields take about {used} tokens"
        ))
        .with_next_action(format!("Give a limit of at least {used}, or none.")));
    }
    Ok(used)
}

pub(crate) struct Fitted<T> {
    pub items: Vec<T>,
    pub truncated: bool,
    pub omitted: u32,
}

/// An item that can be made shorter to fit.
pub(crate) trait Cut {
    /// Shortens the item toward `tokens`; [`fit`] measures the result.
    fn cut_to(&mut self, tokens: usize);
}

/// `tokens` less what the rest of an item takes beside its `inner` part,
/// with one token of slack for the rounding of each estimate.
fn inner_room(tokens: usize, outer: usize, inner: usize) -> usize {
    tokens.saturating_sub(outer.saturating_sub(inner) + 1)
}

fn prefix(text: &str, chars: usize) -> String {
    text.chars().take(chars).collect()
}

/// The field groups a cut memory sheds, in order.
const SHED: [fn(&mut MemoryView); 4] = [
    |_| {},
    |v| {
        v.provenance.extra = None;
        v.tags.clear();
        v.anchors.clear();
    },
    |v| v.title = None,
    |v| {
        v.provenance = Provenance::new(v.provenance.source);
        v.supersedes_id = None;
    },
];

fn json_len<T: Serialize>(item: &T) -> usize {
    serde_json::to_string(item).map_or(0, |s| s.len())
}

/// Cuts `v` toward `tokens`. Its `content_truncated` is set when the
/// content was shortened or a shed group removed a field it had: shedding
/// only removes, so the view without content is then shorter.
fn cut_view(v: &mut MemoryView, tokens: usize) {
    let full = std::mem::take(&mut v.content);
    let total = full.chars().count();
    // Measured with the flag set, its longest form.
    v.content_truncated = true;
    let bare = json_len(v);
    for (i, shed) in SHED.iter().enumerate() {
        shed(v);
        let room = tokens.saturating_sub(tokens_of(v)).saturating_mul(4);
        if room >= MIN_CONTENT_CHARS.min(total) || i + 1 == SHED.len() {
            break;
        }
    }
    // Both measured without content, with the flag set.
    let shed_any = json_len(v) < bare;
    // The longest prefix that fits: escapes make JSON longer than the
    // characters, so the length is searched, not computed.
    let (mut lo, mut hi) = (0, total);
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        v.content = prefix(&full, mid);
        if tokens_of(v) <= tokens {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    v.content = prefix(&full, lo);
    v.content_truncated = lo < total || shed_any;
}

impl Cut for MemoryView {
    fn cut_to(&mut self, tokens: usize) {
        cut_view(self, tokens);
    }
}

impl Cut for LinkedMemory {
    fn cut_to(&mut self, tokens: usize) {
        let room = inner_room(tokens, tokens_of(self), tokens_of(&self.memory));
        cut_view(&mut self.memory, room);
    }
}

impl Cut for JourneyStep {
    fn cut_to(&mut self, tokens: usize) {
        let outer = tokens_of(self);
        if let Some(m) = self.memory.as_mut() {
            let room = inner_room(tokens, outer, tokens_of(m));
            cut_view(m, room);
        }
    }
}

impl Cut for Duplicate {
    fn cut_to(&mut self, tokens: usize) {
        let room = inner_room(tokens, tokens_of(self), tokens_of(&self.memory));
        cut_view(&mut self.memory, room);
    }
}

/// A region keeps its most central memories that fit, the first of them
/// cut if it must be; the rest of the list is dropped (its `size` still
/// counts them).
impl Cut for Region {
    fn cut_to(&mut self, tokens: usize) {
        let central = std::mem::take(&mut self.central);
        let room = tokens.saturating_sub(tokens_of(self));
        self.central = fit_room(central, room).items;
    }
}

/// A duplicate group is cut as a whole, never shortened to fewer members:
/// an agent applies the groups it is shown. Every memory in it gets an
/// equal share.
impl Cut for DuplicateGroup {
    fn cut_to(&mut self, tokens: usize) {
        let n = 1 + self.duplicates.len();
        let views = tokens_of(&self.keep)
            + self
                .duplicates
                .iter()
                .map(|d| tokens_of(&d.memory))
                .sum::<usize>();
        let share = tokens.saturating_sub(tokens_of(self).saturating_sub(views) + n) / n;
        if tokens_of(&self.keep) > share {
            cut_view(&mut self.keep, share);
        }
        for d in &mut self.duplicates {
            if tokens_of(&d.memory) > share {
                cut_view(&mut d.memory, share);
            }
        }
    }
}

/// Keeps items in order within `budget` tokens, of which `used` are taken
/// (the reply's [`fixed_cost`] and any list fitted before this one).
pub(crate) fn fit<T: Serialize + Cut>(items: Vec<T>, budget: usize, used: usize) -> Fitted<T> {
    fit_room(items, budget.saturating_sub(used))
}

fn fit_room<T: Serialize + Cut>(items: Vec<T>, room: usize) -> Fitted<T> {
    let mut left = room;
    let total = items.len();
    let mut kept = Vec::new();
    let mut cut = false;
    for mut item in items {
        let cost = tokens_of(&item) + 1;
        if cost <= left {
            left -= cost;
            kept.push(item);
            continue;
        }
        // Only the first item is cut; the list ends here either way.
        if kept.is_empty() && left > 1 {
            item.cut_to(left - 1);
            if tokens_of(&item) < left {
                kept.push(item);
                cut = true;
            }
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

/// Keeps whole items that cannot be cut (links, points, project counts)
/// in order within the budget.
pub(crate) fn fit_whole<T: Serialize>(items: Vec<T>, budget: usize, used: usize) -> Fitted<T> {
    let mut left = budget.saturating_sub(used);
    let total = items.len();
    let mut kept = Vec::new();
    for item in items {
        let cost = tokens_of(&item) + 1;
        if cost > left {
            break;
        }
        left -= cost;
        kept.push(item);
    }
    let omitted = (total - kept.len()) as u32;
    Fitted {
        truncated: omitted > 0,
        omitted,
        items: kept,
    }
}

#[cfg(test)]
mod tests {
    use sm_core::ids::{MemoryId, ProjectId};
    use sm_core::time::Timestamp;
    use sm_core::types::{
        AnchorKind, AnchorRef, Memory, MemoryType, Origin, ProvenanceSource, Status,
    };

    use super::*;

    fn heavy(content: &str) -> MemoryView {
        let mut provenance = Provenance::new(ProvenanceSource::Agent);
        provenance.author = Some("a".repeat(120));
        provenance.file_path = Some("f".repeat(200));
        provenance.extra = Some(serde_json::json!({ "metadata": "m".repeat(400) }));
        let m = Memory {
            id: MemoryId::new_v7(),
            project_id: ProjectId(1),
            memory_type: MemoryType::Decision,
            status: Status::Active,
            title: Some("t".repeat(200)),
            content: content.to_owned(),
            content_hash: String::new(),
            supersedes_id: Some(MemoryId::new_v7()),
            origin: Origin::Local,
            importance: 0.5,
            tags: (0..20)
                .map(|i| format!("tag-{i}-{}", "x".repeat(40)))
                .collect(),
            imported: None,
            created_at: Timestamp::now(),
            updated_at: Timestamp::now(),
            provenance,
            anchors: (0..20)
                .map(|i| AnchorRef::new(AnchorKind::File, format!("src/{i}/{}", "y".repeat(80))))
                .collect::<Result<_>>()
                .expect("anchors"),
        };
        let mut v = crate::view::view(m, "p".repeat(40));
        v.score = Some(0.5);
        v.rank = Some(1);
        v
    }

    /// B-220: a view cut to its barest form keeps its source tag, though the
    /// cut sheds the provenance the tag was read from.
    #[test]
    fn s8_b220_a_cut_view_keeps_its_source_tag() {
        let mut v = heavy(&"word ".repeat(400));
        v.provenance = Provenance::new(ProvenanceSource::Capture);
        v.provenance.extra = Some(serde_json::json!({
            "hook": "pre-compact",
            "source": "tool_result",
            "metadata": "m".repeat(400),
        }));
        v.source_tag = Some(sm_core::types::SourceTag::of(&v.provenance));
        assert_eq!(v.source_tag, Some(sm_core::types::SourceTag::ToolOutput));
        let mut cut = 0;
        for room in [60, 80, 100, 200, 400] {
            let fitted = fit_room(vec![v.clone()], room);
            for kept in &fitted.items {
                assert_eq!(kept.source_tag, v.source_tag, "room {room}");
                cut += usize::from(kept.provenance.extra.is_none());
            }
        }
        assert!(cut > 0, "some room sheds the provenance extra");
    }

    /// SPEC 8: a first memory far over the minimum budget is cut, other
    /// fields too, until it fits, and the list never exceeds its room.
    #[test]
    fn s8_a_first_memory_that_cannot_fit_whole_is_cut_to_fit() {
        for room in [0, 1, 10, 40, 60, 80, 100, 200, 400, 2000] {
            // Newlines and quotes are escaped in JSON, so they cost double.
            let v = heavy(&"\"\n".repeat(5000));
            let whole = tokens_of(&v);
            let fitted = fit_room(vec![v.clone(), v], room);
            let used: usize = fitted.items.iter().map(|i| tokens_of(i) + 1).sum();
            assert!(used <= room, "room {room}: used {used}");
            assert!(fitted.truncated);
            assert_eq!(fitted.items.len() + fitted.omitted as usize, 2);
            if let Some(first) = fitted.items.first() {
                assert!(tokens_of(first) < whole);
                assert!(first.content_truncated);
                assert_eq!(first.score, Some(0.5), "the score is never cut");
            }
        }
        // At a room the bare memory fits in, it is kept with some content
        // and without the fields it shed.
        let fitted = fit_room(vec![heavy(&"word ".repeat(400))], 100);
        let kept = &fitted.items[0];
        assert!(
            kept.content.chars().count() >= MIN_CONTENT_CHARS,
            "{kept:?}"
        );
        assert!(kept.tags.is_empty() && kept.anchors.is_empty());
        assert!(kept.provenance.extra.is_none());
    }

    /// SPEC 8, stream H 10: a memory with under [`MIN_CONTENT_CHARS`] of
    /// content keeps it whole but sheds its other fields to fit; the view
    /// says it was cut. A cut that removes nothing is not marked.
    #[test]
    fn s8_a_short_memory_that_sheds_fields_is_marked_cut() {
        let short = "deploy fridays";
        assert!(short.chars().count() < MIN_CONTENT_CHARS);
        let v = heavy(short);
        let whole = tokens_of(&v);
        let fitted = fit_room(vec![v], 100);
        let kept = &fitted.items[0];
        assert!(tokens_of(kept) < whole);
        assert_eq!(kept.content, short, "the content is kept whole");
        assert!(kept.tags.is_empty() && kept.anchors.is_empty());
        assert!(kept.content_truncated, "a caller can tell the view was cut");
        assert!(fitted.truncated);

        // Room for the whole view: nothing shed, nothing shortened.
        let mut v = heavy(short);
        let original = v.clone();
        cut_view(&mut v, whole + 20);
        assert_eq!(v, original);
        assert!(!v.content_truncated);
    }

    /// SPEC 8: a whole item that does not fit is never kept, first or not.
    #[test]
    fn s8_a_whole_item_over_the_budget_is_left_out() {
        let f = fit_whole(vec!["x".repeat(1000), "y".to_owned()], 100, 0);
        assert!(f.items.is_empty());
        assert_eq!(f.omitted, 2);
        assert!(f.truncated);
        let f = fit_whole(vec!["x".repeat(100)], 100, 90);
        assert!(f.items.is_empty());
    }

    /// The skeleton check refuses a limit the fixed fields alone exceed.
    #[test]
    fn s8_a_limit_below_the_fixed_fields_is_refused() {
        let long = "z".repeat(1000);
        let e = fixed_cost("status", &long, 100).expect_err("over the limit");
        assert_eq!(e.code, sm_core::error::ErrorCode::InvalidInput);
        assert!(e.next_action.contains("at least"), "{}", e.next_action);
        assert!(fixed_cost("status", &"z", 100).is_ok());
    }
}
