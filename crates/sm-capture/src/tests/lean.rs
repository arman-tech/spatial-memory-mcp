//! B-189 (SPEC section 16, Candidate selection): the lean offering. A
//! compaction summary's sections that are not offered, and a summary unit
//! that repeats the summary units the session offered before it. Every
//! text is synthetic.

use std::path::Path;

use serde_json::{Value, json};
use sm_core::config::CaptureSettings;
use sm_core::protocol::HookEvent;

use super::{project, source_of, temp_dir};
use crate::pipeline::{CaptureInput, CaptureOutcome, TextDrop, capture, replay};
use crate::quote::{SENTENCE_KEY_MIN, sentence_keys};
use crate::select::{split_units, summary_units};
use crate::session::{SUMMARY_KEY_CAP, SUMMARY_REPEAT, SessionState};

fn entry(kind: &str, uuid: &str, content: Value) -> Value {
    json!({
        "type": kind,
        "uuid": uuid,
        "timestamp": format!("t-{uuid}"),
        "message": {"role": kind, "content": content},
    })
}

fn user(uuid: &str, text: &str) -> Value {
    entry("user", uuid, json!(text))
}

fn says(uuid: &str, text: &str) -> Value {
    entry("assistant", uuid, json!([{"type": "text", "text": text}]))
}

fn compact(uuid: &str) -> Value {
    json!({
        "type": "system",
        "subtype": "compact_boundary",
        "uuid": uuid,
        "timestamp": format!("t-{uuid}"),
        "content": "Conversation compacted",
    })
}

fn summary(uuid: &str, text: &str) -> Value {
    let mut v = user(uuid, text);
    v["isCompactSummary"] = json!(true);
    v
}

fn write(dir: &Path, entries: &[Value]) -> String {
    let p = dir.join("session.jsonl");
    let text: String = entries.iter().map(|e| format!("{e}\n")).collect();
    std::fs::write(&p, text).expect("write a transcript");
    p.to_str().expect("a UTF-8 temporary path").to_owned()
}

const PREAMBLE: &str = "This session is being continued from a previous conversation that ran \
out of context. The summary below covers the earlier portion of the conversation.";

const PRIMARY: &str = "## 1. Primary Request and Intent\nThe owner asked for the report cache \
to move to the shared volume. The nightly job writes it there. Reports older than a week are \
deleted.";

const MESSAGES: &str = "## 6. All User Messages\n- \"Move the report cache to the shared \
volume please.\"\n- \"Keep the old path working for one release.\"";

const FOLLOW_UP: &str =
    "### Follow-up\n- \"Also rename the nightly job to nightly-report this week.\"";

const PENDING: &str =
    "## 7. Pending Tasks\n- Rename the nightly job to nightly-report and update the schedule.";

const CURRENT: &str = "## 8. Current Work\nThe cache path now reads from the shared volume, \
because the local disk filled twice last week.";

/// A first compaction summary: three sections offered (with the preamble)
/// and two that are not, one of them with a sub-heading.
fn first_summary() -> String {
    [PREAMBLE, PRIMARY, MESSAGES, FOLLOW_UP, PENDING, CURRENT].join("\n\n")
}

/// The primary request of a later summary: the first summary's, with one
/// sentence added. Three of its four sentence keys were offered before; its
/// heading has no key (B-196).
const PRIMARY_AGAIN: &str = "## 1. Primary Request and Intent\nThe owner asked for the report \
cache to move to the shared volume. The nightly job writes it there. Reports older than a week \
are deleted. It runs at two in the morning.";

const CURRENT_AGAIN: &str = "## 8. Current Work\nThe retention of cached reports is now fourteen \
days, because audits need two weeks of history.";

/// A later compaction summary of the same session: the same preamble, a
/// primary request that repeats the first's, and new current work.
fn second_summary() -> String {
    [PREAMBLE, PRIMARY_AGAIN, CURRENT_AGAIN].join("\n\n")
}

const Q1: &str = "Where should the report cache live from now on?";
const R1: &str = "The report cache moves to the shared volume in the next change.";
const Q2: &str = "Does the old path keep working for a while?";
const R2: &str = "Yes, the old path stays a link to the new one for one release.";
const Q3: &str = "How long are cached reports kept?";
/// A reply that states again what the first summary's current work states.
/// Replies are not compared with summaries, so it is offered.
const R3: &str = "The cache path now reads from the shared volume, because the local disk \
filled twice last week.";

fn transcript() -> Vec<Value> {
    vec![
        user("u1", Q1),
        says("a1", R1),
        compact("cb1"),
        summary("s1", &first_summary()),
        user("u2", Q2),
        says("a2", R2),
        compact("cb2"),
        summary("s2", &second_summary()),
        user("u3", Q3),
        says("a3", R3),
    ]
}

fn input<'a>(
    event: HookEvent,
    payload: &'a Value,
    settings: &'a CaptureSettings,
    project: &'a sm_core::project::ProjectIdentity,
) -> CaptureInput<'a> {
    CaptureInput {
        event,
        payload,
        settings,
        project,
        commit_sha: None,
        transcript_offset: Some(0),
        from_start: true,
    }
}

fn contents(outcome: &CaptureOutcome) -> Vec<&str> {
    outcome
        .candidates
        .iter()
        .map(|c| c.content.as_str())
        .collect()
}

#[test]
fn s16_b189_the_sections_of_a_summary_that_are_not_offered() {
    let text = first_summary();
    assert_eq!(
        split_units(&text),
        [PREAMBLE, PRIMARY, MESSAGES, FOLLOW_UP, PENDING, CURRENT],
        "each section is its own unit"
    );
    // The user's messages, with their sub-heading, and the pending tasks
    // are left out; the preamble, the request and the current work are
    // offered.
    let (units, dropped) = summary_units(&text);
    assert_eq!(units, [PREAMBLE, PRIMARY, CURRENT]);
    assert_eq!(dropped, 3);

    // The same headings written as numbered titles ending in a colon.
    let numbered = [
        PREAMBLE,
        "1. Primary request and intent:\nThe report cache moves to the shared volume.",
        "6. All user messages:\n- \"Move the report cache to the shared volume please.\"",
        "7. Pending tasks:\n- Rename the nightly job to nightly-report.",
        "8. Current work:\nThe cache path now reads from the shared volume.",
    ];
    let (units, dropped) = summary_units(&numbered.join("\n\n"));
    assert_eq!(units, [numbered[0], numbered[1], numbered[4]]);
    assert_eq!(dropped, 2);

    // Units before the first heading named in the list are offered, and a
    // unit under a heading the list does not name stays in the section it
    // is in.
    let unnamed = [
        "The report cache moves to the shared volume next week.",
        "## Background\nThe local disk filled twice last week.",
        "## 7. Pending Tasks\n- Rename the nightly job to nightly-report.",
        "## Notes\n- The schedule changes with the rename of the job.",
    ];
    let (units, dropped) = summary_units(&unnamed.join("\n\n"));
    assert_eq!(units, [unnamed[0], unnamed[1]]);
    assert_eq!(dropped, 2);
    // Each summary is read on its own: the next one starts offered.
    let (units, dropped) = summary_units(&first_summary());
    assert_eq!((units.len(), dropped), (3, 3));

    // A text without headings is split as any other text.
    let plain = "The report cache moves to the shared volume next week.";
    assert_eq!(summary_units(plain), (split_units(plain), 0));
}

#[test]
fn s16_b189_a_summary_unit_that_repeats_earlier_ones_is_not_offered() {
    let mut state = SessionState::new();
    assert!(
        !state.summary_repeats(PRIMARY),
        "nothing was offered before"
    );
    // Three of its four keys were offered before: 0.75 is at least the
    // threshold. The heading is not counted (B-196).
    let (_, body) = PRIMARY_AGAIN.split_once('\n').expect("a heading line");
    let keys = sentence_keys(body);
    assert_eq!(keys.len(), 4, "{keys:?}");
    const { assert!(3.0 / 4.0 >= SUMMARY_REPEAT) };
    assert!(state.summary_repeats(PRIMARY_AGAIN));
    // Its keys were recorded, so the added sentence now counts as seen.
    assert!(state.summary_repeats("It runs at two in the morning."));
    // Two of four seen is below the threshold.
    let half = "The owner asked for the report cache to move to the shared volume. \
The nightly job writes it there. The schedule is kept in the repository. The volume is \
backed up every hour.";
    const { assert!(2.0 / 4.0 < SUMMARY_REPEAT) };
    assert!(!state.summary_repeats(half));
    // A unit without a key of at least SENTENCE_KEY_MIN characters repeats
    // nothing, however often it is seen.
    let short = "## 8. Current Work";
    assert!(sentence_keys(short).is_empty());
    assert!(
        sentence_keys(short)
            .iter()
            .all(|k| k.chars().count() >= SENTENCE_KEY_MIN)
    );
    assert!(!state.summary_repeats(short));
    assert!(!state.summary_repeats(short));
    // Keys are compared as phrases are: case, list markers, emphasis and
    // closing punctuation do not count.
    assert!(state.summary_repeats("- **the nightly job writes it there**"));
    // A daemon restart starts with empty state, and offers it again.
    assert!(!SessionState::new().summary_repeats(PRIMARY_AGAIN));
}

#[test]
fn s16_b196_a_summary_heading_does_not_count_toward_a_repeat() {
    // C12 review finding 2.2: every compaction summary repeats its section
    // headings, so a heading key of at least SENTENCE_KEY_MIN characters
    // was always seen and tipped a unit with new content over the
    // threshold.
    for heading in [
        "## 1. Primary Request and Intent",
        "1. Primary Request and Intent:",
    ] {
        let mut state = SessionState::new();
        let first = format!(
            "{heading}\nThe owner asked for the report cache to move to the shared volume. \
The nightly job writes it there. The schedule is kept in the repository."
        );
        assert!(!state.summary_repeats(&first));
        let later = format!(
            "{heading}\nThe owner asked for the report cache to move to the shared volume. \
The nightly job writes it there. The volume is now encrypted at rest."
        );
        // The heading is a key of its own: with it, three of four keys were
        // seen, at least the threshold; without it, two of three, below it.
        assert_eq!(sentence_keys(&later).len(), 4, "{heading}");
        const { assert!(3.0 / 4.0 >= SUMMARY_REPEAT) };
        const { assert!(2.0 / 3.0 < SUMMARY_REPEAT) };
        assert!(!state.summary_repeats(&later), "{heading}");
        // A unit that is a heading alone has no key, and repeats nothing.
        assert!(!state.summary_repeats(heading), "{heading}");
        // The body alone, all seen, still repeats.
        assert!(state.summary_repeats(
            "The nightly job writes it there. The volume is now encrypted at rest."
        ));
    }
}

#[test]
fn s16_b189_the_summary_keys_of_a_session_are_capped() {
    let mut state = SessionState::new();
    let sentence = |i: usize| format!("Sentence number {i} of the long summary.");
    for i in 0..=SUMMARY_KEY_CAP {
        assert!(!state.summary_repeats(&sentence(i)), "{i}");
    }
    // The newest key is kept, and the oldest one was forgotten.
    assert!(state.summary_repeats(&sentence(SUMMARY_KEY_CAP)));
    assert!(!state.summary_repeats(&sentence(0)));
    assert!(state.summary_repeats(&sentence(2)));
}

#[test]
fn s16_b189_a_replay_and_a_pre_compact_offer_the_lean_summary() {
    let dir = temp_dir();
    let path = write(dir.path(), &transcript());
    let payload = json!({"session_id": "s", "transcript_path": path, "trigger": "manual"});
    let settings = CaptureSettings::default();
    let project = project(dir.path());

    let replayed = replay(
        &input(HookEvent::PreCompact, &payload, &settings, &project),
        &mut SessionState::new(),
    );
    let live = capture(
        &input(HookEvent::PreCompact, &payload, &settings, &project),
        &mut SessionState::new(),
    );
    for (name, outcome) in [("replay", &replayed), ("pre-compact", &live)] {
        // The first summary's two sections that are not offered (three
        // units), the second summary's preamble (the same text, so the
        // content hash) and its repeated primary request.
        let mut drops = outcome.drops.clone();
        drops.sort_by_key(|d| format!("{d:?}"));
        assert_eq!(
            drops,
            [
                TextDrop::Duplicate,
                TextDrop::Repeat,
                TextDrop::Section,
                TextDrop::Section,
                TextDrop::Section,
            ],
            "{name}"
        );
        assert_eq!(
            contents(outcome),
            [
                Q1,
                R1,
                PREAMBLE,
                PRIMARY,
                CURRENT,
                Q2,
                R2,
                CURRENT_AGAIN,
                Q3,
                R3
            ],
            "{name}"
        );
        let sources: Vec<&str> = outcome.candidates.iter().map(source_of).collect();
        assert_eq!(
            sources,
            [
                "prompt", "reply", "summary", "summary", "summary", "prompt", "reply", "summary",
                "prompt", "reply"
            ],
            "{name}: the reply that states the summary's current work again is offered"
        );
    }
}
