//! The memory a judge's spans mark in a candidate's text (B-173, B-174,
//! B-191), and the sentence keys of the lean offering (B-189). A span whose
//! start and end are the same phrase is the quote of B-173. Every text is
//! synthetic.

use crate::quote::{Found, SENTENCE_KEY_MIN, sentence_keys, spanned};

/// Each phrase as a span that starts and ends with it.
fn quotes_of(text: &str, quotes: &[&str]) -> Found {
    spanned(text, quotes.iter().map(|q| (*q, *q)))
}

fn spans_of(text: &str, spans: &[(&str, &str)]) -> Found {
    spanned(text, spans.iter().copied())
}

#[test]
fn s16_a_quote_is_found_whatever_its_markdown_marks_and_spacing() {
    let text = "The **cache** is keyed\n  by `path`.\nIt is cleared on start.";
    let q = quotes_of(text, &["The cache is keyed by path."]);
    assert_eq!(
        q.memory, "The **cache** is keyed\n  by `path`.",
        "copied from the text"
    );
    assert_eq!((q.looked, q.dropped), (1, 0));
    let typographic = quotes_of(
        "The log says \u{201C}idle\u{201D} when the queue is empty.",
        &["The log says \"idle\" when the queue is empty"],
    );
    assert_eq!(
        typographic.memory,
        "The log says \u{201C}idle\u{201D} when the queue is empty."
    );
}

#[test]
fn s16_a_quote_that_is_not_in_the_text_is_dropped() {
    let text = "The cache is keyed by path. Logs rotate daily.";
    let none = quotes_of(text, &["The cache is keyed by hash."]);
    assert_eq!(none.memory, "");
    assert_eq!((none.looked, none.dropped), (1, 1));
    let one = quotes_of(text, &["The cache is keyed by hash.", "Logs rotate daily."]);
    assert_eq!(one.memory, "Logs rotate daily.", "only what the text says");
    assert_eq!((one.looked, one.dropped), (2, 1));
    let reworded = quotes_of(text, &["The cache is keyed by its path."]);
    assert_eq!(reworded.memory, "", "a reworded quote is not in the text");
}

#[test]
fn s16_a_quote_cut_short_brings_its_whole_sentence() {
    let text = "Retries run only for idempotent requests, never for uploads. Logs rotate daily.";
    let q = quotes_of(text, &["Retries run"]);
    assert_eq!(
        q.memory, "Retries run only for idempotent requests, never for uploads.",
        "the qualifiers of the sentence are kept"
    );
    let middle = quotes_of(text, &["for idempotent requests"]);
    assert_eq!(middle.memory, q.memory);
    let across = quotes_of(text, &["never for uploads. Logs rotate"]);
    assert_eq!(
        across.memory, text,
        "a quote across two sentences brings both"
    );
}

#[test]
fn s16_a_quote_starts_and_ends_at_a_word_boundary() {
    let text = "The importer runs first.";
    assert_eq!(quotes_of(text, &["port"]).memory, "");
    assert_eq!(quotes_of(text, &["importer run"]).memory, "");
    assert_eq!(quotes_of(text, &["importer runs"]).memory, text);
}

#[test]
fn s16_an_abbreviation_does_not_end_a_sentence() {
    let text = "Use a fixed seed, e.g. the value 7, in tests. Nothing else changes.";
    let q = quotes_of(text, &["Use a fixed seed"]);
    assert_eq!(q.memory, "Use a fixed seed, e.g. the value 7, in tests.");
}

#[test]
fn s16_quoted_sentences_keep_the_texts_order_and_merge_where_they_touch() {
    let text = "A is first. B is second. C is third. D is fourth.";
    let touching = quotes_of(text, &["C is third", "A is first", "B is second"]);
    assert_eq!(touching.memory, "A is first. B is second. C is third.");
    let apart = quotes_of(text, &["D is fourth", "A is first"]);
    assert_eq!(apart.memory, "A is first.\nD is fourth.");
    let twice = quotes_of(text, &["B is second", "B is second."]);
    assert_eq!(twice.memory, "B is second.");
}

#[test]
fn s16_a_quotes_list_marker_quotation_marks_and_closing_stop_are_ignored() {
    let text = "Rules:\n- Never edit the lock file by hand.\n- Run the formatter first.\n1. Pin every version.";
    let q = quotes_of(
        text,
        &[
            "Never edit the lock file by hand",
            "\"- Run the formatter first.\"",
            "2. Pin every version",
        ],
    );
    assert_eq!(
        q.memory,
        "- Never edit the lock file by hand.\n- Run the formatter first.\n1. Pin every version."
    );
    assert_eq!(q.dropped, 0);
    let blank = quotes_of(text, &["", "  "]);
    assert_eq!(
        (blank.memory.as_str(), blank.looked),
        ("", 0),
        "blank quotes are not counted"
    );
}

/// B-174: a quoted nested item without its parent reads as part of the
/// item the memory puts before it, which changes what it is about.
#[test]
fn s16_a_quoted_nested_line_brings_the_lines_it_is_read_under() {
    let text = "Two defects:\n\
                1. The importer stored a whole session as one memory.\n   \
                - Cause: the turn boundary was missed.\n\
                2. The doctor failed in every project.\n   \
                - Cause: it read the wrong settings key.\n";
    let q = quotes_of(
        text,
        &[
            "The importer stored a whole session as one memory",
            "it read the wrong settings key",
        ],
    );
    assert_eq!(
        q.memory,
        "1. The importer stored a whole session as one memory.\n\
         2. The doctor failed in every project.\n   \
         - Cause: it read the wrong settings key.",
        "the second cause stays under the doctor's item"
    );
    assert_eq!((q.looked, q.dropped), (2, 0), "a parent is not a span");

    let deep = "- Release:\n  - Windows:\n    - Sign the installer first.\n- Logs rotate daily.";
    assert_eq!(
        quotes_of(deep, &["Sign the installer first"]).memory,
        &deep[..deep.find("\n- Logs").unwrap()],
        "every parent up to a line that is not indented"
    );

    let continued = "- The importer runs first.\n  It must finish before the indexer starts.\n- Logs rotate daily.";
    assert_eq!(
        quotes_of(continued, &["It must finish before the indexer starts"]).memory,
        "- The importer runs first.\n  It must finish before the indexer starts.",
        "an indented continuation is read under its item"
    );

    let top = "Rules:\n- Never edit the lock file by hand.";
    assert_eq!(
        quotes_of(top, &["Never edit the lock file by hand"]).memory,
        "- Never edit the lock file by hand.",
        "a line that is not indented brings nothing"
    );
}

// --- spans (B-191) ----------------------------------------------------------

/// B-191: a span marks a whole statement by its first and last words, so
/// every sentence from the one its start is in to the one its end is in is
/// kept, the sentences between them too.
#[test]
fn s16_b191_a_span_brings_every_sentence_from_its_start_to_its_end() {
    let text = "A is first. B is second. C is third. D is fourth.";
    let q = spans_of(text, &[("B is", "C is third")]);
    assert_eq!(q.memory, "B is second. C is third.");
    assert_eq!((q.looked, q.dropped), (1, 0));
    let long = spans_of(text, &[("A is first", "is fourth")]);
    assert_eq!(long.memory, text, "the sentences between are kept");
    let inside = spans_of(text, &[("is second", "second")]);
    assert_eq!(
        inside.memory, "B is second.",
        "a span inside one sentence brings that sentence"
    );
}

/// B-191: a span's end is looked for from its start, so a statement whose
/// last words also occur earlier in the text ends where it does.
#[test]
fn s16_b191_a_span_ends_at_the_first_end_after_its_start() {
    let text = "The old fix was to pin the linker. The new fix, after the update, is to pin \
                the linker. Logs rotate daily.";
    let q = spans_of(text, &[("The new fix", "to pin the linker")]);
    assert_eq!(
        q.memory,
        "The new fix, after the update, is to pin the linker."
    );
    let same = spans_of(text, &[("Logs rotate daily", "Logs rotate daily")]);
    assert_eq!(same.memory, "Logs rotate daily.", "start and end the same");
}

/// B-191: a span whose start or end is not in the text, or whose end is
/// only before its start, is dropped; a span with a blank start or end is
/// not counted.
#[test]
fn s16_b191_a_span_not_in_the_text_or_ending_before_it_starts_is_dropped() {
    let text = "Logs rotate daily. The cache is keyed by path.";
    let backwards = spans_of(text, &[("The cache is keyed", "Logs rotate daily")]);
    assert_eq!(backwards.memory, "");
    assert_eq!((backwards.looked, backwards.dropped), (1, 1));
    let no_end = spans_of(text, &[("The cache is keyed", "by hash")]);
    assert_eq!((no_end.memory.as_str(), no_end.dropped), ("", 1));
    let no_start = spans_of(text, &[("The index is keyed", "by path")]);
    assert_eq!((no_start.memory.as_str(), no_start.dropped), ("", 1));
    let blank = spans_of(text, &[("", "Logs rotate daily"), ("Logs rotate", "  ")]);
    assert_eq!(
        (blank.memory.as_str(), blank.looked, blank.dropped),
        ("", 0, 0),
        "a span with a blank start or end is not counted"
    );
    let one = spans_of(
        text,
        &[("Logs rotate", "by path"), ("The index", "is keyed")],
    );
    assert_eq!(one.memory, text);
    assert_eq!((one.looked, one.dropped), (2, 1));
}

/// B-189: the sentence keys the repeat filter compares are each sentence
/// lower-cased, without its marks and closing punctuation, and only those
/// of at least SENTENCE_KEY_MIN characters.
#[test]
fn s16_b189_sentence_keys_are_the_texts_long_sentences_as_phrases_are_compared() {
    let text = "## Heading\n- **The nightly job writes it there.**\nShort one.\n\
                The cache is keyed by path, not by hash! Logs rotate daily at noon.";
    let mut keys: Vec<String> = sentence_keys(text).into_iter().collect();
    keys.sort();
    assert_eq!(
        keys,
        [
            "logs rotate daily at noon",
            "the cache is keyed by path, not by hash",
            "the nightly job writes it there",
        ]
    );
    assert!(keys.iter().all(|k| k.chars().count() >= SENTENCE_KEY_MIN));
    assert!(sentence_keys("").is_empty());
}
