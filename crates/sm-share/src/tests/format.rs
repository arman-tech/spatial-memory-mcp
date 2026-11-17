//! The shared-file format (SPEC section 9, "Shared file format"; B-78).

use sm_core::ids::MemoryId;
use sm_core::time::Timestamp;
use sm_core::types::{AnchorKind, AnchorRef, MemoryType, Status};

use crate::format::{SharedDoc, render_markdown, shared_path, slug, text_hash};

const ID: &str = "0193a8f0-5c1e-7000-8000-3f2a9c1e5b7d";
const OTHER: &str = "0193a8f0-5c1e-7000-8000-000000000001";

fn doc() -> SharedDoc {
    SharedDoc {
        id: MemoryId::parse(ID).expect("id"),
        memory_type: MemoryType::Decision,
        status: Status::Active,
        date: Timestamp::parse_date("2026-11-15").expect("date"),
        supersedes: None,
        anchors: Vec::new(),
        tags: Vec::new(),
        commit: None,
        title: "Use optimistic concurrency on dbo.Orders".to_owned(),
        body: "Body in Markdown.".to_owned(),
    }
}

/// render, parse, render: the same document and the same bytes.
#[track_caller]
fn round_trip(d: &SharedDoc) -> String {
    let first = d.render();
    let back = SharedDoc::parse(&first).expect("the rendered file parses");
    assert_eq!(&back, d, "parse(render(doc)) is the document");
    assert_eq!(back.render(), first, "render is stable");
    first
}

#[track_caller]
fn refused_at(text: &str, line: usize, reason_part: &str) {
    let e = SharedDoc::parse(text).expect_err("a malformed file is refused");
    assert_eq!(e.line, line, "line of {e}");
    assert!(
        e.reason.contains(reason_part),
        "reason {:?} does not mention {reason_part:?}",
        e.reason
    );
}

fn full_file() -> String {
    let mut d = doc();
    d.supersedes = Some(MemoryId::parse(OTHER).expect("id"));
    d.anchors = vec![
        AnchorRef::new(AnchorKind::Table, "dbo.Orders").expect("anchor"),
        AnchorRef::new(AnchorKind::Service, "order-api").expect("anchor"),
    ];
    d.tags = vec!["concurrency".to_owned(), "sql".to_owned()];
    d.commit = Some("4f2a9c1".to_owned());
    d.render()
}

#[test]
fn s9_render_writes_the_spec_example_layout() {
    let text = full_file();
    assert_eq!(
        text,
        "---\n\
         schema: 1\n\
         id: 0193a8f0-5c1e-7000-8000-3f2a9c1e5b7d\n\
         type: decision\n\
         status: active\n\
         date: 2026-11-15\n\
         supersedes: 0193a8f0-5c1e-7000-8000-000000000001\n\
         anchors: [table:dbo.Orders, service:order-api]\n\
         tags: [concurrency, sql]\n\
         commit: 4f2a9c1\n\
         ---\n\
         # Use optimistic concurrency on dbo.Orders\n\
         \n\
         Body in Markdown.\n"
    );
}

#[test]
fn s9_round_trip_every_type_and_status() {
    for &t in MemoryType::ALL {
        for &s in Status::ALL {
            let mut d = doc();
            d.memory_type = t;
            d.status = s;
            let text = round_trip(&d);
            assert!(text.contains(&format!("\ntype: {t}\n")));
            assert!(text.contains(&format!("\nstatus: {s}\n")));
        }
    }
}

#[test]
fn s9_round_trip_with_and_without_each_optional_key() {
    let bare = doc();
    let text = round_trip(&bare);
    for key in ["supersedes:\n", "anchors:\n", "tags:\n", "commit:\n"] {
        assert!(text.contains(key), "an empty {key:?} is written");
    }

    let mut d = doc();
    d.supersedes = Some(MemoryId::parse(OTHER).expect("id"));
    round_trip(&d);

    let mut d = doc();
    d.anchors = vec![AnchorRef::new(AnchorKind::File, "src/lib.rs").expect("anchor")];
    round_trip(&d);

    let mut d = doc();
    d.tags = vec!["one".to_owned(), "two".to_owned()];
    round_trip(&d);

    let mut d = doc();
    d.commit = Some("0123456789abcdef0123456789abcdef01234567".to_owned());
    round_trip(&d);

    let mut d = doc();
    d.body = String::new();
    let text = round_trip(&d);
    assert!(text.ends_with("---\n# Use optimistic concurrency on dbo.Orders\n"));

    // Every optional key at once.
    let full = full_file();
    let back = SharedDoc::parse(&full).expect("parse");
    assert_eq!(back.render(), full);
}

#[test]
fn s9_omitted_optional_keys_and_empty_lists_read_as_empty() {
    let text = format!(
        "---\nschema: 1\nid: {ID}\ntype: decision\nstatus: active\ndate: 2026-11-15\n\
         anchors: []\n---\n# Use optimistic concurrency on dbo.Orders\n\nBody in Markdown.\n"
    );
    let d = SharedDoc::parse(&text).expect("parse");
    assert_eq!(d, doc());
    assert_eq!(
        d.render(),
        doc().render(),
        "rendering gives the canonical bytes"
    );
}

#[test]
fn s9_round_trip_body_with_headings_and_fenced_code() {
    let mut d = doc();
    d.body = "Intro line.\n\n# Another level-1 heading\n\n## Steps\n\n```rust\nfn main() {\n    \
              println!(\"---\");\n}\n```\n\n- item one\n  continued\n\n---\n\nTrailing paragraph."
        .to_owned();
    let text = round_trip(&d);
    let back = SharedDoc::parse(&text).expect("parse");
    assert_eq!(back.title, d.title, "later headings stay in the body");
    assert!(back.body.contains("# Another level-1 heading"));
    assert!(back.body.contains("    println!(\"---\");"));
}

#[test]
fn s9_crlf_input_reads_as_lf_and_renders_canonically() {
    let lf = full_file();
    let crlf = lf.replace('\n', "\r\n");
    let d = SharedDoc::parse(&crlf).expect("parse CRLF");
    assert_eq!(d.render(), lf);
    let bom = format!("\u{feff}{lf}");
    assert_eq!(SharedDoc::parse(&bom).expect("parse with BOM").render(), lf);
}

#[test]
fn s9_text_hash_is_of_the_markdown_part() {
    let d = doc();
    assert_eq!(d.text_hash(), text_hash(Some(&d.title), &d.body));
    assert_eq!(
        d.text_hash(),
        sm_core::hash::sha256_hex(render_markdown(&d.title, &d.body).as_bytes())
    );
    let mut changed = doc();
    changed.tags = vec!["x".to_owned()];
    changed.status = Status::Superseded;
    assert_eq!(changed.text_hash(), d.text_hash(), "metadata is not text");
    changed.body.push_str(" More.");
    assert_ne!(changed.text_hash(), d.text_hash());
}

#[test]
fn s9_malformed_unknown_key() {
    let text = full_file().replace("commit: 4f2a9c1\n", "commit: 4f2a9c1\nowner: me\n");
    refused_at(&text, 11, "unknown key `owner`");
}

#[test]
fn s9_malformed_nested_value() {
    let text = full_file().replace("commit: 4f2a9c1", "commit: {sha: 4f2a9c1}");
    refused_at(&text, 10, "nested value");
    let text = full_file().replace("tags: [concurrency, sql]", "tags: [[concurrency], sql]");
    refused_at(&text, 9, "nested list");
    let text = full_file().replace("commit: 4f2a9c1\n", "commit:\n  sha: 4f2a9c1\n");
    refused_at(&text, 11, "indented line");
}

#[test]
fn s9_malformed_multi_line_value() {
    let text = full_file().replace("commit: 4f2a9c1\n", "commit: |\n");
    refused_at(&text, 10, "multi-line value");
    let text = full_file().replace("tags: [concurrency, sql]\n", "tags:\n- concurrency\n");
    refused_at(&text, 10, "block list item");
    let text = full_file().replace("tags: [concurrency, sql]", "tags: [concurrency,");
    refused_at(&text, 9, "not closed");
}

#[test]
fn s9_malformed_schema_2() {
    let text = full_file().replace("schema: 1\n", "schema: 2\n");
    refused_at(&text, 2, "schema 2 is not supported");
    // A newer schema is reported as newer even with keys this build lacks.
    let text = text.replace("commit: 4f2a9c1\n", "commit: 4f2a9c1\nreviewers: [a]\n");
    refused_at(&text, 2, "schema 2 is not supported");
}

#[test]
fn s9_malformed_bad_id() {
    let text = full_file().replace(&format!("id: {ID}"), "id: not-a-uuid");
    refused_at(&text, 3, "");
    let upper = ID.to_ascii_uppercase();
    let text = full_file().replace(&format!("id: {ID}"), &format!("id: {upper}"));
    refused_at(&text, 3, "lower-case hyphenated UUID");
    let text = full_file().replace(
        &format!("supersedes: {OTHER}"),
        &format!("supersedes: {ID}"),
    );
    refused_at(&text, 7, "cannot supersede itself");
}

#[test]
fn s9_malformed_bad_tag() {
    let long = "x".repeat(65);
    let text = full_file().replace("tags: [concurrency, sql]", &format!("tags: [{long}, sql]"));
    refused_at(&text, 9, "longer than 64 characters");
    let text = full_file().replace("tags: [concurrency, sql]", "tags: [con	currency, sql]");
    refused_at(&text, 9, "control characters");
    let text = full_file().replace("tags: [concurrency, sql]", "tags: [sql, sql]");
    refused_at(&text, 9, "duplicate tag");
}

#[test]
fn s9_malformed_layout() {
    refused_at("", 1, "front matter line");
    refused_at("# Title only\n", 1, "front matter line");
    let open = "---\nschema: 1\n";
    refused_at(open, 2, "not closed");
    let text = full_file().replace("# Use optimistic", "Use optimistic");
    refused_at(&text, 12, "level-1 heading");
    let text = full_file().replace("commit: 4f2a9c1", "commit: xyz");
    refused_at(&text, 10, "hexadecimal");
    let text = full_file().replace("date: 2026-11-15", "date: 15/11/2026");
    refused_at(&text, 6, "");
    let text = full_file().replace("type: decision", "type: essay");
    refused_at(&text, 4, "");
}

#[test]
fn s9_path_and_slug() {
    let id = MemoryId::parse(ID).expect("id");
    assert_eq!(id.last8(), "9c1e5b7d");
    assert_eq!(
        shared_path(
            MemoryType::Decision,
            "Use optimistic concurrency on dbo.Orders",
            &id
        ),
        ".spatial-memory/shared/decision/use-optimistic-concurrency-on-dbo-orders--9c1e5b7d.md"
    );
    assert_eq!(
        doc().path(),
        ".spatial-memory/shared/decision/use-optimistic-concurrency-on-dbo-orders--9c1e5b7d.md"
    );
    assert_eq!(slug("  Hello, World!  "), "hello-world");
    assert_eq!(slug("Ünïcödé only ✓"), "n-c-d-only");
    assert_eq!(slug("!!!"), "memory");
    assert_eq!(slug(""), "memory");
    let long = "a".repeat(39) + " bcd";
    assert_eq!(slug(&long), "a".repeat(39), "the cut drops a trailing dash");
    assert_eq!(slug(&"x".repeat(100)).len(), 40);
}
