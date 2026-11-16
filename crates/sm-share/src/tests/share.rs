//! `share` (SPEC section 9, Team sharing rules 1 and 2 and "Security";
//! AT-9; B-78).

use std::collections::BTreeSet;

use serde_json::Value;
use sm_core::error::ErrorCode;
use sm_core::ids::MemoryId;
use sm_core::protocol::ShareParams;
use sm_core::traits::SecretScanner;
use sm_core::types::{AnchorKind, AnchorRef, MemoryType, Origin};
use sm_store::{Access, Revision, SyncMark};

use super::{
    Repo, add_note, assert_code, fixture, insert, note_input, project, read_fixture, scanner,
};
use crate::format::{GITATTRIBUTES, SHARED_DIR, SharedDoc};
use crate::share;

fn params(ids: &[MemoryId]) -> ShareParams {
    ShareParams { ids: ids.to_vec() }
}

/// Python's `[REDACTED_<KIND>]` written as version 3's `[REDACTED:<kind>]`
/// (the fixture's `v3_note`).
fn v3(s: &str) -> String {
    const OPEN: &str = "[REDACTED_";
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(at) = rest.find(OPEN) {
        out.push_str(&rest[..at]);
        let after = &rest[at + OPEN.len()..];
        let kind_len = after
            .bytes()
            .take_while(|b| b.is_ascii_uppercase() || *b == b'_')
            .count();
        if kind_len > 0 && after[kind_len..].starts_with(']') {
            out.push_str("[REDACTED:");
            out.push_str(&after[..kind_len].to_ascii_lowercase());
            out.push(']');
            rest = &after[kind_len + 1..];
        } else {
            out.push_str(OPEN);
            rest = after;
        }
    }
    out.push_str(rest);
    out
}

fn strings(v: &Value) -> Vec<String> {
    v.as_array()
        .map(|l| l.iter().filter_map(Value::as_str).map(v3).collect())
        .unwrap_or_default()
}

/// AT-9: every fake credential of `redaction_cases.json` that the scanner
/// redacts is written redacted, and the kinds replaced are listed.
#[test]
fn at9_share_writes_fixture_credentials_redacted_with_kinds() {
    let f = fixture();
    let repo = Repo::init();
    let p = project(&f.store, "demo");
    let cases: Value = serde_json::from_str(&read_fixture("capture/redaction_cases.json"))
        .expect("redaction_cases.json");
    let mut checked = Vec::new();
    for case in cases["cases"].as_array().expect("cases") {
        let kinds: BTreeSet<String> = case["computed_result"]["kinds"]
            .as_array()
            .expect("kinds")
            .iter()
            .map(|k| k.as_str().expect("kind").to_owned())
            .collect();
        if kinds.is_empty() {
            continue;
        }
        let input = case["input"].as_str().expect("input");
        let id = add_note(&f.store, &p, input);
        checked.push((case, kinds, id));
    }
    assert!(checked.len() > 20, "the fixture has credentials to check");

    let ids: Vec<MemoryId> = checked.iter().map(|(_, _, id)| *id).collect();
    let reply = share(&f.store, &scanner(), &p, repo.root(), &params(&ids)).expect("share");
    assert!(
        reply.refused.is_empty(),
        "nothing refused: {:?}",
        reply.refused
    );
    assert_eq!(reply.written.len(), checked.len());

    let scan = scanner();
    for (case, kinds, id) in &checked {
        let name = case["name"].as_str().expect("name");
        let written = reply
            .written
            .iter()
            .find(|w| w.id == *id)
            .unwrap_or_else(|| panic!("{name}: no file"));
        let text = repo.read(&written.path);
        let listed: BTreeSet<String> = written.redactions.iter().cloned().collect();
        assert_eq!(&listed, kinds, "{name}: the kinds replaced are listed");
        assert_eq!(
            written.redactions.len(),
            listed.len(),
            "{name}: each kind is listed once"
        );
        for kind in kinds {
            assert!(
                text.contains(&format!("[REDACTED:{kind}]")),
                "{name}: placeholder for {kind} in\n{text}"
            );
        }
        let expect = &case["expect"];
        for s in strings(&expect["not_contains"]) {
            assert!(!text.contains(&s), "{name}: {s:?} is gone from\n{text}");
        }
        for s in strings(&expect["contains"]) {
            assert!(text.contains(&s), "{name}: {s:?} is kept in\n{text}");
        }
        let any = strings(&expect["contains_any"]);
        if !any.is_empty() {
            assert!(
                any.iter().any(|s| text.contains(s)),
                "{name}: one of {any:?}"
            );
        }
        assert!(scan.scan(&text).is_empty(), "{name}: nothing left to find");
        let doc = SharedDoc::parse(&text).expect("the file parses");
        assert_eq!(doc.id, *id);
        // share writes files only: the memory stays local and unredacted.
        let m = f.store.get_memory(id).expect("memory");
        assert_eq!(m.origin, Origin::Local);
        assert_eq!(m.content, case["input"].as_str().expect("input"));
    }
}

/// AT-9: a credential in a tag or an anchor cannot be redacted, so the
/// memory is refused with `secret_detected` and no file is written.
#[test]
fn at9_share_refuses_credential_in_tag_or_anchor() {
    let f = fixture();
    let repo = Repo::init();
    let p = project(&f.store, "demo");

    let mut tagged = note_input(&p, "Rotate the deploy key every quarter.");
    tagged.memory.tags = vec!["AKIAIOSFODNN7EXAMPLE".to_owned()];
    let tagged = insert(&f.store, tagged);

    let mut anchored = note_input(&p, "The CI token lives in the vault.");
    anchored.memory.anchors = vec![
        AnchorRef::new(
            AnchorKind::Service,
            "ghp_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        )
        .expect("anchor"),
    ];
    let anchored = insert(&f.store, anchored);

    let reply = share(
        &f.store,
        &scanner(),
        &p,
        repo.root(),
        &params(&[tagged, anchored]),
    )
    .expect("share");
    assert!(
        reply.written.is_empty(),
        "nothing written: {:?}",
        reply.written
    );
    assert_eq!(reply.refused.len(), 2);
    for r in &reply.refused {
        assert_eq!(r.error.code, ErrorCode::SecretDetected, "{:?}", r.error);
        assert!(
            r.error.message.contains("front matter"),
            "{}",
            r.error.message
        );
        assert!(!r.error.next_action.is_empty());
    }
    assert!(
        repo.files_under(SHARED_DIR).is_empty(),
        "no file, not even .gitattributes: {:?}",
        repo.files_under(SHARED_DIR)
    );
}

#[test]
fn s9_share_writes_the_file_at_its_path_with_its_slug() {
    let f = fixture();
    let repo = Repo::init();
    let p = project(&f.store, "demo");
    let mut input = note_input(&p, "We retry idempotent calls three times.\n\nWith jitter.");
    input.memory.memory_type = MemoryType::Decision;
    input.memory.title = Some("Retry policy: 3x, with jitter!".to_owned());
    input.memory.tags = vec!["retries".to_owned()];
    input.memory.anchors = vec![AnchorRef::new(AnchorKind::Service, "order-api").expect("anchor")];
    let id = insert(&f.store, input);

    let reply = share(&f.store, &scanner(), &p, repo.root(), &params(&[id])).expect("share");
    assert!(reply.refused.is_empty(), "{:?}", reply.refused);
    let w = &reply.written[0];
    assert_eq!(
        w.path,
        format!(
            ".spatial-memory/shared/decision/retry-policy-3x-with-jitter--{}.md",
            id.last8()
        )
    );
    assert!(w.redactions.is_empty());
    let text = repo.read(&w.path);
    let m = f.store.get_memory(&id).expect("memory");
    assert_eq!(
        text,
        SharedDoc::from_memory(&m).render(),
        "the canonical file"
    );
    let doc = SharedDoc::parse(&text).expect("parse");
    assert_eq!(doc.title, "Retry policy: 3x, with jitter!");
    assert_eq!(
        doc.body,
        "We retry idempotent calls three times.\n\nWith jitter."
    );
    assert_eq!(doc.tags, vec!["retries".to_owned()]);
    assert!(!text.contains('\r'));

    // Sharing again writes the same bytes to the same path.
    let again = share(&f.store, &scanner(), &p, repo.root(), &params(&[id])).expect("share");
    assert_eq!(again.written[0].path, w.path);
    assert_eq!(repo.read(&w.path), text);
    assert_eq!(
        repo.files_under(SHARED_DIR),
        vec![format!("{SHARED_DIR}/.gitattributes"), w.path.clone()]
    );
}

#[test]
fn s9_share_writes_gitattributes_once_and_keeps_an_existing_one() {
    let f = fixture();
    let repo = Repo::init();
    let p = project(&f.store, "demo");
    let a = add_note(&f.store, &p, "First shared memory.");
    share(&f.store, &scanner(), &p, repo.root(), &params(&[a])).expect("share");
    let attrs = format!("{SHARED_DIR}/.gitattributes");
    assert_eq!(repo.read(&attrs), GITATTRIBUTES);
    assert_eq!(GITATTRIBUTES, "* text eol=lf\n");

    repo.write(&attrs, "* text eol=lf\n*.md diff\n");
    let b = add_note(&f.store, &p, "Second shared memory.");
    share(&f.store, &scanner(), &p, repo.root(), &params(&[b])).expect("share");
    assert_eq!(
        repo.read(&attrs),
        "* text eol=lf\n*.md diff\n",
        "left alone"
    );
}

#[test]
fn s9_share_replaces_the_file_after_a_type_change() {
    let f = fixture();
    let repo = Repo::init();
    let p = project(&f.store, "demo");
    let id = add_note(&f.store, &p, "Never run migrations at peak hours.");
    let first = share(&f.store, &scanner(), &p, repo.root(), &params(&[id])).expect("share");
    let old_path = first.written[0].path.clone();
    assert!(old_path.starts_with(".spatial-memory/shared/note/"));

    f.store
        .revise_memory(
            &id,
            Revision {
                memory_type: Some(MemoryType::Pitfall),
                title: Some("Peak-hour migrations".to_owned()),
                ..Revision::default()
            },
            Access::Local,
        )
        .expect("revise");
    let second = share(&f.store, &scanner(), &p, repo.root(), &params(&[id])).expect("share");
    let new_path = second.written[0].path.clone();
    assert_eq!(
        new_path,
        format!(
            ".spatial-memory/shared/pitfall/peak-hour-migrations--{}.md",
            id.last8()
        )
    );
    assert!(!repo.exists(&old_path), "the earlier file is removed");
    let doc = SharedDoc::parse(&repo.read(&new_path)).expect("parse");
    assert_eq!(doc.memory_type, MemoryType::Pitfall);

    // A file with the same suffix that holds another id is left alone.
    let mut stranger = doc.clone();
    stranger.id = MemoryId::new_v7();
    let decoy = format!(".spatial-memory/shared/note/decoy--{}.md", id.last8());
    repo.write(&decoy, &stranger.render());
    f.store
        .revise_memory(
            &id,
            Revision {
                memory_type: Some(MemoryType::Runbook),
                ..Revision::default()
            },
            Access::Local,
        )
        .expect("revise");
    share(&f.store, &scanner(), &p, repo.root(), &params(&[id])).expect("share");
    assert!(
        repo.exists(&decoy),
        "a file holding another id is not removed"
    );
    assert!(!repo.exists(&new_path));
}

#[test]
fn s9_share_refuses_a_path_held_by_another_id() {
    let f = fixture();
    let repo = Repo::init();
    let p = project(&f.store, "demo");
    let id = add_note(&f.store, &p, "Cache keys include the tenant id.");
    let m = f.store.get_memory(&id).expect("memory");
    let mut other = SharedDoc::from_memory(&m);
    let target = other.path();
    other.id = MemoryId::new_v7();
    let foreign = other.render();
    repo.write(&target, &foreign);

    let reply = share(&f.store, &scanner(), &p, repo.root(), &params(&[id])).expect("share");
    assert!(reply.written.is_empty());
    assert_eq!(reply.refused[0].error.code, ErrorCode::InvalidInput);
    assert!(
        reply.refused[0].error.message.contains("not overwritten"),
        "{}",
        reply.refused[0].error.message
    );
    assert_eq!(repo.read(&target), foreign, "the other file is unchanged");

    // A file that does not parse is not overwritten either.
    repo.write(&target, "not a shared file\n");
    let reply = share(&f.store, &scanner(), &p, repo.root(), &params(&[id])).expect("share");
    assert_eq!(reply.refused[0].error.code, ErrorCode::InvalidInput);
    assert_eq!(repo.read(&target), "not a shared file\n");
}

#[test]
fn s9_share_refuses_an_already_shared_memory() {
    let f = fixture();
    let repo = Repo::init();
    let p = project(&f.store, "demo");
    let id = add_note(&f.store, &p, "Shared already.");
    f.store
        .record_sync(
            &id,
            &SyncMark {
                file_path: ".spatial-memory/shared/note/shared-already--00000000.md".to_owned(),
                commit_sha: Some("4f2a9c1".to_owned()),
                blob_id: "0".repeat(40),
                text_hash: "0".repeat(64),
            },
        )
        .expect("record sync");

    let reply = share(&f.store, &scanner(), &p, repo.root(), &params(&[id])).expect("share");
    assert!(reply.written.is_empty());
    assert_eq!(reply.refused[0].error.code, ErrorCode::SharedReadOnly);
    assert!(reply.refused[0].error.message.contains("already shared"));
    assert!(repo.files_under(SHARED_DIR).is_empty());
}

#[test]
fn s9_share_refuses_a_memory_of_another_project() {
    let f = fixture();
    let repo = Repo::init();
    let p = project(&f.store, "demo");
    let q = project(&f.store, "other");
    let id = add_note(&f.store, &q, "Belongs elsewhere.");
    let missing = MemoryId::new_v7();
    let reply = share(
        &f.store,
        &scanner(),
        &p,
        repo.root(),
        &params(&[id, missing]),
    )
    .expect("share");
    assert!(reply.written.is_empty());
    assert_eq!(reply.refused.len(), 2);
    for r in &reply.refused {
        assert_eq!(r.error.code, ErrorCode::NotFound);
    }
}

#[test]
fn s9_share_outside_a_repository_is_not_git_repository() {
    let f = fixture();
    let p = project(&f.store, "demo");
    let id = add_note(&f.store, &p, "Somewhere without git.");
    let plain = tempfile::tempdir().expect("temporary folder");
    let e = assert_code(
        share(&f.store, &scanner(), &p, plain.path(), &params(&[id])),
        ErrorCode::NotGitRepository,
    );
    assert!(!e.next_action.is_empty());
    assert!(!plain.path().join(".spatial-memory").exists());
}

#[test]
fn s9_share_checks_its_parameters() {
    let f = fixture();
    let repo = Repo::init();
    let p = project(&f.store, "demo");
    assert_code(
        share(&f.store, &scanner(), &p, repo.root(), &params(&[])),
        ErrorCode::InvalidInput,
    );
    let many: Vec<MemoryId> = (0..101).map(|_| MemoryId::new_v7()).collect();
    assert_code(
        share(&f.store, &scanner(), &p, repo.root(), &params(&many)),
        ErrorCode::InvalidInput,
    );
}
