//! D-3: FTS5 keyword search finds identifiers containing `.`, `::`, `_` and
//! `-`, and nothing typed into a search is read as query syntax.

use sm_core::traits::{MemoryFilter, QueryScope};
use sm_core::types::Status;

use super::*;
use crate::Revision;

fn kw(store: &Store, text: &str) -> Vec<MemoryId> {
    kw_with(store, text, QueryScope::All, &MemoryFilter::default(), 10)
}

fn kw_with(
    store: &Store,
    text: &str,
    scope: QueryScope,
    filter: &MemoryFilter,
    limit: usize,
) -> Vec<MemoryId> {
    store
        .keyword_search(text, scope, filter, limit)
        .expect("keyword search")
        .into_iter()
        .map(|h| h.memory_id)
        .collect()
}

#[test]
fn d3_identifiers_with_punctuation_are_found() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let m1 = add(&fx.store, p, "Use std::collections::HashMap for the cache");
    let m2 = add(&fx.store, p, "Set config.max_retries to 5");
    let m3 = add(&fx.store, p, "The user_profile_id column is indexed");
    let m4 = add(&fx.store, p, "Send x-request-id on every call");
    // The same words in other orders: a bag-of-words search would find it.
    let decoy = add(
        &fx.store,
        p,
        "HashMap collections std; retries max config; id profile user; id request x",
    );

    assert_eq!(kw(&fx.store, "std::collections::HashMap"), vec![m1]);
    assert_eq!(kw(&fx.store, "config.max_retries"), vec![m2]);
    assert_eq!(kw(&fx.store, "user_profile_id"), vec![m3]);
    assert_eq!(kw(&fx.store, "x-request-id"), vec![m4]);
    // Case does not matter.
    assert_eq!(kw(&fx.store, "STD::Collections::hashmap"), vec![m1]);
    // The words in the decoy's order find the decoy only.
    assert_eq!(kw(&fx.store, "HashMap::collections::std"), vec![decoy]);
    // Part of an identifier still finds it.
    assert_eq!(kw(&fx.store, "max_retries"), vec![m2]);

    // Project scope and the hit's chunk.
    let hits = fx
        .store
        .keyword_search(
            "x-request-id",
            QueryScope::Project(p),
            &MemoryFilter::default(),
            10,
        )
        .expect("search");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].memory_id, m4);
    assert_eq!(hits[0].chunk_seq, 0);
    assert!(hits[0].rank.is_finite());
}

#[test]
fn d3_titles_are_searched_and_follow_revisions() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let mut input = note(p, "the body says nothing about it");
    input.memory.title = Some("Billing ledger reconciliation".to_owned());
    let id = fx
        .store
        .insert_memory(input, Access::Local)
        .expect("insert")
        .id;
    assert_eq!(kw(&fx.store, "ledger reconciliation"), vec![id]);

    fx.store
        .revise_memory(
            &id,
            Revision {
                title: Some("Payments overview".to_owned()),
                ..Revision::default()
            },
            Access::Local,
        )
        .expect("revise the title");
    assert!(kw(&fx.store, "ledger reconciliation").is_empty());
    assert_eq!(kw(&fx.store, "payments overview"), vec![id]);
    assert_eq!(kw(&fx.store, "body says nothing"), vec![id]);
}

#[test]
fn d3_pending_chunks_are_keyword_searched() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let text = "the zebra crossing waits for an embedding";
    let id = fx
        .store
        .insert_memory(
            MemoryInput {
                memory: NewMemory::new(p, text, Provenance::new(ProvenanceSource::Agent)),
                chunks: vec![NewChunk::pending(piece(0, text))],
            },
            Access::Local,
        )
        .expect("insert")
        .id;
    assert_eq!(kw(&fx.store, "zebra crossing"), vec![id]);
}

#[test]
fn d3_status_filter_and_one_hit_per_memory() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let q = project(&fx.store, "beta");
    let mut retracted = note(p, "walrus retracted advice");
    retracted.memory.status = Status::Retracted;
    let retracted = fx
        .store
        .insert_memory(retracted, Access::Local)
        .expect("insert")
        .id;
    let many = fx
        .store
        .insert_memory(
            note_in_parts(p, &["walrus one", "walrus two", "walrus three"]),
            Access::Local,
        )
        .expect("insert")
        .id;
    let other = add(&fx.store, q, "a walrus in another project");

    let found = kw(&fx.store, "walrus");
    assert_eq!(found.len(), 2, "one hit per memory, retracted hidden");
    assert!(found.contains(&many) && found.contains(&other));

    let only_retracted = MemoryFilter {
        statuses: vec![Status::Retracted],
        ..MemoryFilter::default()
    };
    assert_eq!(
        kw_with(&fx.store, "walrus", QueryScope::All, &only_retracted, 10),
        vec![retracted]
    );
    let every_status = MemoryFilter {
        statuses: vec![Status::Active, Status::Superseded, Status::Retracted],
        ..MemoryFilter::default()
    };
    assert_eq!(
        kw_with(&fx.store, "walrus", QueryScope::All, &every_status, 10).len(),
        3
    );

    assert_eq!(
        kw_with(
            &fx.store,
            "walrus",
            QueryScope::Project(p),
            &MemoryFilter::default(),
            10
        ),
        vec![many]
    );
    let exclude = MemoryFilter {
        exclude: vec![many],
        ..MemoryFilter::default()
    };
    assert_eq!(
        kw_with(&fx.store, "walrus", QueryScope::All, &exclude, 10),
        vec![other]
    );
    assert_eq!(
        kw_with(
            &fx.store,
            "walrus",
            QueryScope::All,
            &MemoryFilter::default(),
            1
        )
        .len(),
        1
    );
    assert!(
        kw_with(
            &fx.store,
            "walrus",
            QueryScope::All,
            &MemoryFilter::default(),
            0
        )
        .is_empty()
    );
}

#[test]
fn d3_query_syntax_is_searched_as_text() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let id = add(&fx.store, p, "foo and bar near the col val");
    for q in [
        "\"",
        "\"\"",
        "foo\" OR \"bar",
        "foo OR bar",
        "NEAR(foo bar)",
        "*",
        "foo*",
        "::",
        "-",
        "col:val",
        "title:foo",
        "^foo",
        "AND",
        "(",
        "{text}: foo",
        "   ",
        "",
    ] {
        let r = fx
            .store
            .keyword_search(q, QueryScope::All, &MemoryFilter::default(), 10);
        assert!(r.is_ok(), "query {q:?} failed: {r:?}");
    }
    // Operators are words: `foo OR bar` is a phrase, which this text lacks.
    assert!(kw(&fx.store, "foo OR bar").is_empty());
    assert!(kw(&fx.store, "NEAR(foo bar)").is_empty());
    // `col:val` is the phrase "col val", not a column filter.
    assert_eq!(kw(&fx.store, "col:val"), vec![id]);
    assert!(kw(&fx.store, "   ").is_empty());
}
