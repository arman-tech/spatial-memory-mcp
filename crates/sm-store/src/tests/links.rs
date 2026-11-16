//! SPEC section 4, `links`; section 8, `link` and `related`; decision B-37:
//! links added once and removed, ends checked, staleness from the caller's
//! ancestry oracle, and the two-hop walk over links and shared anchors.

use sm_core::protocol::RELATED_MAX_HOPS;
use sm_core::time::Timestamp;
use sm_core::types::{AnchorKind, AnchorRef, Link, LinkEnd, LinkSource, Relation, Status};

use super::*;
use crate::{ANCHOR_EDGE_EXTRACTOR, WalkHit, head_only};

fn anchor(s: &str) -> AnchorRef {
    s.parse().expect("anchor")
}

fn mem(id: MemoryId) -> LinkEnd {
    LinkEnd::Memory(id)
}

fn anc(s: &str) -> LinkEnd {
    LinkEnd::Anchor(anchor(s))
}

fn link(src: LinkEnd, dst: LinkEnd) -> Link {
    Link {
        src,
        dst,
        relation: Relation::RelatesTo,
        source: LinkSource::Manual,
        extractor: None,
        verified_commit: None,
        verified_at: None,
    }
}

fn anchor_edge(memory: MemoryId, a: &str) -> Link {
    Link {
        source: LinkSource::Extracted,
        extractor: Some(ANCHOR_EDGE_EXTRACTOR.to_owned()),
        ..link(mem(memory), anc(a))
    }
}

fn with_anchors(store: &Store, project: ProjectId, content: &str, anchors: &[&str]) -> MemoryId {
    let mut input = note(project, content);
    input.memory.anchors = anchors.iter().map(|a| anchor(a)).collect();
    store
        .insert_memory(input, Access::Local)
        .expect("insert")
        .id
}

fn hops(hits: &[WalkHit]) -> Vec<(MemoryId, u8)> {
    hits.iter().map(|h| (h.memory_id, h.hops)).collect()
}

/// `(id, hops)` sorted the way the walk reports them: hops, then id.
fn expected(mut v: Vec<(MemoryId, u8)>) -> Vec<(MemoryId, u8)> {
    v.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));
    v
}

#[test]
fn s4_links_are_added_once_and_removed() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let q = project(&fx.store, "beta");
    let m1 = add(&fx.store, p, "the caller");
    let m2 = add(&fx.store, p, "the callee");
    let verified = Link {
        relation: Relation::Calls,
        verified_commit: Some("abc1234".to_owned()),
        verified_at: Some(Timestamp::from_unix_seconds(1_700_000_000)),
        ..link(mem(m1), mem(m2))
    };
    let added = fx.store.add_link(p, verified.clone()).expect("add");
    assert!(added.created);
    assert_eq!(added.link, verified);

    // The same ends and relation: the stored link comes back, unchanged.
    let again = fx
        .store
        .add_link(
            p,
            Link {
                source: LinkSource::Extracted,
                extractor: Some("tree-sitter".to_owned()),
                verified_commit: None,
                verified_at: None,
                ..verified.clone()
            },
        )
        .expect("add again");
    assert!(!again.created);
    assert_eq!(again.link, verified);
    assert_eq!(table_count(&fx.store, "links"), 1);

    // Another relation, or the other direction, is another link.
    assert!(
        fx.store
            .add_link(p, link(mem(m1), mem(m2)))
            .expect("add")
            .created
    );
    assert!(
        fx.store
            .add_link(
                p,
                Link {
                    relation: Relation::Calls,
                    ..link(mem(m2), mem(m1))
                }
            )
            .expect("add")
            .created
    );
    let anchors = link(anc("service:billing"), anc("table:orders"));
    assert!(fx.store.add_link(p, anchors.clone()).expect("add").created);
    // The same anchor link in another project is that project's link.
    assert!(fx.store.add_link(q, anchors.clone()).expect("add").created);
    assert_eq!(table_count(&fx.store, "links"), 5);

    let views = fx
        .store
        .links_of(p, &mem(m1), head_only(None))
        .expect("links");
    let got: Vec<Link> = views.iter().map(|v| v.link.clone()).collect();
    assert_eq!(
        got,
        vec![
            verified.clone(),
            link(mem(m1), mem(m2)),
            Link {
                relation: Relation::Calls,
                ..link(mem(m2), mem(m1))
            },
        ],
        "both directions, oldest first"
    );
    assert!(views.iter().all(|v| v.stale.is_none()));
    let at_orders = fx
        .store
        .links_of(p, &anc("table:orders"), head_only(None))
        .expect("links");
    assert_eq!(at_orders.len(), 1);
    assert_eq!(at_orders[0].link, anchors);

    assert_eq!(
        fx.store
            .remove_link(p, &mem(m1), &mem(m2), Relation::Calls)
            .expect("remove"),
        1
    );
    assert_eq!(
        fx.store
            .remove_link(p, &mem(m1), &mem(m2), Relation::Calls)
            .expect("remove"),
        0
    );
    assert_eq!(
        fx.store
            .remove_link(q, &mem(m1), &mem(m2), Relation::RelatesTo)
            .expect("remove"),
        0
    );
    assert_eq!(
        fx.store
            .remove_link(
                q,
                &anc("service:billing"),
                &anc("table:orders"),
                Relation::RelatesTo
            )
            .expect("remove"),
        1
    );
    assert_eq!(table_count(&fx.store, "links"), 3);
}

#[test]
fn s4_link_ends_are_checked() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let q = project(&fx.store, "beta");
    let a = add(&fx.store, p, "in alpha");
    let b = add(&fx.store, q, "in beta");
    let blank_anchor = LinkEnd::Anchor(AnchorRef {
        kind: AnchorKind::Table,
        name: "  ".to_owned(),
    });
    let cases = [
        (
            "same memory",
            p,
            link(mem(a), mem(a)),
            ErrorCode::InvalidInput,
        ),
        (
            "same anchor",
            p,
            link(anc("table:orders"), anc("table:orders")),
            ErrorCode::InvalidInput,
        ),
        (
            "empty commit",
            p,
            Link {
                verified_commit: Some(" ".to_owned()),
                ..link(mem(a), anc("table:orders"))
            },
            ErrorCode::InvalidInput,
        ),
        (
            "invalid anchor",
            p,
            link(mem(a), blank_anchor),
            ErrorCode::InvalidInput,
        ),
        (
            "another project's memory",
            p,
            link(mem(a), mem(b)),
            ErrorCode::InvalidInput,
        ),
        (
            "missing memory",
            p,
            link(anc("table:orders"), mem(MemoryId::new_v7())),
            ErrorCode::NotFound,
        ),
        (
            "missing project",
            ProjectId(999),
            link(anc("service:billing"), anc("table:orders")),
            ErrorCode::NotFound,
        ),
    ];
    for (what, project, l, code) in cases {
        match fx.store.add_link(project, l) {
            Ok(r) => panic!("{what}: accepted {r:?}"),
            Err(e) => assert_eq!(e.code, code, "{what}: {e:?}"),
        }
    }
    assert_eq!(table_count(&fx.store, "links"), 0);
    assert_code(
        fx.store
            .links_of(p, &mem(MemoryId::new_v7()), head_only(None)),
        ErrorCode::NotFound,
    );
    // An anchor nobody linked has no links.
    assert!(
        fx.store
            .links_of(p, &anc("file:none.rs"), head_only(None))
            .expect("links")
            .is_empty()
    );
}

#[test]
fn s4_head_only_settles_only_the_head() {
    // Decision B-37.
    let head = "0123456789abcdef0123456789abcdef01234567";
    let at = head_only(Some(head));
    assert_eq!(at(head), Some(true));
    assert_eq!(at(&head.to_uppercase()), Some(true));
    assert_eq!(at(&format!("  {head}\n")), Some(true));
    assert_eq!(at("0123456"), Some(true));
    assert_eq!(at("0123456789AB"), Some(true));
    assert_eq!(at("012345"), None, "too short for a prefix");
    assert_eq!(at("1123456"), None, "another commit");
    assert_eq!(at("0123456z"), None, "not hexadecimal");
    assert_eq!(at(&format!("{head}0")), None, "longer than the head");
    assert_eq!(at("0123456é"), None);
    assert_eq!(at(""), None);
    let odd = head_only(Some("ééééééé"));
    assert_eq!(odd("abcdefa"), None, "no panic on a non-ASCII head");
    let unknown = head_only(None);
    assert_eq!(unknown(head), None);
}

#[test]
fn s4_links_of_reports_staleness_from_the_oracle() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let m = add(&fx.store, p, "verified at several commits");
    let commits = [Some("aaaa111"), Some("bbbb222"), Some("cccc333"), None];
    let targets = ["file:a.rs", "file:b.rs", "file:c.rs", "file:d.rs"];
    for (commit, target) in commits.iter().zip(targets) {
        fx.store
            .add_link(
                p,
                Link {
                    verified_commit: commit.map(str::to_owned),
                    ..link(mem(m), anc(target))
                },
            )
            .expect("add");
    }
    let oracle = |c: &str| match c {
        "aaaa111" => Some(true),
        "bbbb222" => Some(false),
        _ => None,
    };
    let stale: Vec<Option<bool>> = fx
        .store
        .links_of(p, &mem(m), oracle)
        .expect("links")
        .iter()
        .map(|v| v.stale)
        .collect();
    assert_eq!(stale, vec![Some(false), Some(true), None, None]);

    let head = "aaaa111ffff0000aaaa111ffff0000aaaa111fff";
    let stale: Vec<Option<bool>> = fx
        .store
        .links_of(p, &mem(m), head_only(Some(head)))
        .expect("links")
        .iter()
        .map(|v| v.stale)
        .collect();
    assert_eq!(stale, vec![Some(false), None, None, None]);
}

#[test]
fn s4_related_walks_links_and_anchors_two_hops() {
    assert_eq!(RELATED_MAX_HOPS, 2);
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let q = project(&fx.store, "beta");
    let x = "file:src/x.rs";
    let billing = "service:billing";

    let s = with_anchors(&fx.store, p, "the start", &[x]);
    let a = add(&fx.store, p, "linked from the start");
    let b = add(&fx.store, p, "two links away");
    let c = add(&fx.store, p, "three links away");
    let d = with_anchors(&fx.store, p, "shares the start's anchor", &[x]);
    let e = add(&fx.store, p, "links to the anchor-sharing memory");
    let mut retracted = note(p, "a retracted memory on the way");
    retracted.memory.status = Status::Retracted;
    let f = fx
        .store
        .insert_memory(retracted, Access::Local)
        .expect("insert")
        .id;
    let g = add(&fx.store, p, "beyond the retracted memory");
    let i = add(&fx.store, p, "reached two ways");
    let j = with_anchors(&fx.store, p, "on a linked anchor", &[billing]);
    let other = with_anchors(&fx.store, q, "same anchor, other project", &[x]);

    for l in [
        link(mem(s), mem(a)),
        link(mem(a), mem(b)),
        link(mem(b), mem(c)),
        link(mem(e), mem(d)),
        link(mem(s), mem(f)),
        link(mem(f), mem(g)),
        link(mem(s), mem(i)),
        link(mem(a), mem(i)),
        link(anc(x), anc(billing)),
    ] {
        fx.store.add_link(p, l).expect("link");
    }

    let hits = fx.store.related_to_memory(&s, &[]).expect("related");
    assert_eq!(
        hops(&hits),
        expected(vec![(a, 1), (d, 1), (i, 1), (b, 2), (e, 2), (g, 2), (j, 2)]),
        "c is three hops away, f is retracted, the start and the other project are left out"
    );
    let via = |id: MemoryId| -> Vec<Link> {
        hits.iter()
            .find(|h| h.memory_id == id)
            .map(|h| h.via.clone())
            .expect("hit")
    };
    assert_eq!(via(a), vec![link(mem(s), mem(a))]);
    assert_eq!(via(b), vec![link(mem(s), mem(a)), link(mem(a), mem(b))]);
    assert_eq!(via(d), vec![anchor_edge(s, x), anchor_edge(d, x)]);
    assert_eq!(
        via(e),
        vec![anchor_edge(s, x), anchor_edge(d, x), link(mem(e), mem(d))]
    );
    assert_eq!(via(g), vec![link(mem(s), mem(f)), link(mem(f), mem(g))]);
    assert_eq!(
        via(j),
        vec![
            anchor_edge(s, x),
            link(anc(x), anc(billing)),
            anchor_edge(j, billing)
        ]
    );

    // Statuses choose what is reported, not what is walked.
    let with_retracted = fx
        .store
        .related_to_memory(&s, &[Status::Active, Status::Retracted])
        .expect("related");
    assert!(hops(&with_retracted).contains(&(f, 1)));
    assert_eq!(with_retracted.len(), hits.len() + 1);
    assert_eq!(
        hops(
            &fx.store
                .related_to_memory(&s, &[Status::Retracted])
                .expect("related")
        ),
        vec![(f, 1)]
    );
    assert_code(
        fx.store.related_to_memory(&MemoryId::new_v7(), &[]),
        ErrorCode::NotFound,
    );

    // From an anchor: its memories are one hop away.
    let from_x = fx
        .store
        .related_to_anchor(p, &anchor(x), &[])
        .expect("related");
    assert_eq!(
        hops(&from_x),
        expected(vec![(s, 1), (d, 1), (a, 2), (e, 2), (i, 2), (j, 2)])
    );
    assert_eq!(
        from_x
            .iter()
            .find(|h| h.memory_id == j)
            .map(|h| h.via.clone()),
        Some(vec![link(anc(x), anc(billing)), anchor_edge(j, billing)])
    );
    assert_eq!(
        hops(
            &fx.store
                .related_to_anchor(q, &anchor(x), &[])
                .expect("related")
        ),
        vec![(other, 1)]
    );
    assert!(
        fx.store
            .related_to_anchor(p, &anchor("table:nothing"), &[])
            .expect("related")
            .is_empty()
    );
    assert_code(
        fx.store.related_to_anchor(ProjectId(999), &anchor(x), &[]),
        ErrorCode::NotFound,
    );

    // Forgetting a memory removes its links from the walk.
    fx.store.forget(&[a], Access::Local).expect("forget");
    let after = hops(&fx.store.related_to_memory(&s, &[]).expect("related"));
    assert!(!after.iter().any(|(id, _)| *id == a || *id == b));
    assert!(after.contains(&(i, 1)));
}
