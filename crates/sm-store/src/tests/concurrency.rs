//! AT-5: two reader threads and one writer thread against one database file
//! for 60 seconds; no lock error reaches any of them. The test runs its full
//! 60 seconds.
//!
//! Each thread has its own `Store` on the file, as separate processes would.
//! The writer inserts, revises, forgets and records injections without
//! pause; the readers run vector search in both scopes, keyword search,
//! listing and point reads. Any error at all fails the test, not only a lock
//! error, so every one is collected and reported rather than panicking
//! mid-run.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use sm_core::time::Timestamp;
use sm_core::traits::{MemoryFilter, QueryScope, VectorQuery};
use sm_core::types::Status;

use super::*;
use crate::{ContentChange, MemoryListFilter, Revision};

const RUN: Duration = Duration::from_secs(60);
/// Writer memories kept alive; older ones are forgotten.
const LIVE: usize = 200;
/// Memories seeded before the run and never changed by the writer.
const STABLE: usize = 40;

/// What one thread did.
#[derive(Debug, Default)]
struct Tally {
    ops: u64,
    errors: Vec<String>,
}

impl Tally {
    fn check<T>(&mut self, what: &str, r: sm_core::error::Result<T>) -> Option<T> {
        self.ops += 1;
        match r {
            Ok(v) => Some(v),
            Err(e) => {
                self.errors
                    .push(format!("{what}: {:?}: {}", e.code, e.message));
                None
            }
        }
    }

    fn fail(&mut self, what: String) {
        self.errors.push(what);
    }
}

/// A one-chunk note embedded with a random unit vector.
fn random_note(rng: &mut Rng, project: ProjectId, content: &str) -> MemoryInput {
    MemoryInput {
        chunks: vec![NewChunk::embedded(piece(0, content), rng.unit_vector())],
        ..note(project, content)
    }
}

fn writer(store: &Store, projects: [ProjectId; 2], start: Instant) -> Tally {
    let mut t = Tally::default();
    let mut rng = Rng::new(1);
    let mut live: VecDeque<MemoryId> = VecDeque::new();
    let mut i: u64 = 0;
    while start.elapsed() < RUN {
        i += 1;
        let p = projects[usize::from(i.is_multiple_of(2))];
        let content = format!("writer memory {i} about orders and billing");
        if let Some(ins) = t.check(
            "insert",
            store.insert_memory(random_note(&mut rng, p, &content), Access::Local),
        ) {
            live.push_back(ins.id);
        }
        if let Some(&target) = live.get(rng.below(live.len().max(1))) {
            let rev = if i.is_multiple_of(3) {
                let text = format!("revised memory {i} about orders and invoices");
                Revision {
                    content: Some(ContentChange {
                        chunks: vec![NewChunk::embedded(piece(0, &text), rng.unit_vector())],
                        content: text,
                    }),
                    ..Revision::default()
                }
            } else {
                Revision {
                    tags: Some(vec![format!("round-{}", i % 7)]),
                    status: Some(if i.is_multiple_of(5) {
                        Status::Retracted
                    } else {
                        Status::Active
                    }),
                    ..Revision::default()
                }
            };
            t.check("revise", store.revise_memory(&target, rev, Access::Local));
        }
        if live.len() > LIVE
            && let Some(old) = live.pop_front()
        {
            t.check("forget", store.forget(&[old], Access::Local));
        }
        let shown: Vec<MemoryId> = live.iter().rev().take(3).copied().collect();
        t.check(
            "record injections",
            store.record_injections(&format!("session-{}", i % 4), &shown, Timestamp::now()),
        );
        if i.is_multiple_of(100) {
            t.check("prune injections", store.prune_injections(Timestamp::now()));
        }
    }
    t
}

fn reader(
    store: &Store,
    seed: u64,
    projects: [ProjectId; 2],
    stable: &[(MemoryId, String)],
    start: Instant,
) -> Tally {
    let mut t = Tally::default();
    let mut rng = Rng::new(seed);
    while start.elapsed() < RUN {
        let p = projects[rng.below(2)];
        for scope in [QueryScope::Project(p), QueryScope::All] {
            let q = rng.unit_vector();
            if let Some(hits) = t.check(
                "vector search",
                store.nearest(&VectorQuery {
                    vector: &q,
                    scope,
                    filter: MemoryFilter::default(),
                    limit: 10,
                }),
            ) && hits.len() != 10
            {
                t.fail(format!(
                    "vector search returned {} hits, not 10",
                    hits.len()
                ));
            }
        }
        if let Some(hits) = t.check(
            "keyword search",
            store.keyword_search(
                "orders",
                QueryScope::Project(p),
                &MemoryFilter::default(),
                10,
            ),
        ) && hits.is_empty()
        {
            t.fail("keyword search found no stable memory".to_owned());
        }
        t.check(
            "keyword search",
            store.keyword_search("billing", QueryScope::All, &MemoryFilter::default(), 10),
        );
        let listed = t.check(
            "list",
            store.list_memories(&MemoryListFilter {
                project: Some(p),
                limit: 20,
                ..MemoryListFilter::default()
            }),
        );
        // A writer memory may be forgotten between two reads: `find` says
        // so with `None`, never with an error.
        if let Some(m) = listed.as_ref().and_then(|l| l.last()) {
            t.check("find", store.find_memory(&m.id));
        }
        let (id, content) = &stable[rng.below(stable.len())];
        if let Some(m) = t.check("get", store.get_memory(id))
            && &m.content != content
        {
            t.fail(format!("stable memory {id} changed"));
        }
        t.check("memory vector", store.memory_vector(id));
    }
    t
}

#[test]
fn at5_two_readers_and_a_writer_for_sixty_seconds() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let q = project(&fx.store, "beta");
    let mut rng = Rng::new(99);
    let mut stable = Vec::new();
    for n in 0..STABLE {
        let project = if n.is_multiple_of(2) { p } else { q };
        let content = format!("stable memory {n} about orders");
        let id = fx
            .store
            .insert_memory(random_note(&mut rng, project, &content), Access::Local)
            .expect("seed")
            .id;
        stable.push((id, content));
    }

    let projects = [p, q];
    let stores = [fx.reopen(stamp()), fx.reopen(stamp()), fx.reopen(stamp())];
    let start = Instant::now();
    let [w, r1, r2] = std::thread::scope(|s| {
        let [ws, rs1, rs2] = &stores;
        let stable = &stable;
        let w = s.spawn(move || writer(ws, projects, start));
        let r1 = s.spawn(move || reader(rs1, 2, projects, stable, start));
        let r2 = s.spawn(move || reader(rs2, 3, projects, stable, start));
        [w, r1, r2].map(|h| h.join().expect("thread"))
    });
    let elapsed = start.elapsed();

    eprintln!(
        "AT-5: {:.1} s; writer {} operations, readers {} and {} operations",
        elapsed.as_secs_f64(),
        w.ops,
        r1.ops,
        r2.ops
    );
    assert!(
        elapsed >= RUN,
        "the test ran {elapsed:?}, not its full 60 s"
    );
    for (name, t) in [("writer", &w), ("reader 1", &r1), ("reader 2", &r2)] {
        assert!(t.ops > 0, "{name} did nothing");
        assert!(
            t.errors.is_empty(),
            "{name}: {} errors, first: {:?}",
            t.errors.len(),
            t.errors.iter().take(5).collect::<Vec<_>>()
        );
    }

    // The file is whole and the writer's bookkeeping holds.
    drop(stores);
    let check: String = fx
        .store
        .read("checking the database", |tx| {
            tx.query_row("PRAGMA integrity_check", [], |r| r.get(0))
                .ctx("checking the database")
        })
        .expect("integrity check");
    assert_eq!(check, "ok");
    let count = fx.store.memory_count().expect("count");
    assert!(
        count > u64::try_from(STABLE).expect("count")
            && count <= u64::try_from(STABLE + LIVE).expect("count"),
        "{count} memories"
    );
    for (id, content) in &stable {
        assert_eq!(&fx.store.get_memory(id).expect("stable").content, content);
    }
}

/// `PRAGMA mmap_size` on the store's connections: readers through
/// [`Store::read`], the writer through [`Store::write`].
fn mmap_sizes(store: &Store) -> (i64, i64) {
    let pragma = |tx: &rusqlite::Transaction<'_>| {
        tx.query_row("PRAGMA mmap_size", [], |r| r.get::<_, i64>(0))
            .ctx("reading mmap_size")
    };
    let reader = store.read("reading mmap_size", pragma).expect("reader");
    let writer = store.write("reading mmap_size", pragma).expect("writer");
    (reader, writer)
}

#[test]
fn s4_readers_map_the_database_and_the_writer_does_not() {
    // Decision B-75: project scope reads every chunk row of the project, so
    // readers read the file through a memory map, up to SQLite's ceiling.
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let id = add(&fx.store, p, "the orders table is partitioned by month");
    assert_eq!(
        mmap_sizes(&fx.store),
        (crate::conn::READER_MMAP_BYTES, 0),
        "readers map the file; the writer does not"
    );
    // A pooled reader keeps the setting, and reads what the writer wrote.
    let hits = fx
        .store
        .nearest(&VectorQuery {
            vector: &fake_vector("the orders table is partitioned by month"),
            scope: QueryScope::Project(p),
            filter: MemoryFilter::default(),
            limit: 1,
        })
        .expect("search");
    assert_eq!(hits[0].memory_id, id);
    assert_eq!(mmap_sizes(&fx.store).0, crate::conn::READER_MMAP_BYTES);
}
