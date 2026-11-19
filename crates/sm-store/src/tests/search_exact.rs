//! D-2: exact vector search returns exactly what a brute-force scan of every
//! chunk returns, in both scopes, with the default filter and with every
//! other filter, ties included.
//!
//! The vectors are [`ExactVector`]s, so every score is exact in `f32` and the
//! reference computes it in integers: the comparison is equality, not a
//! tolerance.

use sm_core::ids::ChunkId;
use sm_core::protocol::RememberOutcome;
use sm_core::traits::{MemoryFilter, MemoryHit, QueryScope, VectorIndex, VectorQuery};
use sm_core::types::{AnchorRef, MemoryType, Status};

use super::*;

/// Memories in the corpus (plus one twin); two thirds go to the first
/// project, so it alone holds more than 1,000 embedded chunks.
const MEMORIES: usize = 1200;

struct CorpusMemory {
    id: MemoryId,
    project: ProjectId,
    status: Status,
    memory_type: MemoryType,
    importance: f64,
    anchors: Vec<AnchorRef>,
    /// Each chunk's vector, by seq; `None` while it waits for an embedding.
    chunks: Vec<Option<ExactVector>>,
}

struct Corpus {
    projects: Vec<ProjectId>,
    memories: Vec<CorpusMemory>,
    /// The vector stored twice in the twin memory (seqs 1 and 2).
    twin_vector: ExactVector,
    twin: MemoryId,
    /// Vectors stored in more than one chunk.
    reused: Vec<ExactVector>,
}

fn anchor(s: &str) -> AnchorRef {
    s.parse().expect("anchor")
}

fn build(store: &Store, rng: &mut Rng) -> Corpus {
    let projects = vec![
        project(store, "alpha"),
        project(store, "beta"),
        project(store, "gamma"),
    ];
    let anchor_pool = [
        anchor("file:src/lib.rs"),
        anchor("symbol:Store::open"),
        anchor("service:billing"),
    ];
    let importances = [0.1, 0.3, 0.5, 0.7, 0.9];
    let mut pool: Vec<ExactVector> = Vec::new();
    let mut reused: Vec<ExactVector> = Vec::new();
    let mut plans: Vec<CorpusMemory> = Vec::new();
    let mut inputs: Vec<MemoryInput> = Vec::new();

    for i in 0..MEMORIES {
        let project = if rng.below(3) < 2 {
            projects[0]
        } else {
            projects[1 + rng.below(2)]
        };
        let status = match rng.below(10) {
            0 => Status::Superseded,
            1 => Status::Retracted,
            // Unjudged captures: stored, never served (B-157).
            2 => Status::Pending,
            _ => Status::Active,
        };
        let memory_type = if rng.below(40) == 0 {
            MemoryType::Runbook
        } else {
            MemoryType::Note
        };
        let importance = importances[rng.below(importances.len())];
        let anchors: Vec<AnchorRef> = anchor_pool
            .iter()
            .filter(|_| rng.below(6) == 0)
            .cloned()
            .collect();
        let n_chunks = 1 + rng.below(3);
        let content = format!("exact memory {i}");
        let mut chunks = Vec::new();
        let mut new_chunks = Vec::new();
        for seq in 0..n_chunks {
            let seq32 = u32::try_from(seq).expect("seq");
            let p = piece(seq32, &format!("exact memory {i} chunk {seq}"));
            if rng.below(20) == 0 {
                new_chunks.push(NewChunk::pending(p));
                chunks.push(None);
                continue;
            }
            let v = if !pool.is_empty() && rng.below(8) == 0 {
                let v = pool[rng.below(pool.len())].clone();
                reused.push(v.clone());
                v
            } else {
                let v = exact_unit(rng);
                pool.push(v.clone());
                v
            };
            new_chunks.push(NewChunk::embedded(p, v.vector.clone()));
            chunks.push(Some(v));
        }
        let mut memory = NewMemory::new(project, content, Provenance::new(ProvenanceSource::Agent));
        memory.status = status;
        memory.memory_type = memory_type;
        memory.importance = importance;
        memory.anchors = anchors.clone();
        inputs.push(MemoryInput {
            memory,
            chunks: new_chunks,
        });
        plans.push(CorpusMemory {
            id: MemoryId::new_v7(),
            project,
            status,
            memory_type,
            importance,
            anchors,
            chunks,
        });
    }

    // The twin: its best chunk is stored twice, at seqs 1 and 2, so the
    // collapse must keep the lower seq.
    let other = exact_unit(rng);
    let twin_vector = exact_unit(rng);
    let twin_chunks = vec![
        Some(other.clone()),
        Some(twin_vector.clone()),
        Some(twin_vector.clone()),
    ];
    let twin_new: Vec<NewChunk> = twin_chunks
        .iter()
        .enumerate()
        .map(|(seq, v)| {
            let seq32 = u32::try_from(seq).expect("seq");
            NewChunk::embedded(
                piece(seq32, &format!("exact twin chunk {seq}")),
                v.as_ref().expect("embedded").vector.clone(),
            )
        })
        .collect();
    inputs.push(MemoryInput {
        memory: NewMemory::new(
            projects[0],
            "exact twin",
            Provenance::new(ProvenanceSource::Agent),
        ),
        chunks: twin_new,
    });
    plans.push(CorpusMemory {
        id: MemoryId::new_v7(),
        project: projects[0],
        status: Status::Active,
        memory_type: MemoryType::Note,
        importance: sm_core::types::DEFAULT_IMPORTANCE,
        anchors: Vec::new(),
        chunks: twin_chunks,
    });

    let inserted = store
        .insert_memories(inputs, Access::Local)
        .expect("insert the corpus");
    assert_eq!(inserted.len(), plans.len());
    for (plan, ins) in plans.iter_mut().zip(&inserted) {
        assert_eq!(ins.outcome, RememberOutcome::Created);
        assert_eq!(ins.chunks as usize, plan.chunks.len());
        plan.id = ins.id;
    }
    let twin = plans.last().expect("twin").id;
    Corpus {
        projects,
        memories: plans,
        twin_vector,
        twin,
        reused,
    }
}

/// What exact search must return: each matching memory's best chunk (the
/// highest score, then the lowest seq), ordered by score, then memory id.
fn brute_force(
    corpus: &Corpus,
    dots: &[Vec<Option<i64>>],
    scope: QueryScope,
    f: &MemoryFilter,
    limit: usize,
) -> Vec<(MemoryId, u32, i64)> {
    // A pending memory is never served, whatever the filter asks for.
    let statuses = f.served_statuses();
    let mut best: Vec<(MemoryId, u32, i64)> = Vec::new();
    for (m, ds) in corpus.memories.iter().zip(dots) {
        if let QueryScope::Project(p) = scope
            && m.project != p
        {
            continue;
        }
        if !statuses.contains(&m.status) {
            continue;
        }
        if !f.types.is_empty() && !f.types.contains(&m.memory_type) {
            continue;
        }
        if let Some(min) = f.min_importance
            && m.importance < min
        {
            continue;
        }
        if let Some(a) = &f.anchor
            && !m.anchors.contains(a)
        {
            continue;
        }
        if f.exclude.contains(&m.id) {
            continue;
        }
        let mut top: Option<(u32, i64)> = None;
        for (seq, d) in ds.iter().enumerate() {
            let Some(k) = *d else { continue };
            if top.is_none_or(|(_, b)| k > b) {
                top = Some((u32::try_from(seq).expect("seq"), k));
            }
        }
        if let Some((seq, k)) = top {
            best.push((m.id, seq, k));
        }
    }
    best.sort_by(|a, b| b.2.cmp(&a.2).then(a.0.cmp(&b.0)));
    best.truncate(limit);
    best
}

fn as_scored(v: &[(MemoryId, u32, i64)]) -> Vec<(MemoryId, u32, f32)> {
    v.iter()
        .map(|&(id, seq, k)| (id, seq, exact_score(k)))
        .collect()
}

fn as_tuples(hits: &[MemoryHit]) -> Vec<(MemoryId, u32, f32)> {
    hits.iter()
        .map(|h| (h.memory_id, h.chunk_seq, h.score))
        .collect()
}

fn search(
    store: &Store,
    q: &[f32],
    scope: QueryScope,
    f: &MemoryFilter,
    limit: usize,
) -> Vec<MemoryHit> {
    store
        .nearest(&VectorQuery {
            vector: q,
            scope,
            filter: f.clone(),
            limit,
        })
        .expect("vector search")
}

/// Through the trait the daemon uses.
fn via_trait<I: VectorIndex>(index: &I, q: &[f32], scope: QueryScope) -> Vec<MemoryHit> {
    index
        .nearest_memories(&VectorQuery {
            vector: q,
            scope,
            filter: MemoryFilter::default(),
            limit: 10,
        })
        .expect("vector search through the trait")
}

#[test]
fn d2_exact_search_matches_brute_force_in_both_scopes() {
    let fx = fixture();
    let mut rng = Rng::new(0x5EED_0002);
    let corpus = build(&fx.store, &mut rng);

    let embedded_in = |p: Option<ProjectId>| -> usize {
        corpus
            .memories
            .iter()
            .filter(|m| p.is_none_or(|p| m.project == p))
            .map(|m| m.chunks.iter().filter(|c| c.is_some()).count())
            .sum()
    };
    assert!(embedded_in(None) >= 1000, "{} chunks", embedded_in(None));
    assert!(
        embedded_in(Some(corpus.projects[0])) >= 1000,
        "{} chunks in the first project",
        embedded_in(Some(corpus.projects[0]))
    );
    assert_eq!(
        fx.store.index_status().expect("status").chunks as usize,
        corpus
            .memories
            .iter()
            .map(|m| m.chunks.len())
            .sum::<usize>()
    );

    let mut queries: Vec<ExactVector> = (0..6).map(|_| exact_unit(&mut rng)).collect();
    queries.push(corpus.twin_vector.clone());
    assert!(corpus.reused.len() >= 3, "too few reused vectors");
    queries.extend(corpus.reused.iter().take(3).cloned());

    let mut scopes: Vec<QueryScope> = corpus
        .projects
        .iter()
        .map(|&p| QueryScope::Project(p))
        .collect();
    scopes.push(QueryScope::All);

    assert!(
        corpus.memories.iter().any(|m| m.status == Status::Pending),
        "the corpus holds no pending memory"
    );
    let all_statuses = Status::ALL.to_vec();
    let mut compared = 0usize;
    let mut ties = 0usize;
    for (qi, q) in queries.iter().enumerate() {
        let dots: Vec<Vec<Option<i64>>> = corpus
            .memories
            .iter()
            .map(|m| {
                m.chunks
                    .iter()
                    .map(|c| c.as_ref().map(|v| q.dot(v)))
                    .collect()
            })
            .collect();
        for &scope in &scopes {
            // Exclude the 30 best memories and 5 random ones.
            let mut exclude: Vec<MemoryId> =
                brute_force(&corpus, &dots, scope, &MemoryFilter::default(), 30)
                    .into_iter()
                    .map(|h| h.0)
                    .collect();
            for _ in 0..5 {
                exclude.push(corpus.memories[rng.below(corpus.memories.len())].id);
            }
            let cases: Vec<(&str, MemoryFilter, usize)> = vec![
                ("default, 10", MemoryFilter::default(), 10),
                ("default, 1", MemoryFilter::default(), 1),
                ("default, 50", MemoryFilter::default(), 50),
                ("default, every memory", MemoryFilter::default(), 2000),
                ("default, 0", MemoryFilter::default(), 0),
                (
                    "every status",
                    MemoryFilter {
                        statuses: all_statuses.clone(),
                        ..MemoryFilter::default()
                    },
                    20,
                ),
                (
                    "pending only",
                    MemoryFilter {
                        statuses: vec![Status::Pending],
                        ..MemoryFilter::default()
                    },
                    10,
                ),
                (
                    "superseded",
                    MemoryFilter {
                        statuses: vec![Status::Superseded],
                        ..MemoryFilter::default()
                    },
                    10,
                ),
                (
                    "runbooks",
                    MemoryFilter {
                        types: vec![MemoryType::Runbook],
                        ..MemoryFilter::default()
                    },
                    10,
                ),
                (
                    "importance >= 0.7",
                    MemoryFilter {
                        min_importance: Some(0.7),
                        ..MemoryFilter::default()
                    },
                    10,
                ),
                (
                    "anchor",
                    MemoryFilter {
                        anchor: Some(anchor("file:src/lib.rs")),
                        ..MemoryFilter::default()
                    },
                    10,
                ),
                (
                    "combined",
                    MemoryFilter {
                        statuses: vec![Status::Active, Status::Retracted],
                        min_importance: Some(0.5),
                        anchor: Some(anchor("symbol:Store::open")),
                        ..MemoryFilter::default()
                    },
                    10,
                ),
                (
                    "exclude",
                    MemoryFilter {
                        exclude: exclude.clone(),
                        ..MemoryFilter::default()
                    },
                    10,
                ),
            ];
            for (name, filter, limit) in &cases {
                let expected = brute_force(&corpus, &dots, scope, filter, *limit);
                if *name == "pending only" {
                    assert!(expected.is_empty(), "the oracle served a pending memory");
                }
                ties += expected.windows(2).filter(|w| w[0].2 == w[1].2).count();
                let got = search(&fx.store, &q.vector, scope, filter, *limit);
                assert_eq!(
                    as_tuples(&got),
                    as_scored(&expected),
                    "query {qi}, scope {scope:?}, case {name}"
                );
                compared += 1;
            }
        }
    }
    assert!(compared >= 400, "{compared} comparisons");
    assert!(ties > 0, "the corpus produced no tied scores");

    // The twin's best chunk is stored twice; the lower seq is reported.
    let top = search(
        &fx.store,
        &corpus.twin_vector.vector,
        QueryScope::All,
        &MemoryFilter::default(),
        10,
    );
    assert_eq!(top[0].memory_id, corpus.twin);
    assert_eq!(top[0].chunk_seq, 1);
    assert_eq!(top[0].score, 1.0);

    // The trait the daemon calls is the same search.
    for &scope in &scopes {
        assert_eq!(
            via_trait(&fx.store, &queries[0].vector, scope),
            search(
                &fx.store,
                &queries[0].vector,
                scope,
                &MemoryFilter::default(),
                10
            )
        );
    }
}

#[test]
fn d2_query_vectors_are_checked() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    add(&fx.store, p, "checked query");
    let short = vec![0.5f32; 10];
    let mut long_ = fake_vector("x");
    long_[0] += 0.5;
    let mut nan = fake_vector("x");
    nan[3] = f32::NAN;
    for bad in [short, long_, nan, vec![0.0; EMBED_DIM]] {
        assert_code(
            fx.store.nearest(&VectorQuery {
                vector: &bad,
                scope: QueryScope::All,
                filter: MemoryFilter::default(),
                limit: 5,
            }),
            ErrorCode::InvalidInput,
        );
    }
}

#[test]
fn d2_pending_chunks_are_not_vector_searched() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let text = "waiting for its embedding";
    let pending = fx
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
    let q = fake_vector(text);
    for scope in [QueryScope::Project(p), QueryScope::All] {
        assert!(search(&fx.store, &q, scope, &MemoryFilter::default(), 10).is_empty());
    }
    let chunks = fx.store.pending_chunks(10).expect("pending");
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].memory_id, pending);
    let done = fx
        .store
        .set_embeddings(
            FAKE_MODEL_ID,
            &[(chunks[0].id, fake_vector(&chunks[0].text))],
        )
        .expect("embed");
    assert_eq!(done, 1);
    for scope in [QueryScope::Project(p), QueryScope::All] {
        let hits = search(&fx.store, &q, scope, &MemoryFilter::default(), 10);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].memory_id, pending);
    }
    // A chunk that already has its embedding is not overwritten.
    let again = fx
        .store
        .set_embeddings(FAKE_MODEL_ID, &[(chunks[0].id, fake_vector("other"))])
        .expect("embed again");
    assert_eq!(again, 0);
}

/// SPEC section 5, Bulk: the queue reads pending chunks, embeds them outside
/// any transaction, then stores the vectors by chunk id. A memory revised or
/// forgotten meanwhile has new chunks, and a chunk id is never given out
/// twice, so the late vectors match no row and the new chunks stay pending.
#[test]
fn s5_a_late_embedding_never_lands_on_a_chunk_that_took_its_place() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let pending = |texts: &[&str]| -> Vec<NewChunk> {
        texts
            .iter()
            .zip(0u32..)
            .map(|(t, seq)| NewChunk::pending(piece(seq, t)))
            .collect()
    };
    let late = |chunks: &[crate::PendingChunk]| -> Vec<(ChunkId, Vec<f32>)> {
        chunks
            .iter()
            .map(|c| (c.id, fake_vector(&c.text)))
            .collect()
    };
    let old = ["the first text, part one", "the first text, part two"];
    let m = fx
        .store
        .insert_memory(
            MemoryInput {
                memory: NewMemory::new(p, old.join(" "), Provenance::new(ProvenanceSource::Agent)),
                chunks: pending(&old),
            },
            Access::Local,
        )
        .expect("insert")
        .id;
    let read = fx.store.pending_chunks(10).expect("pending");
    assert_eq!(read.len(), 2);

    // Revised while its batch is embedded: the old vectors must not land.
    let new = ["the revised text, part one", "the revised text, part two"];
    fx.store
        .revise_memory(
            &m,
            crate::Revision {
                content: Some(crate::ContentChange {
                    content: new.join(" "),
                    chunks: pending(&new),
                }),
                ..crate::Revision::default()
            },
            Access::Local,
        )
        .expect("revise");
    let stored = fx
        .store
        .set_embeddings(FAKE_MODEL_ID, &late(&read))
        .expect("late embeddings");
    assert_eq!(stored, 0, "a late vector landed on a revised chunk");
    let now = fx.store.pending_chunks(10).expect("pending");
    assert_eq!(now.iter().map(|c| c.text.as_str()).collect::<Vec<_>>(), new);
    assert!(now.iter().all(|c| read.iter().all(|r| r.id != c.id)));

    // Forgotten while its batch is embedded, and another memory stored.
    fx.store.forget(&[m], Access::Local).expect("forget");
    let other = ["a different memory"];
    let n = fx
        .store
        .insert_memory(
            MemoryInput {
                memory: NewMemory::new(p, other[0], Provenance::new(ProvenanceSource::Agent)),
                chunks: pending(&other),
            },
            Access::Local,
        )
        .expect("insert other")
        .id;
    let stored = fx
        .store
        .set_embeddings(FAKE_MODEL_ID, &late(&now))
        .expect("late embeddings");
    assert_eq!(stored, 0, "a forgotten memory's vector landed on another");
    let left = fx.store.pending_chunks(10).expect("pending");
    assert_eq!(left.len(), 1);
    assert_eq!(left[0].memory_id, n);
    assert_eq!(left[0].text, other[0]);
}

/// SPEC section 7, Pipeline step 1: a memory's map vector is the mean of its
/// embedded chunk vectors, re-normalized. Pending chunks are left out, a
/// memory with no embedded chunk has no vector, and chunks that cancel out
/// leave the first chunk to stand for the memory.
#[test]
fn s7_memory_vectors_are_normalized_chunk_means() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let q = project(&fx.store, "beta");
    let mut rng = Rng::new(17);
    let a = exact_unit(&mut rng).vector;
    let b = exact_unit(&mut rng).vector;
    let minus_b: Vec<f32> = b.iter().map(|x| -x).collect();
    let insert = |project: ProjectId, content: &str, chunks: Vec<NewChunk>| {
        fx.store
            .insert_memory(
                MemoryInput {
                    chunks,
                    ..note(project, content)
                },
                Access::Local,
            )
            .expect("insert")
            .id
    };
    let mean = insert(
        p,
        "two embedded chunks and a pending one",
        vec![
            NewChunk::embedded(piece(0, "first"), a.clone()),
            NewChunk::pending(piece(1, "second")),
            NewChunk::embedded(piece(2, "third"), b.clone()),
        ],
    );
    let cancel = insert(
        q,
        "chunks that cancel out",
        vec![
            NewChunk::embedded(piece(0, "forward"), b.clone()),
            NewChunk::embedded(piece(1, "backward"), minus_b),
        ],
    );
    let pending = insert(
        p,
        "only a pending chunk",
        vec![NewChunk::pending(piece(0, "waiting"))],
    );

    let sum: Vec<f64> = a
        .iter()
        .zip(&b)
        .map(|(x, y)| f64::from(*x) + f64::from(*y))
        .collect();
    let norm = sum.iter().map(|x| x * x).sum::<f64>().sqrt();
    let got = fx
        .store
        .memory_vector(&mean)
        .expect("vector")
        .expect("embedded");
    assert_eq!(got.len(), EMBED_DIM);
    for (g, s) in got.iter().zip(&sum) {
        assert!(
            (f64::from(*g) - s / norm).abs() < 1e-6,
            "{g} vs {}",
            s / norm
        );
    }
    crate::codec::check_vector("a memory vector", &got).expect("unit length");
    assert_eq!(fx.store.memory_vector(&cancel).expect("vector"), Some(b));
    assert_eq!(fx.store.memory_vector(&pending).expect("vector"), None);
    assert_code(
        fx.store.memory_vector(&MemoryId::new_v7()),
        ErrorCode::NotFound,
    );

    let ids = |scope: QueryScope, filter: &MemoryFilter| -> Vec<MemoryId> {
        let mut v: Vec<MemoryId> = fx
            .store
            .memory_vectors(scope, filter)
            .expect("memory vectors")
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        v.sort();
        v
    };
    let mut both = vec![mean, cancel];
    both.sort();
    assert_eq!(ids(QueryScope::All, &MemoryFilter::default()), both);
    assert_eq!(
        ids(QueryScope::Project(p), &MemoryFilter::default()),
        vec![mean]
    );
    let without_mean = MemoryFilter {
        exclude: vec![mean],
        ..MemoryFilter::default()
    };
    assert_eq!(ids(QueryScope::All, &without_mean), vec![cancel]);
    let all = fx
        .store
        .memory_vectors(QueryScope::All, &MemoryFilter::default())
        .expect("memory vectors");
    for (id, v) in &all {
        let one = fx.store.memory_vector(id).expect("vector").expect("one");
        assert_eq!(v, &one, "memory_vectors and memory_vector agree for {id}");
    }
}

/// B-197 (SPEC section 6, near-identical results): recall reads the vectors
/// of the memories it found in one call. Each is the one `memory_vector`
/// gives, whatever the memory's status; a memory with no embedded chunk, or
/// one that does not exist, is left out; more ids than one statement takes
/// are read in parts.
#[test]
fn s6_b197_memory_vectors_of_listed_ids() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let mut rng = Rng::new(23);
    let mut embedded = Vec::new();
    for i in 0..520 {
        let v = exact_unit(&mut rng).vector;
        let mut input = MemoryInput {
            chunks: vec![NewChunk::embedded(piece(0, &format!("chunk {i}")), v)],
            ..note(p, &format!("memory {i}"))
        };
        if i % 3 == 0 {
            input.memory.status = Status::Pending;
        }
        embedded.push(
            fx.store
                .insert_memory(input, Access::Local)
                .expect("insert")
                .id,
        );
    }
    let waiting = fx
        .store
        .insert_memory(
            MemoryInput {
                chunks: vec![NewChunk::pending(piece(0, "waiting"))],
                ..note(p, "waiting for its embedding")
            },
            Access::Local,
        )
        .expect("insert")
        .id;
    let missing = MemoryId::new_v7();

    let mut ask = embedded.clone();
    ask.push(waiting);
    ask.push(missing);
    let got = fx.store.memory_vectors_of(&ask).expect("vectors");
    assert_eq!(got.len(), embedded.len());
    for id in &embedded {
        let one = fx.store.memory_vector(id).expect("vector").expect("one");
        assert_eq!(got.get(id), Some(&one), "{id}");
    }
    assert!(!got.contains_key(&waiting) && !got.contains_key(&missing));
    assert!(fx.store.memory_vectors_of(&[]).expect("none").is_empty());
}
