//! AT-11 (store part): a stamp mismatch blocks vector and keyword search
//! with `index_mismatch`; after the reindex path writes the new stamp,
//! search works. SPEC section 4, Model stamp and Housekeeping.

use sm_core::ids::ChunkId;
use sm_core::testing::SimpleChunker;
use sm_core::traits::{ChunkParams, Chunker, MemoryFilter, QueryScope, VectorQuery};

use super::*;

const TEXTS: [&str; 3] = [
    "orders table uses soft delete with a deleted_at column",
    "billing service retries webhook delivery five times before giving up",
    "the ingress certificate rotates every ninety days automatically",
];

fn query(v: &[f32], scope: QueryScope) -> VectorQuery<'_> {
    VectorQuery {
        vector: v,
        scope,
        filter: MemoryFilter::default(),
        limit: 10,
    }
}

fn smaller_chunks() -> IndexStamp {
    IndexStamp {
        chunk_max_tokens: 4,
        ..stamp()
    }
}

#[test]
fn at11_stamp_mismatch_blocks_search_until_reindex() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");
    let ids: Vec<MemoryId> = TEXTS.iter().map(|t| add(&fx.store, p, t)).collect();
    let q = fake_vector(TEXTS[0]);

    // A second daemon configured with other chunk settings.
    let changed = smaller_chunks();
    let s2 = fx.reopen(changed.clone());
    let status = s2.index_status().expect("status");
    assert!(!status.matches);
    assert!(!status.differences.is_empty());
    assert_eq!(status.chunks, 3);
    assert_eq!(s2.stored_stamp().expect("stamp"), Some(stamp()));

    for scope in [QueryScope::Project(p), QueryScope::All] {
        assert_code(s2.nearest(&query(&q, scope)), ErrorCode::IndexMismatch);
        assert_code(
            s2.keyword_search("soft delete", scope, &MemoryFilter::default(), 10),
            ErrorCode::IndexMismatch,
        );
        assert_code(
            s2.memory_vectors(scope, &MemoryFilter::default()),
            ErrorCode::IndexMismatch,
        );
    }
    // Even an empty or blank search reports the mismatch.
    assert_code(
        s2.nearest(&VectorQuery {
            limit: 0,
            ..query(&q, QueryScope::All)
        }),
        ErrorCode::IndexMismatch,
    );
    assert_code(
        s2.keyword_search("  ", QueryScope::All, &MemoryFilter::default(), 10),
        ErrorCode::IndexMismatch,
    );
    assert_code(s2.memory_vector(&ids[0]), ErrorCode::IndexMismatch);
    assert_code(
        s2.set_embeddings(FAKE_MODEL_ID, &[(ChunkId(1), fake_vector("x"))]),
        ErrorCode::IndexMismatch,
    );
    assert_code(
        s2.set_embeddings("another-model", &[(ChunkId(1), fake_vector("x"))]),
        ErrorCode::Internal,
    );
    // The first store's stamp still matches what is stored.
    assert_eq!(
        fx.store
            .nearest(&query(&q, QueryScope::All))
            .expect("search")[0]
            .memory_id,
        ids[0]
    );

    // Reindex with the new chunk settings.
    let params = ChunkParams {
        max_tokens: 4,
        overlap: 0.125,
        index_cap: None,
    };
    let report = s2
        .reindex(|m| Ok(SimpleChunker.chunk(&m.content, &params)?.pieces))
        .expect("reindex");
    let backup = PathBuf::from(&report.backup);
    assert!(backup.is_file(), "no backup at {}", backup.display());
    assert!(
        backup
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.contains("reindex")),
        "backup name {}",
        backup.display()
    );
    assert_eq!(report.memories, 3);
    let expected_chunks: u64 = TEXTS
        .iter()
        .map(|t| SimpleChunker.chunk(t, &params).expect("chunk").pieces.len() as u64)
        .sum();
    assert_eq!(report.chunks_queued, expected_chunks);
    assert!(report.chunks_queued > 3);

    let status = s2.index_status().expect("status");
    assert!(status.matches, "{:?}", status.differences);
    assert_eq!(status.chunks, report.chunks_queued);
    assert_eq!(status.pending_embeddings, report.chunks_queued);
    assert_eq!(s2.stored_stamp().expect("stamp"), Some(changed));

    // Keyword search works at once: pending chunks are keyword-searched. A
    // phrase matches inside one chunk; with four-token chunks and no overlap
    // TEXTS[0] splits as "orders table uses soft | delete with a deleted_at |
    // column", so the query is taken from the first chunk.
    assert_eq!(
        s2.keyword_search(
            "orders table uses",
            QueryScope::All,
            &MemoryFilter::default(),
            10
        )
        .expect("keyword search")[0]
            .memory_id,
        ids[0]
    );
    // Vector search works and finds nothing until the chunks are embedded.
    assert!(
        s2.nearest(&query(&q, QueryScope::All))
            .expect("search")
            .is_empty()
    );
    // The store still configured with the old settings now mismatches.
    assert_code(
        fx.store.nearest(&query(&q, QueryScope::All)),
        ErrorCode::IndexMismatch,
    );

    let pending = s2.pending_chunks(1000).expect("pending");
    assert_eq!(pending.len() as u64, report.chunks_queued);
    let vectors: Vec<(ChunkId, Vec<f32>)> = pending
        .iter()
        .map(|c| (c.id, fake_vector(&c.text)))
        .collect();
    assert_eq!(
        s2.set_embeddings(FAKE_MODEL_ID, &vectors).expect("embed"),
        report.chunks_queued
    );
    assert_eq!(s2.index_status().expect("status").pending_embeddings, 0);
    let first_piece = fake_vector(&pending[0].text);
    let hits = s2
        .nearest(&query(&first_piece, QueryScope::Project(p)))
        .expect("search");
    assert_eq!(hits[0].memory_id, pending[0].memory_id);
    assert_eq!(hits[0].chunk_seq, pending[0].seq);
    assert!(s2.memory_vector(&ids[1]).expect("vector").is_some());

    // The backup holds the database as it was before the reindex.
    let restore = tempfile::tempdir().expect("temporary folder");
    let copy = restore.path().join("memory.db");
    std::fs::copy(&backup, &copy).expect("copy the backup");
    let old = Store::open_at(&copy, &restore.path().join("backups")).expect("open the backup");
    assert_eq!(old.stored_stamp().expect("stamp"), Some(stamp()));
    assert!(old.configure_index(stamp()).expect("configure").matches);
    assert_eq!(old.memory_count().expect("count"), 3);
    assert_eq!(old.index_status().expect("status").chunks, 3);
}

#[test]
fn at11_unconfigured_store_refuses_to_search() {
    let dir = tempfile::tempdir().expect("temporary folder");
    let store =
        Store::open_at(&dir.path().join("memory.db"), &dir.path().join("backups")).expect("open");
    let q = fake_vector("anything");
    assert_code(
        store.nearest(&query(&q, QueryScope::All)),
        ErrorCode::Internal,
    );
    assert_code(
        store.keyword_search("anything", QueryScope::All, &MemoryFilter::default(), 10),
        ErrorCode::Internal,
    );
    assert_code(
        store.set_embeddings(FAKE_MODEL_ID, &[(ChunkId(1), q.clone())]),
        ErrorCode::Internal,
    );
    assert_code(store.reindex(|_| Ok(Vec::new())), ErrorCode::Internal);
    // An embedded chunk needs a configured model.
    let p = project(&store, "alpha");
    assert_code(
        store.insert_memory(note(p, "too early"), Access::Local),
        ErrorCode::Internal,
    );
    assert_eq!(store.stored_stamp().expect("stamp"), None);
}

#[test]
fn s4_first_configuration_writes_the_stamp() {
    let dir = tempfile::tempdir().expect("temporary folder");
    let store =
        Store::open_at(&dir.path().join("memory.db"), &dir.path().join("backups")).expect("open");
    let status = store.configure_index(stamp()).expect("configure");
    assert!(status.matches);
    assert_eq!(status.chunks, 0);
    assert_eq!(store.stored_stamp().expect("stamp"), Some(stamp()));

    let p = project(&store, "alpha");
    add(&store, p, "one embedded chunk");
    let text = "one pending chunk";
    store
        .insert_memory(
            MemoryInput {
                memory: NewMemory::new(p, text, Provenance::new(ProvenanceSource::Agent)),
                chunks: vec![NewChunk::pending(piece(0, text))],
            },
            Access::Local,
        )
        .expect("insert");
    let status = store.index_status().expect("status");
    assert_eq!((status.chunks, status.pending_embeddings), (2, 1));

    // Configuring again with the same stamp changes nothing.
    assert!(store.configure_index(stamp()).expect("configure").matches);
    assert_eq!(store.stored_stamp().expect("stamp"), Some(stamp()));
}

#[test]
fn s4_invalid_stamps_are_refused() {
    let dir = tempfile::tempdir().expect("temporary folder");
    let store =
        Store::open_at(&dir.path().join("memory.db"), &dir.path().join("backups")).expect("open");
    assert_code(
        store.configure_index(IndexStamp {
            dim: 512,
            ..stamp()
        }),
        ErrorCode::ConfigInvalid,
    );
    assert_code(
        store.configure_index(IndexStamp {
            model_id: "  ".to_owned(),
            ..stamp()
        }),
        ErrorCode::ConfigInvalid,
    );
    assert_eq!(store.stored_stamp().expect("stamp"), None);
}
