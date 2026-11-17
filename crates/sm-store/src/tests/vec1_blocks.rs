//! How the store keeps `vec1`'s blocks (decision B-74): the search vec1's
//! writer runs before each insert is one lookup in the index of blocks with
//! room, not a scan of every full block, and the index stays right through
//! inserts, re-embeddings and deletes. SPEC section 4, "Cross-project
//! search".

use std::path::Path;

use rusqlite::Connection;

use super::*;
use crate::schema::VEC1_OPEN_BLOCKS_SQL;

/// The statement vec1 prepares to find a block with room
/// (`VEC1_SQL_SEARCH_IDX_FOR_INSERT` in `vendor/vec1/vec1.c`), as it reads
/// there, in two string literals.
const VENDORED_SEARCH: [&str; 2] = [
    r#""SELECT id, first, last, val ""#,
    r#""FROM %Q.'%q_idx' WHERE bucket=? AND length(val)<? LIMIT 1""#,
];

/// The same statement as vec1 expands it for `main.chunk_vectors`.
const SEARCH: &str = "SELECT id, first, last, val FROM 'main'.'chunk_vectors_idx' \
     WHERE bucket=? AND length(val)<? LIMIT 1";

/// vec1's default `blocksize-min`, the value it binds to the search.
const BLOCKSIZE_MIN: i64 = 16 * 1024;

/// The vendored vec1 source.
fn vec1_source() -> String {
    std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../vendor/vec1/vec1.c"))
        .expect("read the vendored vec1 source")
}

/// SQLite's plan for the insert search with `min` bound as vec1 binds
/// `blocksize-min`.
fn plan(conn: &Connection, min: i64) -> String {
    let mut s = conn
        .prepare(&format!("EXPLAIN QUERY PLAN {SEARCH}"))
        .expect("explain");
    let rows = s
        .query_map([0, min], |r| r.get::<_, String>(3))
        .expect("plan rows");
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .expect("plan")
        .join("; ")
}

fn raw(path: &Path) -> Connection {
    Connection::open(path).expect("open the database")
}

#[test]
fn s4_vec1_finds_a_block_with_room_in_one_index_lookup() {
    // The guard holds only while vec1 runs this statement with this bound:
    // a vendored update that changes either fails here first.
    let source = vec1_source();
    for part in VENDORED_SEARCH {
        assert!(
            source.contains(part),
            "vec1's insert search changed ({part} is gone); check that B-74's index still serves it"
        );
    }
    let default = source
        .lines()
        .find_map(|l| l.strip_prefix("#define VEC1_BLOCKSIZE_MIN_DEFAULT"))
        .map(str::trim)
        .expect("vec1 defines its default blocksize-min");
    assert_eq!(default, "(16*1024)", "vec1's default blocksize-min changed");
    assert!(
        VEC1_OPEN_BLOCKS_SQL.contains(&format!("length(val) < {BLOCKSIZE_MIN}")),
        "{VEC1_OPEN_BLOCKS_SQL}"
    );

    let fx = fixture();
    let c = raw(&fx.db_path());
    let used = plan(&c, BLOCKSIZE_MIN);
    assert!(
        used.contains("USING INDEX chunk_vectors_idx_open"),
        "the search must use the index of blocks with room: {used}"
    );
    // Any other bound cannot use it, and falls back to vec1's own index.
    let other = plan(&c, BLOCKSIZE_MIN * 2);
    assert!(!other.contains("chunk_vectors_idx_open"), "{other}");
}

#[test]
fn s4_vec1_blocks_stay_consistent_with_the_index_of_blocks_with_room() {
    let fx = fixture();
    let p = project(&fx.store, "alpha");

    // 120 memories of two chunks: 240 vectors, about 40 blocks of six.
    let mut ids = Vec::new();
    for i in 0..120 {
        let a = format!("chunk a of memory {i} about the orders table");
        let b = format!("chunk b of memory {i} about the billing service");
        let m = fx
            .store
            .insert_memory(note_in_parts(p, &[&a, &b]), Access::Local)
            .expect("insert");
        ids.push(m.id);
    }
    assert_eq!(table_count(&fx.store, "chunk_vectors"), 240);

    // Forget every third memory, so blocks lose vectors in the middle (vec1
    // writes a tombstone into each block in place).
    let gone: Vec<MemoryId> = ids.iter().copied().step_by(3).collect();
    fx.store.forget(&gone, Access::Local).expect("forget");
    let left = 240 - 2 * i64::try_from(gone.len()).expect("count");
    assert_eq!(table_count(&fx.store, "chunk_vectors"), left);

    // Revise some survivors, so their chunks are deleted and stored again.
    for id in ids.iter().filter(|id| !gone.contains(id)).step_by(7) {
        fx.store
            .revise_memory(
                id,
                crate::Revision {
                    content: Some(crate::ContentChange {
                        content: format!("revised {id}"),
                        chunks: vec![NewChunk::embedded(
                            piece(0, &format!("revised {id}")),
                            fake_vector(&format!("revised {id}")),
                        )],
                    }),
                    ..crate::Revision::default()
                },
                Access::Local,
            )
            .expect("revise");
    }

    // New vectors fill the blocks with room, found through the index.
    for i in 0..30 {
        add(&fx.store, p, &format!("a later note {i} on the ingress"));
    }
    let chunks = table_count(&fx.store, "chunks");
    assert_eq!(table_count(&fx.store, "chunk_vectors"), chunks);

    let c = raw(&fx.db_path());
    // SQLite's check covers the partial index's membership and, through
    // vec1's integrity method, every block against vec1's base table.
    let check: String = c
        .query_row("PRAGMA integrity_check", [], |r| r.get(0))
        .expect("integrity check");
    assert_eq!(check, "ok");
    // The index holds exactly the blocks with room.
    let (with_room, indexed, blocks): (i64, i64, i64) = c
        .query_row(
            &format!(
                "SELECT (SELECT count(*) FROM chunk_vectors_idx NOT INDEXED
                         WHERE length(val) < {BLOCKSIZE_MIN}),
                        (SELECT count(*) FROM chunk_vectors_idx INDEXED BY chunk_vectors_idx_open
                         WHERE length(val) < {BLOCKSIZE_MIN}),
                        (SELECT count(*) FROM chunk_vectors_idx)"
            ),
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .expect("count blocks");
    assert_eq!(indexed, with_room);
    assert!(
        with_room >= 1 && with_room < blocks,
        "{with_room} of {blocks}"
    );
}
