//! SPEC 11 AT-7: clustering quality on the labelled fixture
//! (`fixtures/spatial`, 320 real-model vectors in 10 topics of 32).
//!
//! Gate: at most 25% unclustered and no cluster above 40% of the memories.
//! Reported (run with `--nocapture`): the cluster count, the noise share,
//! the largest share and the adjusted Rand index against the topics (SPEC
//! section 7 measured 0.55 to 0.59 with the Python libraries on other data).

mod common;

use std::collections::BTreeMap;

use common::{
    FIXTURE_SHA256, FIXTURE_TEXTS, FIXTURE_TOPICS, fixture_bytes, fixture_memories, fixture_readme,
    fixture_topics, fixture_vectors,
};
use sm_core::EMBED_DIM;
use sm_core::hash::sha256_hex;
use sm_core::ids::MemoryId;
use sm_core::traits::ClusterParams;
use sm_spatial::{MapParams, adjusted_rand_index, regions};

#[test]
fn at7_fixture_hash_matches_readme() {
    assert_eq!(sha256_hex(&fixture_bytes()), FIXTURE_SHA256);
    assert!(
        fixture_readme().contains(FIXTURE_SHA256),
        "fixtures/spatial/README.md records the hash"
    );
}

#[test]
fn at7_fixture_has_ten_topics_of_at_least_30() {
    let topics = fixture_topics();
    assert_eq!(topics.len(), FIXTURE_TEXTS);
    let mut counts = vec![0_usize; FIXTURE_TOPICS];
    for &t in &topics {
        counts[t] += 1;
    }
    assert!(counts.iter().all(|&c| c >= 30), "{counts:?}");
    let vectors = fixture_vectors();
    assert_eq!(vectors.len(), FIXTURE_TEXTS);
    for v in &vectors {
        assert_eq!(v.len(), EMBED_DIM);
        let norm = v.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "{norm}");
    }
}

#[test]
fn at7_clustering_quality_on_the_fixture() {
    let memories = fixture_memories();
    let topics = fixture_topics();
    let run = regions(&memories, &MapParams::default()).unwrap();
    assert!(run.clustered && !run.sampled);
    assert_eq!(run.memories, FIXTURE_TEXTS);
    assert_eq!(
        run.cluster_params,
        Some(ClusterParams::for_n(FIXTURE_TEXTS))
    );

    let row: BTreeMap<MemoryId, usize> = memories
        .iter()
        .enumerate()
        .map(|(i, m)| (m.id, i))
        .collect();
    let mut predicted: Vec<Option<usize>> = vec![None; FIXTURE_TEXTS];
    for r in &run.regions {
        for m in &r.members {
            predicted[row[&m.id]] = Some(r.region as usize);
        }
    }
    let ari = adjusted_rand_index(&topics, &predicted);

    println!(
        "AT-7 fixture: memories={} clusters={} unclustered={} ({:.1}%) largest={} ({:.1}%) ARI={ari:.3} min_cluster_size={} max_cluster_size={}",
        run.memories,
        run.regions.len(),
        run.unclustered.len(),
        100.0 * run.unclustered_share(),
        run.regions.first().map_or(0, |r| r.size()),
        100.0 * run.largest_share(),
        ClusterParams::for_n(FIXTURE_TEXTS).min_cluster_size,
        ClusterParams::for_n(FIXTURE_TEXTS).max_cluster_size,
    );
    for r in &run.regions {
        let mut by_topic = vec![0_usize; FIXTURE_TOPICS];
        for m in &r.members {
            by_topic[topics[row[&m.id]]] += 1;
        }
        println!(
            "AT-7 region {:>2}: size {:>3}, members per topic {by_topic:?}",
            r.region,
            r.size()
        );
    }

    assert!(
        run.unclustered_share() <= 0.25,
        "AT-7: {:.1}% unclustered, above 25%",
        100.0 * run.unclustered_share()
    );
    assert!(
        run.largest_share() <= 0.40,
        "AT-7: the largest cluster holds {:.1}%, above 40%",
        100.0 * run.largest_share()
    );
    assert!(
        run.regions.len() >= 2,
        "AT-7: the fixture splits into clusters"
    );
}
