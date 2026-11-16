//! SPEC 11 AT-6: the same input gives identical region assignments on every
//! run, every thread count and every operating system.
//!
//! - Two runs in one process are identical, to the bit.
//! - Four threads running the pipeline at once each give the same result as
//!   the main thread. `holomap` is serial and `hdbscan` is built without its
//!   `parallel` feature, so there is no thread-count setting to vary: a run
//!   is single-threaded wherever it is called from.
//! - The assignments of the fixture run hash to a golden value computed on
//!   Windows; the Linux check runs the same test against the same value.

mod common;

use std::thread;

use common::{fake_memories, fixture_memories};
use sm_core::hash::sha256_hex;
use sm_core::protocol::MapPoint;
use sm_spatial::{MapParams, MemoryVectors, RegionRun, coordinates, regions};

/// SHA-256 of the fixture run's sorted `"<memory id> <region or ->"` lines.
const FIXTURE_ASSIGNMENTS_SHA256: &str =
    "425378230d48adc5bdb43be6ca9ac18f451f2f49e8300e23c873191b8c21ae0b";

/// SHA-256 of everything in the fixture run that is a number: the
/// assignments, every member's centrality and every centre, as bits, and
/// the 2-dimension coordinates.
const FIXTURE_FULL_SHA256: &str =
    "ab492e4b5c89aa71078739eda9df018392c9d999b8c5f310c60be34b9ba907f8";

fn assignments_digest(run: &RegionRun) -> String {
    let lines: Vec<String> = run
        .assignments()
        .iter()
        .map(|(id, r)| match r {
            Some(r) => format!("{id} {r}"),
            None => format!("{id} -"),
        })
        .collect();
    sha256_hex(lines.join("\n").as_bytes())
}

fn full_digest(run: &RegionRun, coords: &[MapPoint]) -> String {
    let mut lines = vec![run.input_hash.clone()];
    for r in &run.regions {
        lines.push(format!("region {} size {}", r.region, r.size()));
        for m in &r.members {
            lines.push(format!("{} {:08x}", m.id, m.centrality.to_bits()));
        }
        let centre: Vec<String> = r
            .centre
            .iter()
            .map(|x| format!("{:08x}", x.to_bits()))
            .collect();
        lines.push(centre.join(","));
    }
    lines.push(format!("unclustered {:?}", run.unclustered));
    for c in coords {
        lines.push(format!(
            "{} {:08x} {:08x}",
            c.id,
            c.x.to_bits(),
            c.y.to_bits()
        ));
    }
    sha256_hex(lines.join("\n").as_bytes())
}

fn run_with(memories: &[MemoryVectors], params: &MapParams) -> (RegionRun, Vec<MapPoint>) {
    let run = regions(memories, params).unwrap();
    let points = coordinates(memories, params, &run).unwrap();
    (run, points)
}

fn run_both(memories: &[MemoryVectors]) -> (RegionRun, Vec<MapPoint>) {
    run_with(memories, &MapParams::default())
}

fn assert_bit_identical(a: &(RegionRun, Vec<MapPoint>), b: &(RegionRun, Vec<MapPoint>)) {
    assert_eq!(a.0, b.0);
    assert_eq!(a.1, b.1);
    assert_eq!(full_digest(&a.0, &a.1), full_digest(&b.0, &b.1));
}

#[test]
fn at6_fixture_runs_are_identical_on_every_run_and_thread() {
    let memories = fixture_memories();
    let first = run_both(&memories);
    let second = run_both(&memories);
    assert_bit_identical(&first, &second);

    let threaded: Vec<(RegionRun, Vec<MapPoint>)> = thread::scope(|s| {
        let handles: Vec<_> = (0..4).map(|_| s.spawn(|| run_both(&memories))).collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for other in &threaded {
        assert_bit_identical(&first, other);
    }

    let assignments = assignments_digest(&first.0);
    let full = full_digest(&first.0, &first.1);
    println!("AT-6 fixture: assignments sha256 {assignments}; full sha256 {full}");
    assert_eq!(
        assignments, FIXTURE_ASSIGNMENTS_SHA256,
        "AT-6 golden assignments"
    );
    assert_eq!(full, FIXTURE_FULL_SHA256, "AT-6 golden numbers");
}

#[test]
fn at6_fake_embedder_runs_are_identical() {
    let memories = fake_memories(300, 6);
    let first = run_both(&memories);
    assert!(first.0.clustered);
    let second = run_both(&memories);
    assert_bit_identical(&first, &second);
    let threaded: Vec<(RegionRun, Vec<MapPoint>)> = thread::scope(|s| {
        let handles: Vec<_> = (0..4).map(|_| s.spawn(|| run_both(&memories))).collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for other in &threaded {
        assert_bit_identical(&first, other);
    }
}

#[test]
fn at6_input_order_does_not_change_the_run() {
    let memories = fixture_memories();
    let mut reversed = memories.clone();
    reversed.reverse();
    let mut interleaved: Vec<MemoryVectors> = memories.iter().step_by(2).cloned().collect();
    interleaved.extend(memories.iter().skip(1).step_by(2).cloned());
    let a = run_both(&memories);
    assert_bit_identical(&a, &run_both(&reversed));
    assert_bit_identical(&a, &run_both(&interleaved));
}

#[test]
fn at6_sampled_runs_are_identical() {
    // Above the cap: the sample and the nearest-centre assignment are
    // deterministic too (a smaller cap keeps the test fast).
    let memories = fake_memories(400, 5);
    let params = MapParams {
        max_points: 150,
        ..MapParams::default()
    };
    let a = run_with(&memories, &params);
    assert!(a.0.sampled);
    let b = run_with(&memories, &params);
    assert_bit_identical(&a, &b);
}
