//! Release-build timings of the SPEC section 7 pipeline: project and
//! cluster ([`regions`]) and map coordinates ([`coordinates`]) for 300, 2,000
//! and 5,000 fake memories (5,000 is sampled to 2,000) and the 320-memory
//! AT-7 fixture. Single-threaded by construction.
//!
//! ```text
//! cargo run --release -p sm-spatial --example budget
//! ```

#[path = "../tests/common/mod.rs"]
mod common;

use std::time::{Duration, Instant};

use sm_spatial::{MapParams, MemoryVectors, coordinates, regions};

const RUNS: usize = 3;

fn median(mut d: Vec<Duration>) -> Duration {
    d.sort_unstable();
    d[d.len() / 2]
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

fn time(name: &str, memories: &[MemoryVectors]) {
    let params = MapParams::default();
    let mut region_times = Vec::with_capacity(RUNS);
    let mut coord_times = Vec::with_capacity(RUNS);
    let mut last = None;
    for _ in 0..RUNS {
        let start = Instant::now();
        let run = regions(memories, &params).expect("regions");
        region_times.push(start.elapsed());
        let start = Instant::now();
        let coords = coordinates(memories, &params, &run).expect("coordinates");
        coord_times.push(start.elapsed());
        assert_eq!(coords.len(), memories.len());
        last = Some(run);
    }
    let run = last.expect("at least one run");
    println!(
        "{name:<16} n={:>5} projected={:>5} sampled={:<5} clusters={:>3} unclustered={:>5.1}% largest={:>5.1}% | regions median {:>8.1} ms (runs {}) | coordinates median {:>8.1} ms (runs {})",
        run.memories,
        run.projected,
        run.sampled,
        run.regions.len(),
        100.0 * run.unclustered_share(),
        100.0 * run.largest_share(),
        ms(median(region_times.clone())),
        region_times
            .iter()
            .map(|d| format!("{:.1}", ms(*d)))
            .collect::<Vec<_>>()
            .join(", "),
        ms(median(coord_times.clone())),
        coord_times
            .iter()
            .map(|d| format!("{:.1}", ms(*d)))
            .collect::<Vec<_>>()
            .join(", "),
    );
}

fn main() {
    println!(
        "sm-spatial budget: os={} arch={} logical_cpus={} cpu={} profile={}",
        std::env::consts::OS,
        std::env::consts::ARCH,
        std::thread::available_parallelism().map_or(0, |n| n.get()),
        std::env::var("PROCESSOR_IDENTIFIER").unwrap_or_else(|_| "unknown".to_owned()),
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
    );
    time("fixture (AT-7)", &common::fixture_memories());
    for n in [300, 2000, 5000] {
        time(&format!("fake {n}"), &common::fake_memories(n, 14));
    }
}
