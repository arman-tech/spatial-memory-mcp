# Spike: the map at 100,000 memories

This is the owner's thirteenth task (decision B-261; SPEC 7). It measures what the map costs and how much structure it keeps as one project grows from 2,000 to 100,000 memories. The question behind it is a local viewer that draws every memory of a project. Today the map is built from a sample of 2,000 memories. Its projection, holomap 0.3.0, was tested to 2,000 points (SPEC 7 and its risks table). The 5,000-point figures in BENCHMARKS.md ("Clustering") are for the Python server's UMAP, not holomap. A map is cached until the project's memories change.

It is a measuring spike. No product code changed and nothing was installed. It ran on the branch `viewer`, cut from `v3` at `a00c4b2`, on 23 November 2026. It ends in findings; B-261 decides nothing.

## Bottom line

At 100,000 memories in one project the map works, but it is slow to build, rebuilt often, and coarse.

- **Cost.** A full build takes 35.7 s in the library calls alone, 39.0 s with the cache's own reads and hash, and 48.2 s through the daemon. The peak working set is 1.1 GiB in the library and 1.7 GiB in the daemon. The two projections cost about 7 s each at every size, because they only ever see the 2,000 sample. What grows is placing and assigning the other memories, on one thread.
- **Cache.** An identical request is served in 15 to 90 ms. Any change to the project's active memories forces a full rebuild. On the real store, the largest project's active memories changed in 180 separate ten-minute windows over its six days.
- **Regions.** There are about 43 regions at every size, because the 2,000 sample sets the count. They agree with the 200 synthetic sub-topics at ARI 0.53 whatever the project's size.
- **Positions.** The map keeps 54% of a memory's 10 nearest neighbours when everything is sampled (2,000), but only 4.5% at 100,000, or 8.0% counting ties. 36% of placed memories sit on exactly the same point as another.
- **Payload.** 100,000 positions are 1.4 MB in binary or 8.9 MB in JSON. Their text, at the real store's mean length, is 29 MB.
- **Alternatives.**
  - Placing a new memory without a rebuild is exact and takes 0.24 ms for the 98% of new memories that stay outside the sample.
  - Samples of 5,000 and 10,000 recover more sub-topics (ARI 0.74 and 0.79) but cost 2.4 and 5.5 times as much. A sample of 20,000 crashes holomap 0.3.0.
  - Clustering again inside each region over-splits: 763 pieces for 200 sub-topics.
- **The real store cannot test scale.** Its largest project has 786 active memories, below the 2,000 cap, so nothing in it is sampled out.

## Setup

- **Machine.** AMD Ryzen 9 9950X3D: 16 cores, 32 logical processors. 93.6 GB of memory. Windows 11 Pro 10.0.26300. Release build with Rust 1.95.0.
- **Discipline.** Nothing else heavy ran during a measurement. Each measurement ran three times, each run in a fresh child process. Every timing is the median, with the range (lowest to highest) in brackets.
- **Peak memory.** It is the process's working set, sampled every 2 ms by a thread in the child (`map_scale/meter.rs`). A stage's peak is the highest working set seen while that stage ran.
- **The harness.** `crates/sm-cli/examples/map_scale.rs`, with its modules in `crates/sm-cli/examples/map_scale/`. It runs again with:

  ```
  cargo build --release -p sm-cli --bins --examples
  target/release/examples/map_scale prepare cost analyse cache alt7a alt7b alt7c
  target/release/examples/map_scale real <copy of memory.db> <backups folder>
  ```

  Each step appends one JSON line per run to `results.jsonl` in the work folder: `sm-map-scale` under the temporary folder, or `SM_MAP_SCALE_DIR`. The synthetic stores are kept there too, about 1.3 GB in all.
- **The only change outside the example.** `crates/sm-cli/Cargo.toml` takes `sm-spatial` and `holomap` as dev-dependencies, and `Cargo.lock` lists them under sm-cli. Both are existing workspace pins. No library code, schema or configuration changed.
- **No model calls.** The synthetic vectors stand in for embeddings. The real store's vectors are read from a copy. Nothing is embedded on either path.

### Three paths

Each figure names the path it came by.

| Path | What it is | Used for |
| --- | --- | --- |
| Library | `sm_spatial::regions()` and `sm_spatial::coordinates()` called directly on the vectors from `Store::memory_vectors`, as the daemon calls them | Table 1a; the library rows of table 2; the real store |
| Replica | The same pipeline rebuilt stage by stage in `map_scale/pipe.rs`, so that each stage can be timed and its intermediate results kept | Table 1b; items 3 to 7 |
| Daemon | A release daemon on a scratch profile holding a copy of the 100,000-memory store, sent the `regions` operation with `coords: true` over its socket | The daemon rows of table 2 |

The replica matches the library bit for bit. On the first run at each size its regions were identical and no coordinate differed.

## The synthetic data

The generator is `map_scale/synth.rs`. It was fixed before any result was seen, and nothing in it was tuned.

- **Vectors.** Unit vectors in 768 dimensions, in 40 topics of 5 sub-topics each, so 200 sub-topics. A member of sub-topic `s` of topic `t` is:

  ```
  unit( sqrt(0.25) g + sqrt(0.25) T[t] + sqrt(0.15) S[t][s] + sqrt(0.20) L + sqrt(0.15) E )
  ```

  - `g` is one global direction.
  - `T[t]` is a topic direction and `S[t][s]` a sub-topic direction. All of these are fixed random unit directions.
  - `L` mixes 8 fixed random directions of that sub-topic with normal weights.
  - `E` is fresh random noise.

  The expected cosines are about 0.65 inside a sub-topic, 0.50 inside a topic and 0.25 between topics. The measured mean cosine of a random pair is 0.270 at every size.
- **Noise.** 5% of memories are noise, `unit(sqrt(0.25) g + sqrt(0.75) E)`, with no topic.
- **Sizes.** Topic `t` has weight `(t + 1)^-0.8` and sub-topic `s` has weight `(s + 1)^-0.8`. The 200 cells share 95% of the memories by largest remainder. At 100,000 memories the sub-topics run from 87 to 6,055 memories and the topics from 821 to 15,716.
- **Other fields.** Creation times are uniform over 365 days from 2025-11-23. Types are uniform over the six. Every memory is active, with one chunk each.
- **IDs and seeds.** Identifiers are UUIDv7-shaped. The random generator is SplitMix64 with Box-Muller normals, from one structure seed. Each size is an independent draw, so the 10,000 set is not a subset of the 100,000 set.

**The generator's limits.**

- The clusters are isotropic Gaussian blobs, and every topic is equally far from every other. Real topics overlap, nest and differ in spread, so ARI on this data says how well the pipeline recovers clean, equal structure, not how a real project's regions will look.
- Inside a sub-topic, memories vary along 8 latent directions plus full-rank noise. No 2-D layout can keep that at 10-nearest-neighbour resolution once a sub-topic holds hundreds of memories. So the neighbour figures for placed memories partly measure the data, not just the pipeline. The real store's figures are the check on that.
- There is one chunk per memory, so the mean-vector step does nothing here.
- The time, type and status fields are uniform, which is not realistic. They only size the payload.

## The real store

A copy of the live store, used for vectors and counts only. No memory text left the copy, and the copy has been deleted.

1. `spatial-memory stop --hold` stopped the default profile's daemon. Its clean stop had checkpointed the write-ahead log, so the store was a single file.
2. That file was copied into a new folder under `%LOCALAPPDATA%\spatial-memory\analysis\`.
3. `spatial-memory release` lifted the hold straight after the copy.
4. `map_scale real` opened the copy through `Store::open_at` and measured it. It ran a second time, on the same copy, to add the active-only write counts.
5. The copy folder was then deleted, on 23 November 2026, and a listing confirmed it was gone.

| Count | Value |
| --- | --- |
| Memories | 11,235 |
| Active, each with a vector | 1,104 |
| Pending: not on the map, because no search serves them (B-157) | 10,131 |
| Projects with active memories | 3, with 786, 230 and 88 |
| Mean content length, active | 291 bytes (median 240) |
| Mean content length, every status | 645 bytes (median 558) |
| Mean cosine of a random pair, largest project | 0.665 |

- **Everything is in the sample.** The largest project is under the 2,000 cap, so the product maps all of it. Nothing is assigned or placed. The real store therefore checks the projection's neighbours on real vectors, but not scale. One extra run with a sample of 500, which is not the product's setting, shows placement on real vectors (table 4c).
- **Real vectors are concentrated.** A random pair of real memories has cosine 0.665, against 0.270 in the synthetic data.

The map of the largest project, library path, three runs in one process:

| Call | Time, ms | Peak working set |
| --- | ---: | ---: |
| `regions()` | 1,479 (1,440–1,493) | 114 MiB |
| `coordinates()` | 1,463 (1,415–1,473) | 114 MiB |

| Map | Memories | Regions | Unclustered | Largest region |
| --- | ---: | ---: | ---: | ---: |
| Largest project, the product's settings | 786 | 13 | 28.6% | 15.8% |
| Every project pooled, the product's settings | 1,104 | 14 | 26.5% | 18.3% |
| Largest project, sample of 500 | 786 | 13 | 17.9% | 17.8% |

With nothing assigned, HDBSCAN's leftovers stay unclustered: 27–29% of the real memories, against 0.17% for 100,000 synthetic ones.

## 1. Cost per size

**Table 1a.** Library path. Time in ms; peak is the highest working set across the two calls.

| Memories | Read vectors | `regions()` | `coordinates()` | Sum of the medians | Peak working set |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2,000 | 16.2 (16.1–18.5) | 6,955 (6,943–6,985) | 7,043 (7,002–7,055) | 14,014 | 103 MiB |
| 10,000 | 78.4 (77.5–78.5) | 7,251 (7,195–7,293) | 8,456 (8,420–8,881) | 15,785 | 186 MiB |
| 50,000 | 488 (479–592) | 7,962 (7,849–8,127) | 16,552 (16,527–16,890) | 25,002 | 602 MiB |
| 100,000 | 1,041 (1,029–1,046) | 8,759 (8,751–8,782) | 25,879 (25,817–25,900) | 35,679 | 1,121 MiB |

**Table 1b.** Replica path, by stage. Time in ms.

| Stage | 2,000 | 10,000 | 50,000 | 100,000 |
| --- | ---: | ---: | ---: | ---: |
| Read vectors | 16.3 | 86.6 | 488.5 | 1,013.0 (1,004.0–1,016.5) |
| Prepare: mean vectors and sort | 2.3 | 13.0 | 65.1 | 126.7 (126.6–127.7) |
| Input hash | 0.1 | 0.6 | 3.1 | 6.4 |
| Sample 2,000 | 0.9 | 1.5 | 2.6 | 4.7 |
| holomap to 5-D on the sample | 6,962.0 (6,909.0–7,028.5) | 7,064.3 (7,024.2–7,068.8) | 7,359.8 (7,146.9–7,474.7) | 7,083.3 (7,052.6–7,104.9) |
| HDBSCAN | 11.5 | 11.3 | 11.5 | 11.1 |
| Assign the rest to the nearest centre | 0.5 | 126.0 (125.0–131.0) | 715.3 (693.3–751.5) | 1,438.7 (1,433.5–1,445.3) |
| Build regions: rank members | 1.0 | 5.8 | 41.1 | 74.9 |
| Coordinates: prepare and check | 2.6 | 14.1 | 91.2 | 143.1 |
| holomap to 2-D on the sample | 7,026.3 (6,990.4–7,078.0) | 7,091.3 (7,079.2–7,108.2) | 7,303.1 (7,032.7–7,446.0) | 7,072.4 (7,061.0–7,126.3) |
| Place the rest at the mean of their 15 nearest sample rows | 0.0 | 1,225.0 (1,214.6–1,289.6) | 8,794.6 (8,666.0–9,036.5) | 17,743.9 (17,640.9–17,931.1) |

Stages under 20 ms are shown as medians only. Their ranges are within 10% at every size.

- **Projection is flat.** Both projections see only the 2,000-memory sample, so each costs about 7 s at every size. At 2,000 memories they are 99.6% of the time.
- **Placement and assignment grow linearly.** Placement costs about 0.18 ms per placed memory: 0.153 at 10,000, 0.183 at 50,000 and 0.181 at 100,000. Assignment costs about 0.015 ms per memory. Both run on one thread.
- **At 100,000, after the read:** placement is 53% of the time, the two projections together 42%, and assignment 4%.
- **Memory.** Peak working set at 100,000, by stage:

  | Stage | Peak working set |
  | --- | ---: |
  | Read | 737 MiB |
  | Prepare | 1,036 MiB |
  | 5-D projection | 1,115 MiB |
  | Assignment | 1,051 MiB |
  | 2-D projection | 1,120 MiB |
  | Placement | 1,054 MiB |

  The raw vectors are 293 MiB (100,000 × 768 × 4 bytes). The read alone holds about 2.5 times that. The normalised copy made in preparation adds about 300 MiB more. Each projection adds about 70 MiB.

## 2. The cache at 100,000

The daemon's cache (`crates/sm-daemon/src/tools/maps.rs`, `run_for`) has three levels:

1. **An in-memory index by project.** On a hit it reads the stored run by its id.
2. **Otherwise, the stored run.** It reads every vector and every memory row of the project, hashes them, and looks the run up by that state hash.
3. **Otherwise, a rebuild.** It computes the run, stores it, and prunes all but four runs (`KEEP_RUNS`).

The state hash covers every memory's identifier, `updated_at` and vector bytes. So a new memory, an edit, a forget or a re-embedding in the project gives a new key, and the next request rebuilds.

**Table 2a.** Library path: the same steps as `run_for`, as library and store calls. Time in ms.

| Request | Time | Peak working set |
| --- | ---: | ---: |
| First request: nothing stored | 38,969 (38,774–40,544) | 1,128 MiB |
| Identical request, index hit | 14.9 (14.7–14.9) | 29 MiB |
| Identical request with the index dropped, stored run found | 3,787 (3,754–4,006) | 1,071 MiB |
| Insert one memory | 2.9 (2.8–2.9) | 456 MiB |
| Request after that insert: rebuild | 39,152 (38,967–40,400) | 1,142 MiB |

The first request, by stage, in ms:

| Stage | Time |
| --- | ---: |
| Read vectors | 1,001 |
| Read every memory row for `updated_at` | 2,427 |
| State hash | 191 |
| `regions()` | 8,889 |
| `coordinates()` | 25,980 |
| `put_map_run` | 468 |
| Labels and prune | 34 |

The stored-run hit is almost all the first three of those: 1,015 + 2,553 + 191 ms, then 14 ms for the lookup.

**Table 2b.** Daemon path: release daemon, scratch profile, the 100,000-memory store copied in. No embeddings were pending at the start or end of any run.

| Step | Time, ms | Reply | Peak working set |
| --- | ---: | --- | ---: |
| Start, until ready | 1,506 (1,499–1,586) | | 688 MiB (687–699) after |
| A. First request, default limit | 48,231 (48,225–48,409) | 7,562 bytes: 7 of 43 regions, 0 points, 100,036 items omitted | 1,779 MiB (1,777–1,789) |
| B. Identical request | 33.0 (31.4–33.5) | The same | 664 MiB (663–675) |
| C. Every point, `limit` 100,000,000 | 86.9 (71.5–89.4) | 8,656,597 bytes: 43 regions, 100,000 points | 665 MiB (663–676) |
| Stopped; one memory inserted through the store; restarted | 1,497 (1,481–1,516) | | 649 MiB after |
| D. Every point, after the insert: rebuild | 48,709 (48,158–48,778) | 8,656,684 bytes, 100,001 points | 1,747 MiB |
| E. Identical request | 90.1 (82.4–97.0) | | 643 MiB |
| Restarted, nothing changed | 1,508 (1,484–1,575) | | 649 MiB after |
| F. Every point, stored run | 4,782 (4,690–4,829) | | 1,673 MiB (1,527–1,673) |

- **A rebuild is slower through the daemon.** A miss takes 48 s through the daemon against 39 s in the library, about 9 s more. That gap was not investigated.
- **The default reply holds no points.** At the default token budget, the reply at 100,000 carries 7 of the 43 regions and no points. A viewer has to ask with an explicit `limit`.
- **When it rebuilds.** Every write drops the in-memory index for every project, not just its own. The sites are:
  - each embedded batch (`queue.rs:264` and `queue.rs:332`);
  - `revise` (`edit.rs:125`, `edit.rs:237`);
  - `consolidate` (`consolidate.rs:186`);
  - the admin operations (`admin.rs:143`, `admin.rs:194`);
  - every write that stores chunks (`state.rs:344`).

  After a write to another project, the next request at 100,000 pays 3.8 s (library) to 4.8 s (daemon) to re-read and re-hash before it finds the stored run. After a write to the same project, it pays a full rebuild.
- **A rebuild interrupted by a write is not indexed.** `index_map` (`state.rs`) refuses a run when the maps were dropped since its input was read. So a write anywhere during a 48-second rebuild leaves the result stored but not indexed, and the next request takes the stored-run path. This is read from the code, not measured.

Writes to the real store, from the creation and update times of its memories (`map_scale real`):

| Writes | Largest project, active | Every project, active | Largest project, every status |
| --- | ---: | ---: | ---: |
| Days with a write, last 30 | 6 | 6 | 6 |
| Writes: creations plus updates | 1,572 | 2,208 | 7,200 |
| Median per day with a write | 172 | 313 | 978 |
| Hours with a write, last 14 days | 52 | 67 | 54 |
| Ten-minute windows with a write, last 14 days | 180 | 245 | 217 |
| Median gap between writes | 0 minutes | 0 minutes | 0 minutes |

The store is six days old, so the 30- and 14-day windows hold the same six days. A median gap of zero means at least half of the writes share their timestamp with the write before them: they arrive in batches.

**How often the map would be rebuilt while capture runs.** On the real store, the largest project's active memories changed in 180 separate ten-minute windows over six days, so during a working session a map requested more than a few minutes after the last one would almost always need a rebuild, which at 100,000 memories is 39 s in the library and 48 s through the daemon.

## 3. Quality on synthetic data

Agreement is the adjusted Rand index (`sm_spatial::adjusted_rand_index`). A memory in no region counts as one more predicted class. Columns:

- **all**: every memory, with noise as its own true class.
- **members**: noise left out.
- **sampled / placed**: the 2,000 sample alone, then everything else alone.
- **Leading a region**: the true classes that are the largest class among the members of at least one region.
- **Unclustered, sample**: the share of the sample that HDBSCAN left out.
- **Noise unclustered**: the share of the noise memories in no region.

**Table 3.** Replica path, the product's settings.

| Memories | Regions | ARI topics, all | ARI topics, members | ARI sub-topics, all | ARI sub-topics, members | ARI sub-topics, sampled / placed | Topics leading a region | Sub-topics leading a region | Unclustered, all | Unclustered, sample | Noise unclustered | Largest region |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,000 | 45 | 0.469 | 0.496 | 0.591 | 0.660 | n/a | 30 of 40 | 45 of 200 | 7.05% | 7.0% | 1.00% | 6.2% |
| 10,000 | 46 | 0.467 | 0.500 | 0.554 | 0.635 | 0.539 / 0.529 | 30 of 40 | 46 of 200 | 1.89% | 9.4% | 0.60% | 11.3% |
| 50,000 | 42 | 0.447 | 0.482 | 0.497 | 0.568 | 0.516 / 0.492 | 27 of 40 | 42 of 200 | 0.39% | 9.7% | 0.08% | 13.7% |
| 100,000 | 43 | 0.455 | 0.490 | 0.533 | 0.612 | 0.536 / 0.531 | 28 of 40 | 43 of 200 | 0.17% | 8.5% | 0.14% | 10.6% |

- **The region count is set by the sample, not the project.** It is 42 to 46 at every size, because HDBSCAN only ever sees 2,000 points, with `min_cluster_size` 22. Each region's largest class is a different sub-topic: the "sub-topics leading" column equals the region count. With 43 regions for 200 sub-topics, a region holds about 4.7 sub-topics on average. At 100,000, 12 of the 40 topics are the largest class of no region.
- **Agreement does not fall with size.** Agreement is flat between sampled and placed memories too. Assignment to the nearest centre in 768 dimensions keeps the sample's partition.
- **Unclustered collapses through assignment.** HDBSCAN leaves 7–10% of the sample out. Every memory outside the sample is assigned to a region, so at 100,000 only 0.17% are unclustered.
- **Noise is not separated.** Even at 2,000, where nothing is assigned, HDBSCAN puts 99% of the noise memories in a region.

## 4. Neighbours

For 1,000 random memories: the share of each memory's 10 nearest by cosine in 768 dimensions that are also among its 10 nearest on the 2-D map. Memories in the sample and memories placed by the mean of their 15 nearest sample rows are reported separately. A second group of 1,000 is drawn from the sample alone. "Chance" is 10 / (n − 1).

**Table 4a.** Replica path. Ties on the map are broken by row order.

| Memories | Group | Queries | Mean overlap | Median | None shared | Chance |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2,000 | 1,000 random, all sampled | 1,000 | 54.3% | 60% | 0.6% | 0.50% |
| 10,000 | 1,000 random | 1,000 | 16.5% | 10% | 25.3% | 0.10% |
| | of which sampled | 194 | 16.4% | 15% | 22.2% | |
| | of which placed | 806 | 16.6% | 10% | 26.1% | |
| | 1,000 from the sample | 1,000 | 18.0% | 20% | 20.7% | |
| 50,000 | 1,000 random | 1,000 | 6.5% | 0% | 56.4% | 0.02% |
| | of which sampled | 45 | 5.1% | 0% | 66.7% | |
| | of which placed | 955 | 6.6% | 0% | 55.9% | |
| | 1,000 from the sample | 1,000 | 4.4% | 0% | 67.8% | |
| 100,000 | 1,000 random | 1,000 | 4.5% | 0% | 68.5% | 0.01% |
| | of which sampled | 21 | 1.4% | 0% | 85.7% | |
| | of which placed | 979 | 4.6% | 0% | 68.1% | |
| | 1,000 from the sample | 1,000 | 2.0% | 0% | 83.0% | |

Mean cosine of the nearest and the 10th nearest in 768 dimensions:

| Memories | Nearest | 10th nearest |
| ---: | ---: | ---: |
| 2,000 | 0.739 | 0.606 |
| 10,000 | 0.773 | 0.706 |
| 50,000 | 0.801 | 0.764 |
| 100,000 | 0.795 | 0.766 |

**Table 4b.** The same queries, with ties on the map counted. A memory counts as a map neighbour if it is no farther away than the 10th nearest on the map, so memories stacked on one point all count. The tie set is how many memories lie within that distance.

| Memories | Group | Mean overlap, ties counted | None shared, ties counted | Tie set, median | Tie set over 10 | Tie set, largest |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2,000 | 1,000 random, all sampled | 54.3% | 0.6% | 10 | 0.0% | 10 |
| 10,000 | 1,000 random | 18.8% | 24.4% | 10 | 12.1% | 78 |
| | of which sampled | 18.3% | 21.1% | 10 | 8.8% | 78 |
| | of which placed | 18.9% | 25.2% | 10 | 12.9% | 78 |
| | 1,000 from the sample | 19.5% | 20.4% | 10 | 10.6% | 88 |
| 50,000 | 1,000 random | 10.1% | 52.4% | 10 | 25.1% | 274 |
| | of which sampled | 5.8% | 64.4% | 10 | 13.3% | 73 |
| | of which placed | 10.3% | 51.8% | 10 | 25.7% | 274 |
| | 1,000 from the sample | 5.7% | 64.8% | 10 | 21.5% | 278 |
| 100,000 | 1,000 random | 8.0% | 61.0% | 10 | 29.7% | 667 |
| | of which sampled | 3.3% | 76.2% | 10 | 28.6% | 31 |
| | of which placed | 8.1% | 60.7% | 10 | 29.7% | 667 |
| | 1,000 from the sample | 2.9% | 80.1% | 10 | 24.8% | 671 |

Counting ties roughly doubles the overlap for placed memories at 50,000 and 100,000, because so many of them are stacked (table 5). Even so, it stays under 10%. At 100,000, 30% of the queries have more than 10 memories at or inside their 10th map distance, and one has 667.

**Table 4c.** The real store, library path, every active memory of the largest project as a query. Ties made no difference here: every tie set was exactly 10.

| Map | Group | Queries | Mean overlap | Median | None shared | Chance |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Largest project, the product's settings | all, all sampled | 786 | 36.4% | 30% | 2.4% | 1.27% |
| Every project pooled, the product's settings | 1,000 random, all sampled | 1,000 | 34.9% | 30% | 3.1% | 0.91% |
| Largest project, sample of 500 | all | 786 | 24.4% | 20% | 17.8% | 1.27% |
| | of which sampled | 500 | 31.4% | 30% | 3.4% | |
| | of which placed | 286 | 12.1% | 10% | 43.0% | |

The mean cosine of the nearest and the 10th nearest on the real store is 0.830 and 0.768, against 0.665 for a random pair.

- **Sampled memories lose their neighbours too.** A sampled memory's 10 nearest in 768 dimensions are almost all placed memories once the project is large. At 100,000, 98% of memories are placed. So the overlap for a sampled memory falls with size as fast as for a placed one: 2.0% for 1,000 drawn from the sample.
- **In the synthetic data, the true neighbours are close but not distinct.** At 100,000 the 10th nearest in 768 dimensions has cosine 0.766, against 0.795 for the nearest. A memory's ten nearest are hard to tell from its next hundred, and no 2-D layout separates them (see the generator's limits).
- **With everything sampled, the map keeps about a third of the neighbours.** That is 54% on synthetic data at 2,000 and 36% on the real store's largest project.
- **Placement loses most of what is left.** On real vectors with a sample of 500, placed memories keep 12% of their neighbours against 31% for sampled ones. 43% of placed memories keep none.

## 5. Clumping

A placed memory sits at the mean of its 15 nearest sample rows. Memories with the same 15 nearest sample rows therefore land on the same point.

- **Threshold.** The median nearest-neighbour distance among sampled memories on the map, measured among sampled memories only.
- **Placed below.** The share of placed memories whose nearest other memory, of all memories, is closer than that threshold.
- **Same position.** The share of placed memories at exactly the same position as another memory.

**Table 5.** Replica path, the product's settings.

| Memories | Placed | Distinct positions of placed | Placed below the threshold | Placed at the same position as another | Sampled below | Threshold | Placed median nearest distance | Map diagonal |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,000 | 0 | n/a | n/a | n/a | 50.0% | 0.00443 | n/a | 54.1 |
| 10,000 | 8,000 | 6,947 | 78.8% | 18.0% | 57.5% | 0.00468 | 0.00072 | 50.6 |
| 50,000 | 48,000 | 35,681 | 88.5% | 32.8% | 61.0% | 0.00406 | 0.00019 | 54.5 |
| 100,000 | 98,000 | 69,842 | 91.6% | 35.7% | 63.1% | 0.00487 | 0.00014 | 51.3 |

- **Stacking.** At 100,000, 35.7% of placed memories, about 35,000, sit exactly on another memory's position. The 98,000 placed memories occupy 69,842 distinct positions. Drawn as points, the stacked ones hide each other.
- **Crowding.** The median placed memory's nearest neighbour is 35 times closer than the sample's median.
- **The layout itself is tight.** Even the sample's own median distance is about 1/10,000 of the map's diagonal.
- **Real store.** With a sample of 500, which is not the product's setting, the 286 placed memories all have distinct positions. 46.5% are closer than the threshold, against 57.8% of sampled ones. Only 286 memories are placed against 500 sampled, which is far fewer per sample row than in the synthetic runs, so this does not show how real vectors would clump at scale.

## 6. Payload per size

What a viewer needs per memory besides its id is its position, creation day, type, region and status.

- **Binary.** A 16-byte header, the type and status name tables, then 14 bytes per memory: x and y as f32, the day as u16, the type as u8, the region as u16 (0xFFFF for none) and the status as u8. With the id, 16 bytes more per memory.
- **JSON.** One object per memory: `{x, y, day, type, region, status}`, with the region left out for none. Optionally with the id as a string.
- **Encode time.** Encoding from data already in memory.
- **Attribute read.** `Store::get_memories` for every memory, done once before encoding.

**Table 6.** Replica path for positions; library store reads for attributes. Bytes; times in ms.

| Memories | Binary | Binary with id | JSON | JSON with id | Text at the real store's mean length | Encode: binary / JSON / JSON with id | Attribute read |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 2,000 | 28,075 | 60,075 | 176,382 | 264,382 | 581,741 | 0.09 / 0.12 / 0.22 | 47 (47–51) |
| 10,000 | 140,075 | 300,075 | 892,202 | 1,332,202 | 2,908,705 | 0.44 / 0.67 / 1.24 | 250 (246–251) |
| 50,000 | 700,075 | 1,500,075 | 4,469,738 | 6,669,738 | 14,543,524 | 2.38 / 4.39 / 6.88 | 1,244 (1,235–1,263) |
| 100,000 | 1,400,075 | 3,000,075 | 8,943,563 | 13,343,563 | 29,087,047 | 5.31 / 9.02 / 13.87 | 2,515 (2,515–2,650) |

- **Encode times** are medians of three. At 100,000 the ranges are 5.28–5.40, 8.90–9.24 and 13.80–14.13 ms.
- **The figures** come from the second `analyse` run, the one that added the tie-inclusive neighbour columns. Its sizes are identical to the first run's, and its times are within 3%.
- **The daemon's own reply** with every point at 100,000 is 8,656,597 bytes (table 2b, step C), close to the JSON column.
- **Text.** The text column is the memories' content at the real store's mean active length, 290.87 bytes, times the count. It is raw UTF-8, before any JSON escaping. At 100,000 the text is 29 MB: 21 times the binary positions, and 2.2 times the JSON with ids. At every status the real mean is 645 bytes, which would double it.

## 7. Alternatives, by direct library calls

None of these changes the product. Each calls the library crates, or holomap directly, from the example.

### 7a. Larger samples

The 100,000-memory set mapped from a sample of 5,000, 10,000 and 20,000 through the replica. Above 2,000, holomap is called directly, because `MapParams::validate` refuses `max_points` over 2,000. `min_cluster_size` and `max_cluster_size` follow the product's formula for the sample size. Quality, neighbours and clumping come from the first run of each size. To keep memory down, the raw vectors are dropped after preparation. The process's peak working set was 736 MiB at both 5,000 and 10,000. These runs came before the tie-inclusive measure was added, so their neighbour figures count without ties.

**Table 7a.** Time in ms.

| Sample | 2,000 (product) | 5,000 | 10,000 | 20,000 |
| --- | ---: | ---: | ---: | --- |
| `min_cluster_size` / `max_cluster_size` | 22 / 500 | 35 / 1,250 | 50 / 2,500 | |
| holomap to 5-D | 7,083 (7,053–7,105) | 15,872 (15,666–16,077) | 52,163 (51,945–52,181) | Panics |
| HDBSCAN | 11 | 57 (53–57) | 204 (198–277) | |
| Assign | 1,439 (1,434–1,445) | 2,069 (2,051–2,087) | 2,379 (2,368–2,385) | |
| holomap to 2-D | 7,072 (7,061–7,126) | 15,542 (15,370–15,570) | 48,799 (48,483–48,812) | |
| Place | 17,744 (17,641–17,931) | 46,008 (45,151–46,467) | 81,763 (81,184–84,373) | |
| Sum, sampling to placement | 33,429 | 79,230 (78,812–80,287) | 184,905 (184,831–187,996) | |
| Regions | 43 | 63 | 77 | |
| ARI topics, all | 0.455 | 0.497 | 0.445 | |
| ARI sub-topics, all | 0.533 | 0.738 | 0.786 | |
| ARI sub-topics, members | 0.612 | 0.802 | 0.866 | |
| Topics / sub-topics leading a region | 28 / 43 | 38 / 63 | 40 / 77 | |
| Unclustered, all / sample | 0.17% / 8.5% | 0.36% / 7.1% | 1.09% / 10.9% | |
| Neighbour overlap, 1,000 random | 4.5% | 5.4% | 5.5% | |
| None shared | 68.5% | 62.3% | 61.8% | |
| Placed at the same position as another | 35.7% | 13.3% | 5.4% | |
| Placed below the threshold | 91.6% | 84.8% | 77.5% | |

- **Larger samples recover more sub-topics.** Sub-topic agreement rises from 0.53 to 0.74 and 0.79, because more regions are allowed, and most new regions are whole sub-topics. Topic agreement stays near 0.45–0.50.
- **They do not fix neighbours.** Overlap moves from 4.5% to 5.5%.
- **They reduce exact stacking a lot.** Memories at the same position fall from 36% to 5%.
- **They cost a lot.** 79 s at 5,000 and 185 s at 10,000, against 33 s. holomap's time grows faster than its input: 2.2 times for 2.5 times the points, then 3.3 times for 2 times. Placement grows with the sample, because each placed memory scans every sample row.
- **A sample of 20,000 crashes holomap 0.3.0.** All three runs panicked in the 5-D projection after about 2.5 minutes, at `holomap-0.3.0/src/spectral.rs:77`: "range end index 99170 out of range for slice of length 79336". The panic is in `spectral_layout_single`. It asks `smallest_eigenpairs` for 6 eigenvectors (the dimensions plus one), receives 4, and slices the fifth without checking the count. The slice lengths put holomap's graph at 19,834 points, not the 20,000 sampled. Why the graph is smaller, and why the solver returned fewer eigenvectors than it was asked for, was not investigated. The panic is deterministic: it happened at the same point in all three runs, with the same seed. It was not worked around: that would have meant changing the input or the parameters. The product is not affected while its cap stays at 2,000.

### 7b. Placing one new memory without a rebuild

This keeps what the current run already has:

- each region's centre, from its sample members;
- the sample's 2-D positions.

A new memory then gets the region of its nearest centre and the mean position of its 15 nearest sample rows. This is what a rebuild computes for a memory outside the sample. 1,000 new memories were drawn by the same generator, created after the last day.

**Table 7b.** Replica path, 100,000 memories.

| Measure | Value |
| --- | --- |
| Rebuild of the base run: regions and coordinates | 36,103 ms (one run) |
| Centres from the stored run | 0.39–0.85 ms |
| Region and position for 1,000 new memories | 236 ms (232–238) for the 1,000; 0.24 ms each |
| New memories that would enter the sample on a rebuild | 22 of 1,000; expected 2,000 / 100,001 = 2.0% |
| One new memory outside the sample, full rebuild | 36,534 ms |
| Its rebuilt position equals the incremental one | Yes, bit for bit |
| Its rebuilt region equals the incremental one | Yes |
| Every other memory's region and position after that rebuild | Unchanged: same partition, same position bits |
| One new memory inside the sample, full rebuild | 35,964 ms; one old memory leaves the sample |
| Regions before / after | 43 / 42 |
| Agreement of the old and new partitions | ARI 0.665; 13.6% of memories change region |
| Displacement over the map's diagonal | Median 9.3%, 90th percentile 29.5%, maximum 64.4% |
| The same after Procrustes alignment | Median 12.7%, 90th percentile 26.3%, maximum 46.5% |

- **Outside the sample, the incremental placement is exact.** Because the sample is the 2,000 lowest keys and the centres come from sample members only, a memory outside the sample changes nothing else. Placing it costs 0.24 ms instead of 36 s.
- **Inside the sample, the whole map changes.** Each new memory enters with chance about 2,000 / n: 2% at 100,000, 20% at 10,000. When one enters, the 2-D layout is a different layout: a median memory moves 9% of the diagonal, and alignment by rotation, scale and shift does not bring it back. One region in seven changes its members.
- **Not measured: deletions.** By the code, deleting a memory outside the sample changes nothing else, and deleting one inside it brings the next key into the sample, which should behave like an entry. Edits were not measured either. An edit that re-embeds a sampled memory changes the projection's input.

### 7c. Clustering again inside each region

`regions()` at 100,000, then `regions()` again on the members of each of the 43 regions, with the product's settings. Each call samples up to 2,000 of the region's members and sets `min_cluster_size` from that sample. Quality comes from the first run.

**Table 7c.** Library path, 100,000 memories.

| Measure | Value |
| --- | --- |
| `regions()` on the whole project | 8,898 ms (8,701–9,123); peak working set 696 MiB |
| `regions()` inside each of the 43 regions, all together | 222,718 ms (222,712–225,129) |
| One region | Median 5,705 ms (5,676–5,765); slowest 7,755 ms (7,484–8,321) |
| Process peak working set | 736 MiB (736–737) |
| Sub-regions | 763 in all, 9 to 35 per region; every region split |
| ARI sub-topics: top level / sub-regions | 0.533 / 0.097 |
| ARI sub-topics, each region's unclustered memories as one class | 0.290 |
| ARI topics: top level / sub-regions | 0.455 / 0.073 |
| Sub-topics that are the largest class of a sub-region | 191 of 200 |
| Unclustered: top level / sub-regions | 0.17% / 17.2% |

- **It over-splits.** There are 763 sub-regions for 200 sub-topics: about 18 per region, where each region holds about 4.7 sub-topics.
  - `min_cluster_size` comes from the sample, not the region. So every region of 2,000 or more members gets the same minimum as the whole project, 22. The smallest region, 827 members, gets 14.
  - Sub-topic agreement falls from 0.53 to 0.10, or 0.29 if each region's leftovers count as one class.
  - Almost every sub-topic (191 of 200) is the largest class of some sub-region, so the sub-topics are found but cut into pieces.
- **Unclustered rises to 17%.** A region of 2,000 members or fewer is its own sample, so nothing in it is assigned and HDBSCAN's leftovers stay unclustered, as on the real store.
- **It is slow.** It takes 223 s on one thread: 43 calls, a median of 5.7 s each. By table 1b, almost all of a call is holomap's 5-D projection of its sample.
- **Positions not measured.** This alternative computes regions only. Positions inside a sub-region were not computed.
- **Synthetic data.** Real sub-structure may split differently.

## What the figures support

- **The current pipeline runs at 100,000.** On this machine, one project of 100,000 memories maps in 36 to 48 s, with a peak under 1.8 GiB, and nothing fails.
- **Where the time goes.** Both projections cost the same at every size. The cost that grows is placement and assignment, which are linear and run on one thread.
- **The cache does not hold while capture runs.** Any change to the project's active memories means a full rebuild. Any write anywhere means 3.8 to 4.8 s of re-reading and hashing at 100,000. This follows from the code and was measured at both levels.
- **The sample sets the regions.** There are about 43 regions at every size. On this data their agreement with the planted structure does not fall as the project grows.
- **Positions lose neighbours with size.** Overlap falls from 54% to 4.5%. More than a third of placed memories share a point with another at 100,000.
- **Positions are cheap to send; text is not.** 100,000 positions are 1.4 MB in binary and 8.9 MB in JSON. Text at the real mean length is 29 MB, and reading the attributes takes 2.5 s.
- **Placing a new memory outside the sample needs no rebuild.** It is exact, bit for bit, and takes 0.24 ms. A memory that enters the sample changes the whole layout.
- **Larger samples are a partial answer.** They recover more sub-topics and stack less, but do not keep more neighbours. They cost 2.4 and 5.5 times as much. Holomap 0.3.0 cannot take 20,000.
- **Clustering again with the product's settings over-splits** on this data.
- **On real vectors the projection keeps about a third of the neighbours.** That holds when everything is sampled (786 memories). With a sample of 500, placed memories keep 12%.

## What they do not support

- **Region quality on a real project at scale.** The real store's largest project has 786 active memories, below the cap. The synthetic topics are isotropic and equally far apart, so the ARI figures measure recovery of clean structure only.
- **How real vectors behave at 100,000.** Real vectors are far more concentrated than the synthetic ones: a random pair's cosine is 0.665 against 0.270. Sampling, assignment and placement were not measured on vectors like that at scale.
- **What a better layout could reach.** The generator's sub-topics vary along 8 directions plus full-rank noise, so part of the neighbour loss belongs to the data. The figures do not say what any 2-D layout could keep on it, or how much another placement method would gain.
- **That over-splitting is general.** It may come from the generator, or from setting `min_cluster_size` from a region's sample.
- **Other machines.** Everything ran on one 16-core desktop under Windows. The single-thread stages would be slower on a smaller machine, and no other system was measured.
- **The daemon's extra 9 s.** It is measured but not explained, so whether it is inherent is unknown.
- **Sizes and counts beyond those measured.** Nothing ran above 100,000 memories, or with more than one large project, or with more than one request at a time.
- **How the map would look.** Nothing was drawn. The clumping figures count shared points, not what a viewer would show.
- **A rebuild rate under a real viewer.** It is inferred from six days of one store's write times.
- **Any choice.** B-261 records findings and decides nothing.

## Not done

- **`manifolds-rs`.** SPEC 7 names it as the faster alternative, but it is not a workspace dependency. Adding it would change the frozen root pins.
- **A sample of 20,000.** Holomap 0.3.0 panics at that size, and a workaround would have changed the input or the parameters.
- **Concurrency.** Every figure is one request at a time, with no capture writing during it.
- **The daemon's extra 9 s.** It was not investigated.
- **Deleting or editing a sampled memory (7b).** Only reasoned from the code.
- **Positions inside sub-regions (7c).** Only `regions()` was run.
- **Ties for the larger samples (7a).** Those runs came before the tie-inclusive measure was added.
- **Fresh processes for the real store.** Its three runs share one process. The copy was deleted once measuring ended, so they were not repeated.
- **A real project above 2,000 memories.** None exists.
- **More than 100,000 memories, or more than one chunk per memory.**
- **A viewer.** The binary format exists only as an encoder in the example. There is no decoder, no renderer and no product change.
- **Parallel placement or assignment, or any other change to make a number faster.** The spike measures; it does not tune.
- **Linux.** It was not run there.
