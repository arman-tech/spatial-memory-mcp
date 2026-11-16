# Stream C: `sm-spatial`

## Goal

Build `crates/sm-spatial`: the map behind the `regions` and `journey` tools. It implements `sm_core::traits::Projector` with `holomap` and `sm_core::traits::Clusterer` with `hdbscan`, plus the per-project pipeline of SPEC 7 and the `journey` path. It is a pure library: no storage, no embedding, no I/O beyond reading its own test fixture. Stream E calls it, and caches its runs through `sm-store`.

**SPEC 14 done when:** "Determinism and clustering-quality tests pass" (AT-6 and AT-7).

## You own

- `crates/sm-spatial/` (new crate; depends on `sm-core` only, with `sm-core` `test-support` in `[dev-dependencies]`).
- `fixtures/spatial/` (new): the labelled clustering fixture that AT-7 requires you to create and commit, with its README and its generator.

## Read

- **SPEC:** section 7 (all of it: the pipeline, the parameter table, the Rust libraries, labels, `journey`, and the pitfalls; the two-level map is v3.1 and is not built), section 4 rule "Cluster numbers are never keys", section 8 rows `regions` and `journey`, section 11 (AT-6, AT-7), section 13 (the `holomap` row).
- **CONTRACTS.md:** sections 2 and 9 (`Projector`: at most `MAX_PROJECTION_POINTS` 2,000 inputs; `Clusterer`: `None` for noise; `ClusterParams::for_n`, `rust_min_samples()`; no clustering below `MIN_CLUSTER_MEMORIES` 30).
- **DECISIONS:** B-32 (region labels), B-39, B-40.
- **sm-core:** `traits` (`Projector`, `ProjectParams`, `Clusterer`, `ClusterParams`, `VectorIndex`, `VectorQuery`, `MemoryHit`), `protocol` (`RegionsParams`, `RegionsReply`, `Region`, `MapPoint`, `JourneyParams`, `JourneyReply`, `JourneyStep`), `hash` (`fnv1a64`, `sha256`), `ids::MemoryId`.
- **Version 1's journey:** `git show master:spatial_memory/services/spatial.py`, functions `journey` and `_slerp_interpolate`. Port the behaviour, not the structure.
- **Crate sources:** `~/.cargo/registry/src/*/holomap-0.3.0/` and `hdbscan-0.12.0/`. Check how each takes its metric, neighbours, minimum distance and seed, and whether `holomap` stays deterministic when it runs on several threads.

## Build

1. **`Projector` with `holomap`** (SPEC 7 step 2): cosine metric, 15 neighbours, minimum distance 0, seed 42, 5 dimensions, from `ProjectParams`. Refuse more than 2,000 inputs with `invalid_input`.
2. **`Clusterer` with `hdbscan`** (SPEC 7 step 3): `min_cluster_size = max(5, round(sqrt(n) / 2))`, `min_samples` 5 in Python terms, passed as 6 (`rust_min_samples()`), and `max_cluster_size = n / 4`, all from `ClusterParams::for_n`.
3. **The per-project pipeline** (SPEC 7 "Pipeline, per project"):
   - One vector per memory: the mean of its chunk vectors, normalised again.
   - Under 30 memories, no clustering: the project is one group.
   - Over 2,000 memories:
     1. Take a deterministic sample of 2,000. No random-number crate is pinned; derive the sample from the memory ids and the seed, for example by sorting on a seeded `fnv1a64` of each id.
     2. Project and cluster the sample.
     3. Assign every other memory to the nearest cluster centre in the original 768 dimensions.
   - Return each cluster's most central memories (SPEC 7 "Labels"); choose and record how "central" is measured. Return the noise memories as unclustered.
   - Optional 2-dimension coordinates for `regions` (SPEC 8), by the same projection with 2 output dimensions.
   - The `input_hash` that keys the cache (SPEC 7 step 4): a hash of the sorted memory ids and every parameter, so a change in either changes the key.
   - Cluster numbers are local to one run and never leave it (SPEC 4).
4. **`journey`** (SPEC 7): spherical interpolation between two unit vectors over the requested number of steps, and at each step the nearest memory through a `&dyn VectorIndex`, as version 1 did. Decide, from version 1, whether a memory already on the path may appear again, and record it.
5. **Determinism** (AT-6): the same input, in the same order, gives identical assignments on every run, on every thread count, on Windows and Linux. If `holomap` is not deterministic under threading, pin its thread count and record why.

## The AT-7 fixture

AT-7 needs vectors with real topic structure. Fake-embedder vectors have none, so the fixture carries real EmbeddingGemma 2 vectors computed once.

- **`fixtures/spatial/texts.json`**: at least 300 short texts that you write yourself, across 10 topics, at least 30 per topic, each with its topic label. Make them like a developer's memories: decisions, pitfalls, conventions, incidents and runbooks about software. Pick topics that a real store would hold side by side, some of them close (for example two database topics), so the test is not trivial. Do not copy text from a dataset or a website.
- **`fixtures/spatial/vectors.f32le`**: one 768-dimension vector per text, little-endian `f32`, row-major, in `texts.json` order. Embed each with the document prompt `title: none | text: {text}`; each text is one chunk.
- **The generator**: a standalone Cargo project under `fixtures/spatial/generator/` with an empty `[workspace]` table, so it is not a workspace member. `docs/v3/spikes/rmcp-stdio/` follows the same pattern. Base it on `C:\Users\jon\Documents\code-repo\spatial-memory-spikes\spike-embed\src\main.rs`, with the same `ort` and `tokenizers` pins. It reads the model from the per-user cache (`SPATIAL_MEMORY_MODEL_DIR` or the default path); it never copies the model. Its `target` folder is ignored by git; delete it after use.
- **`fixtures/spatial/README.md`**: the model and revision, the prompt, the command that regenerates the vectors, and the SHA-256 of `vectors.f32le`. A test checks that hash, so a changed file fails loudly.

The unit tests read the committed vectors; none loads the model. In phase 3 the lead reruns AT-7 with live embedding.

## Acceptance tests and other tests

- **AT-7**: on the fixture, unclustered memories are at most 25% and no cluster holds more than 40% of memories. Report the cluster count, the noise share, the largest cluster's share, and the adjusted Rand index against the labels; SPEC 7 measured 0.55 to 0.59 at 500 memories and up.
- **AT-6**: the pipeline run twice on the fixture gives identical assignments, and identical again on a second thread count. Also run it on fake-embedder vectors.
- **SPEC 7 rules**: the formula table (n = 500 gives `min_cluster_size` 11; n = 1,000 gives 16; n = 5,000 gives 35); no clustering under 30 memories; the 2,000 cap with deterministic sampling and nearest-centre assignment for the rest (test at a few thousand fake vectors); `input_hash` changes when an id or a parameter changes; the journey path's endpoints are its two memories, its steps are unit length, and its step count is as requested.
- **Budget**: there is no SPEC 11 budget for `regions`. Measure and report the time to project and cluster 300, 2,000 and 5,000 memories, the last with the sampling path, on a release build.
