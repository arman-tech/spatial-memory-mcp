# Stream C report: `sm-spatial`

Branch `arman-tech/stream-c-spatial`, from `v3` at `ea82480`. Commits: `cf81032` (fixture), `c95c12f` (crate and `Cargo.lock`), and this report.

Abbreviations, written out once: ARI is the adjusted Rand index. HDBSCAN is hierarchical density-based spatial clustering of applications with noise. UMAP is uniform manifold approximation and projection; `holomap` implements it. ONNX is Open Neural Network Exchange, and ORT is ONNX Runtime. FNV is Fowler–Noll–Vo, and SHA-256 is the 256-bit Secure Hash Algorithm 2. PCG is the permuted congruential generator. Slerp is spherical linear interpolation, and lerp is linear interpolation. LTO is link-time optimisation. WSL2 is Windows Subsystem for Linux 2.

## 1. Done when

**Yes.** AT-6 (determinism) and AT-7 (clustering quality) pass on Windows and on Linux (Docker). The full gate passes on Windows: fmt, clippy `-D warnings`, `cargo test -p sm-spatial` and `cargo test --workspace`.

## 2. Files built

| Path | What it is |
|---|---|
| `crates/sm-spatial/Cargo.toml` | The new crate. It depends on `sm-core`, `holomap` and `hdbscan` from the workspace pins, plus `sm-core/test-support` and `serde_json` for development. No new pins. |
| `crates/sm-spatial/src/lib.rs` | Crate documentation and re-exports. |
| `crates/sm-spatial/src/projector.rs` | `HolomapProjector`, which implements `sm_core::traits::Projector`: cosine, spectral init, seeded, refuses more than 2,000 inputs. |
| `crates/sm-spatial/src/clusterer.rs` | `HdbscanClusterer`, which implements `sm_core::traits::Clusterer`. It uses Euclidean distance on the projected points, `max_cluster_size`, `min_samples` from `rust_min_samples()`, and noise as `None`. Parameters are checked before they reach the crate. |
| `crates/sm-spatial/src/pipeline.rs` | The SPEC 7 per-project pipeline: `regions`, `coordinates`, `MapParams`, `RegionRun`, `RegionResult`, `RankedMember`, `input_hash`, `sample_order`, and the `ALGORITHM` and `CENTRALITY` texts. |
| `crates/sm-spatial/src/journey.rs` | `journey` over `&dyn VectorIndex`, plus `slerp_path`, `JourneyRequest`, `Endpoint` and `PathStep`. |
| `crates/sm-spatial/src/vector.rs` | The memory vector (the re-normalised mean of its chunks) and cosine. All sums run in a fixed order. |
| `crates/sm-spatial/src/quality.rs` | `adjusted_rand_index` in scikit-learn's pair-confusion form, for the AT-7 report. |
| `crates/sm-spatial/tests/at6_determinism.rs` | AT-6: 4 tests and two golden digests. |
| `crates/sm-spatial/tests/at7_quality.rs` | AT-7: 3 tests, covering the fixture hash, the fixture shape and the quality gate. |
| `crates/sm-spatial/tests/s7_pipeline.rs` | 16 tests of the SPEC 7 rules: formula table, under 30, the 2,000 cap, sampling, `input_hash`, coordinates and journey. |
| `crates/sm-spatial/tests/common/mod.rs` | Fixture readers and fake-embedder memories, shared by the tests and the budget example. |
| `crates/sm-spatial/examples/budget.rs` | Release-build timings for section 5. |
| `fixtures/spatial/texts.json` | 320 developer-memory texts I wrote, in 10 topics of 32. |
| `fixtures/spatial/vectors.f32le` | 320 × 768 little-endian `f32` vectors from EmbeddingGemma 2 at revision `daa72c51…`. SHA-256 `d7845752b6dafaeb4f9976b3bf8238766bc18571618aa2a0421c8815b859ead8`. |
| `fixtures/spatial/README.md` | The model, revision and file pins, the prompt, the regeneration command, the hash and the topics. |
| `fixtures/spatial/generator/` | A standalone Cargo project with an empty `[workspace]` and its own `Cargo.lock`, based on `spike-embed`. It reads the model in place and never copies it. Its `target` folder was deleted. |
| `Cargo.lock` | The new packages listed in section 10. Existing entries changed only where a second major version needed disambiguation: `"syn"` became `"syn 3.0.10"` and `"thiserror"` became `"thiserror 2.0.21"`. No existing version changed. |

## 3. Test commands and exact results

**Windows.**
- Machine: Windows 11 Pro 10.0.26300, x86_64, AMD Ryzen 9 9950X3D (16 cores, 32 logical), rustc 1.95.0, PowerShell.
- `cargo fmt --check` exited 0.
- `cargo clippy --workspace --all-targets --all-features -- -D warnings` exited 0.

`cargo test -p sm-spatial --all-features`:

```
test result: ok. 24 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.64s   (lib)
test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 3.00s    (at6_determinism)
test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.50s    (at7_quality)
test result: ok. 16 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 7.70s   (s7_pipeline)
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s    (doc-tests sm_spatial)
```

`cargo test --workspace --all-features`:

```
test result: ok. 96 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.03s   (sm_core)
test result: ok. 24 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.66s   (sm_spatial lib)
test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 3.48s    (at6_determinism)
test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.65s    (at7_quality)
test result: ok. 16 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 10.11s  (s7_pipeline)
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s    (doc-tests sm_core)
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s    (doc-tests sm_spatial)
```

**Linux (Docker).**
- Command: the RULES section 6 command, `cargo test -p sm-spatial --all-features --locked && cargo clippy -p sm-spatial --all-targets --all-features --locked -- -D warnings`.
- Image: `rust:1.95-trixie`, rustc 1.95.0 (59807616e 2026-04-14), on Linux 6.6.87.2-microsoft-standard-WSL2, x86_64, the same CPU with 32 logical cores visible.
- Docker exited 0, and clippy passed.

```
test result: ok. 24 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.69s   (lib)
test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 3.16s    (at6_determinism)
test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.54s    (at7_quality)
test result: ok. 16 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 9.02s   (s7_pipeline)
test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s    (doc-tests sm_spatial)
```

A `--nocapture` rerun on Linux printed the same AT-6 golden digests and the same AT-7 figures as Windows.

## 4. Acceptance tests

**AT-6, determinism: passes on Windows and Linux.**

| Test | Checks |
|---|---|
| `at6_fixture_runs_are_identical_on_every_run_and_thread` | The fixture run twice, then 4 threads at once, are bit-identical: assignments, centralities, centres and 2-D coordinates. It also matches two golden SHA-256 digests computed on Windows and met unchanged on Linux. Assignments: `425378230d48adc5bdb43be6ca9ac18f451f2f49e8300e23c873191b8c21ae0b`. Every number in the run: `ab492e4b5c89aa71078739eda9df018392c9d999b8c5f310c60be34b9ba907f8`. |
| `at6_fake_embedder_runs_are_identical` | 300 fake-embedder memories, run twice and on 4 threads, are bit-identical. |
| `at6_input_order_does_not_change_the_run` | Reversed and interleaved input give the same run. |
| `at6_sampled_runs_are_identical` | The sampled path, with a cap of 150 on 400 memories, is bit-identical across runs. |
| `at6_projection_is_bit_identical_across_runs` | Unit test: the projector alone. |

`holomap` and `hdbscan` are both serial (decision 1), so there is no thread-count setting to vary. "A second thread count" is tested as concurrent calls from several threads.

**AT-7, clustering quality: passes on Windows and Linux** (`at7_clustering_quality_on_the_fixture`; fixture guards: `at7_fixture_hash_matches_readme`, `at7_fixture_has_ten_topics_of_at_least_30`; ARI check: `at7_ari_matches_scikit_learn`).

| Measure | Value | Gate |
|---|---|---|
| Memories | 320 (10 topics × 32) | |
| `min_cluster_size` / `max_cluster_size` | 9 / 80 (`ClusterParams::for_n(320)`) | |
| Clusters | 9 | |
| Unclustered | 25 (7.8%) | ≤ 25%: pass |
| Largest cluster | 57 (17.8%) | ≤ 40%: pass |
| ARI against the topics | 0.734 | none (SPEC 7 measured 0.55 to 0.59 on other data) |

Topic order: postgres, sqlite, auth, ci, kubernetes, frontend, testing, observability, rust, git.

| Region | Size | Members |
|---|---|---|
| 0 | 57 | ci 26 + git 31. The two topics merged. |
| 1 | 34 | observability 30, plus auth 2, ci 1 and kubernetes 1 |
| 2 | 34 | postgres 32, plus auth 1 and rust 1 |
| 3 | 32 | sqlite 32 |
| 4 | 31 | kubernetes 29, plus ci 2 |
| 5 | 30 | auth 28, plus ci 1 and testing 1 |
| 6 | 28 | testing 26, plus rust 2 |
| 7 | 27 | frontend 26, plus testing 1 |
| 8 | 22 | rust 22 |

The 25 unclustered memories are: rust 7, frontend 6, testing 4, ci 2, kubernetes 2, observability 2, auth 1, git 1.

**SPEC 7 rule tests (all pass).**

| Rule | Tests |
|---|---|
| Formula table (500 → 11, 1,000 → 16, 5,000 → 35) | `s7_min_cluster_size_formula_table` |
| No clustering under 30 | `s7_no_clustering_under_30_memories` (lib and integration) |
| 2,000 cap, deterministic sampling, nearest centre (3,000 fake vectors) | `s7_cap_samples_2000_and_assigns_the_rest_to_the_nearest_centre`, `s7_sample_changes_by_at_most_one_member_when_a_memory_is_added`, `s7_projector_input_is_capped_at_2000`, `s7_projector_refuses_more_than_2000_inputs` |
| `input_hash` sensitivity | `s7_input_hash_changes_with_any_id_or_parameter`, `s7_input_hash_is_order_independent_hex` |
| Journey endpoints, unit steps, step count | `s7_journey_endpoints_are_its_two_memories`, `s7_journey_steps_are_unit_length_and_evenly_spaced`, `s7_journey_step_count_is_as_requested`, `s7_journey_picks_the_nearest_memory_not_already_on_the_path`, `s7_journey_passes_scope_and_exclusions_and_leaves_empty_steps`, `s7_journey_rejects_one_memory_twice_and_opposite_vectors` |
| Regions, central members, coordinates | `s7_regions_are_numbered_largest_first_with_ranked_members`, `s7_central_members_are_ranked_by_cosine_to_the_centre`, `s7_coordinates_cover_every_memory_in_id_order`, `s7_coordinates_for_two_and_three_memories`, `s7_sampled_coordinates_place_the_rest_among_the_sample`, `s7_coordinates_refuse_a_run_of_other_memories_or_parameters` |

sm-spatial has 47 tests in total.

## 5. Measured budgets

SPEC 11 sets no budget for `regions`; these figures are measurements only.

- **Machine:** AMD Ryzen 9 9950X3D, 16 cores (32 logical), Windows 11 Pro 10.0.26300, x86_64.
- **Build:** `release` profile (thin LTO, `codegen-units = 1`).
- **Threads:** one. Both libraries are serial.
- **Method:** median of 3 runs.
- **Command:** `cargo run --release -p sm-spatial --example budget`.
- **Inputs:** fake memories come from the fake embedder in 14 topics, with one chunk each.

`regions` projects to 5 dimensions and clusters. `coordinates` runs the separate 2-D projection.

| Input | Clusters | Unclustered | Largest | `regions` | `coordinates` |
|---|---|---|---|---|---|
| AT-7 fixture, 320 | 9 | 7.8% | 17.8% | 458.1 ms (runs 464.3, 458.1, 454.7) | 443.0 ms |
| Fake, 300 | 25 | 0% | 9.3% | 375.3 ms | 364.6 ms |
| Fake, 2,000 | 39 | 1.7% | 3.8% | 6,697.0 ms | 6,538.5 ms |
| Fake, 5,000 (sampled; 2,000 projected) | 36 | 1.5% | 3.8% | 6,636.9 ms (runs 6,616.0, 6,636.9, 7,218.0) | 6,850.9 ms |

- **Where the time goes:** at 2,000 points, `holomap`'s spectral initialisation does a dense symmetric eigendecomposition, which dominates the time.
- **A cold call at 2,000 memories:** `regions` with `coords: true` is about 13 s.
- **Linux:** I did not time it.

## 6. Decisions

Each needs a B-n entry from the lead.

1. **No thread pinning (SPEC 7, SPEC 11 AT-6).**
   - *What:* `hdbscan` keeps its `parallel` (rayon) feature off; it uses its default serial build. `holomap` 0.3.0 has no threads.
   - *Why:* a run is single-threaded wherever it is called from, so there is no thread count to pin.
   - *Evidence:* `holomap` uses exact k-nearest-neighbours and draws from one seeded PCG64 stream through `libm`. AT-6 passes with 4 concurrent threads, and the golden digests are equal on Windows and Linux.
2. **Centrality (SPEC 7, Labels).** A member's centrality is its cosine to its region's centre. The centre is the re-normalised mean of all members' vectors in 768 dimensions; ties go by memory id.
   - `RegionResult.members` holds every member, fully ranked, and `central(k)` gives the top k. The text is in the `CENTRALITY` constant.
   - *Why:* it is simple, deterministic, uses the original space (not the projection), and the labeller (B-32) wants the most typical members.
   - *Evidence:* `s7_central_members_are_ranked_by_cosine_to_the_centre`.
3. **Sample (SPEC 7, over 2,000).** Each id is keyed by `fnv1a64` of the seed (u64, little-endian) followed by the id's 16 bytes. Ids are sorted on that key ascending, ties by id, and the first 2,000 are taken. The seed is `ProjectParams.seed` (42).
   - *Why:* no random-number crate is pinned, and the sample is stable as a project grows.
   - *Evidence:* adding one memory changes at most one sample member (`s7_sample_changes_by_at_most_one_member_when_a_memory_is_added`).
4. **Nearest-centre assignment (SPEC 7, step 3 and pitfalls).**
   - Each cluster's centre comes from its sample members only.
   - Each non-sample memory goes to the cluster with the highest cosine in 768 dimensions; ties go to the lower cluster.
   - There is no distance threshold, following the SPEC pitfall.
   - Sample noise stays unclustered. A run whose sample has no cluster leaves every memory unclustered.
   - So in a sampled run, the unclustered memories are always sample members.
5. **Region numbering (SPEC 4, SPEC 7).**
   - Regions are numbered by size, largest first, with ties broken by the smallest member id. The numbers are local to one run.
   - Under 30 memories, the whole project is region 0 and `clustered` is false.
6. **`input_hash` (SPEC 7, step 4).**
   - It is the SHA-256 (hex) of: the `ALGORITHM` text (`sm-spatial regions v1; holomap 0.3.0; hdbscan 0.12.0`), every field of both projections, `max_points`, `min_memories`, the cluster-parameter formula and method texts, n, and the sorted ids.
   - It covers ids, not vectors, so stream E must drop a cached run when a member is edited or re-embedded.
7. **Coordinates (SPEC 8, `regions` with `coords`).**
   - *Method:* a second `holomap` projection with the same parameters at 2 dimensions, over the same sample, with neighbours capped at the number projected minus 1.
   - *Memories outside the sample:* each sits at the mean position of its 15 nearest sample memories, by cosine in 768 dimensions.
   - *Small runs:* 1 memory sits at (0, 0). 2 memories sit at (0, 0) and (1 − cos, 0).
   - *Output:* `coordinates(memories, params, run)` returns `sm_core::protocol::MapPoint` with each point's region from `run`. It refuses a run made from other memories or parameters, which `s7_coordinates_refuse_a_run_of_other_memories_or_parameters` tests.
   - *Caveat:* the 2-D map is a different projection from the 5-D one that was clustered, so a region need not look contiguous on it.
8. **Journey: no repeats (SPEC 7, Journey).** Step 0 is `from` and the last step is `to`. Each step in between is the nearest memory that is neither endpoint nor already on the path.
   - Steps fill in order from `from`, so an earlier step wins a shared memory.
   - A step with no candidate left is empty (`memory: None`).
   - The caller's own exclusions are kept.
   - The score is the index's `MemoryHit` score for intermediate steps and the cosine to the endpoint vector at the ends.
   - *Why:* sm-core's `JourneyStep.memory` says "The nearest memory not already on the path, if any." See deviation 1.
9. **Journey geometry (SPEC 7, Journey).**
   - *Interpolation:* `t = i / (steps − 1)`, with slerp in `f64` that falls back to normalised lerp below an angle of 1e-6 rad, version 1's threshold.
   - *Refusals:* opposite endpoints (cos < −1 + 1e-6) and the same memory at both ends give `invalid_input`.
   - *Determinism:* it uses the platform's `sin` and `acos`, so it is not under AT-6.
10. **Input validation (SPEC 8).**
    - The following give `invalid_input` with a next action: duplicate ids, a memory with no chunks, non-finite values, zero vectors, and mismatched lengths.
    - `MapParams::validate` requires:
      - `min_memories` above the projection's neighbours and at least `rust_min_samples()`.
      - `max_points` at most 2,000 and at least `min_memories`.
      - `coords.dims` equal to 2.
    - The clusterer checks its parameters before calling `hdbscan`, whose builder would otherwise print to standard output (the MCP channel) and clamp them.
11. **The fixture (SPEC 11, AT-7).**
    - *Texts:* 320 texts I wrote, with close topic pairs on purpose: postgres and sqlite, ci and testing, ci and kubernetes.
    - *Measured separation:* the nearest neighbour of a text (by cosine) shares its topic for 79.7% of texts.
    - *Reference:* the committed vectors are the reference, because ORT may differ in the last bits between machines.
12. **ARI (SPEC 11, AT-7).** I wrote my own implementation in scikit-learn's pair-confusion form, with unclustered memories as one more label, as the SPEC's measurement did with HDBSCAN's `-1`. It is checked against `adjusted_rand_score` values in `at7_ari_matches_scikit_learn`.

## 7. Deviations

1. **Journey repeats.**
   - *SPEC 7 says* `journey` "stays as in version 1". Version 1's `journey` (`git show master:spatial_memory/services/spatial.py`) searched each step on its own and returned up to `journey_neighbors_per_step` neighbours per step, with no visited set. So one memory could fill several steps.
   - *The contract says* sm-core's `JourneyStep` holds one `memory`, documented as "The nearest memory not already on the path, if any".
   - *What I did:* I followed sm-core, because the code wins (CONTRACTS.md, RULES). The result is one memory per step with no repeats (decision 8).
2. **Opposite endpoints.** Version 1's `_slerp_interpolate` did not guard against them: it divides by `sin(omega)` ≈ 0 when the angle is π. Version 3 refuses them with `invalid_input` (decision 9).
3. **Cluster parameters for sampled runs.**
   - *What I did:* a run over 2,000 memories clusters its 2,000-member sample with `ClusterParams::for_n(2000)`, which gives 22 and 500. `for_n(total)` would give 35 and 1,250 at 5,000.
   - *SPEC 7 says:* its table defines the formula on n memories, but it is silent on which n applies to a sample.
   - *Why:* a maximum of 1,250 on 2,000 points allows a cluster of 62.5%, beyond AT-7's 40% limit, so it would defeat `max_cluster_size`.
   - *Evidence:* the 5,000-memory run gives 36 clusters, 1.5% unclustered and a largest share of 3.8%. `for_n(5000) = 35` is still tested directly. I did not measure `for_n(total)` on a sample for comparison.

## 8. Contract changes requested

None.

Three notes for the lead and stream E:

- **Journey wording.** The disagreement between SPEC 7 ("stays as in version 1") and `JourneyStep.memory` ("not already on the path") should get a B-n entry, so the SPEC text matches the code.
- **Mapping to protocol types.** sm-spatial returns ids, not `MemoryView`s. Stream E maps:
  - `PathStep` to `JourneyStep`.
  - `RegionResult::central(k)` to `Region.central`.
  - `RegionRun.unclustered` to the `u32` count.
- **Invalidation.** `input_hash` keys a run by ids and parameters only. An edit or re-embed must invalidate it (decision 6).

## 9. Unfinished or unverified

- **Journey across operating systems:** bit-for-bit equality is not tested. It uses the platform's `sin` and `acos`, and its results are compared with tolerances.
- **Live embedding:** AT-7 runs on the committed fixture vectors. A rerun with live embedding is the lead's job in phase 3.
- **Budgets:** I timed them on Windows only.
- **`regions` with `coords: true` at 2,000 memories is about 13 s cold.** There is no budget to fail, but it is slow. Stream E should cache runs and compute coordinates only when asked. I did not evaluate `manifolds-rs`, which SPEC 7 names as the alternative.
- **Linux scope:** Linux ran `-p sm-spatial` (RULES section 6), not the whole workspace.

## 10. Licences

New packages in the workspace `Cargo.lock`. All are permissive.

| Crate | Version | Licence | Pulled in by |
|---|---|---|---|
| approx | 0.5.1 | Apache-2.0 | nalgebra, simba |
| autocfg | 1.5.1 | Apache-2.0 OR MIT | num-traits, matrixmultiply (build scripts) |
| bytemuck | 1.25.2 | Zlib OR Apache-2.0 OR MIT | safe_arch, wide |
| glam | 0.30.10, 0.31.1, 0.32.1, 0.33.12 | MIT OR Apache-2.0 | nalgebra's optional `glam*` features. Lock entries only, never compiled (absent from `cargo tree`). |
| hdbscan | 0.12.0 | MIT OR Apache-2.0 | sm-spatial |
| holomap | 0.3.0 | MIT OR Apache-2.0 | sm-spatial |
| kdtree | 0.7.0 | MIT OR Apache-2.0 | hdbscan |
| libm | 0.2.16 | MIT | holomap |
| matrixmultiply | 0.3.11 | MIT/Apache-2.0 | nalgebra |
| nalgebra | 0.35.0 | Apache-2.0 | holomap |
| num-bigint | 0.4.8 | MIT OR Apache-2.0 | num-rational's optional feature. A lock entry only, never compiled. |
| num-complex | 0.4.6 | MIT OR Apache-2.0 | nalgebra, simba |
| num-integer | 0.1.47 | MIT OR Apache-2.0 | num-rational |
| num-rational | 0.4.2 | MIT OR Apache-2.0 | nalgebra |
| num-traits | 0.2.19 | MIT OR Apache-2.0 | hdbscan, holomap, nalgebra |
| rand | 0.9.5 | MIT OR Apache-2.0 | holomap |
| rand_core | 0.9.5 | MIT OR Apache-2.0 | rand, rand_pcg |
| rand_pcg | 0.9.0 | MIT OR Apache-2.0 | holomap |
| rawpointer | 0.2.1 | MIT/Apache-2.0 | matrixmultiply |
| safe_arch | 1.2.0 | Zlib OR Apache-2.0 OR MIT | wide |
| simba | 0.10.2 | Apache-2.0 | nalgebra |
| sm-spatial | 3.0.0 | MIT | this stream |
| syn | 2.0.119 | MIT OR Apache-2.0 | thiserror-impl 1 (the workspace already had syn 3.0.10) |
| thiserror | 1.0.69 | MIT OR Apache-2.0 | kdtree (the workspace already had thiserror 2.0.21) |
| thiserror-impl | 1.0.69 | MIT OR Apache-2.0 | thiserror 1 |
| wide | 1.7.1 | Zlib OR Apache-2.0 OR MIT | simba |

The fixture generator has its own `Cargo.lock` and is not in the workspace. It uses the same `ort` and `tokenizers` pins as `spike-embed`. Its non-MIT/Apache licences are:

- **webpki-roots 1.0.12: CDLA-Permissive-2.0.** This is not on the permissive list. It arrives through the `ort-sys` build script's download of ONNX Runtime, which `docs/v3/LICENSES.md` already flags.
- subtle 2.6.1: BSD-3-Clause.
- ring 0.17.14: Apache-2.0 AND ISC.

The generator is a development tool, run once, and never shipped.
