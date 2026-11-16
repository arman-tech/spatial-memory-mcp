# Benchmark record

Every measurement behind the specification, taken in November 2026 on one machine: 2 virtual cores (Intel Xeon, 2.3 GHz), 7 GB of memory, Linux. Absolute speeds belong to that machine. The comparisons between options are what carry over.

## Embedding models

Ranking quality is nDCG@10 (normalized discounted cumulative gain at 10; 0 to 1, higher is better). Two sets of 150 queries: programming questions against 1,000 documents (prose) and natural-language-to-code search against 2,000 documents (code). Inputs capped at 512 tokens, two threads.

| Model | Prose | Code | Mean | Documents per second (prose) | Query, ms | Peak memory | Licence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EmbeddingGemma 2 | 0.826 | 0.878 | 0.852 | about 2.8 | about 55 | 2.0 GB | Apache 2.0 |
| EmbeddingGemma 1 | 0.838 | 0.793 | 0.815 | 4.0 | 39–48 | 2.1 GB | Gemma terms |
| Arctic Embed M v2 | 0.830 | 0.753 | 0.791 | 3.5 | 39–42 | 3.3 GB | Apache 2.0 |
| Arctic Embed M | 0.805 | 0.750 | 0.778 | 4.9 | 25–31 | 2.0 GB | Apache 2.0 |
| Arctic Embed M Long | 0.808 | 0.736 | 0.772 | 3.6 | 41–44 | 2.3 GB | Apache 2.0 |
| BGE base | 0.789 | 0.734 | 0.761 | 4.9 | 27–31 | 2.0 GB | MIT |
| MiniLM L6 | 0.786 | 0.692 | 0.739 | 39 | 3–4 | 0.6 GB | Apache 2.0 |
| Jina code | 0.625 | 0.839 | 0.732 | 3.7 | 35 | 1.8 GB | Apache 2.0 |
| BGE small | 0.772 | 0.690 | 0.731 | 13.6 | 9–10 | 1.0 GB | MIT |
| Nomic v1.5 | 0.777 | 0.621 | 0.699 | 3.3 | 36–40 | 2.6 GB | Apache 2.0 |
| Keyword search only | 0.631 | 0.507 | 0.569 | n/a | n/a | n/a | n/a |

- **EmbeddingGemma 2 speed** comes from a like-for-like sample of 120 documents: 3.4 documents per second and 55 ms per query, against 4.3 and 42 ms for Arctic Embed M Long on the same sample. Its full run used one thread by mistake.
- **Significance over 300 queries.** EmbeddingGemma 2 beats Arctic Embed M Long by 0.079 (95% interval 0.050 to 0.110) and EmbeddingGemma 1 by 0.036 (0.009 to 0.063). Arctic Embed M, M Long, M v2 and BGE base are within noise of each other.
- **8-bit build of EmbeddingGemma 2:** prose 0.827, 2.7 documents per second, 238 ms per query. Same quality, slower queries.
- **Shorter vectors (EmbeddingGemma 2):** at 768, 512 and 256 dimensions prose scored 0.826, 0.822, 0.825 and code scored 0.878, 0.863, 0.841.
- **Query prompt:** code-retrieval prompt 0.882 against 0.878 for the general prompt on the code set.
- **Correctness:** the Rust harness reproduced the model card's example scores within 0.01, and a text embedded alone and in a padded batch matched at cosine 1.00000.
- **Not run:** `granite-embedding-english-r2` (no ONNX files published). The third set was not run for the larger models.

## Keyword and vector fusion

Reciprocal rank fusion of FTS5 keyword results with vector results, on the prose set, at three vector-to-keyword weightings.

| Model | Vectors alone | 1:1 | 3:1 | 9:1 |
| --- | --- | --- | --- | --- |
| EmbeddingGemma 1 | 0.838 | 0.749 | 0.789 | 0.815 |
| Arctic Embed M Long | 0.808 | 0.745 | 0.779 | 0.796 |
| Arctic Embed M | 0.805 | 0.775 | 0.790 | 0.800 |
| MiniLM L6 | 0.786 | 0.740 | 0.775 | 0.779 |

Fusion lowered quality in every case, on both sets. The test queries are natural language; identifier-heavy queries were not tested. Fusion was not re-run with EmbeddingGemma 2.

## Chunk size

EmbeddingGemma 2, 150 queries. The buried-answer set places about 600 tokens of unrelated text in front of each answer (450 memories). The ordinary set is 600 long question-and-answer documents. Chunks are fixed token windows with one-eighth overlap; a memory's score is its best chunk.

| Approach | Buried answer | Ordinary long documents | Vectors (buried set) | Seconds to embed (buried set) | Peak memory |
| --- | --- | --- | --- | --- | --- |
| 256-token chunks | 0.840 | 0.971 | 2,062 | 690 | 1.5 GB |
| 512-token chunks | 0.756 | 0.979 | 1,112 | 729 | 2.1 GB |
| 1,024-token chunks | 0.528 | not run | 639 | 832 | 4.1 GB |
| Whole memory, up to 2,048 tokens | 0.467 | not run | 450 | 864 | 2.4 GB at batch of 2 |
| Truncate at 512 tokens, no chunking | not run | 0.987 | n/a | n/a | 2.0 GB |

256 against 512 on the buried set: +0.084 (95% interval 0.038 to 0.132). On the ordinary set: -0.008 (-0.019 to +0.002). The ordinary set is close to its ceiling, so it shows only that small chunks do no harm there.

## Cost by input length

EmbeddingGemma 2, full precision, two threads.

| Tokens per input | Seconds per input, alone | Seconds per input, batch of 8 | Peak memory, alone | Peak memory, batch of 8 |
| --- | --- | --- | --- | --- |
| 256 | 0.6 | not run | 0.7 GB | not run |
| 512 | 1.3 | 1.0 | 0.8 GB | 1.7 GB |
| 1,024 | 2.3 | 2.3 | 0.9 GB | 2.7 GB |
| 2,048 | 6.1 | 5.9 | 1.3 GB | 5.9 GB |
| 4,096 | 19.2 | not run | 2.6 GB | not run |
| 8,192 | failed under a 6 GB limit | not run | n/a | not run |

## Similarity scores and the floor

Cosine similarity from EmbeddingGemma 2. Each cell is the 5th percentile, median and 95th percentile across queries.

| Set | Relevant document | Best wrong document | Typical unrelated document |
| --- | --- | --- | --- |
| Prose | 0.72 / 0.80 / 0.88 | 0.70 / 0.76 / 0.82 | 0.57 / 0.61 / 0.64 |
| Code | 0.70 / 0.76 / 0.82 | 0.68 / 0.73 / 0.80 | 0.50 / 0.53 / 0.56 |

| Floor | Relevant kept, prose | Relevant kept, code | Documents passing per query, prose (of 1,000) | Documents passing per query, code (of 2,000) |
| --- | --- | --- | --- | --- |
| 0.65 | 99% | 99% | 149 | 21 |
| 0.70 | 97% | 95% | 25 | 4 |
| 0.75 | 82% | 59% | 4 | 1 |

## Storage engines

Nearest-neighbour search over 100,000 vectors of 768 dimensions, per query.

| Engine | Exact search | Approximate search | Finding |
| --- | --- | --- | --- |
| Plain SQLite table, index on project, distance in code | 3–5 ms within one project | n/a | Chosen for project-scoped search |
| `vec1` (SQLite) | 51–60 ms | 0.8–4.9 ms after 58 seconds of training | Chosen for cross-project search |
| `sqlite-vec` | 100–130 ms; 2–3 ms with a partition key | n/a | Fallback |
| `usearch` | 35–42 ms; 37–49 ms filtered | 0.5–2.8 ms | Fast, but a second store to keep consistent with SQLite |
| `vectorlite` | not run | 0.1 ms after a 170 ms load | Approximate only |
| libSQL | 190–210 ms | Index unusable: 49 seconds and growth from 21 to 788 MB at 5,000 vectors | Rejected |
| Turso | 181–193 ms | n/a | No FTS5 |
| Qdrant Edge | n/a | about 1 ms | Beta, single process |
| LanceDB | 330–460 ms unindexed | not run | 35-minute build, 224 MB binary |

SQLite served concurrent readers in separate processes while another process wrote. The recall of approximate indexes was not measured. Chroma is not embeddable from Rust and was not run.

Added 15 November 2026: exact search written in plain Rust over vectors held in memory took 21.9 ms across 100,000 chunks on one thread, and 0.4 ms within one project of 60 (1,667 chunks), using 307 MB for the vectors. The data was random, not real embeddings. `vec1` stays in version 3.0 by the owner's decision; this figure is recorded as the fallback if compiling `vec1` into the binary fails on Windows.

## Clustering

Labelled text with 14 true classes, Python reference libraries. Agreement is the adjusted Rand index.

| Memories | Pipeline | Clusters | Unclustered | Largest cluster | Agreement |
| --- | --- | --- | --- | --- | --- |
| 500–5,000 | HDBSCAN on raw 768 dimensions | n/a | 75–89% | n/a | n/a |
| 500 | UMAP to 5 dimensions, `min_cluster_size` 11 | 12 | 6.8% | 14.8% | 0.59 |
| 1,000 | UMAP to 5 dimensions, `min_cluster_size` 16 | 17 | 9.0% | 14.6% | 0.59 |
| 5,000 | UMAP to 5 dimensions, `min_cluster_size` 35 | 31 | 6.9% | 10.6% | 0.55 |
| 5,000 | UMAP to 5 dimensions, `min_cluster_size` 5 | 106 | 15.0% | 10.6% | 0.30 |

- Two-level map simulation: agreement 0.67 against 0.43–0.60 for one flat map. A fixed distance threshold on cluster centres failed. About 6% of memories were unclustered and would be lost from the top level unless carried up individually.
- Rust crates: `hdbscan` 0.12.0 matched the Python library once `min_samples` was raised by one. `umap-rs` gave near-random output in four input variations and was rejected, though a harness error cannot be ruled out. `holomap` 0.3.0 was tested to 2,000 points.

## Threads and worker sessions

EmbeddingGemma 2, 400 documents capped at 256 tokens.

| Setup | Documents per second |
| --- | --- |
| One session, 1 thread | 2.0 |
| One session, 2 threads | 3.7 |
| Two sessions, 1 thread each | 4.1 |
| One session, 4 threads (more threads than cores) | 3.3 |
| One session, 2 threads, spinning off | 3.7 |

ONNX Runtime's default thread setting ran one thread on this machine: 1.8 against 3.4 documents per second with two set explicitly.

## Not measured

- Anything on Windows or macOS.
- Scaling beyond two cores, and embedding on a graphics card.
- Real team memories. All retrieval sets are public question-and-answer data.
- Structure-aware chunking, and a 128-token limit.
- Recall of the `vec1` approximate index.
- Identifier-heavy queries against FTS5.

## Later measurements and checks

| What | Result |
| --- | --- |
| Query latency by length, warm, two threads | 57 ms at 16 tokens, 96 at 32, 119 at 64, 200 at 128, 388 at 256 |
| EmbeddingGemma 2 on the repository's own 30-memory, 15-query evaluation set | nDCG@10 0.987, mean reciprocal rank 1.000, hit-at-5 1.000, against a version 2 baseline of 0.974, 0.967 and 1.000 |
| Keyword search on identifiers, porter tokenizer | Quoted phrases matched `dbo.Orders_Archive`, `CS8602` and `OrderService.GetOrderAsync`. Unquoted text with a dot raised a syntax error |
| Version 1's medium-signal captures | Saved, not held for confirmation. The tier only sets importance: at least 0.7 for strong, at least 0.4 for medium |
| Version 2's hook changes | One 200,000-character safety cap replaced per-field limits; the prompt hook returns the top 5 at 200 characters each; five metrics each tied to a future decision |

## Spike results on the owner's machine, 16 November 2026

Windows 11 (build 26300), AMD Ryzen 9 9950X3D (16 cores, 32 logical processors), 93.6 GB of memory. Source: the `spatial-memory-spikes` repository, `results/windows-native/`, commit `30d9138`. Times are medians with the 95th percentile in brackets.

**Embedding, one session, by thread count (s5)**

| Threads | Bulk, chunks per second | 32-token query, ms | 128-token query, ms |
| --- | --- | --- | --- |
| 1 | 3.1 | 43 | 159 |
| 2 | 4.3 | 35 | 119 |
| 4 | 6.7 | 25 | 84 |
| 6 | 8.6 | 24 | 71 |
| 8 | 10.4 | 24 | 63 |
| 12 | 13.2 | 25 | 57 |
| 16 | 15.9 | 26 | 54 |
| 24 | 18.0 | 31 | 57 |
| 32 | 17.6 | 39 | 66 |

Peak memory with one session: 1,467 MB. Model load: 0.33 to 0.46 s.

**A 128-token query on 4 threads while a bulk session runs (s6)**

| Situation | Query, ms | Bulk, chunks per second |
| --- | --- | --- |
| Query alone | 45 (48) | none |
| Bulk on 8 threads, normal priority | 82 (100) | 14.4 |
| Bulk on 8 threads, lower priority | 76 (80) | 14.4 |
| Bulk on 32 threads, lower priority | 135 (141) | 16.8 |

Windows accepted the lower priority for all 40 bulk threads. Peak memory with two sessions: 2,047 MB. The 4-thread query took 84 ms in s5 and 45 ms in s6; the difference is unexplained.

**Process start and local socket**

| Measurement | Windows, ms | Linux containers on the same machine, ms |
| --- | --- | --- |
| Hook program started directly (s1) | 3.5 (4.0) | 0.4 (0.5) |
| Through Git Bash | 44 (51) | not applicable |
| Through Windows PowerShell 5.1 | 111 (122) | not applicable |
| Through PowerShell 7 | 162 (167) | not applicable |
| Python interpreter start | 37 (40) | 6.0 (6.2) |
| Whole hook path through the socket (s8) | 3.6 (5.1) | 0.6 (0.7) |
| Connect and round trip, in process | 0.09 (0.12) | 0.14 (0.18) |
| The same over loopback TCP | 0.27 (0.31) | 0.17 (0.22) |
| Cold daemon auto-start | 21 (24) | 3.2 (3.3) |

**`vec1`, 100,000 vectors of 768 dimensions, exact cosine search (s7)**

| Measurement | Windows | Linux containers on the same machine |
| --- | --- | --- |
| Exact search, ms | 172.5 (186.5) | 29 to 31 |
| 2,000-row range scan through the `vec1` table, ms | 22.5 (29.7) | 15 to 16 |
| Insert 100,000 rows, s | 0.60 | 0.31 to 0.39 |
| Rebuild as a flat index, s | 2.4 | 2.0 to 2.4 |

Both report AVX2 as active and both return the brute-force top 10 for all 20 queries.

**Why the Windows search is slow: checked in the author's Linux sandbox**

| SQLite setting | Exact search, ms | Read calls in one whole run |
| --- | --- | --- |
| Defaults (4 KB pages, 2 MB cache) | 59 | 6,013,463 |
| `PRAGMA mmap_size=1073741824` | 45 | 193 |
| `PRAGMA cache_size=-600000` | 61 | not counted |
| `PRAGMA page_size=65536` | 51 | not counted |

On Linux each query re-reads the whole table through 4 KB read calls, and memory-mapped input/output removes them. This did not explain the Windows figure: see the follow-up below, where the same setting changed nothing on Windows.

## Follow-up spikes on the owner's machine, 16 November 2026

Source: `results/windows-native/` in the `spatial-memory-spikes` repository at commit `2cc4720`. Medians, with the 95th percentile in brackets.

**`vec1` exact search under different SQLite settings, Windows (s7b)**

| Setting | Exact search over 100,000 vectors, ms | Setting confirmed in effect |
| --- | --- | --- |
| Defaults (4 KB pages, 2 MB cache) | 159.6 (174.0) | yes |
| `PRAGMA mmap_size=1073741824` | 157.2 (171.2) | yes |
| `PRAGMA cache_size=-600000` | 167.7 (192.3) | yes |
| `PRAGMA page_size=65536` | 85.4 (109.7) | yes |
| Both memory mapping and 64 KB pages | 90.1 (107.1) | yes |

Every variant returned the brute-force top 10. The cause of the gap to Linux (29 to 31 ms in containers on the same machine) is not known.

**Project-scoped search, Windows (s7b)**

An ordinary table with an index on `project_id`, 100,000 chunks in 60 projects with rows scattered, cosine distance computed in Rust over one project's 1,667 rows: 5.3 ms (5.5) by default and 5.3 ms (6.3) with memory mapping. Results matched brute force.

**Several sessions against one, same total threads (s9)**

| Sessions × threads | Chunks per second | Memory in use, MB |
| --- | --- | --- |
| 1 × 12 | 14.9 | 1,461 |
| 2 × 6 | 14.9 | 2,829 |
| 4 × 3 | 15.5 | 5,562 |
| 6 × 2 | 16.0 | 8,296 |
| 12 × 1 | 16.5 | 16,495 |
| 1 × 16 | 16.5 | 1,470 |
| 2 × 8 | 17.6 | 2,836 |
| 4 × 4 | 19.2 | 5,569 |
| 8 × 2 | 17.1 | 11,033 |
| 16 × 1 | 14.9 | 21,962 |

**The same 128-token query on 4 threads, session by session (s10)**

| Session | Query, ms |
| --- | --- |
| First session created in the process | 43.4 (44.9) |
| Second, after the first was dropped | 89.6 (91.9) |
| Third | 88.3 (92.2) |
| Fourth | 92.5 (94.8) |
| Fifth | 86.7 (91.5) |
| Sixth, before bulk work in the same session | 87.3 (90.4) |
| Sixth, after two bulk batches | 87.0 (89.5) |
| Seventh, with thread spinning on | 81.7 (86.9) |

This explains the earlier disagreement: the 4-thread session in s5 was the third in its process and the query session in s6 was the first. Why later sessions are slower is not known.
