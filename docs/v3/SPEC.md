# Spatial Memory MCP v3 — Rust rewrite specification

Nov 15, 2026 · @Arman

## 1. Summary

Version 3 rewrites Spatial Memory MCP (Model Context Protocol) in Rust as one binary: a long-lived daemon per user that holds the embedding model, plus thin clients for the MCP server and the Claude Code hook. It is built for software engineers who work across many repositories. The reference case is a team with 60+ .NET projects, each in its own git repository, that share databases. Version 3.0 keeps version 1's automatic capture: hooks record decisions and fixes from a session without being asked (section 16).

**Goals**

- **Recall inside the hook budget.** Warm recall returns in 300 ms or less and never blocks a prompt. Version 2 cold-started the hook in 10–11 seconds against a 5-second timeout.
- **Bounded memory.** Long text is chunked and never embedded whole. Version 2 exceeded 6.2 GB on one 8,192-token passage.
- **Engineering memory, not generic notes.** Memories are typed (decision, pitfall, convention, incident, runbook) and anchored to projects, services, databases, tables, endpoints and commits.
- **One store per user across all projects.** Search defaults to the current project and goes cross-project only on request.
- **Team sharing through git.** Shared memories are files in the repository and pass normal pull-request review.
- **Spatial features that earn their place.** `regions` and `journey` stay; the rest of the surface is cut by half.

**Non-goals for v3.0**

- No GraphRAG (graph retrieval-augmented generation) and no LLM (large language model) calls inside the server.
- No server or cloud component, and no multi-user database.
- No approximate vector index until one store passes roughly 100,000 chunks.
- No automatic extraction of links from code. That is v3.1.
- No reuse of stored vectors. Version 1 and 2 data is re-embedded on import.

## 2. Decisions and evidence

Every decision below was confirmed on 15 November 2026 and rests on a measurement, not on a library's reputation. Numbers come from a two-core cloud processor running Linux unless stated; absolute speeds will differ on a developer machine.

| Decision | Choice | Evidence |
| --- | --- | --- |
| Language and shape | Rust, one binary, a daemon plus thin clients | Version 2's hook cold-started in 10–11 seconds against a 5-second timeout. In Rust the model loads once per daemon in 1.5–2.5 seconds. An `rmcp` 3.10.0 MCP server built to 4.5 MB. |
| Storage | SQLite as system of record, `vec1` for vectors, behind a trait | Eight engines benchmarked at 100,000 vectors of 768 dimensions. Project-scoped exact scan 3–5 ms; `vec1` exact 51–60 ms; `sqlite-vec` 100–130 ms; LanceDB 330–460 ms unindexed with a 224 MB binary. |
| Keyword search | SQLite FTS5 as a separate exact-match path | Fusing keyword and vector ranks scored below vectors alone at 1:1, 3:1 and 9:1. |
| Embedding model | EmbeddingGemma 2, text part, full precision | Mean ranking quality 0.852 against 0.772 for Arctic Embed M Long, a gain of 0.079 (95% interval 0.050 to 0.110) over 300 queries. On code it scored 0.878 against 0.736. Apache 2.0. |
| Fallback model | Arctic Embed M through `fastembed` 7.7.0 | 0.778, equal to Arctic Embed M Long within noise, and about 35% faster. `fastembed` reproduced the Python vectors exactly (cosine 1.00000). |
| Chunk limit | 256 tokens by default, configurable 128–1,024 | On mixed-topic memories 256 scored 0.840, 512 scored 0.756, whole-memory embedding 0.467. On single-topic documents 256 and 512 were equal within noise. Section 5. |
| Clustering | UMAP to 5 dimensions, then HDBSCAN, fixed seed | Clustering raw vectors left 75–89% unclustered. The pipeline leaves 7–9% and agrees with true labels at 0.55–0.59. Section 7. |
| Cross-project structure | Two-level map plus a link table | In simulation the two-level map agreed with true topics at 0.67 against 0.43–0.60 for a flat map. Dependencies come from links, not from map distance. |
| GraphRAG | Not used | It needs an LLM at indexing time and does not help fact lookup. Link-walk expansion and agent-written cluster labels are borrowed from it. |
| Runtime threads | Set explicitly | ONNX Runtime's default used one of two cores: 1.8 against 3.4 documents per second. More threads than cores ran 11% slower. Two single-thread sessions beat one two-thread session by about 10%; a worker pool with self-tuning is version 3.1. Section 5. |

**Embedding benchmark, for reference.** Ranking quality on programming questions (prose) and natural-language-to-code search (code), 150 queries each.

| Model | Prose | Code | Mean | Licence |
| --- | --- | --- | --- | --- |
| EmbeddingGemma 2 | 0.826 | 0.878 | 0.852 | Apache 2.0 |
| EmbeddingGemma 1 | 0.838 | 0.793 | 0.815 | Gemma terms |
| Arctic Embed M v2 | 0.830 | 0.753 | 0.791 | Apache 2.0 |
| Arctic Embed M | 0.805 | 0.750 | 0.778 | Apache 2.0 |
| Arctic Embed M Long (version 2's model) | 0.808 | 0.736 | 0.772 | Apache 2.0 |
| BGE base | 0.789 | 0.734 | 0.761 | MIT |
| MiniLM L6 (version 1's model) | 0.786 | 0.692 | 0.739 | Apache 2.0 |
| Jina code | 0.625 | 0.839 | 0.732 | Apache 2.0 |
| Keyword search only | 0.631 | 0.507 | 0.569 | n/a |

## 3. Architecture

One daemon per user and profile owns the model and the database; every other process is a thin client that talks to it over a local socket. This is what removes the cold start: the model loads once (1.5–2.5 seconds measured), not once per hook call.

&#91;embedded content: architecture · 3 clients, 1 daemon, 2 stores\]

Clients reach the daemon only through the socket; only the daemon touches the database and the shared folder in the repository.

**Processes.** All are subcommands of one binary, `spatial-memory`.

| Subcommand | Role | Holds the model |
| --- | --- | --- |
| `daemon` | Serves requests on a local socket. Owns the ONNX (Open Neural Network Exchange) Runtime session and the SQLite database. | Yes |
| `mcp` | MCP server over standard input/output, built on `rmcp`. Forwards each tool call to the daemon. | No |
| `hook` | Claude Code `UserPromptSubmit` and capture hook client. For a prompt it asks the daemon for ambient recall and prints the injection. For PostToolUse, PreCompact, Stop and SessionStart it follows section 16. | No |
| `setup`, `doctor` | Download and verify the model, register the MCP server and hook, diagnose a broken install. | No |
| `stop` | Asks the profile's daemon to shut down and waits until it has gone. Never starts one (B-155). | No |
| `import`, `export`, `migrate`, `reindex` | Bulk paths. They queue work in the daemon and show progress. | No |

**Rules**

1. **Transport.** Unix domain socket on Linux and macOS, named pipe on Windows. Messages are length-prefixed JSON with a protocol version field. The socket lives in the profile directory with user-only permissions. Every request carries the protocol version, the client's working directory and, when the client has one, the session identifier; the daemon resolves the project from the working directory. A client whose version differs from the daemon's asks it to exit and starts a new one.
2. **Auto-start.** A client that finds no daemon starts one detached, guarded by a lock file so two clients cannot start two daemons.
3. **The hook never blocks.** Hard timeout 1,500 ms. On timeout, error or a daemon that is still loading, it prints nothing and exits 0.
4. **One inference at a time.** Each ONNX Runtime session runs one inference at a time. Queries have their own session and never queue behind indexing; bulk work uses a separate session (section 5).
5. **Explicit threads.** Thread counts come from a processor budget (section 5). They are never left to the default and never exceed the physical core count. ONNX Runtime's default used one thread on a two-core machine and halved throughput (1.8 against 3.4 documents per second).
6. **One writer.** SQLite runs in write-ahead logging mode with a single writer task and concurrent readers.
7. **Profiles.** `--profile work` and `--profile personal` use separate directories, databases and sockets. Nothing crosses between them.

**Crates.** One Cargo workspace. The split is also the boundary between build agents (section 14).

| Crate | Responsibility | Depends on |
| --- | --- | --- |
| `sm-core` | Domain types, configuration, errors, and the traits `Embedder`, `Chunker`, `VectorIndex`, `Projector`, `Clusterer`, `SecretScanner`. | none |
| `sm-store` | SQLite schema and migrations, FTS5 (full-text search), `vec1` vector search, anchors, links, map tables. | `sm-core` |
| `sm-embed` | EmbeddingGemma 2 on ONNX Runtime, tokenizer, chunker, model download and checksum. Optional `fastembed` fallback behind a Cargo feature. | `sm-core` |
| `sm-spatial` | UMAP and HDBSCAN pipeline for `regions`, `journey`, map caching. | `sm-core` |
| sm-capture | Automatic capture: per-tool text extraction, transcript reading, signal classification, secret scanner, spool format (section 16). | sm-core |
| `sm-daemon` | Socket server, services (remember, recall, related, share, sync), retrieval gates, indexing queue. | all of the above |
| `sm-cli` | The `spatial-memory` binary: MCP proxy, hook client, setup, doctor, bulk commands. | `sm-core`, `sm-daemon` |

Every external engine sits behind a trait in `sm-core`, so the model, the vector index and the projection library can each be replaced without touching callers.

## 4. Data model

SQLite is the system of record: one database file per user and profile, holding text, vectors, keyword index, anchors, links and maps. Vectors are stored as 768 little-endian 32-bit floats in a BLOB (binary large object) column.

| Table | Holds | Key columns |
| --- | --- | --- |
| `meta` | Schema version, embedding model identifier and dimension, chunk settings used to build the index. | `key`, `value` |
| `projects` | One row per project. Identity is the normalized git remote URL, or the root path when there is no remote. | `id`, `name`, `git_remote`, `root_path` |
| `memories` | The memory itself, stored whole. | `id` (UUID version 7), `project_id`, `type`, `status`, `title`, `content`, `content_hash`, `supersedes_id`, `origin` (local or shared), importance, `tags`, `imported`, `created_at`, `updated_at` |
| `provenance` | Where a memory came from. | `memory_id`, `author`, `source` (agent, human, capture, import, sync), `session_id`, `commit_sha`, `file_path` |
| `chunks` | The embedded pieces of a memory. | `id`, `memory_id`, `project_id`, `seq`, `text`, `token_count`, `embedding`, `model_id` |
| `chunks_fts` | FTS5 keyword index over chunk text and memory title. | columns title and text, one row per chunk, tokenizer porter unicode61, written in the same transaction as `chunks` |
| `anchors` | Things in the codebase a memory is about. | `id`, `kind` (project, service, database, table, endpoint, file, symbol, commit), `name`, `project_id` |
| `memory_anchors` | Which memory is about which anchor. | `memory_id`, `anchor_id` |
| `links` | Directed edges between anchors, or between memories. | `src_kind`, `src_id`, `dst_kind`, `dst_id`, `relation`, `source` (manual or extracted), `extractor`, `verified_commit`, `verified_at` |
| `map_runs`, `map_clusters`, `map_points` | Cached output of one clustering run. | `run_id`, `project_id` (null for the cross-project map), `params`, `input_hash` |
| `injections` | What ambient recall already injected in a session. | `session_id`, `memory_id`, `at` |

**Rules**

- **Types and status.** `type` is one of decision, pitfall, convention, incident, runbook, note. `status` is one of active, superseded, retracted, pending. Setting `supersedes_id` marks the older memory superseded in the same transaction. Recall returns active memories only unless asked otherwise. `pending` is set only by automatic capture and is never served: no search, walk, map or hook returns a pending memory, whatever statuses a caller asks for (B-157, section 16).
- **Memories are whole, chunks are derived.** Search runs on chunks and returns the parent memory. Chunks can always be rebuilt from `memories.content`.
- **Model stamp.** Every chunk records `model_id`. A daemon whose configured model or chunk settings differ from `meta` refuses to search until `reindex` has run.
- **Project-scoped search** reads the vectors `WHERE project_id = ?` through an ordinary index and scores them exactly. Measured at 3–5 ms with 100,000 chunks in the store.
- **Cross-project search** uses `vec1` in exact mode. Measured at 51–60 ms at 100,000 chunks. Its approximate index (0.8–4.9 ms after 58 seconds of training) is a v3.1 switch for stores past 100,000–200,000 chunks.
- **Mapped reads.** Reader connections map the database file with `mmap_size` just under 2 GiB (B-114). The writer is not mapped. The mapping has two limits (B-147):
  - **Only the first 2 GiB of the file is mapped.** That is about 273,000 chunks at the 7.7 KiB per chunk measured with short texts (750 MiB at 100,000 chunks), and about 200,000 full 256-token chunks by estimate. SQLite reads pages beyond it through its page cache, at the speeds measured with mapping off: up to about 4 times slower at 100,000 chunks (B-145). A store that large is also past the point where the approximate index is planned.
  - **An input/output error on a mapped page ends the daemon.** The operating system raises it as an exception that SQLite cannot turn into an error code. The next client starts a new daemon: a prompt hook starts one and injects nothing for that prompt, a capture hook spools its event and starts one, and an MCP call starts one and waits.
- **Links carry their evidence.** A machine-extracted link records the extractor and the commit it was verified against. A link whose `verified_commit` is no longer an ancestor of the project head is reported as stale, not deleted.
- **Cluster numbers are never keys.** They change between runs. Anything that must persist refers to memory identifiers.
- **Relations** are a closed set: calls, reads, writes, depends\_on, relates\_to, caused\_by. `supersedes` lives on the memory row, not in `links`.

* **Forgetting cascades.** Deleting a memory removes its chunks, keyword rows, anchor mappings, links, map points and injection records in one transaction. A shared memory cannot be forgotten locally; its file is removed in the repository.
* **Locations.** Data lives in the platform's per-user data directory under `spatial-memory/<profile>/` (under `%LOCALAPPDATA%` on Windows); `SPATIAL_MEMORY_HOME` overrides it. The profile is `default` when none is given. Model files are cached once per user, outside profiles and outside any repository.
* **Housekeeping.** `injections` rows older than 30 days are pruned. A schema migration or a `reindex` first copies the database file as a backup.

## 5. Embedding and chunking

Memories are embedded with the EmbeddingGemma 2 text model at full precision through ONNX Runtime, in chunks of at most 256 tokens by default, split along the text's own structure. The chunk limit is configurable; the model and the limit are stamped into the index, so changing either requires `reindex`.

**Model, pinned**

| Item | Value |
| --- | --- |
| Source | Hugging Face `onnx-community/embeddinggemma-2-ONNX`, Apache 2.0, not gated |
| Revision | `daa72c51243991dfcaf9f9137d2c573d8f7790c0` |
| `onnx/model.onnx` | 421,421 bytes, SHA-256 `bc47de15f81208a5c99e5ab10f746d5e33b51ea228b7dc0bef9c133a94f1c1c3` |
| `onnx/model.onnx_data` | 1,084,170,240 bytes, SHA-256 `9fd452bfc6916e5a92f080f8dada35598ca4512b21c65897d4c5974d5f9630a2` |
| `tokenizer.json` | 32,170,510 bytes, SHA-256 `4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4` |
| Inputs | `input_ids` and `attention_mask` as 64-bit integers, batch × length. `image_features`, `video_features`, `audio_features` as 32-bit floats of shape 0 × 512 for text-only use. |
| Output | `sentence_embedding`, batch × 768, already unit length |
| Query prompt | `task: search result \| query: {text}` |
| Document prompt | `title: {memory title, or "none"} \| text: {chunk}` |
| Tokenizer | Adds the begin and end tokens itself. Pad on the right with token 0. |
| Runtime | `ort` 2.0.0-rc.15 with its downloaded ONNX Runtime binaries, graph optimization level 3, intra-op threads set explicitly |

- The graph uses Microsoft contrib operators (`RotaryEmbedding`). The standard processor build of ONNX Runtime ran them on Linux; Windows is untested.
- `fastembed` cannot load this model because of the three extra inputs. The direct `ort` implementation is about 70 lines.
- Do not use the 8-bit build. Quality was the same (0.827 against 0.826) but queries took 238 ms against 55 ms.
- Keep all 768 dimensions. Cutting to 256 cost up to 0.037 on code retrieval and the storage saved is not needed at this scale.
- One query prompt is enough. The code-specific prompt scored 0.882 against 0.878 for the general one.

**Chunking rules**

1. Count tokens with the model's own tokenizer. The limit counts chunk content; the prompt and a title of at most 32 tokens come on top.
2. A memory at or under the limit is one chunk. Most are.
3. A longer memory is split on structure first: Markdown headings, then blank-line paragraphs, then sentences. Fenced code blocks stay whole. Adjacent pieces are packed together up to the limit, and never across a heading.
4. A single piece over the limit, such as a long log or code block, is split into token windows with one-eighth overlap.
5. Every chunk is embedded with the memory's title in the document prompt, so a chunk from the middle still carries what the memory is about.
6. No single inference ever exceeds the configured limit. Text is never embedded whole "because it fits the model's 8,192-token window".

**Configuration**

| Key | Default | Range | Changing it |
| --- | --- | --- | --- |
| `chunk.max_tokens` | 256 | 128–1,024 | Requires `reindex` |
| `chunk.overlap` | 0.125 | 0–0.25 | Requires `reindex` |
| `embed.model` | `embeddinggemma-2` | or `arctic-embed-m` with the fallback feature | Requires `reindex` |
| `embed.batch_size` | 4 | 1–16 | Takes effect at once |
| `cpu.budget` | 0.75 | 0.1–1.0 | Share of physical cores that bulk work may use. Takes effect at once |
| `embed.query_threads` | 4, or the physical core count if lower | 1 to physical cores | Takes effect on restart |
| `embed.spinning` | off | on or off | Takes effect on restart |
| `capture.max_index_tokens` | 8,192 | 256 and up | Takes effect on the next capture |

`embed.bulk_workers` and `embed.max_memory_gb` arrive with the worker pool in version 3.1.

**Using the machine's cores**

Thread counts come from a processor budget, not a fixed number, so the same build uses two cores on a small machine and most of sixteen on a workstation without making it unresponsive. The two-core figures in this document describe the test machine, not a limit.

|  | Queries (hook, `recall`) | Bulk work (import, migrate, reindex, sync) |
| --- | --- | --- |
| Input | One short text | Thousands of chunks |
| Sessions | One dedicated session, always loaded | One bulk session, created for the job and released when the queue is empty. A pool of sessions with self-tuning is version 3.1 |
| Threads | `embed.query_threads` | `cpu.budget` × physical cores |
| Priority | Normal | Below normal, so foreground programs and queries run first |

The thread budget counts the threads the daemon itself creates: its service and connection threads, and the ONNX Runtime threads it asks each session for. Threads that the operating system or a library starts on its own are not counted, such as the system thread-pool workers Windows adds to a process that waits on a named pipe (B-131, B-141).

1. **Count physical cores, not logical processors.** On a 16-core, 32-thread processor the ceiling is 16.
2. **Never run more threads than cores.** Four threads on two cores were 11% slower than two.
3. **Leave headroom by default.** With `cpu.budget` at 0.75, bulk work uses at most 12 of 16 cores. Set it to 1.0 for an overnight import.
4. **Bulk work yields.** The bulk session runs at below-normal operating-system priority, and a query never waits behind an indexing batch because it has its own session.
5. **Turn off thread spinning.** By default ONNX Runtime's idle threads spin and burn processor time. With spinning off, throughput under load was unchanged.
6. **Bound memory as well as threads.** One session peaked at about 1.3 GB on 256-token chunks, so version 3.0 needs no memory setting. In version 3.1, `embed.max_memory_gb` caps how many workers start.
7. **Measure, do not guess.** `doctor --bench` is version 3.1: it will time worker and thread combinations on the actual machine and save the fastest. In version 3.0, doctor reports chunks per second and query latency at the configured budget, and nothing is tuned automatically.

**Measured on the owner's 16-core machine, 16 November 2026.** One bulk session embeds 3.1 chunks per second on 1 thread, 6.7 on 4, 10.4 on 8, 13.2 on 12 (the default budget), 15.9 on 16 and 18.0 on 24. A 128-token query takes 159 ms on 1 thread, 84 ms on 4, 63 ms on 8 and 54 ms on 16; a 32-token query is already at its floor of 25 ms on 4 threads. A query session and a bulk session together peak at about 2.0 GB of memory. The defaults in this section are unchanged; the reasoning and the open points are under "Results on the owner's machine" in section 16.

| Setup on two virtual cores, 400 documents capped at 256 tokens | Documents per second |
| --- | --- |
| One session, 1 thread | 2.0 |
| One session, 2 threads | 3.7 |
| Two sessions, 1 thread each | 4.1 |
| One session, 4 threads (more threads than cores) | 3.3 |
| One session, 2 threads, spinning off | 3.7 |

Two single-thread workers beat one two-thread session by about 10%, at twice the memory. That gain is small and is not measured beyond two cores, so version 3.0 uses one bulk session. The worker pool and its self-tuning are version 3.1 (decided 15 November 2026), together with `doctor --bench`.

**Measured on the owner's 16-core machine, 16 November 2026 (spike s9).** With 12 threads in total, one session embedded 14.9 chunks per second and twelve single-thread sessions 16.5, for 16.5 GB of memory against 1.5 GB. Each session adds about 1.37 GB. Decided by the owner on 16 November 2026: the worker pool stays on the version 3.1 list, to be built only if a machine shows a clear gain.

**Why 256 and not 512.** Two tests on EmbeddingGemma 2, 150 queries each, ranking quality as nDCG@10 (normalized discounted cumulative gain at 10; 0 to 1, higher is better).

| Chunk limit | Answer buried after about 600 tokens of other material | Ordinary long documents | Vectors per 450 memories | Peak memory, batch of 8 |
| --- | --- | --- | --- | --- |
| 256 tokens | 0.840 | 0.971 | 2,062 | 1.5 GB |
| 512 tokens | 0.756 | 0.979 | 1,112 | 2.1 GB |
| 1,024 tokens | 0.528 | not run | 639 | 4.1 GB |
| Whole memory, up to 2,048 tokens | 0.467 | not run | 450 | 2.4 GB at batch of 2 |

- On memories that mix topics, 256 beats 512 by 0.084 (95% interval 0.038 to 0.132), and 512 beats embedding the memory whole by 0.289 (0.235 to 0.344). A long vector averages its topics, and the fact you are searching for is diluted.
- On ordinary single-topic documents the difference between 256 and 512 is within noise (0.008, interval -0.019 to +0.002), and the right document was in the top five 99.3% of the time at both sizes.
- Embedding time was the same at 256 and 512 (690 against 729 seconds for the same text). Smaller chunks cost twice the vectors, about 3 KB each, and nothing else.
- Cost grows faster than length above 1,000 tokens: one input of 512 tokens took about 1.2 seconds, 1,024 took 2.3, 2,048 took 6.1, 4,096 took 19, and 8,192 failed under a 6 GB limit.

Limits of this evidence: the buried-answer set is constructed (unrelated text placed in front of each answer) and used fixed token windows, not the structure-aware splitting specified above. Structure-aware splitting should narrow the gap between 256 and 512; that is not measured. A limit of 128 was not tested.

## 6. Retrieval and ambient recall

Vector search is the default path and keyword search is a separate exact-match path; the two are never score-fused. Fusing them scored below vectors alone at every weighting tested (1:1, 3:1 and 9:1) for every general-purpose model.

**Recall pipeline**

1. Embed the query with the query prompt. Queries are cut before embedding: to 128 tokens for the hook and to the chunk limit for explicit recall. Warm latency on the test machine was 57 ms at 16 tokens, 119 ms at 64, 200 ms at 128 and 388 ms at 256.
2. Score chunks exactly by cosine similarity, in the current project unless `scope` is `all`.
3. Collapse to memories: a memory's score is its best chunk's score.
4. Drop memories that are not active, and apply `type` and anchor filters.
5. Explicit recall marks results under the similarity floor as weak and still returns them; only the hook drops them. Return the top `k` (default 5) with scores, types, anchors and provenance.
6. If the query looks like an identifier (an error code, a symbol with `_`, `::` or `.`, a table name), also run FTS5, passing the query as a quoted phrase because raw text containing a dot or colon is a syntax error, and append exact matches that are not already present, marked as keyword hits.

**Similarity floor.** The floor is model-specific and must be recalibrated when the model changes. EmbeddingGemma 2 scores sit in a narrow, high band: unrelated same-domain text scored 0.50–0.64, relevant text 0.70–0.88. The default floor is 0.70, which kept 95–97% of relevant results in testing. A floor of 0.75 kept only 59–82%. Thresholds carried over from other models (0.3–0.5) would pass everything. The floor and the result cap stay provisional until the evaluation on real memories in section 11 has been run.

**Ambient recall (the hook).** The hook injects memory into a prompt without being asked, so it is gated harder than explicit recall.

| Gate | Default | Why |
| --- | --- | --- |
| Scope | Current project only | Cross-project results are noise unless requested. |
| Similarity floor | 0.70 | See above. |
| Result cap | 5 memories, version 2's default | The floor alone still passes several near-misses per query in a dense store. |
| Token budget | One line per memory, cut at 200 characters as in version 2; 800 tokens is the ceiling | Keeps the prompt cheap. Longer memories are cut to their best chunk. |
| No repeats | Once per session | Tracked in `injections` by session identifier. |
| Minimum prompt | 4 words | Skips "yes", "continue" and similar. |
| Deadline | 1,500 ms | On expiry the hook prints nothing. |

**Hook input and output.** The hook reads Claude Code's JSON payload from standard input and takes the prompt, the working directory and the session identifier from it. It writes the injection to standard output as plain text in version 2's format: the header line "\[spatial-memory\] ambient recall:", then one numbered line per memory with its score and its text cut at 200 characters. Version 3 adds the memory identifier to each line. Version 2's switches keep their names: SPATIAL\_MEMORY\_AMBIENT\_RECALL (on by default; any other value disables it) and SPATIAL\_MEMORY\_AMBIENT\_RECALL\_TOP\_K (default 5; zero disables it). SPATIAL\_MEMORY\_AMBIENT\_RECALL\_NAMESPACE has no meaning without namespaces and is replaced by SPATIAL\_MEMORY\_AMBIENT\_RECALL\_SCOPE (project or all). A disabled switch or an empty prompt returns before the daemon is contacted. The reference is `spatial_memory/hooks/user_prompt_submit.py` on the `feature/v2-implementations` branch; field names are confirmed against it and the current Claude Code hook documentation, not assumed.

**Metrics journal.** Version 2's journal design is kept: always on, one JSON line per event, appended atomically in lines under 4,096 bytes to day-stamped per-process files in the profile's metrics folder, pruned after 7 days, and silent on write errors. Each hook call records time, project, latency, top scores, injected memory identifiers and the gate that stopped it. Version 2's five metrics keep their names and their purposes: capture\_size\_tokens and capture\_cap\_hit\_total (how large captures are and how often the index cap cuts them), recall\_result\_count (whether query expansion is worth building), reinforced\_memory\_rank (whether a reranker is worth building), and embedder\_cold\_start\_seconds (set once per daemon start; a cold start on a request is a regression). The status tool returns their 7-day medians and 95th percentiles on request. Following version 2's method, version 3 adds one metric for each of its own unproven defaults: hook\_gate\_total by gate and hook\_top\_score (the similarity floor and the result cap), chunks\_per\_memory (the chunk limit), and recall\_weak\_result\_total (how often explicit recall returns only weak results). Prompt text is not logged unless `journal.include_prompts` is turned on. This is the only evidence for tuning the gates after release.

**Related.** `related` starts from a memory or an anchor and returns two lists: nearest memories by vector, and memories reached by walking `links` and `memory_anchors` up to two hops. This is the graph expansion borrowed from GraphRAG, without the indexing-time LLM cost.

## 7. Spatial map

`regions` reduces each project's memory vectors to 5 dimensions with UMAP (Uniform Manifold Approximation and Projection), then clusters them with HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise). Clustering the raw 768-dimension vectors left 75–89% of points unclustered; this pipeline leaves 7–9%.

**Pipeline, per project**

1. One vector per memory: the mean of its chunk vectors, re-normalized. With fewer than 30 memories, clustering is skipped and the project is returned as one group.
2. UMAP to 5 dimensions: cosine metric, 15 neighbours, minimum distance 0, fixed seed 42. The fixed seed makes repeated runs return the same map.
3. HDBSCAN on the 5-dimension output with `min_cluster_size = max(5, round(sqrt(n) / 2))`, `min_samples = 5` and `max_cluster_size = n / 4`.
4. Cache the run in the map tables, keyed by a hash of the input memory identifiers and parameters. Recompute only when the hash changes.

| Memories | `min_cluster_size` | Clusters | Unclustered | Agreement with true labels |
| --- | --- | --- | --- | --- |
| 500 | 11 | 12 | 6.8% | 0.59 |
| 1,000 | 16 | 17 | 9.0% | 0.59 |
| 5,000 | 35 | 31 | 6.9% | 0.55 |

Measured with the Python reference libraries on labelled text with 14 true classes. Agreement is the adjusted Rand index (0 = random, 1 = identical). A fixed `min_cluster_size` of 5 dropped agreement to 0.30 at 5,000 memories, which is why the value scales with size.

**Rust libraries**

- Clustering: the `hdbscan` crate 0.12.0. It is the only Rust option found with `max_cluster_size`, which prevented a one-giant-cluster collapse seen with one seed. Its `min_samples` counts the point itself, so pass the Python value plus one (6).
- Projection: `holomap` 0.3.0 behind the `Projector` trait. It is deterministic. Cap input at 2,000 memories per run; above that, sample 2,000, cluster, then assign the rest to the nearest cluster centre in the original 768 dimensions. `manifolds-rs` 0.9.1 is the faster alternative if the cap becomes a problem.

**Labels.** The server does not name clusters. `regions` returns each cluster's most central memories; the calling agent names the cluster and may store the label against that run.

**`journey`** interpolates between two memories on the unit sphere and returns the nearest memory at each step. It needs no clustering and stays as in version 1.

**Two-level map (v3.1).** A cross-project map is built from summaries, not from every memory: each project cluster contributes its centre in the original 768 dimensions, and the same pipeline runs on those centres. In simulation this agreed with true topics at 0.67, against 0.43–0.60 for one flat map, and a change in one project left the other projects' clusters untouched.

Pitfalls the design must respect:

- **Unclustered memories vanish upward.** About 6% of memories belong to no cluster. Each must enter the top level as its own single-point summary.
- **Cluster numbers are unstable.** Never store them outside the run that produced them.
- **No fixed distance threshold on cluster centres.** It failed in testing; cluster the centres with the same pipeline instead.
- **The map shows similarity, not dependency.** Which service reads which table comes from the link table, never from proximity on the map.

## 8. MCP tool surface

Version 3 exposes 13 tools, down from 25. Every tool description an agent loads costs context on every session, so batch variants become parameters and administration moves to the command line.

| Tool | Purpose | Replaces |
| --- | --- | --- |
| `remember` | Store one or many memories with type, anchors and provenance. Returns identifiers and any redactions made. | `remember`, `remember_batch`, `extract` |
| `recall` | Search. Parameters: `query`, `scope` (project or all), `type`, `anchor`, `status`, `mode` (auto, vector, keyword), `k`. | `recall`, `hybrid_recall` |
| `related` | Neighbours of a memory or anchor by vector and by link walk. | `nearby`, `discover_connections`, `corpus_bridges` |
| `revise` | Edit a memory, change its status, or supersede it with a new one. | `reinforce`, `decay` |
| `forget` | Delete by identifier or filter. Requires `confirm: true` for more than one memory. | `forget`, `forget_batch` |
| `link` | Add or remove a link between anchors or memories; list links for an anchor. | new |
| `regions` | Clusters for a project, with central memories and optional 2-dimension coordinates. | `regions`, `visualize` |
| `journey` | Path between two memories. | `journey`, `wander` |
| `consolidate` | Find near-duplicate memories and propose merges. Applies only with `apply: true`. | `consolidate` |
| `share` | Write selected memories to the repository's shared folder for review. | `export_memories` |
| `sync` | Load reviewed shared memories from the repository into the local store. | `import_memories` |
| `status` | Daemon health, model, counts per project, index state. | `health`, `stats`, `namespaces` |
| reinforce | Mark a recalled memory as having helped. Records the rank it was found at. Does not change scores in 3.0. | reinforce |

**Moved to the command line:** `setup` (was `setup_hooks`), `export` and `import` of whole stores, project rename and delete (were `rename_namespace`, `delete_namespace`), `reindex`, `migrate`, `doctor`.

**Removed:** `decay` and `reinforce` as scoring mechanisms. Explicit status and `supersedes` replace time-based decay, which hid correct but old decisions. `wander` is dropped; `journey` covers guided exploration.

**Kept after reviewing version 2:** `reinforce` stays, as a signal and not as a score. Version 2 records the rank a memory held when it was reinforced, to decide whether a reranker is worth building. Removing the tool would have removed that evidence. Time-based `decay` stays removed.

**Conventions for every tool**

- Inputs and outputs are typed with JSON Schema generated from Rust types.
- Errors are structured: a stable code, a message, and what the caller can do next.
- Results carry memory identifier, project, type, status, score and provenance, so an agent can cite what it used.
- No tool returns more than a configurable token budget (default 2,000) without an explicit `limit`.

## 9. Scope, team sharing and security

A memory is private to the user's local store until it is shared, and sharing means a file in the project's git repository that a teammate reviews like code.

**Scope**

- The current project is resolved from the working directory: the enclosing git repository's remote URL, or the folder path when there is none.
- Every read defaults to the current project. `scope: all` must be passed explicitly.
- Profiles (`work`, `personal`) are separate stores. A client picks the profile from `--profile`, the `SPATIAL_MEMORY_PROFILE` environment variable, or a per-directory rule in the configuration file, in that order.

**Team sharing**

1. `share` writes each selected memory as one Markdown file with front matter under `.spatial-memory/shared/<type>/<id>.md` in the repository. One file per memory keeps pull-request diffs readable.
2. The author commits and opens a pull request. Review is the gate; the server pushes nothing.
3. After merge, teammates run `sync` (or the daemon syncs on project open, if enabled). Files become memories with `origin = shared` and provenance pointing at the file and commit.
4. Vectors are never committed. Each machine embeds shared files locally, so teammates may run different hardware but must run the same model.
5. A shared file that is deleted or changed in the repository retracts or revises the local copy on the next `sync`. Local edits to a shared memory are refused; the fix goes through the repository.

**Shared file format.** One Markdown file per memory with YAML front matter, the convention that decision records (MADR 4.0) and Claude Code's own skills and agents already use. The title is the first level-1 heading of the body, as in MADR, so it is not repeated in the front matter.

```markdown
---
schema: 1
id: 0193a8f0-5c1e-7000-8000-3f2a9c1e5b7d
type: decision
status: active
date: 2026-11-15
supersedes:
anchors: [table:dbo.Orders, service:order-api]
tags: [concurrency, sql]
commit: 4f2a9c1
---
# Use optimistic concurrency on dbo.Orders

Body in Markdown.
```

| Field | Required | Meaning |
| --- | --- | --- |
| `schema` | Yes | Format version, 1 |
| `id` | Yes | The memory's identifier. Identity lives here, so a file may be renamed or moved |
| `type` | Yes | decision, pitfall, convention, incident, runbook or note |
| `status` | Yes | active, superseded or retracted |
| `date` | Yes | Creation date, year-month-day |
| `supersedes` | No | Identifier of the memory this one replaces |
| `anchors` | No | One-line list of `kind:name` |
| `tags` | No | One-line list |
| `commit` | No | Commit the memory was last verified against |

- **Path.** `.spatial-memory/shared/<type>/<slug>--<last 8 characters of id>.md`. No sequence numbers: they collide when two people add files on separate branches.
- **A strict subset of YAML.** Only single-line scalars and one-line lists. It is read by a small strict parser, not a general YAML library. A file that does not parse is skipped and reported; it never stops a sync.
- **Nothing derived goes in the file.** No vectors, scores, importance, counters or sync timestamps. Authorship comes from git history.
- **Deterministic output.** Fixed key order and LF line endings, with a `.gitattributes` rule in the shared folder so a Windows checkout does not change the bytes.
- **Change detection before reading.** `sync` compares git's blob identifiers with the ones it stored and reads only the files that changed. Listing 10,000 files took 9 ms; reading, parsing and hashing all 10,000 took 115 ms.
- **Re-embed only when the text changes.** A hash of title and body decides. A change to status, anchors or tags updates the row and costs no embedding. Embedding is the real cost: about 67 minutes for 10,000 one-chunk memories on the test machine, against a tenth of a second to parse them.

Considered and not chosen: one JSON-lines file (fast to parse, but every change conflicts on merge and diffs are unreadable in review), and TOML front matter (unambiguous, but unfamiliar and not rendered by GitHub).

**Security**

- **Secret redaction on write.** `remember` and `share` scan content for credentials: private keys, cloud access keys, tokens, connection strings with passwords. Matches are replaced with `[REDACTED:<kind>]` and reported in the result. `share` refuses to write a file that still matches.
- **Local only.** The daemon opens no network listener. The only network access is the one-time model download in `setup`, which can be replaced by a local model directory.
- **Model integrity.** Model files are pinned to a repository revision and verified by SHA-256 before loading (section 5).
- **Socket permissions.** The socket or pipe is restricted to the current user.
- **Retrieved text is data.** Injected memories are wrapped in a clearly delimited block that states they are reference material. Shared memories come from other people and must not be followed as instructions.

## 10. Migration from v1 and v2

Migration is export from Python, import into Rust, and re-embed; version 3 never opens a LanceDB database. Reading LanceDB from Rust would add its 224 MB build to the binary for a one-time job, and the old vectors are useless because the model changes.

1. In the old install, run the existing `export_memories` tool to JSON with `include_vectors` off.
2. Run `spatial-memory migrate --from <file>`. It maps fields, redacts secrets, queues embedding, and prints a report.
3. Verify with `spatial-memory status`, then uninstall the Python package.

| v1 / v2 field | v3 destination |
| --- | --- |
| `id`, `content`, `created_at`, `updated_at`, `content_hash` | `memories`, unchanged |
| `project` | `projects` row, matched by name; unmatched values create a project with no remote |
| `namespace` | A tag, unless it equals the project name |
| `tags`, `metadata`, `source` | Kept as tags and provenance |
| `importance`, `access_count`, `last_accessed`, `expires_at` | Kept in an `imported` JSON column for reference; not used for ranking |
| `vector` | Dropped |
| (none) | `type = note`, `status = active` |

**What to port from the old branches**

- `tests/fixtures/v1_export_sample.json` and `docs/MIGRATION_V1_TO_V2.md` from `ralph/v2-implementation`: the fixture becomes the migration test input.
- `tests/fixtures/retrieval_eval_set.json` and its baseline from `feature/v2-implementations`: the retrieval regression test.
- The ambient-recall hook behaviour and the metrics journal from `feature/v2-implementations`, as behaviour to match, not code to translate.

**What not to carry over:** the unpinned `trust_remote_code` model loading, the hard-coded timestamp in the utility-service test, and the v1-to-v2 migration guide's incorrect steps. Neither `feature/v2-implementations` nor `ralph/v2-implementation` is merged; version 3 supersedes both.

Migration speed is bounded by embedding: about 3 chunks per second on two cores, so 10,000 single-chunk memories take about an hour. The import is resumable and the store is usable for keyword search while it runs.

## 11. Performance budgets and acceptance tests

The build is done when every test below passes on the build machine's operating system, and the other of Windows 11 and Linux is verified before release; budgets are set with headroom over what was measured on a two-core cloud processor.

| Budget | Target | Measured basis |
| --- | --- | --- |
| Hook, warm, end to end | 95th percentile ≤ 300 ms | 200 ms to embed a 128-token query on the two-core test machine, plus 3–5 ms search. On the owner's 16-core Windows machine: 45 to 95 ms to embed, 5.3 ms search, 3.6 ms for the process and pipe |
| Hook hard deadline | 1,500 ms, then empty output and exit 0 | Claude Code's hook timeout is 5 seconds |
| Daemon ready from cold | ≤ 5 seconds | Model load 1.5–2.5 seconds |
| `recall`, project scope, 100,000 chunks | 95th percentile ≤ 250 ms | As for the hook |
| `recall`, all projects, 100,000 chunks | 95th percentile ≤ 350 ms | `vec1` exact search 51–60 ms on Linux; 160–172 ms at the median and up to 197 ms at the 95th percentile on Windows 11 |
| `remember`, one memory of up to one chunk | ≤ 2 seconds | About 0.6 seconds per 256-token chunk |
| Daemon memory, idle after load | ≤ 1 GB of private memory | 0.65–0.8 GB |
| Daemon memory, peak while indexing | ≤ 2 GB | 1.5 GB at 8 chunks of 256 tokens per batch; default batch is 4 |

**Memory.** The idle budget is measured on private memory: `PrivateUsage` on Windows, `RssAnon` on Linux. Total memory, the working set or the resident set, is reported beside it for information. The difference is the mapped database file, which the operating system can reclaim (B-146). The peak budget is judged on total memory (B-114).

**Acceptance tests**

1. **Embedding correctness.** For the query "Which planet is known as the Red Planet?" against the four documents of the first example in README.md of the model repository at the pinned revision, each sent with the document prompt, cosine scores are Venus 0.675, Mars 0.853, Jupiter 0.747, Saturn 0.778, each within 0.01. A text embedded alone and inside a padded batch gives cosine ≥ 0.9999.
2. **Retrieval regression.** On the ported `retrieval_eval_set.json`, ranking quality stays within the fixture's own tolerances of the stored baseline (0.02 on nDCG@10 and on mean reciprocal rank, 0.07 on hit-at-5). EmbeddingGemma 2 measured 0.987, 1.000 and 1.000 on this set, against a baseline of 0.974, 0.967 and 1.000 from version 2's model.
3. **Hook latency and safety.** 100 warm hook calls meet the budget, and meet it again while a bulk import is running. During that import the daemon never runs more threads than the configured budget. With the daemon killed, the hook exits 0 with empty output inside 1,500 ms.
4. **Long input.** Storing a 50,000-character memory succeeds, produces multiple chunks, and keeps peak memory under 2 GB. This is the case that killed version 2.
5. **Concurrency.** Two readers and one writer run for 60 seconds with no lock error reaching a client.
6. **Determinism.** `regions` run twice on the same data returns identical assignments.
7. **Clustering quality.** On a labelled fixture that the spatial stream creates and commits (at least 300 short texts across 10 topics), unclustered memories are at most 25% and no cluster holds more than 40%.
8. **Migration.** The v1 export fixture imports completely, every memory is re-embedded, and the report matches the row count.
9. **Redaction.** Fixtures with fake credentials are redacted on `remember`, and `share` refuses a file that still matches.
10. **Model integrity.** A model file with one changed byte is never loaded: the daemon stays up in a failed state, reports a clear error, and embeds nothing.
11. **Index mismatch.** Changing the model identifier or chunk size in configuration blocks search until `reindex` completes.
12. **Scope.** A memory stored in project A is not returned from project B without `scope: all`, and never across profiles.

13) **Capture from tools.** Fixture PostToolUse payloads for Edit, Write and Bash: strong-signal text is stored as an active memory with capture provenance, weak-signal text stores nothing, and a fake credential in the payload is redacted.
14) **Capture from the transcript.** With a fixture transcript, PreCompact and Stop store assistant text that PostToolUse missed and do not store what it already captured. Stop does nothing when `stop_hook_active` is true, and prints its continue response. PreCompact prints nothing.
15) **Capture never blocks and never loses.** With the daemon stopped, each capture hook exits 0 within 200 ms and its payload is in the spool. After the daemon starts, the spooled item is processed.
16) **Both tiers are saved, pending.** A medium-signal capture is stored as a pending memory with importance of at least 0.4. A strong-signal capture is stored pending with importance of at least 0.7. Neither is served until it is judged (B-157, which replaces "stored as an active memory, as in version 1"). Importance does not change ranking.

17. **Compatibility with version 2's hooks.** With `SPATIAL_MEMORY_AMBIENT_RECALL=off` the prompt hook prints nothing and does not contact the daemon. The injected block matches version 2's format. A capture longer than the index cap is stored whole, indexed up to the cap, and recorded as a cap hit in the journal.

**Evaluation on real memories (after the build, the owner's task).** Every retrieval number in this document comes from public question-and-answer data. Before any default is tuned, the owner writes 30 to 50 queries against a store built up by real use of version 3, each with the memories it should find, in the format of `retrieval_eval_set.json`. `spatial-memory doctor --eval <file>` runs them against the live store and reports nDCG@10, hit-at-5, mean reciprocal rank, and the score distributions of right and wrong results. The build delivers the command; the similarity floor, the result cap and the chunk limit are revisited only with its output.

**Test conventions**

- Unit tests use a fake `Embedder` that returns deterministic vectors. No unit test loads the real model.
- Real-model tests live in one integration test binary that loads the model once, and are skipped unless the model directory is present.
- Every documented command runs unchanged in PowerShell and Bash.

## 12. Phasing

Version 3.0 ships the daemon, the full data model and per-project maps; version 3.1 adds everything that needs a populated multi-project store to be worth building. The anchor and link tables ship in 3.0 so that 3.1 adds behaviour, not a schema change.

| Area | v3.0 | v3.1 |
| --- | --- | --- |
| Runtime | Daemon, MCP proxy, hook, setup, doctor | Background sync on project open; worker pool with self-tuning for bulk embedding |
| Storage | SQLite, exact vector search, FTS5 | `vec1` approximate index for stores past 100,000–200,000 chunks |
| Memory model | Types, status, supersedes, anchors, provenance | None planned |
| Links | Link table and `link` tool, manual links, stale detection | Automatic extraction from code (Roslyn-based for .NET), stamped with the verifying commit |
| Maps | Per-project `regions`, `journey` | Cross-project two-level map |
| Sharing | `share` and `sync` through git | Conflict report when a shared memory contradicts a local one |
| Migration | From v1 JSON export | None planned |
| Models | EmbeddingGemma 2; Arctic Embed M behind a Cargo feature | Re-evaluate against newer models with the benchmark harness. Trial embedding on the graphics card for bulk work |
| Distribution | `cargo install` from the repository | Prebuilt binaries, which need continuous-integration runners and are costed first |
| Model download | Built-in root certificates; proxy from `ALL_PROXY`, `HTTPS_PROXY` or `HTTP_PROXY`; the files can be placed by hand | Use the operating system's certificate store and proxy settings for the model download (B-148) |
| Capture | PostToolUse, PreCompact, Stop and SessionStart hooks; Claude Code plugin package; Cursor event mapping | Signal patterns tuned from the metrics journal; query expansion and a reranker decided from version 2's metrics |

**Exit condition for 3.0:** every acceptance test in section 11 pass on the build machine, and the other of Windows 11 and Linux is verified or listed as unverified. Migrating a real version 1 store is not part of the exit condition: version 3 is a clean break and no existing memories are carried over.

## 13. Risks and unverified items

Everything in this list was either not tested or tested only on Linux; each has a named fallback so the build does not stall on it.

| Risk | Status | Fallback or check |
| --- | --- | --- |
| Windows build: ONNX Runtime binaries, the `vec1` C file compiled by the `cc` crate, named pipes | Not tested. All measurements are from Linux. | First build task runs `cargo build` and the embedding self-test on Windows before anything else is written. |
| EmbeddingGemma 2 is about 40 days old (released 6 October 2026) | Benchmarked on two public sets, 300 queries. Not tested on real team memories. | Arctic Embed M through `fastembed` behind the same trait. Switching is a configuration change plus `reindex`. |
| The ONNX export is community-maintained and can change | Pinned by revision and SHA-256 (section 5). | Mirror the three files to storage you control. |
| `vec1` is at version 0.8 | Exact mode benchmarked at 100,000 vectors. Approximate mode benchmarked for speed only; its recall was not measured. | `sqlite-vec` behind the same trait; approximate mode stays off until recall is measured. |
| `interprocess` crate for sockets and named pipes | Chosen from documentation, not tested. | Loopback TCP with a per-user token file. |
| `holomap` projection above 2,000 points | Tested to that size only. | Sampling plus nearest-centre assignment, or `manifolds-rs`. |
| Similarity floor of 0.70 | Calibrated on public data. In a dense single-topic store it still passes many near-misses. | The 5-memory cap and token budget bound the damage; `doctor` reports the score distribution of the user's own store. |
| Ambient recall precision | No ground truth exists for "should nothing be injected here". | Ship with the metrics journal on, review injections after two weeks of use. |
| Identifier search through FTS5 | Tokenizer behaviour on names like `dbo.Orders_Archive` not tested. | Acceptance test with identifier fixtures; switch tokenizer settings if it fails. |
| Embedding speed on real developer machines | Measured on two cloud cores only. | `doctor` prints chunks per second and query latency on the local machine. |
| Removed tools (`wander`, `decay`, `reinforce`) | Design judgment, not measured. | wander can return with no schema change; decay and reinforce would need their counters restored from the imported column. |
| Scaling beyond two cores | Not measured. The test machine had two virtual cores. One to two threads scaled 1.85 times. | Version 3.0 uses one bulk session inside the budget, and doctor reports its speed on the real machine. The self-tuning worker pool is version 3.1. |
| Below-normal priority for bulk workers on Windows | Not tested. | If priority cannot be set, halve the processor budget for bulk work. |
| Embedding on a graphics card | Not tested. A community repository publishes separately modified DirectML graphs for this model, which suggests the stock graph may not run on DirectML unchanged. | The processor path is the supported one in 3.0. A graphics-card embedder is a 3.1 trial behind the same trait. |
| vec1 compiled into the binary | Tested only as a runtime-loadable library on Linux, not compiled in through the cc crate. Its source is the repository at sqlite.org/vec1; no release archive or checksum was recorded. | Vendor the source at a recorded check-in with its SHA-256. If compiling it in fails, load it as an extension, or use sqlite-vec. |
| Long prompts in the hook | A 256-token query took 388 ms on the test machine, over the 300 ms budget. | Hook queries are cut to their first 128 tokens (200 ms). Whether the first 128 tokens are the right part of a long prompt is untested. |
| Identifier search, update | Now tested: with the porter tokenizer, quoted phrases matched dbo.Orders\_Archive, CS8602 and OrderService.GetOrderAsync. Unquoted text with a dot raised a syntax error. | Always quote keyword queries (section 6). |

## 14. Build plan

The build runs in four phases: one agent freezes the contracts, five agents build crates in parallel worktrees, two more build on top of those, and the lead integrates and has the result reviewed by an agent that wrote none of it. Parallel work is safe only because every crate depends on `sm-core` traits and not on each other.

| Phase | Stream | Deliverable | Depends on | Done when |
| --- | --- | --- | --- | --- |
| 0 | Lead: contracts | Workspace, `sm-core` (types, traits, configuration, error codes), socket message schema, fixtures, `docs/v3/CONTRACTS.md` | none | Workspace builds on Windows and Linux; a spike proves ONNX Runtime, `vec1` and the socket crate compile and run on Windows |
| 1 | A: store | `sm-store` | 0 | Schema, migrations, exact vector search, FTS5, links; storage tests pass |
| 1 | B: embed | `sm-embed` | 0 | Embedding self-test and chunker tests pass; model download verifies checksums; the thread budget and priority rules of section 5 work |
| 1 | C: spatial | `sm-spatial` | 0 | Determinism and clustering-quality tests pass |
| 1 | D: clients | `sm-cli`: MCP proxy, hook client, `setup`, `doctor` | 0 | All five hook events and the plugin package run against a mock daemon that speaks the frozen message schema; the daemon subcommand is wired in phase 3 |
| 1 | G: capture | sm-capture: extraction, transcript reader, signal classifier, secret scanner, spool | 0 | Version 1's hook test cases, ported as fixtures, pass against the new crate |
| 2 | E: daemon | `sm-daemon`: services, indexing queue, retrieval gates | A, B, C | Every tool and every hook event works end to end over the socket |
| 2 | F: sharing and migration | Redaction, `share`, `sync`, `migrate` | A, B | Sharing and migration tests pass, using the secret scanner delivered by stream G |
| 3 | Lead: integration | Merge, acceptance tests, performance tests, README and migration guide | D, E, F | Every acceptance test passes on the build machine |
| 3 | Reviewer | Independent review against this specification | integration | Findings fixed or listed as known gaps |

**Rules for every stream**

- Contracts are frozen after phase 0. A stream that needs a contract change asks the lead; it does not edit `sm-core` itself.
- Each stream works in its own git worktree and branch, and finishes with `cargo fmt --check`, `cargo clippy -- -D warnings` and its crate's tests green.
- No stream adds or changes continuous-integration workflows, publishes a crate, or pushes to a remote. A proposed workflow goes in `docs/v3/CI_PROPOSAL.md` with its expected runner minutes, for a human to cost and enable.
- Work lands on a `v3` branch. `master` and the Python code stay untouched until version 3.0 is accepted. On that branch the Rust workspace sits at the repository root and the Python tree is removed from the working tree; it remains on master and in history. The two are never built side by side.
- Anything in section 13 that fails is reported with evidence and its fallback applied; it is not worked around silently.

**Requirement identifiers.** Version 2's commits cite the story and section they implement; version 3 keeps that discipline. Acceptance test n in section 11 is `AT-n`. The decisions in section 2 are `D-1` to `D-10` in table order: language and shape, storage, keyword search, embedding model, fallback model, chunk limit, clustering, cross-project structure, GraphRAG, runtime threads. Any other rule is cited by section and rule name, for example "section 16, Never loses". Every commit message and test name cites the identifier it serves, and `docs/v3/TRACE.md` maps each identifier to its tests and commits. A requirement with no test is a reported gap.

The prompt that starts this build is in the Build prompt tab.

## 15. Open decisions and tested versions

These points came out of the review of 15 November 2026. Each has a default so the build is not blocked, but the first two change scope and should be decided before the build starts.

| Decision | Why it is open | Default if undecided |
| --- | --- | --- |
| Automatic capture | Version 1 ships hooks on `master` (PostToolUse, PreCompact, Stop) that extract memories from sessions without being asked, and a Claude Code plugin package under `plugin/`. This specification covers recall only. | 3.0 ships recall and explicit `remember`. Decided 15 November 2026: capture and the plugin package are part of 3.0 (section 16). |
| What a project is | Identity is the git repository. Decided 15 November 2026: each of the 60+ .NET projects is its own git repository, so each is its own project. | Project = repository. Sub-projects are anchors of kind project. A `.spatial-memory/project.toml` file may name the project explicitly. |
| Recall below the floor | Dropping weak results from explicit recall would return nothing in a sparse store. | Explicit recall returns them marked weak; only the hook drops them. |
| Daemon when idle | The loaded model holds about 0.7 GB for as long as the daemon runs. | The daemon never exits on its own. `daemon.idle_exit_minutes` can change that; a restart costs 1.5 to 2.5 seconds. |
| One user, several machines | Private memories do not travel between a user's own machines. | Not supported in 3.0. `export` and `import` move a whole store by hand. |
| Removed tools | `wander`, `decay` and `reinforce` were part of the product's original identity. | Removed, as in section 8. |
| Distribution | Version 1 installs as a zero-configuration plugin. Version 3.0 needs a Rust toolchain and C++ build tools to install. Prebuilt binaries need continuous-integration runners, which cost money on a private repository. | Agreed 15 November 2026: decided before any public release. 3.0 is not announced publicly until the runner cost in the build's CI proposal has been reviewed and the binaries question is settled. |

**Versions tested.** Anything else is the build agent's choice, recorded in its decisions file.

| Component | Version | How it was tested |
| --- | --- | --- |
| `ort` | 2.0.0-rc.15 | EmbeddingGemma 2 end to end, Linux |
| `tokenizers` | 0.23.2 | With the model's `tokenizer.json` |
| `fastembed` | 7.7.0 | Fallback and comparison models |
| `rmcp` | 3.10.0 | One-tool server over standard input/output |
| `rusqlite` | 0.43, bundled SQLite | Storage benchmark, FTS5 |
| `vec1` | 0.8 | Loaded as an extension, exact and approximate search |
| `sqlite-vec` | 0.1.9 | Storage benchmark |
| `hdbscan` | 0.12.0 | Matched the Python reference |
| `holomap` | 0.3.0 | To 2,000 points |
| `interprocess` | 2.4.8 | Not tested; version checked only |
| `directories` | 6.0.0 | Not tested; version checked only |

## 16. Automatic capture

Version 3.0 keeps version 1's automatic capture: hooks watch a session and store decisions, fixes and conventions without being asked. What changes is where the work happens. The hook hands its payload over and exits; the daemon extracts, classifies, redacts and stores.

| Event | What the hook does | What it prints |
| --- | --- | --- |
| PostToolUse | Sends the tool call (Edit, Write, Bash, NotebookEdit) to the daemon. Registered as asynchronous. | The same continue response as Stop |
| PreCompact | Sends the transcript path, so assistant text that PostToolUse missed is scanned before the context is compacted. | Nothing |
| Stop | Sends the final assistant text from the payload's last\_assistant\_message field. It does not read the transcript, which Claude Code does not guarantee to hold the final message at that moment. Skips all work when `stop_hook_active` is true. | `{"continue": true, "suppressOutput": true}` |
| SessionStart | On startup and resume only, adds a short note that project memory is available. | Additional context |
| UserPromptSubmit | Ambient recall (section 6). | The injection |

**Pipeline in the daemon**

1. Extract memory-worthy text from the tool call, or assistant text from the transcript since the offset last recorded for that session.
2. Classify it with the signal patterns ported from version 1. A score of 0.8 or more is a strong signal, 0.5 to 0.79 is a medium signal, and anything lower is dropped.
3. Redact secrets with the scanner of section 9.
4. Drop duplicates: the same content hash as an existing memory, or as an item already captured in this session.
5. Store it as a pending memory, whatever its tier (B-157, which replaces "store it as an active memory, as version 1 does for both tiers"). A pending memory is never served: ambient recall, `recall`, `related`, `regions`, `journey` and `consolidate` never return it, `recall` and `revise` refuse `pending` by name, `journey` refuses it as an endpoint, and `share` refuses it. Only a judged capture becomes active. A save of the same text in the same project replaces a pending capture. The tier sets the stored importance: the higher of 0.7 and the score for a strong signal, and the higher of 0.4 and 0.8 times the score for a medium one.
6. Embed it on the bulk path, at low priority.

**Rules**

- **Never blocks.** Every capture hook exits 0 within 200 ms, whatever happens. It never exits with a blocking code.
- **Never loses.** When the daemon is not reachable, the hook writes the payload to a spool directory: a file written under `tmp/`, then renamed into `new/`. The daemon drains the spool when it starts. This is version 1's queue protocol.
- **Importance is stored, not used.** Version 1 saves both tiers and uses the tier only to set importance. Version 3 stores both tiers pending (B-157). Importance does not affect ranking or the prompt hook until the journal shows that it should; `recall` accepts a minimum importance as a filter.
- **Type and provenance.** The pattern family sets the type (decision, pitfall, convention) and anything else is a note. Provenance records capture as the source, with the session, the tool, the file path and the current commit. A capture from Edit or Write is anchored to its file.
- **Controls.** `capture.enabled` (on by default), `capture.tools`, and ignore patterns for paths such as `.env` files and key files. A repository opts out in `.spatial-memory/project.toml`. Captured memories are private until shared.
- **Clients.** `--client claude-code` and `--client cursor`, with version 1's event-name mapping. Cursor gets the capture events only: it has no prompt event and ignores hook output. Cursor is tested with fixture payloads only.
- **Packaging.** A Claude Code plugin under `plugin/` (manifest, MCP entry, hooks file) that calls the installed `spatial-memory` binary. `setup` remains the other way to register. The plugin does not carry the binary in 3.0.

**Carried over from version 2**

- **No per-field size limits.** Version 1 cut captures at 3,000 to 10,000 characters and lost content. Version 2 replaced those limits with one safety cap of 200,000 characters at extraction. Version 3 keeps that cap and stores the captured text whole.
- **Index cap.** Version 2 embedded only the first 8,192 tokens of a capture. Version 3 indexes the same span, as chunks: `capture.max_index_tokens` defaults to 8,192, which is at most 32 chunks. The two capture metrics record the size before and after the cap and each cap hit by tool. Explicit `remember` is not capped.
- **Registration.** The commands are version 2's: `spatial-memory hook <event> --client claude-code`, with events `session-start`, `post-tool-use`, `pre-compact`, `stop` and `user-prompt-submit`. Timeouts are 5 seconds for the prompt hook, 10 for PostToolUse (asynchronous), and 15 for PreCompact and Stop. SessionStart is registered through settings and not through the plugin, as in version 2.
- **Preload at startup.** The model loads when the daemon starts, never on a request. `embedder_cold_start_seconds` verifies it.

**Deliberate differences from version 2**

- **Relevance gates on the prompt hook.** Version 2 injected its top 5 whatever their scores. Version 3 adds the similarity floor, the minimum prompt length and the no-repeat rule.
- **Vector search, not hybrid.** Version 2's hook called hybrid recall. Fusion scored below vectors alone in every test here (section 6).
- **Chunking replaces the token cut.** Version 2's capture metrics existed to decide whether chunking was worth building. The chunk-size test in section 5 decided it.

**How the server and hooks are registered**

- **`setup` is the supported route for 3.0 (B-156, which replaces "the plugin is the primary route").** The marketplace still serves the version 1 plugin, so the 3.0 plugin has never been installed through `/plugin install`; the README says the plugin route is not verified. Hooks that `setup` writes go live in every Claude Code session already open, because Claude Code reads its settings file again when it changes; the README says what such a session does (RC1 of `CAPTURE-RCA.md` applies to it).
- **The plugin.** One install registers the MCP server (`.mcp.json`) and every hook (`hooks/hooks.json`), is versioned, and is disabled or removed cleanly. The repository carries a marketplace file so it installs with `/plugin marketplace add` and `/plugin install`. `claude plugin validate` runs in the tests. It does not download the model: `setup --model-only` does that, registers nothing, and is allowed while the plugin is enabled (B-153).
- **Exec form for every hook, never shell form.** Each hook is `"command": "spatial-memory"` with `"args": ["hook", "<event>", "--client", "claude-code"]`. With `args`, Claude Code starts the program directly. Without them it starts a shell first, which on Windows is Git Bash or PowerShell, on every prompt and every tool call. The program and arguments are version 2's; only the form changes.
- **Asynchronous where nothing is returned.** PostToolUse is registered with `"async": true`. No hook sets `onFailure` to block. Timeouts stay as in version 2.
- **Plugin options.** The profile and the ambient-recall switch are declared as `userConfig`, which Claude Code exports to hook processes as `CLAUDE_PLUGIN_OPTION_<KEY>`. The hook reads those, then version 2's environment variables.
- **What `setup` does.** It registers the server by calling `claude mcp add` at user scope, not by editing Claude Code's files itself. It merges hooks into `~/.claude/settings.json` after writing a backup, is safe to run twice, has a remove option, and refuses to run while the plugin is enabled so nothing is registered twice. A plugin disabled at any scope, local included, counts as disabled, and the version 1 plugin is never taken for this one (B-150). `setup --remove` unregisters, then stops the profile's daemon the way `stop` does (B-155).
- **The binary is found on `PATH`.** The plugin does not carry it in 3.0. `doctor` checks that the version on `PATH` matches the plugin's.

Considered and not the default:

- **HTTP hooks** post the event straight to a local address, with no process started at all. They are the fastest option, but a hook whose endpoint is down is a non-blocking error, so a capture made while the daemon is stopped would be lost, and the daemon would need a loopback listener. They are a version 3.1 option, and only if the measured start time of the hook program on Windows shows a need.
- **MCP-tool hooks** call a tool on an already-connected MCP server. They would reuse the running proxy and start nothing. The official reference says they work on every event once the servers are connected, take string inputs from the payload, and have the tool's text treated like a command's output. They would also add one tool that the model can see, and they do nothing when the server is not connected. The build's spike measures them against exec form on the prompt hook; exec form stays the default until that evidence exists.

**Verified against the official hooks reference (15 November 2026)**

- **Exec form.** With `args`, the command is started directly with no shell on any platform. Without `args` it goes through `sh -c`, Git Bash on Windows, or PowerShell when Git Bash is absent. On Windows the command must be a real executable.
- **Timeouts are in seconds.** The default is 600, lowered to 30 for UserPromptSubmit. Ours stay at version 2's values. A timeout is not enforced on an asynchronous hook.
- **Asynchronous hooks.** Only command hooks can be asynchronous. Each firing is its own process. In a non-interactive run, an asynchronous hook still running when the session ends is killed, so the hand-off to the daemon or the spool must finish in milliseconds.
- **Output.** Plain text on standard output with exit code 0 becomes context for UserPromptSubmit and SessionStart. Hook output is capped at 10,000 characters.
- **Double registration.** The same handler in several settings files runs once, but a plugin's copy runs separately. This is why `setup` refuses to run while the plugin is enabled.
- **SessionStart** accepts only command and MCP-tool hooks, and fires for `startup`, `resume`, `clear`, `compact` and `fork`.
- **Payload fields.** Every event carries `session_id`, `transcript_path`, `cwd`, `permission_mode` and `hook_event_name`. UserPromptSubmit adds `prompt`. PostToolUse adds `tool_name`, `tool_input`, `tool_response`, `tool_use_id` and `duration_ms`. PreCompact adds `trigger` and `custom_instructions`. Stop adds `stop_hook_active` and `last_assistant_message`. SessionStart adds `source`.
- **HTTP hooks.** A connection failure or a non-2xx reply is a non-blocking error, and a plain-text reply is an error whose text is not added to context.
- **Server registration.** The command is `claude mcp add [options] <name> -- <command> [args...]`, with `--scope user` for all projects.

**Shell-agnostic rules**

Versions 1 and 2 run their hooks through Python with shell-form commands, and ship a pair of scripts (`find_python.cmd` and `find_python.sh`) to locate the interpreter on each platform. Version 3 removes the problem instead of working around it: one native executable, started with no shell.

- **No shell anywhere.** Hooks use exec form. `setup` and the daemon start other programs directly, never through a command string. The same hooks file works on Windows, macOS and Linux.
- **No interpreter to find.** There is no Python or Node on the hook path, so nothing depends on `python` against `python3`, on a virtual environment, or on which shell is installed.
- **Where a client only accepts a command string** (Cursor), the string is the program name and plain arguments: no quotes, variables, pipes or paths. Such a string means the same in Bash, PowerShell and the Windows command prompt.
- **Finding the binary.** The plugin uses the bare name `spatial-memory`, resolved on `PATH`. `setup` writes the absolute path instead. The plugin has no setting for the binary's path, because Claude Code does not substitute a plugin option in a hook's command (B-85). Where `PATH` differs between the terminal and the application that starts Claude Code, which happens for applications launched from the desktop on macOS, disable the plugin and run `spatial-memory setup`, which registers the absolute path. `doctor` checks that the hook command resolves.
- **Input and output.** JSON in on standard input as UTF-8; UTF-8 out with no byte-order mark. Nothing depends on the console's code page.
- **Paths.** Both separators are accepted on Windows. Paths are canonicalised before they are compared and are never built by joining strings.
- **Git without a shell or a subprocess.** The daemon reads the repository's configuration files to find the remote, and caches the answer per working directory. It must handle a worktree, where `.git` is a file that points to the real directory; Orca creates worktrees.
- **Windows specifics.** Starting the daemon opens no console window. Claude Code under Windows and under the Windows Subsystem for Linux are two installs with two stores.
- **Documentation.** Every documented command is a program and its flags, with no shell syntax, so it runs unchanged in PowerShell and Bash.

**Pre-build spikes on the build machine**

Nine spikes (`s0` to `s8`) settle what could not be measured off Windows. They live in their own private repository, `spatial-memory-spikes`, together with the kick-off prompt (`PROMPT.md`), the method and decision rule for each spike (`docs/SPIKES.md`) and one result format (`docs/RESULTS-FORMAT.md`). Claude Code runs them on the owner's Windows 11 machine, natively and in Linux containers: latest Fedora, latest Ubuntu, Arch Linux standing in for Omarchy, and Ubuntu 22.04 as a control. Each spike program writes its own result file, and nobody edits a result by hand. The reviewer reads `results/SUMMARY.md` from that repository and then updates this specification and the build prompt. The table below shows three of the nine; the spike repository is the source of truth for all of them.

| Spike | Question | Decision rule |
| --- | --- | --- |
| `timing` | How long does a hook program take to start and exit: directly, through Git Bash, through PowerShell? | Keep exec-form command hooks if the median is at most 25 ms and the 95th percentile at most 60 ms. Otherwise the prompt hook becomes an MCP-tool hook. |
| `mcp-scope` | Where does `claude mcp add --scope user` store a server, and do `get` and `remove` work? | If add, get and remove all succeed, `setup` uses the command and never edits the file. The file's path is used only by `doctor`. |
| `claude-hook` | Does Claude Code run an exec-form hook given only a bare program name? | If yes, the plugin names the program `spatial-memory`. If only the absolute path runs, users on that platform register with `spatial-memory setup`, which writes the absolute path (B-85). |

Measured so far, on Linux with Claude Code 2.1.325: a user-scope server is stored in `~/.claude.json`, and add, get and remove all work. A stand-in hook program took 0.9 ms to start and exit directly, 1.8 ms through `sh`, and the Python interpreter alone took 9.7 ms. None of these is evidence for Windows.

**Results on the owner's machine, 16 November 2026**

Windows 11 (build 26300), AMD Ryzen 9 9950X3D with 16 cores and 32 logical processors, 93.6 GB of memory, Rust 1.92.0, Microsoft compiler tools 14.52, Claude Code 2.1.325. All nine spikes have a Windows result and every check passes. The 42 result files validate. Source: `results/` in the `spatial-memory-spikes` repository at commit `30d9138`. The reviewer read the result files, not only the agent's report.

| Spike | Windows result | What it settles |
| --- | --- | --- |
| s0 preflight | Ready. Long paths enabled, Docker runs Linux containers, Defender real-time protection is on | The machine can run the build |
| s1 hook timing | Program started directly: 3.5 ms median, 4.0 ms at the 95th percentile. Through Git Bash 44 ms, Windows PowerShell 111 ms, PowerShell 7 162 ms. Starting Python alone: 37 ms | Keep exec-form hooks. A shell would cost 40 to 160 ms per hook |
| s2 server registration | Add, get and remove all work. Stored in `.claude.json` in the home folder | `setup` uses `claude mcp add` |
| s3 hooks in Claude Code | The exec-form hook ran by bare program name. SessionStart, Stop and SessionEnd ran. The Stop payload has `last_assistant_message`. A daemon started from a hook outlived the session | The plugin names the program `spatial-memory` on Windows |
| s4 embedding self-test | Scores within 0.0005 of the Linux reference, identical tokens, load 0.33 s | EmbeddingGemma 2 through ONNX Runtime works on Windows |
| s5 thread scaling | One bulk session: 3.1 chunks per second on 1 thread, 6.7 on 4, 10.4 on 8, 13.2 on 12, 15.9 on 16, 18.0 on 24. A 128-token query: 159 ms on 1 thread, 84 ms on 4, 63 ms on 8, 54 ms on 16 | See the reading below |
| s6 bulk priority | Query alone 45 ms. With bulk on 8 threads: 82 ms at normal priority, 76 ms at lower priority. With bulk on all 32 logical processors: 135 ms | See the reading below |
| s7 `vec1` | Compiles with the Microsoft compiler, AVX2 is active, results equal brute force. Exact search over 100,000 vectors: 172.5 ms median, against 29 to 31 ms in Linux containers on the same machine | Correct on Windows, but slow. See the reading below |
| s8 daemon and pipe | Whole hook path (start, connect, request, reply, exit): 3.6 ms median, 5.1 ms at the 95th percentile. Round trip 0.09 ms. Cold auto-start 21 ms. The lock refuses a second daemon | The named-pipe design holds |

In Linux containers on the same machine (Fedora 44, Ubuntu 26.04, Arch Linux), s0, s1 and s4 to s8 pass. On the Ubuntu 22.04 control, s4 to s6 cannot link, as predicted, and s7 and s8 pass.

**The reviewer's reading**

- **Query threads (s5).** The spike recorded `change_thread_defaults` because its rule picks the smallest thread count within 10% of the best time, which is 12. That rule was too mechanical. Four threads give 84 ms at the 128-token cap and 25 ms for a 32-token query, against a budget of 200 ms. Decided by the owner on 16 November 2026: keep `embed.query_threads` at 4.
- **One session scales poorly, and a pool does not fix it (s5, s9).** Twenty-four threads give 5.7 times the throughput of one. The follow-up spike s9 then tried several small sessions with the same total threads. At 12 threads, one session gave 14.9 chunks per second and twelve single-thread sessions gave 16.5, which is 1.11 times as much for 11 times the memory (16.5 GB against 1.5 GB). At 16 threads the best arrangement, four sessions of four threads, gave 19.2 against 16.5. Throughput on this machine is capped near 15 to 19 chunks per second however the threads are arranged. Decided by the owner on 16 November 2026: the worker pool stays on the version 3.1 list, to be built only if a machine shows a clear gain; spike s9 is the test for that.
- **Bulk work and queries (s6).** The spike recorded `move_bulk_to_child_process` because its rule required the query median to stay within 1.5 times the query-alone median. In absolute terms the 95th percentile was 100, 80 and 141 ms, all inside the 200 ms budget. Priority changed little because 4 cores stayed free; bulk throughput was the same at both priorities. A child process shares the same memory and processor cache, so it is not expected to help. Decided by the owner on 16 November 2026: keep bulk embedding in the daemon, keep the 0.75 budget and the lower priority, and never use every logical processor (17% more throughput for three times the query time).
- **Only the first session a process creates is fast (s5, s6, s10).** The same 128-token query on 4 threads took 84 ms in s5 and 45 ms in s6. The follow-up spike s10 found why the numbers differ: the first ONNX Runtime session created in a process answered in 43 ms, and the next four sessions, each created after the previous one was dropped, took 87 to 93 ms. Bulk work in the same session made no difference (87 ms before, 87 ms after), and thread spinning made little (82 ms). The cause is not known. Rule for the build: the daemon creates the query session first and keeps it for its whole life. Whether the query session stays fast once a second session exists beside it was not measured; both figures are inside the 200 ms budget.
- **`vec1` exact search is slow on Windows and the cause is not known (s7, s7b).** AVX2 is active on both systems. The reviewer's first explanation, the cost of file read calls, was wrong for Windows: the follow-up spike s7b confirmed memory-mapped input/output was in effect and the search still took 157 ms against 160 ms by default. A 600 MB page cache did not help either (168 ms). A 64 KB page size halved it (85 ms), but that setting applies to the whole store and was not tested against anything else. Decided by the owner on 16 November 2026: keep `vec1` with default settings in version 3.0 and accept about 160 ms for search across all projects on Windows. The budget for that search still holds (see section 11), and the hook does not use this path.
- **The project-scoped path is confirmed on Windows (s7b).** The first spike's 2,000-row scan (22.5 ms) queried a row range of the `vec1` table, which is not this specification's path. The follow-up measured the real one: an ordinary table read with `WHERE project_id = ?` through an index, scored in Rust. With 100,000 chunks in 60 projects, rows scattered, one project of 1,667 rows took 5.3 ms at the median and 5.5 ms at the 95th percentile, and every result matched brute force.
- **Pipe security (s8).** With default settings the named pipe lets Everyone and Anonymous read. The daemon must create the pipe with a security descriptor that admits only the current user.
- **Detaching (s8, s3).** Windows refused to let the daemon leave its parent's job object. The daemon outlived its launcher and the Claude Code session anyway. The design must not depend on leaving the job.
- **Paths (s3).** In one session the shell-form hook received `CLAUDE_PROJECT_DIR` with forward slashes and the exec-form hooks received backslashes. Project identity must not depend on slash direction.
- **One deviation (s3).** The spike ran twice. The first run misread its own log because two parallel hooks interleaved partial lines. Every log, journal and spool line must go out in one write call.

**Follow-up spikes s7b, s9 and s10** ran on Windows on 16 November 2026 (repository commit `2cc4720`, 45 result files, none invalid). Their results are folded into the points above. Two more findings came with them:

- **Build from PowerShell on this machine, not from Git Bash.** In Git Bash, Git's own `link.exe` comes before the Microsoft linker on the search path, and linking fails. Claude Code's shell tool on Windows is Git Bash, so the build prompt tells every worker to run `cargo` through PowerShell.
- **The budgets in section 11 hold on this machine, two of them narrowly.** A hook is about 100 ms end to end at worst (4 to 5 ms for the process and pipe, up to 95 ms to embed, 5 to 6 ms to search) against 300 ms. Search across all projects is about 290 ms at the 95th percentile (up to 95 ms to embed, up to 197 ms in `vec1`) against 350 ms. Peak memory with a query session and a bulk session embedding batches of 8 was 2,047 MB against 2 GB; the default batch of 4 was not measured.

**Terminal close, checked by the owner on 16 November 2026.** The owner started the spike daemon in one PowerShell window, closed that window, and asked for its status from a new one. The daemon answered on its named pipe after 72 seconds of running, and then stopped cleanly on request. A daemon started this way survives its terminal being closed on Windows 11.

**Still not verified**

- Hooks packaged in a plugin on Windows, including SessionStart. The spike registered hooks in a settings file.
- Exec-form hooks by bare name on Linux. Claude Code was not run in the containers.
- Native Linux timings (the containers ran in a virtual machine), and anything on macOS.

**Checked on 15 November 2026 while reviewing what else to spike**

- **Minimum C library on Linux (observed).** The prebuilt ONNX Runtime library that the `ort` crate downloads needs the GNU C Library (glibc) 2.38 or later and a matching C++ library. In an Ubuntu 22.04 container (glibc 2.35) the embedding spike did not link, and a binary built on Ubuntu 24.04 refused to start with `version GLIBC_2.38 not found`. The same spike built and passed its self-test on Ubuntu 24.04 and in Fedora 45, Arch Linux and Ubuntu 26.04 containers. Version 1 lists Ubuntu and Linux Mint as supported without naming a version. Decided by the owner on 16 November 2026: version 3.0 requires glibc 2.38 or later on Linux, and the README says so. That includes Ubuntu 24.04 and later, Debian 13, Red Hat Enterprise Linux 10, and current Fedora and Arch Linux. It excludes Ubuntu 22.04, the Linux Mint releases based on it, Debian 12 and Red Hat Enterprise Linux 9. Support for older systems is looked at only if the need arises.
- **Lower priority for the bulk session (run on Linux, not on Windows).** `ort` 2.0.0-rc.15 has `SessionBuilder::with_thread_manager`. On Linux, ONNX Runtime handed thread creation to the program and the system accepted a lower priority for every bulk thread. The thread that calls `run` also does inference work, so it is lowered too. Two cores cannot show contention; the 16-core run decides.
- **Memory (measured on Linux).** One loaded session of the full-precision model peaked at about 1.5 GB. A query session and a bulk session together peaked at about 2.2 GB.
- **`vec1` (built with GCC, not with the Microsoft compiler).** `vec1.c` compiled as two objects, one with AVX2 (Advanced Vector Extensions 2) and one scalar, chosen at run time. It linked statically with SQLite, registered through `sqlite3_vec1_extra_init`, reported AVX2 as active, and answered exact and row-range queries. With the Microsoft compiler the AVX2 object needs `/arch:AVX2`; without it only the scalar code is built.
- **Linux testing in containers.** The owner has Docker on the Windows 11 machine and has used it to test on Fedora and Omarchy. Omarchy is a configuration on top of Arch Linux; one search found no Omarchy container image, so the official `archlinux` image stands in for it. A container tests the build, the C library, local sockets and the acceptance tests. It does not test Claude Code's hook integration unless Claude Code is signed in inside the container, and its timings are not native Linux timings. The spike repository's container script ran in the author's sandbox against all four images.

**The spikes, all approved by the owner on 15 November 2026:** `s0-preflight` (is the machine ready), `s1-hook-timing`, `s2-mcp-scope`, `s3-claude-hook` (now also the SessionStart, Stop and SessionEnd events, the payload's key names and working-directory form, and whether a daemon started by a hook outlives the session), `s4-embed-selftest`, `s5-embed-scale` (throughput and query time at 1 to 32 threads), `s6-embed-priority` (query time while bulk work runs at lower priority), `s7-vec1`, and `s8-daemon-pipe` (auto-start, lock, local socket, and the full hook path time). On Linux, s1, s2, s4, s5 and s6 have run. s0, s7, s8 and the additions to s3 are specified but not yet written; writing them is part of the spike prompt.

**Reference on `master`:** `spatial_memory/hooks/` (dispatcher, pipeline, signal detection, content extractor, redaction, overlap detector, transcript pipeline, and one module per event), `plugin/hooks/hooks.json`, `spatial_memory/services/queue_processor.py` and `ingest_pipeline.py`, and the hook tests. The patterns and test payloads are ported; the Python structure is not.

**Not verified**

- Version 1 leaves SessionStart out of its plugin hooks because of a Claude Code bug on Windows at version 2.1.37. On 16 November 2026 a SessionStart hook registered in a project settings file ran on Windows with Claude Code 2.1.325. Hooks packaged in a plugin were not tested, so the build checks them in phase 0.
- The payload fields were checked against the official hooks reference on 15 November 2026 and observed on Windows on 16 November 2026 (spike s3). Hook start time on Windows is measured: 3.5 ms for the program alone and 3.6 ms for the whole path through the named pipe.
- The 200 ms limit is a design choice, not a measurement.
- An earlier draft of this section hid medium-signal captures as "proposed". That was dropped on 15 November 2026 after checking version 1, which saves both tiers. The capture analysis of 17 November 2026 overturned it: every capture now waits as `pending` (B-157, `CAPTURE-RCA.md`).
