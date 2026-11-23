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
| `stop --hold`, `release` | `stop --hold` writes the profile's hold file, `daemon.hold`, then stops the daemon as `stop` does. `release` removes the hold. For upgrades and uninstalls (B-180). | No |
| `import`, `export`, `migrate`, `reindex` | Bulk paths. They queue work in the daemon and show progress. | No |
| `replay` | Runs stored Claude Code session transcripts through the current capture pipeline, on request only (section 16, Replay; B-172). | No |

**Rules**

1. **Transport.** Unix domain socket on Linux and macOS, named pipe on Windows. Messages are length-prefixed JSON with a protocol version field. The socket lives in the profile directory with user-only permissions. Every request carries the protocol version, the client's working directory and, when the client has one, the session identifier; the daemon resolves the project from the working directory. A client whose version differs from the daemon's asks it to exit and starts a new one, unless the client's build ranks older, by protocol version and then by release number: an older build never replaces a newer daemon. Such a client does not ask it to exit and starts no daemon: it fails at once with `version_mismatch`, naming how to run the installed build, and a hook goes on without the daemon, its capture left in the spool (B-232). The daemon refuses the request from an older client that still asks, one from before B-232, and keeps serving. A build whose number does not rank is not refused, and a client replaces a daemon of such a build. `spatial-memory stop` still stops the daemon, from any build that has the command (B-228). The commit a build was made from is reported, in `hello`, `status` and the daemon's start line, and is never part of the version check: a client and a daemon of one protocol and release match whatever their commits, so a reinstall of the same release does not make the old program's clients replace the daemon. Only `doctor` compares it, with the program on `PATH` (section 16; B-231).
2. **Auto-start.** A client that finds no daemon starts one detached, guarded by a lock file so two clients cannot start two daemons. While the profile's hold file exists, no client starts a daemon: a hook spools its capture or recalls nothing, and any other client fails at once with `daemon_unavailable`, naming the hold and `spatial-memory release`. A daemon started anyway reads the hold after taking the lock and exits 0 without serving (B-180).
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
| `capture_sessions`, `capture_seen` | Each session's transcript read offset and whether capture read the session from its first line (schema version 6, B-184), and the hash of every candidate each project's capture has handled (schema version 3, B-163). | `session_id`, `project_id`, `content_hash` |
| `capture_judge`, `judge_usage` | The judge's queue, with one row per pending automatic capture: its context and its failed attempts, and, once the judge rejects it, when and why (schema version 7, B-190). Each UTC day's judge and duplicate-check calls, verdicts, failures and cost (schema version 4, B-167). | `memory_id`, `context`, `queued_at`, `attempts`, `rejected_at`, `rejected_reason`; `day`, `calls`, `cost_usd` |

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

**Near-identical results take one place** (B-197). Explicit recall and the hook walk their results best first. A result whose memory vector (the unit mean of its chunk vectors) has a cosine of at least 0.91 with a result kept before it is hidden, and the next result takes its place, so the best-scoring memory of each near-identical set is shown. Explicit recall walks its vector results, then its keyword results; each search asks for twice its places. The hook counts a memory already shown in the session as kept, so its near-identical memories are not shown in that session either. Nothing in the store changes, and a hidden memory is still shown by a search it leads. 0.91 is the lowest two-decimal cosine, from the judge's near-duplicate threshold (0.90) up, at which no pair of active memories holding different labelled knowledge in the stored design data is at or above it. It is provisional, and `recall_collapsed_total` is its metric.

**Similarity floor.** The floor is model-specific and must be recalibrated when the model changes. EmbeddingGemma 2 scores sit in a narrow, high band: unrelated same-domain text scored 0.50–0.64, relevant text 0.70–0.88. The default floor is 0.70, which kept 95–97% of relevant results in testing. A floor of 0.75 kept only 59–82%. Thresholds carried over from other models (0.3–0.5) would pass everything. The floor and the result cap stay provisional until the evaluation on real memories in section 11 has been run.

**Ambient recall (the hook).** The hook injects memory into a prompt without being asked, so it is gated harder than explicit recall.

| Gate | Default | Why |
| --- | --- | --- |
| Scope | Current project only | Cross-project results are noise unless requested. |
| Similarity floor | 0.70 | See above. |
| Result cap | 5 memories, version 2's default | The floor alone still passes several near-misses per query in a dense store. |
| Token budget | One line per memory, shown whole up to 1,200 characters, else cut at the last space before 1,200 and marked ` [cut]`; 1,700 tokens is the ceiling (B-198, replacing version 2's 200-character cut) | Shows a decision with its reason: at least 95% of the active memories of every stored evaluation run fit, and 1,200 is the longest capture unit. Five full lines fit with the longest source tag (B-220), and the block stays at or under 6,800 characters, 68% of Claude Code's 10,000-character limit on hook output. A memory over 1,200 characters with more than one chunk is cut to its best chunk, also marked. |
| No repeats | Once per session | Tracked in `injections` by session identifier. |
| Minimum prompt | 4 words | Skips "yes", "continue" and similar. |
| Deadline | 1,500 ms | On expiry the hook prints nothing. |

**Hook input and output.** The hook reads Claude Code's JSON payload from standard input and takes the prompt, the working directory and the session identifier from it. It writes the injection to standard output as plain text in version 2's format: the header line "\[spatial-memory\] ambient recall:", then one numbered line per memory with its score and its text. Version 2 cut the text at 200 characters; version 3 shows it whole up to 1,200 characters and cuts a longer one between words, marked ` [cut]` (B-198). Version 3 adds the memory identifier to each line and, after it, the memory's source tag (B-220, below). Version 2's switches keep their names: SPATIAL\_MEMORY\_AMBIENT\_RECALL (on by default; any other value disables it) and SPATIAL\_MEMORY\_AMBIENT\_RECALL\_TOP\_K (default 5; zero disables it). SPATIAL\_MEMORY\_AMBIENT\_RECALL\_NAMESPACE has no meaning without namespaces and is replaced by SPATIAL\_MEMORY\_AMBIENT\_RECALL\_SCOPE (project or all). A disabled switch or an empty prompt returns before the daemon is contacted. The reference is `spatial_memory/hooks/user_prompt_submit.py` on the `feature/v2-implementations` branch; field names are confirmed against it and the current Claude Code hook documentation, not assumed.

**Source tags (B-220).** Each line of ambient recall says where its memory came from, in one short fixed tag: `  <rank>. (<score>) [<id>] [source: <tag>] <text>`. Every memory a tool returns carries the same tag as `source_tag` (section 8). The tag is worked out from the memory's provenance each time it is shown; nothing stored changes.

| Provenance | Tag |
| --- | --- |
| `remember`, by the agent or a person | `saved on request` |
| Capture of a message the user wrote (`prompt`) | `user message` |
| Capture of a compaction summary (`summary`) | `session summary` |
| Capture of a turn's final text (`reply`) | `assistant reply` |
| Capture of text written to a file by Edit, MultiEdit or Write (`tool_input`) | `written to a file` |
| Capture of the prose of a Bash, Read or Grep result (`tool_result`), or a PostToolUse capture of any tool but Edit, MultiEdit, Write and NotebookEdit | `file or command output, unverified` |
| `migrate` from version 2, and the evaluation set of `doctor --eval` | `imported` (`import` keeps the provenance each memory was exported with, so its tag too) |
| `sync`, from the repository's shared folder | `repository shared file, unverified` |
| A capture whose kind of text was not recorded | by its hook: PostToolUse as above by its tool, PreCompact and Stop as `assistant reply` (the only text they read before B-179); otherwise `unknown source, unverified` |

"Unverified" marks text that whoever can write a file the session reads, or the repository's shared folder, could have planted (section 9). The longest tag is 34 characters. Five full lines with it come to at most about 6,660 characters, or 1,665 tokens, under the 1,700-token ceiling.

**Metrics journal.** Version 2's journal design is kept: always on, one JSON line per event, appended atomically in lines under 4,096 bytes to day-stamped per-process files in the profile's metrics folder, pruned after 7 days, and silent on write errors. Each hook call records time, project, latency, top scores, injected memory identifiers and the gate that stopped it. Version 2's five metrics keep their names and their purposes: capture\_size\_tokens and capture\_cap\_hit\_total (how large captures are and how often the index cap cuts them), recall\_result\_count (whether query expansion is worth building), reinforced\_memory\_rank (whether a reranker is worth building), and embedder\_cold\_start\_seconds (set once per daemon start; a cold start on a request is a regression). The status tool returns their 7-day medians and 95th percentiles on request. Following version 2's method, version 3 adds one metric for each of its own unproven defaults: hook\_gate\_total by gate and hook\_top\_score (the similarity floor and the result cap), chunks\_per\_memory (the chunk limit), recall\_weak\_result\_total (how often explicit recall returns only weak results) and recall\_collapsed\_total (how many near-identical results recall hid, B-197). Automatic capture adds capture\_forgotten\_total, the captured memories removed with `forget` (B-200). Prompt text is not logged unless `journal.include_prompts` is turned on. This is the only evidence for tuning the gates after release.

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
- Results carry memory identifier, project, type, status, score and provenance, so an agent can cite what it used. Each memory also carries its source tag (`source_tag`, section 6, B-220), which the token budget never sheds.
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
- **Captured file and command output is unverified (B-220, B-221, B-236).** With `capture.tool_results` on, capture offers the prose of what a session read with Bash, Read and Grep (section 16). It is off by default (B-236). A statement planted in a file of a cloned repository, or in fetched content a command prints, can be judged worth keeping and then be recalled in every later session of the project. Recall tags such a memory `file or command output, unverified` (section 6). With `capture.tool_results` off, capture never offers that text, and `forget` removes a memory by the identifier each recalled line shows. The judge is a filter for worth, not for trust.

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

13) **Capture from tools.** Fixture PostToolUse payloads for Edit, Write and Bash, with those tools named in `capture.tools` (none is named by default, B-160): strong-signal text is stored as a pending memory (B-157) with capture provenance, weak-signal text stores nothing, and a fake credential in the payload is redacted.
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
| Capture judge | The capture analysis of 17 November 2026 (`CAPTURE-RCA.md`) showed that the signal rules alone store narration, status reports and tool output (RC4, RC6, RC7). | Decided 17 November 2026: for capture only, the owner withdrew the rule that no model other than the embedding model is used. A Claude model judges each capture through Claude Code, under the user's own login (section 16, The judge). Defaults: on, model `haiku`, at most 200 calls a day (600 since B-209), and pending captures deleted after 14 days (B-165), since narrowed to captures the judge gave up on or never takes (B-178). |
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
| Claude Code | 2.1.325 | The capture judge's call, flags, isolation and cost, on Windows (`SPIKE-JUDGE.md`, B-158) |

## 16. Automatic capture

Version 3.0 keeps version 1's automatic capture: hooks watch a session and store decisions, fixes and conventions without being asked. What changes is where the work happens. The hook hands its payload over and exits. The daemon extracts, classifies, redacts and stores, and a Claude model judges what was stored before any of it is served (The judge, below).

| Event | What the hook does | What it prints |
| --- | --- | --- |
| PostToolUse | Sends the tool call to the daemon when `capture.tools` names its tool. None is named by default (B-160, which replaces "Edit, Write, Bash, NotebookEdit"): a tool's material is a command, its output or code, not a conclusion. Registered as asynchronous. | The same continue response as Stop |
| PreCompact | Sends the transcript path, so the texts not yet read (step 1) are scanned before the context is compacted. | Nothing |
| Stop | Sends the transcript path and the final assistant text from the payload's last\_assistant\_message field. The daemon reads the transcript since the offset (step 1), then offers the message as the turn's reply, since Claude Code does not guarantee the transcript holds it at that moment (B-179, which replaces "it does not read the transcript"). Skips all work when `stop_hook_active` is true. | `{"continue": true, "suppressOutput": true}` |
| SessionStart | On startup and resume only, adds a short note that project memory is available. | Additional context |
| UserPromptSubmit | Ambient recall (section 6). | The injection |

**Pipeline in the daemon**

1. Extract the texts offered to the judge from the tool call, or from the transcript since the offset the store holds for that session (B-162, B-163). From the transcript (B-179, which replaces "the final text of each turn"): each message the user wrote, without Claude Code's `<system-reminder>` blocks; each compaction summary, in a session seen from its start (B-184: capture has read every text before the summary, from the transcript's first line; a session capture joined part-way, read only after a boundary or open before setup, never has its summaries offered, because a summary restates history capture did not read), less two of its sections and its repeats (B-189: the units under an "All user messages", "User messages" or "Pending tasks" heading are not offered, nor a unit at least 0.7 of whose sentences of 20 characters or more the same session offered before in a summary, its heading lines left out, since every summary repeats its section headings (B-196); those sentence keys are held in memory only, at most 8,192 a session; each unit not offered is journalled `capture_dropped_total` with `stage` `select`); each turn's final assistant text; and the prose of the new text of each Edit, MultiEdit and Write (up to 4,000 characters a call) and of the output of each Bash, Read and Grep call that did not fail (up to 1,500 characters a result). Prose is a run of lines of words, at least 120 characters long; code, tables of values, paths and logs are not prose, and raw command output and file contents are never offered whole. Text written before a tool call announces the next action and is not a turn's final text. Each text is split into units of at most 1,200 characters at paragraph, line and sentence ends, a heading starting a new unit; a unit under 20 characters is dropped. Each unit is one candidate. A session with no recorded offset, such as one resumed under a new id, is read only after its last compaction boundary. A session open before setup gets no history: its first Stop reads only its own turn and records the end of the transcript as its offset. The safety cap counts from the newest text back (B-161). A turn's final text carries, as its context, the user prompt that started its turn, cut to 2,000 characters. Any other text from a transcript carries a `Source:` line naming its kind, then that prompt. A text from a tool call has no context (B-166). Provenance records the kind of text as `source`: `reply`, `prompt`, `summary`, `tool_input` or `tool_result`. With `capture.tool_results` off (B-221), the default (B-236), no `tool_result` text is offered, on replay too: each one is left out of the read and journalled `capture_dropped_total` with `stage` `select` and reason `tool_results_off`, and the read offset still passes it. PostToolUse then skips any tool but Edit, MultiEdit, Write and NotebookEdit, even one `capture.tools` names, as it skips a tool not named.
2. Classify it with the signal patterns ported from version 1. A score of 0.8 or more is a strong signal, 0.5 to 0.79 is a medium signal, and anything lower is weak. For a text from the transcript the patterns only set the type, tags and importance: a weak text is offered at importance 0.4, because the patterns passed only 16% of the statements worth keeping (B-179, which replaces "anything lower is dropped"). For a tool call (PostToolUse, opt-in) a weak text is dropped.
3. Redact secrets with the scanner of section 9.
4. Drop duplicates: a content hash the project's capture has handled before, whatever became of its memory, so a forgotten capture is never captured again from the same text (B-163); the same content hash as an existing memory, or as an item already captured in this session; and a near duplicate, whose cosine to an active or pending memory of the project, or to an earlier candidate of the same read, is 0.93 or more (B-169, which replaces B-164's 0.85; provisional, journalled as `capture_dropped_total`). A pending capture that is not in the judge's queue was made by the rules before the judge and is never judged, so it does not count here, even when its hash was handled, and a capture of its text replaces it (B-176). This check runs before anything has judged the candidate, so its threshold is set to drop no item the labellers called worth keeping.
5. Store it as a pending memory, whatever its tier (B-157, which replaces "store it as an active memory, as version 1 does for both tiers"). A pending memory is never served: ambient recall, `recall`, `related`, `regions`, `journey` and `consolidate` never return it, `recall` and `revise` refuse `pending` by name, `journey` refuses it as an endpoint, and `share` refuses it. `related` and `link` refuse it as their start or end, no walk passes through it, and `link list` leaves out a link to it. `export` never writes it or a link to it. `import` never stores a `pending` line, an active capture from an export of store schema below 2, or a link to a stored pending capture. `forget` deletes it but only counts it in its reply (`pending`), and no error the judge's answer causes repeats the answer's text (B-181). `revise` refuses to make it superseded or retracted, or to change it in any other way unless the change makes it active; `remember` refuses to supersede it; and `consolidate` refuses to merge it or keep it in a merge, and checks every group before it merges any (B-176). Only a capture the judge keeps becomes active (step 7), or one the user revises to `active`. The capture waits in the judge's queue with its context (B-167). A save of the same text in the same project replaces a pending capture. The tier sets the stored importance: the higher of 0.7 and the score for a strong signal, and the higher of 0.4 and 0.8 times the score for a medium one.
6. Embed it on the bulk path, at low priority. This happens before the store, for the near-duplicate check of step 4, and the memory is stored with its chunks embedded. When the embedder fails, the candidate is stored without that check and the queue embeds it later (B-164).
7. Judge it in the background, as described below. A kept capture becomes active, and its content is the sentences of its own text that the judge marked (B-173, B-191). A rejected one, by the judge or at settling, stays pending, unserved and never judged again, and is deleted 30 days after its rejection (B-190, which replaces "is deleted").

**The judge (B-165 to B-170, B-173, B-182, B-189 to B-194)**

On 17 November 2026, for capture only, the owner withdrew the rule that no model other than the embedding model is used. The signal rules no longer filter transcript texts (B-179). A Claude model decides, run through Claude Code under the user's own login.

- **What it returns.** For each candidate the judge returns four things:
  - keep or reject;
  - a reason code: `knowledge` for a kept capture, and `narration`, `status`, `tool_output`, `transient`, `restatement`, `fragment`, `duplicate` or `other` for a rejected one;
  - the type, one of the six;
  - for a kept capture, its spans: one to six statements worth keeping, each given by a start phrase and an end phrase, four to ten words each, copied word for word from its text (B-191, which replaces quotes of single sentences).

  The judge never writes the memory (B-173, which replaces a rewrite of at most 4,000 characters). An answer that does not judge each candidate exactly once, that keeps a capture without a span, that gives a span without both phrases, or that breaks any of these rules, settles nothing (`bad_answer`).
- **The memory is the text's own sentences.**
  - **Looking up the spans.** Each span's start phrase is looked up in the capture's redacted text, then the first place at or after it where its end phrase is found. The comparison ignores `*`, backticks, typographic quotation marks and runs of whitespace, plus a phrase's leading list marker and closing punctuation. A phrase must start and end at a word boundary. A span whose start is not found, or whose end is not found at or after its start, is dropped.
  - **Building the memory.** Each span found brings every whole sentence from the one its start touches to the one its end touches (B-191). A sentence on an indented line, such as a nested list item or the continuation of a list item, is read under the nearest line above it that is less indented. So it also brings the opening sentence of that line, and of that line's own parent, up to a line that is not indented (B-174). The memory is those sentences, in the text's order, joined by line breaks.
  - **When nothing can be built.** A kept capture with no span found (`unquoted`), or whose sentences exceed 4,000 characters (`too_long`), is not settled. It stays pending and unserved, counts an attempt, and is journalled with that verdict. It is not a failed call.
- **When it runs.** One daemon thread makes the calls, one at a time, never inside a hook's time budget. The hooks still only hand over their payload and exit 0.
  - It calls when 25 captures wait, or when the oldest has waited 5 minutes.
  - A call takes at most 25 captures and 100,000 characters.
  - Fresh captures go first, oldest first.
  - After a bad answer, the batch is split in two and each half judged again, down to a single capture; a single capture with two bad answers is marked failed judgement and waits only to be pruned (B-206). A capture whose verdict could not be applied is retried alone; after 3 such attempts it waits only to be pruned.
- **The call.** It runs `claude -p` with the model, the prompt and the JSON schema the spike proved (B-158, `SPIKE-JUDGE.md`). The prompt is version 10 (`docs/v3/spikes/judge/prompt.txt`, B-191, which replaces version 6 of B-182).
  - There are no tools and no settings sources, and no session is saved.
  - It runs in an empty folder under the data root, at lower priority, and is stopped after 180 seconds.
  - Prompt caching is off (`DISABLE_PROMPT_CACHING=1`): every call's candidates are new, so writing them to the cache only costs more (B-182).
  - Candidates are numbered `c01` upward, never by memory id.
  - The schema is fixed to the batch: exactly one verdict per candidate, each `id` one of the batch's (B-208). With a schema that allowed any number of verdicts, Haiku sometimes stopped one verdict short (B-207).
  - A failed call, other than a stopped one, is kept for diagnosis in `judge/failures/` under the data root: its input, schema, exit code, failure and output, the newest 20 for up to 30 days (B-207). It never leaves the machine.
- **Redaction.** Text and context are redacted when captured, and again before the call. The memory is cut from the redacted text, and is redacted again before it is stored (section 9).
- **Settling.** Verdicts are settled in one transaction, and only for a capture that is still pending.
  - A kept capture keeps its id, provenance, tags, importance and anchors. It takes the memory's content, title and chunks, and the judge's type, and its provenance records the judge's model and reason.
  - A kept capture's neighbours are the closest 5, at cosine 0.70 or more, of the project's active memories and the memories of the same call, for the same project, that are made active before it (B-196). Kept captures are settled in the order of the call, so a capture rejected for any reason is never a neighbour of a later one.
  - A kept capture whose memory's cosine to its closest neighbour is 0.90 or more is rejected as `near_duplicate` (B-169).
  - **The duplicate check (B-192).** Every other kept capture whose memory has neighbours goes into one more call for the batch. The call is made before the batch is settled, so each capture is asked about with the neighbours it would have if every earlier kept capture that is not a near duplicate were made active (B-196). It runs as the judge's call does, with its own prompt and schema (`docs/v3/spikes/judge/repeats-prompt.txt`, `repeats-schema.json`), and asks whether the memory adds anything to its neighbours. One that adds nothing is rejected as `repeats`; one that adds anything is made active whole, unchanged. The answer is used only when it holds for the neighbours the capture has at settling: when they are the ones it was asked with, or, for an answer that it adds something, some of them. Otherwise the capture stays pending, counts a failed attempt and is judged again alone; it is journalled `capture_judged_total` with the verdict `recheck` (B-196). A failed or malformed answer settles none of the call's captures, which wait and retry as after a failed call.
  - A kept capture is rejected as `duplicate` if a memory of the project that is not pending already holds its content hash.
  - Each rejection is kept as step 7 says and journalled as `capture_dropped_total` with `stage` `judge` and its reason (B-190).
- **When it cannot run, captures wait.** The judge cannot run when:
  - the judge is off;
  - capture is off;
  - Claude Code is not found or not signed in;
  - the day's call cap is reached;
  - the day's spending cap is reached (B-193);
  - a call failed.

  A failed call backs off for 5 minutes, doubling up to 1 hour. A rate limit waits for its reset time. The call cap counts every call per UTC day, the duplicate check's included. The spending cap compares the day's recorded cost with `capture.judge.max_usd_per_day` before each batch; a batch started under it finishes, duplicate check included, so a day can end over it by one batch's cost.
- **Pruning (B-178).** Waiting alone never deletes a capture the judge can still take: with the judge off, out of calls, or unable to run, every queued capture waits, however long. A capture the judge gave up on, marked failed judgement after two bad answers alone (B-206) or with three verdicts that could not be applied, is deleted `capture.judge.pending_max_days` after the last of them. Captures that a version before schema version 4 left pending have no context and are never judged; each is deleted `pending_max_days` after the upgrade (B-167). Schema version 5 records when a bad answer came; an earlier bad answer is dated to the upgrade, so the upgrade itself deletes nothing. A rejected capture is deleted, with its queue row and chunks, 30 days after its rejection and not before (B-190).
- **Cost (B-182, B-193).** Measured with haiku and prompt version 10 on the design data of B-177:
  - about 0.00019 US dollars a candidate for the judge and 0.00006 for the duplicate check, about 0.00025 together;
  - about 0.43 a day on the owner's typical day, about 1,730 candidates, measured from 31 days of transcripts; about 1.40 on the busiest of them;
  - the call cap is 1,000 a day (B-209, B-250). Over the 30 days of transcripts, the 95th percentile day needs about 410 calls under `verdicts`, duplicate checks included, and the busiest about 500; under `statements` about 770 and 940, both calls included. A 95th-percentile `statements` day costs about 1.92 to 3.32 US dollars, so the dollar cap of 3.00 can stop it; 4.00 is proposed and not made (B-250). On a busier day the rest wait. After a bad answer, the failed batch is split in two and each half judged again, down to a single capture. A single capture with two bad answers is set aside as failed judgement (B-206). A rechecked capture is judged alone;
  - `capture.judge.max_usd_per_day` (3.0) stops new batches for the rest of the UTC day.

  These are Claude Code's estimates at API prices. Under a subscription they are plan usage. `setup` says so before it finishes, with how to turn the judge off, and the README says the same (B-170).
- **What it shows.** `status` and `doctor` say whether the judge can run, and why not. They show the pending count, the rejected count (B-190), and today's calls and estimated cost beside the day's spending cap (B-193). The journal records `capture_judged_total`, `capture_judge_failures_total`, `capture_judge_cost_usd` (one line a call, a failed one with the label `failed`, B-202), `capture_pending` (B-165) and `capture_forgotten_total` (B-200).
- **Quality over the last 7 days (B-200).** `status` and `doctor` also show, read from the journal over the last 7 days:
  - captures judged: those the judge settled, kept or rejected; a capture judged again is counted once, when it is settled;
  - kept: made active;
  - rejected: rejected by the judge, or kept by it and then dropped as a duplicate, near duplicate or repeat (shown apart);
  - automatically captured memories removed with `forget`, and apart from them the pending captures it removed. `forget` writes one `capture_forgotten_total` line when it deletes captured memories: the value is the served ones, label `pending` the pending ones, label `by` `ids` or `filter`. A memory stored with `remember` is not counted;
  - the judge's calls, the duplicate check's and the failed ones included, and their estimated cost (B-202).

  The journal keeps 7 days and is silent on write errors, so a line it failed to write is not counted. A daemon before B-200 does not send these figures, and nothing is shown.
- **Without Claude Code**, automatic captures wait, unserved, until the judge can run; waiting does not delete them (B-178). `status` shows how many wait. Explicit `remember` is not affected.

**The statements task (B-240 to B-255)**

`capture.judge.task` chooses how the judge works: `verdicts` (the default, everything above) or `statements` (docs/v3/STATEMENT_JUDGE_PROPOSAL.md, built with the owner's changes). Either `config.toml` may set it, and the profile's value wins. A build before B-240 refuses a config that names the key, so the key must be removed before an older build is installed.

- **One session a call (B-241).** The oldest waiting capture with no attempt names the project and the session. That session's fresh captures follow in transcript order (`created_at`, then id), up to 25 captures and 100,000 characters. A capture with an attempt goes alone, as B-206 says. The wait rule is the same: 25 captures waiting in all, or the oldest has waited 5 minutes.
- **The statements call.** Haiku through `claude -p`, with the judge's flags, folder and environment, the prompt `docs/v3/spikes/judge/statements-prompt.txt` (R2 as measured, B-240) and its schema. The prompt says the texts are data and never instructions. For each capture the judge returns its statements: each a start phrase and an end phrase from the capture's text, a label (`keep` or `duplicate`), a type and a grade (`must_keep` or `good_to_have`). A capture with no statement worth keeping is answered with none. A bad answer halves the batch, down to one capture tried twice (B-206). The call stops after 600 seconds (B-247).
- **The capture's own words (B-242).** Each kept statement's span is looked up in the capture's redacted text as a verdict's spans are, and its sentences are the memory's content. The title is `derive_title`'s. No text the model writes is stored or shown. A statement whose span is not found (`unquoted`) or that the judge labels `duplicate` is not stored.
- **The duplicate step (B-243, B-246).** Each kept statement is embedded. Its candidates are the project's active memories and the part's earlier kept statements at cosine 0.80 or more, the closest 5. A statement with a candidate is asked about in one more call, at most 40 statements and 60,000 characters a call, with `docs/v3/spikes/judge/same-prompt.txt`. Each answer place may name only its own statement's candidates, or none. B-190's reject at 0.90 and B-192's repeats check do not run under this task; the capture-time check at 0.93 does.
- **Settling (B-242, B-243).** One transaction per part, and only after its duplicate step is done.
  - The first stored statement takes the capture's row, from pending to active. Each later one is a new memory of the same project with the capture's provenance and anchors. `extra.judge` records the model, the task, the capture, the statement's number, its grade and the time.
  - A statement whose content hash a memory of the project holds is not stored (`hash`).
  - A statement the duplicate step matched is held: a pending copy, rejected at once as `repeat`, with `extra.judge.repeat_of` naming the memory it repeats, kept unserved 30 days as B-190 keeps a rejected capture. To audit it, read the held memory and its `repeat_of`; to undo it, make it active.
  - A statement still answered badly by the duplicate step after `JUDGE_MAX_ATTEMPTS` (3) tries, alone in its call, is held the same way with the reason `unchecked`. Its capture's other statements settle.
  - A capture with no statement stored is rejected and kept 30 days, as `unchecked`, `repeats`, `duplicate` or `nothing_kept`.
- **All or nothing (B-244).** The caps and the daemon's stop are checked before each call of a part. A cap reached, a stop, a crash or a failed duplicate step call leaves every capture of the part waiting whole, to be judged again from its statements call. Nothing is stored unchecked for that reason.
- **A source switched off (B-245).** Under either task, a waiting capture whose source is a tool result is rejected without a call while `capture.tool_results` is off, as `tool_results_off`, kept 30 days.
- **Reading a transcript again (B-242).** The capture's content hash as read and the session offsets guard it, so a second read, a new session on the same file or a replay stores nothing again.
- **What it shows.** `status` and `doctor` name the task; under `statements` they give today's statements stored, held as repeats and held unchecked. The journal adds `capture_statements_total`, one line a statement, labelled by `outcome` (`stored`, `repeat`, `duplicate`, `unquoted`, `hash` or `unchecked`), `grade`, `type` and `project`, and `capture_judge_cost_usd` labels each call `statements` or `same`.
- **Measured (B-248, B-252 to B-254).** Through the daemon on the design set and the first hold-out, with `capture.tool_results` on as in the harness, must keep is 105 and 99 of 172 and 104 and 103 of 182; on the same input the daemon matches B-239's harness within the statements call's own run-to-run variance, and leaves no statement unchecked (B-252). On h2, which the design has not seen, with today's defaults and two runs of each task: must keep 118 and 123 of 307 under `statements` against 85 and 71 under `verdicts`; active 1,102 and 1,103 against 224 and 201; junk by direct labelling 16.8% (67 of a sample of 400) against 4.9% (11 of 224); none unchecked (B-253). With one query for each must-keep group either store holds, the `statements` store returns the group for 111 of 153 queries against 84, with fewer repeats among its places (B-254). The default stays `verdicts`.

**Replay (B-172)**

`spatial-memory replay <session>...` or `spatial-memory replay --all` runs stored session transcripts through the current pipeline. It runs only when asked; nothing replays on its own.

- **What it reads.** Claude Code keeps a session's transcript at `<config>/projects/<folder>/<session id>.jsonl`. `<config>` is `CLAUDE_CONFIG_DIR`, or `~/.claude` when that is unset. `<folder>` is the session's working folder with every character other than an ASCII letter or digit replaced by `-`. An argument is a session id of the folder the command runs in, or a transcript file. `--all` takes every transcript of that folder.
- **Which project.** The working folder the transcript records names the project, whichever folder the command runs in. A transcript that records none is skipped.
- **How it is read.** The whole transcript is read, before and after each compaction boundary, as step 1 reads a session seen from its start: every kind of text step 1 offers, compaction summaries included, each with its context and split into units. Each text has the 200,000-character cap; there is no cap over all of them together.
- **What happens next.** Steps 2 to 7 are those of any capture. Each candidate is stored pending and queued for the judge, and nothing is served until the judge keeps it. A content hash the project's capture has handled is dropped, so a replay never stores again what live capture or an earlier replay stored, and never brings back a forgotten capture. The one exception is a capture made by the rules before the judge, which waits outside the judge's queue: a replay of its text replaces it (B-176).
- **What it leaves alone.** It never moves the session's read position, so live capture goes on from where it was.
- **The request.** The command sends one `replay` request per transcript. The daemon replies as soon as the work is queued, on the same queue and spool as captures. With capture off, it queues nothing and the command says so. Provenance records the trigger `replay`. `status` shows the jobs still queued.

**Rules**

- **Never blocks.** Every capture hook exits 0 within 200 ms, whatever happens. It never exits with a blocking code.
- **Never loses.** When the daemon is not reachable, the hook writes the payload to a spool directory: a file written under `tmp/`, then renamed into `new/`. The daemon drains the spool when it starts. This is version 1's queue protocol.
- **The capture state outlives the daemon.** The store keeps each session's read offset and the hash of every candidate each project's capture has handled (schema version 3, B-163). A row unused for 90 days is pruned; Claude Code deletes a transcript 30 days after its last use by default.
- **Importance is stored, not used.** Version 1 saves both tiers and uses the tier only to set importance. Version 3 stores both tiers pending (B-157). Importance does not affect ranking or the prompt hook until the journal shows that it should; `recall` accepts a minimum importance as a filter.
- **Type and provenance.** The pattern family sets a pending capture's type (decision, pitfall, convention), and anything else is a note. The judge sets the type of a kept capture. Provenance records capture as the source, with the session, the tool, the file path and the current commit. For a kept capture it also records the judge's model and reason. A capture from Edit or Write is anchored to its file.
- **Controls.** These are the controls, with their defaults:
  - `capture.enabled` (on);
  - `capture.tools` (empty, B-160);
  - `capture.tool_results` (off, B-221, B-236): off, capture offers no text read from a file or command output (step 1); on, it offers the prose of each Bash, Read and Grep result, and PostToolUse may capture any tool `capture.tools` names;
  - the judge's settings (B-165): `capture.judge.enabled` (on), `capture.judge.model` (`haiku`), `capture.judge.task` (`verdicts`, B-240), `capture.judge.max_calls_per_day` (1000, B-209, B-250), `capture.judge.max_usd_per_day` (3.0, B-193), `capture.judge.pending_max_days` (14, provisional, metric `capture_pending`; it applies only to captures the judge gave up on or never takes, B-178) and `capture.judge.command` (found on `PATH` when absent);
  - ignore patterns for paths such as `.env` files and key files.

  A repository opts out in `.spatial-memory/project.toml`. Captured memories are private until shared.
- **Clients.** `--client claude-code` and `--client cursor`, with version 1's event-name mapping. Cursor gets the capture events only: it has no prompt event and ignores hook output. Cursor is tested with fixture payloads only.
- **Packaging.** A Claude Code plugin under `plugin/` (manifest, MCP entry, hooks file) that calls the installed `spatial-memory` binary. `setup` remains the other way to register. The plugin does not carry the binary in 3.0.

**Carried over from version 2**

- **No per-field size limits.** Version 1 cut captures at 3,000 to 10,000 characters and lost content. Version 2 replaced those limits with one safety cap of 200,000 characters at extraction. Version 3 keeps that cap, counted from the newest text back (B-161), and stores the captured text whole.
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
- **What `setup` does.** It registers the server by calling `claude mcp add` at user scope, not by editing Claude Code's files itself. It merges hooks into `~/.claude/settings.json` after writing a backup, is safe to run twice, has a remove option, and refuses to run while the plugin is enabled so nothing is registered twice. A plugin disabled at any scope, local included, counts as disabled, and the version 1 plugin is never taken for this one (B-150). `setup --remove` unregisters, then stops the profile's daemon the way `stop` does (B-155). Claude Code's /mcp toggle disables a server for one folder by naming it in that project's `disabledMcpServers` in `~/.claude.json`, and an entry left from version 1 hides this server too. `setup` and `doctor` read those lists and never write them. `setup` notes each project whose list names the server. `doctor` fails while the server is registered and the current folder's list names it, and it names any other such folders without failing (B-226).
- **The binary is found on `PATH`.** The plugin does not carry it in 3.0. `spatial-memory --version` prints the release, then `+` and the commit the program was built from, such as `3.0.0+1b5b54a` (B-231). `doctor` checks that the release on `PATH`, the part before `+`, matches the plugin's. It also checks that the daemon was built from the same commit as the program on `PATH`. It does not compare them when the program on `PATH` prints no commit, when either was built without git (`unknown`), when the daemon's `hello` names another build than `doctor` itself, which the build check reports, or when the program on `PATH` is another release than the daemon, which the path check reports. A daemon that reports no commit was built before B-231 and fails. A daemon whose replies carry another build does not reach these checks: one of an older build, of the same rank or of a build that does not rank is replaced by `doctor`'s own first, and one of a newer build is left running, so the daemon check fails with `version_mismatch` and names how to run the installed build, and neither the build check nor the commit check runs (B-232).

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
