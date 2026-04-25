# Migrating from v1.x to v2.0

> **Status:** Current — applies to spatial-memory-mcp 2.0.0. See [`internal-docs/v2-plan.md`](../internal-docs/v2-plan.md) and [`internal-docs/embedding-model-upgrade.md`](../internal-docs/embedding-model-upgrade.md) for design notes.

## TL;DR

v2.0 is a **clean break**. Your v1.x database is not opened in place. Instead:

1. **While still on v1.x**, ask your LLM to export your memories to a file.
2. **Upgrade** spatial-memory-mcp to v2.0.
3. **Delete the old database directory.**
4. **Restart your client** so the v2.0 server starts with a fresh database.
5. **Ask your LLM to import** the export file. v2.0 re-embeds every memory under the new default model.

No memories are lost. The export contains your content, tags, importance, namespaces, and project — everything except the raw vectors, which are regenerated under the new model.

---

## Why a clean break?

v1.x stores embeddings as 384-dim vectors from `all-MiniLM-L6-v2` with a 256-token context window. v2.0's new default is **`snowflake-arctic-embed-m-long`** — 768-dim vectors with an **8192-token context window** (32× more). The vector width and the model itself are different, so v2.0 cannot read v1.x vectors at all — they're in a different representation space.

Rather than ship a one-shot in-place migration tool that gets used once and then carries forward as legacy code forever, v2.0 leans on the export/import path that already exists in v1.x. The import already re-embeds content under whatever model the running server is configured with — it's the natural cutover.

You only do this once, per database, when you upgrade.

## What changes for you

| Axis | v1.x default | v2.0 default | Impact |
|---|---|---|---|
| Embedding model | `all-MiniLM-L6-v2` | `snowflake-arctic-embed-m-long` | Higher retrieval quality |
| Vector dimension | 384 | 768 | 2× storage per vector (still tiny) |
| Context window | 256 tokens (~200 words) | up to 8192 tokens (~6000 words) | Long memories no longer truncated |
| CPU embed latency | ~0.2 ms | ~2–3 ms | Invisible against LLM turn latency |
| Cold-start model load | ~0.5 s | ~2.5–3.5 s | One-time cost on server start |
| On-disk model size | ~80 MB | ~550 MB | Larger first download |

The headline win is **context window**: memories larger than ~200 words no longer need chunking. A whole conversation, design doc, or root-cause analysis fits in one memory.

If you want different trade-offs (faster embedding, smaller storage, constrained hardware), v2.0 ships several alternates as first-class aliases — see [Picking a different model](#picking-a-different-model).

---

## Migration steps

### Step 1 — While still on v1.x, export your memories

In your LLM client (Claude Code, Cursor, etc.) connected to the v1.x server, ask:

> "Export all my memories to `~/v1-backup.parquet`."

The LLM calls the `export_memories` MCP tool, which writes the file. Parquet is recommended because it preserves all fields with full fidelity. JSON works too if you prefer to inspect the file by hand.

You can verify the export by checking the file exists and is non-empty:

```bash
ls -lh ~/v1-backup.parquet
```

The export includes content, tags, importance, namespaces, project, timestamps, and access counts. It also includes the v1.x vectors, but **v2.0 ignores those during import** — it regenerates them with the new model.

### Step 2 — Upgrade spatial-memory-mcp to v2.0

Pick whichever applies to your install:

```bash
# pip
pip install -U spatial-memory-mcp

# pipx
pipx upgrade spatial-memory-mcp

# uvx (refresh the cached version)
uvx --refresh spatial-memory-mcp --version
```

### Step 3 — Delete the old database directory

The v1.x database lives at `.spatial-memory/` (or wherever `SPATIAL_MEMORY_MEMORY_PATH` points). v2.0 will refuse to open it — your safe move is to delete it now that you have an export.

```bash
# Default location
rm -rf .spatial-memory/

# Or, if you set a custom path
rm -rf "$SPATIAL_MEMORY_MEMORY_PATH"
```

If you're nervous, rename it instead — you can always remove the rename later:

```bash
mv .spatial-memory .spatial-memory-v1-backup
```

### Step 4 — Restart your client

Quit and relaunch Claude Code / Cursor / your MCP client so it spawns a fresh v2.0 server process. The first server startup will:

- Download the new default model (`snowflake-arctic-embed-m-long`, ~550 MB) — this is a one-time cost.
- Create a fresh empty database at the configured path.
- Verify the embedding-model registry resolves correctly.

You can confirm the new server is healthy by asking your LLM:

> "Run the spatial-memory `doctor` command and show me the output."

The output should show the configured model is `arctic-m-long` and the database is fresh (zero stored vectors).

### Step 5 — Import your memories

Back in the LLM client connected to v2.0, ask:

> "Import memories from `~/v1-backup.parquet`. Set `regenerate_embeddings=true` and `dry_run=false`."

Both flags matter:

- **`regenerate_embeddings=true`** — re-embeds every memory's content with the v2.0 model. Without this, the import would try to use the v1.x vectors, which have the wrong dimension and would be rejected.
- **`dry_run=false`** — the import tool defaults to dry-run (preview only) for safety. You must explicitly say "actually import."

The import streams through the file in batches. Throughput depends on hardware:
- ~50–200 memories/second on commodity CPU with the new default model.
- A 5,000-memory database imports in 30s–2min.

When import completes, the LLM should report `memories_imported`, `memories_skipped`, and `memories_failed` counts. If anything failed, the response includes per-record validation errors.

You're done. Your memories are back, semantically searchable under the new model, with whole-document context where applicable.

---

## CLI alternative (v2.0+)

v2.0 ships `export`, `import`, and `doctor` subcommands so the migration can be
scripted end-to-end. The CLI flips one default vs. the MCP tool: **`--dry-run`
defaults to `False` on the CLI** (opt-in via `--dry-run` for a preview), the
opposite of the MCP tool. This matches typical Unix-tool ergonomics where
invoking the command does the thing.

```bash
# Step 1 (must run while v1.x is still installed)
python -m spatial_memory export --output ~/v1-backup.parquet

# Step 2: upgrade
pip install -U spatial-memory-mcp

# Step 3: delete old DB
rm -rf .spatial-memory/

# Step 4 (optional): preview the import
python -m spatial_memory import --source ~/v1-backup.parquet --regenerate-embeddings --dry-run

# Step 5: actually import (no flag needed — write is the default)
python -m spatial_memory import --source ~/v1-backup.parquet --regenerate-embeddings

# Step 6 (optional): verify the new DB matches the configured model
python -m spatial_memory doctor
```

---

## Recovery: I already upgraded without exporting first

If you upgraded the package and tried to start v2.0 against your v1.x database, you'll see an error like:

```
ERROR: This looks like a v1.x database (N memories, 384-dim vectors, no embedding metadata).
v2.0 defaults to 768-dim (arctic-m-long) and cannot open v1.x data in place.
```

Your data is **not lost** — v2.0 refused to touch it precisely so you could recover. To export from v1.x:

```bash
# 1. Install v1.x alongside v2.0 with a different command name
pipx install spatial-memory-mcp==1.11.4 --suffix=v1

# 2. Point spatial-memory-v1 at your existing data
SPATIAL_MEMORY_MEMORY_PATH=.spatial-memory \
spatial-memory-v1 serve &

# 3. Connect Claude Code (or your client) to the v1 server temporarily,
#    and ask it to export to a file:
#       "Export all memories to ~/v1-backup.parquet"

# 4. Stop the v1 server, uninstall it, and proceed with the normal migration:
pipx uninstall spatial-memory-mcp-v1
rm -rf .spatial-memory
# ... then continue from Step 4 of the main migration above.
```

If `pipx` isn't available, the same thing works with a virtualenv:

```bash
python -m venv /tmp/sm-v1-recovery
/tmp/sm-v1-recovery/bin/pip install spatial-memory-mcp==1.11.4
SPATIAL_MEMORY_MEMORY_PATH=.spatial-memory \
/tmp/sm-v1-recovery/bin/python -m spatial_memory
```

---

## FAQ

### My v1.x install was using a custom embedding model, not the default. Does this still work?

Yes, with one extra consideration. The export contains the **content** of every memory, not just the v1.x vectors. v2.0's import re-embeds that content with whatever model v2.0 is configured to use. So your custom v1.x model is irrelevant to the migrated database.

If you want v2.0 to keep using your custom model, set `SPATIAL_MEMORY_EMBEDDING_MODEL=<your-model>` in your environment **before** the first v2.0 ingest. Once v2.0 has stored even one memory, the model is pinned to whatever was active at first write.

### What about my hooks and plugin configuration?

Unchanged. v2.0 keeps the v1.x hook surface, MCP tool names, and plugin config format. No re-setup needed.

### Will v2.0 work with my v1.x `.env` file?

Mostly. `SPATIAL_MEMORY_EMBEDDING_MODEL=all-MiniLM-L6-v2` resolves to the legacy `minilm` alias automatically, so you won't get a "model not found" error. But it will also block the v2.0 default — if you want the new `arctic-m-long` default, remove that line (or set it to `arctic-m-long` explicitly).

The `SPATIAL_MEMORY_EMBEDDING_DIMENSIONS` setting has been removed (the registry owns dimensions now). If you have it in your `.env`, v2.0 will ignore it.

### How big are the export files?

Roughly:
- Parquet: ~1 KB per memory (compressed, full fidelity including v1.x vectors).
- JSON: ~3–5 KB per memory (human-readable).
- CSV: ~1–3 KB per memory (no vectors by default — fine for migration since we re-embed anyway).

A 10,000-memory database exports as ~10 MB Parquet, ~30–50 MB JSON.

### Can I migrate in pieces (one namespace at a time)?

Yes. The export tool supports `namespace` and `project` filters; the import tool supports `namespace_override`. So you can:

1. Export only the `default` namespace from v1: `"Export memories from namespace 'default' to ~/default.parquet"`
2. Upgrade and start v2.0.
3. Import that namespace: `"Import ~/default.parquet, regenerate embeddings, no dry run"`
4. Export and import each remaining namespace separately.

This is useful if you want to migrate critical namespaces first and test before bringing the rest over.

### What if I want to roll back to v1.x?

v2.0 doesn't write to the v1.x data directory at all (it refuses to open it). So if you didn't delete the v1.x directory yet, just downgrade the package and point the server at it:

```bash
pip install spatial-memory-mcp==1.11.4
# v1.x picks up your old .spatial-memory/ directory unchanged.
```

If you already deleted the v1.x directory but kept the export file, you can downgrade and re-import into a fresh v1.x database — but you'll lose the v1.x vectors and have to let v1.x re-embed under MiniLM. That's identical to the v2.0 path, just with the smaller model.

### Picking a different model

The new default is `arctic-m-long` (768-dim, 8K context). If that's not the right trade-off for you, v2.0 ships several alternates. Set `SPATIAL_MEMORY_EMBEDDING_MODEL=<alias>` before first ingest:

| Alias | Dim | Context | Best for |
|---|---|---|---|
| `arctic-m-long` *(default)* | 768 | 8192 tok | Most users — long memories, good quality |
| `nomic` | 768 | 8192 tok | Same size class, Matryoshka-trained |
| `arctic-s` | 384 | 512 tok | Fast embedding, short memories, half the storage |
| `arctic-xs` | 384 | 512 tok | Raspberry Pi / constrained hardware |
| `bge-small` | 384 | 512 tok | Community-familiar 384-dim alternate |
| `e5-small` | 384 | 512 tok | Symmetric query+passage prefix |
| `minilm` | 384 | 256 tok | v1.x continuity (not recommended for new data) |

You can also pass any HuggingFace model path directly (e.g. `BAAI/bge-base-en-v1.5`) — the registry treats unknown names as passthrough.

### What if import fails partway through?

The import is record-by-record. If it fails after importing some records, those records are in the v2.0 database. You can:

- Re-run the import on the same file. By default the importer doesn't deduplicate, so you'd get duplicates — pass `deduplicate=true, dedup_threshold=0.95` to skip already-present records.
- Or, delete the v2.0 database and start over: `rm -rf .spatial-memory/` then re-run import from scratch.

The import response includes per-record validation errors, which usually point at the failing field.

---

## Error message decoder

| Error | What it means | Fix |
|---|---|---|
| `This looks like a v1.x database (N memories, 384-dim vectors, no embedding metadata)` | v2.0 won't open v1.x data in place | Follow [Recovery](#recovery-i-already-upgraded-without-exporting-first) |
| `Embedding model mismatch: configured=X, stored=Y (both 768-dim)` | You changed `SPATIAL_MEMORY_EMBEDDING_MODEL` after v2.0 had already stored memories under a different alias of the same dim | Either set the env var back to `Y`, or in v2.1+ run `spatial-memory reembed --to=X` |
| `Embedding model mismatch: configured=X (dim=768), stored=Y (dim=384)` | You changed the model to a different dimension | Use export/import (same flow as v1→v2 migration) |
| `Cannot detect format from path: ...` on import | File extension isn't `.parquet`/`.json`/`.csv` | Add the extension or pass `format=parquet` explicitly |
| `Dimension mismatch` during import | Source has v1.x vectors, target uses v2.0 model | Pass `regenerate_embeddings=true` to the import tool |
| `dry_run: true` in the import response, but you wanted a real import | Default is dry-run for safety | Re-run with `dry_run=false` explicitly |

---

## Reporting problems

If something during migration doesn't work as documented, please open an issue at <https://github.com/your-org/spatial-memory-mcp/issues> with:

- Your v1.x version (`pip show spatial-memory-mcp` before upgrade) and v2.0 version (after upgrade).
- The exact error message.
- Approximate corpus size (number of memories).
- Whether you're using the LLM/MCP path or the planned CLI path.

Tag the issue with `migration:v1-to-v2` so it routes correctly.
