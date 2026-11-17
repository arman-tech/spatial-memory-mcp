# Migrating from Spatial Memory 1.x or 2.x

Version 3 does not open the old LanceDB database. To migrate, you export from the Python install, import into version 3, and let version 3 re-embed everything with its own model (SPEC 10). The old vectors are not used, because the model has changed.

Expect a short export, a migrate that takes about a second for 10,000 records, and then embedding in the background. Embedding takes about an hour per 10,000 single-chunk memories on two cores. Keyword search works the whole time.

## Before you start

- Install version 3 and run `spatial-memory setup`, so that the model is downloaded and checked (README, "Install").
- Leave the Python install in place until step 3 passes.
- Know where your agent ran the Python server. Version 1 resolves export paths against the server's working directory. That is usually the project folder the agent was opened in.

## Step 1: export from the old install

Ask the agent that still has the Python server to call `export_memories` with these arguments:

```json
{
  "output_path": "exports/spatial-memory-v1.json",
  "format": "json",
  "include_vectors": false,
  "project": "*"
}
```

- **`project: "*"` exports every project.** Without it, version 1 exports only the project it detects for the current folder. Memories of other projects would be left behind without any warning.
- **`output_path` must be inside `exports/` or `backups/`.** Those folders are relative to the Python server's working directory (the defaults of `SPATIAL_MEMORY_EXPORT_ALLOWED_PATHS`), and the folder is created if it is missing. Any other path is refused. The tool's reply gives the absolute `output_path` it wrote and `memories_exported`. Note both down.
- **JSON only.** `migrate` reads the JSON array that `export_memories` writes. Version 2's export format is identical to version 1's (B-91), so the same call works there.
- **Over 1,000,000 memories:** version 1 stops at `max_export_records` (1,000,000 by default). Export one project at a time with `"project": "<name>"`, and migrate each file.

## Step 2: run `migrate` from inside your main repository

Open a terminal in the repository you use most, then run:

```
spatial-memory migrate --from <the output_path from step 1>
```

The command is the same in PowerShell and Bash. Put the path in quotes if it contains spaces. If the daemon is not already running, `migrate` starts it.

### Where each record goes

Each record's `project` value is matched to a version 3 project in this order (B-79, B-91):

1. A project whose git remote equals the value. Version 1 stored normalised remotes such as `github.com/owner/repo`.
2. The one project with that name. If two projects share the name, the migration fails with `invalid_input` before anything is written.
3. The working directory's project, when its name equals the value. This is why you run `migrate` from inside the main repository: memories that version 1 stored under that repository's name land in that repository's project.
4. Otherwise, a new project. If the value looks like a normalised remote, the project gets that remote, and opening any clone of the repository finds it. Otherwise it is a project known only by its name. `recall` and `ambient` find such a project only with `scope: all`, or when you run inside a repository whose project has the same name.

A record with no `project` value goes to `--default-project <name>` when you give it. Otherwise it goes to the working directory's project. If neither applies, `migrate` fails with `project_unresolved` before it writes anything (B-35).

### What is kept

| Version 1 or 2 field | In version 3 |
| --- | --- |
| `id`, `content`, `created_at`, `updated_at`, `content_hash` | Unchanged |
| `namespace` | A tag, unless it equals the project name |
| `tags`, `metadata`, `source` | Tags and provenance. `metadata` is kept as given. |
| `importance`, `access_count`, `last_accessed`, `expires_at` | Kept for reference only, and not used for ranking |
| `vector` | Dropped |
| (none) | `type` is `note`, and `status` is `active` |

- **Secrets are redacted on the way in.** The same scanner that protects capture replaces each secret with `[REDACTED:<kind>]`. Version 1's own `[REDACTED_<KIND>]` placeholders stay as they are.
- **The report's "with secrets redacted"** counts the inserted records that had something removed.

### Reading the report

```
Imported <file>
  records read:        <n>
  inserted:            <n>
  already present:     <n>
  with secrets redacted: <n>
  pending embeddings:  <n>
  projects created:    <names>
  issues (<n>):
    record <index> (<id>): <reason>
```

- **The counts always balance:** `records read` equals `inserted`, plus `already present`, plus the number of issues (AT-8). `records read` should equal step 1's `memories_exported`.
- **An issue** is a record that cannot be stored:
  - it is not a JSON object;
  - its id is missing or appears earlier in the file;
  - its content is empty;
  - its content hash matches another memory of the same project, and the issue names that memory.

  Nothing else is dropped.
- **Running `migrate` again is safe.** Records already stored count as `already present` and are not changed.

## Step 3: verify, then uninstall the Python package

```
spatial-memory status
```

- **Each project's memory count** should add up to `inserted` plus what you had before.
- **`pending_embeddings`** falls to zero as the daemon embeds in the background. Until then, `recall` works in keyword mode on what is still waiting.
- **If the daemon stops part-way,** the chunks it has not embedded stay pending in the store. The next daemon picks them up. There is nothing to re-run.
- **`spatial-memory doctor`** checks the model, the index and the hooks, and gives a next action for anything that fails.

When the counts match, uninstall the Python package with the tool you installed it with (for example, `pip uninstall spatial-memory-mcp` or `uv tool uninstall spatial-memory-mcp`). Remove its server entry from your agent's configuration, then delete its `.spatial-memory` data folder when you no longer need it.

## Known limits

- **Not yet tried on a real store.** The migration has been tested on the version 1 fixture and on a generated export of 10,000 records (81 MB), which took 946 ms on Windows and 828 ms on Linux. It has not yet been run against a real version 1 store (SPEC 12's exit condition; B-91).
- **Embedding time** is set by the CPU: about 3 chunks per second on two cores (SPEC 10). Faster machines finish sooner.
