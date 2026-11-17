# Spatial Memory 3.0

Spatial Memory gives a coding agent a memory that lasts between sessions. It captures decisions, fixes, pitfalls and conventions as you work, stores them per project on your machine, and recalls them by meaning. It works with Claude Code and with Cursor.

- **It is one program,** `spatial-memory`, which you build from this repository.
- **Nothing leaves your machine.** It talks to no network service. The only download is the embedding model, once, during setup.
- **Memories are private** until you share one. Sharing writes a file into the repository, and a teammate reviews it like code.

Version 3.0 is a rewrite in Rust of the Python server (versions 1 and 2), and a clean break from it: it starts with an empty store and carries no memories over. Migration is optional. `spatial-memory migrate` can load a version 1 or 2 export ([docs/v3/MIGRATION.md](docs/v3/MIGRATION.md)). It has been tested on the export fixtures only, never on a real store.

## Supported systems

| System | Status |
| --- | --- |
| Windows 11, x64 | Supported |
| Linux, x64, glibc 2.38 or later: Ubuntu 24.04 or later, Debian 13, Red Hat Enterprise Linux 10, current Fedora, current Arch Linux | Supported |
| Ubuntu 22.04, Debian 12, Red Hat Enterprise Linux 9 | Not supported. Their glibc is older than the ONNX Runtime binaries need, so the program does not link. |
| macOS | Not tested. The code paths exist, but 3.0 makes no promise for macOS. |

The model takes about 1.1 GB of disk in the per-user cache. While the background process (the daemon) runs, it holds about 0.7 GB of memory. Searching a large store also maps the database file into memory, so the daemon's working set grows with the store: after searches over 100,000 chunks it measured up to about 1.6 GB, of which about 0.7 GB was the daemon's own memory and the rest the mapped file (`docs/v3/RESULTS.md`, section 2). The 1 GB budget for an idle daemon counts its own (private) memory only, because the operating system can take the mapped file's pages back whenever it needs them.

## Install

3.0 has no prebuilt binaries yet. You build it with the Rust toolchain.

1. **Install Rust** with [rustup](https://rustup.rs). The repository pins the compiler version in `rust-toolchain.toml`, and `rustup` installs that version the first time you build.
2. **Install a C compiler.** SQLite, the vector extension and the tokenizer's regular-expression library are compiled from C.
   - **Windows:** Visual Studio Build Tools with the "Desktop development with C++" workload.
   - **Linux:** `gcc` or `clang`, for example the `build-essential` package on Ubuntu and Debian.
3. **Build and install the binary** from the repository folder:

   ```
   cargo install --path crates/sm-cli --locked
   ```

   This puts `spatial-memory` in Cargo's `bin` folder, which `rustup` added to your `PATH`.

   **On Windows, run `cargo` from PowerShell or Command Prompt, not from Git Bash.** Git's own `link.exe` comes first on Git Bash's `PATH` and hides the Microsoft linker.

4. **Connect it to your agent** (next section). Then check the installation with `spatial-memory doctor`. Until the model is downloaded, `doctor` reports `model_missing`.

## Connect it to your agent

**`spatial-memory setup` is the supported way to connect version 3.0.** The Claude Code plugin route is not verified for 3.0: the plugin marketplace still serves the version 1 plugin. Don't use both: they register the same server and hooks. `setup` refuses to run while the plugin is enabled, and `doctor` reports a failure if both are registered.

- **`spatial-memory setup`** (supported):

  ```
  spatial-memory setup                    # Claude Code
  spatial-memory setup --client cursor    # Cursor
  spatial-memory setup --remove           # undo, then stop the daemon
  ```

  - **What it does.** It downloads the model if it is missing, checks the SHA-256 of every model file, and registers the server and the hooks with the absolute path of the binary. Because the path is absolute, it works even when the application that starts Claude Code sees a different `PATH` from your terminal, as desktop launchers often do.
  - **Claude Code.** The server is registered with `claude mcp add` at user scope. The hooks are merged into `~/.claude/settings.json`, after a backup is taken.
  - **Cursor.** It writes `~/.cursor/mcp.json` and `~/.cursor/hooks.json`. Cursor gets the capture hooks only: it has no prompt hook.
  - **Running it twice** changes nothing.

- **The Claude Code plugin** (not verified for 3.0). The 3.0 plugin is in [plugin/](plugin/), and [plugin/README.md](plugin/README.md) lists what it registers and its options. It runs `spatial-memory` by name from `PATH`. Until the marketplace serves it, `/plugin install spatial-memory --marketplace arman-tech/spatial-memory-mcp` installs version 1 instead.

  The plugin does not download the model. Download and verify it with:

  ```
  spatial-memory setup --model-only
  ```

  This registers nothing, so it runs while the plugin is enabled.

**Sessions that are already open.** Claude Code reads `~/.claude/settings.json` again when the file changes. The hooks therefore go live in every Claude Code session that is open the moment `setup` writes the file, without a restart. In such a session, expect this:

- Its next prompt gets recalled memories, and its next replies are captured.
- Nothing written before `setup` is captured, with one exception. Capture starts from the reply that ends the session's next turn. If the session compacts before that turn ends, capture reads the part of the transcript after the last compaction, which the conversation still holds. A resumed session is read the same way. Text compacted away before `setup` is never captured ([docs/v3/CAPTURE-RCA.md](docs/v3/CAPTURE-RCA.md), RC1).
- The MCP server is registered separately, with `claude mcp add`. A session that was already open may not show the 13 tools until you start a new one.

**Check the installation** once it is connected:

```
spatial-memory doctor
```

If no daemon is running, `doctor` starts one. It then waits for the daemon to load the model, for up to 5 seconds after the daemon started. If the model still isn't loaded by then, for example on a first start after a reboot, the `state` check fails and says how long the daemon has been loading. Run `doctor` again.

## What it does in a session

- **Capture.** Hooks hand the final text of each of the assistant's turns to the daemon. Text written before a tool call is not handed over. Tool calls themselves are not captured unless you name their tools in `capture.tools`. The daemon:
  - picks out what may be worth keeping, using version 1's signal patterns;
  - redacts secrets;
  - drops duplicates: the same text, a restatement of a memory the project already holds, and text it has handled before, so a memory you forgot does not come back;
  - stores the rest as `decision`, `pitfall`, `convention` or `note`, **pending**.

  A pending capture is never served: ambient recall, `recall` and the other tools skip it. You can keep one by hand with `revise` and `status: active`, or delete it with `forget`.

  A capture hook exits within 200 ms and never blocks the agent. When the daemon is not running, the hook writes the capture to a spool folder, and the daemon reads it later.

  **Known defect.** In its first real use, 3.0's capture stored 222 memories in 25 minutes, and 188 of them (84.7%) were not worth keeping: mostly the agent's progress narration, much of it from history compacted before `setup` ran. [docs/v3/CAPTURE-RCA.md](docs/v3/CAPTURE-RCA.md) gives the causes. Every capture now waits as pending, and the causes in how capture reads and deduplicates are fixed. The signal patterns still cannot tell a finding from narration, so nothing captured is served until a judge that can is in place.
- **Ambient recall.** When you send a prompt, the hook adds the few memories of the current project that are most relevant to it, marked as reference material.
- **The daemon** is started automatically by the first client that needs it, and it opens no network listener. It loads the model once and stays running. To stop it, run `spatial-memory stop`. To have it stop after a period with no requests, set `daemon.idle_exit_minutes`.

## Tools

The agent sees 13 tools:

| Tool | What it does |
| --- | --- |
| `remember` | Store one or many memories with a type, anchors and provenance. The reply lists any redactions. |
| `recall` | Search by meaning or by keyword. Filters: `scope` (`project` or `all`), `type`, `anchor`, `status`, `mode`, `k`. |
| `related` | Neighbours of a memory or an anchor, by similarity and by links |
| `revise` | Edit a memory, change its status, or supersede it with a new one |
| `forget` | Delete by id or by filter. Deleting more than one memory needs `confirm: true`. |
| `link` | Add, remove or list links between anchors and memories |
| `regions` | Clusters of a project's memories, with the central memories of each |
| `journey` | A path of memories between two memories |
| `consolidate` | Find near-duplicates and propose merges. It changes nothing without `apply: true`. |
| `share` | Write chosen memories into the repository's `.spatial-memory/shared/` folder, for review |
| `sync` | Load reviewed shared memories from the repository |
| `status` | Daemon health, the model, counts per project, the index state |
| `reinforce` | Mark a recalled memory as having helped. The mark is recorded and does not change ranking. |

Every reply fits a token budget of 2,000 by default. A reply that was cut says what it left out. Every error has a stable code, a message, and what to do next.

## Command line

| Command | What it does |
| --- | --- |
| `spatial-memory doctor` | Check the binary, the model, the daemon, the index, the hooks and the spool, with a next action for each failure. Then print the embedder's numbers from the daemon: bulk chunks per second and the median query time, with the thread counts and batch size they ran at |
| `spatial-memory doctor --eval <set> [<baseline>]` | Run a retrieval evaluation set against the store, and compare it with a baseline |
| `spatial-memory status` | The daemon, the model and the counts per project |
| `spatial-memory export <file> [--project <name>]` | Write the store, or one project, to a new file |
| `spatial-memory import <file>` | Load a file written by `export` |
| `spatial-memory migrate --from <file> [--default-project <name>]` | Optional: load an export from version 1 or 2. Tested on the export fixtures only, never on a real store ([MIGRATION.md](docs/v3/MIGRATION.md)) |
| `spatial-memory reindex` | Rebuild the index after the model or the chunk settings changed. It takes a backup first. |
| `spatial-memory project rename <from> <to>` | Rename a project |
| `spatial-memory project delete <name> [--confirm]` | Delete a project and its memories. Without `--confirm`, it shows what would be deleted. |
| `spatial-memory setup` | Register with the agent, as above. With `--remove`, unregister and then stop the daemon. |
| `spatial-memory setup --model-only` | Download the model if a file is missing or wrong, and verify every file. Registers nothing, so it runs while the plugin is enabled. |
| `spatial-memory stop` | Stop the daemon cleanly: it gives running requests up to 5 seconds to finish, writes queued captures to the spool, and exits. The command waits until the daemon has gone, up to 15 seconds, and never starts one. |
| `spatial-memory daemon` | Run the daemon in the foreground. Clients start it on their own, so you rarely need this. |

Every command takes `--profile <name>`. The `mcp` and `hook` subcommands are for the agent to run.

## Where things are kept

| What | Windows | Linux |
| --- | --- | --- |
| Data root | `%LOCALAPPDATA%\spatial-memory` | `$XDG_DATA_HOME/spatial-memory`, or `~/.local/share/spatial-memory` |
| Model | `%LOCALAPPDATA%\spatial-memory\_models` | `$XDG_CACHE_HOME/spatial-memory/_models`, or `~/.cache/spatial-memory/_models` |

- `SPATIAL_MEMORY_HOME` replaces the data root.
- `SPATIAL_MEMORY_MODEL_DIR` replaces the model folder, for example to use a copy you downloaded yourself.
- Both must be absolute paths. A relative path is refused with `config_invalid`, because the hooks, the MCP server and the daemon each start in a different folder.
- `SPATIAL_MEMORY_MODEL_URL` makes `setup` download the model's files from a mirror instead of Hugging Face. Each file is fetched from `<address>/<file path>`. The pinned sizes and SHA-256 values still apply, so a mirror cannot supply different files.

Each profile has its own folder under the data root:

```
config.toml          settings for this profile
memory.db            the store (SQLite)
daemon.log           the daemon's log
metrics/             the metrics journal
spool/               captures waiting for the daemon
backups/             copies taken before migrate and reindex
```

## Configuration

Settings are read from two files: `config.toml` in the data root applies to every profile, and the profile's own `config.toml` overrides it. Every setting is optional. The values below are the defaults.

```toml
[recall]
k = 5                     # results per recall
similarity_floor = 0.70   # below this, results are marked weak

[ambient]                 # recall added to each prompt
enabled = true
top_k = 5
scope = "project"         # or "all"

[capture]
enabled = true
tools = []                # none by default; version 1 captured ["Edit", "Write", "Bash", "NotebookEdit"]
# ignore = [...]          # path patterns never captured; the default list covers .env and key files

[cpu]
budget = 0.75             # share of physical cores that background embedding may use

[daemon]
idle_exit_minutes = 0     # 0: the daemon never exits on its own

[output]
token_budget = 2000       # largest reply, in tokens (at least 200)

[journal]
include_prompts = false   # write prompt text into the metrics journal
```

A value out of range, or a misspelt key, is refused with `config_invalid`, and the message names the key.

- **Ambient recall from the environment.** Three variables override the `[ambient]` settings, and the plugin's `ambient_recall` option overrides all three:
  - `SPATIAL_MEMORY_AMBIENT_RECALL`: `on` turns ambient recall on, and any other value turns it off.
  - `SPATIAL_MEMORY_AMBIENT_RECALL_TOP_K`: 0 turns it off.
  - `SPATIAL_MEMORY_AMBIENT_RECALL_SCOPE`.
- **Changing the chunk or model settings** (`[chunk]`, `[embed]`) makes the store's index stale. Search is then refused with `index_mismatch` until you run `spatial-memory reindex`.

### Projects

A project is a git repository, identified by its remote URL. A folder with no repository is identified by its path. On Windows, paths that differ only in case or slash direction are the same project. Every read is limited to the current project unless you pass `scope: all`.

A repository can name its project, or turn capture off, in `.spatial-memory/project.toml`:

```toml
name = "order-service"

[capture]
enabled = false
```

### Profiles

A profile is a separate store, such as `work` and `personal`. The profile is chosen by the first of these that is set:

1. `--profile`;
2. the plugin's `profile` option;
3. `SPATIAL_MEMORY_PROFILE`;
4. a folder rule in the root `config.toml`;
5. the profile `default`.

```toml
[[profile_rule]]
path = "C:/work"
profile = "work"
```

## Sharing with a team

1. **`share`** writes each chosen memory as one Markdown file under `.spatial-memory/shared/<type>/`. Each file has a short front-matter block and the memory as its body. The format is in SPEC section 9.
2. **Review.** You commit the files and open a pull request, as with code. Spatial Memory never commits or pushes.
3. **`sync`.** After the merge, teammates run `sync`. It loads the shared memories, and it retracts or revises local copies whose files were deleted or changed.
4. **Shared memories are read-only locally:** a fix goes through the repository.
5. **Secrets.** `share` refuses to write a memory that still contains a secret after redaction.
6. **Vectors are never committed.** Each machine embeds shared memories with its own copy of the model.

## Troubleshooting

Start with `spatial-memory doctor`. It runs every check and gives a next action for each failure. The daemon writes its own log to `daemon.log` in the profile folder.

| Code or symptom | What to do |
| --- | --- |
| `daemon_unavailable` | No daemon answered and none could start. Run `spatial-memory doctor`, then read `daemon.log`. |
| `daemon_loading` | The daemon is still loading the model, which takes a few seconds after a start. Try again. |
| `model_missing` | Run `spatial-memory setup --model-only` to download the model, or set `SPATIAL_MEMORY_MODEL_DIR` to a folder that holds it. |
| The model download fails behind a company proxy | `setup` takes its proxy from the first of `ALL_PROXY`, `HTTPS_PROXY` and `HTTP_PROXY` that is set, in upper or lower case: an `http://` or `https://` address, optionally with `user:password@`. It honours `NO_PROXY`. It does not read the Windows proxy settings and does not support SOCKS. It trusts only the root certificates built into the binary, so a proxy that inspects TLS with a company certificate makes the download fail. Then download the files by hand (next row), or point `SPATIAL_MEMORY_MODEL_URL` at a mirror inside the company. Version 3.1 plans to use the operating system's certificate store and proxy settings. |
| Downloading the model by hand | Download three files with a browser or any tool that follows redirects. Each address is `https://huggingface.co/onnx-community/embeddinggemma-2-ONNX/resolve/daa72c51243991dfcaf9f9137d2c573d8f7790c0/` followed by `onnx/model.onnx`, `onnx/model.onnx_data` or `tokenizer.json`. Put them in `embeddinggemma-2/daa72c51243991dfcaf9f9137d2c573d8f7790c0/` inside the model folder (see "Where things are kept"), keeping the `onnx` subfolder for the first two. Instead, you can put them in any folder and name it with `SPATIAL_MEMORY_MODEL_DIR`. Then run `spatial-memory setup --model-only`. It checks each file's size and SHA-256 and downloads only a file that is missing or wrong. |
| `model_integrity` | A model file's size or SHA-256 is wrong. `doctor` names the file. Remove the model folder and run `spatial-memory setup --model-only`. |
| `index_mismatch` | The model or chunk settings changed. Run `spatial-memory reindex`. |
| `version_mismatch` | A daemon from another build was running. The client asks it to exit and starts a new one, so a retry normally works. If it keeps happening, run `doctor`. |
| `project_unresolved` | Run from inside the project's folder, or name the project in `.spatial-memory/project.toml`. |
| `config_invalid` | Fix the key that the message names, in the root or the profile `config.toml`. |
| `shared_read_only` | The memory came from the repository. Change its file there, then run `sync`. |
| `secret_detected` | `share` found a secret that redaction could not remove. Edit the memory first. |
| The hooks do nothing in a desktop-launched agent | The application's `PATH` does not have `spatial-memory`. Use `spatial-memory setup` instead of the plugin. |
| `cargo` fails to link on Windows | You ran it from Git Bash. Run it from PowerShell. |
| The store's database is larger than 2 GiB | The daemon maps only the first 2 GiB of `memory.db` in the profile folder into memory. It reads the rest through SQLite's page cache, which is slower: at 100,000 chunks, reading without the mapping made `recall` up to about 4 times slower (`docs/v3/RESULTS.md`, section 4). The database measured 750 MiB at 100,000 short chunks. 2 GiB is about 273,000 short chunks, or by estimate about 200,000 full chunks of 256 tokens. The speed budgets were measured at 100,000 chunks only. An approximate index for large stores is planned for version 3.1. |
| The daemon stopped with no error in `daemon.log` | If the operating system cannot read part of the mapped database file, for example because the disk is failing or a network or removable drive went away, the daemon ends at once: SQLite cannot catch that error. The next hook or tool call starts a new daemon. A prompt hook gets no recalled memories for that one prompt, a capture hook keeps its event in the spool, and an MCP call waits for the new daemon. Keep the data root on a local disk, and run `spatial-memory doctor` if it happens again. |
| The prompt hook takes about 100 ms on Windows | Windows power throttling (EcoQoS) slows the background daemon within seconds of its start, because it has no window. Measured: the query embedding takes about 85 ms instead of 47.6 ms, and the prompt hook's median is 100.1 ms instead of 61.4 ms, still inside its 300 ms budget. Version 3.0 does not opt out of throttling. |

## Not verified

These were not run or measured for 3.0, or only in part, as each item says. You may meet any of them:

- **The fallback model, Arctic Embed M** (the `arctic-embed-m` Cargo feature). It builds and its unit tests pass, but it has never been loaded or run.
- **macOS.** Nothing was built or run on it.
- **Timings on native Linux.** Every Linux figure was measured in Docker on a Windows machine. The plugin's hooks were run by bare program name inside Claude Code on Windows only.
- **The plugin route.** The marketplace still serves the version 1 plugin, so the 3.0 plugin in `plugin/` has never been installed through `/plugin install`.
- **Inside Claude Code and Cursor.** The tests run the binary as the clients do, with their hook input and MCP messages, but not under the clients. `setup`'s tests replace `claude mcp add` and `remove` with a stand-in.
- **The model download** ran once, on Windows: 3 files, 1.1 GB, in 29 seconds, with every checksum passing. It was not run on Linux, behind a proxy, or on a network that inspects TLS. It reads `ALL_PROXY`, `HTTPS_PROXY`, `HTTP_PROXY` and `NO_PROXY` but not the Windows proxy settings, and it trusts the root certificates built into the binary, not the operating system's. The proxy variables were read from the code, not tried through a proxy. Downloading the three files by hand and running `setup` was tested once, on Windows: `setup` verified them and downloaded nothing.
- **A real two-core machine.** Two processors were simulated in Docker only.
- **A first start after a reboot on Windows.** The daemon's start time, about 1.5 s, was measured with the model already in the file cache.
- **The idle exit with the real model.** It was tested with a stand-in embedder only.
- **A daemon running as administrator on Windows.**
- **Migrating a real version 1 or 2 store.** `migrate` was tested on the export fixtures only.
- **Retrieval quality on your own memories.** It was measured on the fixture evaluation set only.

## Uninstall

1. Run `spatial-memory setup --remove`. It unregisters the server and the hooks, then stops the daemon and waits until it has gone. If you used the plugin, uninstall it with `/plugin` instead, then run `spatial-memory stop`.
2. Close Claude Code and Cursor, then run `spatial-memory stop` again. A session that is still open keeps its own MCP server process, which starts the daemon again on its next tool call. If you used other profiles, run `spatial-memory stop --profile <name>` for each.
3. Run `cargo uninstall sm-cli`. On Windows it fails while a `spatial-memory.exe` is running, because Windows does not let the file of a running program be deleted.
4. Delete the data root and the model folder (see "Where things are kept") if you no longer want your memories.

## Development

Contributors: start with [CLAUDE.md](CLAUDE.md) and `docs/v3/`. [SPEC.md](docs/v3/SPEC.md) is what to build, and [CONTRACTS.md](docs/v3/CONTRACTS.md) is how the crates connect.

The 1.x documentation is in [docs/archive/python/](docs/archive/python/).

## Licence

MIT. Every dependency is under a permissive licence. [docs/v3/LICENSES.md](docs/v3/LICENSES.md) lists them. The EmbeddingGemma 2 ONNX export is under Apache-2.0 and is downloaded at run time, never stored in the repository.
