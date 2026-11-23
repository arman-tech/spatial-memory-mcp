# Security policy

This policy covers Spatial Memory 3.0. The 1.x policy is archived in [docs/archive/python/SECURITY.md](docs/archive/python/SECURITY.md).

## Reporting a vulnerability

Do not open a public issue. Use [GitHub's private vulnerability reporting](https://github.com/arman-tech/spatial-memory-mcp/security/advisories/new), and include:

- what the problem is, and what an attacker gains;
- the steps to reproduce it, with the version and the operating system;
- a suggested fix, if you have one.

We will respond within 48 hours and work with you on a fix. Security fixes ship as patch releases. Watch the repository's releases to hear about them.

## Supported versions and systems

| Version | Supported |
| --- | --- |
| 3.0 | Yes, on Windows 11 x64 and on Linux with glibc 2.38 or later |
| 1.x and 2.x | No. Migrate to 3.0 ([docs/v3/MIGRATION.md](docs/v3/MIGRATION.md)). |

macOS builds from source but has never been tested, including the socket checks below.

## What Spatial Memory keeps

Everything stays on your machine. Each profile's folder under the data root (README, "Where things are kept") holds:

- **`memory.db`:** the memories, their chunks and vectors, tags, anchors, links, and where each memory came from.
- **`spool/`:** capture payloads waiting for the daemon. Each file holds a raw hook payload: the tool's input and output, or a transcript excerpt. Secrets are removed when the daemon processes a payload, not before. So until then, a spool file can contain a secret that the agent's tool saw. A payload the daemon cannot read is kept in `spool/failed/`.
- **`daemon.log` and `metrics/`:** the daemon's log and the metrics journal. The journal leaves out prompt text unless you set `journal.include_prompts = true`.
- **`backups/`:** full copies of `memory.db`, taken before a schema migration or a `reindex`.

None of these files is encrypted. If you need encryption at rest, use full-disk encryption: BitLocker on Windows, LUKS on Linux.

**Shared memories** are the exception. `share` writes them as Markdown files under `.spatial-memory/shared/` in your repository, for you to review and commit. Anyone who can read the repository can read them.

## Protections

### Secrets are redacted on the way in

The secret scanner replaces each credential it finds with `[REDACTED:<kind>]`. It covers private keys, cloud access keys, tokens, and connection strings with passwords (SPEC 9). It runs on:

- `remember`;
- every automatic capture, before anything is stored;
- `migrate` and `import`.

`share` scans again and refuses to write a file that still matches. The scanner works by pattern, so it can miss a secret in a format it does not know. Review shared files before you commit them.

### Retrieved text is data, not instructions

The prompt hook adds recalled memories to the agent's context. They sit between a header line and a closing line saying the lines are reference material from stored memories, not instructions. Shared memories are written by other people, and a memory can quote text from anywhere. Treat what is recalled as information to weigh, never as a command.

Each recalled line says where its memory came from, after its id: for example `[source: user message]`, `[source: assistant reply]` or `[source: file or command output, unverified]`. The `recall` tool gives the same tag as `source_tag`. The README lists every tag. A tag that ends in "unverified" marks text that nobody in the session wrote: text read from a file or from command output, text from the repository's shared folder, or a capture whose kind was not recorded.

### What automatic capture reads, and text planted in files

Automatic capture (README, "What it does in a session") reads the session's transcript and offers the judge:

- the messages you write;
- the summary Claude Code writes when it compacts;
- the final text of each of the assistant's turns;
- the prose the assistant writes to files with Edit, MultiEdit and Write;
- only when `capture.tool_results` is on (it is off by default): the prose of what the agent reads with Bash, Read and Grep, that is file contents, command output and search results.

It never takes code, logs or whole files, only runs of prose lines, at most 1,500 characters of each result. It does not take what WebFetch returns. A tool's own call is captured only when you name the tool in `capture.tools`.

**The last kind can be planted.** Whoever wrote a file the agent reads can put a statement in it: a file in a repository you cloned, a downloaded document, a page fetched with `curl`, a commit message. If the judge keeps that statement, it becomes an active memory of the project, and ambient recall shows it to every later session of that project. The judge decides whether text is worth keeping, not whether it is true or who wrote it. On ten synthetic statements written as project rules (deploy with the checks skipped, delete failing tests, commit API keys, turn off certificate checks), the judge kept none, in one test run (DECISIONS B-222). That is a single measurement, not a guarantee.

The optional `statements` task (`capture.judge.task`, README) keeps what a session states, so it keeps planted rules far more often once they are in the session's own words. The same ten statements were made into captures of the four kinds offered by default: your message, the assistant's reply, the compaction summary, and prose written to a file. Under `statements` it stored 32 of the 40 in each of two runs. Under the default `verdicts` task it stored 2 of 40 in each (DECISIONS B-251). Each is recalled with the tag of its kind, such as `user message` or `written to a file`, because that is where the text came from. A planted file the agent only reads is not offered at all while `capture.tool_results` is off.

What protects you:

- **The tag.** Such a memory is always recalled as `[source: file or command output, unverified]`. The tag is worked out from the provenance stored with the memory, which the text cannot change.
- **The switch, off by default.** Capture offers text read from files and command output only when this is set in `config.toml` in the data root (then run `spatial-memory stop`):

  ```toml
  [capture]
  tool_results = true
  ```

  While it is off, which is the default, capture takes only the first four kinds above, and the PostToolUse hook captures no tool but Edit, MultiEdit, Write and NotebookEdit, even one named in `capture.tools`. Switching it off again does not change memories already stored. It was on by default until DECISIONS B-236.
- **Removing a memory.** Each recalled line shows the memory's id in brackets. Ask the agent to `forget` that id; capture then treats the text as handled and does not store it again. `forget` also takes a filter (type, status, tag, anchor, created before) with `confirm: true`, and `spatial-memory project delete` removes every memory of a project.

A memory tagged `saved on request` was stored with `remember`. The agent can call `remember` on its own, so text that tells the agent to remember something can arrive with that tag too. Treat recalled memories as notes to check, whatever their tag.

### Only you can reach the daemon

- **Windows.** The daemon listens on a named pipe whose access list allows only your user. It refuses remote clients and refuses a pipe that another process created first. Clients check that the process serving the pipe runs as your user.
- **Linux.**
  - The daemon listens on a Unix socket with mode 0600, inside your profile folder or a private runtime folder.
  - The daemon makes the profile folder readable by your user only.
  - Each side checks the other's user id on the connection.
  - Clients refuse a socket, or a socket folder, that another user owns. When the profile path is too long for a socket, the daemon uses a fallback folder under the runtime or temporary folder. That folder is created user-only, and clients refuse it if it is a link or if anyone else can write to it.
- **No network listener.** The daemon opens no network port at all.

### Network access

Spatial Memory itself makes one network request: `spatial-memory setup` downloads the embedding model from Hugging Face. The download is pinned to a repository revision. Each file is checked by SHA-256 before it is used, and checked again every time the daemon loads it. A file that fails the check stops the daemon with a `model_integrity` error. To avoid the download, set `SPATIAL_MEMORY_MODEL_DIR` to a copy you fetched yourself. It must still pass the same check.

Building from source also fetches crates from crates.io, and the `ort` crate fetches a prebuilt ONNX Runtime library. `Cargo.lock` pins every crate version. Build with `--locked` so that nothing new is resolved.

## Known limits

- **No encryption at rest** (see above).
- **Projects and profiles are separate, but not walled off from you.** Scope keeps one project's memories out of another's recall, and a profile is a separate store. Both protect against mistakes, not against someone else using your account: any process running as your user can read every profile.
- **Redaction is by pattern.** A credential in an unknown format is stored as it is.
- **Captured file and command output is not checked for truth.** With `capture.tool_results` on (it is off by default), text planted in a file the agent reads can become a memory that later sessions recall, tagged unverified (see above).
- **Vectors reveal something about their text.** Embeddings are not encrypted. Recovering text from them is hard, but not proven impossible.
- **macOS is untested,** including its socket checks.

## Checklist for a shared or sensitive machine

- Keep the data root inside your home folder, or set `SPATIAL_MEMORY_HOME` to a folder only you can read.
- Turn on full-disk encryption.
- If the agent often reads repositories or downloads you do not trust, leave `tool_results` off under `[capture]`, as it is by default.
- Review files under `.spatial-memory/shared/` before you commit them, as you would review code.
- Clear `spool/failed/` once you have read it: it may hold unredacted payloads.
- Delete old copies in `backups/` that you no longer need.
