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
- **Vectors reveal something about their text.** Embeddings are not encrypted. Recovering text from them is hard, but not proven impossible.
- **macOS is untested,** including its socket checks.

## Checklist for a shared or sensitive machine

- Keep the data root inside your home folder, or set `SPATIAL_MEMORY_HOME` to a folder only you can read.
- Turn on full-disk encryption.
- Review files under `.spatial-memory/shared/` before you commit them, as you would review code.
- Clear `spool/failed/` once you have read it: it may hold unredacted payloads.
- Delete old copies in `backups/` that you no longer need.
