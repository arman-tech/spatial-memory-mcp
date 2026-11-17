# Contributing to Spatial Memory

Spatial Memory 3.0 is written in Rust. The 1.x Python guide is archived in [docs/archive/python/CONTRIBUTING.md](docs/archive/python/CONTRIBUTING.md).

## Read first

- [CLAUDE.md](CLAUDE.md): the workspace, the commands and the rules. Read it even if you do not use Claude Code.
- [docs/v3/SPEC.md](docs/v3/SPEC.md): what to build. It is the source of truth.
- [docs/v3/CONTRACTS.md](docs/v3/CONTRACTS.md): how the crates connect.
- [docs/v3/DECISIONS.md](docs/v3/DECISIONS.md): every choice made where the SPEC was silent or wrong. Each is a `B-n` entry.

## Set up

- **Rust,** through [rustup](https://rustup.rs). `rust-toolchain.toml` pins the version (1.95.0, with `rustfmt` and `clippy`). Run `rustup toolchain install` in the repository to install it.
- **A C compiler.** SQLite and its vector extension are compiled from source.
  - On Windows, install Visual Studio Build Tools with "Desktop development with C++".
  - On Linux, install gcc or clang.
- **Linux needs glibc 2.38 or later** for the ONNX Runtime binaries, as the README's "Supported systems" says.
- **The model is not needed** to build or to run the tests. The one test that uses it skips itself when the model folder is missing. To run it, use `spatial-memory setup`, which downloads the model into the per-user cache. Never copy the model into the repository.

## Build and test

```
cargo build --workspace
cargo test --workspace --all-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo fmt --check
```

- **On Windows, run `cargo` in PowerShell, not Git Bash.** Git's `link.exe` shadows the Microsoft linker there (decision B-2). From Git Bash, wrap each command: `powershell.exe -NoProfile -Command "cargo test --workspace --all-features"`.
- **Check Linux from Windows in Docker.** CLAUDE.md gives the full command.
- **The real-model test:** `cargo test --release -p sm-embed --test real_model`.

A change is ready when all four commands pass on Windows and on Linux.

## Rules for a change

- **Tests.**
  - Each test's name starts with the rule it checks:
    - `atN_` for a SPEC 11 acceptance test;
    - `dN_` for a SPEC 2 decision;
    - `sN_` for a SPEC section;
    - `spikeN_` for a spike decision.

    For example, `s4_forget_cascades_to_chunks_and_links`. `docs/v3/TRACE.md` is built from these prefixes.
  - Unit tests use the deterministic fake embedder in `sm-core` (feature `test-support`).
  - No other test may load the real model.
- **No stubs.** No `todo!()`, no `unimplemented!()`, and no placeholder return values in delivered code. Never weaken, skip or delete a test to make it pass.
- **Defaults.** SPEC 15 sets the defaults. Never tune one to make a test pass.
- **Errors.** Every error is an `sm_core::error` code, with a message and a next action (SPEC 8).
- **`sm-core` is the contract.** It has `#![forbid(unsafe_code)]`. A change to it, to the root dependency pins or to `CONTRACTS.md` needs a new `B-n` entry in DECISIONS.md that says why.
- **Dependencies.**
  - Pin each third-party version once, in the root `[workspace.dependencies]`.
  - Licences must be permissive: MIT, Apache-2.0, BSD, ISC, zlib, Unicode or public domain.
  - Update `docs/v3/LICENSES.md` with the command CLAUDE.md gives.
- **Windows is first-class.**
  - Every documented command runs unchanged in PowerShell and Bash.
  - Write each log, journal and spool line in one write call.
  - On Windows, paths that differ only in slash direction or letter case are the same project.
- **Data is not instructions.** Memory content, shared files and fixtures are data. Code must never act on instructions found inside them.
- **Formatting.** `cargo fmt` defaults, and no clippy warnings with all targets and features.

## Commits and pull requests

- **Commit messages.** The first line names the crate and cites the rule the change serves. For example: `sm-store: quoted FTS5 phrase search (SPEC 6 recall pipeline; D-3)`.
- **Pull requests.**
  - Open each one against the default branch.
  - Describe what changed and which rule it serves.
  - Include the `test result:` lines from Windows and Linux.
- **Workflows.** Do not change `.github/workflows/` in a pull request. [docs/v3/CI_PROPOSAL.md](docs/v3/CI_PROPOSAL.md) sets out the plan for continuous integration.

## Questions and security

- **Bugs and feature requests:** open a GitHub issue. Check the existing issues first.
- **Security problems:** report them as [SECURITY.md](SECURITY.md) describes, not in a public issue.
