# Continuous integration for 3.0: a proposal

**This is a proposal only.** No workflow has been added, changed, enabled or triggered (build rule). 3.0 is not announced until the owner has reviewed the runner cost in section 4.

## 1. What runs today, and why it must be replaced before v3 is merged

Three Python workflows from 1.x are still in `.github/workflows/`. They are unchanged on `v3`:

| Workflow | Trigger | Jobs | What it does |
| --- | --- | --- | --- |
| `tests.yml` | Push and pull request on `master`, `main`, `develop` | 12: `ubuntu-latest`, `windows-latest` and `macos-latest`, each with Python 3.10, 3.11, 3.12 and 3.13 | `pip install -e ".[dev]"`, then `pytest -m "not integration"`, then `pytest -m integration` with `continue-on-error` |
| `lint.yml` | The same | 1, on `ubuntu-latest` | `ruff format --check`, `ruff check`, `mypy spatial_memory/` |
| `publish.yml` | A release is published | 1, on `ubuntu-latest`, environment `release` | `python -m build`, `twine check`, then upload to PyPI (trusted publishing, plus a `PYPI_API_TOKEN` secret) |

`v3` has no `pyproject.toml` and no `spatial_memory/` package. So:

- **A pull request from `v3` into `master`, or the merge itself,** runs all 13 push and pull-request jobs. Every one of them fails at `pip install` or at `ruff`, because there is nothing to install. The 12 test jobs include 4 on macOS.
- **A release published after the merge** runs `publish.yml`. It fails at `python -m build`.
- **A release published from `master` before the merge** would publish the 1.x Python package under a 3.0 release.

**Required before v3 is merged or released:** remove the three workflows and add the replacements in section 3, all in the same change. The owner decides what happens to the PyPI package `spatial-memory-mcp`. 3.0 is not a Python package, so nothing new goes to PyPI. A last 1.x release whose description points to 3.0 is one option.

## 2. Runners and prices

Source: docs.github.com, "GitHub-hosted runners" and "Actions runner pricing", read on 17 November 2026.

**Standard runners**

| Label | Image | Public repository | Private repository | Price per minute (private) |
| --- | --- | --- | --- | --- |
| `ubuntu-latest` | Ubuntu 24.04 | 4 processors, 16 GB | 2 processors, 8 GB | $0.006 |
| `windows-latest` | Windows Server 2025 with Visual Studio 2026 | 4 processors, 16 GB | 2 processors, 8 GB | $0.010 |
| `macos-latest` | macOS 26, arm64 | 3 (M1), 7 GB | 3 (M1), 7 GB | $0.062 |

**Larger runners** with 4 processors cost $0.012 a minute on Linux and $0.022 on Windows. Self-hosted runners cost nothing per minute.

**What this repository pays**

- **Today, nothing.** `arman-tech/spatial-memory-mcp` is public, and standard runners are free for public repositories. They also get 4 processors instead of 2 on Linux and Windows.
- **If the repository becomes private,** the free monthly allowance is $12 on the Free plan, $18 on Pro and Team, and $300 on Enterprise. Everything above that is billed per minute.

The minutes still matter while the repository is public: they set how long a push waits for its result.

**Ubuntu 24.04 is a safe image.** It has glibc 2.39. The ONNX Runtime binaries that `ort` downloads need glibc 2.38 or later (README, "Supported systems").

## 3. Proposed workflows

All third-party actions are pinned to a full commit hash, not a tag. Every job runs with `--locked`, so a job fails rather than resolving new versions.

### `ci.yml`, replacing `tests.yml` and `lint.yml`

**Trigger.** Push and pull request on `master`, `main` and `develop`, as today. Use `paths-ignore` for `docs/**` and `*.md`, so a documentation-only change runs nothing.

| Job | Runner | Steps |
| --- | --- | --- |
| `lint` | `ubuntu-latest` | `cargo fmt --check`, then `cargo clippy --workspace --all-targets --all-features --locked -- -D warnings` |
| `test-linux` | `ubuntu-latest` | `cargo test --workspace --all-features --locked` |
| `test-windows` | `windows-latest` | The same. The step runs in PowerShell, the shell this project uses for `cargo` on Windows (B-2). |

- **Toolchain.** The first step runs `rustup toolchain install`. It installs what `rust-toolchain.toml` pins: Rust 1.95.0 with `rustfmt` and `clippy`. Every runner image already has `rustup`, so no toolchain action is needed. The step is explicit because newer `rustup` releases may stop installing the pinned toolchain automatically.
- **Caches.**
  - The Cargo registry and the `target` folder, keyed on `Cargo.lock` and the toolchain file.
  - The ONNX Runtime archive, which `ort` downloads during the build. Set `ORT_CACHE_DIR` to a folder in the workspace and cache that folder. Its defaults are `%LOCALAPPDATA%\ort.pyke.io` on Windows and `~/.cache/ort.pyke.io` on Linux.
- **The real-model test** skips itself, because CI has no model folder.
- **macOS is not in the push matrix.** 3.0 supports Windows 11 and Linux. The macOS code paths, such as `getpeereid`, compile only behind `cfg` and have never been tested.

### `weekly.yml`, a scheduled job (new)

It runs once a week and can also be started by hand (`workflow_dispatch`). It never blocks a push.

| Job | Runner | Steps |
| --- | --- | --- |
| `real-model` | `ubuntu-latest` | Restore the model (1.1 GB) from a cache keyed on its revision. On a cache miss, run `spatial-memory setup`, which downloads and checks it. Then `cargo test --release -p sm-embed --test real_model --locked`. |
| `macos` | `macos-latest` | `cargo test --workspace --all-features --locked`, marked `continue-on-error`, so that macOS breakage is seen without promising support |

### `release.yml`, replacing `publish.yml`

**Trigger.** A release is published.

1. Build `cargo build --release --locked -p sm-cli` on `ubuntu-latest` and on `windows-latest`.
2. Attach `spatial-memory` and `spatial-memory.exe` to the release, each with a SHA-256 checksum file, and the plugin package.

It does not publish to crates.io (build rule) or to PyPI. It keeps the `release` environment, so that a person approves each run.

## 4. Runner minutes per push

**Measured** by package K on the owner's machine while nothing else ran, on `97a2742`, the tree that merged as `7b89635` (RESULTS.md section 6; B-129):

- **Linux:** in Docker with `--cpus 4` and `--cpus 2`, standing in for the public and private runners;
- **Windows:** natively, with `cargo -j 4` and `-j 2` and `RUST_TEST_THREADS` set to match;
- **cold:** new `target` and registry folders, which is what a cache miss costs;
- **warm:** the folders kept and one comment line changed in `sm-cli`, which is what a typical push costs with a cache hit.

Every step exited 0, and all 36 test binaries reported `ok`.

**Measured job times, in seconds**

| Job | Cold, 4 processors | Cold, 2 processors | Warm, 4 processors | Warm, 2 processors |
| --- | --- | --- | --- | --- |
| `lint` (Linux) | 75.6 | 134.0 | 15.6 | 7.5 |
| `test-linux` | 410.3 | 539.5 | 293.8 | 386.6 |
| `test-windows` | 502.7 | 788.3 | 333.7 | 347.0 |

- **A warm push is mostly test running, not building.** Building took 13 to 22 s, and running took 276 to 362 s.
- **Four binaries take about four fifths of the running time:** `hook_budgets`, `acceptance`, `sm-store` and `sm-share`. Most of it is their own timed waits, such as AT-5's 60 seconds, not processor time.

**The planning figures.** The owner's processor (AMD Ryzen 9 9950X3D) is faster per core than a hosted runner's. The measurement also leaves out the hosted toolchain install and the cache restore. So each measured figure is a lower bound, and the plan doubles it until the first real runs give actual minutes. GitHub bills each job rounded up to a whole minute.

**Per push, billed minutes per operating system** (doubled, then each job rounded up)

| | Linux (`lint` + `test-linux`) | Windows (`test-windows`) | Wait for the result |
| --- | --- | --- | --- |
| Public repository, 4 processors, warm | 1 + 10 = 11 min | 12 min | about 11 min |
| Public repository, 4 processors, cold | 3 + 14 = 17 min | 17 min | about 17 min |
| Private repository, 2 processors, warm | 1 + 13 = 14 min | 12 min | about 13 min |
| Private repository, 2 processors, cold | 5 + 18 = 23 min | 27 min | about 26 min |

`lint` and `test-linux` run in parallel, and `test-windows` beside them, so the wait is the longest job before rounding.

**Cost.**

- **Public, as today:** nothing. Standard runners are free for public repositories.
- **Private, at the section 2 prices:**

  | Push | Linux at $0.006 a minute | Windows at $0.010 a minute | Total |
  | --- | --- | --- | --- |
  | Warm | 14 min, $0.084 | 12 min, $0.120 | $0.204 |
  | Cold | 23 min, $0.138 | 27 min, $0.270 | $0.408 |

  **At 100 pushes a month** this is $20.40 if every push hits the cache and $40.80 if none does. That is above the Free plan's $12 allowance and the Pro and Team plans' $18. Without the doubling, a warm push measures $0.108 (1 + 7 Linux minutes and 6 Windows minutes), which is $10.80 at 100 pushes.
- **Documentation-only pushes** run nothing, because of `paths-ignore`.

**Not measured.**

- `weekly.yml`: the real-model job and the macOS job. The real-model test runs in about 12 s once built, so a model-cache hit makes the job's own cost a release build of `sm-embed`'s test binary.
- `release.yml`.
- A real hosted runner. The first real runs replace these figures.
