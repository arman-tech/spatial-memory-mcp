# Phase 0 spikes

Results of the phase 0 spikes the build prompt asks for, run by the lead on 16 November 2026. The pre-build spike repository `spatial-memory-spikes` (summarised in `SPIKE-WINDOWS.md` and in SPEC section 16, "Results on the owner's machine") already measured `s0` to `s10` on this machine with Rust 1.92. Its decisions are applied as they stand and its measurements are not repeated, with two exceptions: spikes a, b and c were rebuilt and re-run once on the pinned toolchain (`DECISIONS.md` B-1), and spikes d, e and f are new or re-timed.

**Machine.** Windows 11 Pro build 26300, AMD Ryzen 9 9950X3D (16 cores, 32 logical processors), 93.6 GB of memory. Rust 1.95.0. Microsoft C/C++ compiler 19.52.37029 (tools 14.52). Claude Code 2.1.325.

**Sources.** Spikes a to c are the programs of the spike repository, copied unchanged except for the toolchain file. Spikes d to f are in `docs/v3/spikes/` in this repository: the `rmcp` program (`rmcp-stdio/`), the two test plugins (`plugin-exec/`, `plugin-mcp/`), and the drivers (`tools/drive50.sh`, `tools/e50.ps1`, `tools/hookstats.py`). None of them is part of the build.

## Summary

| Spike | Question | Result | Applied |
| --- | --- | --- | --- |
| P0-0 | Does `cargo` find the Microsoft linker when run through PowerShell? | Yes | B-2 |
| P0-1 | Model files in the per-user cache, hashes verified | Yes, all three match SPEC 5 | B-3 |
| a | `ort` loads EmbeddingGemma 2 and passes the self-test on Rust 1.95 | Pass | Stream B |
| b | `vec1` compiled in through `cc`, exact search correct, source vendored at a recorded check-in | Pass, `vec1_ok` | B-7, stream A |
| c | Local socket between processes, detached auto-start, lock | Pass, `local_socket_ok` | Streams D and E |
| d | An `rmcp` 3.10.0 stdio server with one tool, called by an MCP client and by Claude Code | Pass | Stream D |
| e | Exec-form hook by bare name, 50 launches, start to exit | 5.4 ms median, 6.0 ms 95th percentile | B-6 |
| f | `claude plugin validate`; plugin hooks on Windows; prompt hook as `mcp_tool` against exec form over 50 prompts | Validate passes; plugin hooks run, SessionStart included; exec form kept | B-5, B-6 |

## P0-0 Trivial build

`cargo +1.95 new` of a binary crate, built and run through `powershell.exe -NoProfile -Command "cargo run"`, printed `Hello, world!`. In PowerShell `where.exe link.exe` lists only Git's `link.exe`, yet the build links, because `rustc` finds the Microsoft linker itself. In Git Bash the same build fails (SPIKE-WINDOWS s7b and s9). Every `cargo` command in this build runs through PowerShell.

## P0-1 Model cache

The three pinned files were copied from the spike repository's verified download into `%LOCALAPPDATA%\spatial-memory\_models\embeddinggemma-2\daa72c51243991dfcaf9f9137d2c573d8f7790c0\` and re-hashed:

| File | Bytes | SHA-256 |
| --- | --- | --- |
| `onnx/model.onnx` | 421,421 | `bc47de15f81208a5c99e5ab10f746d5e33b51ea228b7dc0bef9c133a94f1c1c3` |
| `onnx/model.onnx_data` | 1,084,170,240 | `9fd452bfc6916e5a92f080f8dada35598ca4512b21c65897d4c5974d5f9630a2` |
| `tokenizer.json` | 32,170,510 | `4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4` |

All three equal SPEC section 5. Nothing model-related is in the repository or in any worktree.

## a. Embedding self-test on Rust 1.95

`spike-embed selftest` with `SM_SPIKE_MODEL_DIR` pointing at the cache:

```
crates: ort 2.0.0-rc.15 (prebuilt ONNX Runtime, statically linked), tokenizers 0.23.2
model inputs: ["input_ids", "attention_mask", "image_features", "video_features", "audio_features"]
load 0.35 s, vector length 768, norm 1.0000, peak memory 662 MB
scores    venus 0.675  mars 0.853  jupiter 0.747  saturn 0.778
reference venus 0.675  mars 0.853  jupiter 0.747  saturn 0.778  (Linux x86-64; tolerance 0.01)
same text alone and in a batch: cosine 1.00000
token fingerprint 434a9df05b5b5fc5, reference 434a9df05b5b5fc5
RESULT selftest: PASS
```

This is AT-1's check, run outside the workspace. The `RotaryEmbedding` contrib operator runs on Windows (SPEC 5's open point).

## b. vec1 compiled in

`spike-vec1 run`:

```
c compiler: MSVC cl.exe 19.52.37029 for x64; two objects: VEC1SIMD=SCALAR, and VEC1SIMD=AVX2 with /arch:AVX2; NDEBUG, VEC1_STATIC, opt level 3
sqlite 3.54.1, rusqlite 0.43.0, libsqlite3-sys 0.40.0
vec1_info: version 0.8 (AVX2, multi-threaded)
insert 100,000 vectors in one transaction: 0.60 s
rebuild to flat, cosine: 2.22 s
exact query correctness: 20 of 20 queries return the brute-force top 10 as a set; 20 of 20 in the same order
exact query over 100,000 vectors     median   160.61 ms   p95   169.28 ms
scoped scan over 2,000 rows          median    20.94 ms   p95    27.33 ms
RESULT vec1: vec1_ok
```

The exact search is as slow on Windows as the spike repository found (157 to 172 ms); the owner accepted that on 16 November 2026 (SPEC 16, "The reviewer's reading"). The project-scoped path is an ordinary indexed table read scored in Rust, not the `vec1` row-range scan above; SPIKE-WINDOWS s7b measured it at 5.3 ms median.

**Provenance of the vendored source.** The spike repository vendored `vec1.c` without recording its check-in. Phase 0 recovered it: the file's SHA3-256 was looked up in SQLite's Fossil repository (`https://sqlite.org/src/whatis/<hash>`), which names it as artifact `ext/vec1/vec1.c` of a trunk check-in.

| Item | Value |
| --- | --- |
| Upstream path | `ext/vec1/vec1.c` in https://sqlite.org/src |
| Check-in | `60cf9357e8a60bc1de92068ed10e090cf432552f8040ba9f0294155051416347` (trunk, 2026-11-12 15:39:04 UTC, "Avoid a potential FP divide-by-zero condition in vec1 training") |
| Version | 0.8 (`vec1_info()`), file header dated 2026-02-04 |
| Size | 379,695 bytes |
| SHA-256 | `93b8e1848ea3c11faaa9b533338a7e0a2404edb0acee47507b4fbee60c7a4b62` |
| SHA3-256 | `de33d11affa682b3dfa44220a06f167679465447cec7b719cfd711ea22b6d75f` |
| Licence | Public domain (the SQLite blessing) |

The file is vendored unchanged at `vendor/vec1/vec1.c` with a `.gitattributes` rule that turns off line-ending conversion, so the hash holds on a Windows checkout.

Checked in the other direction as well: downloading the file at that check-in from upstream (`https://sqlite.org/src/raw/ext/vec1/vec1.c?ci=<check-in>`, and the equivalent `https://sqlite.org/src/raw?name=ext/vec1/vec1.c&ci=<check-in>`) returned 379,695 bytes with the same SHA-256. `vendor/vec1/README.md` records the procedure for moving to a newer check-in.

## c. Local socket

`spike-pipe run`:

```
socket: named pipe \\.\pipe\sm-spike-pipe-12264
pipe default security: ... (A;;FR;;;WD)(A;;FR;;;AN)
  Everyone (WD): allow FR; Anonymous (AN): allow FR; ...
connect and round trip, in process   median   0.093 ms   p95   0.121 ms   (1000 of 1000 replies match)
full hook path, new client process   median   3.806 ms   p95   4.246 ms   (200 of 200 replies match)
second daemon: refused
daemon outlives its launcher: true
auto-start, cold                     median  17.781 ms   p95  22.143 ms   (5 of 5 replies match)
shutdown cleans up: true
RESULT daemon-pipe: local_socket_ok
```

The default pipe security still lets Everyone and Anonymous read, so the binding rule stands: the daemon creates the pipe with a security descriptor that admits only the current user (stream E; checked by a test).

## d. rmcp stdio server

`docs/v3/spikes/rmcp-stdio`: `rmcp` 3.10.0 with features `server`, `client`, `macros`, `schemars`, `transport-io`, `transport-child-process`; one server with the tools `echo_upper` and `ambient_recall`, built with `#[tool_router]`, `#[tool]`, `Parameters<T>` and `#[tool_handler]`. Release binary 5.5 MB.

- **MCP client.** The same binary in `client` mode spawned the server with `TokioChildProcess`, listed the tools (input schema generated from the Rust type through `schemars`), and called `echo_upper` 50 times: start and initialise 6.39 ms; call median 0.057 ms, 95th percentile 0.098 ms.
- **Claude Code.** Registered through an `--mcp-config` file, Claude Code 2.1.325 called the tool and answered with its result (`PHASE ZERO`).

## e. Exec-form hook by bare name

`docs/v3/spikes/tools/e50.ps1` starts `spatial-memory-spike hook user-prompt-submit --client claude-code` by bare name (resolved on `PATH`, no shell, `UseShellExecute = false`), writes a UserPromptSubmit payload to its standard input, reads its output and waits for exit; two warm-up launches are discarded.

| Measure | n | Median | 95th percentile | Max |
| --- | --- | --- | --- | --- |
| Start to exit, launched directly by bare name | 50 | 5.38 ms | 6.00 ms | 7.08 ms |
| Same program inside Claude Code, as reported by Claude Code (`durationMs`, spike f) | 50 | 31 ms | 40 ms | 45 ms |

SPEC 16's rule ("keep exec-form command hooks if the median is at most 25 ms and the 95th percentile at most 60 ms") applies to the program's own start and exit, which passes with a wide margin. The Claude Code figure adds Claude Code's own process handling and is reported for comparison with spike f only. The spike stand-in does not contact a daemon; the full path through the named pipe is spike c's 3.8 ms.

## f. Plugin validation, plugin hooks on Windows, and mcp_tool against exec form

Two test plugins, identical except for the prompt hook (`docs/v3/spikes/plugin-exec`, `docs/v3/spikes/plugin-mcp`). Both declare the three `userConfig` options of SPEC 16 (`profile`, `ambient_recall` with `options`, `binary_path`) and an `.mcp.json` entry for the stand-in server.

**Validation.** `claude plugin validate <dir>` passes for both (`✔ Validation passed`, exit 0). It prints two plugin-directory warnings, a missing README of at least 40 words and a missing icon, which "do not change the result" and only fail under `--profile directory`. The real plugin carries both (stream D).

**Plugin hooks on Windows.** With `--plugin-dir`, the plugin's exec-form hooks ran by bare name on Windows: SessionStart for `startup` (176 ms, the first launch in a fresh session) and for `resume` (`claude -p --resume <session>`; the hook log recorded `session-start SessionStart resume`), and UserPromptSubmit on every prompt. This closes SPEC 16's "Hooks packaged in a plugin on Windows, including SessionStart" and the version 1 Windows bug at Claude Code 2.1.37.

**Output reaches the model only in exec form.**

- Exec form: the session transcript records a `hook_success` attachment whose `content` is the hook's standard output, and the model quoted the `[spatial-memory] ambient recall:` block back.
- `mcp_tool` form (`"type": "mcp_tool"`, `"server": "plugin:sm-spike-mcp:spatial-memory"`, `"tool": "ambient_recall"`, `"input": {"prompt": "${prompt}"}`): the tool ran and its text is in the attachment's `stdout`, but the attachment's `content` is `UserPromptSubmit completed` and the model did not see the block. On Claude Code 2.1.325, a plain-text tool result from an `mcp_tool` prompt hook is not injected, contrary to the hooks reference SPEC 16 quotes. A JSON `hookSpecificOutput` result was not tried.
- `mcp_tool` form with the MCP server not connected (observed under `--strict-mcp-config`, which excludes plugin servers): the hook is skipped as "not connected". Exec form does not depend on the server.

**50 prompts each.** `docs/v3/spikes/tools/drive50.sh` runs one `claude -p` session per variant with `--input-format stream-json`, sends 50 prompts one at a time (each waits for the previous turn's result), model Haiku. Durations are Claude Code's own `durationMs` per UserPromptSubmit `hook_success` attachment in the session transcript (`tools/hookstats.py`).

| Variant | n | Median | 95th percentile | Max | Output injected |
| --- | --- | --- | --- | --- | --- |
| Exec form, bare name | 50 | 31 ms | 40 ms | 45 ms | Yes |
| `mcp_tool` on the plugin's server | 50 | 14 ms | 19 ms | 22 ms | No |

The `mcp_tool` hook saves about 17 ms at the median and 21 ms at the 95th percentile, 6 to 7% of the 300 ms hook budget, but its output is not delivered and it does nothing when the server is not connected. Following the build prompt's rule (keep exec form unless the difference is large and nothing is lost), the prompt hook stays in exec form (B-6). The server's own connection at session start took 143 ms and is not on the prompt path.

## Not verified in phase 0

- Exec-form hooks by bare name on Linux inside Claude Code (Claude Code is not signed in inside the containers). The Linux build and tests run in Docker; hook integration on Linux is listed as unverified in RESULTS.md unless it is run there.
- Native Linux timings and macOS.
