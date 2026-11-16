# Licences

The build prompt allows permissive licences only: MIT, Apache-2.0, BSD, ISC, zlib, Unicode and public domain. Anything else is flagged here.

- **Project.** MIT (`Cargo.toml`, `[workspace.package]`).
- **Last regenerated.** Phase 0, 16 November 2026, from `Cargo.lock` with `cargo metadata`. It is regenerated after every stream merge and in phase 3.
- **To regenerate.** The command runs unchanged in PowerShell and Bash:

  ```
  cargo tree --workspace --all-features --target all -e normal,build,dev --prefix none --format "{p} {l}" --locked
  ```

## Flags

| Crate | Licence | Reaches the build through | State |
| --- | --- | --- | --- |
| `option-ext` 0.2.0 | MPL-2.0 (weak copyleft) | `directories` 6.0.0 → `dirs-sys` 0.5.0 in phase 0's first tree | **Removed.** `sm-core` resolves the folders itself (DECISIONS B-41). |
| `option-ext` 0.2.0 | MPL-2.0 (weak copyleft) | `fastembed` 7.7.0 → `hf-hub` 0.5.0 → `dirs` 6.0.0, only with the `arctic-embed-m` feature (D-5) | **Flagged for the owner.** The feature is off by default; the default build does not contain it. Stream B builds the feature because SPEC 12 requires Arctic Embed M behind a Cargo feature. |
| `webpki-roots` 1.0.12 | CDLA-Permissive-2.0 | `ort-sys` 2.0.0-rc.15 build script → `ureq`, to download the ONNX Runtime binaries over TLS (`download-binaries`, `tls-rustls`); and, from stream B, `sm-embed` → `ureq` 3.5.2 (`rustls`) for the model download (B-42) | **Flagged: permissive but not on the list.** CDLA-Permissive-2.0 is a permissive licence for data (here, Mozilla's list of root certificates). Through `sm-embed` it is linked into the binary, not only used by a build script (B-42). |
| `webpki-root-certs` 1.0.12 | CDLA-Permissive-2.0 | `ureq`, through `fastembed` → `hf-hub` and through the `ort-sys` build script, only with `arctic-embed-m` (the feature turns it on in the shared `ureq`) | Flagged as above. |
| `libfuzzer-sys` 0.4.13 | (MIT OR Apache-2.0) AND NCSA | `fastembed` → `image` 0.25.10 → `ravif` → `rav1e`, only with `arctic-embed-m` | **Flagged: permissive but not on the list.** NCSA is the University of Illinois licence, a BSD-style permissive licence. |

The last four rows come from a scratch crate holding the dependencies the streams will add, pinned as in the root `Cargo.toml` (`ort`, `tokenizers`, `rusqlite`, `hdbscan`, `holomap`, `interprocess`, `tokio`, `rmcp`, `regex`, and `fastembed` behind a feature): 223 crates without the feature, 394 with it. They are confirmed or cleared when the streams merge.

The `ureq` and `clap` pins (B-42) add, in that scratch crate: `ureq` 3.5.2, `ureq-proto` 0.7.1, `rustls` 0.23.46, `rustls-pki-types` 1.16.0, `http` 1.5.0, `httparse` 1.10.1, `log` 0.4.35, `percent-encoding` 2.3.2, `base64` 0.23.1, `utf8-zero` 0.8.1, `zeroize` 1.9.1 and the `clap` crates (`clap`, `clap_builder`, `clap_derive` 4.6.7, `clap_lex` 1.1.1, `anstyle` 1.0.14, `heck` 0.5.0), all MIT OR Apache-2.0 or Apache-2.0 OR ISC OR MIT; `strsim` 0.11.1 (MIT); `ring` 0.17.14 (Apache-2.0 AND ISC); `rustls-webpki` 0.103.17 and `untrusted` 0.9.0 (ISC); `subtle` 2.6.1 (BSD-3-Clause). All permissive and on the list.

## Not Rust crates

| Item | Licence | Where it comes from |
| --- | --- | --- |
| `vendor/vec1/vec1.c` (SQLite vector extension, version 0.8) | Public domain (the SQLite blessing) | Vendored unchanged, B-7 and `vendor/vec1/README.md` |
| SQLite, bundled by `libsqlite3-sys` 0.40.0 (stream A) | Public domain | Compiled into the binary |
| ONNX Runtime (stream B) | MIT | Downloaded by `ort-sys` at build time and linked |
| EmbeddingGemma 2 ONNX export | Apache-2.0 (SPEC 2, model table) | Downloaded at run time into the per-user cache; never in the repository |
| Snowflake Arctic Embed M (feature `arctic-embed-m`) | Apache-2.0 (SPEC 2, model table) | Downloaded at run time by `fastembed` |
| `fixtures/` | MIT, this project's own | Ported from the project's Python history (B-10) |

## Rust crates in the phase 0 tree

62 crates, every platform. All are permissive. Where a crate offers a choice, this project takes MIT.

| Crate | Version | Licence |
| --- | --- | --- |
| bitflags | 2.13.2 | MIT OR Apache-2.0 |
| block-buffer | 0.12.1 | MIT OR Apache-2.0 |
| bumpalo | 3.20.4 | MIT OR Apache-2.0 |
| cfg-if | 1.0.5 | MIT OR Apache-2.0 |
| const-oid | 0.10.3 | Apache-2.0 OR MIT |
| cpufeatures | 0.3.1 | MIT OR Apache-2.0 |
| crypto-common | 0.2.2 | MIT OR Apache-2.0 |
| digest | 0.11.3 | MIT OR Apache-2.0 |
| dyn-clone | 1.0.20 | MIT OR Apache-2.0 |
| equivalent | 1.0.3 | Apache-2.0 OR MIT |
| errno | 0.3.15 | MIT OR Apache-2.0 |
| fastrand | 2.5.0 | Apache-2.0 OR MIT |
| futures-core | 0.3.34 | MIT OR Apache-2.0 |
| futures-task | 0.3.34 | MIT OR Apache-2.0 |
| futures-util | 0.3.34 | MIT OR Apache-2.0 |
| getrandom | 0.4.3 | MIT OR Apache-2.0 |
| hashbrown | 0.17.1 | MIT OR Apache-2.0 |
| hybrid-array | 0.4.15 | MIT OR Apache-2.0 |
| indexmap | 2.14.2 | Apache-2.0 OR MIT |
| itoa | 1.0.19 | MIT OR Apache-2.0 |
| js-sys | 0.3.107 | MIT OR Apache-2.0 |
| libc | 0.2.191 | MIT OR Apache-2.0 |
| linux-raw-sys | 0.12.1 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT |
| memchr | 2.8.3 | Unlicense OR MIT |
| once_cell | 1.21.4 | MIT OR Apache-2.0 |
| pin-project-lite | 0.2.17 | Apache-2.0 OR MIT |
| proc-macro2 | 1.0.107 | MIT OR Apache-2.0 |
| quote | 1.0.47 | MIT OR Apache-2.0 |
| r-efi | 6.0.0 | MIT OR Apache-2.0 OR LGPL-2.1-or-later (MIT taken; UEFI targets only) |
| ref-cast | 1.0.27 | MIT OR Apache-2.0 |
| ref-cast-impl | 1.0.27 | MIT OR Apache-2.0 |
| rustix | 1.1.6 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT |
| rustversion | 1.0.23 | MIT OR Apache-2.0 |
| schemars | 1.2.3 | MIT |
| schemars_derive | 1.2.3 | MIT |
| serde | 1.0.229 | MIT OR Apache-2.0 |
| serde_core | 1.0.229 | MIT OR Apache-2.0 |
| serde_derive | 1.0.229 | MIT OR Apache-2.0 |
| serde_derive_internals | 0.30.0 | MIT OR Apache-2.0 |
| serde_json | 1.0.154 | MIT OR Apache-2.0 |
| serde_spanned | 1.1.3 | MIT OR Apache-2.0 |
| sha2 | 0.11.1 | MIT OR Apache-2.0 |
| slab | 0.4.12 | MIT |
| syn | 3.0.10 | MIT OR Apache-2.0 |
| tempfile | 3.27.0 | MIT OR Apache-2.0 (tests only) |
| thiserror | 2.0.21 | MIT OR Apache-2.0 |
| thiserror-impl | 2.0.21 | MIT OR Apache-2.0 |
| toml | 1.2.1+spec-1.1.0 | MIT OR Apache-2.0 |
| toml_datetime | 1.1.3+spec-1.1.0 | MIT OR Apache-2.0 |
| toml_parser | 1.1.5+spec-1.1.0 | MIT OR Apache-2.0 |
| toml_writer | 1.2.0+spec-1.1.0 | MIT OR Apache-2.0 |
| typenum | 1.20.1 | MIT OR Apache-2.0 |
| unicode-ident | 1.0.27 | (MIT OR Apache-2.0) AND Unicode-3.0 |
| uuid | 1.29.2 | Apache-2.0 OR MIT |
| wasm-bindgen | 0.2.130 | MIT OR Apache-2.0 |
| wasm-bindgen-macro | 0.2.130 | MIT OR Apache-2.0 |
| wasm-bindgen-macro-support | 0.2.130 | MIT OR Apache-2.0 |
| wasm-bindgen-shared | 0.2.130 | MIT OR Apache-2.0 |
| windows-link | 0.2.1 | MIT OR Apache-2.0 |
| windows-sys | 0.61.2 | MIT OR Apache-2.0 |
| winnow | 1.0.5 | MIT |
| zmij | 1.0.23 | MIT |
