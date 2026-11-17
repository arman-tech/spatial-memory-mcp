# Licences

The build prompt allows permissive licences only: MIT, Apache-2.0, BSD, ISC, zlib, Unicode and public domain. Anything else is flagged here.

- **Project.** MIT (`Cargo.toml`, `[workspace.package]`).
- **Last regenerated.** Phase 1, 16 November 2026, after the stream D merge, with the command below. It is regenerated after every stream merge and in phase 3.
- **To regenerate.** The command runs unchanged in PowerShell and Bash:

  ```
  cargo tree --workspace --all-features --target all -e normal,build,dev --prefix none --format "{p} {l}" --locked
  ```

## Flags

| Crate | Licence | Reaches the build through | State |
| --- | --- | --- | --- |
| `option-ext` 0.2.0 | MPL-2.0 (weak copyleft) | `directories` 6.0.0 → `dirs-sys` 0.5.0 in phase 0's first tree | **Removed.** `sm-core` resolves the folders itself (DECISIONS B-41). |
| `option-ext` 0.2.0 | MPL-2.0 (weak copyleft) | `sm-embed` → `fastembed` 7.7.0 → `hf-hub` 0.5.0 → `dirs` 6.0.0 → `dirs-sys` 0.5.0, only with the `arctic-embed-m` feature (D-5) | **Flagged for the owner; confirmed by the stream B merge.** The feature is off by default; `cargo tree -i option-ext` without it finds nothing. It is built because SPEC 12 requires Arctic Embed M behind a Cargo feature. |
| `webpki-roots` 1.0.12 | CDLA-Permissive-2.0 | `ort-sys` 2.0.0-rc.15 build script → `ureq`, to download the ONNX Runtime binaries over TLS (`download-binaries`, `tls-rustls`); and, from stream B, `sm-embed` → `ureq` 3.5.2 (`rustls`) for the model download (B-42) | **Flagged: permissive but not on the list; confirmed by the stream B merge.** CDLA-Permissive-2.0 is a permissive licence for data (here, Mozilla's list of root certificates). Through `sm-embed` it is linked into the default binary, not only used by a build script (B-42). |
| `webpki-root-certs` 1.0.12 | CDLA-Permissive-2.0 | `ureq`, through `fastembed` → `hf-hub` and through the `ort-sys` build script, only with `arctic-embed-m` (the feature turned it on in the shared `ureq`) | **Removed** by the stream B merge: `fastembed` is pinned without its default features (B-50). |
| `libfuzzer-sys` 0.4.13 | (MIT OR Apache-2.0) AND NCSA | `fastembed` → `image` 0.25.10 → `ravif` → `rav1e`, only with `arctic-embed-m` | **Removed** by the stream B merge: without its default features `fastembed` brings no `image` stack (B-50). |

The last four rows came from a scratch crate holding the dependencies the streams would add, pinned as in the root `Cargo.toml`. The stream B merge confirmed `option-ext` and `webpki-roots` in the workspace tree and cleared `webpki-root-certs` and `libfuzzer-sys` (B-50, which also removed the system OpenSSL crates `openssl`, `openssl-sys` and `native-tls`).

The `ureq` and `clap` pins (B-42) add, in that scratch crate: `ureq` 3.5.2, `ureq-proto` 0.7.1, `rustls` 0.23.46, `rustls-pki-types` 1.16.0, `http` 1.5.0, `httparse` 1.10.1, `log` 0.4.35, `percent-encoding` 2.3.2, `base64` 0.23.1, `utf8-zero` 0.8.1, `zeroize` 1.9.1 and the `clap` crates (`clap`, `clap_builder`, `clap_derive` 4.6.7, `clap_lex` 1.1.1, `anstyle` 1.0.14, `heck` 0.5.0), all MIT OR Apache-2.0 or Apache-2.0 OR ISC OR MIT; `strsim` 0.11.1 (MIT); `ring` 0.17.14 (Apache-2.0 AND ISC); `rustls-webpki` 0.103.17 and `untrusted` 0.9.0 (ISC); `subtle` 2.6.1 (BSD-3-Clause). All permissive and on the list.

## Not Rust crates

| Item | Licence | Where it comes from |
| --- | --- | --- |
| `vendor/vec1/vec1.c` (SQLite vector extension, version 0.8) | Public domain (the SQLite blessing) | Vendored unchanged, B-7 and `vendor/vec1/README.md` |
| SQLite, bundled by `libsqlite3-sys` 0.40.0 (stream A) | Public domain | Compiled into the binary |
| ONNX Runtime (stream B) | MIT | Downloaded by `ort-sys` at build time and linked |
| Oniguruma regular-expression library (stream B) | BSD-2-Clause | Compiled from C by `onig_sys` 69.9.3 for `tokenizers` (`onig` feature) and linked |
| `ring` 0.17.14's C and assembly (stream B) | Apache-2.0 AND ISC, as the crate | Compiled by its build script for `rustls` |
| EmbeddingGemma 2 ONNX export | Apache-2.0 (SPEC 2, model table) | Downloaded at run time into the per-user cache; never in the repository |
| Snowflake Arctic Embed M (feature `arctic-embed-m`) | Apache-2.0 (SPEC 2, model table) | Downloaded at run time by `fastembed` |
| `fixtures/` | MIT, this project's own | Ported from the project's Python history (B-10) |

## Rust crates in the workspace tree

262 third-party crates, every platform, with all features of the merged crates (`sm-core`, `sm-spatial`, `sm-embed`, `sm-capture`, `sm-store`, `sm-cli`, `sm-share`). The figure of 261 given at the stream D merge was one short: the tree without `sm-share` also lists 262. All are permissive and on the list except the two flagged above. Where a crate offers a choice, this project takes MIT; `ryu` offers Apache-2.0 OR BSL-1.0 and Apache-2.0 is taken.

Stream C (`sm-spatial`, B-45) added `hdbscan` 0.12.0, `holomap` 0.3.0 and their numeric dependencies. Stream B (`sm-embed`) added `ort` and `ort-sys` 2.0.0-rc.15, `tokenizers` 0.23.2 with `onig`, `ndarray` 0.17.2, `ureq` 3.5.2 with `rustls`, `num_cpus`, and, behind `arctic-embed-m`, `fastembed` 7.7.0 with `hf-hub` 0.5.0. Stream G (`sm-capture`) added no third-party crate: `regex` 1.13.1, `regex-automata` 0.4.18, `regex-syntax` 0.8.11 and `aho-corasick` 1.1.5 were already in the tree through `tokenizers`, and `sm-capture` takes `regex` through the root pin with its default features. Stream A (`sm-store`) added `rusqlite` 0.43.0 and `libsqlite3-sys` 0.40.0 (`bundled`, MIT; SQLite itself is listed above), `hashlink` 0.12.2 with `hashbrown` 0.17.1 (previously an unreached lock entry), `fallible-iterator`, `fallible-streaming-iterator`, and `vcpkg` 0.2.15 as a build dependency of `libsqlite3-sys`; `sqlite-wasm-rs` 0.6.3 and `rsqlite-vfs` 0.2.0 (MIT) are reached only on the `wasm32-unknown-unknown` target and are never built here. `cc`, `pkg-config` and the other build crates were already in the tree. Stream D (`sm-cli`) added 50 crates: the `clap` 4.6.7 crates (B-42); `interprocess` 2.4.8 (0BSD OR Apache-2.0) with `recvmsg` 1.0.0 and `doctest-file` 1.1.1, both 0BSD (Zero-Clause BSD, a BSD licence with no conditions, equivalent to public domain); `rmcp` and `rmcp-macros` 3.10.0 (Apache-2.0) with `tokio` 1.54.1, `tokio-util`, `tokio-stream`, the `futures` crates, `tracing`, `chrono` 0.4.45, `process-wrap` 10.0.1 and `darling` 0.24.1; `nix` 0.31.3 on Unix; and the `windows` 0.62.2 crates on Windows. Eight crates appear in more than one version: `syn` 2 and 3, `thiserror` 1 and 2, `getrandom` 0.2, 0.3 and 0.4, `base64` 0.13 and 0.23, `windows-sys` 0.52 and 0.61, `r-efi` 5 and 6, `hashbrown` 0.16 and 0.17, and `darling` 0.20 and 0.24; all MIT, or MIT OR Apache-2.0. Stream F (`sm-share`, MIT) added no third-party crate: `serde`, `serde_json`, `windows-sys` 0.61.2 and, for tests, `tempfile` were already in the tree.

`Cargo.lock` also holds 71 entries that `cargo tree` does not reach on any target with all features: optional features of the crates above that are never compiled. They are `reqwest`, `hyper` and its `tower` stack, `url` and `idna` with the `icu_*`, `yoke`, `zerofrom`, `zerovec`, `zerotrie`, `tinystr`, `litemap`, `writeable` and `potential_utf` crates (Unicode-3.0), `quinn`, `quinn-proto`, `quinn-udp`, `cookie`, `cookie_store`, `time` and its helpers, `glam` (four versions), `num-bigint`, `zlib-rs` (Zlib), `tinyvec` and the `wasm` helpers; every one is MIT, Apache-2.0, Zlib or Unicode-3.0, alone or as a choice. `cargo metadata --locked --all-features` lists them with their licences.

The AT-7 fixture generator (`fixtures/spatial/generator/`) has its own lock file and is not in the workspace or the binary; besides MIT and Apache-2.0 it uses `webpki-roots` 1.0.12 (CDLA-Permissive-2.0, flagged above), `subtle` 2.6.1 (BSD-3-Clause) and `ring` 0.17.14 (Apache-2.0 AND ISC).

| Crate | Version | Licence |
| --- | --- | --- |
| adler2 | 2.0.1 | 0BSD OR MIT OR Apache-2.0 |
| ahash | 0.8.12 | MIT OR Apache-2.0 |
| aho-corasick | 1.1.5 | Unlicense OR MIT |
| allocator-api2 | 0.2.21 | MIT OR Apache-2.0 |
| android_system_properties | 0.1.6 | MIT OR Apache-2.0 |
| anstyle | 1.0.14 | MIT OR Apache-2.0 |
| approx | 0.5.1 | Apache-2.0 |
| autocfg | 1.5.1 | Apache-2.0 OR MIT |
| base64 | 0.13.1 | MIT/Apache-2.0 |
| base64 | 0.23.1 | MIT OR Apache-2.0 |
| bitflags | 2.13.2 | MIT OR Apache-2.0 |
| block-buffer | 0.12.1 | MIT OR Apache-2.0 |
| bumpalo | 3.20.4 | MIT OR Apache-2.0 |
| bytemuck | 1.25.2 | Zlib OR Apache-2.0 OR MIT |
| bytes | 1.12.1 | MIT |
| castaway | 0.2.4 | MIT |
| cc | 1.11.2 | MIT OR Apache-2.0 |
| cfg-if | 1.0.5 | MIT OR Apache-2.0 |
| cfg_aliases | 0.2.2 | MIT |
| chrono | 0.4.45 | MIT OR Apache-2.0 |
| clap | 4.6.7 | MIT OR Apache-2.0 |
| clap_builder | 4.6.7 | MIT OR Apache-2.0 |
| clap_derive | 4.6.7 | MIT OR Apache-2.0 |
| clap_lex | 1.1.1 | MIT OR Apache-2.0 |
| compact_str | 0.9.1 | MIT |
| console | 0.16.6 | MIT |
| const-oid | 0.10.3 | Apache-2.0 OR MIT |
| core-foundation-sys | 0.8.7 | MIT OR Apache-2.0 |
| cpufeatures | 0.3.1 | MIT OR Apache-2.0 |
| crc32fast | 1.5.5 | MIT OR Apache-2.0 |
| crossbeam-deque | 0.8.8 | MIT OR Apache-2.0 |
| crossbeam-epoch | 0.9.21 | MIT OR Apache-2.0 |
| crossbeam-utils | 0.8.23 | MIT OR Apache-2.0 |
| crypto-common | 0.2.2 | MIT OR Apache-2.0 |
| daachorse | 3.0.3 | MIT OR Apache-2.0 |
| darling | 0.20.11 | MIT |
| darling | 0.24.1 | MIT |
| darling_core | 0.20.11 | MIT |
| darling_core | 0.24.1 | MIT |
| darling_macro | 0.20.11 | MIT |
| darling_macro | 0.24.1 | MIT |
| dary_heap | 0.3.9 | MIT OR Apache-2.0 |
| derive_builder | 0.20.2 | MIT OR Apache-2.0 |
| derive_builder_core | 0.20.2 | MIT OR Apache-2.0 |
| derive_builder_macro | 0.20.2 | MIT OR Apache-2.0 |
| digest | 0.11.3 | MIT OR Apache-2.0 |
| dirs | 6.0.0 | MIT OR Apache-2.0 |
| dirs-sys | 0.5.0 | MIT OR Apache-2.0 |
| doctest-file | 1.1.1 | 0BSD |
| dyn-clone | 1.0.20 | MIT OR Apache-2.0 |
| either | 1.19.0 | MIT OR Apache-2.0 |
| encode_unicode | 1.0.0 | Apache-2.0 OR MIT |
| equivalent | 1.0.3 | Apache-2.0 OR MIT |
| errno | 0.3.15 | MIT OR Apache-2.0 |
| esaxx-rs | 0.1.10 | Apache-2.0 |
| fallible-iterator | 0.3.0 | MIT/Apache-2.0 |
| fallible-streaming-iterator | 0.1.9 | MIT/Apache-2.0 |
| fastembed | 7.7.0 | Apache-2.0 |
| fastrand | 2.5.0 | Apache-2.0 OR MIT |
| find-msvc-tools | 0.1.15 | MIT OR Apache-2.0 |
| flate2 | 1.1.11 | MIT OR Apache-2.0 |
| fnv | 1.0.7 | Apache-2.0 / MIT |
| foldhash | 0.2.0 | Zlib |
| futures | 0.3.34 | MIT OR Apache-2.0 |
| futures-channel | 0.3.34 | MIT OR Apache-2.0 |
| futures-core | 0.3.34 | MIT OR Apache-2.0 |
| futures-executor | 0.3.34 | MIT OR Apache-2.0 |
| futures-io | 0.3.34 | MIT OR Apache-2.0 |
| futures-macro | 0.3.34 | MIT OR Apache-2.0 |
| futures-sink | 0.3.34 | MIT OR Apache-2.0 |
| futures-task | 0.3.34 | MIT OR Apache-2.0 |
| futures-util | 0.3.34 | MIT OR Apache-2.0 |
| getrandom | 0.2.17 | MIT OR Apache-2.0 |
| getrandom | 0.3.4 | MIT OR Apache-2.0 |
| getrandom | 0.4.3 | MIT OR Apache-2.0 |
| hashbrown | 0.16.1 | MIT OR Apache-2.0 |
| hashbrown | 0.17.1 | MIT OR Apache-2.0 |
| hashlink | 0.12.2 | MIT OR Apache-2.0 |
| hdbscan | 0.12.0 | MIT OR Apache-2.0 |
| heck | 0.5.0 | MIT OR Apache-2.0 |
| hermit-abi | 0.5.3 | MIT OR Apache-2.0 |
| hf-hub | 0.5.0 | Apache-2.0 |
| hmac-sha256 | 1.1.15 | ISC |
| holomap | 0.3.0 | MIT OR Apache-2.0 |
| http | 1.5.0 | MIT OR Apache-2.0 |
| httparse | 1.10.1 | MIT OR Apache-2.0 |
| hybrid-array | 0.4.15 | MIT OR Apache-2.0 |
| iana-time-zone | 0.1.65 | MIT OR Apache-2.0 |
| iana-time-zone-haiku | 0.1.2 | MIT OR Apache-2.0 |
| ident_case | 1.0.1 | MIT/Apache-2.0 |
| indexmap | 2.14.2 | Apache-2.0 OR MIT |
| indicatif | 0.18.6 | MIT |
| interprocess | 2.4.8 | 0BSD OR Apache-2.0 |
| itertools | 0.14.0 | MIT OR Apache-2.0 |
| itoa | 1.0.19 | MIT OR Apache-2.0 |
| js-sys | 0.3.107 | MIT OR Apache-2.0 |
| kdtree | 0.7.0 | MIT OR Apache-2.0 |
| libc | 0.2.191 | MIT OR Apache-2.0 |
| libm | 0.2.16 | MIT |
| libredox | 0.1.25 | MIT |
| libsqlite3-sys | 0.40.0 | MIT |
| linux-raw-sys | 0.12.1 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT |
| log | 0.4.35 | MIT OR Apache-2.0 |
| lzma-rust2 | 0.15.8 | Apache-2.0 |
| macro_rules_attribute | 0.2.3 | Apache-2.0 OR MIT OR Zlib |
| macro_rules_attribute-proc_macro | 0.2.3 | Apache-2.0 OR MIT OR Zlib |
| matrixmultiply | 0.3.11 | MIT/Apache-2.0 |
| memchr | 2.8.3 | Unlicense OR MIT |
| minimal-lexical | 0.2.1 | MIT/Apache-2.0 |
| miniz_oxide | 0.9.1 | MIT OR Zlib OR Apache-2.0 |
| mio | 1.2.5 | MIT |
| monostate | 0.1.18 | MIT OR Apache-2.0 |
| monostate-impl | 0.1.18 | MIT OR Apache-2.0 |
| nalgebra | 0.35.0 | Apache-2.0 |
| ndarray | 0.17.2 | MIT OR Apache-2.0 |
| nix | 0.31.3 | MIT |
| nom | 7.1.3 | MIT |
| num-complex | 0.4.6 | MIT OR Apache-2.0 |
| num-integer | 0.1.47 | MIT OR Apache-2.0 |
| num-rational | 0.4.2 | MIT OR Apache-2.0 |
| num-traits | 0.2.19 | MIT OR Apache-2.0 |
| num_cpus | 1.17.0 | MIT OR Apache-2.0 |
| once_cell | 1.21.4 | MIT OR Apache-2.0 |
| onig | 6.5.3 | MIT |
| onig_sys | 69.9.3 | MIT |
| option-ext | 0.2.0 | MPL-2.0 |
| ort | 2.0.0-rc.15 | MIT OR Apache-2.0 |
| ort-sys | 2.0.0-rc.15 | MIT OR Apache-2.0 |
| paste | 1.0.15 | MIT OR Apache-2.0 |
| pastey | 0.2.3 | MIT OR Apache-2.0 |
| percent-encoding | 2.3.2 | MIT OR Apache-2.0 |
| pin-project-lite | 0.2.17 | Apache-2.0 OR MIT |
| pkg-config | 0.3.34 | MIT OR Apache-2.0 |
| portable-atomic | 1.15.0 | Apache-2.0 OR MIT |
| portable-atomic-util | 0.2.8 | Apache-2.0 OR MIT |
| ppv-lite86 | 0.2.21 | MIT OR Apache-2.0 |
| proc-macro2 | 1.0.107 | MIT OR Apache-2.0 |
| process-wrap | 10.0.1 | Apache-2.0 OR MIT |
| quote | 1.0.47 | MIT OR Apache-2.0 |
| r-efi | 5.3.0 | MIT OR Apache-2.0 OR LGPL-2.1-or-later (MIT taken; UEFI targets only) |
| r-efi | 6.0.0 | MIT OR Apache-2.0 OR LGPL-2.1-or-later (MIT taken; UEFI targets only) |
| rand | 0.9.5 | MIT OR Apache-2.0 |
| rand_chacha | 0.9.0 | MIT OR Apache-2.0 |
| rand_core | 0.9.5 | MIT OR Apache-2.0 |
| rand_pcg | 0.9.0 | MIT OR Apache-2.0 |
| rawpointer | 0.2.1 | MIT/Apache-2.0 |
| rayon | 1.12.0 | MIT OR Apache-2.0 |
| rayon-cond | 0.4.0 | Apache-2.0/MIT |
| rayon-core | 1.13.0 | MIT OR Apache-2.0 |
| recvmsg | 1.0.0 | 0BSD |
| redox_users | 0.5.3 | MIT |
| ref-cast | 1.0.27 | MIT OR Apache-2.0 |
| ref-cast-impl | 1.0.27 | MIT OR Apache-2.0 |
| regex | 1.13.1 | MIT OR Apache-2.0 |
| regex-automata | 0.4.18 | MIT OR Apache-2.0 |
| regex-syntax | 0.8.11 | MIT OR Apache-2.0 |
| ring | 0.17.14 | Apache-2.0 AND ISC (both apply; both on the list) |
| rmcp | 3.10.0 | Apache-2.0 |
| rmcp-macros | 3.10.0 | Apache-2.0 |
| rsqlite-vfs | 0.2.0 | MIT (wasm32 only) |
| rusqlite | 0.43.0 | MIT |
| rustix | 1.1.6 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT |
| rustls | 0.23.46 | Apache-2.0 OR ISC OR MIT |
| rustls-pki-types | 1.16.0 | MIT OR Apache-2.0 |
| rustls-webpki | 0.103.17 | ISC |
| rustversion | 1.0.23 | MIT OR Apache-2.0 |
| ryu | 1.0.23 | Apache-2.0 OR BSL-1.0 (Apache-2.0 taken) |
| safe_arch | 1.2.0 | Zlib OR Apache-2.0 OR MIT |
| safetensors | 0.8.0 | Apache-2.0 |
| schemars | 1.2.3 | MIT |
| schemars_derive | 1.2.3 | MIT |
| serde | 1.0.229 | MIT OR Apache-2.0 |
| serde_core | 1.0.229 | MIT OR Apache-2.0 |
| serde_derive | 1.0.229 | MIT OR Apache-2.0 |
| serde_derive_internals | 0.30.0 | MIT OR Apache-2.0 |
| serde_json | 1.0.154 | MIT OR Apache-2.0 |
| serde_spanned | 1.1.3 | MIT OR Apache-2.0 |
| sha2 | 0.11.1 | MIT OR Apache-2.0 |
| shlex | 2.0.1 | MIT OR Apache-2.0 |
| signal-hook-registry | 1.4.8 | MIT OR Apache-2.0 |
| simba | 0.10.2 | Apache-2.0 |
| simd-adler32 | 0.3.10 | MIT |
| slab | 0.4.12 | MIT |
| smallvec | 1.16.5 | MIT OR Apache-2.0 |
| socket2 | 0.6.5 | MIT OR Apache-2.0 |
| spm_precompiled | 0.1.4 | Apache-2.0 |
| sqlite-wasm-rs | 0.6.3 | MIT (wasm32 only) |
| static_assertions | 1.1.0 | MIT OR Apache-2.0 |
| strsim | 0.11.1 | MIT |
| subtle | 2.6.1 | BSD-3-Clause |
| syn | 2.0.119 | MIT OR Apache-2.0 |
| syn | 3.0.10 | MIT OR Apache-2.0 |
| tempfile | 3.27.0 | MIT OR Apache-2.0 (tests only) |
| thiserror | 1.0.69 | MIT OR Apache-2.0 |
| thiserror | 2.0.21 | MIT OR Apache-2.0 |
| thiserror-impl | 1.0.69 | MIT OR Apache-2.0 |
| thiserror-impl | 2.0.21 | MIT OR Apache-2.0 |
| tokenizers | 0.23.2 | Apache-2.0 |
| tokio | 1.54.1 | MIT |
| tokio-macros | 2.7.2 | MIT |
| tokio-stream | 0.1.19 | MIT |
| tokio-util | 0.7.20 | MIT |
| toml | 1.2.1+spec-1.1.0 | MIT OR Apache-2.0 |
| toml_datetime | 1.1.3+spec-1.1.0 | MIT OR Apache-2.0 |
| toml_parser | 1.1.5+spec-1.1.0 | MIT OR Apache-2.0 |
| toml_writer | 1.2.0+spec-1.1.0 | MIT OR Apache-2.0 |
| tracing | 0.1.44 | MIT |
| tracing-attributes | 0.1.31 | MIT |
| tracing-core | 0.1.36 | MIT |
| typenum | 1.20.1 | MIT OR Apache-2.0 |
| unicode-ident | 1.0.27 | (MIT OR Apache-2.0) AND Unicode-3.0 |
| unicode-normalization-alignments | 0.1.12 | MIT/Apache-2.0 |
| unicode-segmentation | 1.13.3 | MIT OR Apache-2.0 |
| unicode-width | 0.2.3 | MIT OR Apache-2.0 |
| unicode_categories | 0.1.1 | MIT OR Apache-2.0 |
| unit-prefix | 0.5.2 | MIT |
| untrusted | 0.9.0 | ISC |
| ureq | 3.5.2 | MIT OR Apache-2.0 |
| ureq-proto | 0.7.1 | MIT OR Apache-2.0 |
| utf8-zero | 0.8.1 | MIT OR Apache-2.0 |
| uuid | 1.29.2 | Apache-2.0 OR MIT |
| vcpkg | 0.2.15 | MIT/Apache-2.0 |
| version_check | 0.9.5 | MIT/Apache-2.0 |
| wasi | 0.11.1+wasi-snapshot-preview1 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT |
| wasip2 | 1.0.4+wasi-0.2.12 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT |
| wasm-bindgen | 0.2.130 | MIT OR Apache-2.0 |
| wasm-bindgen-macro | 0.2.130 | MIT OR Apache-2.0 |
| wasm-bindgen-macro-support | 0.2.130 | MIT OR Apache-2.0 |
| wasm-bindgen-shared | 0.2.130 | MIT OR Apache-2.0 |
| web-time | 1.1.0 | MIT OR Apache-2.0 |
| webpki-roots | 1.0.12 | CDLA-Permissive-2.0 |
| wide | 1.7.1 | Zlib OR Apache-2.0 OR MIT |
| widestring | 1.2.1 | MIT OR Apache-2.0 |
| windows | 0.62.2 | MIT OR Apache-2.0 |
| windows-collections | 0.3.2 | MIT OR Apache-2.0 |
| windows-core | 0.62.2 | MIT OR Apache-2.0 |
| windows-future | 0.3.2 | MIT OR Apache-2.0 |
| windows-implement | 0.60.2 | MIT OR Apache-2.0 |
| windows-interface | 0.59.3 | MIT OR Apache-2.0 |
| windows-link | 0.2.1 | MIT OR Apache-2.0 |
| windows-numerics | 0.3.1 | MIT OR Apache-2.0 |
| windows-result | 0.4.1 | MIT OR Apache-2.0 |
| windows-strings | 0.5.1 | MIT OR Apache-2.0 |
| windows-sys | 0.52.0 | MIT OR Apache-2.0 |
| windows-sys | 0.61.2 | MIT OR Apache-2.0 |
| windows-targets | 0.52.6 | MIT OR Apache-2.0 |
| windows-threading | 0.2.1 | MIT OR Apache-2.0 |
| windows_aarch64_gnullvm | 0.52.6 | MIT OR Apache-2.0 |
| windows_aarch64_msvc | 0.52.6 | MIT OR Apache-2.0 |
| windows_i686_gnu | 0.52.6 | MIT OR Apache-2.0 |
| windows_i686_gnullvm | 0.52.6 | MIT OR Apache-2.0 |
| windows_i686_msvc | 0.52.6 | MIT OR Apache-2.0 |
| windows_x86_64_gnu | 0.52.6 | MIT OR Apache-2.0 |
| windows_x86_64_gnullvm | 0.52.6 | MIT OR Apache-2.0 |
| windows_x86_64_msvc | 0.52.6 | MIT OR Apache-2.0 |
| winnow | 1.0.5 | MIT |
| wit-bindgen | 0.57.1 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT |
| zerocopy | 0.8.63 | BSD-2-Clause OR Apache-2.0 OR MIT |
| zerocopy-derive | 0.8.63 | BSD-2-Clause OR Apache-2.0 OR MIT |
| zeroize | 1.9.1 | Apache-2.0 OR MIT |
| zmij | 1.0.23 | MIT |
