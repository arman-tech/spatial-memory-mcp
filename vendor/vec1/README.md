# vec1

`vec1.c` is SQLite's vector search extension, vendored unchanged. `sm-store` compiles it into the binary with the `cc` crate.

| Item | Value |
| --- | --- |
| Upstream | `ext/vec1/vec1.c` in the SQLite source repository, https://sqlite.org/src |
| Check-in | `60cf9357e8a60bc1de92068ed10e090cf432552f8040ba9f0294155051416347` (trunk, 2026-11-12 15:39:04 UTC) |
| Download | https://sqlite.org/src/raw/ext/vec1/vec1.c?ci=60cf9357e8a60bc1de92068ed10e090cf432552f8040ba9f0294155051416347 |
| Version | 0.8 (reported by `vec1_info()`); file header dated 2026-02-04 |
| Size | 379,695 bytes |
| SHA-256 | `93b8e1848ea3c11faaa9b533338a7e0a2404edb0acee47507b4fbee60c7a4b62` |
| SHA3-256 | `de33d11affa682b3dfa44220a06f167679465447cec7b719cfd711ea22b6d75f` |
| Licence | Public domain. The author disclaims copyright; the file carries the SQLite blessing in place of a licence. |

The check-in was recovered by looking the file's SHA3-256 up in SQLite's Fossil repository (`https://sqlite.org/src/whatis/<sha3-256>`); see `docs/v3/SPIKE.md`, spike b, and `docs/v3/DECISIONS.md`, B-7.

The repository's `.gitattributes` turns off line-ending conversion under `vendor/`, so the hash is the same on a Windows checkout. To update, replace the file with a newer check-in's copy, record the new check-in and hashes here and in `DECISIONS.md`, and rerun the `sm-store` tests.
