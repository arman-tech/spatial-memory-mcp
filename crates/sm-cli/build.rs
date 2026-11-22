//! Stamps the binary with the commit it was built from (B-231):
//! `SM_COMMIT` is git's short hash, with `.dirty` after it when the build
//! had uncommitted changes, or `unknown` when git cannot say (see
//! `build/stamp.rs`). `spatial-memory --version` prints
//! `<release>+<SM_COMMIT>`.
//!
//! Cargo runs this again when the checked-out commit moves (git's `HEAD`,
//! branch refs, `packed-refs` or `reftable`), when a file under the stamped
//! paths changes, or when a rule that decides which files git ignores or
//! how it compares them changes (the root's `.gitignore` and
//! `.gitattributes`, git's `info/exclude`), and only then.

use std::path::PathBuf;
use std::process::Command;

#[path = "build/stamp.rs"]
mod stamp;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let manifest = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").expect("set by cargo"));
    let stamp = stamp::stamp(&manifest, &|| Command::new("git"));
    for path in &stamp.watch {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    println!("cargo:rustc-env=SM_COMMIT={}", stamp.commit);
}
