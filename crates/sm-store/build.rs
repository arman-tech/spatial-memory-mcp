//! Compiles SQLite's vec1 vector extension (`vendor/vec1/vec1.c`, vendored
//! unchanged, decision B-7) and links it statically into `sm-store`.
//!
//! On x86-64 the file is compiled twice, as spike b did: a scalar object
//! (`VEC1SIMD=SCALAR`), which holds the entry points and picks a code path at
//! run time with CPUID, and an AVX2 object (`VEC1SIMD=AVX2`) built with
//! `/arch:AVX2` (MSVC) or `-mavx2 -mfma` (GCC and Clang). Other targets get
//! one plain object. Both are compiled against the bundled SQLite header that
//! `libsqlite3-sys` exports as `DEP_SQLITE3_INCLUDE`, with `VEC1_STATIC` so
//! that vec1 calls the linked SQLite directly rather than through a loadable
//! extension's API table.

use std::env;
use std::path::PathBuf;

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let source = manifest.join("../../vendor/vec1/vec1.c");
    println!("cargo:rerun-if-changed={}", source.display());
    println!("cargo:rerun-if-changed=build.rs");

    let include = env::var("DEP_SQLITE3_INCLUDE").expect(
        "DEP_SQLITE3_INCLUDE: libsqlite3-sys with the bundled feature must be a direct dependency",
    );
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let msvc = env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc");

    let base = || {
        let mut build = cc::Build::new();
        build
            .file(&source)
            .include(&include)
            .define("NDEBUG", None)
            .define("VEC1_STATIC", None)
            // vec1 is the hot loop of cross-project search; it is optimised
            // even in debug builds of the workspace.
            .opt_level(3)
            .warnings(false);
        build
    };

    if arch == "x86_64" {
        base().define("VEC1SIMD", "SCALAR").compile("vec1_scalar");
        let mut avx2 = base();
        avx2.define("VEC1SIMD", "AVX2");
        if msvc {
            avx2.flag("/arch:AVX2");
        } else {
            avx2.flag("-mavx2").flag("-mfma");
        }
        avx2.compile("vec1_avx2");
    } else {
        base().compile("vec1");
    }
}
