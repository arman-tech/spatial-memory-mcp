//! Registration of the statically linked vec1 extension. This is the only
//! `unsafe` code in `sm-store`.

use std::ffi::{c_char, c_int};
use std::sync::OnceLock;

use sm_core::error::{CoreError, Result};

unsafe extern "C" {
    /// Defined in `vendor/vec1/vec1.c` when compiled with `VEC1_STATIC`
    /// (see `build.rs`). It ignores its argument and calls
    /// `sqlite3_auto_extension(initExtension)`, so that every SQLite
    /// connection the process opens afterwards has the `vec1` module and its
    /// functions. Returns `SQLITE_OK` (0).
    fn sqlite3_vec1_extra_init(z: *const c_char) -> c_int;
}

static REGISTERED: OnceLock<c_int> = OnceLock::new();

/// Registers vec1 as an automatic extension, once per process. Must run
/// before the store opens a connection; `Store::open` calls it first.
pub(crate) fn register() -> Result<()> {
    let rc = *REGISTERED.get_or_init(|| {
        // SAFETY: `sqlite3_vec1_extra_init` takes no ownership and does not
        // read its argument (`UNUSED_PARAMETER(z)` in vec1.c), so a null
        // pointer is valid. It only calls `sqlite3_auto_extension`, which
        // initialises SQLite if needed and appends a function pointer to a
        // list guarded by SQLite's own static mutex (the bundled library is
        // built with SQLITE_THREADSAFE=1), so it is sound from any thread.
        // The `OnceLock` makes it run once, so the entry point is never
        // registered twice. The registered function, vec1's `initExtension`,
        // lives in this binary's static code and is never unloaded.
        unsafe { sqlite3_vec1_extra_init(std::ptr::null()) }
    });
    if rc == 0 {
        Ok(())
    } else {
        Err(CoreError::storage(format!(
            "registering the vec1 extension failed with SQLite code {rc}"
        )))
    }
}
