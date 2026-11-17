//! The `spatial-memory` binary. Everything lives in the `sm_cli` library so
//! the integration tests can drive the same code with a mock daemon.

fn main() {
    let code = sm_cli::main_entry();
    // `exit` rather than returning: a hook may leave an exchange thread
    // blocked on a daemon that never answers, and the process must not wait
    // for it (CONTRACTS section 7, Never blocks).
    std::process::exit(code);
}
