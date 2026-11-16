//! The one ONNX Runtime environment of the process.
//!
//! ONNX Runtime allows one environment per process, and ort refuses to build
//! a second even after the first is dropped, so it is built once and kept.
//! Telemetry is off.

use std::sync::OnceLock;

use ort::environment::Environment;
use sm_core::error::{CoreError, Result};

/// The process environment, built on first use.
pub fn environment() -> Result<Environment> {
    static ENV: OnceLock<std::result::Result<Environment, String>> = OnceLock::new();
    ENV.get_or_init(|| {
        ort::init()
            .with_name("spatial-memory")
            .with_telemetry(false)
            .build()
            .map_err(|e| e.to_string())
    })
    .clone()
    .map_err(|e| {
        CoreError::internal(format!("cannot start ONNX Runtime: {e}"))
            .with_next_action("Run `spatial-memory doctor` and report the error it shows.")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s5_the_environment_is_built_once_and_shared() {
        let a = environment().unwrap();
        let b = environment().unwrap();
        drop(a);
        drop(b);
        assert!(environment().is_ok());
    }
}
