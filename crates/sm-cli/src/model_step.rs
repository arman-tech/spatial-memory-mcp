//! The model step of `setup` and `doctor` (decision B-43).
//!
//! SPEC section 3 says `setup` and `doctor` download and verify the model.
//! Only `sm-embed` holds the download code and the pinned hashes, and
//! `sm-cli` takes `sm-embed` in phase 3. Until then both commands go through
//! this trait, private to `sm-cli`, and the binary uses [`Unwired`], which
//! says so and fails. Tests supply their own implementation.

use sm_core::config::Settings;
use sm_core::error::{CoreError, ErrorCode};

/// Downloads and verifies the embedding model.
pub trait ModelStep: Send + Sync {
    /// Makes the configured model present and verified, downloading it when
    /// it is missing (`setup`). Returns a line describing the model.
    fn ensure(&self, settings: &Settings) -> Result<String, CoreError>;

    /// Verifies the configured model without downloading (`doctor`).
    /// Returns a line describing the model.
    fn check(&self, settings: &Settings) -> Result<String, CoreError>;
}

/// The phase 1 build: no model step is wired, and both calls say so.
#[derive(Debug, Clone, Copy, Default)]
pub struct Unwired;

/// The message [`Unwired`] fails with.
pub const UNWIRED_MESSAGE: &str = "the model download and checksum step is not wired in this build \
     (decision B-43: sm-cli takes sm-embed in phase 3)";

/// The next action [`Unwired`] fails with.
pub const UNWIRED_NEXT_ACTION: &str =
    "Use a phase 3 build of spatial-memory to download and verify the model.";

fn unwired() -> CoreError {
    CoreError::new(ErrorCode::ModelMissing, UNWIRED_MESSAGE).with_next_action(UNWIRED_NEXT_ACTION)
}

impl ModelStep for Unwired {
    fn ensure(&self, _settings: &Settings) -> Result<String, CoreError> {
        Err(unwired())
    }

    fn check(&self, _settings: &Settings) -> Result<String, CoreError> {
        Err(unwired())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s3_unwired_model_step_fails_and_says_why() {
        let s = Settings::default();
        let e = Unwired.ensure(&s).unwrap_err();
        assert_eq!(e.code, ErrorCode::ModelMissing);
        assert!(e.message.contains("not wired in this build"));
        assert_eq!(e.next_action, UNWIRED_NEXT_ACTION);
        assert_eq!(Unwired.check(&s).unwrap_err(), e);
    }
}
