//! Structured errors: a stable code, a message and what the caller can do
//! next (SPEC section 8, Conventions for every tool). The same type travels
//! over the socket, so a client sees exactly what the daemon produced.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

string_enum! {
    /// Stable error codes. Agents and tests match on these, never on the
    /// message text. Adding a code is a contract change (SPEC section 14).
    pub enum ErrorCode {
        /// A parameter is missing, malformed or out of range.
        InvalidInput => "invalid_input",
        /// The memory, project, anchor or link does not exist.
        NotFound => "not_found",
        /// The operation touches more than one memory and needs `confirm: true`
        /// (SPEC section 8, `forget`).
        ConfirmRequired => "confirm_required",
        /// A shared memory cannot be edited or forgotten locally (SPEC section 4,
        /// Forgetting cascades; section 9, Team sharing rule 5).
        SharedReadOnly => "shared_read_only",
        /// The configured model or chunk settings differ from the index stamp;
        /// search is refused until `reindex` runs (SPEC section 4, Model stamp; AT-11).
        IndexMismatch => "index_mismatch",
        /// The model files are not in the model directory.
        ModelMissing => "model_missing",
        /// A model file's size or SHA-256 differs from the pinned value (AT-10).
        ModelIntegrity => "model_integrity",
        /// The daemon is running but has not finished loading the model.
        DaemonLoading => "daemon_loading",
        /// No daemon answered and none could be started.
        DaemonUnavailable => "daemon_unavailable",
        /// Client and daemon builds or protocol versions differ.
        VersionMismatch => "version_mismatch",
        /// The working directory does not resolve to a project.
        ProjectUnresolved => "project_unresolved",
        /// Content still matches the secret scanner after redaction; `share`
        /// refuses to write it (SPEC section 9, Security).
        SecretDetected => "secret_detected",
        /// The operation needs a git repository and the project has none.
        NotGitRepository => "not_git_repository",
        /// The database failed (I/O, corruption, constraint).
        Storage => "storage",
        /// The operation did not finish within its deadline.
        Timeout => "timeout",
        /// A configuration file failed to parse or holds an out-of-range value.
        ConfigInvalid => "config_invalid",
        /// A bug. The message says where.
        Internal => "internal",
    }
}

impl ErrorCode {
    /// The default advice for the caller, used when the producer has nothing
    /// more specific to say.
    pub const fn default_next_action(self) -> &'static str {
        match self {
            ErrorCode::InvalidInput => "Correct the parameter named in the message and retry.",
            ErrorCode::NotFound => "Check the identifier, or search with recall to find it.",
            ErrorCode::ConfirmRequired => {
                "Review what will be affected, then retry with confirm: true."
            }
            ErrorCode::SharedReadOnly => {
                "Change the shared file in the repository and run sync; local edits to shared memories are refused."
            }
            ErrorCode::IndexMismatch => "Run `spatial-memory reindex`, then retry.",
            ErrorCode::ModelMissing => "Run `spatial-memory setup` to download the model.",
            ErrorCode::ModelIntegrity => {
                "Delete the model directory named in the message and run `spatial-memory setup` again."
            }
            ErrorCode::DaemonLoading => "Retry in a few seconds; the model is loading.",
            ErrorCode::DaemonUnavailable => {
                "Run `spatial-memory doctor` to see why the daemon cannot start."
            }
            ErrorCode::VersionMismatch => {
                "Retry; the client restarts the daemon with the current build."
            }
            ErrorCode::ProjectUnresolved => {
                "Run from inside a project folder, or pass scope: all to search every project."
            }
            ErrorCode::SecretDetected => "Remove the credential from the content and retry.",
            ErrorCode::NotGitRepository => "Run from inside a git repository.",
            ErrorCode::Storage => {
                "Run `spatial-memory doctor`; if the database is damaged, restore the latest backup."
            }
            ErrorCode::Timeout => "Retry; if it keeps happening, run `spatial-memory doctor`.",
            ErrorCode::ConfigInvalid => "Fix the configuration file named in the message.",
            ErrorCode::Internal => "Report this as a bug with the message text.",
        }
    }
}

/// The one error type every crate returns across a contract boundary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema, thiserror::Error)]
#[error("{code}: {message}")]
pub struct CoreError {
    /// Stable code to match on.
    pub code: ErrorCode,
    /// What went wrong, for a human.
    pub message: String,
    /// What the caller can do next.
    pub next_action: String,
}

/// Result alias used throughout the workspace.
pub type Result<T, E = CoreError> = std::result::Result<T, E>;

impl CoreError {
    /// An error with the code's default next action.
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        CoreError {
            code,
            message: message.into(),
            next_action: code.default_next_action().to_owned(),
        }
    }

    /// Replaces the next action with something more specific.
    #[must_use]
    pub fn with_next_action(mut self, next_action: impl Into<String>) -> Self {
        self.next_action = next_action.into();
        self
    }

    pub fn invalid_input(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::InvalidInput, message)
    }

    pub fn not_found(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::NotFound, message)
    }

    pub fn storage(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::Storage, message)
    }

    pub fn config_invalid(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::ConfigInvalid, message)
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::Internal, message)
    }
}

impl From<std::io::Error> for CoreError {
    fn from(e: std::io::Error) -> Self {
        CoreError::storage(format!("I/O error: {e}"))
    }
}

impl From<serde_json::Error> for CoreError {
    fn from(e: serde_json::Error) -> Self {
        CoreError::invalid_input(format!("invalid JSON: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s8_error_codes_round_trip_as_snake_case_strings() {
        for code in ErrorCode::ALL {
            let json = serde_json::to_string(code).unwrap();
            assert_eq!(json, format!("\"{}\"", code.as_str()));
            let back: ErrorCode = serde_json::from_str(&json).unwrap();
            assert_eq!(back, *code);
            assert!(!code.default_next_action().is_empty());
        }
    }

    #[test]
    fn s8_error_serialises_code_message_and_next_action() {
        let e = CoreError::new(ErrorCode::ConfirmRequired, "3 memories match");
        let v = serde_json::to_value(&e).unwrap();
        assert_eq!(v["code"], "confirm_required");
        assert_eq!(v["message"], "3 memories match");
        assert_eq!(
            v["next_action"],
            ErrorCode::ConfirmRequired.default_next_action()
        );
        assert_eq!(e.to_string(), "confirm_required: 3 memories match");
    }

    #[test]
    fn s8_unknown_error_code_is_rejected() {
        let r: std::result::Result<ErrorCode, _> = serde_json::from_str("\"nope\"");
        assert!(r.is_err());
        let e = "nope".parse::<ErrorCode>().unwrap_err();
        assert_eq!(e.code, ErrorCode::InvalidInput);
    }
}
