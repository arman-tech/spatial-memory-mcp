//! Team sharing and migration for Spatial Memory 3.0 (SPEC sections 9 and
//! 10; decisions B-77, B-78 and B-79).
//!
//! - [`format`]: the shared-file format. A strict writer and parser for one
//!   Markdown file per memory with a small front matter, the file's path,
//!   and the text hash that decides re-embedding.
//! - [`share()`]: writes memories of a project into the repository's working
//!   tree, redacted, for the author to commit and review. It never commits
//!   or pushes.
//! - [`sync()`]: applies the committed shared folder at `HEAD` to the store:
//!   adds, revises and retracts shared memories, and adopts the author's own
//!   copy once the pull request has merged.
//! - [`migrate()`]: imports a version 1 or 2 JSON export.
//! - [`export()`] and [`import()`]: move a whole store as JSON Lines.
//!
//! Every function is synchronous and keeps no process-global state. Errors
//! are `sm_core::error` codes with a message and a next action (SPEC 8).
//! Memory content, shared files and import files are data: nothing in them
//! is followed as an instruction.

#![forbid(unsafe_code)]

mod export;
pub mod format;
mod git;
mod migrate;
mod share;
mod sync;

#[cfg(test)]
mod tests;

pub use export::{EXPORT_FORMAT, EXPORT_VERSION, export, import};
pub use migrate::migrate;
pub use share::share;
pub use sync::{MAX_SHARED_FILE_BYTES, sync};

/// An error about one item's data (`invalid_input`, `not_found`), which is
/// reported for that item, as opposed to the store failing (`storage`,
/// `internal`), which stops the call.
pub(crate) fn is_item_error(e: &sm_core::CoreError) -> bool {
    matches!(
        e.code,
        sm_core::ErrorCode::InvalidInput | sm_core::ErrorCode::NotFound
    )
}
