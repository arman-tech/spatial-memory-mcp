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
mod redact;
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

/// Whether consolidation merged this memory into another (B-73):
/// `sm-store`'s `merge_memories` sets it `superseded` and records
/// provenance `extra.merged_into`, with no `supersedes_id` pointing at it.
/// No superseder holds it, so the liveness rule alone would make it active.
/// `sync`'s settling and `import`'s status restore never change its status,
/// and `sync` takes its file's `active` as `superseded`. A file that says
/// `retracted`, or a deleted file, still retracts it.
pub(crate) fn is_merged(m: &sm_core::types::Memory) -> bool {
    m.provenance
        .extra
        .as_ref()
        .and_then(|e| e.get("merged_into"))
        .is_some_and(|v| !v.is_null())
}
