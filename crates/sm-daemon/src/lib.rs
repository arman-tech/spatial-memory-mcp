//! `sm-daemon`: the library behind `spatial-memory daemon` (SPEC 14,
//! stream E).
//!
//! The daemon is the only process that holds the embedding model. It binds
//! the profile's endpoint, answers length-prefixed JSON requests
//! (CONTRACTS 3.1), embeds queries on a session created first and kept for
//! life, embeds stored chunks on a bulk queue, drains the hook spool and
//! keeps the journal metrics.
//!
//! Concurrency is plain threads, not an async runtime: one accept thread,
//! one thread per connection, a loader thread, a bulk embedding worker, a
//! capture worker and a maintenance thread (spool drain, idle exit, daily
//! pruning). Every request is short and blocking (SQLite, ONNX Runtime), so
//! a runtime would add a dependency and nothing else. There is no
//! process-global state: two daemons for two profiles run in one test
//! process.

#![deny(unsafe_code)]

mod admin;
mod ambient;
mod budget;
mod capture;
mod dispatch;
pub mod endpoint;
mod loader;
mod lock;
mod log;
mod maintenance;
mod options;
mod project;
mod queue;
mod server;
mod share_adapter;
mod sharing;
mod state;
mod tools;
mod view;

pub use endpoint::connect;
pub use loader::{GemmaLoader, LoadedModel, ModelLoader};
pub use lock::DaemonLock;
pub use log::DaemonLog;
pub use options::DaemonOptions;
pub use server::{DaemonHandle, Exit, Started, run, start};
pub use share_adapter::ShareAdapter;
pub use sharing::{ResolvedProject, SharingContext, SharingOps};
