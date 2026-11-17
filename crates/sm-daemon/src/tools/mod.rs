//! The 13 MCP tools (SPEC 8). Every write passes `Access::Local`: shared
//! memories change only through `sync` (B-72). Every reply that lists
//! memories uses `MemoryView` and fits `limit` or `output.token_budget`.

pub(crate) mod consolidate;
pub(crate) mod edit;
pub(crate) mod link;
pub(crate) mod maps;
pub(crate) mod recall;
pub(crate) mod related;
pub(crate) mod remember;
pub(crate) mod status;
