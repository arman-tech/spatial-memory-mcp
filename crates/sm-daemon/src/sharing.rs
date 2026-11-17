//! The seam to stream F (decision B-79): share, sync, migrate, import and
//! export run in the daemon process but are implemented by `sm-share`.
//! The daemon calls them through [`SharingOps`]; the implementation the
//! `spatial-memory daemon` subcommand passes in is
//! [`crate::ShareAdapter`], which calls `sm-share`'s functions. Tests pass
//! their own implementations.

use sm_core::error::Result;
use sm_core::project::ProjectIdentity;
use sm_core::protocol::{
    ExportParams, ExportReport, ImportParams, ImportReport, MigrateParams, ShareParams, ShareReply,
    SyncParams, SyncReply,
};
use sm_core::traits::{ChunkParams, Chunker, SecretScanner};
use sm_core::types::Project;
use sm_store::Store;

/// The project the request resolved to, with the identity it came from.
/// `identity.repo` gives the repository root for share and sync.
#[derive(Clone, Copy)]
pub struct ResolvedProject<'a> {
    pub project: &'a Project,
    pub identity: &'a ProjectIdentity,
}

/// What a sharing operation may use. The chunker is `None` while the model
/// loads or after it failed; memories written then are stored with pending
/// chunks only if the operation can chunk them some other way, otherwise it
/// must refuse them.
pub struct SharingContext<'a> {
    pub store: &'a Store,
    pub chunker: Option<&'a dyn Chunker>,
    pub scanner: &'a dyn SecretScanner,
    pub chunk_params: ChunkParams,
    pub project: Option<ResolvedProject<'a>>,
}

/// Stream F's operations. After each call the daemon wakes the embedding
/// queue and drops the map cache, so imported and synced memories are
/// embedded and the maps recomputed.
pub trait SharingOps: Send + Sync {
    fn share(&self, ctx: &SharingContext<'_>, params: &ShareParams) -> Result<ShareReply>;
    fn sync(&self, ctx: &SharingContext<'_>, params: &SyncParams) -> Result<SyncReply>;
    fn migrate(&self, ctx: &SharingContext<'_>, params: &MigrateParams) -> Result<ImportReport>;
    fn import(&self, ctx: &SharingContext<'_>, params: &ImportParams) -> Result<ImportReport>;
    fn export(&self, ctx: &SharingContext<'_>, params: &ExportParams) -> Result<ExportReport>;
}
