//! The seam to stream F (decision B-79): share, sync, migrate, import and
//! export run in the daemon process but are implemented by `sm-share`.
//! `sm-daemon` does not depend on `sm-share`; the CLI passes the real
//! implementation in when it starts the daemon.

use sm_core::error::{CoreError, ErrorCode, Result};
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

/// The implementation used until stream F's `sm-share` is wired in by the
/// CLI. Every call fails with `internal` naming the missing piece; nothing
/// pretends to succeed.
#[derive(Debug, Default, Clone, Copy)]
pub struct UnwiredSharing;

fn unwired(op: &str) -> CoreError {
    CoreError::new(
        ErrorCode::Internal,
        format!("{op} is not available: the daemon was started without the sharing implementation"),
    )
    .with_next_action("Start the daemon with `spatial-memory daemon`, which wires in sharing.")
}

impl SharingOps for UnwiredSharing {
    fn share(&self, _: &SharingContext<'_>, _: &ShareParams) -> Result<ShareReply> {
        Err(unwired("share"))
    }
    fn sync(&self, _: &SharingContext<'_>, _: &SyncParams) -> Result<SyncReply> {
        Err(unwired("sync"))
    }
    fn migrate(&self, _: &SharingContext<'_>, _: &MigrateParams) -> Result<ImportReport> {
        Err(unwired("migrate"))
    }
    fn import(&self, _: &SharingContext<'_>, _: &ImportParams) -> Result<ImportReport> {
        Err(unwired("import"))
    }
    fn export(&self, _: &SharingContext<'_>, _: &ExportParams) -> Result<ExportReport> {
        Err(unwired("export"))
    }
}
