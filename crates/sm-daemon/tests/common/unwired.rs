//! A [`SharingOps`] that is not there (decision B-101).
//!
//! It lived in the delivered library until the `spatial-memory daemon`
//! subcommand wired in `sm_daemon::ShareAdapter`; it is kept here for the
//! tests that start a daemon without sharing, and for
//! `s9_import_export_and_migrate_without_the_implementation_say_so`. Every
//! call fails with `internal` naming the missing piece; nothing pretends to
//! succeed.

use sm_core::error::{CoreError, ErrorCode, Result};
use sm_core::protocol::{
    ExportParams, ExportReport, ImportParams, ImportReport, MigrateParams, ShareParams, ShareReply,
    SyncParams, SyncReply,
};
use sm_daemon::{SharingContext, SharingOps};

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
