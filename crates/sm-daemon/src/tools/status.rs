//! `status` (SPEC 8; SPEC 6, the metrics journal): the daemon's state and
//! `state_error`, the model, the index stamp and pending embeddings,
//! counts per project, the spool, and, when asked, the journal's 7-day
//! medians and 95th percentiles. It needs no model, so it is served while
//! the model loads or after it failed (B-36).

use sm_core::error::Result;
use sm_core::journal::{SUMMARY_WINDOW_DAYS, summarize};
use sm_core::protocol::{ModelStatus, StatusParams, StatusReply};
use sm_core::spool::list_spool;
use sm_core::time::Timestamp;

use crate::budget::{fit_whole, fixed_cost};
use crate::dispatch::{Ctx, hello};

pub(crate) fn status(ctx: &Ctx<'_>, p: &StatusParams) -> Result<StatusReply> {
    let sh = ctx.sh;
    let daemon = hello(sh);
    let model = ModelStatus {
        model_id: sh.info.model_id.clone(),
        revision: sh.info.revision.clone(),
        dim: sh.info.dim as u32,
        loaded: sh.ready_opt().is_some(),
    };
    let index = sh.store.index_status()?;
    // A working directory outside any project is not an error here, and a
    // project the store has not seen is not created: status only reads
    // (B-151).
    let current_project = crate::project::lookup(sh, &ctx.req.cwd)
        .ok()
        .flatten()
        .map(|e| e.project.name);
    let metrics = p.metrics.then(|| {
        summarize(
            &sh.paths.metrics_dir(),
            Timestamp::now(),
            SUMMARY_WINDOW_DAYS,
        )
    });
    let spool_pending = list_spool(&sh.paths.spool_new()).map_or(0, |l| l.len() as u64);
    // Every field but the project list is fixed; a limit those alone
    // exceed is refused (e-report decision 24).
    let mut reply = StatusReply {
        daemon,
        model,
        index,
        current_project,
        projects: Vec::new(),
        database: sh.paths.database().display().to_string(),
        database_bytes: sh.store.database_bytes(),
        spool_pending,
        judge: crate::judge::status(sh)?,
        metrics,
        omitted: u32::MAX,
    };
    let budget = ctx.budget(p.limit);
    let used = fixed_cost("status", &reply, budget)?;
    let projects = fit_whole(sh.store.project_statuses()?, budget, used);
    reply.projects = projects.items;
    reply.omitted = projects.omitted;
    Ok(reply)
}
