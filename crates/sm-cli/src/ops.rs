//! The command-line operations: `status`, `import`, `export`, `migrate`,
//! `reindex`, `project rename` and `project delete` (SPEC section 3, command
//! table; SPEC section 10, step 3). Each sends one op through the waiting
//! client and prints the report.

use std::fmt::Write as _;

use sm_core::error::{CoreError, ErrorCode};
use sm_core::eval::{EvalReport, ScoreDistribution};
use sm_core::protocol::{
    DaemonState, ExportReport, ImportReport, JudgeBlocked, JudgeStatus, Op, ProjectReply,
    ReindexReport, Reply, StatusReply,
};

/// What an operation printed and how it ended.
#[derive(Debug, Clone, PartialEq)]
pub struct OpOutcome {
    /// Text for stdout.
    pub stdout: String,
    /// Text for stderr: the `error:` and `next:` lines of an error the
    /// reply reports, such as a daemon whose model failed to load.
    pub stderr: String,
    /// The process exit code.
    pub code: i32,
}

/// Runs one operation through `call` (the waiting client in the binary) and
/// renders its reply. `project delete` without `--confirm` is a preview:
/// the daemon refuses with `confirm_required` and says what would be
/// removed (B-30), which is printed as the command's result.
pub fn run(
    op: Op,
    call: impl FnOnce(Op) -> Result<Reply, CoreError>,
) -> Result<OpOutcome, CoreError> {
    let preview = matches!(&op, Op::ProjectDelete(p) if !p.confirm);
    match call(op) {
        Ok(reply) => Ok(OpOutcome {
            stdout: render(&reply),
            stderr: reported_errors(&reply)
                .iter()
                .map(crate::cli::error_text)
                .collect(),
            code: exit_code(&reply),
        }),
        Err(e) if preview && e.code == ErrorCode::ConfirmRequired => Ok(OpOutcome {
            stdout: format!(
                "Nothing was deleted. {}\nRun again with --confirm to delete.\n",
                e.message
            ),
            stderr: String::new(),
            code: 0,
        }),
        Err(e) => Err(e),
    }
}

/// The exit code of a reply: `doctor --eval` fails on a regression against
/// the baseline, and `status` when the daemon reports an error; everything
/// else that arrived succeeded.
pub fn exit_code(reply: &Reply) -> i32 {
    match reply {
        Reply::Eval(r) if !r.regressions.is_empty() => 1,
        _ if !reported_errors(reply).is_empty() => 1,
        _ => 0,
    }
}

/// The errors a successful reply reports. For `status`: a model that failed
/// to load (the daemon's `state_error`), and an index stamp that differs
/// from the configuration, on which every search fails with
/// `index_mismatch`. A daemon still loading reports no error.
pub fn reported_errors(reply: &Reply) -> Vec<CoreError> {
    let Reply::Status(r) = reply else {
        return Vec::new();
    };
    let mut errors = Vec::new();
    if r.daemon.state == DaemonState::Failed {
        errors.push(r.daemon.state_error.clone().unwrap_or_else(|| {
            CoreError::internal("the daemon reports that its model failed without saying why")
                .with_next_action(
                    "Read daemon.log in the profile folder, then run `spatial-memory doctor`.",
                )
        }));
    }
    if !r.index.matches {
        errors.push(
            CoreError::new(
                ErrorCode::IndexMismatch,
                format!(
                    "the index does not match the configuration: {}",
                    r.index.differences.join("; ")
                ),
            )
            .with_next_action(
                "Run `spatial-memory reindex`, or set the model and chunk settings back.",
            ),
        );
    }
    errors
}

/// The text form of a reply.
pub fn render(reply: &Reply) -> String {
    match reply {
        Reply::Status(r) => render_status(r),
        Reply::Import(r) => render_import(r),
        Reply::Export(r) => render_export(r),
        Reply::Reindex(r) => render_reindex(r),
        Reply::Project(r) => render_project(r),
        Reply::Eval(r) => render_eval(r),
        other => {
            let mut s = serde_json::to_string_pretty(other).unwrap_or_default();
            s.push('\n');
            s
        }
    }
}

fn counts(map: &std::collections::BTreeMap<String, u64>) -> String {
    let parts: Vec<String> = map.iter().map(|(k, v)| format!("{k} {v}")).collect();
    if parts.is_empty() {
        String::from("none")
    } else {
        parts.join(", ")
    }
}

/// Why the capture judge cannot run, in words (SPEC section 16, The judge;
/// B-165).
pub(crate) fn judge_blocked_text(b: JudgeBlocked) -> &'static str {
    match b {
        JudgeBlocked::Disabled => "it is turned off (capture.judge.enabled is false)",
        JudgeBlocked::CaptureDisabled => "automatic capture is off (capture.enabled is false)",
        JudgeBlocked::NoJudge => "this daemon was started without one",
        JudgeBlocked::Unavailable => "Claude Code is not available",
        JudgeBlocked::DailyCap => "today's calls are used (capture.judge.max_calls_per_day)",
        JudgeBlocked::DailyBudget => "today's spending is used (capture.judge.max_usd_per_day)",
        JudgeBlocked::BackingOff => "a call failed, and it waits before the next",
    }
}

/// Whether the judge can run, why not, and until when.
pub(crate) fn judge_state(j: &JudgeStatus) -> String {
    let mut s = match j.blocked {
        None => String::from("can run"),
        Some(b) => format!("cannot run: {}", judge_blocked_text(b)),
    };
    if let Some(at) = &j.retry_at {
        let _ = write!(s, ", until {at}");
    }
    s
}

/// The model and the Claude Code executable the judge runs.
pub(crate) fn judge_runs(j: &JudgeStatus) -> String {
    match &j.command {
        Some(c) => format!("model {} through {c}", j.model),
        None => format!("model {}, Claude Code not found", j.model),
    }
}

/// The pending captures, how many of them wait for the judge, the rejected
/// captures kept for 30 days (B-190) and those the judge gave up on
/// (B-206), each when there are any.
pub(crate) fn judge_pending(j: &JudgeStatus) -> String {
    let mut s = format!("{} captures, {} waiting for the judge", j.pending, j.queued);
    if j.rejected > 0 {
        s.push_str(&format!(", {} rejected (kept 30 days)", j.rejected));
    }
    if j.failed > 0 {
        s.push_str(&format!(", {} failed judgement (not served)", j.failed));
    }
    s
}

/// Today's calls against the cap, the captures judged and the cost,
/// against the spending cap when the daemon reports one (B-193).
pub(crate) fn judge_today(j: &JudgeStatus) -> String {
    let mut s = format!(
        "{} of {} calls, {} judged, about {:.4} US dollars",
        j.calls_today, j.max_calls_per_day, j.judged_today, j.cost_today_usd
    );
    if let Some(max) = j.max_usd_per_day {
        s.push_str(&format!(" of {max:.2}"));
    }
    s.push_str(" (Claude Code's estimate)");
    s
}

/// Automatic capture over the last 7 days, from the journal (B-200): the
/// captures the judge settled, kept and rejected, the captured memories
/// removed with `forget`, and the judge's calls and cost. `None` from a
/// daemon before it.
pub(crate) fn judge_week(j: &JudgeStatus) -> Option<String> {
    let q = j.quality.as_ref()?;
    let mut s = format!(
        "{} judged: {} kept, {} rejected",
        q.judged, q.kept, q.rejected
    );
    if q.dropped_after_keep > 0 {
        let _ = write!(
            s,
            " ({} of them kept, then dropped as duplicates)",
            q.dropped_after_keep
        );
    }
    let _ = write!(s, "; {} captured memories removed with forget", q.forgotten);
    if q.forgotten_pending > 0 {
        let _ = write!(s, ", and {} pending captures", q.forgotten_pending);
    }
    let _ = write!(
        s,
        "; {} judge calls, about {:.4} US dollars (Claude Code's estimate)",
        q.calls, q.cost_usd
    );
    Some(s)
}

/// `spatial-memory status`: the daemon, the model, the index and queue, the
/// database and spool, the capture judge, and the memories of each project.
pub fn render_status(r: &StatusReply) -> String {
    let d = &r.daemon;
    let mut s = String::new();
    let _ = writeln!(
        s,
        "Daemon: {} (pid {}, profile {}, build {}, protocol {})",
        d.state, d.pid, d.profile, d.build, d.protocol
    );
    let _ = writeln!(s, "  endpoint:  {}", d.endpoint);
    let ready = d
        .ready_seconds
        .map(|t| format!(", ready after {t:.1} s"))
        .unwrap_or_default();
    let _ = writeln!(s, "  started:   {}{ready}", d.started_at);
    if let Some(e) = &d.state_error {
        let _ = writeln!(s, "  error:     {}: {}", e.code, e.message);
    }
    let m = &r.model;
    let _ = writeln!(
        s,
        "Model: {} at revision {}, {} dimensions, {}",
        m.model_id,
        m.revision,
        m.dim,
        if m.loaded { "loaded" } else { "not loaded" }
    );
    let i = &r.index;
    let _ = writeln!(
        s,
        "Index: {}; {} chunks, {} waiting for embedding",
        if i.matches {
            String::from("matches the configuration")
        } else {
            format!(
                "does not match the configuration ({})",
                i.differences.join("; ")
            )
        },
        i.chunks,
        i.pending_embeddings
    );
    let _ = writeln!(s, "Database: {} ({} bytes)", r.database, r.database_bytes);
    let _ = writeln!(s, "Spool: {} captures waiting", r.spool_pending);
    let _ = writeln!(
        s,
        "Capture queue: {} jobs queued or running",
        r.captures_queued
    );
    let j = &r.judge;
    let _ = writeln!(s, "Judge: {}; {}", judge_state(j), judge_runs(j));
    let _ = writeln!(s, "  pending:    {}", judge_pending(j));
    let _ = writeln!(s, "  today:      {}", judge_today(j));
    if let (Some(week), Some(q)) = (judge_week(j), &j.quality) {
        let _ = writeln!(s, "  {:<12}{week}", format!("{} days:", q.days));
    }
    if let Some(e) = &j.last_error {
        let _ = writeln!(s, "  last error: {e}");
    }
    if let Some(p) = &r.current_project {
        let _ = writeln!(s, "Current project: {p}");
    }
    // The reply counts memories only for the projects it lists, so with
    // projects left out the total is said to be theirs (B-138).
    let shown = r.projects.len();
    let all = shown + r.omitted as usize;
    let total: u64 = r.projects.iter().map(|p| p.memories).sum();
    let _ = match (r.omitted, shown) {
        (0, _) => writeln!(s, "Projects: {all} with {total} memories"),
        (_, 0) => writeln!(s, "Projects: {all}, none shown"),
        _ => writeln!(
            s,
            "Projects: {all}; the {shown} shown hold {total} memories"
        ),
    };
    for p in &r.projects {
        let _ = writeln!(
            s,
            "  {}: {} memories, {} shared",
            p.name, p.memories, p.shared
        );
        let _ = writeln!(s, "    by type:   {}", counts(&p.by_type));
        let _ = writeln!(s, "    by status: {}", counts(&p.by_status));
        if let Some(remote) = &p.git_remote {
            let _ = writeln!(s, "    git remote: {remote}");
        }
    }
    if r.omitted > 0 {
        let _ = writeln!(s, "  ({} more projects not shown)", r.omitted);
    }
    for metric in r.metrics.iter().flatten() {
        let num = |v: Option<f64>| v.map_or_else(|| String::from("-"), |x| format!("{x:.3}"));
        let _ = writeln!(
            s,
            "Metric {}: median {} p95 {} (n={})",
            metric.metric,
            num(metric.median),
            num(metric.p95),
            metric.count
        );
    }
    s
}

fn render_import(r: &ImportReport) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "Imported {}", r.source);
    let _ = writeln!(s, "  records read:        {}", r.read);
    let _ = writeln!(s, "  inserted:            {}", r.inserted);
    let _ = writeln!(s, "  already present:     {}", r.skipped_existing);
    let _ = writeln!(s, "  with secrets redacted: {}", r.redacted);
    let _ = writeln!(s, "  pending embeddings:  {}", r.pending_embeddings);
    if !r.projects_created.is_empty() {
        let _ = writeln!(
            s,
            "  projects created:    {}",
            r.projects_created.join(", ")
        );
    }
    if !r.issues.is_empty() {
        let _ = writeln!(s, "  issues ({}):", r.issues.len());
        for issue in &r.issues {
            match &issue.id {
                Some(id) => {
                    let _ = writeln!(s, "    record {} ({id}): {}", issue.index, issue.message);
                }
                None => {
                    let _ = writeln!(s, "    record {}: {}", issue.index, issue.message);
                }
            }
        }
    }
    s
}

fn render_export(r: &ExportReport) -> String {
    format!(
        "Exported {} memories and {} links from {} projects to {}\n",
        r.memories, r.links, r.projects, r.path
    )
}

fn render_reindex(r: &ReindexReport) -> String {
    format!(
        "Reindexed {} memories; {} chunks queued for embedding.\nBackup: {}\n",
        r.memories, r.chunks_queued, r.backup
    )
}

fn render_project(r: &ProjectReply) -> String {
    match (&r.renamed_from, r.deleted) {
        (Some(from), _) => format!(
            "Renamed project {from} to {}; {} memories moved.\n",
            r.name, r.memories
        ),
        (None, true) => format!(
            "Deleted project {} and its {} memories.\n",
            r.name, r.memories
        ),
        (None, false) => format!("Project {}: nothing changed.\n", r.name),
    }
}

fn number(v: Option<f64>) -> String {
    v.map_or_else(|| String::from("-"), |x| format!("{x:.4}"))
}

fn distribution(label: &str, d: &ScoreDistribution) -> String {
    format!(
        "{label} (n={}): min {} p5 {} p25 {} median {} p75 {} p95 {} max {}\n",
        d.count,
        number(d.min),
        number(d.p5),
        number(d.p25),
        number(d.median),
        number(d.p75),
        number(d.p95),
        number(d.max)
    )
}

/// `doctor --eval`: the three AT-2 metrics, the score distributions, the
/// baseline comparison and the queries that missed.
pub fn render_eval(r: &EvalReport) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "Evaluation: {} queries", r.queries);
    let _ = writeln!(s, "  nDCG@10  {:.4}", r.metrics.ndcg_at_10);
    let _ = writeln!(s, "  hit@5    {:.4}", r.metrics.hit_at_5);
    let _ = writeln!(s, "  MRR      {:.4}", r.metrics.mrr);
    s.push_str(&distribution("Relevant scores", &r.relevant_scores));
    s.push_str(&distribution("Irrelevant scores", &r.irrelevant_scores));
    if let Some(b) = &r.baseline {
        let _ = writeln!(
            s,
            "Baseline{}: nDCG@10 {:.4}, hit@5 {:.4}, MRR {:.4}",
            b.model
                .as_deref()
                .map(|m| format!(" ({m})"))
                .unwrap_or_default(),
            b.ndcg_at_10,
            b.hit_at_5,
            b.mrr
        );
        if r.regressions.is_empty() {
            let _ = writeln!(s, "No regressions against the baseline.");
        }
    }
    for reg in &r.regressions {
        let _ = writeln!(s, "REGRESSION: {reg}");
    }
    let misses: Vec<_> = r.per_query.iter().filter(|q| !q.hit_at_5).collect();
    if !misses.is_empty() {
        let _ = writeln!(
            s,
            "Queries without a relevant memory in the top 5 ({}):",
            misses.len()
        );
        for q in misses {
            let rank = q
                .first_relevant_rank
                .map_or_else(|| String::from("not found"), |r| format!("first at {r}"));
            let _ = writeln!(s, "  {:?}: {rank}", q.text);
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use sm_core::protocol::{ImportIssue, ProjectDeleteParams, ReindexParams, StatusParams};

    use super::*;

    /// A `status` reply for tests: a ready daemon with one project, or `state`
    /// with `state_error`.
    fn sample_status(state: DaemonState, state_error: Option<CoreError>) -> StatusReply {
        use sm_core::protocol::{DaemonInfo, IndexStatus, ModelStatus, ProjectStatus};
        let ready = state == DaemonState::Ready;
        StatusReply {
            daemon: DaemonInfo {
                protocol: sm_core::protocol::PROTOCOL_VERSION,
                build: sm_core::BUILD.to_owned(),
                pid: 4242,
                profile: "default".into(),
                state,
                state_error,
                started_at: sm_core::time::Timestamp::now(),
                ready_seconds: ready.then_some(2.5),
                model_id: "embeddinggemma-2".into(),
                endpoint: "sm-test-endpoint".into(),
            },
            model: ModelStatus {
                model_id: "embeddinggemma-2".into(),
                revision: "daa72c5".into(),
                dim: 768,
                loaded: ready,
            },
            index: IndexStatus {
                matches: true,
                differences: Vec::new(),
                chunks: 40,
                pending_embeddings: 3,
            },
            current_project: Some("acme".into()),
            projects: vec![ProjectStatus {
                name: "acme".into(),
                git_remote: Some("github.com/acme/app".into()),
                memories: 12,
                by_type: [("decision".to_owned(), 7), ("fact".to_owned(), 5)].into(),
                by_status: [("active".to_owned(), 12)].into(),
                shared: 2,
            }],
            database: "/data/default/memory.db".into(),
            database_bytes: 65536,
            spool_pending: 1,
            captures_queued: 2,
            judge: sm_core::protocol::JudgeStatus {
                enabled: true,
                can_run: true,
                blocked: None,
                retry_at: None,
                model: "haiku".into(),
                command: Some("claude".into()),
                pending: 4,
                queued: 4,
                rejected: 0,
                failed: 0,
                calls_today: 3,
                max_calls_per_day: 200,
                judged_today: 41,
                cost_today_usd: 0.0061,
                max_usd_per_day: None,
                last_error: None,
                quality: None,
            },
            metrics: None,
            omitted: 0,
        }
    }

    fn status_op() -> Op {
        Op::Status(StatusParams {
            metrics: false,
            limit: None,
        })
    }

    #[test]
    fn s10_status_of_a_ready_daemon_reads_as_a_report_and_succeeds() {
        let r = sample_status(DaemonState::Ready, None);
        let out = run(status_op(), |_| Ok(Reply::Status(Box::new(r.clone())))).unwrap();
        assert_eq!(out.code, 0);
        assert_eq!(out.stderr, "");
        let text = out.stdout;
        assert!(
            text.starts_with("Daemon: ready (pid 4242, profile default, "),
            "{text}"
        );
        assert!(text.contains("  endpoint:  sm-test-endpoint\n"), "{text}");
        assert!(text.contains(", ready after 2.5 s\n"), "{text}");
        assert!(
            text.contains("Model: embeddinggemma-2 at revision daa72c5, 768 dimensions, loaded\n"),
            "{text}"
        );
        assert!(
            text.contains("Index: matches the configuration; 40 chunks, 3 waiting for embedding\n"),
            "{text}"
        );
        assert!(
            text.contains("Database: /data/default/memory.db (65536 bytes)\n"),
            "{text}"
        );
        assert!(text.contains("Spool: 1 captures waiting\n"), "{text}");
        assert!(
            text.contains("Capture queue: 2 jobs queued or running\n"),
            "{text}"
        );
        assert!(
            text.contains("Judge: can run; model haiku through claude\n"),
            "{text}"
        );
        assert!(
            text.contains("  pending:    4 captures, 4 waiting for the judge\n"),
            "{text}"
        );
        assert!(
            text.contains(
                "  today:      3 of 200 calls, 41 judged, about 0.0061 US dollars (Claude Code's \
                 estimate)\n"
            ),
            "{text}"
        );
        assert!(text.contains("Current project: acme\n"), "{text}");
        assert!(text.contains("Projects: 1 with 12 memories\n"), "{text}");
        assert!(text.contains("  acme: 12 memories, 2 shared\n"), "{text}");
        assert!(
            text.contains("    by type:   decision 7, fact 5\n"),
            "{text}"
        );
        assert!(text.contains("    by status: active 12\n"), "{text}");
        assert!(
            text.contains("    git remote: github.com/acme/app\n"),
            "{text}"
        );
        assert!(!text.contains("error"), "{text}");
    }

    /// A judge that cannot run says why, until when, and its last error
    /// (SPEC section 16, The judge; B-165).
    #[test]
    fn s16_status_says_why_the_judge_cannot_run() {
        let mut r = sample_status(DaemonState::Ready, None);
        r.judge.can_run = false;
        r.judge.blocked = Some(JudgeBlocked::BackingOff);
        r.judge.retry_at = Some("2026-11-17T10:05:00Z".into());
        r.judge.last_error = Some("Claude Code exited with status 1".into());
        let text = render_status(&r);
        assert!(
            text.contains(
                "Judge: cannot run: a call failed, and it waits before the next, until \
                 2026-11-17T10:05:00Z; model haiku through claude\n"
            ),
            "{text}"
        );
        assert!(
            text.contains("  last error: Claude Code exited with status 1\n"),
            "{text}"
        );
        r.judge.blocked = Some(JudgeBlocked::Unavailable);
        r.judge.command = None;
        r.judge.retry_at = None;
        r.judge.last_error = None;
        let text = render_status(&r);
        assert!(
            text.contains(
                "Judge: cannot run: Claude Code is not available; model haiku, Claude Code not \
                 found\n"
            ),
            "{text}"
        );
        assert!(!text.contains("last error"), "{text}");
    }

    /// Today's cost shows against `capture.judge.max_usd_per_day` when the
    /// daemon reports it, and a judge stopped by it says so (B-193).
    #[test]
    fn s16_b193_status_shows_the_spending_cap() {
        let mut r = sample_status(DaemonState::Ready, None);
        r.judge.max_usd_per_day = Some(3.0);
        let text = render_status(&r);
        assert!(
            text.contains(
                "  today:      3 of 200 calls, 41 judged, about 0.0061 US dollars of 3.00 \
                 (Claude Code's estimate)\n"
            ),
            "{text}"
        );
        r.judge.can_run = false;
        r.judge.blocked = Some(JudgeBlocked::DailyBudget);
        r.judge.retry_at = Some("2026-11-19T00:00:00Z".into());
        let text = render_status(&r);
        assert!(
            text.contains(
                "Judge: cannot run: today's spending is used (capture.judge.max_usd_per_day), \
                 until 2026-11-19T00:00:00Z; model haiku through claude\n"
            ),
            "{text}"
        );
    }

    /// Rejected captures show as a count beside the pending ones, only when
    /// there are any (B-190).
    #[test]
    fn s16_b190_status_counts_rejected_captures() {
        let mut r = sample_status(DaemonState::Ready, None);
        let text = render_status(&r);
        assert!(!text.contains("rejected"), "{text}");
        r.judge.rejected = 7;
        let text = render_status(&r);
        assert!(
            text.contains(
                "  pending:    4 captures, 4 waiting for the judge, 7 rejected (kept 30 days)\n"
            ),
            "{text}"
        );
    }

    /// Captures marked failed judgement show as a count beside the pending
    /// ones, only when there are any (B-206).
    #[test]
    fn s16_b206_status_counts_captures_that_failed_judgement() {
        let mut r = sample_status(DaemonState::Ready, None);
        let text = render_status(&r);
        assert!(!text.contains("failed judgement"), "{text}");
        r.judge.queued = 2;
        r.judge.failed = 2;
        let text = render_status(&r);
        assert!(
            text.contains(
                "  pending:    4 captures, 2 waiting for the judge, 2 failed judgement (not \
                 served)\n"
            ),
            "{text}"
        );
    }

    /// The last 7 days of automatic capture show under the judge when the
    /// daemon reports them, and not from a daemon before B-200.
    #[test]
    fn s16_b200_status_shows_the_last_7_days_of_automatic_capture() {
        let mut r = sample_status(DaemonState::Ready, None);
        let text = render_status(&r);
        assert!(!text.contains("days:"), "{text}");
        r.judge.quality = Some(sm_core::journal::CaptureQuality {
            days: 7,
            judged: 120,
            kept: 80,
            rejected: 40,
            dropped_after_keep: 6,
            forgotten: 3,
            forgotten_pending: 4,
            calls: 15,
            cost_usd: 0.0336,
        });
        let text = render_status(&r);
        assert!(
            text.contains(
                "  today:      3 of 200 calls, 41 judged, about 0.0061 US dollars (Claude Code's \
                 estimate)\n  7 days:     120 judged: 80 kept, 40 rejected (6 of them kept, then \
                 dropped as duplicates); 3 captured memories removed with forget, and 4 pending \
                 captures; 15 judge calls, about 0.0336 US dollars (Claude Code's estimate)\n"
            ),
            "{text}"
        );
        r.judge.quality = Some(sm_core::journal::CaptureQuality {
            days: 7,
            ..Default::default()
        });
        let text = render_status(&r);
        assert!(
            text.contains(
                "  7 days:     0 judged: 0 kept, 0 rejected; 0 captured memories removed with \
                 forget; 0 judge calls, about 0.0000 US dollars (Claude Code's estimate)\n"
            ),
            "{text}"
        );
    }

    #[test]
    fn s10_status_of_a_failed_model_exits_1_with_the_error_and_next_lines() {
        let failure = CoreError::new(
            ErrorCode::ModelMissing,
            "model file onnx/model.onnx is missing",
        )
        .with_next_action("Run `spatial-memory setup` to download the model.");
        let r = sample_status(DaemonState::Failed, Some(failure.clone()));
        let out = run(status_op(), |_| Ok(Reply::Status(Box::new(r.clone())))).unwrap();
        assert_eq!(out.code, 1);
        assert_eq!(out.stderr, crate::cli::error_text(&failure));
        assert!(out.stdout.starts_with("Daemon: failed "), "{}", out.stdout);
        assert!(
            out.stdout
                .contains("  error:     model_missing: model file onnx/model.onnx is missing\n"),
            "{}",
            out.stdout
        );
        assert!(out.stdout.contains(", not loaded\n"), "{}", out.stdout);
        // Failed without a reason still gives a next action.
        let r = sample_status(DaemonState::Failed, None);
        let errors = reported_errors(&Reply::Status(Box::new(r)));
        assert_eq!(errors.len(), 1);
        assert!(errors[0].next_action.contains("daemon.log"), "{errors:?}");
    }

    /// B-138: the reply counts memories only for the projects it lists, so
    /// with projects left out the total is said to be theirs alone.
    #[test]
    fn s10_status_totals_only_the_projects_it_shows() {
        let mut r = sample_status(DaemonState::Ready, None);
        assert_eq!(r.projects.len(), 1);
        r.omitted = 2;
        let out = run(status_op(), |_| Ok(Reply::Status(Box::new(r.clone())))).unwrap();
        assert_eq!(out.code, 0);
        assert!(
            out.stdout
                .contains("Projects: 3; the 1 shown hold 12 memories\n"),
            "{}",
            out.stdout
        );
        assert!(
            out.stdout.contains("  (2 more projects not shown)\n"),
            "{}",
            out.stdout
        );
    }

    #[test]
    fn s10_status_with_an_index_mismatch_exits_1_and_says_to_reindex() {
        let mut r = sample_status(DaemonState::Ready, None);
        r.index.matches = false;
        r.index.differences = vec!["chunk_max_tokens 512 != 256".into()];
        let out = run(status_op(), |_| Ok(Reply::Status(Box::new(r.clone())))).unwrap();
        assert_eq!(out.code, 1);
        assert!(
            out.stderr.starts_with(
                "error: index_mismatch: the index does not match the configuration: \
                 chunk_max_tokens 512 != 256\nnext: Run `spatial-memory reindex`"
            ),
            "{}",
            out.stderr
        );
        assert!(
            out.stdout
                .contains("Index: does not match the configuration (chunk_max_tokens 512 != 256)"),
            "{}",
            out.stdout
        );
    }

    #[test]
    fn s10_status_while_loading_is_not_an_error() {
        let mut r = sample_status(DaemonState::Loading, None);
        r.projects.clear();
        r.omitted = 3;
        r.metrics = Some(vec![sm_core::journal::MetricSummary {
            metric: "hook_ms".into(),
            median: Some(4.0),
            p95: None,
            count: 9,
        }]);
        let out = run(status_op(), |_| Ok(Reply::Status(Box::new(r.clone())))).unwrap();
        assert_eq!((out.code, out.stderr.as_str()), (0, ""));
        assert!(out.stdout.starts_with("Daemon: loading "), "{}", out.stdout);
        // B-138: not "with 0 memories", which the three projects may hold.
        assert!(
            out.stdout.contains("Projects: 3, none shown\n"),
            "{}",
            out.stdout
        );
        assert!(
            out.stdout.contains("  (3 more projects not shown)\n"),
            "{}",
            out.stdout
        );
        assert!(
            out.stdout
                .contains("Metric hook_ms: median 4.000 p95 - (n=9)\n"),
            "{}",
            out.stdout
        );
    }

    #[test]
    fn s3_project_delete_without_confirm_prints_the_preview_and_succeeds() {
        let op = Op::ProjectDelete(ProjectDeleteParams {
            name: "acme".into(),
            confirm: false,
        });
        let out = run(op, |_| {
            Err(CoreError::new(
                ErrorCode::ConfirmRequired,
                "project acme holds 12 memories",
            ))
        })
        .unwrap();
        assert_eq!(out.code, 0);
        assert!(
            out.stdout
                .contains("Nothing was deleted. project acme holds 12 memories")
        );
        assert!(out.stdout.contains("--confirm"));
    }

    #[test]
    fn s3_other_errors_are_returned_unchanged() {
        let e = CoreError::new(ErrorCode::Storage, "disk full");
        let got = run(Op::Reindex(ReindexParams {}), |_| Err(e.clone())).unwrap_err();
        assert_eq!(got, e);
        let confirmed = Op::ProjectDelete(ProjectDeleteParams {
            name: "acme".into(),
            confirm: true,
        });
        let refused = CoreError::new(ErrorCode::ConfirmRequired, "x");
        assert_eq!(
            run(confirmed, |_| Err(refused.clone())).unwrap_err(),
            refused
        );
    }

    #[test]
    fn s3_reports_render_their_counts() {
        let r = Reply::Import(ImportReport {
            source: "/tmp/x.jsonl".into(),
            read: 3,
            inserted: 2,
            skipped_existing: 1,
            redacted: 0,
            projects_created: vec!["acme".into()],
            issues: vec![ImportIssue {
                index: 2,
                id: Some("abc".into()),
                message: "bad type".into(),
            }],
            pending_embeddings: 4,
        });
        let text = render(&r);
        assert!(text.contains("inserted:            2"));
        assert!(text.contains("record 2 (abc): bad type"));
        assert!(text.contains("projects created:    acme"));
        let p = Reply::Project(ProjectReply {
            name: "new".into(),
            renamed_from: Some("old".into()),
            deleted: false,
            memories: 5,
        });
        assert_eq!(
            render(&p),
            "Renamed project old to new; 5 memories moved.\n"
        );
    }
}
