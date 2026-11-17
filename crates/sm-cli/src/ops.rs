//! The command-line operations: `import`, `export`, `migrate`, `reindex`,
//! `project rename` and `project delete` (SPEC section 3, command table).
//! Each sends one op through the waiting client and prints the report.

use std::fmt::Write as _;

use sm_core::error::{CoreError, ErrorCode};
use sm_core::eval::{EvalReport, ScoreDistribution};
use sm_core::protocol::{ExportReport, ImportReport, Op, ProjectReply, ReindexReport, Reply};

/// What an operation printed and how it ended.
#[derive(Debug, Clone, PartialEq)]
pub struct OpOutcome {
    /// Text for stdout.
    pub stdout: String,
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
            code: exit_code(&reply),
        }),
        Err(e) if preview && e.code == ErrorCode::ConfirmRequired => Ok(OpOutcome {
            stdout: format!(
                "Nothing was deleted. {}\nRun again with --confirm to delete.\n",
                e.message
            ),
            code: 0,
        }),
        Err(e) => Err(e),
    }
}

/// The exit code of a reply: `doctor --eval` fails on a regression against
/// the baseline; everything else that arrived succeeded.
pub fn exit_code(reply: &Reply) -> i32 {
    match reply {
        Reply::Eval(r) if !r.regressions.is_empty() => 1,
        _ => 0,
    }
}

/// The text form of a reply.
pub fn render(reply: &Reply) -> String {
    match reply {
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
    use sm_core::protocol::{ImportIssue, ProjectDeleteParams, ReindexParams};

    use super::*;

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
