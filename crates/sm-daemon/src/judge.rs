//! The capture judge (SPEC section 16, The judge; B-168; RC4, RC6 and RC7
//! in `docs/v3/CAPTURE-RCA.md`).
//!
//! The capture rules are a cheap first filter. Every capture they pass is
//! stored `pending` (B-157), and only the judge decides what becomes of
//! it: a kept capture becomes `active` with the sentences of its own text
//! that the judge quoted as its content and the judge's type, and a
//! rejected one is deleted. Nothing unjudged is ever served.
//!
//! - **One worker, in the background.** One thread, `sm-judge`, makes at
//!   most one call at a time, never on a request's path, so no hook waits
//!   for it. It waits for the model (the memory is chunked and embedded),
//!   for a backoff to pass, for the daily cap to reset, and for a batch:
//!   it calls as soon as [`JUDGE_BATCH_MAX`] captures wait, or once the
//!   oldest has waited `DaemonOptions::judge_wait`.
//! - **Redaction.** A capture's text and context were redacted when it was
//!   captured; both are redacted again before the call, and so is each
//!   memory built from the quotes before it is stored (SPEC section 9).
//! - **Checks.** An answer must judge each candidate exactly once, with a
//!   reason of the schema, `keep` exactly when the reason is `knowledge`,
//!   and a type of the six and at least one quote for a kept one. An
//!   answer that fails any check fails the whole batch.
//! - **Quotes, not rewrites** (B-173). The judge never writes a memory: it
//!   quotes the candidate's sentences worth keeping, and the memory is made
//!   of the sentences of the candidate's own text that the quotes found
//!   (`quote.rs`). A quote not in the text is dropped. A kept capture
//!   none of whose quotes is found, or whose sentences come to more than
//!   [`MAX_REWRITE_CHARS`] characters, is not settled: it stays pending,
//!   counts an attempt, and is journalled with the verdict `unquoted` or
//!   `too_long`.
//! - **Failures.** Any failure leaves every capture of the batch pending,
//!   is counted against the daily cap and in the journal, and makes the
//!   judge wait: until the limit's reset time when Claude Code gives one,
//!   otherwise `judge_backoff`, doubled after each failure up to
//!   `judge_backoff_max`. A bad answer also counts against each capture
//!   of the batch; one that has had [`sm_store::JUDGE_MAX_ATTEMPTS`] bad
//!   answers waits only to be pruned.
//! - **Near duplicates** (step 12). Each kept memory is embedded, and one
//!   whose cosine with an active memory of its project, or with a memory
//!   kept earlier in the batch for the same project, is at least
//!   [`NEAR_DUPLICATE_COSINE`] is deleted instead of kept,
//!   and journalled as `capture_dropped_total` with `stage` `judge`. When
//!   the embedder fails, the memory is kept without the check, its chunks
//!   left for the embedding queue.
//! - **Metrics.** `capture_judged_total` per verdict applied,
//!   `capture_judge_cost_usd` per answered call,
//!   `capture_judge_failures_total` per failed call and `capture_pending`
//!   after each call.

mod claude;
mod quote;
#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, HashMap};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sm_core::error::{CoreError, Result as CoreResult};
use sm_core::ids::{MemoryId, ProjectId};
use sm_core::journal::metrics;
use sm_core::protocol::{JudgeBlocked, JudgeStatus};
use sm_core::time::Timestamp;
use sm_core::traits::SecretScanner;
use sm_core::types::{MemoryType, derive_title};
use sm_store::{JUDGE_BATCH_MAX, JudgeItem, NewChunk, Settle, Settled};

pub use claude::{
    CALL_TIMEOUT, ClaudeJudge, PASS_ENV, PROMPT, SCHEMA, find_claude_in, parse_output,
};

use crate::capture::{cosine, embed_docs};
use crate::state::{Ready, Shared, lock, panic_text};
use crate::tools::remember::pending_chunks;

/// The cosine at or above which a kept memory is a near duplicate of an
/// active memory of its project, or of a memory kept before it in the same
/// batch, and is deleted instead of kept (step 12, B-169). Chosen from the
/// labelled export: the highest two-place cosine that still catches both of
/// its labelled duplicates (their cosines are 0.9075 and 0.9393). Simulated
/// in capture order with a perfect judge, it costs 4 of the 34 "worth
/// keeping" items, which the labellers did not call duplicates. Provisional:
/// two labelled duplicates are thin evidence, and the measure was of the
/// captures' whole text, not of the sentences the judge quotes (B-173).
/// Its metric is `capture_dropped_total`
/// with `stage` `judge` and the cosine.
pub(crate) const NEAR_DUPLICATE_COSINE: f32 = 0.90;

/// Characters of content and context in one batch after its first capture.
pub(crate) const JUDGE_BATCH_CHARS: usize = 100_000;

/// The longest memory kept, in characters: the sentences the quotes found
/// (B-173). The prompt asks for one to six sentences.
pub const MAX_REWRITE_CHARS: usize = 4_000;

/// The reasons of the schema; `knowledge` is the only reason to keep.
pub const REASONS: &[&str] = &[
    "knowledge",
    "narration",
    "status",
    "tool_output",
    "transient",
    "restatement",
    "fragment",
    "duplicate",
    "other",
];

/// The request id of the judge's journal lines and log lines.
const REQUEST_ID: &str = "judge";

/// The longest the worker sleeps before it looks again.
const POLL: Duration = Duration::from_secs(60);

/// The pause after an error of the store or a panic in the worker.
const ERROR_PAUSE: Duration = Duration::from_secs(60);

const MICROS_PER_DAY: i64 = 86_400 * 1_000_000;

/// One capture as the judge sees it. `id` is its place in the batch
/// (`c01` to `c25`), never the memory id.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct JudgeCandidate {
    pub id: String,
    pub text: String,
    pub context: String,
}

/// One verdict, as the schema gives it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JudgeVerdict {
    pub id: String,
    pub keep: bool,
    pub reason: String,
    #[serde(rename = "type")]
    pub memory_type: String,
    /// The candidate's sentences worth keeping, copied from its text; empty
    /// for a rejected candidate (B-173).
    pub quotes: Vec<String>,
}

/// An answered call.
#[derive(Debug, Clone, PartialEq)]
pub struct JudgeAnswer {
    pub verdicts: Vec<JudgeVerdict>,
    /// The call's cost in US dollars, as Claude Code reports it.
    pub cost_usd: f64,
    pub duration_ms: u64,
    /// The model that answered, when the output named it.
    pub model: Option<String>,
}

/// Why a call failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JudgeFailureKind {
    /// Claude Code could not be started, or is not signed in.
    Unavailable,
    /// A rate limit or the plan's usage limit was reached.
    RateLimited,
    /// No answer within the call's timeout.
    Timeout,
    /// Claude Code reported any other error, or ended without an answer.
    CallFailed,
    /// The answer broke the schema or did not judge each capture once.
    BadAnswer,
    /// The daemon stopped during the call.
    Stopped,
}

impl JudgeFailureKind {
    pub fn as_str(self) -> &'static str {
        match self {
            JudgeFailureKind::Unavailable => "unavailable",
            JudgeFailureKind::RateLimited => "rate_limited",
            JudgeFailureKind::Timeout => "timeout",
            JudgeFailureKind::CallFailed => "call_failed",
            JudgeFailureKind::BadAnswer => "bad_answer",
            JudgeFailureKind::Stopped => "stopped",
        }
    }
}

/// A failed call. `message` never holds memory text.
#[derive(Debug, Clone, PartialEq)]
pub struct JudgeFailure {
    pub kind: JudgeFailureKind,
    pub message: String,
    /// When to try again, when the failure says.
    pub retry_at: Option<Timestamp>,
    /// What the call cost, when it got that far.
    pub cost_usd: f64,
}

/// What judges a batch of captures. The daemon's is [`ClaudeJudge`];
/// tests give a scripted one.
pub trait Judge: Send + Sync + std::fmt::Debug {
    /// The model setting, for `status` and the provenance.
    fn model(&self) -> &str;
    /// The executable a call would run now; `None` when there is none.
    fn command(&self) -> Option<String>;
    /// Judges one batch. Returns early with [`JudgeFailureKind::Stopped`]
    /// once `stop` is set.
    fn judge(
        &self,
        batch: &[JudgeCandidate],
        stop: &AtomicBool,
    ) -> Result<JudgeAnswer, JudgeFailure>;
}

/// The judge worker's state in this daemon.
#[derive(Debug, Clone, Default)]
pub(crate) struct JudgeRuntime {
    /// No call before this.
    pub retry_at: Option<Timestamp>,
    /// Calls failed since the last answer.
    pub failures: u32,
    pub last_error: Option<String>,
    pub last_kind: Option<JudgeFailureKind>,
    /// A call is running.
    pub busy: bool,
    /// The UTC day whose cap was logged as reached.
    pub capped_day: Option<String>,
}

/// One checked verdict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Checked {
    pub keep: bool,
    pub reason: String,
    pub memory_type: MemoryType,
    /// The sentences of the candidate's text that the quotes found; empty
    /// for a rejected capture and for a kept one that is not settled.
    pub memory: String,
    /// Quotes given and quotes not found in the text.
    pub quotes: usize,
    pub dropped: usize,
    /// Why a kept capture is not settled: `unquoted` (no quote found) or
    /// `too_long` (the sentences found exceed [`MAX_REWRITE_CHARS`]).
    pub unsettled: Option<&'static str>,
}

/// Checks an answer against its batch and returns one verdict per
/// candidate, in the batch's order. The error never holds memory text, nor
/// any value the answer wrote that is not one of the batch's ids, an id of
/// the batch's shape or a known reason: the model writes the answer after
/// reading the captures, and the error reaches `status` as the judge's last
/// error (B-181).
pub(crate) fn check(
    candidates: &[JudgeCandidate],
    verdicts: &[JudgeVerdict],
) -> Result<Vec<Checked>, String> {
    let mut slots: Vec<Option<Checked>> = vec![None; candidates.len()];
    for v in verdicts {
        let Some(i) = candidates.iter().position(|c| c.id == v.id) else {
            return Err(if id_shaped(&v.id) {
                format!("a verdict names {}, which is not in the batch", v.id)
            } else {
                "a verdict names an id that is not in the batch".to_owned()
            });
        };
        if slots[i].is_some() {
            return Err(format!("{} was judged twice", v.id));
        }
        if !REASONS.contains(&v.reason.as_str()) {
            return Err(format!("the verdict on {} has an unknown reason", v.id));
        }
        if v.keep != (v.reason == "knowledge") {
            return Err(format!(
                "the verdict on {} has keep {} with the reason {}",
                v.id, v.keep, v.reason
            ));
        }
        let checked = if v.keep {
            let memory_type = v
                .memory_type
                .parse::<MemoryType>()
                .map_err(|_| format!("the verdict on {} has an unknown type", v.id))?;
            if v.quotes.iter().all(|q| q.trim().is_empty()) {
                return Err(format!("{} was kept without a quote", v.id));
            }
            let q = quote::quoted(&candidates[i].text, &v.quotes);
            let unsettled = if q.memory.is_empty() {
                Some("unquoted")
            } else if q.memory.chars().count() > MAX_REWRITE_CHARS {
                Some("too_long")
            } else {
                None
            };
            Checked {
                keep: true,
                reason: v.reason.clone(),
                memory_type,
                memory: if unsettled.is_none() {
                    q.memory
                } else {
                    String::new()
                },
                quotes: q.quotes,
                dropped: q.dropped,
                unsettled,
            }
        } else {
            Checked {
                keep: false,
                reason: v.reason.clone(),
                memory_type: MemoryType::Note,
                memory: String::new(),
                quotes: 0,
                dropped: 0,
                unsettled: None,
            }
        };
        slots[i] = Some(checked);
    }
    slots
        .into_iter()
        .zip(candidates)
        .map(|(s, c)| s.ok_or_else(|| format!("{} was not judged", c.id)))
        .collect()
}

/// Whether `id` has the shape of a batch id: `c` and one to four digits.
/// Such a value cannot carry memory text, so an error may name it (B-181).
fn id_shaped(id: &str) -> bool {
    id.strip_prefix('c')
        .is_some_and(|d| (1..=4).contains(&d.len()) && d.bytes().all(|b| b.is_ascii_digit()))
}

/// The `sm-judge` thread: runs until the daemon stops.
pub(crate) fn run(sh: &Arc<Shared>, judge: &dyn Judge) {
    sh.log.line(format!(
        "judge: on, model {}, at most {} calls a day",
        judge.model(),
        sh.settings.capture.judge.max_calls_per_day
    ));
    while !sh.stopping() {
        let next = match catch_unwind(AssertUnwindSafe(|| step(sh, judge))) {
            Ok(Ok(next)) => next,
            Ok(Err(e)) => {
                sh.log.line(format!(
                    "judge: {}: {}; trying again in {} s",
                    e.code.as_str(),
                    e.message,
                    ERROR_PAUSE.as_secs()
                ));
                Some(ERROR_PAUSE)
            }
            Err(p) => {
                lock(&sh.judge_state).busy = false;
                sh.log.line(format!(
                    "judge: panicked: {}; trying again in {} s",
                    panic_text(p.as_ref()),
                    ERROR_PAUSE.as_secs()
                ));
                Some(ERROR_PAUSE)
            }
        };
        if let Some(d) = next {
            sh.judge_wake.wait(d);
        }
    }
}

fn micros(d: Duration) -> i64 {
    i64::try_from(d.as_micros()).unwrap_or(i64::MAX)
}

/// The wait from `now` to `at`, at most [`POLL`].
fn until(now: Timestamp, at: Timestamp) -> Duration {
    let us = at.as_micros().saturating_sub(now.as_micros()).max(0);
    Duration::from_micros(u64::try_from(us).unwrap_or(0))
        .min(POLL)
        .max(Duration::from_millis(10))
}

/// The start of the UTC day after `now`.
fn next_day(now: Timestamp) -> Timestamp {
    Timestamp::from_micros((now.as_micros().div_euclid(MICROS_PER_DAY) + 1) * MICROS_PER_DAY)
}

/// One pass of the worker: `None` to go on at once, otherwise how long to
/// wait (a wake-up cuts it short).
fn step(sh: &Shared, judge: &dyn Judge) -> CoreResult<Option<Duration>> {
    let Some(ready) = sh.ready_opt() else {
        return Ok(Some(if sh.is_loading() {
            Duration::from_millis(200)
        } else {
            POLL
        }));
    };
    let now = Timestamp::now();
    if let Some(at) = lock(&sh.judge_state).retry_at.filter(|at| *at > now) {
        return Ok(Some(until(now, at)));
    }
    if judge.command().is_none() {
        let mut rt = lock(&sh.judge_state);
        if rt.last_kind != Some(JudgeFailureKind::Unavailable) {
            sh.log.line(
                "judge: Claude Code was not found; captures stay pending until it is installed \
                 or named in capture.judge.command",
            );
        }
        rt.last_kind = Some(JudgeFailureKind::Unavailable);
        rt.last_error = Some(
            "Claude Code was not found: install it, or name its executable in \
             capture.judge.command"
                .to_owned(),
        );
        rt.retry_at = Some(now.add_micros(micros(sh.opts.judge_backoff)));
        return Ok(Some(until(
            now,
            now.add_micros(micros(sh.opts.judge_backoff)),
        )));
    }
    let day = now.date_string();
    let cap = sh.settings.capture.judge.max_calls_per_day;
    if sh.store.judge_usage(&day)?.calls >= cap {
        let mut rt = lock(&sh.judge_state);
        if rt.capped_day.as_deref() != Some(day.as_str()) {
            sh.log.line(format!(
                "judge: today's {cap} calls (capture.judge.max_calls_per_day) are used; \
                 captures stay pending until the next UTC day"
            ));
            rt.capped_day = Some(day);
        }
        return Ok(Some(until(now, next_day(now))));
    }
    let stats = sh.store.judge_queue_stats()?;
    if stats.queued == 0 {
        return Ok(Some(POLL));
    }
    if stats.queued < JUDGE_BATCH_MAX as u64
        && let Some(oldest) = stats.oldest
    {
        let due = oldest.add_micros(micros(sh.opts.judge_wait));
        if due > now {
            return Ok(Some(until(now, due)));
        }
    }
    let items = sh.store.judge_queue(JUDGE_BATCH_MAX, JUDGE_BATCH_CHARS)?;
    if items.is_empty() {
        return Ok(Some(POLL));
    }
    call(sh, judge, &ready, &day, &items)?;
    journal_pending(sh, REQUEST_ID);
    Ok(None)
}

/// Clears `busy` however the call ends.
struct Busy<'a>(&'a Shared);

impl<'a> Busy<'a> {
    fn start(sh: &'a Shared) -> Busy<'a> {
        lock(&sh.judge_state).busy = true;
        Busy(sh)
    }
}

impl Drop for Busy<'_> {
    fn drop(&mut self) {
        lock(&self.0.judge_state).busy = false;
    }
}

/// The candidates of a batch: text and context redacted again.
pub(crate) fn candidates_of(
    scanner: &dyn SecretScanner,
    items: &[JudgeItem],
) -> Vec<JudgeCandidate> {
    items
        .iter()
        .enumerate()
        .map(|(i, it)| JudgeCandidate {
            id: format!("c{:02}", i + 1),
            text: scanner.redact(&it.content).text,
            context: scanner.redact(&it.context).text,
        })
        .collect()
}

fn call(
    sh: &Shared,
    judge: &dyn Judge,
    ready: &Ready,
    day: &str,
    items: &[JudgeItem],
) -> CoreResult<()> {
    let candidates = candidates_of(&sh.scanner, items);
    let answer = {
        let _busy = Busy::start(sh);
        judge.judge(&candidates, &sh.stop)
    };
    let checked = answer.and_then(|a| match check(&candidates, &a.verdicts) {
        Ok(c) => Ok((a, c)),
        Err(message) => Err(JudgeFailure {
            kind: JudgeFailureKind::BadAnswer,
            message,
            retry_at: None,
            cost_usd: a.cost_usd,
        }),
    });
    match checked {
        Ok((answer, checked)) => {
            sh.store
                .record_judge_call(day, items.len() as u64, false, answer.cost_usd)?;
            let mut labels = BTreeMap::new();
            labels.insert("candidates".to_owned(), Value::from(items.len()));
            labels.insert("duration_ms".to_owned(), Value::from(answer.duration_ms));
            sh.journal.record(
                REQUEST_ID,
                metrics::CAPTURE_JUDGE_COST_USD,
                answer.cost_usd,
                labels,
            );
            {
                let mut rt = lock(&sh.judge_state);
                if rt.failures > 0 {
                    sh.log.line("judge: answering again");
                }
                rt.failures = 0;
                rt.retry_at = None;
                rt.last_error = None;
                rt.last_kind = None;
            }
            let model = answer
                .model
                .clone()
                .unwrap_or_else(|| judge.model().to_owned());
            settle(sh, ready, items, &checked, &model)
        }
        Err(f) => failed(sh, day, items, &f),
    }
}

fn failed(sh: &Shared, day: &str, items: &[JudgeItem], f: &JudgeFailure) -> CoreResult<()> {
    sh.store.record_judge_call(day, 0, true, f.cost_usd)?;
    if f.kind == JudgeFailureKind::Stopped {
        return Ok(());
    }
    if f.kind == JudgeFailureKind::BadAnswer {
        let ids: Vec<MemoryId> = items.iter().map(|i| i.id).collect();
        sh.store.judge_failed(&ids)?;
    }
    let mut labels = BTreeMap::new();
    labels.insert("kind".to_owned(), Value::from(f.kind.as_str()));
    labels.insert("candidates".to_owned(), Value::from(items.len()));
    sh.journal.record(
        REQUEST_ID,
        metrics::CAPTURE_JUDGE_FAILURES_TOTAL,
        1.0,
        labels,
    );
    let now = Timestamp::now();
    let mut rt = lock(&sh.judge_state);
    rt.failures = rt.failures.saturating_add(1);
    let doubled = sh
        .opts
        .judge_backoff
        .saturating_mul(1u32 << rt.failures.saturating_sub(1).min(16))
        .min(sh.opts.judge_backoff_max);
    let retry = f
        .retry_at
        .filter(|at| *at > now)
        .unwrap_or_else(|| now.add_micros(micros(doubled)));
    rt.retry_at = Some(retry);
    rt.last_error = Some(f.message.clone());
    rt.last_kind = Some(f.kind);
    sh.log.line(format!(
        "judge: a call about {} captures failed ({}): {}; they stay pending; next call at {}",
        items.len(),
        f.kind.as_str(),
        f.message,
        retry.to_rfc3339()
    ));
    Ok(())
}

/// One kept memory, built from the judge's quotes (B-173), on its way to
/// the store.
struct Rewrite {
    index: usize,
    project: ProjectId,
    content: String,
    title: Option<String>,
    memory_type: MemoryType,
    chunks: Vec<NewChunk>,
}

fn guarded<T>(f: impl FnOnce() -> CoreResult<T>) -> CoreResult<T> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(r) => r,
        Err(p) => Err(CoreError::internal(format!(
            "panicked: {}",
            panic_text(p.as_ref())
        ))),
    }
}

/// The closest cosine of each kept memory to an active memory of its
/// project or to a memory kept before it in the batch for the same
/// project. All
/// `None`, with the embeddings cleared for the queue, when the embedder or
/// the lookup fails.
fn closest(sh: &Shared, ready: &Ready, rewrites: &mut [Rewrite]) -> Vec<Option<f32>> {
    let looked_up = guarded(|| {
        let means = {
            let mut docs: Vec<(Option<&str>, &mut [NewChunk])> = rewrites
                .iter_mut()
                .map(|r| (r.title.as_deref(), r.chunks.as_mut_slice()))
                .collect();
            embed_docs(sh, ready, &mut docs)?
        };
        let mut by_project: BTreeMap<ProjectId, Vec<usize>> = BTreeMap::new();
        for (i, r) in rewrites.iter().enumerate() {
            by_project.entry(r.project).or_default().push(i);
        }
        let mut found: Vec<Option<f32>> = vec![None; rewrites.len()];
        for (project, indexes) in by_project {
            let vectors: Vec<Vec<f32>> = indexes.iter().map(|&i| means[i].clone()).collect();
            let near = sh.store.active_nearest(project, &vectors)?;
            for (&i, n) in indexes.iter().zip(near) {
                found[i] = n.map(|n| n.cosine);
            }
        }
        Ok((means, found))
    });
    let (means, found) = match looked_up {
        Ok(v) => v,
        Err(e) => {
            sh.log.line(format!(
                "judge: cannot check kept captures for near duplicates: {}: {}; kept without \
                 the check, embedded later by the queue",
                e.code.as_str(),
                e.message
            ));
            for r in rewrites.iter_mut() {
                for c in &mut r.chunks {
                    c.embedding = None;
                }
            }
            return vec![None; rewrites.len()];
        }
    };
    let mut out = Vec::with_capacity(rewrites.len());
    let mut kept: Vec<(ProjectId, &[f32])> = Vec::new();
    for (i, r) in rewrites.iter().enumerate() {
        let c = found[i]
            .into_iter()
            .chain(
                kept.iter()
                    .filter(|(p, _)| *p == r.project)
                    .map(|(_, m)| cosine(m, &means[i])),
            )
            .reduce(f32::max);
        if !c.is_some_and(|c| c >= NEAR_DUPLICATE_COSINE) {
            kept.push((r.project, &means[i]));
        }
        out.push(c);
    }
    out
}

/// Applies a checked answer: rejected captures are deleted, kept ones made
/// active with the sentences their quotes found unless those are a near
/// duplicate; a kept one whose quotes found nothing stays pending.
fn settle(
    sh: &Shared,
    ready: &Ready,
    items: &[JudgeItem],
    checked: &[Checked],
    model: &str,
) -> CoreResult<()> {
    let cap = Some(sh.settings.capture.max_index_tokens as usize);
    let mut plan: Vec<Option<Settle>> = vec![None; items.len()];
    let mut near_drops: Vec<(usize, f32)> = Vec::new();
    let mut rewrites: Vec<Rewrite> = Vec::new();
    let mut unchunked: Vec<MemoryId> = Vec::new();
    let mut unsettled: Vec<usize> = Vec::new();
    for (i, (item, c)) in items.iter().zip(checked).enumerate() {
        if !c.keep {
            plan[i] = Some(Settle::Delete { id: item.id });
            continue;
        }
        if let Some(why) = c.unsettled {
            sh.log.line(format!(
                "judge: capture {} was kept, but {}; it stays pending",
                item.id,
                if why == "too_long" {
                    format!("the sentences its quotes found exceed {MAX_REWRITE_CHARS} characters")
                } else {
                    format!("none of its {} quotes is in its text", c.quotes)
                }
            ));
            unsettled.push(i);
            continue;
        }
        let content = sh.scanner.redact(&c.memory).text;
        let title = derive_title(&content);
        match guarded(|| pending_chunks(sh, ready, &content, cap)) {
            Ok((chunks, _)) => rewrites.push(Rewrite {
                index: i,
                project: item.project,
                content,
                title,
                memory_type: c.memory_type,
                chunks,
            }),
            Err(e) => {
                sh.log.line(format!(
                    "judge: capture {}: its rewrite cannot be chunked: {}: {}; it stays pending",
                    item.id,
                    e.code.as_str(),
                    e.message
                ));
                unchunked.push(item.id);
            }
        }
    }
    let near = closest(sh, ready, &mut rewrites);
    for (r, c) in rewrites.into_iter().zip(near) {
        let id = items[r.index].id;
        plan[r.index] = Some(match c {
            Some(c) if c >= NEAR_DUPLICATE_COSINE => {
                near_drops.push((r.index, c));
                Settle::Delete { id }
            }
            _ => Settle::Keep {
                id,
                content: r.content,
                title: r.title,
                memory_type: r.memory_type,
                chunks: r.chunks,
                judge_model: model.to_owned(),
                reason: "knowledge".to_owned(),
            },
        });
    }
    let (order, settles): (Vec<usize>, Vec<Settle>) = plan
        .into_iter()
        .enumerate()
        .filter_map(|(i, s)| s.map(|s| (i, s)))
        .unzip();
    let settled = sh.store.settle_captures(settles)?;
    sh.after_write();
    let mut failed_ids = unchunked;
    failed_ids.extend(unsettled.iter().map(|&i| items[i].id));
    if !failed_ids.is_empty() {
        sh.store.judge_failed(&failed_ids)?;
    }
    let mut names: HashMap<ProjectId, String> = HashMap::new();
    let mut name_of = |project: ProjectId| match names.get(&project) {
        Some(n) => n.clone(),
        None => {
            let n = sh
                .store
                .project(project)
                .map(|p| p.name)
                .unwrap_or_else(|_| project.to_string());
            names.insert(project, n.clone());
            n
        }
    };
    let judged_labels = |name: &str, c: &Checked, verdict: &str| {
        let mut labels = BTreeMap::new();
        labels.insert("project".to_owned(), Value::from(name));
        labels.insert("verdict".to_owned(), Value::from(verdict));
        labels.insert("reason".to_owned(), Value::from(c.reason.as_str()));
        if c.keep {
            labels.insert("type".to_owned(), Value::from(c.memory_type.as_str()));
            labels.insert("quotes".to_owned(), Value::from(c.quotes));
            labels.insert("quotes_dropped".to_owned(), Value::from(c.dropped));
        }
        labels
    };
    for &i in &unsettled {
        let c = &checked[i];
        let name = name_of(items[i].project);
        let labels = judged_labels(&name, c, c.unsettled.unwrap_or("unquoted"));
        sh.journal
            .record(REQUEST_ID, metrics::CAPTURE_JUDGED_TOTAL, 1.0, labels);
    }
    for (i, outcome) in order.into_iter().zip(settled) {
        let name = name_of(items[i].project);
        let c = &checked[i];
        let labels = judged_labels(&name, c, if c.keep { "keep" } else { "reject" });
        sh.journal
            .record(REQUEST_ID, metrics::CAPTURE_JUDGED_TOTAL, 1.0, labels);
        let dropped = near_drops
            .iter()
            .find(|(j, _)| *j == i)
            .map(|(_, c)| ("near_duplicate", Some(*c)))
            .or(match outcome {
                Settled::Duplicate { .. } => Some(("duplicate", None)),
                _ => None,
            });
        if let Some((reason, cosine)) = dropped {
            let mut labels = BTreeMap::new();
            labels.insert("project".to_owned(), Value::from(name.as_str()));
            labels.insert("stage".to_owned(), Value::from("judge"));
            labels.insert("reason".to_owned(), Value::from(reason));
            if let Some(c) = cosine {
                labels.insert("cosine".to_owned(), Value::from(f64::from(c)));
            }
            sh.journal
                .record(REQUEST_ID, metrics::CAPTURE_DROPPED_TOTAL, 1.0, labels);
        }
    }
    Ok(())
}

/// One `capture_pending` line: the pending count, with how many the judge
/// can still take as the label `queued`.
pub(crate) fn journal_pending(sh: &Shared, request_id: &str) {
    match sh.store.judge_queue_stats() {
        Ok(stats) => {
            let mut labels = BTreeMap::new();
            labels.insert("queued".to_owned(), Value::from(stats.queued));
            sh.journal.record(
                request_id,
                metrics::CAPTURE_PENDING,
                stats.pending as f64,
                labels,
            );
        }
        Err(e) => sh.log.line(format!(
            "{request_id}: cannot count pending captures: {}: {}",
            e.code.as_str(),
            e.message
        )),
    }
}

/// Whether a judge call is running.
pub(crate) fn busy(sh: &Shared) -> bool {
    lock(&sh.judge_state).busy
}

/// The judge's part of `status` (B-165).
pub(crate) fn status(sh: &Shared) -> CoreResult<JudgeStatus> {
    let settings = &sh.settings.capture;
    let now = Timestamp::now();
    let usage = sh.store.judge_usage(&now.date_string())?;
    let stats = sh.store.judge_queue_stats()?;
    let command = sh.opts.judge.as_ref().and_then(|j| j.command());
    let rt = lock(&sh.judge_state).clone();
    let waiting = rt.retry_at.filter(|at| *at > now);
    let cap = settings.judge.max_calls_per_day;
    let blocked = if !settings.judge.enabled {
        Some(JudgeBlocked::Disabled)
    } else if !settings.enabled {
        Some(JudgeBlocked::CaptureDisabled)
    } else if sh.opts.judge.is_none() {
        Some(JudgeBlocked::NoJudge)
    } else if command.is_none()
        || (waiting.is_some() && rt.last_kind == Some(JudgeFailureKind::Unavailable))
    {
        Some(JudgeBlocked::Unavailable)
    } else if usage.calls >= cap {
        Some(JudgeBlocked::DailyCap)
    } else if waiting.is_some() {
        Some(JudgeBlocked::BackingOff)
    } else {
        None
    };
    let retry_at = match blocked {
        Some(JudgeBlocked::DailyCap) => Some(next_day(now)),
        Some(JudgeBlocked::BackingOff | JudgeBlocked::Unavailable) => waiting,
        _ => None,
    };
    let last_error = match blocked {
        Some(JudgeBlocked::Unavailable) if command.is_none() => Some(
            "Claude Code was not found: install it, or name its executable in \
             capture.judge.command"
                .to_owned(),
        ),
        _ => rt.last_error,
    };
    Ok(JudgeStatus {
        enabled: settings.judge.enabled,
        can_run: blocked.is_none(),
        blocked,
        retry_at: retry_at.map(Timestamp::to_rfc3339),
        model: settings.judge.model.clone(),
        command,
        pending: stats.pending,
        queued: stats.queued,
        calls_today: usage.calls,
        max_calls_per_day: cap,
        judged_today: usage.judged,
        cost_today_usd: usage.cost_usd,
        last_error,
    })
}
