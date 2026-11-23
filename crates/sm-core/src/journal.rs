//! The metrics journal (SPEC section 6, Metrics journal).
//!
//! Version 2's design is kept (`spatial_memory/core/metrics_journal.py` on
//! `feature/v2-implementations`): always on, one JSON line per event,
//! per-process files named `<YYYY-MM-DD>-<pid>-<start epoch>.jsonl` in the
//! profile's `metrics` folder, pruned after 7 days, silent on write errors.
//! Version 3 enforces the 4,096-byte line limit that version 2 only assumed,
//! and writes each line with one `write` call (spike decision 3).
//!
//! The hook client and the daemon both write here, each to its own file.

use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use crate::time::Timestamp;

/// Lines are shorter than this, newline included.
pub const MAX_LINE_BYTES: usize = 4096;
/// Label string values are cut to this many characters (version 2).
pub const MAX_LABEL_CHARS: usize = 64;
/// Files older than this many days are removed by [`prune`].
pub const RETENTION_DAYS: i64 = 7;
/// Window for the medians and 95th percentiles `status` returns.
pub const SUMMARY_WINDOW_DAYS: i64 = 7;
/// `request_id` for events outside a request (version 2).
pub const STARTUP_REQUEST_ID: &str = "startup";

/// Metric names. Version 2's five keep their names; the others are
/// version 3's, one per unproven default (SPEC sections 6 and 16).
pub mod metrics {
    pub const CAPTURE_SIZE_TOKENS: &str = "capture_size_tokens";
    pub const CAPTURE_CAP_HIT_TOTAL: &str = "capture_cap_hit_total";
    pub const RECALL_RESULT_COUNT: &str = "recall_result_count";
    pub const REINFORCED_MEMORY_RANK: &str = "reinforced_memory_rank";
    pub const EMBEDDER_COLD_START_SECONDS: &str = "embedder_cold_start_seconds";
    pub const HOOK_GATE_TOTAL: &str = "hook_gate_total";
    pub const HOOK_TOP_SCORE: &str = "hook_top_score";
    pub const CHUNKS_PER_MEMORY: &str = "chunks_per_memory";
    pub const RECALL_WEAK_RESULT_TOTAL: &str = "recall_weak_result_total";
    /// One line per prompt-hook call with its latency, top scores, injected
    /// ids and the gate that stopped it (SPEC section 6: "Each hook call
    /// records time, project, latency, top scores, injected memory
    /// identifiers and the gate that stopped it").
    pub const HOOK_LATENCY_MS: &str = "hook_latency_ms";
    /// One line per capture candidate that was not stored, value 1, label
    /// `reason`: `seen` when the project's capture state holds its hash
    /// (B-163), `near_duplicate` when an active or pending memory of the
    /// project, or an earlier candidate of the same read, is at least as
    /// close as the near-duplicate threshold, with that `cosine` as a label
    /// (B-164; the threshold is provisional and this is its metric), and
    /// `duplicate` when the store found a memory of the project with the
    /// same content hash (B-29, B-159). A line with the label `stage`
    /// `judge` is a capture the judge kept that is rejected after its
    /// verdict, kept pending and never served (B-190): reason
    /// `near_duplicate` when its rewrite is as close as the judge stage's
    /// threshold to an active memory of the project, or to a rewrite made
    /// active before it in the same call (B-196), with the `cosine`;
    /// `duplicate` when a memory of the project that is not pending holds
    /// the rewrite's content hash (B-167, B-169); `repeats` when the
    /// duplicate check found nothing in it that its nearest active
    /// memories, or the rewrites made active before it in the same call, do
    /// not already say (B-192, B-196). A line with
    /// `stage` `select` is a text that was not offered: a compaction
    /// summary unit (B-189), reason `summary_section` when it is in a
    /// section that is not offered, `summary_repeat` when the session
    /// offered most of its sentences before; or text read from a file or
    /// command output, reason `tool_results_off`, while
    /// `capture.tool_results` is off (B-221). Lines without `stage` are the
    /// capture stage's.
    pub const CAPTURE_DROPPED_TOTAL: &str = "capture_dropped_total";
    /// One line per capture the judge decided, value 1 (SPEC section 16,
    /// The judge; B-165). Labels: `verdict` (`keep` or `reject`; for a
    /// kept capture that is not settled and is judged again, `unquoted`
    /// or `too_long` when no memory could be built from it, B-191, and
    /// `recheck` when the duplicate check's answer does not hold for the
    /// neighbours it has at settling, B-196), `reason`
    /// (the judge's reason code: `knowledge` for a kept capture; for a
    /// rejected one `narration`, `status`, `tool_output`, `transient`,
    /// `restatement`, `fragment`, `duplicate` or `other`), `type` (the
    /// type of a kept capture) and `project` (the project's name). A kept
    /// capture that is then dropped as a
    /// near duplicate is counted here as kept and under
    /// `capture_dropped_total` with `stage` `judge`.
    pub const CAPTURE_JUDGED_TOTAL: &str = "capture_judged_total";
    /// One line per failed judge call, value 1, label `kind`: `unavailable`
    /// (Claude Code could not be started, or is not signed in),
    /// `rate_limited`, `timeout`, `call_failed` (Claude Code ran and
    /// reported any other error, or exited without an answer) or
    /// `bad_answer` (the answer broke the schema or did not judge each
    /// capture once), with `candidates`. Every capture of a failed call
    /// stays pending (B-165).
    pub const CAPTURE_JUDGE_FAILURES_TOTAL: &str = "capture_judge_failures_total";
    /// One line per judge call Claude Code answered, value the call's cost
    /// in US dollars as Claude Code reports it, labels `candidates` and
    /// `duration_ms` (B-165). The duplicate check's call (B-192) adds the
    /// label `call` `repeats`, its `candidates` the kept memories it was
    /// asked about. Under a Claude subscription the cost is an estimate of
    /// plan usage, not a charge. Under the statement task (B-241) `call`
    /// is `statements` for the call that returns a batch's statements, its
    /// `candidates` the captures, and `same` for the duplicate step's call,
    /// its `candidates` the statements asked about.
    pub const CAPTURE_JUDGE_COST_USD: &str = "capture_judge_cost_usd";
    /// One line per statement the judge returned under the statement task,
    /// value 1 (SPEC section 16, The judge; B-241). Labels: `outcome` (a
    /// [`super::StatementOutcome`]), `grade` (`must_keep` or
    /// `good_to_have`, haiku's own; recorded, never used), `type` and
    /// `project`. Each capture is still one `capture_judged_total` line,
    /// `keep` when at least one of its statements was stored.
    pub const CAPTURE_STATEMENTS_TOTAL: &str = "capture_statements_total";
    /// The number of pending captures, written after each judge call and
    /// each maintenance run, label `queued`: how many of them the judge can
    /// still take (B-165). The others are captures from before schema
    /// version 4 or captures with three bad answers; they wait only to be
    /// pruned (B-167).
    pub const CAPTURE_PENDING: &str = "capture_pending";
    /// One line per recall that hid near-identical results, value the
    /// number hidden, label `path` (`ambient` for the prompt hook's search,
    /// `recall` for the tool): results whose memory vector is at least as
    /// close as the collapse threshold to a better result of the same read
    /// (SPEC section 6, near-identical results; B-197). The threshold is
    /// provisional and this is its metric.
    pub const RECALL_COLLAPSED_TOTAL: &str = "recall_collapsed_total";
    /// One line per `forget` that deleted automatically captured memories
    /// (provenance source `capture`), value the number of them that had
    /// been served (any status but `pending`), label `pending` the number
    /// of pending captures it deleted (waiting for the judge, or rejected
    /// and kept 30 days; never served), and label `by` (`ids` or `filter`).
    /// `status` sums it over 7 days (B-200).
    pub const CAPTURE_FORGOTTEN_TOTAL: &str = "capture_forgotten_total";

    /// Every metric `status` summarises, in this order.
    pub const ALL: &[&str] = &[
        CAPTURE_SIZE_TOKENS,
        CAPTURE_CAP_HIT_TOTAL,
        RECALL_RESULT_COUNT,
        REINFORCED_MEMORY_RANK,
        EMBEDDER_COLD_START_SECONDS,
        HOOK_GATE_TOTAL,
        HOOK_TOP_SCORE,
        CHUNKS_PER_MEMORY,
        RECALL_WEAK_RESULT_TOTAL,
        HOOK_LATENCY_MS,
        CAPTURE_DROPPED_TOTAL,
        CAPTURE_JUDGED_TOTAL,
        CAPTURE_JUDGE_FAILURES_TOTAL,
        CAPTURE_JUDGE_COST_USD,
        CAPTURE_PENDING,
        RECALL_COLLAPSED_TOTAL,
        CAPTURE_FORGOTTEN_TOTAL,
        CAPTURE_STATEMENTS_TOTAL,
    ];
}

string_enum! {
    /// What became of one statement under the statement task
    /// (`capture_statements_total` label `outcome`; B-241).
    pub enum StatementOutcome {
        /// Stored as an active memory.
        Stored => "stored",
        /// The duplicate step found it says what a memory already says; kept
        /// unserved 30 days with the id of that memory (B-190).
        Repeat => "repeat",
        /// The judge itself labelled it a duplicate; not stored.
        Duplicate => "duplicate",
        /// Its span was not found in the capture; not stored.
        Unquoted => "unquoted",
        /// A memory of the project that is not pending holds its content
        /// hash; not stored.
        Hash => "hash",
        /// The duplicate step's answer for it failed every try; kept
        /// unserved 30 days, not stored.
        Unchecked => "unchecked",
    }
}

string_enum! {
    /// Which gate stopped (or passed) a prompt-hook call (`hook_gate_total`
    /// label `gate`; SPEC section 6, Hook gates).
    pub enum HookGate {
        /// The memories were injected.
        Injected => "injected",
        /// The switch is off or `TOP_K` is 0.
        Disabled => "disabled",
        /// The prompt is empty or shorter than the minimum.
        ShortPrompt => "short_prompt",
        /// No result reached the similarity floor.
        Floor => "floor",
        /// Every result above the floor was already injected this session.
        Repeat => "repeat",
        /// The project has no memories, or the scope is empty.
        Empty => "empty",
        /// The daemon is still loading the model.
        Loading => "loading",
        /// The deadline passed.
        Timeout => "timeout",
        /// The daemon could not be reached or returned an error.
        Error => "error",
        /// The project could not be resolved from the working directory.
        NoProject => "no_project",
    }
}

/// One journal line.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Entry {
    /// Python `isoformat()` of the UTC time, as version 2 writes it.
    pub ts: String,
    pub request_id: String,
    pub metric: String,
    pub value: f64,
    #[serde(default)]
    pub labels: BTreeMap<String, serde_json::Value>,
}

impl Entry {
    pub fn new(
        at: Timestamp,
        request_id: &str,
        metric: &str,
        value: f64,
        labels: BTreeMap<String, serde_json::Value>,
    ) -> Self {
        Entry {
            ts: at.to_python_isoformat(),
            request_id: request_id.to_owned(),
            metric: metric.to_owned(),
            value,
            labels,
        }
    }

    /// The compact JSON line with its newline, under [`MAX_LINE_BYTES`].
    /// String label values are cut to 64 characters; string items inside
    /// array labels (injected ids, top scores) are cut the same way. If the
    /// line is still too long, array labels lose items from the end, then
    /// labels are dropped from the last key back, and a `truncated: true`
    /// label records that it happened. `None` only if the line cannot fit
    /// even without labels (an absurd metric name or request id).
    pub fn to_line(&self) -> Option<String> {
        let mut e = self.clone();
        if !e.value.is_finite() {
            e.value = 0.0;
            e.labels
                .insert("non_finite".to_owned(), serde_json::Value::Bool(true));
        }
        for v in e.labels.values_mut() {
            cut_label(v);
        }
        let mut cut = false;
        loop {
            let mut line = serde_json::to_string(&e).ok()?;
            line.push('\n');
            if line.len() < MAX_LINE_BYTES {
                return Some(line);
            }
            if !cut {
                cut = true;
                e.labels
                    .insert("truncated".to_owned(), serde_json::Value::Bool(true));
                continue;
            }
            let longest_array = e
                .labels
                .iter()
                .filter(|(k, v)| {
                    k.as_str() != "truncated" && v.as_array().is_some_and(|a| !a.is_empty())
                })
                .max_by_key(|(_, v)| v.to_string().len())
                .map(|(k, _)| k.clone());
            if let Some(k) = longest_array {
                if let Some(serde_json::Value::Array(a)) = e.labels.get_mut(&k) {
                    a.pop();
                }
                continue;
            }
            let last = e
                .labels
                .keys()
                .rev()
                .find(|k| k.as_str() != "truncated")
                .cloned();
            match last {
                Some(k) => {
                    e.labels.remove(&k);
                }
                None => return None,
            }
        }
    }
}

fn cut_label(v: &mut serde_json::Value) {
    match v {
        serde_json::Value::String(s) if s.chars().count() > MAX_LABEL_CHARS => {
            *s = s.chars().take(MAX_LABEL_CHARS).collect();
        }
        serde_json::Value::Array(items) => {
            for item in items {
                if let serde_json::Value::String(s) = item
                    && s.chars().count() > MAX_LABEL_CHARS
                {
                    *s = s.chars().take(MAX_LABEL_CHARS).collect();
                }
            }
        }
        _ => {}
    }
}

/// `<YYYY-MM-DD>-<pid>-<start epoch>.jsonl`.
pub fn file_name(day: Timestamp, pid: u32, start_epoch: i64) -> String {
    format!("{}-{pid}-{start_epoch}.jsonl", day.date_string())
}

/// The date at the start of a journal file name, if it has one.
pub fn file_date(name: &str) -> Option<Timestamp> {
    let stem = name.strip_suffix(".jsonl")?;
    Timestamp::parse_date(stem.get(..10)?).ok()
}

/// A process's journal writer. Cheap to share; every method is silent on
/// error, as the journal must never break a request (SPEC section 6).
#[derive(Debug)]
pub struct Journal {
    dir: PathBuf,
    pid: u32,
    start_epoch: i64,
    include_prompts: bool,
    lock: Mutex<()>,
}

impl Journal {
    /// A writer for this process into `dir` (the profile's `metrics`
    /// folder). `include_prompts` is `journal.include_prompts`.
    pub fn new(dir: PathBuf, include_prompts: bool) -> Self {
        Journal {
            dir,
            pid: std::process::id(),
            start_epoch: Timestamp::now().unix_seconds(),
            include_prompts,
            lock: Mutex::new(()),
        }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Whether prompt text may be written into labels.
    pub fn include_prompts(&self) -> bool {
        self.include_prompts
    }

    /// Appends one line, creating the folder if needed. Silent on error.
    pub fn record(
        &self,
        request_id: &str,
        metric: &str,
        value: f64,
        labels: BTreeMap<String, serde_json::Value>,
    ) {
        let now = Timestamp::now();
        let Some(line) = Entry::new(now, request_id, metric, value, labels).to_line() else {
            return;
        };
        let path = self.dir.join(file_name(now, self.pid, self.start_epoch));
        let _guard = self.lock.lock();
        let _ = std::fs::create_dir_all(&self.dir);
        if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(&path) {
            // One write call per line (spike decision 3).
            let _ = f.write_all(line.as_bytes());
        }
    }
}

/// Removes journal files whose name dates are more than 7 days before
/// `today`. Silent on error.
pub fn prune(dir: &Path, today: Timestamp) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(date) = name.to_str().and_then(file_date) else {
            continue;
        };
        if today.days_since(date) > RETENTION_DAYS {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

/// Python's `statistics.quantiles(data, n=100)[p - 1]` (method
/// `exclusive`), which version 2 used for its p50 and p95; a single sample
/// is returned as is. `None` for no samples or `p` outside 1 to 99.
pub fn percentile(samples: &[f64], p: u32) -> Option<f64> {
    if !(1..=99).contains(&p) {
        return None;
    }
    let mut data: Vec<f64> = samples.iter().copied().filter(|v| v.is_finite()).collect();
    match data.len() {
        0 => return None,
        1 => return Some(data[0]),
        _ => {}
    }
    data.sort_by(f64::total_cmp);
    let ld = data.len() as i64;
    let n: i64 = 100;
    let m = ld + 1;
    let i = i64::from(p);
    let j = (i * m / n).clamp(1, ld - 1);
    let delta = i * m - j * n;
    let (lo, hi) = (data[(j - 1) as usize], data[j as usize]);
    Some((lo * (n - delta) as f64 + hi * delta as f64) / n as f64)
}

/// Median and 95th percentile of one metric over the window.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct MetricSummary {
    pub metric: String,
    pub count: u64,
    pub median: Option<f64>,
    pub p95: Option<f64>,
}

/// Calls `f` with every line of every journal file in `dir` written in the
/// `window_days` before `now`. Unreadable files and lines are skipped.
fn each_entry(dir: &Path, now: Timestamp, window_days: i64, mut f: impl FnMut(Entry)) {
    let cutoff = now.add_micros(-window_days * 86_400 * 1_000_000);
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        for line in text.lines() {
            let Ok(e) = serde_json::from_str::<Entry>(line) else {
                continue;
            };
            let Ok(ts) = Timestamp::parse(&e.ts) else {
                continue;
            };
            if ts >= cutoff {
                f(e);
            }
        }
    }
}

/// Reads every journal file in `dir` and summarises each metric in
/// [`metrics::ALL`] over the `window_days` before `now`. Unreadable files
/// and lines are skipped.
pub fn summarize(dir: &Path, now: Timestamp, window_days: i64) -> Vec<MetricSummary> {
    let mut values: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    each_entry(dir, now, window_days, |e| {
        values.entry(e.metric).or_default().push(e.value);
    });
    metrics::ALL
        .iter()
        .map(|m| {
            let v = values.get(*m).map(Vec::as_slice).unwrap_or(&[]);
            MetricSummary {
                metric: (*m).to_owned(),
                count: v.len() as u64,
                median: percentile(v, 50),
                p95: percentile(v, 95),
            }
        })
        .collect()
}

/// Automatic capture over the last days, as `status` shows it (SPEC
/// section 16, Quality over the last 7 days; B-200). Read from the journal,
/// so a line the journal failed to write is not counted.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct CaptureQuality {
    /// The window, in days before now.
    pub days: u32,
    /// Captures the judge settled: verdict `keep` or `reject`. A kept
    /// capture judged again (`unquoted`, `too_long`, `recheck`) is counted
    /// when it is settled.
    pub judged: u64,
    /// Settled captures made active: kept, and not then dropped as a
    /// duplicate, near duplicate or repeat.
    pub kept: u64,
    /// Settled captures not made active: rejected by the verdict, or kept
    /// and then dropped. Kept 30 days, never served (B-190).
    pub rejected: u64,
    /// The part of `rejected` the judge kept that was then dropped
    /// (`capture_dropped_total` with `stage` `judge`).
    pub dropped_after_keep: u64,
    /// Automatically captured memories that had been served and were
    /// removed with `forget`.
    pub forgotten: u64,
    /// Pending captures removed with `forget`: never served.
    pub forgotten_pending: u64,
    /// Judge calls, the duplicate check's and the failed ones included, as
    /// the day's usage counts them (B-202).
    pub calls: u64,
    /// Their cost in US dollars, as Claude Code reports it: under a Claude
    /// subscription an estimate of plan usage, not a charge.
    pub cost_usd: f64,
    /// Statements the judge returned under the statement task (B-241),
    /// every outcome.
    #[serde(default)]
    pub statements: u64,
    /// The part of `statements` stored as active memories.
    #[serde(default)]
    pub statements_stored: u64,
    /// The part the duplicate step held back as repeats.
    #[serde(default)]
    pub statements_repeat: u64,
    /// The part the duplicate step could not check, held back.
    #[serde(default)]
    pub statements_unchecked: u64,
}

/// [`CaptureQuality`] over the `window_days` before `now`, from the
/// journal files in `dir`.
pub fn capture_quality(dir: &Path, now: Timestamp, window_days: i64) -> CaptureQuality {
    let mut q = CaptureQuality {
        days: u32::try_from(window_days).unwrap_or(0),
        ..CaptureQuality::default()
    };
    let (mut keep, mut reject) = (0_u64, 0_u64);
    let count = |v: f64| {
        if v.is_finite() && v > 0.0 {
            v as u64
        } else {
            0
        }
    };
    let label = |e: &Entry, k: &str| e.labels.get(k).and_then(|v| v.as_str()).map(str::to_owned);
    each_entry(dir, now, window_days, |e| match e.metric.as_str() {
        metrics::CAPTURE_JUDGED_TOTAL => match label(&e, "verdict").as_deref() {
            Some("keep") => keep += count(e.value),
            Some("reject") => reject += count(e.value),
            _ => {}
        },
        metrics::CAPTURE_DROPPED_TOTAL if label(&e, "stage").as_deref() == Some("judge") => {
            q.dropped_after_keep += count(e.value);
        }
        metrics::CAPTURE_FORGOTTEN_TOTAL => {
            q.forgotten += count(e.value);
            q.forgotten_pending += e
                .labels
                .get("pending")
                .and_then(serde_json::Value::as_f64)
                .map_or(0, count);
        }
        metrics::CAPTURE_JUDGE_COST_USD => {
            q.calls += 1;
            if e.value.is_finite() && e.value > 0.0 {
                q.cost_usd += e.value;
            }
        }
        metrics::CAPTURE_STATEMENTS_TOTAL => {
            let n = count(e.value);
            q.statements += n;
            match label(&e, "outcome").as_deref() {
                Some("stored") => q.statements_stored += n,
                Some("repeat") => q.statements_repeat += n,
                Some("unchecked") => q.statements_unchecked += n,
                _ => {}
            }
        }
        _ => {}
    });
    // Each drop after a keep is journalled with its verdict, so it cannot
    // outnumber the keeps unless the journal lost a line.
    q.dropped_after_keep = q.dropped_after_keep.min(keep);
    q.judged = keep + reject;
    q.kept = keep - q.dropped_after_keep;
    q.rejected = reject + q.dropped_after_keep;
    q
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn labels(v: serde_json::Value) -> BTreeMap<String, serde_json::Value> {
        serde_json::from_value(v).unwrap()
    }

    #[test]
    fn s6_journal_line_matches_version_2_shape() {
        let at = Timestamp::parse("2026-11-16T09:21:07.512340Z").unwrap();
        let e = Entry::new(
            at,
            "a1b2c3d4e5f6",
            metrics::HOOK_TOP_SCORE,
            0.81,
            labels(json!({"project": "x"})),
        );
        assert_eq!(
            e.to_line().unwrap(),
            "{\"ts\":\"2026-11-16T09:21:07.512340+00:00\",\"request_id\":\"a1b2c3d4e5f6\",\
             \"metric\":\"hook_top_score\",\"value\":0.81,\"labels\":{\"project\":\"x\"}}\n"
        );
    }

    #[test]
    fn s6_journal_lines_stay_under_4096_bytes() {
        let at = Timestamp::from_unix_seconds(1_800_000_000);
        let long = "y".repeat(500);
        let e = Entry::new(
            at,
            "r",
            "m",
            1.0,
            labels(json!({"a": long, "ids": vec!["x".repeat(60); 200]})),
        );
        let line = e.to_line().unwrap();
        assert!(line.len() < MAX_LINE_BYTES);
        let back: Entry = serde_json::from_str(line.trim_end()).unwrap();
        assert_eq!(
            back.labels["a"].as_str().unwrap().chars().count(),
            MAX_LABEL_CHARS
        );
        assert_eq!(back.labels["truncated"], json!(true));
        assert!(!back.labels["ids"].as_array().unwrap().is_empty());
        let nan = Entry::new(at, "r", "m", f64::NAN, BTreeMap::new())
            .to_line()
            .unwrap();
        assert!(nan.contains("\"value\":0.0") && nan.contains("non_finite"));
        assert!(
            Entry::new(at, &"r".repeat(5000), "m", 1.0, BTreeMap::new())
                .to_line()
                .is_none()
        );
    }

    #[test]
    fn s6_percentile_matches_python_statistics_quantiles() {
        // Values from CPython 3.13: statistics.quantiles(data, n=100).
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        assert_eq!(percentile(&data, 50), Some(5.5));
        // The exclusive method extrapolates past the largest sample.
        assert!((percentile(&data, 95).unwrap() - 10.45).abs() < 1e-12);
        assert!((percentile(&[3.0, 1.0, 2.0], 50).unwrap() - 2.0).abs() < 1e-12);
        assert!((percentile(&[1.0, 2.0, 3.0], 95).unwrap() - 3.8).abs() < 1e-12);
        // quantiles([10, 20], n=100)[49] == 15.0
        assert!((percentile(&[10.0, 20.0], 50).unwrap() - 15.0).abs() < 1e-12);
        assert_eq!(percentile(&[7.0], 95), Some(7.0));
        assert_eq!(percentile(&[], 50), None);
        let many: Vec<f64> = (1..=200).map(f64::from).collect();
        // quantiles(range(1, 201), n=100)[94] == 190.95
        assert!((percentile(&many, 95).unwrap() - 190.95).abs() < 1e-9);
    }

    #[test]
    fn s6_journal_files_are_per_process_and_pruned_after_7_days() {
        let dir = tempfile::tempdir().unwrap();
        let j = Journal::new(dir.path().join("metrics"), false);
        j.record("r1", metrics::RECALL_RESULT_COUNT, 3.0, BTreeMap::new());
        j.record("r2", metrics::RECALL_RESULT_COUNT, 5.0, BTreeMap::new());
        let files: Vec<_> = std::fs::read_dir(j.dir()).unwrap().flatten().collect();
        assert_eq!(files.len(), 1);
        let name = files[0].file_name().into_string().unwrap();
        assert!(name.contains(&format!("-{}-", std::process::id())) && name.ends_with(".jsonl"));
        let summary = summarize(j.dir(), Timestamp::now(), SUMMARY_WINDOW_DAYS);
        let rc = summary
            .iter()
            .find(|s| s.metric == metrics::RECALL_RESULT_COUNT)
            .unwrap();
        assert_eq!((rc.count, rc.median), (2, Some(4.0)));

        let old = j.dir().join("2026-11-01-1-1.jsonl");
        let recent = j.dir().join("2026-11-09-1-1.jsonl");
        std::fs::write(&old, "").unwrap();
        std::fs::write(&recent, "").unwrap();
        std::fs::write(j.dir().join("notes.txt"), "").unwrap();
        prune(j.dir(), Timestamp::parse("2026-11-16T12:00:00Z").unwrap());
        assert!(!old.exists());
        assert!(recent.exists());
        assert!(j.dir().join("notes.txt").exists());
        // Unwritable folder: silent.
        Journal::new(dir.path().join("metrics").join(&name).join("x"), false).record(
            "r",
            "m",
            1.0,
            BTreeMap::new(),
        );
    }

    /// `status`'s 7-day view of automatic capture adds up the judge's
    /// verdicts, the drops after a keep, the forgotten captures and the
    /// judge's cost, and leaves out older lines (B-200).
    #[test]
    fn s16_b200_capture_quality_counts_the_last_7_days() {
        let dir = tempfile::tempdir().unwrap();
        let now = Timestamp::parse("2026-11-18T12:00:00Z").unwrap();
        let recent = Timestamp::parse("2026-11-12T12:00:00Z").unwrap();
        let old = Timestamp::parse("2026-11-11T11:59:59Z").unwrap();
        let line = |at: Timestamp, metric: &str, value: f64, l: serde_json::Value| {
            Entry::new(at, "judge", metric, value, labels(l))
                .to_line()
                .unwrap()
        };
        let judged = metrics::CAPTURE_JUDGED_TOTAL;
        let mut text = String::new();
        for _ in 0..5 {
            text += &line(recent, judged, 1.0, json!({"verdict": "keep"}));
        }
        for _ in 0..3 {
            text += &line(now, judged, 1.0, json!({"verdict": "reject"}));
        }
        // Judged again later, not settled: not counted.
        text += &line(now, judged, 1.0, json!({"verdict": "recheck"}));
        text += &line(now, judged, 1.0, json!({"verdict": "unquoted"}));
        // Two kept captures dropped after the verdict; capture-stage drops
        // are not the judge's.
        for reason in ["near_duplicate", "repeats"] {
            text += &line(
                now,
                metrics::CAPTURE_DROPPED_TOTAL,
                1.0,
                json!({"stage": "judge", "reason": reason}),
            );
        }
        text += &line(
            now,
            metrics::CAPTURE_DROPPED_TOTAL,
            1.0,
            json!({"reason": "seen"}),
        );
        text += &line(
            now,
            metrics::CAPTURE_FORGOTTEN_TOTAL,
            2.0,
            json!({"pending": 1, "by": "ids"}),
        );
        text += &line(
            now,
            metrics::CAPTURE_FORGOTTEN_TOTAL,
            0.0,
            json!({"pending": 4, "by": "filter"}),
        );
        text += &line(
            recent,
            metrics::CAPTURE_JUDGE_COST_USD,
            0.0021,
            json!({"candidates": 8}),
        );
        text += &line(
            now,
            metrics::CAPTURE_JUDGE_COST_USD,
            0.0004,
            json!({"candidates": 2, "call": "repeats"}),
        );
        // Older than 7 days: none of these count.
        text += &line(old, judged, 1.0, json!({"verdict": "keep"}));
        text += &line(
            old,
            metrics::CAPTURE_FORGOTTEN_TOTAL,
            3.0,
            json!({"pending": 0}),
        );
        text += &line(old, metrics::CAPTURE_JUDGE_COST_USD, 1.0, json!({}));
        text += "not json\n";
        std::fs::write(dir.path().join("2026-11-18-1-1.jsonl"), text).unwrap();

        let q = capture_quality(dir.path(), now, SUMMARY_WINDOW_DAYS);
        assert_eq!(
            (q.days, q.judged, q.kept, q.rejected, q.dropped_after_keep),
            (7, 8, 3, 5, 2),
            "{q:?}"
        );
        assert_eq!(
            (q.forgotten, q.forgotten_pending, q.calls),
            (2, 5, 2),
            "{q:?}"
        );
        assert!((q.cost_usd - 0.0025).abs() < 1e-12, "{q:?}");

        let empty = capture_quality(&dir.path().join("missing"), now, SUMMARY_WINDOW_DAYS);
        assert_eq!(
            empty,
            CaptureQuality {
                days: 7,
                ..CaptureQuality::default()
            }
        );
    }

    /// B-241: under the statement task each statement the judge returned is
    /// one `capture_statements_total` line with its outcome, and the 7-day
    /// view counts them by outcome beside the captures.
    #[test]
    fn s16_capture_statements_total_counts_each_outcome() {
        assert_eq!(
            metrics::CAPTURE_STATEMENTS_TOTAL,
            "capture_statements_total"
        );
        assert!(metrics::ALL.contains(&metrics::CAPTURE_STATEMENTS_TOTAL));
        let names: Vec<&str> = StatementOutcome::ALL.iter().map(|o| o.as_str()).collect();
        assert_eq!(
            names,
            [
                "stored",
                "repeat",
                "duplicate",
                "unquoted",
                "hash",
                "unchecked"
            ]
        );
        let dir = tempfile::tempdir().unwrap();
        let now = Timestamp::parse("2026-11-23T12:00:00Z").unwrap();
        let old = Timestamp::parse("2026-11-16T11:59:59Z").unwrap();
        let line = |at: Timestamp, outcome: &str| {
            Entry::new(
                at,
                "judge",
                metrics::CAPTURE_STATEMENTS_TOTAL,
                1.0,
                labels(json!({"outcome": outcome, "grade": "must_keep",
                              "type": "decision", "project": "p"})),
            )
            .to_line()
            .unwrap()
        };
        let mut text = String::new();
        for outcome in [
            "stored",
            "stored",
            "stored",
            "repeat",
            "repeat",
            "duplicate",
            "unquoted",
            "hash",
            "unchecked",
        ] {
            text += &line(now, outcome);
        }
        text += &line(old, "stored");
        text += &Entry::new(
            now,
            "judge",
            metrics::CAPTURE_JUDGED_TOTAL,
            1.0,
            labels(json!({"verdict": "keep"})),
        )
        .to_line()
        .unwrap();
        std::fs::write(dir.path().join("2026-11-23-1-1.jsonl"), text).unwrap();
        let q = capture_quality(dir.path(), now, SUMMARY_WINDOW_DAYS);
        assert_eq!(
            (
                q.statements,
                q.statements_stored,
                q.statements_repeat,
                q.statements_unchecked
            ),
            (9, 3, 2, 1),
            "{q:?}"
        );
        assert_eq!((q.judged, q.kept), (1, 1), "{q:?}");
    }

    #[test]
    fn s6_hook_gates_have_stable_names() {
        let names: Vec<&str> = HookGate::ALL.iter().map(|g| g.as_str()).collect();
        assert_eq!(
            names,
            [
                "injected",
                "disabled",
                "short_prompt",
                "floor",
                "repeat",
                "empty",
                "loading",
                "timeout",
                "error",
                "no_project"
            ]
        );
        assert_eq!(
            file_name(Timestamp::parse_date("2026-11-16").unwrap(), 42, 7),
            "2026-11-16-42-7.jsonl"
        );
        assert_eq!(
            file_date("2026-11-16-42-7.jsonl"),
            Some(Timestamp::parse_date("2026-11-16").unwrap())
        );
        assert_eq!(file_date("x.jsonl"), None);
    }
}
