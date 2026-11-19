//! Judges for the daemon's tests (SPEC 16, The judge; B-168).
//!
//! - [`RecordedJudge`] replays the real judge's verdicts recorded in the
//!   spike (`fixtures/judge/recorded-verdicts.json`, `SPIKE-JUDGE.md`).
//!   A test that uses it shows what the daemon does with the model's own
//!   answers on those synthetic texts. It does not measure the model: that
//!   is done on the labelled export (`CAPTURE-RCA.md` section 8.5).
//! - [`ScriptedJudge`] answers as a test tells it to, records every batch it
//!   is given, and can hold a call open, for the paths a recording cannot
//!   reach: failures, the cap, near duplicates, redaction and timing. Its
//!   duplicate check (B-192) finds something new in every memory unless a
//!   test sets another answer.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use sm_daemon::{
    DaemonOptions, Judge, JudgeAnswer, JudgeCandidate, JudgeFailure, JudgeFailureKind, JudgeSpan,
    JudgeVerdict, RepeatAnswer, RepeatItem, RepeatVerdict,
};

use super::{Rig, fixture};

/// Short judge waits for tests: a partial batch is judged after 20 ms, and
/// a failed call is retried after 300 ms, doubling up to 2 s.
pub fn judge_options(opts: &mut DaemonOptions, judge: Arc<dyn Judge>) {
    opts.judge = Some(judge);
    opts.judge_wait = Duration::from_millis(20);
    opts.judge_backoff = Duration::from_millis(300);
    opts.judge_backoff_max = Duration::from_secs(2);
}

impl Rig {
    /// Runs the daemons of this rig with `judge` and the short waits of
    /// [`judge_options`].
    pub fn use_judge(&mut self, judge: Arc<dyn Judge>) {
        judge_options(&mut self.opts, judge);
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

// --- the recorded judge ----------------------------------------------------

/// One verdict of the recording.
#[derive(Debug, Clone)]
pub struct Recorded {
    pub keep: bool,
    pub reason: String,
    pub memory_type: String,
    /// The statements the judge marked in the text, by their first and
    /// last words (B-191).
    pub spans: Vec<JudgeSpan>,
}

/// Replays the recorded verdicts, matched on the candidate's text. A text
/// that is not in the recording fails the whole batch as a bad answer and
/// is listed by [`RecordedJudge::misses`]: a test never passes on a verdict
/// made up for it. The duplicate check (B-192) was not recorded, so every
/// call of it fails the same way, each memory it was asked about listed as
/// a miss.
#[derive(Debug)]
pub struct RecordedJudge {
    verdicts: HashMap<String, Recorded>,
    model: String,
    cost_per_candidate: f64,
    misses: Mutex<Vec<String>>,
}

/// The text the judged barrier stores: candidate c14 of the recording, a
/// decision with its reason, which the real judge kept.
pub const RECORDED_SENTINEL: &str = "We decided that the importer runs before the indexer, because the indexer reads the imported rows.";

impl RecordedJudge {
    pub fn new() -> Arc<RecordedJudge> {
        let doc = fixture("judge/recorded-verdicts.json");
        let model = doc["model"]
            .as_str()
            .expect("the recording names its model");
        let runs = doc["runs"].as_object().expect("the recorded runs");
        let (cost, n) = runs.values().fold((0.0, 0.0), |(c, n), r| {
            (
                c + r["cost_usd"].as_f64().expect("a run's cost"),
                n + r["candidates"].as_f64().expect("a run's size"),
            )
        });
        let mut verdicts = HashMap::new();
        for v in doc["verdicts"].as_array().expect("the verdicts") {
            let s = |k: &str| v[k].as_str().unwrap_or_else(|| panic!("{k}")).to_owned();
            let spans = v["spans"]
                .as_array()
                .expect("spans")
                .iter()
                .map(|q| JudgeSpan {
                    start: q["start"].as_str().expect("a span's start").to_owned(),
                    end: q["end"].as_str().expect("a span's end").to_owned(),
                })
                .collect();
            verdicts.insert(
                s("text"),
                Recorded {
                    keep: v["keep"].as_bool().expect("keep"),
                    reason: s("reason"),
                    memory_type: s("type"),
                    spans,
                },
            );
        }
        Arc::new(RecordedJudge {
            verdicts,
            model: model.to_owned(),
            cost_per_candidate: cost / n,
            misses: Mutex::new(Vec::new()),
        })
    }

    /// The recorded verdict on `text`.
    pub fn recorded(&self, text: &str) -> Recorded {
        self.verdicts
            .get(text)
            .unwrap_or_else(|| panic!("{text:?} is not in the recording"))
            .clone()
    }

    /// The texts it was given that are not in the recording.
    pub fn misses(&self) -> Vec<String> {
        lock(&self.misses).clone()
    }
}

impl Judge for RecordedJudge {
    fn model(&self) -> &str {
        "haiku"
    }

    fn command(&self) -> Option<String> {
        Some("recorded".to_owned())
    }

    fn judge(
        &self,
        batch: &[JudgeCandidate],
        _stop: &AtomicBool,
    ) -> Result<JudgeAnswer, JudgeFailure> {
        let mut verdicts = Vec::with_capacity(batch.len());
        let mut missed = Vec::new();
        for c in batch {
            match self.verdicts.get(&c.text) {
                Some(r) => verdicts.push(JudgeVerdict {
                    id: c.id.clone(),
                    keep: r.keep,
                    reason: r.reason.clone(),
                    memory_type: r.memory_type.clone(),
                    spans: r.spans.clone(),
                }),
                None => missed.push(c.text.clone()),
            }
        }
        if !missed.is_empty() {
            let n = missed.len();
            lock(&self.misses).extend(missed);
            return Err(JudgeFailure {
                kind: JudgeFailureKind::BadAnswer,
                message: format!("{n} candidates are not in the recording"),
                retry_at: None,
                cost_usd: 0.0,
            });
        }
        Ok(JudgeAnswer {
            verdicts,
            cost_usd: self.cost_per_candidate * batch.len() as f64,
            duration_ms: 1,
            model: Some(self.model.clone()),
        })
    }

    fn repeats(
        &self,
        items: &[RepeatItem],
        _stop: &AtomicBool,
    ) -> Result<RepeatAnswer, JudgeFailure> {
        lock(&self.misses).extend(items.iter().map(|i| i.text.clone()));
        Err(JudgeFailure {
            kind: JudgeFailureKind::BadAnswer,
            message: format!(
                "the duplicate check was not recorded; {} memories were asked",
                items.len()
            ),
            retry_at: None,
            cost_usd: 0.0,
        })
    }
}

// --- the scripted judge ----------------------------------------------------

/// What a [`ScriptedJudge`] answers a batch with.
pub type Script = Box<dyn Fn(&[JudgeCandidate]) -> Result<JudgeAnswer, JudgeFailure> + Send + Sync>;

/// A kept verdict on `c` that marks one statement, by its first and last
/// words (B-191).
pub fn keep(c: &JudgeCandidate, span: (&str, &str), memory_type: &str) -> JudgeVerdict {
    keep_spans(c, &[span], memory_type)
}

/// A kept verdict on `c` that marks each of `spans`.
pub fn keep_spans(c: &JudgeCandidate, spans: &[(&str, &str)], memory_type: &str) -> JudgeVerdict {
    JudgeVerdict {
        id: c.id.clone(),
        keep: true,
        reason: "knowledge".to_owned(),
        memory_type: memory_type.to_owned(),
        spans: spans
            .iter()
            .map(|(start, end)| JudgeSpan {
                start: (*start).to_owned(),
                end: (*end).to_owned(),
            })
            .collect(),
    }
}

/// A rejected verdict on `c`.
pub fn reject(c: &JudgeCandidate, reason: &str) -> JudgeVerdict {
    JudgeVerdict {
        id: c.id.clone(),
        keep: false,
        reason: reason.to_owned(),
        memory_type: "note".to_owned(),
        spans: Vec::new(),
    }
}

/// An answer of one verdict per candidate, at 0.0001 US dollars each.
pub fn answer(
    batch: &[JudgeCandidate],
    f: impl Fn(&JudgeCandidate) -> JudgeVerdict,
) -> Result<JudgeAnswer, JudgeFailure> {
    Ok(JudgeAnswer {
        verdicts: batch.iter().map(f).collect(),
        cost_usd: 0.0001 * batch.len() as f64,
        duration_ms: 5,
        model: Some("claude-test-judge".to_owned()),
    })
}

/// Keeps every candidate whole, one span over its whole text, with `memory_type`.
pub fn keep_all(memory_type: &'static str) -> Script {
    Box::new(move |batch| answer(batch, |c| keep(c, (&c.text, &c.text), memory_type)))
}

/// What a [`ScriptedJudge`] answers a duplicate check with (B-192).
pub type RepeatScript =
    Box<dyn Fn(&[RepeatItem]) -> Result<RepeatAnswer, JudgeFailure> + Send + Sync>;

/// A duplicate check's answer: `new(item)` for each item, at 0.0001 US
/// dollars each.
pub fn repeat_answer(
    items: &[RepeatItem],
    new: impl Fn(&RepeatItem) -> bool,
) -> Result<RepeatAnswer, JudgeFailure> {
    Ok(RepeatAnswer {
        verdicts: items
            .iter()
            .map(|i| RepeatVerdict {
                id: i.id.clone(),
                new: new(i),
                same_as: Vec::new(),
                spans: Vec::new(),
            })
            .collect(),
        cost_usd: 0.0001 * items.len() as f64,
        duration_ms: 5,
    })
}

/// Finds something new in every memory: the default.
pub fn all_new() -> RepeatScript {
    Box::new(|items| repeat_answer(items, |_| true))
}

/// Finds nothing new in any memory.
pub fn nothing_new() -> RepeatScript {
    Box::new(|items| repeat_answer(items, |_| false))
}

/// Fails every duplicate check with `kind`.
pub fn failing_repeats(kind: JudgeFailureKind, message: &'static str) -> RepeatScript {
    Box::new(move |_| {
        Err(JudgeFailure {
            kind,
            message: message.to_owned(),
            retry_at: None,
            cost_usd: 0.0,
        })
    })
}

/// Fails every call with `kind`.
pub fn failing(kind: JudgeFailureKind, message: &'static str) -> Script {
    Box::new(move |_| {
        Err(JudgeFailure {
            kind,
            message: message.to_owned(),
            retry_at: None,
            cost_usd: 0.0,
        })
    })
}

/// A judge that runs a script the test can change, records every batch,
/// and can hold calls open until released.
pub struct ScriptedJudge {
    script: Mutex<Script>,
    repeats: Mutex<RepeatScript>,
    command: Mutex<Option<String>>,
    batches: Mutex<Vec<Vec<JudgeCandidate>>>,
    checks: Mutex<Vec<Vec<RepeatItem>>>,
    times: Mutex<Vec<Instant>>,
    held: AtomicBool,
    in_flight: AtomicUsize,
    max_in_flight: AtomicUsize,
}

impl std::fmt::Debug for ScriptedJudge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScriptedJudge")
            .field("batches", &lock(&self.batches).len())
            .finish_non_exhaustive()
    }
}

impl ScriptedJudge {
    pub fn new(script: Script) -> Arc<ScriptedJudge> {
        Arc::new(ScriptedJudge {
            script: Mutex::new(script),
            repeats: Mutex::new(all_new()),
            command: Mutex::new(Some("claude-test".to_owned())),
            batches: Mutex::new(Vec::new()),
            checks: Mutex::new(Vec::new()),
            times: Mutex::new(Vec::new()),
            held: AtomicBool::new(false),
            in_flight: AtomicUsize::new(0),
            max_in_flight: AtomicUsize::new(0),
        })
    }

    pub fn set(&self, script: Script) {
        *lock(&self.script) = script;
    }

    /// What the duplicate check answers from now on.
    pub fn set_repeats(&self, script: RepeatScript) {
        *lock(&self.repeats) = script;
    }

    /// Every duplicate check it was asked, in order (B-192).
    pub fn checks(&self) -> Vec<Vec<RepeatItem>> {
        lock(&self.checks).clone()
    }

    /// `None`: Claude Code is not found.
    pub fn set_command(&self, command: Option<&str>) {
        *lock(&self.command) = command.map(str::to_owned);
    }

    /// Calls made from now on wait until [`ScriptedJudge::release`].
    pub fn hold(&self) {
        self.held.store(true, Ordering::SeqCst);
    }

    pub fn release(&self) {
        self.held.store(false, Ordering::SeqCst);
    }

    /// Every batch it was called with, in order.
    pub fn batches(&self) -> Vec<Vec<JudgeCandidate>> {
        lock(&self.batches).clone()
    }

    /// When each call started, in order.
    pub fn times(&self) -> Vec<Instant> {
        lock(&self.times).clone()
    }

    pub fn calls(&self) -> usize {
        lock(&self.batches).len()
    }

    /// Whether a call is running now.
    pub fn in_call(&self) -> bool {
        self.in_flight.load(Ordering::SeqCst) > 0
    }

    /// The most calls that ran at once.
    pub fn max_in_flight(&self) -> usize {
        self.max_in_flight.load(Ordering::SeqCst)
    }
}

impl Judge for ScriptedJudge {
    fn model(&self) -> &str {
        "haiku"
    }

    fn command(&self) -> Option<String> {
        lock(&self.command).clone()
    }

    fn judge(
        &self,
        batch: &[JudgeCandidate],
        stop: &AtomicBool,
    ) -> Result<JudgeAnswer, JudgeFailure> {
        let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_in_flight.fetch_max(now, Ordering::SeqCst);
        lock(&self.times).push(Instant::now());
        lock(&self.batches).push(batch.to_vec());
        while self.held.load(Ordering::SeqCst) {
            if stop.load(Ordering::SeqCst) {
                self.in_flight.fetch_sub(1, Ordering::SeqCst);
                return Err(JudgeFailure {
                    kind: JudgeFailureKind::Stopped,
                    message: "the daemon is stopping".to_owned(),
                    retry_at: None,
                    cost_usd: 0.0,
                });
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        let out = (lock(&self.script))(batch);
        self.in_flight.fetch_sub(1, Ordering::SeqCst);
        out
    }

    fn repeats(
        &self,
        items: &[RepeatItem],
        _stop: &AtomicBool,
    ) -> Result<RepeatAnswer, JudgeFailure> {
        lock(&self.checks).push(items.to_vec());
        (lock(&self.repeats))(items)
    }
}
