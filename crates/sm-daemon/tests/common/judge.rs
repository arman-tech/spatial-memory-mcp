//! Judges for the daemon's tests (SPEC 16, The judge; B-168).
//!
//! - [`RecordedJudge`] replays the real judge's verdicts recorded in the
//!   spike (`fixtures/judge/recorded-verdicts.json`, `SPIKE-JUDGE.md`).
//!   A test that uses it shows what the daemon does with the model's own
//!   answers on those synthetic texts. It does not measure the model: that
//!   is done on the labelled export (`CAPTURE-RCA.md` section 8.5).
//! - [`ScriptedJudge`] answers as a test tells it to, records every batch it
//!   is given, and can hold a call open, for the paths a recording cannot
//!   reach: failures, the cap, near duplicates, redaction and timing.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use sm_daemon::{
    DaemonOptions, Judge, JudgeAnswer, JudgeCandidate, JudgeFailure, JudgeFailureKind, JudgeVerdict,
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
    /// The sentences the judge quoted from the text (B-173).
    pub quotes: Vec<String>,
}

/// Replays the recorded verdicts, matched on the candidate's text. A text
/// that is not in the recording fails the whole batch as a bad answer and
/// is listed by [`RecordedJudge::misses`]: a test never passes on a verdict
/// made up for it.
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
            let quotes = v["quotes"]
                .as_array()
                .expect("quotes")
                .iter()
                .map(|q| q.as_str().expect("a quote").to_owned())
                .collect();
            verdicts.insert(
                s("text"),
                Recorded {
                    keep: v["keep"].as_bool().expect("keep"),
                    reason: s("reason"),
                    memory_type: s("type"),
                    quotes,
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
                    quotes: r.quotes.clone(),
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
}

// --- the scripted judge ----------------------------------------------------

/// What a [`ScriptedJudge`] answers a batch with.
pub type Script = Box<dyn Fn(&[JudgeCandidate]) -> Result<JudgeAnswer, JudgeFailure> + Send + Sync>;

/// A kept verdict on `c` that quotes `quote` (B-173).
pub fn keep(c: &JudgeCandidate, quote: &str, memory_type: &str) -> JudgeVerdict {
    keep_quotes(c, &[quote], memory_type)
}

/// A kept verdict on `c` that quotes each of `quotes`.
pub fn keep_quotes(c: &JudgeCandidate, quotes: &[&str], memory_type: &str) -> JudgeVerdict {
    JudgeVerdict {
        id: c.id.clone(),
        keep: true,
        reason: "knowledge".to_owned(),
        memory_type: memory_type.to_owned(),
        quotes: quotes.iter().map(|q| (*q).to_owned()).collect(),
    }
}

/// A rejected verdict on `c`.
pub fn reject(c: &JudgeCandidate, reason: &str) -> JudgeVerdict {
    JudgeVerdict {
        id: c.id.clone(),
        keep: false,
        reason: reason.to_owned(),
        memory_type: "note".to_owned(),
        quotes: Vec::new(),
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

/// Keeps every candidate whole, quoting its whole text, with `memory_type`.
pub fn keep_all(memory_type: &'static str) -> Script {
    Box::new(move |batch| answer(batch, |c| keep(c, &c.text, memory_type)))
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
    command: Mutex<Option<String>>,
    batches: Mutex<Vec<Vec<JudgeCandidate>>>,
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
            command: Mutex::new(Some("claude-test".to_owned())),
            batches: Mutex::new(Vec::new()),
            times: Mutex::new(Vec::new()),
            held: AtomicBool::new(false),
            in_flight: AtomicUsize::new(0),
            max_in_flight: AtomicUsize::new(0),
        })
    }

    pub fn set(&self, script: Script) {
        *lock(&self.script) = script;
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
}
