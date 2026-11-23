//! The statements task (B-240 to B-247; `docs/v3/STATEMENT_JUDGE_PROPOSAL.md`
//! as built, with the owner's changes): `capture.judge.task = "statements"`.
//!
//! - **One session per call** ([`sm_store::Store::judge_queue_session`]).
//!   One call ([`Judge::statements`]) marks every statement of each capture
//!   of the batch by its first and last words, with a label (`keep` or
//!   `duplicate`), a type and a grade (`must_keep` or `good_to_have`).
//! - **The capture's own words** (A6). Each statement labelled `keep` whose
//!   span is found in the capture's text ([`spanned`]) is redacted again,
//!   titled by `derive_title` and chunked; one whose span is not found is
//!   `unquoted`, and one the judge labels `duplicate` is not stored. Nothing
//!   the model wrote is stored or shown.
//! - **The duplicate step** (B-243). Each kept statement is embedded, and
//!   its candidates are the active memories of its project and the kept
//!   statements before it in the part of the batch, for the same project,
//!   whose cosine with it is at least [`SAME_FLOOR`], the closest
//!   [`SAME_CANDIDATES`]. A statement with none is not asked about. The
//!   others are asked in calls ([`Judge::same`]) of at most
//!   [`SAME_CALL_MAX`] statements and [`SAME_CALL_CHARS`] characters,
//!   with the harness's ids: `s01` upward over the part's kept statements,
//!   `k01` upward for the memories by first appearance. A statement the
//!   step matches is kept unserved for 30 days with the id of what it
//!   repeats (A4); one whose answer is still bad after its call was halved
//!   to it alone and tried [`JUDGE_MAX_ATTEMPTS`] times is kept unserved as
//!   `unchecked` (A2). The capture's other statements settle as usual.
//! - **All or nothing per part** (B-244, A3). The daily caps and the
//!   daemon's stop are checked before each duplicate step call; reached,
//!   or a call that fails other than by a bad answer, and nothing of the
//!   part is settled: its captures wait whole, to be judged again.
//! - **Metrics.** `capture_statements_total` per statement with its
//!   outcome (`stored`, `repeat`, `hash`, `unchecked`, `duplicate`,
//!   `unquoted`), grade, type and project; `capture_judged_total` per
//!   capture with `task` `statements` and its count of statements; and
//!   `capture_judge_cost_usd` per call with `call` `statements` or `same`.

use std::collections::HashSet;
use std::ops::Range;

use sm_core::journal::StatementOutcome;
use sm_store::{Hold, JUDGE_MAX_ATTEMPTS, RepeatOf, StatementSettle, StatementSettled};

use super::*;

/// The cosine at or above which an active memory, or a kept statement
/// before it in the part, is a candidate of a statement in the duplicate
/// step (B-243). The harness's floor (g10, `FLOOR`), measured on the design
/// set and the first hold-out (B-239). Provisional; its metric is
/// `capture_statements_total` with `outcome` `repeat`.
pub(crate) const SAME_FLOOR: f32 = 0.80;

/// The most candidates one statement is asked with (the harness's `K`).
pub(crate) const SAME_CANDIDATES: usize = 5;

/// The most statements in one duplicate step call (the harness's).
pub(crate) const SAME_CALL_MAX: usize = 40;

/// Characters of statements and candidates in one duplicate step call
/// after its first statement (the harness's).
pub(crate) const SAME_CALL_CHARS: usize = 60_000;

/// The `call` label of each call's cost line.
const STATEMENTS_CALL: &str = "statements";
const SAME_CALL: &str = "same";

const LABELS: &[&str] = &["keep", "duplicate"];
const GRADES: &[&str] = &["must_keep", "good_to_have"];

/// One statement of an answer, checked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CheckedStatement {
    /// Labelled `keep`; `false` for `duplicate`.
    pub keep: bool,
    pub memory_type: MemoryType,
    pub grade: String,
    pub start: String,
    pub end: String,
}

/// Checks a statements answer against its batch and returns each
/// capture's statements, in the batch's order (B-241). Each capture must be
/// answered exactly once; each statement must have a known label, type and
/// grade. As [`check`]'s, the error names only the batch's ids, ids of
/// their shape, and the place of a statement in its unit (B-181).
pub(crate) fn check_units(
    candidates: &[JudgeCandidate],
    units: &[StatementUnit],
) -> Result<Vec<Vec<CheckedStatement>>, String> {
    let mut slots: Vec<Option<Vec<CheckedStatement>>> = vec![None; candidates.len()];
    for u in units {
        let Some(i) = candidates.iter().position(|c| c.id == u.id) else {
            return Err(if id_shaped(&u.id) {
                format!("an answer names {}, which is not in the batch", u.id)
            } else {
                "an answer names an id that is not in the batch".to_owned()
            });
        };
        if slots[i].is_some() {
            return Err(format!("{} was answered twice", u.id));
        }
        let mut out = Vec::with_capacity(u.statements.len());
        for (n, s) in u.statements.iter().enumerate() {
            let place = n + 1;
            if !LABELS.contains(&s.label.as_str()) {
                return Err(format!(
                    "statement {place} of {} has an unknown label",
                    u.id
                ));
            }
            let memory_type = s
                .memory_type
                .parse::<MemoryType>()
                .map_err(|_| format!("statement {place} of {} has an unknown type", u.id))?;
            if !GRADES.contains(&s.importance.as_str()) {
                return Err(format!(
                    "statement {place} of {} has an unknown importance",
                    u.id
                ));
            }
            out.push(CheckedStatement {
                keep: s.label == "keep",
                memory_type,
                grade: s.importance.clone(),
                start: s.start.clone(),
                end: s.end.clone(),
            });
        }
        slots[i] = Some(out);
    }
    slots
        .into_iter()
        .zip(candidates)
        .map(|(s, c)| s.ok_or_else(|| format!("{} was not answered", c.id)))
        .collect()
}

/// Checks a duplicate step answer against its items and returns, in the
/// items' order, the id of the candidate each matched, if any (B-246). Each
/// item must be answered exactly once, with a match among its own
/// candidates or none. The error names only ids (B-181).
pub(crate) fn check_same(
    items: &[SameItem],
    answers: &[SameMatch],
) -> Result<Vec<Option<String>>, String> {
    let mut slots: Vec<Option<Option<String>>> = vec![None; items.len()];
    for a in answers {
        let Some(i) = items.iter().position(|it| it.id == a.id) else {
            return Err(if shaped(&a.id, 's') {
                format!(
                    "a duplicate step answer names {}, which was not asked",
                    a.id
                )
            } else {
                "a duplicate step answer names an id that was not asked".to_owned()
            });
        };
        if slots[i].is_some() {
            return Err(format!("{} was answered twice", a.id));
        }
        if let Some(m) = &a.matched
            && !items[i].candidates.iter().any(|c| &c.id == m)
        {
            return Err(format!(
                "the answer on {} matches an id that is not among its candidates",
                a.id
            ));
        }
        slots[i] = Some(a.matched.clone());
    }
    slots
        .into_iter()
        .zip(items)
        .map(|(s, it)| s.ok_or_else(|| format!("{} was not answered", it.id)))
        .collect()
}

/// A candidate of a statement in the duplicate step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum Cand {
    /// An active memory of the statement's project.
    Active(MemoryId),
    /// A kept statement before it in the part: its place among them.
    Earlier(usize),
}

/// The duplicate step's items for a part (B-243).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Asked {
    /// One per statement with a candidate, in the statements' order.
    pub items: Vec<SameItem>,
    /// For each item, its statement's place among the kept statements.
    pub of: Vec<usize>,
    /// What each candidate id names.
    pub names: HashMap<String, Cand>,
}

/// The items of the duplicate step, with the harness's ids (g10 `dup`):
/// `s01` upward for the kept statements of the part, in order, and `k01`
/// upward for the active memories, by first appearance. `tops` gives each
/// statement's candidates, closest first; an active memory whose text is
/// not in `active` (forgotten since the lookup) is left out, and a
/// statement left with no candidate is not asked about.
pub(crate) fn same_items(
    statements: &[String],
    tops: &[Vec<(Cand, f32)>],
    active: &HashMap<MemoryId, String>,
) -> Asked {
    let sid = |j: usize| format!("s{:02}", j + 1);
    let mut names: HashMap<String, Cand> = (0..statements.len())
        .map(|j| (sid(j), Cand::Earlier(j)))
        .collect();
    let mut kid: HashMap<MemoryId, String> = HashMap::new();
    let mut items = Vec::new();
    let mut of = Vec::new();
    for (j, top) in tops.iter().enumerate() {
        let mut candidates = Vec::with_capacity(top.len());
        for (c, _) in top {
            match *c {
                Cand::Earlier(k) => candidates.push(SameCandidate {
                    id: sid(k),
                    text: statements[k].clone(),
                }),
                Cand::Active(id) => {
                    let Some(text) = active.get(&id) else {
                        continue;
                    };
                    let next = format!("k{:02}", kid.len() + 1);
                    let k = kid.entry(id).or_insert(next).clone();
                    names.insert(k.clone(), Cand::Active(id));
                    candidates.push(SameCandidate {
                        id: k,
                        text: text.clone(),
                    });
                }
            }
        }
        if candidates.is_empty() {
            continue;
        }
        items.push(SameItem {
            id: sid(j),
            statement: statements[j].clone(),
            candidates,
        });
        of.push(j);
    }
    Asked { items, of, names }
}

/// The duplicate step's calls, as the harness split them: a new call
/// before an item when the current one holds [`SAME_CALL_MAX`] items or
/// the item's characters, its statement's and its candidates', would take
/// it past [`SAME_CALL_CHARS`].
pub(crate) fn split_calls(items: &[SameItem]) -> Vec<Range<usize>> {
    let mut out = Vec::new();
    let (mut start, mut size) = (0, 0);
    for (i, it) in items.iter().enumerate() {
        let n = it.statement.chars().count()
            + it.candidates
                .iter()
                .map(|c| c.text.chars().count())
                .sum::<usize>();
        if i > start && (i - start >= SAME_CALL_MAX || size + n > SAME_CALL_CHARS) {
            out.push(start..i);
            start = i;
            size = 0;
        }
        size += n;
    }
    if start < items.len() {
        out.push(start..items.len());
    }
    out
}

/// How one part of a batch ended.
#[derive(Debug)]
enum Part {
    /// Settled; the last bad answer when a statement was held unchecked.
    Done(Option<JudgeFailure>),
    /// The statements answer failed its checks: counted, nothing settled.
    Bad(JudgeFailure),
    /// A call failed some other way, or was stopped: counted and, unless
    /// stopped, backed off. Nothing of the part was settled.
    Failed,
    /// A daily cap was reached, or the daemon is stopping, before a
    /// duplicate step call: nothing of the part was settled (B-244).
    Paused,
}

/// Judges one session's batch under the statements task. A bad statements
/// answer halves it, the first half first, down to a single capture, which
/// is tried twice and then marked failed judgement (B-206, as under the
/// verdicts task). Another failure, a cap or a stop ends the batch; what
/// was not settled stays queued.
pub(super) fn call(
    sh: &Shared,
    judge: &dyn Judge,
    ready: &Ready,
    day: &str,
    items: &[JudgeItem],
) -> CoreResult<()> {
    let mut todo: Vec<(&[JudgeItem], u8)> = vec![(items, 0)];
    let mut day = day.to_owned();
    let mut first = true;
    let mut marked: Option<JudgeFailure> = None;
    let mut unchecked: Option<JudgeFailure> = None;
    while let Some((part, tries)) = todo.pop() {
        if !first {
            day = Timestamp::now().date_string();
            if !may_call(sh, &day)? {
                break;
            }
        }
        first = false;
        match attempt(sh, judge, ready, &day, part)? {
            Part::Done(u) => {
                if u.is_some() {
                    unchecked = u;
                }
            }
            Part::Failed => return Ok(()),
            Part::Paused => break,
            Part::Bad(f) => {
                bad_call(sh, &day, part, tries, &f, Some(STATEMENTS_CALL))?;
                if part.len() > 1 {
                    let (a, b) = part.split_at(part.len() / 2);
                    todo.push((b, 0));
                    todo.push((a, 0));
                } else if tries == 0 {
                    todo.push((part, 1));
                } else {
                    let ids: Vec<MemoryId> = part.iter().map(|i| i.id).collect();
                    sh.store.mark_failed_judgement(&ids)?;
                    sh.log.line(format!(
                        "judge: capture {} was answered badly twice alone; it is marked failed \
                         judgement and stays pending, unserved, until it is pruned",
                        part[0].id
                    ));
                    marked = Some(f);
                }
            }
        }
    }
    if let Some(f) = marked.or(unchecked) {
        let retry = back_off(sh, &f);
        sh.log.line(format!(
            "judge: a capture was marked failed judgement or a statement was held unchecked; \
             next call at {}",
            retry.to_rfc3339()
        ));
    }
    Ok(())
}

/// One statements call about `items`, and the duplicate step's calls.
fn attempt(
    sh: &Shared,
    judge: &dyn Judge,
    ready: &Ready,
    day: &str,
    items: &[JudgeItem],
) -> CoreResult<Part> {
    let candidates = candidates_of(&sh.scanner, items);
    let answer = {
        let _busy = Busy::start(sh);
        judge.statements(&candidates, &sh.stop)
    };
    let checked = answer.and_then(|a| match check_units(&candidates, &a.units) {
        Ok(c) => Ok((a, c)),
        Err(message) => Err(JudgeFailure {
            kind: JudgeFailureKind::BadAnswer,
            message,
            retry_at: None,
            cost_usd: a.cost_usd,
        }),
    });
    let (answer, units) = match checked {
        Ok(v) => v,
        Err(f) => {
            return Ok(
                match fail(sh, judge, day, items, f, Some(STATEMENTS_CALL))? {
                    Attempt::Bad(f) => Part::Bad(f),
                    Attempt::Done | Attempt::Failed => Part::Failed,
                },
            );
        }
    };
    sh.store
        .record_judge_call(day, items.len() as u64, false, answer.cost_usd)?;
    cost_line(
        sh,
        STATEMENTS_CALL,
        items.len(),
        answer.duration_ms,
        answer.cost_usd,
    );
    let model = answer
        .model
        .clone()
        .unwrap_or_else(|| judge.model().to_owned());
    settle(sh, judge, ready, items, &candidates, &units, &model)
}

fn cost_line(sh: &Shared, call: &str, n: usize, duration_ms: u64, cost_usd: f64) {
    let mut labels = BTreeMap::new();
    labels.insert("call".to_owned(), Value::from(call));
    labels.insert("candidates".to_owned(), Value::from(n));
    labels.insert("duration_ms".to_owned(), Value::from(duration_ms));
    sh.journal.record(
        REQUEST_ID,
        metrics::CAPTURE_JUDGE_COST_USD,
        cost_usd,
        labels,
    );
}

/// One kept statement of a part, on its way to the store.
struct Kept {
    /// Its capture's place in the part.
    item: usize,
    /// Its place in its capture's answer.
    index: u32,
    project: ProjectId,
    content: String,
    title: Option<String>,
    memory_type: MemoryType,
    grade: String,
    chunks: Vec<NewChunk>,
}

/// What a statement journals: its outcome, grade and type.
#[derive(Debug, Clone)]
struct Line {
    outcome: StatementOutcome,
    grade: String,
    memory_type: MemoryType,
}

/// Settles a checked statements answer for one part (B-242 to B-244).
fn settle(
    sh: &Shared,
    judge: &dyn Judge,
    ready: &Ready,
    items: &[JudgeItem],
    candidates: &[JudgeCandidate],
    units: &[Vec<CheckedStatement>],
    model: &str,
) -> CoreResult<Part> {
    let cap = Some(sh.settings.capture.max_index_tokens as usize);
    // Per capture: the lines of the statements not passed to the store.
    let mut early: Vec<Vec<Line>> = vec![Vec::new(); items.len()];
    let mut kept: Vec<Kept> = Vec::new();
    let mut unchunked: Vec<usize> = Vec::new();
    for (i, (item, unit)) in items.iter().zip(units).enumerate() {
        let mut mine = Vec::new();
        let mut chunked = true;
        for (n, st) in unit.iter().enumerate() {
            let line = |outcome| Line {
                outcome,
                grade: st.grade.clone(),
                memory_type: st.memory_type,
            };
            if !st.keep {
                early[i].push(line(StatementOutcome::Duplicate));
                continue;
            }
            let quoted = if st.start.trim().is_empty() || st.end.trim().is_empty() {
                String::new()
            } else {
                spanned(
                    &candidates[i].text,
                    std::iter::once((st.start.as_str(), st.end.as_str())),
                )
                .memory
            };
            if quoted.is_empty() {
                early[i].push(line(StatementOutcome::Unquoted));
                continue;
            }
            let content = sh.scanner.redact(&quoted).text;
            let title = derive_title(&content);
            match guarded(|| pending_chunks(sh, ready, &content, cap)) {
                Ok((chunks, _)) => mine.push(Kept {
                    item: i,
                    index: u32::try_from(n).unwrap_or(u32::MAX),
                    project: item.project,
                    content,
                    title,
                    memory_type: st.memory_type,
                    grade: st.grade.clone(),
                    chunks,
                }),
                Err(e) => {
                    sh.log.line(format!(
                        "judge: capture {}: a statement cannot be chunked: {}: {}; it stays \
                         pending",
                        item.id,
                        e.code.as_str(),
                        e.message
                    ));
                    chunked = false;
                    break;
                }
            }
        }
        if chunked {
            kept.extend(mine);
        } else {
            unchunked.push(i);
        }
    }

    let means = guarded(|| {
        let mut docs: Vec<(Option<&str>, &mut [NewChunk])> = kept
            .iter_mut()
            .map(|k| (k.title.as_deref(), k.chunks.as_mut_slice()))
            .collect();
        embed_docs(sh, ready, &mut docs)
    });
    let means = match means {
        Ok(m) => m,
        Err(e) => {
            // Without vectors there is no duplicate step: nothing of the
            // part is settled, and each capture counts an attempt.
            sh.log.line(format!(
                "judge: cannot embed the statements of {} captures for the duplicate step: {}: \
                 {}; they stay pending",
                items.len(),
                e.code.as_str(),
                e.message
            ));
            let ids: Vec<MemoryId> = items.iter().map(|i| i.id).collect();
            sh.store.judge_failed(&ids)?;
            return Ok(Part::Done(None));
        }
    };

    let tops = candidates_near(sh, &kept, &means)?;
    let active: Vec<MemoryId> = {
        let mut seen = HashSet::new();
        tops.iter()
            .flatten()
            .filter_map(|(c, _)| match c {
                Cand::Active(id) => seen.insert(*id).then_some(*id),
                Cand::Earlier(_) => None,
            })
            .collect()
    };
    let texts: HashMap<MemoryId, String> = if active.is_empty() {
        HashMap::new()
    } else {
        sh.store
            .get_memories(&active)?
            .into_iter()
            .map(|m| {
                let text = sh.scanner.redact(&m.content).text;
                (m.id, clip_chars(&text, MAX_REWRITE_CHARS))
            })
            .collect()
    };
    let contents: Vec<String> = kept.iter().map(|k| k.content.clone()).collect();
    let asked = same_items(&contents, &tops, &texts);

    let (matches, unchecked, last_bad) =
        match same_step(sh, judge, items.len(), &asked, kept.len())? {
            Step::Answered {
                matches,
                unchecked,
                last_bad,
            } => (matches, unchecked, last_bad),
            Step::Failed => return Ok(Part::Failed),
            Step::Paused => return Ok(Part::Paused),
        };
    answered(sh);

    // The store's input, one entry per capture that was chunked, and what
    // each kept statement journals.
    let place: Vec<(usize, u32)> = kept.iter().map(|k| (k.item, k.index)).collect();
    let mut per_item: Vec<Vec<StatementSettle>> = vec![Vec::new(); items.len()];
    let mut stored_lines: Vec<Vec<Line>> = vec![Vec::new(); items.len()];
    for (j, k) in kept.into_iter().enumerate() {
        let hold = if unchecked[j] {
            Some(Hold::Unchecked)
        } else {
            matches[j].map(|c| {
                Hold::Repeat(match c {
                    Cand::Active(id) => RepeatOf::Memory(id),
                    Cand::Earlier(e) => RepeatOf::Statement {
                        capture: items[place[e].0].id,
                        index: place[e].1,
                    },
                })
            })
        };
        stored_lines[k.item].push(Line {
            outcome: StatementOutcome::Stored,
            grade: k.grade.clone(),
            memory_type: k.memory_type,
        });
        per_item[k.item].push(StatementSettle {
            index: k.index,
            content: k.content,
            title: k.title,
            memory_type: k.memory_type,
            chunks: k.chunks,
            grade: k.grade,
            hold,
        });
    }
    let order: Vec<usize> = (0..items.len())
        .filter(|i| !unchunked.contains(i))
        .collect();
    let settles: Vec<Settle> = order
        .iter()
        .map(|&i| Settle::Statements {
            id: items[i].id,
            judge_model: model.to_owned(),
            statements: std::mem::take(&mut per_item[i]),
        })
        .collect();
    let settled = sh.store.settle_captures(settles)?;
    sh.after_write();
    if !unchunked.is_empty() {
        let ids: Vec<MemoryId> = unchunked.iter().map(|&i| items[i].id).collect();
        sh.store.judge_failed(&ids)?;
    }

    let mut names: HashMap<ProjectId, String> = HashMap::new();
    for (i, outcome) in order.into_iter().zip(settled) {
        let Settled::Statements(results) = outcome else {
            continue;
        };
        let project = items[i].project;
        let name = match names.get(&project) {
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
        let mut lines = std::mem::take(&mut early[i]);
        for (mut line, r) in std::mem::take(&mut stored_lines[i])
            .into_iter()
            .zip(&results)
        {
            line.outcome = match r {
                StatementSettled::Stored(_) => StatementOutcome::Stored,
                StatementSettled::Held {
                    repeat_of: Some(_), ..
                } => StatementOutcome::Repeat,
                StatementSettled::Held {
                    repeat_of: None, ..
                } => StatementOutcome::Unchecked,
                StatementSettled::Duplicate { .. } => StatementOutcome::Hash,
            };
            lines.push(line);
        }
        let has = |o: StatementOutcome| lines.iter().any(|l| l.outcome == o);
        let reason = if has(StatementOutcome::Stored) {
            "knowledge"
        } else if has(StatementOutcome::Unchecked) {
            "unchecked"
        } else if has(StatementOutcome::Repeat) {
            "repeats"
        } else if has(StatementOutcome::Hash) {
            "duplicate"
        } else {
            "nothing_kept"
        };
        for l in &lines {
            let mut labels = BTreeMap::new();
            labels.insert("outcome".to_owned(), Value::from(l.outcome.as_str()));
            labels.insert("grade".to_owned(), Value::from(l.grade.as_str()));
            labels.insert("type".to_owned(), Value::from(l.memory_type.as_str()));
            labels.insert("project".to_owned(), Value::from(name.as_str()));
            sh.journal
                .record(REQUEST_ID, metrics::CAPTURE_STATEMENTS_TOTAL, 1.0, labels);
        }
        let mut labels = BTreeMap::new();
        labels.insert("project".to_owned(), Value::from(name.as_str()));
        labels.insert(
            "verdict".to_owned(),
            Value::from(if reason == "knowledge" {
                "keep"
            } else {
                "reject"
            }),
        );
        labels.insert("reason".to_owned(), Value::from(reason));
        labels.insert("statements".to_owned(), Value::from(units[i].len()));
        labels.insert("task".to_owned(), Value::from(STATEMENTS_CALL));
        sh.journal
            .record(REQUEST_ID, metrics::CAPTURE_JUDGED_TOTAL, 1.0, labels);
    }
    Ok(Part::Done(last_bad))
}

/// Each kept statement's candidates, closest first, at most
/// [`SAME_CANDIDATES`]: the active memories of its project and the kept
/// statements before it in the part for the same project, at cosine
/// [`SAME_FLOOR`] or more.
fn candidates_near(
    sh: &Shared,
    kept: &[Kept],
    means: &[Vec<f32>],
) -> CoreResult<Vec<Vec<(Cand, f32)>>> {
    let mut by_project: BTreeMap<ProjectId, Vec<usize>> = BTreeMap::new();
    for (j, k) in kept.iter().enumerate() {
        by_project.entry(k.project).or_default().push(j);
    }
    let mut found: Vec<Vec<Nearest>> = vec![Vec::new(); kept.len()];
    for (project, indexes) in by_project {
        let vectors: Vec<Vec<f32>> = indexes.iter().map(|&j| means[j].clone()).collect();
        let near = sh
            .store
            .active_near(project, &vectors, SAME_FLOOR, SAME_CANDIDATES)?;
        for (&j, n) in indexes.iter().zip(near) {
            found[j] = n;
        }
    }
    Ok((0..kept.len())
        .map(|j| {
            let mut top: Vec<(Cand, f32)> = found[j]
                .iter()
                .map(|n| (Cand::Active(n.id), n.cosine))
                .chain(
                    (0..j)
                        .filter(|&k| kept[k].project == kept[j].project)
                        .map(|k| (Cand::Earlier(k), cosine(&means[k], &means[j])))
                        .filter(|(_, c)| *c >= SAME_FLOOR),
                )
                .collect();
            top.sort_by(|a, b| b.1.total_cmp(&a.1));
            top.truncate(SAME_CANDIDATES);
            top
        })
        .collect())
}

/// How the duplicate step ended for a part.
enum Step {
    /// Every item answered or held unchecked: for each kept statement, what
    /// it matched and whether it is unchecked; the last bad answer when one
    /// is.
    Answered {
        matches: Vec<Option<Cand>>,
        unchecked: Vec<bool>,
        last_bad: Option<JudgeFailure>,
    },
    Failed,
    Paused,
}

/// The duplicate step's calls for a part (B-243, B-244). Before each call
/// the caps and the stop are checked. A bad answer halves the call, the
/// first half first; a single statement is tried [`JUDGE_MAX_ATTEMPTS`]
/// times and then held unchecked (A2). Any other failure ends the step.
fn same_step(
    sh: &Shared,
    judge: &dyn Judge,
    captures: usize,
    asked: &Asked,
    statements: usize,
) -> CoreResult<Step> {
    let place: HashMap<&str, usize> = asked
        .items
        .iter()
        .zip(&asked.of)
        .map(|(it, &j)| (it.id.as_str(), j))
        .collect();
    let mut matches: Vec<Option<Cand>> = vec![None; statements];
    let mut unchecked = vec![false; statements];
    let mut last_bad = None;
    let mut todo: Vec<(&[SameItem], u32)> = split_calls(&asked.items)
        .into_iter()
        .rev()
        .map(|r| (&asked.items[r], 0))
        .collect();
    while let Some((part, tries)) = todo.pop() {
        let day = Timestamp::now().date_string();
        if !may_call(sh, &day)? {
            sh.log.line(format!(
                "judge: a daily cap was reached, or the daemon is stopping, before a duplicate \
                 step call; the {captures} captures of the part stay queued whole (B-244)"
            ));
            return Ok(Step::Paused);
        }
        let answer = {
            let _busy = Busy::start(sh);
            judge.same(part, &sh.stop)
        };
        let checked = answer.and_then(|a| match check_same(part, &a.answers) {
            Ok(c) => Ok((a, c)),
            Err(message) => Err(JudgeFailure {
                kind: JudgeFailureKind::BadAnswer,
                message,
                retry_at: None,
                cost_usd: a.cost_usd,
            }),
        });
        match checked {
            Ok((a, got)) => {
                sh.store.record_judge_call(&day, 0, false, a.cost_usd)?;
                cost_line(sh, SAME_CALL, part.len(), a.duration_ms, a.cost_usd);
                for (it, m) in part.iter().zip(got) {
                    let j = place[it.id.as_str()];
                    matches[j] = m.and_then(|id| asked.names.get(&id).copied());
                }
            }
            Err(f) if f.kind == JudgeFailureKind::BadAnswer => {
                judge.keep_failure(&f);
                count_failure(sh, &day, part.len(), &f, Some(SAME_CALL))?;
                {
                    let mut rt = lock(&sh.judge_state);
                    rt.last_error = Some(f.message.clone());
                    rt.last_kind = Some(f.kind);
                }
                let next = if part.len() > 1 {
                    let (a, b) = part.split_at(part.len() / 2);
                    todo.push((b, 0));
                    todo.push((a, 0));
                    "each half is asked again"
                } else if tries + 1 < JUDGE_MAX_ATTEMPTS {
                    todo.push((part, tries + 1));
                    "it is asked again alone"
                } else {
                    unchecked[place[part[0].id.as_str()]] = true;
                    last_bad = Some(f.clone());
                    "it is held unchecked, unserved, for 30 days"
                };
                sh.log.line(format!(
                    "judge: the duplicate step's answer about {} statements was bad: {}; {next}",
                    part.len(),
                    f.message
                ));
            }
            Err(f) => {
                if f.kind != JudgeFailureKind::Stopped {
                    judge.keep_failure(&f);
                }
                count_failure(sh, &day, part.len(), &f, Some(SAME_CALL))?;
                if f.kind != JudgeFailureKind::Stopped {
                    let retry = back_off(sh, &f);
                    sh.log.line(format!(
                        "judge: a duplicate step call about {} statements failed ({}): {}; the \
                         {captures} captures of the part stay pending; next call at {}",
                        part.len(),
                        f.kind.as_str(),
                        f.message,
                        retry.to_rfc3339()
                    ));
                }
                return Ok(Step::Failed);
            }
        }
    }
    Ok(Step::Answered {
        matches,
        unchecked,
        last_bad,
    })
}
