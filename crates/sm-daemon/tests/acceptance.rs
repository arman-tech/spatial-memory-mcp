//! The acceptance tests that reach the daemon, over the real socket with
//! the fake embedder (SPEC 14, stream E, Tests; SPEC 11): AT-3 (daemon
//! side), AT-4, AT-5, AT-9 (`remember` side), AT-10, AT-11, AT-12 and
//! AT-16's ranking clause; the SPEC 6 recall rules (identifier queries,
//! weak results); the SPEC 7 map cache; and B-79's sharing seam dropping
//! the maps.
//!
//! Every daemon runs on a temporary profile with the endpoint passed in,
//! so nothing here reads the real profile folder or the environment.

mod common;

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use common::*;
use serde_json::Value;
use sm_core::config::ProfileName;
use sm_core::error::{ErrorCode, Result};
use sm_core::ids::MemoryId;
use sm_core::journal::HookGate;
use sm_core::paths::{Endpoint, ProfilePaths};
use sm_core::protocol::{
    AmbientParams, AmbientReply, DaemonState, ForgetParams, ImportParams, Op, RecallParams,
    RecallReply, RegionsParams, RegionsReply, ReindexParams, RememberItem, RememberParams,
    RememberReply, Reply, Request, ReviseParams,
};
use sm_core::testing::{FakeEmbedder, SimpleChunker, dot, fake_info, fake_vector};
use sm_core::traits::{DocInput, Embedder, EmbedderInfo, EmbedderProvider, SessionKind};
use sm_core::types::{IndexStamp, MatchKind, RecallMode, Scope};
use sm_daemon::{LoadedModel, ModelLoader, SharingOps};
use sm_embed::ModelFile;
use sm_embed::model::{hash_file, verify_files};

// --- helpers ---------------------------------------------------------------

fn query(q: &str, mode: RecallMode, k: u32) -> RecallParams {
    let mut p = RecallParams::new(q);
    p.mode = Some(mode);
    p.k = Some(k);
    p
}

fn ids(r: &RecallReply) -> Vec<MemoryId> {
    r.results.iter().map(|v| v.id).collect()
}

fn remember_many(d: &Daemon, cwd: &str, items: Vec<RememberItem>) -> RememberReply {
    match d.ok(cwd, Op::Remember(RememberParams { memories: items })) {
        Reply::Remember(r) => r,
        other => panic!("remember replied {other:?}"),
    }
}

/// `regions` with one central memory per region and a budget large enough
/// that nothing is cut, so the reply is the whole run.
fn regions(d: &Daemon, cwd: &str, coords: bool) -> RegionsReply {
    match d.ok(
        cwd,
        Op::Regions(RegionsParams {
            coords,
            central: Some(1),
            label: None,
            limit: Some(100_000),
        }),
    ) {
        Reply::Regions(r) => r,
        other => panic!("regions replied {other:?}"),
    }
}

/// Nearest-rank percentile of sorted durations.
fn percentile(sorted: &[Duration], p: f64) -> Duration {
    let n = sorted.len();
    let rank = ((p / 100.0) * n as f64).ceil() as usize;
    sorted[rank.clamp(1, n) - 1]
}

/// Version 1's placeholders `[REDACTED_<KIND>]` written as version 3's
/// markers `[REDACTED:<kind>]` (`redaction_patterns.json`, `v3_mapping`).
fn v3_markers(s: &str) -> String {
    const OPEN: &str = "[REDACTED_";
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find(OPEN) {
        out.push_str(&rest[..i]);
        let after = &rest[i + OPEN.len()..];
        let kind_len = after
            .find(|c: char| !(c.is_ascii_uppercase() || c == '_'))
            .unwrap_or(after.len());
        if kind_len > 0 && after[kind_len..].starts_with(']') {
            out.push_str(&format!(
                "[REDACTED:{}]",
                after[..kind_len].to_ascii_lowercase()
            ));
            rest = &after[kind_len + 1..];
        } else {
            out.push_str(OPEN);
            rest = after;
        }
    }
    out.push_str(rest);
    out
}

fn strings(v: &Value) -> Vec<String> {
    v.as_array()
        .map(|l| {
            l.iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

// --- SPEC 7: the map cache --------------------------------------------------

/// SPEC 7 and AT-6 at the daemon: `regions` twice on the same data returns
/// the cached run with identical assignments; coordinates are computed only
/// when asked; an edit drops the cached run; the same data after a restart
/// is served from the stored run.
#[test]
fn s7_regions_cached_until_an_edit_and_coordinates_only_when_asked() {
    let rig = Rig::new();
    let cwd = rig.project("maps");
    let d = rig.start_fake();
    let topics = [
        "postgres index vacuum autovacuum bloat",
        "react component render hook state",
        "docker image layer cache build",
        "rust borrow lifetime trait generic",
    ];
    let mut all = Vec::new();
    for (t, topic) in topics.iter().enumerate() {
        for i in 0..10 {
            all.push(remember(
                &d,
                &cwd,
                &format!("{topic} note{i} topic{t} detail{i}"),
            ));
        }
    }
    d.wait_embedded(&cwd);

    let first = regions(&d, &cwd, false);
    assert!(!first.cached, "the first run is computed");
    assert_eq!(first.memories, 40);
    assert!(first.points.is_none(), "no coordinates unless asked");
    let second = regions(&d, &cwd, false);
    assert!(second.cached, "the same data gives the cached run");
    assert_eq!(second.run, first.run);
    assert_eq!(second.regions, first.regions, "identical assignments");
    assert_eq!(second.unclustered, first.unclustered);

    let with = regions(&d, &cwd, true);
    assert!(!with.cached, "coordinates are computed on the first ask");
    assert!(!with.truncated, "the whole map fits the budget");
    let points = with.points.as_ref().expect("coordinates when asked");
    let point_ids: HashSet<MemoryId> = points.iter().map(|p| p.id).collect();
    assert_eq!(point_ids, all.iter().copied().collect::<HashSet<_>>());
    let again = regions(&d, &cwd, true);
    assert!(again.cached);
    assert_eq!(again.run, with.run);
    assert_eq!(again.points, with.points);
    let plain = regions(&d, &cwd, false);
    assert!(
        plain.cached,
        "a plain request is served by the coordinates run"
    );
    assert!(plain.points.is_none());

    let mut edit = ReviseParams::new(all[0]);
    edit.title = Some("a renamed postgres note".to_owned());
    match d.ok(&cwd, Op::Revise(edit)) {
        Reply::Revise(_) => {}
        other => panic!("revise replied {other:?}"),
    }
    d.wait_embedded(&cwd);
    let edited = regions(&d, &cwd, false);
    assert!(!edited.cached, "an edit drops the cached run");
    assert_ne!(edited.run, first.run);
    assert_ne!(edited.run, with.run);
    let edited_again = regions(&d, &cwd, false);
    assert!(edited_again.cached);
    assert_eq!(edited_again.run, edited.run);

    drop(d);
    let d = rig.start_fake();
    let restarted = regions(&d, &cwd, false);
    assert!(restarted.cached, "the stored run survives a restart");
    assert_eq!(restarted.run, edited.run);
    assert_eq!(restarted.regions, edited.regions);
}

/// B-79 and SPEC 9: after a sharing operation writes, the daemon drops the
/// maps and wakes the queue, so the imported memory is embedded and the
/// next `regions` is a new run that includes it.
#[test]
fn s9_import_drops_the_maps_and_its_memory_is_embedded() {
    let rig = Rig::new();
    let cwd = rig.project("shared");
    let sharing = RecordingSharing::new();
    let d = rig.start(
        FakeLoader::new(),
        Arc::clone(&sharing) as Arc<dyn SharingOps>,
    );
    d.wait_ready();
    for i in 0..5 {
        remember(
            &d,
            &cwd,
            &format!("deploy note {i} about the staging cluster"),
        );
    }
    d.wait_embedded(&cwd);
    let before = regions(&d, &cwd, false);
    assert!(regions(&d, &cwd, false).cached);

    match d.ok(
        &cwd,
        Op::Import(ImportParams {
            path: rig.tmp.path().join("backup.json").display().to_string(),
        }),
    ) {
        Reply::Import(r) => assert_eq!(r.inserted, 1),
        other => panic!("import replied {other:?}"),
    }
    d.wait_embedded(&cwd);
    let after = regions(&d, &cwd, false);
    assert!(!after.cached, "the import dropped the maps");
    assert_ne!(after.run, before.run);
    assert_eq!(after.memories, before.memories + 1, "the import is mapped");

    let calls = sharing.calls();
    assert_eq!(calls.len(), 1, "{calls:?}");
    assert_eq!(calls[0].op, "import");
    assert!(calls[0].had_chunker);
}

// --- SPEC 6: recall ---------------------------------------------------------

/// SPEC 6: an identifier query runs the keyword search too; its hits come
/// after the vector hits, without repeats, marked `keyword` with no score.
/// A plain query and vector mode run no keyword search.
#[test]
fn s6_identifier_query_appends_marked_keyword_hits_after_vector_hits() {
    let rig = Rig::new();
    let cwd = rig.project("identifiers");
    let d = rig.start_fake();
    // The vector neighbours hold both words but not the phrase
    // `billing service`, so only `named` is a keyword hit.
    let exact_text = "service billing";
    let close_text = "service billing notes";
    let named_text = "the billing_service crate owns invoices ledgers payouts taxes \
                      currencies receipts statements and dunning";
    let far_text = "unrelated note about garden tomatoes and summer rain";
    let exact = remember(&d, &cwd, exact_text);
    let close = remember(&d, &cwd, close_text);
    let named = remember(&d, &cwd, named_text);
    remember(&d, &cwd, far_text);
    d.wait_embedded(&cwd);

    let q = "billing_service";
    let qv = fake_vector(q);
    let score = |t: &str| dot(&qv, &fake_vector(t));
    assert!(score(exact_text) > score(close_text));
    assert!(
        score(close_text) > score(named_text),
        "named is not a top-2 vector hit"
    );
    assert!(score(close_text) > score(far_text));

    let r = recall(&d, &cwd, query(q, RecallMode::Auto, 2));
    assert_eq!(ids(&r), vec![exact, close, named]);
    for (i, v) in r.results.iter().enumerate() {
        assert_eq!(v.rank, Some(i as u32 + 1));
    }
    for v in &r.results[..2] {
        assert_eq!(v.matched_by, Some(MatchKind::Vector));
        assert!(v.score.is_some());
    }
    let kw = &r.results[2];
    assert_eq!(kw.matched_by, Some(MatchKind::Keyword), "marked");
    assert_eq!(kw.score, None);
    assert!(!kw.weak, "a keyword hit has no score to be weak by");

    let v = recall(&d, &cwd, query(q, RecallMode::Vector, 2));
    assert_eq!(
        ids(&v),
        vec![exact, close],
        "vector mode runs no keyword search"
    );
    let plain = recall(&d, &cwd, query("billing service", RecallMode::Auto, 2));
    assert_eq!(ids(&plain), vec![exact, close], "not an identifier");
    let k = recall(&d, &cwd, query(q, RecallMode::Keyword, 5));
    assert_eq!(ids(&k), vec![named]);
    assert_eq!(k.results[0].matched_by, Some(MatchKind::Keyword));
}

/// SPEC 6: results under the similarity floor are returned and marked
/// weak, not dropped; `weak_only` says when every result is weak.
#[test]
fn s6_weak_results_are_marked_not_dropped() {
    let rig = Rig::new();
    let cwd = rig.project("weak");
    let d = rig.start_fake();
    let texts = [
        "release checklist for the billing deploy",
        "release checklist for the billing deploy on friday",
        "garden tomatoes need summer rain",
        "the cat sat on the warm mat",
    ];
    for t in texts {
        remember(&d, &cwd, t);
    }
    d.wait_embedded(&cwd);
    let floor = rig.settings.recall.similarity_floor as f32;

    let q = texts[0];
    let mixed = recall(&d, &cwd, query(q, RecallMode::Auto, 4));
    assert_eq!(mixed.results.len(), 4, "nothing is dropped");
    let mut weak = 0;
    for v in &mixed.results {
        let s = v.score.expect("a vector hit has a score");
        assert_eq!(v.weak, s < floor, "{v:?}");
        weak += usize::from(v.weak);
    }
    assert!((1..4).contains(&weak), "some weak, some strong: {weak}");
    assert!(!mixed.weak_only);

    let q = "quantum lighthouse zebra orchestra";
    let qv = fake_vector(q);
    for t in texts {
        assert!(
            dot(&qv, &fake_vector(t)) < floor,
            "every text is under the floor"
        );
    }
    let all_weak = recall(&d, &cwd, query(q, RecallMode::Auto, 4));
    assert_eq!(all_weak.results.len(), 4, "weak results are still returned");
    assert!(all_weak.results.iter().all(|v| v.weak));
    assert!(all_weak.weak_only);
}

// --- AT-3: hook latency, idle and during a bulk import ----------------------

/// Times `n` warm `ambient_recall` calls, each over a new connection, as
/// the hook makes them; sorted.
fn timed(d: &Daemon, req: &dyn Fn() -> Request, n: usize) -> Vec<Duration> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let r = req();
        let t = Instant::now();
        let resp = d.send(&r);
        out.push(t.elapsed());
        match resp.into_result() {
            Ok(Reply::Ambient(a)) => assert!(a.text.is_some(), "a block: {a:?}"),
            other => panic!("ambient_recall replied {other:?}"),
        }
    }
    out.sort();
    out
}

/// AT-3, daemon side: 100 warm `ambient_recall` calls meet the hook budget
/// (95th percentile ≤ 300 ms), and meet it again while a bulk import of at
/// least 2,000 chunks is being embedded; the bulk session never runs more
/// threads than `cpu.budget` allows. The fake embedder sleeps per document
/// so the import outlasts the measurement. The real model's figures are
/// stream C's (`ThreadReport`) and the owner's.
#[test]
fn at3_warm_ambient_latency_idle_and_during_a_bulk_import_within_the_thread_budget() {
    const BUDGET: Duration = Duration::from_millis(300);
    let rig = Rig::new();
    let cwd = rig.project("latency");
    let provider = RecordingProvider::new(Duration::from_millis(3));
    let d = rig.start(
        FakeLoader::with(Arc::clone(&provider) as Arc<dyn EmbedderProvider>),
        Arc::new(UnwiredSharing),
    );
    d.wait_ready();
    let prompt = "release checklist steps for the billing service deploy";
    for i in 0..5 {
        remember(&d, &cwd, &format!("{prompt} variant {i}"));
    }
    d.wait_embedded(&cwd);
    let req = || {
        d.req(
            &cwd,
            Op::AmbientRecall(AmbientParams {
                prompt: prompt.to_owned(),
                top_k: 3,
                scope: Scope::Project,
            }),
        )
    };
    timed(&d, &req, 5);
    let idle = timed(&d, &req, 100);

    let mut chunks = 0u32;
    for b in 0..20 {
        let items = (0..100)
            .map(|i| RememberItem::new(format!("bulk import record {b} {i} for the archive")))
            .collect();
        chunks += remember_many(&d, &cwd, items)
            .memories
            .iter()
            .map(|m| m.chunks)
            .sum::<u32>();
    }
    assert!(
        chunks >= 2000,
        "a bulk import of at least 2,000 chunks: {chunks}"
    );
    assert!(
        d.status(&cwd).index.pending_embeddings > 0,
        "the import runs"
    );
    let busy = timed(&d, &req, 100);
    let left = d.status(&cwd).index.pending_embeddings;
    assert!(left > 0, "the import still ran after the 100 calls");

    let (idle_p50, idle_p95) = (percentile(&idle, 50.0), percentile(&idle, 95.0));
    let (busy_p50, busy_p95) = (percentile(&busy, 50.0), percentile(&busy, 95.0));
    println!(
        "AT-3 ambient_recall, 100 warm calls: idle median {idle_p50:?} p95 {idle_p95:?}; \
         during a bulk import of {chunks} chunks median {busy_p50:?} p95 {busy_p95:?} \
         ({left} chunks still pending after)"
    );
    assert!(idle_p95 <= BUDGET, "idle p95 {idle_p95:?}");
    assert!(busy_p95 <= BUDGET, "p95 during the import {busy_p95:?}");

    d.wait_embedded(&cwd);
    let rec = provider.recorded();
    let bulk = rig.settings.bulk_threads(8);
    let query_threads = rig.settings.query_threads(8);
    assert!(rec.max_live_bulk_threads >= 1, "the bulk session ran");
    assert!(
        rec.max_live_bulk_threads <= bulk,
        "{} bulk threads alive at once, budget {bulk}",
        rec.max_live_bulk_threads
    );
    for (kind, threads) in &rec.opened {
        match kind {
            SessionKind::Bulk => assert_eq!(*threads, bulk),
            SessionKind::Query => assert_eq!(*threads, query_threads),
        }
    }
    assert!(rec.documents >= chunks as usize + 5);
    println!(
        "AT-3 threads: bulk budget {bulk} of 8 physical cores, at most {} alive; \
         query session {query_threads}; sessions opened {:?}",
        rec.max_live_bulk_threads, rec.opened
    );
}

/// AT-3's thread rule while `doctor_eval` embeds its own set during a bulk
/// import (SPEC 3 rule 5; e-report decision 35): the evaluation embeds
/// through the daemon's one bulk session, so the bulk threads alive at
/// once never pass `cpu.budget`.
#[test]
fn at3_doctor_eval_during_a_bulk_import_stays_within_the_thread_budget() {
    let rig = Rig::new();
    let cwd = rig.project("eval-busy");
    let provider = RecordingProvider::new(Duration::from_millis(3));
    let d = rig.start(
        FakeLoader::with(Arc::clone(&provider) as Arc<dyn EmbedderProvider>),
        Arc::new(UnwiredSharing),
    );
    d.wait_ready();
    for b in 0..6 {
        let items = (0..100)
            .map(|i| RememberItem::new(format!("bulk import record {b} {i} for the archive")))
            .collect();
        remember_many(&d, &cwd, items);
    }
    let before = d.status(&cwd).index.pending_embeddings;
    assert!(before > 0, "the import runs");
    let set = fixture_path("retrieval/retrieval_eval_set.json");
    let Reply::Eval(report) = d.ok(
        &cwd,
        Op::DoctorEval(sm_core::protocol::DoctorEvalParams {
            set_path: set.to_string_lossy().into_owned(),
            baseline_path: None,
        }),
    ) else {
        panic!("not an eval reply")
    };
    assert!(report.queries > 0);
    let after = d.status(&cwd).index.pending_embeddings;
    assert!(
        after > 0,
        "the import still ran when the evaluation ended ({before} pending before, {after} after)"
    );

    d.wait_embedded(&cwd);
    let rec = provider.recorded();
    let bulk = rig.settings.bulk_threads(8);
    assert!(rec.max_live_bulk_threads >= 1, "the bulk session ran");
    assert!(
        rec.max_live_bulk_threads <= bulk,
        "{} bulk threads alive at once, budget {bulk}; sessions opened {:?}",
        rec.max_live_bulk_threads,
        rec.opened
    );
    println!(
        "AT-3 doctor_eval during an import: bulk budget {bulk}, at most {} alive; \
         {before} chunks pending before the evaluation, {after} after",
        rec.max_live_bulk_threads
    );
}

// --- AT-4: long input ---------------------------------------------------------

/// AT-4: a 50,000-character `remember` succeeds, gives several chunks, and
/// the queue embeds every one of them. (Peak memory with the real model is
/// the owner's measurement; the fake embedder holds no model.)
#[test]
fn at4_a_50000_character_memory_is_stored_in_several_chunks_and_embedded() {
    let rig = Rig::new();
    let cwd = rig.project("long");
    let d = rig.start_fake();
    let mut content = String::with_capacity(50_100);
    let mut i = 0;
    while content.len() < 50_000 {
        content.push_str(&format!("word{i} "));
        i += 1;
    }
    content.truncate(50_000);
    assert_eq!(content.chars().count(), 50_000);

    let reply = remember_many(&d, &cwd, vec![RememberItem::new(content.clone())]);
    let m = &reply.memories[0];
    assert!(m.chunks >= 10, "several chunks: {}", m.chunks);
    d.wait_embedded(&cwd);
    let status = d.status(&cwd);
    assert_eq!(status.index.chunks, u64::from(m.chunks));
    assert_eq!(status.index.pending_embeddings, 0);
    let store = rig.store();
    let info = fake_info();
    let index = store
        .configure_index(IndexStamp {
            model_id: info.model_id,
            revision: info.revision,
            dim: info.dim,
            chunk_max_tokens: rig.settings.chunk.max_tokens,
            chunk_overlap: rig.settings.chunk.overlap,
        })
        .expect("the daemon's stamp");
    assert!(index.matches, "{:?}", index.differences);
    assert_eq!(store.get_memory(&m.id).expect("stored").content, content);
    assert!(store.memory_vector(&m.id).expect("a vector").is_some());
    let found = recall(
        &d,
        &cwd,
        query("word120 word121 word122", RecallMode::Vector, 1),
    );
    assert_eq!(
        ids(&found),
        vec![m.id],
        "a vector search reaches its chunks"
    );
    println!("AT-4: 50,000 characters, {} chunks, all embedded", m.chunks);
}

// --- AT-5: concurrency ----------------------------------------------------------

/// Sends `op(n)` over a new connection until `until`; the count and every
/// error that reached the client.
fn hammer(
    endpoint: &Endpoint,
    profile: &ProfileName,
    cwd: &str,
    until: Instant,
    stop: &AtomicBool,
    op: &dyn Fn(usize) -> Op,
) -> (usize, Vec<String>) {
    let mut n = 0;
    let mut errors = Vec::new();
    while Instant::now() < until && !stop.load(Ordering::SeqCst) {
        let req = Request::new(cwd, profile.clone(), op(n));
        let mut c = Client::connect(endpoint);
        if let Err(e) = c.call(&req).into_result() {
            errors.push(format!(
                "{} {}: {}",
                req.call.name(),
                e.code.as_str(),
                e.message
            ));
            if errors.len() >= 20 {
                stop.store(true, Ordering::SeqCst);
            }
        }
        n += 1;
    }
    (n, errors)
}

/// AT-5: two readers and one writer over the socket for 60 seconds, the
/// embedding queue writing beside them; no lock error, and no error of any
/// kind, reaches a client.
#[test]
fn at5_two_readers_and_one_writer_for_60_seconds_see_no_lock_error() {
    let rig = Rig::new();
    let cwd = rig.project("busy");
    let d = rig.start_fake();
    for i in 0..20 {
        remember(&d, &cwd, &format!("seed note {i} about the release train"));
    }
    d.wait_embedded(&cwd);
    let endpoint = d.endpoint.clone();
    let profile = d.profile.clone();
    let stop = AtomicBool::new(false);
    let until = Instant::now() + Duration::from_secs(60);

    let reader_a = |n: usize| -> Op {
        let mode = if n % 3 == 2 {
            RecallMode::Keyword
        } else {
            RecallMode::Auto
        };
        Op::Recall(query(&format!("release train note {}", n % 20), mode, 5))
    };
    let reader_b = |n: usize| -> Op {
        if n.is_multiple_of(2) {
            Op::Status(Default::default())
        } else {
            Op::AmbientRecall(AmbientParams {
                prompt: format!("what do we know about the release train {n}"),
                top_k: 3,
                scope: Scope::All,
            })
        }
    };
    let writer = |n: usize| -> Op {
        Op::Remember(RememberParams {
            memories: vec![RememberItem::new(format!(
                "writer note {n}: the release train leaves at {} past the hour",
                n % 60
            ))],
        })
    };
    let (a, b, w) = std::thread::scope(|s| {
        let a = s.spawn(|| hammer(&endpoint, &profile, &cwd, until, &stop, &reader_a));
        let b = s.spawn(|| hammer(&endpoint, &profile, &cwd, until, &stop, &reader_b));
        let w = s.spawn(|| hammer(&endpoint, &profile, &cwd, until, &stop, &writer));
        (
            a.join().expect("reader a"),
            b.join().expect("reader b"),
            w.join().expect("writer"),
        )
    });
    println!(
        "AT-5: 60 seconds, reader a {} requests, reader b {} requests, writer {} requests",
        a.0, b.0, w.0
    );
    assert!(a.1.is_empty(), "reader a: {:?}", a.1);
    assert!(b.1.is_empty(), "reader b: {:?}", b.1);
    assert!(w.1.is_empty(), "writer: {:?}", w.1);
    assert!(a.0 > 0 && b.0 > 0 && w.0 > 0);
    d.wait_embedded(&cwd);
    assert_eq!(
        d.status(&cwd)
            .projects
            .iter()
            .map(|p| p.memories)
            .sum::<u64>(),
        20 + w.0 as u64,
        "every write was stored"
    );
}

// --- AT-9: redaction on remember -------------------------------------------------

/// AT-9, `remember` side: every case of `redaction_cases.json` stored
/// through `remember` is redacted as the fixture says (version 3's
/// markers), the fake credentials never reach the store, and the reply
/// lists the kinds. The empty input is refused (`remember` needs content).
#[test]
fn at9_remember_redacts_the_fixture_credentials_and_lists_the_kinds() {
    let rig = Rig::new();
    let cwd = rig.project("secrets");
    let d = rig.start_fake();
    let store = rig.store();
    let f = fixture("capture/redaction_cases.json");
    let mut credentials = 0;
    let mut clean = 0;
    for case in f["cases"].as_array().expect("cases") {
        let name = case["name"].as_str().expect("a name");
        let input = case["input"].as_str().expect("a text input");
        let computed = &case["computed_result"];
        let kinds = strings(&computed["kinds"]);
        if input.trim().is_empty() {
            let e = d.err(
                &cwd,
                Op::Remember(RememberParams {
                    memories: vec![RememberItem::new(input)],
                }),
            );
            assert_code(&e, ErrorCode::InvalidInput);
            continue;
        }
        let reply = remember_many(&d, &cwd, vec![RememberItem::new(input)]);
        let m = &reply.memories[0];
        assert_eq!(m.redactions, kinds, "{name}: the kinds in the reply");
        let stored = store.get_memory(&m.id).expect("stored");
        let want = v3_markers(computed["redacted_text"].as_str().expect("a text"));
        assert_eq!(stored.content, want, "{name}: the stored content");
        let title = stored.title.clone().unwrap_or_default();
        for s in strings(&case["expect"]["not_contains"]) {
            assert!(!stored.content.contains(&s), "{name}: {s} stored");
            assert!(!title.contains(&s), "{name}: {s} in the title");
        }
        for s in strings(&case["expect"]["contains"]) {
            assert!(stored.content.contains(&v3_markers(&s)), "{name}: {s}");
        }
        if kinds.is_empty() {
            clean += 1;
        } else {
            credentials += 1;
        }
    }
    println!("AT-9: {credentials} cases with fake credentials redacted, {clean} left unchanged");
    assert!(credentials >= 25, "{credentials}");
}

// --- AT-10: model integrity ---------------------------------------------------------

/// AT-10 with the real check: `sm_embed::model::verify_files` over a small
/// file set (one 4 KiB file, its size and SHA-256 pinned) runs in the
/// loader before the fake model. Intact, the daemon is ready; after one
/// changed byte it is `failed` with `model_integrity` in `hello` and
/// `status`, every model request fails with that error, and `forget`,
/// `status` and keyword `recall` are still served (B-36).
#[test]
fn at10_one_changed_byte_fails_the_model_and_the_rest_is_served() {
    let rig = Rig::new();
    let cwd = rig.project("integrity");
    let dir = rig.tmp.path().join("model");
    std::fs::create_dir_all(&dir).expect("a model folder");
    let file = dir.join("model.onnx");
    let bytes: Vec<u8> = (0..4096u32).map(|i| (i % 251) as u8).collect();
    std::fs::write(&file, &bytes).expect("write the model file");
    let size = bytes.len() as u64;
    let sha = hash_file(&file).expect("hash the model file");
    let loader: Arc<dyn ModelLoader> = Arc::new(FnLoader({
        let dir = dir.clone();
        move || -> Result<LoadedModel> {
            verify_files(
                &dir,
                &[ModelFile {
                    path: "model.onnx",
                    size,
                    sha256: &sha,
                }],
            )?;
            Ok(fake_model())
        }
    }));

    let d = rig.start(Arc::clone(&loader), Arc::new(UnwiredSharing));
    d.wait_ready();
    let kept = remember(&d, &cwd, "the release checklist lives in docs/release.md");
    let gone = remember(&d, &cwd, "a note to forget about the staging rotation");
    d.wait_embedded(&cwd);
    drop(d);

    let mut changed = bytes.clone();
    changed[2048] ^= 0x01;
    std::fs::write(&file, &changed).expect("change one byte");
    assert_eq!(std::fs::metadata(&file).expect("metadata").len(), size);

    let d = rig.start(Arc::clone(&loader), Arc::new(UnwiredSharing));
    assert!(wait_until(Duration::from_secs(10), || d.hello().state
        != DaemonState::Loading));
    let hello = d.hello();
    assert_eq!(hello.state, DaemonState::Failed);
    let e = hello.state_error.expect("the error in hello");
    assert_code(&e, ErrorCode::ModelIntegrity);
    assert!(e.message.contains("model.onnx"), "{}", e.message);
    let status = d.status(&cwd);
    assert_eq!(status.daemon.state, DaemonState::Failed);
    assert_code(
        status
            .daemon
            .state_error
            .as_ref()
            .expect("the error in status"),
        ErrorCode::ModelIntegrity,
    );
    assert!(!status.model.loaded);

    assert_code(
        &d.err(&cwd, Op::Recall(RecallParams::new("release checklist"))),
        ErrorCode::ModelIntegrity,
    );
    assert_code(
        &d.err(
            &cwd,
            Op::Remember(RememberParams {
                memories: vec![RememberItem::new("another note on the release checklist")],
            }),
        ),
        ErrorCode::ModelIntegrity,
    );
    assert_code(
        &d.err(&cwd, Op::Reindex(ReindexParams {})),
        ErrorCode::ModelIntegrity,
    );

    let kw = recall(&d, &cwd, query("release checklist", RecallMode::Keyword, 5));
    assert_eq!(ids(&kw), vec![kept], "keyword recall is served");
    match d.ok(
        &cwd,
        Op::Forget(ForgetParams {
            ids: vec![gone],
            filter: None,
            confirm: false,
        }),
    ) {
        Reply::Forget(f) => assert_eq!(f.deleted, vec![gone]),
        other => panic!("forget replied {other:?}"),
    }
    let after = d.status(&cwd);
    assert_eq!(
        after.projects.iter().map(|p| p.memories).sum::<u64>(),
        1,
        "forget was served"
    );
}

// --- AT-11: index mismatch ---------------------------------------------------------

fn renamed_info() -> EmbedderInfo {
    EmbedderInfo {
        model_id: "fake-renamed".to_owned(),
        ..fake_info()
    }
}

/// The fake embedder under another model identifier, so the queue's
/// embeddings carry the identifier the index is configured for.
struct RenamedProvider {
    info: EmbedderInfo,
}

impl EmbedderProvider for RenamedProvider {
    fn info(&self) -> &EmbedderInfo {
        &self.info
    }
    fn open(&self, kind: SessionKind, threads: usize) -> Result<Box<dyn Embedder>> {
        Ok(Box::new(RenamedEmbedder {
            inner: FakeEmbedder::with_session(kind, threads),
            info: self.info.clone(),
        }))
    }
}

struct RenamedEmbedder {
    inner: FakeEmbedder,
    info: EmbedderInfo,
}

impl Embedder for RenamedEmbedder {
    fn info(&self) -> &EmbedderInfo {
        &self.info
    }
    fn embed_query(&self, text: &str, max_tokens: usize) -> Result<Vec<f32>> {
        self.inner.embed_query(text, max_tokens)
    }
    fn embed_documents(&self, docs: &[DocInput<'_>]) -> Result<Vec<Vec<f32>>> {
        self.inner.embed_documents(docs)
    }
}

struct RenamedLoader;

impl ModelLoader for RenamedLoader {
    fn info(&self) -> EmbedderInfo {
        renamed_info()
    }
    fn load(&self) -> Result<LoadedModel> {
        Ok(LoadedModel {
            provider: Arc::new(RenamedProvider {
                info: renamed_info(),
            }),
            chunker: Arc::new(SimpleChunker),
        })
    }
}

/// Starts a daemon whose configuration differs from the index by
/// `difference`; search is blocked with `index_mismatch` until `reindex`
/// completes, then serves `want` again.
fn blocked_until_reindex(
    rig: &Rig,
    cwd: &str,
    loader: Arc<dyn ModelLoader>,
    q: &str,
    want: MemoryId,
    difference: &str,
) {
    let d = rig.start(loader, Arc::new(UnwiredSharing));
    d.wait_ready();
    let s = d.status(cwd);
    assert!(!s.index.matches);
    assert_eq!(s.index.differences, vec![difference.to_owned()]);
    for mode in [RecallMode::Auto, RecallMode::Vector, RecallMode::Keyword] {
        assert_code(
            &d.err(cwd, Op::Recall(query(q, mode, 1))),
            ErrorCode::IndexMismatch,
        );
    }
    let report = match d.ok(cwd, Op::Reindex(ReindexParams {})) {
        Reply::Reindex(r) => r,
        other => panic!("reindex replied {other:?}"),
    };
    assert_eq!(report.memories, 2);
    assert!(report.chunks_queued >= 2, "{report:?}");
    d.wait_embedded(cwd);
    assert!(d.status(cwd).index.matches);
    assert_eq!(
        ids(&recall(&d, cwd, query(q, RecallMode::Auto, 1))),
        vec![want]
    );
    assert_eq!(
        ids(&recall(&d, cwd, query("billing", RecallMode::Keyword, 1))),
        vec![want]
    );
}

/// AT-11: changing the chunk size, then the model identifier, blocks
/// search with `index_mismatch` until `reindex` completes.
#[test]
fn at11_a_changed_chunk_size_or_model_blocks_search_until_reindex() {
    let mut rig = Rig::new();
    let cwd = rig.project("stamp");
    let q = "release checklist for the billing deploy";
    let want = {
        let d = rig.start_fake();
        let id = remember(&d, &cwd, q);
        remember(&d, &cwd, "garden tomatoes need summer rain");
        d.wait_embedded(&cwd);
        assert_eq!(
            ids(&recall(&d, &cwd, query(q, RecallMode::Auto, 1))),
            vec![id]
        );
        id
    };

    let before = rig.settings.chunk.max_tokens;
    rig.settings.chunk.max_tokens = before / 2;
    blocked_until_reindex(
        &rig,
        &cwd,
        FakeLoader::new(),
        q,
        want,
        &format!(
            "chunk.max_tokens {before} in the index, {} configured",
            before / 2
        ),
    );

    blocked_until_reindex(
        &rig,
        &cwd,
        Arc::new(RenamedLoader),
        q,
        want,
        &format!(
            "model {} in the index, fake-renamed configured",
            fake_info().model_id
        ),
    );
}

// --- AT-12: scope ------------------------------------------------------------------

/// AT-12: a memory stored from project A's folder is not returned from
/// project B without `scope: all`, by recall (vector and keyword) or by the
/// ambient hook.
#[test]
fn at12_a_memory_of_project_a_is_not_returned_from_b_without_scope_all() {
    let rig = Rig::new();
    let a = rig.project("alpha-app");
    let b = rig.project("beta-app");
    let d = rig.start_fake();
    let text = "release checklist steps for the billing service deploy";
    let id = remember(&d, &a, text);
    d.wait_embedded(&a);

    for mode in [RecallMode::Auto, RecallMode::Keyword] {
        let from_a = recall(&d, &a, query(text, mode, 5));
        assert_eq!(ids(&from_a), vec![id], "{mode:?} from A");
        let from_b = recall(&d, &b, query(text, mode, 5));
        assert!(from_b.results.is_empty(), "{mode:?} from B: {from_b:?}");
        let mut all = query(text, mode, 5);
        all.scope = Some(Scope::All);
        let from_b_all = recall(&d, &b, all);
        assert_eq!(ids(&from_b_all), vec![id], "{mode:?} from B, scope all");
        assert_eq!(from_b_all.results[0].project, "alpha-app");
    }

    let ambient = |scope: Scope| match d
        .call(
            &b,
            Op::AmbientRecall(AmbientParams {
                prompt: text.to_owned(),
                top_k: 3,
                scope,
            }),
        )
        .into_result()
    {
        Ok(Reply::Ambient(r)) => r,
        other => panic!("ambient_recall replied {other:?}"),
    };
    assert!(ambient(Scope::Project).injected.is_empty(), "project scope");
    assert_eq!(ambient(Scope::All).injected, vec![id], "all projects");
}

/// AT-12: two daemons on two profiles under one root, the same project
/// folder, never see each other's memories, even with `scope: all`.
#[test]
fn at12_two_profiles_never_see_each_others_memories() {
    let alpha = Rig::with_profile("alpha");
    let mut beta = Rig::with_profile("beta");
    beta.paths = ProfilePaths::new(
        alpha.paths.root.clone(),
        ProfileName::parse("beta").expect("a profile name"),
    );
    let cwd = alpha.project("shared-folder");
    let da = alpha.start_fake();
    let db = beta.start_fake();
    let in_alpha = "alpha only: the release checklist lives in docs";
    let in_beta = "beta only: the release checklist lives in the wiki";
    let ia = remember(&da, &cwd, in_alpha);
    let ib = remember(&db, &cwd, in_beta);
    da.wait_embedded(&cwd);
    db.wait_embedded(&cwd);

    for mode in [RecallMode::Auto, RecallMode::Keyword] {
        let mut p = query("release checklist", mode, 10);
        p.scope = Some(Scope::All);
        assert_eq!(
            ids(&recall(&da, &cwd, p.clone())),
            vec![ia],
            "{mode:?} alpha"
        );
        assert_eq!(ids(&recall(&db, &cwd, p)), vec![ib], "{mode:?} beta");
    }
    assert_eq!(alpha.store().memory_count().expect("count"), 1);
    assert_eq!(beta.store().memory_count().expect("count"), 1);
    assert_ne!(alpha.paths.database(), beta.paths.database());
}

// --- AT-16: importance does not change ranking -------------------------------

/// What a recall ranks: each result's id, score, rank, match kind and weak
/// mark, in order.
type Ranked = (MemoryId, Option<f32>, Option<u32>, Option<MatchKind>, bool);

fn ranking(r: &RecallReply) -> Vec<Ranked> {
    r.results
        .iter()
        .map(|v| (v.id, v.score, v.rank, v.matched_by, v.weak))
        .collect()
}

/// A ranking without the rank numbers, which a filter renumbers.
fn unnumbered(r: &[Ranked]) -> Vec<(MemoryId, Option<f32>, Option<MatchKind>, bool)> {
    r.iter().map(|x| (x.0, x.1, x.3, x.4)).collect()
}

/// Sets a memory's importance with `revise`, and nothing else.
fn set_importance(d: &Daemon, cwd: &str, id: MemoryId, importance: f64) {
    let mut p = ReviseParams::new(id);
    p.importance = Some(importance);
    match d.ok(cwd, Op::Revise(p)) {
        Reply::Revise(r) => {
            assert_eq!(r.memory.importance, importance);
            assert!(!r.reembedded, "an importance change re-embeds nothing");
        }
        other => panic!("revise replied {other:?}"),
    }
}

/// AT-16 through the daemon (SPEC 11 AT-16 "Importance does not change
/// ranking"; SPEC 16 "Importance is stored, not used"; B-70): importance
/// changes neither `recall`, in any mode, nor what the prompt hook's
/// `ambient_recall` injects, and `min_importance` only filters. The same
/// calls are made with every memory at the default importance, then with
/// importances the reverse of the ranking, then in its order, and give the
/// same answers. Two memories with the same words in another order tie on
/// score; the lower id ranks first both when it has the lower importance
/// and when it has the higher (B-70: score, then id).
#[test]
fn at16_importance_does_not_change_recall_ranking() {
    let rig = Rig::new();
    let cwd = rig.project("importance");
    let d = rig.start_fake();
    let q = "release checklist for the billing_service deploy";
    let texts = [
        "release checklist for the billing_service deploy",
        "release checklist for the billing_service",
        "release checklist for the billing_service deploy on friday",
        "the billing_service deploy runbook and its rollback steps",
        "deploy checklist for the garden shed",
        "the garden shed checklist for deploy",
        "summer rain and garden tomatoes",
    ];
    let qv = fake_vector(q);
    let mut stored: Vec<(MemoryId, f32, &str)> = texts
        .iter()
        .map(|t| (remember(&d, &cwd, t), dot(&qv, &fake_vector(t)), *t))
        .collect();
    d.wait_embedded(&cwd);
    let tie = (stored[4].0.min(stored[5].0), stored[4].0.max(stored[5].0));
    assert_eq!(stored[4].1, stored[5].1, "the same words tie on score");
    // The ranking B-70 gives: score, then id.
    stored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    for w in stored.windows(2) {
        assert!(
            w[0].1 == w[1].1 || w[0].1 - w[1].1 > 1e-3,
            "every other score stands apart: {stored:?}"
        );
    }
    let expected: Vec<MemoryId> = stored.iter().map(|s| s.0).collect();
    let at = |id: MemoryId| expected.iter().position(|e| *e == id).expect("ranked");
    assert_eq!(at(tie.1), at(tie.0) + 1, "the tie is ranked by id");
    let n = expected.len();
    let floor = rig.settings.recall.similarity_floor as f32;
    let above: Vec<MemoryId> = stored
        .iter()
        .filter(|s| s.1 >= floor)
        .map(|s| s.0)
        .collect();
    assert!(
        above.len() >= 3,
        "three memories reach the floor: {stored:?}"
    );
    let phrase: HashSet<MemoryId> = stored
        .iter()
        .filter(|s| s.2.contains("release checklist"))
        .map(|s| s.0)
        .collect();
    assert_eq!(phrase.len(), 3);

    let mut sessions = 0;
    let mut answers = |importance: &str| -> (Vec<Vec<Ranked>>, AmbientReply) {
        let calls = [
            query(q, RecallMode::Vector, n as u32),
            query(q, RecallMode::Auto, 2),
            query("release checklist", RecallMode::Keyword, n as u32),
        ];
        let recalls: Vec<Vec<Ranked>> = calls
            .into_iter()
            .map(|p| ranking(&recall(&d, &cwd, p)))
            .collect();
        assert_eq!(
            recalls[0].iter().map(|r| r.0).collect::<Vec<_>>(),
            expected,
            "vector recall, {importance}: score then id"
        );
        let keyword: HashSet<MemoryId> = recalls[2].iter().map(|r| r.0).collect();
        assert_eq!(keyword, phrase, "keyword recall, {importance}");
        // A new session each time, so nothing is held back as a repeat.
        sessions += 1;
        let req = d
            .req(
                &cwd,
                Op::AmbientRecall(AmbientParams {
                    prompt: q.to_owned(),
                    top_k: 3,
                    scope: Scope::Project,
                }),
            )
            .with_session(Some(format!("at16-{sessions}")));
        let ambient = match d.send(&req).into_result() {
            Ok(Reply::Ambient(r)) => r,
            other => panic!("ambient_recall replied {other:?}"),
        };
        assert_eq!(
            ambient.gate,
            HookGate::Injected,
            "{importance}: {ambient:?}"
        );
        assert_eq!(ambient.injected, above[..3], "ambient, {importance}");
        (recalls, ambient)
    };
    let mut filtered = query(q, RecallMode::Vector, n as u32);
    filtered.min_importance = Some(0.5);

    let default = answers("the default importance");
    let full = default.0[0].clone();
    // The best result has the least importance, so the tie's lower id has
    // the lower importance.
    for (i, id) in expected.iter().enumerate() {
        set_importance(&d, &cwd, *id, i as f64 / (n - 1) as f64);
    }
    let reversed = answers("importance against the ranking");
    assert_eq!(
        unnumbered(&ranking(&recall(&d, &cwd, filtered.clone()))),
        unnumbered(&full[n / 2..]),
        "min_importance keeps the lower half, in ranking order"
    );
    // The best result has the most importance, so the tie's lower id has
    // the higher importance.
    for (i, id) in expected.iter().enumerate() {
        set_importance(&d, &cwd, *id, (n - 1 - i) as f64 / (n - 1) as f64);
    }
    let aligned = answers("importance with the ranking");
    assert_eq!(
        unnumbered(&ranking(&recall(&d, &cwd, filtered))),
        unnumbered(&full[..=n / 2]),
        "min_importance keeps the upper half, in ranking order"
    );

    assert_eq!(reversed, default, "importance against the ranking");
    assert_eq!(aligned, default, "importance with the ranking");
}
