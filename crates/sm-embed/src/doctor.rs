//! The numbers `doctor` prints: bulk chunks per second and query latency at
//! the configured thread budget, on a fixed sample (SPEC 5, rule 7). Nothing
//! is tuned from them.

use std::time::{Duration, Instant};

use sm_core::config::Settings;
use sm_core::error::{CoreError, Result};
use sm_core::traits::{ChunkParams, Chunker, DocInput, Embedder, EmbedderProvider, SessionKind};

/// Chunks embedded in the timed bulk run.
pub const SAMPLE_CHUNKS: usize = 16;
/// Query length for the latency figure, in tokens.
pub const QUERY_TOKENS: usize = 128;
/// Timed query runs; the median is reported.
pub const QUERY_RUNS: usize = 5;

/// Words of the fixed sample: plain engineering prose.
const WORDS: &[&str] = &[
    "the",
    "order",
    "service",
    "retries",
    "a",
    "failed",
    "payment",
    "when",
    "database",
    "connection",
    "pool",
    "is",
    "exhausted",
    "and",
    "writes",
    "an",
    "audit",
    "row",
    "before",
    "returning",
    "we",
    "decided",
    "to",
    "use",
    "optimistic",
    "concurrency",
    "on",
    "dbo.Orders",
    "because",
    "deadlocks",
    "appeared",
    "under",
    "load",
    "in",
    "release",
    "migration",
    "adds",
    "index",
    "customer",
    "id",
    "queue",
    "consumer",
    "must",
    "be",
    "idempotent",
    "cache",
    "invalidation",
    "happens",
    "after",
    "commit",
    "not",
    "timeout",
    "for",
    "report",
    "export",
    "was",
    "raised",
    "from",
    "thirty",
    "seconds",
    "two",
    "minutes",
];

/// `words` words of the fixed sample, the same on every machine: a 64-bit
/// linear congruential generator picks each word, and every 17th word ends
/// a sentence.
pub fn sample_text(words: usize) -> String {
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    let mut out = String::new();
    for i in 0..words {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let word = WORDS[((state >> 33) % WORDS.len() as u64) as usize];
        if i > 0 {
            out.push_str(if i % 17 == 0 { ". " } else { " " });
        }
        out.push_str(word);
    }
    if words > 0 {
        out.push('.');
    }
    out
}

/// What `doctor` reports for the embedder.
#[derive(Debug, Clone, PartialEq)]
pub struct DoctorNumbers {
    pub physical_cores: usize,
    pub query_threads: usize,
    pub bulk_threads: usize,
    pub batch_size: usize,
    /// Chunks in the timed bulk run.
    pub sample_chunks: usize,
    pub bulk_elapsed: Duration,
    pub bulk_chunks_per_sec: f64,
    /// Tokens in the timed query, before the prompt.
    pub query_tokens: usize,
    pub query_runs: usize,
    pub query_latency_median: Duration,
}

/// Opens one bulk and one query session at the configured budget and times
/// them with [`time_sessions`].
pub fn doctor_numbers(
    provider: &dyn EmbedderProvider,
    chunker: &dyn Chunker,
    settings: &Settings,
    physical_cores: usize,
) -> Result<DoctorNumbers> {
    let bulk = provider.open(SessionKind::Bulk, settings.bulk_threads(physical_cores))?;
    let query = provider.open(SessionKind::Query, settings.query_threads(physical_cores))?;
    time_sessions(
        bulk.as_ref(),
        query.as_ref(),
        chunker,
        settings,
        physical_cores,
    )
}

/// Times the fixed sample on sessions already open: one untimed warm-up
/// each, then [`SAMPLE_CHUNKS`] chunks in bulk on `bulk` and the median of
/// [`QUERY_RUNS`] queries of [`QUERY_TOKENS`] tokens on `query`. The
/// thread counts reported are the configured budget for `physical_cores`,
/// the counts both sessions are opened with (the daemon's included; B-125).
pub fn time_sessions(
    bulk: &dyn Embedder,
    query: &dyn Embedder,
    chunker: &dyn Chunker,
    settings: &Settings,
    physical_cores: usize,
) -> Result<DoctorNumbers> {
    let max_tokens = settings.chunk.max_tokens as usize;
    let params = ChunkParams {
        max_tokens,
        overlap: settings.chunk.overlap,
        index_cap: None,
    };
    let text = sample_text(SAMPLE_CHUNKS * max_tokens);
    let plan = chunker.chunk(&text, &params)?;
    let chunks: Vec<&str> = plan
        .pieces
        .iter()
        .take(SAMPLE_CHUNKS)
        .map(|p| p.text.as_str())
        .collect();
    if chunks.len() < SAMPLE_CHUNKS {
        return Err(CoreError::internal(format!(
            "the doctor sample made {} chunks, not {SAMPLE_CHUNKS}",
            chunks.len()
        )));
    }
    let docs: Vec<DocInput<'_>> = chunks
        .iter()
        .map(|t| DocInput {
            title: None,
            text: t,
        })
        .collect();

    let bulk_threads = settings.bulk_threads(physical_cores);
    let warm = settings.embed.batch_size.max(1) as usize;
    bulk.embed_documents(&docs[..warm.min(docs.len())])?;
    let started = Instant::now();
    let vectors = bulk.embed_documents(&docs)?;
    let bulk_elapsed = started.elapsed();
    if vectors.len() != docs.len() {
        return Err(CoreError::internal(
            "the bulk session returned the wrong number of vectors",
        ));
    }

    let query_threads = settings.query_threads(physical_cores);
    let question = sample_text(QUERY_TOKENS * 2);
    let query_tokens = chunker.count_tokens(&question)?.min(QUERY_TOKENS);
    query.embed_query(&question, QUERY_TOKENS)?;
    let mut times = Vec::with_capacity(QUERY_RUNS);
    for _ in 0..QUERY_RUNS {
        let started = Instant::now();
        query.embed_query(&question, QUERY_TOKENS)?;
        times.push(started.elapsed());
    }
    times.sort_unstable();

    Ok(DoctorNumbers {
        physical_cores,
        query_threads,
        bulk_threads,
        batch_size: settings.embed.batch_size as usize,
        sample_chunks: docs.len(),
        bulk_elapsed,
        bulk_chunks_per_sec: docs.len() as f64 / bulk_elapsed.as_secs_f64().max(1e-9),
        query_tokens,
        query_runs: QUERY_RUNS,
        query_latency_median: times[QUERY_RUNS / 2],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use sm_core::testing::{FakeProvider, SimpleChunker};

    #[test]
    fn s5_rule7_sample_is_fixed() {
        let a = sample_text(40);
        assert_eq!(a, sample_text(40));
        assert_eq!(a.split_whitespace().count(), 40);
        assert!(a.ends_with('.'));
        assert_eq!(a.matches(". ").count(), 2);
        assert!(sample_text(0).is_empty());
        assert!(sample_text(1000).split_whitespace().all(|w| {
            let w = w.trim_end_matches('.');
            WORDS.contains(&w)
        }));
    }

    #[test]
    fn s5_rule7_doctor_reports_both_numbers_at_the_budget() {
        let provider = FakeProvider::new();
        let mut settings = Settings::default();
        settings.cpu.budget = 0.5;
        let n = doctor_numbers(&provider, &SimpleChunker, &settings, 8).unwrap();
        assert_eq!(n.physical_cores, 8);
        assert_eq!(n.bulk_threads, 4);
        assert_eq!(n.query_threads, 4);
        assert_eq!(n.sample_chunks, SAMPLE_CHUNKS);
        assert_eq!(n.query_tokens, QUERY_TOKENS);
        assert_eq!(n.query_runs, QUERY_RUNS);
        assert!(n.bulk_chunks_per_sec > 0.0);
        assert_eq!(provider.opened(), 2);
    }
}
