//! EmbeddingGemma 2 through ONNX Runtime behind
//! `sm_core::traits::{EmbedderProvider, Embedder}` (SPEC 5).
//!
//! [`GemmaProvider::load`] checks the pinned files (AT-10) and loads the
//! tokenizer once; every session it opens shares that tokenizer. A query
//! session runs inference on the calling thread. A bulk session owns a
//! runner thread, lowered below normal priority like its pool threads, and
//! runs every batch there (D-10).

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread::JoinHandle;

use ndarray::Array2;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;
use sm_core::EMBED_DIM;
use sm_core::config::Settings;
use sm_core::error::{CoreError, ErrorCode, Result};
use sm_core::traits::{DocInput, Embedder, EmbedderInfo, EmbedderProvider, SessionKind};

use crate::chunker::StructuredChunker;
use crate::input::{
    InputBuilder, PAD_ID, PaddedBatch, Prepared, pad_batch, plan_batches, split_output,
};
use crate::model::{self, MODEL_ID, MODEL_ONNX, REVISION, TOKENIZER, VerifyReport};
use crate::runtime;
use crate::threads::{
    CountingThreads, ThreadPlan, ThreadReport, physical_cores, plan_threads, priority_probe,
};
use crate::tokenize::ModelTokenizer;

/// The model's inputs, in its order.
pub const INPUT_NAMES: [&str; 5] = [
    "input_ids",
    "attention_mask",
    "image_features",
    "video_features",
    "audio_features",
];
/// The model output used: one unit vector per input.
pub const OUTPUT_NAME: &str = "sentence_embedding";
/// Width of the empty image, video and audio feature inputs.
pub const FEATURE_DIM: usize = 512;
/// The padding token, which must have id [`PAD_ID`].
pub const PAD_TOKEN: &str = "<pad>";

/// The settings a session uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmbedConfig {
    /// `chunk.max_tokens`: the most content tokens in one input.
    pub chunk_max_tokens: usize,
    /// `embed.batch_size`: inputs per inference.
    pub batch_size: usize,
    /// `embed.spinning`: let intra-op and inter-op threads spin.
    pub spinning: bool,
}

impl EmbedConfig {
    pub fn from_settings(settings: &Settings) -> Self {
        EmbedConfig {
            chunk_max_tokens: settings.chunk.max_tokens as usize,
            batch_size: settings.embed.batch_size as usize,
            spinning: settings.embed.spinning,
        }
    }
}

impl Default for EmbedConfig {
    fn default() -> Self {
        EmbedConfig::from_settings(&Settings::default())
    }
}

/// Opens EmbeddingGemma 2 sessions from one verified model folder.
#[derive(Debug)]
pub struct GemmaProvider {
    dir: PathBuf,
    tok: Arc<ModelTokenizer>,
    config: EmbedConfig,
    info: EmbedderInfo,
    physical: usize,
    verified: VerifyReport,
}

impl GemmaProvider {
    /// Checks every pinned file in `dir` (size, then SHA-256), loads the
    /// tokenizer and starts ONNX Runtime. Sessions are opened later with
    /// [`EmbedderProvider::open`].
    pub fn load(dir: &Path, config: EmbedConfig) -> Result<Self> {
        let verified = model::verify(dir)?;
        let tok = ModelTokenizer::from_file(&TOKENIZER.local_path(dir))?;
        match tok.token_id(PAD_TOKEN) {
            Some(id) if i64::from(id) == PAD_ID => {}
            other => {
                return Err(model::integrity(
                    dir,
                    format!(
                        "the tokenizer maps {PAD_TOKEN} to {other:?}, not {PAD_ID}; inputs are padded with {PAD_ID}"
                    ),
                ));
            }
        }
        runtime::environment()?;
        Ok(GemmaProvider {
            dir: dir.to_path_buf(),
            tok: Arc::new(tok),
            config,
            info: EmbedderInfo {
                model_id: MODEL_ID.to_owned(),
                revision: REVISION.to_owned(),
                dim: EMBED_DIM,
            },
            physical: physical_cores(),
            verified,
        })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn config(&self) -> EmbedConfig {
        self.config
    }

    /// What the integrity check found when the provider was loaded.
    pub fn verify_report(&self) -> &VerifyReport {
        &self.verified
    }

    /// The model tokenizer, shared by every session.
    pub fn tokenizer(&self) -> &Arc<ModelTokenizer> {
        &self.tok
    }

    /// A chunker counting with the model tokenizer (SPEC 5, rule 1).
    pub fn chunker(&self) -> StructuredChunker {
        StructuredChunker::new(self.tok.clone())
    }

    /// Opens a session with its concrete type, for its thread report.
    pub fn open_gemma(&self, kind: SessionKind, threads: usize) -> Result<GemmaEmbedder> {
        let plan = plan_threads(kind, threads, self.physical, priority_probe());
        let manager = CountingThreads::new(plan.lower);
        let exec = match kind {
            SessionKind::Query => {
                let session =
                    build_session(&self.dir, &plan, self.config.spinning, manager.clone())?;
                Exec::Inline(Mutex::new(session))
            }
            SessionKind::Bulk => Exec::Runner(Runner::start(
                self.dir.clone(),
                plan.clone(),
                self.config.spinning,
                manager.clone(),
            )?),
        };
        Ok(GemmaEmbedder {
            info: self.info.clone(),
            tok: Arc::clone(&self.tok),
            config: self.config,
            plan,
            threads: manager,
            truncated: AtomicUsize::new(0),
            exec,
        })
    }
}

impl EmbedderProvider for GemmaProvider {
    fn info(&self) -> &EmbedderInfo {
        &self.info
    }

    fn open(&self, kind: SessionKind, threads: usize) -> Result<Box<dyn Embedder>> {
        Ok(Box::new(self.open_gemma(kind, threads)?))
    }
}

/// One session: its own ONNX Runtime session and thread pool, the
/// provider's tokenizer.
#[derive(Debug)]
pub struct GemmaEmbedder {
    info: EmbedderInfo,
    tok: Arc<ModelTokenizer>,
    config: EmbedConfig,
    plan: ThreadPlan,
    threads: CountingThreads,
    truncated: AtomicUsize,
    exec: Exec,
}

impl GemmaEmbedder {
    /// The session's threads as planned and as seen so far (AT-3, D-10).
    pub fn threads(&self) -> ThreadReport {
        ThreadReport {
            plan: self.plan.clone(),
            pool: self.threads.snapshot(),
        }
    }

    /// Intra-op threads the session was built with.
    pub fn intra_threads(&self) -> usize {
        self.plan.intra_threads
    }

    /// Inputs whose content was cut to respect the per-input cap.
    pub fn truncated_inputs(&self) -> usize {
        self.truncated.load(Ordering::Relaxed)
    }

    pub fn config(&self) -> EmbedConfig {
        self.config
    }

    fn builder(&self) -> InputBuilder<'_> {
        InputBuilder::new(&*self.tok, self.config.chunk_max_tokens)
    }

    fn count_truncated(&self, prepared: &[Prepared]) {
        let n = prepared.iter().filter(|p| p.truncated).count();
        if n > 0 {
            self.truncated.fetch_add(n, Ordering::Relaxed);
        }
    }
}

impl Embedder for GemmaEmbedder {
    fn info(&self) -> &EmbedderInfo {
        &self.info
    }

    fn embed_query(&self, text: &str, max_tokens: usize) -> Result<Vec<f32>> {
        let prepared = self.builder().query(text, max_tokens)?;
        self.count_truncated(std::slice::from_ref(&prepared));
        let mut out = self.exec.run(pad_batch(&[&prepared.ids]))?;
        out.pop()
            .ok_or_else(|| CoreError::internal("the model returned no vector for the query"))
    }

    fn embed_documents(&self, docs: &[DocInput<'_>]) -> Result<Vec<Vec<f32>>> {
        let builder = self.builder();
        let prepared = docs
            .iter()
            .map(|d| builder.document(d))
            .collect::<Result<Vec<_>>>()?;
        self.count_truncated(&prepared);
        embed_in_batches(&prepared, self.config.batch_size, |batch| {
            self.exec.run(batch)
        })
    }
}

/// Embeds `prepared` in batches of at most `batch_size`, shortest first,
/// and returns the vectors in input order.
pub fn embed_in_batches(
    prepared: &[Prepared],
    batch_size: usize,
    mut run: impl FnMut(PaddedBatch) -> Result<Vec<Vec<f32>>>,
) -> Result<Vec<Vec<f32>>> {
    let lengths: Vec<usize> = prepared.iter().map(|p| p.ids.len()).collect();
    let mut out: Vec<Option<Vec<f32>>> = vec![None; prepared.len()];
    for group in plan_batches(&lengths, batch_size) {
        let rows: Vec<&[i64]> = group.iter().map(|&i| prepared[i].ids.as_slice()).collect();
        let vectors = run(pad_batch(&rows))?;
        if vectors.len() != group.len() {
            return Err(CoreError::internal(format!(
                "a batch of {} inputs returned {} vectors",
                group.len(),
                vectors.len()
            )));
        }
        for (i, v) in group.into_iter().zip(vectors) {
            out[i] = Some(v);
        }
    }
    out.into_iter()
        .map(|v| v.ok_or_else(|| CoreError::internal("an input was not embedded")))
        .collect()
}

/// Where a session's inference runs.
#[derive(Debug)]
enum Exec {
    /// On the calling thread (query sessions).
    Inline(Mutex<Session>),
    /// On the session's own lowered thread (bulk sessions).
    Runner(Runner),
}

impl Exec {
    fn run(&self, batch: PaddedBatch) -> Result<Vec<Vec<f32>>> {
        match self {
            Exec::Inline(session) => {
                let mut session = session
                    .lock()
                    .map_err(|_| CoreError::internal("the query session is poisoned"))?;
                run_batch(&mut session, batch)
            }
            Exec::Runner(runner) => runner.run(batch),
        }
    }
}

type Reply = mpsc::Sender<Result<Vec<Vec<f32>>>>;

/// A bulk session's thread. It lowers itself, builds the session (so ONNX
/// Runtime's pool threads are created through the lowered manager) and
/// runs every batch sent to it.
#[derive(Debug)]
struct Runner {
    jobs: Option<mpsc::Sender<(PaddedBatch, Reply)>>,
    handle: Option<JoinHandle<()>>,
}

impl Runner {
    fn start(
        dir: PathBuf,
        plan: ThreadPlan,
        spinning: bool,
        manager: CountingThreads,
    ) -> Result<Self> {
        let (jobs, inbox) = mpsc::channel::<(PaddedBatch, Reply)>();
        let (ready_tx, ready_rx) = mpsc::channel::<Result<()>>();
        let handle = std::thread::Builder::new()
            .name("sm-embed-bulk".to_owned())
            .spawn(move || {
                manager.adopt_current();
                let mut session = match build_session(&dir, &plan, spinning, manager) {
                    Ok(s) => {
                        let _ = ready_tx.send(Ok(()));
                        s
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                for (batch, reply) in inbox {
                    let _ = reply.send(run_batch(&mut session, batch));
                }
            })
            .map_err(|e| {
                CoreError::internal(format!("cannot start the bulk embedding thread: {e}"))
            })?;
        let ready = ready_rx.recv().unwrap_or_else(|_| {
            Err(CoreError::internal(
                "the bulk embedding thread stopped while loading",
            ))
        });
        match ready {
            Ok(()) => Ok(Runner {
                jobs: Some(jobs),
                handle: Some(handle),
            }),
            Err(e) => {
                let _ = handle.join();
                Err(e)
            }
        }
    }

    fn run(&self, batch: PaddedBatch) -> Result<Vec<Vec<f32>>> {
        let stopped = || CoreError::internal("the bulk embedding thread has stopped");
        let (reply, answer) = mpsc::channel();
        self.jobs
            .as_ref()
            .ok_or_else(stopped)?
            .send((batch, reply))
            .map_err(|_| stopped())?;
        answer.recv().map_err(|_| stopped())?
    }
}

impl Drop for Runner {
    fn drop(&mut self) {
        // Closing the queue ends the thread's loop; the session drops there.
        self.jobs.take();
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn ort_error(what: &'static str) -> impl Fn(ort::Error) -> CoreError {
    move |e| CoreError::internal(format!("{what}: {e}"))
}

fn builder_error<R>(e: ort::Error<R>) -> CoreError {
    CoreError::internal(format!("cannot configure the ONNX Runtime session: {e}"))
}

/// One ONNX Runtime session: graph optimisation level 3, the planned intra-op
/// threads through `manager`, one inter-op thread (sequential execution),
/// spinning only when configured.
fn build_session(
    dir: &Path,
    plan: &ThreadPlan,
    spinning: bool,
    manager: CountingThreads,
) -> Result<Session> {
    let env = runtime::environment()?;
    let path = MODEL_ONNX.local_path(dir);
    let mut builder = Session::builder(&env)
        .map_err(ort_error("cannot create an ONNX Runtime session"))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(builder_error)?
        .with_independent_thread_pool()
        .map_err(builder_error)?
        .with_parallel_execution(false)
        .map_err(builder_error)?
        .with_intra_threads(plan.intra_threads)
        .map_err(builder_error)?
        .with_inter_threads(1)
        .map_err(builder_error)?
        .with_intra_op_spinning(spinning)
        .map_err(builder_error)?
        .with_inter_op_spinning(spinning)
        .map_err(builder_error)?
        .with_thread_manager(manager)
        .map_err(builder_error)?;
    let session = builder.commit_from_file(&path).map_err(|e| {
        CoreError::internal(format!("cannot load {}: {e}", path.display())).with_next_action(
            "Run `spatial-memory doctor`; if the model check passes, report this error.",
        )
    })?;
    check_signature(dir, &session)?;
    Ok(session)
}

/// The model must take the five pinned inputs and give `sentence_embedding`.
fn check_signature(dir: &Path, session: &Session) -> Result<()> {
    let mut inputs: Vec<&str> = session.inputs().iter().map(|o| o.name()).collect();
    inputs.sort_unstable();
    let mut expected = INPUT_NAMES.to_vec();
    expected.sort_unstable();
    let has_output = session.outputs().iter().any(|o| o.name() == OUTPUT_NAME);
    if inputs != expected || !has_output {
        return Err(CoreError::new(
            ErrorCode::ModelIntegrity,
            format!(
                "the model in {} takes {inputs:?} and gives {:?}; expected {INPUT_NAMES:?} and {OUTPUT_NAME}",
                dir.display(),
                session.outputs().iter().map(|o| o.name()).collect::<Vec<_>>()
            ),
        )
        .with_next_action("Run `spatial-memory setup` to download the pinned model again."));
    }
    Ok(())
}

/// Runs one padded batch and returns its unit vectors in row order.
fn run_batch(session: &mut Session, batch: PaddedBatch) -> Result<Vec<Vec<f32>>> {
    let rows = batch.rows;
    if rows == 0 {
        return Ok(Vec::new());
    }
    let shape = [rows, batch.cols];
    let ids =
        Tensor::from_array((shape, batch.ids)).map_err(ort_error("cannot build input_ids"))?;
    let mask = Tensor::from_array((shape, batch.mask))
        .map_err(ort_error("cannot build attention_mask"))?;
    let empty = || {
        Tensor::from_array(Array2::<f32>::zeros((0, FEATURE_DIM)))
            .map_err(ort_error("cannot build an empty feature input"))
    };
    let outputs = session
        .run(ort::inputs![
            "input_ids" => ids,
            "attention_mask" => mask,
            "image_features" => empty()?,
            "video_features" => empty()?,
            "audio_features" => empty()?,
        ])
        .map_err(ort_error("inference failed"))?;
    let output = outputs
        .get(OUTPUT_NAME)
        .ok_or_else(|| CoreError::internal(format!("the model gave no {OUTPUT_NAME}")))?;
    let (shape, data) = output
        .try_extract_tensor::<f32>()
        .map_err(ort_error("cannot read sentence_embedding"))?;
    split_output(shape, data, rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prepared(lens: &[usize]) -> Vec<Prepared> {
        lens.iter()
            .enumerate()
            .map(|(i, &n)| Prepared {
                ids: std::iter::once(i as i64 + 100)
                    .chain((1..n).map(|_| 7))
                    .collect(),
                truncated: false,
            })
            .collect()
    }

    /// A fake model: each vector holds its row's first id and the batch
    /// width, so order and padding can be checked.
    fn fake_run(batch: &PaddedBatch) -> Vec<Vec<f32>> {
        (0..batch.rows)
            .map(|r| vec![batch.ids[r * batch.cols] as f32, batch.cols as f32])
            .collect()
    }

    #[test]
    fn s5_config_follows_the_settings() {
        let c = EmbedConfig::default();
        assert_eq!(
            (c.chunk_max_tokens, c.batch_size, c.spinning),
            (256, 4, false)
        );
        let mut s = Settings::default();
        s.embed.batch_size = 9;
        s.embed.spinning = true;
        s.chunk.max_tokens = 512;
        let c = EmbedConfig::from_settings(&s);
        assert_eq!(
            (c.chunk_max_tokens, c.batch_size, c.spinning),
            (512, 9, true)
        );
    }

    #[test]
    fn s5_batches_never_exceed_batch_size_and_keep_input_order() {
        let docs = prepared(&[9, 3, 5, 3, 12, 1, 7]);
        let mut sizes = Vec::new();
        let out = embed_in_batches(&docs, 3, |b| {
            sizes.push(b.rows);
            Ok(fake_run(&b))
        })
        .unwrap();
        assert_eq!(sizes, [3, 3, 1]);
        for (i, v) in out.iter().enumerate() {
            assert_eq!(v[0], i as f32 + 100.0, "vector {i} is out of order");
        }
        // Shortest first: the first batch is padded only to its longest.
        assert_eq!(out[5][1], 3.0);
        assert_eq!(out[4][1], 12.0);
        assert!(
            embed_in_batches(&[], 4, |_| unreachable!())
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn s5_a_short_batch_result_is_an_internal_error() {
        let docs = prepared(&[2, 2]);
        let err = embed_in_batches(&docs, 4, |_| Ok(vec![vec![0.0]])).unwrap_err();
        assert_eq!(err.code, ErrorCode::Internal);
    }

    #[test]
    fn at10_load_refuses_a_missing_model() {
        let dir = tempfile::tempdir().unwrap();
        let err = GemmaProvider::load(dir.path(), EmbedConfig::default()).unwrap_err();
        assert_eq!(err.code, ErrorCode::ModelMissing);
    }
}
