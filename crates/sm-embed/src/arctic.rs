//! The fallback model: Snowflake Arctic Embed M through `fastembed` 7.7.0,
//! behind the same traits (D-5, B-23). Feature `arctic-embed-m`, off by
//! default.
//!
//! fastembed downloads its own ONNX export into a cache folder set here
//! under the per-user model cache, and does not expose a revision, so the
//! revision recorded is the fastembed version (B-23). fastembed builds its
//! sessions itself and offers no thread-creation hook, so bulk work cannot
//! be lowered below normal priority: a bulk session takes the SPEC 13
//! fallback (half the budget) and reports it.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use sm_core::EMBED_DIM;
use sm_core::config::EmbedModel;
use sm_core::error::{CoreError, ErrorCode, Result};
use sm_core::paths::{MODEL_DIR_ENV, model_dir};
use sm_core::traits::{DocInput, Embedder, EmbedderInfo, EmbedderProvider, SessionKind};

use crate::chunker::StructuredChunker;
use crate::gemma::EmbedConfig;
use crate::input::split_output;
use crate::runtime;
use crate::threads::{
    PoolSnapshot, PriorityProbe, ThreadPlan, ThreadReport, physical_cores, plan_threads,
};
use crate::tokenize::{ModelTokenizer, TextTokenizer, cut_to_tokens};

/// The model id recorded in the index stamp.
pub const ARCTIC_MODEL_ID: &str = EmbedModel::ArcticEmbedM.as_str();
/// The revision recorded: the fastembed version (B-23).
pub const ARCTIC_REVISION: &str = "fastembed-7.7.0";
/// The query prefix from the model card. Documents have none.
pub const ARCTIC_QUERY_PREFIX: &str = "Represent this sentence for searching relevant passages: ";
/// The model's position limit.
pub const ARCTIC_MAX_LENGTH: usize = 512;
/// Room kept for the query prefix and the special tokens.
pub const ARCTIC_PROMPT_ROOM: usize = 32;
/// Why bulk work takes the SPEC 13 fallback with this model.
pub const NO_THREAD_HOOK: &str = "fastembed builds its own ONNX Runtime sessions and exposes no thread-creation hook, so bulk threads cannot be lowered below normal priority";

/// fastembed's cache folder: under the per-user model cache, never at
/// `SPATIAL_MEMORY_MODEL_DIR`, which names the EmbeddingGemma folder.
pub fn default_cache_dir() -> Result<PathBuf> {
    cache_dir_from(|k| std::env::var(k).ok())
}

fn cache_dir_from(get_env: impl Fn(&str) -> Option<String>) -> Result<PathBuf> {
    model_dir(
        |k| if k == MODEL_DIR_ENV { None } else { get_env(k) },
        ARCTIC_MODEL_ID,
        ARCTIC_REVISION,
    )
}

/// `chunk.max_tokens` must leave room for the prompt inside the model's 512
/// positions; fastembed would otherwise cut the input without saying so.
pub fn check_chunk_limit(chunk_max_tokens: usize) -> Result<()> {
    let most = ARCTIC_MAX_LENGTH - ARCTIC_PROMPT_ROOM;
    if chunk_max_tokens > most {
        return Err(CoreError::invalid_input(format!(
            "chunk.max_tokens is {chunk_max_tokens}; arctic-embed-m takes at most {most} tokens a chunk"
        ))
        .with_next_action(format!(
            "Set chunk.max_tokens to {most} or less, or use embed.model = \"embeddinggemma-2\"."
        )));
    }
    Ok(())
}

/// The query as the model expects it.
pub fn arctic_query(text: &str) -> String {
    format!("{ARCTIC_QUERY_PREFIX}{text}")
}

fn fastembed_error(what: &'static str) -> impl Fn(fastembed::Error) -> CoreError {
    move |e| CoreError::internal(format!("{what}: {e}"))
}

fn open_model(cache: &Path, config: &EmbedConfig, threads: usize) -> Result<TextEmbedding> {
    let spin = if config.spinning { "1" } else { "0" };
    let options = TextInitOptions::new(EmbeddingModel::SnowflakeArcticEmbedM)
        .with_cache_dir(cache.to_path_buf())
        .with_show_download_progress(false)
        .with_max_length(ARCTIC_MAX_LENGTH)
        .with_intra_threads(threads)
        .with_session_config("session.intra_op.allow_spinning", spin)
        .with_session_config("session.inter_op.allow_spinning", spin)
        .with_environment(runtime::environment()?);
    TextEmbedding::try_new(options).map_err(|e| {
        CoreError::new(
            ErrorCode::ModelMissing,
            format!("cannot load arctic-embed-m from {}: {e}", cache.display()),
        )
        .with_next_action("Check the network connection, then run `spatial-memory setup`.")
    })
}

/// Opens Arctic Embed M sessions. Loading fetches the model on first use.
#[derive(Debug)]
pub struct ArcticProvider {
    cache: PathBuf,
    tok: Arc<ModelTokenizer>,
    config: EmbedConfig,
    info: EmbedderInfo,
    physical: usize,
}

impl ArcticProvider {
    /// Loads the model once into `cache` (fastembed downloads it if
    /// absent) and keeps its tokenizer for the chunker and the query cut.
    pub fn load(cache: &Path, config: EmbedConfig) -> Result<Self> {
        check_chunk_limit(config.chunk_max_tokens)?;
        let model = open_model(cache, &config, 1)?;
        let tok = ModelTokenizer::from_tokenizer(model.tokenizer.clone())?;
        Ok(ArcticProvider {
            cache: cache.to_path_buf(),
            tok: Arc::new(tok),
            config,
            info: EmbedderInfo {
                model_id: ARCTIC_MODEL_ID.to_owned(),
                revision: ARCTIC_REVISION.to_owned(),
                dim: EMBED_DIM,
            },
            physical: physical_cores(),
        })
    }

    /// A chunker counting with this model's tokenizer.
    pub fn chunker(&self) -> StructuredChunker {
        StructuredChunker::new(self.tok.clone())
    }

    pub fn open_arctic(&self, kind: SessionKind, threads: usize) -> Result<ArcticEmbedder> {
        let plan = plan_threads(kind, threads, self.physical, &no_hook());
        let model = open_model(&self.cache, &self.config, plan.intra_threads)?;
        Ok(ArcticEmbedder {
            info: self.info.clone(),
            tok: Arc::clone(&self.tok),
            config: self.config,
            plan,
            model: Mutex::new(model),
        })
    }
}

/// The probe result for this model: lowering is never possible.
fn no_hook() -> PriorityProbe {
    PriorityProbe {
        lowered_to: None,
        refused: Some(NO_THREAD_HOOK.to_owned()),
    }
}

impl EmbedderProvider for ArcticProvider {
    fn info(&self) -> &EmbedderInfo {
        &self.info
    }

    fn open(&self, kind: SessionKind, threads: usize) -> Result<Box<dyn Embedder>> {
        Ok(Box::new(self.open_arctic(kind, threads)?))
    }
}

/// One Arctic Embed M session.
pub struct ArcticEmbedder {
    info: EmbedderInfo,
    tok: Arc<ModelTokenizer>,
    config: EmbedConfig,
    plan: ThreadPlan,
    model: Mutex<TextEmbedding>,
}

impl std::fmt::Debug for ArcticEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArcticEmbedder")
            .field("info", &self.info)
            .field("plan", &self.plan)
            .finish_non_exhaustive()
    }
}

impl ArcticEmbedder {
    /// The session's threads as planned. fastembed's pool is not observable.
    pub fn threads(&self) -> ThreadReport {
        ThreadReport {
            plan: self.plan.clone(),
            pool: PoolSnapshot::default(),
        }
    }

    fn embed(&self, texts: Vec<String>, batch: usize) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let rows = texts.len();
        let vectors = self
            .model
            .lock()
            .map_err(|_| CoreError::internal("the arctic session is poisoned"))?
            .embed(texts, Some(batch.max(1)))
            .map_err(fastembed_error("arctic-embed-m inference failed"))?;
        let flat: Vec<f32> = vectors.concat();
        let width = if rows == vectors.len() && flat.len() == rows * EMBED_DIM {
            EMBED_DIM as i64
        } else {
            -1
        };
        split_output(&[vectors.len() as i64, width], &flat, rows)
    }
}

impl Embedder for ArcticEmbedder {
    fn info(&self) -> &EmbedderInfo {
        &self.info
    }

    fn embed_query(&self, text: &str, max_tokens: usize) -> Result<Vec<f32>> {
        let max = max_tokens.min(self.config.chunk_max_tokens.max(1));
        let cut = cut_to_tokens(&*self.tok, text, max)?;
        self.embed(vec![arctic_query(cut)], 1)?
            .pop()
            .ok_or_else(|| CoreError::internal("the model returned no vector for the query"))
    }

    /// Documents are embedded as given; this model has no document prompt,
    /// so the title is not used.
    fn embed_documents(&self, docs: &[DocInput<'_>]) -> Result<Vec<Vec<f32>>> {
        let limit = self.config.chunk_max_tokens.max(1);
        let mut texts = Vec::with_capacity(docs.len());
        for doc in docs {
            let n = self.tok.count(doc.text)?;
            if n > limit {
                return Err(CoreError::invalid_input(format!(
                    "a chunk of {n} tokens is over the chunk limit of {limit} tokens (SPEC 5, rule 6)"
                ))
                .with_next_action("Split the text with the chunker before embedding it."));
            }
            texts.push(doc.text.to_owned());
        }
        self.embed(texts, self.config.batch_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn d5_fallback_cache_is_under_the_model_cache_not_the_override() {
        let env = |k: &str| match k {
            "SPATIAL_MEMORY_MODEL_DIR" => Some("/elsewhere/gemma".to_owned()),
            "XDG_CACHE_HOME" | "LOCALAPPDATA" => Some("/cache".to_owned()),
            "HOME" => Some("/home/u".to_owned()),
            _ => None,
        };
        let dir = cache_dir_from(env).unwrap();
        let s = dir.to_string_lossy().replace('\\', "/");
        assert!(!s.starts_with("/elsewhere"), "{s}");
        assert!(s.contains("arctic-embed-m"), "{s}");
        assert!(s.ends_with(ARCTIC_REVISION), "{s}");
    }

    #[test]
    fn d5_chunk_limit_leaves_room_for_the_prompt() {
        assert!(check_chunk_limit(256).is_ok());
        assert!(check_chunk_limit(480).is_ok());
        let err = check_chunk_limit(481).unwrap_err();
        assert_eq!(err.code, ErrorCode::InvalidInput);
    }

    #[test]
    fn d5_query_prompt_and_bulk_fallback() {
        assert_eq!(
            arctic_query("red planet"),
            "Represent this sentence for searching relevant passages: red planet"
        );
        let plan = plan_threads(SessionKind::Bulk, 8, 16, &no_hook());
        assert_eq!(plan.intra_threads, 4);
        assert!(plan.priority_fallback);
        let plan = plan_threads(SessionKind::Query, 4, 16, &no_hook());
        assert_eq!((plan.intra_threads, plan.priority_fallback), (4, false));
    }
}
