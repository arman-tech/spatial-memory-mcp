//! The injectable model loader (SPEC 14, stream E, Entry point).
//!
//! The daemon learns the model's identity (`info`) before loading it, so the
//! index stamp is configured at start and keyword recall, forget, link and
//! status are served while the model loads or after it fails (B-36).

use std::path::PathBuf;
use std::sync::Arc;

use sm_core::config::{EmbedModel, Settings};
use sm_core::error::{CoreError, Result};
use sm_core::traits::{Chunker, EmbedderInfo, EmbedderProvider};
use sm_embed::{EmbedConfig, GemmaProvider};

/// A loaded model: the provider that opens sessions and the chunker that
/// shares its tokenizer.
#[derive(Clone)]
pub struct LoadedModel {
    pub provider: Arc<dyn EmbedderProvider>,
    pub chunker: Arc<dyn Chunker>,
}

/// Loads the embedding model. `load` runs on a background thread while the
/// daemon answers `hello` with `loading`.
pub trait ModelLoader: Send + Sync {
    /// The identity the loaded model will report. Must not touch the files.
    fn info(&self) -> EmbedderInfo;
    /// Verifies and loads the model.
    fn load(&self) -> Result<LoadedModel>;
}

/// The production loader: `GemmaProvider::load`, which verifies the files
/// itself, then the tokenizer-backed chunker.
pub struct GemmaLoader {
    dir: PathBuf,
    config: EmbedConfig,
    model: EmbedModel,
}

impl GemmaLoader {
    pub fn new(dir: PathBuf, settings: &Settings) -> Self {
        GemmaLoader {
            dir,
            config: EmbedConfig::from_settings(settings),
            model: settings.embed.model,
        }
    }
}

impl ModelLoader for GemmaLoader {
    fn info(&self) -> EmbedderInfo {
        EmbedderInfo {
            model_id: self.model.as_str().to_owned(),
            revision: match self.model {
                EmbedModel::EmbeddingGemma2 => sm_embed::model::REVISION.to_owned(),
                EmbedModel::ArcticEmbedM => String::new(),
            },
            dim: sm_core::EMBED_DIM,
        }
    }

    fn load(&self) -> Result<LoadedModel> {
        if self.model != EmbedModel::EmbeddingGemma2 {
            // The fallback model sits behind sm-embed's `arctic-embed-m`
            // feature, which this crate does not enable (B-41).
            return Err(CoreError::config_invalid(format!(
                "embed.model = {:?} is not built into this daemon",
                self.model.as_str()
            ))
            .with_next_action("Set embed.model = \"embeddinggemma-2\" in config.toml."));
        }
        let provider = GemmaProvider::load(&self.dir, self.config)?;
        let chunker = provider.chunker();
        Ok(LoadedModel {
            provider: Arc::new(provider),
            chunker: Arc::new(chunker),
        })
    }
}
