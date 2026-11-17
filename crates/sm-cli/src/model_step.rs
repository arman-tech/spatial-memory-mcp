//! The model step of `setup` and `doctor` (SPEC section 3 and 5; decision
//! B-43).
//!
//! `setup` makes the configured model present and verified: when a pinned
//! file is missing or does not match its size and SHA-256, it downloads the
//! files into the per-user model cache (`SPATIAL_MEMORY_MODEL_DIR`, or
//! `<cache>/spatial-memory/_models/<id>/<revision>/`) and verifies every
//! file again. `doctor` only verifies. Both go through [`ModelStep`], so
//! tests supply their own; the binary uses [`EmbedModelStep`] over
//! `sm-embed`'s download and checksums.
//!
//! The files come from the pinned Hugging Face revision, or from the
//! address in `SPATIAL_MEMORY_MODEL_URL` ([`MODEL_URL_ENV`]) when it is set:
//! the mirror that SPEC section 13 suggests for a community export that can
//! change. The address only says where; the sizes and SHA-256 values are
//! still the pinned ones, so a mirror cannot supply different files.

use std::path::PathBuf;
use std::sync::Arc;

use sm_core::config::{EmbedModel, Settings};
use sm_core::error::{CoreError, ErrorCode};
use sm_embed::download::{self, DownloadEvent};
use sm_embed::model::{self, ModelFile};

/// Downloads and verifies the embedding model.
pub trait ModelStep: Send + Sync {
    /// Makes the configured model present and verified, downloading it when
    /// it is missing (`setup`). Returns a line describing the model.
    fn ensure(&self, settings: &Settings) -> Result<String, CoreError>;

    /// Verifies the configured model without downloading (`doctor`).
    /// Returns a line describing the model.
    fn check(&self, settings: &Settings) -> Result<String, CoreError>;
}

/// Reads one environment variable; `None` when it is unset.
pub type SharedEnv = Arc<dyn Fn(&str) -> Option<String> + Send + Sync>;

/// The folder address the model files are downloaded from, in place of the
/// pinned Hugging Face revision: each file is `<address>/<file path>`, for
/// example `<address>/onnx/model.onnx`. Unset or blank means the pinned
/// revision.
pub const MODEL_URL_ENV: &str = "SPATIAL_MEMORY_MODEL_URL";

/// The model step over `sm-embed`: the pinned EmbeddingGemma 2 files, from
/// the pinned Hugging Face revision, into the model cache.
pub struct EmbedModelStep {
    /// Where the model folder is found (`SPATIAL_MEMORY_MODEL_DIR` and the
    /// platform cache folder).
    pub get_env: SharedEnv,
    /// `<base_url>/<file path>` is each file's address.
    pub base_url: String,
    /// The files, with their sizes and SHA-256 values.
    pub files: Vec<ModelFile<'static>>,
    /// Called for each download event; `setup` prints them on stderr.
    pub progress: fn(&DownloadEvent<'_>),
}

impl EmbedModelStep {
    /// The binary's step: the process environment, the pinned files, and
    /// progress on stderr.
    pub fn pinned() -> Self {
        Self::from_env(Arc::new(crate::profile::process_env))
    }

    /// The pinned files, from the address [`MODEL_URL_ENV`] names in
    /// `get_env` or else the pinned revision, into the model folder
    /// `get_env` names.
    pub fn from_env(get_env: SharedEnv) -> Self {
        let base_url = get_env(MODEL_URL_ENV)
            .map(|v| v.trim().to_owned())
            .filter(|v| !v.is_empty())
            .unwrap_or_else(download::pinned_base_url);
        EmbedModelStep {
            get_env,
            base_url,
            files: model::MODEL_FILES.to_vec(),
            progress: print_progress,
        }
    }

    /// The model folder for `settings`. Only EmbeddingGemma 2 is built into
    /// this program, as in the daemon's loader (B-41).
    pub fn model_dir(&self, settings: &Settings) -> Result<PathBuf, CoreError> {
        if settings.embed.model != EmbedModel::EmbeddingGemma2 {
            return Err(CoreError::config_invalid(format!(
                "embed.model = {:?} is not built into this program",
                settings.embed.model.as_str()
            ))
            .with_next_action("Set embed.model = \"embeddinggemma-2\" in config.toml."));
        }
        let get_env = Arc::clone(&self.get_env);
        sm_core::paths::model_dir(move |k| get_env(k), model::MODEL_ID, model::REVISION)
    }

    fn verified(&self, settings: &Settings) -> Result<(PathBuf, model::VerifyReport), CoreError> {
        let dir = self.model_dir(settings)?;
        let report = model::verify_files(&dir, &self.files)?;
        Ok((dir, report))
    }
}

/// One line per file started and finished, on stderr: the weights are a
/// gigabyte, so `setup` says what it is waiting for.
fn print_progress(event: &DownloadEvent<'_>) {
    match event {
        DownloadEvent::Started { path, size } => {
            eprintln!("downloading {path} ({size} bytes)");
        }
        DownloadEvent::Finished { path } => eprintln!("verified {path}"),
        DownloadEvent::Skipped { .. } | DownloadEvent::Progress { .. } => {}
    }
}

fn describe(dir: &std::path::Path, report: &model::VerifyReport) -> String {
    format!(
        "{} {} in {}: {} files, {} bytes, SHA-256 verified",
        model::MODEL_ID,
        model::REVISION,
        dir.display(),
        report.files.len(),
        report.bytes()
    )
}

impl ModelStep for EmbedModelStep {
    fn ensure(&self, settings: &Settings) -> Result<String, CoreError> {
        let dir = self.model_dir(settings)?;
        match model::verify_files(&dir, &self.files) {
            Ok(report) => return Ok(describe(&dir, &report)),
            Err(e) if matches!(e.code, ErrorCode::ModelMissing | ErrorCode::ModelIntegrity) => {}
            Err(e) => return Err(e),
        }
        let progress = self.progress;
        let fetched = download::download_files(&dir, &self.base_url, &self.files, &mut |e| {
            progress(e);
        })?;
        let report = model::verify_files(&dir, &self.files)?;
        Ok(format!(
            "{} (downloaded {} of {} files, {} bytes)",
            describe(&dir, &report),
            fetched.fetched.len(),
            self.files.len(),
            fetched.bytes
        ))
    }

    fn check(&self, settings: &Settings) -> Result<String, CoreError> {
        let (dir, report) = self.verified(settings)?;
        Ok(describe(&dir, &report))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use sm_core::hash::sha256_hex;

    use super::*;

    /// An HTTP/1.1 server on 127.0.0.1 that answers from a table and counts
    /// requests, as `sm-embed`'s download tests do. Nothing leaves the
    /// machine.
    struct Server {
        base: String,
        requests: Arc<AtomicUsize>,
        table: Arc<Mutex<HashMap<String, Vec<u8>>>>,
    }

    impl Server {
        fn start() -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let base = format!("http://{}/repo", listener.local_addr().unwrap());
            let requests = Arc::new(AtomicUsize::new(0));
            let table: Arc<Mutex<HashMap<String, Vec<u8>>>> = Arc::default();
            let (r, t) = (Arc::clone(&requests), Arc::clone(&table));
            std::thread::spawn(move || {
                for stream in listener.incoming() {
                    let Ok(mut stream) = stream else { return };
                    r.fetch_add(1, Ordering::SeqCst);
                    let mut reader = BufReader::new(stream.try_clone().unwrap());
                    let mut line = String::new();
                    if reader.read_line(&mut line).is_err() {
                        continue;
                    }
                    let path = line.split_whitespace().nth(1).unwrap_or("").to_owned();
                    loop {
                        let mut h = String::new();
                        if reader.read_line(&mut h).is_err() || h == "\r\n" || h.is_empty() {
                            break;
                        }
                    }
                    let body = t.lock().unwrap().get(&path).cloned();
                    let (status, body) = match body {
                        Some(b) => (200, b),
                        None => (404, Vec::new()),
                    };
                    let head = format!(
                        "HTTP/1.1 {status} X\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        body.len()
                    );
                    let _ = stream.write_all(head.as_bytes());
                    let _ = stream.write_all(&body);
                    let _ = stream.flush();
                }
            });
            Server {
                base,
                requests,
                table,
            }
        }

        fn serve(&self, path: &str, body: &[u8]) {
            self.table
                .lock()
                .unwrap()
                .insert(format!("/repo/{path}"), body.to_vec());
        }

        fn requests(&self) -> usize {
            self.requests.load(Ordering::SeqCst)
        }
    }

    /// Two synthetic model files, served by a local server, and a model
    /// folder named by `SPATIAL_MEMORY_MODEL_DIR` inside a temporary folder.
    struct Rig {
        _tmp: tempfile::TempDir,
        dir: PathBuf,
        server: Server,
        step: EmbedModelStep,
    }

    const GRAPH: &[u8] = b"synthetic graph bytes";
    const VOCAB: &[u8] = b"{\"synthetic\": \"tokenizer\"}";

    fn leak(s: String) -> &'static str {
        Box::leak(s.into_boxed_str())
    }

    fn file(path: &'static str, body: &[u8]) -> ModelFile<'static> {
        ModelFile {
            path,
            size: body.len() as u64,
            sha256: leak(sha256_hex(body)),
        }
    }

    fn quiet(_: &DownloadEvent<'_>) {}

    fn rig() -> Rig {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("models").join("gemma");
        let server = Server::start();
        server.serve("onnx/model.onnx", GRAPH);
        server.serve("tokenizer.json", VOCAB);
        let model_dir = dir.to_string_lossy().into_owned();
        let step = EmbedModelStep {
            get_env: Arc::new(move |k: &str| {
                (k == sm_core::paths::MODEL_DIR_ENV).then(|| model_dir.clone())
            }),
            base_url: server.base.clone(),
            files: vec![
                file("onnx/model.onnx", GRAPH),
                file("tokenizer.json", VOCAB),
            ],
            progress: quiet,
        };
        Rig {
            _tmp: tmp,
            dir,
            server,
            step,
        }
    }

    #[test]
    fn s5_setup_downloads_a_missing_model_into_the_model_folder_and_verifies_it() {
        let r = rig();
        let s = Settings::default();
        let line = r.step.ensure(&s).unwrap();
        assert!(line.contains("downloaded 2 of 2 files"), "{line}");
        assert!(line.contains("SHA-256 verified"), "{line}");
        assert!(line.contains(&r.dir.display().to_string()), "{line}");
        assert_eq!(
            std::fs::read(r.dir.join("onnx").join("model.onnx")).unwrap(),
            GRAPH
        );
        assert_eq!(std::fs::read(r.dir.join("tokenizer.json")).unwrap(), VOCAB);
        assert_eq!(r.server.requests(), 2);
        // Present and verified: nothing is fetched again.
        let again = r.step.ensure(&s).unwrap();
        assert!(!again.contains("downloaded"), "{again}");
        assert_eq!(r.server.requests(), 2);
        assert_eq!(r.step.check(&s).unwrap(), again);
    }

    #[test]
    fn s5_setup_replaces_a_file_whose_hash_does_not_match() {
        let r = rig();
        let s = Settings::default();
        r.step.ensure(&s).unwrap();
        // Same size, one byte changed.
        let mut bad = VOCAB.to_vec();
        bad[3] ^= 0x01;
        std::fs::write(r.dir.join("tokenizer.json"), &bad).unwrap();
        let e = r.step.check(&s).unwrap_err();
        assert_eq!(e.code, ErrorCode::ModelIntegrity);
        let line = r.step.ensure(&s).unwrap();
        assert!(line.contains("downloaded 1 of 2 files"), "{line}");
        assert_eq!(std::fs::read(r.dir.join("tokenizer.json")).unwrap(), VOCAB);
        assert_eq!(r.server.requests(), 3);
    }

    #[test]
    fn s5_setup_refuses_a_download_that_does_not_match_the_pinned_hash() {
        let r = rig();
        r.server
            .serve("tokenizer.json", b"{\"something\": \"else\"}...");
        let e = r.step.ensure(&Settings::default()).unwrap_err();
        assert_eq!(e.code, ErrorCode::ModelIntegrity, "{e:?}");
        assert!(!e.next_action.is_empty());
        assert!(!r.dir.join("tokenizer.json").exists());
    }

    #[test]
    fn s5_doctor_verifies_without_downloading_and_says_what_to_do() {
        let r = rig();
        let s = Settings::default();
        let e = r.step.check(&s).unwrap_err();
        assert_eq!(e.code, ErrorCode::ModelMissing);
        assert!(e.message.contains("onnx/model.onnx"), "{}", e.message);
        assert!(e.next_action.contains("spatial-memory setup"), "{e:?}");
        assert_eq!(r.server.requests(), 0);
        assert!(!r.dir.exists());
    }

    #[test]
    fn s5_a_model_that_is_not_built_in_is_refused_with_a_next_action() {
        let r = rig();
        let mut s = Settings::default();
        s.embed.model = EmbedModel::ArcticEmbedM;
        for e in [
            r.step.ensure(&s).unwrap_err(),
            r.step.check(&s).unwrap_err(),
        ] {
            assert_eq!(e.code, ErrorCode::ConfigInvalid);
            assert!(e.next_action.contains("embeddinggemma-2"), "{e:?}");
        }
        assert_eq!(r.server.requests(), 0);
    }

    #[test]
    fn s5_the_binary_step_is_the_pinned_revision_and_files() {
        // The test process's own environment may name a mirror; the step
        // built from an empty one is the pinned revision.
        let step = EmbedModelStep::from_env(Arc::new(|_: &str| None));
        assert_eq!(step.base_url, download::pinned_base_url());
        assert!(step.base_url.ends_with(model::REVISION));
        assert_eq!(step.files, model::MODEL_FILES);
        let blank = EmbedModelStep::from_env(Arc::new(|k: &str| {
            (k == MODEL_URL_ENV).then(|| "  ".to_owned())
        }));
        assert_eq!(blank.base_url, download::pinned_base_url());
        assert_eq!(EmbedModelStep::pinned().files, model::MODEL_FILES);
    }

    #[test]
    fn s5_a_mirror_address_is_used_and_the_pinned_hashes_still_hold() {
        // SPEC 13: "Mirror the three files to storage you control." The
        // mirror here serves other bytes under the pinned names; nothing is
        // kept, and the error says which file and what to do.
        let r = rig();
        for f in model::MODEL_FILES {
            r.server.serve(f.path, b"not the pinned file");
        }
        let dir = r.dir.to_string_lossy().into_owned();
        let base = format!("{}/", r.server.base);
        let mut step = EmbedModelStep::from_env(Arc::new(move |k: &str| match k {
            sm_core::paths::MODEL_DIR_ENV => Some(dir.clone()),
            MODEL_URL_ENV => Some(base.clone()),
            _ => None,
        }));
        step.progress = quiet;
        assert_eq!(step.base_url, format!("{}/", r.server.base));
        assert_eq!(step.files, model::MODEL_FILES);
        let e = step.ensure(&Settings::default()).unwrap_err();
        assert_eq!(e.code, ErrorCode::ModelIntegrity, "{e:?}");
        assert!(e.message.contains(&r.server.base), "{}", e.message);
        assert!(!e.next_action.is_empty());
        assert_eq!(r.server.requests(), 1);
        for f in model::MODEL_FILES {
            assert!(!f.local_path(&r.dir).exists(), "{}", f.path);
        }
    }
}
