//! The pinned EmbeddingGemma 2 files and the integrity check (SPEC 5,
//! "Model, pinned"; AT-10).
//!
//! [`verify`] checks the size of every file, then its SHA-256. The daemon
//! runs it before loading, so one changed byte stops the start with
//! `model_integrity` and a next action.

use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};
use sm_core::config::{EMBEDDINGGEMMA_REVISION, EmbedModel};
use sm_core::error::{CoreError, ErrorCode, Result};
use sm_core::hash::to_hex;

/// The model id stamped into the index.
pub const MODEL_ID: &str = EmbedModel::EmbeddingGemma2.as_str();
/// The pinned revision of the Hugging Face repository.
pub const REVISION: &str = EMBEDDINGGEMMA_REVISION;
/// The Hugging Face repository the files come from.
pub const HF_REPO: &str = "onnx-community/embeddinggemma-2-ONNX";

/// One file of the model: where it lives under the model folder, its size
/// and its SHA-256 (lower-case hex).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelFile<'a> {
    /// Relative path with `/` separators, as in the repository.
    pub path: &'a str,
    pub size: u64,
    pub sha256: &'a str,
}

impl ModelFile<'_> {
    /// The file's location under `dir`.
    pub fn local_path(&self, dir: &Path) -> PathBuf {
        let mut p = dir.to_path_buf();
        for part in self.path.split('/') {
            p.push(part);
        }
        p
    }
}

/// The ONNX graph (session entry point).
pub const MODEL_ONNX: ModelFile<'static> = ModelFile {
    path: "onnx/model.onnx",
    size: 421_421,
    sha256: "bc47de15f81208a5c99e5ab10f746d5e33b51ea228b7dc0bef9c133a94f1c1c3",
};
/// The external weights the graph refers to.
pub const MODEL_DATA: ModelFile<'static> = ModelFile {
    path: "onnx/model.onnx_data",
    size: 1_084_170_240,
    sha256: "9fd452bfc6916e5a92f080f8dada35598ca4512b21c65897d4c5974d5f9630a2",
};
/// The tokenizer.
pub const TOKENIZER: ModelFile<'static> = ModelFile {
    path: "tokenizer.json",
    size: 32_170_510,
    sha256: "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4",
};
/// Every pinned file, smallest first so a broken download shows quickly.
pub const MODEL_FILES: [ModelFile<'static>; 3] = [MODEL_ONNX, TOKENIZER, MODEL_DATA];

/// The model folder: `SPATIAL_MEMORY_MODEL_DIR`, or the per-user cache
/// (`sm_core::paths::model_dir`, decision B-3).
pub fn default_model_dir() -> Result<PathBuf> {
    sm_core::paths::model_dir(|k| std::env::var(k).ok(), MODEL_ID, REVISION)
}

/// Read buffer for hashing: large enough that the hash, not the system
/// call, is the cost.
const HASH_BUFFER: usize = 4 << 20;

/// How one file checked out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileCheck {
    pub path: String,
    pub bytes: u64,
    /// Time to hash the file.
    pub elapsed: Duration,
}

/// What [`verify`] checked and how long it took.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifyReport {
    pub dir: PathBuf,
    pub files: Vec<FileCheck>,
    /// The whole check: sizes and hashes.
    pub elapsed: Duration,
}

impl VerifyReport {
    /// Bytes hashed.
    pub fn bytes(&self) -> u64 {
        self.files.iter().map(|f| f.bytes).sum()
    }
}

/// Checks the pinned model files in `dir`: every size first, then every
/// SHA-256. A missing file is `model_missing`; a wrong size or hash is
/// `model_integrity`. Each error names the file and the folder.
pub fn verify(dir: &Path) -> Result<VerifyReport> {
    verify_files(dir, &MODEL_FILES)
}

/// [`verify`] for any list of files (tests use synthetic ones).
pub fn verify_files(dir: &Path, files: &[ModelFile<'_>]) -> Result<VerifyReport> {
    let started = Instant::now();
    for file in files {
        check_size(dir, file)?;
    }
    let mut checks = Vec::with_capacity(files.len());
    for file in files {
        let t = Instant::now();
        let actual = hash_file(&file.local_path(dir)).map_err(|e| unreadable(dir, file, &e))?;
        if actual != file.sha256 {
            return Err(hash_mismatch(dir, file, &actual));
        }
        checks.push(FileCheck {
            path: file.path.to_owned(),
            bytes: file.size,
            elapsed: t.elapsed(),
        });
    }
    Ok(VerifyReport {
        dir: dir.to_path_buf(),
        files: checks,
        elapsed: started.elapsed(),
    })
}

/// Whether one file is present with the pinned size and hash. Used by the
/// download to skip files that are already in place.
pub fn file_is_verified(dir: &Path, file: &ModelFile<'_>) -> bool {
    check_size(dir, file).is_ok()
        && hash_file(&file.local_path(dir)).is_ok_and(|h| h == file.sha256)
}

fn check_size(dir: &Path, file: &ModelFile<'_>) -> Result<()> {
    let path = file.local_path(dir);
    let meta = match std::fs::metadata(&path) {
        Ok(m) => m,
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            return Err(missing(dir, file));
        }
        Err(e) => return Err(unreadable(dir, file, &e)),
    };
    if !meta.is_file() {
        return Err(missing(dir, file));
    }
    if meta.len() != file.size {
        return Err(integrity(
            dir,
            format!(
                "model file {} in {} is {} bytes; the pinned size is {} bytes",
                file.path,
                dir.display(),
                meta.len(),
                file.size
            ),
        ));
    }
    Ok(())
}

/// SHA-256 of a file, lower-case hex.
pub fn hash_file(path: &Path) -> io::Result<String> {
    let mut f = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; HASH_BUFFER];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(to_hex(hasher.finalize().as_slice()))
}

pub(crate) fn missing(dir: &Path, file: &ModelFile<'_>) -> CoreError {
    CoreError::new(
        ErrorCode::ModelMissing,
        format!("model file {} is missing from {}", file.path, dir.display()),
    )
    .with_next_action(format!(
        "Run `spatial-memory setup` to download the model, or set {} to the folder holding it.",
        sm_core::paths::MODEL_DIR_ENV
    ))
}

fn unreadable(dir: &Path, file: &ModelFile<'_>, e: &io::Error) -> CoreError {
    CoreError::new(
        ErrorCode::ModelMissing,
        format!(
            "model file {} in {} cannot be read: {e}",
            file.path,
            dir.display()
        ),
    )
    .with_next_action(format!(
        "Check the permissions on {}, or run `spatial-memory setup` again.",
        dir.display()
    ))
}

pub(crate) fn hash_mismatch(dir: &Path, file: &ModelFile<'_>, actual: &str) -> CoreError {
    integrity(
        dir,
        format!(
            "model file {} in {} has SHA-256 {actual}; the pinned value is {}",
            file.path,
            dir.display(),
            file.sha256
        ),
    )
}

pub(crate) fn integrity(dir: &Path, message: String) -> CoreError {
    CoreError::new(ErrorCode::ModelIntegrity, message).with_next_action(format!(
        "Delete the model directory {} and run `spatial-memory setup` again.",
        dir.display()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sm_core::hash::sha256_hex;

    struct Synthetic {
        dir: tempfile::TempDir,
        files: Vec<(String, u64, String)>,
    }

    impl Synthetic {
        fn new(contents: &[(&str, &[u8])]) -> Self {
            let dir = tempfile::tempdir().unwrap();
            let mut files = Vec::new();
            for (path, bytes) in contents {
                let f = ModelFile {
                    path,
                    size: 0,
                    sha256: "",
                };
                let p = f.local_path(dir.path());
                std::fs::create_dir_all(p.parent().unwrap()).unwrap();
                std::fs::write(&p, bytes).unwrap();
                files.push((path.to_string(), bytes.len() as u64, sha256_hex(bytes)));
            }
            Synthetic { dir, files }
        }

        fn pinned(&self) -> Vec<ModelFile<'_>> {
            self.files
                .iter()
                .map(|(p, s, h)| ModelFile {
                    path: p,
                    size: *s,
                    sha256: h,
                })
                .collect()
        }
    }

    fn sample() -> Synthetic {
        let big: Vec<u8> = (0..(HASH_BUFFER + 4097)).map(|i| (i % 251) as u8).collect();
        Synthetic::new(&[("onnx/model.onnx", b"graph bytes"), ("weights.bin", &big)])
    }

    #[test]
    fn at10_verify_accepts_matching_files() {
        let s = sample();
        let report = verify_files(s.dir.path(), &s.pinned()).unwrap();
        assert_eq!(report.files.len(), 2);
        assert_eq!(report.bytes(), s.files.iter().map(|f| f.1).sum::<u64>());
        for f in s.pinned() {
            assert!(file_is_verified(s.dir.path(), &f));
        }
    }

    #[test]
    fn at10_one_changed_byte_is_model_integrity() {
        let s = sample();
        let path = s.dir.path().join("weights.bin");
        let mut bytes = std::fs::read(&path).unwrap();
        // The byte after the first hash buffer: a sampled check that read
        // only the head would miss it.
        bytes[HASH_BUFFER + 1] ^= 0x01;
        std::fs::write(&path, &bytes).unwrap();
        let err = verify_files(s.dir.path(), &s.pinned()).unwrap_err();
        assert_eq!(err.code, ErrorCode::ModelIntegrity);
        assert!(err.message.contains("weights.bin"), "{}", err.message);
        assert!(err.message.contains("SHA-256"), "{}", err.message);
        assert!(
            err.next_action
                .contains(&s.dir.path().display().to_string()),
            "{}",
            err.next_action
        );
        assert!(!file_is_verified(s.dir.path(), &s.pinned()[1]));
    }

    #[test]
    fn at10_wrong_size_is_model_integrity_before_any_hash() {
        let s = sample();
        std::fs::write(s.dir.path().join("weights.bin"), b"short").unwrap();
        let err = verify_files(s.dir.path(), &s.pinned()).unwrap_err();
        assert_eq!(err.code, ErrorCode::ModelIntegrity);
        assert!(err.message.contains("weights.bin"));
        assert!(err.message.contains("5 bytes"), "{}", err.message);
    }

    #[test]
    fn at10_missing_file_is_model_missing() {
        let s = sample();
        std::fs::remove_file(s.dir.path().join("onnx").join("model.onnx")).unwrap();
        let err = verify_files(s.dir.path(), &s.pinned()).unwrap_err();
        assert_eq!(err.code, ErrorCode::ModelMissing);
        assert!(err.message.contains("onnx/model.onnx"), "{}", err.message);
        assert!(err.next_action.contains("setup"));
        // A folder in place of a file is missing too.
        std::fs::create_dir(s.dir.path().join("onnx").join("model.onnx")).unwrap();
        let err = verify_files(s.dir.path(), &s.pinned()).unwrap_err();
        assert_eq!(err.code, ErrorCode::ModelMissing);
    }

    #[test]
    fn s5_pinned_files_match_the_spec_table() {
        assert_eq!(MODEL_ID, "embeddinggemma-2");
        assert_eq!(REVISION, "daa72c51243991dfcaf9f9137d2c573d8f7790c0");
        let total: u64 = MODEL_FILES.iter().map(|f| f.size).sum();
        assert_eq!(total, 421_421 + 1_084_170_240 + 32_170_510);
        for f in MODEL_FILES {
            assert_eq!(f.sha256.len(), 64);
            assert!(
                f.sha256
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
            );
        }
        let p = MODEL_DATA.local_path(Path::new("root"));
        assert_eq!(p, Path::new("root").join("onnx").join("model.onnx_data"));
    }
}
