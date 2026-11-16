//! Downloads the pinned model files (B-42, B-43): ureq with rustls, one
//! file at a time, hashing as it writes.
//!
//! Each file streams into a temporary name inside its own folder of the
//! model directory and is renamed into place only when its size and SHA-256
//! match the pinned values. A mismatch deletes the partial file and returns
//! `model_integrity`. A file that is already present and verified is
//! skipped. `setup` and `doctor` call [`download_model`] in phase 3.

use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};
use sm_core::error::{CoreError, ErrorCode, Result};
use sm_core::hash::to_hex;

use crate::model::{self, HF_REPO, MODEL_FILES, ModelFile, REVISION};

/// Hugging Face's file endpoint for the pinned repository; the revision and
/// the file path follow.
pub const HF_RESOLVE: &str = "https://huggingface.co/onnx-community/embeddinggemma-2-ONNX/resolve";

/// `<HF_RESOLVE>/<revision>`: the base URL for the pinned files.
pub fn pinned_base_url() -> String {
    debug_assert!(HF_RESOLVE.contains(HF_REPO));
    format!("{HF_RESOLVE}/{REVISION}")
}

/// Connection and first-byte timeouts. The body has none: the weights are
/// 1.08 GB and a slow line is not an error.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
const RESPONSE_TIMEOUT: Duration = Duration::from_secs(60);
/// Copy buffer.
const CHUNK: usize = 1 << 20;

/// Progress, for `setup` to print.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DownloadEvent<'a> {
    /// The file is already present and verified.
    Skipped {
        path: &'a str,
    },
    Started {
        path: &'a str,
        size: u64,
    },
    /// Bytes written so far.
    Progress {
        path: &'a str,
        done: u64,
        size: u64,
    },
    /// Verified and renamed into place.
    Finished {
        path: &'a str,
    },
}

/// What a download did.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DownloadReport {
    pub fetched: Vec<String>,
    pub skipped: Vec<String>,
    /// Bytes transferred.
    pub bytes: u64,
    pub elapsed: Duration,
}

/// Downloads the pinned EmbeddingGemma 2 files into `dir` (normally
/// [`model::default_model_dir`]) and returns once every file is verified.
pub fn download_model(
    dir: &Path,
    progress: &mut dyn FnMut(&DownloadEvent<'_>),
) -> Result<DownloadReport> {
    download_files(dir, &pinned_base_url(), &MODEL_FILES, progress)
}

/// Downloads `files` from `<base_url>/<path>` into `dir`.
pub fn download_files(
    dir: &Path,
    base_url: &str,
    files: &[ModelFile<'_>],
    progress: &mut dyn FnMut(&DownloadEvent<'_>),
) -> Result<DownloadReport> {
    let started = Instant::now();
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_connect(Some(CONNECT_TIMEOUT))
        .timeout_recv_response(Some(RESPONSE_TIMEOUT))
        .build()
        .into();
    let mut report = DownloadReport::default();
    for file in files {
        if model::file_is_verified(dir, file) {
            progress(&DownloadEvent::Skipped { path: file.path });
            report.skipped.push(file.path.to_owned());
            continue;
        }
        let url = format!("{}/{}", base_url.trim_end_matches('/'), file.path);
        report.bytes += fetch_one(&agent, &url, dir, file, progress)?;
        report.fetched.push(file.path.to_owned());
    }
    report.elapsed = started.elapsed();
    Ok(report)
}

/// Removes the temporary file unless the download completed.
struct Partial {
    path: PathBuf,
    keep: bool,
}

impl Drop for Partial {
    fn drop(&mut self) {
        if !self.keep {
            // Best effort: the file may never have been created.
            let _ = fs::remove_file(&self.path);
        }
    }
}

fn fetch_one(
    agent: &ureq::Agent,
    url: &str,
    dir: &Path,
    file: &ModelFile<'_>,
    progress: &mut dyn FnMut(&DownloadEvent<'_>),
) -> Result<u64> {
    let target = file.local_path(dir);
    let folder = target
        .parent()
        .ok_or_else(|| CoreError::internal(format!("model file {} has no folder", file.path)))?;
    fs::create_dir_all(folder).map_err(|e| write_error(dir, file, &e))?;
    let name = target
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mut partial = Partial {
        path: folder.join(format!("{name}.download-{}", std::process::id())),
        keep: false,
    };

    progress(&DownloadEvent::Started {
        path: file.path,
        size: file.size,
    });
    let response = agent
        .get(url)
        .call()
        .map_err(|e| network_error(url, file, &e.to_string()))?;
    let mut body = response.into_body().into_reader();

    let mut out = File::create(&partial.path).map_err(|e| write_error(dir, file, &e))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; CHUNK];
    let mut done: u64 = 0;
    loop {
        let n = match body.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(network_error(url, file, &e.to_string())),
        };
        done += n as u64;
        if done > file.size {
            return Err(model::integrity(
                dir,
                format!(
                    "download of {} from {url} sent more than the pinned {} bytes; the partial file was deleted",
                    file.path, file.size
                ),
            ));
        }
        hasher.update(&buf[..n]);
        out.write_all(&buf[..n])
            .map_err(|e| write_error(dir, file, &e))?;
        progress(&DownloadEvent::Progress {
            path: file.path,
            done,
            size: file.size,
        });
    }
    out.sync_all().map_err(|e| write_error(dir, file, &e))?;
    drop(out);

    if done != file.size {
        return Err(model::integrity(
            dir,
            format!(
                "download of {} from {url} gave {done} bytes; the pinned size is {} bytes; the partial file was deleted",
                file.path, file.size
            ),
        ));
    }
    let actual = to_hex(hasher.finalize().as_slice());
    if actual != file.sha256 {
        return Err(model::integrity(
            dir,
            format!(
                "download of {} from {url} has SHA-256 {actual}; the pinned value is {}; the partial file was deleted",
                file.path, file.sha256
            ),
        ));
    }
    fs::rename(&partial.path, &target).map_err(|e| write_error(dir, file, &e))?;
    partial.keep = true;
    progress(&DownloadEvent::Finished { path: file.path });
    Ok(done)
}

fn network_error(url: &str, file: &ModelFile<'_>, detail: &str) -> CoreError {
    CoreError::new(
        ErrorCode::ModelMissing,
        format!("could not download model file {} from {url}: {detail}", file.path),
    )
    .with_next_action(format!(
        "Check the network connection and run `spatial-memory setup` again, or place the files by hand and set {}.",
        sm_core::paths::MODEL_DIR_ENV
    ))
}

fn write_error(dir: &Path, file: &ModelFile<'_>, e: &io::Error) -> CoreError {
    CoreError::new(
        ErrorCode::ModelMissing,
        format!(
            "could not write model file {} into {}: {e}",
            file.path,
            dir.display()
        ),
    )
    .with_next_action(format!(
        "Check the free space and permissions of {}, then run `spatial-memory setup` again.",
        dir.display()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sm_core::hash::sha256_hex;
    use std::collections::HashMap;
    use std::io::{BufRead, BufReader};
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    /// What the local server sends for one path.
    #[derive(Clone)]
    enum Reply {
        Body(Vec<u8>),
        /// Declares `declared` bytes and sends `body`, then closes.
        Short {
            body: Vec<u8>,
            declared: usize,
        },
        Status(u16),
    }

    /// An HTTP/1.1 server on 127.0.0.1 that answers from a table.
    struct Server {
        base: String,
        requests: Arc<AtomicUsize>,
        table: Arc<Mutex<HashMap<String, Reply>>>,
    }

    impl Server {
        fn start(table: HashMap<String, Reply>) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let base = format!("http://{}/repo", listener.local_addr().unwrap());
            let requests = Arc::new(AtomicUsize::new(0));
            let table = Arc::new(Mutex::new(table));
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
                    let reply = t.lock().unwrap().get(&path).cloned();
                    let (status, body, declared) = match reply {
                        Some(Reply::Body(b)) => (200, b.clone(), b.len()),
                        Some(Reply::Short { body, declared }) => (200, body, declared),
                        Some(Reply::Status(s)) => (s, Vec::new(), 0),
                        None => (404, Vec::new(), 0),
                    };
                    let head = format!(
                        "HTTP/1.1 {status} X\r\nContent-Length: {declared}\r\nConnection: close\r\n\r\n"
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

        fn requests(&self) -> usize {
            self.requests.load(Ordering::SeqCst)
        }

        fn set(&self, path: &str, reply: Reply) {
            self.table
                .lock()
                .unwrap()
                .insert(format!("/repo/{path}"), reply);
        }
    }

    struct Fixture {
        good: Vec<(String, Vec<u8>, String)>,
    }

    impl Fixture {
        fn new() -> Self {
            let graph = b"pretend onnx graph".to_vec();
            let weights: Vec<u8> = (0..(3 * CHUNK + 17)).map(|i| (i * 7 % 256) as u8).collect();
            let good = vec![
                ("onnx/model.onnx".to_owned(), graph),
                ("onnx/model.onnx_data".to_owned(), weights),
            ]
            .into_iter()
            .map(|(p, b)| {
                let h = sha256_hex(&b);
                (p, b, h)
            })
            .collect();
            Fixture { good }
        }

        fn files(&self) -> Vec<ModelFile<'_>> {
            self.good
                .iter()
                .map(|(p, b, h)| ModelFile {
                    path: p,
                    size: b.len() as u64,
                    sha256: h,
                })
                .collect()
        }

        fn serving_good(&self) -> Server {
            Server::start(
                self.good
                    .iter()
                    .map(|(p, b, _)| (format!("/repo/{p}"), Reply::Body(b.clone())))
                    .collect(),
            )
        }
    }

    fn leftovers(dir: &Path) -> Vec<String> {
        let mut found = Vec::new();
        let mut stack = vec![dir.to_path_buf()];
        while let Some(d) = stack.pop() {
            for e in fs::read_dir(&d).unwrap() {
                let p = e.unwrap().path();
                if p.is_dir() {
                    stack.push(p);
                } else if p.to_string_lossy().contains(".download-") {
                    found.push(p.display().to_string());
                }
            }
        }
        found
    }

    #[test]
    fn s5_download_writes_verified_files_and_skips_them_next_time() {
        let fx = Fixture::new();
        let server = fx.serving_good();
        let dir = tempfile::tempdir().unwrap();
        let mut events = Vec::new();
        let report = download_files(dir.path(), &server.base, &fx.files(), &mut |e| {
            events.push(format!("{e:?}"))
        })
        .unwrap();
        assert_eq!(report.fetched.len(), 2);
        assert!(report.skipped.is_empty());
        assert_eq!(
            report.bytes,
            fx.good.iter().map(|g| g.1.len() as u64).sum::<u64>()
        );
        model::verify_files(dir.path(), &fx.files()).unwrap();
        assert!(leftovers(dir.path()).is_empty());
        assert!(events.iter().any(|e| e.starts_with("Finished")));
        assert_eq!(server.requests(), 2);

        // Present and verified: nothing is fetched.
        let again = download_files(dir.path(), &server.base, &fx.files(), &mut |_| {}).unwrap();
        assert!(again.fetched.is_empty());
        assert_eq!(again.skipped.len(), 2);
        assert_eq!(server.requests(), 2, "no request for verified files");
    }

    #[test]
    fn at10_download_with_one_changed_byte_is_model_integrity_and_leaves_nothing() {
        let fx = Fixture::new();
        let server = fx.serving_good();
        let mut bad = fx.good[1].1.clone();
        bad[2 * CHUNK + 5] ^= 0x80;
        server.set("onnx/model.onnx_data", Reply::Body(bad));
        let dir = tempfile::tempdir().unwrap();
        let err = download_files(dir.path(), &server.base, &fx.files(), &mut |_| {}).unwrap_err();
        assert_eq!(err.code, ErrorCode::ModelIntegrity);
        assert!(
            err.message.contains("onnx/model.onnx_data"),
            "{}",
            err.message
        );
        assert!(err.message.contains("SHA-256"));
        assert!(leftovers(dir.path()).is_empty(), "partial file deleted");
        assert!(!dir.path().join("onnx").join("model.onnx_data").exists());
        // The good file before it was kept.
        assert!(model::file_is_verified(dir.path(), &fx.files()[0]));
    }

    #[test]
    fn at10_download_of_wrong_length_is_model_integrity() {
        let fx = Fixture::new();
        let dir = tempfile::tempdir().unwrap();

        // Longer than pinned: stops at the pinned size.
        let server = fx.serving_good();
        let mut long = fx.good[0].1.clone();
        long.extend_from_slice(b"extra");
        server.set("onnx/model.onnx", Reply::Body(long));
        let err = download_files(dir.path(), &server.base, &fx.files(), &mut |_| {}).unwrap_err();
        assert_eq!(err.code, ErrorCode::ModelIntegrity);
        assert!(err.message.contains("more than"), "{}", err.message);
        assert!(leftovers(dir.path()).is_empty());

        // Shorter, with an honest length.
        let server = fx.serving_good();
        server.set("onnx/model.onnx", Reply::Body(fx.good[0].1[..4].to_vec()));
        let err = download_files(dir.path(), &server.base, &fx.files(), &mut |_| {}).unwrap_err();
        assert_eq!(err.code, ErrorCode::ModelIntegrity);
        assert!(err.message.contains("gave 4 bytes"), "{}", err.message);
        assert!(leftovers(dir.path()).is_empty());
    }

    #[test]
    fn s5_download_cut_short_is_an_error_and_leaves_nothing() {
        let fx = Fixture::new();
        let server = fx.serving_good();
        let body = fx.good[1].1[..CHUNK].to_vec();
        server.set(
            "onnx/model.onnx_data",
            Reply::Short {
                body,
                declared: fx.good[1].1.len(),
            },
        );
        let dir = tempfile::tempdir().unwrap();
        let err = download_files(dir.path(), &server.base, &fx.files(), &mut |_| {}).unwrap_err();
        assert!(
            matches!(
                err.code,
                ErrorCode::ModelMissing | ErrorCode::ModelIntegrity
            ),
            "{err}"
        );
        assert!(leftovers(dir.path()).is_empty());
        assert!(!dir.path().join("onnx").join("model.onnx_data").exists());
    }

    #[test]
    fn s5_download_http_error_is_model_missing() {
        let fx = Fixture::new();
        let server = fx.serving_good();
        server.set("onnx/model.onnx", Reply::Status(404));
        let dir = tempfile::tempdir().unwrap();
        let err = download_files(dir.path(), &server.base, &fx.files(), &mut |_| {}).unwrap_err();
        assert_eq!(err.code, ErrorCode::ModelMissing);
        assert!(err.message.contains("onnx/model.onnx"), "{}", err.message);
        assert!(err.next_action.contains("setup"));
        assert!(leftovers(dir.path()).is_empty());
    }

    #[test]
    fn s5_download_replaces_a_corrupt_file() {
        let fx = Fixture::new();
        let server = fx.serving_good();
        let dir = tempfile::tempdir().unwrap();
        let target = fx.files()[0].local_path(dir.path());
        fs::create_dir_all(target.parent().unwrap()).unwrap();
        fs::write(&target, b"pretend onnx grapX").unwrap();
        let report = download_files(dir.path(), &server.base, &fx.files(), &mut |_| {}).unwrap();
        assert_eq!(report.fetched.len(), 2);
        model::verify_files(dir.path(), &fx.files()).unwrap();
    }

    #[test]
    fn s5_pinned_url_names_repository_and_revision() {
        assert_eq!(
            pinned_base_url(),
            "https://huggingface.co/onnx-community/embeddinggemma-2-ONNX/resolve/daa72c51243991dfcaf9f9137d2c573d8f7790c0"
        );
    }
}
