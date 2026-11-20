//! The requirement trace (SPEC 14, Requirement identifiers; B-40). Every
//! test name `docs/v3/TRACE.md` cites in backticks is a function under
//! `crates/`, found as the file's header says: a name that follows `fn `
//! and starts with `atN_`, `dN_`, `sN_` or `spikeN_`. A cited name that ends
//! in `*` stands for the tests that start with it, so at least one must.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn is_name_byte(c: u8) -> bool {
    c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'_'
}

/// The length of the test name at the start of `s`, if one starts there:
/// `(at|d|s|spike)[0-9]+_[a-z0-9_]+`, as far as it runs.
fn name_len(s: &[u8]) -> Option<usize> {
    ["spike", "at", "d", "s"].iter().find_map(|prefix| {
        let rest = s.strip_prefix(prefix.as_bytes())?;
        let digits = rest.iter().take_while(|c| c.is_ascii_digit()).count();
        if digits == 0 || rest.get(digits) != Some(&b'_') {
            return None;
        }
        let tail = rest[digits + 1..]
            .iter()
            .take_while(|&&c| is_name_byte(c))
            .count();
        (tail > 0).then_some(prefix.len() + digits + 1 + tail)
    })
}

/// Every name that follows `fn ` in `text`.
fn defined_names(text: &str, names: &mut BTreeSet<String>) {
    let mut rest = text.as_bytes();
    while let Some(at) = rest.windows(3).position(|w| w == b"fn ") {
        rest = &rest[at + 3..];
        if let Some(n) = name_len(rest) {
            names.insert(String::from_utf8_lossy(&rest[..n]).into_owned());
        }
    }
}

fn rust_files(dir: &Path, files: &mut Vec<PathBuf>) {
    let entries =
        std::fs::read_dir(dir).unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()));
    for entry in entries {
        let entry = entry.unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()));
        let path = entry.path();
        let kind = entry
            .file_type()
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        if kind.is_dir() {
            rust_files(&path, files);
        } else if kind.is_file() && path.extension().is_some_and(|x| x == "rs") {
            files.push(path);
        }
    }
}

/// The test names cited inside backticks in `trace`, a line at a time; a
/// name followed by `*` is kept with its `*`, as a prefix.
fn cited_names(trace: &str) -> BTreeSet<String> {
    let mut cited = BTreeSet::new();
    for line in trace.lines() {
        for span in line.split('`').skip(1).step_by(2) {
            let b = span.as_bytes();
            let mut i = 0;
            while i < b.len() {
                let starts_a_word =
                    i == 0 || !(b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_');
                match name_len(&b[i..]).filter(|_| starts_a_word) {
                    Some(n) => {
                        let end = n + usize::from(b.get(i + n) == Some(&b'*'));
                        cited.insert(String::from_utf8_lossy(&b[i..i + end]).into_owned());
                        i += end;
                    }
                    None => i += 1,
                }
            }
        }
    }
    cited
}

#[test]
fn s14_trace_cites_only_test_names_that_exist() {
    let root = repo();
    let trace_path = root.join("docs/v3/TRACE.md");
    let trace = std::fs::read_to_string(&trace_path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", trace_path.display()));
    let cited = cited_names(&trace);
    assert!(
        !cited.is_empty(),
        "docs/v3/TRACE.md cites no test name in backticks; the check would prove nothing"
    );

    let mut files = Vec::new();
    rust_files(&root.join("crates"), &mut files);
    let mut defined = BTreeSet::new();
    for file in &files {
        let text = std::fs::read_to_string(file)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", file.display()));
        defined_names(&text, &mut defined);
    }

    let missing: Vec<&str> = cited
        .iter()
        .map(String::as_str)
        .filter(|name| match name.strip_suffix('*') {
            Some(prefix) => !defined.iter().any(|d| d.starts_with(prefix)),
            None => !defined.contains(*name),
        })
        .collect();
    assert!(
        missing.is_empty(),
        "docs/v3/TRACE.md cites {} test name(s) that no `fn` under crates/ defines \
         ({} cited, {} defined in {} files):\n  {}\n\
         Cite each test by its current name, or write the test (B-40).",
        missing.len(),
        cited.len(),
        defined.len(),
        files.len(),
        missing.join("\n  ")
    );
}
