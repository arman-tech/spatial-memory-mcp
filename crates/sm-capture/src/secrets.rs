//! The secret scanner (SPEC section 16, Pipeline in the daemon, step 3;
//! AT-9): every tier 1, 2 and 3 pattern of version 1, with its false
//! positive filter, and version 3's additions ([`V3_VALUE_PATTERNS`],
//! B-86), behind [`SecretScanner`].
//!
//! Version 1 applied the patterns one after another, each to the text the
//! previous one had already rewritten with `[REDACTED_<KIND>]`
//! placeholders, so a later pattern saw the earlier placeholders (an
//! `Authorization: Bearer <JWT>` header is left as a redacted JSON Web
//! Token (JWT) because the header pattern no longer matches once the token
//! is a placeholder). This scanner keeps that working text exactly as
//! Python had it, so every pattern matches where Python's did, and maps
//! each match back to a byte range of the original text. Each range lies
//! on `char` boundaries; [`sm_core::traits::redact_matches`] then writes
//! `[REDACTED:<kind>]`.
//!
//! Where Python replaced only part of a match, the match is only that part:
//! the password of `scheme://user:password@host`, and the value of a tier 3
//! `key = value`. Python also dropped a tier 3 value's closing quote; here
//! the quote stays.
//!
//! Text that already holds `[REDACTED:<kind>]` markers is treated as
//! already redacted: each marker stands in the working text as Python's
//! placeholder would, and a match made only of markers is not reported.
//! That makes redaction stable: [`SecretScanner::scan`] finds nothing in
//! the output of [`SecretScanner::redact`], and redacting twice changes
//! nothing.

use std::sync::LazyLock;

use regex::Regex;
use sm_core::hash::python_strip;
use sm_core::traits::{Redaction, SecretMatch, SecretScanner, redact_matches};

/// A pattern that matches a credential by its shape (tiers 1 and 2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyPattern {
    /// Version 1's name for the pattern (`redaction_patterns.json`).
    pub name: &'static str,
    /// 1 for prefix patterns, 2 for structural ones.
    pub tier: u8,
    /// The version 3 kind written into `[REDACTED:<kind>]`.
    pub kind: &'static str,
    /// Whether a match makes the whole capture skipped (a private key).
    pub private_key: bool,
    /// Whether only the text between the two capture groups is the secret
    /// (`scheme://user:` and `@` stay).
    pub url_password: bool,
    /// The `regex` crate expression.
    pub regex: &'static str,
}

/// A tier 3 pattern: a keyword, `=` or `:`, then a value in group 1, which
/// is a secret unless the false positive filter says otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValuePattern {
    /// Version 1's name for the pattern.
    pub name: &'static str,
    /// The version 3 kind.
    pub kind: &'static str,
    /// The `regex` crate expression, `(?i)` included.
    pub regex: &'static str,
}

/// Version 1's tier 1 and tier 2 patterns, in the order they are applied.
pub const KEY_PATTERNS: [KeyPattern; 19] = [
    KeyPattern {
        name: "aws_access_key_id",
        tier: 1,
        kind: "aws_key",
        private_key: false,
        url_password: false,
        regex: r#"AKIA[0-9A-Z]{16}"#,
    },
    KeyPattern {
        name: "github_token",
        tier: 1,
        kind: "github_token",
        private_key: false,
        url_password: false,
        regex: r#"(?:ghp|gho|ghs|ghr)_[A-Za-z0-9_]{36,255}"#,
    },
    KeyPattern {
        name: "github_pat",
        tier: 1,
        kind: "github_token",
        private_key: false,
        url_password: false,
        regex: r#"github_pat_[A-Za-z0-9_]{22,255}"#,
    },
    KeyPattern {
        name: "stripe_key",
        tier: 1,
        kind: "stripe_key",
        private_key: false,
        url_password: false,
        regex: r#"(?:sk|pk)_(?:live|test)_[A-Za-z0-9]{20,99}"#,
    },
    KeyPattern {
        name: "slack_token",
        tier: 1,
        kind: "slack_token",
        private_key: false,
        url_password: false,
        regex: r#"xox[bp]-[0-9]{10,13}-[0-9A-Za-z-]+"#,
    },
    KeyPattern {
        name: "slack_webhook",
        tier: 1,
        kind: "slack_webhook",
        private_key: false,
        url_password: false,
        regex: r#"https://hooks\.slack\.com/services/T[A-Z0-9]+/B[A-Z0-9]+/[A-Za-z0-9]+"#,
    },
    KeyPattern {
        name: "openai_project_key",
        tier: 1,
        kind: "openai_key",
        private_key: false,
        url_password: false,
        regex: r#"sk-proj-[A-Za-z0-9_-]{20,255}"#,
    },
    KeyPattern {
        name: "openai_legacy_key",
        tier: 1,
        kind: "openai_key",
        private_key: false,
        url_password: false,
        regex: r#"sk-[A-Za-z0-9]{20,48}T3BlbkFJ[A-Za-z0-9]*"#,
    },
    KeyPattern {
        name: "google_api_key",
        tier: 1,
        kind: "google_key",
        private_key: false,
        url_password: false,
        regex: r#"AIza[0-9A-Za-z_-]{35}"#,
    },
    KeyPattern {
        name: "sendgrid_key",
        tier: 1,
        kind: "sendgrid_key",
        private_key: false,
        url_password: false,
        regex: r#"SG\.[A-Za-z0-9_-]{22}\.[A-Za-z0-9_-]{43}"#,
    },
    KeyPattern {
        name: "pypi_token",
        tier: 1,
        kind: "pypi_token",
        private_key: false,
        url_password: false,
        regex: r#"pypi-[A-Za-z0-9_-]{16,255}"#,
    },
    KeyPattern {
        name: "npm_token",
        tier: 1,
        kind: "npm_token",
        private_key: false,
        url_password: false,
        regex: r#"npm_[A-Za-z0-9]{36,64}"#,
    },
    KeyPattern {
        name: "anthropic_key",
        tier: 1,
        kind: "anthropic_key",
        private_key: false,
        url_password: false,
        regex: r#"sk-ant-[A-Za-z0-9_-]{20,255}"#,
    },
    KeyPattern {
        name: "huggingface_token",
        tier: 1,
        kind: "hf_token",
        private_key: false,
        url_password: false,
        regex: r#"hf_[A-Za-z0-9]{34,}"#,
    },
    KeyPattern {
        name: "private_key_pem",
        tier: 2,
        kind: "private_key",
        private_key: true,
        url_password: false,
        regex: r#"(?s)-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----[\s\S]*?-----END (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"#,
    },
    KeyPattern {
        name: "pgp_private_key_block",
        tier: 2,
        kind: "private_key",
        private_key: true,
        url_password: false,
        regex: r#"(?s)-----BEGIN PGP PRIVATE KEY BLOCK-----[\s\S]*?-----END PGP PRIVATE KEY BLOCK-----"#,
    },
    KeyPattern {
        name: "jwt",
        tier: 2,
        kind: "jwt",
        private_key: false,
        url_password: false,
        regex: r#"eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"#,
    },
    KeyPattern {
        name: "auth_header",
        tier: 2,
        kind: "auth_header",
        private_key: false,
        url_password: false,
        regex: r#"(?:Authorization|authorization)\s*[:=]\s*(?:Basic|Bearer)\s+[A-Za-z0-9+/=_-]{20,}"#,
    },
    KeyPattern {
        name: "password_in_url",
        tier: 2,
        kind: "password_url",
        private_key: false,
        url_password: true,
        regex: r#"(://[^:\s]+:)[^@\s]{3,}(@)"#,
    },
];

/// Version 1's tier 3 patterns, applied after every key pattern.
pub const VALUE_PATTERNS: [ValuePattern; 3] = [
    ValuePattern {
        name: "password",
        kind: "password",
        regex: r#"(?i)(?:password|passwd|pwd)\s*[=:]\s*['"]?([^\s'";\n]{3,})['"]?"#,
    },
    ValuePattern {
        name: "secret",
        kind: "secret",
        regex: r#"(?i)(?:secret|api_key|apikey|api_secret|access_token)\s*[=:]\s*['"]?([^\s'";\n]{3,})['"]?"#,
    },
    ValuePattern {
        name: "token",
        kind: "token",
        regex: r#"(?i)(?:token|auth_token)\s*[=:]\s*['"]?([^\s'";\n]{8,})['"]?"#,
    },
];

/// Version 3's additions to the tier 3 patterns, applied after version 1's,
/// with the same value group and false positive filter (B-86).
///
/// Version 1's `secret` pattern needs `=` or `:` straight after the word,
/// so it never saw the secret half of an AWS key pair
/// (`AWS_SECRET_ACCESS_KEY=...`), which SPEC 9 counts among the cloud access
/// keys to redact. The keyword may be followed by a closing quote, as in
/// JSON. Prose such as "the secret access key" has spaces and is not
/// matched.
pub const V3_VALUE_PATTERNS: [ValuePattern; 1] = [ValuePattern {
    name: "aws_secret_access_key",
    kind: "aws_key",
    regex: r#"(?i)(?:secret_?access_?key|aws_?secret_?key)['"]?\s*[=:]\s*['"]?([^\s'";\n]{3,})['"]?"#,
}];

/// Version 1's false positive filter for tier 3 values: template
/// expressions, angle-bracket and asterisk placeholders, values already
/// redacted, and stop words such as `changeme` or `example`. `$` is written
/// `\n?$` because Python's `$` also matches before a final newline.
pub const FALSE_POSITIVE_REGEX: &str = r#"(?i)^\$\{|^\{\{|^\$\(|^<[^>]+>\n?$|^\*{2,}\n?$|^\[REDACTED_|^(changeme|example|replace|your[_-]|my[_-]|test|dummy|fake|xxx|placeholder)"#;

static KEY_COMPILED: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    KEY_PATTERNS
        .iter()
        .map(|p| Regex::new(p.regex).expect("key patterns are valid regular expressions"))
        .collect()
});

/// Version 1's value patterns, then version 3's, in the order applied.
fn value_patterns() -> impl Iterator<Item = &'static ValuePattern> {
    VALUE_PATTERNS.iter().chain(V3_VALUE_PATTERNS.iter())
}

static VALUE_COMPILED: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    value_patterns()
        .map(|p| Regex::new(p.regex).expect("value patterns are valid regular expressions"))
        .collect()
});

static FALSE_POSITIVE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(FALSE_POSITIVE_REGEX).expect("the false positive filter is a valid expression")
});

/// A version 3 redaction marker, `[REDACTED:<kind>]`.
static V3_MARKER: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\[REDACTED:([a-z_]+)\]").expect("the marker expression is valid")
});

/// Whether a tier 3 value is a placeholder rather than a secret: blank or
/// shorter than three characters once surrounding whitespace and quotes are
/// removed, or caught by [`FALSE_POSITIVE_REGEX`].
pub fn is_false_positive(value: &str) -> bool {
    let stripped = python_strip(value).trim_matches(['\'', '"']);
    if stripped.is_empty() || stripped.chars().count() < 3 {
        return true;
    }
    FALSE_POSITIVE.is_match(stripped)
}

/// Version 1's placeholder for a version 3 kind: `[REDACTED_<KIND>]`.
pub fn python_placeholder(kind: &str) -> String {
    format!("[REDACTED_{}]", kind.to_ascii_uppercase())
}

/// Everything the scanner found in one text, with version 1's skip rule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SecretReport {
    /// The redacted text and the kinds replaced.
    pub redaction: Redaction,
    /// Every match, in the order the patterns found them.
    pub matches: Vec<SecretMatch>,
    /// Whether a private key was found.
    pub has_private_key: bool,
    /// Version 1's `should_skip`: a private key was found, or more than half
    /// of the original characters were redacted.
    pub should_skip: bool,
}

impl SecretReport {
    /// The number of matches; version 1's `redaction_count`.
    pub fn count(&self) -> usize {
        self.matches.len()
    }
}

/// The scanner: version 1's patterns over a working text that keeps
/// Python's placeholders.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RegexSecretScanner;

impl RegexSecretScanner {
    /// A scanner. The patterns are compiled once per process, on first use.
    pub fn new() -> Self {
        RegexSecretScanner
    }

    /// Scans and redacts `text` and applies version 1's skip rule.
    pub fn report(&self, text: &str) -> SecretReport {
        let scan = scan_text(text);
        let redaction = redact_matches(text, scan.matches.clone());
        let should_skip = if scan.has_private_key {
            true
        } else if scan.matches.is_empty() {
            false
        } else {
            let original = text.chars().count();
            let kept = V3_MARKER.replace_all(&redaction.text, "").chars().count();
            original > 0 && 1.0 - (kept as f64 / original as f64) > 0.5
        };
        SecretReport {
            redaction,
            matches: scan.matches,
            has_private_key: scan.has_private_key,
            should_skip,
        }
    }
}

impl SecretScanner for RegexSecretScanner {
    fn scan(&self, text: &str) -> Vec<SecretMatch> {
        scan_text(text).matches
    }
}

struct Scan {
    matches: Vec<SecretMatch>,
    has_private_key: bool,
}

fn scan_text(text: &str) -> Scan {
    let mut scan = Scan {
        matches: Vec::new(),
        has_private_key: false,
    };
    if text.is_empty() {
        return scan;
    }
    let mut working = Working::new(text);
    for (pattern, regex) in KEY_PATTERNS.iter().zip(KEY_COMPILED.iter()) {
        let mut hits = Vec::new();
        if pattern.url_password {
            for c in regex.captures_iter(&working.rendered) {
                if let (Some(user), Some(at)) = (c.get(1), c.get(2)) {
                    let span = (user.end(), at.start());
                    hits.push(Hit {
                        mask: span,
                        secret: span,
                    });
                }
            }
        } else {
            for m in regex.find_iter(&working.rendered) {
                let span = (m.start(), m.end());
                hits.push(Hit {
                    mask: span,
                    secret: span,
                });
            }
        }
        let found = working.apply(&hits, pattern.kind, &mut scan.matches);
        if found && pattern.private_key {
            scan.has_private_key = true;
        }
    }
    for (pattern, regex) in value_patterns().zip(VALUE_COMPILED.iter()) {
        let mut hits = Vec::new();
        for c in regex.captures_iter(&working.rendered) {
            let (Some(whole), Some(value)) = (c.get(0), c.get(1)) else {
                continue;
            };
            if is_false_positive(value.as_str()) {
                continue;
            }
            // Python replaced from the value to the end of the match, the
            // closing quote included; the working text does the same so
            // later patterns see what Python saw. The reported secret is the
            // value alone, so the quote stays in the redacted text.
            hits.push(Hit {
                mask: (value.start(), whole.end()),
                secret: (value.start(), value.end()),
            });
        }
        working.apply(&hits, pattern.kind, &mut scan.matches);
    }
    scan
}

/// One match in the working text: what Python replaced (`mask`) and what is
/// reported as the secret (`secret`, inside `mask`).
#[derive(Debug, Clone, Copy)]
struct Hit {
    mask: (usize, usize),
    secret: (usize, usize),
}

/// A run of the working text: original text, or a placeholder standing for
/// a range of the original.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Piece {
    Text {
        start: usize,
        end: usize,
    },
    Mask {
        start: usize,
        end: usize,
        label: String,
    },
}

impl Piece {
    fn start(&self) -> usize {
        match self {
            Piece::Text { start, .. } | Piece::Mask { start, .. } => *start,
        }
    }

    fn end(&self) -> usize {
        match self {
            Piece::Text { end, .. } | Piece::Mask { end, .. } => *end,
        }
    }
}

/// The working text as Python had it, with the map back to the original.
/// The pieces cover the original text in order, without gaps, and none is
/// empty.
struct Working<'a> {
    text: &'a str,
    pieces: Vec<Piece>,
    rendered: String,
    /// Where each piece starts in `rendered`.
    offsets: Vec<usize>,
}

impl<'a> Working<'a> {
    fn new(text: &'a str) -> Self {
        let mut pieces = Vec::new();
        let mut pos = 0;
        for c in V3_MARKER.captures_iter(text) {
            let (Some(whole), Some(kind)) = (c.get(0), c.get(1)) else {
                continue;
            };
            if whole.start() > pos {
                pieces.push(Piece::Text {
                    start: pos,
                    end: whole.start(),
                });
            }
            pieces.push(Piece::Mask {
                start: whole.start(),
                end: whole.end(),
                label: python_placeholder(kind.as_str()),
            });
            pos = whole.end();
        }
        if pos < text.len() {
            pieces.push(Piece::Text {
                start: pos,
                end: text.len(),
            });
        }
        let mut working = Working {
            text,
            pieces,
            rendered: String::new(),
            offsets: Vec::new(),
        };
        working.render();
        working
    }

    fn render(&mut self) {
        self.rendered.clear();
        self.offsets.clear();
        for piece in &self.pieces {
            self.offsets.push(self.rendered.len());
            match piece {
                Piece::Text { start, end } => self.rendered.push_str(&self.text[*start..*end]),
                Piece::Mask { label, .. } => self.rendered.push_str(label),
            }
        }
    }

    /// The piece holding byte `w` of the working text.
    fn piece_at(&self, w: usize) -> usize {
        self.offsets.partition_point(|&o| o <= w).saturating_sub(1)
    }

    /// The original range for a working range, widened to whole
    /// placeholders. `None` when the range holds no original text, that is
    /// when it lies wholly in placeholders.
    fn to_original(&self, (ws, we): (usize, usize)) -> Option<(usize, usize)> {
        if ws >= we || we > self.rendered.len() {
            return None;
        }
        let i = self.piece_at(ws);
        let j = self.piece_at(we - 1);
        let start = match &self.pieces[i] {
            Piece::Text { start, .. } => start + (ws - self.offsets[i]),
            Piece::Mask { start, .. } => *start,
        };
        let end = match &self.pieces[j] {
            Piece::Text { start, .. } => start + (we - self.offsets[j]),
            Piece::Mask { end, .. } => *end,
        };
        let has_text = self.pieces[i..=j]
            .iter()
            .any(|p| matches!(p, Piece::Text { .. }));
        has_text.then_some((start, end))
    }

    /// Replaces each hit with `kind`'s placeholder, as Python's `sub` did,
    /// and appends the secrets to `out`. Returns whether anything was
    /// replaced.
    fn apply(&mut self, hits: &[Hit], kind: &str, out: &mut Vec<SecretMatch>) -> bool {
        let mut spans: Vec<(usize, usize)> = Vec::new();
        for hit in hits {
            let (Some(mask), Some(secret)) =
                (self.to_original(hit.mask), self.to_original(hit.secret))
            else {
                continue;
            };
            match spans.last_mut() {
                Some(last) if mask.0 < last.1 => last.1 = last.1.max(mask.1),
                _ => spans.push(mask),
            }
            out.push(SecretMatch {
                kind: kind.to_owned(),
                start: secret.0,
                end: secret.1,
            });
        }
        if spans.is_empty() {
            return false;
        }
        let label = python_placeholder(kind);
        let mut old = std::mem::take(&mut self.pieces);
        let mut new = Vec::with_capacity(old.len() + 2 * spans.len());
        let mut k = 0;
        for (start, end) in spans {
            while k < old.len() && old[k].end() <= start {
                new.push(old[k].clone());
                k += 1;
            }
            if k < old.len()
                && old[k].start() < start
                && let Piece::Text { start: s, end: e } = old[k]
            {
                new.push(Piece::Text {
                    start: s,
                    end: start,
                });
                old[k] = Piece::Text { start, end: e };
            }
            while k < old.len() && old[k].end() <= end {
                k += 1;
            }
            if k < old.len()
                && old[k].start() < end
                && let Piece::Text { end: e, .. } = old[k]
            {
                old[k] = Piece::Text { start: end, end: e };
            }
            new.push(Piece::Mask {
                start,
                end,
                label: label.clone(),
            });
        }
        new.extend(old.drain(k..));
        self.pieces = new;
        self.render();
        true
    }
}
