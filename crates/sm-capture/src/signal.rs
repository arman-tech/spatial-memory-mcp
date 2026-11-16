//! Signal classification: version 1's 22 patterns, the tiers, the stored
//! importance and the memory type (SPEC section 16, Pipeline in the daemon,
//! steps 2 and 5; Type and provenance).
//!
//! The patterns are `fixtures/capture/signal_patterns.json`, in its order.
//! The 15 patterns that end in Python's `(?:\.|$)` end in `(?:\.|\n?$)` here,
//! because Python's `$` without MULTILINE also matches before a final
//! newline and the `regex` crate's does not. Each pattern carries `(?i)`.
//! A pattern runs only when one of its keywords is a substring of the
//! lower-cased text, as in version 1.

use std::sync::LazyLock;

use regex::Regex;
use sm_core::hash::python_strip;
use sm_core::types::MemoryType;

/// One signal pattern, as listed in `signal_patterns.json`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SignalPattern {
    /// The pattern's identifier in the fixture (`decision`, `solution_fix`, ...).
    pub id: &'static str,
    /// The family reported in `patterns_matched` and used for the type.
    pub family: &'static str,
    /// The score a match contributes.
    pub confidence: f64,
    /// Lower-case substrings; the regex runs only when one is present.
    pub keywords: &'static [&'static str],
    /// The `regex` crate expression, `(?i)` included.
    pub regex: &'static str,
}

/// Version 1's 22 signal patterns, in evaluation order.
pub const SIGNAL_PATTERNS: [SignalPattern; 22] = [
    SignalPattern {
        id: "decision",
        family: "decision",
        confidence: 0.8,
        keywords: &["decided", "chose", "going with", "selected", "will use"],
        regex: r#"(?i)\b(?:decided|chose|going with|selected|will use)\s+.+?(?:\.|\n?$)"#,
    },
    SignalPattern {
        id: "important",
        family: "important",
        confidence: 0.9,
        keywords: &["important", "note", "remember", "key point"],
        regex: r#"(?i)\b(?:important|note|remember|key point)[:\s]+.+?(?:\.|\n?$)"#,
    },
    SignalPattern {
        id: "solution_fix",
        family: "solution",
        confidence: 0.85,
        keywords: &["the fix", "the solution", "the approach"],
        regex: r#"(?i)the (?:fix|solution|approach) (?:is|was)\s+.+?(?:\.|\n?$)"#,
    },
    SignalPattern {
        id: "error_diagnosis",
        family: "error",
        confidence: 0.8,
        keywords: &["the issue", "the problem", "the bug"],
        regex: r#"(?i)the (?:issue|problem|bug) was\s+.+?(?:\.|\n?$)"#,
    },
    SignalPattern {
        id: "explicit_save",
        family: "explicit",
        confidence: 0.95,
        keywords: &["save", "remember", "note", "store"],
        regex: r#"(?i)\b(?:save|remember|note|store)(?:\s+that)?\s+.+?(?:\.|\n?$)"#,
    },
    SignalPattern {
        id: "pattern_learning",
        family: "pattern",
        confidence: 0.85,
        keywords: &["the trick is", "the key is", "pattern:"],
        regex: r#"(?i)(?:the trick is|the key is|pattern:)\s+.+?(?:\.|\n?$)"#,
    },
    SignalPattern {
        id: "resolved_by",
        family: "solution",
        confidence: 0.8,
        keywords: &["resolved by"],
        regex: r#"(?i)resolved by\s+.+?(?:\.|\n?$)"#,
    },
    SignalPattern {
        id: "broke_failed_because",
        family: "error",
        confidence: 0.8,
        keywords: &["broke because", "failed because"],
        regex: r#"(?i)(?:broke|failed) because\s+.+?(?:\.|\n?$)"#,
    },
    SignalPattern {
        id: "error_due_to",
        family: "error",
        confidence: 0.8,
        keywords: &["the error was due to"],
        regex: r#"(?i)the error was due to\s+.+?(?:\.|\n?$)"#,
    },
    SignalPattern {
        id: "architecture",
        family: "decision",
        confidence: 0.8,
        keywords: &["the architecture will be"],
        regex: r#"(?i)the architecture will be\s+.+?(?:\.|\n?$)"#,
    },
    SignalPattern {
        id: "definition",
        family: "definition",
        confidence: 0.6,
        keywords: &[" is a ", " is an ", " is the ", " means ", " refers to "],
        regex: r#"(?i)\b\w+\s+(?:is\s+(?:a|an|the)\s+|means\s+|refers\s+to\s+)\S.+?(?:\.|\n?$)"#,
    },
    SignalPattern {
        id: "convention",
        family: "convention",
        confidence: 0.7,
        keywords: &["convention here is"],
        regex: r#"(?i)convention here is\s+.+?(?:\.|\n?$)"#,
    },
    SignalPattern {
        id: "watch_out_for",
        family: "workaround",
        confidence: 0.7,
        keywords: &["watch out for"],
        regex: r#"(?i)watch out for\s+.+?(?:\.|\n?$)"#,
    },
    SignalPattern {
        id: "workaround_is",
        family: "workaround",
        confidence: 0.75,
        keywords: &["the workaround is"],
        regex: r#"(?i)the workaround is\s+.+?(?:\.|\n?$)"#,
    },
    SignalPattern {
        id: "configuration",
        family: "configuration",
        confidence: 0.65,
        keywords: &["you need to set"],
        regex: r#"(?i)you need to set\s+.+?(?:\.|\n?$)"#,
    },
    SignalPattern {
        id: "testing",
        family: "testing",
        confidence: 0.7,
        keywords: &[
            "test failed",
            "test passed",
            "tests pass",
            "test fail",
            "test coverage",
            "assertion",
        ],
        regex: r#"(?i)(?:tests?\s+(?:failed|passed|pass|fail)|test coverage|assertion\s+(?:error|failed))"#,
    },
    SignalPattern {
        id: "performance",
        family: "performance",
        confidence: 0.7,
        keywords: &[
            "bottleneck",
            "latency",
            "throughput",
            "optimization",
            "faster when",
            "slower when",
        ],
        regex: r#"(?i)(?:bottleneck|latency|throughput|optimization|(?:faster|slower)\s+when)"#,
    },
    SignalPattern {
        id: "environment",
        family: "environment",
        confidence: 0.6,
        keywords: &[
            "requires version",
            "upgraded to",
            "compatible with",
            "python version",
            "node version",
        ],
        regex: r#"(?i)(?:requires?\s+version|upgraded?\s+to|compatible\s+with|(?:python|node|npm|java)\s+version)"#,
    },
    SignalPattern {
        id: "dependency",
        family: "dependency",
        confidence: 0.6,
        keywords: &[
            "dependency",
            "dependencies",
            "depends on",
            "package version",
            "library version",
            "pip install",
            "npm install",
        ],
        regex: r#"(?i)(?:dependenc(?:y|ies)|depends\s+on|(?:package|library)\s+version|(?:pip|npm|yarn)\s+install)"#,
    },
    SignalPattern {
        id: "api",
        family: "api",
        confidence: 0.6,
        keywords: &[
            "endpoint",
            "api returns",
            "api return",
            "request body",
            "response body",
            "webhook",
            "rest api",
            "graphql",
        ],
        regex: r#"(?i)(?:endpoint|api\s+returns?|(?:request|response)\s+body|webhook|rest\s+api|graphql)"#,
    },
    SignalPattern {
        id: "procedure",
        family: "procedure",
        confidence: 0.6,
        keywords: &["steps to", "step to", "to reproduce", "step 1", "first,"],
        regex: r#"(?i)(?:steps?\s+to\s+|to\s+reproduce|step\s+\d|first,\s+.+?then)"#,
    },
    SignalPattern {
        id: "workflow",
        family: "workflow",
        confidence: 0.6,
        keywords: &[
            "process for",
            "pipeline for",
            "workflow",
            "run before",
            "run after",
        ],
        regex: r#"(?i)(?:(?:process|pipeline)\s+for|workflow\s+(?:is|for)|run\s+(?:before|after))"#,
    },
];

/// The lowest score of a strong signal.
pub const STRONG_THRESHOLD: f64 = 0.8;
/// The lowest score of a medium signal. Anything lower is dropped.
pub const MEDIUM_THRESHOLD: f64 = 0.5;

static COMPILED: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    SIGNAL_PATTERNS
        .iter()
        .map(|p| Regex::new(p.regex).expect("signal patterns are valid regular expressions"))
        .collect()
});

/// How strong a signal is: version 1's tiers 1, 2 and 3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Tier {
    /// Tier 1: a score of 0.8 or more.
    Strong,
    /// Tier 2: a score from 0.5 to below 0.8.
    Medium,
    /// Tier 3: anything lower, which is dropped.
    Weak,
}

impl Tier {
    /// The tier for a score. A score that is not a number is weak.
    pub fn from_score(score: f64) -> Tier {
        if score >= STRONG_THRESHOLD {
            Tier::Strong
        } else if score >= MEDIUM_THRESHOLD {
            Tier::Medium
        } else {
            Tier::Weak
        }
    }

    /// Version 1's tier number: 1, 2 or 3.
    pub fn number(self) -> u8 {
        match self {
            Tier::Strong => 1,
            Tier::Medium => 2,
            Tier::Weak => 3,
        }
    }

    /// The tier for version 1's number; `None` for anything but 1, 2 or 3.
    pub fn from_number(n: i64) -> Option<Tier> {
        match n {
            1 => Some(Tier::Strong),
            2 => Some(Tier::Medium),
            3 => Some(Tier::Weak),
            _ => None,
        }
    }
}

/// The result of classifying a text.
#[derive(Debug, Clone, PartialEq)]
pub struct Signal {
    /// The tier of `score`.
    pub tier: Tier,
    /// The highest confidence among the matching patterns, or 0.
    pub score: f64,
    /// The families of the matching patterns, first match first, without
    /// repeats.
    pub families: Vec<String>,
}

impl Signal {
    /// No signal: tier 3, score 0, no families.
    pub fn none() -> Signal {
        Signal {
            tier: Tier::Weak,
            score: 0.0,
            families: Vec::new(),
        }
    }
}

/// Classifies a text with the 22 patterns. Empty or blank text has no
/// signal. The keyword check uses the lower-cased text; the regex runs on
/// the text as given.
pub fn classify(text: &str) -> Signal {
    if python_strip(text).is_empty() {
        return Signal::none();
    }
    let lower = text.to_lowercase();
    let mut score: f64 = 0.0;
    let mut families: Vec<String> = Vec::new();
    for (pattern, regex) in SIGNAL_PATTERNS.iter().zip(COMPILED.iter()) {
        if !pattern.keywords.iter().any(|k| lower.contains(k)) {
            continue;
        }
        if regex.is_match(text) {
            if !families.iter().any(|f| f == pattern.family) {
                families.push(pattern.family.to_owned());
            }
            if pattern.confidence > score {
                score = pattern.confidence;
            }
        }
    }
    Signal {
        tier: Tier::from_score(score),
        score,
        families,
    }
}

/// The identifiers of the patterns whose regex matches `text`, ignoring the
/// keyword check. Used to compare the rewritten expressions with Python's.
pub fn matching_pattern_ids(text: &str) -> Vec<&'static str> {
    SIGNAL_PATTERNS
        .iter()
        .zip(COMPILED.iter())
        .filter(|(_, r)| r.is_match(text))
        .map(|(p, _)| p.id)
        .collect()
}

/// The stored importance (SPEC section 16, step 5): the higher of 0.7 and
/// the score for a strong signal, the higher of 0.4 and 0.8 times the score
/// for a medium one, clamped to 0 to 1. `None` for a weak signal or a score
/// that is not finite, which gives no candidate.
pub fn importance(tier: Tier, score: f64) -> Option<f64> {
    if !score.is_finite() {
        return None;
    }
    let raw = match tier {
        Tier::Strong => score.max(0.7),
        Tier::Medium => (0.8 * score).max(0.4),
        Tier::Weak => return None,
    };
    Some(raw.clamp(0.0, 1.0))
}

/// Version 1's ordered family-to-namespace map (`derive_namespace`).
pub const NAMESPACE_MAP: [(&[&str], &str); 7] = [
    (&["decision"], "decisions"),
    (&["error", "solution", "testing"], "troubleshooting"),
    (
        &[
            "pattern",
            "convention",
            "workaround",
            "configuration",
            "performance",
        ],
        "patterns",
    ),
    (&["environment", "dependency"], "context"),
    (&["procedure", "workflow"], "procedures"),
    (&["definition", "api"], "references"),
    (&["important", "explicit"], "notes"),
];

/// Version 1's `derive_namespace`: the first map entry, in map order, that
/// holds any of the families; `notes` when none does.
pub fn v1_namespace<S: AsRef<str>>(families: &[S]) -> &'static str {
    for (keys, namespace) in NAMESPACE_MAP {
        if families.iter().any(|f| keys.contains(&f.as_ref())) {
            return namespace;
        }
    }
    "notes"
}

/// The memory type for a set of families (SPEC section 16, Type and
/// provenance), starting from version 1's namespace: `decisions` is a
/// decision, `troubleshooting` a pitfall, `patterns` a convention, and
/// everything else a note.
pub fn memory_type_for<S: AsRef<str>>(families: &[S]) -> MemoryType {
    match v1_namespace(families) {
        "decisions" => MemoryType::Decision,
        "troubleshooting" => MemoryType::Pitfall,
        "patterns" => MemoryType::Convention,
        _ => MemoryType::Note,
    }
}
