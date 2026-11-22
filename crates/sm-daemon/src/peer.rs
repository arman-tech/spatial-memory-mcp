//! Which of two builds is newer (B-228; SPEC 3, Rules, rule 1).
//!
//! A client that meets another build asks the daemon to exit with the
//! reason `version_mismatch` and starts a daemon of its own build. When the
//! client is the older one, that would replace a newer daemon with an older
//! one, so the daemon refuses it (`dispatch`). This module only ranks: the
//! protocol version first, then the build's `major.minor.patch`.

use std::cmp::Ordering;

use sm_core::protocol::PROTOCOL_VERSION;

/// How a peer that speaks protocol `v` with build `build` ranks against
/// this build: `Some(Less)` when the peer is older, `Some(Greater)` when it
/// is newer, `Some(Equal)` when both rank the same.
///
/// A lower or higher protocol decides on its own. With the same protocol,
/// the builds are compared as `major.minor.patch`, ignoring any
/// `+metadata`; `None` when either build is not of that form (a
/// pre-release such as `3.0.0-rc1` included), so it cannot be ranked.
pub fn rank_peer(v: u32, build: &str) -> Option<Ordering> {
    rank(v, build, PROTOCOL_VERSION, sm_core::BUILD)
}

fn rank(v: u32, build: &str, our_v: u32, our_build: &str) -> Option<Ordering> {
    match v.cmp(&our_v) {
        Ordering::Equal => Some(release(build)?.cmp(&release(our_build)?)),
        other => Some(other),
    }
}

/// `major.minor.patch` of `build`, its `+metadata` ignored; `None` unless
/// it is exactly three dot-separated runs of ASCII digits.
fn release(build: &str) -> Option<[u64; 3]> {
    let core = build.split_once('+').map_or(build, |(core, _)| core);
    let mut parts = core.split('.');
    let mut out = [0u64; 3];
    for slot in &mut out {
        let part = parts.next()?;
        if part.is_empty() || !part.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        *slot = part.parse().ok()?;
    }
    parts.next().is_none().then_some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s3_b228_the_protocol_ranks_first() {
        assert_eq!(rank(1, "9.9.9", 3, "3.0.0"), Some(Ordering::Less));
        assert_eq!(rank(2, "3.0.0", 3, "3.0.0"), Some(Ordering::Less));
        assert_eq!(rank(4, "0.0.0", 3, "3.0.0"), Some(Ordering::Greater));
        // Another protocol ranks even when its build cannot be read.
        assert_eq!(rank(1, "not a version", 3, "3.0.0"), Some(Ordering::Less));
        assert_eq!(rank(4, "", 3, "3.0.0"), Some(Ordering::Greater));
    }

    #[test]
    fn s3_b228_with_one_protocol_the_release_numbers_rank() {
        assert_eq!(rank(3, "2.9.9", 3, "3.0.0"), Some(Ordering::Less));
        assert_eq!(rank(3, "3.0.1", 3, "3.0.0"), Some(Ordering::Greater));
        assert_eq!(rank(3, "3.0.0", 3, "3.0.0"), Some(Ordering::Equal));
        // Numbers, not text: 3.10.0 is newer than 3.9.0.
        assert_eq!(rank(3, "3.10.0", 3, "3.9.0"), Some(Ordering::Greater));
        // Build metadata is ignored, on either side.
        assert_eq!(rank(3, "3.0.0+abc1234", 3, "3.0.0"), Some(Ordering::Equal));
        assert_eq!(
            rank(3, "2.9.9+abc1234", 3, "3.0.0+def5678"),
            Some(Ordering::Less)
        );
    }

    #[test]
    fn s3_b228_a_build_that_is_not_three_numbers_does_not_rank() {
        for build in [
            "0.0.0-other",
            "3.0.0-rc1",
            "3.0",
            "3.0.0.0",
            "3..0",
            "v3.0.0",
            "",
            "+abc",
            "3.0.x",
            " 3.0.0",
            "99999999999999999999.0.0",
        ] {
            assert_eq!(rank(3, build, 3, "3.0.0"), None, "{build:?}");
            assert_eq!(rank(3, "3.0.0", 3, build), None, "ours {build:?}");
        }
    }

    #[test]
    fn s3_b228_this_build_ranks_equal_to_itself() {
        assert_eq!(
            rank_peer(PROTOCOL_VERSION, sm_core::BUILD),
            Some(Ordering::Equal)
        );
        assert!(release(sm_core::BUILD).is_some(), "{}", sm_core::BUILD);
    }
}
