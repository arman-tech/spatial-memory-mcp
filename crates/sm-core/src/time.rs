//! UTC timestamps with microsecond precision.
//!
//! Microseconds, because version 1 exports carry six fractional digits and
//! SPEC section 10 copies `created_at` and `updated_at` unchanged. The
//! civil-date conversion is Howard Hinnant's `days_from_civil` and
//! `civil_from_days`, valid for years 0000 to 9999.
//!
//! Text forms:
//! - JSON and the database: RFC 3339 with `Z`, the fraction omitted when it is
//!   zero and otherwise six digits (`2025-08-14T09:21:07.512340Z`).
//! - The metrics journal: the same with `+00:00`, which is what Python's
//!   `datetime.isoformat()` writes for an aware UTC time (version 2 journal).
//! - Parsing accepts RFC 3339 with `Z` or a `+HH:MM`/`-HH:MM` offset, and a
//!   naive ISO 8601 time with no offset, read as UTC (version 1 exports).
//!   `T`, `t` or a space separate date and time. One to nine fractional
//!   digits are accepted and truncated to microseconds.

use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::error::{CoreError, Result};

const MICROS_PER_SEC: i64 = 1_000_000;
const SECS_PER_DAY: i64 = 86_400;

/// Microseconds since 1970-01-01T00:00:00Z.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Timestamp(i64);

impl Timestamp {
    pub const UNIX_EPOCH: Timestamp = Timestamp(0);

    pub const fn from_micros(micros: i64) -> Self {
        Timestamp(micros)
    }

    pub const fn as_micros(self) -> i64 {
        self.0
    }

    pub const fn from_unix_seconds(secs: i64) -> Self {
        Timestamp(secs * MICROS_PER_SEC)
    }

    /// Whole seconds since the epoch, rounded down.
    pub const fn unix_seconds(self) -> i64 {
        self.0.div_euclid(MICROS_PER_SEC)
    }

    /// The current time. A clock before 1970 reads as the epoch.
    pub fn now() -> Self {
        let micros = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| i64::try_from(d.as_micros()).unwrap_or(i64::MAX))
            .unwrap_or(0);
        Timestamp(micros)
    }

    /// This time plus a signed number of microseconds, saturating.
    #[must_use]
    pub const fn add_micros(self, micros: i64) -> Self {
        Timestamp(self.0.saturating_add(micros))
    }

    /// Whole days from `earlier` to `self`, rounded down (negative when
    /// `earlier` is later).
    pub const fn days_since(self, earlier: Timestamp) -> i64 {
        (self.0 - earlier.0).div_euclid(SECS_PER_DAY * MICROS_PER_SEC)
    }

    /// `YYYY-MM-DDTHH:MM:SS[.ffffff]Z`.
    pub fn to_rfc3339(self) -> String {
        self.format("Z")
    }

    /// `YYYY-MM-DDTHH:MM:SS[.ffffff]+00:00`, Python's `isoformat()` for an
    /// aware UTC time (metrics journal `ts`).
    pub fn to_python_isoformat(self) -> String {
        self.format("+00:00")
    }

    /// `YYYY-MM-DD` (shared file front matter `date`, journal file names).
    pub fn date_string(self) -> String {
        let (y, m, d) = civil_from_days(self.0.div_euclid(SECS_PER_DAY * MICROS_PER_SEC));
        format!("{y:04}-{m:02}-{d:02}")
    }

    fn format(self, suffix: &str) -> String {
        let days = self.0.div_euclid(SECS_PER_DAY * MICROS_PER_SEC);
        let rem = self.0.rem_euclid(SECS_PER_DAY * MICROS_PER_SEC);
        let (y, mo, d) = civil_from_days(days);
        let secs = rem / MICROS_PER_SEC;
        let micros = rem % MICROS_PER_SEC;
        let (h, mi, s) = (secs / 3600, (secs / 60) % 60, secs % 60);
        if micros == 0 {
            format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}{suffix}")
        } else {
            format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}.{micros:06}{suffix}")
        }
    }

    /// Parses the forms listed in the module documentation.
    pub fn parse(text: &str) -> Result<Self> {
        parse_timestamp(text).ok_or_else(|| {
            CoreError::invalid_input(format!(
                "invalid timestamp {text:?}; expected ISO 8601 such as 2025-08-14T09:21:07Z"
            ))
        })
    }

    /// Parses `YYYY-MM-DD` as midnight UTC (shared file front matter `date`).
    pub fn parse_date(text: &str) -> Result<Self> {
        let b = text.as_bytes();
        let parsed = if b.len() == 10 {
            parse_date_bytes(b)
        } else {
            None
        };
        let (y, m, d) = parsed.ok_or_else(|| {
            CoreError::invalid_input(format!("invalid date {text:?}; expected YYYY-MM-DD"))
        })?;
        Ok(Timestamp(
            days_from_civil(y, m, d) * SECS_PER_DAY * MICROS_PER_SEC,
        ))
    }
}

impl fmt::Display for Timestamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_rfc3339())
    }
}

impl Serialize for Timestamp {
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        s.serialize_str(&self.to_rfc3339())
    }
}

impl<'de> Deserialize<'de> for Timestamp {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        Timestamp::parse(&s).map_err(|e| serde::de::Error::custom(e.message))
    }
}

impl JsonSchema for Timestamp {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed("Timestamp")
    }

    fn json_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({
            "type": "string",
            "format": "date-time",
            "description": "UTC time in RFC 3339 form, for example 2026-11-16T09:21:07.512340Z."
        })
    }
}

/// Days since 1970-01-01 for a proleptic Gregorian date.
pub fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = i64::from((m + 9) % 12);
    let doy = (153 * mp + 2) / 5 + i64::from(d) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// Proleptic Gregorian date for a count of days since 1970-01-01.
pub fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

fn is_leap(y: i64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

fn days_in_month(y: i64, m: u32) -> u32 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        _ if is_leap(y) => 29,
        _ => 28,
    }
}

fn digits(b: &[u8]) -> Option<u32> {
    if b.is_empty() || !b.iter().all(u8::is_ascii_digit) {
        return None;
    }
    Some(b.iter().fold(0u32, |acc, d| acc * 10 + u32::from(d - b'0')))
}

fn parse_date_bytes(b: &[u8]) -> Option<(i64, u32, u32)> {
    if b.len() < 10 || b[4] != b'-' || b[7] != b'-' {
        return None;
    }
    let y = i64::from(digits(&b[0..4])?);
    let m = digits(&b[5..7])?;
    let d = digits(&b[8..10])?;
    if !(1..=12).contains(&m) || d == 0 || d > days_in_month(y, m) {
        return None;
    }
    Some((y, m, d))
}

fn parse_timestamp(text: &str) -> Option<Timestamp> {
    let b = text.as_bytes();
    let (y, mo, d) = parse_date_bytes(b)?;
    if b.len() < 19 || !matches!(b[10], b'T' | b't' | b' ') || b[13] != b':' || b[16] != b':' {
        return None;
    }
    let h = digits(&b[11..13])?;
    let mi = digits(&b[14..16])?;
    let s = digits(&b[17..19])?;
    if h > 23 || mi > 59 || s > 59 {
        return None;
    }
    let mut i = 19;
    let mut micros: i64 = 0;
    if i < b.len() && b[i] == b'.' {
        let start = i + 1;
        let mut end = start;
        while end < b.len() && b[end].is_ascii_digit() {
            end += 1;
        }
        let n = end - start;
        if n == 0 || n > 9 {
            return None;
        }
        let mut frac = digits(&b[start..end.min(start + 6)])? as i64;
        for _ in n.min(6)..6 {
            frac *= 10;
        }
        micros = frac;
        i = end;
    }
    let offset_secs: i64 = match &b[i..] {
        [] => 0,
        [b'Z' | b'z'] => 0,
        [sign @ (b'+' | b'-'), rest @ ..] if rest.len() == 5 && rest[2] == b':' => {
            let oh = i64::from(digits(&rest[0..2])?);
            let om = i64::from(digits(&rest[3..5])?);
            if oh > 23 || om > 59 {
                return None;
            }
            let total = oh * 3600 + om * 60;
            if *sign == b'+' { total } else { -total }
        }
        _ => return None,
    };
    let days = days_from_civil(y, mo, d);
    let secs =
        days * SECS_PER_DAY + i64::from(h) * 3600 + i64::from(mi) * 60 + i64::from(s) - offset_secs;
    Some(Timestamp(secs * MICROS_PER_SEC + micros))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s10_parses_v1_naive_timestamps_as_utc() {
        let t = Timestamp::parse("2025-08-14T09:21:07.512340").unwrap();
        assert_eq!(t.to_rfc3339(), "2025-08-14T09:21:07.512340Z");
        let t = Timestamp::parse("2025-09-01T00:00:00").unwrap();
        assert_eq!(t.to_rfc3339(), "2025-09-01T00:00:00Z");
        let t = Timestamp::parse("2025-01-01T00:00:00.000001").unwrap();
        assert_eq!(t.to_rfc3339(), "2025-01-01T00:00:00.000001Z");
        assert_eq!(t.as_micros() % 1_000_000, 1);
    }

    #[test]
    fn s10_v1_fixture_timestamps_round_trip_unchanged() {
        // AT-8: created_at and updated_at are copied unchanged.
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/migration/v1_export_full.json"
        );
        let records: Vec<serde_json::Value> =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        for rec in &records {
            for key in ["created_at", "updated_at", "last_accessed"] {
                let s = rec[key].as_str().unwrap();
                let t = Timestamp::parse(s).unwrap();
                assert_eq!(t.to_rfc3339(), format!("{s}Z"), "{key} of {}", rec["id"]);
            }
        }
    }

    #[test]
    fn s10_timestamps_parse_offsets_and_separators() {
        let a = Timestamp::parse("2026-11-16T10:00:00+02:00").unwrap();
        let b = Timestamp::parse("2026-11-16 08:00:00Z").unwrap();
        let c = Timestamp::parse("2026-11-16t03:30:00-04:30").unwrap();
        assert_eq!(a, b);
        assert_eq!(a, c);
        let n = Timestamp::parse("2026-11-16T08:00:00.123456789Z").unwrap();
        assert_eq!(n.as_micros() - b.as_micros(), 123_456);
        let m = Timestamp::parse("2026-11-16T08:00:00.5Z").unwrap();
        assert_eq!(m.as_micros() - b.as_micros(), 500_000);
    }

    #[test]
    fn s10_invalid_timestamps_are_rejected() {
        for bad in [
            "",
            "2026-11-16",
            "2026-13-01T00:00:00",
            "2026-02-29T00:00:00",
            "2026-11-16T24:00:00",
            "2026-11-16T00:00:60",
            "2026-11-16T00:00:00.",
            "2026-11-16T00:00:00.1234567890",
            "2026-11-16T00:00:00+0200",
            "2026-11-16T00:00:00 junk",
            "２026-11-16T00:00:00",
        ] {
            assert!(Timestamp::parse(bad).is_err(), "{bad:?} should fail");
        }
        assert!(Timestamp::parse("2024-02-29T00:00:00").is_ok());
    }

    #[test]
    fn s6_journal_form_matches_python_isoformat() {
        let t = Timestamp::parse("2026-11-16T09:05:03.000250Z").unwrap();
        assert_eq!(t.to_python_isoformat(), "2026-11-16T09:05:03.000250+00:00");
        let t = Timestamp::parse("2026-11-16T09:05:03Z").unwrap();
        assert_eq!(t.to_python_isoformat(), "2026-11-16T09:05:03+00:00");
    }

    #[test]
    fn s4_timestamp_civil_conversion_round_trips() {
        for days in (days_from_civil(0, 1, 1)..=days_from_civil(9999, 12, 31)).step_by(997) {
            let (y, m, d) = civil_from_days(days);
            assert_eq!(days_from_civil(y, m, d), days);
        }
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(civil_from_days(-1), (1969, 12, 31));
        assert_eq!(
            Timestamp::from_micros(-1).to_rfc3339(),
            "1969-12-31T23:59:59.999999Z"
        );
    }

    #[test]
    fn s6_journal_date_string_and_days_since() {
        let t = Timestamp::parse("2026-11-16T23:59:59Z").unwrap();
        assert_eq!(t.date_string(), "2026-11-16");
        assert_eq!(
            Timestamp::parse_date("2026-11-16").unwrap().date_string(),
            "2026-11-16"
        );
        assert!(Timestamp::parse_date("2026-11-1").is_err());
        let earlier = Timestamp::parse("2026-11-09T00:00:00Z").unwrap();
        assert_eq!(t.days_since(earlier), 7);
        assert_eq!(earlier.days_since(t), -8);
    }

    #[test]
    fn s4_timestamps_serialise_as_rfc3339() {
        let t = Timestamp::parse("2026-11-16T09:05:03.25Z").unwrap();
        let json = serde_json::to_string(&t).unwrap();
        assert_eq!(json, "\"2026-11-16T09:05:03.250000Z\"");
        let back: Timestamp = serde_json::from_str(&json).unwrap();
        assert_eq!(back, t);
    }
}
