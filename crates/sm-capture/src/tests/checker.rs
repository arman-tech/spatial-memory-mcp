//! The fixture checker of `fixtures/capture/README.md`: the generators
//! (`$concat`, `$repeat`, `$float`) and the `{path, op, value}` checks.

use regex::Regex;
use serde_json::Value;

use crate::extract::python_truthy;

/// The value with every `$concat` and `$repeat` generator expanded.
/// `$float` and `$py_literal` objects are left as they are.
pub fn resolve(v: &Value) -> Value {
    match v {
        Value::Object(map) => {
            if let Some(parts) = map.get("$concat") {
                let mut out = String::new();
                for part in parts.as_array().expect("$concat takes a list") {
                    match resolve(part) {
                        Value::String(s) => out.push_str(&s),
                        other => panic!("$concat part is not a string: {other}"),
                    }
                }
                return Value::String(out);
            }
            if let Some(unit) = map.get("$repeat") {
                let unit = resolve(unit);
                let unit = unit.as_str().expect("$repeat takes a string");
                let count = map
                    .get("count")
                    .and_then(Value::as_u64)
                    .expect("$repeat takes a count");
                return Value::String(unit.repeat(count as usize));
            }
            Value::Object(map.iter().map(|(k, v)| (k.clone(), resolve(v))).collect())
        }
        Value::Array(items) => Value::Array(items.iter().map(resolve).collect()),
        other => other.clone(),
    }
}

/// A string value, generators expanded.
pub fn text(v: &Value) -> String {
    match resolve(v) {
        Value::String(s) => s,
        other => panic!("not a string: {other}"),
    }
}

/// A number, or a `$float` generator.
pub fn float(v: &Value) -> Option<f64> {
    if let Some(f) = v.as_f64() {
        return Some(f);
    }
    match v.get("$float").and_then(Value::as_str)? {
        "NaN" => Some(f64::NAN),
        "Infinity" => Some(f64::INFINITY),
        "-Infinity" => Some(f64::NEG_INFINITY),
        other => panic!("unknown $float {other}"),
    }
}

/// Whether the value holds a `$py_literal`, a value only Python can hold.
pub fn has_py_literal(v: &Value) -> bool {
    match v {
        Value::Object(map) => map.contains_key("$py_literal") || map.values().any(has_py_literal),
        Value::Array(items) => items.iter().any(has_py_literal),
        _ => false,
    }
}

/// Python's `==` on the values `json.loads` makes: `1 == 1.0`.
pub fn json_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => match (x.as_i64(), y.as_i64()) {
            (Some(i), Some(j)) => i == j,
            _ => x.as_f64() == y.as_f64(),
        },
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| json_eq(p, q))
        }
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len()
                && x.iter()
                    .all(|(k, v)| y.get(k).is_some_and(|w| json_eq(v, w)))
        }
        _ => a == b,
    }
}

/// The value at a dot-separated `path`: list indexes are numbers, and a
/// final `length` is the length (code points for a string). An empty path
/// is the value itself.
pub fn lookup(root: &Value, path: &str) -> Option<Value> {
    if path.is_empty() {
        return Some(root.clone());
    }
    let parts: Vec<&str> = path.split('.').collect();
    let mut current = root.clone();
    for (i, part) in parts.iter().enumerate() {
        let last = i + 1 == parts.len();
        let next = match &current {
            Value::Array(items) => match part.parse::<usize>() {
                Ok(n) => items.get(n).cloned(),
                Err(_) if last && *part == "length" => Some(Value::from(items.len())),
                Err(_) => None,
            },
            Value::Object(map) => match map.get(*part) {
                Some(v) => Some(v.clone()),
                None if last && *part == "length" => Some(Value::from(map.len())),
                None => None,
            },
            Value::String(s) if last && *part == "length" => Some(Value::from(s.chars().count())),
            _ => None,
        };
        current = next?;
    }
    Some(current)
}

fn contains(container: &Value, item: &Value) -> bool {
    match container {
        Value::String(s) => item.as_str().is_some_and(|i| s.contains(i)),
        Value::Array(items) => items.iter().any(|x| json_eq(x, item)),
        Value::Object(map) => item.as_str().is_some_and(|k| map.contains_key(k)),
        _ => false,
    }
}

fn approx(a: f64, b: f64) -> bool {
    if a == b {
        return true;
    }
    (a - b).abs() <= 1e-6 * a.abs().max(b.abs())
}

/// Runs one `{path, op, value}` check against `observed`.
pub fn check(observed: &Value, check: &Value) -> Result<(), String> {
    let path = check.get("path").and_then(Value::as_str).unwrap_or("");
    let op = check
        .get("op")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("check without op: {check}"))?;
    let expected = check.get("value").map(resolve).unwrap_or(Value::Null);
    let actual = lookup(observed, path);
    let num = |v: &Option<Value>| v.as_ref().and_then(Value::as_f64);
    let ok = match op {
        "eq" => actual.as_ref().is_some_and(|a| json_eq(a, &expected)),
        "ne" => !actual.as_ref().is_some_and(|a| json_eq(a, &expected)),
        // Not in the README's list, but used once
        // (`cc_post_tool_use_edit_fix_round_trip`): equal to one of the
        // listed values.
        "one_of" => match (actual.as_ref(), expected.as_array()) {
            (Some(a), Some(options)) => options.iter().any(|o| json_eq(a, o)),
            _ => false,
        },
        "is_null" => actual.as_ref().is_none_or(Value::is_null),
        "not_null" => actual.as_ref().is_some_and(|a| !a.is_null()),
        "contains" => actual.as_ref().is_some_and(|a| contains(a, &expected)),
        "not_contains" => actual.as_ref().is_some_and(|a| !contains(a, &expected)),
        "has_key" => actual
            .as_ref()
            .and_then(Value::as_object)
            .is_some_and(|m| expected.as_str().is_some_and(|k| m.contains_key(k))),
        "not_has_key" => actual
            .as_ref()
            .and_then(Value::as_object)
            .is_some_and(|m| expected.as_str().is_some_and(|k| !m.contains_key(k))),
        "approx" => match (num(&actual), expected.as_f64()) {
            (Some(a), Some(b)) => approx(a, b),
            _ => false,
        },
        "le" => match (num(&actual), expected.as_f64()) {
            (Some(a), Some(b)) => a <= b,
            _ => false,
        },
        "ge" => match (num(&actual), expected.as_f64()) {
            (Some(a), Some(b)) => a >= b,
            _ => false,
        },
        "ends_with" => match (actual.as_ref().and_then(Value::as_str), expected.as_str()) {
            (Some(a), Some(b)) => a.ends_with(b),
            _ => false,
        },
        "matches" => match (actual.as_ref().and_then(Value::as_str), expected.as_str()) {
            (Some(a), Some(pattern)) => Regex::new(&format!("^(?:{pattern})"))
                .map_err(|e| format!("bad pattern {pattern:?}: {e}"))?
                .is_match(a),
            _ => false,
        },
        "type" => {
            let a = actual.as_ref();
            match expected.as_str().unwrap_or("") {
                "int" => a.is_some_and(|v| v.is_i64() || v.is_u64()),
                "float" => a.is_some_and(Value::is_f64),
                "str" => a.is_some_and(Value::is_string),
                "list" => a.is_some_and(Value::is_array),
                "dict" => a.is_some_and(Value::is_object),
                "bool" => a.is_some_and(Value::is_boolean),
                other => return Err(format!("unknown type {other}")),
            }
        }
        "truthy" => actual.as_ref().is_some_and(python_truthy),
        "falsy" => !actual.as_ref().is_some_and(python_truthy),
        "path_eq" => match (actual.as_ref().and_then(Value::as_str), expected.as_str()) {
            (Some(a), Some(b)) => a.replace('\\', "/") == b.replace('\\', "/"),
            _ => false,
        },
        "set_eq" => match (
            actual.as_ref().and_then(Value::as_array),
            expected.as_array(),
        ) {
            (Some(a), Some(b)) => {
                let unique = a
                    .iter()
                    .enumerate()
                    .all(|(i, x)| !a[..i].iter().any(|y| json_eq(x, y)));
                unique && a.len() == b.len() && a.iter().all(|x| b.iter().any(|y| json_eq(x, y)))
            }
            _ => false,
        },
        other => return Err(format!("unknown op {other}")),
    };
    if ok {
        Ok(())
    } else {
        Err(format!(
            "{path} {op} {}: got {}",
            super::short(&expected),
            actual.as_ref().map_or("<missing>".to_owned(), super::short)
        ))
    }
}

/// Runs every check of a list and returns the failures.
pub fn check_all(observed: &Value, checks: &Value) -> Vec<String> {
    checks
        .as_array()
        .map(|list| {
            list.iter()
                .filter_map(|c| check(observed, c).err())
                .collect()
        })
        .unwrap_or_else(|| vec![format!("expect is not a list: {}", super::short(checks))])
}
