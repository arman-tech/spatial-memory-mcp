//! Time and memory of this process, child processes and figures.
//!
//! A thread reads this process's working set and private bytes every 2 ms
//! and keeps the highest since the last reset; [`measure`] resets it at the
//! start of a stage, so a stage's peak is the highest reading during it
//! (or at its start or end). Readings on Windows are
//! `K32GetProcessMemoryInfo` on the current process; elsewhere
//! `/proc/self/status` (`VmRSS`, `RssAnon`, `VmHWM`).

use std::io::{BufRead, BufReader, Write as _};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

static PEAK_WS: AtomicU64 = AtomicU64::new(0);
static PEAK_PRIVATE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy, Default)]
pub struct Mem {
    pub working_set: u64,
    pub private: u64,
    /// The process's own lifetime peak of the working set.
    pub peak_working_set: u64,
}

#[cfg(windows)]
pub fn now_mem() -> Mem {
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS, PROCESS_MEMORY_COUNTERS_EX,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let cb = u32::try_from(size_of::<PROCESS_MEMORY_COUNTERS_EX>()).unwrap_or(0);
    let mut c = PROCESS_MEMORY_COUNTERS_EX {
        cb,
        ..Default::default()
    };
    // SAFETY: the current-process pseudo handle needs no closing; `c` is a
    // writable PROCESS_MEMORY_COUNTERS_EX of `cb` bytes, which starts with
    // the fields of PROCESS_MEMORY_COUNTERS.
    let ok = unsafe {
        K32GetProcessMemoryInfo(
            GetCurrentProcess(),
            (&raw mut c).cast::<PROCESS_MEMORY_COUNTERS>(),
            cb,
        )
    };
    if ok == 0 {
        return Mem::default();
    }
    Mem {
        working_set: c.WorkingSetSize as u64,
        private: c.PrivateUsage as u64,
        peak_working_set: c.PeakWorkingSetSize as u64,
    }
}

#[cfg(not(windows))]
pub fn now_mem() -> Mem {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let kb = |key: &str| -> u64 {
        status
            .lines()
            .find_map(|l| l.strip_prefix(key))
            .and_then(|rest| rest.trim_start_matches(':').split_whitespace().next())
            .and_then(|v| v.parse::<u64>().ok())
            .map_or(0, |v| v * 1024)
    };
    Mem {
        working_set: kb("VmRSS"),
        private: kb("RssAnon"),
        peak_working_set: kb("VmHWM"),
    }
}

/// Starts the sampling thread; call once per process.
pub fn start_sampler() {
    std::thread::spawn(|| {
        loop {
            let m = now_mem();
            PEAK_WS.fetch_max(m.working_set, Ordering::Relaxed);
            PEAK_PRIVATE.fetch_max(m.private, Ordering::Relaxed);
            std::thread::sleep(Duration::from_millis(2));
        }
    });
}

/// One measured stage.
#[derive(Debug, Clone)]
pub struct Rec {
    pub stage: String,
    pub ms: f64,
    pub start_ws: u64,
    pub peak_ws: u64,
    pub peak_private: u64,
    pub end_ws: u64,
}

impl Rec {
    pub fn json(&self) -> Value {
        json!({
            "stage": self.stage,
            "ms": self.ms,
            "start_ws": self.start_ws,
            "peak_ws": self.peak_ws,
            "peak_private": self.peak_private,
            "end_ws": self.end_ws,
        })
    }
}

/// Runs `f` as one stage: its wall time and this process's peak memory.
pub fn measure<T>(stage: &str, f: impl FnOnce() -> T) -> (T, Rec) {
    let m0 = now_mem();
    PEAK_WS.store(m0.working_set, Ordering::Relaxed);
    PEAK_PRIVATE.store(m0.private, Ordering::Relaxed);
    let t0 = Instant::now();
    let out = f();
    let elapsed = t0.elapsed();
    let m1 = now_mem();
    PEAK_WS.fetch_max(m1.working_set, Ordering::Relaxed);
    PEAK_PRIVATE.fetch_max(m1.private, Ordering::Relaxed);
    let rec = Rec {
        stage: stage.to_owned(),
        ms: ms(elapsed),
        start_ws: m0.working_set,
        peak_ws: PEAK_WS.load(Ordering::Relaxed),
        peak_private: PEAK_PRIVATE.load(Ordering::Relaxed),
        end_ws: m1.working_set,
    };
    (out, rec)
}

pub fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

pub fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

/// Prints one result line for the parent.
pub fn emit(v: &Value) {
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "RESULT {v}");
    let _ = out.flush();
}

/// Runs this program as a child with `args` after `child`, echoes its other
/// lines and returns its results. A child that fails is reported and gives
/// no results.
pub fn child(args: &[String]) -> Vec<Value> {
    child_timed(args, None).0
}

/// [`child`] with a deadline: a child still running after `timeout` is
/// killed. Returns the results it gave and whether it was killed.
pub fn child_timed(args: &[String], timeout: Option<Duration>) -> (Vec<Value>, bool) {
    let exe = std::env::current_exe().expect("own path");
    let mut cmd = Command::new(exe);
    cmd.arg("child").args(args);
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());
    let started = Instant::now();
    let mut proc = cmd.spawn().expect("spawn a child");
    let stdout = proc.stdout.take().expect("child stdout");
    let reader = std::thread::spawn(move || {
        let mut results = Vec::new();
        for line in BufReader::new(stdout).lines() {
            let Ok(line) = line else { break };
            if let Some(rest) = line.strip_prefix("RESULT ") {
                match serde_json::from_str::<Value>(rest) {
                    Ok(v) => results.push(v),
                    Err(e) => println!("  | unreadable result: {e}"),
                }
            } else {
                println!("  | {line}");
            }
        }
        results
    });
    let mut timed_out = false;
    let status = loop {
        if let Some(s) = proc.try_wait().expect("poll a child") {
            break s;
        }
        if timeout.is_some_and(|t| started.elapsed() > t) {
            let _ = proc.kill();
            timed_out = true;
            break proc.wait().expect("wait for a killed child");
        }
        std::thread::sleep(Duration::from_millis(200));
    };
    let results = reader.join().unwrap_or_default();
    if timed_out {
        println!(
            "  child {args:?} killed after {:.0} s",
            started.elapsed().as_secs_f64()
        );
    } else if !status.success() {
        println!("  child {args:?} failed: {status}");
    }
    (results, timed_out)
}

/// The median, minimum and maximum.
pub fn mrange(xs: &[f64]) -> (f64, f64, f64) {
    let mut v: Vec<f64> = xs.to_vec();
    v.sort_by(f64::total_cmp);
    if v.is_empty() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let n = v.len();
    let med = if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    };
    (med, v[0], v[n - 1])
}

/// `median [min-max]` with `digits` decimals.
pub fn fmt_range(xs: &[f64], digits: usize) -> String {
    let (m, lo, hi) = mrange(xs);
    format!("{m:.digits$} [{lo:.digits$}-{hi:.digits$}]")
}

/// Per stage, across runs: time and peak working set, median and range.
pub fn stage_table(title: &str, runs: &[Value]) {
    println!("{title}");
    let Some(first) = runs.first() else {
        println!("  (no runs)");
        return;
    };
    let stages: Vec<String> = first["recs"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|r| r["stage"].as_str().map(str::to_owned))
                .collect()
        })
        .unwrap_or_default();
    for s in stages {
        let pick = |field: &str| -> Vec<f64> {
            runs.iter()
                .filter_map(|r| {
                    r["recs"]
                        .as_array()?
                        .iter()
                        .find(|x| x["stage"] == s.as_str())
                        .and_then(|x| x[field].as_f64())
                })
                .collect()
        };
        let peak: Vec<f64> = pick("peak_ws").iter().map(|b| *b / 1_048_576.0).collect();
        let start: Vec<f64> = pick("start_ws").iter().map(|b| *b / 1_048_576.0).collect();
        println!(
            "  {s:<24} {} ms | peak WS {} MiB | start WS {} MiB | runs {}",
            fmt_range(&pick("ms"), 1),
            fmt_range(&peak, 0),
            fmt_range(&start, 0),
            peak.len()
        );
    }
}
