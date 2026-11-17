//! Readings of another process from outside: its working set, its private
//! memory and its threads (stream K, Part 2, Readings; B-58, B-114).
//!
//! - Windows: `K32GetProcessMemoryInfo` with `PROCESS_MEMORY_COUNTERS_EX`
//!   (`WorkingSetSize`, `PeakWorkingSetSize`, `PrivateUsage`); threads from a
//!   Toolhelp snapshot, their names from `GetThreadDescription`.
//! - Linux: `/proc/<pid>/status` (`VmRSS`, `VmHWM`, `RssAnon`, `RssFile`);
//!   threads from `/proc/<pid>/task`, their names from `comm`.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::Duration;

/// One reading of a process.
#[derive(Debug, Clone, Copy, Default)]
pub struct Reading {
    /// The working set (Windows) or resident set (Linux), in bytes.
    pub working_set: u64,
    /// The peak working set (Windows) or `VmHWM` (Linux) since the process
    /// started.
    pub peak_working_set: u64,
    /// Private memory: `PrivateUsage` on Windows (the commit charge),
    /// `RssAnon` on Linux.
    pub private: u64,
    /// `RssFile` on Linux: file-backed resident pages. `None` on Windows.
    pub file_backed: Option<u64>,
    /// Threads in the process.
    pub threads: usize,
}

/// Reads the process, or `None` once it has gone.
pub fn read(pid: u32) -> Option<Reading> {
    let mut r = imp::memory(pid)?;
    r.threads = imp::thread_ids(pid).len();
    Some(r)
}

/// Windows: turns power throttling (EcoQoS) off for the process from
/// outside, a diagnostic only; the daemon itself is left as built.
#[cfg(windows)]
pub fn no_power_throttling(pid: u32) -> Result<(), String> {
    imp::no_power_throttling(pid)
}

/// The name of each thread of the process, counted by name. A thread without
/// a name is `(unnamed: <module>)` on Windows, after the module its start
/// address is in, or `(unnamed)`.
pub fn thread_names(pid: u32) -> BTreeMap<String, usize> {
    let mut out = BTreeMap::new();
    let modules = imp::modules(pid);
    for tid in imp::thread_ids(pid) {
        let name = imp::thread_name(pid, tid)
            .filter(|n| !n.is_empty())
            .unwrap_or_else(|| match imp::start_module(tid, &modules) {
                Some(m) => format!("(unnamed: {m})"),
                None => "(unnamed)".to_owned(),
            });
        *out.entry(name).or_insert(0) += 1;
    }
    out
}

/// `name x count` for each name, in name order.
pub fn names_line(names: &BTreeMap<String, usize>) -> String {
    names
        .iter()
        .map(|(n, c)| format!("{n} x{c}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Bytes as MiB with one decimal.
pub fn mib(bytes: u64) -> String {
    format!("{:.1} MiB", bytes as f64 / (1024.0 * 1024.0))
}

/// The highest figures seen while sampling.
#[derive(Debug, Clone, Default)]
pub struct Peak {
    pub samples: usize,
    pub max_working_set: u64,
    pub max_private: u64,
    pub max_threads: usize,
    /// The threads by name at the first sample that reached `max_threads`.
    pub names_at_max: BTreeMap<String, usize>,
    /// The most threads the daemon's own code had at once ([`own`]).
    pub max_own: usize,
    /// The threads by name at the first sample that reached `max_own`.
    pub names_at_max_own: BTreeMap<String, usize>,
    /// The last reading taken.
    pub last: Option<Reading>,
}

/// The threads the daemon's own code names: `sm-` and then a role (the
/// main thread is left out: Linux gives it the process's name, as it gives
/// every thread without a name of its own).
pub fn own(names: &BTreeMap<String, usize>) -> usize {
    names
        .iter()
        .filter(|(n, _)| n.starts_with("sm-"))
        .map(|(_, c)| c)
        .sum()
}

/// Reads a process every `every` on its own thread until stopped.
pub struct Sampler {
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<Peak>>,
}

impl Sampler {
    pub fn start(pid: u32, every: Duration) -> Sampler {
        let stop = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&stop);
        let handle = std::thread::Builder::new()
            .name("budgets-sampler".to_owned())
            .spawn(move || {
                let mut peak = Peak::default();
                while !flag.load(Ordering::SeqCst) {
                    let Some(r) = read(pid) else { break };
                    peak.samples += 1;
                    peak.max_working_set = peak.max_working_set.max(r.working_set);
                    peak.max_private = peak.max_private.max(r.private);
                    let names = thread_names(pid);
                    if r.threads > peak.max_threads {
                        peak.max_threads = r.threads;
                        peak.names_at_max = names.clone();
                    }
                    let mine = own(&names);
                    if mine > peak.max_own {
                        peak.max_own = mine;
                        peak.names_at_max_own = names;
                    }
                    peak.last = Some(r);
                    std::thread::sleep(every);
                }
                peak
            })
            .expect("start the sampler thread");
        Sampler {
            stop,
            handle: Some(handle),
        }
    }

    pub fn stop(mut self) -> Peak {
        self.stop.store(true, Ordering::SeqCst);
        self.handle
            .take()
            .map(|h| h.join().expect("the sampler thread"))
            .unwrap_or_default()
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

#[cfg(windows)]
mod imp {
    use super::Reading;
    use windows_sys::Wdk::System::Threading::{
        NtQueryInformationThread, ThreadQuerySetWin32StartAddress,
    };
    use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE, LocalFree};
    use windows_sys::Win32::System::Diagnostics::ToolHelp::{
        CreateToolhelp32Snapshot, MODULEENTRY32W, Module32FirstW, Module32NextW, TH32CS_SNAPMODULE,
        TH32CS_SNAPTHREAD, THREADENTRY32, Thread32First, Thread32Next,
    };
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS, PROCESS_MEMORY_COUNTERS_EX,
    };
    use windows_sys::Win32::System::Threading::{
        GetThreadDescription, OpenProcess, OpenThread, PROCESS_QUERY_LIMITED_INFORMATION,
        PROCESS_VM_READ, THREAD_QUERY_INFORMATION, THREAD_QUERY_LIMITED_INFORMATION,
    };

    /// Turns Windows power throttling (EcoQoS execution speed) off for
    /// process `pid`, from outside it: `Err` says why it could not.
    pub(super) fn no_power_throttling(pid: u32) -> Result<(), String> {
        use windows_sys::Win32::System::Threading::{
            PROCESS_POWER_THROTTLING_CURRENT_VERSION, PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
            PROCESS_POWER_THROTTLING_STATE, PROCESS_SET_INFORMATION, ProcessPowerThrottling,
            SetProcessInformation,
        };
        let state = PROCESS_POWER_THROTTLING_STATE {
            Version: PROCESS_POWER_THROTTLING_CURRENT_VERSION,
            ControlMask: PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
            StateMask: 0,
        };
        let size = u32::try_from(size_of::<PROCESS_POWER_THROTTLING_STATE>())
            .map_err(|e| e.to_string())?;
        // SAFETY: the handle is checked before use and closed after; `state`
        // is a readable PROCESS_POWER_THROTTLING_STATE of `size` bytes.
        unsafe {
            let h = OpenProcess(PROCESS_SET_INFORMATION, 0, pid);
            if h.is_null() {
                return Err(format!("OpenProcess: {}", std::io::Error::last_os_error()));
            }
            let ok =
                SetProcessInformation(h, ProcessPowerThrottling, (&raw const state).cast(), size);
            let err = std::io::Error::last_os_error();
            CloseHandle(h);
            if ok == 0 {
                return Err(format!("SetProcessInformation: {err}"));
            }
        }
        Ok(())
    }

    pub(super) fn memory(pid: u32) -> Option<Reading> {
        let cb = u32::try_from(size_of::<PROCESS_MEMORY_COUNTERS_EX>()).ok()?;
        let mut c = PROCESS_MEMORY_COUNTERS_EX {
            cb,
            ..Default::default()
        };
        // SAFETY: the handle is checked before use and closed after; `c` is
        // a writable PROCESS_MEMORY_COUNTERS_EX of `cb` bytes, which starts
        // with the fields of PROCESS_MEMORY_COUNTERS.
        let ok = unsafe {
            let h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ, 0, pid);
            if h.is_null() {
                return None;
            }
            let ok = K32GetProcessMemoryInfo(h, (&raw mut c).cast::<PROCESS_MEMORY_COUNTERS>(), cb);
            CloseHandle(h);
            ok
        };
        (ok != 0).then_some(Reading {
            working_set: c.WorkingSetSize as u64,
            peak_working_set: c.PeakWorkingSetSize as u64,
            private: c.PrivateUsage as u64,
            file_backed: None,
            threads: 0,
        })
    }

    pub(super) fn thread_ids(pid: u32) -> Vec<u32> {
        let mut ids = Vec::new();
        // SAFETY: plain calls; the snapshot handle is checked and closed.
        unsafe {
            let snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
            if snap.is_null() || snap == INVALID_HANDLE_VALUE {
                return ids;
            }
            let mut entry = THREADENTRY32 {
                dwSize: u32::try_from(size_of::<THREADENTRY32>()).unwrap_or(0),
                ..Default::default()
            };
            let mut ok = Thread32First(snap, &mut entry);
            while ok != 0 {
                if entry.th32OwnerProcessID == pid {
                    ids.push(entry.th32ThreadID);
                }
                ok = Thread32Next(snap, &mut entry);
            }
            CloseHandle(snap);
        }
        ids
    }

    pub(super) fn thread_name(_pid: u32, tid: u32) -> Option<String> {
        // SAFETY: the thread handle is checked and closed; the description
        // is a NUL-terminated wide string the system allocated, read up to
        // its NUL and freed with LocalFree.
        unsafe {
            let h = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, 0, tid);
            if h.is_null() {
                return None;
            }
            let mut p: windows_sys::core::PWSTR = std::ptr::null_mut();
            let hr = GetThreadDescription(h, &mut p);
            CloseHandle(h);
            if hr < 0 || p.is_null() {
                return None;
            }
            let mut len = 0;
            while *p.add(len) != 0 {
                len += 1;
            }
            let name = String::from_utf16_lossy(std::slice::from_raw_parts(p, len));
            LocalFree(p.cast());
            Some(name)
        }
    }

    /// The process's modules: base address, size and file name.
    pub(super) type Modules = Vec<(usize, usize, String)>;

    pub(super) fn modules(pid: u32) -> Modules {
        let mut out = Vec::new();
        // SAFETY: plain calls; the snapshot handle is checked and closed, and
        // `szModule` is a NUL-terminated array inside the entry.
        unsafe {
            let snap = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, pid);
            if snap.is_null() || snap == INVALID_HANDLE_VALUE {
                return out;
            }
            let mut entry = MODULEENTRY32W {
                dwSize: u32::try_from(size_of::<MODULEENTRY32W>()).unwrap_or(0),
                ..Default::default()
            };
            let mut ok = Module32FirstW(snap, &mut entry);
            while ok != 0 {
                let len = entry
                    .szModule
                    .iter()
                    .position(|&c| c == 0)
                    .unwrap_or(entry.szModule.len());
                out.push((
                    entry.modBaseAddr as usize,
                    entry.modBaseSize as usize,
                    String::from_utf16_lossy(&entry.szModule[..len]),
                ));
                ok = Module32NextW(snap, &mut entry);
            }
            CloseHandle(snap);
        }
        out
    }

    /// The module the thread's start address is in.
    pub(super) fn start_module(tid: u32, modules: &Modules) -> Option<String> {
        let mut start: usize = 0;
        // SAFETY: the thread handle is checked and closed; the call writes
        // one pointer-sized value into `start`, whose size it is given.
        let status = unsafe {
            let h = OpenThread(THREAD_QUERY_INFORMATION, 0, tid);
            if h.is_null() {
                return None;
            }
            let status = NtQueryInformationThread(
                h,
                ThreadQuerySetWin32StartAddress,
                (&raw mut start).cast(),
                u32::try_from(size_of::<usize>()).unwrap_or(0),
                std::ptr::null_mut(),
            );
            CloseHandle(h);
            status
        };
        if status < 0 || start == 0 {
            return None;
        }
        modules
            .iter()
            .find(|(base, size, _)| start >= *base && start < base + size)
            .map(|(_, _, name)| name.clone())
            .or_else(|| Some(format!("{start:#x}")))
    }
}

#[cfg(not(windows))]
mod imp {
    use super::Reading;

    fn kb(status: &str, key: &str) -> Option<u64> {
        status.lines().find_map(|line| {
            let value = line.strip_prefix(key)?.strip_prefix(':')?;
            let kb = value
                .trim()
                .strip_suffix("kB")?
                .trim()
                .parse::<u64>()
                .ok()?;
            Some(kb * 1024)
        })
    }

    pub(super) fn memory(pid: u32) -> Option<Reading> {
        let status = std::fs::read_to_string(format!("/proc/{pid}/status")).ok()?;
        Some(Reading {
            working_set: kb(&status, "VmRSS")?,
            peak_working_set: kb(&status, "VmHWM")?,
            private: kb(&status, "RssAnon")?,
            file_backed: kb(&status, "RssFile"),
            threads: 0,
        })
    }

    pub(super) fn thread_ids(pid: u32) -> Vec<u32> {
        std::fs::read_dir(format!("/proc/{pid}/task"))
            .map(|dir| {
                dir.filter_map(Result::ok)
                    .filter_map(|e| e.file_name().to_str()?.parse().ok())
                    .collect()
            })
            .unwrap_or_default()
    }

    pub(super) fn thread_name(pid: u32, tid: u32) -> Option<String> {
        std::fs::read_to_string(format!("/proc/{pid}/task/{tid}/comm"))
            .ok()
            .map(|s| s.trim().to_owned())
    }

    /// Linux names every thread (`comm`, the process's name by default), so
    /// no module is looked up.
    pub(super) type Modules = Vec<(usize, usize, String)>;

    pub(super) fn modules(_pid: u32) -> Modules {
        Vec::new()
    }

    pub(super) fn start_module(_tid: u32, _modules: &Modules) -> Option<String> {
        None
    }
}
