//! What the process reads back from the operating system: peak and current
//! memory (AT-4 and the memory budgets), thread counts, and the priority of
//! a thread (D-10; SPEC 5, "Using the machine's cores").
//!
//! Windows uses the Win32 API, Linux reads `/proc`. Elsewhere every reading
//! is `None` and lowering a thread fails, so bulk work takes the SPEC 13
//! fallback (half the threads, reported).

/// The priority a lowered bulk thread reads back as: Windows
/// `THREAD_PRIORITY_BELOW_NORMAL` (-1); on Linux a nice value this much
/// above the process's own (capped at 19). Both are absolute targets, so
/// lowering a thread twice, or a thread that inherited a lowered nice value
/// from the thread that started it, still leaves it one step below normal.
pub const LINUX_NICE_STEP: i32 = 10;

/// The calling thread's id as the OS knows it.
pub fn current_thread_id() -> u64 {
    imp::current_thread_id()
}

/// Peak resident memory of this process in bytes: `PeakWorkingSetSize` on
/// Windows, `VmHWM` on Linux.
pub fn peak_memory_bytes() -> Option<u64> {
    imp::peak_memory_bytes()
}

/// Resident memory of this process in bytes now: `WorkingSetSize` on
/// Windows, `VmRSS` on Linux.
pub fn current_memory_bytes() -> Option<u64> {
    imp::current_memory_bytes()
}

/// Threads of this process as the OS counts them.
pub fn process_thread_count() -> Option<usize> {
    imp::process_thread_count()
}

/// The priority of thread `tid` of this process, read from the OS: the
/// Windows relative thread priority (0 normal, -1 below normal), or the
/// Linux nice value (0 normal, higher is lower priority).
pub fn thread_priority(tid: u64) -> Option<i32> {
    imp::thread_priority(tid)
}

/// Lowers the calling thread below normal priority and returns the
/// priority read back from the OS, or why it could not.
pub fn lower_current_thread() -> Result<i32, String> {
    imp::lower_current_thread()
}

/// Whether `priority`, as [`thread_priority`] reads it, is below `normal`.
pub fn is_below(priority: i32, normal: i32) -> bool {
    if cfg!(windows) {
        priority < normal
    } else {
        priority > normal
    }
}

/// The processor's name, for reports: `PROCESSOR_IDENTIFIER` on Windows,
/// the `model name` line of `/proc/cpuinfo` on Linux.
pub fn cpu_name() -> String {
    imp::cpu_name().unwrap_or_else(|| "unknown processor".to_owned())
}

#[cfg(windows)]
mod imp {
    use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Diagnostics::ToolHelp::{
        CreateToolhelp32Snapshot, TH32CS_SNAPTHREAD, THREADENTRY32, Thread32First, Thread32Next,
    };
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::{
        GetCurrentProcess, GetCurrentProcessId, GetCurrentThread, GetCurrentThreadId,
        GetThreadPriority, OpenThread, SetThreadPriority, THREAD_PRIORITY_BELOW_NORMAL,
        THREAD_QUERY_LIMITED_INFORMATION,
    };

    /// `THREAD_PRIORITY_ERROR_RETURN`.
    const PRIORITY_ERROR: i32 = 0x7fff_ffff;

    pub(super) fn current_thread_id() -> u64 {
        // SAFETY: no arguments; always succeeds.
        u64::from(unsafe { GetCurrentThreadId() })
    }

    fn counters() -> Option<PROCESS_MEMORY_COUNTERS> {
        let mut c = PROCESS_MEMORY_COUNTERS::default();
        let cb = u32::try_from(size_of::<PROCESS_MEMORY_COUNTERS>()).ok()?;
        // SAFETY: `c` is a writable PROCESS_MEMORY_COUNTERS of `cb` bytes;
        // the current-process pseudo handle needs no closing.
        let ok = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, cb) };
        (ok != 0).then_some(c)
    }

    pub(super) fn peak_memory_bytes() -> Option<u64> {
        counters().map(|c| c.PeakWorkingSetSize as u64)
    }

    pub(super) fn current_memory_bytes() -> Option<u64> {
        counters().map(|c| c.WorkingSetSize as u64)
    }

    pub(super) fn process_thread_count() -> Option<usize> {
        // SAFETY: plain calls; the snapshot handle is checked and closed.
        unsafe {
            let pid = GetCurrentProcessId();
            let snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
            if snap.is_null() || snap == INVALID_HANDLE_VALUE {
                return None;
            }
            let mut entry = THREADENTRY32 {
                dwSize: u32::try_from(size_of::<THREADENTRY32>()).unwrap_or(0),
                ..Default::default()
            };
            let mut n = 0;
            let mut ok = Thread32First(snap, &mut entry);
            while ok != 0 {
                if entry.th32OwnerProcessID == pid {
                    n += 1;
                }
                ok = Thread32Next(snap, &mut entry);
            }
            CloseHandle(snap);
            Some(n)
        }
    }

    pub(super) fn thread_priority(tid: u64) -> Option<i32> {
        let tid = u32::try_from(tid).ok()?;
        // SAFETY: the handle is checked before use and closed after.
        unsafe {
            let h = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, 0, tid);
            if h.is_null() {
                return None;
            }
            let p = GetThreadPriority(h);
            CloseHandle(h);
            (p != PRIORITY_ERROR).then_some(p)
        }
    }

    pub(super) fn lower_current_thread() -> Result<i32, String> {
        // SAFETY: the current-thread pseudo handle is always valid.
        let (ok, back) = unsafe {
            let ok = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
            (ok, GetThreadPriority(GetCurrentThread()))
        };
        if ok == 0 {
            return Err(format!(
                "SetThreadPriority(THREAD_PRIORITY_BELOW_NORMAL) failed: {}",
                std::io::Error::last_os_error()
            ));
        }
        if back != THREAD_PRIORITY_BELOW_NORMAL {
            return Err(format!(
                "the thread priority reads back as {back}, not {THREAD_PRIORITY_BELOW_NORMAL}"
            ));
        }
        Ok(back)
    }

    pub(super) fn cpu_name() -> Option<String> {
        std::env::var("PROCESSOR_IDENTIFIER").ok()
    }
}

#[cfg(target_os = "linux")]
mod imp {
    use super::LINUX_NICE_STEP;

    pub(super) fn current_thread_id() -> u64 {
        // SAFETY: gettid has no preconditions.
        let tid = unsafe { libc::gettid() };
        u64::try_from(tid).unwrap_or(0)
    }

    fn status_bytes(key: &str) -> Option<u64> {
        let status = std::fs::read_to_string("/proc/self/status").ok()?;
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

    pub(super) fn peak_memory_bytes() -> Option<u64> {
        status_bytes("VmHWM")
    }

    pub(super) fn current_memory_bytes() -> Option<u64> {
        status_bytes("VmRSS")
    }

    pub(super) fn process_thread_count() -> Option<usize> {
        Some(std::fs::read_dir("/proc/self/task").ok()?.count())
    }

    /// The nice value, field 19 of `/proc/self/task/<tid>/stat`.
    pub(super) fn thread_priority(tid: u64) -> Option<i32> {
        let stat = std::fs::read_to_string(format!("/proc/self/task/{tid}/stat")).ok()?;
        // The command name is in parentheses and may hold spaces; fields
        // after it start at field 3 (state).
        let rest = &stat[stat.rfind(')')? + 1..];
        rest.split_whitespace().nth(19 - 3)?.parse().ok()
    }

    /// The process's own nice value: that of its main thread, whose id is
    /// the process id.
    fn process_nice() -> Option<i32> {
        thread_priority(u64::from(std::process::id()))
    }

    /// Sets the calling thread's nice value to the process's plus
    /// [`LINUX_NICE_STEP`], capped at 19. Linux threads inherit the nice
    /// value of the thread that starts them, and ONNX Runtime starts its
    /// pool from the lowered bulk runner, so the target is absolute: a
    /// thread already at or past it is left as it is, never raised.
    pub(super) fn lower_current_thread() -> Result<i32, String> {
        // SAFETY: gettid has no preconditions.
        let tid = unsafe { libc::gettid() };
        let id = u64::try_from(tid).map_err(|_| "gettid returned a negative id".to_owned())?;
        let before =
            thread_priority(id).ok_or_else(|| "cannot read the thread's nice value".to_owned())?;
        let normal =
            process_nice().ok_or_else(|| "cannot read the process's nice value".to_owned())?;
        let target = (normal + LINUX_NICE_STEP).min(19);
        if before >= target {
            return Ok(before);
        }
        let who = libc::id_t::try_from(tid).map_err(|_| "thread id out of range".to_owned())?;
        // SAFETY: plain call; on Linux PRIO_PROCESS with a thread id sets
        // that one thread's nice value.
        let rc = unsafe { libc::setpriority(libc::PRIO_PROCESS as _, who, target) };
        if rc != 0 {
            return Err(format!(
                "setpriority(PRIO_PROCESS, {tid}, {target}) failed: {}",
                std::io::Error::last_os_error()
            ));
        }
        let back = thread_priority(id)
            .ok_or_else(|| "cannot read the thread's nice value back".to_owned())?;
        if back != target {
            return Err(format!("the nice value reads back as {back}, not {target}"));
        }
        Ok(back)
    }

    pub(super) fn cpu_name() -> Option<String> {
        let info = std::fs::read_to_string("/proc/cpuinfo").ok()?;
        info.lines().find_map(|line| {
            let (key, value) = line.split_once(':')?;
            (key.trim() == "model name").then(|| value.trim().to_owned())
        })
    }
}

#[cfg(not(any(windows, target_os = "linux")))]
mod imp {
    pub(super) fn current_thread_id() -> u64 {
        0
    }

    pub(super) fn peak_memory_bytes() -> Option<u64> {
        None
    }

    pub(super) fn current_memory_bytes() -> Option<u64> {
        None
    }

    pub(super) fn process_thread_count() -> Option<usize> {
        None
    }

    pub(super) fn thread_priority(_tid: u64) -> Option<i32> {
        None
    }

    pub(super) fn lower_current_thread() -> Result<i32, String> {
        Err("thread priority is not supported on this operating system".to_owned())
    }

    pub(super) fn cpu_name() -> Option<String> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(any(windows, target_os = "linux"))]
    fn s5_os_readings_are_available() {
        // Current first: other tests allocate in parallel, and the peak only
        // grows, so a peak read afterwards is at least the current reading.
        let now = current_memory_bytes().unwrap();
        let peak = peak_memory_bytes().unwrap();
        assert!(peak >= now && now > 0, "peak {peak}, now {now}");
        assert!(process_thread_count().unwrap() >= 1);
        let tid = current_thread_id();
        assert!(tid > 0);
        assert!(thread_priority(tid).is_some());
    }

    #[test]
    #[cfg(any(windows, target_os = "linux"))]
    fn d10_a_lowered_thread_reads_back_below_normal() {
        let main = thread_priority(current_thread_id()).unwrap();
        let (tid, lowered, back) = std::thread::spawn(|| {
            let tid = current_thread_id();
            let lowered = lower_current_thread();
            (tid, lowered, thread_priority(tid))
        })
        .join()
        .unwrap();
        assert!(tid != current_thread_id());
        match lowered {
            Ok(p) => {
                assert_eq!(back, Some(p));
                assert!(is_below(p, main) || p == 19, "{p} vs {main}");
            }
            // A system that refuses is the SPEC 13 fallback, not a failure
            // here; the bulk session reports it.
            Err(e) => eprintln!("thread priority refused: {e}"),
        }
        // The calling thread is untouched.
        assert_eq!(thread_priority(current_thread_id()), Some(main));
    }

    /// Found by the Linux real-model run: pool threads inherit the lowered
    /// runner's nice value, and lowering them again took them from 10 to 19.
    /// Lowering is one step below normal however often it is applied and
    /// whatever the thread inherited.
    #[test]
    #[cfg(any(windows, target_os = "linux"))]
    fn d10_lowering_is_one_step_however_often_it_runs() {
        let main = thread_priority(current_thread_id()).unwrap();
        let (first, second, child) = std::thread::spawn(|| {
            let first = lower_current_thread();
            let second = lower_current_thread();
            // A thread started by a lowered thread, then lowered itself, as
            // ONNX Runtime's pool threads are under the bulk runner.
            let child = std::thread::spawn(lower_current_thread).join().unwrap();
            (first, second, child)
        })
        .join()
        .unwrap();
        match first {
            Ok(p) => {
                assert!(is_below(p, main) || p == 19, "{p} vs {main}");
                assert_eq!(second, Ok(p));
                assert_eq!(child, Ok(p));
            }
            Err(e) => eprintln!("thread priority refused: {e}"),
        }
        assert_eq!(thread_priority(current_thread_id()), Some(main));
    }

    #[test]
    fn s5_priority_direction_matches_the_os() {
        if cfg!(windows) {
            assert!(is_below(-1, 0));
            assert!(!is_below(0, 0));
        } else {
            assert!(is_below(10, 0));
            assert!(!is_below(0, 0));
        }
    }
}
