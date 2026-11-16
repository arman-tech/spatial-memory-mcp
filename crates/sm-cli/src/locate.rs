//! Finding a program on `PATH`, the way a shell would, without running it.
//! `setup` uses it for `claude`; `doctor` uses it for `spatial-memory`
//! (SPEC section 16, Finding the binary).

use std::path::{Path, PathBuf};

use crate::profile::GetEnv;

/// The extensions tried on Windows when `PATHEXT` is unset.
const DEFAULT_PATHEXT: &str = ".COM;.EXE;.BAT;.CMD";

/// The first file named `name` (with a `PATHEXT` extension on Windows) in
/// the folders of `PATH`.
pub fn on_path(name: &str, get_env: GetEnv) -> Option<PathBuf> {
    let path = get_env("PATH")?;
    let exts: Vec<String> = if cfg!(windows) {
        get_env("PATHEXT")
            .filter(|v| !v.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_PATHEXT.to_owned())
            .split(';')
            .map(str::trim)
            .filter(|e| !e.is_empty())
            .map(str::to_ascii_lowercase)
            .collect()
    } else {
        Vec::new()
    };
    for dir in std::env::split_paths(&path) {
        if dir.as_os_str().is_empty() {
            continue;
        }
        if cfg!(windows) {
            for ext in &exts {
                let candidate = dir.join(format!("{name}{ext}"));
                if is_program(&candidate) {
                    return Some(candidate);
                }
            }
        } else {
            let candidate = dir.join(name);
            if is_program(&candidate) {
                return Some(candidate);
            }
        }
    }
    None
}

#[cfg(unix)]
fn is_program(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(path)
        .map(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
        .unwrap_or(false)
}

#[cfg(not(unix))]
fn is_program(path: &Path) -> bool {
    path.is_file()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s16_program_is_found_in_the_path_folders_in_order() {
        let tmp = tempfile::tempdir().unwrap();
        let first = tmp.path().join("a");
        let second = tmp.path().join("b");
        std::fs::create_dir_all(&first).unwrap();
        std::fs::create_dir_all(&second).unwrap();
        let file = if cfg!(windows) { "tool.cmd" } else { "tool" };
        let target = second.join(file);
        std::fs::write(&target, "").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&target, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        let joined = std::env::join_paths([&first, &second]).unwrap();
        let joined = joined.to_string_lossy().into_owned();
        let get = move |k: &str| (k == "PATH").then(|| joined.clone());
        assert_eq!(on_path("tool", &get), Some(target));
        assert_eq!(on_path("absent", &get), None);
        let none = |_: &str| None;
        assert_eq!(on_path("tool", &none), None);
    }
}
