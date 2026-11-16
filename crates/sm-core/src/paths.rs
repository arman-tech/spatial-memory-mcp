//! Where things live on disk (SPEC section 4, Locations; section 3, rule 1,
//! Transport; decisions B-3, B-4 and B-17).
//!
//! ```text
//! <root>/config.toml                 settings shared by every profile
//! <root>/<profile>/config.toml       the profile's settings
//! <root>/<profile>/memory.db         the store (SQLite, WAL)
//! <root>/<profile>/daemon.lock       auto-start guard
//! <root>/<profile>/daemon.sock       the socket (Linux and macOS)
//! <root>/<profile>/daemon.log        the daemon's own log
//! <root>/<profile>/metrics/          the metrics journal
//! <root>/<profile>/spool/tmp|new|failed/  capture hand-off when the daemon is down
//! <root>/<profile>/backups/          copies taken before migrate and reindex
//! ```
//!
//! `<root>` is `SPATIAL_MEMORY_HOME`, or `<platform data-local dir>/spatial-memory`.
//! The model cache is separate: `<platform cache dir>/spatial-memory/_models/<id>/<revision>/`,
//! or `SPATIAL_MEMORY_MODEL_DIR`.

use std::path::{Path, PathBuf};

use crate::config::ProfileName;
use crate::error::{CoreError, ErrorCode, Result};
use crate::hash::sha256_hex;
use crate::project::path_key;

/// Folder name under the platform directories.
pub const APP_DIR: &str = "spatial-memory";
/// Replaces `<platform data-local dir>/spatial-memory` (decision B-4).
pub const HOME_ENV: &str = "SPATIAL_MEMORY_HOME";
/// Replaces the whole model directory path (decision B-3).
pub const MODEL_DIR_ENV: &str = "SPATIAL_MEMORY_MODEL_DIR";
/// Folder under the cache root that holds models.
pub const MODELS_DIR: &str = "_models";

/// Longest Unix socket path used in the profile directory. `sun_path` is
/// 104 bytes on macOS and 108 on Linux, including the terminating zero.
pub const MAX_SOCKET_PATH_BYTES: usize = 100;

fn env_path(get_env: &impl Fn(&str) -> Option<String>, key: &str) -> Option<PathBuf> {
    get_env(key)
        .map(|v| v.trim().to_owned())
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

/// The data root: `SPATIAL_MEMORY_HOME`, or the platform's local data folder
/// plus `spatial-memory`.
pub fn data_root(get_env: impl Fn(&str) -> Option<String>) -> Result<PathBuf> {
    if let Some(p) = env_path(&get_env, HOME_ENV) {
        return Ok(p);
    }
    base_dir(BaseDir::DataLocal, &get_env)
        .map(|d| d.join(APP_DIR))
        .ok_or_else(|| {
            CoreError::new(ErrorCode::ConfigInvalid, "cannot find the home folder")
                .with_next_action(format!("Set {HOME_ENV} to a folder for the store."))
        })
}

/// The model directory for a model and revision: `SPATIAL_MEMORY_MODEL_DIR`,
/// or the per-user cache.
pub fn model_dir(
    get_env: impl Fn(&str) -> Option<String>,
    model_id: &str,
    revision: &str,
) -> Result<PathBuf> {
    if let Some(p) = env_path(&get_env, MODEL_DIR_ENV) {
        return Ok(p);
    }
    base_dir(BaseDir::Cache, &get_env)
        .map(|d| {
            d.join(APP_DIR)
                .join(MODELS_DIR)
                .join(model_id)
                .join(revision)
        })
        .ok_or_else(|| {
            CoreError::new(ErrorCode::ModelMissing, "cannot find the cache folder")
                .with_next_action(format!(
                    "Set {MODEL_DIR_ENV} to the folder holding the model files."
                ))
        })
}

/// A per-user platform folder. These are the two folders the build took from
/// the `directories` crate, resolved by the same rules without it (decision
/// B-41).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BaseDir {
    /// Windows `%LOCALAPPDATA%`, Linux `$XDG_DATA_HOME` or `~/.local/share`,
    /// macOS `~/Library/Application Support`.
    DataLocal,
    /// Windows `%LOCALAPPDATA%`, Linux `$XDG_CACHE_HOME` or `~/.cache`, macOS
    /// `~/Library/Caches`.
    Cache,
}

/// The user's home folder: `std::env::home_dir`, which reads `HOME` and then
/// the password database on Unix, and `USERPROFILE` and then the profile
/// folder on Windows. An empty value counts as unknown.
pub fn home_dir() -> Option<PathBuf> {
    std::env::home_dir().filter(|p| !p.as_os_str().is_empty())
}

/// The platform folder of this kind, or `None` when it cannot be found.
pub fn base_dir(kind: BaseDir, get_env: &impl Fn(&str) -> Option<String>) -> Option<PathBuf> {
    let home = home_dir();
    if cfg!(windows) {
        windows_base_dir(get_env, home.as_deref())
    } else if cfg!(target_os = "macos") {
        home.map(|h| macos_base_dir(kind, &h))
    } else {
        xdg_base_dir(kind, get_env, home.as_deref())
    }
}

/// Windows: `LOCALAPPDATA` when it holds an absolute path, else
/// `<home>\AppData\Local`. Windows sets the variable at logon from the
/// `FOLDERID_LocalAppData` known folder that `directories` read through the
/// shell; reading the variable keeps `sm-core` free of unsafe code.
pub fn windows_base_dir(
    get_env: &impl Fn(&str) -> Option<String>,
    home: Option<&Path>,
) -> Option<PathBuf> {
    get_env("LOCALAPPDATA")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .or_else(|| home.map(|h| h.join("AppData").join("Local")))
}

/// The XDG rule used on Linux and the BSDs: the variable when it holds an
/// absolute path, else the default under the home folder.
pub fn xdg_base_dir(
    kind: BaseDir,
    get_env: &impl Fn(&str) -> Option<String>,
    home: Option<&Path>,
) -> Option<PathBuf> {
    let (var, default) = match kind {
        BaseDir::DataLocal => ("XDG_DATA_HOME", Path::new(".local").join("share")),
        BaseDir::Cache => ("XDG_CACHE_HOME", PathBuf::from(".cache")),
    };
    get_env(var)
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .or_else(|| home.map(|h| h.join(default)))
}

/// The macOS folders under `~/Library`.
pub fn macos_base_dir(kind: BaseDir, home: &Path) -> PathBuf {
    let library = home.join("Library");
    match kind {
        BaseDir::DataLocal => library.join("Application Support"),
        BaseDir::Cache => library.join("Caches"),
    }
}

/// How a client reaches the daemon.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Endpoint {
    /// A Unix domain socket file, created with user-only permissions.
    UnixSocket(PathBuf),
    /// A Windows named pipe, by name without the `\\.\pipe\` prefix. The
    /// daemon creates it with a security descriptor that admits only the
    /// current user and as the first instance (spike result 1).
    NamedPipe(String),
}

impl Endpoint {
    /// The full pipe path for a named pipe, or the socket path.
    pub fn display_path(&self) -> String {
        match self {
            Endpoint::UnixSocket(p) => p.to_string_lossy().into_owned(),
            Endpoint::NamedPipe(n) => format!(r"\\.\pipe\{n}"),
        }
    }
}

/// The files of one profile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfilePaths {
    /// The data root.
    pub root: PathBuf,
    pub profile: ProfileName,
    /// `<root>/<profile>`.
    pub dir: PathBuf,
}

impl ProfilePaths {
    pub fn new(root: PathBuf, profile: ProfileName) -> Self {
        let dir = root.join(profile.as_str());
        ProfilePaths { root, profile, dir }
    }

    /// Resolves the root from the environment.
    pub fn from_env(
        get_env: impl Fn(&str) -> Option<String>,
        profile: ProfileName,
    ) -> Result<Self> {
        Ok(ProfilePaths::new(data_root(get_env)?, profile))
    }

    pub fn root_config(&self) -> PathBuf {
        self.root.join(crate::config::CONFIG_FILE)
    }

    pub fn profile_config(&self) -> PathBuf {
        self.dir.join(crate::config::CONFIG_FILE)
    }

    pub fn database(&self) -> PathBuf {
        self.dir.join("memory.db")
    }

    pub fn lock_file(&self) -> PathBuf {
        self.dir.join("daemon.lock")
    }

    pub fn daemon_log(&self) -> PathBuf {
        self.dir.join("daemon.log")
    }

    pub fn metrics_dir(&self) -> PathBuf {
        self.dir.join("metrics")
    }

    /// Spool files are written here first...
    pub fn spool_tmp(&self) -> PathBuf {
        self.dir.join("spool").join("tmp")
    }

    /// ...then renamed here, where the daemon picks them up.
    pub fn spool_new(&self) -> PathBuf {
        self.dir.join("spool").join("new")
    }

    /// Spool files the daemon could not read are moved here, not deleted.
    pub fn spool_failed(&self) -> PathBuf {
        self.dir.join("spool").join("failed")
    }

    pub fn backups_dir(&self) -> PathBuf {
        self.dir.join("backups")
    }

    /// The daemon's endpoint on this platform.
    pub fn endpoint(&self) -> Endpoint {
        if cfg!(windows) {
            Endpoint::NamedPipe(pipe_name(&self.dir))
        } else {
            let runtime = std::env::var_os("XDG_RUNTIME_DIR").map(PathBuf::from);
            Endpoint::UnixSocket(unix_socket_path(&self.dir, runtime.as_deref()))
        }
    }
}

/// 16 hex characters identifying a profile directory. On Windows the path is
/// compared without regard to slash direction or case (spike result 4).
pub fn profile_hash(dir: &Path, windows: bool) -> String {
    let key = path_key(&dir.to_string_lossy(), windows);
    sha256_hex(key.as_bytes())[..16].to_owned()
}

/// `spatial-memory-<hash>`: one pipe per user and profile.
pub fn pipe_name(dir: &Path) -> String {
    format!("{APP_DIR}-{}", profile_hash(dir, true))
}

/// `<dir>/daemon.sock`, or a hashed name in the runtime folder (or `/tmp`)
/// when that path would not fit in `sun_path`.
pub fn unix_socket_path(dir: &Path, runtime_dir: Option<&Path>) -> PathBuf {
    let direct = dir.join("daemon.sock");
    if direct.as_os_str().len() <= MAX_SOCKET_PATH_BYTES {
        return direct;
    }
    let base = runtime_dir
        .filter(|p| p.has_root())
        .unwrap_or(Path::new("/tmp"));
    base.join(format!("{APP_DIR}-{}.sock", profile_hash(dir, false)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(pairs: &'static [(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> {
        move |k: &str| {
            pairs
                .iter()
                .find(|(n, _)| *n == k)
                .map(|(_, v)| (*v).to_owned())
        }
    }

    #[test]
    fn s4_home_override_replaces_the_root_and_keeps_profiles_apart() {
        let root = data_root(env(&[(HOME_ENV, "/srv/sm")])).unwrap();
        assert_eq!(root, PathBuf::from("/srv/sm"));
        let a = ProfilePaths::new(root.clone(), ProfileName::parse("a").unwrap());
        let b = ProfilePaths::new(root, ProfileName::parse("b").unwrap());
        assert_eq!(a.database(), PathBuf::from("/srv/sm/a/memory.db"));
        assert_ne!(a.database(), b.database());
        assert_ne!(a.endpoint(), b.endpoint());
        assert_eq!(a.spool_new(), PathBuf::from("/srv/sm/a/spool/new"));
        assert_eq!(a.root_config(), PathBuf::from("/srv/sm/config.toml"));
        assert_eq!(a.profile_config(), PathBuf::from("/srv/sm/a/config.toml"));
        // Without the override the platform folder is used.
        let default = data_root(env(&[(HOME_ENV, "  ")])).unwrap();
        assert!(default.ends_with(APP_DIR));
    }

    #[test]
    fn s4_model_dir_override_and_cache_layout() {
        let rev = crate::config::EMBEDDINGGEMMA_REVISION;
        let d = model_dir(
            env(&[(MODEL_DIR_ENV, "/models/gemma")]),
            "embeddinggemma-2",
            rev,
        )
        .unwrap();
        assert_eq!(d, PathBuf::from("/models/gemma"));
        let d = model_dir(env(&[]), "embeddinggemma-2", rev).unwrap();
        assert!(
            d.ends_with(
                Path::new(APP_DIR)
                    .join(MODELS_DIR)
                    .join("embeddinggemma-2")
                    .join(rev)
            )
        );
    }

    #[test]
    fn s4_platform_dirs_follow_the_xdg_rules() {
        let home = Path::new("/home/me");
        let none = env(&[]);
        assert_eq!(
            xdg_base_dir(BaseDir::DataLocal, &none, Some(home)),
            Some(PathBuf::from("/home/me/.local/share"))
        );
        assert_eq!(
            xdg_base_dir(BaseDir::Cache, &none, Some(home)),
            Some(PathBuf::from("/home/me/.cache"))
        );
        // An absolute variable wins; a relative or empty one is ignored, as
        // the XDG specification and `dirs-sys` 0.5 require.
        if cfg!(unix) {
            let set = env(&[("XDG_DATA_HOME", "/data"), ("XDG_CACHE_HOME", "/cache")]);
            assert_eq!(
                xdg_base_dir(BaseDir::DataLocal, &set, Some(home)),
                Some(PathBuf::from("/data"))
            );
            assert_eq!(
                xdg_base_dir(BaseDir::Cache, &set, Some(home)),
                Some(PathBuf::from("/cache"))
            );
        }
        let relative = env(&[("XDG_DATA_HOME", "data"), ("XDG_CACHE_HOME", "")]);
        assert_eq!(
            xdg_base_dir(BaseDir::DataLocal, &relative, Some(home)),
            Some(PathBuf::from("/home/me/.local/share"))
        );
        assert_eq!(
            xdg_base_dir(BaseDir::Cache, &relative, Some(home)),
            Some(PathBuf::from("/home/me/.cache"))
        );
        assert_eq!(xdg_base_dir(BaseDir::Cache, &none, None), None);
        // Windows: LOCALAPPDATA when absolute, else under the home folder.
        let win_home = Path::new(r"C:\Users\me");
        assert_eq!(
            windows_base_dir(&none, Some(win_home)),
            Some(win_home.join("AppData").join("Local"))
        );
        assert_eq!(
            windows_base_dir(&env(&[("LOCALAPPDATA", "Local")]), Some(win_home)),
            Some(win_home.join("AppData").join("Local"))
        );
        assert_eq!(windows_base_dir(&none, None), None);
        if cfg!(windows) {
            assert_eq!(
                windows_base_dir(&env(&[("LOCALAPPDATA", r"D:\Local")]), Some(win_home)),
                Some(PathBuf::from(r"D:\Local"))
            );
        }
        assert_eq!(
            macos_base_dir(BaseDir::DataLocal, Path::new("/Users/me")),
            PathBuf::from("/Users/me/Library/Application Support")
        );
        assert_eq!(
            macos_base_dir(BaseDir::Cache, Path::new("/Users/me")),
            PathBuf::from("/Users/me/Library/Caches")
        );
    }

    #[test]
    fn s4_platform_dirs_resolve_on_this_machine() {
        let get = |k: &str| std::env::var(k).ok();
        let home = home_dir().expect("home folder");
        assert!(home.is_absolute());
        for kind in [BaseDir::DataLocal, BaseDir::Cache] {
            let dir = base_dir(kind, &get).expect("platform folder");
            assert!(dir.is_absolute(), "{dir:?}");
            if cfg!(windows)
                && let Some(local) = std::env::var_os("LOCALAPPDATA")
            {
                let key = |p: &Path| path_key(&p.to_string_lossy(), true);
                assert_eq!(key(&dir), key(Path::new(&local)));
            }
        }
    }

    #[test]
    fn spike1_pipe_name_ignores_slash_direction_and_case() {
        let a = pipe_name(Path::new(
            r"C:\Users\Me\AppData\Local\spatial-memory\default",
        ));
        let b = pipe_name(Path::new(
            "c:/users/me/appdata/local/spatial-memory/default/",
        ));
        assert_eq!(a, b);
        assert!(a.starts_with("spatial-memory-") && a.len() == "spatial-memory-".len() + 16);
        let c = pipe_name(Path::new(r"C:\Users\Me\AppData\Local\spatial-memory\work"));
        assert_ne!(a, c);
        assert_eq!(
            Endpoint::NamedPipe(a.clone()).display_path(),
            format!(r"\\.\pipe\{a}")
        );
    }

    #[test]
    fn s3_long_socket_paths_move_to_the_runtime_dir() {
        let short = Path::new("/home/me/.local/share/spatial-memory/default");
        assert_eq!(unix_socket_path(short, None), short.join("daemon.sock"));
        let long = PathBuf::from(format!("/home/{}/spatial-memory/default", "x".repeat(90)));
        let p = unix_socket_path(&long, Some(Path::new("/run/user/1000")));
        assert!(p.starts_with("/run/user/1000"));
        assert!(p.to_string_lossy().ends_with(".sock"));
        assert_eq!(
            p,
            unix_socket_path(&long, Some(Path::new("/run/user/1000")))
        );
        let tmp = unix_socket_path(&long, Some(Path::new("relative")));
        assert!(tmp.starts_with("/tmp"));
    }
}
