//! Profile resolution shared by every subcommand (SPEC section 9, Profiles).
//!
//! The environment is a parameter so tests never read the user's variables
//! or the real data folder.

use std::path::{Path, PathBuf};

use sm_core::config::{self, ProfileSource, RootConfig, Settings};
use sm_core::error::CoreError;
use sm_core::paths::{self, ProfilePaths};

/// Reads one environment variable. The binary passes `std::env::var`;
/// tests pass a map.
pub type GetEnv<'a> = &'a dyn Fn(&str) -> Option<String>;

/// The real environment, with non-Unicode values treated as unset.
pub fn process_env(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

/// A resolved profile: its folders, what selected it, and the root
/// configuration it was read from.
#[derive(Debug, Clone)]
pub struct Resolved {
    pub paths: ProfilePaths,
    pub source: ProfileSource,
    pub root_config: RootConfig,
}

impl Resolved {
    /// The profile's validated settings (root file, then profile file).
    pub fn settings(&self) -> Result<Settings, CoreError> {
        config::load_settings(&self.paths.root, &self.root_config, &self.paths.profile)
    }
}

/// `--profile`, then the plugin option, then `SPATIAL_MEMORY_PROFILE`, then
/// the longest `[[profile_rule]]` containing `cwd`, then `default`.
pub fn resolve(flag: Option<&str>, get_env: GetEnv, cwd: &Path) -> Result<Resolved, CoreError> {
    let root = paths::data_root(get_env)?;
    let root_config = config::load_root_config(&root)?;
    let (profile, source) =
        config::resolve_profile(flag, get_env, Some(cwd), &root_config.profile_rules)?;
    Ok(Resolved {
        paths: ProfilePaths::new(root, profile),
        source,
        root_config,
    })
}

/// A path made absolute against `cwd` before it is sent to the daemon,
/// which has its own working directory (stream D: paths are made absolute).
pub fn absolute(path: &Path, cwd: &Path) -> PathBuf {
    if path.is_absolute() {
        return path.to_path_buf();
    }
    let joined = cwd.join(path);
    std::path::absolute(&joined).unwrap_or(joined)
}

/// What the profile source is called in `doctor` output.
pub fn source_name(source: ProfileSource) -> &'static str {
    match source {
        ProfileSource::Flag => "--profile",
        ProfileSource::PluginOption => config::PLUGIN_PROFILE_ENV,
        ProfileSource::Environment => config::PROFILE_ENV,
        ProfileSource::DirectoryRule => "a [[profile_rule]] in config.toml",
        ProfileSource::Default => "the default",
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    fn env(pairs: Vec<(&'static str, String)>) -> impl Fn(&str) -> Option<String> {
        let map: HashMap<&str, String> = pairs.into_iter().collect();
        move |k: &str| map.get(k).cloned()
    }

    #[test]
    fn s9_profile_flag_then_environment_then_rule_then_default() {
        let tmp = tempfile::tempdir().unwrap();
        let home = tmp.path().to_string_lossy().into_owned();
        let project = tmp.path().join("work").join("client-a");
        std::fs::write(
            tmp.path().join("config.toml"),
            format!(
                "[[profile_rule]]\npath = {:?}\nprofile = \"client-a\"\n",
                tmp.path().join("work").join("client-a").to_string_lossy()
            ),
        )
        .unwrap();

        let get = env(vec![(paths::HOME_ENV, home.clone())]);
        let r = resolve(Some("flagged"), &get, &project).unwrap();
        assert_eq!(r.paths.profile.as_str(), "flagged");
        assert_eq!(r.source, ProfileSource::Flag);

        let get = env(vec![
            (paths::HOME_ENV, home.clone()),
            (config::PROFILE_ENV, "envd".into()),
        ]);
        let r = resolve(None, &get, &project).unwrap();
        assert_eq!(r.paths.profile.as_str(), "envd");

        let get = env(vec![(paths::HOME_ENV, home.clone())]);
        let r = resolve(None, &get, &project).unwrap();
        assert_eq!(r.paths.profile.as_str(), "client-a");
        assert_eq!(r.source, ProfileSource::DirectoryRule);
        assert_eq!(r.paths.dir, tmp.path().join("client-a"));

        let r = resolve(None, &get, tmp.path()).unwrap();
        assert_eq!(r.paths.profile.as_str(), "default");
        assert_eq!(r.settings().unwrap(), Settings::default());
    }

    #[test]
    fn s9_invalid_profile_is_an_error_not_a_fallback() {
        let tmp = tempfile::tempdir().unwrap();
        let get = env(vec![(paths::HOME_ENV, tmp.path().to_string_lossy().into())]);
        assert!(resolve(Some("Not Valid"), &get, tmp.path()).is_err());
    }

    #[test]
    fn s9_relative_paths_are_made_absolute() {
        let tmp = tempfile::tempdir().unwrap();
        let p = absolute(Path::new("out/x.jsonl"), tmp.path());
        assert!(p.is_absolute());
        assert!(p.starts_with(tmp.path()));
        let abs = tmp.path().join("y");
        assert_eq!(absolute(&abs, Path::new("/elsewhere")), abs);
    }
}
