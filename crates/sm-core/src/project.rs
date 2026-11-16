//! Project identity (SPEC section 9, Scope; section 15, What a project is;
//! section 16, Shell-agnostic rules).
//!
//! A project is a git repository. Its identity is the normalised remote URL,
//! or the root folder when the repository has no remote or there is no
//! repository. The remote is read from the repository's own configuration
//! files, never by running git, and worktrees (where `.git` is a file) are
//! followed to the common directory. On Windows two paths that differ only in
//! slash direction or letter case are the same project (spike result 4).

use std::fs;
use std::path::{Component, Path, PathBuf, Prefix};

use serde::{Deserialize, Serialize};

use crate::error::{CoreError, ErrorCode, Result};

/// The folder inside a repository that holds `project.toml` and `shared/`.
pub const SPATIAL_DIR: &str = ".spatial-memory";
/// The per-repository settings file (SPEC section 15, What a project is).
pub const PROJECT_TOML: &str = "project.toml";

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

/// Removes Windows' verbatim prefix that `fs::canonicalize` adds:
/// `\\?\C:\x` becomes `C:\x` and `\\?\UNC\server\share` becomes
/// `\\server\share`. Other paths are returned unchanged.
pub fn strip_verbatim(path: PathBuf) -> PathBuf {
    let Some(s) = path.to_str() else { return path };
    if let Some(rest) = s.strip_prefix(r"\\?\UNC\") {
        return PathBuf::from(format!(r"\\{rest}"));
    }
    if let Some(rest) = s.strip_prefix(r"\\?\") {
        // Only strip when what remains is an ordinary drive path.
        let b = rest.as_bytes();
        if b.len() >= 2 && b[0].is_ascii_alphabetic() && b[1] == b':' {
            return PathBuf::from(rest);
        }
    }
    path
}

/// The canonical form of an existing folder: symbolic links resolved,
/// absolute, without the verbatim prefix.
pub fn canonical_dir(path: &Path) -> Result<PathBuf> {
    let c = fs::canonicalize(path).map_err(|e| {
        CoreError::new(
            ErrorCode::ProjectUnresolved,
            format!("cannot resolve the folder {}: {e}", path.display()),
        )
    })?;
    Ok(strip_verbatim(c))
}

/// A comparison key for a path. With `windows` set, backslashes become `/`
/// and the text is lower-cased, so `C:\Repo` and `c:/repo` share a key. A
/// trailing `/` is removed except from a root (`/`, `c:/`).
pub fn path_key(path: &str, windows: bool) -> String {
    let mut s = if windows {
        let mut t = path.replace('\\', "/");
        if let Some(rest) = t.strip_prefix("//?/UNC/") {
            t = format!("//{rest}");
        } else if let Some(rest) = t.strip_prefix("//?/") {
            t = rest.to_owned();
        }
        t.to_lowercase()
    } else {
        path.to_owned()
    };
    while s.ends_with('/') && !is_root_key(&s) {
        s.pop();
    }
    s
}

fn is_root_key(s: &str) -> bool {
    if s == "/" {
        return true;
    }
    let b = s.as_bytes();
    b.len() == 3 && b[0].is_ascii_alphabetic() && b[1] == b':' && b[2] == b'/'
}

/// `path_key` for the platform this build runs on.
pub fn project_path_key(path: &Path) -> String {
    path_key(&path.to_string_lossy(), cfg!(windows))
}

/// One path component as text, with the platform's prefix forms rewritten
/// to forward slashes.
fn prefix_text(prefix: Prefix<'_>) -> String {
    match prefix {
        Prefix::Disk(d) | Prefix::VerbatimDisk(d) => format!("{}:", char::from(d)),
        Prefix::UNC(server, share) | Prefix::VerbatimUNC(server, share) => {
            format!("//{}/{}", server.to_string_lossy(), share.to_string_lossy())
        }
        Prefix::Verbatim(p) => format!("//?/{}", p.to_string_lossy()),
        Prefix::DeviceNS(p) => format!("//./{}", p.to_string_lossy()),
    }
}

/// A path split lexically into an optional prefix, a root flag and normal
/// components, with `.` removed and `..` applied. Nothing touches the disk.
#[derive(Debug, Clone, PartialEq, Eq)]
struct LexicalPath {
    prefix: Option<String>,
    rooted: bool,
    parts: Vec<String>,
}

impl LexicalPath {
    fn parse(path: &Path) -> Self {
        let mut out = LexicalPath {
            prefix: None,
            rooted: false,
            parts: Vec::new(),
        };
        for c in path.components() {
            match c {
                Component::Prefix(p) => out.prefix = Some(prefix_text(p.kind())),
                Component::RootDir => out.rooted = true,
                Component::CurDir => {}
                Component::ParentDir => {
                    if out.parts.last().is_some_and(|p| p != "..") {
                        out.parts.pop();
                    } else if !out.rooted {
                        out.parts.push("..".to_owned());
                    }
                }
                Component::Normal(n) => out.parts.push(n.to_string_lossy().into_owned()),
            }
        }
        out
    }

    fn is_absolute(&self) -> bool {
        self.rooted
    }

    fn to_slash_string(&self) -> String {
        let mut s = self.prefix.clone().unwrap_or_default();
        if self.rooted {
            s.push('/');
        }
        s.push_str(&self.parts.join("/"));
        s
    }

    /// The components of `self` after `root`, when `root` is a prefix of it.
    fn strip_root(&self, root: &LexicalPath, windows: bool) -> Option<Vec<String>> {
        let eq = |a: &str, b: &str| {
            if windows {
                a.to_lowercase() == b.to_lowercase()
            } else {
                a == b
            }
        };
        let prefix_eq = match (&self.prefix, &root.prefix) {
            (None, None) => true,
            (Some(a), Some(b)) => eq(a, b),
            _ => false,
        };
        if !prefix_eq || self.rooted != root.rooted || self.parts.len() < root.parts.len() {
            return None;
        }
        if !self.parts.iter().zip(&root.parts).all(|(a, b)| eq(a, b)) {
            return None;
        }
        Some(self.parts[root.parts.len()..].to_vec())
    }
}

/// The anchor name for a file (SPEC section 16, Type and provenance): the
/// path relative to the project root with `/` separators when the file is
/// inside the project, otherwise the absolute path with `/` separators. A
/// relative `file` is taken relative to `root`. Normalisation is lexical:
/// `.` and `..` are applied and nothing is read from disk.
pub fn normalize_file_anchor(root: &Path, file: &str) -> String {
    normalize_file_anchor_for(root, file, cfg!(windows))
}

fn normalize_file_anchor_for(root: &Path, file: &str, windows: bool) -> String {
    let file_path = Path::new(file);
    let joined = if file_path.is_absolute() || file_path.has_root() {
        file_path.to_path_buf()
    } else {
        root.join(file_path)
    };
    let f = LexicalPath::parse(&joined);
    let r = LexicalPath::parse(root);
    match f.strip_root(&r, windows) {
        Some(rel) if !rel.is_empty() && f.is_absolute() => rel.join("/"),
        _ => f.to_slash_string(),
    }
}

// ---------------------------------------------------------------------------
// Remote URLs (ported from version 1, spatial_memory/adapters/git_utils.py)
// ---------------------------------------------------------------------------

/// The parts of a git remote URL, as version 1's `ParsedRemoteURL`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedRemote {
    /// Lower-cased host.
    pub host: String,
    /// Path as written, without a trailing `.git` or `/`.
    pub path: String,
    /// `https`, `ssh` or `git`.
    pub scheme: &'static str,
    /// User before `@`; `git` by default for SSH forms, empty otherwise.
    pub user: String,
}

/// Version 1's `(?P<path>.+?)(?:\.git)?/?$`: the shortest non-empty path
/// whose remainder is `.git/`, `.git`, `/` or nothing. `.` does not match a
/// newline, so a path containing one fails.
fn lazy_path_git(rest: &str) -> Option<&str> {
    for suffix in [".git/", ".git", "/", ""] {
        if let Some(path) = rest.strip_suffix(suffix)
            && !path.is_empty()
        {
            return (!path.contains('\n')).then_some(path);
        }
    }
    None
}

/// Version 1's Azure tail `(?P<path>.+?)/?$`.
fn lazy_path_slash(rest: &str) -> Option<&str> {
    for suffix in ["/", ""] {
        if let Some(path) = rest.strip_suffix(suffix)
            && !path.is_empty()
        {
            return (!path.contains('\n')).then_some(path);
        }
    }
    None
}

/// `(?P<host>[^/:]+)(?::(?P<port>\d+))?/` followed by the lazy path.
/// Returns `(host, path)`.
fn host_port_path(s: &str) -> Option<(&str, &str)> {
    let end = s.find(['/', ':']).unwrap_or(s.len());
    if end == 0 || end == s.len() {
        return None;
    }
    let host = &s[..end];
    let mut rest = &s[end..];
    if let Some(after_colon) = rest.strip_prefix(':') {
        let digits = after_colon.bytes().take_while(u8::is_ascii_digit).count();
        if digits == 0 {
            return None;
        }
        rest = &after_colon[digits..];
    }
    let rest = rest.strip_prefix('/')?;
    Some((host, lazy_path_git(rest)?))
}

/// `(?:(?P<user>[^@]+)@)?` then `host_port_path`, trying with a user first.
fn user_host_port_path(s: &str) -> Option<(Option<&str>, &str, &str)> {
    if let Some(at) = s.find('@')
        && at > 0
        && let Some((host, path)) = host_port_path(&s[at + 1..])
    {
        return Some((Some(&s[..at]), host, path));
    }
    host_port_path(s).map(|(h, p)| (None, h, p))
}

fn strip_http_scheme(url: &str) -> Option<&str> {
    url.strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))
}

fn parse_azure(url: &str) -> Option<ParsedRemote> {
    let s = strip_http_scheme(url)?;
    let (host, rest) = if let Some(rest) = s.strip_prefix("dev.azure.com/") {
        ("dev.azure.com", rest)
    } else {
        // `[^.]+\.visualstudio\.com/`: the text up to the first dot, then
        // the literal domain.
        let dot = s.find('.')?;
        if dot == 0 {
            return None;
        }
        let rest = s[dot..].strip_prefix(".visualstudio.com/")?;
        (&s[..dot + ".visualstudio.com".len()], rest)
    };
    let path = lazy_path_slash(rest)?;
    Some(ParsedRemote {
        host: host.to_lowercase(),
        path: path.replace("/_git/", "/"),
        scheme: "https",
        user: String::new(),
    })
}

fn parse_scp(url: &str) -> Option<ParsedRemote> {
    let attempt = |user: Option<&str>, s: &str| -> Option<ParsedRemote> {
        let colon = s.find(':')?;
        if colon == 0 {
            return None;
        }
        let rest = &s[colon + 1..];
        if rest.starts_with('/') {
            return None;
        }
        let path = lazy_path_git(rest)?;
        Some(ParsedRemote {
            host: s[..colon].to_lowercase(),
            path: path.to_owned(),
            scheme: "ssh",
            user: user.unwrap_or("git").to_owned(),
        })
    };
    if let Some(at) = url.find('@')
        && at > 0
        && let Some(p) = attempt(Some(&url[..at]), &url[at + 1..])
    {
        return Some(p);
    }
    attempt(None, url)
}

/// Parses a git remote URL exactly as version 1 did: Azure DevOps first,
/// then HTTPS, `ssh://`, SCP-like (`git@host:org/repo`) and `git://`. The
/// URL is stripped of surrounding whitespace first. `None` when no form
/// matches (a local path, for example).
pub fn parse_remote_url(url: &str) -> Option<ParsedRemote> {
    let url = crate::hash::python_strip(url);
    if let Some(p) = parse_azure(url) {
        return Some(p);
    }
    if let Some(s) = strip_http_scheme(url)
        && let Some((user, host, path)) = user_host_port_path(s)
    {
        return Some(ParsedRemote {
            host: host.to_lowercase(),
            path: path.to_owned(),
            scheme: "https",
            user: user.unwrap_or_default().to_owned(),
        });
    }
    if let Some(s) = url.strip_prefix("ssh://")
        && let Some((user, host, path)) = user_host_port_path(s)
    {
        return Some(ParsedRemote {
            host: host.to_lowercase(),
            path: path.to_owned(),
            scheme: "ssh",
            user: user.unwrap_or("git").to_owned(),
        });
    }
    if let Some(p) = parse_scp(url) {
        return Some(p);
    }
    if let Some(s) = url.strip_prefix("git://")
        && let Some((host, path)) = host_port_path(s)
    {
        return Some(ParsedRemote {
            host: host.to_lowercase(),
            path: path.to_owned(),
            scheme: "git",
            user: String::new(),
        });
    }
    None
}

/// Version 1's `normalize_project_id`: `host/path` with trailing slashes
/// and one `.git` removed. The host is lower-case; the path keeps its case.
pub fn normalize_remote_url(url: &str) -> Option<String> {
    let parsed = parse_remote_url(url)?;
    let mut path = parsed.path.trim_end_matches('/');
    if let Some(p) = path.strip_suffix(".git") {
        path = p;
    }
    Some(format!("{}/{}", parsed.host, path))
}

// ---------------------------------------------------------------------------
// Git configuration
// ---------------------------------------------------------------------------

/// Reads one git configuration value: quotes removed, escapes applied, and a
/// `#` or `;` comment outside quotes dropped.
fn git_config_value(raw: &str) -> String {
    let mut out = String::new();
    let mut in_quote = false;
    let mut pending_space = String::new();
    let mut chars = raw.trim_start().chars();
    while let Some(c) = chars.next() {
        match c {
            '"' => in_quote = !in_quote,
            '\\' => {
                out.push_str(&pending_space);
                pending_space.clear();
                match chars.next() {
                    Some('n') => out.push('\n'),
                    Some('t') => out.push('\t'),
                    Some('b') => {
                        out.pop();
                    }
                    Some(other) => out.push(other),
                    None => {}
                }
            }
            '#' | ';' if !in_quote => break,
            c if c.is_whitespace() && !in_quote => pending_space.push(c),
            c => {
                out.push_str(&pending_space);
                pending_space.clear();
                out.push(c);
            }
        }
    }
    out
}

/// Every `remote.<name>.url` in a git configuration file, in the order the
/// remotes first appear. A remote with several `url` lines keeps the first,
/// which is the one git fetches from.
pub fn parse_git_config_remotes(text: &str) -> Vec<(String, String)> {
    let mut remotes: Vec<(String, String)> = Vec::new();
    let mut section = String::new();
    let mut subsection: Option<String> = None;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with(';') {
            continue;
        }
        if let Some(inner) = line.strip_prefix('[') {
            let Some(end) = inner.rfind(']') else {
                continue;
            };
            let inner = &inner[..end];
            if let Some(q) = inner.find('"') {
                section = inner[..q].trim().to_lowercase();
                let sub = &inner[q + 1..];
                let sub = sub.strip_suffix('"').unwrap_or(sub);
                subsection = Some(sub.replace("\\\"", "\"").replace("\\\\", "\\"));
            } else if let Some((s, sub)) = inner.split_once('.') {
                section = s.trim().to_lowercase();
                subsection = Some(sub.trim().to_lowercase());
            } else {
                section = inner.trim().to_lowercase();
                subsection = None;
            }
            continue;
        }
        if section != "remote" {
            continue;
        }
        let Some(name) = &subsection else { continue };
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        if !key.trim().eq_ignore_ascii_case("url") {
            continue;
        }
        let value = git_config_value(value);
        if value.is_empty() || remotes.iter().any(|(n, _)| n == name) {
            continue;
        }
        remotes.push((name.clone(), value));
    }
    remotes
}

/// Version 1's remote priority: `upstream`, then `origin`, then the first.
pub fn select_remote(remotes: &[(String, String)]) -> Option<&str> {
    for preferred in ["upstream", "origin"] {
        if let Some((_, url)) = remotes.iter().find(|(n, _)| n == preferred) {
            return Some(url);
        }
    }
    remotes.first().map(|(_, u)| u.as_str())
}

/// A git repository found on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GitRepo {
    /// The working-tree root: the folder that contains `.git`.
    pub root: PathBuf,
    /// The git directory for this working tree (`.git`, or
    /// `<main>/.git/worktrees/<name>` for a worktree).
    pub git_dir: PathBuf,
    /// The directory holding `config`, `refs` and `packed-refs`; the same as
    /// `git_dir` except in a worktree.
    pub common_dir: PathBuf,
}

impl GitRepo {
    /// Opens the repository whose working tree is `root`, following a `.git`
    /// file (`gitdir: <path>`) and a `commondir` file. `None` when `root`
    /// has no `.git` entry or it points nowhere.
    pub fn open(root: &Path) -> Option<GitRepo> {
        let dot_git = root.join(".git");
        let meta = fs::metadata(&dot_git).ok()?;
        let git_dir = if meta.is_dir() {
            dot_git
        } else {
            let text = fs::read_to_string(&dot_git).ok()?;
            let target = text.trim().strip_prefix("gitdir:")?.trim();
            let target = Path::new(target);
            let resolved = if target.is_absolute() {
                target.to_path_buf()
            } else {
                root.join(target)
            };
            let resolved = strip_verbatim(fs::canonicalize(&resolved).ok()?);
            if !resolved.is_dir() {
                return None;
            }
            resolved
        };
        let common_dir = match fs::read_to_string(git_dir.join("commondir")) {
            Ok(text) => {
                let rel = Path::new(text.trim());
                let joined = if rel.is_absolute() {
                    rel.to_path_buf()
                } else {
                    git_dir.join(rel)
                };
                strip_verbatim(fs::canonicalize(&joined).unwrap_or(joined))
            }
            Err(_) => git_dir.clone(),
        };
        Some(GitRepo {
            root: root.to_path_buf(),
            git_dir,
            common_dir,
        })
    }

    /// Walks up from `start` to the nearest folder with a `.git` entry.
    pub fn discover(start: &Path) -> Option<GitRepo> {
        start.ancestors().find_map(GitRepo::open)
    }

    /// The preferred remote URL as written in the configuration, if any.
    pub fn remote_url(&self) -> Option<String> {
        let text = fs::read_to_string(self.common_dir.join("config")).ok()?;
        let remotes = parse_git_config_remotes(&text);
        select_remote(&remotes).map(str::to_owned)
    }

    /// The commit `HEAD` points at, as hex, read from `HEAD`, the loose ref
    /// and `packed-refs`. `None` for an unborn branch or an unreadable ref.
    pub fn head_commit(&self) -> Option<String> {
        let head = fs::read_to_string(self.git_dir.join("HEAD")).ok()?;
        let head = head.trim();
        let Some(reference) = head.strip_prefix("ref:") else {
            return is_object_id(head).then(|| head.to_ascii_lowercase());
        };
        let reference = reference.trim();
        for dir in [&self.git_dir, &self.common_dir] {
            if let Ok(text) = fs::read_to_string(dir.join(reference)) {
                let id = text.trim();
                if is_object_id(id) {
                    return Some(id.to_ascii_lowercase());
                }
            }
        }
        let packed = fs::read_to_string(self.common_dir.join("packed-refs")).ok()?;
        packed.lines().find_map(|line| {
            let (id, name) = line.split_once(' ')?;
            (name.trim() == reference && is_object_id(id)).then(|| id.to_ascii_lowercase())
        })
    }
}

/// A SHA-1 (40) or SHA-256 (64) object id in hex.
pub fn is_object_id(s: &str) -> bool {
    (s.len() == 40 || s.len() == 64) && s.bytes().all(|b| b.is_ascii_hexdigit())
}

// ---------------------------------------------------------------------------
// project.toml and resolution
// ---------------------------------------------------------------------------

/// `.spatial-memory/project.toml`. Unknown keys are ignored so a newer
/// file does not break an older build.
///
/// ```toml
/// name = "order-api"
///
/// [capture]
/// enabled = false
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectToml {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub capture: ProjectCapture,
}

/// The `[capture]` table of `project.toml`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectCapture {
    /// `false` opts the repository out of automatic capture.
    #[serde(default)]
    pub enabled: Option<bool>,
}

impl ProjectToml {
    pub fn parse(text: &str) -> Result<Self> {
        toml::from_str(text)
            .map_err(|e| CoreError::config_invalid(format!("{PROJECT_TOML}: {}", e.message())))
    }
}

/// Where the working directory resolved to. The daemon caches this per
/// working directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectIdentity {
    /// Canonical project root.
    pub root: PathBuf,
    /// `root` written with `/` separators, for display and storage.
    pub root_path: String,
    /// `path_key` of the root, for comparison.
    pub root_key: String,
    /// Normalised remote (`host/path`), when the repository has one that
    /// parses.
    pub git_remote: Option<String>,
    /// The repository, when there is one.
    pub repo: Option<GitRepo>,
    /// Display name: `project.toml`'s `name`, else the remote's last path
    /// segment, else the root folder's name.
    pub name: String,
    /// `false` when `project.toml` opts out of capture, or cannot be read
    /// (the conservative reading of a broken opt-out file).
    pub capture_enabled: bool,
    /// Problems worth reporting that did not stop resolution.
    pub warnings: Vec<String>,
}

impl ProjectIdentity {
    /// The unique identity of the project: `git:<remote>` when there is a
    /// remote, else `path:<root key>`. Stores look projects up by this.
    pub fn key(&self) -> String {
        match &self.git_remote {
            Some(r) => format!("git:{r}"),
            None => format!("path:{}", self.root_key),
        }
    }
}

/// The default display name for a project.
pub fn default_project_name(git_remote: Option<&str>, root: &Path) -> String {
    if let Some(remote) = git_remote
        && let Some(last) = remote.rsplit('/').find(|s| !s.is_empty())
    {
        return last.to_owned();
    }
    root.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .filter(|n| !n.is_empty())
        .unwrap_or_else(|| LexicalPath::parse(root).to_slash_string())
}

/// Resolves the project for a working directory: the nearest enclosing git
/// repository, else the nearest folder holding `.spatial-memory/project.toml`,
/// else the folder itself. The user's home folder and a filesystem root are
/// not projects (`project_unresolved`), as in version 1.
pub fn resolve_project(cwd: &Path) -> Result<ProjectIdentity> {
    let cwd = canonical_dir(cwd)?;
    let repo = GitRepo::discover(&cwd);
    let root = match &repo {
        Some(r) => r.root.clone(),
        None => cwd
            .ancestors()
            .find(|a| a.join(SPATIAL_DIR).join(PROJECT_TOML).is_file())
            .map(Path::to_path_buf)
            .unwrap_or_else(|| cwd.clone()),
    };
    if repo.is_none() && is_too_broad(&root) {
        return Err(CoreError::new(
            ErrorCode::ProjectUnresolved,
            format!("{} is not inside a project folder", root.display()),
        ));
    }
    let git_remote = repo
        .as_ref()
        .and_then(GitRepo::remote_url)
        .and_then(|u| normalize_remote_url(&u));
    let mut warnings = Vec::new();
    let mut capture_enabled = true;
    let mut name = None;
    let toml_path = root.join(SPATIAL_DIR).join(PROJECT_TOML);
    match fs::read_to_string(&toml_path) {
        Ok(text) => match ProjectToml::parse(&text) {
            Ok(pt) => {
                capture_enabled = pt.capture.enabled.unwrap_or(true);
                name = pt
                    .name
                    .map(|n| n.trim().to_owned())
                    .filter(|n| !n.is_empty());
            }
            Err(e) => {
                capture_enabled = false;
                warnings.push(format!(
                    "{}: {}; capture is off for this project until it parses",
                    toml_path.display(),
                    e.message
                ));
            }
        },
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => {
            capture_enabled = false;
            warnings.push(format!(
                "{}: {e}; capture is off for this project",
                toml_path.display()
            ));
        }
    }
    let name = name.unwrap_or_else(|| default_project_name(git_remote.as_deref(), &root));
    Ok(ProjectIdentity {
        root_path: LexicalPath::parse(&root).to_slash_string(),
        root_key: project_path_key(&root),
        git_remote,
        repo,
        name,
        capture_enabled,
        warnings,
        root,
    })
}

fn is_too_broad(root: &Path) -> bool {
    if root.parent().is_none() {
        return true;
    }
    let Some(home) = crate::paths::home_dir() else {
        return false;
    };
    let home = fs::canonicalize(&home).map(strip_verbatim).unwrap_or(home);
    project_path_key(&home) == project_path_key(root)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn norm(url: &str) -> Option<String> {
        normalize_remote_url(url)
    }

    #[test]
    fn s9_remote_forms_normalise_like_version_1() {
        let cases = [
            (
                "https://github.com/org/repo.git",
                Some("github.com/org/repo"),
            ),
            ("https://github.com/org/repo", Some("github.com/org/repo")),
            ("https://github.com/org/repo/", Some("github.com/org/repo")),
            (
                "https://GitHub.com/Org/Repo.git",
                Some("github.com/Org/Repo"),
            ),
            ("http://host:8080/org/repo.git", Some("host/org/repo")),
            (
                "https://user:token@github.com/org/repo.git",
                Some("github.com/org/repo"),
            ),
            (
                "ssh://git@github.com/org/repo.git",
                Some("github.com/org/repo"),
            ),
            (
                "ssh://git@github.com:22/org/repo.git",
                Some("github.com/org/repo"),
            ),
            ("git@github.com:org/repo.git", Some("github.com/org/repo")),
            ("github.com:org/repo", Some("github.com/org/repo")),
            ("git://github.com/org/repo.git", Some("github.com/org/repo")),
            (
                "  git@github.com:org/repo.git\n",
                Some("github.com/org/repo"),
            ),
            (
                "https://dev.azure.com/org/proj/_git/repo",
                Some("dev.azure.com/org/proj/repo"),
            ),
            (
                "https://org.visualstudio.com/proj/_git/repo/",
                Some("org.visualstudio.com/proj/repo"),
            ),
            ("/srv/git/repo.git", None),
            ("file:///srv/git/repo.git", None),
            ("C:/repos/x.git", None),
            ("", None),
        ];
        for (url, expected) in cases {
            assert_eq!(norm(url).as_deref(), expected, "{url:?}");
        }
    }

    #[test]
    fn s9_remote_quirks_are_kept_from_version_1() {
        // A user before dev.azure.com misses the Azure pattern and keeps
        // `_git`, exactly as version 1 did.
        assert_eq!(
            norm("https://org@dev.azure.com/org/proj/_git/repo").as_deref(),
            Some("dev.azure.com/org/proj/_git/repo")
        );
        // A user before visualstudio.com is swallowed into the host.
        assert_eq!(
            norm("https://user@org.visualstudio.com/proj/_git/repo").as_deref(),
            Some("user@org.visualstudio.com/proj/repo")
        );
        // A non-numeric port fails the URL forms and nothing else matches.
        assert_eq!(norm("https://host:abc/org/repo"), None);
        // An '@' after the host is part of the path.
        assert_eq!(norm("https://host/a@b").as_deref(), Some("host/a@b"));
        // A Windows path with backslashes looks SCP-like, as in version 1.
        assert_eq!(norm(r"C:\repos\x.git").as_deref(), Some(r"c/\repos\x"));
        // The parser removes one `.git` and the normaliser a second, as
        // version 1 does (checked against master's git_utils.py).
        assert_eq!(
            parse_remote_url("https://h/o/r.git.git").unwrap().path,
            "o/r.git"
        );
        assert_eq!(norm("https://h/o/r.git.git").as_deref(), Some("h/o/r"));
        assert_eq!(
            norm("https://h/o/r.git.git.git").as_deref(),
            Some("h/o/r.git")
        );
        let p = parse_remote_url("ssh://h/o/r").unwrap();
        assert_eq!((p.scheme, p.user.as_str()), ("ssh", "git"));
    }

    #[test]
    fn s9_git_config_remote_priority_is_upstream_origin_first() {
        let text = r#"
[core]
	repositoryformatversion = 0
[remote "fork"]
	url = git@github.com:me/repo.git
	fetch = +refs/heads/*:refs/remotes/fork/*
	fetch = +refs/tags/*:refs/tags/*
[remote "origin"]
	url = "https://github.com/org/repo.git" ; a comment
[branch "main"]
	remote = origin
"#;
        let remotes = parse_git_config_remotes(text);
        assert_eq!(remotes.len(), 2);
        assert_eq!(
            remotes[1],
            (
                "origin".to_owned(),
                "https://github.com/org/repo.git".to_owned()
            )
        );
        assert_eq!(
            select_remote(&remotes),
            Some("https://github.com/org/repo.git")
        );
        let with_upstream = format!("{text}\n[remote \"upstream\"]\n\turl = git@h:u/r.git\n");
        assert_eq!(
            select_remote(&parse_git_config_remotes(&with_upstream)),
            Some("git@h:u/r.git")
        );
        let only_fork = "[remote \"fork\"]\nurl = a:b/c\n";
        assert_eq!(
            select_remote(&parse_git_config_remotes(only_fork)),
            Some("a:b/c")
        );
        assert_eq!(select_remote(&[]), None);
    }

    #[test]
    fn spike4_windows_paths_differing_in_slash_or_case_share_a_key() {
        assert_eq!(path_key(r"C:\Users\Me\Repo\", true), "c:/users/me/repo");
        assert_eq!(path_key("c:/users/me/repo", true), "c:/users/me/repo");
        assert_eq!(path_key(r"\\?\C:\Repo", true), "c:/repo");
        assert_eq!(path_key(r"\\?\UNC\srv\share\x", true), "//srv/share/x");
        assert_eq!(path_key("C:/", true), "c:/");
        assert_eq!(path_key("/home/Me/Repo/", false), "/home/Me/Repo");
        assert_eq!(path_key("/", false), "/");
        assert_ne!(
            path_key("/home/me/repo", false),
            path_key("/home/Me/Repo", false)
        );
    }

    #[test]
    fn spike4_strip_verbatim_removes_only_drive_and_unc_prefixes() {
        assert_eq!(
            strip_verbatim(PathBuf::from(r"\\?\C:\x")),
            PathBuf::from(r"C:\x")
        );
        assert_eq!(
            strip_verbatim(PathBuf::from(r"\\?\UNC\s\sh")),
            PathBuf::from(r"\\s\sh")
        );
        assert_eq!(
            strip_verbatim(PathBuf::from(r"\\?\Volume{x}\a")),
            PathBuf::from(r"\\?\Volume{x}\a")
        );
        assert_eq!(
            strip_verbatim(PathBuf::from("/tmp/x")),
            PathBuf::from("/tmp/x")
        );
    }

    #[test]
    fn s16_file_anchors_are_repository_relative() {
        let dir = tempfile::tempdir().unwrap();
        let root = canonical_dir(dir.path()).unwrap();
        let inside = root.join("src").join("lib.rs");
        assert_eq!(
            normalize_file_anchor(&root, &inside.to_string_lossy()),
            "src/lib.rs"
        );
        assert_eq!(
            normalize_file_anchor(&root, "src/./x/../lib.rs"),
            "src/lib.rs"
        );
        let outside = root.parent().unwrap().join("other.txt");
        let a = normalize_file_anchor(&root, &outside.to_string_lossy());
        assert!(a.ends_with("/other.txt") && !a.contains('\\'), "{a}");
        if cfg!(windows) {
            let upper = inside.to_string_lossy().to_uppercase().replace('\\', "/");
            assert_eq!(normalize_file_anchor(&root, &upper), "SRC/LIB.RS");
        }
    }

    #[test]
    fn s16_worktree_git_file_is_followed_to_the_common_dir() {
        let dir = tempfile::tempdir().unwrap();
        let base = canonical_dir(dir.path()).unwrap();
        let main = base.join("main");
        let main_git = main.join(".git");
        let wt_git = main_git.join("worktrees").join("feature");
        fs::create_dir_all(&wt_git).unwrap();
        fs::create_dir_all(main_git.join("refs").join("heads")).unwrap();
        fs::write(
            main_git.join("config"),
            "[remote \"origin\"]\n\turl = git@github.com:org/app.git\n",
        )
        .unwrap();
        let commit = "0123456789abcdef0123456789abcdef01234567";
        fs::write(main_git.join("refs/heads/feature"), format!("{commit}\n")).unwrap();
        fs::write(wt_git.join("HEAD"), "ref: refs/heads/feature\n").unwrap();
        fs::write(wt_git.join("commondir"), "../..\n").unwrap();
        let worktree = base.join("feature-wt");
        fs::create_dir_all(worktree.join("sub")).unwrap();
        fs::write(
            worktree.join(".git"),
            format!("gitdir: {}\n", wt_git.display()),
        )
        .unwrap();

        let id = resolve_project(&worktree.join("sub")).unwrap();
        assert_eq!(id.git_remote.as_deref(), Some("github.com/org/app"));
        assert_eq!(id.key(), "git:github.com/org/app");
        assert_eq!(id.name, "app");
        assert_eq!(id.root, worktree);
        let repo = id.repo.unwrap();
        assert_eq!(repo.head_commit().as_deref(), Some(commit));
        assert_eq!(
            project_path_key(&repo.common_dir),
            project_path_key(&main_git)
        );
    }

    #[test]
    fn s16_head_commit_reads_packed_refs_and_detached_heads() {
        let dir = tempfile::tempdir().unwrap();
        let root = canonical_dir(dir.path()).unwrap();
        let git = root.join(".git");
        fs::create_dir_all(&git).unwrap();
        let a = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        fs::write(git.join("HEAD"), "ref: refs/heads/main\n").unwrap();
        fs::write(
            git.join("packed-refs"),
            format!("# pack-refs with: peeled\n{a} refs/heads/main\n"),
        )
        .unwrap();
        let repo = GitRepo::open(&root).unwrap();
        assert_eq!(repo.head_commit().as_deref(), Some(a));
        assert_eq!(repo.remote_url(), None);
        let b = "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB";
        fs::write(git.join("HEAD"), format!("{b}\n")).unwrap();
        assert_eq!(repo.head_commit(), Some(b.to_ascii_lowercase()));
        fs::write(git.join("HEAD"), "ref: refs/heads/unborn\n").unwrap();
        assert_eq!(repo.head_commit(), None);
    }

    #[test]
    fn s9_folder_without_repository_is_identified_by_path() {
        let dir = tempfile::tempdir().unwrap();
        let root = canonical_dir(dir.path()).unwrap().join("Plain Folder");
        fs::create_dir_all(root.join("deep")).unwrap();
        fs::create_dir_all(root.join(SPATIAL_DIR)).unwrap();
        fs::write(
            root.join(SPATIAL_DIR).join(PROJECT_TOML),
            "name = \"Orders\"\nfuture_key = 1\n",
        )
        .unwrap();
        let id = resolve_project(&root.join("deep")).unwrap();
        assert_eq!(id.root, root);
        assert_eq!(id.git_remote, None);
        assert_eq!(id.name, "Orders");
        assert!(id.capture_enabled);
        assert!(id.key().starts_with("path:"));
        assert_eq!(id.key(), format!("path:{}", project_path_key(&root)));
    }

    #[test]
    fn s16_project_toml_opts_out_of_capture_and_broken_file_is_conservative() {
        let dir = tempfile::tempdir().unwrap();
        let root = canonical_dir(dir.path()).unwrap();
        fs::create_dir_all(root.join(".git")).unwrap();
        fs::create_dir_all(root.join(SPATIAL_DIR)).unwrap();
        let toml_path = root.join(SPATIAL_DIR).join(PROJECT_TOML);
        fs::write(&toml_path, "[capture]\nenabled = false\n").unwrap();
        let id = resolve_project(&root).unwrap();
        assert!(!id.capture_enabled);
        assert!(id.warnings.is_empty());
        fs::write(&toml_path, "name = [unclosed\n").unwrap();
        let id = resolve_project(&root).unwrap();
        assert!(!id.capture_enabled);
        assert_eq!(id.warnings.len(), 1);
        assert_eq!(id.name, default_project_name(None, &root));
    }

    #[test]
    fn s9_filesystem_root_is_not_a_project() {
        let root = std::env::temp_dir();
        let top = root.ancestors().last().unwrap();
        if GitRepo::open(top).is_none() {
            let e = resolve_project(top).unwrap_err();
            assert_eq!(e.code, ErrorCode::ProjectUnresolved);
        }
    }

    #[test]
    fn s9_default_name_prefers_remote_then_folder() {
        assert_eq!(
            default_project_name(Some("github.com/org/repo"), Path::new("/x/y")),
            "repo"
        );
        assert_eq!(default_project_name(None, Path::new("/x/my-app")), "my-app");
    }
}
