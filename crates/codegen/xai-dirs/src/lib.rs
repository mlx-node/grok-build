//! Home-directory resolution generally: USERPROFILE-first `home_dir`, plus
//! the agent home (`$GROK_HOME` or `<home>/.mlx-agent`) and the project agent
//! folder (`<project>/.mlx-agent`). Shared by `xai-grok-config` and
//! `xai-fast-worktree`.
//!
//! Which function to call:
//! - [`grok_home`]: the usual choice, a cached, created path to build on.
//! - [`user_grok_home`]: `None` instead of a cwd fallback when no home resolves.
//! - [`default_grok_home`]: the `<home>/.mlx-agent` default, ignoring `$GROK_HOME`, so callers can detect an override.
//! - [`resolve_grok_home`]: a fresh, uncached resolve.
//! - [`resolve_grok_home_with_source`]: [`resolve_grok_home`] plus where the path came from.
//! - [`home_dir`]: the home directory itself, for sibling dot dirs (`~/.claude`, `~/.agents`, ...).
//! - [`project_agent_dir`]: the agent folder of a project, `None` where that folder would be the agent home.
//! - [`ensure_project_agent_dir`]: the same folder, created with a `.gitignore` of `*` when the agent makes it.
//!
//! The agent never reads or writes grok-build's `~/.grok` or a project's `.grok/`.
//!
//! TODO: collapse these getters by threading the path through config as an
//! explicit value.

#![deny(clippy::indexing_slicing)]

use std::ffi::OsStr;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// Name of the agent's own folder, both the home (`~/.mlx-agent`) and a project's (`<project>/.mlx-agent`).
pub const AGENT_DIR_NAME: &str = ".mlx-agent";

/// What the agent writes as `.gitignore` in a project folder it creates, so the folder stays out of git.
const PROJECT_AGENT_GITIGNORE: &[u8] = b"*\n";

/// Where a resolved grok home came from, so "why did grok pick this
/// directory?" is answerable in diagnostics without re-reading the
/// environment at the asking site.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrokHomeSource {
    /// A non-empty `$GROK_HOME` override.
    EnvOverride,
    /// `<home>/.mlx-agent` derived from the home directory.
    HomeDefault,
}

/// The user's home directory via [`std::env::home_dir`]: `HOME` on Unix, `USERPROFILE` on Windows.
/// Not `dirs::home_dir()`: on Windows `dirs` ignores a redirected `USERPROFILE`.
/// Every home-anchored path must come from this one function.
#[allow(deprecated, clippy::disallowed_methods)] // the one sanctioned std::env::home_dir call
pub fn home_dir() -> Option<PathBuf> {
    std::env::home_dir()
}

/// `<home>/.mlx-agent`, canonicalized via `dunce` (not `std::fs::canonicalize`,
/// which yields Windows `\\?\` verbatim paths).
fn grok_home_in(home: &Path) -> PathBuf {
    dunce::canonicalize(home)
        .unwrap_or_else(|_| home.to_path_buf())
        .join(AGENT_DIR_NAME)
}

/// `$GROK_HOME` verbatim when non-empty, else `<home>/.mlx-agent`.
/// Used as-is (not canonicalized) so literal prefix checks and symlink guards still see original components.
fn resolve_grok_home_from(
    grok_home_env: Option<&OsStr>,
    os_home: Option<&Path>,
) -> Option<(PathBuf, GrokHomeSource)> {
    if let Some(env) = grok_home_env.filter(|env| !env.is_empty()) {
        return Some((PathBuf::from(env), GrokHomeSource::EnvOverride));
    }
    os_home.map(|home| (grok_home_in(home), GrokHomeSource::HomeDefault))
}

/// Resolve the grok home from the environment (fresh, no cache); `None` if neither resolves.
pub fn resolve_grok_home() -> Option<PathBuf> {
    resolve_grok_home_with_source().map(|(home, _)| home)
}

/// [`resolve_grok_home`] plus the [`GrokHomeSource`] the path came from.
pub fn resolve_grok_home_with_source() -> Option<(PathBuf, GrokHomeSource)> {
    resolve_grok_home_from(
        std::env::var_os("GROK_HOME").as_deref(),
        home_dir().as_deref(),
    )
}

/// The default `<home>/.mlx-agent`, used when `$GROK_HOME` is unset.
pub fn default_grok_home() -> PathBuf {
    grok_home_in(&home_dir().unwrap_or_else(|| PathBuf::from(".")))
}

/// The grok home, created if missing and cached for the process; falls back to
/// [`default_grok_home`] when neither `$GROK_HOME` nor a home resolves.
pub fn grok_home() -> PathBuf {
    static GROK_HOME: OnceLock<PathBuf> = OnceLock::new();
    GROK_HOME
        .get_or_init(|| {
            let home = resolve_grok_home().unwrap_or_else(default_grok_home);
            if let Err(err) = std::fs::create_dir_all(&home) {
                tracing::warn!(path = %home.display(), %err, "failed to create grok home");
            }
            home
        })
        .clone()
}

/// Like [`grok_home`], but `None` when no home resolves (no cwd fallback).
pub fn user_grok_home() -> Option<PathBuf> {
    resolve_grok_home().is_some().then(grok_home)
}

/// The agent home as text for the user and the model: `~/.mlx-agent` for the default home, else the absolute path.
pub fn display_grok_home() -> String {
    display_grok_home_in(resolve_grok_home().as_deref(), home_dir().as_deref())
}

fn display_grok_home_in(agent_home: Option<&Path>, home: Option<&Path>) -> String {
    let is_default =
        |agent_home: &Path| home.is_some_and(|home| same_path(agent_home, &grok_home_in(home)));
    match agent_home {
        Some(agent_home) if !is_default(agent_home) => agent_home.display().to_string(),
        _ => format!("~/{AGENT_DIR_NAME}"),
    }
}

/// True for the agent folder's name, `.mlx-agent`.
pub fn is_agent_dir_name(name: &OsStr) -> bool {
    name == AGENT_DIR_NAME
}

/// `<dir>/.mlx-agent` in `dir`'s own spelling, or `None` when `dir` is the home folder or that folder is the agent home.
/// Every project path the agent reads or writes is built from this.
pub fn project_agent_dir(dir: &Path) -> Option<PathBuf> {
    project_agent_dir_in(dir, home_dir().as_deref(), resolve_grok_home().as_deref())
}

/// [`project_agent_dir`] against an injected home folder and agent home.
pub fn project_agent_dir_in(
    dir: &Path,
    home: Option<&Path>,
    agent_home: Option<&Path>,
) -> Option<PathBuf> {
    if home.is_some_and(|home| same_path(dir, home)) {
        return None;
    }
    let folder = dir.join(AGENT_DIR_NAME);
    if agent_home.is_some_and(|agent_home| same_path(&folder, agent_home)) {
        return None;
    }
    Some(folder)
}

/// [`project_agent_dir`], creating the folder when it is missing.
/// Only a folder this call creates gets the `.gitignore` of `*`; an existing one, or a symlink, is left as it is.
///
/// # Errors
/// Creating the folder (its parent must exist) or writing the `.gitignore` fails.
pub fn ensure_project_agent_dir(dir: &Path) -> std::io::Result<Option<PathBuf>> {
    ensure_project_agent_dir_in(dir, home_dir().as_deref(), resolve_grok_home().as_deref())
}

/// [`ensure_project_agent_dir`] against an injected home folder and agent home.
///
/// # Errors
/// As [`ensure_project_agent_dir`].
pub fn ensure_project_agent_dir_in(
    dir: &Path,
    home: Option<&Path>,
    agent_home: Option<&Path>,
) -> std::io::Result<Option<PathBuf>> {
    let Some(folder) = project_agent_dir_in(dir, home, agent_home) else {
        return Ok(None);
    };
    match std::fs::create_dir(&folder) {
        Ok(()) => write_project_agent_gitignore(&folder)?,
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(error) => return Err(error),
    }
    Ok(Some(folder))
}

/// Give a project agent folder that something else just created (the model's `mkdir`) the same `.gitignore`.
/// The caller knows the folder was absent before; here it must now be a real directory, not a symlink.
///
/// # Errors
/// Reading the folder's metadata or writing the `.gitignore` fails.
pub fn adopt_created_project_agent_dir(folder: &Path) -> std::io::Result<()> {
    match std::fs::symlink_metadata(folder) {
        Ok(meta) if meta.is_dir() => write_project_agent_gitignore(folder),
        Ok(_) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

/// `create_new`, so a `.gitignore` already there (or a symlink in its place) is never overwritten.
fn write_project_agent_gitignore(folder: &Path) -> std::io::Result<()> {
    let opened = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(folder.join(".gitignore"));
    match opened {
        Ok(mut file) => file.write_all(PROJECT_AGENT_GITIGNORE),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
        Err(error) => Err(error),
    }
}

/// Same file or folder, through symlinks. A missing leaf is compared through its canonical parent.
fn same_path(a: &Path, b: &Path) -> bool {
    a == b || canonical_leaf(a) == canonical_leaf(b)
}

fn canonical_leaf(path: &Path) -> PathBuf {
    if let Ok(canonical) = dunce::canonicalize(path) {
        return canonical;
    }
    match (path.parent(), path.file_name()) {
        (Some(parent), Some(name)) => dunce::canonicalize(parent)
            .map(|parent| parent.join(name))
            .unwrap_or_else(|_| path.to_path_buf()),
        _ => path.to_path_buf(),
    }
}

/// Rewrite grok-build's paths in the shipped docs: `~/.grok` becomes `display_home`, and a `.grok` folder name
/// followed by `/`, `` ` ``, `\`, `,`, a space, a line end or the end becomes `.mlx-agent`. Names such as
/// `.grok-plugin` and dotted names such as `ai.x.grok` are kept.
pub fn rewrite_doc_paths(text: &str, display_home: &str) -> String {
    const HOME: &str = "~/.grok";
    const NAME: &str = ".grok";
    let ends_name = |rest: &str| {
        rest.chars()
            .next()
            .is_none_or(|c| matches!(c, '/' | '`' | '\\' | ',' | ' ' | '\n'))
    };
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(c) = rest.chars().next() {
        if let Some(after) = rest.strip_prefix(HOME)
            && ends_name(after)
        {
            out.push_str(display_home);
            rest = after;
        } else if let Some(after) = rest.strip_prefix(NAME)
            && !out.chars().next_back().is_some_and(char::is_alphanumeric)
            && ends_name(after)
        {
            out.push_str(AGENT_DIR_NAME);
            rest = after;
        } else {
            out.push(c);
            let mut chars = rest.chars();
            chars.next();
            rest = chars.as_str();
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;
    use std::ffi::OsString;

    #[test]
    fn env_wins_over_os_home() {
        let resolved =
            resolve_grok_home_from(Some(OsStr::new("/custom/home")), Some(Path::new("/home/u")));
        assert_eq!(
            resolved,
            Some((PathBuf::from("/custom/home"), GrokHomeSource::EnvOverride))
        );
    }

    #[test]
    fn env_used_verbatim_even_when_it_exists() {
        // A real, existing dir whose canonical form differs (macOS symlinks
        // `/var` -> `/private/var`): the env value must come back unchanged.
        let tmp = tempfile::tempdir().unwrap();
        let resolved = resolve_grok_home_from(Some(tmp.path().as_os_str()), None);
        assert_eq!(
            resolved,
            Some((tmp.path().to_path_buf(), GrokHomeSource::EnvOverride))
        );
    }

    #[test]
    fn empty_env_falls_through_to_os_home() {
        let tmp = tempfile::tempdir().unwrap();
        let resolved = resolve_grok_home_from(Some(&OsString::new()), Some(tmp.path()));
        assert_eq!(
            resolved,
            Some((
                dunce::canonicalize(tmp.path()).unwrap().join(".mlx-agent"),
                GrokHomeSource::HomeDefault
            ))
        );
    }

    #[test]
    fn default_grok_home_has_no_verbatim_prefix() {
        // The reason we canonicalize via dunce: std::fs::canonicalize yields
        // `\\?\` verbatim paths on Windows that break git and byte-exact
        // comparisons. No-op assertion on Unix.
        let home = default_grok_home();
        assert!(!home.to_string_lossy().starts_with(r"\\?\"));
        assert!(home.ends_with(".mlx-agent"));
    }

    fn canonical_tempdir() -> (tempfile::TempDir, PathBuf) {
        let tmp = tempfile::tempdir().unwrap();
        let path = dunce::canonicalize(tmp.path()).unwrap();
        (tmp, path)
    }

    #[test]
    fn project_agent_dir_skips_home_and_agent_home() {
        let (_tmp, root) = canonical_tempdir();
        let home = root.join("home");
        let agent_home = home.join(AGENT_DIR_NAME);
        std::fs::create_dir_all(&agent_home).unwrap();
        let sub = home.join("sub");
        std::fs::create_dir_all(&sub).unwrap();

        assert_eq!(
            project_agent_dir_in(&home, Some(&home), Some(&agent_home)),
            None
        );
        assert_eq!(
            project_agent_dir_in(&sub, Some(&home), Some(&agent_home)),
            Some(sub.join(AGENT_DIR_NAME))
        );

        let custom_project = root.join("proj");
        std::fs::create_dir_all(&custom_project).unwrap();
        let custom_home = custom_project.join(AGENT_DIR_NAME);
        assert_eq!(
            project_agent_dir_in(&custom_project, Some(&home), Some(&custom_home)),
            None,
            "a custom agent home at <proj>/.mlx-agent is not a project folder, even before it exists"
        );
        assert_eq!(
            project_agent_dir_in(&home, Some(&home), Some(&custom_home)),
            None,
            "the home folder has no project folder whatever the agent home is"
        );
    }

    #[cfg(unix)]
    #[test]
    fn project_agent_dir_compares_through_symlinks() {
        let (_tmp, root) = canonical_tempdir();
        let home = root.join("home");
        let agent_home = root.join("agent-home");
        std::fs::create_dir_all(&home).unwrap();
        std::fs::create_dir_all(&agent_home).unwrap();

        let home_link = root.join("home-link");
        std::os::unix::fs::symlink(&home, &home_link).unwrap();
        assert_eq!(
            project_agent_dir_in(&home_link, Some(&home), Some(&agent_home)),
            None
        );

        let proj = root.join("proj");
        std::fs::create_dir_all(&proj).unwrap();
        std::os::unix::fs::symlink(&agent_home, proj.join(AGENT_DIR_NAME)).unwrap();
        assert_eq!(
            project_agent_dir_in(&proj, Some(&home), Some(&agent_home)),
            None
        );
    }

    #[test]
    fn ensure_writes_gitignore_only_for_a_folder_it_creates() {
        let (_tmp, root) = canonical_tempdir();
        let home = root.join("home");
        let agent_home = home.join(AGENT_DIR_NAME);
        std::fs::create_dir_all(&agent_home).unwrap();

        let fresh = root.join("fresh");
        std::fs::create_dir_all(&fresh).unwrap();
        let folder = ensure_project_agent_dir_in(&fresh, Some(&home), Some(&agent_home))
            .unwrap()
            .unwrap();
        assert_eq!(folder, fresh.join(AGENT_DIR_NAME));
        assert_eq!(std::fs::read(folder.join(".gitignore")).unwrap(), b"*\n");

        std::fs::write(folder.join(".gitignore"), "edited\n").unwrap();
        ensure_project_agent_dir_in(&fresh, Some(&home), Some(&agent_home)).unwrap();
        assert_eq!(
            std::fs::read_to_string(folder.join(".gitignore")).unwrap(),
            "edited\n"
        );

        let existing = root.join("existing");
        std::fs::create_dir_all(existing.join(AGENT_DIR_NAME)).unwrap();
        ensure_project_agent_dir_in(&existing, Some(&home), Some(&agent_home)).unwrap();
        assert!(!existing.join(AGENT_DIR_NAME).join(".gitignore").exists());

        assert_eq!(
            ensure_project_agent_dir_in(&home, Some(&home), Some(&agent_home)).unwrap(),
            None
        );
        assert_eq!(std::fs::read_dir(&agent_home).unwrap().count(), 0);
        let other_home = root.join("other-home");
        std::fs::create_dir_all(&other_home).unwrap();
        assert_eq!(
            ensure_project_agent_dir_in(&other_home, Some(&other_home), Some(&agent_home)).unwrap(),
            None
        );
        assert!(!other_home.join(AGENT_DIR_NAME).exists());
    }

    #[cfg(unix)]
    #[test]
    fn ensure_and_adopt_write_nothing_through_a_symlink() {
        let (_tmp, root) = canonical_tempdir();
        let home = root.join("home");
        std::fs::create_dir_all(&home).unwrap();
        let target = root.join("target");
        std::fs::create_dir_all(&target).unwrap();
        let proj = root.join("proj");
        std::fs::create_dir_all(&proj).unwrap();
        let link = proj.join(AGENT_DIR_NAME);
        std::os::unix::fs::symlink(&target, &link).unwrap();

        ensure_project_agent_dir_in(&proj, Some(&home), None).unwrap();
        adopt_created_project_agent_dir(&link).unwrap();
        assert!(!target.join(".gitignore").exists());
    }

    #[test]
    fn adopt_writes_gitignore_once_and_keeps_an_existing_one() {
        let (_tmp, root) = canonical_tempdir();
        let made = root.join("made").join(AGENT_DIR_NAME);
        std::fs::create_dir_all(&made).unwrap();
        adopt_created_project_agent_dir(&made).unwrap();
        assert_eq!(std::fs::read(made.join(".gitignore")).unwrap(), b"*\n");

        let own = root.join("own").join(AGENT_DIR_NAME);
        std::fs::create_dir_all(&own).unwrap();
        std::fs::write(own.join(".gitignore"), "mine\n").unwrap();
        adopt_created_project_agent_dir(&own).unwrap();
        assert_eq!(
            std::fs::read_to_string(own.join(".gitignore")).unwrap(),
            "mine\n"
        );

        adopt_created_project_agent_dir(&root.join("missing").join(AGENT_DIR_NAME)).unwrap();
        assert!(!root.join("missing").exists());
    }

    #[test]
    fn agent_dir_name_is_mlx_agent_only() {
        assert!(is_agent_dir_name(OsStr::new(".mlx-agent")));
        assert!(!is_agent_dir_name(OsStr::new(".grok")));
    }

    #[test]
    fn display_home_abbreviates_only_the_default() {
        let (_tmp, home) = canonical_tempdir();
        assert_eq!(
            display_grok_home_in(Some(&home.join(".mlx-agent")), Some(&home)),
            "~/.mlx-agent"
        );
        assert_eq!(display_grok_home_in(None, Some(&home)), "~/.mlx-agent");
        let custom = home.join("elsewhere");
        assert_eq!(
            display_grok_home_in(Some(&custom), Some(&home)),
            custom.display().to_string()
        );
    }

    #[test]
    fn rewrite_doc_paths_renames_grok_build_paths_only() {
        let text = "See ~/.grok/config.toml, `~/.grok` and ~/.grok, then <repo>/.grok/skills, `.grok`, \
                    .grok\\hooks and .grok lists. Keep .grok-plugin/plugin.json, `ai.x.grok`, grok_ws_url and \
                    $GROK_HOME.\nDisk usage for ~/.grok\n%USERPROFILE%\\.grok";
        assert_eq!(
            rewrite_doc_paths(text, "/h/.mlx-agent"),
            "See /h/.mlx-agent/config.toml, `/h/.mlx-agent` and /h/.mlx-agent, then <repo>/.mlx-agent/skills, \
             `.mlx-agent`, .mlx-agent\\hooks and .mlx-agent lists. Keep .grok-plugin/plugin.json, `ai.x.grok`, \
             grok_ws_url and $GROK_HOME.\nDisk usage for /h/.mlx-agent\n%USERPROFILE%\\.mlx-agent"
        );
    }

    #[test]
    fn none_when_nothing_resolves() {
        assert_eq!(
            resolve_grok_home_from(/* grok_home_env */ None, /* os_home */ None),
            None
        );
    }
}
