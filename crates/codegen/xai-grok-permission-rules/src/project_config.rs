//! Project config-file discovery: locating repo-local `.mcp.json` and `.mlx-agent/config.toml` files by walking from `cwd` up to the git root.
//!
//! These pure `git2` and filesystem walks are shared by the shell's config loaders and the folder-trust gate's `repo_configs_present`.

use std::path::{Path, PathBuf};

use crate::repo::RepoDirChain;

/// Filename of the project-local MCP server config.
pub const MCP_JSON_FILENAME: &str = ".mcp.json";

/// Candidate `.mcp.json` paths from repo root to `cwd`, whether or not they exist.
/// Useful for file watching so newly created files are detected after startup.
pub fn mcp_json_candidate_paths(cwd: &Path) -> Vec<PathBuf> {
    mcp_json_candidate_paths_in(&RepoDirChain::resolve(cwd).dirs)
}

/// [`mcp_json_candidate_paths`] over a precomputed cwd-to-git-root dir chain ([`RepoDirChain`]), repo-root-first.
fn mcp_json_candidate_paths_in(chain_dirs: &[PathBuf]) -> Vec<PathBuf> {
    chain_dirs
        .iter()
        .rev()
        .map(|dir| dir.join(MCP_JSON_FILENAME))
        .collect()
}

/// Find existing `.mcp.json` files from `cwd` up to the git root (repo-root-first order).
pub fn find_mcp_json_files(cwd: &Path) -> Vec<PathBuf> {
    find_mcp_json_files_in(&RepoDirChain::resolve(cwd).dirs)
}

/// [`find_mcp_json_files`] over a precomputed dir chain. See [`RepoDirChain`].
/// `pub` so the folder-trust gate's `repo_configs_present` can call it.
pub fn find_mcp_json_files_in(chain_dirs: &[PathBuf]) -> Vec<PathBuf> {
    mcp_json_candidate_paths_in(chain_dirs)
        .into_iter()
        .filter(|path| path.is_file())
        .collect()
}

/// Find `.mlx-agent/config.toml` from `cwd` up to the git repo root, repo-root (lowest) to cwd (highest), matching skills and AGENTS.md discovery.
/// No repo: only `cwd/.mlx-agent/config.toml`. The home folder and a folder whose `.mlx-agent` is the agent home have none,
/// so the user config is never a project overlay.
pub fn find_project_configs(cwd: &Path) -> Vec<PathBuf> {
    find_project_configs_under(
        cwd,
        xai_dirs::home_dir().as_deref(),
        xai_dirs::resolve_grok_home().as_deref(),
    )
}

pub fn find_project_configs_under(
    cwd: &Path,
    home: Option<&Path>,
    grok_home: Option<&Path>,
) -> Vec<PathBuf> {
    find_project_configs_in(
        &RepoDirChain::resolve_under_home(cwd, home).dirs,
        home,
        grok_home,
    )
}

/// [`find_project_configs`] over a precomputed [`RepoDirChain`], repo-root-first. `pub` for the folder-trust gate.
pub fn find_project_configs_in(
    chain_dirs: &[PathBuf],
    home: Option<&Path>,
    grok_home: Option<&Path>,
) -> Vec<PathBuf> {
    // `dirs` is cwd-first; reverse so repo root comes first (lowest priority) and cwd last (highest)
    chain_dirs
        .iter()
        .rev()
        .filter_map(|dir| xai_dirs::project_agent_dir_in(dir, home, grok_home))
        .map(|folder| folder.join("config.toml"))
        .filter(|config_path| config_path.is_file())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_project_configs_reads_only_the_agent_folder() {
        let tmp = tempfile::tempdir().unwrap();
        let root = dunce::canonicalize(tmp.path()).unwrap();
        let home = root.join("home");
        let agent_home = home.join(".mlx-agent");
        std::fs::create_dir_all(&agent_home).unwrap();
        std::fs::write(agent_home.join("config.toml"), "# user\n").unwrap();
        std::fs::create_dir_all(home.join(".grok")).unwrap();
        std::fs::write(home.join(".grok/config.toml"), "# GROKMARK\n").unwrap();
        assert!(
            find_project_configs_under(&home, Some(&home), Some(&agent_home)).is_empty(),
            "the home folder has no project config"
        );
        assert!(
            find_project_configs_under(&home, Some(&home), Some(&root.join("agent-home")))
                .is_empty(),
            "the home folder has no project config, whatever the agent home is"
        );

        let project = home.join("repo");
        std::fs::create_dir_all(project.join(".grok")).unwrap();
        std::fs::write(project.join(".grok/config.toml"), "# GROKMARK\n").unwrap();
        std::fs::create_dir_all(project.join(".mlx-agent")).unwrap();
        std::fs::write(project.join(".mlx-agent/config.toml"), "# MLXMARK\n").unwrap();
        let found = find_project_configs_under(&project, Some(&home), Some(&agent_home));
        assert_eq!(found, vec![project.join(".mlx-agent").join("config.toml")]);

        let custom_home = home.join("work").join(".mlx-agent");
        std::fs::create_dir_all(&custom_home).unwrap();
        std::fs::write(custom_home.join("config.toml"), "# user\n").unwrap();
        assert!(
            find_project_configs_under(&home.join("work"), Some(&home), Some(&custom_home))
                .is_empty(),
            "a custom agent home's config is the user tier, not a project overlay"
        );
    }
}
