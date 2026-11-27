// The upstream sync drops `docs/internal/`, so the shipped config reference is the operator page this checks.

use xai_grok_shell::agent::config::FEATURES;

const CONFIG_REFERENCE: &str = include_str!("../docs/user-guide/26-config-reference.md");

#[test]
fn every_registered_feature_reaches_the_operator() {
    for spec in FEATURES {
        let row_start = format!("| `{}` |", spec.path);
        let row = CONFIG_REFERENCE
            .lines()
            .find(|line| line.starts_with(&row_start))
            .unwrap_or_else(|| panic!("{} has no row in 26-config-reference.md", spec.path));
        assert!(
            row.contains(&format!("`{}`", spec.env)),
            "the {} row of 26-config-reference.md does not name {}",
            spec.path,
            spec.env,
        );
    }
}
