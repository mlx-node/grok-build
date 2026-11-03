pub(crate) mod host_service;
pub(crate) mod listing;
pub(crate) mod manager;
pub(crate) mod notify;
pub(crate) mod registry;
pub(crate) mod request_service;
pub(crate) mod schema_contract;
pub(crate) mod store;
pub(crate) mod tracker;

#[cfg(test)]
mod builtin_tests {
    #[test]
    fn every_builtin_validates_and_matches_its_registered_name() {
        for builtin in super::registry::BUILTIN_WORKFLOWS {
            let meta = xai_workflow::extract_meta(builtin.script)
                .unwrap_or_else(|e| panic!("builtin '{}' must validate: {e}", builtin.name));
            assert_eq!(
                meta.name, builtin.name,
                "registry key must equal meta.name for '{}'",
                builtin.name
            );
            assert!(
                meta.when_to_use
                    .as_deref()
                    .is_some_and(|text| !text.is_empty()),
                "builtin '{}' needs meta.when_to_use",
                builtin.name
            );
            let keys: Vec<&str> = builtin
                .args_hint
                .match_indices("\":")
                .filter_map(|(end, _)| {
                    let head = &builtin.args_hint[..end];
                    head.rfind('"').map(|start| &head[start + 1..])
                })
                .collect();
            assert!(
                !keys.is_empty(),
                "builtin '{}' needs an args hint",
                builtin.name
            );
            for key in keys {
                let access = format!("args.{key}");
                let reads_key = builtin.script.match_indices(&access).any(|(at, _)| {
                    builtin.script[at + access.len()..]
                        .chars()
                        .next()
                        .is_none_or(|next| !(next.is_alphanumeric() || next == '_'))
                });
                assert!(
                    reads_key,
                    "builtin '{}' args hint names `{key}` but the script never reads args.{key}",
                    builtin.name
                );
            }
        }
    }

    #[test]
    fn deep_research_binds_shards_and_renders_verified_claims() {
        let script = super::registry::BUILTIN_WORKFLOWS
            .iter()
            .find(|builtin| builtin.name == "deep-research")
            .map(|builtin| builtin.script)
            .expect("deep-research builtin registered");
        assert!(script.contains("expected_ids[shard_idx]"));
        assert!(script.contains("verification_results[assigned_shard]"));
        assert!(script.contains("verified_claim_ids"));
        assert!(script.contains("**Status: Partial**"));
        assert!(!script.contains("label: \"research-reporter\""));
        assert!(script.contains("label: \"report-synthesizer\""));
        assert!(script.contains("<report-body>"));
        assert!(!script.contains("output_schema: synthesis_schema"));
        assert!(script.contains("failed citation validation"));
        assert!(script.contains("let findings_fallback"));
        assert!(script.contains("full_report += \"\\n## Sources\\n\""));
        assert!(script.contains("report: chat_report"));
        assert!(!script.contains("chat_report += \"\\n## Sources\\n\""));
    }
}
