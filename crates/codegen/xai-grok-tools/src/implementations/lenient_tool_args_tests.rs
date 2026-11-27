//! Every integer, number, and boolean a tool advertises must also accept its string form.
//!
//! Local runtimes (mlx, llama.cpp chat templates) render `<parameter=k>v</parameter>` without
//! consulting the schema, so `"timeout_ms": "300000"` reaches deserialization. Walks each tool's
//! advertised JSON schema, builds a minimal valid input, swaps one scalar for its string form,
//! and asserts the input still deserializes.

use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Map, Value, json};

use crate::types::tool_io::ToolInput;

const MAX_DEPTH: usize = 8;

fn resolve<'a>(schema: &'a Value, root: &'a Value) -> &'a Value {
    let mut current = schema;
    for _ in 0..MAX_DEPTH {
        let Some(reference) = current.get("$ref").and_then(Value::as_str) else {
            return current;
        };
        let pointer = reference.trim_start_matches('#');
        current = root
            .pointer(pointer)
            .unwrap_or_else(|| panic!("unresolvable $ref {reference}"));
    }
    current
}

fn variants(schema: &Value) -> Option<&Vec<Value>> {
    schema
        .get("oneOf")
        .or_else(|| schema.get("anyOf"))
        .and_then(Value::as_array)
}

fn is_null_schema(schema: &Value) -> bool {
    schema.get("type").and_then(Value::as_str) == Some("null")
}

/// The schema's non-null `type`, if it names one (`"integer"` or `["integer", "null"]`).
fn primary_type(schema: &Value) -> Option<&str> {
    match schema.get("type")? {
        Value::String(t) => Some(t.as_str()),
        Value::Array(types) => types
            .iter()
            .filter_map(Value::as_str)
            .find(|t| *t != "null"),
        _ => None,
    }
}

fn scalar_sample(schema: &Value, kind: &str) -> Value {
    match kind {
        "boolean" => json!(true),
        _ => {
            let min = schema.get("minimum").and_then(Value::as_f64).unwrap_or(1.0);
            let max = schema
                .get("maximum")
                .and_then(Value::as_f64)
                .unwrap_or(f64::MAX);
            let n = min.max(1.0).min(max);
            json!(n as i64)
        }
    }
}

/// A non-whole value inside the schema's bounds, so `number` fields prove they keep fractions.
fn fractional_sample(schema: &Value) -> Value {
    let max = schema.get("maximum").and_then(Value::as_f64);
    let min = schema
        .get("minimum")
        .and_then(Value::as_f64)
        .unwrap_or_else(|| max.map_or(1.0, |m| (m - 1.0).min(1.0)));
    let max = max.unwrap_or(min + 2.0);
    let n = [
        min.max(1.0) + 0.5,
        min + 0.5,
        (min + max) / 2.0,
        min + (max - min) / 4.0,
    ]
    .into_iter()
    .find(|n| n.fract() != 0.0 && (min..=max).contains(n))
    .unwrap_or_else(|| panic!("no fractional value inside [{min}, {max}] for {schema}"));
    json!(n)
}

#[test]
fn fractional_sample_is_fractional_within_bounds() {
    for schema in [
        json!({}),
        json!({"maximum": 1.0}),
        json!({"minimum": 0.0}),
        json!({"minimum": 0.0, "maximum": 1.0}),
        json!({"minimum": 2.0, "maximum": 2.25}),
        json!({"minimum": -3.0, "maximum": -2.0}),
    ] {
        let n = fractional_sample(&schema).as_f64().expect("a number");
        assert_ne!(n.fract(), 0.0, "{schema} sampled whole {n}");
        let min = schema
            .get("minimum")
            .and_then(Value::as_f64)
            .unwrap_or(f64::MIN);
        let max = schema
            .get("maximum")
            .and_then(Value::as_f64)
            .unwrap_or(f64::MAX);
        assert!(
            (min..=max).contains(&n),
            "{schema} sampled {n} out of bounds"
        );
    }
}

/// A minimal value the schema accepts: required properties only, first non-null variant.
fn sample(schema: &Value, root: &Value, depth: usize) -> Value {
    assert!(depth < MAX_DEPTH, "schema nests too deep: {schema}");
    let schema = resolve(schema, root);
    if let Some(value) = schema.get("const") {
        return value.clone();
    }
    if let Some(value) = schema
        .get("enum")
        .and_then(Value::as_array)
        .and_then(|values| values.iter().find(|v| !v.is_null()))
    {
        return value.clone();
    }
    if let Some(options) = variants(schema) {
        let option = options
            .iter()
            .find(|o| !is_null_schema(resolve(o, root)))
            .expect("a non-null variant");
        let mut value = sample(option, root, depth + 1);
        if let (Some(extra), Value::Object(map)) = (schema.get("properties"), &mut value) {
            let required = required_names(schema);
            fill_required(map, extra, &required, root, depth);
        }
        return value;
    }
    if let Some(parts) = schema.get("allOf").and_then(Value::as_array) {
        let mut merged = Map::new();
        for part in parts {
            if let Value::Object(map) = sample(part, root, depth + 1) {
                merged.extend(map);
            }
        }
        return Value::Object(merged);
    }
    match primary_type(schema) {
        Some("string") => json!("x"),
        Some(kind @ ("integer" | "number" | "boolean")) => scalar_sample(schema, kind),
        Some("array") => {
            let min_items = schema.get("minItems").and_then(Value::as_u64).unwrap_or(0);
            let item = schema.get("items").cloned().unwrap_or(json!({}));
            Value::Array(
                (0..min_items)
                    .map(|_| sample(&item, root, depth + 1))
                    .collect(),
            )
        }
        Some("null") => Value::Null,
        _ => {
            let mut map = Map::new();
            if let Some(properties) = schema.get("properties") {
                fill_required(&mut map, properties, &required_names(schema), root, depth);
            }
            Value::Object(map)
        }
    }
}

fn required_names(schema: &Value) -> Vec<String> {
    schema
        .get("required")
        .and_then(Value::as_array)
        .map(|names| {
            names
                .iter()
                .filter_map(|n| n.as_str().map(str::to_owned))
                .collect()
        })
        .unwrap_or_default()
}

fn fill_required(
    map: &mut Map<String, Value>,
    properties: &Value,
    required: &[String],
    root: &Value,
    depth: usize,
) {
    for name in required {
        if let Some(property) = properties.get(name) {
            map.insert(name.clone(), sample(property, root, depth + 1));
        }
    }
}

/// One scalar the model can send: where it sits in a sample input and the value it takes there.
struct ScalarSlot {
    path: Vec<PathStep>,
    kind: String,
    typed: Value,
}

#[derive(Clone)]
enum PathStep {
    Key(String),
    Index,
}

fn describe(path: &[PathStep]) -> String {
    path.iter()
        .map(|step| match step {
            PathStep::Key(k) => k.clone(),
            PathStep::Index => "[0]".to_owned(),
        })
        .collect::<Vec<_>>()
        .join(".")
}

/// Collect every integer/number/boolean property reachable through objects and arrays, plus a
/// base input in which each array on the way holds one element so nested slots exist.
fn collect_slots(
    schema: &Value,
    root: &Value,
    path: &mut Vec<PathStep>,
    base: &mut Value,
    slots: &mut Vec<ScalarSlot>,
    depth: usize,
) {
    if depth >= MAX_DEPTH {
        return;
    }
    let schema = resolve(schema, root);
    if let Some(options) = variants(schema)
        && schema.get("properties").is_none()
    {
        // Nullable wrapper (`anyOf: [T, null]`): walk T. Tagged unions: walk the variant the
        // sample picked, since the other variants need a different tag.
        let non_null: Vec<&Value> = options
            .iter()
            .filter(|o| !is_null_schema(resolve(o, root)))
            .collect();
        if let Some(first) = non_null.first() {
            collect_slots(first, root, path, base, slots, depth + 1);
        }
        return;
    }
    if let Some(kind) = primary_type(schema)
        && matches!(kind, "integer" | "number" | "boolean")
    {
        let enum_value = schema
            .get("enum")
            .and_then(Value::as_array)
            .and_then(|values| values.iter().find(|v| !v.is_null()).cloned());
        if kind == "number" && enum_value.is_none() {
            slots.push(ScalarSlot {
                path: path.clone(),
                kind: "number, fractional".to_owned(),
                typed: fractional_sample(schema),
            });
        }
        slots.push(ScalarSlot {
            path: path.clone(),
            kind: kind.to_owned(),
            typed: enum_value.unwrap_or_else(|| scalar_sample(schema, kind)),
        });
        return;
    }
    if primary_type(schema) == Some("array") {
        let item = schema.get("items").cloned().unwrap_or(json!({}));
        if let Value::Array(items) = base {
            if items.is_empty() {
                items.push(sample(&item, root, depth + 1));
            }
            path.push(PathStep::Index);
            if let Some(first) = items.first_mut() {
                collect_slots(&item, root, path, first, slots, depth + 1);
            }
            path.pop();
        }
        return;
    }
    let Some(properties) = schema.get("properties").and_then(Value::as_object) else {
        return;
    };
    let Value::Object(map) = base else {
        return;
    };
    for (name, property) in properties {
        let property_schema = resolve(property, root);
        let is_container = matches!(primary_type(property_schema), Some("array" | "object"))
            || variants(property_schema).is_some()
            || property_schema.get("properties").is_some();
        if !map.contains_key(name) && is_container {
            let value = sample(property, root, depth + 1);
            if value.is_null() {
                continue;
            }
            map.insert(name.clone(), value);
        }
        path.push(PathStep::Key(name.clone()));
        let mut scratch = Value::Null;
        let target = map.get_mut(name).unwrap_or(&mut scratch);
        collect_slots(property, root, path, target, slots, depth + 1);
        path.pop();
    }
}

fn set_at(value: &mut Value, path: &[PathStep], new: Value) {
    let Some((last, parents)) = path.split_last() else {
        *value = new;
        return;
    };
    let mut current = value;
    for step in parents {
        current = match step {
            PathStep::Key(k) => current
                .get_mut(k.as_str())
                .unwrap_or_else(|| panic!("missing key {k}")),
            PathStep::Index => current.get_mut(0).expect("non-empty array"),
        };
    }
    match last {
        PathStep::Key(k) => {
            current
                .as_object_mut()
                .expect("object parent")
                .insert(k.clone(), new);
        }
        PathStep::Index => {
            *current.get_mut(0).expect("non-empty array") = new;
        }
    }
}

fn stringify(scalar: &Value) -> Value {
    match scalar {
        Value::Bool(b) => json!(b.to_string()),
        Value::Number(n) => json!(n.to_string()),
        other => other.clone(),
    }
}

#[derive(Default)]
struct Report {
    /// `Type.path` of every scalar the walk reached.
    walked: Vec<String>,
    failures: Vec<String>,
}

fn check<T: DeserializeOwned + Serialize + schemars::JsonSchema>() -> Report {
    let label = std::any::type_name::<T>();
    let root = serde_json::to_value(schemars::schema_for!(T)).expect("schema serializes");
    let mut base = sample(&root, &root, 0);
    let mut slots = Vec::new();
    collect_slots(&root, &root, &mut Vec::new(), &mut base, &mut slots, 0);

    let mut report = Report::default();
    if let Err(err) = serde_json::from_value::<T>(base.clone()) {
        report.failures.push(format!(
            "{label}: sample input {base} does not deserialize ({err}); teach `sample` this schema shape"
        ));
        return report;
    }

    for slot in &slots {
        let where_ = format!("{label}.{}", describe(&slot.path));
        report.walked.push(where_.clone());
        let mut typed = base.clone();
        set_at(&mut typed, &slot.path, slot.typed.clone());
        let mut stringly = base.clone();
        set_at(&mut stringly, &slot.path, stringify(&slot.typed));
        let expected = match serde_json::from_value::<T>(typed.clone()) {
            Ok(parsed) => serde_json::to_value(parsed).expect("input serializes"),
            Err(err) => {
                report.failures.push(format!(
                    "{where_} ({}): typed sample {typed} does not deserialize ({err})",
                    slot.kind
                ));
                continue;
            }
        };
        match serde_json::from_value::<T>(stringly.clone()) {
            Ok(parsed) => {
                let actual = serde_json::to_value(parsed).expect("input serializes");
                if actual != expected {
                    report.failures.push(format!(
                        "{where_} ({}): string form {stringly} parsed to {actual}, typed form to {expected}",
                        slot.kind
                    ));
                }
            }
            Err(err) => report.failures.push(format!(
                "{where_} ({}): rejects its string form {stringly}: {err}",
                slot.kind
            )),
        }
    }
    report
}

/// Lists each built-in `ToolInput` variant with its input type, and fails to compile when a
/// variant is added to `ToolInput` without being listed here.
macro_rules! builtin_inputs {
    ($($variant:ident => $ty:ty),* $(,)?) => {{
        #[allow(dead_code)]
        fn exhaustive(input: &ToolInput) {
            match input {
                $(ToolInput::$variant(_) => {})*
                ToolInput::MCPTool(_) | ToolInput::Dynamic(_) => {}
            }
        }
        vec![$(check::<$ty> as fn() -> Report),*]
    }};
}

#[test]
fn every_scalar_tool_argument_accepts_its_string_form() {
    use crate::implementations::grok_build::{
        ask_user_question, enter_plan_mode, exit_plan_mode, grep, image_edit, image_gen, list_dir,
        monitor, read_file, scheduler, search_replace, send_feedback, send_subagent_message, todo,
        update_goal, video_gen, web_fetch, web_search, workflow,
    };
    use crate::implementations::{codex, grok_build_hashline, lsp, memory, opencode};

    let mut checks = builtin_inputs! {
        ReadFile => read_file::ReadFileInput,
        SearchReplace => search_replace::SearchReplaceInput,
        Bash => crate::implementations::BashToolInput,
        Grep => grep::GrepSearchInput,
        ListDir => list_dir::ListDirInput,
        TodoWrite => todo::TodoWriteInput,
        Skill => crate::implementations::skills::skill::SkillInput,
        TaskOutput => xai_tool_types::TaskOutputToolInput,
        WaitTasks => xai_tool_types::WaitTasksToolInput,
        KillTask => xai_tool_types::KillTaskToolInput,
        Task => xai_tool_types::TaskToolInput,
        WebSearch => web_search::WebSearchInput,
        ImageGen => image_gen::ImageGenInput,
        ImageEdit => image_edit::ImageEditInput,
        ImageToVideo => video_gen::ImageToVideoInput,
        ReferenceToVideo => video_gen::ReferenceToVideoInput,
        WebFetch => web_fetch::WebFetchInput,
        Write => opencode::write::WriteInput,
        ApplyPatch => codex::apply_patch::tool::ApplyPatchInput,
        HashlineEdit => grok_build_hashline::edit::types::HashlineEditInput,
        CodexListDir => codex::list_dir::tool::CodexListDirInput,
        CodexGrepFiles => codex::grep_files::tool::CodexGrepFilesInput,
        CodexReadFile => codex::read_file::tool::CodexReadFileInput,
        MemorySearch => memory::types::MemorySearchInput,
        MemoryGet => memory::types::MemoryGetInput,
        SearchTool => crate::implementations::search_tool::SearchToolInput,
        UseTool => crate::implementations::UseToolInput,
        EnterPlanMode => enter_plan_mode::EnterPlanModeInput,
        ExitPlanMode => exit_plan_mode::ExitPlanModeInput,
        AskUserQuestion => ask_user_question::AskUserQuestionInput,
        SendSubagentMessage => send_subagent_message::SendSubagentMessageInput,
        SendFeedback => send_feedback::SendFeedbackInput,
        Lsp => lsp::LspToolInput,
        Monitor => monitor::types::MonitorInput,
        SchedulerCreate => scheduler::create::SchedulerCreateInput,
        SchedulerDelete => scheduler::delete::SchedulerDeleteInput,
        SchedulerList => scheduler::list::SchedulerListInput,
        UpdateGoal => update_goal::UpdateGoalInput,
        Workflow => workflow::WorkflowToolInput,
    };
    // Tools whose input has no `ToolInput` variant.
    checks.extend([
        check::<opencode::bash::BashInput> as fn() -> Report,
        check::<opencode::edit::EditInput>,
        check::<opencode::glob::GlobInput>,
        check::<opencode::grep::GrepInput>,
        check::<opencode::read::ReadInput>,
        check::<opencode::skill::SkillInput>,
        check::<opencode::todowrite::TodoWriteInput>,
    ]);

    let reports: Vec<Report> = checks.iter().map(|check| check()).collect();
    let walked: Vec<&str> = reports
        .iter()
        .flat_map(|r| r.walked.iter().map(String::as_str))
        .collect();
    // The fields from the failing mlx session; if the walk stops reaching them it proves nothing.
    for field in [
        "xai_tool_types::task::TaskOutputToolInput.timeout_ms",
        "xai_grok_tools::implementations::grok_build::read_file::ReadFileInput.limit",
        "xai_grok_tools::implementations::grok_build::video_gen::ReferenceToVideoInput.keyframes.[0].timestamp_s",
    ] {
        assert!(walked.contains(&field), "walk no longer reaches {field}");
    }
    let failures: Vec<&str> = reports
        .iter()
        .flat_map(|r| r.failures.iter().map(String::as_str))
        .collect();
    assert!(
        failures.is_empty(),
        "tool arguments that reject their string form:\n{}",
        failures.join("\n")
    );
}

#[test]
fn walker_finds_nested_and_nullable_scalars() {
    #[derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
    struct Item {
        weight: u32,
    }
    #[derive(serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
    struct Strict {
        name: String,
        limit: Option<u64>,
        flag: bool,
        items: Vec<Item>,
    }
    let report = check::<Strict>();
    let field = |line: &str| -> String {
        let head = line.split_once(" (").map_or(line, |(head, _)| head);
        head.rsplit_once("Strict.")
            .map_or(head, |(_, field)| field)
            .to_owned()
    };
    let mut walked: Vec<String> = report.walked.iter().map(|w| field(w)).collect();
    let mut failed: Vec<String> = report.failures.iter().map(|f| field(f)).collect();
    walked.sort_unstable();
    failed.sort_unstable();
    let expected = ["flag", "items.[0].weight", "limit"];
    assert_eq!(walked, expected);
    assert_eq!(failed, expected, "{:#?}", report.failures);
}

#[test]
fn read_file_accepts_string_limit_from_mlx_session() {
    use crate::implementations::grok_build::read_file::ReadFileInput;
    let input: ReadFileInput =
        serde_json::from_value(json!({"target_file": "/x", "limit": "900"})).unwrap();
    assert_eq!(input.path, "/x");
    assert_eq!(input.limit, Some(900));
}
