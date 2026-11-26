//! Project and agent-home paths are built only through this crate: no other non-test code line under `crates/codegen`
//! names grok-build's `.grok` folder or spells `.mlx-agent` itself. Comments are not scanned.

use std::path::{Path, PathBuf};

/// `(path under crates/codegen, text on the line)` for the only lines that may keep a needle.
const ALLOWED: &[(&str, &str)] = &[
    (
        "xai-grok-permission-rules/src/shell_access.rs",
        "*name == \".grok\"",
    ),
    ("xai-grok-sandbox/src/command/protected.rs", "\".grok\","),
    ("xai-file-utils/src/workspace_classifier.rs", "\".grok\","),
    (
        "xai-grok-shell/src/builtin.rs",
        "content.replace(&home_prefix, \"~/.grok/\")",
    ),
    ("xai-grok-update/src/auto_update.rs", "\".grok/downloads/\""),
    (
        "xai-grok-workspace/src/worktree/mod.rs",
        "temp_dir().join(xai_dirs::AGENT_DIR_NAME)",
    ),
];

/// Crates that only tests depend on.
const TEST_CRATES: &[&str] = &["xai-grok-test-support", "xai-grok-pager-pty-harness"];

const TEST_ATTRS: &[&str] = &[
    "#[cfg(test)]",
    "#[cfg(all(test",
    "#[cfg(any(test",
    "#[test]",
    "#[tokio::test",
];

/// The needle a code line breaks, if any.
fn needle(line: &str) -> Option<&'static str> {
    for needle in ["\".grok\"", "\".grok/", "\".grok\\\\", ".grok/"] {
        if line.contains(needle) {
            return Some(needle);
        }
    }
    if line.contains("\".mlx-agent") {
        return Some("\".mlx-agent");
    }
    if line.contains("join(") && line.contains("AGENT_DIR_NAME") {
        return Some("join( with AGENT_DIR_NAME");
    }
    None
}

struct Line {
    number: usize,
    code: String,
    test: bool,
}

#[derive(Default)]
struct Lexed {
    lines: Vec<Line>,
    /// `(mod name, #[path] value)` of every `mod name;` under a test attribute.
    test_mods: Vec<(String, Option<String>)>,
    whole_file_test: bool,
}

#[derive(Clone, Copy, PartialEq)]
enum State {
    Code,
    LineComment,
    BlockComment(u32),
    Str,
    RawStr(usize),
}

fn is_ident(c: Option<char>) -> bool {
    c.is_some_and(|c| c.is_alphanumeric() || c == '_')
}

fn starts_with_at(chars: &[char], at: usize, text: &str) -> bool {
    let mut i = at;
    for t in text.chars() {
        if chars.get(i) != Some(&t) {
            return false;
        }
        i += 1;
    }
    true
}

/// Hashes of a raw string opener (`r"`, `r#"`, `br##"`) whose `r` is at `at`, or `None`.
fn raw_string_hashes(chars: &[char], at: usize) -> Option<usize> {
    let before = at.checked_sub(1).and_then(|i| chars.get(i)).copied();
    let prefix_ok = !is_ident(before)
        || (before == Some('b')
            && !is_ident(at.checked_sub(2).and_then(|i| chars.get(i)).copied()));
    if !prefix_ok {
        return None;
    }
    let mut i = at + 1;
    while chars.get(i) == Some(&'#') {
        i += 1;
    }
    (chars.get(i) == Some(&'"')).then_some(i - at - 1)
}

/// `mod name;` and its `#[path]` from the text of one attributed item.
fn test_mod_decl(item: &str) -> Option<(String, Option<String>)> {
    let path = item.find("#[path = \"").and_then(|start| {
        let rest = item.get(start + "#[path = \"".len()..)?;
        rest.find('"')
            .and_then(|end| rest.get(..end))
            .map(str::to_owned)
    });
    let after_attrs = item.rfind(']').and_then(|end| item.get(end + 1..))?.trim();
    let decl = after_attrs
        .strip_prefix("pub(crate) ")
        .or_else(|| after_attrs.strip_prefix("pub(super) "))
        .or_else(|| after_attrs.strip_prefix("pub "))
        .unwrap_or(after_attrs);
    let name = decl.strip_prefix("mod ")?.trim_end_matches(';').trim();
    (!name.is_empty() && name.chars().all(|c| c.is_alphanumeric() || c == '_'))
        .then(|| (name.to_owned(), path))
}

fn lex(src: &str) -> Lexed {
    let chars: Vec<char> = src.chars().collect();
    let mut out = Lexed::default();
    let mut state = State::Code;
    let mut line = String::new();
    let mut line_test = false;
    let mut number = 1;
    let mut depth: u32 = 0;
    let mut test_depth: Option<u32> = None;
    let mut pending: Option<(String, i32, u32)> = None;
    let mut i = 0;
    while let Some(&c) = chars.get(i) {
        if c == '\n' {
            if state == State::LineComment {
                state = State::Code;
            }
            let trimmed = line.trim_start();
            out.whole_file_test |=
                trimmed.starts_with("#![cfg(test)]") || trimmed.starts_with("#![cfg(all(test");
            out.lines.push(Line {
                number,
                code: std::mem::take(&mut line),
                test: line_test || pending.is_some() || test_depth.is_some(),
            });
            line_test = pending.is_some() || test_depth.is_some();
            number += 1;
            if let Some((item, _, _)) = pending.as_mut() {
                item.push('\n');
            }
            i += 1;
            continue;
        }
        let next = chars.get(i + 1).copied();
        let mut take = 1;
        match state {
            State::LineComment => {
                i += 1;
                continue;
            }
            State::BlockComment(level) => {
                if c == '*' && next == Some('/') {
                    state = if level == 1 {
                        State::Code
                    } else {
                        State::BlockComment(level - 1)
                    };
                    i += 2;
                } else if c == '/' && next == Some('*') {
                    state = State::BlockComment(level + 1);
                    i += 2;
                } else {
                    i += 1;
                }
                continue;
            }
            State::Str => {
                if c == '\\' && next != Some('\n') {
                    take = 2;
                } else if c == '"' {
                    state = State::Code;
                }
            }
            State::RawStr(hashes) => {
                if c == '"' && (1..=hashes).all(|h| chars.get(i + h) == Some(&'#')) {
                    state = State::Code;
                    take = 1 + hashes;
                }
            }
            State::Code => {
                if c == '/' && next == Some('/') {
                    state = State::LineComment;
                    i += 2;
                    continue;
                }
                if c == '/' && next == Some('*') {
                    state = State::BlockComment(1);
                    i += 2;
                    continue;
                }
                if c == '"' {
                    state = State::Str;
                } else if c == 'r'
                    && let Some(hashes) = raw_string_hashes(&chars, i)
                {
                    state = State::RawStr(hashes);
                    take = 2 + hashes;
                } else if c == '\'' {
                    if next == Some('\\') {
                        let mut end = i + 3;
                        while chars.get(end).is_some_and(|&ch| ch != '\'' && ch != '\n') {
                            end += 1;
                        }
                        take = end + 1 - i;
                    } else if chars.get(i + 2) == Some(&'\'') {
                        take = 3;
                    }
                } else if c == '#'
                    && test_depth.is_none()
                    && pending.is_none()
                    && TEST_ATTRS
                        .iter()
                        .any(|attr| starts_with_at(&chars, i, attr))
                {
                    pending = Some((String::new(), 0, depth));
                    line_test = true;
                } else if c == '{' {
                    depth += 1;
                    if pending.take().is_some() {
                        test_depth = Some(depth);
                    }
                } else if c == '}' {
                    if test_depth == Some(depth) {
                        test_depth = None;
                    }
                    if pending
                        .as_ref()
                        .is_some_and(|(_, _, start)| *start == depth)
                    {
                        pending = None;
                    }
                    depth = depth.saturating_sub(1);
                } else if let Some((item, nest, _)) = pending.as_mut() {
                    match c {
                        '(' | '[' => *nest += 1,
                        ')' | ']' => *nest -= 1,
                        ';' if *nest == 0 => {
                            item.push(';');
                            if let Some(decl) = test_mod_decl(item) {
                                out.test_mods.push(decl);
                            }
                            pending = None;
                        }
                        ',' if *nest == 0 => pending = None,
                        _ => {}
                    }
                }
            }
        }
        for k in i..i + take {
            if let Some(&ch) = chars.get(k).filter(|&&ch| ch != '\n') {
                line.push(ch);
                if let Some((item, _, _)) = pending.as_mut() {
                    item.push(ch);
                }
            }
        }
        i += take;
    }
    out.lines.push(Line {
        number,
        code: line,
        test: line_test || pending.is_some() || test_depth.is_some(),
    });
    out
}

fn rust_files(dir: &Path, files: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if path.is_dir() {
            if name != "target" && !name.starts_with('.') {
                rust_files(&path, files);
            }
        } else if name.ends_with(".rs") {
            files.push(path);
        }
    }
}

/// Files and folders a `#[cfg(test)] mod name;` in `file` points at.
fn test_mod_targets(file: &Path, name: &str, path_attr: Option<&str>) -> Vec<PathBuf> {
    let dir = file.parent().unwrap_or(Path::new(""));
    if let Some(path) = path_attr {
        let target = dir.join(path);
        let folder = target.with_extension("");
        return vec![target, folder];
    }
    let stem = file.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let base = if matches!(stem, "mod" | "lib" | "main") {
        dir.to_path_buf()
    } else {
        dir.join(stem)
    };
    vec![base.join(format!("{name}.rs")), base.join(name)]
}

#[test]
fn only_the_resolver_names_the_project_or_agent_folder() {
    let codegen = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/codegen")
        .to_path_buf();
    let mut files = Vec::new();
    rust_files(&codegen, &mut files);
    files.sort();
    assert!(files.len() > 1000, "found only {} files", files.len());

    let lexed: Vec<(PathBuf, String, Lexed)> = files
        .into_iter()
        .filter_map(|path| {
            let rel = path
                .strip_prefix(&codegen)
                .ok()?
                .to_string_lossy()
                .replace('\\', "/");
            let src = std::fs::read_to_string(&path).ok()?;
            Some((path, rel, lex(&src)))
        })
        .collect();
    let mut test_targets: Vec<PathBuf> = Vec::new();
    for (path, _, file) in &lexed {
        for (name, path_attr) in &file.test_mods {
            test_targets.extend(test_mod_targets(path, name, path_attr.as_deref()));
        }
    }
    let is_test_file = |path: &Path, rel: &str| {
        let mut parts = rel.split('/');
        let krate = parts.next().unwrap_or("");
        TEST_CRATES.contains(&krate)
            || matches!(parts.next(), Some("tests" | "benches" | "examples"))
            || test_targets
                .iter()
                .any(|t| path == t || path.starts_with(t))
    };

    let mut used = vec![false; ALLOWED.len()];
    let mut offenders = Vec::new();
    let mut scanned = 0;
    for (path, rel, file) in &lexed {
        if rel.starts_with("xai-dirs/") || file.whole_file_test || is_test_file(path, rel) {
            continue;
        }
        scanned += 1;
        for line in file.lines.iter().filter(|line| !line.test) {
            let Some(hit) = needle(&line.code) else {
                continue;
            };
            let allowed = ALLOWED
                .iter()
                .position(|(file, text)| rel == file && line.code.contains(text));
            match allowed {
                Some(index) => {
                    if let Some(slot) = used.get_mut(index) {
                        *slot = true;
                    }
                }
                None => offenders.push(format!(
                    "{rel}:{}: {hit}: {}",
                    line.number,
                    line.code.trim()
                )),
            }
        }
    }
    assert!(scanned > 1000, "scanned only {scanned} files");
    assert!(
        offenders.is_empty(),
        "build project paths with xai_dirs::project_agent_dir and agent-home paths with the agent home; \
         name grok-build's folder nowhere:\n{}",
        offenders.join("\n")
    );
    let stale: Vec<_> = ALLOWED
        .iter()
        .zip(&used)
        .filter(|(_, used)| !**used)
        .map(|(entry, _)| entry)
        .collect();
    assert!(
        stale.is_empty(),
        "allowed lines that no longer exist: {stale:?}"
    );
}

#[test]
fn the_lexer_skips_comments_and_test_code_but_not_strings() {
    let src = "// \".grok\"\n\
               let a = \".grok\"; /* \".grok/ */\n\
               let b = r#\"x/.grok/y\"#;\n\
               let c = '{';\n\
               #[cfg(test)]\n\
               mod tests {\n\
                   fn f() { let _ = \".grok\"; }\n\
               }\n\
               #[cfg(test)]\n\
               #[path = \"x_tests.rs\"]\n\
               mod x;\n\
               let d = \".mlx-agent\";\n";
    let lexed = lex(src);
    let hits: Vec<(usize, &'static str)> = lexed
        .lines
        .iter()
        .filter(|line| !line.test)
        .filter_map(|line| needle(&line.code).map(|hit| (line.number, hit)))
        .collect();
    assert_eq!(
        hits,
        vec![(2, "\".grok\""), (3, ".grok/"), (12, "\".mlx-agent")]
    );
    assert_eq!(
        lexed.test_mods,
        vec![("x".to_owned(), Some("x_tests.rs".to_owned()))]
    );
}
