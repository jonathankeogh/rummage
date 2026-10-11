//! Finds the files to index under the chosen folders, and what kind each is.
//!
//! Skip rules follow DigUp's (MIT), learned on a real developer's Mac: never hidden folders, keys or password files,
//! dependency folders or caches; inside a code project (a folder with .git, Cargo.toml, package.json …) nothing but
//! code, and code only with `--code`, minus what git ignores and vendored or built folders.

use crate::store::Kind;
use ignore::WalkBuilder;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

pub struct Found {
    pub path: PathBuf,
    pub kind: Kind,
    pub size: u64,
    pub mtime: i64,
}

#[derive(Default)]
pub struct Options {
    /// Also index source code inside code projects (searched with "code: …").
    pub code: bool,
    pub excluded: Vec<PathBuf>,
}

/// Plain text documents. Rich formats (PDF, docx, HTML) come in a later milestone.
const DOC_EXTENSIONS: &[&str] = &["txt", "text", "md", "markdown", "rst", "org", "adoc"];

const SOURCE_EXTENSIONS: &[&str] = &[
    "py", "pyi", "pyx", "c", "h", "cc", "cpp", "cxx", "hpp", "hh", "hxx", "m", "mm", "swift", "ts", "tsx", "mts",
    "cts", "js", "jsx", "mjs", "cjs", "go", "rs", "java", "kt", "kts", "scala", "groovy", "rb", "php", "cs", "fs",
    "sh", "bash", "zsh", "fish", "ps1", "lua", "pl", "pm", "r", "jl", "dart", "sql", "vue", "svelte", "astro", "html",
    "htm", "css", "scss", "sass", "less", "gradle", "cmake", "mk", "proto", "graphql", "gql", "tf", "hcl", "nix",
    "zig", "hs", "ex", "exs", "erl", "clj", "cljs", "elm", "ml", "mli", "vim", "el", "lisp", "asm", "s", "metal",
    "glsl", "hlsl", "wgsl", "cu", "cuh", "ino", "v", "sv", "tex",
];
/// What goes with code: a repo's docs and settings.
const CODE_TEXT_EXTENSIONS: &[&str] = &["md", "markdown", "rst", "adoc", "org", "txt", "yaml", "yml", "toml"];
const CODE_FILE_NAMES: &[&str] = &[
    "Makefile",
    "makefile",
    "GNUmakefile",
    "Dockerfile",
    "Containerfile",
    "Justfile",
    "justfile",
    "Rakefile",
    "Gemfile",
    "Podfile",
    "Brewfile",
    "Procfile",
    "Vagrantfile",
    "Jenkinsfile",
];

/// Folders never worth walking into (hidden folders are skipped by the walker itself).
const SKIPPED_FOLDERS: &[&str] =
    &["node_modules", "bower_components", "venv", "site-packages", "__pycache__", "DerivedData", "Pods", "Carthage"];
/// Inside a code project: other people's code and what builds leave.
const CODE_SKIPPED_FOLDERS: &[&str] = &[
    "vendor",
    "Vendor",
    "vendored",
    "third_party",
    "third-party",
    "thirdparty",
    "3rdparty",
    "external",
    "extern",
    "deps",
    "build",
    "Build",
    "dist",
    "out",
    "target",
    "obj",
    "coverage",
    "htmlcov",
    "generated",
    "__generated__",
    "wandb",
    "mlruns",
    "lightning_logs",
];
const PROJECT_MARKERS: &[&str] = &[
    ".git",
    "package.json",
    "Package.swift",
    "pyproject.toml",
    "setup.py",
    "requirements.txt",
    "Cargo.toml",
    "go.mod",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "Gemfile",
    "composer.json",
    "CMakeLists.txt",
];

const MAX_DOC_BYTES: u64 = 20 << 20;
const MAX_CODE_BYTES: u64 = 1 << 20;

/// Every file to index under `roots`, sorted by path.
pub fn walk(roots: &[PathBuf], options: &Options) -> Vec<Found> {
    let mut found = Vec::new();
    for root in roots {
        let excluded = options.excluded.clone();
        let home = std::env::var_os("HOME").map(PathBuf::from);
        let walker = WalkBuilder::new(root)
            .hidden(true)
            .git_ignore(true)
            .git_global(false)
            .git_exclude(true)
            .parents(true)
            .follow_links(false)
            .filter_entry(move |entry| {
                let path = entry.path();
                if !entry.file_type().is_some_and(|t| t.is_dir()) || entry.depth() == 0 {
                    return true;
                }
                let name = entry.file_name().to_string_lossy();
                if SKIPPED_FOLDERS.contains(&name.as_ref()) || name.ends_with(".noindex") {
                    return false;
                }
                if home.as_ref().is_some_and(|h| path == h.join("Library")) {
                    return false;
                }
                if excluded.iter().any(|e| path.starts_with(e)) {
                    return false;
                }
                // A submodule or worktree (.git is a file there): someone else's code, or a copy of this repo's.
                if path.join(".git").is_file() {
                    return false;
                }
                !(CODE_SKIPPED_FOLDERS.contains(&name.as_ref()) && in_project(path))
            })
            .build();
        for entry in walker.filter_map(|e| e.ok()) {
            if !entry.file_type().is_some_and(|t| t.is_file()) {
                continue;
            }
            let path = entry.path();
            let Some(kind) = classify(path, options) else { continue };
            let Ok(meta) = entry.metadata() else { continue };
            let limit = if kind == Kind::Code { MAX_CODE_BYTES } else { MAX_DOC_BYTES };
            if meta.len() == 0 || meta.len() > limit {
                continue;
            }
            let mtime = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0);
            found.push(Found { path: path.to_path_buf(), kind, size: meta.len(), mtime });
        }
    }
    found.sort_by(|a, b| a.path.cmp(&b.path));
    found.dedup_by(|a, b| a.path == b.path);
    found
}

/// What kind of file this is for rummage, or None to leave it alone.
fn classify(path: &Path, options: &Options) -> Option<Kind> {
    let name = path.file_name()?.to_string_lossy();
    if is_secret(&name) {
        return None;
    }
    let ext = path.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
    if in_project(path.parent()?) {
        if !options.code {
            return None;
        }
        let lower = name.to_lowercase();
        if lower.ends_with(".min.js") || lower.ends_with(".min.css") || lower.ends_with(".min.mjs") {
            return None;
        }
        if ["license", "licence", "copying", "notice"].iter().any(|p| lower.starts_with(p)) {
            return None;
        }
        let code = SOURCE_EXTENSIONS.contains(&ext.as_str())
            || CODE_TEXT_EXTENSIONS.contains(&ext.as_str())
            || CODE_FILE_NAMES.contains(&name.as_ref())
            || name.starts_with("Dockerfile");
        return code.then_some(Kind::Code);
    }
    DOC_EXTENSIONS.contains(&ext.as_str()).then_some(Kind::Doc)
}

/// Keys, certificates, password and credential files: never read, wherever they are.
fn is_secret(name: &str) -> bool {
    let lower = name.to_lowercase();
    const TYPES: &[&str] = &[
        "pem",
        "key",
        "p12",
        "pfx",
        "keystore",
        "jks",
        "ppk",
        "kdbx",
        "gpg",
        "asc",
        "crt",
        "cer",
        "der",
        "p8",
        "mobileprovision",
    ];
    if let Some((_, ext)) = lower.rsplit_once('.') {
        if TYPES.contains(&ext) {
            return true;
        }
    }
    const PREFIXES: &[&str] = &[
        "id_rsa",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        ".env",
        "secret",
        "credential",
        ".netrc",
        ".npmrc",
        ".pypirc",
        "htpasswd",
        ".htpasswd",
    ];
    PREFIXES.iter().any(|p| lower.starts_with(p))
}

/// Whether `folder` or a folder above it is a code project.
fn in_project(folder: &Path) -> bool {
    folder.ancestors().any(|dir| PROJECT_MARKERS.iter().any(|m| dir.join(m).exists()) && dir != Path::new("/"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn secrets_are_never_read() {
        for name in ["id_rsa", "id_ed25519.pub", "server.pem", ".env.local", "secrets.txt", "credentials.md"] {
            assert!(is_secret(name), "{name}");
        }
        for name in ["notes.md", "keyboard.txt", "README.md"] {
            assert!(!is_secret(name), "{name}");
        }
    }
}
