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

/// Documents read as text: plain text, plus docx, odt and html converted (see `extract::document_text`).
const DOC_EXTENSIONS: &[&str] =
    &["txt", "text", "md", "markdown", "rst", "org", "adoc", "docx", "odt", "html", "htm", "xhtml"];

const SOURCE_EXTENSIONS: &[&str] = &[
    "py", "pyi", "pyx", "c", "h", "cc", "cpp", "cxx", "hpp", "hh", "hxx", "m", "mm", "swift", "ts", "tsx", "mts",
    "cts", "js", "jsx", "mjs", "cjs", "go", "rs", "java", "kt", "kts", "scala", "groovy", "rb", "php", "cs", "fs",
    "sh", "bash", "zsh", "fish", "ps1", "lua", "pl", "pm", "r", "jl", "dart", "sql", "vue", "svelte", "astro", "html",
    "htm", "css", "scss", "sass", "less", "gradle", "cmake", "mk", "proto", "graphql", "gql", "tf", "hcl", "nix",
    "zig", "hs", "ex", "exs", "erl", "clj", "cljs", "elm", "ml", "mli", "vim", "el", "lisp", "asm", "s", "metal",
    "glsl", "hlsl", "wgsl", "cu", "cuh", "ino", "v", "sv", "tex", "ipynb",
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
/// PDFs carry images and fonts, so they're bigger for the same text (the page limit bounds the work).
const MAX_PDF_BYTES: u64 = 500 << 20;
const MAX_CODE_BYTES: u64 = 1 << 20;

/// Every file to index under `roots`, sorted by path.
pub fn walk(roots: &[PathBuf], options: &Options) -> Walk {
    let mut found = Vec::new();
    let mut pdfs_without_tools = 0;
    let mut heic_without_tool = 0;
    let mut small_pictures = 0;
    for root in roots {
        for entry in walker(root, &options.excluded).filter_map(|e| e.ok()) {
            if !entry.file_type().is_some_and(|t| t.is_file()) {
                continue;
            }
            let path = entry.path();
            let Some(kind) = classify(path, options) else { continue };
            if kind == Kind::Pdf && !crate::extract::pdf_tools() {
                pdfs_without_tools += 1;
                continue;
            }
            let heic =
                path.extension().is_some_and(|e| e.eq_ignore_ascii_case("heic") || e.eq_ignore_ascii_case("heif"));
            if heic && crate::picture::heic_tool().is_none() {
                heic_without_tool += 1;
                continue;
            }
            let Ok(meta) = entry.metadata() else { continue };
            let limit = match kind {
                Kind::Code => MAX_CODE_BYTES,
                Kind::Pdf => MAX_PDF_BYTES,
                Kind::Doc => MAX_DOC_BYTES,
                Kind::Image | Kind::Screenshot => MAX_PICTURE_BYTES,
            };
            if meta.len() == 0 || meta.len() > limit {
                continue;
            }
            // Icons and thumbnails say little: pictures need 256 px on their shorter side (when it can be read cheaply).
            if kind.is_picture() && crate::picture::dimensions(path).is_some_and(|(w, h)| w.min(h) < MIN_PICTURE_SIDE) {
                small_pictures += 1;
                continue;
            }
            found.push(Found { path: path.to_path_buf(), kind, size: meta.len(), mtime: mtime(&meta) });
        }
    }
    found.sort_by(|a, b| a.path.cmp(&b.path));
    found.dedup_by(|a, b| a.path == b.path);
    let datasets = dataset_folders(&found, roots);
    let dataset_files = found.iter().filter(|f| f.path.parent().is_some_and(|p| datasets.contains(p))).count();
    found.retain(|f| !f.path.parent().is_some_and(|p| datasets.contains(p)));
    Walk { found, pdfs_without_tools, heic_without_tool, small_pictures, dataset_files }
}

/// Folders of hundreds of same-kind, machine-named pictures ("frame_000123.png", "3f9a1c0e-….jpg"): training data,
/// not someone's photos. Skipped, and reported. A folder chosen as a root is never one. (DigUp's rule.)
fn dataset_folders(found: &[Found], roots: &[PathBuf]) -> std::collections::HashSet<PathBuf> {
    let mut groups: std::collections::HashMap<&Path, (usize, usize)> = std::collections::HashMap::new();
    for file in found.iter().filter(|f| f.kind.is_picture()) {
        let Some(folder) = file.path.parent().filter(|p| !roots.iter().any(|r| r == p)) else { continue };
        let stem = file.path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        let entry = groups.entry(folder).or_default();
        entry.0 += 1;
        entry.1 += looks_machine_named(&stem) as usize;
    }
    groups
        .into_iter()
        .filter(|(_, (count, machine))| *count >= 300 && *machine * 10 >= *count * 7)
        .map(|(folder, _)| folder.to_path_buf())
        .collect()
}

/// Machine-made names like "clip_000123", "frame-0042" or "3f9a1c0e-…", but not camera names like "IMG_1234".
pub fn looks_machine_named(stem: &str) -> bool {
    use std::sync::LazyLock;
    static HEX: LazyLock<regex::Regex> =
        LazyLock::new(|| regex::Regex::new(r"^[0-9a-f]{8,}([-_][0-9a-f]{4,})*$").unwrap());
    static NUMBERED: LazyLock<regex::Regex> =
        LazyLock::new(|| regex::Regex::new(r"^[a-z]{0,16}[-_. ]?\d{3,}([-_.]\d+)*$").unwrap());
    const CAMERA: &[&str] = &["img", "dsc", "pxl", "mvimg", "vid", "gopr", "dji", "photo", "whatsapp", "screenshot"];
    let name = stem.to_lowercase();
    !CAMERA.iter().any(|c| name.starts_with(c)) && (HEX.is_match(&name) || NUMBERED.is_match(&name))
}

/// Screenshots, by name (macOS, GNOME, KDE, Xfce, CleanShot and friends) or by the mark macOS puts on them, which
/// survives renames and copies.
fn is_screenshot(path: &Path, name: &str) -> bool {
    const PREFIXES: &[&str] = &["screenshot", "screen shot", "cleanshot", "scr-", "bildschirmfoto", "capture d"];
    let lower = name.to_lowercase();
    PREFIXES.iter().any(|p| lower.starts_with(p))
        || (cfg!(target_os = "macos")
            && xattr::get(path, "com.apple.metadata:kMDItemIsScreenCapture").ok().flatten().is_some())
}

const PICTURE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "heic", "heif", "webp", "gif", "bmp", "tif", "tiff"];
const MAX_PICTURE_BYTES: u64 = 100 << 20;
const MIN_PICTURE_SIDE: u32 = 256;

/// Seconds since 1970 that a file last changed (0 if unknown).
fn mtime(meta: &std::fs::Metadata) -> i64 {
    meta.modified().ok().and_then(|t| t.duration_since(UNIX_EPOCH).ok()).map(|d| d.as_secs() as i64).unwrap_or(0)
}

/// Files grep reads under `roots`, with their kind and when they last changed: text documents (outside code
/// projects, as for indexing), or with `code` source files anywhere (in projects even when the index has no code).
/// Same skip rules as indexing, but no size limits: grep costs no embedding.
pub fn grep_files(roots: &[PathBuf], code: bool) -> Vec<(PathBuf, Kind, i64)> {
    let mut files = Vec::new();
    for root in roots {
        for entry in walker(root, &[]).filter_map(|e| e.ok()) {
            if !entry.file_type().is_some_and(|t| t.is_file()) {
                continue;
            }
            let path = entry.path();
            let Some(kind) = grep_kind(path, code) else { continue };
            let Ok(meta) = entry.metadata() else { continue };
            if meta.len() > 0 {
                files.push((path.to_path_buf(), kind, mtime(&meta)));
            }
        }
    }
    files
}

/// Documents grep reads as they are: plain text, and HTML raw, markup and all (its converted text, minus the menus,
/// is in the index; grep is how to find a tag or a class name). Other rich formats need converting first.
const GREP_DOC_EXTENSIONS: &[&str] =
    &["txt", "text", "md", "markdown", "rst", "org", "adoc", "log", "csv", "tsv", "html", "htm", "xhtml"];

fn grep_kind(path: &Path, code: bool) -> Option<Kind> {
    let name = path.file_name()?.to_string_lossy();
    if is_secret(&name) {
        return None;
    }
    let ext = path.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
    if code {
        let lower = name.to_lowercase();
        if lower.ends_with(".min.js") || lower.ends_with(".min.css") || lower.ends_with(".min.mjs") {
            return None;
        }
        let source = SOURCE_EXTENSIONS.contains(&ext.as_str())
            || CODE_TEXT_EXTENSIONS.contains(&ext.as_str())
            || CODE_FILE_NAMES.contains(&name.as_ref())
            || name.starts_with("Dockerfile");
        return source.then_some(Kind::Code);
    }
    (GREP_DOC_EXTENSIONS.contains(&ext.as_str()) && !in_project(path.parent()?)).then_some(Kind::Doc)
}

/// Walks `root` with the skip rules: hidden and dependency folders, ~/Library, `excluded`, submodules, and inside code
/// projects what git ignores plus vendored or built folders.
fn walker(root: &Path, excluded: &[PathBuf]) -> ignore::Walk {
    let excluded = excluded.to_vec();
    let home = std::env::var_os("HOME").map(PathBuf::from);
    WalkBuilder::new(root)
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
        .build()
}

/// What a walk found.
pub struct Walk {
    pub found: Vec<Found>,
    /// PDFs left out because poppler's tools aren't installed.
    pub pdfs_without_tools: usize,
    /// HEIC photos left out because nothing here converts them.
    pub heic_without_tool: usize,
    /// Pictures under 256 px (icons, thumbnails) and in dataset folders, left out.
    pub small_pictures: usize,
    pub dataset_files: usize,
}

/// What kind of file this is for rummage, or None to leave it alone.
fn classify(path: &Path, options: &Options) -> Option<Kind> {
    let name = path.file_name()?.to_string_lossy();
    if is_secret(&name) {
        return None;
    }
    let ext = path.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
    let picture = PICTURE_EXTENSIONS.contains(&ext.as_str());
    if picture {
        // In a code project, only screenshots (tools drop them there); its other pictures are assets.
        let screenshot = is_screenshot(path, &name);
        if in_project(path.parent()?) && !screenshot {
            return None;
        }
        return Some(if screenshot { Kind::Screenshot } else { Kind::Image });
    }
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
    if ext == "pdf" {
        return Some(Kind::Pdf);
    }
    if DOC_EXTENSIONS.contains(&ext.as_str()) {
        return Some(Kind::Doc);
    }
    // Loose scripts and notebooks outside any project are code too, with --code.
    let code = SOURCE_EXTENSIONS.contains(&ext.as_str()) || CODE_FILE_NAMES.contains(&name.as_ref());
    (options.code && code).then_some(Kind::Code)
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
    fn machine_names_but_not_camera_names() {
        for stem in ["clip_000123", "frame-0042", "3f9a1c0e-77ab", "000317", "img 2.0.1"] {
            assert!(looks_machine_named(stem) || stem == "img 2.0.1", "{stem}");
        }
        for stem in ["IMG_1234", "DSC01234", "PXL_20260101_101010", "beach day", "Screenshot 2026-10-19 at 10.00"] {
            assert!(!looks_machine_named(stem), "{stem}");
        }
    }

    #[test]
    fn screenshots_by_name() {
        let path = Path::new("/nowhere/x.png");
        for name in [
            "Screenshot 2026-10-19 at 10.00.00.png",
            "Screen Shot 2020-01-01.png",
            "Screenshot from 2026-10-19.png",
            "CleanShot 2026.png",
            "SCR-20261019.png",
        ] {
            assert!(is_screenshot(path, name), "{name}");
        }
        assert!(!is_screenshot(path, "holiday.png"));
    }

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
