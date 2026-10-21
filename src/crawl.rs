//! Finds the files to index under the chosen folders, and what kind each is.
//!
//! Skip rules follow DigUp's (MIT), learned on a real developer's Mac: never hidden folders, keys or password files,
//! dependency folders or caches; inside a code project (a folder with .git, Cargo.toml, package.json …) nothing but
//! code, and code only with `--code`, minus what git ignores and vendored or built folders.

use crate::store::Kind;
use ignore::WalkBuilder;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
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
    /// Also songs, music videos and album art in music folders (left out unless asked for: a music library is days
    /// of sound to embed, rarely what's searched for).
    pub music: bool,
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

/// Every file to index under `roots`, sorted by path. `looking` hears how many have been found so far, now and then.
pub fn walk(roots: &[PathBuf], options: &Options, looking: &mut dyn FnMut(usize)) -> Walk {
    let mut found: Vec<Found> = Vec::new();
    let mut music_skipped = 0;
    let mut bad_names = 0;
    let music_folders = music_folders();
    let mut pdfs_without_tools = 0;
    let mut heic_without_tool = 0;
    let mut media_without_ffmpeg = 0;
    let mut small_pictures = 0;
    let mut cloud = Vec::new();
    let placeholders = Arc::new(AtomicUsize::new(0));
    let mut remote_folders = Vec::new();
    let mut refused: Vec<PathBuf> = Vec::new();
    let mut unreadable: Vec<(PathBuf, String)> = Vec::new();
    for root in roots {
        let remote = remote_mounts_below(root, &remote_mounts());
        remote_folders.extend(remote.iter().cloned());
        // A folder chosen inside a music folder is read whole: it was asked for.
        let skip_music = !options.music && !music_folders.iter().any(|m| root.starts_with(m));
        let walk = walker(root, &options.excluded, &remote, Some(placeholders.clone()));
        for entry in walk {
            let entry = match entry {
                Ok(entry) => entry,
                Err(error) => {
                    // A folder that can't be listed (no permission, a drive that errored): its files are kept as they
                    // were, not taken for deleted. macOS keeps the Photos library private until the terminal has Full
                    // Disk Access.
                    if let Some((path, why)) = unreadable_path(&error) {
                        match photos_library(&path) {
                            Some(library) if !refused.contains(&library.to_path_buf()) => {
                                refused.push(library.to_path_buf())
                            }
                            Some(_) => {}
                            None => unreadable.push((path, why)),
                        }
                    }
                    continue;
                }
            };
            if !entry.file_type().is_some_and(|t| t.is_file()) {
                continue;
            }
            let path = entry.path();
            // A name that isn't text (bytes from another system's encoding) can't be kept in the index as itself.
            if path.to_str().is_none() {
                bad_names += 1;
                continue;
            }
            // In the cloud only: reading anything of it (even a picture's size) would download it.
            let Ok(meta) = entry.metadata() else { continue };
            if in_cloud_only(&meta) {
                if indexable_name(path) {
                    cloud.push(Found {
                        path: path.to_path_buf(),
                        kind: Kind::Doc,
                        size: meta.len(),
                        mtime: mtime(&meta),
                    });
                }
                continue;
            }
            let Some(kind) = classify(path, options) else { continue };
            if skip_music && (kind.is_timed() || kind.is_picture()) && in_music_library(path, &music_folders) {
                music_skipped += 1;
                continue;
            }
            if kind == Kind::Pdf && !crate::extract::pdf_tools() {
                pdfs_without_tools += 1;
                continue;
            }
            if kind.is_timed() && !crate::media::available() {
                media_without_ffmpeg += 1;
                continue;
            }
            let heic =
                path.extension().is_some_and(|e| e.eq_ignore_ascii_case("heic") || e.eq_ignore_ascii_case("heif"));
            if heic && crate::picture::heic_tool().is_none() {
                heic_without_tool += 1;
                continue;
            }
            let limit = match kind {
                Kind::Code => MAX_CODE_BYTES,
                Kind::Pdf => MAX_PDF_BYTES,
                Kind::Doc => MAX_DOC_BYTES,
                Kind::Image | Kind::Screenshot => MAX_PICTURE_BYTES,
                Kind::Audio | Kind::Video => u64::MAX,
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
            if found.len().is_multiple_of(500) {
                looking(found.len());
            }
        }
    }
    found.sort_by(|a, b| a.path.cmp(&b.path));
    found.dedup_by(|a, b| a.path == b.path);
    let datasets = dataset_folders(&found, roots);
    let dataset_files = found.iter().filter(|f| f.path.parent().is_some_and(|p| datasets.contains(p))).count();
    found.retain(|f| !f.path.parent().is_some_and(|p| datasets.contains(p)));
    Walk {
        found,
        pdfs_without_tools,
        heic_without_tool,
        media_without_ffmpeg,
        small_pictures,
        dataset_files,
        cloud,
        placeholders: placeholders.load(Ordering::Relaxed),
        remote_folders,
        refused,
        unreadable,
        music_skipped,
        bad_names,
    }
}

/// The music folders: ~/Music, and on Linux the one ~/.config/user-dirs.dirs names (XDG_MUSIC_DIR="$HOME/Musik").
fn music_folders() -> Vec<PathBuf> {
    let Some(home) = std::env::var_os("HOME").map(PathBuf::from) else { return Vec::new() };
    let mut folders = vec![home.join("Music")];
    let dirs = std::fs::read_to_string(home.join(".config/user-dirs.dirs")).unwrap_or_default();
    for line in dirs.lines() {
        if let Some(value) = line.strip_prefix("XDG_MUSIC_DIR=") {
            let value = value.trim_matches('"').replace("$HOME", &home.to_string_lossy());
            folders.push(PathBuf::from(value));
        }
    }
    folders
}

/// In a music folder, or in an Apple Music or iTunes library wherever it is.
fn in_music_library(path: &Path, music_folders: &[PathBuf]) -> bool {
    music_folders.iter().any(|m| path.starts_with(m))
        || path.ancestors().any(|a| {
            let name = a.file_name().map(|n| n.to_string_lossy()).unwrap_or_default();
            name == "iTunes Media" || name == "Media.localized" || name.ends_with(".musiclibrary")
        })
}

/// The system's own folders (and the root of everything): never indexed, as they're not someone's files.
pub fn is_system_folder(path: &Path) -> bool {
    const SYSTEM: &[&str] = &[
        "/System",
        "/Library",
        "/Applications",
        "/usr",
        "/bin",
        "/sbin",
        "/etc",
        "/private/etc",
        "/opt",
        "/boot",
        "/dev",
        "/proc",
        "/sys",
        "/run",
        "/lib",
        "/lib32",
        "/lib64",
        "/snap",
        "/nix",
        "/cores",
    ];
    path == Path::new("/") || SYSTEM.iter().any(|s| path.starts_with(s))
}

/// The Apple Photos library a path is in, if any ("~/Pictures/Photos Library.photoslibrary").
pub fn photos_library(path: &Path) -> Option<&Path> {
    path.ancestors().find(|a| a.extension().is_some_and(|e| e == "photoslibrary"))
}

/// Inside a Photos library only its originals are read ("originals/", or "Masters/" in libraries before 2019): the
/// rest is thumbnails, edited copies and its database.
const PHOTOS_ORIGINALS: &[&str] = &["originals", "Masters"];

/// The path a walk couldn't read, and why ("permission denied"): a folder it couldn't list, usually.
fn unreadable_path(error: &ignore::Error) -> Option<(PathBuf, String)> {
    let why = error.io_error()?;
    let why = match why.kind() {
        std::io::ErrorKind::PermissionDenied => "permission denied".to_string(),
        _ => why.to_string(),
    };
    let mut error = error;
    loop {
        match error {
            ignore::Error::WithPath { path, .. } => return Some((path.clone(), why)),
            ignore::Error::WithDepth { err, .. } | ignore::Error::WithLineNumber { err, .. } => error = err,
            _ => return None,
        }
    }
}

/// The file's content isn't on this disk, only in the cloud: macOS marks such files "dataless" (iCloud Drive with
/// Optimize Mac Storage; Dropbox, OneDrive and Google Drive through File Provider). Reading one downloads it.
pub fn in_cloud_only(meta: &std::fs::Metadata) -> bool {
    #[cfg(target_os = "macos")]
    {
        use std::os::macos::fs::MetadataExt;
        const SF_DATALESS: u32 = 0x4000_0000;
        meta.st_flags() & SF_DATALESS != 0
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = meta;
        false
    }
}

/// Whether a file's name says it's something rummage reads (without reading it), for counting cloud-only files.
fn indexable_name(path: &Path) -> bool {
    let ext = path.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
    let ext = ext.as_str();
    ext == "pdf"
        || DOC_EXTENSIONS.contains(&ext)
        || PICTURE_EXTENSIONS.contains(&ext)
        || AUDIO_EXTENSIONS.contains(&ext)
        || VIDEO_EXTENSIONS.contains(&ext)
}

/// Network and cloud file systems: reading a file there fetches it over the network (Linux's mount types).
const REMOTE_FILE_SYSTEMS: &[&str] = &[
    "nfs",
    "nfs4",
    "cifs",
    "smb3",
    "smbfs",
    "afs",
    "ceph",
    "glusterfs",
    "davfs",
    "fuse.rclone",
    "fuse.sshfs",
    "fuse.s3fs",
    "fuse.gcsfuse",
    "fuse.goofys",
    "fuse.onedriver",
    "fuse.gvfsd-fuse",
    "fuse.davfs2",
    "fuse.dropbox",
    // macOS's
    "smbfs",
    "afpfs",
    "webdav",
];

/// Where network and cloud file systems are mounted.
fn remote_mounts() -> Vec<PathBuf> {
    mounts().into_iter().filter(|(_, kind)| REMOTE_FILE_SYSTEMS.contains(&kind.as_str())).map(|(p, _)| p).collect()
}

/// The network file system `path` is on ("nfs", "smbfs" …), if it's on one.
pub fn network_file_system(path: &Path) -> Option<String> {
    let path = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    let (_, kind) =
        mounts().into_iter().filter(|(point, _)| path.starts_with(point)).max_by_key(|(p, _)| p.as_os_str().len())?;
    REMOTE_FILE_SYSTEMS.contains(&kind.as_str()).then_some(kind)
}

/// Every mount: where, and its file system's type. Linux says in /proc/self/mountinfo; macOS and the BSDs in `mount`'s
/// output.
fn mounts() -> Vec<(PathBuf, String)> {
    if let Ok(text) = std::fs::read_to_string("/proc/self/mountinfo") {
        return mounts_in_mountinfo(&text);
    }
    let output = std::process::Command::new("mount").output().map(|o| String::from_utf8_lossy(&o.stdout).into_owned());
    mounts_in_mount_output(&output.unwrap_or_default())
}

/// The mounts in a /proc/self/mountinfo: "id parent dev root mountpoint options … - type source …".
fn mounts_in_mountinfo(mountinfo: &str) -> Vec<(PathBuf, String)> {
    mountinfo
        .lines()
        .filter_map(|line| {
            let (before, after) = line.split_once(" - ")?;
            let point = before.split(' ').nth(4)?;
            let kind = after.split(' ').next()?;
            Some((PathBuf::from(unescape_mount(point)), kind.to_string()))
        })
        .collect()
}

/// The mounts in `mount`'s output: "//sam@nas/share on /Volumes/share (smbfs, nodev, nosuid, mounted by sam)".
fn mounts_in_mount_output(output: &str) -> Vec<(PathBuf, String)> {
    output
        .lines()
        .filter_map(|line| {
            let (before, options) = line.rsplit_once(" (")?;
            let (_, point) = before.split_once(" on ")?;
            let kind = options.split([',', ')']).next()?.trim();
            Some((PathBuf::from(point), kind.to_string()))
        })
        .collect()
}

/// The remote mount points in a /proc/self/mountinfo.
#[cfg(test)]
fn remote_mounts_in(mountinfo: &str) -> Vec<PathBuf> {
    mounts_in_mountinfo(mountinfo)
        .into_iter()
        .filter(|(_, kind)| REMOTE_FILE_SYSTEMS.contains(&kind.as_str()))
        .map(|(p, _)| p)
        .collect()
}

/// Mount points escape spaces, tabs, newlines and backslashes as octal (\040).
fn unescape_mount(point: &str) -> String {
    let bytes = point.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'\\' && i + 3 < bytes.len() && bytes[i + 1..i + 4].iter().all(|b| (b'0'..=b'7').contains(b)) {
            out.push(u8::from_str_radix(&point[i + 1..i + 4], 8).unwrap_or(b'?'));
            i += 4;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// The remote mounts inside `root` that a walk of it leaves out. A root that is itself on one (or is one) was chosen
/// on purpose, so what's inside it is read.
fn remote_mounts_below(root: &Path, mounts: &[PathBuf]) -> Vec<PathBuf> {
    if mounts.iter().any(|m| root.starts_with(m)) {
        return Vec::new();
    }
    mounts.iter().filter(|m| m.starts_with(root)).cloned().collect()
}

/// Folders of hundreds of same-kind, machine-named pictures, clips or videos ("frame_000123.png", "clip_0042.wav",
/// "3f9a1c0e-….jpg"): training data, not someone's files. Skipped, and reported. A folder chosen as a root is never
/// one. (DigUp's rule.)
fn dataset_folders(found: &[Found], roots: &[PathBuf]) -> std::collections::HashSet<PathBuf> {
    let mut groups: std::collections::HashMap<(&Path, Kind), (usize, usize)> = std::collections::HashMap::new();
    // A Photos library names its originals by code, but they're someone's photos.
    for file in found.iter().filter(|f| (f.kind.is_picture() || f.kind.is_timed()) && photos_library(&f.path).is_none())
    {
        let Some(folder) = file.path.parent().filter(|p| !roots.iter().any(|r| r == p)) else { continue };
        let stem = file.path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        let entry = groups.entry((folder, file.kind)).or_default();
        entry.0 += 1;
        entry.1 += looks_machine_named(&stem) as usize;
    }
    groups
        .into_iter()
        .filter(|(_, (count, machine))| *count >= 300 && *machine * 10 >= *count * 7)
        .map(|((folder, _), _)| folder.to_path_buf())
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

const AUDIO_EXTENSIONS: &[&str] =
    &["mp3", "m4a", "aac", "wav", "flac", "ogg", "oga", "opus", "wma", "aiff", "aif", "caf"];
/// .ts / .mts / .m2ts count only when the bytes say MPEG transport stream (else they're TypeScript).
const VIDEO_EXTENSIONS: &[&str] = &["mp4", "mov", "m4v", "mkv", "webm", "avi", "wmv", "flv", "3gp", "mpg", "mpeg"];
const TRANSPORT_EXTENSIONS: &[&str] = &["ts", "mts", "m2ts"];

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
        let remote = remote_mounts_below(root, &remote_mounts());
        for entry in walker(root, &[], &remote, None).filter_map(|e| e.ok()) {
            if !entry.file_type().is_some_and(|t| t.is_file()) {
                continue;
            }
            let path = entry.path();
            let Some(kind) = grep_kind(path, code) else { continue };
            let Ok(meta) = entry.metadata() else { continue };
            if path.to_str().is_none() {
                continue;
            }
            // Never downloads a file to search it.
            if meta.len() > 0 && !in_cloud_only(&meta) {
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

/// Walks `root` with the skip rules: hidden files and folders, dependency folders, ~/Library, `excluded`, `remote`
/// mounts, submodules, and inside code projects what git ignores plus vendored or built folders. Old-style iCloud
/// placeholders (".Report.pdf.icloud", hidden) are counted into `placeholders`.
fn walker(
    root: &Path,
    excluded: &[PathBuf],
    remote: &[PathBuf],
    placeholders: Option<Arc<AtomicUsize>>,
) -> ignore::Walk {
    let excluded = excluded.to_vec();
    let remote = remote.to_vec();
    let home = std::env::var_os("HOME").map(PathBuf::from);
    WalkBuilder::new(root)
        .hidden(false)
        .git_ignore(true)
        .git_global(false)
        .git_exclude(true)
        .parents(true)
        .follow_links(false)
        .filter_entry(move |entry| {
            let path = entry.path();
            if entry.depth() == 0 {
                return true;
            }
            let name = entry.file_name().to_string_lossy();
            let is_dir = entry.file_type().is_some_and(|t| t.is_dir());
            if path.parent().is_some_and(|p| p.extension().is_some_and(|e| e == "photoslibrary")) {
                return is_dir && PHOTOS_ORIGINALS.contains(&name.as_ref());
            }
            if name.starts_with('.') {
                if name.ends_with(".icloud")
                    && let Some(count) = &placeholders
                {
                    count.fetch_add(1, Ordering::Relaxed);
                }
                return false;
            }
            if !is_dir {
                return true;
            }
            // Lightroom's previews: copies of photos that are indexed where they are.
            // And Safari's downloads still coming in.
            if remote.iter().any(|m| path == m) || name.ends_with(".lrdata") || name.ends_with(".download") {
                return false;
            }
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
    /// Sound and video left out because ffmpeg isn't installed.
    pub media_without_ffmpeg: usize,
    /// Pictures under 256 px (icons, thumbnails) and in dataset folders, left out.
    pub small_pictures: usize,
    pub dataset_files: usize,
    /// Files in the cloud only (not downloaded), not read: their kind isn't known, so it's Doc.
    pub cloud: Vec<Found>,
    /// Old-style iCloud placeholder files.
    pub placeholders: usize,
    /// Network and cloud mounts inside the folders, left out.
    pub remote_folders: Vec<PathBuf>,
    /// Photos libraries macOS didn't let it read.
    pub refused: Vec<PathBuf>,
    /// Other places it couldn't read, and why: what's indexed under them stays.
    pub unreadable: Vec<(PathBuf, String)>,
    /// Songs, music videos and album art left out of music folders (`Options::music` reads them).
    pub music_skipped: usize,
    /// Files whose names aren't text (another system's encoding), left out.
    pub bad_names: usize,
}

/// What kind of file this is for rummage, or None to leave it alone.
fn classify(path: &Path, options: &Options) -> Option<Kind> {
    let name = path.file_name()?.to_string_lossy();
    if is_secret(&name) {
        return None;
    }
    let ext = path.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
    // Sound and video, anywhere but inside code projects (there they're assets or fixtures).
    let timed = if AUDIO_EXTENSIONS.contains(&ext.as_str()) {
        Some(Kind::Audio)
    } else if VIDEO_EXTENSIONS.contains(&ext.as_str())
        || (TRANSPORT_EXTENSIONS.contains(&ext.as_str()) && crate::media::is_mpeg_transport_stream(path))
    {
        Some(Kind::Video)
    } else {
        None
    };
    if let Some(kind) = timed {
        return (!in_project(path.parent()?)).then_some(kind);
    }
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

    #[test]
    fn network_and_cloud_mounts_are_found_and_left_out_unless_chosen() {
        let mountinfo = "\
22 1 259:2 / / rw,relatime shared:1 - ext4 /dev/nvme0n1p2 rw
40 22 0:35 / /home/sam/Drive rw,nosuid shared:20 - fuse.rclone gdrive: rw,user_id=1000
41 22 0:36 / /home/sam/My\\040Server rw shared:21 - fuse.sshfs sam@box:/ rw
42 22 0:37 / /mnt/nas rw shared:22 - nfs4 nas:/export rw
43 22 0:38 / /home/sam/usb rw shared:23 - vfat /dev/sda1 rw
";
        let mounts = remote_mounts_in(mountinfo);
        assert_eq!(
            mounts,
            vec![PathBuf::from("/home/sam/Drive"), PathBuf::from("/home/sam/My Server"), PathBuf::from("/mnt/nas")]
        );
        // Inside home: left out. Chosen as a folder itself, or a folder inside it: read.
        assert_eq!(
            remote_mounts_below(Path::new("/home/sam"), &mounts),
            vec![PathBuf::from("/home/sam/Drive"), PathBuf::from("/home/sam/My Server")]
        );
        assert!(remote_mounts_below(Path::new("/home/sam/Drive"), &mounts).is_empty());
        assert!(remote_mounts_below(Path::new("/home/sam/Drive/Work"), &mounts).is_empty());
    }

    #[test]
    fn photos_library_originals_are_not_a_dataset() {
        let found = |folder: &str| -> Vec<Found> {
            (0..400)
                .map(|i| Found {
                    path: PathBuf::from(format!("{folder}/{i:08X}-0B7D-4E5A-9C3B-2D1E0F4A5B6C.jpeg").to_lowercase()),
                    kind: Kind::Image,
                    size: 1,
                    mtime: 0,
                })
                .collect()
        };
        let roots = [PathBuf::from("/home")];
        assert_eq!(dataset_folders(&found("/home/training/frames"), &roots).len(), 1);
        assert!(dataset_folders(&found("/home/Pictures/Photos Library.photoslibrary/originals/4"), &roots).is_empty());
    }

    #[test]
    fn system_folders_are_not_indexed() {
        for system in ["/", "/usr", "/usr/share/doc", "/System/Library", "/Applications/Safari.app", "/etc"] {
            assert!(is_system_folder(Path::new(system)), "{system}");
        }
        for mine in ["/Users/sam", "/home/sam/notes", "/Volumes/Photos", "/mnt/nas", "/private/var/folders/x", "/srv"] {
            assert!(!is_system_folder(Path::new(mine)), "{mine}");
        }
    }

    #[test]
    fn macos_mounts_are_read_from_mount() {
        let output = "/dev/disk3s1s1 on / (apfs, sealed, local, read-only, journaled)
//sam@nas._smb._tcp.local/Photos on /Volumes/Photos (smbfs, nodev, nosuid, mounted by sam)
nas:/export/home on /Users/sam/Shared Home (nfs, asynchronous)
";
        let mounts = mounts_in_mount_output(output);
        assert_eq!(mounts[1], (PathBuf::from("/Volumes/Photos"), "smbfs".to_string()));
        assert_eq!(mounts[2], (PathBuf::from("/Users/sam/Shared Home"), "nfs".to_string()));
        assert!(mounts.iter().filter(|(_, k)| REMOTE_FILE_SYSTEMS.contains(&k.as_str())).count() == 2);
    }

    #[test]
    fn music_libraries_are_left_out_unless_asked_for() {
        let dir = std::env::temp_dir().join(format!("rummage-music-{}", std::process::id()));
        let album = dir.join("Music Library/Media.localized/Nina Simone/Pastel Blues");
        std::fs::create_dir_all(&album).unwrap();
        std::fs::write(album.join("01 Be My Husband.mp3"), "ID3 not really a song").unwrap();
        std::fs::write(dir.join("voice memo.m4a"), "not really a recording").unwrap();
        let skipped = walk(std::slice::from_ref(&dir), &Options::default(), &mut |_| {});
        assert_eq!(skipped.music_skipped, 1);
        let asked = walk(std::slice::from_ref(&dir), &Options { music: true, ..Options::default() }, &mut |_| {});
        assert_eq!(asked.music_skipped, 0);
        assert!(in_music_library(Path::new("/home/sam/Musik/a.flac"), &[PathBuf::from("/home/sam/Musik")]));
        assert!(!in_music_library(Path::new("/home/sam/Podcasts/a.mp3"), &[PathBuf::from("/home/sam/Musik")]));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn names_that_arent_text_are_left_out() {
        use std::os::unix::ffi::OsStrExt;
        let dir = std::env::temp_dir().join(format!("rummage-names-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // "café.md" in Latin-1, as an old Windows zip leaves it. macOS won't create such a name; Linux will.
        let latin1 = dir.join(std::ffi::OsStr::from_bytes(b"caf\xe9.md"));
        if std::fs::write(&latin1, "Notes from the café.").is_err() {
            eprintln!("skipped: this file system only takes UTF-8 names");
            std::fs::remove_dir_all(&dir).unwrap();
            return;
        }
        std::fs::write(dir.join("menu.md"), "The menu.").unwrap();
        let walk = walk(std::slice::from_ref(&dir), &Options::default(), &mut |_| {});
        assert_eq!((walk.found.len(), walk.bad_names), (1, 1));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn hidden_files_are_skipped_and_icloud_placeholders_counted() {
        let dir = std::env::temp_dir().join(format!("rummage-crawl-{}", std::process::id()));
        std::fs::create_dir_all(dir.join(".hidden")).unwrap();
        std::fs::write(dir.join("note.md"), "A note.").unwrap();
        std::fs::write(dir.join(".secret-note.md"), "Hidden.").unwrap();
        std::fs::write(dir.join(".hidden/inside.md"), "Hidden too.").unwrap();
        std::fs::write(dir.join(".Report.pdf.icloud"), "placeholder").unwrap();
        let walk = walk(std::slice::from_ref(&dir), &Options::default(), &mut |_| {});
        let names: Vec<_> = walk.found.iter().map(|f| f.path.file_name().unwrap().to_owned()).collect();
        assert_eq!(names, vec!["note.md"]);
        assert_eq!(walk.placeholders, 1);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
