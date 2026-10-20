//! Where rummage keeps things. The same on every Unix: XDG locations, with the Hugging Face cache for the model.

use anyhow::{Result, bail};
use std::path::PathBuf;

/// ggml-org's Q8_0 conversion of EmbeddingGemma 2 at one pinned revision (never F16: the model's activations overflow
/// float16). The same files DigUp pins.
pub const MODEL_REPOSITORY: &str = "ggml-org/embeddinggemma-2-GGUF";
pub const MODEL_REVISION: &str = "bfcd298762cc34d0357ece5ebdd31791a3a374d8";
pub const TEXT_MODEL: &str = "embeddinggemma-2-Q8_0.gguf";
pub const MEDIA_MODEL: &str = "mmproj-embeddinggemma-2-Q8_0.gguf";

fn home() -> PathBuf {
    std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("/"))
}

/// Whether a program of this name is on PATH.
pub fn on_path(program: &str) -> bool {
    std::env::var_os("PATH").is_some_and(|path| std::env::split_paths(&path).any(|dir| dir.join(program).is_file()))
}

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name).filter(|v| !v.is_empty()).map(PathBuf::from)
}

/// The index folder: `$RUMMAGE_INDEX`, else `$XDG_DATA_HOME/rummage`, else `~/.local/share/rummage`.
pub fn index_dir() -> PathBuf {
    env_path("RUMMAGE_INDEX")
        .or_else(|| env_path("XDG_DATA_HOME").map(|d| d.join("rummage")))
        .unwrap_or_else(|| home().join(".local/share/rummage"))
}

/// The model files rummage runs.
pub struct ModelFiles {
    pub text: PathBuf,
    /// The image and audio encoders; not needed for text.
    pub media: Option<PathBuf>,
}

impl ModelFiles {
    /// `$RUMMAGE_MODELS` (and only there, when set); else the pinned revision in the Hugging Face cache, where
    /// Hugging Face's own tool puts it, then rummage's download folder (`download_folder`).
    pub fn locate() -> Result<ModelFiles> {
        let folders = match env_path("RUMMAGE_MODELS") {
            Some(folder) => vec![folder],
            None => vec![huggingface_snapshot(), download_folder()],
        };
        let Some(folder) = folders.iter().find(|f| f.join(TEXT_MODEL).exists()) else {
            let wanted = [&crate::download::TEXT, &crate::download::MEDIA];
            let folder = folders.last().expect("a folder");
            bail!("no model yet\n{}", crate::download::instructions(&wanted, folder));
        };
        let text = folder.join(TEXT_MODEL);
        // The encoder may have come later, into the other folder.
        let media = folders.iter().map(|f| f.join(MEDIA_MODEL)).find(|p| p.exists());
        Ok(ModelFiles { text, media })
    }
}

/// Where rummage downloads the model: `$XDG_DATA_HOME/rummage/models` or `~/.local/share/rummage/models`, one folder
/// per revision (whatever `RUMMAGE_INDEX` says: the model serves every index).
pub fn download_folder() -> PathBuf {
    env_path("RUMMAGE_MODELS").unwrap_or_else(|| {
        env_path("XDG_DATA_HOME")
            .unwrap_or_else(|| home().join(".local/share"))
            .join("rummage/models")
            .join(MODEL_REVISION)
    })
}

/// `$HF_HUB_CACHE`, `$HF_HOME/hub` or `~/.cache/huggingface/hub`, then the pinned snapshot in it.
fn huggingface_snapshot() -> PathBuf {
    let hub = env_path("HF_HUB_CACHE")
        .or_else(|| env_path("HF_HOME").map(|h| h.join("hub")))
        .unwrap_or_else(|| home().join(".cache/huggingface/hub"));
    hub.join(format!("models--{}", MODEL_REPOSITORY.replace('/', "--"))).join("snapshots").join(MODEL_REVISION)
}

/// `~/x/y` for display.
/// A file's name for result lists: a Photos library original is "Photos Library › 4F2A…E1.heic" (its own name is a
/// code, and its folder is deep inside the library).
pub fn display_name(path: &std::path::Path) -> String {
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    match crate::crawl::photos_library(path).and_then(|l| l.file_stem()) {
        Some(library) => format!("{} › {name}", library.to_string_lossy()),
        None => name,
    }
}

/// Where other copies are, by folder: "~/Downloads, ~/Backup".
pub fn folders(paths: &[String]) -> String {
    let mut folders: Vec<String> = Vec::new();
    for path in paths {
        let path = std::path::Path::new(path);
        let folder = tilde(path.parent().unwrap_or(path));
        if !folders.contains(&folder) {
            folders.push(folder);
        }
    }
    folders.join(", ")
}

pub fn tilde(path: &std::path::Path) -> String {
    let home = home();
    match path.strip_prefix(&home) {
        Ok(rest) if !home.as_os_str().is_empty() && home != PathBuf::from("/") => format!("~/{}", rest.display()),
        _ => path.display().to_string(),
    }
}
