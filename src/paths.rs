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
    /// `$RUMMAGE_MODELS` (and only there, when set), else the pinned revision in the Hugging Face cache.
    pub fn locate() -> Result<ModelFiles> {
        let folder = env_path("RUMMAGE_MODELS").unwrap_or_else(huggingface_snapshot);
        let text = folder.join(TEXT_MODEL);
        if !text.exists() {
            bail!(
                "no {TEXT_MODEL} in {}\nget it with: uvx --from huggingface_hub hf download {MODEL_REPOSITORY} \
                 {TEXT_MODEL} {MEDIA_MODEL} --revision {MODEL_REVISION}",
                folder.display()
            );
        }
        let media = Some(folder.join(MEDIA_MODEL)).filter(|p| p.exists());
        Ok(ModelFiles { text, media })
    }
}

/// `$HF_HUB_CACHE`, `$HF_HOME/hub` or `~/.cache/huggingface/hub`, then the pinned snapshot in it.
fn huggingface_snapshot() -> PathBuf {
    let hub = env_path("HF_HUB_CACHE")
        .or_else(|| env_path("HF_HOME").map(|h| h.join("hub")))
        .unwrap_or_else(|| home().join(".cache/huggingface/hub"));
    hub.join(format!("models--{}", MODEL_REPOSITORY.replace('/', "--"))).join("snapshots").join(MODEL_REVISION)
}

/// `~/x/y` for display.
pub fn tilde(path: &std::path::Path) -> String {
    let home = home();
    match path.strip_prefix(&home) {
        Ok(rest) if !home.as_os_str().is_empty() && home != PathBuf::from("/") => format!("~/{}", rest.display()),
        _ => path.display().to_string(),
    }
}
