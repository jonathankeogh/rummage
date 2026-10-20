//! Getting the model: EmbeddingGemma 2 from Hugging Face, at the pinned revision, the first time rummage needs it.
//! It asks first (865 MB), downloads with `curl` (on every Mac and nearly every Linux), resuming a broken download,
//! and checks each file's SHA-256 before using it. Declined, it says where to put the files and how to get them.

use crate::paths::{self, MEDIA_MODEL, MODEL_REPOSITORY, MODEL_REVISION, ModelFiles, TEXT_MODEL};
use anyhow::{Context, Result, bail};
use std::io::{BufRead, IsTerminal, Write};
use std::path::Path;
use std::process::Command;

/// A file of the model, as Hugging Face serves it at the pinned revision.
pub struct ModelFile {
    pub name: &'static str,
    pub bytes: u64,
    pub sha256: &'static str,
    /// What it's for, for the question.
    pub what: &'static str,
}

pub const TEXT: ModelFile = ModelFile {
    name: TEXT_MODEL,
    bytes: 309_855_456,
    sha256: "2188ac1deca4b77dffefd603c2776a9d76d9d74ec01841392982ebb840b09135",
    what: "text",
};

pub const MEDIA: ModelFile = ModelFile {
    name: MEDIA_MODEL,
    bytes: 554_821_024,
    sha256: "c4a8a52691ecef40618438928bdf9e68379b854e24166f292592353db0aab64f",
    what: "pictures, sound and video",
};

/// Where Hugging Face serves a file of the pinned revision.
pub fn url(file: &ModelFile) -> String {
    format!("https://huggingface.co/{MODEL_REPOSITORY}/resolve/{MODEL_REVISION}/{}", file.name)
}

fn megabytes(bytes: u64) -> u64 {
    (bytes as f64 / 1e6).round() as u64
}

/// The model files there are, after downloading the missing ones if the person agrees: the text model always, the
/// picture and sound encoder too when `media` (the first download fetches both, so a second one doesn't surprise
/// anyone later). Without a terminal to ask in, or declined, it says how to get them instead.
pub fn ensure(media: bool) -> Result<ModelFiles> {
    let have = ModelFiles::locate().ok();
    let mut missing: Vec<&ModelFile> = Vec::new();
    if have.is_none() {
        missing.push(&TEXT);
        missing.push(&MEDIA);
    } else if media && have.as_ref().is_some_and(|m| m.media.is_none()) {
        missing.push(&MEDIA);
    }
    if missing.is_empty() {
        return Ok(have.expect("located"));
    }
    let folder = paths::download_folder();
    let interactive = std::io::stdin().is_terminal() && std::io::stderr().is_terminal();
    if !interactive || std::env::var_os("RUMMAGE_MODELS").is_some() {
        // Asked for later (an encoder for pictures): carry on without it, as before.
        if let Some(have) = have {
            eprintln!("{}", instructions(&missing, &folder));
            return Ok(have);
        }
        bail!("{}", instructions(&missing, &folder));
    }

    let total: u64 = missing.iter().map(|f| f.bytes).sum();
    eprintln!(
        "rummage needs its model to read your files: EmbeddingGemma 2 from Hugging Face (Apache 2.0, {MODEL_REPOSITORY})."
    );
    for file in &missing {
        eprintln!("  {} MB for {}", megabytes(file.bytes), file.what);
    }
    eprint!("Download {} MB now, into {}? [Y/n] ", megabytes(total), paths::tilde(&folder));
    std::io::stderr().flush()?;
    let mut answer = String::new();
    std::io::stdin().lock().read_line(&mut answer)?;
    if answer.trim().to_lowercase().starts_with('n') {
        eprintln!("\nOK. {}", instructions(&missing, &folder));
        if let Some(have) = have {
            return Ok(have);
        }
        bail!("no model yet");
    }
    std::fs::create_dir_all(&folder).with_context(|| format!("can't create {}", folder.display()))?;
    for file in &missing {
        fetch(file, &folder)?;
    }
    ModelFiles::locate()
}

/// Where the files go and how to get them, for someone who'd rather do it themselves.
pub fn instructions(missing: &[&ModelFile], folder: &Path) -> String {
    // Paths as a shell takes them in quotes: $HOME for the home folder.
    let quoted = |path: &Path| match std::env::var_os("HOME").map(std::path::PathBuf::from) {
        Some(home) if path.starts_with(&home) => {
            format!("\"$HOME/{}\"", path.strip_prefix(&home).unwrap_or(path).display())
        }
        _ => format!("\"{}\"", path.display()),
    };
    let mut text = format!(
        "rummage looks for the model in {} (or the folder RUMMAGE_MODELS names, or the Hugging Face cache). To put it \
         there yourself:\n\n    mkdir -p {}\n",
        paths::tilde(folder),
        quoted(folder)
    );
    for file in missing {
        text += &format!("    curl -L -C - -o {} {}\n", quoted(&folder.join(file.name)), url(file));
    }
    text += &format!(
        "\nor with Hugging Face's tool: uvx --from huggingface_hub hf download {MODEL_REPOSITORY} {} --revision \
         {MODEL_REVISION}",
        missing.iter().map(|f| f.name).collect::<Vec<_>>().join(" ")
    );
    text
}

/// Downloads one file into `folder` (resuming a part already there), checking its size and SHA-256.
fn fetch(file: &ModelFile, folder: &Path) -> Result<()> {
    let target = folder.join(file.name);
    let part = folder.join(format!("{}.part", file.name));
    let have = std::fs::metadata(&part).map_or(0, |m| m.len());
    if have < file.bytes {
        eprintln!("{} ({} MB)", file.name, megabytes(file.bytes));
        let status = Command::new("curl")
            .args(["--location", "--fail", "--retry", "3", "--continue-at", "-", "--progress-bar", "--output"])
            .arg(&part)
            .arg(url(file))
            .status()
            .context("can't run curl (install it, or download the files yourself: see `rummage status`)")?;
        if !status.success() {
            bail!("the download of {} stopped ({status}); run the command again to resume it", file.name);
        }
    }
    let size = std::fs::metadata(&part)?.len();
    if size != file.bytes {
        std::fs::remove_file(&part).ok();
        bail!("{} came out {size} bytes, not {}: deleted it, try again", file.name, file.bytes);
    }
    match sha256(&part) {
        Some(hash) if hash != file.sha256 => {
            std::fs::remove_file(&part).ok();
            bail!("{} isn't the file it should be (SHA-256 {hash}): deleted it, try again", file.name);
        }
        Some(_) => {}
        None => eprintln!("(no shasum or sha256sum here to check {}: its size is right)", file.name),
    }
    std::fs::rename(&part, &target)?;
    eprintln!("{}: done, checked", file.name);
    Ok(())
}

/// A file's SHA-256, through `sha256sum` (Linux) or `shasum` (macOS); None without either.
fn sha256(path: &Path) -> Option<String> {
    let output = Command::new("sha256sum").arg(path).output().ok().filter(|o| o.status.success()).or_else(|| {
        Command::new("shasum").args(["-a", "256"]).arg(path).output().ok().filter(|o| o.status.success())
    })?;
    String::from_utf8_lossy(&output.stdout).split_whitespace().next().map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn urls_and_instructions_name_the_pinned_files() {
        assert_eq!(
            url(&TEXT),
            "https://huggingface.co/ggml-org/embeddinggemma-2-GGUF/resolve/bfcd298762cc34d0357ece5ebdd31791a3a374d8/embeddinggemma-2-Q8_0.gguf"
        );
        let text = instructions(&[&TEXT, &MEDIA], Path::new("/srv/models"));
        assert!(text.contains("mkdir -p \"/srv/models\""), "{text}");
        assert!(text.contains("curl -L -C - -o \"/srv/models/embeddinggemma-2-Q8_0.gguf\""), "{text}");
        assert!(text.contains("mmproj-embeddinggemma-2-Q8_0.gguf"));
    }

    #[test]
    fn hashes_files() {
        let path = std::env::temp_dir().join(format!("rummage-sha-{}", std::process::id()));
        std::fs::write(&path, b"abc").unwrap();
        let hash = sha256(&path);
        std::fs::remove_file(&path).unwrap();
        if let Some(hash) = hash {
            assert_eq!(hash, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        }
    }
}
