# rummage

Instant multimodal semantic file search for the Unix terminal. Describe what you're looking for, and rummage finds it
by meaning and by exact words, locally, on any GPU or CPU. Type `rummage` for a yazi-style interface, or use it inside
yazi with the plugin.

**Status: milestone 1 of 9.** Text search works end to end: plain text, Markdown and (with `--code`) source code.
Images, PDFs, audio, video, the interface and the yazi plugin are coming.

## Build

You need Rust, CMake, git and a C++ compiler (Xcode Command Line Tools on macOS, `build-essential` and `clang` on
Linux).

```sh
git clone --recurse-submodules https://github.com/jonathankeogh/rummage && cd rummage
cargo build --release                    # macOS: Metal. Linux: CPU, or add --features cuda / --features vulkan
```

The model is [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) (Apache 2.0). Download it once (310 MB
for text; the 555 MB image and audio part comes later):

```sh
uvx --from huggingface_hub hf download ggml-org/embeddinggemma-2-GGUF embeddinggemma-2-Q8_0.gguf \
  --revision bfcd298762cc34d0357ece5ebdd31791a3a374d8
```

## Use

```sh
rummage index ~/Documents ~/notes        # add folders and index them (later: `rummage index` alone re-syncs)
rummage search "the clause about pets in the lease"
rummage search INV20931                  # exact words count too
rummage index --code ~/code              # also read code in code projects...
rummage search "code: retry a failed call with backoff"   # ...and search it with "code:"
rummage status
```

`--json` on search prints results for scripts. The index lives in `$XDG_DATA_HOME/rummage` (or
`~/.local/share/rummage`); set `RUMMAGE_INDEX` or `--index` to use another. `RUMMAGE_MODELS` points at a folder with
the model files instead of the Hugging Face cache.

## What it reads

Only the folders you add. It never reads hidden folders, keys, certificates or password files, dependency folders
(`node_modules`, `venv` …) or `~/Library`. Inside a code project (a folder with `.git`, `Cargo.toml`, `package.json`
…) it reads nothing unless you pass `--code`, and then skips what git ignores and vendored or built folders.

## How it works

Text is cut into ~1,800-character passages, each embedded by EmbeddingGemma 2 through llama.cpp (pinned in
`vendor/llama.cpp`, built by `crates/llama-sys`). One SQLite file holds a vector per passage plus an FTS5 keyword
index. A search scores every vector, keeps passages that stand out for that query, and boosts files that contain the
query's words.

Credits: [DigUp](https://github.com/ARahim3/DigUp) for the approach, skip rules and ranking (see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)), [llama.cpp](https://github.com/ggml-org/llama.cpp) and
[yazi](https://github.com/sxyazi/yazi).
