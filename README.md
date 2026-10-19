# rummage

Instant multimodal semantic file search for the Unix terminal. Describe what you're looking for, and rummage finds it
by meaning and by exact words, locally, on any GPU or CPU. Type `rummage` for a yazi-style interface, or use it inside
yazi with the plugin.

**Status: milestones 1–6 of 10.** Search works end to end over text, Markdown, PDFs (page by page), Word
(docx), OpenDocument (odt) and HTML files, and, with `--code`, source code, through a background service that keeps
searches instant, with a search-as-you-type interface and live grep beside meaning search. Photos and screenshots
are found by what they show (and screenshots and scanned pages by their words, through OCR). Audio, video and the
yazi plugin are coming.

## Build

You need Rust, CMake, git and a C++ compiler (Xcode Command Line Tools on macOS, `build-essential` and `clang` on
Linux). For PDFs, install poppler (`brew install poppler`, `apt install poppler-utils`, `dnf install poppler-utils`);
for the words in screenshots and scans, tesseract (`brew install tesseract`, `apt install tesseract-ocr`); for iPhone
HEIC photos on Linux, libheif (`apt install libheif-examples`). Without them those parts are left out, and
`rummage index` says so.

```sh
git clone --recurse-submodules https://github.com/jonathankeogh/rummage && cd rummage
cargo build --release                    # macOS: Metal. Linux: CPU, or add --features cuda / --features vulkan
```

The model is [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) (Apache 2.0). Download it once: 310 MB
for text, and 555 MB more for pictures (without it, pictures wait):

```sh
uvx --from huggingface_hub hf download ggml-org/embeddinggemma-2-GGUF embeddinggemma-2-Q8_0.gguf \
  mmproj-embeddinggemma-2-Q8_0.gguf --revision bfcd298762cc34d0357ece5ebdd31791a3a374d8
```

## Use

```sh
rummage                                  # the interface: type to search, ↑↓ to move, ⏎ to open, F1 for help
                                         #   (^v on a PDF shows the page itself; RUMMAGE_PICTURES=halfblocks if
                                         #    your terminal doesn't answer its graphics query)
rummage index ~/Documents ~/notes        # add folders and index them (later: `rummage index` alone re-syncs)
rummage search "the clause about pets in the lease"
rummage search INV20931                  # exact words count too
rummage search "/INV\d{5}/"              # a regex: grep runs over text and code files beside meaning search
rummage index --code ~/code              # also read code in code projects...
rummage search "code: retry a failed call with backoff"   # ...and search it with "code:"
rummage status
```

Searches go through a small background service that keeps the model loaded, so after the first one they take about
10 ms. The first search starts it; it quits after 15 idle minutes and picks up re-indexing on its own. `rummage serve`
runs it in the foreground (`--linger 0` never quits), `rummage stop` stops it, and `rummage search --no-serve` skips
it. Its log is `serve.log` in the index folder.

`--json` on search prints results for scripts. The index lives in `$XDG_DATA_HOME/rummage` (or
`~/.local/share/rummage`); set `RUMMAGE_INDEX` or `--index` to use another. `RUMMAGE_MODELS` points at a folder with
the model files instead of the Hugging Face cache.

## What it reads

Text and Markdown, PDFs a page at a time (scanned pages as pictures, read by OCR), docx, odt and html, photos and
screenshots (JPEG, PNG, HEIC, WebP, GIF, BMP, TIFF; icons under 256 px and dataset folders of machine-named pictures
are left out), and with `--code` source code and notebooks, in projects and as loose scripts. Long files go in two
passes: the first 30 pages of a big PDF or 40 passages of a long document right away, the rest once everything else
is in, so a 500-page PDF can't hold up the first pass. Scanned PDFs (no text) wait for OCR.

Only the folders you add. It never reads hidden folders, keys, certificates or password files, dependency folders
(`node_modules`, `venv` …) or `~/Library`. Inside a code project (a folder with `.git`, `Cargo.toml`, `package.json`
…) it reads nothing unless you pass `--code`, and then skips what git ignores and vendored or built folders.

## How it works

Every search does two things at once. Meaning search: the query is embedded and compared with every passage
(passages that are a little similar to everything, like menus and boilerplate, are held back). Live grep: the query's
exact text (or a `/regex/`) in the text and code files themselves, newest first, up to 20 files or 3 seconds, so it
also finds files too big to index. One score orders both: meaning, plus a bonus for the query's words or a grep match,
plus a little for being recent. The interface shows meaning results at once and merges grep's as they arrive.

Text is cut into ~1,800-character passages, each embedded by EmbeddingGemma 2 through llama.cpp (pinned in
`vendor/llama.cpp`, built by `crates/llama-sys`). One SQLite file holds a vector per passage plus an FTS5 keyword
index. A search scores every vector, keeps passages that stand out for that query, and boosts files that contain the
query's words.

## Tests

`cargo test --release` runs the unit tests everywhere. With the model files present it also runs
`tests/reference_vectors.rs` (rummage's vectors must match DigUp's reference vectors, cosine ≥ 0.999) and
`tests/end_to_end.rs` (index, re-sync, long files in two passes, the service, grep, crash recovery); without them
(as on CI) those skip. `scripts/eval.py` measures ranking on DigUp's eval sets: build the text parts of its testbed
somewhere outside a code project (e.g. `~/.cache/rummage-eval`) and see the script's header.

## License

MIT (see [LICENSE](LICENSE)).

Credits: [DigUp](https://github.com/ARahim3/DigUp) for the approach, skip rules and ranking (see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)), [llama.cpp](https://github.com/ggml-org/llama.cpp) and
[yazi](https://github.com/sxyazi/yazi).
