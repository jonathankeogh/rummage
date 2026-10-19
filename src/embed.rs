//! EmbeddingGemma 2 on llama.cpp: the text model's mean-pooled, L2-normalized output is the vector.
//!
//! Inputs are framed as the reference frames them (`<bos> text <eos>`). The model is non-causal with no KV cache, so
//! each batch of inputs goes through one `llama_process` call, one sequence per input. Ported from DigUp's
//! LlamaEmbedder (MIT); images and audio come in a later milestone through the same batch.

use crate::paths::ModelFiles;
use anyhow::{Context, Result, bail};
use rummage_llama_sys as sys;
use std::ffi::{CString, c_char, c_void};
use std::sync::Once;
use std::sync::atomic::{AtomicU32, Ordering};

/// A picture for the image encoder.
pub enum Picture {
    /// A file llama.cpp decodes itself (see `picture::stb_reads`).
    File(std::path::PathBuf),
    /// Pixels decoded here.
    Pixels(crate::picture::Rgb),
}

/// Every vector has this many numbers.
pub const DIM: usize = 768;

/// EmbeddingGemma 2's task prompts (model card).
pub mod prompts {
    pub fn query(text: &str) -> String {
        format!("task: search result | query: {text}")
    }

    pub fn document(title: Option<&str>, text: &str) -> String {
        let title = title.filter(|t| !t.is_empty()).unwrap_or("none");
        format!("title: {title} | text: {text}")
    }
}

pub struct Embedder {
    model: *mut sys::llama_model,
    context: *mut sys::llama_context,
    batch: *mut sys::llama_batch_ext,
    vocab: *const sys::llama_vocab,
    embedding_size: usize,
    /// The width of the encoder's output rows, which go into the text model in place of tokens.
    row_size: usize,
    /// The image and audio encoders (mtmd), when loaded (not for queries).
    media: *mut sys::mtmd_context,
    max_tokens: usize,
    sequences: i32,
    tokens: usize,
    position: sys::llama_pos,
}

// The llama.cpp objects belong to this Embedder alone and are only used through &mut self, so moving it to another
// thread is safe (the service keeps it behind a Mutex).
unsafe impl Send for Embedder {}

const MAX_SEQUENCES: i32 = 16;

impl Embedder {
    /// Loads the text model. `query_only` reserves a smaller context (queries are short): 512 tokens, not 2048.
    pub fn load(models: &ModelFiles, query_only: bool) -> Result<Embedder> {
        init_backend();
        let path = CString::new(models.text.to_string_lossy().as_bytes())?;
        unsafe {
            let mut model_params = sys::llama_model_default_params();
            model_params.n_gpu_layers = 999;
            let model = sys::llama_model_load_from_file(path.as_ptr(), model_params);
            if model.is_null() {
                bail!("can't load {}", models.text.display());
            }
            // One input must fit one micro-batch: 2048 tokens, the reference's cut.
            let max_tokens = if query_only { 512 } else { 2048 };
            let mut params = sys::llama_context_default_params();
            params.n_ctx = max_tokens as u32;
            params.n_batch = max_tokens as u32;
            params.n_ubatch = max_tokens as u32;
            params.n_seq_max = MAX_SEQUENCES as u32;
            params.kv_unified = true;
            params.embeddings = true;
            params.pooling_type = sys::LLAMA_POOLING_TYPE_MEAN;
            let context = sys::llama_init_from_model(model, params);
            if context.is_null() {
                sys::llama_model_free(model);
                bail!("can't create a llama.cpp context");
            }
            let mut embedder = Embedder {
                model,
                context,
                batch: sys::llama_batch_ext_init(context),
                vocab: sys::llama_model_get_vocab(model),
                embedding_size: sys::llama_model_n_embd_out(model) as usize,
                row_size: sys::llama_model_n_embd_inp(model) as usize,
                media: std::ptr::null_mut(),
                max_tokens,
                sequences: 0,
                tokens: 0,
                position: 0,
            };
            if embedder.embedding_size != DIM {
                bail!("expected {DIM}-number vectors, the model makes {}", embedder.embedding_size);
            }
            // The image and audio encoders, for indexing (queries are text).
            if let (false, Some(media)) = (query_only, &models.media) {
                let path = CString::new(media.to_string_lossy().as_bytes())?;
                let mut params = sys::mtmd_context_params_default();
                params.use_gpu = true;
                params.print_timings = false;
                params.warmup = true;
                // Never rescale: pictures arrive at their final size (`picture::target_size`).
                params.image_min_tokens = 1;
                params.image_max_tokens = 1120;
                embedder.media = sys::mtmd_init_from_file(path.as_ptr(), model, params);
                if embedder.media.is_null() {
                    bail!("can't load {}", media.display());
                }
            }
            // The first call builds the GPU pipelines: pay for that now.
            embedder.embed_texts(&[prompts::query("warm up")]).context("warming up the model")?;
            Ok(embedder)
        }
    }

    /// One vector per text, in order.
    pub fn embed_texts(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.reset();
        let mut vectors = Vec::with_capacity(texts.len());
        for text in texts {
            let tokens = self.tokenize(text);
            if self.sequences == MAX_SEQUENCES || self.tokens + tokens.len() > self.max_tokens {
                vectors.extend(self.run()?);
            }
            self.begin();
            for token in tokens {
                self.add_token(token)?;
            }
        }
        if self.sequences > 0 {
            vectors.extend(self.run()?);
        }
        Ok(vectors)
    }

    /// `<bos> text <eos>`, cut to `max_tokens` keeping both special tokens (as transformers cuts).
    fn tokenize(&self, text: &str) -> Vec<sys::llama_token> {
        let bytes = text.as_bytes();
        let mut tokens = vec![0; bytes.len() + 8];
        let mut count = unsafe {
            sys::llama_tokenize(
                self.vocab,
                bytes.as_ptr() as *const c_char,
                bytes.len() as i32,
                tokens.as_mut_ptr(),
                tokens.len() as i32,
                true,
                true,
            )
        };
        if count < 0 {
            tokens = vec![0; (-count) as usize];
            count = unsafe {
                sys::llama_tokenize(
                    self.vocab,
                    bytes.as_ptr() as *const c_char,
                    bytes.len() as i32,
                    tokens.as_mut_ptr(),
                    tokens.len() as i32,
                    true,
                    true,
                )
            };
        }
        tokens.truncate(count.max(0) as usize);
        if tokens.len() > self.max_tokens {
            tokens.truncate(self.max_tokens - 1);
            tokens.push(unsafe { sys::llama_vocab_eos(self.vocab) });
        }
        tokens
    }

    fn begin(&mut self) {
        self.sequences += 1;
        self.position = 0;
    }

    fn reset(&mut self) {
        unsafe { sys::llama_batch_ext_clear(self.batch) };
        self.sequences = 0;
        self.tokens = 0;
    }

    /// Whether the image encoder is loaded.
    pub fn sees_images(&self) -> bool {
        !self.media.is_null()
    }

    /// One picture's vector at `budget` vision tokens: from a file llama.cpp decodes itself (JPEG, PNG …, exactly as
    /// the reference), or from decoded pixels. Framed as the reference frames it: `<bos> <|image> rows <image|> <eos>`.
    pub fn embed_image(&mut self, picture: Picture, budget: usize) -> Result<Vec<f32>> {
        if self.media.is_null() {
            bail!("the image encoder isn't loaded");
        }
        let bitmap = self.load_bitmap(picture, budget)?;
        let result = self.embed_bitmap(bitmap);
        unsafe { sys::mtmd_bitmap_free(bitmap) };
        result
    }

    fn load_bitmap(&mut self, picture: Picture, budget: usize) -> Result<*mut sys::mtmd_bitmap> {
        let bitmap = unsafe {
            match &picture {
                Picture::File(path) => {
                    let name = CString::new(path.to_string_lossy().as_bytes())?;
                    let wrapper = sys::mtmd_helper_bitmap_init_from_file(
                        self.media,
                        name.as_ptr(),
                        false,
                        sys::mtmd_helper_init_opt_default(),
                    );
                    if !wrapper.video_ctx.is_null() {
                        if !wrapper.bitmap.is_null() {
                            sys::mtmd_bitmap_free(wrapper.bitmap);
                        }
                        sys::mtmd_helper_video_free(wrapper.video_ctx);
                        bail!("a video, not a picture: {}", path.display());
                    }
                    if wrapper.bitmap.is_null() {
                        bail!("can't read {}", path.display());
                    }
                    if sys::mtmd_bitmap_is_audio(wrapper.bitmap) {
                        sys::mtmd_bitmap_free(wrapper.bitmap);
                        bail!("sound, not a picture: {}", path.display());
                    }
                    wrapper.bitmap
                }
                Picture::Pixels(rgb) => sys::mtmd_bitmap_init(rgb.width as u32, rgb.height as u32, rgb.data.as_ptr()),
            }
        };
        if bitmap.is_null() {
            bail!("can't make a bitmap");
        }
        // Resize to fill the budget, exactly as the reference does.
        let (width, height) =
            unsafe { (sys::mtmd_bitmap_get_nx(bitmap) as usize, sys::mtmd_bitmap_get_ny(bitmap) as usize) };
        let Some((to_height, to_width)) = crate::picture::target_size(height, width, budget) else {
            unsafe { sys::mtmd_bitmap_free(bitmap) };
            bail!("too thin to embed");
        };
        if (to_height, to_width) == (height, width) {
            return Ok(bitmap);
        }
        let pixels = unsafe { std::slice::from_raw_parts(sys::mtmd_bitmap_get_data(bitmap), width * height * 3) };
        let resized = crate::picture::resize_bicubic(pixels, width, height, to_width, to_height);
        unsafe { sys::mtmd_bitmap_free(bitmap) };
        let bitmap = unsafe { sys::mtmd_bitmap_init(to_width as u32, to_height as u32, resized.as_ptr()) };
        if bitmap.is_null() {
            bail!("can't resize");
        }
        Ok(bitmap)
    }

    fn embed_bitmap(&mut self, bitmap: *mut sys::mtmd_bitmap) -> Result<Vec<f32>> {
        unsafe {
            let chunks = sys::mtmd_input_chunks_init();
            let marker = std::ffi::CStr::from_ptr(sys::mtmd_default_marker());
            let text = sys::mtmd_input_text {
                text: marker.as_ptr(),
                text_len: marker.to_bytes().len(),
                add_special: true,
                parse_special: true,
            };
            let bitmaps = [bitmap as *const sys::mtmd_bitmap];
            let status = sys::mtmd_tokenize(self.media, chunks, &text, bitmaps.as_ptr(), 1);
            let result = (|| {
                if status != 0 {
                    bail!("can't tokenize the picture ({status})");
                }
                self.reset();
                self.begin();
                for index in 0..sys::mtmd_input_chunks_size(chunks) {
                    let chunk = sys::mtmd_input_chunks_get(chunks, index);
                    if sys::mtmd_input_chunk_get_type(chunk) == sys::MTMD_INPUT_CHUNK_TYPE_TEXT {
                        let mut count = 0usize;
                        let tokens = sys::mtmd_input_chunk_get_tokens_text(chunk, &mut count);
                        if tokens.is_null() {
                            continue;
                        }
                        for &token in std::slice::from_raw_parts(tokens, count) {
                            self.add_token(token)?;
                        }
                    } else {
                        if sys::mtmd_encode_chunk(self.media, chunk) != 0 {
                            bail!("the encoder failed");
                        }
                        let rows = sys::mtmd_get_output_embd(self.media);
                        if rows.is_null() {
                            bail!("the encoder gave nothing");
                        }
                        self.add_rows(rows, sys::mtmd_input_chunk_get_n_tokens(chunk))?;
                    }
                }
                Ok(self.run()?.remove(0))
            })();
            sys::mtmd_input_chunks_free(chunks);
            result
        }
    }

    /// Encoder output rows into the batch, one position each (the batch copies them).
    fn add_rows(&mut self, rows: *const f32, count: usize) -> Result<()> {
        for row in 0..count {
            let embd =
                sys::llama_embd { data: unsafe { rows.add(row * self.row_size) }, n_rows: 1, n_embd: self.row_size };
            let index = unsafe { sys::llama_batch_ext_add_embd(self.batch, self.sequences - 1, embd) };
            self.place(index)?;
        }
        Ok(())
    }

    fn add_token(&mut self, token: sys::llama_token) -> Result<()> {
        let index = unsafe { sys::llama_batch_ext_add_token(self.batch, self.sequences - 1, token) };
        self.place(index)
    }

    /// Positions the entry the batch just took and asks for its output.
    fn place(&mut self, index: i32) -> Result<()> {
        if index < 0 {
            bail!("batch full ({index})");
        }
        let pos: [sys::llama_pos; 4] = [self.position, 0, 0, 0];
        let placed = unsafe {
            sys::llama_batch_ext_set_pos(self.batch, index, pos.as_ptr())
                && sys::llama_batch_ext_set_output_embd(self.batch, index, true)
        };
        if !placed {
            bail!("can't place token {index}");
        }
        self.position += 1;
        self.tokens += 1;
        Ok(())
    }

    /// Runs the model over the batch: one L2-normalized vector per input, in order.
    fn run(&mut self) -> Result<Vec<Vec<f32>>> {
        let status = unsafe { sys::llama_process(self.context, sys::LLAMA_PROCESS_TYPE_ENCODE, self.batch) };
        let result = if status != 0 {
            Err(anyhow::anyhow!("llama_process failed ({status})"))
        } else {
            (0..self.sequences)
                .map(|sequence| {
                    let pooled = unsafe { sys::llama_get_embeddings_seq(self.context, sequence) };
                    if pooled.is_null() {
                        bail!("no pooled embedding for input {sequence}");
                    }
                    let pooled = unsafe { std::slice::from_raw_parts(pooled, self.embedding_size) };
                    let norm = pooled.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
                    let scale = if norm > 0.0 { (1.0 / norm) as f32 } else { 0.0 };
                    Ok(pooled.iter().map(|x| x * scale).collect())
                })
                .collect()
        };
        self.reset();
        result
    }
}

impl Drop for Embedder {
    fn drop(&mut self) {
        unsafe {
            if !self.media.is_null() {
                sys::mtmd_free(self.media);
            }
            sys::llama_batch_ext_free(self.batch);
            sys::llama_free(self.context);
            sys::llama_model_free(self.model);
        }
    }
}

/// llama.cpp's process-wide setup, once. Its log is chatty at load time (and warns about EmbeddingGemma's tokenizer on
/// every load): only errors reach stderr, warnings too with RUMMAGE_DEBUG set.
fn init_backend() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| unsafe {
        sys::llama_log_set(Some(log), std::ptr::null_mut());
        sys::mtmd_helper_log_set(Some(log), std::ptr::null_mut());
        sys::llama_backend_init();
    });
}

static LOG_LEVEL: AtomicU32 = AtomicU32::new(sys::GGML_LOG_LEVEL_INFO);

unsafe extern "C" fn log(level: sys::ggml_log_level, text: *const c_char, _: *mut c_void) {
    if text.is_null() {
        return;
    }
    // CONT continues the previous message, at its level.
    if level != sys::GGML_LOG_LEVEL_CONT {
        LOG_LEVEL.store(level, Ordering::Relaxed);
    }
    static DEBUG: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var_os("RUMMAGE_DEBUG").is_some());
    let threshold = if *DEBUG { sys::GGML_LOG_LEVEL_WARN } else { sys::GGML_LOG_LEVEL_ERROR };
    if LOG_LEVEL.load(Ordering::Relaxed) >= threshold {
        eprint!("{}", unsafe { std::ffi::CStr::from_ptr(text) }.to_string_lossy());
    }
}
