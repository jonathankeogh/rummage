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
    max_tokens: usize,
    sequences: i32,
    tokens: usize,
    position: sys::llama_pos,
}

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
                max_tokens,
                sequences: 0,
                tokens: 0,
                position: 0,
            };
            if embedder.embedding_size != DIM {
                bail!("expected {DIM}-number vectors, the model makes {}", embedder.embedding_size);
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

    fn add_token(&mut self, token: sys::llama_token) -> Result<()> {
        let index = unsafe { sys::llama_batch_ext_add_token(self.batch, self.sequences - 1, token) };
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
