//! rummage's vectors against DigUp's reference vectors (tests/reference/vectors.json, made with the same llama.cpp
//! build and model files). Matching them shows the bindings, tokenizing, prompts and pooling are right, and catches a
//! llama.cpp upgrade that quietly changes the vectors (every index would have to embed again).
//!
//! Needs the model files (see README); skipped without them, as on CI.

use rummage::embed::{Embedder, Picture, prompts};
use rummage::paths::ModelFiles;
use std::collections::HashMap;

/// Runtimes that agree measure ≥ 0.999 (DigUp's threshold); less is a change people would see in their results.
const AGREEMENT: f32 = 0.999;

#[derive(serde::Deserialize)]
struct Reference {
    made_with: String,
    vectors: HashMap<String, Vec<f32>>,
}

/// The text inputs, exactly as DigUp's ReferenceVectorTests builds them.
fn texts() -> Vec<(&'static str, String)> {
    let long: Vec<String> = (1..=60)
        .map(|i| {
            format!(
                "Note {i}: the review moved to room {} because the projector in the main hall failed again, and the team \
                 agreed to send the slides the night before.",
                i % 7 + 2
            )
        })
        .collect();
    vec![
        ("query", prompts::query("payment declined error")),
        ("query-bn", prompts::query("ইলিশ মাছ রান্না")),
        ("query-ar", prompts::query("متى تغلق المكتبة")),
        (
            "document",
            prompts::document(
                Some("Tokyo itinerary"),
                "Flight to Tokyo Haneda on November 3. Then the train to Kamakura to see the Great Buddha, and back for \
                 dinner in Shibuya.",
            ),
        ),
        // About 1,800 tokens: past the sliding attention window, so the global layers count as well.
        ("document-long", prompts::document(None, &long.join(" "))),
    ]
}

#[test]
fn text_vectors_match_digups() {
    check(|embedder| {
        let inputs = texts();
        let made = embedder.embed_texts(&inputs.iter().map(|(_, t)| t.clone()).collect::<Vec<_>>()).expect("embed");
        inputs.iter().map(|(name, _)| name.to_string()).zip(made).collect()
    });
}

#[test]
fn image_vectors_match_digups() {
    // A screenshot at 280 vision tokens, a photo at 140, and the photo as a video frame at 70: read by llama.cpp's
    // own decoder and resized here exactly as the reference resizes them.
    let folder = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/reference");
    check(|embedder| {
        if !embedder.sees_images() {
            eprintln!("skipped: no image encoder");
            return Vec::new();
        }
        [("screen-280", "screen.png", 280), ("photo-140", "photo.jpg", 140), ("frame-70", "photo.jpg", 70)]
            .into_iter()
            .map(|(name, file, budget)| {
                (name.to_string(), embedder.embed_image(Picture::File(folder.join(file)), budget).expect(file))
            })
            .collect()
    });
}

#[test]
fn audio_vector_matches_digups() {
    // A speech clip, decoded by ffmpeg to 16 kHz mono as indexing decodes any recording.
    let speech = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/reference/speech.wav");
    check(|embedder| {
        if !embedder.sees_images() || !rummage::media::available() {
            eprintln!("skipped: no audio encoder or ffmpeg");
            return Vec::new();
        }
        let mut windows = Vec::new();
        rummage::media::audio_windows(&speech, None, |w| {
            windows.push(w.samples);
            Ok(())
        })
        .expect("decode speech.wav");
        assert_eq!(windows.len(), 1, "a short clip is one window");
        vec![("speech".to_string(), embedder.embed_audio(&windows[0]).expect("embed"))]
    });
}

/// Embeds with `make` and compares each vector with its reference.
fn check(make: impl FnOnce(&mut Embedder) -> Vec<(String, Vec<f32>)>) {
    let models = match ModelFiles::locate() {
        Ok(models) => models,
        // On CI a missing model is a failure, not a skip (RUMMAGE_REQUIRE_MODEL=1).
        Err(error) if std::env::var_os("RUMMAGE_REQUIRE_MODEL").is_some() => panic!("no model files: {error:#}"),
        Err(_) => {
            eprintln!("skipped: no model files");
            return;
        }
    };
    let reference: Reference =
        serde_json::from_str(include_str!("reference/vectors.json")).expect("tests/reference/vectors.json");
    let mut embedder = Embedder::load(&models, false).expect("load the model");
    let made = make(&mut embedder);
    let mut worst = (String::new(), 1.0f32);
    for (name, vector) in &made {
        let expected = &reference.vectors[name];
        let cosine: f32 = vector.iter().zip(expected).map(|(a, b)| a * b).sum();
        eprintln!("{name:<14} cosine {cosine:.5}");
        if cosine < worst.1 {
            worst = (name.to_string(), cosine);
        }
    }
    assert!(
        worst.1 >= AGREEMENT,
        "{}: cosine {} with DigUp's vectors ({}): rummage's pipeline makes other vectors",
        worst.0,
        worst.1,
        reference.made_with
    );
}
