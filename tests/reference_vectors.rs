//! rummage's vectors against DigUp's reference vectors (tests/reference/vectors.json, made with the same llama.cpp
//! build and model files). Matching them shows the bindings, tokenizing, prompts and pooling are right, and catches a
//! llama.cpp upgrade that quietly changes the vectors (every index would have to embed again).
//!
//! Needs the model files (see README); skipped without them, as on CI. The image and audio references (photo.jpg,
//! screen.png, speech.wav) are checked once rummage embeds images and audio.

use rummage::embed::{Embedder, prompts};
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
    let Ok(models) = ModelFiles::locate() else {
        eprintln!("skipped: no model files");
        return;
    };
    let reference: Reference =
        serde_json::from_str(include_str!("reference/vectors.json")).expect("tests/reference/vectors.json");
    let inputs = texts();
    let mut embedder = Embedder::load(&models, false).expect("load the model");
    let made = embedder.embed_texts(&inputs.iter().map(|(_, t)| t.clone()).collect::<Vec<_>>()).expect("embed");
    let mut worst = (String::new(), 1.0f32);
    for ((name, _), vector) in inputs.iter().zip(&made) {
        let expected = &reference.vectors[*name];
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
