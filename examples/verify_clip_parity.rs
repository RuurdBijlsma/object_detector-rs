#![allow(clippy::cast_precision_loss)]
use ndarray::Array2;
use ndarray_npy::read_npy;
use open_clip_inference::TextEmbedder;
use std::path::Path;

#[tokio::main]
async fn main() -> color_eyre::Result<()> {
    // 1. Load the Python Reference
    let ref_path = "py-yolo/text_prompt/clip_reference.npy";
    if !Path::new(ref_path).exists() {
        println!("Error: Reference file not found. Run 'python export_clip_reference.py' first.");
        return Ok(());
    }
    let py_embeddings: Array2<f32> = read_npy(ref_path)?;
    println!("Loaded Python reference: {:?}", py_embeddings.dim());

    // 2. Initialize Rust Embedder
    // Note: Use the MobileCLIP S2 variant which matches mobileclip2_b.ts architecture
    let model_id = "RuteNL/MobileCLIP2-B-OpenCLIP-ONNX";
    println!("Initializing Rust TextEmbedder ({model_id})...");

    let text_embedder = TextEmbedder::from_hf(model_id).build().await?;

    // 3. Generate Embeddings in Rust
    let labels = vec!["cat", "car", "van", "sign", "person", "lamp", "watermelon"];
    println!("Encoding labels in Rust...");

    let rust_embeddings = text_embedder.embed_texts(&labels)?;

    // 4. Compare
    println!("\n--- Parity Results ---");
    let mut total_diff = 0.0;

    for (i, label) in labels.iter().enumerate() {
        let py_row = py_embeddings.row(i);
        let rust_row = rust_embeddings.row(i);

        // Calculate Cosine Similarity
        let dot = py_row.dot(&rust_row);
        let py_norm = py_row.dot(&py_row).sqrt();
        let rust_norm = rust_row.dot(&rust_row).sqrt();
        let cosine_sim = dot / (py_norm * rust_norm);

        // Calculate Mean Absolute Error
        let mut abs_diff = 0.0;
        for j in 0..512 {
            abs_diff += (py_row[j] - rust_row[j]).abs();
        }
        let mae = abs_diff / 512.0;
        total_diff += mae;

        println!("Label: '{label}'");
        println!("  Cosine Similarity: {cosine_sim:.6}");
        println!("  Mean Absolute Error: {mae:.6}");
    }

    let avg_mae = total_diff / labels.len() as f32;
    println!("\nFinal Average MAE: {avg_mae:.6}");

    if avg_mae < 1e-4 {
        println!("✅ PARITY CHECK PASSED: Embeddings are virtually identical.");
    } else if avg_mae < 1e-2 {
        println!("⚠️ PARITY CHECK WARNING: Small differences detected (check normalization).");
    } else {
        println!("❌ PARITY CHECK FAILED: Significant deviation.");
    }

    Ok(())
}
