use color_eyre::Result;
use ndarray::{s, Array4};
use ort::session::Session;
use ort::value::Value;
use serde::Deserialize;
use std::fs;

#[derive(Deserialize)]
struct PythonMeta {
    shape: Vec<usize>,
    pad: (f32, f32),
    ratio: f32,
}

fn calculate_iou(box1: &[f32; 4], box2: &[f32; 4]) -> f32 {
    let x1 = box1[0].max(box2[0]);
    let y1 = box1[1].max(box2[1]);
    let x2 = box1[2].min(box2[2]);
    let y2 = box1[3].min(box2[3]);
    let intersection = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]);
    let area2 = (box2[2] - box2[0]) * (box2[3] - box2[1]);
    intersection / (area1 + area2 - intersection)
}

fn nms(candidates: &Vec<([f32; 4], f32, usize)>, iou_threshold: f32) -> Vec<usize> {
    if candidates.is_empty() { return vec![]; }
    let mut indices: Vec<usize> = (0..candidates.len()).collect();
    indices.sort_by(|&a, &b| candidates[b].1.partial_cmp(&candidates[a].1).unwrap());

    let mut kept = Vec::new();
    while !indices.is_empty() {
        let current = indices.remove(0);
        kept.push(current);
        indices.retain(|&idx| {
            calculate_iou(&candidates[current].0, &candidates[idx].0) <= iou_threshold
        });
    }
    kept
}

pub fn main() -> Result<()> {
    color_eyre::install()?;

    let model_path = "assets/model/dynamic-onnx/yoloe-26l-seg-pf-dynamic-try-3.onnx";
    let vocab_path = "assets/model/dynamic-onnx/vocabulary-dynamic.json";
    let tensor_bin = "assets/model/dynamic-onnx/debug_data/cat_tensor.bin";
    let tensor_json = "assets/model/dynamic-onnx/debug_data/cat_meta.json";

    let vocab: Vec<String> = serde_json::from_str(&fs::read_to_string(vocab_path)?)?;
    let meta: PythonMeta = serde_json::from_str(&fs::read_to_string(tensor_json)?)?;

    // Load Tensor
    let bytes = fs::read(tensor_bin)?;
    let f32_vec: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    let input_tensor = Array4::from_shape_vec((meta.shape[0], meta.shape[1], meta.shape[2], meta.shape[3]), f32_vec)?;

    // Run Inference
    let mut session = Session::builder()?.commit_from_file(model_path)?;
    let outputs = session.run(ort::inputs!["images" => Value::from_array(input_tensor)?])?;
    let preds = outputs["detections"].try_extract_array::<f32>()?;
    let preds = preds.slice(s![0, .., ..]);

    // 1. Filter and Collect Candidates
    let conf_threshold = 0.4;
    let mut candidates = Vec::new();
    for i in 0..preds.shape()[0] {
        let score = preds[[i, 4]];
        if score > conf_threshold {
            let bbox = [preds[[i, 0]], preds[[i, 1]], preds[[i, 2]], preds[[i, 3]]];
            let class_id = preds[[i, 5]] as usize;
            candidates.push((bbox, score, class_id));
        }
    }

    // 2. Run NMS
    let iou_threshold = 0.7;
    let kept_indices = nms(&candidates, iou_threshold);

    // 3. Scale and Print Final Results
    println!("\n==================== FINAL CLEAN DETECTIONS (RUST) ====================");
    println!("Post-NMS Count: {}", kept_indices.len());

    for idx in kept_indices {
        let (bbox, score, class_id) = &candidates[idx];
        let tag = vocab.get(*class_id).map(|s| s.as_str()).unwrap_or("unknown");

        // Scaling logic from run_onnx.rs
        let x1 = (bbox[0] - meta.pad.0) / meta.ratio;
        let y1 = (bbox[1] - meta.pad.1) / meta.ratio;
        let x2 = (bbox[2] - meta.pad.0) / meta.ratio;
        let y2 = (bbox[3] - meta.pad.1) / meta.ratio;

        println!("TAG: {:<20} | CONF: {:.4} | BBOX: [{:.2}, {:.2}, {:.2}, {:.2}]",
                 tag, score, x1, y1, x2, y2);
    }

    Ok(())
}