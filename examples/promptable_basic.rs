use color_eyre::eyre::Context;
use image::{DynamicImage, GenericImage, GenericImageView, Rgba};
use ndarray::{s, Array4, Axis, Ix2};
use open_clip_inference::TextEmbedder;
use ort::session::Session;
use ort::value::Value;

#[derive(Debug)]
struct Detection {
    bbox: [f32; 4],
    score: f32,
    class_id: usize,
    label: String,
}

fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;

    let model_path = "yoloe-26x-pure-clip.onnx";
    let img_path = "assets/img/market.jpg";
    let clip_model_id = "RuteNL/MobileCLIP2-B-OpenCLIP-ONNX";
    let labels = vec!["cat", "car", "van", "sign", "person", "lamp", "watermelon"];
    let conf_threshold = 0.15;
    let iou_threshold = 0.7;

    println!("--- Initializing Promptable Pipeline ---");

    let rt = tokio::runtime::Runtime::new()?;
    let text_embedder = rt.block_on(async {
        TextEmbedder::from_hf(clip_model_id).build().await
    })?;

    let mut session = Session::builder()?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3).unwrap()
        .commit_from_file(model_path)?;

    let img = image::open(img_path).wrap_err("Failed to load image")?;
    let (img_tensor, ratio, pad, (orig_w, orig_h)) = preprocess(&img, 640);

    println!("Generating embeddings for {} labels...", labels.len());
    let text_embs = text_embedder.embed_texts(&labels)?;
    let text_tensor = text_embs.insert_axis(Axis(0));

    println!("Running YOLOE inference...");
    let outputs = session.run(ort::inputs![
        "images" => Value::from_array(img_tensor)?,
        "text_embeddings" => Value::from_array(text_tensor)?
    ])?;

    // --- FIX: Explicitly handle dimensionality ---
    let raw_output = outputs["output0"].try_extract_array::<f32>()?;

    // Convert dynamic shape [1, features, 8400] to static 2D [features, 8400]
    let preds_2d = raw_output
        .slice(s![0, .., ..])
        .into_dimensionality::<Ix2>()?
        .reversed_axes(); // Result: [8400, features]

    let num_classes = labels.len();
    let mut candidates = Vec::new();

    for i in 0..preds_2d.shape()[0] {
        let row = preds_2d.row(i); // Returns ArrayView1, which is indexable by usize

        // Scores start at index 4
        let scores = row.slice(s![4..4 + num_classes]);

        let mut max_score = 0.0f32;
        let mut max_cls_id = 0;
        for (idx, &s) in scores.iter().enumerate() {
            if s > max_score {
                max_score = s;
                max_cls_id = idx;
            }
        }

        if max_score > conf_threshold {
            let cx = row[0];
            let cy = row[1];
            let w = row[2];
            let h = row[3];

            candidates.push(Detection {
                bbox: [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0],
                score: max_score,
                class_id: max_cls_id,
                label: labels[max_cls_id].to_string(),
            });
        }
    }

    let kept_indices = nms(&candidates, iou_threshold);
    let mut final_detections = Vec::new();

    for idx in kept_indices {
        let mut det = &candidates[idx];

        let x1 = ((det.bbox[0] - pad.0) / ratio).clamp(0.0, orig_w as f32);
        let y1 = ((det.bbox[1] - pad.1) / ratio).clamp(0.0, orig_h as f32);
        let x2 = ((det.bbox[2] - pad.0) / ratio).clamp(0.0, orig_w as f32);
        let y2 = ((det.bbox[3] - pad.1) / ratio).clamp(0.0, orig_h as f32);

        final_detections.push(Detection {
            bbox: [x1, y1, x2, y2],
            score: det.score,
            class_id: det.class_id,
            label: det.label.clone(),
        });
    }

    println!("\n--- Result Summary ---");
    println!("Objects detected: {}", final_detections.len());
    for det in &final_detections {
        println!("[{:>10}] Score: {:.4} | Box: [{:.1}, {:.1}, {:.1}, {:.1}]",
                 det.label, det.score, det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3]);
    }

    Ok(())
}

fn preprocess(img: &DynamicImage, target_size: u32) -> (Array4<f32>, f32, (f32, f32), (u32, u32)) {
    let (w, h) = img.dimensions();
    let ratio = target_size as f32 / w.max(h) as f32;
    let new_w = (w as f32 * ratio).round() as u32;
    let new_h = (h as f32 * ratio).round() as u32;

    let pad_w = (target_size - new_w) as f32 / 2.0;
    let pad_h = (target_size - new_h) as f32 / 2.0;
    // todo: image preprocess should be same as in src/ crate code

    let resized = img.resize_exact(new_w, new_h, image::imageops::FilterType::Triangle);
    let mut canvas = DynamicImage::new_rgb8(target_size, target_size);
    for y in 0..target_size {
        for x in 0..target_size {
            canvas.put_pixel(x, y, Rgba([114, 114, 114, 255]));
        }
    }
    image::imageops::overlay(&mut canvas, &resized, pad_w as i64, pad_h as i64);

    let mut array = Array4::zeros((1, 3, target_size as usize, target_size as usize));
    let rgb_canvas = canvas.to_rgb8();
    for (x, y, pixel) in rgb_canvas.enumerate_pixels() {
        array[[0, 0, y as usize, x as usize]] = pixel[0] as f32 / 255.0;
        array[[0, 1, y as usize, x as usize]] = pixel[1] as f32 / 255.0;
        array[[0, 2, y as usize, x as usize]] = pixel[2] as f32 / 255.0;
    }

    (array, ratio, (pad_w, pad_h), (w, h))
}

fn nms(candidates: &[Detection], iou_threshold: f32) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..candidates.len()).collect();
    indices.sort_by(|&a, &b| candidates[b].score.partial_cmp(&candidates[a].score).unwrap());

    let mut kept = Vec::new();
    let mut suppressed = vec![false; candidates.len()];

    for i in 0..indices.len() {
        let idx = indices[i];
        if suppressed[idx] { continue; }

        kept.push(idx);
        for j in (i + 1)..indices.len() {
            let next_idx = indices[j];
            if suppressed[next_idx] { continue; }

            if calculate_iou(&candidates[idx].bbox, &candidates[next_idx].bbox) > iou_threshold {
                suppressed[next_idx] = true;
            }
        }
    }
    kept
}

fn calculate_iou(box1: &[f32; 4], box2: &[f32; 4]) -> f32 {
    let x1 = box1[0].max(box2[0]);
    let y1 = box1[1].max(box2[1]);
    let x2 = box1[2].min(box2[2]);
    let y2 = box1[3].min(box2[3]);
    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]);
    let area2 = (box2[2] - box2[0]) * (box2[3] - box2[1]);
    inter / (area1 + area2 - inter)
}