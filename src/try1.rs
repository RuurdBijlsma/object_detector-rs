use color_eyre::eyre::Result;
use image::{DynamicImage, GenericImageView, imageops::FilterType};
use ndarray::{Array4, ArrayView, IxDyn};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Value;
use std::fs;
use std::path::Path;
use std::time::Instant;

pub fn try1() -> Result<()> {
    // 1. Load Vocabulary
    let vocab_raw = fs::read_to_string("../assets/model/first-try-onnx/vocabulary.json")?;
    let vocabulary: Vec<String> = serde_json::from_str(&vocab_raw)?;

    let mut session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .expect("with_optimization_level")
        .with_intra_threads(num_cpus::get())
        .expect("with_intra_threads")
        .commit_from_file("../assets/model/first-try-onnx/yoloe-26l-seg-pf.onnx")?;

    let img_dir = Path::new("assets/img");

    println!("--- Starting inference on images in {:?} ---\n", img_dir);

    for entry in fs::read_dir(img_dir)? {
        let entry = entry?;
        let path = entry.path();
        process_image(&mut session, &path, &vocabulary)?;
    }

    Ok(())
}
// Define a structure to hold our detection results
#[derive(Debug, Clone)]
struct Detection {
    box_coords: [f32; 4], // x1, y1, x2, y2
    score: f32,
    class_id: usize,
}

fn process_image(session: &mut Session, img_path: &Path, vocabulary: &[String]) -> Result<()> {
    let start_time = Instant::now();
    let img = image::open(img_path)?;
    let input_size = 640;
    let input_tensor = preprocess(&img, input_size)?;
    let input_value = Value::from_array(input_tensor)?;

    let outputs = session.run(ort::inputs!["images" => input_value])?;
    let (shape, data) = outputs["output0"].try_extract_tensor::<f32>()?;
    let view = ArrayView::from_shape(
        IxDyn(&shape.iter().map(|&x| x as usize).collect::<Vec<_>>()),
        data,
    )?;
    let detections_raw = view.into_dimensionality::<ndarray::Ix3>()?;

    let mut candidates = Vec::new();
    let conf_threshold = 0.45; // Slightly higher threshold helps with RAM++ models

    // 1. Collect all candidates
    for i in 0..300 {
        let score = detections_raw[[0, i, 4]];
        if score > conf_threshold {
            candidates.push(Detection {
                box_coords: [
                    detections_raw[[0, i, 0]],
                    detections_raw[[0, i, 1]],
                    detections_raw[[0, i, 2]],
                    detections_raw[[0, i, 3]],
                ],
                score,
                class_id: detections_raw[[0, i, 5]] as usize,
            });
        }
    }

    // 2. Sort candidates by score descending
    candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

    // 3. Perform Class-Agnostic NMS
    let mut kept_detections: Vec<Detection> = Vec::new();
    let iou_threshold = 0.45;

    for cand in candidates {
        let mut keep = true;
        for kept in &kept_detections {
            if calculate_iou(&cand.box_coords, &kept.box_coords) > iou_threshold {
                keep = false;
                break;
            }
        }
        if keep {
            kept_detections.push(cand);
        }
    }

    // 4. Print Results
    let tags: Vec<String> = kept_detections
        .iter()
        .map(|d| vocabulary[d.class_id].clone())
        .collect();

    let elapsed = start_time.elapsed();
    println!("Image: {}", img_path.file_name().unwrap().to_string_lossy());
    println!("  - Time: {:.2?}ms", elapsed.as_secs_f64() * 1000.0);
    println!("  - Objects Found: {}", kept_detections.len());
    println!("  - Tags: {}", tags.join(", "));
    println!("{}", "-".repeat(30));

    Ok(())
}

// Helper function to calculate Intersection over Union
fn calculate_iou(box_a: &[f32; 4], box_b: &[f32; 4]) -> f32 {
    let x_min = box_a[0].max(box_b[0]);
    let y_min = box_a[1].max(box_b[1]);
    let x_max = box_a[2].min(box_b[2]);
    let y_max = box_a[3].min(box_b[3]);

    let intersection_area = (x_max - x_min).max(0.0) * (y_max - y_min).max(0.0);

    let area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]);
    let area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]);

    intersection_area / (area_a + area_b - intersection_area)
}

fn letterbox(img: &DynamicImage, size: u32) -> (DynamicImage, f32, u32, u32) {
    let (w, h) = img.dimensions();
    let scale = (size as f32 / w.max(h) as f32).min(1.0);
    let new_w = (w as f32 * scale) as u32;
    let new_h = (h as f32 * scale) as u32;

    let resized = img.resize(new_w, new_h, FilterType::Triangle);

    // 114, 114, 114 is the specific "Gray" color YOLO uses for padding
    let mut canvas = DynamicImage::new_rgb8(size, size);
    for pixel in canvas.as_mut_rgb8().unwrap().pixels_mut() {
        *pixel = image::Rgb([114, 114, 114]);
    }

    let top = (size - new_h) / 2;
    let left = (size - new_w) / 2;
    image::imageops::replace(&mut canvas, &resized, left as i64, top as i64);

    (canvas, scale, left, top)
}

fn preprocess(img: &DynamicImage, size: u32) -> Result<Array4<f32>> {
    let (letterboxed, _, _, _) = letterbox(img, size);
    let rgb = letterboxed.to_rgb8();

    let mut array = Array4::<f32>::zeros((1, 3, size as usize, size as usize));
    for (x, y, pixel) in rgb.enumerate_pixels() {
        // BACK TO BASIC SCALING: Just x / 255.0
        array[[0, 0, y as usize, x as usize]] = f32::from(pixel[0]) / 255.0;
        array[[0, 1, y as usize, x as usize]] = f32::from(pixel[1]) / 255.0;
        array[[0, 2, y as usize, x as usize]] = f32::from(pixel[2]) / 255.0;
    }
    Ok(array)
}
