#![allow(clippy::significant_drop_tightening)]
use color_eyre::eyre::Result;
use criterion::{Criterion, criterion_group, criterion_main};
use ndarray::s;
use object_detector::predictor::nms::non_maximum_suppression;
use object_detector::predictor::{preprocess_image, reconstruct_mask};
use object_detector::{ObjectBBox, PromptFreeDetector, PromptableDetector};
use ort::value::Value;
use std::hint::black_box;

#[allow(clippy::too_many_lines)]
fn benchmark_components(
    c: &mut Criterion,
    pf_seg: &PromptFreeDetector,
    _prompt_seg: &PromptableDetector,
) -> Result<()> {
    let img_path = "assets/img/market.jpg";
    let img = image::open(img_path).expect("Failed to open benchmark image. Ensure image exists.");

    // --- PROMPT-FREE SEGMENTATION ---
    c.bench_function("preprocess", |b| {
        b.iter(|| {
            black_box(preprocess_image(
                black_box(&img),
                pf_seg.engine.image_size,
                pf_seg.engine.stride,
            ))
        });
    });

    let (input_tensor, meta) =
        preprocess_image(&img, pf_seg.engine.image_size, pf_seg.engine.stride);

    c.bench_function("inference_seg", |b| {
        b.iter(|| {
            let mut session = pf_seg.engine.session.lock().unwrap();
            let outputs = session
                .run(ort::inputs!["images" => Value::from_array(input_tensor.clone()).unwrap()])
                .unwrap();
            let preds = outputs["detections"].try_extract_array::<f32>().unwrap();
            let protos = outputs["protos"].try_extract_array::<f32>().unwrap();
            black_box((preds, protos));
        });
    });

    // Extract data for NMS/Mask benchmarks
    let (preds, protos) = {
        let mut session = pf_seg.engine.session.lock().unwrap();
        let outputs = session
            .run(ort::inputs!["images" => Value::from_array(input_tensor.clone()).unwrap()])?;
        (
            outputs["detections"].try_extract_array::<f32>()?.to_owned(),
            outputs["protos"].try_extract_array::<f32>()?.to_owned(),
        )
    };

    let preds_view = preds.slice(s![0, .., ..]);
    let protos_view = protos.slice(s![0, .., .., ..]);

    c.bench_function("nms_and_filtering", |b| {
        b.iter(|| {
            let mut boxes = Vec::new();
            let mut scores = Vec::new();
            for i in 0..preds_view.shape()[0] {
                let score = preds_view[[i, 4]];
                if score > 0.25 {
                    boxes.push(ObjectBBox {
                        x1: preds_view[[i, 0]],
                        y1: preds_view[[i, 1]],
                        x2: preds_view[[i, 2]],
                        y2: preds_view[[i, 3]],
                    });
                    scores.push(score);
                }
            }
            black_box(non_maximum_suppression(&boxes, &scores, 0.45));
        });
    });

    // Mask processing benchmark
    let mut boxes = Vec::new();
    let mut scores = Vec::new();
    let mut weights_vec = Vec::new();
    for i in 0..preds_view.shape()[0] {
        let score = preds_view[[i, 4]];
        if score > 0.25 {
            boxes.push(ObjectBBox {
                x1: preds_view[[i, 0]],
                y1: preds_view[[i, 1]],
                x2: preds_view[[i, 2]],
                y2: preds_view[[i, 3]],
            });
            scores.push(score);
            weights_vec.push(preds_view.slice(s![i, 6..38]).to_owned());
        }
    }
    let kept = non_maximum_suppression(&boxes, &scores, 0.45);

    if let Some(&idx) = kept.first() {
        let sample_bbox = boxes[idx];
        let weights = &weights_vec[idx];
        c.bench_function("process_mask_single", |b| {
            b.iter(|| {
                black_box(reconstruct_mask(
                    black_box(&protos_view),
                    black_box(weights),
                    black_box(&meta),
                    black_box(&sample_bbox),
                ));
            });
        });
    }

    // Full predict benchmarks removed - moved to full_benchmarks.rs

    Ok(())
}

fn benchmark_wrapper(c: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("Failed to create Tokio runtime");

    // Initialize only necessary detector variants for component benchmarking
    let (pf_seg, prompt_seg) = runtime.block_on(async {
        println!("Downloading/Loading models for component benchmarking...");

        let pf_seg = PromptFreeDetector::from_hf()
            .build()
            .await
            .expect("Failed to load pf_seg");

        let prompt_seg = PromptableDetector::from_hf()
            .build()
            .await
            .expect("Failed to load prompt_seg");

        (pf_seg, prompt_seg)
    });

    benchmark_components(c, &pf_seg, &prompt_seg).unwrap();
}

criterion_group!(benches, benchmark_wrapper);
criterion_main!(benches);
