use color_eyre::eyre::Result;
use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::s;
use object_detector::predictor::nms::non_maximum_suppression;
use object_detector::predictor::{preprocess_image, reconstruct_mask};
use object_detector::{ObjectBBox, PromptFreeDetector};
use ort::value::Value;
use std::hint::black_box;

fn benchmark_predict_components(c: &mut Criterion) -> Result<()> {
    // Model Paths
    let seg_model_path = "assets/model/prompt_free/yoloe-26l-seg-pf.onnx";
    let det_model_path = "assets/model/prompt_free/yoloe-26l-det-pf.onnx";
    let vocab_path = "assets/model/prompt_free/vocabulary_4585.json";
    let img_path = "assets/img/fridge.jpg";

    let img = image::open(img_path).expect("Failed to open image");

    // --- SEGMENTATION MODEL BENCHMARKS ---
    let mut seg_predictor = PromptFreeDetector::builder(seg_model_path, vocab_path).build()?;

    c.bench_function("preprocess", |b| {
        b.iter(|| {
            black_box(preprocess_image(
                black_box(&img),
                seg_predictor.engine.image_size,
                seg_predictor.engine.stride,
            ))
        });
    });

    let (input_tensor, meta) = preprocess_image(&img, seg_predictor.engine.image_size, seg_predictor.engine.stride);

    c.bench_function("inference_seg", |b| {
        b.iter(|| {
            let outputs = seg_predictor
                .engine
                .session
                .run(ort::inputs!["images" => Value::from_array(input_tensor.clone()).unwrap()])
                .unwrap();
            let preds = outputs["detections"].try_extract_array::<f32>().unwrap();
            let protos = outputs["protos"].try_extract_array::<f32>().unwrap();
            black_box((preds, protos));
        });
    });

    // Extract data for component benchmarks
    let (preds, protos) = {
        let outputs = seg_predictor
            .engine
            .session
            .run(ort::inputs!["images" => Value::from_array(input_tensor.clone()).unwrap()])?;
        let preds = outputs["detections"].try_extract_array::<f32>()?.to_owned();
        let protos = outputs["protos"].try_extract_array::<f32>()?.to_owned();
        (preds, protos)
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

    c.bench_function("predict_full_seg", |b| {
        b.iter(|| {
            seg_predictor
                .predict(black_box(&img))
                .call()
                .expect("Predict failed");
        });
    });

    // --- DETECTION MODEL BENCHMARK ---
    let mut det_predictor = PromptFreeDetector::builder(det_model_path, vocab_path).build()?;

    c.bench_function("predict_full_det", |b| {
        b.iter(|| {
            det_predictor
                .predict(black_box(&img))
                .call()
                .expect("Predict failed");
        });
    });

    Ok(())
}

fn benchmark_wrapper(c: &mut Criterion) {
    benchmark_predict_components(c).unwrap();
}

criterion_group!(benches, benchmark_wrapper);
criterion_main!(benches);