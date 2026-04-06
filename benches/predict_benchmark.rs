#![allow(clippy::significant_drop_tightening)]
use color_eyre::eyre::Result;
use criterion::{Criterion, criterion_group, criterion_main};
use ndarray::s;
use object_detector::predictor::nms::non_maximum_suppression;
use object_detector::predictor::{preprocess_image, reconstruct_mask};
use object_detector::{ObjectBBox, PromptFreeDetector, PromptableDetector};
use open_clip_inference::TextEmbedder;
use ort::value::Value;
use std::hint::black_box;

#[allow(clippy::too_many_lines)]
async fn benchmark_predict_components(c: &mut Criterion) -> Result<()> {
    // Model Paths
    let pf_seg_model_path = "assets/model/prompt_free/yoloe-26l-seg-pf.onnx";
    let pf_det_model_path = "assets/model/prompt_free/yoloe-26l-det-pf.onnx";
    let prompt_seg_model_path = "assets/model/promptable/yoloe-26l-seg-promptable.onnx";
    let prompt_det_model_path = "assets/model/promptable/yoloe-26l-det-promptable.onnx";
    let vocab_path = "assets/model/prompt_free/vocabulary_4585.json";
    let img_path = "assets/img/fridge.jpg";

    let img = image::open(img_path).expect("Failed to open image");
    let labels = ["lamp", "person", "bottle", "shelf"];

    // --- PROMPT-FREE SEGMENTATION MODEL BENCHMARKS ---
    let pf_seg_predictor =
        PromptFreeDetector::builder(pf_seg_model_path, vocab_path).build()?;

    c.bench_function("preprocess", |b| {
        b.iter(|| {
            black_box(preprocess_image(
                black_box(&img),
                pf_seg_predictor.engine.image_size,
                pf_seg_predictor.engine.stride,
            ))
        });
    });

    let (input_tensor, meta) = preprocess_image(
        &img,
        pf_seg_predictor.engine.image_size,
        pf_seg_predictor.engine.stride,
    );

    c.bench_function("inference_seg", |b| {
        b.iter(|| {
            let mut session = pf_seg_predictor.engine.session.lock().unwrap();
            let outputs = session
                .run(ort::inputs!["images" => Value::from_array(input_tensor.clone()).unwrap()])
                .unwrap();
            let preds = outputs["detections"].try_extract_array::<f32>().unwrap();
            let protos = outputs["protos"].try_extract_array::<f32>().unwrap();
            black_box((preds, protos));
        });
    });

    // Extract data for component benchmarks
    let (preds, protos) = {
        let mut session = pf_seg_predictor.engine.session.lock().unwrap();
        let outputs = session
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

    c.bench_function("predict_full_pf_seg", |b| {
        b.iter(|| {
            pf_seg_predictor
                .predict(black_box(&img))
                .call()
                .expect("Predict failed");
        });
    });

    // --- PROMPT-FREE DETECTION MODEL BENCHMARK ---
    let pf_det_predictor =
        PromptFreeDetector::builder(pf_det_model_path, vocab_path).build()?;

    c.bench_function("predict_full_pf_det", |b| {
        b.iter(|| {
            pf_det_predictor
                .predict(black_box(&img))
                .call()
                .expect("Predict failed");
        });
    });

    // --- PROMPTABLE SEGMENTATION MODEL BENCHMARKS ---
    // TextEmbedder cannot be cloned, so we initialize two (or one per detector)
    let embedder_seg = TextEmbedder::from_hf("RuteNL/MobileCLIP2-B-OpenCLIP-ONNX")
        .build()
        .await?;
    let prompt_seg_predictor =
        PromptableDetector::builder(prompt_seg_model_path, embedder_seg).build()?;
    c.bench_function("predict_full_promptable_seg", |b| {
        b.iter(|| {
            prompt_seg_predictor
                .predict(black_box(&img), black_box(&labels))
                .call()
                .expect("Predict failed");
        });
    });

    // --- PROMPTABLE DETECTION MODEL BENCHMARKS ---
    let embedder_det = TextEmbedder::from_hf("RuteNL/MobileCLIP2-B-OpenCLIP-ONNX")
        .build()
        .await?;
    let prompt_det_predictor =
        PromptableDetector::builder(prompt_det_model_path, embedder_det).build()?;
    c.bench_function("predict_full_promptable_det", |b| {
        b.iter(|| {
            prompt_det_predictor
                .predict(black_box(&img), black_box(&labels))
                .call()
                .expect("Predict failed");
        });
    });

    Ok(())
}

fn benchmark_wrapper(c: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("Failed to create Tokio runtime");

    runtime.block_on(benchmark_predict_components(c)).unwrap();
}

criterion_group!(benches, benchmark_wrapper);
criterion_main!(benches);
