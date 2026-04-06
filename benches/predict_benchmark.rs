#![allow(clippy::significant_drop_tightening)]
use color_eyre::eyre::Result;
use criterion::{Criterion, criterion_group, criterion_main};
use ndarray::s;
use object_detector::model_manager::HfModel;
use object_detector::predictor::nms::non_maximum_suppression;
use object_detector::predictor::{preprocess_image, reconstruct_mask};
use object_detector::{
    DetectorType, ModelScale, ObjectBBox, PromptFreeDetector, PromptableDetector,
};
use ort::value::Value;
use std::hint::black_box;

#[allow(clippy::too_many_lines)]
fn benchmark_predict_components(
    c: &mut Criterion,
    pf_seg: &PromptFreeDetector,
    pf_det: &PromptFreeDetector,
    prompt_seg: &PromptableDetector,
    prompt_det: &PromptableDetector,
) -> Result<()> {
    let img_path = "assets/img/market.jpg";
    let img = image::open(img_path).expect("Failed to open benchmark image. Ensure image exists.");
    let labels = [
        "lamp",
        "person",
        "watermelon",
        "cat",
        "keyboard",
        "sausage",
        "jar",
        "car",
        "van",
    ];

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

    c.bench_function("predict_full_pf_seg", |b| {
        b.iter(|| {
            pf_seg
                .predict(black_box(&img))
                .call()
                .expect("Predict failed");
        });
    });

    // --- PROMPT-FREE DETECTION ---
    c.bench_function("predict_full_pf_det", |b| {
        b.iter(|| {
            pf_det
                .predict(black_box(&img))
                .call()
                .expect("Predict failed");
        });
    });

    // --- PROMPTABLE SEGMENTATION ---
    c.bench_function("predict_full_promptable_seg", |b| {
        b.iter(|| {
            prompt_seg
                .predict(black_box(&img), black_box(&labels))
                .call()
                .expect("Predict failed");
        });
    });

    // --- PROMPTABLE DETECTION ---
    c.bench_function("predict_full_promptable_det", |b| {
        b.iter(|| {
            prompt_det
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

    // Initialize 4 detector variants from HuggingFace
    let (pf_seg, pf_det, prompt_seg, prompt_det) = runtime.block_on(async {
        println!("Downloading/Loading models from Hugging Face for benchmarking...");

        // Helper to get non-default HfModel paths (Detection only variants)
        let pf_det_file =
            HfModel::get_model_file_path(DetectorType::PromptFree, ModelScale::Large, false);
        let prompt_det_file =
            HfModel::get_model_file_path(DetectorType::Promptable, ModelScale::Large, false);

        let pf_seg = PromptFreeDetector::from_hf()
            .build()
            .await
            .expect("Failed to load pf_seg");

        let pf_det = PromptFreeDetector::from_hf()
            .model(HfModel {
                id: HfModel::DEFAULT_REPO_ID.to_string(),
                file: pf_det_file.clone(),
            })
            .data_model(HfModel {
                id: HfModel::DEFAULT_REPO_ID.to_string(),
                file: format!("{pf_det_file}.data"),
            })
            .build()
            .await
            .expect("Failed to load pf_det");

        let prompt_seg = PromptableDetector::from_hf()
            .build()
            .await
            .expect("Failed to load prompt_seg");

        let prompt_det = PromptableDetector::from_hf()
            .model(HfModel {
                id: HfModel::DEFAULT_REPO_ID.to_string(),
                file: prompt_det_file.clone(),
            })
            .data_model(HfModel {
                id: HfModel::DEFAULT_REPO_ID.to_string(),
                file: format!("{prompt_det_file}.data"),
            })
            .build()
            .await
            .expect("Failed to load prompt_det");

        (pf_seg, pf_det, prompt_seg, prompt_det)
    });

    benchmark_predict_components(c, &pf_seg, &pf_det, &prompt_seg, &prompt_det).unwrap();
}

criterion_group!(benches, benchmark_wrapper);
criterion_main!(benches);
