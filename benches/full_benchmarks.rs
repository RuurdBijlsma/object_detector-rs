use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use object_detector::{DetectorType, ModelScale, ObjectDetector};
use ort::ep::CUDA;
use std::hint::black_box;

fn benchmark_full(c: &mut Criterion) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("Failed to create Tokio runtime");

    let img_path = "assets/img/van.jpg";
    let img = image::open(img_path).expect("Failed to open benchmark image.");
    let labels = ["van"];

    let scales = [
        ModelScale::Nano,
        ModelScale::Small,
        ModelScale::Medium,
        ModelScale::Large,
        ModelScale::XLarge,
    ];

    let mut group = c.benchmark_group("full_predict");

    for scale in scales {
        for detector_type in [DetectorType::PromptFree, DetectorType::Promptable] {
            for with_mask in [true, false] {
                let type_str = match detector_type {
                    DetectorType::PromptFree => "prompt_free",
                    DetectorType::Promptable => "promptable",
                };
                let task = if with_mask { "seg" } else { "det" };
                let scale_str = format!("{scale:?}").to_lowercase();

                let mut detector = runtime
                    .block_on(async {
                        ObjectDetector::from_hf(detector_type)
                            .with_execution_providers(&[CUDA::default().build().error_on_failure()])
                            .scale(scale)
                            .include_mask(with_mask)
                            .build()
                            .await
                    })
                    .expect("Failed to load ObjectDetector");

                group.bench_with_input(
                    BenchmarkId::new(type_str, format!("{scale_str}/{task}")),
                    &img,
                    |b, i| {
                        b.iter(|| {
                            let res = match detector_type {
                                DetectorType::PromptFree => detector.predict(black_box(i)).call(),
                                DetectorType::Promptable => detector
                                    .predict(black_box(i))
                                    .labels(black_box(&labels))
                                    .call(),
                            };
                            res.expect("Predict failed");
                        });
                    },
                );
                drop(detector);
            }
        }
    }
    group.finish();
}

criterion_group!(benches, benchmark_full);
criterion_main!(benches);
