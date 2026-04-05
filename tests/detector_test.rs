use image::DynamicImage;
use object_detector::{DetectorType, ModelScale, ObjectDetector};

#[tokio::test]
async fn test_object_detector_prompt_free() {
    // Nano scale to speed up download
    let mut detector = ObjectDetector::from_hf(DetectorType::PromptFree)
        .scale(ModelScale::Nano)
        .include_mask(false)
        .build()
        .await
        .expect("Failed to build PromptFree detector");

    let img = DynamicImage::new_rgb8(640, 640);
    let results = detector
        .predict(&img)
        .call()
        .expect("Failed to predict with PromptFree detector");

    println!("Detected {} objects with PromptFree", results.len());
}

#[tokio::test]
async fn test_object_detector_promptable() {
    // Nano scale to speed up download
    let mut detector = ObjectDetector::from_hf(DetectorType::Promptable)
        .scale(ModelScale::Nano)
        .include_mask(false)
        .build()
        .await
        .expect("Failed to build Promptable detector");

    let img = DynamicImage::new_rgb8(640, 640);
    let labels = ["cat", "dog", "person"];
    let results = detector
        .predict(&img)
        .labels(&labels)
        .call()
        .expect("Failed to predict with Promptable detector");

    println!("Detected {} objects with Promptable", results.len());
}

#[tokio::test]
async fn test_object_detector_errors() {
    let mut prompt_free_detector = ObjectDetector::from_hf(DetectorType::PromptFree)
        .scale(ModelScale::Nano)
        .include_mask(false)
        .build()
        .await
        .expect("Failed to build PromptFree detector");

    let img = DynamicImage::new_rgb8(640, 640);

    // Should error if labels are provided to PromptFree
    let labels = ["cat"];
    let result = prompt_free_detector.predict(&img).labels(&labels).call();
    assert!(
        result.is_err(),
        "Expected error when providing labels to PromptFree detector"
    );

    let mut promptable_detector = ObjectDetector::from_hf(DetectorType::Promptable)
        .scale(ModelScale::Nano)
        .include_mask(false)
        .build()
        .await
        .expect("Failed to build Promptable detector");

    // Should error if labels are NOT provided to Promptable
    let result = promptable_detector.predict(&img).call();
    assert!(
        result.is_err(),
        "Expected error when not providing labels to Promptable detector"
    );
}
