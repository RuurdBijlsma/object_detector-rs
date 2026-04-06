use object_detector::{DetectorType, ObjectDetector};
use std::path::Path;

#[tokio::main]
async fn main() -> color_eyre::Result<()> {
    let image_path = Path::new("assets/img/fridge.jpg");
    let img = image::open(image_path)?;

    let detector = ObjectDetector::from_hf(DetectorType::PromptFree)
        .build()
        .await?;
    let results = detector.predict(&img).call()?;
    for det in results {
        println!("[{:>10}] Score: {:.4}", det.tag, det.score);
    }

    Ok(())
}
