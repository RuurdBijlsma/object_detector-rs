use color_eyre::Result;
use object_detector::{DetectorType, ObjectDetector};
use std::path::Path;

#[tokio::main]
async fn main() -> Result<()> {
    color_eyre::install()?;

    let image_path = Path::new("assets/img/fridge.jpg");
    let img = image::open(image_path)?;

    let mut detector = ObjectDetector::from_hf(DetectorType::PromptFree)
        .build()
        .await?;
    let results = detector.predict(&img).call()?;
    for det in results {
        println!("[{:>10}] Score: {:.4}", det.tag, det.score);
    }

    Ok(())
}
