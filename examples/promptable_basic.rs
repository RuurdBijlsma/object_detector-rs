use object_detector::{DetectorType, ObjectDetector};
use std::path::Path;

#[tokio::main]
async fn main() -> color_eyre::Result<()> {
    let image_path = Path::new("assets/img/market.jpg");
    let img = image::open(image_path)?;
    let labels = ["lamp", "person"];

    let detector = ObjectDetector::from_hf(DetectorType::Promptable)
        .cache_dir(Path::new("MY_CACHE"))
        .build()
        .await?;

    println!(
        "Running inference on {} for labels: {:?}...",
        image_path.display(),
        labels
    );
    let results = detector.predict(&img).labels(&labels).call()?;
    for det in results {
        println!("[{:>10}] Score: {:.4}", det.tag, det.score);
    }

    Ok(())
}
