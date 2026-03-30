use color_eyre::Result;
use object_detector::YOLO26Predictor;
use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::time::Instant;

fn main() -> Result<()> {
    color_eyre::install()?;
    let mut predictor = YOLO26Predictor::new(
        "assets/model/yoloe-26l-seg-pf.onnx",
        "assets/model/vocabulary.json",
    )?;

    println!("--- YOLO26 Rust Optimized ---");
    let img_dir = Path::new("assets/img");
    for entry in fs::read_dir(img_dir)? {
        let path = entry?.path();
        if path.extension().map_or(false, |e| e == "jpg" || e == "png") {
            let img = image::open(&path)?;

            let start = Instant::now();
            let results = predictor.predict(&img, 0.4, 0.7)?;

            let tags = results
                .iter()
                .map(|r| r.tag.clone())
                .collect::<HashSet<_>>();

            println!(
                "Image: {} ({:?}) - Objects: {:?}",
                path.file_name().unwrap().to_string_lossy(),
                start.elapsed(),
                tags
            );
        }
    }
    Ok(())
}