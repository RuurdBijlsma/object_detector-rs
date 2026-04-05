use ab_glyph::{FontVec, PxScale};
use color_eyre::Result;
use image::{DynamicImage, Rgba, RgbaImage};
use imageproc::drawing::{draw_filled_rect_mut, draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use object_detector::{DetectedObject, DetectorType, ModelScale, ObjectDetector, ObjectMask};
use std::fs;
use std::path::Path;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<()> {
    color_eyre::install()?;

    let img_dir = Path::new("assets/img");
    let font_path = Path::new("assets/Roboto-Regular.ttf");
    let font = FontVec::try_from_vec(fs::read(font_path)?)?;
    let labels = ["lamp", "person"];

    let config_specs = [
        (DetectorType::PromptFree, true, "output/prompt_free_masked"),
        (
            DetectorType::PromptFree,
            false,
            "output/prompt_free_no_mask",
        ),
        (DetectorType::Promptable, true, "output/promptable_masked"),
        (DetectorType::Promptable, false, "output/promptable_no_mask"),
    ];

    for (dtype, include_mask, out_dir_str) in config_specs {
        let out_dir = Path::new(out_dir_str);
        fs::create_dir_all(out_dir)?;

        println!("\n--- Initializing Detector: {dtype:?} (Scale: Large, Mask: {include_mask}) ---");
        let mut detector = ObjectDetector::from_hf(dtype)
            .scale(ModelScale::Large)
            .include_mask(include_mask)
            .build()
            .await?;

        let start_config = Instant::now();
        let mut img_count = 0;

        for entry in fs::read_dir(img_dir)? {
            let path = entry?.path();

            println!("  Processing: {}", path.display());
            let img = image::open(&path)?;
            let file_name = path.file_name().unwrap();

            let results = match dtype {
                DetectorType::PromptFree => detector.predict(&img).call()?,
                DetectorType::Promptable => detector.predict(&img).labels(&labels).call()?,
            };

            let mut out_path = out_dir.join(file_name);
            out_path.set_extension("png");
            visualize_results(&img, &results, &font, &out_path)?;
            img_count += 1;
        }

        println!(
            ">>> Configuration {:?} {} finished. Processed {} images in {:?}",
            dtype,
            if include_mask { "[mask]" } else { "[no mask]" },
            img_count,
            start_config.elapsed()
        );
    }

    Ok(())
}

fn visualize_results(
    img: &DynamicImage,
    results: &[DetectedObject],
    font: &FontVec,
    out_path: &Path,
) -> Result<()> {
    let mut output_img = img.to_rgba8();

    for det in results {
        let color = get_color(det.class_id);

        if let Some(mask) = &det.mask {
            apply_mask(&mut output_img, mask, color);
        }

        let b = det.bbox;
        let (w, h) = ((b.x2 - b.x1).max(1.0) as u32, (b.y2 - b.y1).max(1.0) as u32);

        // Draw bbox with 3-pixel thickness
        for i in 0..3 {
            let rect = Rect::at(b.x1 as i32 + i, b.y1 as i32 + i).of_size(
                w.saturating_sub((i * 2) as u32).max(1),
                h.saturating_sub((i * 2) as u32).max(1),
            );
            draw_hollow_rect_mut(&mut output_img, rect, color);
        }

        let label = format!("{} {:.2}", det.tag, det.score);
        let scale = PxScale::from(24.0);
        let text_y = (b.y1 as i32 - 28).max(0);
        let box_width = (label.len() as u32 * 13).max(40);

        // Draw background box for text
        draw_filled_rect_mut(
            &mut output_img,
            Rect::at(b.x1 as i32, text_y).of_size(box_width, 28),
            color,
        );

        // Draw text
        draw_text_mut(
            &mut output_img,
            Rgba([255, 255, 255, 255]),
            b.x1 as i32 + 4,
            text_y + 2,
            scale,
            font,
            &label,
        );
    }

    output_img.save(out_path)?;
    Ok(())
}

const fn get_color(class_id: usize) -> Rgba<u8> {
    let colors = [
        [255, 56, 56],
        [255, 112, 31],
        [255, 178, 29],
        [72, 249, 10],
        [26, 147, 238],
        [20, 54, 243],
        [146, 204, 23],
        [128, 0, 255],
    ];
    let c = colors[class_id % colors.len()];
    Rgba([c[0], c[1], c[2], 255])
}

fn apply_mask(img: &mut RgbaImage, mask: &ObjectMask, color: Rgba<u8>) {
    let (img_w, img_h) = img.dimensions();
    for y in 0..mask.height.min(img_h) {
        for x in 0..mask.width.min(img_w) {
            if mask.get(x, y) {
                let p = img.get_pixel_mut(x, y);
                // Blend with background
                p[0] = ((u32::from(p[0]) + u32::from(color[0])) / 2) as u8;
                p[1] = ((u32::from(p[1]) + u32::from(color[1])) / 2) as u8;
                p[2] = ((u32::from(p[2]) + u32::from(color[2])) / 2) as u8;
            }
        }
    }
}
