use color_eyre::Result;
use fast_image_resize::images::Image; // Fixed import for v6
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
use image::{DynamicImage, GenericImageView, imageops::FilterType as ImageFilter};
use std::fs;

// --- Helper: image crate ---
fn resize_with_image(img: &DynamicImage, w: u32, h: u32, filter: ImageFilter) -> Vec<u8> {
    img.resize_exact(w, h, filter).to_rgb8().into_raw()
}

fn resize_with_fast_image_resize(
    img: &DynamicImage,
    w: u32,
    h: u32,
    filter: FilterType,
) -> Result<Vec<u8>> {
    let (src_w, src_h) = img.dimensions();
    let src_rgb = img.to_rgb8();

    let src_image = Image::from_vec_u8(src_w, src_h, src_rgb.into_raw(), PixelType::U8x3)?;
    let mut dst_image = Image::new(w, h, PixelType::U8x3);

    let mut resizer = Resizer::new();
    let options = ResizeOptions::new().resize_alg(ResizeAlg::Convolution(filter));

    resizer.resize(&src_image, &mut dst_image, &options)?;

    Ok(dst_image.into_vec())
}

fn calculate_mse(a: &[u8], b: &[u8]) -> f64 {
    if a.len() != b.len() {
        return f64::MAX;
    }
    let sum: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&p1, &p2)| {
            let diff = p1 as f64 - p2 as f64;
            diff * diff
        })
        .sum();
    sum / a.len() as f64
}

fn main() -> Result<()> {
    color_eyre::install()?;

    let target_w = 640;
    let target_h = 360;
    let img = image::open("assets/img/cat.jpg")?;
    let reference = fs::read("assets/model/dynamic-onnx/opencv_ref_u8_rgb.bin")
        .expect("Please run 'python yolo26/gen_opencv_ref.py' first.");

    println!("{:<25} | {:<18} | {:<10}", "Library", "Algorithm", "MSE");
    println!("{}", "-".repeat(60));

    // 1. Test 'image' crate
    let image_filters = [
        ("Nearest", ImageFilter::Nearest),
        ("Triangle/Bilinear", ImageFilter::Triangle),
        ("CatmullRom/Bicubic", ImageFilter::CatmullRom),
        ("Gaussian", ImageFilter::Gaussian),
        ("Lanczos3", ImageFilter::Lanczos3),
    ];

    for (name, filter) in image_filters {
        let result = resize_with_image(&img, target_w, target_h, filter);
        let mse = calculate_mse(&result, &reference);
        println!("{:<25} | {:<18} | {:.4}", "image crate", name, mse);
    }

    // 2. Test 'fast_image_resize' v6
    let fir_filters = [
        ("Box/Nearest", FilterType::Box),
        ("Bilinear", FilterType::Bilinear),
        ("Hamming", FilterType::Hamming),
        ("CatmullRom", FilterType::CatmullRom),
        ("Mitchell", FilterType::Mitchell),
        ("Lanczos3", FilterType::Lanczos3),
        ("Gaussian", FilterType::Gaussian),
    ];

    for (name, filter) in fir_filters {
        let result = resize_with_fast_image_resize(&img, target_w, target_h, filter)?;
        let mse = calculate_mse(&result, &reference);
        println!("{:<25} | {:<18} | {:.4}", "fast_image_resize", name, mse);
    }

    Ok(())
}