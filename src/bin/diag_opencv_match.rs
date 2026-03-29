use color_eyre::Result;
use image::{GenericImageView, RgbImage};
use std::fs;

/// Manual implementation of OpenCV's INTER_LINEAR (Bilinear) for u8 images.
/// Replicates the center-aligned, non-filtering (sampling) behavior.
fn naive_bilinear_opencv(img: &RgbImage, dst_w: u32, dst_h: u32) -> Vec<u8> {
    let (src_w, src_h) = img.dimensions();
    let scale_x = src_w as f32 / dst_w as f32;
    let scale_y = src_h as f32 / dst_h as f32;
    let mut out = vec![0u8; (dst_w * dst_h * 3) as usize];

    for y in 0..dst_h {
        for x in 0..dst_w {
            // 1. Calculate center-aligned source coordinates (The OpenCV Secret)
            let src_xf = (x as f32 + 0.5) * scale_x - 0.5;
            let src_yf = (y as f32 + 0.5) * scale_y - 0.5;

            // 2. Identify the 4 surrounding pixels
            let x1 = src_xf.floor() as i32;
            let y1 = src_yf.floor() as i32;
            let x2 = x1 + 1;
            let y2 = y1 + 1;

            // 3. Calculate interpolation weights
            let dx = src_xf - x1 as f32;
            let dy = src_yf - y1 as f32;

            // 4. Clamping logic to match OpenCV border handling (BORDER_REPLICATE style)
            let x1_u = x1.clamp(0, src_w as i32 - 1) as u32;
            let y1_u = y1.clamp(0, src_h as i32 - 1) as u32;
            let x2_u = x2.clamp(0, src_w as i32 - 1) as u32;
            let y2_u = y2.clamp(0, src_h as i32 - 1) as u32;

            // Get pixels from source
            let p11 = img.get_pixel(x1_u, y1_u);
            let p21 = img.get_pixel(x2_u, y1_u);
            let p12 = img.get_pixel(x1_u, y2_u);
            let p22 = img.get_pixel(x2_u, y2_u);

            for c in 0..3 {
                let v11 = p11[c] as f32;
                let v21 = p21[c] as f32;
                let v12 = p12[c] as f32;
                let v22 = p22[c] as f32;

                // Bilinear Interpolation formula
                let val = v11 * (1.0 - dx) * (1.0 - dy)
                    + v21 * dx * (1.0 - dy)
                    + v12 * (1.0 - dx) * dy
                    + v22 * dx * dy;

                // Index calculation using consistent usize casting
                let dst_idx = ((y as usize * dst_w as usize) + x as usize) * 3 + c;

                // OpenCV uses 0.5 rounding for u8 (standard float to int rounding)
                out[dst_idx] = (val + 0.5) as u8;
            }
        }
    }
    out
}

fn main() -> Result<()> {
    color_eyre::install()?;

    // Load your high-res image
    let img = image::open("assets/img/cat.jpg")?.to_rgb8();
    let (dst_w, dst_h) = (640, 360);

    println!("Running Manual Naive Bilinear (OpenCV Style)...");
    let result = naive_bilinear_opencv(&img, dst_w, dst_h);

    // Load the OpenCV Reference from your path
    let reference = fs::read("assets/model/dynamic-onnx/opencv_ref_u8_rgb.bin")
        .expect("OpenCV reference bin not found at the expected path.");

    let mse = calculate_mse(&result, &reference);
    println!("\nManual Aligned MSE: {:.6}", mse);

    let get_patch = |data: &[u8], w: usize| {
        let mut p = Vec::new();
        for y in 0..3 { for x in 0..3 { p.push(data[(y * w + x) * 3]); } }
        p
    };

    println!("\nOpenCV Reference (Top-Left 3x3 Red): [204, 135, 204, 175, 203, 207, 153, 195, 190]");
    println!("Rust Manual      (Top-Left 3x3 Red): {:?}", get_patch(&result, dst_w as usize));

    if mse < 1.0 {
        println!("\nSUCCESS: Manual Naive Bilinear matches OpenCV! Accuracy restored.");
    } else {
        println!("\nSTILL DIFFERENT: This is unexpected if the reference was generated with cv2.resize INTER_LINEAR.");
    }

    Ok(())
}

fn calculate_mse(a: &[u8], b: &[u8]) -> f64 {
    a.iter().zip(b.iter()).map(|(&p1, &p2)| (p1 as f64 - p2 as f64).powi(2)).sum::<f64>() / a.len() as f64
}