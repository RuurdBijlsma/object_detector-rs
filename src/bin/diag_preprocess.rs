use color_eyre::Result;
use image::{imageops::FilterType, GenericImageView, ImageBuffer, Rgb};
use ndarray::{s, Array4};
use std::fs;

fn main() -> Result<()> {
    let img_path = "assets/img/cat_640x360_ref.png";
    let img = image::open(img_path)?;
    let (w0, h0) = img.dimensions();
    let (imgsz, stride) = (640, 32);

    // 1. Calculate dims
    let r = imgsz as f32 / (w0.max(h0) as f32);
    let new_unpad_w = (w0 as f32 * r).round() as u32;
    let new_unpad_h = (h0 as f32 * r).round() as u32;
    let w_pad = ((new_unpad_w as f32 / stride as f32).ceil() * stride as f32) as u32;
    let h_pad = ((new_unpad_h as f32 / stride as f32).ceil() * stride as f32) as u32;

    // 2. Resize (CRITICAL: Triangle matches cv2 INTER_LINEAR)
    let resized = img.resize_exact(new_unpad_w, new_unpad_h, FilterType::Triangle);

    // 3. Padding
    let mut canvas = ImageBuffer::from_pixel(w_pad, h_pad, Rgb([114, 114, 114]));
    let left = (w_pad - new_unpad_w) / 2;
    let top = (h_pad - new_unpad_h) / 2;
    image::imageops::overlay(&mut canvas, &resized.to_rgb8(), left as i64, top as i64);

    // 4. Normalization to CHW
    let mut input = Array4::zeros((1, 3, h_pad as usize, w_pad as usize));
    for (x, y, rgb) in canvas.enumerate_pixels() {
        input[[0, 0, y as usize, x as usize]] = (rgb[0] as f32) / 255.0;
        input[[0, 1, y as usize, x as usize]] = (rgb[1] as f32) / 255.0;
        input[[0, 2, y as usize, x as usize]] = (rgb[2] as f32) / 255.0;
    }

    println!("--- Rust Preprocess Diag: cat.jpg ---");
    println!("Tensor Shape: {:?}", input.dim());

    let mean = input.mean().unwrap();
    let std = input.std(0.0);
    println!("Mean: {:.8}", mean);
    println!("Std:  {:.8}", std);

    let sample = input.slice(s![0, 0, top as usize..top as usize + 3, left as usize..left as usize + 3]);
    println!("Top-Left 3x3 (Red Channel):\n{:?}", sample);

    // 5. Direct Comparison with Python file
    if let Ok(py_bytes) = fs::read("assets/model/dynamic-onnx/python_preprocess.bin") {
        let py_f32: Vec<f32> = py_bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        let py_tensor = Array4::from_shape_vec((1, 3, h_pad as usize, w_pad as usize), py_f32)?;

        let diff = &input - &py_tensor;
        let mse = diff.mapv(|x| x * x).mean().unwrap();
        println!("\nMean Squared Error vs Python: {:.10}", mse);
        if mse > 1e-6 {
            println!("CRITICAL: Preprocessing differs significantly!");
        } else {
            println!("SUCCESS: Preprocessing is identical (within float tolerance).");
        }
    }

    Ok(())
}