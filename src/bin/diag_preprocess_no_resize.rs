use color_eyre::Result;
use image::{GenericImageView, ImageBuffer, Rgb};
use ndarray::{Array4, s};
use std::fs;

fn main() -> Result<()> {
    // Load the SAME PRE-RESIZED reference
    let img = image::open("assets/img/cat_640x360_ref.png")?;
    let (w, h) = img.dimensions(); // 640, 360
    dbg!(w, h);

    let (w_pad, h_pad) = (640, 384);
    let top = (h_pad - h) / 2;
    let left = (w_pad - w) / 2;

    let mut canvas = ImageBuffer::from_pixel(w_pad, h_pad, Rgb([114, 114, 114]));
    image::imageops::overlay(&mut canvas, &img.to_rgb8(), left as i64, top as i64);

    let mut input = Array4::zeros((1, 3, h_pad as usize, w_pad as usize));
    for (x, y, rgb) in canvas.enumerate_pixels() {
        input[[0, 0, y as usize, x as usize]] = (rgb[0] as f32) / 255.0;
        input[[0, 1, y as usize, x as usize]] = (rgb[1] as f32) / 255.0;
        input[[0, 2, y as usize, x as usize]] = (rgb[2] as f32) / 255.0;
    }

    println!("--- Rust (No Resize) ---");
    println!("Mean: {:.8}", input.mean().unwrap());
    let sample = input.slice(s![0, 0, top as usize..top as usize + 3, left as usize..left as usize + 3]);
    println!("Top-Left 3x3 (Red):\n{:?}", sample);

    if let Ok(py_bytes) = fs::read("assets/model/dynamic-onnx/python_ref_padded.bin") {
        let py_f32: Vec<f32> = py_bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        let py_tensor = Array4::from_shape_vec((1, 3, h_pad as usize, w_pad as usize), py_f32)?;
        let mse = (&input - &py_tensor).mapv(|x| x * x).mean().unwrap();
        println!("MSE vs Python: {:.10}", mse);
    }

    Ok(())
}