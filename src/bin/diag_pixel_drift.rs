use color_eyre::Result;
use fast_image_resize::images::Image;
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
use image::{DynamicImage, GenericImageView, imageops::FilterType as ImageFilter};
use std::fs;

fn main() -> Result<()> {
    let target_w = 640;
    let target_h = 360;
    let img = image::open("assets/img/cat.jpg")?;
    let ref_bytes = fs::read("assets/model/dynamic-onnx/opencv_ref_u8_rgb.bin")?;

    // Helper to get 5x5 red patch from raw bytes
    let get_patch = |data: &[u8], w: usize| {
        let mut patch = Vec::new();
        for y in 0..5 {
            for x in 0..5 {
                patch.push(data[(y * w + x) * 3]); // Red channel
            }
        }
        patch
    };

    let py_patch = get_patch(&ref_bytes, target_w as usize);
    println!("OpenCV Reference (Top-Left 5x5 Red):");
    print_patch(&py_patch);

    // Test image crate Bilinear
    let mut resizer = img.resize_exact(target_w, target_h, ImageFilter::Triangle).to_rgb8();
    let rust_patch = get_patch(resizer.as_raw(), target_w as usize);
    println!("\nimage crate Bilinear:");
    print_patch(&rust_patch);

    // Test fast_image_resize Bilinear
    let fir_res = resize_fir(&img, target_w, target_h, FilterType::Bilinear)?;
    let fir_patch = get_patch(&fir_res, target_w as usize);
    println!("\nfast_image_resize Bilinear:");
    print_patch(&fir_patch);

    Ok(())
}

fn print_patch(p: &[u8]) {
    for i in 0..5 {
        println!("{:?}", &p[i*5 .. (i+1)*5]);
    }
}

fn resize_fir(img: &DynamicImage, w: u32, h: u32, filter: FilterType) -> Result<Vec<u8>> {
    let (src_w, src_h) = img.dimensions();
    let src_image = Image::from_vec_u8(src_w, src_h, img.to_rgb8().into_raw(), PixelType::U8x3)?;
    let mut dst_image = Image::new(w, h, PixelType::U8x3);
    let mut resizer = Resizer::new();
    resizer.resize(&src_image, &mut dst_image, &ResizeOptions::new().resize_alg(ResizeAlg::Convolution(filter)))?;
    Ok(dst_image.into_vec())
}