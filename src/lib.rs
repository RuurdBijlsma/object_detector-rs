//! # `object_detector`
//!
//! Easy-to-use object detection and instance segmentation in Rust, powered by ONNX Runtime and the YOLOE-26
//! (Real-Time Seeing Anything) model family.
//!
//! ![Object Detector](https://github.com/RuurdBijlsma/object_detector-rs/raw/main/.github/img_market.jpg)
//!
//! `object_detector` allows you to detect and segment virtually any object in an image. It supports two main modes:
//! - **Prompt-Free**: Uses a built-in vocabulary of 4,500+ tags.
//! - **Promptable**: Describe what to find in natural language (e.g., "blue toaster", "person with a hat").
//!
//! ## Key Features
//!
//! - **Open-Vocabulary Detection**: Detect objects using arbitrary text prompts.
//! - **Built-in Vocabulary**: Detect 4,585 categories out-of-the-box in Prompt-Free mode.
//! - **Instance Segmentation**: High-quality pixel-level masks for every detection.
//! - **Automatic Model Management**: Models are automatically downloaded from Hugging Face on demand.
//! - **Hardware Acceleration**: Full support for `CUDA`, `TensorRT`, `CoreML`, `DirectML`, and more via `ort`.
//!
//! ## Usage
//!
//! The following example demonstrates how to use the **Prompt-Free** mode to detect objects using the default vocabulary.
//!
//! ```rust
//! use object_detector::{DetectorType, ObjectDetector};
//!
//! #[tokio::main]
//! async fn main() -> color_eyre::Result<()> {
//!     color_eyre::install()?;
//!
//!     // Initialize the detector (automatically pulls models from Hugging Face)
//!     let mut detector = ObjectDetector::from_hf(DetectorType::PromptFree)
//!         .build()
//!         .await?;
//!
//!     // Load an image
//!     let img = image::open("assets/img/fridge.jpg")?;
//!
//!     // Run inference
//!     let results = detector.predict(&img).call()?;
//!
//!     for det in results {
//!         println!("[{:>10}] Score: {:.4}", det.tag, det.score);
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Configuration
//!
//! You can customize the model scale (Nano to `XLarge`), enable/disable masks for performance, and configure hardware acceleration.
//!
//! ```rust
//! use object_detector::{DetectorType, ModelScale, ObjectDetector};
//! use ort::ep::CUDA;
//!
//! #[tokio::main]
//! async fn main() -> color_eyre::Result<()> {
//!     let mut detector = ObjectDetector::from_hf(DetectorType::Promptable) // Choose Promptable or PromptFree
//!         .scale(ModelScale::Large)      // Choose from Nano to XLarge
//!         .include_mask(true)            // Set to false for faster bounding-box-only detection
//!         .with_execution_providers(&[
//!             // Choose execution provider (error_on_failure is optional, it helps detect failure to use EP)
//!             CUDA::default().build().error_on_failure()
//!         ])
//!         .build()
//!         .await?;
//!
//!     let img = image::open("assets/img/market.jpg")?;
//!
//!     let results = detector
//!         .predict(&img)
//!         .labels(&["lamp", "person"])   // Only required for Promptable mode
//!         .confidence_threshold(0.35)    // Filter out low-certainty detections
//!         .intersection_over_union(0.7)  // Control overlap handling (NMS)
//!         .call()?;
//!
//!     for det in results {
//!         println!("Detected: {} with score {}", det.tag, det.score);
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Model Selection Guide
//!
//! This crate utilizes **YOLOE-26 (Real-Time Seeing Anything)**, a state-of-the-art open-vocabulary model family built upon
//! the [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/) architecture. Unlike traditional YOLO models
//! limited to a fixed set of categories (like COCO's 80 classes), YOLOE-26 can detect and segment virtually any object.
//!
//! ### Performance Benchmarks
//!
//! The following results demonstrate the execution time (latency) across different scales and modes.
//!
//! ![YOLOE-26 Benchmarks](https://github.com/RuurdBijlsma/object_detector-rs/raw/main/.github/benchmarks/benchmark_grid.png)
//!
//! > **Benchmark Environment:** Ryzen 5800X3D CPU | RTX 2080Ti GPU | CUDA Execution Provider.
//! > For **Promptable** modes, text embeddings (CLIP) are cached, as they would be when performing repeated inference
//! > during normal use.
//!
//! ### Model Scales: Speed vs. Accuracy
//!
//! The crate supports five model scales. Choosing the right one depends on your hardware and accuracy requirements:
//!
//! | Scale          | Parameters | Description                            | Best For                                                  |
//! |:---------------|:-----------|:---------------------------------------|:----------------------------------------------------------|
//! | **`Nano` (N)**   | ~4.8M      | Fastest inference speed.               | Edge devices, mobile applications, and low-power CPUs.    |
//! | **`Small` (S)**  | ~13.1M     | Balanced efficiency and accuracy.      | Real-time desktop applications and mid-range `IoT` devices. |
//! | **`Medium` (M)** | ~27.9M     | High accuracy with moderate latency.   | GPU inference where precision is a priority.              |
//! | **`Large` (L)**  | ~32.3M     | **(Default)** High-fidelity detection. | Server-side processing and high-precision robotics.       |
//! | **`XLarge` (X)** | ~69.9M     | Maximum accuracy available.            | Non-real-time analysis and maximum-precision tasks.       |
//!
//! ### Operating Modes: Prompt-Free vs. Promptable
//!
//! #### **Prompt-Free Mode (`DetectorType::PromptFree`)**
//!
//! * **Mechanism:** Uses a built-in vocabulary of **4,585 classes** based on the RAM++ tag set.
//! * **Characteristics:** This mode is highly efficient as it does not require external text encoding. It functions like a
//!   traditional YOLO model but with a much larger pre-defined label set.
//! * **Constraint:** You are limited to the built-in vocabulary; you cannot define custom classes at runtime in this mode.
//!
//! #### **Promptable Mode (`DetectorType::Promptable`)**
//!
//! * **Mechanism:** Uses a text-alignment module to compare image features against **CLIP text embeddings**.
//! * **Characteristics:** This mode provides high flexibility. You can prompt the model with specific strings such as
//!   `"blue toaster"` or `"peace symbol"`.
//! * **Constraint:** Requires a CLIP model (handled automatically via `open_clip_inference`) to generate embeddings for
//!   your labels, which adds a small initial overhead.
//!
//! ### Task Selection: Mask (Segmentation) vs. Detection
//!
//! You can toggle the `include_mask(bool)` parameter during builder initialization.
//!
//! * **Instance Segmentation (`include_mask(true)`):** Includes a pixel-level `ObjectMask` for every detected object. This
//!   is useful for tasks like background removal, object isolation, or precise spatial analysis.
//! * **Object Detection (`include_mask(false)`):** Returns only the bounding boxes, tags and scores for detected objects.
//! * **Performance Impact:**
//! * On **GPU**, detecting only bounding boxes is approximately **15-33% faster** than segmentation.
//! * On **CPU**, the difference is more significant, running up to **2x faster** when masks are disabled.
//!
//! ---
//!
//! ### Understanding Thresholds
//!
//! Fine-tuning the detection thresholds is essential for balancing precision (reducing false positives) and recall (finding
//! all objects).
//!
//! #### **Confidence Threshold (`confidence_threshold`)**
//!
//! This parameter determines the minimum certainty required for a detection to be returned.
//!
//! * **Default:** `0.25`
//! * **Low values (e.g., 0.1):** The model will be more sensitive and detect more objects, but it will also return more "
//!   noise" or incorrect detections.
//! * **High values (e.g., 0.7):** The model will only return objects it is very certain about. This reduces false positives
//!   but may cause it to miss partially obscured or small objects.
//!
//! #### **Intersection Over Union (`intersection_over_union`)**
//!
//! This parameter controls the Non-Maximum Suppression (NMS) process, which decides how to handle multiple overlapping
//! boxes for the same object.
//!
//! * **Default:** `0.7`
//! * **Mechanism:** When two boxes overlap significantly, the IOU value measures the ratio of the overlap area to the total
//!   area of both boxes. If the IOU is higher than your threshold, the box with the lower confidence score is discarded.
//! * **High values (e.g., 0.8):** More tolerant of overlapping boxes. Useful in crowded scenes where objects are physically
//!   close to each other.
//! * **Low values (e.g., 0.3):** More aggressive at removing overlaps. Useful if the model is producing "double" detections
//!   for a single object.
//!
//! ---
//!
//! ## Cargo Features
//!
//! | Feature  | Default | Description                                             | Dependencies      |
//! |:---------|:-------:|:--------------------------------------------------------|:------------------|
//! | `hf-hub` | **Yes** | Enable automatic model downloading from Hugging Face.   | `hf-hub`, `tokio` |
//! | `serde`  | **Yes** | Enable `Serialize`/`Deserialize` for detection structs. | `serde`           |
//!
//! The main `ort` Cargo features are also forwarded.
//!
//! * ORT Cargo features: <https://ort.pyke.io/setup/cargo-features>
//! * ORT Execution providers: <https://ort.pyke.io/perf/execution-providers>
//!
//! ## Troubleshooting
//!
//! ### Link error - ORT
//!
//! If a link error happens while building, this is probably due to ORT. You can try the `load-dynamic` Cargo feature to
//! resolve this. You'll need to point to an instance of the `ONNXRuntime` library on your system via an environment variable.
//! See the next section for more info.
//!
//! ### [When using `load-dynamic` feature] ONNX Runtime Library Not Found
//!
//! `OnnxRuntime` is dynamically loaded, so if it's not found correctly, then download the correct onnxruntime library
//! from [GitHub Releases](http://github.com/microsoft/onnxruntime/releases).
//!
//! Then put the dll/so/dylib location in your `PATH`, or point the `ORT_DYLIB_PATH` env var to it.
//!
//! ## Output samples
//!
//! ### Car park - no masks - prompted: `["car"]`
//!
//! ![Car park](https://github.com/RuurdBijlsma/object_detector-rs/raw/main/.github/parking_lot.jpg)
//!
//! ### Street view - with mask - prompt free
//!
//! ![Street view](https://github.com/RuurdBijlsma/object_detector-rs/raw/main/.github/streetview.jpg)
//!
//! ### Cat - with mask - prompted `["cat"]`
//!
//! ![Cat](https://github.com/RuurdBijlsma/object_detector-rs/raw/main/.github/img_cat.jpg)

#![allow(
    clippy::missing_panics_doc,
    clippy::missing_errors_doc,
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]

mod error;
pub mod model_manager;
pub mod object_detector;
pub mod predictor;
mod structs;

pub use error::ObjectDetectorError;
pub use object_detector::ObjectDetector;
pub use predictor::{PromptFreeDetector, PromptableDetector, YoloPreprocessMeta};
pub use structs::*;
