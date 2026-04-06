## Model Selection Guide

This crate utilizes **YOLOE-26 (Real-Time Seeing Anything)**, a state-of-the-art open-vocabulary model family built upon
the [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/) architecture. Unlike traditional YOLO models
limited to a fixed set of categories (like COCO's 80 classes), YOLOE-26 can detect and segment virtually any object.

### Performance Benchmarks

The following results demonstrate the execution time (latency) across different scales and modes.

![YOLOE-26 Benchmarks](.github/benchmarks/benchmark_grid.png)

> [!NOTE]
> **Benchmark Environment:** Ryzen 5800X3D CPU | RTX 2080Ti GPU | CUDA Execution Provider.
> For **Promptable** modes, text embeddings (CLIP) are cached, as they would be when performing repeated inference
> during normal use.

---

### 1. Model Scales: Speed vs. Accuracy

The crate supports five model scales. Choosing the right one depends on your hardware and accuracy requirements:

| Scale          | Description                                             | Best For                                            |
|:---------------|:--------------------------------------------------------|:----------------------------------------------------|
| **Nano (N)**   | ~4.8M Parameters. Fastest inference.                    | Edge devices, high-FPS mobile apps, low-power CPUs. |
| **Small (S)**  | Balanced efficiency. Significantly higher AP than Nano. | Real-time desktop apps, mid-range IoT devices.      |
| **Medium (M)** | High accuracy with moderate latency.                    | Standard GPU inference where precision matters.     |
| **Large (L)**  | **(Default)** High-fidelity detection and segmentation. | Server-side processing and high-precision robotics. |
| **XLarge (X)** | State-of-the-art accuracy. Highest resource usage.      | Non-real-time analysis and maximum-precision tasks. |

### 2. Operating Modes: Prompt-Free vs. Promptable

You can initialize the detector in one of two distinct modes:

#### **Prompt-Free Mode (`DetectorType::PromptFree`)**

* **How it works:** Uses a massive internal vocabulary of **4,585 classes** (based on the RAM++ tag list).
* **Pros:** Extremely fast and "zero-config." It works like a traditional YOLO model but recognizes thousands of objects
  out of the box.
* **Cons:** You cannot add new classes at runtime; you are limited to the built-in vocabulary.

#### **Promptable Mode (`DetectorType::Promptable`)**

* **How it works:** Uses a text-alignment module (RepRTA) to compare image features against **CLIP text embeddings**.
* **Pros:** Infinite flexibility. You can prompt the model with specific strings like `"vintage blue toaster"` or
  `"peace symbol"`.
* **Cons:** Slightly higher overhead. It requires a CLIP model (handled automatically via `open_clip_inference`) to
  generate embeddings for your labels.

### 3. Task Selection: Mask (Segmentation) vs. Detection

When building your detector, you can toggle `include_mask(bool)`.

* **Instance Segmentation (`include_mask(true)`):** Returns a pixel-perfect `ObjectMask` for every detected object. This
  is essential for background removal, object isolation, or measurement tasks.
* **Object Detection (`include_mask(false)`):** Returns only the bounding boxes.
* **Performance Impact:** Detecting only bounding boxes is **10-25% faster** depending on the scale. Mask reconstruction
  requires processing "protos" (mask prototypes) and performing bilinear upsampling, which adds CPU/GPU overhead. If you
  only need to know *where* an object is (box), disable masks for a significant speed boost.

### 🛠 Technical Highlights

* **NMS-Free (End-to-End):** YOLO26 architecture is natively end-to-end. It uses a one-to-one matching strategy during
  training that allows it to predict the final objects directly. This eliminates the traditional "Non-Maximum
  Suppression" bottleneck during export and inference.
* **Embedding Caching:** In Promptable mode, this crate automatically caches the embeddings for your labels. If you call
  `.predict()` repeatedly with the same labels, the text-encoding step is skipped, allowing for true real-time
  performance even when using natural language prompts.
* **Memory Efficiency:** The crate utilizes ONNX Runtime's memory arena and specialized preprocessing to ensure that
  even the `XLarge` models maintain a stable memory footprint during long-running processes.

# Troubleshooting

## Link error - ORT

If a link error happens while building, this is probably due to ORT. You can try the `load-dynamic` cargo feature to
resolve this. You'll need to point to an instance of the ONNXRuntime library on your system via an environment variable.
See the next section for more info.

## [When using `load-dynamic` feature] ONNX Runtime Library Not Found

OnnxRuntime is dynamically loaded, so if it's not found correctly, then download the correct onnxruntime library
from [GitHub Releases](http://github.com/microsoft/onnxruntime/releases).

Then put the dll/so/dylib location in your `PATH`, or point the `ORT_DYLIB_PATH` env var to it.

**PowerShell example:**

* Adjust path to where the dll is.

```powershell
$env:ORT_DYLIB_PATH = "C:/Apps/onnxruntime/lib/onnxruntime.dll"
```

**Shell example:**

```shell
export ORT_DYLIB_PATH="/usr/local/lib/libonnxruntime.so"
```

![img_cat.png](.github/img_cat.png)