# /// script
# requires-python = "==3.12.*"
# dependencies = [
#    "onnxruntime==1.24.4",
#    "opencv-python==4.13.0.92",
#    "numpy",
#    "torch==2.11.0",
#    "ultralytics>=8.4.31",
#    "clip @ git+https://github.com/ultralytics/CLIP.git",
# ]
# ///
import cv2
import numpy as np
import onnxruntime as ort
import torch
import torchvision
from ultralytics import YOLOE
from pathlib import Path

def letterbox_rectangular(img, new_shape=640, color=(114, 114, 114), stride=32):
    shape = img.shape[:2]
    r = min(new_shape / shape[0], new_shape / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape - new_unpad[0], new_shape - new_unpad[1]
    dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img

def xywh2xyxy(x):
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right."""
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def debug_onnx_market_person_count():
    onnx_path = "assets/prompt_model/yoloe-26x-text-dynamic.onnx"
    img_path = Path("assets/img/market.jpg")
    conf_threshold = 0.15
    iou_threshold = 0.7

    # 1. Setup
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    yoloe_helper = YOLOE("yoloe-26x-seg.pt")

    # 2. Preprocess
    img0 = cv2.imread(str(img_path))
    canvas = letterbox_rectangular(img0, 640)
    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    input_tensor = canvas_rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

    # 3. Inference
    with torch.no_grad():
        embeddings = yoloe_helper.get_text_pe(["person"]).cpu().numpy()

    outputs = session.run(None, {'images': input_tensor, 'text_embeddings': embeddings})
    preds = np.squeeze(outputs[0]).T # [5040, 37]

    # 4. Torch-based NMS (Replicating Ultralytics exactly)
    preds = torch.from_numpy(preds)

    # Filter by confidence
    # Column 4 is 'person' score
    keep = preds[:, 4] > conf_threshold
    preds = preds[keep]

    if preds.shape[0] > 0:
        # Convert xywh to xyxy
        boxes = xywh2xyxy(preds[:, :4])
        scores = preds[:, 4]

        # torchvision.ops.nms is the engine behind YOLO
        # It handles overlaps with high floating-point precision
        indices = torchvision.ops.nms(boxes, scores, iou_threshold)

        # Limit to max_det=300 (YOLO default)
        indices = indices[:300]
        person_count = len(indices)
    else:
        person_count = 0

    print(f"\n{'='*30}")
    print(f"ONNX DEBUG RESULTS (TORCH NMS)")
    print(f"{'='*30}")
    print(f"Image: {img_path.name}")
    print(f"Detected Persons: {person_count}")
    print(f"{'='*30}\n")

if __name__ == "__main__":
    debug_onnx_market_person_count()