# /// script
# requires-python = "==3.12.*"
# dependencies = [
#    "onnxruntime",
#    "numpy",
#    "opencv-python",
# ]
# ///
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path


def nms_sweep():
    # 1. Setup
    onnx_path = "../../assets/prompt_model/yoloe-26x-text-dynamic.onnx"

    # Load the tensors we verified earlier
    try:
        img_tensor = np.load("ref_img_tensor.npy")
        text_pe = np.load("ref_text_pe.npy")
    except:
        print("Run diag_export_reference.py first!")
        return

    # 2. Run Inference
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    outputs = session.run(None, {'images': img_tensor, 'text_embeddings': text_pe})

    # 3. Prepare Boxes
    preds = np.squeeze(outputs[0]).T
    boxes = preds[:, :4]
    scores = preds[:, 4]

    # Convert xywh (center) to xyxy
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    # 4. Sweep
    print(f"\n{'=' * 40}")
    print(f"NMS THRESHOLD SWEEP")
    print(f"{'=' * 40}")
    print(f"Confidence Threshold: 0.15 (Fixed)")
    print(f"{'IOU Thresh':<12} | {'Detected People'}")
    print("-" * 30)

    # Test various IOU thresholds
    # 0.45 is your current. 0.7 is the official YOLO default.
    for iou in [0.3, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9]:
        indices = cv2.dnn.NMSBoxes(
            boxes_xyxy.tolist(),
            scores.tolist(),
            score_threshold=0.15,
            nms_threshold=iou
        )
        count = len(indices)
        marker = "<-- YOUR CURRENT" if iou == 0.45 else ("<-- YOLO DEFAULT" if iou == 0.7 else "")
        print(f"{iou:<12.2f} | {count:<15} {marker}")

    print("=" * 40)
    print("If 0.7 results in ~43, update your onnx_basic.py logic.")


if __name__ == "__main__":
    nms_sweep()