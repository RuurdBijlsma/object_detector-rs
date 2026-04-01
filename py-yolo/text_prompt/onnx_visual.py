# /// script
# requires-python = "==3.12.*"
# dependencies = [
#    "onnxruntime==1.24.4",
#    "opencv-python==4.13.0.92",
#    "numpy",
#    "torch==2.11.0",
#    "torchvision",
#    "ultralytics>=8.4.31",
#    "clip @ git+https://github.com/ultralytics/CLIP.git",
# ]
# ///
import cv2
import numpy as np
import onnxruntime as ort
import torch
import torchvision
import torch.nn.functional as F
from ultralytics import YOLOE
from ultralytics.utils.plotting import Annotator, colors
from pathlib import Path


def letterbox_rectangular(img, new_shape=640, color=(114, 114, 114), stride=32):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw = np.mod(dw, stride)
    dh = np.mod(dh, stride)

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img, (r, r), (dw, dh)


def process_mask(protos, masks_in, bboxes, shape):
    c, mh, mw = protos.shape
    masks = (torch.from_numpy(masks_in) @ torch.from_numpy(protos).view(c, -1)).view(-1, mh, mw)
    masks = torch.sigmoid(masks)

    canvas_h, canvas_w = int(mh * 4), int(mw * 4)
    masks = F.interpolate(masks[None], (canvas_h, canvas_w), mode='bilinear', align_corners=False)[0]
    masks = crop_mask(masks, torch.from_numpy(bboxes))

    return masks.gt(0.5)


def crop_mask(masks, boxes):
    n, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]
    c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]
    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


class YOLOE_ONNX_Visualizer:
    def __init__(self, onnx_path, model_pt="yoloe-26x-seg.pt"):
        self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        self.yoloe_helper = YOLOE(model_pt)
        self.img_size = 640

    def run_inference(self, img_path, classes, conf_threshold=0.15, iou_threshold=0.7):
        img0 = cv2.imread(str(img_path))
        if img0 is None:
            return None, []

        img_canvas, ratio, pad = letterbox_rectangular(img0, new_shape=self.img_size)
        img_rgb = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2RGB)
        img_tensor = img_rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

        with torch.no_grad():
            text_pe = self.yoloe_helper.get_text_pe(classes).cpu().numpy()

        outputs = self.session.run(None, {
            'images': img_tensor,
            'text_embeddings': text_pe
        })

        preds = np.squeeze(outputs[0]).T
        protos = np.squeeze(outputs[1])
        num_classes = len(classes)

        boxes_raw = preds[:, :4]
        scores_matrix = preds[:, 4: 4 + num_classes]
        coeffs_raw = preds[:, 4 + num_classes:]

        max_scores = np.max(scores_matrix, axis=1)
        class_indices = np.argmax(scores_matrix, axis=1)

        mask = max_scores > conf_threshold
        if not np.any(mask):
            return img0, []

        boxes_raw = boxes_raw[mask]
        scores = max_scores[mask]
        class_indices = class_indices[mask]
        coeffs_raw = coeffs_raw[mask]

        x1 = boxes_raw[:, 0] - boxes_raw[:, 2] / 2
        y1 = boxes_raw[:, 1] - boxes_raw[:, 3] / 2
        x2 = boxes_raw[:, 0] + boxes_raw[:, 2] / 2
        y2 = boxes_raw[:, 1] + boxes_raw[:, 3] / 2
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        keep_indices = torchvision.ops.nms(
            torch.from_numpy(boxes_xyxy).float(),
            torch.from_numpy(scores).float(),
            iou_threshold
        ).cpu().numpy()

        if len(keep_indices) == 0:
            return img0, []

        final_boxes = np.atleast_2d(boxes_xyxy[keep_indices])
        final_scores = np.atleast_1d(scores[keep_indices])
        final_cls = np.atleast_1d(class_indices[keep_indices])
        final_coeffs = np.atleast_2d(coeffs_raw[keep_indices])

        # Process Masks (Canvas Space)
        masks = process_mask(protos, final_coeffs, final_boxes, img_canvas.shape[:2])

        ih, iw = img0.shape[:2]
        pad_w, pad_h = int(pad[0]), int(pad[1])
        unpad_h, unpad_w = img_canvas.shape[0] - 2 * pad_h, img_canvas.shape[1] - 2 * pad_w

        # Slice padding and resize to original image
        masks = masks[:, pad_h:pad_h + unpad_h, pad_w:pad_w + unpad_w]
        masks = F.interpolate(masks[None].float(), (ih, iw), mode='bilinear', align_corners=False)[0].gt(0.5)

        # Scale boxes back to original image
        final_boxes[:, [0, 2]] -= pad[0]
        final_boxes[:, [1, 3]] -= pad[1]
        final_boxes /= ratio[0]

        return img0, list(zip(final_boxes, final_scores, final_cls, masks))

    def visualize(self, img0, detections, classes, output_path):
        annotator = Annotator(img0.copy(), line_width=2, font_size=12)

        if len(detections) > 0:
            boxes, scores, cls_ids, masks = zip(*detections)

            # --- FIX: Convert stacked masks to NumPy for Annotator ---
            masks_np = torch.stack(list(masks)).cpu().numpy()

            # Draw Masks
            annotator.masks(masks_np,
                            colors=[colors(int(x), True) for x in cls_ids],
                            alpha=0.5)

            # Draw Boxes and Labels
            for i, (box, score, cls_id) in enumerate(zip(boxes, scores, cls_ids)):
                label = f"{classes[int(cls_id)]} {score:.2f}"
                color = colors(int(cls_id), True)
                annotator.box_label(box, label, color=color)

        cv2.imwrite(str(output_path), annotator.result())


def main():
    ONNX_MODEL = "../../assets/prompt_model/yoloe-26x-text-dynamic.onnx"
    IMG_DIR = Path("img")
    OUT_DIR = Path("output/onnx_visual_results")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    MY_CLASSES = ["cat", "car", "van", "sign", "person", "lamp", "watermelon"]

    print(f"--- Initializing ONNX Segmentation Visualizer ---")
    viz = YOLOE_ONNX_Visualizer(ONNX_MODEL)

    image_extensions = [".jpg", ".jpeg", ".png", ".webp"]
    images = [f for f in IMG_DIR.iterdir() if f.suffix.lower() in image_extensions]

    for img_p in images:
        print(f"Processing: {img_p.name}")
        original_img, detections = viz.run_inference(img_p, MY_CLASSES)

        if original_img is None:
            continue

        out_p = OUT_DIR / f"onnx_pred_{img_p.name}"
        viz.visualize(original_img, detections, MY_CLASSES, out_p)

        found_labels = [MY_CLASSES[int(d[2])] for d in detections]
        print(f"  - Found: {found_labels if found_labels else 'Nothing'}")
        print(f"  - Saved to: {out_p}")

    print(f"\nDone! Visual results saved to: {OUT_DIR.absolute()}")


if __name__ == "__main__":
    main()