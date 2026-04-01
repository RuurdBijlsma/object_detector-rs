# /// script
# requires-python = "==3.12.*"
# dependencies = [
#    "onnxruntime==1.24.4",
#    "opencv-python==4.13.0.92",
#    "numpy",
#    "torch==2.11.0",
#    "torchvision",
#    "clip @ git+https://github.com/ultralytics/CLIP.git",
# ]
# ///
import cv2
import numpy as np
import onnxruntime as ort
import torch
import torchvision
import torch.nn.functional as F
import clip
from pathlib import Path


# --- HELPERS ---

def get_color(i):
    """YOLO-style palette."""
    palette = [(255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29), (207, 210, 49), (72, 249, 10)]
    color = palette[i % len(palette)]
    return (color[2], color[1], color[0])


def letterbox_rectangular(img, new_shape=640, stride=32):
    shape = img.shape[:2]  # h, w
    r = min(new_shape / shape[0], new_shape / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

    dw, dh = (new_shape - new_unpad[0]) % stride, (new_shape - new_unpad[1]) % stride
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, (r, r), (dw, dh)


def crop_mask(masks, boxes):
    """Zeroes out mask pixels outside the bounding box to reduce noise."""
    n, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # [N, 1, 1]
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]
    c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]
    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


# --- CORE ---

class PureCLIPEncoder:
    def __init__(self, model_path="mobileclip2_b.ts"):
        print(f"Loading CLIP: {model_path}")
        self.model = torch.jit.load(model_path, map_location="cpu").eval()

    def get_embeddings(self, classes):
        tokens = clip.tokenize(classes)
        with torch.no_grad():
            output = self.model(tokens)
            embeddings = output[0] if isinstance(output, (list, tuple)) else output
            embeddings = embeddings.float()
        return embeddings.numpy()[None]


class YOLOE_Pure_Inference:
    def __init__(self, onnx_path, clip_path="mobileclip2_b.ts"):
        self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        self.clip = PureCLIPEncoder(clip_path)

    def run(self, img_path, classes, conf_threshold=0.15):
        img0 = cv2.imread(str(img_path))
        if img0 is None: return None, []

        # 1. Preprocess
        canvas, ratio, pad = letterbox_rectangular(img0)
        c_h, c_w = canvas.shape[:2]
        img_tensor = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0

        # 2. CLIP Embeddings
        text_pe = self.clip.get_embeddings(classes)

        # 3. ONNX Run
        outputs = self.session.run(None, {'images': img_tensor, 'text_embeddings': text_pe})
        preds, protos = np.squeeze(outputs[0]).T, np.squeeze(outputs[1])
        nc = len(classes)

        # 4. Filter & Decode
        scores_matrix = preds[:, 4: 4 + nc]
        max_scores = np.max(scores_matrix, axis=1)
        class_ids = np.argmax(scores_matrix, axis=1)

        mask = max_scores > conf_threshold
        if not np.any(mask): return img0, []

        boxes_raw = preds[mask, :4]
        x1 = boxes_raw[:, 0] - boxes_raw[:, 2] / 2
        y1 = boxes_raw[:, 1] - boxes_raw[:, 3] / 2
        x2 = boxes_raw[:, 0] + boxes_raw[:, 2] / 2
        y2 = boxes_raw[:, 1] + boxes_raw[:, 3] / 2
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        keep = torchvision.ops.nms(torch.from_numpy(boxes_xyxy).float(),
                                   torch.from_numpy(max_scores[mask]).float(), 0.7).numpy()

        if len(keep) == 0: return img0, []

        # 5. Finalize Bboxes & Scores
        final_boxes = boxes_xyxy[keep]
        final_scores = max_scores[mask][keep]
        final_cids = class_ids[mask][keep]
        final_coeffs = preds[mask][keep, 4 + nc:]

        # 6. MASK PROCESSING (FIXED)
        num_dets = len(keep)
        c_protos, mh, mw = protos.shape

        # Matrix multiply [N, 32] @ [32, 160*160] -> [N, 160, 160]
        masks = (torch.from_numpy(final_coeffs) @ torch.from_numpy(protos).view(c_protos, -1)).view(num_dets, mh, mw)
        masks = torch.sigmoid(masks)

        # Upscale to ACTUAL CANVAS SIZE (not just 640x640)
        masks = F.interpolate(masks[None], (c_h, c_w), mode='bilinear', align_corners=False)[0]

        # Crop noise outside boxes (using canvas-space boxes)
        masks = crop_mask(masks, torch.from_numpy(final_boxes))

        # Rescale back to original image pixels
        ih, iw = img0.shape[:2]
        ph, pw = int(pad[1]), int(pad[0])
        unh, unw = c_h - 2 * ph, c_w - 2 * pw

        # Slice padding -> [N, unh, unw]
        masks = masks[:, ph:ph + unh, pw:pw + unw]
        # Resize to original
        masks = F.interpolate(masks[None], (ih, iw), mode='bilinear', align_corners=False)[0].gt(0.5).numpy()

        # Rescale boxes
        final_boxes[:, [0, 2]] = (final_boxes[:, [0, 2]] - pad[0]) / ratio[0]
        final_boxes[:, [1, 3]] = (final_boxes[:, [1, 3]] - pad[1]) / ratio[1]

        return img0, list(zip(final_boxes, final_scores, final_cids, masks))


def visualize(img, detections, classes, out_path):
    canvas = img.copy()
    overlay = canvas.copy()
    for box, score, cid, mask in detections:
        color = get_color(int(cid))
        # Fill mask
        overlay[mask] = color
        # Draw box and label
        cv2.rectangle(canvas, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
        label = f"{classes[cid]} {score:.2f}"
        cv2.putText(canvas, label, (int(box[0]), int(box[1] - 5)), 0, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # 0.4 opacity for masks
    cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0, canvas)
    cv2.imwrite(str(out_path), canvas)


def main():
    ONNX = "yoloe-26x-pure-clip.onnx"
    IMG_DIR = Path("img")
    OUT_DIR = Path("output/pure_onnx")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    CLASSES = ["cat", "car", "van", "sign", "person", "lamp", "watermelon"]

    inf = YOLOE_Pure_Inference(ONNX)

    # List images
    imgs = list(IMG_DIR.glob("*.jpg")) + list(IMG_DIR.glob("*.png"))

    for img_p in imgs:
        print(f"Processing {img_p.name}...")
        img, dets = inf.run(img_p, CLASSES)
        if img is not None:
            visualize(img, dets, CLASSES, OUT_DIR / f"pure_{img_p.name}")


if __name__ == "__main__":
    main()