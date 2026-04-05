# /// script
# requires-python = "==3.12.*"
# dependencies = [
#    "torch==2.11.0",
#    "ultralytics==8.4.31",
#    "onnxscript==0.6.2",
# ]
# ///

import torch
import json
from pathlib import Path
from ultralytics import YOLO

# Wrapper for FULL segmentation (Boxes + Coefficients + Prototypes)
class YOLO26SegWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # YOLOE-26 raw output [0] is the inference tuple
        inf_out = self.model(x)[0]
        # Return [Batch, 300, 38] and [Batch, 32, 160, 160]
        return inf_out[0], inf_out[1]

# Wrapper for DETECTION ONLY (Strip masks and prune proto-head)
class YOLO26DetWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        inf_out = self.model(x)[0]
        # inf_out[0] is [Batch, 300, 38]
        # Columns 0-3: Box, 4: Score, 5: Class, 6-37: Mask Coeffs
        # We only take 0 through 5 (Box + Score + Class)
        detections = inf_out[0][..., :6]
        return detections

def export_all_variants():
    scales = ["n", "s", "m", "l", "x"]
    output_dir = Path("assets/model/prompt_free")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Common settings
    img_size = 640
    fake_input = torch.randn(1, 3, img_size, img_size)

    for scale in scales:
        base_name = f"yoloe-26{scale}-seg-pf"
        pt_file = f"{base_name}.pt"

        print(f"\n{'='*50}\nProcessing Scale: {scale.upper()}\n{'='*50}")

        try:
            # 1. Load the underlying YOLOE-26 Prompt-Free model
            yolo_model = YOLO(pt_file)

            # 2. Save vocabulary once (it's the same for all 26-pf models)
            if scale == "n":
                vocab = [yolo_model.names[i] for i in range(len(yolo_model.names))]
                with open(output_dir / "vocabulary_4585.json", "w", encoding="utf-8") as f:
                    json.dump(vocab, f, indent=2)

            # --- VARIANT A: MASKED (Segmentation) ---
            seg_onnx = output_dir / f"yoloe-26{scale}-seg-pf.onnx"
            print(f"Exporting SEGMENTATION variant to {seg_onnx.name}...")
            torch.onnx.export(
                YOLO26SegWrapper(yolo_model.model).eval(),
                fake_input,
                str(seg_onnx),
                export_params=True,
                opset_version=18,
                do_constant_folding=True,
                input_names=['images'],
                output_names=['detections', 'protos'],
                dynamic_axes={
                    'images': {0: 'batch', 2: 'height', 3: 'width'},
                    'detections': {0: 'batch', 1: 'anchors'},
                    'protos': {0: 'batch', 2: 'p_height', 3: 'p_width'}
                }
            )

            # --- VARIANT B: MASKLESS (Detection Only) ---
            det_onnx = output_dir / f"yoloe-26{scale}-det-pf.onnx"
            print(f"Exporting DETECTION variant to {det_onnx.name}...")
            torch.onnx.export(
                YOLO26DetWrapper(yolo_model.model).eval(),
                fake_input,
                str(det_onnx),
                export_params=True,
                opset_version=18,
                do_constant_folding=True,
                input_names=['images'],
                output_names=['detections'], # 'protos' is NOT outputted here
                dynamic_axes={
                    'images': {0: 'batch', 2: 'height', 3: 'width'},
                    'detections': {0: 'batch', 1: 'anchors'}
                }
            )

            print(f"Done scale {scale}.")

        except Exception as e:
            print(f"Failed {scale}: {e}")

if __name__ == "__main__":
    export_all_variants()