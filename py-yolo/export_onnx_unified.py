# /// script
# requires-python = "==3.12.*"
# dependencies = [
#    "torch==2.11.0",
#    "ultralytics==8.4.31",
#    "onnx==1.21.0",
#    "numpy",
#    "onnxscript==0.6.2",
#    "clip @ git+https://github.com/ultralytics/CLIP.git",
# ]
# ///

import torch
import json
import argparse
from pathlib import Path
from ultralytics import YOLO, YOLOE


class PromptableWrapper(torch.nn.Module):
    """Handles the manual forward loop required for TPE injection."""

    def __init__(self, model_pt, export_mask=True):
        super().__init__()
        self.model = model_pt.model
        self.export_mask = export_mask
        self.head = self.model.model[-1]
        self.head.end2end = False
        self.head.export = True

    def forward(self, x, text_embeddings):
        # 1. Generate Text Projection
        projected_text = self.head.get_tpe(text_embeddings)

        # 2. Manual Forward Loop (Preserves skip connections)
        y = []
        feat = x
        for m in self.model.model:
            if m == self.head:
                head_input = [y[j] for j in m.f] + [projected_text]
                out = m(head_input)
                # out[0]: [Batch, 300, 4 + num_classes + 32]
                if self.export_mask:
                    return out
                else:
                    # Slicing from end because num_classes is dynamic
                    return out[0][..., :-32]

            xi = feat if m.f == -1 else [y[j] for j in m.f] if isinstance(m.f,
                                                                          list) else y[
                m.f]
            feat = m(xi)
            y.append(feat)


class PromptFreeWrapper(torch.nn.Module):
    """Simple wrapper for standard fixed-vocabulary models."""

    def __init__(self, model_pt, export_mask=True):
        super().__init__()
        self.model = model_pt.model
        self.export_mask = export_mask

    def forward(self, x):
        inf_out = self.model(x)[0]
        if self.export_mask:
            return inf_out[0], inf_out[1]
        else:
            # Slicing from start to get Box + Score + Class (Fixed shape)
            return inf_out[0][..., :6]


class YOLOEExporter:
    def __init__(self, output_root: Path):
        self.output_root = output_root
        self.img_size = 640

    def export_promptable(self, scale: str):
        print(f"--- Exporting Promptable: {scale.upper()} ---")
        path_dir = self.output_root / "promptable"
        path_dir.mkdir(parents=True, exist_ok=True)

        model = YOLOE(f"yoloe-26{scale}-seg.pt")
        dummy_img = torch.randn(1, 3, self.img_size, self.img_size)
        dummy_txt = torch.randn(1, 5, 512)  # 5 classes for tracing

        for is_seg in [True, False]:
            suffix = "seg" if is_seg else "det"
            out_path = path_dir / f"yoloe-26{scale}-{suffix}-promptable.onnx"

            torch.onnx.export(
                PromptableWrapper(model, export_mask=is_seg).eval(),
                (dummy_img, dummy_txt),
                str(out_path),
                opset_version=18,
                input_names=['images', 'text_embeddings'],
                output_names=['output0', 'protos'] if is_seg else ['output0'],
                dynamic_axes={
                    'images': {0: 'batch', 2: 'height', 3: 'width'},
                    'text_embeddings': {0: 'batch', 1: 'num_classes'},
                    'output0': {0: 'batch', 1: 'anchors'},
                    'protos': {0: 'batch'}
                } if is_seg else {
                    'images': {0: 'batch', 2: 'height', 3: 'width'},
                    'text_embeddings': {0: 'batch', 1: 'num_classes'},
                    'output0': {0: 'batch', 1: 'anchors'}
                }
            )

    def export_prompt_free(self, scale: str):
        print(f"--- Exporting Prompt-Free: {scale.upper()} ---")
        path_dir = self.output_root / "prompt_free"
        path_dir.mkdir(parents=True, exist_ok=True)

        model = YOLO(f"yoloe-26{scale}-seg-pf.pt")
        dummy_img = torch.randn(1, 3, self.img_size, self.img_size)

        if scale == "n":
            vocab = [model.names[i] for i in range(len(model.names))]
            with open(path_dir / "vocabulary_4585.json", "w") as f:
                json.dump(vocab, f, indent=2)

        for is_seg in [True, False]:
            suffix = "seg" if is_seg else "det"
            out_path = path_dir / f"yoloe-26{scale}-{suffix}-pf.onnx"

            torch.onnx.export(
                PromptFreeWrapper(model, export_mask=is_seg).eval(),
                dummy_img,
                str(out_path),
                opset_version=18,
                do_constant_folding=True,
                input_names=['images'],
                output_names=['detections', 'protos'] if is_seg else ['detections'],
                dynamic_axes={
                    'images': {0: 'batch', 2: 'height', 3: 'width'},
                    'detections': {0: 'batch', 1: 'anchors'},
                    'protos': {0: 'batch', 2: 'p_height', 3: 'p_width'}
                } if is_seg else {
                    'images': {0: 'batch', 2: 'height', 3: 'width'},
                    'detections': {0: 'batch', 1: 'anchors'}
                }
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", choices=["n", "s", "m", "l", "x", "all"],
                        default="all")
    parser.add_argument("--type", choices=["promptable", "free", "both"],
                        default="both")
    args = parser.parse_args()

    exporter = YOLOEExporter(Path("assets/model"))
    scales = ["n", "s", "m", "l", "x"] if args.scale == "all" else [args.scale]

    for s in scales:
        if args.type in ["promptable", "both"]:
            exporter.export_promptable(s)
        if args.type in ["free", "both"]:
            exporter.export_prompt_free(s)
