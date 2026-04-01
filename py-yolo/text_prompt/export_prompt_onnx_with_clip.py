# /// script
# requires-python = "==3.12.*"
# dependencies = [
#    "torch==2.11.0",
#    "ultralytics>=8.4.31",
#    "onnx==1.21.0",
#    "numpy",
#    "onnxscript==0.6.2",
#    "clip @ git+https://github.com/ultralytics/CLIP.git",
# ]
# ///


import torch
from pathlib import Path
from ultralytics import YOLOE


class YOLOE_Complete_Wrapper(torch.nn.Module):
    def __init__(self, yoloe_model):
        super().__init__()
        self.model = yoloe_model.model
        # Force One-to-Many head
        self.model.model[-1].end2end = False

    def forward(self, x, raw_clip_embeddings):
        """
        x: Images [1, 3, 640, 640]
        raw_clip_embeddings: Raw output from mobileclip2_b.ts [1, N, 512]
        """
        # 1. APPLY PROJECTION HEAD INSIDE ONNX
        # This is the 'get_tpe' logic from tasks.py
        head = self.model.model[-1]
        projected_text = head.get_tpe(raw_clip_embeddings)

        # 2. STANDARD FORWARD
        y = []
        for m in self.model.model:
            if m.f != -1:
                feat = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            else:
                feat = x

            if m == head:
                # Pass the already projected text to the head
                head_input = feat + [projected_text]
                return m(head_input)

            x = m(feat)
            y.append(x if m.i in self.model.save else None)


def export_yoloe_final():
    model_scale = "x"
    pt_file = f"yoloe-26{model_scale}-seg.pt"
    onnx_path = f"yoloe-26{model_scale}-pure-clip.onnx"

    print(f"--- Loading {pt_file} ---")
    yolo = YOLOE(pt_file)

    # Wrap with the internal projection head included
    wrapper = YOLOE_Complete_Wrapper(yolo).eval()

    dummy_img = torch.randn(1, 3, 640, 640)
    dummy_txt = torch.randn(1, 5, 512)

    print(f"--- Exporting to: {onnx_path} ---")
    torch.onnx.export(
        wrapper,
        (dummy_img, dummy_txt),
        onnx_path,
        opset_version=18,
        input_names=['images', 'text_embeddings'],
        output_names=['output0', 'protos'],
        dynamic_axes={
            'images': {0: 'batch', 2: 'height', 3: 'width'},
            'text_embeddings': {0: 'batch', 1: 'num_classes'},
            'output0': {0: 'batch', 1: 'anchors'},
            'protos': {0: 'batch'}
        }
    )
    print("SUCCESS: ONNX now accepts RAW CLIP embeddings.")


if __name__ == "__main__":
    export_yoloe_final()