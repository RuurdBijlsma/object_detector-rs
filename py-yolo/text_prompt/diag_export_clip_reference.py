import torch
import numpy as np
import clip
from pathlib import Path


def export_reference():
    # 1. Load the same MobileCLIP model YOLOE uses
    model_path = "mobileclip2_b.ts"
    if not Path(model_path).exists():
        print(f"Error: {model_path} not found. Please ensure it's in the current directory.")
        return

    print(f"Loading {model_path}...")
    model = torch.jit.load(model_path, map_location="cpu").eval()

    # 2. Define test labels
    labels = ["cat", "car", "van", "sign", "person", "lamp", "watermelon"]
    print(f"Generating embeddings for: {labels}")

    # 3. Tokenize and Encode
    tokens = clip.tokenize(labels)
    with torch.no_grad():
        # MobileCLIP returns a tuple or a single tensor depending on the export
        output = model(tokens)
        embeddings = output[0] if isinstance(output, (list, tuple)) else output
        # Ensure it is float32 and [N, 512]
        embeddings = embeddings.float().cpu().numpy()

    # 4. Save to Disk
    output_file = "clip_reference.npy"
    np.save(output_file, embeddings)

    print(f"Successfully saved reference to {output_file}")
    print(f"Shape: {embeddings.shape}")
    # Print a tiny slice for manual visual check
    print(f"First 5 values of first embedding: {embeddings[0, :5]}")


if __name__ == "__main__":
    export_reference()
