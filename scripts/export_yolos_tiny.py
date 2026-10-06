"""Export the pinned Apache-2.0 YOLOS-Tiny weights to the browser locator ONNX."""

import argparse
import hashlib
from pathlib import Path

import torch
from transformers import AutoModelForObjectDetection

WEIGHTS_SHA256 = "5a6a017a20cb522dd347271fa5bd670467e456176aaccd940090e50985ac6e74"


class Detector(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        result = self.model(pixel_values=pixel_values)
        return result.logits, result.pred_boxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="Directory containing the pinned Hugging Face weights and config")
    parser.add_argument("output", type=Path, help="ONNX output path")
    args = parser.parse_args()
    actual = hashlib.sha256((args.source / "model.safetensors").read_bytes()).hexdigest()
    if actual != WEIGHTS_SHA256:
        raise ValueError("YOLOS-Tiny source weights SHA-256 mismatch")
    model = AutoModelForObjectDetection.from_pretrained(args.source).eval()
    torch.manual_seed(0)
    torch.onnx.export(
        Detector(model), torch.randn(1, 3, 416, 416), str(args.output),
        input_names=["pixel_values"], output_names=["logits", "pred_boxes"], opset_version=17,
    )
    print(hashlib.sha256(args.output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
