"""Export pinned official BiRefNet_lite weights to a browser-oriented ONNX graph.

The deformable convolution decomposition follows the mathematical approach
documented by CoderViking (MIT): https://huggingface.co/CoderViking/birefnet-lite-onnx
No ONNX weights or graph are copied from that repository.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoModelForImageSegmentation

SOURCE_REPOSITORY = "ZhengPeng7/BiRefNet_lite"
SOURCE_REVISION = "7838f1c3472f827cd8ce13ab5ccc2ce48077360f"
SOURCE_SHA256 = "4417d89795250e698c3cb0ae8df15743810065f646f48a694fdfa7ca052d0815"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deform_as_grid_samples(layer, image):
    """Evaluate a deformable convolution as one GridSample/1x1 Conv per tap."""
    offsets = layer.offset_conv(image)
    modulation = 2 * torch.sigmoid(layer.modulator_conv(image))
    height, width = image.shape[-2:]
    assert layer.stride == (1, 1) and offsets.shape[-2:] == (height, width)
    kernel_height, kernel_width = layer.regular_conv.kernel_size
    ys = torch.arange(height, device=image.device, dtype=image.dtype)
    xs = torch.arange(width, device=image.device, dtype=image.dtype)
    base_y, base_x = torch.meshgrid(ys, xs, indexing="ij")
    base_y = (2 * base_y + 1) / height - 1
    base_x = (2 * base_x + 1) / width - 1
    result = None
    for row in range(kernel_height):
        for column in range(kernel_width):
            tap = row * kernel_width + column
            sample_y = base_y + (offsets[:, 2 * tap] + row - layer.padding) * (2 / height)
            sample_x = base_x + (offsets[:, 2 * tap + 1] + column - layer.padding) * (2 / width)
            grid = torch.stack((sample_x, sample_y), dim=-1)
            sampled = F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            weight = layer.regular_conv.weight[:, :, row, column, None, None]
            part = F.conv2d(sampled * modulation[:, tap:tap + 1], weight)
            result = part if result is None else result + part
    if layer.regular_conv.bias is not None:
        result += layer.regular_conv.bias[None, :, None, None]
    return result


class FinalLogits(torch.nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, input_image):
        return self.network(input_image)[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, help="Pinned upstream snapshot; downloaded if omitted")
    parser.add_argument("--output", type=Path, default=Path("models/birefnet-lite-512.onnx"))
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()
    if args.size < 128 or args.size % 32:
        parser.error("--size must be at least 128 and divisible by 32")

    source_dir = args.source_dir or Path(snapshot_download(
        repo_id=SOURCE_REPOSITORY,
        revision=SOURCE_REVISION,
        allow_patterns=["model.safetensors", "config.json", "BiRefNet_config.py", "birefnet.py"],
    ))
    actual_source_hash = sha256(source_dir / "model.safetensors")
    if actual_source_hash != SOURCE_SHA256:
        raise ValueError(f"Upstream weights changed: {actual_source_hash}")

    torch.set_grad_enabled(False)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    network = AutoModelForImageSegmentation.from_pretrained(str(source_dir), trust_remote_code=True).eval().float()
    module = sys.modules[type(network).__module__]
    deform_class = module.DeformableConv2d

    # Check the replacement with non-zero offsets before tracing the real model.
    torch.manual_seed(7)
    sample_layer = deform_class(4, 5, kernel_size=3, padding=1)
    for parameter in sample_layer.parameters():
        torch.nn.init.normal_(parameter, std=0.1)
    sample_image = torch.randn(1, 4, 12, 14)
    patch_error = (sample_layer(sample_image) - deform_as_grid_samples(sample_layer, sample_image)).abs().max().item()
    print(f"Deformable convolution max error: {patch_error:.6g}", flush=True)
    if patch_error > 1e-4:
        raise RuntimeError("GridSample decomposition does not match upstream")

    torch.manual_seed(11)
    probe = torch.rand(1, 3, args.size, args.size)
    reference = FinalLogits(network)(probe).detach().numpy()
    deform_class.forward = deform_as_grid_samples
    patched = FinalLogits(network)(probe).detach().numpy()
    model_error = float(np.max(np.abs(reference - patched)))
    print(f"Patched model max error: {model_error:.6g}", flush=True)
    if model_error > 1e-3:
        raise RuntimeError("Patched model differs from upstream")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        FinalLogits(network), (probe,), str(args.output),
        input_names=["input_image"], output_names=["logits"],
        opset_version=17, do_constant_folding=True, dynamo=False,
    )
    graph = onnx.load(str(args.output))
    onnx.checker.check_model(graph)
    if any(node.op_type == "DeformConv" for node in graph.graph.node):
        raise RuntimeError("Export contains a nonstandard DeformConv operator")
    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    actual = session.run(["logits"], {"input_image": probe.numpy()})[0]
    onnx_error = float(np.max(np.abs(reference - actual)))
    print(f"ONNX vs upstream max error: {onnx_error:.6g}", flush=True)
    if onnx_error > 1e-2:
        raise RuntimeError("ONNX output differs from upstream")

    versions = {name: importlib.metadata.version(name) for name in (
        "torch", "torchvision", "transformers", "timm", "kornia", "einops", "onnx", "onnxruntime"
    )}
    manifest = {
        "modelId": f"birefnet-lite-{args.size}-fp32",
        "sourceRepository": f"https://huggingface.co/{SOURCE_REPOSITORY}",
        "sourceRevision": SOURCE_REVISION,
        "sourceWeightsSHA256": actual_source_hash,
        "sourceLicense": "MIT",
        "exportMethod": "PyTorch ONNX opset 17, fixed shape, GridSample deformable convolution",
        "exportToolVersions": versions,
        "exportedSHA256": sha256(args.output),
        "inputShape": [1, 3, args.size, args.size],
        "outputShape": list(actual.shape),
        "precision": "fp32",
        "runtime": {
            "id": f"birefnet-lite-{args.size}-fp32",
            "url": f"/models/{args.output.name}",
            "inputName": "input_image",
            "outputName": "logits",
            "inputWidth": args.size,
            "inputHeight": args.size,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "output": "logits",
        },
        "validation": {"deformMaxError": patch_error, "patchedMaxError": model_error, "onnxMaxError": onnx_error},
    }
    if args.size == 512:
        manifest["runtime"]["locatorUrl"] = "/models/yolos-tiny-416.onnx"
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Exported {args.output} ({args.output.stat().st_size:,} bytes)", flush=True)


if __name__ == "__main__":
    main()
