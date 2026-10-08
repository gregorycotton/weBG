"""Quantize the pinned 512 FP32 export to the tested uint8-weight browser graph."""

import argparse
import hashlib
import json
import shutil
import tempfile
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import onnx
from PIL import Image, ImageOps
from onnx import helper
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static

CALIBRATION = Path(__file__).with_name('quantization-calibration.json')
BASE_SHA256 = 'eba7f32d81b4ea697334d467f44d373094633f3dd02eeb000e5f592510f79164'
OUTPUT_SHA256 = '72446f88aacb13f0f3dfd17d653cc34c427adc465fbd76097742e87b64f3fdd7'


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


class Reader(CalibrationDataReader):
    def __init__(self, entries, directory):
        self.entries = iter(entries)
        self.directory = directory

    def get_next(self):
        item = next(self.entries, None)
        if item is None:
            return None
        if self.directory:
            raw = (self.directory / item['name']).read_bytes()
        else:
            request = Request(item['url'], headers={'User-Agent': 'weBG model export/0.1 (https://github.com/gregorycotton/weBG)'})
            for attempt in range(3):
                try:
                    with urlopen(request, timeout=60) as response:
                        raw = response.read()
                    break
                except HTTPError as error:
                    if error.code not in (429, 503) or attempt == 2:
                        raise
        if sha256(raw) != item['sha256']:
            raise ValueError(f"Calibration thumbnail changed: {item['name']}")
        with Image.open(BytesIO(raw)) as source:
            image = ImageOps.exif_transpose(source).convert('RGB').resize((512, 512), Image.Resampling.BILINEAR)
        pixels = np.asarray(image, dtype=np.float32) / 255
        pixels = (pixels - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        print('Calibrating', item['name'], flush=True)
        return {'input_image': pixels.transpose(2, 0, 1)[None].copy()}


def keep_quantized_weights_only(model):
    producers = {name: node for node in model.graph.node for name in node.output}
    consumers = defaultdict(list)
    for node in model.graph.node:
        for name in node.input:
            consumers[name].append(node)
    aliases = {}
    removed = set()
    for node in model.graph.node:
        if node.op_type != 'DequantizeLinear':
            continue
        quantize = producers.get(node.input[0])
        if quantize is None or quantize.op_type != 'QuantizeLinear':
            continue
        if not all(other.op_type == 'DequantizeLinear' for other in consumers[quantize.output[0]]):
            raise ValueError('Activation quantizer has another consumer')
        aliases[node.output[0]] = quantize.input[0]
        removed.update((*node.output, *quantize.output))

    def resolve(name):
        while name in aliases:
            name = aliases[name]
        return name

    remaining = [node for node in model.graph.node if not any(name in removed for name in node.output)]
    for node in remaining:
        node.input[:] = [resolve(name) for name in node.input]
    for output in model.graph.output:
        if output.name in aliases:
            remaining.append(helper.make_node('Identity', [resolve(output.name)], [output.name]))
    del model.graph.node[:]
    model.graph.node.extend(remaining)
    used = {name for node in remaining for name in node.input}
    initializers = [tensor for tensor in model.graph.initializer if tensor.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(initializers)
    if len(aliases) != 1120:
        raise ValueError(f'Unexpected activation quantizer count: {len(aliases)}')
    onnx.checker.check_model(model)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True, help='Pinned FP32 512 ONNX export')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--calibration-dir', type=Path, help='Optional local copy of the 12 hash-pinned Commons thumbnails')
    args = parser.parse_args()
    if not args.source.is_file():
        parser.error(f'Source ONNX does not exist: {args.source}')
    with args.source.open('rb') as source:
        if hashlib.file_digest(source, 'sha256').hexdigest() != BASE_SHA256:
            raise ValueError('Source ONNX does not match the pinned FP32 export')
    entries = json.loads(CALIBRATION.read_text())
    if len(entries) != 12 or len({entry['name'] for entry in entries}) != 12:
        raise ValueError('Expected 12 distinct calibration thumbnails')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='webg-quantize-') as directory:
        temporary = Path(directory)
        source = temporary / 'fp32.onnx'
        qdq = temporary / 'qdq.onnx'
        shutil.copyfile(args.source, source)
        quantize_static(str(source), str(qdq), Reader(entries, args.calibration_dir),
                        quant_format=QuantFormat.QDQ, op_types_to_quantize=['Conv', 'MatMul'],
                        per_channel=True, activation_type=QuantType.QUInt8,
                        weight_type=QuantType.QUInt8, calibrate_method=CalibrationMethod.MinMax)
        model = keep_quantized_weights_only(onnx.load(str(qdq)))
        onnx.save(model, args.output)
    with args.output.open('rb') as output:
        digest = hashlib.file_digest(output, 'sha256').hexdigest()
    if digest != OUTPUT_SHA256:
        raise ValueError(f'Quantized model differs from tested asset: {digest}')
    print('SHA-256', digest, flush=True)


if __name__ == '__main__':
    main()
