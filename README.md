# weBG

Standalone, browser-side image segmentation library. It returns a soft, one-byte-per-pixel subject mask and the mapping back to the source image. The browser editor is not part of this package.

The reference backend is single-threaded ONNX Runtime Web WASM. The runtime and model are self-hosted, with no IMG.LY dependency. Image bytes are processed in a worker; the library does not upload them.

## Current status

- Working browser path: inspect JPEG, PNG, or WebP dimensions before decode; resize to the model's fixed input; run WASM inference; return a soft alpha mask and subject bounds in source coordinates. An optional PNG export applies the mask to the original image.
- Initial model candidate: our 512 × 512 FP32 ONNX export of the official BiRefNet_lite weights. The model is **183,755,169 bytes**; the mask is **262,144 bytes** before any storage compression.
- `models/birefnet-lite-512.json` records the source revision, upstream and exported SHA-256 hashes, license, export method, tool versions, tensor shapes, and validation errors. The `.onnx` file is deliberately excluded from Git and from the npm package; host it as a separate static asset.
- This is a functional baseline, not yet a qualified replacement for every IMG.LY image. Hair, translucent edges, varied photos, browsers, and memory pressure still need comparative testing.

## Build and try it

Requires Node.js with `--experimental-strip-types` support and Python 3 for the local static server.

```sh
npm install
npm test
npm run build
npm run verify:model
python3 -m http.server 8765 --bind 127.0.0.1
```

Open `http://127.0.0.1:8765/test/subject.html` and choose an image, or run the built-in synthetic sample. `test/browser.html` checks the worker and WASM runtime using a tiny deterministic ONNX model. Serve over HTTP; `file://` will not work for the worker and WASM asset requests.

If the binary is missing, generate it as described below. `npm run verify:model` compares the model's SHA-256 with the manifest before deployment. Publish `dist/index.js`, `dist/worker.js`, both `dist/ort-wasm-simd-threaded.*` files, the model, and its manifest on the same origin. Keep the worker and WASM files adjacent to `index.js`. The model URL in the manifest is `/models/birefnet-lite-512.onnx`; adjust it for the host path if necessary.

## API

```js
import { createSegmenter } from "/dist/index.js";

const manifest = await fetch("/models/birefnet-lite-512.json").then(r => r.json());
const segmenter = createSegmenter({ model: manifest.runtime });
try {
  const mask = await segmenter.segment(imageBlob, stage => console.log(stage));
  // mask.data: row-major Uint8Array of soft alpha, mask.width × mask.height.
  // mask.sourceWidth/sourceHeight: original decoded image dimensions.
  // mask.subjectBounds: source-coordinate rectangle; empty means no subject.
  const pngBlob = await segmenter.exportCutout(imageBlob, mask);
} finally {
  segmenter.dispose();
}
```

Reuse one segmenter for multiple images to reuse its loaded session. Only one operation may run on a segmenter at a time; a concurrent call rejects. `initialize()` can preload the model. `exportCutout()` can also use a previously saved mask without loading the model. `dispose()` terminates its worker and rejects in-flight work. The default input limits are 25 MiB compressed and 40 million decoded pixels; pass `maxInputBytes` or `maxInputPixels` to tighten them. JPEG, PNG, and WebP dimensions are checked from the file header before full decode. Invalid, truncated, oversized, and mismatched mask inputs raise an error. Model files must be same-origin.

Keep the source image and `mask.data` separately. To preview, sample the mask at the canvas's *display* resolution and use it as alpha for the source image. `exportCutout()` allocates a full-source-resolution canvas only when called; it scales the native mask to the source coordinates and multiplies existing image transparency. A 6000 × 4000 source still produces a 512 × 512 mask with an explicit full-source coordinate mapping. The small mask cannot recover hair detail absent from 512-pixel inference. Large final PNG exports may require substantially more memory than normal segmentation.

## Reproduce the candidate model

The export uses the official [BiRefNet_lite weights](https://huggingface.co/ZhengPeng7/BiRefNet_lite) at the revision in the manifest, checks their hash, converts the deformable convolution to standard ONNX operators, and checks both the patched PyTorch network and ONNX output against the original network. The model card labels the weights MIT. The export script uses upstream model code only as an export-time input. Read and review that pinned source before running it because Transformers loads its custom Python code.

```sh
python3.11 -m venv .venv
.venv/bin/pip install -r scripts/requirements-export.txt
.venv/bin/python scripts/export_birefnet_lite.py
npm run verify:model
```

Export tools are not browser dependencies. Direct tool versions and the source revision are pinned; the script validates numeric equivalence and writes the final hash. A second export in the same environment produced the identical SHA-256. Re-exporting under a different Python/platform environment may change ONNX bytes, so compare the generated manifest and browser behavior before distributing a replacement binary.

## Qualification still needed

The included graph is fixed at 512 × 512. Changing inference resolution requires another exported and tested model artifact; resizing a mask after inference cannot restore edge detail. A 512-pixel synthetic browser check passed. With no concurrent export workload on the development machine, the observed cold and warm runs took about 7.6 and 4.7 seconds; an overlapping PyTorch export slowed one run to 38 seconds. Measure cold and warm inference time, peak memory, mask quality, and failure rate on representative images and target devices. Decide whether this model is acceptable before calling it the replacement. Benchmark other permissively licensed models if it is not. WebGPU and threaded WASM are future optimizations; the latter requires cross-origin isolation.

See [third-party notices](THIRD_PARTY_NOTICES.md) for the model and runtime licenses.
