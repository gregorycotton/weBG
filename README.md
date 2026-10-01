# weBG

Standalone, browser-side image segmentation library. It returns a soft, one-byte-per-pixel subject mask and the mapping back to the source image. The browser editor is not part of this package.

The reference backend is single-threaded ONNX Runtime Web WASM. The runtime and model are self-hosted, with no IMG.LY dependency. Image bytes are processed in a worker; the library does not upload them.

## Current status

- Working browser path: inspect JPEG, PNG, or WebP dimensions before decode; resize to the model's fixed input; run WASM inference; return a soft alpha mask and subject bounds in source coordinates. An optional PNG export applies the mask to the original image.
- Model candidates: our fixed-shape 512, 768, and 1024 FP32 ONNX exports of the same official BiRefNet_lite weights. The 512 model remains the default and completed sample runs on an iPhone 16. The 768 model completed a 50-photo Chrome batch but repeatedly reloaded the iPhone Safari page; treat it as experimental there. The 1024 model failed during Chrome WASM inference with `std::bad_alloc` and is not a supported browser option.
- Each `models/birefnet-lite-<size>.json` records the source revision, upstream and exported SHA-256 hashes, license, export method, tool versions, tensor shapes, and validation errors. The `.onnx` files are deliberately excluded from Git and the npm package; host a chosen model as a separate static asset.
- This is a functional baseline, not yet a qualified replacement for every IMG.LY image. The 768 review still has scene-selection and disconnected-subject failures. iPhone 16 testing found that 512 segmentation worked on several photos, while full-size export of a 39.5-megapixel photo reloaded Safari.

## Build and try it

Requires Node.js with `--experimental-strip-types` support and Python 3 for the local static server.

```sh
npm install
npm test
npm run build
npm run verify:model
python3 -m http.server 8765 --bind 127.0.0.1
```

Open `http://127.0.0.1:8765/test/subject.html` and choose an image, or run the built-in synthetic sample. Click **Run self-hosted subject model**, choose full-size or a smaller PNG, then **Generate PNG** and **Download PNG** to inspect the actual export. The full-size option reports a clear error for sources over 16 million pixels. If Safari reloads during a run, the page displays the last reported stage when it reopens, provided browser storage is available. `test/browser.html` checks export geometry, transparency, resizing, malformed inputs, and the worker and WASM runtime using a tiny deterministic ONNX model. Serve over HTTP; `file://` will not work for the worker and WASM asset requests.

## Batch cutouts for review

After adding photos to `photoset/`, run `npm run batch` from the weBG root for the 512 model, or `npm run batch:768` for the 768 model. They write to `outputs/` and `outputs-768/` respectively. Each JPEG, PNG, or WebP produces `<original-name>-CUTOUT.png` (for example, `dog.jpg` becomes `dog-CUTOUT.png`). The batch reuses one model session, skips output files that already exist, and reports failures and timings per image. It does not alter input photos or existing cutouts. The command uses a separate headless Google Chrome instance and requires Chrome to be installed. It builds the library and checks the selected model hash before processing. See [the 50-photo comparison](outputs-768/comparison.md) for current results.

To test the batch command without touching either folder, run `npm run test:batch`. It uses a temporary synthetic image and removes its results afterward. The script also accepts input, output, and optional model-manifest paths: `npm run batch -- /path/to/input /path/to/output models/birefnet-lite-768.json`.

If a binary is missing, generate it as described below. `npm run verify:model` checks 512; `npm run verify:model -- models/birefnet-lite-768.json` checks 768. Publish `dist/index.js`, `dist/worker.js`, both `dist/ort-wasm-simd-threaded.*` files, the chosen model, and its manifest on the same origin. Keep the worker and WASM files adjacent to `index.js`. Model URLs in the manifests use `/models/`; adjust them for the host path if necessary.

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
  // Omit the fourth argument for a full-size PNG when the source is at most 16 MP.
  const pngBlob = await segmenter.exportCutout(imageBlob, mask, undefined, { maxOutputPixels: 8_000_000 });
} finally {
  segmenter.dispose();
}
```

Reuse one segmenter for multiple images to reuse its loaded session. Only one operation may run on a segmenter at a time; a concurrent call rejects. `initialize()` can preload the model. `exportCutout()` can also use a previously saved mask without loading the model. `dispose()` terminates its worker and rejects in-flight work. The default input limits are 25 MiB compressed and 40 million decoded pixels; pass `maxInputBytes` or `maxInputPixels` to tighten them. Full-size PNG export has a separate 16-million-pixel limit, configurable with `maxExportPixels` in `createSegmenter()`. A per-call `maxOutputPixels` scales the output within that limit while preserving the source aspect ratio. JPEG, PNG, and WebP dimensions are checked from the file header before full decode. Invalid, truncated, oversized, and mismatched mask inputs raise an error. Model files must be same-origin.

Keep the source image and `mask.data` separately. To preview, sample the mask at the canvas's *display* resolution and use it as alpha for the source image. `exportCutout()` allocates a canvas at the requested output size, scales the native mask to those coordinates, and multiplies existing image transparency. For a smaller PNG, it requests resized decoding from the browser; the browser may still need temporary memory to decode the original. A 6000 × 4000 source produces a 512 × 512 or 768 × 768 mask, depending on the model, with an explicit full-source coordinate mapping. Upscaling a mask cannot recover edge detail absent from inference. Large final PNG exports may require substantially more memory than normal segmentation.

## iPhone 16 follow-up

For an isolated 768 test, close Safari, reopen the test page, select **768** before running any 512 image, leave preview off, and run one small photo through `segment()` without exporting. If the page reloads, note the previous-run stage shown on the page and the time. On the iPhone, check **Settings → Privacy & Security → Analytics & Improvements → Analytics Data** for a `JetsamEvent` at that time. A matching `com.apple.WebKit.WebContent` entry would support memory termination; the stage marker alone cannot establish the cause. Repeat with 512 on the same photo as a control. For the large skeleton photo, select **Up to 8 MP** before generating a PNG. The scaled export reduces its final canvas and encoder workload, but must still be tested on the phone.

## Reproduce the candidate model

The export uses the official [BiRefNet_lite weights](https://huggingface.co/ZhengPeng7/BiRefNet_lite) at the revision in the manifest, checks their hash, converts the deformable convolution to standard ONNX operators, and checks both the patched PyTorch network and ONNX output against the original network. The model card labels the weights MIT. The export script uses upstream model code only as an export-time input. Read and review that pinned source before running it because Transformers loads its custom Python code.

```sh
python3.11 -m venv .venv
.venv/bin/pip install -r scripts/requirements-export.txt
.venv/bin/python scripts/export_birefnet_lite.py
npm run verify:model
.venv/bin/python scripts/export_birefnet_lite.py --size 768 --output models/birefnet-lite-768.onnx
npm run verify:model -- models/birefnet-lite-768.json
```

Export tools are not browser dependencies. Direct tool versions and the source revision are pinned; the script validates numeric equivalence and writes the final hash. A second export in the same environment produced the identical SHA-256. Re-exporting under a different Python/platform environment may change ONNX bytes, so compare the generated manifest and browser behavior before distributing a replacement binary.

## Qualification still needed

All three ONNX graphs are fixed-shape and hash-pinned. The 1024 graph passes upstream numeric validation, but Chrome single-threaded WASM failed on the cyclist photo with `std::bad_alloc`; do not use it as a browser default. In a separate 50-photo browser batch on this Mac, 768 completed without errors and reduced entirely blank exports from eight to two. It still missed the city and river skylines and the soccer ball; the chain-link fence remained unusable. On the same bike photo in separate fresh Chrome instances, model initialization took 2.90 seconds at 512 and 2.94 seconds at 768; warm segmentation took 5.22 and 10.72 seconds respectively. The 768 batch's sampled summed Chrome process-tree RSS peaked at about 6.0 GiB, including possible double-counting of shared pages. These local timings do not include real network download latency. Measure quality and memory on intended lower-memory devices before changing the default. WebGPU and threaded WASM remain separate qualifications; the latter requires cross-origin isolation.

See [third-party notices](THIRD_PARTY_NOTICES.md) for the model and runtime licenses.
