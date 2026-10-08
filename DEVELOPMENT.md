# Development notes

This document is for working on weBG. The [README](README.md) describes the desktop beta as a library consumer sees it. The [beta decision record](BETA_RELEASE.md) contains the frozen quality rule and results; [performance measurements](PERFORMANCE.md) are separate. None of these development records or test files is included in the npm tarball.

## Local setup

Use a Node.js version that supports `--experimental-strip-types`. The ONNX model files are distributed separately from Git and the npm tarball. Put the 512 segmentation and locator files in `models/` before running the browser page or model verification.

```sh
npm ci
npm test
npm run build
npm run verify:model
npm run test:batch
python3 -m http.server 8765 --bind 127.0.0.1
```

Open `http://127.0.0.1:8765/test/subject.html` for manual experiments. The page supports preview, 512 crop refinement, and PNG export; these controls are development tools, not part of the first product's one-click path. `test/browser.html` checks the worker and WASM runtime, image validation, geometry, transparency, resizing, one-click equivalence, and refinement with a tiny deterministic ONNX model. Serve these pages over HTTP because workers and WASM do not work from `file://`.

## Batch review

`npm run batch` reads the local ignored `img-tests/photosets/photoset-1/` directory and writes `<original-name>-CUTOUT.png` files to `img-tests/outputs/outputs-1/`. It reuses one Chrome model session and skips existing output files. To choose another input, output, or manifest, run:

```sh
npm run batch -- /path/to/photos /path/to/outputs models/birefnet-lite-512.json
```

`npm run batch:768` is an exploratory comparison, not a supported beta path. The script rebuilds the library and verifies the selected model hash before processing. `npm run test:batch` uses a temporary synthetic image and leaves no output in the photo folders. All photo sets, cutouts, screenshots, and local release binaries remain Git-ignored.

## Model reproduction

The 512 BiRefNet_lite and YOLOS-Tiny manifests pin upstream revisions, source weight SHA-256, export tools, ONNX SHA-256, and tensor shapes. Export tools are development dependencies only. The BiRefNet export uses official weights and checks its patched PyTorch network and ONNX output against the original network. Review the pinned upstream custom Python code before running it.

```sh
python3.11 -m venv .venv
.venv/bin/pip install -r scripts/requirements-export.txt
.venv/bin/python scripts/export_birefnet_lite.py
npm run verify:model
```

For the locator, obtain `model.safetensors`, `config.json`, and `preprocessor_config.json` from the revision recorded in `models/yolos-tiny-416.json`, then run `python scripts/export_yolos_tiny.py /path/to/source models/yolos-tiny-416.onnx`. The script checks the source weight hash. Both 512 ONNX files reproduced the manifest hashes in the current export environment; a different environment may produce different bytes and needs browser validation.

The 768 and 1024 manifests and exports remain for research. The 768 graph is slower and previously exceeded the iPhone 16 Safari WebContent limit. The 1024 graph passed export validation but failed desktop Chrome WASM inference with `std::bad_alloc`. Neither is part of the beta bundle.

## Release check

Run `npm run release:prepare` to build the package and create `release/v<version>/webg-<version>-browser.tar.gz` plus the npm tarball and checksums. The browser archive contains only packaged library files and the two hash-verified model assets. Run `npm run release:check` to unpack that browser archive into a temporary directory, verify both ONNX hashes, and complete one Chrome background removal using the image included in the archive. The check serves no files from the source tree or the Git-ignored local model directory. Publish both tarballs and `SHA256SUMS` as versioned release assets only after this check passes.

## History

The earlier 50-photo, 75-photo, holdout, iPhone, and model-pilot reviews are summarized in `BETA_RELEASE.md` and `PERFORMANCE.md`; their per-image records live under ignored `img-tests/outputs/`. Those viewed sets are regression material, not unseen qualification sets for a changed candidate.
