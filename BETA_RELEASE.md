# First desktop beta decision

This release is a **one-click, automatic** background remover for a browser image editor on laptops and desktops. The editor passes a JPEG, PNG, or WebP `Blob` to `segmenter.removeBackground()` and receives a transparent PNG `Blob` or an explicit error. The initial product does not ask users to select, refine, or rescue a subject. iPhone and other mobile devices are outside this first release.

## Promise and limits

The intended photo has one visually dominant foreground subject: a person, animal, road or air vehicle, boat, flower, or discrete physical object. The subject should be recognizable when the *source image* is displayed at about 512 pixels on its long side. A physically held item or rider and vehicle can be part of that subject. A separately positioned ball, skateboard, second animal or vehicle, dense transparent structure, or entire landscape is not part of the automatic promise. The library may still return a result for such images; quality is unqualified.

The caller supplies a file within 25 MiB and 40 million source pixels. The one-click method scales images above the default 16-million-pixel export cap, preserving aspect ratio. Callers can request a smaller PNG with `maxOutputPixels` or configure a higher desktop export limit after measuring memory. An explicit no-subject or input-limit error is preferable to a blank PNG.

## Locked qualification method

These rules are fixed **before** the next unseen set is run. Select 90 new Wikimedia Commons photos from source thumbnails alone: 15 people, 15 animals, 15 vehicles, 15 boats, 15 flowers, and 15 objects. Record the intended subject, source, stated license, and local file hash before inference. Exclude prior photos and near duplicates. Apply the promise above to source previews only; reject images with no clearly dominant subject, an equally prominent rival, or a source that is already a transparent cutout. Do not remove an accepted image from the denominator after seeing its output.

Score the first automatic 512 result at 512-pixel display size, with no crop refinement, manual prompts, or per-photo retries. A result is usable when the intended subject is substantially intact and recognizable, including essential attached parts, without a large opaque patch of unrelated background or another equally prominent subject. Minor edge softness is acceptable. A blank mask, explicit no-subject error, wrong subject, essential missing part, or material background fragment is a miss. Publish each image's result and reason for a miss in a text-only qualification record.

The product quality gate is **at least 81/90 usable results overall and at least 12/15 in each group**. Do not change it after the run or generate another qualification set to replace an unfavorable result. The 512 candidate must also complete all 90 attempts without a browser crash, and must never silently produce a blank PNG. Each successful PNG must decode with the expected aspect ratio and nonzero alpha.

## Desktop acceptance and release

After the quality gate passes, test a cold model load and 20 sequential mixed-size photos in the target desktop Chrome and Safari versions, using the same one-click method. Include several 8 MP exports and a near-limit input. Record completion, export dimensions, errors, and browser reloads. No crash, reload, or invalid PNG passes. Then freeze the library build, runtime files, both 512 model assets and manifests, hashes, hosting paths, package version, and notices. Install its tarball in an isolated consumer page and check a cold one-click call. Only then tag the beta. No editor integration is part of this release work.

If the frozen quality gate fails, keep the set as viewed development evidence. Do not tag a broad automatic beta. The next choice is a documented change in the model/pipeline with another development regression run, or a separately agreed narrower product promise. The current viewed sets cannot qualify a later fix.

## Model decision before the frozen run

Keep the self-exported, hash-pinned BiRefNet_lite 512 plus the sparse-mask car/boat locator as the candidate. Development review used the 23 documented misses in holdout 4 and previously passing controls. Moving the same model to 768 recovered some sparse subjects and parts, but lost much of a cyclist's bicycle in a previously usable photo and did not resolve multi-subject selection. The official general-use BiRefNet tiny checkpoint generated new empty masks on passing boats and vehicles. IS-Net general-use improved a few misses but left broad low-alpha background haze and regressed several controls. BEN2 base produced some better native cutouts, but its 1024 ONNX graph failed in desktop Chrome WASM with `std::bad_alloc`. A full BiRefNet 512 fp16 ONNX graph ran in desktop Chrome, but the local asset was about 473 MB and one zero-input WASM inference took about 19 seconds; it still kept multiple comparable subjects in several misses. That community export also lacks our own upstream-to-ONNX provenance chain. None offered a justified replacement for the first one-click candidate.

The earlier MobileSAM experiment required a person to draw a box, which the first editor will not ask users to do. An automatic detector can propose boxes for some photos, but when two people, boats, or objects are similarly prominent it cannot reliably infer which the user meant. Such scenes remain outside the stated single-dominant-subject promise. All model comparisons above were on viewed development images; they are not qualification scores.

## Frozen desktop result (October 7th 2026)

The 90 Commons sources were frozen with intended subjects, source pages, stated licenses, and local hashes in `img-tests/photosets/photoset-beta-desktop-5/intent.md` before inference. The unchanged 512 candidate produced **80/90 usable automatic cutouts**: people 11/15, animals 14/15, vehicles 14/15, boats 13/15, flowers 14/15, and objects 14/15. The 81/90 overall and 12/15 people gates both failed. Chrome completed all 90 attempts without a reload; 89 yielded correctly sized, nonblank RGBA PNGs, and one returned an explicit no-subject error. The per-image results and reasons are in `img-tests/outputs/outputs-beta-desktop-5-512/results.md` (Git-ignored). This set became viewed development evidence, so that candidate was not tagged.

A separate Chrome one-click repeated-use run on 20 already viewed large photos (about 9.6–39.5 MP) reused one session and the default 16 MP PNG cap. It produced 19 valid PNGs and one explicit no-subject error for a cyclist, without a tab reload. That is useful reliability evidence, but it does not change the quality score or replace the planned Chrome and Safari release acceptance after qualification.

After the frozen score, a browser regression exposed a YOLOS class-index mistake in a provisional person-locator experiment: the upstream model labels `person` as class 1, not class 0. The corrected detector now proposes and applies crops on two viewed person failures. A Chrome rerun changed exactly those two alpha masks; 87 other PNGs were pixel identical and the explicit no-subject case remained. The recovered walking woman is usable. Replacing the original sparse mask for a confident person crop also removed the residual railing from the silhouetted skater; the skater is recognizable, though the tiny skateboard edge remains uncertain. The previously passing Boat 8 rescue stayed byte identical. This was a **new development candidate**, not part of the frozen 80/90 result, and required a new unseen qualification set under the same rules.

## Qualified candidate (October 7th 2026)

For that corrected candidate, a second 90-photo Wikimedia Commons set was chosen from source previews and frozen **before** inference. Its intended subjects, source pages, stated licenses, image hashes, and exact library and model asset hashes were recorded in `img-tests/photosets/photoset-beta-desktop-6/intent.md` before the run. The first Chrome run scored **81/90 usable**, exactly meeting the unchanged overall gate; the groups scored people 15/15, animals 13/15, vehicles 12/15, boats 12/15, flowers 15/15, and objects 14/15. All 90 attempts completed in one tab: 89 nonblank, correctly sized PNGs and one explicit no-subject error, with no reload or silently blank export. The [tracked qualification record](QUALIFICATION.md) lists all 90 sources, intended subjects, scores, observations, and source/output hashes without storing the images. The original photos and PNGs remain local and Git-ignored; the text record does not by itself permit a visual review of the output edges.

This is an exact pass, with no margin in the vehicle and boat groups. Five borderline passes are documented in the per-image record, as are nine misses including missing bicycle/boat parts, retained dock posts, an incorrect chair selection, and one parked car with no subject found. The convenience sample is evidence for the stated single-dominant-subject promise, not a measured success rate for arbitrary uploads. It is now viewed and cannot qualify a future pipeline change as unseen.

## Desktop acceptance and beta package

The same one-click build completed 20 already viewed, mixed-size photos (about 9.6–39.5 MP) sequentially in desktop Chrome, Playwright WebKit 26.6, and actual Safari 26.6.2. Each run produced **19 valid PNGs and one explicit no-subject error** for `cyclist.jpg`. PNGs decoded with nonzero transparency, preserved aspect ratio, and respected the default 16 MP output cap. The large skeleton photo exported at 3620 × 4419. No browser crashed or reloaded; the actual Safari run had exactly one page load. These are reliability checks on viewed photos, not additional quality scores.

The beta package is `webg@0.1.0-beta.2`. It remains `UNLICENSED`, and its npm manifest is private to prevent registry publication; the versioned GitHub release assets are publicly downloadable. The npm tarball contains the library build, WASM runtime, both model manifests, the README image, and third-party licenses. The browser bundle adds both exact tested ONNX files under `/models/` and is deployable as a static-site asset tree. Serve the bundle on one origin and deploy it as one versioned set; an old cached worker or model manifest can produce a different cutout. The tested asset SHA-256 values are:

| Asset | SHA-256 |
| --- | --- |
| `dist/index.js` | `5e06d083cc2d154d185d03ec476d20197b577b70e824e6bf69905fcaed358f53` |
| `dist/worker.js` | `1c890b4ec284ef4161bf32e59efc481ce2bb403f579649ddd4fad676a276c0c1` |
| `dist/ort-wasm-simd-threaded.mjs` | `e13f7f94fc51b4ca72b12faeb1ee95f4ace6dfbc8939bc718aabdc0a27c4299b` |
| `dist/ort-wasm-simd-threaded.wasm` | `3398c10d07d229bd91b364548e130e0e51a8e5704b88c7c083ebbeb78842dee2` |
| `models/birefnet-lite-512.onnx` | `eba7f32d81b4ea697334d467f44d373094633f3dd02eeb000e5f592510f79164` |
| `models/yolos-tiny-416.onnx` | `b12c56df09c905ae7ace9944b7981a88a20e2b9a006f1b052861b05c6e4362c1` |

The PNG and model limitations above remain part of the beta contract. Integration into the browser editor is separate work and has not been started here.
