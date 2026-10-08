# Performance measurements

These are local measurements from development runs, not download benchmarks or a performance guarantee. Hardware, browser version, image mix, and model settings affect the result. See [BETA_RELEASE.md](BETA_RELEASE.md) for the visual quality decision and browser acceptance record.

## Earlier 50-photo Mac comparison

Separate headless Chrome runs used the same 50 photos. Warm full-PNG median processing times were **4.93 s** for weBG 512, **10.60 s** for weBG 768, and **8.90 s** for IMG.LY 1.7.0 medium on CPU. All three completed the set without execution failures. This comparison excludes first-visit downloads and uses different models; it is not a claim of equivalent visual quality. IMG.LY recovered a skyline and a soccer ball that weBG missed, and also produced visible defects on other photos. Per-image quality scores for that comparison were not completed.

On one bike image in separate fresh Chrome processes, 512 and 768 session initialization took **2.90 s** and **2.94 s**. Warm segmentation took **5.22 s** and **10.72 s**. A 50-photo 768 run completed, but a sampled Chrome process-tree RSS reached about **6.0 GiB**, potentially double-counting shared pages. This is not a peak-memory measurement.

## Browser acceptance for the desktop beta

The frozen 512 candidate processed 20 previously viewed photos of about 9.6–39.5 MP sequentially in Chrome, Playwright WebKit 26.6, and Safari 26.6.2. Each browser produced 19 valid PNGs and one explicit no-subject error, with no page reload. The default one-click export cap limited output to 16 MP; the large skeleton photo exported at 3620 × 4419. This was a reliability check, not a quality score or cold-download measurement.

The iPhone 16 is outside the desktop beta. Its 512 runs and some 8 MP exports passed, while 768 triggered a WebContent `per-process-limit` termination in one fresh small-sample run. Later successful 512 runs still reported memory pressure. The sampled resident-memory values in those logs are not measured peaks.

## Weight-only uint8 beta.3 candidate (October 8, 2026)

The BiRefNet_lite 512 ONNX file shrank from **183.8 MB** FP32 to **54.1 MB** with per-channel uint8 weights and FP32 activations. Gzip sizes were **164.8 MB** and **41.7 MB**. The YOLOS locator is loaded only for sparse-mask fallback and is unchanged. Actual transfer depends on hosting compression and caching.

In six paired desktop Chrome photos, the median warm one-click time was about **4.70 s FP32** versus **4.77 s uint8 weights** (roughly 1.5% slower). The 90-photo uint8 qualification run had a similar warm processing pattern. These are local measurements, not a speed guarantee.

For two photos (one regular, one invoking a second inference), three fresh Chrome profiles per model were sampled every 200 ms with `ps`. The sum of browser-process RSS peaked around **2.57 GiB FP32** versus **1.69–1.79 GiB uint8 weights**, about 30–34% lower. This sum counts shared mappings in multiple processes and is not a unique physical-memory peak; the repeated paired difference is the useful signal. No PNG export was performed in this memory test.

The new frozen 90-photo quality run scored **85/90 usable**, with 89 valid PNGs and one explicit no-subject error. The same photos under FP32 produced the same explicit error; the 89 binary masks had median overlap of **99.92%**. The full uint8 activation-and-weight graph was rejected: it was slower and introduced material quality regressions on the previously viewed set. See [QUALIFICATION_BETA3.md](QUALIFICATION_BETA3.md) for the per-photo record.

On one viewed boat photo, WebKit and actual Safari invoked a crop/locator fallback and kept the full boat, while Chrome kept only fragments under both FP32 and uint8. The uint8 Chrome fragment was smaller. This is an existing model/pipeline limitation and the 20-photo execution check did not count visual usability.
