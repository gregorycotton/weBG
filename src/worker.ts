import * as ort from "onnxruntime-web/wasm";
import { imageDimensions } from "./image";
import { locateWeakSubjects } from "./locator";
import { mergeRefinement, outputToMask, pixelsToTensor, type Mask, type ModelConfig, type Rect, type RefineMode } from "./mask";

type Request =
  | { type: "initialize"; model: ModelConfig }
  | { type: "segment"; image: Blob; maxInputPixels: number }
  | { type: "refine"; image: Blob; mask: Mask; region: Rect; mode: RefineMode; maxInputPixels: number }
  | { type: "exportCutout"; image: Blob; mask: Mask; maxInputPixels: number; maxOutputPixels: number };

let session: ort.InferenceSession | undefined;
let model: ModelConfig | undefined;
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = new URL("./", import.meta.url).href;

async function readDimensions(image: Blob, maxInputPixels: number): Promise<{ width: number; height: number }> {
  self.postMessage({ type: "progress", stage: "reading-image-header" });
  const { width, height } = imageDimensions(new Uint8Array(await image.arrayBuffer()));
  if (width * height > maxInputPixels) throw new Error("Image dimensions exceed the configured pixel limit.");
  return { width, height };
}

async function decodeImage(image: Blob, maxInputPixels: number, expected?: { width: number; height: number }, resize?: { width: number; height: number }): Promise<ImageBitmap> {
  const { width, height } = await readDimensions(image, maxInputPixels);
  // JPEG EXIF rotation can swap header dimensions relative to the decoded bitmap.
  if (expected && !((width === expected.width && height === expected.height) || (width === expected.height && height === expected.width))) throw new Error("Mask source dimensions do not match the image.");
  self.postMessage({ type: "progress", stage: "decoding-image" });
  let bitmap: ImageBitmap;
  try { bitmap = resize ? await createImageBitmap(image, { resizeWidth: resize.width, resizeHeight: resize.height }) : await createImageBitmap(image); }
  catch { throw new Error("Could not decode JPEG, PNG, or WebP image."); }
  if (!bitmap.width || !bitmap.height || bitmap.width * bitmap.height > maxInputPixels || resize && (bitmap.width !== resize.width || bitmap.height !== resize.height)) {
    bitmap.close();
    throw new Error("Decoded image dimensions do not match the requested size or pixel limit.");
  }
  if (expected && !resize && (bitmap.width !== expected.width || bitmap.height !== expected.height)) {
    bitmap.close();
    throw new Error("Mask source dimensions do not match the image.");
  }
  return bitmap;
}

async function infer(bitmap: ImageBitmap, sourceWidth: number, sourceHeight: number): Promise<Mask> {
  let input: ort.Tensor;
  try {
    if (!session || !model) throw new Error("Model is not initialized.");
    const canvas = new OffscreenCanvas(model.inputWidth, model.inputHeight);
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) throw new Error("Canvas 2D is unavailable in the worker.");
    context.fillStyle = "#fff";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
    input = new ort.Tensor("float32", pixelsToTensor(pixels, canvas.width, canvas.height, model.mean, model.std), [1, 3, canvas.height, canvas.width]);
  } finally { bitmap.close(); }
  self.postMessage({ type: "progress", stage: "running-model" });
  try {
    const results = await session.run({ [model.inputName]: input });
    try {
      const output = results[model.outputName];
      if (!output || output.dims.at(-2) !== model.inputHeight || output.dims.at(-1) !== model.inputWidth || !(output.data instanceof Float32Array)) throw new Error("Model output does not match its configuration.");
      return outputToMask(output.data, model.inputWidth, model.inputHeight, sourceWidth, sourceHeight, model.output);
    } finally { for (const result of Object.values(results)) result.dispose(); }
  } finally { input.dispose(); }
}

async function inferRegion(image: Blob, region: Rect): Promise<Mask> {
  if (!model) throw new Error("Model is not initialized.");
  let bitmap: ImageBitmap;
  try { bitmap = await createImageBitmap(image, region.x, region.y, region.width, region.height, { resizeWidth: model.inputWidth, resizeHeight: model.inputHeight }); }
  catch { throw new Error("Could not decode the selected image region."); }
  if (bitmap.width !== model.inputWidth || bitmap.height !== model.inputHeight) {
    bitmap.close();
    throw new Error("Decoded region size does not match the model.");
  }
  return infer(bitmap, region.width, region.height);
}

async function locate(image: Blob, sourceWidth: number, sourceHeight: number): Promise<{ region: Rect; person: boolean }[]> {
  if (!model?.locatorUrl) return [];
  self.postMessage({ type: "progress", stage: "locating-subject" });
  const bitmap = await createImageBitmap(image, { resizeWidth: 416, resizeHeight: 416 });
  let input: ort.Tensor;
  try {
    const canvas = new OffscreenCanvas(416, 416);
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) throw new Error("Canvas 2D is unavailable in the worker.");
    context.drawImage(bitmap, 0, 0, 416, 416);
    input = new ort.Tensor("float32", pixelsToTensor(context.getImageData(0, 0, 416, 416).data, 416, 416, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), [1, 3, 416, 416]);
  } finally { bitmap.close(); }
  let locator: ort.InferenceSession | undefined;
  try {
    locator = await ort.InferenceSession.create(model.locatorUrl, { executionProviders: ["wasm"] });
    const result = await locator.run({ pixel_values: input });
    try {
      const logits = result.logits?.data, boxes = result.pred_boxes?.data;
      if (!(logits instanceof Float32Array) || !(boxes instanceof Float32Array)) throw new Error("Locator output does not match its configuration.");
      return locateWeakSubjects(logits, boxes, sourceWidth, sourceHeight);
    } finally { for (const output of Object.values(result)) output.dispose(); }
  } finally { input.dispose(); await locator?.release(); }
}

async function exportCutout(image: Blob, mask: Mask, maxInputPixels: number, maxOutputPixels: number): Promise<Blob> {
  const scale = Math.min(1, Math.sqrt(maxOutputPixels / (mask.sourceWidth * mask.sourceHeight)));
  let width = Math.max(1, Math.floor(mask.sourceWidth * scale));
  let height = Math.max(1, Math.floor(mask.sourceHeight * scale));
  if (width * height > maxOutputPixels) {
    if (width >= height) width = Math.floor(maxOutputPixels / height);
    else height = Math.floor(maxOutputPixels / width);
  }
  const resize = width === mask.sourceWidth && height === mask.sourceHeight ? undefined : { width, height };
  const bitmap = await decodeImage(image, maxInputPixels, { width: mask.sourceWidth, height: mask.sourceHeight }, resize);
  try {
    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    const context = canvas.getContext("2d");
    if (!context) throw new Error("Canvas 2D is unavailable in the worker.");
    context.drawImage(bitmap, 0, 0);
    const maskCanvas = new OffscreenCanvas(mask.width, mask.height);
    const maskContext = maskCanvas.getContext("2d");
    if (!maskContext) throw new Error("Canvas 2D is unavailable in the worker.");
    const pixels = maskContext.createImageData(mask.width, mask.height);
    for (let i = 0; i < mask.data.length; i++) {
      const offset = i * 4;
      pixels.data[offset] = pixels.data[offset + 1] = pixels.data[offset + 2] = 255;
      pixels.data[offset + 3] = mask.data[i];
    }
    maskContext.putImageData(pixels, 0, 0);
    context.globalCompositeOperation = "destination-in";
    context.drawImage(maskCanvas, 0, 0, bitmap.width, bitmap.height);
    self.postMessage({ type: "progress", stage: "encoding-png" });
    return await canvas.convertToBlob({ type: "image/png" });
  } finally { bitmap.close(); }
}

self.onmessage = async (event: MessageEvent<Request>) => {
  try {
    const request = event.data;
    if (request.type === "initialize") {
      if (!session) {
        self.postMessage({ type: "progress", stage: "loading-model" });
        session = await ort.InferenceSession.create(request.model.url, { executionProviders: ["wasm"] });
        model = request.model;
      }
      self.postMessage({ type: "ready" });
      return;
    }
    if (request.type === "exportCutout") {
      const blob = await exportCutout(request.image, request.mask, request.maxInputPixels, request.maxOutputPixels);
      self.postMessage({ type: "exported", blob });
      return;
    }
    if (!session || !model) throw new Error("Model is not initialized.");
    if (request.type === "refine") {
      const dimensions = await readDimensions(request.image, request.maxInputPixels);
      if (!((dimensions.width === request.mask.sourceWidth && dimensions.height === request.mask.sourceHeight) || (dimensions.width === request.mask.sourceHeight && dimensions.height === request.mask.sourceWidth))) throw new Error("Mask source dimensions do not match the image.");
      self.postMessage({ type: "progress", stage: "decoding-image-region" });
      const detail = await inferRegion(request.image, request.region);
      if (!detail.subjectBounds.width || !detail.subjectBounds.height) throw new Error("No subject found in the selected region.");
      self.postMessage({ type: "progress", stage: "merging-mask" });
      const refined = mergeRefinement(request.mask, detail, request.region, request.mode);
      self.postMessage({ type: "done", mask: refined }, { transfer: [refined.data.buffer] });
      return;
    }
    const bitmap = await decodeImage(request.image, request.maxInputPixels);
    const sourceWidth = bitmap.width, sourceHeight = bitmap.height;
    // TODO: A centered crop can miss off-center subjects; use a qualified locator before expanding this fallback.
    const region = sourceWidth > model.inputWidth || sourceHeight > model.inputHeight ? {
      x: Math.floor(sourceWidth * 0.1), y: Math.floor(sourceHeight * 0.2),
      width: Math.floor(sourceWidth * 0.8), height: Math.floor(sourceHeight * 0.7),
    } : undefined;
    let centeredCrop: ImageBitmap | undefined;
    try {
      if (region) {
        try {
          centeredCrop = await createImageBitmap(bitmap, region.x, region.y, region.width, region.height, { resizeWidth: model.inputWidth, resizeHeight: model.inputHeight });
          if (centeredCrop.width !== model.inputWidth || centeredCrop.height !== model.inputHeight) {
            centeredCrop.close(); centeredCrop = undefined;
          }
        } catch { /* A blank result can still retry the crop from the source Blob. */ }
      }
      let mask = await infer(bitmap, sourceWidth, sourceHeight);
      if (!mask.subjectBounds.width && region) {
        self.postMessage({ type: "progress", stage: "checking-centered-crop" });
        let detail: Mask | undefined;
        try {
          if (centeredCrop) {
            const crop = centeredCrop; centeredCrop = undefined;
            detail = await infer(crop, region.width, region.height);
          } else detail = await inferRegion(request.image, region);
        } catch { /* A failed optional crop leaves the original mask. */ }
        if (detail) {
          let strong = 0;
          for (const alpha of detail.data) if (alpha >= 16) strong++;
          if (strong >= detail.data.length * 0.02) mask = mergeRefinement(mask, detail, region, "add");
        }
      }
      // TODO: This deliberately checks only sparse masks. Recovering a large wrong subject
      // (for example a crane instead of a boat) needs a semantic choice UI or a stronger joint model.
      let occupied = 0;
      for (const alpha of mask.data) if (alpha >= 16) occupied++;
      if (model.locatorUrl && occupied < mask.data.length * 0.03) {
        let located: { region: Rect; person: boolean }[] = [];
        try {
          located = await locate(request.image, sourceWidth, sourceHeight);
          if (!located.length) self.postMessage({ type: "progress", stage: "locator-no-match" });
        }
        catch (error) {
          self.postMessage({ type: "progress", stage: `locator-failed: ${error instanceof Error ? error.message : String(error)}` });
        }
        for (const { region, person } of located) {
          self.postMessage({ type: "progress", stage: `locator-found: ${JSON.stringify(region)}` });
          let detail: Mask | undefined;
          try { detail = await inferRegion(request.image, region); }
          catch (error) {
            self.postMessage({ type: "progress", stage: `locator-crop-failed: ${error instanceof Error ? error.message : String(error)}` });
          }
          if (detail) {
            let strong = 0;
            for (const alpha of detail.data) if (alpha >= 16) strong++;
            if (strong >= detail.data.length * 0.02) {
              // A high-confidence person crop replaces a sparse, often unrelated original fragment.
              const base = person ? { ...mask, data: new Uint8Array(mask.data.length) } : mask;
              const merged = mergeRefinement(base, detail, region, "add");
              let rescued = 0;
              for (const alpha of merged.data) if (alpha >= 16) rescued++;
              if (rescued > occupied * 2 && rescued >= merged.data.length * 0.005) {
                mask = merged;
                self.postMessage({ type: "progress", stage: "locator-rescue-applied" });
                break;
              } else self.postMessage({ type: "progress", stage: "locator-rescue-rejected" });
            } else self.postMessage({ type: "progress", stage: "locator-crop-weak" });
          }
        }
      }
      self.postMessage({ type: "done", mask }, { transfer: [mask.data.buffer] });
    } finally { centeredCrop?.close(); }
  } catch (error) {
    self.postMessage({ type: "error", error: error instanceof Error ? error.message : String(error) });
  }
};
