import * as ort from "onnxruntime-web/wasm";
import { imageDimensions } from "./image";
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
  if (!session || !model) throw new Error("Model is not initialized.");
  const canvas = new OffscreenCanvas(model.inputWidth, model.inputHeight);
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("Canvas 2D is unavailable in the worker.");
  context.fillStyle = "#fff";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
  const input = new ort.Tensor("float32", pixelsToTensor(pixels, canvas.width, canvas.height, model.mean, model.std), [1, 3, canvas.height, canvas.width]);
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
      let bitmap: ImageBitmap;
      try { bitmap = await createImageBitmap(request.image, request.region.x, request.region.y, request.region.width, request.region.height, { resizeWidth: model.inputWidth, resizeHeight: model.inputHeight }); }
      catch { throw new Error("Could not decode the selected image region."); }
      try {
        if (bitmap.width !== model.inputWidth || bitmap.height !== model.inputHeight) throw new Error("Decoded region size does not match the model.");
        const detail = await infer(bitmap, request.region.width, request.region.height);
        if (!detail.subjectBounds.width || !detail.subjectBounds.height) throw new Error("No subject found in the selected region.");
        self.postMessage({ type: "progress", stage: "merging-mask" });
        const refined = mergeRefinement(request.mask, detail, request.region, request.mode);
        self.postMessage({ type: "done", mask: refined }, { transfer: [refined.data.buffer] });
      } finally { bitmap.close(); }
      return;
    }
    const bitmap = await decodeImage(request.image, request.maxInputPixels);
    try {
      const mask = await infer(bitmap, bitmap.width, bitmap.height);
      self.postMessage({ type: "done", mask }, { transfer: [mask.data.buffer] });
    } finally { bitmap.close(); }
  } catch (error) {
    self.postMessage({ type: "error", error: error instanceof Error ? error.message : String(error) });
  }
};
