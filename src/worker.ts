import * as ort from "onnxruntime-web/wasm";
import { imageDimensions } from "./image";
import { outputToMask, pixelsToTensor, type Mask, type ModelConfig } from "./mask";

type Request =
  | { type: "initialize"; model: ModelConfig }
  | { type: "segment"; image: Blob; maxInputPixels: number }
  | { type: "exportCutout"; image: Blob; mask: Mask; maxInputPixels: number };

let session: ort.InferenceSession | undefined;
let model: ModelConfig | undefined;
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = new URL("./", import.meta.url).href;

async function decodeImage(image: Blob, maxInputPixels: number): Promise<ImageBitmap> {
  self.postMessage({ type: "progress", stage: "reading-image-header" });
  const { width, height } = imageDimensions(new Uint8Array(await image.arrayBuffer()));
  if (width * height > maxInputPixels) throw new Error("Image dimensions exceed the configured pixel limit.");
  self.postMessage({ type: "progress", stage: "decoding-image" });
  let bitmap: ImageBitmap;
  try { bitmap = await createImageBitmap(image); }
  catch { throw new Error("Could not decode JPEG, PNG, or WebP image."); }
  if (!bitmap.width || !bitmap.height || bitmap.width * bitmap.height > maxInputPixels) {
    bitmap.close();
    throw new Error("Image dimensions exceed the configured pixel limit.");
  }
  return bitmap;
}

async function exportCutout(image: Blob, mask: Mask, maxInputPixels: number): Promise<Blob> {
  const bitmap = await decodeImage(image, maxInputPixels);
  try {
    if (bitmap.width !== mask.sourceWidth || bitmap.height !== mask.sourceHeight) throw new Error("Mask source dimensions do not match the image.");
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
      const blob = await exportCutout(request.image, request.mask, request.maxInputPixels);
      self.postMessage({ type: "exported", blob });
      return;
    }
    if (!session || !model) throw new Error("Model is not initialized.");
    const bitmap = await decodeImage(request.image, request.maxInputPixels);
    try {
      const sourceWidth = bitmap.width, sourceHeight = bitmap.height;
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
          const mask = outputToMask(output.data, model.inputWidth, model.inputHeight, sourceWidth, sourceHeight, model.output);
          self.postMessage({ type: "done", mask }, { transfer: [mask.data.buffer] });
        } finally { for (const result of Object.values(results)) result.dispose(); }
      } finally { input.dispose(); }
    } finally { bitmap.close(); }
  } catch (error) {
    self.postMessage({ type: "error", error: error instanceof Error ? error.message : String(error) });
  }
};
