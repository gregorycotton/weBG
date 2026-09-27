import * as ort from "onnxruntime-web/wasm";
import { outputToMask, pixelsToTensor, type ModelConfig } from "./mask";

type Request =
  | { type: "initialize"; model: ModelConfig }
  | { type: "segment"; image: Blob; maxInputPixels: number };

let session: ort.InferenceSession | undefined;
let model: ModelConfig | undefined;
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = new URL("./", import.meta.url).href;

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
    if (!session || !model) throw new Error("Model is not initialized.");
    self.postMessage({ type: "progress", stage: "decoding-image" });
    // TODO: createImageBitmap may decode a very large image before the pixel-limit check.
    // Read dimensions from image headers or ImageDecoder metadata before decoding if untrusted huge files become common.
    const bitmap = await createImageBitmap(request.image);
    try {
      const sourceWidth = bitmap.width, sourceHeight = bitmap.height;
      if (!sourceWidth || !sourceHeight || sourceWidth * sourceHeight > request.maxInputPixels) throw new Error("Image dimensions exceed the configured pixel limit.");
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
