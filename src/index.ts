import { validateModel, type Mask, type ModelConfig, type Rect, type RefineMode } from "./mask";

export type { Mask, ModelConfig, Rect, RefineMode } from "./mask";

export interface SegmenterOptions {
  model: ModelConfig;
  maxInputBytes?: number;
  maxInputPixels?: number;
  maxExportPixels?: number;
}

export interface ExportOptions { maxOutputPixels?: number }

export interface Segmenter {
  initialize(onProgress?: (stage: string) => void): Promise<void>;
  segment(image: Blob, onProgress?: (stage: string) => void): Promise<Mask>;
  refine(image: Blob, mask: Mask, region: Rect, mode: RefineMode, onProgress?: (stage: string) => void): Promise<Mask>;
  exportCutout(image: Blob, mask: Mask, onProgress?: (stage: string) => void, options?: ExportOptions): Promise<Blob>;
  dispose(): void;
}

type WorkerReply =
  | { type: "progress"; stage: string }
  | { type: "ready" }
  | { type: "done"; mask: Mask }
  | { type: "exported"; blob: Blob }
  | { type: "error"; error: string };

export function createSegmenter(options: SegmenterOptions): Segmenter {
  validateModel(options.model);
  const modelUrl = new URL(options.model.url, location.href);
  if (modelUrl.origin !== location.origin) throw new Error("Model must be hosted on the application's origin.");
  const model = { ...options.model, url: modelUrl.href };
  const maxInputBytes = options.maxInputBytes ?? 25 * 1024 * 1024;
  const maxInputPixels = options.maxInputPixels ?? 40_000_000;
  const maxExportPixels = options.maxExportPixels ?? 16_000_000;
  if (![maxInputBytes, maxInputPixels, maxExportPixels].every(value => Number.isSafeInteger(value) && value > 0)) throw new Error("Input and export limits must be positive integers.");
  let worker: Worker | undefined;
  let initializing: Promise<void> | undefined;
  let pending: { resolve(value: WorkerReply): void; reject(error: Error): void; onProgress?: (stage: string) => void } | undefined;
  let disposed = false;

  function validateImage(image: Blob): void {
    if (!image || typeof image.size !== "number" || typeof image.arrayBuffer !== "function") throw new Error("Input must be an image Blob.");
    if (!Number.isSafeInteger(image.size) || !image.size || image.size > maxInputBytes) throw new Error(`Image must be between 1 and ${maxInputBytes} bytes.`);
    if (image.type && !["image/jpeg", "image/png", "image/webp"].includes(image.type)) throw new Error("Use a JPEG, PNG, or WebP image.");
  }

  function validateMask(mask: Mask): void {
    if (!mask || !(mask.data instanceof Uint8Array) || ![mask.width, mask.height, mask.sourceWidth, mask.sourceHeight].every(value => Number.isSafeInteger(value) && value > 0) || mask.width > 2048 || mask.height > 2048 || mask.width * mask.height > 2_097_152 || mask.sourceWidth * mask.sourceHeight > maxInputPixels || mask.data.length !== mask.width * mask.height) throw new Error("Mask has invalid dimensions or pixel data.");
  }

  function getWorker(): Worker {
    if (disposed) throw new Error("Segmenter has been disposed.");
    if (!worker) {
      worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
      worker.onmessage = event => {
        const reply = event.data as WorkerReply;
        if (!pending) return;
        if (reply.type === "progress") { pending.onProgress?.(reply.stage); return; }
        const current = pending; pending = undefined;
        if (reply.type === "error") current.reject(new Error(reply.error));
        else current.resolve(reply);
      };
      worker.onerror = event => {
        pending?.reject(new Error(event.message || "Segmentation worker failed."));
        pending = undefined;
        worker?.terminate(); worker = undefined; initializing = undefined;
      };
      worker.onmessageerror = () => {
        pending?.reject(new Error("Could not read the segmentation result."));
        pending = undefined;
        worker?.terminate(); worker = undefined; initializing = undefined;
      };
    }
    return worker;
  }

  function send(message: object, onProgress?: (stage: string) => void): Promise<WorkerReply> {
    if (pending) return Promise.reject(new Error("Another operation is running."));
    return new Promise((resolve, reject) => {
      pending = { resolve, reject, onProgress };
      try { getWorker().postMessage(message); }
      catch (error) { pending = undefined; reject(error); }
    });
  }

  return {
    initialize(onProgress) {
      initializing ??= send({ type: "initialize", model }, onProgress).then(() => undefined).catch(error => { initializing = undefined; throw error; });
      return initializing;
    },
    async segment(image, onProgress) {
      validateImage(image);
      await this.initialize(onProgress);
      const reply = await send({ type: "segment", image, maxInputPixels }, onProgress);
      if (reply.type !== "done") throw new Error("Segmentation worker returned an unexpected response.");
      return reply.mask;
    },
    async refine(image, mask, region, mode, onProgress) {
      validateImage(image);
      validateMask(mask);
      if (!region || ![region.x, region.y, region.width, region.height].every(Number.isSafeInteger)
        || region.x < 0 || region.y < 0 || region.width <= 0 || region.height <= 0
        || region.x + region.width > mask.sourceWidth || region.y + region.height > mask.sourceHeight) throw new Error("Refinement region must be inside the source image.");
      if (mode !== "replace" && mode !== "add") throw new Error("Refinement mode must be replace or add.");
      await this.initialize(onProgress);
      const reply = await send({ type: "refine", image, mask, region, mode, maxInputPixels }, onProgress);
      if (reply.type !== "done") throw new Error("Refinement worker returned an unexpected response.");
      return reply.mask;
    },
    async exportCutout(image, mask, onProgress, options) {
      validateImage(image);
      validateMask(mask);
      const maxOutputPixels = options?.maxOutputPixels;
      if (maxOutputPixels !== undefined && (!Number.isSafeInteger(maxOutputPixels) || maxOutputPixels <= 0)) throw new Error("maxOutputPixels must be a positive integer.");
      const outputPixels = Math.min(mask.sourceWidth * mask.sourceHeight, maxOutputPixels ?? Number.MAX_SAFE_INTEGER);
      if (outputPixels > maxExportPixels) throw new Error(`PNG export exceeds the ${maxExportPixels}-pixel limit. Request a smaller PNG with maxOutputPixels or raise maxExportPixels.`);
      const reply = await send({ type: "exportCutout", image, mask, maxInputPixels, maxOutputPixels: outputPixels }, onProgress);
      if (reply.type !== "exported") throw new Error("Export worker returned an unexpected response.");
      return reply.blob;
    },
    dispose() {
      disposed = true;
      pending?.reject(new Error("Segmenter was disposed."));
      pending = undefined;
      worker?.terminate(); worker = undefined;
    }
  };
}
