import { validateModel, type Mask, type ModelConfig } from "./mask";

export type { Mask, ModelConfig, Rect } from "./mask";

export interface SegmenterOptions {
  model: ModelConfig;
  maxInputBytes?: number;
  maxInputPixels?: number;
}

export interface Segmenter {
  initialize(onProgress?: (stage: string) => void): Promise<void>;
  segment(image: Blob, onProgress?: (stage: string) => void): Promise<Mask>;
  dispose(): void;
}

type WorkerReply =
  | { type: "progress"; stage: string }
  | { type: "ready" }
  | { type: "done"; mask: Mask }
  | { type: "error"; error: string };

export function createSegmenter(options: SegmenterOptions): Segmenter {
  validateModel(options.model);
  const modelUrl = new URL(options.model.url, location.href);
  if (modelUrl.origin !== location.origin) throw new Error("Model must be hosted on the application's origin.");
  const model = { ...options.model, url: modelUrl.href };
  const maxInputBytes = options.maxInputBytes ?? 25 * 1024 * 1024;
  const maxInputPixels = options.maxInputPixels ?? 40_000_000;
  if (![maxInputBytes, maxInputPixels].every(value => Number.isSafeInteger(value) && value > 0)) throw new Error("Input limits must be positive integers.");
  let worker: Worker | undefined;
  let initializing: Promise<void> | undefined;
  let pending: { resolve(value: WorkerReply): void; reject(error: Error): void; onProgress?: (stage: string) => void } | undefined;
  let disposed = false;

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
    if (pending) return Promise.reject(new Error("Another segmentation operation is running."));
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
      if (!image || typeof image.size !== "number" || typeof image.arrayBuffer !== "function") throw new Error("Input must be an image Blob.");
      if (!image.size || image.size > maxInputBytes) throw new Error(`Image must be between 1 and ${maxInputBytes} bytes.`);
      if (image.type && !["image/jpeg", "image/png", "image/webp"].includes(image.type)) throw new Error("Use a JPEG, PNG, or WebP image.");
      await this.initialize(onProgress);
      const reply = await send({ type: "segment", image, maxInputPixels }, onProgress);
      if (reply.type !== "done") throw new Error("Segmentation worker returned an unexpected response.");
      return reply.mask;
    },
    dispose() {
      disposed = true;
      pending?.reject(new Error("Segmenter was disposed."));
      pending = undefined;
      worker?.terminate(); worker = undefined;
    }
  };
}
