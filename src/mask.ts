export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface Mask {
  data: Uint8Array;
  width: number;
  height: number;
  sourceWidth: number;
  sourceHeight: number;
  subjectBounds: Rect;
}

export type RefineMode = "replace" | "add";

export interface ModelConfig {
  id: string;
  url: string;
  locatorUrl?: string;
  inputName: string;
  outputName: string;
  inputWidth: number;
  inputHeight: number;
  mean: [number, number, number];
  std: [number, number, number];
  output: "logits" | "probabilities";
}

export function validateModel(model: ModelConfig): void {
  if (!model || ![model.id, model.url, model.inputName, model.outputName].every(value => typeof value === "string" && value.length > 0)) throw new Error("Model ID, URL, and tensor names are required.");
  if (![model.inputWidth, model.inputHeight].every(value => Number.isInteger(value) && value > 0 && value <= 2048) || model.inputWidth * model.inputHeight > 2_097_152) throw new Error("Model input exceeds its dimension or pixel budget.");
  if (!Array.isArray(model.mean) || !Array.isArray(model.std) || model.mean.length !== 3 || model.std.length !== 3 || ![...model.mean, ...model.std].every(Number.isFinite) || model.std.some(value => value <= 0)) throw new Error("Model normalization must contain three finite means and positive standard deviations.");
  if (model.output !== "logits" && model.output !== "probabilities") throw new Error("Model output must be logits or probabilities.");
  if (model.locatorUrl !== undefined && (typeof model.locatorUrl !== "string" || !model.locatorUrl)) throw new Error("Locator URL must be a nonempty string.");
}

export function pixelsToTensor(pixels: Uint8ClampedArray, width: number, height: number, mean: ModelConfig["mean"], std: ModelConfig["std"]): Float32Array {
  const plane = width * height;
  if (pixels.length !== plane * 4) throw new Error("Unexpected RGBA image size.");
  const tensor = new Float32Array(plane * 3);
  for (let pixel = 0; pixel < plane; pixel++) {
    for (let channel = 0; channel < 3; channel++) tensor[channel * plane + pixel] = (pixels[pixel * 4 + channel] / 255 - mean[channel]) / std[channel];
  }
  return tensor;
}

export function outputToMask(values: Float32Array, width: number, height: number, sourceWidth: number, sourceHeight: number, output: ModelConfig["output"]): Mask {
  if (values.length !== width * height) throw new Error("Unexpected model output size.");
  const data = new Uint8Array(values.length);
  let minX = width, minY = height, maxX = -1, maxY = -1;
  for (let i = 0; i < values.length; i++) {
    const value = output === "logits" ? 1 / (1 + Math.exp(-values[i])) : values[i];
    if (!Number.isFinite(value)) throw new Error("Model output contains a non-finite value.");
    const alpha = Math.round(Math.max(0, Math.min(1, value)) * 255);
    data[i] = alpha;
    if (alpha >= 16) {
      const x = i % width, y = Math.floor(i / width);
      minX = Math.min(minX, x); minY = Math.min(minY, y);
      maxX = Math.max(maxX, x); maxY = Math.max(maxY, y);
    }
  }
  const subjectBounds = maxX < 0 ? { x: 0, y: 0, width: 0, height: 0 } : (() => {
    const x = Math.floor(minX * sourceWidth / width), y = Math.floor(minY * sourceHeight / height);
    return { x, y, width: Math.ceil((maxX + 1) * sourceWidth / width) - x, height: Math.ceil((maxY + 1) * sourceHeight / height) - y };
  })();
  return { data, width, height, sourceWidth, sourceHeight, subjectBounds };
}

export function mergeRefinement(base: Mask, detail: Mask, region: Rect, mode: RefineMode): Mask {
  const scale = Math.max(1, 1024 / Math.max(base.width, base.height));
  const width = Math.round(base.width * scale), height = Math.round(base.height * scale);
  const data = new Uint8Array(width * height);
  const sample = (mask: Mask, u: number, v: number): number => {
    const x = Math.max(0, Math.min(mask.width - 1, u * mask.width - 0.5));
    const y = Math.max(0, Math.min(mask.height - 1, v * mask.height - 0.5));
    const x0 = Math.floor(x), y0 = Math.floor(y), x1 = Math.min(x0 + 1, mask.width - 1), y1 = Math.min(y0 + 1, mask.height - 1);
    const dx = x - x0, dy = y - y0;
    return (mask.data[y0 * mask.width + x0] * (1 - dx) + mask.data[y0 * mask.width + x1] * dx) * (1 - dy)
      + (mask.data[y1 * mask.width + x0] * (1 - dx) + mask.data[y1 * mask.width + x1] * dx) * dy;
  };
  const feather = Math.max(1, Math.min(region.width, region.height) * 0.08);
  let minX = width, minY = height, maxX = -1, maxY = -1;
  for (let y = 0; y < height; y++) {
    const sy = (y + 0.5) * base.sourceHeight / height;
    for (let x = 0; x < width; x++) {
      const sx = (x + 0.5) * base.sourceWidth / width;
      const original = sample(base, sx / base.sourceWidth, sy / base.sourceHeight);
      let alpha = original;
      if (sx >= region.x && sx < region.x + region.width && sy >= region.y && sy < region.y + region.height) {
        const refined = sample(detail, (sx - region.x) / region.width, (sy - region.y) / region.height);
        const edge = Math.min(sx - region.x, region.x + region.width - sx, sy - region.y, region.y + region.height - sy);
        const weight = Math.min(1, edge / feather);
        alpha = mode === "add" ? Math.max(original, refined * weight) : original * (1 - weight) + refined * weight;
      }
      const value = Math.round(alpha);
      data[y * width + x] = value;
      if (value >= 16) { minX = Math.min(minX, x); minY = Math.min(minY, y); maxX = Math.max(maxX, x); maxY = Math.max(maxY, y); }
    }
  }
  const subjectBounds = maxX < 0 ? { x: 0, y: 0, width: 0, height: 0 } : (() => {
    const x = Math.floor(minX * base.sourceWidth / width), y = Math.floor(minY * base.sourceHeight / height);
    return { x, y, width: Math.ceil((maxX + 1) * base.sourceWidth / width) - x, height: Math.ceil((maxY + 1) * base.sourceHeight / height) - y };
  })();
  return { data, width, height, sourceWidth: base.sourceWidth, sourceHeight: base.sourceHeight, subjectBounds };
}
