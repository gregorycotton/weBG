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

export interface ModelConfig {
  id: string;
  url: string;
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
