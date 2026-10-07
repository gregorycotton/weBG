import type { Rect } from "./mask";

// YOLOS COCO classes: car=3, boat=9. Only these known weak cases trigger automatic rescue.
export function locateWeakSubjects(logits: Float32Array, boxes: Float32Array, width: number, height: number): Rect[] {
  if (logits.length !== 100 * 92 || boxes.length !== 100 * 4) throw new Error("Locator output has an unexpected shape.");
  const candidates: { score: number; box: number[] }[] = [];
  for (let i = 0; i < 100; i++) {
    const offset = i * 92;
    let largest = -Infinity, total = 0;
    for (let j = 0; j < 92; j++) largest = Math.max(largest, logits[offset + j]);
    for (let j = 0; j < 92; j++) total += Math.exp(logits[offset + j] - largest);
    const car = Math.exp(logits[offset + 3] - largest) / total;
    const boat = Math.exp(logits[offset + 9] - largest) / total;
    const score = Math.max(car, boat);
    if (!Number.isFinite(score)) throw new Error("Locator output contains a non-finite value.");
    const box = Array.from(boxes.subarray(i * 4, i * 4 + 4));
    if (score >= 0.6 && box.every(Number.isFinite) && box[2] > 0 && box[3] > 0) candidates.push({ score, box });
  }
  const regions: Rect[] = [];
  for (const candidate of candidates.sort((a, b) => b.score * b.box[2] * b.box[3] - a.score * a.box[2] * a.box[3])) {
    const [cx, cy, bw, bh] = candidate.box;
    const x = Math.max(0, Math.round((cx - bw * 0.65) * width));
    const y = Math.max(0, Math.round((cy - bh * 0.65) * height));
    const right = Math.min(width, Math.round((cx + bw * 0.65) * width));
    const bottom = Math.min(height, Math.round((cy + bh * 0.65) * height));
    if (right - x < 16 || bottom - y < 16) continue;
    const region = { x, y, width: right - x, height: bottom - y };
    if (regions.some(previous => {
      const overlap = Math.max(0, Math.min(right, previous.x + previous.width) - Math.max(x, previous.x))
        * Math.max(0, Math.min(bottom, previous.y + previous.height) - Math.max(y, previous.y));
      return overlap / (region.width * region.height + previous.width * previous.height - overlap) > 0.8;
    })) continue;
    regions.push(region);
    // TODO: A third crop could recover more scenes but costs another 512 inference on mobile.
    if (regions.length === 2) break;
  }
  return regions;
}
