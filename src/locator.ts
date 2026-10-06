import type { Rect } from "./mask";

// YOLOS COCO classes: car=3, boat=9. Only these known weak cases trigger automatic rescue.
export function locateWeakSubject(logits: Float32Array, boxes: Float32Array, width: number, height: number): Rect | undefined {
  if (logits.length !== 100 * 92 || boxes.length !== 100 * 4) throw new Error("Locator output has an unexpected shape.");
  let bestScore = 0;
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
    bestScore = Math.max(bestScore, score);
    candidates.push({ score, box: Array.from(boxes.subarray(i * 4, i * 4 + 4)) });
  }
  if (bestScore < 0.7) return;
  const chosen = candidates.filter(candidate => candidate.score >= Math.max(0.7, bestScore - 0.08))
    .sort((a, b) => b.box[2] * b.box[3] - a.box[2] * a.box[3])[0];
  const [cx, cy, bw, bh] = chosen.box;
  if (![cx, cy, bw, bh].every(Number.isFinite) || bw <= 0 || bh <= 0) return;
  const x = Math.max(0, Math.round((cx - bw * 0.65) * width));
  const y = Math.max(0, Math.round((cy - bh * 0.65) * height));
  const right = Math.min(width, Math.round((cx + bw * 0.65) * width));
  const bottom = Math.min(height, Math.round((cy + bh * 0.65) * height));
  if (right - x < 16 || bottom - y < 16) return;
  return { x, y, width: right - x, height: bottom - y };
}
