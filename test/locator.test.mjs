import assert from "node:assert/strict";
import test from "node:test";
import { locateWeakSubject } from "../src/locator.ts";

test("locator prefers the full boat among nearby scores and ignores uncertain boxes", () => {
  const logits = new Float32Array(100 * 92);
  logits.fill(-10);
  const boxes = new Float32Array(100 * 4);
  for (let i = 0; i < 100; i++) logits[i * 92 + 91] = 12; // no object
  logits[91] = 0; logits[9] = 3;
  boxes.set([0.45, 0.5, 0.1, 0.1], 0);
  logits[92 + 91] = 0; logits[92 + 9] = 2.6;
  boxes.set([0.5, 0.5, 0.4, 0.5], 4);
  assert.deepEqual(locateWeakSubject(logits, boxes, 1000, 800), { x: 240, y: 140, width: 520, height: 520 });
  logits[9] = -10; logits[92 + 9] = -10;
  assert.equal(locateWeakSubject(logits, boxes, 1000, 800), undefined);
});
