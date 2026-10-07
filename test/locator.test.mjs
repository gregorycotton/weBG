import assert from "node:assert/strict";
import test from "node:test";
import { locateWeakSubjects } from "../src/locator.ts";

test("locator offers distinct substantial boxes and ignores uncertain boxes", () => {
  const logits = new Float32Array(100 * 92);
  logits.fill(-10);
  const boxes = new Float32Array(100 * 4);
  for (let i = 0; i < 100; i++) logits[i * 92 + 91] = 12; // no object
  logits[91] = 0; logits[9] = 3;
  boxes.set([0.45, 0.5, 0.1, 0.1], 0);
  logits[92 + 91] = 0; logits[92 + 9] = 0.7; // Lower-confidence but much larger boat.
  boxes.set([0.5, 0.5, 0.4, 0.5], 4);
  assert.deepEqual(locateWeakSubjects(logits, boxes, 1000, 800), [
    { region: { x: 240, y: 140, width: 520, height: 520 }, person: false },
    { region: { x: 385, y: 348, width: 130, height: 104 }, person: false },
  ]);
  logits[9] = -10; logits[92 + 9] = -10;
  assert.deepEqual(locateWeakSubjects(logits, boxes, 1000, 800), []);
  logits[0] = 4; // COCO class 0 is N/A, not person.
  assert.deepEqual(locateWeakSubjects(logits, boxes, 1000, 800), []);
  logits[0] = -10;
  logits[1] = 4;
  boxes.set([0.5, 0.5, 0.2, 0.4], 0);
  assert.deepEqual(locateWeakSubjects(logits, boxes, 1000, 800), [
    { region: { x: 370, y: 192, width: 260, height: 416 }, person: true },
  ]);
  logits[1] = 2;
  assert.deepEqual(locateWeakSubjects(logits, boxes, 1000, 800), []);
});
