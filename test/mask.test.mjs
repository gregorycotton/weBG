import assert from "node:assert/strict";
import { test } from "node:test";
import { mergeRefinement, outputToMask, pixelsToTensor, validateModel } from "../src/mask.ts";

test("RGB input is normalized in NCHW order", () => {
  const tensor = pixelsToTensor(new Uint8ClampedArray([255, 0, 128, 255, 0, 255, 64, 255]), 2, 1, [0, 0, 0], [1, 1, 1]);
  assert.deepEqual([...tensor], [1, 0, 0, 1, 128 / 255, 64 / 255].map(Math.fround));
});

test("soft alpha and bounds map from mask pixels to the source", () => {
  const mask = outputToMask(new Float32Array([-100, 0, 100, -100]), 2, 2, 6000, 4000, "logits");
  assert.deepEqual([...mask.data], [0, 128, 255, 0]);
  assert.deepEqual(mask.subjectBounds, { x: 0, y: 0, width: 6000, height: 4000 });
  assert.equal(mask.data.length, 4);
  const corner = outputToMask(new Float32Array([-100, 100, -100, -100]), 2, 2, 6000, 4000, "logits");
  assert.deepEqual(corner.subjectBounds, { x: 3000, y: 0, width: 3000, height: 2000 });
});

test("empty masks have empty bounds and malformed output fails", () => {
  assert.deepEqual(outputToMask(new Float32Array([0, 0, 0, 0]), 2, 2, 6000, 4000, "probabilities").subjectBounds, { x: 0, y: 0, width: 0, height: 0 });
  assert.throws(() => outputToMask(new Float32Array([NaN]), 1, 1, 1, 1, "probabilities"), /non-finite/);
});

test("model configuration rejects invalid input shapes", () => {
  const model = { id: "x", url: "/x.onnx", inputName: "x", outputName: "y", inputWidth: 512, inputHeight: 512, mean: [0, 0, 0], std: [1, 1, 1], output: "logits" };
  assert.throws(() => validateModel({ ...model, inputWidth: 0 }), /dimension/);
  assert.throws(() => validateModel({ ...model, inputWidth: 2048, inputHeight: 2048 }), /pixel budget/);
  assert.throws(() => validateModel({ ...model, mean: undefined }), /normalization/);
});

test("refinement adds or replaces only inside its feathered source region", () => {
  const empty = { data: new Uint8Array(16), width: 4, height: 4, sourceWidth: 4, sourceHeight: 4, subjectBounds: { x: 0, y: 0, width: 0, height: 0 } };
  const detail = { ...empty, data: new Uint8Array(16).fill(255) };
  const region = { x: 1, y: 1, width: 2, height: 2 };
  const added = mergeRefinement(empty, detail, region, "add");
  assert.equal(added.width, 1024);
  assert.equal(added.data[512 * 1024 + 512], 255);
  assert.equal(added.data[512 * 1024 + 100], 0);
  assert.deepEqual(added.subjectBounds, region);
  assert.equal(empty.data[5], 0);
  const solid = { ...empty, data: new Uint8Array(16).fill(255) };
  const removed = mergeRefinement(solid, empty, region, "replace");
  assert.equal(removed.data[512 * 1024 + 512], 0);
  assert.equal(removed.data[512 * 1024 + 100], 255);
});
