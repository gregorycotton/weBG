import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { createReadStream, readFileSync } from "node:fs";

const manifest = JSON.parse(readFileSync("models/birefnet-lite-512.json", "utf8"));
const hash = createHash("sha256");
for await (const chunk of createReadStream("models/birefnet-lite-512.onnx")) hash.update(chunk);
assert.equal(hash.digest("hex"), manifest.exportedSHA256, "Model SHA-256 differs from the provenance manifest");
assert.deepEqual(manifest.inputShape, [1, 3, manifest.runtime.inputHeight, manifest.runtime.inputWidth]);
assert.deepEqual(manifest.outputShape, [1, 1, manifest.runtime.inputHeight, manifest.runtime.inputWidth]);
console.log(`Verified ${manifest.modelId}: ${manifest.exportedSHA256}`);
