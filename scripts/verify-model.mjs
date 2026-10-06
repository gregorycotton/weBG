import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { createReadStream, readFileSync } from "node:fs";
import { basename, dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

export async function verifyModel(path = "models/birefnet-lite-512.json") {
  const manifestPath = resolve(path);
  const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
  assert.match(manifest.runtime.url, /^\/models\/[\w.-]+\.onnx$/);
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(join(dirname(manifestPath), basename(manifest.runtime.url)))) hash.update(chunk);
  assert.equal(hash.digest("hex"), manifest.exportedSHA256, "Model SHA-256 differs from the provenance manifest");
  assert.deepEqual(manifest.inputShape, [1, 3, manifest.runtime.inputHeight, manifest.runtime.inputWidth]);
  assert.deepEqual(manifest.outputShape, [1, 1, manifest.runtime.inputHeight, manifest.runtime.inputWidth]);
  if (manifest.runtime.locatorUrl) {
    assert.match(manifest.runtime.locatorUrl, /^\/models\/[\w.-]+\.onnx$/);
    const locatorManifest = JSON.parse(readFileSync(join(dirname(manifestPath), basename(manifest.runtime.locatorUrl).replace(/\.onnx$/, ".json")), "utf8"));
    const locatorHash = createHash("sha256");
    for await (const chunk of createReadStream(join(dirname(manifestPath), basename(manifest.runtime.locatorUrl)))) locatorHash.update(chunk);
    assert.equal(locatorHash.digest("hex"), locatorManifest.exportedSHA256, "Locator SHA-256 differs from the provenance manifest");
  }
  return manifest;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const manifest = await verifyModel(process.argv[2]);
  console.log(`Verified ${manifest.modelId}: ${manifest.exportedSHA256}`);
}
