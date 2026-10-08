import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { mkdtemp, readFile, rm, stat } from "node:fs/promises";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { chromium } from "playwright-core";

const { version } = JSON.parse(await readFile(new URL("../package.json", import.meta.url), "utf8"));
const archive = resolve(process.argv[2] ?? `release/v${version}/webg-${version}-browser.tar.gz`);
const temporary = await mkdtemp(join(tmpdir(), "webg-consumer-"));
let server;
let browser;

async function sha256(path) {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest("hex");
}

try {
  const extracted = spawnSync("tar", ["-xzf", archive, "-C", temporary], { stdio: "inherit" });
  if (extracted.error) throw extracted.error;
  assert.equal(extracted.status, 0, "Could not unpack the browser release");
  const packageJson = JSON.parse(await readFile(join(temporary, "webg/package.json"), "utf8"));
  assert.equal(packageJson.name, "webg");
  for (const asset of ["dist/index.js", "dist/worker.js", "dist/ort-wasm-simd-threaded.mjs", "dist/ort-wasm-simd-threaded.wasm", "docs/weBG-README.png"]) {
    assert((await stat(join(temporary, "webg", asset))).size > 0, `Missing ${asset}`);
  }
  const manifest = JSON.parse(await readFile(join(temporary, "models/birefnet-lite-512.json"), "utf8"));
  const locator = JSON.parse(await readFile(join(temporary, "models/yolos-tiny-416.json"), "utf8"));
  assert.equal(await sha256(join(temporary, "models/birefnet-lite-512.onnx")), manifest.exportedSHA256);
  assert.equal(await sha256(join(temporary, "models/yolos-tiny-416.onnx")), locator.exportedSHA256);
  assert.equal(manifest.runtime.locatorUrl, "/models/yolos-tiny-416.onnx");

  const page = `<!doctype html><script type="module">
    import { createSegmenter } from "/webg/dist/index.js";
    try {
      const image = await createImageBitmap(await (await fetch("/webg/docs/weBG-README.png")).blob());
      const canvas = document.createElement("canvas");
      canvas.width = Math.floor(image.width / 2); canvas.height = image.height;
      canvas.getContext("2d").drawImage(image, 0, 0);
      image.close();
      const input = await new Promise(resolve => canvas.toBlob(resolve, "image/png"));
      const manifest = await (await fetch("/models/birefnet-lite-512.json")).json();
      const segmenter = createSegmenter({ model: manifest.runtime });
      try {
        const output = await segmenter.removeBackground(input);
        const bitmap = await createImageBitmap(output);
        const result = document.createElement("canvas");
        result.width = bitmap.width; result.height = bitmap.height;
        const context = result.getContext("2d", { willReadFrequently: true });
        context.drawImage(bitmap, 0, 0);
        bitmap.close();
        const alpha = context.getImageData(0, 0, result.width, result.height).data;
        let transparent = false, opaque = false;
        for (let i = 3; i < alpha.length; i += 4) { transparent ||= alpha[i] < 128; opaque ||= alpha[i] > 128; }
        if (output.type !== "image/png" || result.width !== canvas.width || result.height !== canvas.height || !transparent || !opaque) throw new Error("Invalid cutout PNG");
        window.releaseResult = { width: result.width, height: result.height };
      } finally { segmenter.dispose(); }
    } catch (error) { window.releaseResult = { error: String(error) }; }
  </script>`;
  server = createServer(async (request, response) => {
    const pathname = new URL(request.url, "http://localhost").pathname;
    if (pathname === "/") { response.writeHead(200, { "Content-Type": "text/html" }); response.end(page); return; }
    if (!/^\/(?:webg\/(?:dist|docs)\/|models\/)[\w./-]+$/.test(pathname) || pathname.includes("..")) { response.writeHead(404); response.end(); return; }
    const file = join(temporary, pathname);
    try {
      const info = await stat(file);
      const type = pathname.endsWith(".js") || pathname.endsWith(".mjs") ? "text/javascript" : pathname.endsWith(".json") ? "application/json" : pathname.endsWith(".wasm") ? "application/wasm" : pathname.endsWith(".png") ? "image/png" : "application/octet-stream";
      response.writeHead(200, { "Content-Type": type, "Content-Length": info.size });
      createReadStream(file).pipe(response);
    } catch { response.writeHead(404); response.end(); }
  });
  await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
  browser = await chromium.launch({ channel: "chrome", headless: true });
  const tab = await browser.newPage();
  await tab.goto(`http://127.0.0.1:${server.address().port}/`);
  await tab.waitForFunction(() => window.releaseResult, null, { timeout: 120_000 });
  const result = await tab.evaluate(() => window.releaseResult);
  assert(!result.error, result.error);
  console.log(`PASS: ${packageJson.name}@${packageJson.version} produced a ${result.width}×${result.height} cutout from the unpacked release alone`);
} finally {
  await browser?.close();
  if (server) await new Promise(resolve => server.close(resolve));
  await rm(temporary, { recursive: true, force: true });
}
