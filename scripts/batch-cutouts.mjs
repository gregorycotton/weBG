import { createReadStream } from "node:fs";
import { access, link, mkdir, readdir, stat, unlink } from "node:fs/promises";
import { createServer } from "node:http";
import { randomUUID } from "node:crypto";
import { basename, dirname, extname, join, resolve } from "node:path";
import { pipeline } from "node:stream/promises";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright-core";
import { verifyModel } from "./verify-model.mjs";

const root = fileURLToPath(new URL("..", import.meta.url));
const mime = { ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp" };
const args = process.argv.slice(2);
if (![0, 2, 3].includes(args.length)) throw new Error("Usage: npm run batch -- [input-folder output-folder [model-manifest]]");
const inputDir = resolve(args[0] ?? join(root, "photoset"));
const outputDir = resolve(args[1] ?? join(root, "outputs"));
const manifestPath = resolve(args[2] ?? join(root, "models/birefnet-lite-512.json"));

async function exists(path) {
  try { await access(path); return true; }
  catch (error) { if (error.code === "ENOENT") return false; throw error; }
}

async function main() {
  const entries = await readdir(inputDir, { withFileTypes: true });
  const photos = entries.filter(entry => entry.isFile() && mime[extname(entry.name).toLowerCase()]).map(entry => entry.name).sort();
  if (!photos.length) { console.log(`No JPEG, PNG, or WebP files in ${inputDir}`); return; }
  const names = new Set();
  for (const photo of photos) {
    const output = `${basename(photo, extname(photo))}-CUTOUT.png`.toLowerCase();
    if (names.has(output)) throw new Error(`Two photos would produce the same output name: ${output}`);
    names.add(output);
  }
  await mkdir(outputDir, { recursive: true });
  const pending = [];
  for (const photo of photos) {
    const output = join(outputDir, `${basename(photo, extname(photo))}-CUTOUT.png`);
    if (await exists(output)) console.log(`SKIP existing ${basename(output)}`);
    else pending.push({ photo, output });
  }
  if (!pending.length) { console.log("All cutouts already exist."); return; }
  await access(join(root, "dist/index.js"));
  const { runtime } = await verifyModel(manifestPath);
  const modelFile = join(dirname(manifestPath), basename(runtime.url));
  let activePhoto;
  const server = createServer((request, response) => {
    const path = new URL(request.url, "http://localhost").pathname;
    let file, type;
    if (request.method === "GET" && path === "/") {
      response.writeHead(200, { "Content-Type": "text/html" });
      response.end('<!doctype html><meta charset="utf-8"><a id="download">Download cutout</a>');
      return;
    }
    if (request.method === "GET" && path === "/input" && activePhoto) {
      file = activePhoto;
      type = mime[extname(file).toLowerCase()];
    } else if (request.method === "GET" && /^\/dist\/[\w.-]+$/.test(path)) {
      file = join(root, path);
      type = path.endsWith(".wasm") ? "application/wasm" : "text/javascript";
    } else if (request.method === "GET" && path === runtime.url) {
      file = modelFile;
      type = "application/octet-stream";
    } else {
      response.writeHead(404); response.end("Not found"); return;
    }
    (async () => {
      try {
        const info = await stat(file);
        response.writeHead(200, { "Content-Type": type, "Content-Length": info.size, "Cache-Control": "no-store" });
        await pipeline(createReadStream(file), response);
      } catch (error) {
        if (response.headersSent) response.destroy(error);
        else { response.writeHead(500); response.end(`Could not read ${path}`); }
      }
    })();
  });
  await new Promise((resolve, reject) => { server.once("error", reject); server.listen(0, "127.0.0.1", resolve); });
  let browser;
  try {
    browser = await chromium.launch({ channel: "chrome", headless: true });
    const context = await browser.newContext({ acceptDownloads: true });
    const page = await context.newPage();
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    await page.evaluate(async model => {
      const { createSegmenter } = await import("/dist/index.js");
      globalThis.batchSegmenter = createSegmenter({ model, maxExportPixels: 40_000_000 });
    }, runtime);
    let failed = 0;
    for (const [index, { photo, output }] of pending.entries()) {
      activePhoto = join(inputDir, photo);
      const temporary = `${output}.partial-${randomUUID()}`;
      console.log(`[${index + 1}/${pending.length}] ${photo}`);
      try {
        const details = await page.evaluate(async () => {
          const response = await fetch("/input");
          if (!response.ok) throw new Error(`Could not load input (${response.status})`);
          const image = await response.blob();
          const start = performance.now();
          const mask = await globalThis.batchSegmenter.segment(image);
          const segmented = performance.now();
          const png = await globalThis.batchSegmenter.exportCutout(image, mask);
          const exported = performance.now();
          const download = document.querySelector("#download");
          download.href = URL.createObjectURL(png);
          download.download = "cutout.png";
          return { width: mask.sourceWidth, height: mask.sourceHeight, bytes: png.size, segmentMs: Math.round(segmented - start), exportMs: Math.round(exported - segmented) };
        });
        const downloadPromise = page.waitForEvent("download", { timeout: 120000 });
        await page.locator("#download").click();
        const download = await downloadPromise;
        await download.saveAs(temporary);
        await link(temporary, output); // Never overwrite a reviewed result.
        console.log(`  WROTE ${basename(output)} (${details.width}×${details.height}, ${(details.bytes / 1024 / 1024).toFixed(1)} MiB; segment ${details.segmentMs} ms, PNG ${details.exportMs} ms)`);
      } catch (error) {
        failed++;
        console.error(`  FAILED ${photo}: ${error.message}`);
        if (page.isClosed()) break;
      } finally {
        await page.evaluate(() => {
          const download = document.querySelector("#download");
          if (download.href.startsWith("blob:")) URL.revokeObjectURL(download.href);
          download.removeAttribute("href");
        }).catch(() => {});
        if (await exists(temporary)) await unlink(temporary);
      }
    }
    await page.evaluate(() => globalThis.batchSegmenter.dispose()).catch(() => {});
    if (failed) throw new Error(`${failed} image(s) failed; completed cutouts were kept.`);
  } finally {
    await browser?.close();
    await new Promise(resolve => server.close(resolve));
  }
}

main().catch(error => { console.error(error.message); process.exitCode = 1; });
