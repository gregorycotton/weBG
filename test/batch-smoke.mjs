import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdtemp, mkdir, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { deflateSync } from "node:zlib";

const root = fileURLToPath(new URL("..", import.meta.url));
const temp = await mkdtemp(join(tmpdir(), "webg-batch-"));
const input = join(temp, "photoset"), output = join(temp, "outputs");

function pngChunk(name, data) {
  const type = Buffer.from(name);
  const bytes = Buffer.concat([type, data]);
  let crc = 0xffffffff;
  for (const byte of bytes) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit++) crc = (crc >>> 1) ^ (0xedb88320 & -(crc & 1));
  }
  const chunk = Buffer.alloc(data.length + 12);
  chunk.writeUInt32BE(data.length, 0);
  bytes.copy(chunk, 4);
  chunk.writeUInt32BE((crc ^ 0xffffffff) >>> 0, data.length + 8);
  return chunk;
}

async function run() {
  const child = spawn(process.platform === "win32" ? "npm.cmd" : "npm", ["run", "batch", "--", input, output], { cwd: root });
  let stdout = "", stderr = "";
  child.stdout.on("data", chunk => { stdout += chunk; });
  child.stderr.on("data", chunk => { stderr += chunk; });
  const code = await new Promise((resolve, reject) => { child.once("error", reject); child.once("close", resolve); });
  assert.equal(code, 0, stdout + stderr);
  return stdout;
}

try {
  await mkdir(input);
  const header = Buffer.alloc(13);
  const size = 256;
  header.writeUInt32BE(size, 0); header.writeUInt32BE(size, 4);
  header[8] = 8; header[9] = 6;
  const pixels = Buffer.alloc(size * (size * 4 + 1));
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const offset = y * (size * 4 + 1) + 1 + x * 4;
    const subject = ((x - 128) / 68) ** 2 + ((y - 135) / 92) ** 2 <= 1;
    pixels.set(subject ? [48, 75, 121, 255] : [240, 232, 216, 255], offset);
  }
  const png = Buffer.concat([
    Buffer.from("89504e470d0a1a0a", "hex"),
    pngChunk("IHDR", header),
    pngChunk("IDAT", deflateSync(pixels)),
    pngChunk("IEND", Buffer.alloc(0))
  ]);
  await writeFile(join(input, "sample.png"), png);
  await run();
  assert.deepEqual(await readdir(output), ["sample-CUTOUT.png"]);
  const cutout = await readFile(join(output, "sample-CUTOUT.png"));
  assert.equal(cutout.subarray(0, 8).toString("hex"), "89504e470d0a1a0a");
  assert.equal(cutout.readUInt32BE(16), size);
  assert.equal(cutout.readUInt32BE(20), size);
  assert.match(await run(), /SKIP existing sample-CUTOUT\.png/);
  assert.deepEqual(await readFile(join(output, "sample-CUTOUT.png")), cutout);
  console.log("PASS: batch PNG export, naming, and no-overwrite behavior");
} finally {
  await rm(temp, { recursive: true, force: true });
}
