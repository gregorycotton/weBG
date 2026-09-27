import assert from "node:assert/strict";
import { test } from "node:test";
import { imageDimensions } from "../src/image.ts";

test("PNG, progressive JPEG, and WebP headers expose dimensions", () => {
  const png = new Uint8Array(24);
  png.set([137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82]);
  new DataView(png.buffer).setUint32(16, 6000);
  new DataView(png.buffer).setUint32(20, 4000);
  assert.deepEqual(imageDimensions(png), { width: 6000, height: 4000 });

  const jpeg = new Uint8Array([255, 216, 255, 225, 0, 4, 0, 0, 255, 194, 0, 8, 8, 15, 160, 23, 112, 3]);
  assert.deepEqual(imageDimensions(jpeg), { width: 6000, height: 4000 });

  const webp = (fourcc, data) => {
    const bytes = new Uint8Array(20 + data.length + (data.length & 1));
    bytes.set([82, 73, 70, 70, 0, 0, 0, 0, 87, 69, 66, 80], 0);
    new DataView(bytes.buffer).setUint32(4, bytes.length - 8, true);
    bytes.set([...fourcc].map(char => char.charCodeAt(0)), 12);
    new DataView(bytes.buffer).setUint32(16, data.length, true);
    bytes.set(data, 20);
    return bytes;
  };
  assert.deepEqual(imageDimensions(webp("VP8X", [0, 0, 0, 0, 111, 23, 0, 159, 15, 0])), { width: 6000, height: 4000 });
  assert.deepEqual(imageDimensions(webp("VP8L", [47, 111, 7, 231, 3])), { width: 1904, height: 3997 });
  assert.deepEqual(imageDimensions(webp("VP8 ", [0, 0, 0, 157, 1, 42, 112, 23, 160, 15])), { width: 6000, height: 4000 });
});

test("truncated and malformed headers are rejected", () => {
  for (const bytes of [new Uint8Array(), new Uint8Array([255, 216, 255, 218]), new Uint8Array([82, 73, 70, 70]), new Uint8Array(24)]) {
    assert.throws(() => imageDimensions(bytes), /Invalid or truncated/);
  }
});
