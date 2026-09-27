export interface ImageDimensions { width: number; height: number }

export function imageDimensions(bytes: Uint8Array): ImageDimensions {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const invalid = (): never => { throw new Error("Invalid or truncated JPEG, PNG, or WebP image header."); };
  const dimensions = (width: number, height: number) => {
    if (!width || !height) invalid();
    return { width, height };
  };

  if (bytes.length >= 8 && [137, 80, 78, 71, 13, 10, 26, 10].every((value, i) => bytes[i] === value)) {
    if (bytes.length < 24 || view.getUint32(8) !== 13 || view.getUint32(12) !== 0x49484452) invalid();
    return dimensions(view.getUint32(16), view.getUint32(20));
  }

  if (bytes[0] === 0xff && bytes[1] === 0xd8) {
    let offset = 2;
    while (offset < bytes.length) {
      if (bytes[offset++] !== 0xff) invalid();
      while (bytes[offset] === 0xff) offset++;
      const marker = bytes[offset++];
      if (marker === undefined || marker === 0 || marker === 0xda || marker === 0xd9) invalid();
      if (marker === 0xd8 || marker === 0x01 || marker >= 0xd0 && marker <= 0xd7) continue;
      if (offset + 2 > bytes.length) invalid();
      const length = view.getUint16(offset);
      if (length < 2 || offset + length > bytes.length) invalid();
      if ([0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf].includes(marker)) {
        if (length < 8) invalid();
        return dimensions(view.getUint16(offset + 5), view.getUint16(offset + 3));
      }
      offset += length;
    }
    return invalid();
  }

  if (bytes.length >= 12 && view.getUint32(0) === 0x52494646 && view.getUint32(8) === 0x57454250) {
    const end = Math.min(bytes.length, view.getUint32(4, true) + 8);
    for (let offset = 12; offset + 8 <= end;) {
      const length = view.getUint32(offset + 4, true);
      const data = offset + 8;
      if (data + length > end) invalid();
      const type = view.getUint32(offset);
      if (type === 0x56503858) { // VP8X
        if (length < 10) invalid();
        return dimensions(1 + bytes[data + 4] + (bytes[data + 5] << 8) + (bytes[data + 6] << 16), 1 + bytes[data + 7] + (bytes[data + 8] << 8) + (bytes[data + 9] << 16));
      }
      if (type === 0x5650384c) { // VP8L
        if (length < 5 || bytes[data] !== 0x2f) invalid();
        return dimensions(1 + bytes[data + 1] + ((bytes[data + 2] & 0x3f) << 8), 1 + (bytes[data + 2] >> 6) + (bytes[data + 3] << 2) + ((bytes[data + 4] & 0x0f) << 10));
      }
      if (type === 0x56503820) { // VP8
        if (length < 10 || bytes[data + 3] !== 0x9d || bytes[data + 4] !== 0x01 || bytes[data + 5] !== 0x2a) invalid();
        return dimensions(view.getUint16(data + 6, true) & 0x3fff, view.getUint16(data + 8, true) & 0x3fff);
      }
      offset = data + length + (length & 1);
    }
    return invalid();
  }

  return invalid();
}
