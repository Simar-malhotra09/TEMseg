export interface LabeledMask {
  width: number;
  height: number;
  labels: Uint16Array;
}

export function parseNumpyLabels(buffer: ArrayBuffer): LabeledMask {
  const bytes = new Uint8Array(buffer);
  const magic = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59];
  if (magic.some((value, index) => bytes[index] !== value)) {
    throw new Error("Instance labels are not a valid NumPy file.");
  }

  const majorVersion = bytes[6];
  const headerLengthOffset = majorVersion === 1 ? 8 : 8;
  const headerLengthBytes = majorVersion === 1 ? 2 : 4;
  if (majorVersion < 1 || majorVersion > 3) {
    throw new Error(`Unsupported NumPy file version: ${majorVersion}.`);
  }

  const view = new DataView(buffer);
  const headerLength = headerLengthBytes === 2
    ? view.getUint16(headerLengthOffset, true)
    : view.getUint32(headerLengthOffset, true);
  const dataOffset = headerLengthOffset + headerLengthBytes + headerLength;
  const header = new TextDecoder().decode(bytes.subarray(headerLengthOffset + headerLengthBytes, dataOffset));
  const dtype = header.match(/'descr'\s*:\s*'([^']+)'/)?.[1];
  const shape = header.match(/'shape'\s*:\s*\((\d+)\s*,\s*(\d+)\s*\)/);
  if (dtype !== "<u2" && dtype !== "=u2" && dtype !== "|u2") {
    throw new Error(`Unsupported instance label dtype: ${dtype ?? "missing"}.`);
  }
  if (!shape || header.includes("'fortran_order': True")) {
    throw new Error("Instance labels must be a C-order, two-dimensional NumPy array.");
  }

  const height = Number(shape[1]);
  const width = Number(shape[2]);
  const pixelCount = width * height;
  if (dataOffset + pixelCount * 2 !== buffer.byteLength) {
    throw new Error("Instance label dimensions do not match the NumPy data size.");
  }

  const labels = new Uint16Array(pixelCount);
  for (let index = 0; index < pixelCount; index += 1) {
    labels[index] = view.getUint16(dataOffset + index * 2, true);
  }
  return { width, height, labels };
}
