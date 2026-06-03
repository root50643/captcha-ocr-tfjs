const fs = require("fs");
const path = require("path");
const { requirePackage } = require("./dependencies");

const { PNG } = requirePackage("pngjs");
const jpeg = requirePackage("jpeg-js");

const IMAGE_EXTENSIONS = new Set([".png", ".jpg", ".jpeg"]);

const DEFAULT_PREPROCESSING = {
  imageSize: 32,
  padding: 4,
  adaptiveBlockSize: 21,
  adaptiveC: 7,
  minComponentArea: 3,
  segmentCount: 5
};

function isImageFile(filePath) {
  return IMAGE_EXTENSIONS.has(path.extname(filePath).toLowerCase());
}

function walkImageFiles(dir) {
  if (!fs.existsSync(dir)) return [];
  const out = [];
  const stack = [dir];
  while (stack.length) {
    const current = stack.pop();
    for (const entry of fs.readdirSync(current, { withFileTypes: true })) {
      const full = path.join(current, entry.name);
      if (entry.isDirectory()) {
        stack.push(full);
      } else if (entry.isFile() && isImageFile(full)) {
        out.push(full);
      }
    }
  }
  return out.sort((a, b) => a.localeCompare(b));
}

function ensureDir(dir) {
  fs.mkdirSync(dir, { recursive: true });
}

function decodeImage(filePath) {
  const ext = path.extname(filePath).toLowerCase();
  const buffer = fs.readFileSync(filePath);
  const isPng =
    buffer.length >= 8 &&
    buffer[0] === 0x89 &&
    buffer[1] === 0x50 &&
    buffer[2] === 0x4e &&
    buffer[3] === 0x47;
  const isJpeg = buffer.length >= 2 && buffer[0] === 0xff && buffer[1] === 0xd8;

  if (isPng || ext === ".png") {
    const png = PNG.sync.read(buffer);
    return {
      width: png.width,
      height: png.height,
      data: png.data,
      sourcePath: filePath
    };
  }

  if (isJpeg || ext === ".jpg" || ext === ".jpeg") {
    const decoded = jpeg.decode(buffer, { useTArray: true });
    return {
      width: decoded.width,
      height: decoded.height,
      data: decoded.data,
      sourcePath: filePath
    };
  }

  throw new Error(`Unsupported image extension: ${filePath}`);
}

function rgbaToGray(image) {
  const { width, height, data } = image;
  const gray = new Uint8ClampedArray(width * height);

  for (let i = 0, p = 0; i < gray.length; i++, p += 4) {
    const alpha = data[p + 3] == null ? 255 : data[p + 3];
    const invAlpha = 255 - alpha;
    const r = (data[p] * alpha + 255 * invAlpha) / 255;
    const g = (data[p + 1] * alpha + 255 * invAlpha) / 255;
    const b = (data[p + 2] * alpha + 255 * invAlpha) / 255;
    gray[i] = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
  }

  return gray;
}

function normalizeBlockSize(blockSize) {
  const value = Math.max(3, Math.round(blockSize));
  return value % 2 === 0 ? value + 1 : value;
}

function adaptiveThreshold(gray, width, height, options = {}) {
  const blockSize = normalizeBlockSize(
    options.adaptiveBlockSize ?? DEFAULT_PREPROCESSING.adaptiveBlockSize
  );
  const c = options.adaptiveC ?? DEFAULT_PREPROCESSING.adaptiveC;
  const radius = Math.floor(blockSize / 2);
  const integralWidth = width + 1;
  const integral = new Float64Array((width + 1) * (height + 1));

  for (let y = 1; y <= height; y++) {
    let rowSum = 0;
    for (let x = 1; x <= width; x++) {
      rowSum += gray[(y - 1) * width + (x - 1)];
      integral[y * integralWidth + x] =
        integral[(y - 1) * integralWidth + x] + rowSum;
    }
  }

  const binary = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) {
    const y0 = Math.max(0, y - radius);
    const y1 = Math.min(height - 1, y + radius);
    for (let x = 0; x < width; x++) {
      const x0 = Math.max(0, x - radius);
      const x1 = Math.min(width - 1, x + radius);
      const a = y0 * integralWidth + x0;
      const b = y0 * integralWidth + (x1 + 1);
      const cIdx = (y1 + 1) * integralWidth + x0;
      const d = (y1 + 1) * integralWidth + (x1 + 1);
      const area = (x1 - x0 + 1) * (y1 - y0 + 1);
      const mean = (integral[d] - integral[b] - integral[cIdx] + integral[a]) / area;
      const idx = y * width + x;
      binary[idx] = gray[idx] < mean - c ? 1 : 0;
    }
  }

  return binary;
}

function removeSmallComponents(binary, width, height, minArea = 3) {
  if (minArea <= 1) return binary;

  const visited = new Uint8Array(binary.length);
  const cleaned = new Uint8Array(binary.length);
  const queue = [];
  const component = [];
  const neighbors = [
    [-1, -1],
    [0, -1],
    [1, -1],
    [-1, 0],
    [1, 0],
    [-1, 1],
    [0, 1],
    [1, 1]
  ];

  for (let start = 0; start < binary.length; start++) {
    if (!binary[start] || visited[start]) continue;

    queue.length = 0;
    component.length = 0;
    queue.push(start);
    visited[start] = 1;

    for (let head = 0; head < queue.length; head++) {
      const idx = queue[head];
      component.push(idx);
      const x = idx % width;
      const y = Math.floor(idx / width);

      for (const [dx, dy] of neighbors) {
        const nx = x + dx;
        const ny = y + dy;
        if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
        const next = ny * width + nx;
        if (binary[next] && !visited[next]) {
          visited[next] = 1;
          queue.push(next);
        }
      }
    }

    if (component.length >= minArea) {
      for (const idx of component) cleaned[idx] = 1;
    }
  }

  return cleaned;
}

function binarizeDecodedImage(image, options = {}) {
  const gray = rgbaToGray(image);
  const binary = adaptiveThreshold(gray, image.width, image.height, options);
  return removeSmallComponents(
    binary,
    image.width,
    image.height,
    options.minComponentArea ?? DEFAULT_PREPROCESSING.minComponentArea
  );
}

function findBoundingBox(binary, width, height, bounds = null, margin = 0) {
  const xStart = bounds ? Math.max(0, bounds.x) : 0;
  const yStart = bounds ? Math.max(0, bounds.y) : 0;
  const xEnd = bounds ? Math.min(width, bounds.x + bounds.width) : width;
  const yEnd = bounds ? Math.min(height, bounds.y + bounds.height) : height;

  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;

  for (let y = yStart; y < yEnd; y++) {
    for (let x = xStart; x < xEnd; x++) {
      if (!binary[y * width + x]) continue;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
  }

  if (maxX < minX || maxY < minY) return null;

  minX = Math.max(0, minX - margin);
  minY = Math.max(0, minY - margin);
  maxX = Math.min(width - 1, maxX + margin);
  maxY = Math.min(height - 1, maxY + margin);

  return {
    x: minX,
    y: minY,
    width: maxX - minX + 1,
    height: maxY - minY + 1
  };
}

function cropBinary(binary, width, height, bbox) {
  const out = new Uint8Array(bbox.width * bbox.height);
  for (let y = 0; y < bbox.height; y++) {
    const srcStart = (bbox.y + y) * width + bbox.x;
    const dstStart = y * bbox.width;
    for (let x = 0; x < bbox.width; x++) {
      out[dstStart + x] = binary[srcStart + x] ? 1 : 0;
    }
  }
  return out;
}

function normalizeBinary(crop, cropWidth, cropHeight, options = {}) {
  const imageSize = options.imageSize ?? DEFAULT_PREPROCESSING.imageSize;
  const padding = options.padding ?? DEFAULT_PREPROCESSING.padding;
  const maxContent = Math.max(1, imageSize - padding * 2);
  const output = new Float32Array(imageSize * imageSize);

  if (cropWidth <= 0 || cropHeight <= 0) return output;

  const scale = Math.min(maxContent / cropWidth, maxContent / cropHeight);
  const resizedWidth = Math.max(1, Math.round(cropWidth * scale));
  const resizedHeight = Math.max(1, Math.round(cropHeight * scale));
  const offsetX = Math.floor((imageSize - resizedWidth) / 2);
  const offsetY = Math.floor((imageSize - resizedHeight) / 2);

  for (let y = 0; y < resizedHeight; y++) {
    const srcY = Math.min(cropHeight - 1, Math.floor(((y + 0.5) * cropHeight) / resizedHeight));
    for (let x = 0; x < resizedWidth; x++) {
      const srcX = Math.min(cropWidth - 1, Math.floor(((x + 0.5) * cropWidth) / resizedWidth));
      const value = crop[srcY * cropWidth + srcX] ? 1 : 0;
      output[(offsetY + y) * imageSize + (offsetX + x)] = value;
    }
  }

  return output;
}

function preprocessDigitFile(filePath, options = {}) {
  const image = decodeImage(filePath);
  const binary = binarizeDecodedImage(image, options);
  const bbox =
    findBoundingBox(binary, image.width, image.height, null, 1) || {
      x: 0,
      y: 0,
      width: image.width,
      height: image.height
    };
  const crop = cropBinary(binary, image.width, image.height, bbox);
  const input = normalizeBinary(crop, bbox.width, bbox.height, options);

  return {
    input,
    bbox,
    width: image.width,
    height: image.height,
    binary
  };
}

function projectionSegments(binary, width, height, count) {
  const projection = new Int32Array(width);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (binary[y * width + x]) projection[x]++;
    }
  }

  const groups = [];
  let start = -1;
  for (let x = 0; x < width; x++) {
    if (projection[x] > 0 && start < 0) {
      start = x;
    } else if (projection[x] === 0 && start >= 0) {
      groups.push({ x0: start, x1: x - 1 });
      start = -1;
    }
  }
  if (start >= 0) groups.push({ x0: start, x1: width - 1 });

  let merged = [];
  for (const group of groups) {
    const last = merged[merged.length - 1];
    if (last && group.x0 - last.x1 <= 2) {
      last.x1 = group.x1;
    } else {
      merged.push({ ...group });
    }
  }

  while (merged.length > count) {
    let bestGapIndex = -1;
    let bestGap = Infinity;
    for (let i = 0; i < merged.length - 1; i++) {
      const gap = merged[i + 1].x0 - merged[i].x1;
      if (gap < bestGap) {
        bestGap = gap;
        bestGapIndex = i;
      }
    }
    if (bestGapIndex < 0) break;
    merged.splice(bestGapIndex, 2, {
      x0: merged[bestGapIndex].x0,
      x1: merged[bestGapIndex + 1].x1
    });
  }

  while (merged.length < count) {
    let widestIndex = -1;
    let widestWidth = 0;
    for (let i = 0; i < merged.length; i++) {
      const groupWidth = merged[i].x1 - merged[i].x0 + 1;
      if (groupWidth > widestWidth) {
        widestWidth = groupWidth;
        widestIndex = i;
      }
    }

    if (widestIndex < 0 || widestWidth < 8) break;
    const group = merged[widestIndex];
    const minSide = Math.max(3, Math.floor(widestWidth * 0.2));
    let splitX = -1;
    let bestScore = Infinity;
    const center = (group.x0 + group.x1) / 2;

    for (let x = group.x0 + minSide; x <= group.x1 - minSide; x++) {
      const score =
        (projection[x - 1] || 0) + projection[x] + (projection[x + 1] || 0) +
        Math.abs(x - center) * 0.04;
      if (score < bestScore) {
        bestScore = score;
        splitX = x;
      }
    }

    if (splitX <= group.x0 || splitX >= group.x1) break;
    merged.splice(
      widestIndex,
      1,
      { x0: group.x0, x1: splitX },
      { x0: splitX + 1, x1: group.x1 }
    );
  }

  const bboxes = merged
    .map((group) =>
      findBoundingBox(
        binary,
        width,
        height,
        { x: group.x0, y: 0, width: group.x1 - group.x0 + 1, height },
        1
      )
    )
    .filter(Boolean)
    .filter((bbox) => bbox.width * bbox.height >= 8);

  if (bboxes.length !== count) return null;
  return bboxes.sort((a, b) => a.x - b.x);
}

function slotSegments(binary, width, height, count) {
  const out = [];
  const slotWidth = width / count;
  for (let i = 0; i < count; i++) {
    const x0 = Math.floor(i * slotWidth);
    const x1 = Math.floor((i + 1) * slotWidth);
    const bbox =
      findBoundingBox(
        binary,
        width,
        height,
        { x: x0, y: 0, width: Math.max(1, x1 - x0), height },
        1
      ) || {
        x: x0,
        y: 0,
        width: Math.max(1, x1 - x0),
        height
      };
    out.push(bbox);
  }
  return out;
}

function segmentRawImageFile(filePath, options = {}) {
  const image = decodeImage(filePath);
  const binary = binarizeDecodedImage(image, options);
  const count = options.segmentCount ?? DEFAULT_PREPROCESSING.segmentCount;
  const projected = projectionSegments(binary, image.width, image.height, count);
  const bboxes = projected || slotSegments(binary, image.width, image.height, count);
  const method = projected ? "projection" : "slots";

  const segments = bboxes.map((bbox, index) => {
    const crop = cropBinary(binary, image.width, image.height, bbox);
    return {
      index,
      bbox,
      input: normalizeBinary(crop, bbox.width, bbox.height, options),
      crop,
      cropWidth: bbox.width,
      cropHeight: bbox.height
    };
  });

  return {
    filePath,
    width: image.width,
    height: image.height,
    binary,
    segments,
    method
  };
}

function writeBinaryPng(values, width, height, outPath) {
  ensureDir(path.dirname(outPath));
  const png = new PNG({ width, height });
  for (let i = 0, p = 0; i < values.length; i++, p += 4) {
    const foreground = values[i] > 0.5;
    const shade = foreground ? 0 : 255;
    png.data[p] = shade;
    png.data[p + 1] = shade;
    png.data[p + 2] = shade;
    png.data[p + 3] = 255;
  }
  fs.writeFileSync(outPath, PNG.sync.write(png));
}

function writeNormalizedPng(values, imageSize, outPath) {
  writeBinaryPng(values, imageSize, imageSize, outPath);
}

module.exports = {
  DEFAULT_PREPROCESSING,
  IMAGE_EXTENSIONS,
  binarizeDecodedImage,
  cropBinary,
  decodeImage,
  ensureDir,
  findBoundingBox,
  isImageFile,
  normalizeBinary,
  preprocessDigitFile,
  segmentRawImageFile,
  walkImageFiles,
  writeBinaryPng,
  writeNormalizedPng
};
