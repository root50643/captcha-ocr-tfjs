const fs = require("fs");
const path = require("path");
const tf = require("../src/tf");
const { loadLayersModel } = require("../src/tfjs-io");
const {
  DEFAULT_PREPROCESSING,
  ensureDir,
  segmentRawImageFile,
  walkImageFiles,
  writeBinaryPng,
  writeNormalizedPng
} = require("../src/image-utils");

const PROJECT_ROOT = path.resolve(__dirname, "..");

function parseArgs(argv) {
  const out = {
    modelDir: path.join(PROJECT_ROOT, "model"),
    rawDir: path.join(PROJECT_ROOT, "data", "raw"),
    outputDir: path.join(PROJECT_ROOT, "reports", "raw-predictions")
  };

  for (const arg of argv.slice(2)) {
    const [rawKey, rawValue] = arg.replace(/^--/, "").split("=");
    if (!rawKey || rawValue == null) continue;
    const key = rawKey.replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
    if (["modelDir", "rawDir", "outputDir"].includes(key)) {
      out[key] = path.resolve(PROJECT_ROOT, rawValue);
    }
  }

  return out;
}

function safeBaseName(filePath) {
  return path.basename(filePath, path.extname(filePath)).replace(/[^a-zA-Z0-9_-]/g, "_");
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function relForHtml(fromFile, targetFile) {
  return path.relative(path.dirname(fromFile), targetFile).replace(/\\/g, "/");
}

function expectedFromFile(filePath) {
  const name = path.basename(filePath, path.extname(filePath));
  return /^[1-9]{5}$/.test(name) ? name : "";
}

function argMax(values) {
  let bestIndex = 0;
  let bestValue = values[0];
  for (let i = 1; i < values.length; i++) {
    if (values[i] > bestValue) {
      bestValue = values[i];
      bestIndex = i;
    }
  }
  return { index: bestIndex, value: bestValue };
}

function csvEscape(value) {
  const text = String(value ?? "");
  if (!/[",\n]/.test(text)) return text;
  return `"${text.replace(/"/g, '""')}"`;
}

function writeCsv(outPath, records) {
  const headers = [
    "file",
    "expected",
    "prediction",
    "correct",
    "segment_method",
    "digit_index",
    "digit",
    "confidence",
    "bbox_x",
    "bbox_y",
    "bbox_width",
    "bbox_height"
  ];
  const lines = [headers.join(",")];
  for (const record of records) {
    for (const digit of record.digits) {
      lines.push(
        [
          record.file,
          record.expected,
          record.prediction,
          record.correct,
          record.segmentMethod,
          digit.index,
          digit.label,
          digit.confidence.toFixed(6),
          digit.bbox.x,
          digit.bbox.y,
          digit.bbox.width,
          digit.bbox.height
        ]
          .map(csvEscape)
          .join(",")
      );
    }
  }
  fs.writeFileSync(outPath, `${lines.join("\n")}\n`);
}

function writeHtmlReport(outPath, records, summary) {
  const rows = records
    .map((record) => {
      const digitCells = record.digits
        .map(
          (digit) => `<div class="digit">
  <img src="${escapeHtml(relForHtml(outPath, digit.normalizedPath))}" alt="">
  <div class="digit-label">${escapeHtml(digit.label)}</div>
  <div class="confidence">${(digit.confidence * 100).toFixed(1)}%</div>
</div>`
        )
        .join("");
      const correctness = record.expected
        ? record.correct
          ? '<span class="ok">correct</span>'
          : '<span class="bad">mismatch</span>'
        : '<span class="muted">unknown</span>';

      return `<tr>
  <td class="file">${escapeHtml(record.file)}</td>
  <td><img class="raw" src="${escapeHtml(relForHtml(outPath, record.rawPath))}" alt=""></td>
  <td><img class="raw" src="${escapeHtml(relForHtml(outPath, record.binarizedPath))}" alt=""></td>
  <td class="prediction">${escapeHtml(record.prediction)}</td>
  <td>${escapeHtml(record.expected || "-")}</td>
  <td>${correctness}</td>
  <td><div class="digits">${digitCells}</div></td>
</tr>`;
    })
    .join("\n");

  const html = `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Raw digit predictions</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; color: #1f2933; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border-bottom: 1px solid #d7dde5; padding: 8px; text-align: left; vertical-align: middle; }
    th { background: #f3f5f7; position: sticky; top: 0; z-index: 1; }
    .raw { width: 250px; height: 50px; image-rendering: pixelated; border: 1px solid #d7dde5; }
    .digits { display: flex; gap: 6px; align-items: center; }
    .digit { width: 52px; text-align: center; }
    .digit img { width: 32px; height: 32px; image-rendering: pixelated; border: 1px solid #d7dde5; background: white; }
    .digit-label, .prediction { font-size: 24px; font-weight: 700; letter-spacing: 0; }
    .confidence { font-size: 11px; color: #657080; }
    .file { white-space: nowrap; font-family: Consolas, monospace; }
    .ok { color: #176b45; font-weight: 700; }
    .bad { color: #a33838; font-weight: 700; }
    .muted { color: #657080; }
    .metrics { display: flex; gap: 14px; flex-wrap: wrap; margin: 14px 0 18px; }
    .metric { border: 1px solid #d7dde5; border-radius: 6px; padding: 10px 12px; min-width: 150px; }
    .metric b { display: block; font-size: 22px; margin-top: 4px; }
  </style>
</head>
<body>
  <h1>Raw digit predictions</h1>
  <p>Generated at ${escapeHtml(summary.generatedAt)}</p>
  <div class="metrics">
    <div class="metric">Images<b>${summary.totalImages}</b></div>
    <div class="metric">Labeled images<b>${summary.labeledImages}</b></div>
    <div class="metric">Labeled accuracy<b>${summary.labeledImages ? `${(summary.labeledAccuracy * 100).toFixed(2)}%` : "-"}</b></div>
  </div>
  <table>
    <thead>
      <tr>
        <th>File</th>
        <th>Raw</th>
        <th>Adaptive binary</th>
        <th>Prediction</th>
        <th>Expected</th>
        <th>Status</th>
        <th>Digits</th>
      </tr>
    </thead>
    <tbody>${rows}</tbody>
  </table>
</body>
</html>
`;

  fs.writeFileSync(outPath, html);
}

async function main() {
  const config = parseArgs(process.argv);
  const labels = JSON.parse(fs.readFileSync(path.join(config.modelDir, "labels.json"), "utf8"));
  const preprocessing = {
    ...DEFAULT_PREPROCESSING,
    ...JSON.parse(fs.readFileSync(path.join(config.modelDir, "preprocessing.json"), "utf8"))
  };

  await tf.ready();
  const model = await loadLayersModel(tf, config.modelDir);

  const files = walkImageFiles(config.rawDir);
  if (!files.length) throw new Error(`No raw images found in ${config.rawDir}`);

  ensureDir(config.outputDir);
  const binarizedDir = path.join(config.outputDir, "binarized");
  const digitsDir = path.join(config.outputDir, "digits");
  ensureDir(binarizedDir);
  ensureDir(digitsDir);

  const segmentEntries = [];
  const records = [];

  for (const filePath of files) {
    const base = safeBaseName(filePath);
    const segmented = segmentRawImageFile(filePath, preprocessing);
    const binarizedPath = path.join(binarizedDir, `${base}.png`);
    writeBinaryPng(segmented.binary, segmented.width, segmented.height, binarizedPath);

    const record = {
      file: path.basename(filePath),
      rawPath: filePath,
      binarizedPath,
      expected: expectedFromFile(filePath),
      prediction: "",
      correct: "",
      segmentMethod: segmented.method,
      digits: []
    };

    for (const segment of segmented.segments) {
      const normalizedPath = path.join(digitsDir, `${base}_${segment.index}.png`);
      writeNormalizedPng(segment.input, preprocessing.imageSize, normalizedPath);

      const digit = {
        index: segment.index,
        bbox: segment.bbox,
        normalizedPath,
        label: "",
        confidence: 0
      };
      record.digits.push(digit);
      segmentEntries.push({ input: segment.input, digit });
    }

    records.push(record);
  }

  const imageSize = preprocessing.imageSize;
  const pixelsPerImage = imageSize * imageSize;
  const xsData = new Float32Array(segmentEntries.length * pixelsPerImage);
  segmentEntries.forEach((entry, index) => xsData.set(entry.input, index * pixelsPerImage));

  const xs = tf.tensor4d(xsData, [segmentEntries.length, imageSize, imageSize, 1]);
  const probabilities = await model.predict(xs).array();
  xs.dispose();

  probabilities.forEach((probs, index) => {
    const best = argMax(probs);
    const digit = segmentEntries[index].digit;
    digit.label = labels[best.index];
    digit.confidence = best.value;
    digit.probabilities = Object.fromEntries(labels.map((label, labelIndex) => [label, probs[labelIndex]]));
  });

  let labeledImages = 0;
  let labeledCorrect = 0;
  for (const record of records) {
    record.prediction = record.digits.map((digit) => digit.label).join("");
    if (record.expected) {
      labeledImages++;
      record.correct = record.prediction === record.expected;
      if (record.correct) labeledCorrect++;
    } else {
      record.correct = "";
    }
  }

  const summary = {
    generatedAt: new Date().toISOString(),
    totalImages: records.length,
    labeledImages,
    labeledCorrect,
    labeledAccuracy: labeledImages ? labeledCorrect / labeledImages : null,
    modelDir: config.modelDir,
    rawDir: config.rawDir
  };

  const jsonPath = path.join(config.outputDir, "predictions.json");
  const csvPath = path.join(config.outputDir, "predictions.csv");
  const htmlPath = path.join(config.outputDir, "report.html");

  fs.writeFileSync(jsonPath, JSON.stringify({ summary, records }, null, 2));
  writeCsv(csvPath, records);
  writeHtmlReport(htmlPath, records, summary);

  model.dispose();

  console.log(`Predicted ${records.length} raw images.`);
  if (labeledImages) {
    console.log(
      `Labeled raw accuracy: ${labeledCorrect}/${labeledImages} (${(summary.labeledAccuracy * 100).toFixed(2)}%)`
    );
  }
  console.log(`Report written to ${htmlPath}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
