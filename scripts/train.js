const fs = require("fs");
const path = require("path");
const tf = require("../src/tf");
const { saveLayersModel } = require("../src/tfjs-io");
const {
  DEFAULT_PREPROCESSING,
  ensureDir,
  preprocessDigitFile,
  walkImageFiles
} = require("../src/image-utils");

const PROJECT_ROOT = path.resolve(__dirname, "..");
const LABELS = ["1", "2", "3", "4", "5", "6", "7", "8", "9"];

function parseArgs(argv) {
  const out = {
    dataDir: path.join(PROJECT_ROOT, "data", "segmented"),
    modelDir: path.join(PROJECT_ROOT, "model"),
    reportDir: path.join(PROJECT_ROOT, "reports", "training"),
    epochs: 20,
    batchSize: 32,
    imageSize: DEFAULT_PREPROCESSING.imageSize,
    seed: 42,
    validationRatio: 0.15,
    testRatio: 0.15
  };

  for (const arg of argv.slice(2)) {
    const [rawKey, rawValue] = arg.replace(/^--/, "").split("=");
    if (!rawKey || rawValue == null) continue;
    const key = rawKey.replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
    if (["epochs", "batchSize", "imageSize", "seed"].includes(key)) {
      out[key] = Number(rawValue);
    } else if (["validationRatio", "testRatio"].includes(key)) {
      out[key] = Number(rawValue);
    } else if (["dataDir", "modelDir", "reportDir"].includes(key)) {
      out[key] = path.resolve(PROJECT_ROOT, rawValue);
    }
  }

  return out;
}

function mulberry32(seed) {
  let state = seed >>> 0;
  return function next() {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffleInPlace(items, random) {
  for (let i = items.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [items[i], items[j]] = [items[j], items[i]];
  }
}

function loadSplitSamples(config) {
  const random = mulberry32(config.seed);
  const splits = { train: [], validation: [], test: [] };
  const counts = {};

  for (const label of LABELS) {
    const files = walkImageFiles(path.join(config.dataDir, label));
    if (!files.length) throw new Error(`No images found for label ${label}`);
    shuffleInPlace(files, random);
    counts[label] = files.length;

    const testCount = Math.max(1, Math.round(files.length * config.testRatio));
    const validationCount = Math.max(1, Math.round(files.length * config.validationRatio));
    const labelIndex = LABELS.indexOf(label);

    files.forEach((filePath, index) => {
      const sample = { filePath, label, labelIndex };
      if (index < testCount) {
        splits.test.push(sample);
      } else if (index < testCount + validationCount) {
        splits.validation.push(sample);
      } else {
        splits.train.push(sample);
      }
    });
  }

  shuffleInPlace(splits.train, random);
  shuffleInPlace(splits.validation, random);
  shuffleInPlace(splits.test, random);

  return { splits, counts };
}

function tensorFromSamples(tfInstance, samples, preprocessing) {
  const imageSize = preprocessing.imageSize;
  const pixelsPerImage = imageSize * imageSize;
  const xsData = new Float32Array(samples.length * pixelsPerImage);
  const labelIndexes = new Int32Array(samples.length);

  samples.forEach((sample, index) => {
    const processed = preprocessDigitFile(sample.filePath, preprocessing);
    xsData.set(processed.input, index * pixelsPerImage);
    labelIndexes[index] = sample.labelIndex;
  });

  const xs = tfInstance.tensor4d(xsData, [samples.length, imageSize, imageSize, 1]);
  const labels = tfInstance.tensor1d(labelIndexes, "int32");
  const ys = tfInstance.oneHot(labels, LABELS.length);
  labels.dispose();

  return { xs, ys, labelIndexes: Array.from(labelIndexes) };
}

function createModel(tfInstance, imageSize) {
  const model = tfInstance.sequential();
  model.add(tfInstance.layers.flatten({ inputShape: [imageSize, imageSize, 1] }));
  model.add(tfInstance.layers.dense({ units: 128, activation: "relu" }));
  model.add(tfInstance.layers.dropout({ rate: 0.25 }));
  model.add(tfInstance.layers.dense({ units: 64, activation: "relu" }));
  model.add(tfInstance.layers.dense({ units: LABELS.length, activation: "softmax" }));

  model.compile({
    optimizer: tfInstance.train.adam(0.001),
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"]
  });

  return model;
}

function computeConfusionMatrix(actual, predicted) {
  const matrix = Array.from({ length: LABELS.length }, () =>
    Array.from({ length: LABELS.length }, () => 0)
  );
  for (let i = 0; i < actual.length; i++) {
    matrix[actual[i]][predicted[i]] += 1;
  }
  return matrix;
}

function matrixToCsv(matrix) {
  const lines = [];
  lines.push(["actual\\predicted", ...LABELS].join(","));
  matrix.forEach((row, index) => {
    lines.push([LABELS[index], ...row].join(","));
  });
  return `${lines.join("\n")}\n`;
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function writeTrainingReport(outPath, metrics, matrix) {
  const maxCell = Math.max(1, ...matrix.flat());
  const rows = matrix
    .map((row, rowIndex) => {
      const cells = row
        .map((value, colIndex) => {
          const intensity = value / maxCell;
          const color =
            rowIndex === colIndex
              ? `rgba(40, 150, 95, ${0.18 + intensity * 0.65})`
              : `rgba(190, 65, 65, ${0.08 + intensity * 0.55})`;
          return `<td style="background:${color}">${value}</td>`;
        })
        .join("");
      return `<tr><th>${LABELS[rowIndex]}</th>${cells}</tr>`;
    })
    .join("\n");

  const counts = Object.entries(metrics.classCounts)
    .map(([label, count]) => `<li>${escapeHtml(label)}: ${count}</li>`)
    .join("");

  const html = `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Digit model training report</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; color: #1f2933; }
    table { border-collapse: collapse; margin-top: 12px; }
    th, td { border: 1px solid #ccd2da; padding: 8px 12px; text-align: center; }
    th { background: #f3f5f7; }
    .metrics { display: flex; gap: 16px; flex-wrap: wrap; margin: 16px 0; }
    .metric { border: 1px solid #d6dbe1; border-radius: 6px; padding: 10px 12px; min-width: 150px; }
    .metric b { display: block; font-size: 22px; margin-top: 4px; }
  </style>
</head>
<body>
  <h1>Digit model training report</h1>
  <p>Generated at ${escapeHtml(metrics.generatedAt)}</p>
  <div class="metrics">
    <div class="metric">Test accuracy<b>${(metrics.testAccuracy * 100).toFixed(2)}%</b></div>
    <div class="metric">Validation accuracy<b>${(metrics.finalValidationAccuracy * 100).toFixed(2)}%</b></div>
    <div class="metric">Train samples<b>${metrics.splitSizes.train}</b></div>
    <div class="metric">Test samples<b>${metrics.splitSizes.test}</b></div>
  </div>
  <h2>Confusion matrix</h2>
  <table>
    <thead><tr><th>Actual / Predicted</th>${LABELS.map((label) => `<th>${label}</th>`).join("")}</tr></thead>
    <tbody>${rows}</tbody>
  </table>
  <h2>Class counts</h2>
  <ul>${counts}</ul>
</body>
</html>
`;

  fs.writeFileSync(outPath, html);
}

async function main() {
  const config = parseArgs(process.argv);
  const preprocessing = {
    ...DEFAULT_PREPROCESSING,
    imageSize: config.imageSize
  };

  await tf.ready();
  console.log(`TensorFlow.js ${tf.version.tfjs}, backend=${tf.getBackend()}`);

  const { splits, counts } = loadSplitSamples(config);
  console.log(
    `Samples: train=${splits.train.length}, validation=${splits.validation.length}, test=${splits.test.length}`
  );

  const train = tensorFromSamples(tf, splits.train, preprocessing);
  const validation = tensorFromSamples(tf, splits.validation, preprocessing);
  const test = tensorFromSamples(tf, splits.test, preprocessing);

  const model = createModel(tf, preprocessing.imageSize);
  model.summary();

  const history = await model.fit(train.xs, train.ys, {
    epochs: config.epochs,
    batchSize: config.batchSize,
    validationData: [validation.xs, validation.ys],
    shuffle: true,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        const acc = logs.acc ?? logs.accuracy ?? 0;
        const valAcc = logs.val_acc ?? logs.val_accuracy ?? 0;
        console.log(
          `epoch ${String(epoch + 1).padStart(2, "0")}/${config.epochs} ` +
            `loss=${logs.loss.toFixed(4)} acc=${acc.toFixed(4)} ` +
            `val_loss=${logs.val_loss.toFixed(4)} val_acc=${valAcc.toFixed(4)}`
        );
      }
    }
  });

  const evalResult = model.evaluate(test.xs, test.ys);
  const testLoss = (await evalResult[0].data())[0];
  const testAccuracy = (await evalResult[1].data())[0];
  evalResult.forEach((tensor) => tensor.dispose());

  const predictionTensor = model.predict(test.xs).argMax(-1);
  const predicted = Array.from(await predictionTensor.data());
  predictionTensor.dispose();

  const matrix = computeConfusionMatrix(test.labelIndexes, predicted);
  const last = history.history;
  const finalValidationAccuracy =
    (last.val_acc || last.val_accuracy || [0]).slice(-1)[0] ?? 0;
  const finalTrainAccuracy = (last.acc || last.accuracy || [0]).slice(-1)[0] ?? 0;

  const modelDir = config.modelDir;
  const trainingDir = config.reportDir;
  ensureDir(modelDir);
  ensureDir(trainingDir);

  await saveLayersModel(tf, model, modelDir);
  fs.writeFileSync(path.join(modelDir, "labels.json"), JSON.stringify(LABELS, null, 2));
  fs.writeFileSync(
    path.join(modelDir, "preprocessing.json"),
    JSON.stringify(preprocessing, null, 2)
  );

  const metrics = {
    generatedAt: new Date().toISOString(),
    tfjsVersion: tf.version.tfjs,
    backend: tf.getBackend(),
    labels: LABELS,
    classCounts: counts,
    splitSizes: {
      train: splits.train.length,
      validation: splits.validation.length,
      test: splits.test.length
    },
    epochs: config.epochs,
    batchSize: config.batchSize,
    finalTrainAccuracy,
    finalValidationAccuracy,
    testLoss,
    testAccuracy,
    confusionMatrix: matrix
  };

  fs.writeFileSync(path.join(trainingDir, "metrics.json"), JSON.stringify(metrics, null, 2));
  fs.writeFileSync(path.join(trainingDir, "confusion-matrix.csv"), matrixToCsv(matrix));
  writeTrainingReport(path.join(trainingDir, "training-report.html"), metrics, matrix);

  train.xs.dispose();
  train.ys.dispose();
  validation.xs.dispose();
  validation.ys.dispose();
  test.xs.dispose();
  test.ys.dispose();
  model.dispose();

  console.log(`Test accuracy: ${(testAccuracy * 100).toFixed(2)}%`);
  console.log(`Model written to ${modelDir}`);
  console.log(`Confusion matrix written to ${path.join(trainingDir, "confusion-matrix.csv")}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
