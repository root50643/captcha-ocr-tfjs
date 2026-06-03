const fs = require("fs");
const path = require("path");
const tf = require("./tf");
const { loadLayersModel } = require("./tfjs-io");
const {
  DEFAULT_PREPROCESSING,
  segmentRawImageFile
} = require("./image-utils");

let cachedModel = null;

async function loadRecognizer(modelDir = path.join(__dirname, "..", "model")) {
  if (!cachedModel || cachedModel.modelDir !== modelDir) {
    await tf.ready();
    cachedModel = {
      modelDir,
      model: await loadLayersModel(tf, modelDir),
      labels: JSON.parse(fs.readFileSync(path.join(modelDir, "labels.json"), "utf8")),
      preprocessing: {
        ...DEFAULT_PREPROCESSING,
        ...JSON.parse(fs.readFileSync(path.join(modelDir, "preprocessing.json"), "utf8"))
      }
    };
  }
  return cachedModel;
}

function bestPrediction(values) {
  let index = 0;
  let confidence = values[0];
  for (let i = 1; i < values.length; i++) {
    if (values[i] > confidence) {
      index = i;
      confidence = values[i];
    }
  }
  return { index, confidence };
}

async function solveCaptchaFile(filePath, options = {}) {
  const recognizer = await loadRecognizer(options.modelDir);
  const segmented = segmentRawImageFile(filePath, recognizer.preprocessing);
  const imageSize = recognizer.preprocessing.imageSize;
  const pixelsPerImage = imageSize * imageSize;
  const xsData = new Float32Array(segmented.segments.length * pixelsPerImage);
  segmented.segments.forEach((segment, index) => xsData.set(segment.input, index * pixelsPerImage));

  const xs = tf.tensor4d(xsData, [segmented.segments.length, imageSize, imageSize, 1]);
  const probabilities = await recognizer.model.predict(xs).array();
  xs.dispose();

  const digits = probabilities.map((probs, index) => {
    const best = bestPrediction(probs);
    return {
      index,
      label: recognizer.labels[best.index],
      confidence: best.confidence,
      bbox: segmented.segments[index].bbox,
      probabilities: Object.fromEntries(
        recognizer.labels.map((label, labelIndex) => [label, probs[labelIndex]])
      )
    };
  });

  return {
    file: filePath,
    text: digits.map((digit) => digit.label).join(""),
    confidence: digits.reduce((sum, digit) => sum + digit.confidence, 0) / digits.length,
    segmentMethod: segmented.method,
    digits
  };
}

module.exports = {
  loadRecognizer,
  solveCaptchaFile
};
