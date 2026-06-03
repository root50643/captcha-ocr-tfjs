function loadTensorFlow() {
  try {
    return require("@tensorflow/tfjs");
  } catch {
    return require("../vendor/tf.min.js");
  }
}

module.exports = loadTensorFlow();
