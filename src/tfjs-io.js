const fs = require("fs");
const path = require("path");

function arrayBufferToBuffer(arrayBuffer) {
  return Buffer.from(new Uint8Array(arrayBuffer));
}

function bufferToArrayBuffer(buffer) {
  return buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
}

async function saveLayersModel(tf, model, modelDir) {
  fs.mkdirSync(modelDir, { recursive: true });

  let artifacts = null;
  await model.save({
    save: async (modelArtifacts) => {
      artifacts = modelArtifacts;
      return {
        modelArtifactsInfo: {
          dateSaved: new Date(),
          modelTopologyType: "JSON",
          modelTopologyBytes: Buffer.byteLength(JSON.stringify(modelArtifacts.modelTopology)),
          weightDataBytes: modelArtifacts.weightData ? modelArtifacts.weightData.byteLength : 0
        }
      };
    }
  });

  if (!artifacts) throw new Error("TensorFlow.js did not provide model artifacts.");

  const weightFileName = "group1-shard1of1.bin";
  const modelJson = {
    format: "layers-model",
    generatedBy: `TensorFlow.js v${tf.version.tfjs}`,
    convertedBy: null,
    modelTopology: artifacts.modelTopology,
    weightsManifest: [
      {
        paths: [weightFileName],
        weights: artifacts.weightSpecs
      }
    ]
  };

  fs.writeFileSync(path.join(modelDir, "model.json"), JSON.stringify(modelJson, null, 2));
  fs.writeFileSync(path.join(modelDir, weightFileName), arrayBufferToBuffer(artifacts.weightData));
}

async function loadLayersModel(tf, modelDir) {
  const modelJsonPath = path.join(modelDir, "model.json");
  const modelJson = JSON.parse(fs.readFileSync(modelJsonPath, "utf8"));
  const manifest = modelJson.weightsManifest || [];
  const weightBuffers = [];
  const weightSpecs = [];

  for (const group of manifest) {
    weightSpecs.push(...group.weights);
    for (const relPath of group.paths) {
      weightBuffers.push(fs.readFileSync(path.join(modelDir, relPath)));
    }
  }

  const weightData = bufferToArrayBuffer(Buffer.concat(weightBuffers));
  return tf.loadLayersModel(
    tf.io.fromMemory({
      modelTopology: modelJson.modelTopology,
      weightSpecs,
      weightData
    })
  );
}

module.exports = { loadLayersModel, saveLayersModel };

