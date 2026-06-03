const path = require("path");
const { solveCaptchaFile } = require("../src/ocr");

const PROJECT_ROOT = path.resolve(__dirname, "..");

function parseArgs(argv) {
  const out = {
    file: "",
    modelDir: path.join(PROJECT_ROOT, "model")
  };

  for (const arg of argv.slice(2)) {
    if (!arg.startsWith("--") && !out.file) {
      out.file = path.resolve(PROJECT_ROOT, arg);
      continue;
    }
    const [rawKey, rawValue] = arg.replace(/^--/, "").split("=");
    if (!rawKey || rawValue == null) continue;
    const key = rawKey.replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
    if (key === "file") out.file = path.resolve(PROJECT_ROOT, rawValue);
    if (key === "modelDir") out.modelDir = path.resolve(PROJECT_ROOT, rawValue);
  }

  return out;
}

async function main() {
  const config = parseArgs(process.argv);
  if (!config.file) {
    throw new Error("Usage: node scripts/solve-file.js data/raw/14662.png");
  }

  const result = await solveCaptchaFile(config.file, { modelDir: config.modelDir });
  console.log(JSON.stringify(result, null, 2));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
