const assert = require("assert");
const path = require("path");
const { solveCaptchaFile } = require("../src/ocr");

async function main() {
  const root = path.resolve(__dirname, "..");
  const cases = [
    ["14662.png", "14662"],
    ["99511.png", "99511"]
  ];

  for (const [fileName, expected] of cases) {
    const result = await solveCaptchaFile(path.join(root, "data", "raw", fileName), {
      modelDir: path.join(root, "model")
    });
    assert.strictEqual(result.text, expected, `${fileName} should solve as ${expected}`);
  }

  console.log("Smoke tests passed.");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
