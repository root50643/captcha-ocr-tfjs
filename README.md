# Captcha OCR TFJS

這是一個獨立的 TensorFlow.js 驗證碼辨識專案，包含：

- 訓練程式：使用 `data/segmented/1` 到 `data/segmented/9` 的切割數字樣本訓練模型。
- 樣本資料：保留已標記的單字樣本與五碼 RawData 範例。
- 推論程式：可對單張或整個 `data/raw` 目錄做自適應二值化、切割 5 個數字並辨識。
- 已訓練模型：目前 baseline 放在 `model/`，可直接推論，也可重新訓練覆蓋。

本專案只辨識數字 `1` 到 `9`，不包含 `0`。

## 專案結構

```text
captcha-ocr-tfjs/
  .github/workflows/ci.yml      GitHub Actions 基本檢查
  data/
    segmented/                  訓練資料，子目錄 1-9 為 label
    raw/                        五碼驗證碼原圖範例
  docs/                         維護文件
  model/                        TensorFlow.js model.json 與權重
  scripts/
    train.js                    訓練模型並輸出混淆矩陣
    predict.js                  批次推論 RawData 並輸出 HTML/CSV/JSON 報告
    solve-file.js               單張驗證碼推論
  src/
    ocr.js                      可被其他 Node.js 專案引用的 OCR API
    image-utils.js              二值化、切割、正規化
    tfjs-io.js                  TensorFlow.js 模型讀寫
  test/smoke.js                 基本推論測試
  vendor/tf.min.js              無 npm 環境的 TensorFlow.js fallback
```

## 安裝

```bash
npm install
```

如果是在 Codex workspace 且沒有 `npm`，也可以直接使用 bundled Node 跑程式；專案會 fallback 到 `vendor/tf.min.js`。

## 單張推論

```bash
npm run solve -- data/raw/14662.png
```

或：

```bash
node scripts/solve-file.js data/raw/14662.png
```

輸出會是 JSON，包含整體辨識結果、每個 digit 的信心分數與 bounding box。

## 批次推論

```bash
npm run predict
```

預設讀取：

- 模型：`model/`
- 原圖：`data/raw/`
- 輸出：`reports/raw-predictions/`

主要輸出：

- `reports/raw-predictions/report.html`
- `reports/raw-predictions/predictions.csv`
- `reports/raw-predictions/predictions.json`
- `reports/raw-predictions/binarized/`
- `reports/raw-predictions/digits/`

## 訓練

```bash
npm run train
```

預設讀取 `data/segmented/1` 到 `data/segmented/9`，並輸出：

- TensorFlow.js 模型：`model/`
- 混淆矩陣：`reports/training/confusion-matrix.csv`
- 訓練摘要：`reports/training/training-report.html`
- 指標 JSON：`reports/training/metrics.json`

常用參數：

```bash
node scripts/train.js --epochs=20 --batch-size=32 --image-size=32
node scripts/train.js --data-dir=data/segmented --model-dir=model --report-dir=reports/training
```

## 在其他 Node.js 程式引用

```js
const { solveCaptchaFile } = require("./src/ocr");

async function main() {
  const result = await solveCaptchaFile("data/raw/14662.png", {
    modelDir: "model"
  });
  console.log(result.text);
}

main();
```

## 維護建議

- 新增樣本時，將切割好的單字圖片放到 `data/segmented/<label>/`。
- 若驗證碼樣式改變，先增加樣本，再執行 `npm run train` 重訓。
- 重訓後執行 `npm test` 和 `npm run predict` 檢查 baseline 是否仍穩定。
- `reports/` 是產物，不建議提交到 GitHub。

## 授權

MIT
