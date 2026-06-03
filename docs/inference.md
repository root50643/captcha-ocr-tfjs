# Inference

## Single File

```bash
npm run solve -- data/raw/14662.png
```

The output is JSON:

```json
{
  "text": "14662",
  "confidence": 0.9994,
  "segmentMethod": "projection",
  "digits": []
}
```

## Batch

```bash
npm run predict
```

The batch predictor writes:

- `reports/raw-predictions/report.html`
- `reports/raw-predictions/predictions.csv`
- `reports/raw-predictions/predictions.json`
- `reports/raw-predictions/binarized/*.png`
- `reports/raw-predictions/digits/*.png`

The report is useful for checking whether failures are caused by segmentation or model classification.
