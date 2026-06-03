# Training

Run:

```bash
npm run train
```

The trainer:

1. Reads `data/segmented/1` through `data/segmented/9`.
2. Applies the same preprocessing used during inference.
3. Splits each class into train, validation, and test sets.
4. Trains a small TensorFlow.js dense model.
5. Writes a TensorFlow.js model to `model/`.
6. Writes metrics and a confusion matrix to `reports/training/`.

Useful options:

```bash
node scripts/train.js --epochs=30
node scripts/train.js --batch-size=16
node scripts/train.js --image-size=32
node scripts/train.js --seed=123
node scripts/train.js --data-dir=data/segmented --model-dir=model --report-dir=reports/training
```

The current baseline uses `imageSize=32` and labels `1` to `9`.
