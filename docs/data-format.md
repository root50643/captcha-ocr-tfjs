# Data Format

## Segmented Training Data

Training data lives under:

```text
data/segmented/
  1/
  2/
  ...
  9/
```

Each subdirectory name is the label. Images inside that folder should contain one digit only.

Rules:

- Valid labels are `1` through `9`.
- Label `0` is intentionally unsupported.
- PNG and JPEG are supported.
- The image can have color and transparency; preprocessing converts it to grayscale and adaptive binary.

## Raw Captcha Data

Raw five-digit captcha examples live under:

```text
data/raw/
```

The inference pipeline expects each raw image to contain five digits. If the file name is exactly five digits, such as `14662.png`, the batch report treats it as ground truth for accuracy reporting.
