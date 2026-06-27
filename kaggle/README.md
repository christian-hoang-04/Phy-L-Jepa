# Kaggle Run Guide

Use the laptop for smoke tests and code edits. Use Kaggle for full training.

## Expected Kaggle Setup

1. Create a Kaggle notebook with GPU enabled.
2. Add this repo or upload the working branch contents.
3. Add the ColoPola NPZ dataset as a Kaggle dataset.
4. Run a smoke test first.
5. Run the full training command only after the smoke test works.

## Local Smoke Test

From the repo root:

```bash
python tools/smoke_forward.py
```

This uses synthetic data and checks that the model can perform one forward/backward step.

## Kaggle Smoke Test

From the repo root in Kaggle:

```bash
python tools/smoke_forward.py --device auto --batch-size 8
```

## Full Training Template

Replace `/kaggle/input/YOUR_DATASET_DIR` with the Kaggle input folder containing train/test `.npz` files.

```bash
python kaggle/train_kaggle.py \
  --data-root /kaggle/input/YOUR_DATASET_DIR \
  --epochs 150 \
  --batch-size 256 \
  --max-samples 4096 \
  --output-dir /kaggle/working/results/phys_jepa_cloude_transformer
```

## First Rule

Do not start with the full run.

Start with:

```bash
python kaggle/train_kaggle.py \
  --data-root /kaggle/input/YOUR_DATASET_DIR \
  --epochs 1 \
  --batch-size 64 \
  --max-samples 128 \
  --output-dir /kaggle/working/results/smoke
```

If that fails, fix data paths and shape assumptions before changing architecture.

