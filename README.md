# Phy-L JEPA

Mueller-matrix JEPA pipeline for the ColoPola dataset.

## Scope

This repository is organized around a compact, publication-style training stack:

- direct physical features from the coherency matrix real and imaginary parts
- lightweight transformer context encoder for the Cloude feature path
- CPU training entrypoint plus Colab/GPU-friendly launch path
- ColoPola NPZ training data
- masked JEPA and physics-retention JEPA variants

The goal is to keep the codebase focused on the stage work and on a reproducible training workflow.

## Main components

- `colopola_dataset.py`: ColoPola NPZ loader
- `physics_features.py`: coherency extractor plus MLP and transformer encoders
- `architectures.py`: lightweight Mueller encoder and MLP predictor
- `hybrid_physics_jepa.py`: masked JEPA with direct physical retention
- `scripts/`: runnable entrypoints for training, evaluation, probing, and comparison
- `scripts/pretrain_adaptive_hybrid.py`: CPU-only training entrypoint
- `scripts/train_jepa_cpu_150.py`: dedicated 150-epoch training launcher for the Cloude transformer JEPA
- `scripts/train_probe_mlp.py`: frozen-JEPA linear and MLP probe suite for healthy/cancer prediction

## Data

Training in this repository expects the ColoPola data in NPZ format.

### Official source (Zenodo)

The dataset is available on Zenodo:  
https://zenodo.org/records/10554304

Current release files are provided in **HDF5 (`.h5`)** format.

### Convert `.h5` to `.npz` (code-compatible format)

This codebase expects train/test NPZ files readable by `colopola_dataset.py`.  
If your downloaded files are `.h5`, convert them first.

#### 1) Install dependencies

```bash
py -3.13 -m pip install h5py numpy
```

#### 2) Example conversion script

Create `tools/convert_h5_to_npz.py`:

```python
from pathlib import Path
import argparse
import numpy as np
import h5py

def find_first_existing(h5f, candidates):
    for k in candidates:
        if k in h5f:
            return np.array(h5f[k])
    raise KeyError(f"None of these keys were found: {candidates}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input .h5 file")
    parser.add_argument("--output", type=str, required=True, help="Path to output .npz file")
    parser.add_argument("--x-key", type=str, default="", help="Optional explicit key for features/images")
    parser.add_argument("--y-key", type=str, default="", help="Optional explicit key for labels")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(in_path, "r") as f:
        # Common key names used in H5 datasets
        x_candidates = [args.x_key] if args.x_key else []
        x_candidates += ["X", "x", "images", "data", "inputs", "mueller", "features"]

        y_candidates = [args.y_key] if args.y_key else []
        y_candidates += ["y", "Y", "labels", "target", "targets", "classes"]

        X = find_first_existing(f, [k for k in x_candidates if k])
        y = find_first_existing(f, [k for k in y_candidates if k])

    # Save with standard key names expected by most training loaders
    np.savez_compressed(out_path, X=X, y=y)
    print(f"[OK] Saved: {out_path}")
    print(f"     X shape: {X.shape}, dtype: {X.dtype}")
    print(f"     y shape: {y.shape}, dtype: {y.dtype}")

if __name__ == "__main__":
    main()
```

#### 3) Run conversion

```bash
py -3.13 tools/convert_h5_to_npz.py --input path/to/train.h5 --output path/to/train.npz
py -3.13 tools/convert_h5_to_npz.py --input path/to/test.h5  --output path/to/test.npz
```

If your H5 file uses different dataset names, pass explicit keys:

```bash
py -3.13 tools/convert_h5_to_npz.py --input path/to/train.h5 --output path/to/train.npz --x-key images --y-key labels
```

### Expected local path

By default, the launcher looks for:

`data/GIGADATASET_COLAB_NPZ`

On other machines, place the converted NPZ train/test files at `data/GIGADATASET_COLAB_NPZ` inside the repo, or set `PHY_L_JEPA_DATA_ROOT` to the dataset folder, or pass `--data-root` explicitly.

## Training

Run the 150-epoch Cloude transformer JEPA:

```bash
py -3.13 scripts/train_jepa_cpu_150.py --epochs 150 --output-dir results/phys_jepa_cloude_transformer
```

Outputs are written under `results/phys_jepa_cloude_transformer/`.

## Alternative entrypoint

The more general CPU-only pretraining script is:

```bash
py -3.13 scripts/pretrain_adaptive_hybrid.py --config config.yaml --epochs 150
```

It uses the same ColoPola loader and direct coherency features.

## Colab

See `COLAB.md` for a minimal GPU launch path in Google Colab.

## Repository layout

- `config.yaml`: default CPU configuration
- `config.mueller.local.yaml`: local override
- `results/`: training logs and checkpoints, ignored by Git
- `benchmark_results/`: legacy benchmark outputs kept for reference, ignored by Git

## Notes

- The transformer path is intentionally lightweight for CPU or Colab execution.
- The training pipeline assumes Mueller tensors with 16 channels.
- No image assets are required for the main workflow.
