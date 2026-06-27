# Project Map

This file is Christian's working map for contributing safely to this repo.

## Current Working Branch

```text
christian/architecture-experiments
```

Remotes:

- `origin` - Christian's fork: <https://github.com/christian-hoang-04/Phy-L-Jepa>
- `upstream` - original repo: <https://github.com/ayouboa30/Phy-L-Jepa>

## Main Local Files

- `architectures.py` - Mueller encoders, token pooling, `MaskedMuellerJEPA`, and `TokenMLPPredictor`.
- `physics_features.py` - Cloude coherency feature extraction and Cloude transformer encoder.
- `hybrid_physics_jepa.py` - masked JEPA plus physical retention and collapse-control losses.
- `train_jepa_cpu_150.py` - current transformer training entrypoint.
- `pretrain_adaptive_hybrid.py` - CPU-only hybrid pretraining entrypoint.
- `train_probe_mlp.py` / `eval_probe_mlp.py` - downstream probing.

## New Helper Areas

- `tools/` - small local diagnostics and smoke tests.
- `kaggle/` - Kaggle-specific setup notes and training wrapper.

## Current Architecture Questions

1. Is the requested transformer change about the encoder, the predictor, or both?
2. Which pooled representation should change from mean pooling to max pooling?
3. Should the feature extractor remain physically grounded in Cloude coherency, or be replaced by a learned front-end?
4. What is the smallest experiment that can show an improvement under limited compute?

## Local Laptop Policy

Use the laptop for:

- import checks
- shape checks
- one-batch forward/backward smoke tests
- reading code
- editing code

Do not expect full 150-epoch training to be comfortable on CPU-only laptop hardware.

## Kaggle Policy

Use Kaggle for:

- full training runs
- GPU experiments
- comparing architecture variants
- saving checkpoints and metrics

