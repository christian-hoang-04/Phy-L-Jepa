from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a tiny synthetic NPZ dataset for smoke tests.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/synthetic_smoke"))
    parser.add_argument("--train-samples", type=int, default=128)
    parser.add_argument("--test-samples", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_x = rng.normal(size=(args.train_samples, 16, args.image_size, args.image_size)).astype("float32")
    train_y = rng.integers(0, 2, size=(args.train_samples,), dtype="int64")
    test_x = rng.normal(size=(args.test_samples, 16, args.image_size, args.image_size)).astype("float32")
    test_y = rng.integers(0, 2, size=(args.test_samples,), dtype="int64")

    np.savez_compressed(args.output_dir / "train_smoke.npz", X=train_x, label=train_y)
    np.savez_compressed(args.output_dir / "test_smoke.npz", X=test_x, label=test_y)

    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()

