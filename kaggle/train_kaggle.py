from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_jepa_cpu_150 import train_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Kaggle-friendly launcher for Phy-L-Jepa training.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/results/phys_jepa_cloude_transformer"))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=4096)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    print(f"[kaggle] torch={torch.__version__}")
    print(f"[kaggle] cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[kaggle] gpu={torch.cuda.get_device_name(0)}")
    print(f"[kaggle] data_root={args.data_root}")
    print(f"[kaggle] output_dir={args.output_dir}")

    if not args.data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {args.data_root}")

    train_run(
        data_root=args.data_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        smoke_test=args.smoke_test,
    )


if __name__ == "__main__":
    main()
