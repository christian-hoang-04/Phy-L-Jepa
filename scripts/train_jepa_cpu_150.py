from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from phy_l_jepa.architectures import MaskedMuellerJEPA
from phy_l_jepa.colopola_dataset import ColoPolaDataset
from phy_l_jepa.paths import default_data_root
from phy_l_jepa.physics_features import CloudeTransformerEncoder


DEFAULT_DATA_ROOT = default_data_root()
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "phys_jepa_cloude_transformer"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def build_loader(
    data_root: Path,
    split: str,
    batch_size: int,
    max_samples: int | None,
    shuffle: bool,
    smoke_test: bool,
) -> tuple[DataLoader, torch.Tensor]:
    dataset = ColoPolaDataset(
        data_dir=data_root,
        split=split,
        max_samples=64 if smoke_test else max_samples,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No samples found for split '{split}' in {data_root}")
    sample, _ = dataset[0]
    if not isinstance(sample, torch.Tensor):
        sample = torch.as_tensor(sample)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=False), sample


def build_model(sample: torch.Tensor) -> MaskedMuellerJEPA:
    if sample.ndim != 3 or sample.shape[0] != 16:
        raise ValueError(f"Expected a Mueller tensor shaped [16,H,W], got {tuple(sample.shape)}")

    image_size = int(max(sample.shape[1], sample.shape[2]))
    encoder = CloudeTransformerEncoder(
        patch_size=1,
        embed_dim=64,
        mlp_hidden_dim=128,
        num_heads=4,
        depth=2,
        dropout=0.1,
        image_size=image_size,
    )
    return MaskedMuellerJEPA(
        encoder=encoder,
        predictor_depth=2,
        dropout=0.1,
        mask_ratio=0.5,
        ema_momentum=0.99,
        loss="smooth_l1",
        predictor_token_dim=16,
        predictor_hidden_dim=128,
    )


@torch.no_grad()
def evaluate_split(model: MaskedMuellerJEPA, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    losses = []
    with torch.no_grad():
        for view_target, view_context in loader:
            view_target = view_target.to(device, non_blocking=True)
            view_context = view_context.to(device, non_blocking=True)
            out = model(view_context, view_target)
            losses.append(float(out["loss"].mean().item()))
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "num_batches": len(losses),
        "num_samples": len(loader.dataset),
    }


def train_run(
    *,
    data_root: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    max_samples: int,
    smoke_test: bool,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    train_loader, train_sample = build_loader(
        data_root=data_root,
        split="train",
        batch_size=batch_size,
        max_samples=max_samples,
        shuffle=True,
        smoke_test=smoke_test,
    )
    model = build_model(train_sample).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=3e-4, weight_decay=0.01)

    run_dir = output_dir / "phys_jepa_cloude_transformer"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "latest.pth.tar"

    print(
        f"[phys_jepa_cloude_transformer] start | epochs={epochs} | batch_size={batch_size} | "
        f"train_samples={len(train_loader.dataset)} | device={device} | data_root={data_root}",
        flush=True,
    )
    t0 = time.perf_counter()
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        losses = []
        for batch_idx, (view_target, view_context) in enumerate(train_loader):
            view_target = view_target.to(device, non_blocking=True)
            view_context = view_context.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = model(view_context, view_target)
            loss = out["loss"].mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            model.update_target()
            losses.append(float(loss.item()))

            if batch_idx % 25 == 0:
                print(
                    f"[phys_jepa_cloude_transformer] epoch {epoch:03d} batch {batch_idx:04d} "
                    f"loss={loss.item():.6f}",
                    flush=True,
                )

        avg_loss = float(np.mean(losses)) if losses else float("nan")
        history.append({
            "epoch": epoch,
            "loss": avg_loss,
            "elapsed_sec": time.perf_counter() - epoch_start,
        })
        print(f"[phys_jepa_cloude_transformer] epoch {epoch:03d}/{epochs} loss={avg_loss:.6f}", flush=True)
        save_json(run_dir / "history.json", {"history": history})
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            checkpoint_path,
        )

    test_loader, _ = build_loader(
        data_root=data_root,
        split="test",
        batch_size=batch_size,
        max_samples=None,
        shuffle=False,
        smoke_test=smoke_test,
    )
    test_metrics = evaluate_split(model, test_loader, device)
    save_json(run_dir / "test_metrics.json", test_metrics)

    result = {
        "epochs": epochs,
        "batch_size": batch_size,
        "train_samples": len(train_loader.dataset),
        "test_samples": test_metrics["num_samples"],
        "elapsed_sec": time.perf_counter() - t0,
        "checkpoint": str(checkpoint_path),
        "test_metrics": test_metrics,
        "device": str(device),
    }
    save_json(output_dir / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=4096)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        args.epochs = min(args.epochs, 1)
        args.batch_size = min(args.batch_size, 64)
        args.max_samples = min(args.max_samples, 64)
        args.output_dir = PROJECT_ROOT / "results" / "phys_jepa_transformer_smoke"

    train_run(
        data_root=args.data_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        smoke_test=args.smoke_test,
    )
    print(f"done -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
