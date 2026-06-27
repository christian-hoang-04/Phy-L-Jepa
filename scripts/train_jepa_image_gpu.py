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

from phy_l_jepa.architectures import ImageMuellerTransformerEncoder, MaskedMuellerJEPA
from phy_l_jepa.colopola_dataset import ColoPolaDataset
from phy_l_jepa.paths import default_data_root


DEFAULT_DATA_ROOT = default_data_root()
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "masked_image_jepa_gpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def build_model(
    sample: torch.Tensor,
    *,
    patch_size: int,
    embed_dim: int,
    depth: int,
    num_heads: int,
    mlp_hidden_dim: int,
    dropout: float,
    mask_ratio: float,
) -> MaskedMuellerJEPA:
    if sample.ndim != 3 or sample.shape[0] != 16:
        raise ValueError(f"Expected a Mueller tensor shaped [16,H,W], got {tuple(sample.shape)}")

    image_size = int(max(sample.shape[1], sample.shape[2]))
    encoder = ImageMuellerTransformerEncoder(
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_hidden_dim=mlp_hidden_dim,
        dropout=dropout,
        image_size=image_size,
    )
    return MaskedMuellerJEPA(
        encoder=encoder,
        predictor_depth=2,
        dropout=dropout,
        mask_ratio=mask_ratio,
        ema_momentum=0.99,
        loss="smooth_l1",
        predictor_token_dim=max(16, embed_dim // 4),
        predictor_hidden_dim=max(128, embed_dim * 2),
    )


@torch.no_grad()
def evaluate_split(model: MaskedMuellerJEPA, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    losses = []
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


def train_run(args: argparse.Namespace) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_loader, train_sample = build_loader(
        data_root=args.data_root,
        split="train",
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        shuffle=True,
        smoke_test=args.smoke_test,
    )
    model = build_model(
        train_sample,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_hidden_dim=args.mlp_hidden_dim,
        dropout=args.dropout,
        mask_ratio=args.mask_ratio,
    ).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    run_dir = args.output_dir / "masked_image_jepa"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "latest.pth.tar"

    print(
        f"[masked_image_jepa] start | epochs={args.epochs} | batch_size={args.batch_size} | "
        f"train_samples={len(train_loader.dataset)} | device={device} | data_root={args.data_root}",
        flush=True,
    )
    t0 = time.perf_counter()
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        losses = []
        for batch_idx, (view_target, view_context) in enumerate(train_loader):
            view_target = view_target.to(device, non_blocking=True)
            view_context = view_context.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out = model(view_context, view_target)
                loss = out["loss"].mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            model.update_target()
            losses.append(float(loss.item()))

            if batch_idx % args.log_every == 0:
                print(f"[masked_image_jepa] epoch {epoch:03d} batch {batch_idx:04d} loss={loss.item():.6f}", flush=True)

        avg_loss = float(np.mean(losses)) if losses else float("nan")
        history.append({"epoch": epoch, "loss": avg_loss, "elapsed_sec": time.perf_counter() - epoch_start})
        print(f"[masked_image_jepa] epoch {epoch:03d}/{args.epochs} loss={avg_loss:.6f}", flush=True)
        save_json(run_dir / "history.json", {"history": history})
        torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}, checkpoint_path)

    test_loader, _ = build_loader(
        data_root=args.data_root,
        split="test",
        batch_size=args.batch_size,
        max_samples=None,
        shuffle=False,
        smoke_test=args.smoke_test,
    )
    test_metrics = evaluate_split(model, test_loader, device)
    save_json(run_dir / "test_metrics.json", test_metrics)

    result = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "train_samples": len(train_loader.dataset),
        "test_samples": test_metrics["num_samples"],
        "elapsed_sec": time.perf_counter() - t0,
        "checkpoint": str(checkpoint_path),
        "test_metrics": test_metrics,
        "device": str(device),
        "encoder_type": "image",
    }
    save_json(args.output_dir / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=1)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mask-ratio", type=float, default=0.6)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        args.epochs = min(args.epochs, 1)
        args.batch_size = min(args.batch_size, 64)
        args.max_samples = 64
        args.output_dir = PROJECT_ROOT / "results" / "masked_image_jepa_smoke"

    train_run(args)
    print(f"done -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
