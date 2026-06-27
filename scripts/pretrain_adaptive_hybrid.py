"""Pretrain the physics-aware JEPA on the ColoPola dataset.

This version is CPU-only, removes the transformer path, and uses direct
Cloude coherency real/imaginary features as the retention target.
"""

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
import yaml
from torch.utils.data import DataLoader

from phy_l_jepa.architectures import MuellerPatchEncoder
from phy_l_jepa.colopola_dataset import ColoPolaDataset
from phy_l_jepa.hybrid_physics_jepa import HybridRetentionPhysicsJEPA
from phy_l_jepa.physics_features import FrozenLNPIVAEReference


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DATA_DIR = PROJECT_ROOT / "data" / "GIGADATASET_COLAB_NPZ"
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


def load_yaml(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_json(path: str | Path, value: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--ckpt-dir", type=str, default=None)
    parser.add_argument("--variance-weight", type=float, default=None)
    parser.add_argument("--covariance-weight", type=float, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    cfg["seed"] = args.seed
    cfg.setdefault("optim", {})
    cfg.setdefault("data", {})
    cfg.setdefault("model", {})
    cfg.setdefault("output", {})
    if args.epochs is not None:
        cfg["optim"]["epochs"] = args.epochs
    if args.data_root is not None:
        cfg["data"]["data_root"] = args.data_root
    if args.ckpt_dir is not None:
        cfg["output"]["results_dir"] = args.ckpt_dir
        cfg["output"]["checkpoint"] = str(Path(args.ckpt_dir) / "latest.pth.tar")
    if args.variance_weight is not None:
        cfg["model"]["variance_weight"] = args.variance_weight
    if args.covariance_weight is not None:
        cfg["model"]["covariance_weight"] = args.covariance_weight

    if args.smoke_test:
        cfg["optim"]["epochs"] = min(int(cfg["optim"].get("epochs", 1)), 1)
        cfg["data"]["batch_size"] = min(int(cfg.get("data", {}).get("batch_size", 8)), 2)
        cfg["data"]["num_workers"] = 0
        cfg["output"]["results_dir"] = str(PROJECT_ROOT / "results" / "mueller_smoke")
        cfg["output"]["checkpoint"] = str(Path(cfg["output"]["results_dir"]) / "latest.pth.tar")

    set_seed(int(cfg.get("seed", 42)))
    device = torch.device("cpu")
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    data_root = Path(
        cfg.get("data", {}).get("data_root", str(DATA_DIR))
    )
    print(f"[mueller-hybrid] Initializing ColoPolaDataset from {data_root}...")
    dataset = ColoPolaDataset(data_dir=data_root, split="train")
    if args.smoke_test and len(dataset) > 32:
        from torch.utils.data import Subset

        dataset = Subset(dataset, list(range(32)))
        print("[mueller-hybrid] Smoke test enabled: limiting dataset to 32 samples.")

    sample = dataset[0]
    sample_tensor = sample[0] if isinstance(sample, tuple) else sample
    if not isinstance(sample_tensor, torch.Tensor):
        sample_tensor = torch.as_tensor(sample_tensor)
    if sample_tensor.ndim != 3 or sample_tensor.shape[0] != 16:
        raise ValueError(
            f"Expected a Mueller sample shaped [16,H,W], got {tuple(sample_tensor.shape)}"
        )

    sample_h, sample_w = int(sample_tensor.shape[1]), int(sample_tensor.shape[2])
    image_size = int(cfg.get("model", {}).get("image_size", max(sample_h, sample_w)))
    patch_size = int(cfg.get("model", {}).get("patch_size", 1))
    if image_size != sample_h or image_size != sample_w:
        image_size = max(sample_h, sample_w)
    if patch_size <= 0 or image_size % patch_size != 0:
        patch_size = 1

    loader = DataLoader(
        dataset,
        batch_size=int(cfg.get("data", {}).get("batch_size", 32)),
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )

    encoder = MuellerPatchEncoder(
        in_channels=16,
        patch_size=patch_size,
        embed_dim=int(cfg.get("model", {}).get("embed_dim", 128)),
        depth=int(cfg.get("model", {}).get("depth", 2)),
        dropout=float(cfg.get("model", {}).get("dropout", 0.1)),
        image_size=image_size,
    )

    reference = FrozenLNPIVAEReference(model_path=None, patch_size=patch_size)
    reference = reference.to(device)

    jepa = HybridRetentionPhysicsJEPA(
        encoder=encoder,
        reference=reference,
        predictor_depth=int(cfg.get("model", {}).get("predictor_depth", 2)),
        dropout=float(cfg.get("model", {}).get("dropout", 0.1)),
        mask_ratio=float(cfg.get("masking", {}).get("ratio", 0.5)),
        ema_momentum=float(cfg.get("model", {}).get("ema_momentum", 0.996)),
        prediction_loss=str(cfg.get("model", {}).get("loss", "smooth_l1")),
        retention_weight=float(cfg.get("model", {}).get("retention_weight", 1.0)),
        encoder_retention_weight=float(cfg.get("model", {}).get("encoder_retention_weight", 1.0)),
        variance_weight=float(cfg.get("model", {}).get("variance_weight", 1.0)),
        covariance_weight=float(cfg.get("model", {}).get("covariance_weight", 0.01)),
    ).to(device)

    base_jepa = jepa
    params_to_optimize = (
        list(base_jepa.context_encoder.parameters())
        + list(base_jepa.jepa.predictor.parameters())
        + [base_jepa.jepa.mask_token]
        + list(base_jepa.retention_head.parameters())
        + list(base_jepa.encoder_retention_head.parameters())
    )

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=float(cfg.get("optim", {}).get("lr", 1e-3)),
        weight_decay=float(cfg.get("optim", {}).get("weight_decay", 0.05)),
    )

    epochs = int(cfg.get("optim", {}).get("epochs", 50))
    start_epoch = 0
    history = []
    output_dir = Path(cfg.get("output", {}).get("results_dir", PROJECT_ROOT / "results" / "mueller_hybrid"))
    checkpoint_path = Path(cfg.get("output", {}).get("checkpoint", output_dir / "latest.pth.tar"))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if args.resume and checkpoint_path.exists():
        print(f"[mueller-hybrid] Resuming from checkpoint: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device)
        base_jepa.context_encoder.load_state_dict(state["encoder"])
        base_jepa.target_encoder.load_state_dict(state["target_encoder"])
        base_jepa.jepa.predictor.load_state_dict(state["predictor"])
        base_jepa.retention_head.load_state_dict(state["retention_head"])
        base_jepa.encoder_retention_head.load_state_dict(state["encoder_retention_head"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state.get("epoch", 0))

    print(f"[mueller-hybrid] Starting training for {epochs} epochs on {device} (start={start_epoch})")
    max_batches = args.max_batches

    for epoch in range(start_epoch, epochs):
        jepa.train()
        losses = []
        epoch_start = time.perf_counter()
        for batch_idx, (view_target, view_context) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            view_target = view_target.to(device)
            view_context = view_context.to(device)

            optimizer.zero_grad(set_to_none=True)
            result = jepa(view_context, view_target)
            loss = result["loss"].mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_to_optimize, 1.0)
            optimizer.step()
            base_jepa.update_target()
            losses.append(float(loss.item()))

        avg_loss = float(np.mean(losses)) if losses else float("nan")
        history.append(
            {
                "epoch": epoch + 1,
                "loss": avg_loss,
                "elapsed_sec": time.perf_counter() - epoch_start,
            }
        )
        print(f"[mueller-hybrid] epoch {epoch + 1:03d} loss={avg_loss:.6f}")

        save_json(output_dir / "pretrain_history.json", {"history": history, "config": cfg})
        torch.save(
            {
                "epoch": epoch + 1,
                "encoder": base_jepa.context_encoder.state_dict(),
                "target_encoder": base_jepa.target_encoder.state_dict(),
                "predictor": base_jepa.jepa.predictor.state_dict(),
                "retention_head": base_jepa.retention_head.state_dict(),
                "encoder_retention_head": base_jepa.encoder_retention_head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": cfg,
            },
            checkpoint_path,
        )

    print(f"[mueller-hybrid] Finished. Checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
