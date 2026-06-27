from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

from .architectures import ImageMuellerTransformerEncoder
from .colopola_dataset import ColoPolaDataset
from .physics_features import CloudeTransformerEncoder


class ProbeHead(nn.Module):
    def __init__(self, in_dim: int, kind: str = "linear", hidden_dim: int = 32, dropout: float = 0.2) -> None:
        super().__init__()
        kind = kind.lower()
        if kind not in {"linear", "mlp"}:
            raise ValueError(f"Unknown head kind: {kind}")
        self.kind = kind
        if kind == "linear":
            self.net = nn.Linear(in_dim, 2)
        else:
            self.net = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 2),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def build_encoder(sample: torch.Tensor, encoder_type: str) -> nn.Module:
    image_size = int(max(sample.shape[1], sample.shape[2]))
    if encoder_type == "image":
        return ImageMuellerTransformerEncoder(
            patch_size=1,
            embed_dim=128,
            depth=4,
            num_heads=8,
            mlp_hidden_dim=256,
            dropout=0.1,
            image_size=image_size,
        )
    return CloudeTransformerEncoder(
        patch_size=1,
        embed_dim=64,
        mlp_hidden_dim=128,
        num_heads=4,
        depth=2,
        dropout=0.1,
        image_size=image_size,
    )


@torch.no_grad()
def encode_dataset(encoder: nn.Module, loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    feats = []
    labels = []
    encoder.eval()
    for x, y in loader:
        x = x.to(device)
        z = encoder.represent(x)
        feats.append(z.cpu())
        labels.append(y.cpu())
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


def make_split_indices(labels: torch.Tensor, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    labels_np = labels.numpy()
    rng = np.random.default_rng(seed)
    train_idx = []
    val_idx = []
    for cls in sorted(np.unique(labels_np).tolist()):
        cls_idx = np.where(labels_np == cls)[0]
        rng.shuffle(cls_idx)
        n_val = max(1, int(round(len(cls_idx) * val_fraction)))
        val_idx.extend(cls_idx[:n_val].tolist())
        train_idx.extend(cls_idx[n_val:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return np.asarray(train_idx, dtype=np.int64), np.asarray(val_idx, dtype=np.int64)


def build_base_dataset(data_root: Path, split: str, max_samples: int | None, smoke_test: bool) -> ColoPolaDataset:
    effective_max = 256 if smoke_test else max_samples
    return ColoPolaDataset(
        data_dir=data_root,
        split=split,
        max_samples=effective_max,
        return_labels=True,
        allowed_labels=(0, 1),
    )


def build_loader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=False)


@torch.no_grad()
def compute_metrics(head: nn.Module, features: torch.Tensor, labels: torch.Tensor) -> dict:
    head.eval()
    loader = DataLoader(TensorDataset(features, labels), batch_size=1024, shuffle=False, num_workers=0)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    correct = 0
    tp = fp = fn = 0
    for x, y in loader:
        logits = head(x)
        loss = criterion(logits, y)
        preds = logits.argmax(dim=1)
        bs = int(y.shape[0])
        total_loss += float(loss.item()) * bs
        total += bs
        correct += int((preds == y).sum().item())
        tp += int(((preds == 1) & (y == 1)).sum().item())
        fp += int(((preds == 1) & (y == 0)).sum().item())
        fn += int(((preds == 0) & (y == 1)).sum().item())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "num_samples": total,
    }


def fit_standardizer(train_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = train_features.mean(dim=0)
    std = train_features.std(dim=0, unbiased=False).clamp_min(1e-6)
    return mean, std


def standardize(features: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (features - mean) / std


def infer_probe_head_hidden_dim(state_dict: dict[str, torch.Tensor], kind: str) -> int:
    kind = kind.lower()
    if kind == "linear":
        return 0
    weight = state_dict.get("net.1.weight")
    if weight is None:
        raise RuntimeError("Cannot infer MLP hidden_dim from checkpoint: missing net.1.weight")
    return int(weight.shape[0])


def train_probe(
    *,
    kind: str,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    hidden_dim: int,
    dropout: float,
    weight_decay: float,
    patience: int,
    lr: float,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    head = ProbeHead(train_features.shape[1], kind=kind, hidden_dim=hidden_dim, dropout=dropout)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(TensorDataset(train_features, train_labels), batch_size=batch_size, shuffle=True, num_workers=0)
    best_state = None
    best_val_f1 = -1.0
    best_epoch = 0
    best_val_metrics = None
    stalled = 0
    history = []

    for epoch in range(1, epochs + 1):
        head.train()
        for x, y in train_loader:
            logits = head(x)
            loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        train_metrics = compute_metrics(head, train_features, train_labels)
        val_metrics = compute_metrics(head, val_features, val_labels)
        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        })

        improved = val_metrics["f1"] > best_val_f1 + 1e-4
        if improved:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            best_val_metrics = copy.deepcopy(val_metrics)
            best_state = copy.deepcopy(head.state_dict())
            torch.save({
                "epoch": epoch,
                "head": best_state,
                "head_config": {"kind": kind, "hidden_dim": hidden_dim, "dropout": dropout},
                "best_val_f1": best_val_f1,
            }, output_dir / "best.pth.tar")
            stalled = 0
        else:
            stalled += 1

        print(
            f"[probe_{kind}] epoch {epoch:03d}/{epochs} "
            f"train_acc={train_metrics['accuracy']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['f1']:.4f}",
            flush=True,
        )
        save_json(output_dir / "history.json", {"history": history})
        torch.save({
            "epoch": epoch,
            "head": head.state_dict(),
            "head_config": {"kind": kind, "hidden_dim": hidden_dim, "dropout": dropout},
            "optimizer": optimizer.state_dict(),
            "best_epoch": best_epoch,
            "best_val_f1": best_val_f1,
        }, output_dir / "latest.pth.tar")

        if stalled >= patience:
            break

    if best_state is not None:
        head.load_state_dict(best_state)

    train_metrics = compute_metrics(head, train_features, train_labels)
    val_metrics = compute_metrics(head, val_features, val_labels)
    test_metrics = compute_metrics(head, test_features, test_labels)
    result = {
        "kind": kind,
        "best_epoch": best_epoch,
        "best_val": best_val_metrics,
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
    }
    save_json(output_dir / "metrics.json", result)
    return result


def load_encoder(ckpt_path: Path, sample: torch.Tensor, encoder_type: str, device: torch.device) -> nn.Module:
    encoder = build_encoder(sample, encoder_type).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_state = state.get("model", state)
    encoder_state = {
        k.removeprefix("context_encoder."): v
        for k, v in model_state.items()
        if k.startswith("context_encoder.")
    }
    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Unexpected checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    encoder.eval()
    encoder.requires_grad_(False)
    return encoder


def evaluate_kind(kind: str, suite_dir: Path, test_features: torch.Tensor, test_labels: torch.Tensor) -> dict:
    probe_path = suite_dir / kind / "best.pth.tar"
    if not probe_path.exists():
        probe_path = suite_dir / kind / "latest.pth.tar"
    payload = torch.load(probe_path, map_location="cpu", weights_only=False)
    hidden_dim = infer_probe_head_hidden_dim(payload, kind)
    head = ProbeHead(test_features.shape[1], kind=kind, hidden_dim=hidden_dim if hidden_dim > 0 else 32, dropout=0.2)
    head.load_state_dict(payload["head"])
    metrics = compute_metrics(head, test_features, test_labels)
    out = {"kind": kind, "checkpoint": str(probe_path), "metrics": metrics}
    save_json(suite_dir / kind / "test_only_metrics.json", out)
    return out

