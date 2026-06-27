from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

from phy_l_jepa.architectures import ImageMuellerTransformerEncoder
from phy_l_jepa.colopola_dataset import ColoPolaDataset
from phy_l_jepa.physics_features import CloudeTransformerEncoder
DEFAULT_DATA_ROOT = Path(r"C:\Users\ayoub\Desktop\Stage\Project\data\GIGADATASET_COLAB_NPZ")
DEFAULT_JEPA_CKPT = PROJECT_ROOT / "results" / "phys_jepa_cloude_transformer" / "phys_jepa_cloude_transformer" / "latest.pth.tar"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "phys_jepa_probe_suite"


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
        batch_losses = []
        for x, y in train_loader:
            logits = head(x)
            loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--jepa-ckpt", type=Path, default=DEFAULT_JEPA_CKPT)
    parser.add_argument("--encoder-type", choices=("cloude", "image"), default="cloude")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    set_seed(42)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_train = build_base_dataset(args.data_root, "train", args.max_train_samples, args.smoke_test)
    test_ds = build_base_dataset(args.data_root, "test", None, args.smoke_test)

    train_idx, val_idx = make_split_indices(torch.tensor(base_train.labels), args.val_fraction, seed=42)
    train_ds = Subset(base_train, train_idx.tolist())
    val_ds = Subset(base_train, val_idx.tolist())

    train_loader = build_loader(train_ds, args.batch_size, shuffle=False)
    val_loader = build_loader(val_ds, args.batch_size, shuffle=False)
    test_loader = build_loader(test_ds, args.batch_size, shuffle=False)

    sample, _ = base_train[0]
    encoder = build_encoder(sample, args.encoder_type).to(device)
    state = torch.load(args.jepa_ckpt, map_location=device, weights_only=False)
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

    print(
        f"[probe_suite] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
        f"ckpt={args.jepa_ckpt}",
        flush=True,
    )

    t0 = time.perf_counter()
    train_features, train_labels = encode_dataset(encoder, train_loader, device)
    val_features, val_labels = encode_dataset(encoder, val_loader, device)
    test_features, test_labels = encode_dataset(encoder, test_loader, device)

    suite_dir = args.output_dir
    suite_dir.mkdir(parents=True, exist_ok=True)
    mean, std = fit_standardizer(train_features)
    torch.save({"mean": mean, "std": std}, suite_dir / "standardizer.pt")
    train_features = standardize(train_features, mean, std)
    val_features = standardize(val_features, mean, std)
    test_features = standardize(test_features, mean, std)

    save_json(suite_dir / "split_stats.json", {
        "train_samples": int(train_labels.numel()),
        "val_samples": int(val_labels.numel()),
        "test_samples": int(test_labels.numel()),
        "train_class_counts": {str(int(k)): int(v) for k, v in zip(*torch.unique(train_labels, return_counts=True))},
        "val_class_counts": {str(int(k)): int(v) for k, v in zip(*torch.unique(val_labels, return_counts=True))},
        "test_class_counts": {str(int(k)): int(v) for k, v in zip(*torch.unique(test_labels, return_counts=True))},
    })

    results = []
    for kind in ("linear", "mlp"):
        kind_dir = suite_dir / kind
        kind_hidden = 0 if kind == "linear" else args.hidden_dim
        kind_dropout = 0.0 if kind == "linear" else args.dropout
        print(f"[probe_suite] training {kind}", flush=True)
        results.append(
            train_probe(
                kind=kind,
                train_features=train_features,
                train_labels=train_labels,
                val_features=val_features,
                val_labels=val_labels,
                test_features=test_features,
                test_labels=test_labels,
                output_dir=kind_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                hidden_dim=kind_hidden,
                dropout=kind_dropout,
                weight_decay=args.weight_decay,
                patience=args.patience,
                lr=args.lr,
            )
        )

    final = {
        "elapsed_sec": time.perf_counter() - t0,
        "jepa_ckpt": str(args.jepa_ckpt),
        "encoder_type": args.encoder_type,
        "standardizer": str(suite_dir / "standardizer.pt"),
        "results": results,
    }
    save_json(suite_dir / "metrics.json", final)
    print(f"done -> {suite_dir}", flush=True)


if __name__ == "__main__":
    main()



