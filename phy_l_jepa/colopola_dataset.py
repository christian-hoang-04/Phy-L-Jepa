from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class ColoPolaDataset(Dataset):
    """ColoPola NPZ loader with in-memory caching."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        max_samples: Optional[int] = None,
        return_labels: bool = False,
        allowed_labels: Optional[Sequence[int]] = None,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.return_labels = bool(return_labels)

        files = sorted([f for f in self.data_dir.glob("*.npz") if split in f.name.lower()])
        if not files:
            raise FileNotFoundError(f"No NPZ files found for split '{split}' in {self.data_dir}")
        if len(files) > 1:
            print(f"[ColoPolaDataset] Found {len(files)} files for {split}. Using the first one for speed.")

        file_path = files[0]
        print(f"[ColoPolaDataset] Loading {file_path.name} into RAM...")
        with np.load(file_path, mmap_mode="r") as data:
            keys = list(data.keys())
            target_key = keys[0]
            for k in keys:
                if "16" in k or "batch" in k or "x" in k.lower():
                    target_key = k
                    break

            matrix_data = np.asarray(data[target_key])
            labels = np.asarray(data["label"]) if "label" in data else None

        if matrix_data.ndim != 4 or matrix_data.shape[1] != 16:
            raise ValueError(f"Expected [N,16,H,W] data in {file_path}, got {matrix_data.shape}")
        if self.return_labels and labels is None:
            raise ValueError(f"No 'label' array available in {file_path}")

        if labels is not None and allowed_labels is not None:
            allowed = np.asarray(list(allowed_labels), dtype=labels.dtype)
            keep = np.isin(labels, allowed)
            matrix_data = matrix_data[keep]
            labels = labels[keep]

        if max_samples is not None:
            matrix_data = matrix_data[: int(max_samples)]
            if labels is not None:
                labels = labels[: int(max_samples)]

        self.matrix_data = matrix_data
        self.labels = labels
        self.total_samples = int(matrix_data.shape[0])
        print(f"[ColoPolaDataset] Cached {self.total_samples} samples from key '{target_key}'.")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        matrix = self.matrix_data[idx]
        tensor = torch.from_numpy(np.asarray(matrix)).float()
        if not self.return_labels:
            return tensor, tensor

        label = int(self.labels[idx])
        return tensor, torch.tensor(label, dtype=torch.long)
