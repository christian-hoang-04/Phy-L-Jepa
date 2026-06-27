"""Direct physical features for Mueller matrices.

The public pipeline now uses the real and imaginary parts of the Cloude
coherency matrix as the physical retention target, instead of freezing a
separate LN-PIVAE reference model.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def pauli_basis(device: torch.device | None = None) -> torch.Tensor:
    dtype = torch.complex64
    sigma = torch.stack(
        [
            torch.tensor([[1, 0], [0, 1]], dtype=dtype, device=device),
            torch.tensor([[1, 0], [0, -1]], dtype=dtype, device=device),
            torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=device),
            torch.tensor([[0, -1j], [1j, 0]], dtype=dtype, device=device),
        ]
    )
    return torch.stack(
        [torch.kron(sigma[i], sigma[j].conj()) for i in range(4) for j in range(4)]
    ).reshape(4, 4, 4, 4)


@torch.no_grad()
def mueller_to_coherency(mueller: torch.Tensor) -> torch.Tensor:
    """Convert real Mueller matrices to Hermitian Cloude coherency matrices."""
    if mueller.ndim < 2 or mueller.shape[-1] != 16:
        raise ValueError(f"Expected [..., 16] Mueller vectors, got {tuple(mueller.shape)}")
    basis = pauli_basis(mueller.device)
    m = mueller.to(torch.complex64).reshape(*mueller.shape[:-1], 4, 4)
    h = torch.einsum("...ij,ijab->...ab", m, basis) / 4.0
    return 0.5 * (h + h.mH)


class CloudeCoherencyFeatureExtractor(nn.Module):
    """Direct coherency-feature extractor for Mueller patches.

    The output is the concatenation of the real and imaginary parts of the
    4x4 coherency matrix flattened to 32 features per spatial location.
    """

    def __init__(self, patch_size: int = 1):
        super().__init__()
        self.patch_size = int(patch_size)
        self.feature_dim = 32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 16:
            raise ValueError(f"Expected [B, 16, H, W], got {tuple(x.shape)}")

        batch, _, height, width = x.shape
        mueller = x.permute(0, 2, 3, 1).reshape(-1, 16)
        coherency = mueller_to_coherency(mueller)
        features = torch.cat(
            [coherency.real.reshape(-1, 16), coherency.imag.reshape(-1, 16)],
            dim=-1,
        )
        features = features.view(batch, height, width, self.feature_dim).permute(0, 3, 1, 2)

        if self.patch_size > 1:
            features = F.avg_pool2d(
                features,
                kernel_size=self.patch_size,
                stride=self.patch_size,
            )

        return features.flatten(2).transpose(1, 2)


class FrozenLNPIVAEReference(CloudeCoherencyFeatureExtractor):
    """Backward-compatible alias for the new direct coherency target."""

    def __init__(self, model_path: str | Path | None = None, patch_size: int = 1):
        super().__init__(patch_size=patch_size)
        self.model_path = None if model_path is None else str(model_path)


class CloudeMLPLatentEncoder(nn.Module):
    """Cloude coherency real/imag features followed by a small MLP projection."""

    def __init__(
        self,
        patch_size: int = 1,
        embed_dim: int = 32,
        hidden_dim: int = 64,
        depth: int = 1,
        dropout: float = 0.0,
        image_size: int = 5,
    ) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)
        self.image_size = int(image_size)
        self.n_patches_h = max(1, self.image_size // self.patch_size)
        self.n_patches_w = max(1, self.image_size // self.patch_size)
        self.num_patches = self.n_patches_h * self.n_patches_w
        self.extractor = CloudeCoherencyFeatureExtractor(patch_size=self.patch_size)

        layers: list[nn.Module] = [
            nn.LayerNorm(self.extractor.feature_dim),
            nn.Linear(self.extractor.feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        for _ in range(max(depth - 1, 0)):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(hidden_dim, self.embed_dim))
        self.projection = nn.Sequential(*layers)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(
        self, x: torch.Tensor, visible_indices: torch.Tensor | None = None
    ) -> torch.Tensor:
        tokens = self.projection(self.extractor(x))
        tokens = tokens + self.pos_embed[:, : tokens.shape[1], :]
        if visible_indices is not None:
            tokens = torch.gather(
                tokens,
                1,
                visible_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim),
            )
        return tokens

    def represent(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).mean(dim=1)


class CloudeTransformerEncoder(nn.Module):
    """Cloude coherency real/imag features followed by a lightweight transformer."""

    def __init__(
        self,
        patch_size: int = 1,
        embed_dim: int = 64,
        mlp_hidden_dim: int = 128,
        num_heads: int = 4,
        depth: int = 2,
        dropout: float = 0.1,
        image_size: int = 5,
    ) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)
        self.image_size = int(image_size)
        self.n_patches_h = max(1, self.image_size // self.patch_size)
        self.n_patches_w = max(1, self.image_size // self.patch_size)
        self.num_patches = self.n_patches_h * self.n_patches_w
        self.extractor = CloudeCoherencyFeatureExtractor(patch_size=self.patch_size)

        self.input_proj = nn.Sequential(
            nn.LayerNorm(self.extractor.feature_dim),
            nn.Linear(self.extractor.feature_dim, self.embed_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.final_norm = nn.LayerNorm(self.embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(
        self, x: torch.Tensor, visible_indices: torch.Tensor | None = None
    ) -> torch.Tensor:
        tokens = self.input_proj(self.extractor(x))
        tokens = self.transformer(tokens + self.pos_embed[:, : tokens.shape[1], :])
        tokens = self.final_norm(tokens)
        if visible_indices is not None:
            tokens = torch.gather(
                tokens,
                1,
                visible_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim),
            )
        return tokens

    def represent(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).mean(dim=1)
