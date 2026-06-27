from __future__ import annotations

from typing import Optional

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class MuellerPatchEncoder(nn.Module):
    """Lightweight spatial encoder for Mueller matrices.

    Inputs are `[B, 16, H, W]`. The output is a token sequence
    `[B, num_patches, D]`.
    """

    def __init__(
        self,
        in_channels: int = 16,
        patch_size: int = 1,
        embed_dim: int = 128,
        depth: int = 4,
        dropout: float = 0.1,
        image_size: int = 128,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)
        self.image_size = int(image_size)
        self.n_patches_h = max(1, self.image_size // self.patch_size)
        self.n_patches_w = max(1, self.image_size // self.patch_size)
        self.num_patches = self.n_patches_h * self.n_patches_w

        self.patch_encoder = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def embed_patches(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f"Expected [B,{self.in_channels},H,W], got {tuple(x.shape)}")
        tokens = self.patch_encoder(x).flatten(2).transpose(1, 2)
        if tokens.shape[1] != self.num_patches:
            # Allow small grids whose size differs from the configuration at runtime.
            pos = self.pos_embed[:, : tokens.shape[1], :]
        else:
            pos = self.pos_embed
        return tokens + pos

    def forward(
        self, x: torch.Tensor, visible_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        tokens = self.embed_patches(x)
        if visible_indices is not None:
            tokens = torch.gather(
                tokens,
                1,
                visible_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim),
            )
        return tokens

    def represent(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).mean(dim=1)


class ImageMuellerTransformerEncoder(nn.Module):
    """Image-scale transformer encoder over raw 16-channel Mueller patches."""

    def __init__(
        self,
        in_channels: int = 16,
        patch_size: int = 1,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 8,
        mlp_hidden_dim: int = 256,
        dropout: float = 0.1,
        image_size: int = 5,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)
        self.image_size = int(image_size)
        self.n_patches_h = max(1, self.image_size // self.patch_size)
        self.n_patches_w = max(1, self.image_size // self.patch_size)
        self.num_patches = self.n_patches_h * self.n_patches_w

        self.patch_embed = nn.Sequential(
            nn.Conv2d(self.in_channels, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size),
            nn.GELU(),
            nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=1),
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
        self, x: torch.Tensor, visible_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f"Expected [B,{self.in_channels},H,W], got {tuple(x.shape)}")

        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_embed[:, : tokens.shape[1], :]
        tokens = self.final_norm(self.transformer(tokens))
        if visible_indices is not None:
            tokens = torch.gather(
                tokens,
                1,
                visible_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim),
            )
        return tokens

    def represent(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).mean(dim=1)




class MuellerMatrixEncoder(nn.Module):
    """Matrix-level encoder for Mueller samples.

    Each spatial position is treated as one 16D Mueller matrix token.
    This avoids any spatial patch convolution and keeps the masked JEPA
    prediction problem at the matrix level.
    """

    def __init__(
        self,
        in_channels: int = 16,
        patch_size: int = 1,
        embed_dim: int = 64,
        depth: int = 1,
        dropout: float = 0.05,
        image_size: int = 5,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)
        self.image_size = int(image_size)
        self.n_patches_h = max(1, self.image_size)
        self.n_patches_w = max(1, self.image_size)
        self.num_patches = self.n_patches_h * self.n_patches_w
        self.hidden_dim = int(hidden_dim)

        layers: list[nn.Module] = [
            nn.LayerNorm(self.in_channels),
            nn.Linear(self.in_channels, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        for _ in range(max(depth - 1, 0)):
            layers.extend(
                [
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
        layers.append(nn.Linear(self.hidden_dim, self.embed_dim))
        self.matrix_encoder = nn.Sequential(*layers)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def embed_matrices(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f"Expected [B,{self.in_channels},H,W], got {tuple(x.shape)}")
        batch, _, height, width = x.shape
        tokens = x.permute(0, 2, 3, 1).reshape(batch, height * width, self.in_channels)
        tokens = self.matrix_encoder(tokens)
        if tokens.shape[1] != self.num_patches:
            pos = self.pos_embed[:, : tokens.shape[1], :]
        else:
            pos = self.pos_embed
        return tokens + pos

    def forward(
        self, x: torch.Tensor, visible_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        tokens = self.embed_matrices(x)
        if visible_indices is not None:
            tokens = torch.gather(
                tokens,
                1,
                visible_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim),
            )
        return tokens

    def represent(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).mean(dim=1)

class TokenMLPPredictor(nn.Module):
    """No-attention token mixer implemented as a compact MLP."""

    def __init__(
        self,
        num_tokens: int,
        embed_dim: int,
        *,
        token_dim: int = 32,
        hidden_dim: int = 512,
        depth: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.embed_dim = int(embed_dim)
        self.token_dim = int(token_dim)
        self.input_proj = nn.Linear(self.embed_dim, self.token_dim)
        self.output_proj = nn.Linear(self.token_dim, self.embed_dim)
        seq_dim = self.num_tokens * self.token_dim
        blocks: list[nn.Module] = []
        current = seq_dim
        for _ in range(max(depth - 1, 0)):
            blocks.extend([nn.Linear(current, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
            current = hidden_dim
        blocks.append(nn.Linear(current, seq_dim))
        self.mixer = nn.Sequential(*blocks)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.input_proj(tokens).reshape(tokens.shape[0], -1)
        mixed = self.mixer(hidden).reshape(tokens.shape[0], self.num_tokens, self.token_dim)
        return self.output_proj(mixed)


class MaskedMuellerJEPA(nn.Module):
    """EMA-target masked JEPA operating on Mueller matrices."""

    def __init__(
        self,
        encoder: MuellerPatchEncoder,
        predictor_depth: int = 2,
        dropout: float = 0.1,
        mask_ratio: float = 0.5,
        ema_momentum: float = 0.996,
        loss: str = "smooth_l1",
        predictor_token_dim: int = 32,
        predictor_hidden_dim: int = 512,
    ):
        super().__init__()
        self.context_encoder = encoder
        self.target_encoder = copy.deepcopy(encoder)
        self.target_encoder.requires_grad_(False)
        self.target_encoder.eval()
        self.mask_ratio = float(mask_ratio)
        self.ema_momentum = float(ema_momentum)
        self.loss = loss
        self.mask_token = nn.Parameter(torch.zeros(1, 1, encoder.embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.predictor = TokenMLPPredictor(
            encoder.num_patches,
            encoder.embed_dim,
            token_dim=predictor_token_dim,
            hidden_dim=predictor_hidden_dim,
            depth=predictor_depth,
            dropout=dropout,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_encoder.eval()
        return self

    @torch.no_grad()
    def update_target(self) -> None:
        for context, target in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            target.mul_(self.ema_momentum).add_(context, alpha=1 - self.ema_momentum)

    def sample_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        n = self.context_encoder.num_patches
        n_mask = max(1, min(n - 1, round(self.mask_ratio * n)))
        scores = torch.rand(batch_size, n, device=device)
        mask_indices = scores.topk(n_mask, dim=1).indices
        mask = torch.zeros(batch_size, n, dtype=torch.bool, device=device)
        return mask.scatter_(1, mask_indices, True)

    def forward(
        self,
        context_view: torch.Tensor,
        target_view: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        target_view = context_view if target_view is None else target_view
        batch = context_view.shape[0]
        mask = self.sample_mask(batch, context_view.device) if mask is None else mask

        visible = (~mask).nonzero(as_tuple=False)[:, 1].view(batch, -1)
        hidden = mask.nonzero(as_tuple=False)[:, 1].view(batch, -1)
        context = self.context_encoder(context_view, visible)

        full = self.mask_token.expand(batch, self.context_encoder.num_patches, -1).clone()
        full.scatter_(1, visible.unsqueeze(-1).expand_as(context), context)

        pos = self.context_encoder.pos_embed.expand(batch, -1, -1)
        predictions = self.predictor(full + pos)
        predictions = torch.gather(
            predictions, 1, hidden.unsqueeze(-1).expand(-1, -1, predictions.shape[-1])
        )
        with torch.no_grad():
            targets = self.target_encoder(target_view)
            targets = torch.gather(
                targets, 1, hidden.unsqueeze(-1).expand(-1, -1, targets.shape[-1])
            )

        if self.loss == "mse":
            loss = F.mse_loss(predictions, targets)
        else:
            loss = F.smooth_l1_loss(predictions, targets)

        return {
            "loss": loss,
            "predictions": predictions,
            "targets": targets,
            "mask": mask,
        }

