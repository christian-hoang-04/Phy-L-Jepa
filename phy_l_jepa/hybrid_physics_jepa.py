"""Masked Mueller JEPA with direct physical retention and collapse control."""

from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F

from .architectures import MuellerPatchEncoder, MaskedMuellerJEPA
from .physics_features import CloudeCoherencyFeatureExtractor


class HybridRetentionPhysicsJEPA(nn.Module):
    """Classical EMA-target JEPA plus direct coherency retention and VC regularization."""

    def __init__(
        self,
        encoder: MuellerPatchEncoder,
        reference: CloudeCoherencyFeatureExtractor,
        *,
        predictor_depth: int = 2,
        dropout: float = 0.1,
        mask_ratio: float = 0.5,
        ema_momentum: float = 0.996,
        prediction_loss: str = "smooth_l1",
        retention_weight: float = 1.0,
        encoder_retention_weight: float = 1.0,
        variance_weight: float = 1.0,
        covariance_weight: float = 0.01,
        variance_margin: float = 1.0,
    ) -> None:
        super().__init__()
        self.jepa = MaskedMuellerJEPA(
            encoder,
            predictor_depth=predictor_depth,
            dropout=dropout,
            mask_ratio=mask_ratio,
            ema_momentum=ema_momentum,
            loss=prediction_loss,
        )
        self.reference = reference
        self.feature_dim = int(getattr(reference, "feature_dim", 32))
        self.retention_head = nn.Sequential(
            nn.LayerNorm(encoder.embed_dim),
            nn.Linear(encoder.embed_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
        )
        self.encoder_retention_head = copy.deepcopy(self.retention_head)
        self.reference_norm = nn.LayerNorm(self.feature_dim, elementwise_affine=False)
        self.retention_weight = float(retention_weight)
        self.encoder_retention_weight = float(encoder_retention_weight)
        self.variance_weight = float(variance_weight)
        self.covariance_weight = float(covariance_weight)
        self.variance_margin = float(variance_margin)

    @property
    def context_encoder(self) -> MuellerPatchEncoder:
        return self.jepa.context_encoder

    @property
    def target_encoder(self) -> MuellerPatchEncoder:
        return self.jepa.target_encoder

    @torch.no_grad()
    def update_target(self) -> None:
        self.jepa.update_target()

    def train(self, mode: bool = True):
        super().train(mode)
        self.reference.eval()
        self.jepa.target_encoder.eval()
        return self

    def _variance_covariance(
        self, embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat = embeddings.reshape(-1, embeddings.shape[-1])
        centered = flat - flat.mean(dim=0, keepdim=True)
        std = torch.sqrt(centered.var(dim=0, unbiased=False) + 1e-4)
        variance_loss = F.relu(self.variance_margin - std).mean()
        if flat.shape[0] < 2:
            covariance_loss = flat.new_zeros(())
        else:
            covariance = centered.T @ centered / (flat.shape[0] - 1)
            covariance.fill_diagonal_(0)
            covariance_loss = covariance.square().sum() / flat.shape[1]
        return variance_loss, covariance_loss

    def forward(
        self,
        context_view: torch.Tensor,
        target_view: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        target_view = context_view if target_view is None else target_view
        result = self.jepa(context_view, target_view, mask=mask)

        hidden = result["mask"].nonzero(as_tuple=False)[:, 1].view(
            context_view.shape[0], -1
        )
        patch_ids = hidden

        with torch.no_grad():
            reference_all = self.reference(target_view)
            reference_targets = torch.gather(
                reference_all,
                1,
                patch_ids.unsqueeze(-1).expand(-1, -1, self.feature_dim),
            )
            reference_targets = self.reference_norm(reference_targets)

        retained = self.retention_head(result["predictions"])
        retention_loss = F.mse_loss(retained, reference_targets)

        online_full = self.context_encoder(context_view)
        encoder_retained = self.encoder_retention_head(online_full)
        encoder_reference = self.reference_norm(reference_all)
        encoder_retention_loss = F.mse_loss(encoder_retained, encoder_reference)

        encoder_recording_proxy = online_full.mean(dim=1)
        variance_loss, covariance_loss = self._variance_covariance(
            encoder_recording_proxy
        )
        total = (
            result["loss"]
            + self.retention_weight * retention_loss
            + self.encoder_retention_weight * encoder_retention_loss
            + self.variance_weight * variance_loss
            + self.covariance_weight * covariance_loss
        )
        return {
            **result,
            "loss": total,
            "jepa_loss": result["loss"],
            "retention_loss": retention_loss,
            "encoder_retention_loss": encoder_retention_loss,
            "variance_loss": variance_loss,
            "covariance_loss": covariance_loss,
            "retained": retained,
            "encoder_retained": encoder_retained,
            "encoder_recording_proxy": encoder_recording_proxy,
            "reference_targets": reference_targets,
        }
