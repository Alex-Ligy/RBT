"""
Loss functions for the RBT segmentation pipeline.

Two components are combined:

1. **Weighted Cross-Entropy Loss** — standard pixel-wise classification loss
   with per-class weights to counter the severe class imbalance (Class 1,
   "stripped aggregate", is rare).

2. **Local Contrastive Loss** — a pixel-level contrastive objective operating
   in the learned feature space.  For each anchor pixel *i*:
       • *Positive* pixels (i+): same-class neighbours within a local window.
       • *Negative* pixels (i−): different-class neighbours.
   The loss pulls same-class feature vectors closer together and pushes
   different-class feature vectors apart, which sharpens the boundary between
   visually similar bitumen and basalt textures.

   Mathematical formulation (InfoNCE-style):
       L_contrast = -log[ exp(sim(z_i, z_i+) / τ) /
                          Σ_k exp(sim(z_i, z_k) / τ) ]
   where sim(·,·) = cosine similarity and τ is a temperature scalar.

Reference:
    Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020.
    Adapted here for *pixel-level, local-neighbourhood* contrastive learning.
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# 1. Weighted Cross-Entropy
# ======================================================================
class WeightedCrossEntropyLoss(nn.Module):
    """Standard cross-entropy with per-class weights and ignore index.

    Args:
        class_weights: List of per-class weights (e.g. [0.5, 5.0, 1.0]).
            Higher weight on class 1 (aggregate) compensates for scarcity.
        ignore_index: Label index to ignore (default 255).
    """

    def __init__(
        self,
        class_weights: Optional[List[float]] = None,
        ignore_index: int = 255,
    ):
        super().__init__()
        weight = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, C, H, W) raw class scores.
            targets: (B, H, W) integer class labels.
        """
        return self.ce(logits, targets)


# ======================================================================
# 2. Local Contrastive Loss (pixel-level)
# ======================================================================
class LocalContrastiveLoss(nn.Module):
    """Pixel-level contrastive loss within local neighbourhoods.

    For each sampled anchor pixel, we extract its feature vector and the
    feature vectors of all pixels within a local window.  An InfoNCE-style
    loss then encourages the anchor to be similar to same-class neighbours
    (positives) and dissimilar to different-class neighbours (negatives).

    This is especially effective at boundaries between bitumen and exposed
    aggregate, where spectral overlap causes confusion.

    Args:
        temperature: τ in the InfoNCE denominator (lower → sharper).
        neighborhood: Side-length of the local window (must be odd).
        num_samples: Number of anchor pixels to sample per image.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        neighborhood: int = 15,
        num_samples: int = 256,
    ):
        super().__init__()
        self.temperature = temperature
        self.neighborhood = neighborhood
        self.num_samples = num_samples

    def forward(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features: (B, D, H, W) L2-normalised feature embeddings
                      (e.g., from an intermediate decoder layer).
            targets:  (B, H, W) ground-truth class labels.

        Returns:
            Scalar contrastive loss averaged over the batch.
        """
        B, D, H, W = features.shape
        half = self.neighborhood // 2
        total_loss = torch.tensor(0.0, device=features.device)
        valid_count = 0

        for b in range(B):
            feat = features[b]    # (D, H, W)
            label = targets[b]    # (H, W)

            # Sample anchor positions (avoid border region)
            valid_h = torch.arange(half, H - half, device=features.device)
            valid_w = torch.arange(half, W - half, device=features.device)
            if valid_h.numel() == 0 or valid_w.numel() == 0:
                continue

            # Random anchor indices
            num_anchors = min(self.num_samples, valid_h.numel() * valid_w.numel())
            idx = torch.randperm(valid_h.numel() * valid_w.numel(), device=features.device)[:num_anchors]
            anchor_h = valid_h[idx // valid_w.numel()]
            anchor_w = valid_w[idx % valid_w.numel()]

            for i in range(num_anchors):
                ah, aw = anchor_h[i].item(), anchor_w[i].item()
                anchor_feat = feat[:, ah, aw]                       # (D,)
                anchor_label = label[ah, aw].item()

                # Extract local neighbourhood features and labels
                patch_feat = feat[:, ah - half : ah + half + 1, aw - half : aw + half + 1]  # (D, n, n)
                patch_label = label[ah - half : ah + half + 1, aw - half : aw + half + 1]   # (n, n)

                patch_feat_flat = patch_feat.reshape(D, -1).T  # (n*n, D)
                patch_label_flat = patch_label.reshape(-1)       # (n*n,)

                # Cosine similarity between anchor and all neighbours
                sim = F.cosine_similarity(
                    anchor_feat.unsqueeze(0), patch_feat_flat, dim=1
                ) / self.temperature  # (n*n,)

                # Positive mask: same class (exclude self at centre)
                pos_mask = (patch_label_flat == anchor_label)
                centre_idx = (self.neighborhood * self.neighborhood) // 2
                pos_mask[centre_idx] = False

                if pos_mask.sum() == 0:
                    continue  # No positives in window — skip

                # InfoNCE: for each positive, denominator is all neighbours
                # L = -mean_over_positives [ log( exp(sim_pos) / sum_all exp(sim) ) ]
                log_sum_exp = torch.logsumexp(sim, dim=0)  # scalar
                pos_sims = sim[pos_mask]
                loss_i = -(pos_sims - log_sum_exp).mean()

                total_loss = total_loss + loss_i
                valid_count += 1

        if valid_count == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        return total_loss / valid_count


# ======================================================================
# 3. Composite Loss (used during training)
# ======================================================================
class CompositeSegmentationLoss(nn.Module):
    """Combined loss: L_total = L_CE + λ · L_contrastive.

    During forward, it accepts the model's raw logits.  The feature map
    for contrastive loss is derived by L2-normalising the logits themselves
    (a lightweight proxy — in practice one could tap an intermediate layer).

    Args:
        class_weights: Per-class CE weights.
        contrastive_weight: λ scaling the contrastive term.
        temperature: Contrastive temperature τ.
        neighborhood: Contrastive local window size.
        num_samples: Anchors to sample per image.
    """

    def __init__(
        self,
        class_weights: Optional[List[float]] = None,
        contrastive_weight: float = 0.1,
        temperature: float = 0.07,
        neighborhood: int = 15,
        num_samples: int = 256,
    ):
        super().__init__()
        self.ce_loss = WeightedCrossEntropyLoss(class_weights=class_weights)
        self.contrastive_loss = LocalContrastiveLoss(
            temperature=temperature,
            neighborhood=neighborhood,
            num_samples=num_samples,
        )
        self.lam = contrastive_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict:
        """
        Returns dict with 'total', 'ce', and 'contrastive' loss values.
        """
        loss_ce = self.ce_loss(logits, targets)

        # Use L2-normalised logits as feature embeddings for contrastive loss
        features = F.normalize(logits, p=2, dim=1)
        loss_contrast = self.contrastive_loss(features, targets)

        loss_total = loss_ce + self.lam * loss_contrast

        return {
            "total": loss_total,
            "ce": loss_ce.detach(),
            "contrastive": loss_contrast.detach(),
        }
