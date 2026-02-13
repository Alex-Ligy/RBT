"""
Evaluation metrics for RBT semantic segmentation.

Standard metrics:
    - Overall Accuracy (OA): fraction of correctly classified pixels.
    - Mean Intersection-over-Union (mIoU): averaged across all classes.
    - Per-class IoU.

Engineering metric (EN 12697-11):
    - Bitumen Coverage (%):
        Coverage = Pixels_Bitumen / (Pixels_Bitumen + Pixels_Aggregate) × 100
      Background pixels are *excluded* from the denominator, isolating the
      stone surface for engineering evaluation.
"""

from typing import Dict

import numpy as np
import torch


class SegmentationMetrics:
    """Accumulates a confusion matrix and derives segmentation metrics.

    Args:
        num_classes: Total number of classes (including background).
        class_names: Optional readable names for logging.
    """

    def __init__(self, num_classes: int = 3, class_names=None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.confusion_matrix = np.zeros(
            (num_classes, num_classes), dtype=np.int64
        )

    def reset(self):
        self.confusion_matrix[:] = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """Update the confusion matrix with a batch of predictions.

        Args:
            preds:   (B, H, W) predicted class indices (argmax of logits).
            targets: (B, H, W) ground-truth class indices.
        """
        preds_np = preds.cpu().numpy().astype(np.int64).ravel()
        targets_np = targets.cpu().numpy().astype(np.int64).ravel()

        # Filter out ignore-label pixels (255)
        valid = targets_np < self.num_classes
        preds_np = preds_np[valid]
        targets_np = targets_np[valid]

        # Accumulate into confusion matrix
        indices = targets_np * self.num_classes + preds_np
        cm_flat = np.bincount(indices, minlength=self.num_classes ** 2)
        self.confusion_matrix += cm_flat.reshape(self.num_classes, self.num_classes)

    # ------------------------------------------------------------------
    # Derived metrics
    # ------------------------------------------------------------------
    def overall_accuracy(self) -> float:
        """OA = correct / total."""
        correct = np.diag(self.confusion_matrix).sum()
        total = self.confusion_matrix.sum()
        return float(correct / total) if total > 0 else 0.0

    def per_class_iou(self) -> np.ndarray:
        """IoU_c = TP_c / (TP_c + FP_c + FN_c) for each class c."""
        tp = np.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(axis=0) - tp
        fn = self.confusion_matrix.sum(axis=1) - tp
        denom = tp + fp + fn
        iou = np.where(denom > 0, tp / denom, 0.0)
        return iou

    def mean_iou(self) -> float:
        """mIoU averaged over all classes."""
        return float(self.per_class_iou().mean())

    def summary(self) -> Dict[str, float]:
        """Return a dict of all metrics for logging."""
        iou = self.per_class_iou()
        result = {
            "overall_accuracy": self.overall_accuracy(),
            "mean_iou": self.mean_iou(),
        }
        for i, name in enumerate(self.class_names):
            result[f"iou_{name}"] = float(iou[i])
        return result


# ======================================================================
# Engineering metric: Bitumen Coverage
# ======================================================================
def calculate_bitumen_coverage(
    mask: np.ndarray,
    aggregate_class: int = 1,
    bitumen_class: int = 2,
) -> float:
    """Calculate the degree of bitumen coverage per EN 12697-11.

    Formula:
        Coverage(%) = Pixels_Bitumen / (Pixels_Bitumen + Pixels_Aggregate) × 100

    Background (class 0) pixels are **excluded** from both numerator and
    denominator, so the metric reflects only the stone surface.

    Args:
        mask: (H, W) integer class-index array (prediction or ground truth).
        aggregate_class: Class index for exposed aggregate (default 1).
        bitumen_class: Class index for bitumen coating (default 2).

    Returns:
        Coverage percentage in [0, 100].  Returns 0.0 if no relevant pixels.
    """
    n_aggregate = int((mask == aggregate_class).sum())
    n_bitumen = int((mask == bitumen_class).sum())
    total = n_aggregate + n_bitumen
    if total == 0:
        return 0.0
    return (n_bitumen / total) * 100.0
