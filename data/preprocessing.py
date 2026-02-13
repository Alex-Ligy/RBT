"""
Preprocessing utilities for the RBT segmentation pipeline.

Key component — **Centroid Sampling**:
    Instead of random crops, we locate the centroid of the minority class
    (Class 1: exposed aggregate / stripped area) and crop around it.
    This ensures the network sees sufficient minority-class examples during
    training, directly addressing the severe class imbalance.

Reference: Section 3.2 of the paper — "Centroid Sampling Strategy".
"""

import logging
from typing import Tuple, Optional

import cv2
import numpy as np
import albumentations as A

logger = logging.getLogger(__name__)


# ======================================================================
# Centroid Sampling Crop
# ======================================================================
class CentroidCropTransform:
    """Crop a (crop_size × crop_size) patch centred on the minority-class centroid.

    Algorithm:
        1. Compute binary mask for the minority class.
        2. If the class is present, calculate its centroid (centre of mass).
        3. Add random jitter to avoid overfitting to the exact centre.
        4. Extract a crop centred at the (jittered) centroid, clamped to
           image boundaries.
        5. If the minority class is absent, fall back to a random crop.

    Args:
        crop_size: Side length of the square crop.
        minority_class: Class index of the minority class.
        jitter: Maximum random displacement (in pixels) around the centroid.
        fallback_to_random: Whether to use a random crop when the minority
            class is not present in the image.
    """

    def __init__(
        self,
        crop_size: int = 512,
        minority_class: int = 1,
        jitter: int = 50,
        fallback_to_random: bool = True,
    ):
        self.crop_size = crop_size
        self.minority_class = minority_class
        self.jitter = jitter
        self.fallback_to_random = fallback_to_random

    def _compute_centroid(self, mask: np.ndarray) -> Optional[Tuple[int, int]]:
        """Compute the centroid (cy, cx) of *minority_class* pixels in the mask.

        Uses OpenCV moments for efficiency.  Returns None if the class is
        not present.
        """
        binary = (mask == self.minority_class).astype(np.uint8)
        moments = cv2.moments(binary)
        if moments["m00"] == 0:
            return None
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        return cy, cx

    def _crop_around(
        self, image: np.ndarray, mask: np.ndarray, cy: int, cx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract a crop_size × crop_size patch centred at (cy, cx)."""
        h, w = mask.shape[:2]
        half = self.crop_size // 2

        # Clamp to valid range
        y1 = max(0, cy - half)
        x1 = max(0, cx - half)
        y2 = y1 + self.crop_size
        x2 = x1 + self.crop_size

        # Shift back if we exceed image bounds
        if y2 > h:
            y2 = h
            y1 = max(0, h - self.crop_size)
        if x2 > w:
            x2 = w
            x1 = max(0, w - self.crop_size)

        crop_img = image[y1:y2, x1:x2]
        crop_msk = mask[y1:y2, x1:x2]

        # If image is smaller than crop_size, pad with zeros
        if crop_img.shape[0] < self.crop_size or crop_img.shape[1] < self.crop_size:
            crop_img = _pad_to_size(crop_img, self.crop_size)
            crop_msk = _pad_to_size(crop_msk, self.crop_size)

        return crop_img, crop_msk

    def _random_crop(
        self, image: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback: random crop when minority class is absent."""
        h, w = mask.shape[:2]
        max_y = max(0, h - self.crop_size)
        max_x = max(0, w - self.crop_size)
        y1 = np.random.randint(0, max_y + 1)
        x1 = np.random.randint(0, max_x + 1)
        return self._crop_around(image, mask, y1 + self.crop_size // 2, x1 + self.crop_size // 2)

    def __call__(
        self, image: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        centroid = self._compute_centroid(mask)

        if centroid is not None:
            cy, cx = centroid
            # Apply random jitter to avoid overfitting to exact centroid
            if self.jitter > 0:
                cy += np.random.randint(-self.jitter, self.jitter + 1)
                cx += np.random.randint(-self.jitter, self.jitter + 1)
            return self._crop_around(image, mask, cy, cx)

        if self.fallback_to_random:
            return self._random_crop(image, mask)

        # Last resort: centre crop
        h, w = mask.shape[:2]
        return self._crop_around(image, mask, h // 2, w // 2)


def _pad_to_size(arr: np.ndarray, size: int) -> np.ndarray:
    """Zero-pad array so that both spatial dims are at least *size*."""
    h, w = arr.shape[:2]
    pad_h = max(0, size - h)
    pad_w = max(0, size - w)
    if arr.ndim == 3:
        return np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
    return np.pad(arr, ((0, pad_h), (0, pad_w)), mode="constant")


# ======================================================================
# Albumentations augmentation pipelines
# ======================================================================

def get_train_augmentation(cfg: dict) -> A.Compose:
    """Build the training augmentation pipeline from config.

    Augmentations simulate real-world variability in RBT imaging:
        - Flips / rotations: aggregates are orientation-invariant.
        - Gaussian noise: camera sensor grain under lab lighting.
        - Colour jitter: slight illumination changes between samples.

    The final Resize guarantees a fixed spatial size for batching.
    """
    aug_cfg = cfg.get("augmentation", {})
    input_size = cfg["data"]["input_size"]
    transforms = []

    if aug_cfg.get("horizontal_flip", False):
        transforms.append(A.HorizontalFlip(p=0.5))

    if aug_cfg.get("vertical_flip", False):
        transforms.append(A.VerticalFlip(p=0.5))

    if aug_cfg.get("random_rotate_90", False):
        transforms.append(A.RandomRotate90(p=0.5))

    noise_cfg = aug_cfg.get("gaussian_noise", {})
    if noise_cfg.get("enabled", False):
        var_limit = noise_cfg.get("var_limit", [10.0, 50.0])
        # albumentations >=2.0 uses std_range (values in [0, 1]) instead of
        # var_limit (pixel-level variance).  Convert: std ≈ sqrt(var) / 255.
        import inspect
        gauss_params = inspect.signature(A.GaussNoise.__init__).parameters
        if "std_range" in gauss_params:
            std_lo = min(1.0, (var_limit[0] ** 0.5) / 255.0)
            std_hi = min(1.0, (var_limit[1] ** 0.5) / 255.0)
            transforms.append(A.GaussNoise(std_range=(std_lo, std_hi), p=0.3))
        else:
            transforms.append(A.GaussNoise(var_limit=tuple(var_limit), p=0.3))

    jitter_cfg = aug_cfg.get("color_jitter", {})
    if jitter_cfg.get("enabled", False):
        transforms.append(
            A.ColorJitter(
                brightness=jitter_cfg.get("brightness", 0.2),
                contrast=jitter_cfg.get("contrast", 0.2),
                saturation=0.0,
                hue=0.0,
                p=0.3,
            )
        )

    # Final resize to guarantee consistent tensor size
    transforms.append(A.Resize(height=input_size, width=input_size))

    return A.Compose(transforms)


def get_val_augmentation(cfg: dict) -> A.Compose:
    """Validation pipeline: deterministic resize only."""
    input_size = cfg["data"]["input_size"]
    return A.Compose([A.Resize(height=input_size, width=input_size)])
