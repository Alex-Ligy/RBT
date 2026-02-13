"""
Custom Dataset for Rolling Bottle Test (RBT) Image Segmentation.

Loads pairs of (image, mask) for semantic segmentation of bitumen-coated
aggregates according to EN 12697-11.

Classes:
    0 - Background
    1 - Aggregate / Stone (exposed, stripped area — minority class)
    2 - Bitumen (coated area)
"""

import os
import logging
from pathlib import Path
from typing import Optional, Callable, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class RBTDataset(Dataset):
    """Rolling Bottle Test semantic segmentation dataset.

    Expects directory layout:
        root_dir/
            images/   *.png or *.jpg
            masks/    *.png  (single-channel, pixel values = class indices)

    Args:
        root_dir: Path to dataset root.
        image_dir: Subdirectory name for images.
        mask_dir: Subdirectory name for masks.
        file_list: Optional explicit list of stem names to include.
        transform: Albumentations-style transform applied to image+mask.
        centroid_crop: Optional centroid-based cropping callable applied
            *before* augmentation (see preprocessing.CentroidCropTransform).
    """

    EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

    def __init__(
        self,
        root_dir: str,
        image_dir: str = "images",
        mask_dir: str = "masks",
        file_list: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        centroid_crop: Optional[Callable] = None,
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / image_dir
        self.mask_dir = self.root_dir / mask_dir
        self.transform = transform
        self.centroid_crop = centroid_crop

        # Discover image / mask pairs
        self.samples: List[Tuple[Path, Path]] = self._discover_pairs(file_list)
        logger.info(
            "RBTDataset initialised: %d samples from %s", len(self.samples), root_dir
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _discover_pairs(self, file_list: Optional[List[str]]) -> List[Tuple[Path, Path]]:
        """Pair each image with its corresponding mask by stem name."""
        if not self.image_dir.is_dir():
            logger.warning("Image directory does not exist: %s", self.image_dir)
            return []

        # Build a lookup: stem -> mask_path
        mask_lookup = {}
        if self.mask_dir.is_dir():
            for p in self.mask_dir.iterdir():
                if p.suffix.lower() in self.EXTENSIONS:
                    mask_lookup[p.stem] = p

        pairs = []
        for img_path in sorted(self.image_dir.iterdir()):
            if img_path.suffix.lower() not in self.EXTENSIONS:
                continue
            stem = img_path.stem
            if file_list is not None and stem not in file_list:
                continue
            mask_path = mask_lookup.get(stem)
            if mask_path is None:
                logger.warning("No mask found for image %s — skipping.", img_path.name)
                continue
            pairs.append((img_path, mask_path))

        return pairs

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.samples[idx]

        # Read image as BGR → RGB (uint8)
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read mask as single-channel (uint8, pixel values = class ids)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # --- Centroid-based cropping (applied before augmentation) ---
        if self.centroid_crop is not None:
            image, mask = self.centroid_crop(image, mask)

        # --- Albumentations augmentation ---
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Convert to tensors
        # image: (H, W, 3) uint8 → (3, H, W) float32 [0, 1]
        image = torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32) / 255.0)
        # mask: (H, W) int64
        mask = torch.from_numpy(mask.astype(np.int64))

        return image, mask
