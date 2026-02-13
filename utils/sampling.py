"""
Centroid Sampling utility — factory function for building the
CentroidCropTransform from a config dict.

This module acts as a thin bridge between the YAML config and the
CentroidCropTransform implemented in data/preprocessing.py.

Reference: Section 3.2 of the paper — "Centroid Sampling Strategy".
"""

from typing import Optional

from data.preprocessing import CentroidCropTransform


def build_centroid_crop(cfg: dict) -> Optional[CentroidCropTransform]:
    """Construct a CentroidCropTransform from the config, or None if disabled.

    Args:
        cfg: Full parsed YAML config dict.

    Returns:
        CentroidCropTransform instance, or None if centroid sampling is
        disabled in the config.
    """
    cs_cfg = cfg.get("centroid_sampling", {})
    if not cs_cfg.get("enabled", False):
        return None

    return CentroidCropTransform(
        crop_size=cs_cfg.get("crop_size", cfg["data"]["input_size"]),
        minority_class=cs_cfg.get("minority_class", 1),
        jitter=cs_cfg.get("centroid_jitter", 50),
        fallback_to_random=cs_cfg.get("fallback_to_random", True),
    )
