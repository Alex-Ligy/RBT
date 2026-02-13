"""
Inference and evaluation script for the RBT segmentation model.

Usage:
    # Evaluate on a single image
    python evaluate.py --config configs/config.yaml \\
                       --checkpoint checkpoints/best_model.pth \\
                       --image test_image.jpg \\
                       --output result.png

    # Evaluate on a directory of images
    python evaluate.py --config configs/config.yaml \\
                       --checkpoint checkpoints/best_model.pth \\
                       --image_dir test_images/ \\
                       --output_dir results/

Outputs:
    1. Segmentation mask overlay visualisation.
    2. Bitumen Coverage Percentage (EN 12697-11).
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.deeplabv3_plus import DeepLabV3Plus
from utils.metrics import SegmentationMetrics, calculate_bitumen_coverage

logger = logging.getLogger("RBT-Eval")

# Colour map for overlay visualisation
# Class 0 (Background): transparent / black
# Class 1 (Aggregate):  Red — visually highlights stripping
# Class 2 (Bitumen):    Green — indicates successful coating
CLASS_COLOURS = np.array([
    [0,   0,   0],    # Background
    [255, 0,   0],    # Aggregate (stripped) — red
    [0,   200, 0],    # Bitumen (coated) — green
], dtype=np.uint8)


# ======================================================================
# Sliding Window Inference
# ======================================================================
def sliding_window_inference(
    model: torch.nn.Module,
    image: np.ndarray,
    window_size: int = 512,
    stride: int = 256,
    num_classes: int = 3,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """Run inference on a full-resolution image using overlapping patches.

    Handles images larger than the training crop size by:
        1. Extracting overlapping windows.
        2. Predicting each window.
        3. Averaging overlapping predictions for smooth boundaries.

    Args:
        model: Trained DeepLabV3+ model.
        image: (H, W, 3) uint8 RGB image.
        window_size: Patch size (should match training input_size).
        stride: Step between windows (< window_size → overlap).
        num_classes: Number of output classes.
        device: Torch device.

    Returns:
        (H, W) int array of predicted class indices.
    """
    model.eval()
    h, w = image.shape[:2]

    # Pad image so that it is divisible by stride
    pad_h = (window_size - h % window_size) % window_size if h % stride != 0 else 0
    pad_w = (window_size - w % window_size) % window_size if w % stride != 0 else 0
    # Ensure image is at least window_size
    pad_h = max(pad_h, window_size - h) if h < window_size else pad_h
    pad_w = max(pad_w, window_size - w) if w < window_size else pad_w

    padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    ph, pw = padded.shape[:2]

    # Accumulation buffers
    logit_sum = np.zeros((num_classes, ph, pw), dtype=np.float32)
    count = np.zeros((ph, pw), dtype=np.float32)

    with torch.no_grad():
        for y in range(0, ph - window_size + 1, stride):
            for x in range(0, pw - window_size + 1, stride):
                patch = padded[y : y + window_size, x : x + window_size]
                tensor = (
                    torch.from_numpy(patch.transpose(2, 0, 1).astype(np.float32) / 255.0)
                    .unsqueeze(0)
                    .to(device)
                )
                logits = model(tensor)  # (1, C, H, W)
                logits_np = logits.squeeze(0).cpu().numpy()

                logit_sum[:, y : y + window_size, x : x + window_size] += logits_np
                count[y : y + window_size, x : x + window_size] += 1.0

    # Average and argmax
    count = np.maximum(count, 1.0)
    logit_avg = logit_sum / count[np.newaxis, :, :]
    pred = logit_avg.argmax(axis=0)

    # Remove padding
    return pred[:h, :w]


# ======================================================================
# Visualisation
# ======================================================================
def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend a colour-coded segmentation mask on top of the original image.

    Args:
        image: (H, W, 3) uint8 RGB image.
        mask:  (H, W) predicted class indices.
        alpha: Transparency of the overlay.

    Returns:
        (H, W, 3) blended image.
    """
    colour_mask = CLASS_COLOURS[mask]  # (H, W, 3)
    # Only overlay non-background pixels
    fg = mask > 0
    blended = image.copy()
    blended[fg] = (
        (1 - alpha) * image[fg].astype(np.float32)
        + alpha * colour_mask[fg].astype(np.float32)
    ).astype(np.uint8)
    return blended


# ======================================================================
# Single-image evaluation
# ======================================================================
def evaluate_image(
    model: torch.nn.Module,
    image_path: str,
    cfg: dict,
    device: torch.device,
) -> dict:
    """Run inference on a single image and compute coverage.

    Returns dict with keys: 'mask', 'overlay', 'coverage'.
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    eval_cfg = cfg.get("evaluation", {})
    sw_cfg = eval_cfg.get("sliding_window", {})

    if sw_cfg.get("enabled", True):
        mask = sliding_window_inference(
            model, image,
            window_size=sw_cfg.get("window_size", cfg["data"]["input_size"]),
            stride=sw_cfg.get("stride", cfg["data"]["input_size"] // 2),
            num_classes=cfg["model"]["num_classes"],
            device=device,
        )
    else:
        # Simple resize → predict → resize back
        input_size = cfg["data"]["input_size"]
        resized = cv2.resize(image, (input_size, input_size))
        tensor = (
            torch.from_numpy(resized.transpose(2, 0, 1).astype(np.float32) / 255.0)
            .unsqueeze(0)
            .to(device)
        )
        with torch.no_grad():
            logits = model(tensor)
        mask_small = logits.argmax(dim=1).squeeze(0).cpu().numpy()
        mask = cv2.resize(
            mask_small.astype(np.uint8), (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    alpha = eval_cfg.get("overlay_alpha", 0.5)
    overlay = overlay_mask(image, mask, alpha=alpha)
    coverage = calculate_bitumen_coverage(mask)

    return {"mask": mask, "overlay": overlay, "coverage": coverage}


# ======================================================================
# Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="Evaluate RBT Segmentation Model")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--image", type=str, default=None, help="Single image path")
    parser.add_argument("--image_dir", type=str, default=None, help="Directory of images")
    parser.add_argument("--output", type=str, default="result.png", help="Output path (single image)")
    parser.add_argument("--output_dir", type=str, default="results", help="Output dir (batch)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")

    # Load model
    model = DeepLabV3Plus(
        num_classes=cfg["model"]["num_classes"],
        output_stride=cfg["model"]["output_stride"],
        pretrained_backbone=False,  # Load weights from checkpoint
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info(
        "Loaded checkpoint from epoch %d (mIoU=%.4f)",
        checkpoint.get("epoch", -1), checkpoint.get("best_miou", -1),
    )

    # Determine image list
    image_paths = []
    if args.image:
        image_paths.append(args.image)
    elif args.image_dir:
        img_dir = Path(args.image_dir)
        image_paths = sorted(
            str(p) for p in img_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".bmp"}
        )
    else:
        parser.error("Provide either --image or --image_dir")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in image_paths:
        logger.info("Processing: %s", img_path)
        result = evaluate_image(model, img_path, cfg, device)

        coverage = result["coverage"]
        logger.info("Result: %.1f%% Bitumen Coverage", coverage)

        # Save overlay
        if len(image_paths) == 1:
            out_path = args.output
        else:
            out_path = str(out_dir / (Path(img_path).stem + "_result.png"))

        overlay_bgr = cv2.cvtColor(result["overlay"], cv2.COLOR_RGB2BGR)

        # Add coverage text to image
        text = f"Coverage: {coverage:.1f}%"
        cv2.putText(
            overlay_bgr, text, (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA,
        )
        cv2.imwrite(out_path, overlay_bgr)
        logger.info("Saved → %s", out_path)

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
