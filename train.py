"""
Training script for RBT Semantic Segmentation.

Usage:
    python train.py --config configs/config.yaml

Implements:
    - Poly learning-rate schedule:  lr = lr_base × (1 − iter/max_iter)^power
    - Composite loss: Weighted CE + λ · Local Contrastive Loss
    - Centroid Sampling for minority-class focus
    - Best-model checkpointing based on validation mIoU
    - TensorBoard logging of losses and metrics
"""

import os
import sys
import logging
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.deeplabv3_plus import DeepLabV3Plus
from data.dataset import RBTDataset
from data.preprocessing import CentroidCropTransform, get_train_augmentation, get_val_augmentation
from utils.losses import CompositeSegmentationLoss
from utils.metrics import SegmentationMetrics
from utils.sampling import build_centroid_crop


# ======================================================================
# Utilities
# ======================================================================
def set_seed(seed: int):
    """Reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def poly_lr_lambda(current_iter: int, max_iter: int, power: float = 0.9):
    """Poly learning rate decay factor."""
    return (1 - current_iter / max_iter) ** power


def build_dataloaders(cfg: dict):
    """Create train/val DataLoaders with centroid sampling + augmentation."""
    centroid_crop = build_centroid_crop(cfg)
    train_aug = get_train_augmentation(cfg)
    val_aug = get_val_augmentation(cfg)

    full_dataset = RBTDataset(
        root_dir=cfg["data"]["root_dir"],
        image_dir=cfg["data"]["image_dir"],
        mask_dir=cfg["data"]["mask_dir"],
        centroid_crop=centroid_crop,
        transform=train_aug,
    )

    # Split into train / val
    n_total = len(full_dataset)
    n_train = int(n_total * cfg["data"]["train_split"])
    n_val = n_total - n_train

    if n_total == 0:
        logging.warning("Dataset is empty — using synthetic data for testing.")
        return _build_synthetic_loaders(cfg)

    train_set, val_set = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.get("seed", 42)),
    )

    # Override transform for validation subset (no augmentation, no centroid crop)
    val_dataset = RBTDataset(
        root_dir=cfg["data"]["root_dir"],
        image_dir=cfg["data"]["image_dir"],
        mask_dir=cfg["data"]["mask_dir"],
        centroid_crop=None,
        transform=val_aug,
    )
    # Restrict to validation indices
    val_dataset.samples = [full_dataset.samples[i] for i in val_set.indices]

    train_loader = DataLoader(
        train_set,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"].get("pin_memory", True),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"].get("pin_memory", True),
    )
    return train_loader, val_loader


class SyntheticRBTDataset(torch.utils.data.Dataset):
    """Tiny synthetic dataset for smoke-testing the pipeline."""

    def __init__(self, size: int = 16, input_size: int = 512, num_classes: int = 3):
        self.size = size
        self.input_size = input_size
        self.num_classes = num_classes

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image = torch.randn(3, self.input_size, self.input_size)
        mask = torch.randint(0, self.num_classes, (self.input_size, self.input_size))
        return image, mask


def _build_synthetic_loaders(cfg: dict):
    input_size = cfg["data"]["input_size"]
    num_classes = cfg["model"]["num_classes"]
    batch_size = cfg["training"]["batch_size"]

    train_set = SyntheticRBTDataset(size=16, input_size=input_size, num_classes=num_classes)
    val_set = SyntheticRBTDataset(size=4, input_size=input_size, num_classes=num_classes)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


# ======================================================================
# Training & Validation Loops
# ======================================================================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: CompositeSegmentationLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    global_step: int,
    log_interval: int = 10,
) -> int:
    model.train()
    running = {"total": 0.0, "ce": 0.0, "contrastive": 0.0}

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        losses = criterion(logits, masks)

        optimizer.zero_grad()
        losses["total"].backward()
        optimizer.step()

        # Accumulate
        for k in running:
            running[k] += losses[k].item()

        # Logging
        if (batch_idx + 1) % log_interval == 0:
            n = batch_idx + 1
            pbar.set_postfix(
                loss=f"{running['total']/n:.4f}",
                ce=f"{running['ce']/n:.4f}",
                cl=f"{running['contrastive']/n:.4f}",
            )

        writer.add_scalar("train/loss_total", losses["total"].item(), global_step)
        writer.add_scalar("train/loss_ce", losses["ce"].item(), global_step)
        writer.add_scalar("train/loss_contrastive", losses["contrastive"].item(), global_step)
        global_step += 1

    return global_step


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: CompositeSegmentationLoss,
    device: torch.device,
    num_classes: int,
) -> dict:
    model.eval()
    metrics = SegmentationMetrics(
        num_classes=num_classes,
        class_names=["background", "aggregate", "bitumen"],
    )
    total_loss = 0.0
    n_batches = 0

    for images, masks in tqdm(loader, desc="Validating", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        losses = criterion(logits, masks)
        total_loss += losses["total"].item()
        n_batches += 1

        preds = logits.argmax(dim=1)
        metrics.update(preds, masks)

    result = metrics.summary()
    result["val_loss"] = total_loss / max(n_batches, 1)
    return result


# ======================================================================
# Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="Train RBT Segmentation Model")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="Path to YAML config"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    # Logging setup
    log_dir = Path(cfg["logging"]["log_dir"])
    ckpt_dir = Path(cfg["logging"]["checkpoint_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "train.log"),
        ],
    )
    logger = logging.getLogger("RBT-Train")
    logger.info("Configuration:\n%s", yaml.dump(cfg, default_flow_style=False))

    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Model
    model = DeepLabV3Plus(
        num_classes=cfg["model"]["num_classes"],
        output_stride=cfg["model"]["output_stride"],
        pretrained_backbone=cfg["model"]["pretrained_backbone"],
    ).to(device)
    logger.info("Model: DeepLabV3+ with ResNet-101 backbone (OS=%d)", cfg["model"]["output_stride"])

    # Data
    train_loader, val_loader = build_dataloaders(cfg)
    logger.info("Train batches: %d | Val batches: %d", len(train_loader), len(val_loader))

    # Loss
    loss_cfg = cfg["training"]["loss"]
    criterion = CompositeSegmentationLoss(
        class_weights=loss_cfg.get("ce_weight"),
        contrastive_weight=loss_cfg.get("contrastive_weight", 0.1),
        temperature=loss_cfg.get("contrastive_temperature", 0.07),
        neighborhood=loss_cfg.get("contrastive_neighborhood", 15),
        num_samples=loss_cfg.get("contrastive_num_samples", 256),
    ).to(device)

    # Optimizer
    opt_cfg = cfg["training"]["optimizer"]
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=opt_cfg["lr"],
        momentum=opt_cfg["momentum"],
        weight_decay=opt_cfg["weight_decay"],
    )

    # LR scheduler (poly)
    total_iters = cfg["training"]["epochs"] * len(train_loader)
    lr_power = cfg["training"]["lr_scheduler"].get("power", 0.9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda it: poly_lr_lambda(it, total_iters, lr_power),
    )

    # TensorBoard
    writer = SummaryWriter(log_dir=str(log_dir / "tensorboard"))

    # Training loop
    best_miou = 0.0
    global_step = 0
    log_interval = cfg["logging"].get("log_interval", 10)

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        logger.info("=== Epoch %d/%d ===", epoch, cfg["training"]["epochs"])

        global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch, writer, global_step, log_interval,
        )
        scheduler.step()

        # Validation
        val_metrics = validate(model, val_loader, criterion, device, cfg["model"]["num_classes"])
        logger.info(
            "Val — mIoU: %.4f | OA: %.4f | Loss: %.4f",
            val_metrics["mean_iou"],
            val_metrics["overall_accuracy"],
            val_metrics["val_loss"],
        )

        # TensorBoard
        writer.add_scalar("val/miou", val_metrics["mean_iou"], epoch)
        writer.add_scalar("val/oa", val_metrics["overall_accuracy"], epoch)
        writer.add_scalar("val/loss", val_metrics["val_loss"], epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        # Checkpointing — save best model
        if val_metrics["mean_iou"] > best_miou:
            best_miou = val_metrics["mean_iou"]
            ckpt_path = ckpt_dir / "best_model.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_miou": best_miou,
                    "config": cfg,
                },
                ckpt_path,
            )
            logger.info("Saved best model (mIoU=%.4f) → %s", best_miou, ckpt_path)

    writer.close()
    logger.info("Training complete. Best mIoU: %.4f", best_miou)


if __name__ == "__main__":
    main()
