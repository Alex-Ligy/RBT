"""
DeepLabv3+ Semantic Segmentation Network.

Reference:
    Chen et al., "Encoder-Decoder with Atrous Separable Convolution for
    Semantic Image Segmentation", ECCV 2018.

Architecture overview:
    Encoder
    ├── ResNet-101 backbone (dilated, output stride 16 or 8)
    └── ASPP module (multi-scale atrous convolutions)
    Decoder
    ├── Low-level feature projection (1×1 conv, 48 channels)
    ├── Concatenation with upsampled ASPP output
    └── 3×3 refinement convolutions → final classifier

The ASPP module uses rates {6, 12, 18} (for OS=16) or {12, 24, 36} (OS=8)
to capture multi-scale context — essential for jointly recognising large
aggregate surfaces and fine boundary detail between bitumen and stone.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet_backbone import resnet101_backbone


# ======================================================================
# ASPP — Atrous Spatial Pyramid Pooling
# ======================================================================
class _ASPPConv(nn.Sequential):
    """Single ASPP branch: atrous conv → BN → ReLU."""

    def __init__(self, in_ch: int, out_ch: int, dilation: int):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class _ASPPPooling(nn.Module):
    """Image-level (global) pooling branch of ASPP."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[2:]
        out = self.gap(x)          # (B, C, 1, 1)
        out = self.conv(out)
        # Upsample *before* BN so spatial dims > 1 (avoids BN error at B=1)
        out = F.interpolate(out, size=size, mode="bilinear", align_corners=False)
        out = self.bn(out)
        out = self.relu(out)
        return out


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling.

    Branches:
        1. 1×1 convolution
        2-4. 3×3 convolutions with dilation rates (e.g., 6, 12, 18)
        5. Image-level global average pooling

    All branches are concatenated and projected to 256 channels.
    """

    def __init__(self, in_ch: int, out_ch: int = 256, rates: List[int] = None):
        super().__init__()
        if rates is None:
            rates = [6, 12, 18]

        modules = [
            # Branch 1: 1×1 conv
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        ]
        # Branches 2-4: atrous convolutions at different rates
        for rate in rates:
            modules.append(_ASPPConv(in_ch, out_ch, rate))
        # Branch 5: global pooling
        modules.append(_ASPPPooling(in_ch, out_ch))

        self.branches = nn.ModuleList(modules)

        # Projection after concatenation
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * (len(rates) + 2), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_outs = [branch(x) for branch in self.branches]
        x = torch.cat(branch_outs, dim=1)
        return self.project(x)


# ======================================================================
# DeepLabv3+ Decoder
# ======================================================================
class Decoder(nn.Module):
    """DeepLabv3+ decoder that fuses low-level and ASPP features.

    Steps:
        1. Project low-level features (256 ch) to 48 channels.
        2. Upsample ASPP output to match low-level spatial size.
        3. Concatenate → two 3×3 convolutions → final 1×1 classifier.
    """

    def __init__(
        self,
        low_level_ch: int = 256,
        aspp_ch: int = 256,
        num_classes: int = 3,
    ):
        super().__init__()
        # 1×1 projection for low-level features (reduce channel noise)
        self.low_level_proj = nn.Sequential(
            nn.Conv2d(low_level_ch, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        # Refinement convolutions after concatenation
        self.refine = nn.Sequential(
            nn.Conv2d(aspp_ch + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        # Final pixel-wise classifier
        self.classifier = nn.Conv2d(256, num_classes, 1)

    def forward(
        self,
        aspp_out: torch.Tensor,
        low_level_feat: torch.Tensor,
    ) -> torch.Tensor:
        # Project low-level
        low = self.low_level_proj(low_level_feat)  # (B, 48, H/4, W/4)

        # Upsample ASPP output to low-level resolution
        aspp_up = F.interpolate(
            aspp_out, size=low.shape[2:], mode="bilinear", align_corners=False
        )

        # Concatenate and refine
        fused = torch.cat([aspp_up, low], dim=1)  # (B, 256+48, H/4, W/4)
        fused = self.refine(fused)

        return self.classifier(fused)  # (B, num_classes, H/4, W/4)


# ======================================================================
# Full DeepLabv3+ Model
# ======================================================================
class DeepLabV3Plus(nn.Module):
    """DeepLabv3+ with ResNet-101 backbone for RBT segmentation.

    Forward pass returns logits at the original input resolution.

    Args:
        num_classes: Number of semantic classes (default 3).
        output_stride: Backbone output stride (16 or 8).
        pretrained_backbone: Use ImageNet-pretrained ResNet-101 weights.
    """

    def __init__(
        self,
        num_classes: int = 3,
        output_stride: int = 16,
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        self.backbone = resnet101_backbone(
            output_stride=output_stride, pretrained=pretrained_backbone
        )

        # ASPP rates depend on output stride
        aspp_rates = [6, 12, 18] if output_stride == 16 else [12, 24, 36]
        self.aspp = ASPP(in_ch=2048, out_ch=256, rates=aspp_rates)

        self.decoder = Decoder(
            low_level_ch=256, aspp_ch=256, num_classes=num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[2:]  # (H, W)

        # Encoder
        low_level_feat, high_level_feat = self.backbone(x)
        aspp_out = self.aspp(high_level_feat)

        # Decoder
        logits = self.decoder(aspp_out, low_level_feat)  # (B, C, H/4, W/4)

        # Upsample to original resolution
        logits = F.interpolate(
            logits, size=input_size, mode="bilinear", align_corners=False
        )

        return logits
