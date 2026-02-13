"""
ResNet-101 Backbone with Dilated (Atrous) Convolutions.

Reference:
    He et al., "Deep Residual Learning for Image Recognition", CVPR 2016.
    Chen et al., "Rethinking Atrous Convolution for Semantic Image
    Segmentation", arXiv 2017 — introduced dilated ResNet for dense prediction.

For semantic segmentation we modify the standard ResNet by:
    1. Replacing stride-2 convolutions in later layers with dilated convolutions
       to maintain spatial resolution (output stride = 16 or 8).
    2. Returning both low-level features (from layer1, stride 4) and high-level
       features (from layer4) needed by the DeepLabv3+ decoder.
"""

from typing import Tuple

import torch
import torch.nn as nn
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.models.resnet import Bottleneck


def _replace_stride_with_dilation(
    layer: nn.Sequential, dilation: int
) -> nn.Sequential:
    """Replace stride-2 with dilation in every Bottleneck of *layer*."""
    for module in layer.modules():
        if isinstance(module, Bottleneck):
            # The 3×3 conv in the bottleneck
            if module.conv2.stride == (2, 2):
                module.conv2.stride = (1, 1)
                module.conv2.dilation = (dilation, dilation)
                module.conv2.padding = (dilation, dilation)
            # The optional down-sampling shortcut
            if module.downsample is not None:
                for sub in module.downsample.modules():
                    if isinstance(sub, nn.Conv2d) and sub.stride == (2, 2):
                        sub.stride = (1, 1)
    return layer


class ResNet101Backbone(nn.Module):
    """Modified ResNet-101 that outputs multi-scale feature maps.

    Returns:
        low_level_feat : feature map from layer1 (stride 4, 256 channels)
        high_level_feat: feature map from layer4 (stride ``output_stride``)
    """

    def __init__(self, output_stride: int = 16, pretrained: bool = True):
        super().__init__()
        assert output_stride in (8, 16), "output_stride must be 8 or 16"

        weights = ResNet101_Weights.DEFAULT if pretrained else None
        base = resnet101(weights=weights)

        # Stem: conv1 + bn1 + relu + maxpool → stride 4
        self.stem = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool
        )

        # layer1 → stride 4  (low-level features for decoder)
        self.layer1 = base.layer1   # 256 channels

        # layer2 → stride 8
        self.layer2 = base.layer2   # 512 channels

        # layer3 / layer4 — apply dilation depending on output_stride
        if output_stride == 16:
            self.layer3 = base.layer3                                    # stride 16
            self.layer4 = _replace_stride_with_dilation(base.layer4, dilation=2)
        else:  # output_stride == 8
            self.layer3 = _replace_stride_with_dilation(base.layer3, dilation=2)
            self.layer4 = _replace_stride_with_dilation(base.layer4, dilation=4)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        low_level = self.layer1(x)       # (B, 256, H/4, W/4)
        x = self.layer2(low_level)
        x = self.layer3(x)
        high_level = self.layer4(x)      # (B, 2048, H/OS, W/OS)
        return low_level, high_level


def resnet101_backbone(output_stride: int = 16, pretrained: bool = True) -> ResNet101Backbone:
    """Factory function for convenience."""
    return ResNet101Backbone(output_stride=output_stride, pretrained=pretrained)
