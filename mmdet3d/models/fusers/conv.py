from typing import List

import torch
from torch import nn

from mmdet3d.models.builder import FUSERS

__all__ = ["ConvFuser"]


@FUSERS.register_module()
class ConvFuser(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        self.in_channels = in_channels # [80, 256]
        self.out_channels = out_channels # 256
        super().__init__(
            nn.Conv2d(sum(in_channels), out_channels, 3, padding=1, bias=False), # 336-->256
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        # inputs:(2, 80, 180, 180)和(2, 256, 180, 180) --> (2, 336, 180, 180)
        return super().forward(torch.cat(inputs, dim=1)) # (2, 256, 180, 180)
