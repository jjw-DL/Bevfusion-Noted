from typing import Tuple

import torch
from mmcv.runner import force_fp32
from torch import nn

from mmdet3d.models.builder import VTRANSFORMS

from .base import BaseDepthTransform

__all__ = ["DepthLSSTransform"]


@VTRANSFORMS.register_module()
class DepthLSSTransform(BaseDepthTransform):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        downsample: int = 1,
    ) -> None:
        super().__init__(
            in_channels=in_channels, # 256
            out_channels=out_channels, # 80
            image_size=image_size, # (256, 704)
            feature_size=feature_size, # (32, 88)
            xbound=xbound, # [-54.0, 54.0, 0.3]
            ybound=ybound, # [-54.0, 54.0, 0.3]
            zbound=zbound, # [-10.0, 10.0, 20.0]
            dbound=dbound, # [1.0, 60.0, 0.5]
        )
        self.dtransform = nn.Sequential(
            nn.Conv2d(1, 8, 1), # 1 --> 8
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 32, 5, stride=4, padding=2), # 8 --> 32
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2), # 32 --> 64
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        ) # 将深度从1-->64
        self.depthnet = nn.Sequential(
            nn.Conv2d(in_channels + 64, in_channels, 3, padding=1), # 256+64 --> 256 
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1), # 256-->256
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, self.D + self.C, 1), # 256 --> 118 + 88 = 198
        ) # 对拼接后的特征进一步提取信息, 获取深度和类别特征
        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False), # 80 --> 80
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels, # 80
                    out_channels, # 80
                    3,
                    stride=downsample, # 2
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False), # 80 --> 80
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

    @force_fp32()
    def get_cam_feats(self, x, d):
        B, N, C, fH, fW = x.shape # 2, 6, 256, 32, 88
        # d:(2, 6, 1, 256, 704) --> (12, 1, 256, 704)
        d = d.view(B * N, *d.shape[2:]) # lidar点在相机投影位置处的深度，没有投影的位置深度为0
        x = x.view(B * N, C, fH, fW) # (12, 256, 32, 88)

        d = self.dtransform(d) # (12, 64, 32, 88) 对投影深度信息进行编码，深度增加到64, 长宽缩小8倍，
        x = torch.cat([d, x], dim=1) # 将深度信息与语意信息拼接 --> (12, 320, 32, 88)
        x = self.depthnet(x) # 320-->256-->198: (12, 198, 32, 88) 产生深度(118)和上下文信息(80)

        depth = x[:, : self.D].softmax(dim=1) # 计算深度分布 (12, 118, 32, 88)
        # (12, 1, 118, 32, 88) * (12, 80, 1, 32, 88) --> (12, 80, 118, 32, 88)
        # 将深度分布与上下文信息进行外积(每个深度分布都包含80维的上下文特征)
        x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)

        x = x.view(B, N, self.C, self.D, fH, fW) # (2, 6, 80, 118, 32, 88)
        x = x.permute(0, 1, 3, 4, 5, 2) # (2, 6, 118, 32, 88, 80) # 表示每个3维空间点有80维特征（118, 32, 88）是空间索引
        return x # (2, 6, 118, 32, 88, 80)

    def forward(self, *args, **kwargs):
        x = super().forward(*args, **kwargs) # (2, 80, 360, 360)
        x = self.downsample(x) # (2, 80, 180, 180)
        return x # (2, 80, 180, 180)