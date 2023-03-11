# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import BaseModule
from torch import nn as nn

from mmdet.models import BACKBONES

__all__ = ["SECOND"]


@BACKBONES.register_module()
class SECOND(BaseModule):
    """Backbone network for SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (int): Input channels.
        out_channels (list[int]): Output channels for multi-scale feature maps.
        layer_nums (list[int]): Number of layers in each stage.
        layer_strides (list[int]): Strides of each stage.
        norm_cfg (dict): Config dict of normalization layers.
        conv_cfg (dict): Config dict of convolutional layers.
    """

    def __init__(
        self,
        in_channels=128, # 256
        out_channels=[128, 128, 256], # [128, 256]
        layer_nums=[3, 5, 5], # [5, 5]
        layer_strides=[2, 2, 2], # [1, 2]
        norm_cfg=dict(type="BN", eps=1e-3, momentum=0.01),
        conv_cfg=dict(type="Conv2d", bias=False),
        init_cfg=None,
        pretrained=None,
    ):
        super().__init__(init_cfg=init_cfg)
        assert len(layer_strides) == len(layer_nums)
        assert len(out_channels) == len(layer_nums)

        in_filters = [in_channels, *out_channels[:-1]] # 计算每一个block的起始输入channle[256, 128]
        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        blocks = []
        for i, layer_num in enumerate(layer_nums):
            block = [
                build_conv_layer(
                    conv_cfg,
                    in_filters[i], # 256, 128
                    out_channels[i], # 128, 256
                    3,
                    stride=layer_strides[i], # 1, 2
                    padding=1, # 构建conv层
                ),
                build_norm_layer(norm_cfg, out_channels[i])[1], # 这里取1的原因是返回值有两个一个是name，一个是layer
                nn.ReLU(inplace=True),
            ]
            # 按照配置文件构建n个输入输出相同的层
            for j in range(layer_num):
                block.append(
                    build_conv_layer(
                        conv_cfg, out_channels[i], out_channels[i], 3, padding=1
                    )
                )
                block.append(build_norm_layer(norm_cfg, out_channels[i])[1])
                block.append(nn.ReLU(inplace=True))

            block = nn.Sequential(*block)
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)

        assert not (
            init_cfg and pretrained
        ), "init_cfg and pretrained cannot be setting at the same time"
        if isinstance(pretrained, str):
            warnings.warn(
                "DeprecationWarning: pretrained is a deprecated, "
                'please use "init_cfg" instead'
            )
            self.init_cfg = dict(type="Pretrained", checkpoint=pretrained)
        else:
            self.init_cfg = dict(type="Kaiming", layer="Conv2d")

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input with shape (N, C, H, W). (2, 256, 180, 180)

        Returns:
            tuple[torch.Tensor]: Multi-scale features.
        """
        outs = []
        for i in range(len(self.blocks)):
            # (2, 128, 180, 180)和(2, 256, 90, 90)
            x = self.blocks[i](x)
            outs.append(x)
        return tuple(outs) # (2, 128, 180, 180)和(2, 256, 90, 90)
