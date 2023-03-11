import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16

from mmdet.models.builder import NECKS

__all__ = ["GeneralizedLSSFPN"]


@NECKS.register_module()
class GeneralizedLSSFPN(BaseModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_outs,
        start_level=0,
        end_level=-1,
        no_norm_on_lateral=False,
        conv_cfg=None,
        norm_cfg=dict(type="BN2d"),
        act_cfg=dict(type="ReLU"),
        upsample_cfg=dict(mode="bilinear", align_corners=True),
    ) -> None:
        super().__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels # [192, 384, 728]
        self.out_channels = out_channels # 256
        self.num_ins = len(in_channels) # 3
        self.num_outs = num_outs # 3
        self.no_norm_on_lateral = no_norm_on_lateral # False
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy() # bilinear, align_corners=False

        if end_level == -1:
            self.backbone_end_level = self.num_ins - 1 # 2
            # assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level # 0
        self.end_level = end_level # -1

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level): # (0, 2)
            l_conv = ConvModule(
                in_channels[i]
                + (
                    in_channels[i + 1]
                    if i == self.backbone_end_level - 1
                    else out_channels
                ), # 448, 1152, 
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False,
            )
            fpn_conv = ConvModule(
                out_channels, # 256
                out_channels, # 256
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False,
            )

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        # upsample -> cat -> conv1x1 -> conv3x3
        assert len(inputs) == len(self.in_channels) # (12, 192, 32, 88)和(12, 384, 16, 44)和(12, 768, 8, 22)

        # build laterals
        laterals = [inputs[i + self.start_level] for i in range(len(inputs))] # self.start_level = 0

        # build top-down path
        used_backbone_levels = len(laterals) - 1 # 2
        for i in range(used_backbone_levels - 1, -1, -1): # 从倒数第二层开始
            x = F.interpolate(
                laterals[i + 1],
                size=laterals[i].shape[2:],
                **self.upsample_cfg,
            ) # 先进行上采样
            laterals[i] = torch.cat([laterals[i], x], dim=1) # 将上采样后的特征拼接
            laterals[i] = self.lateral_convs[i](laterals[i]) # 通过lateral卷积层-->256
            laterals[i] = self.fpn_convs[i](laterals[i]) # 在通过fpn卷积层-->256

        # build outputs
        outs = [laterals[i] for i in range(used_backbone_levels)] # 组合输出:(12, 256, 32, 88)和(12, 256, 16, 44)
        return tuple(outs) # ((12, 256, 32, 88)和(12, 256, 16, 44))
