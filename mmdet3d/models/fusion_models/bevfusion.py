from typing import Any, Dict

import torch
from mmcv.runner import auto_fp16, force_fp32
from torch import nn
from torch.nn import functional as F

from mmdet3d.models.builder import (
    build_backbone,
    build_fuser,
    build_head,
    build_neck,
    build_vtransform,
)
from mmdet3d.ops import Voxelization
from mmdet3d.models import FUSIONMODELS

from .base import Base3DFusionModel

__all__ = ["BEVFusion"]


@FUSIONMODELS.register_module()
class BEVFusion(Base3DFusionModel):
    def __init__(
        self,
        encoders: Dict[str, Any],
        fuser: Dict[str, Any],
        decoder: Dict[str, Any],
        heads: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__()

        self.encoders = nn.ModuleDict()
        if encoders.get("camera") is not None: # 构建camera的encoder
            self.encoders["camera"] = nn.ModuleDict(
                {
                    "backbone": build_backbone(encoders["camera"]["backbone"]), # 构建backbone, swin transformer
                    "neck": build_neck(encoders["camera"]["neck"]), # 
                    "vtransform": build_vtransform(encoders["camera"]["vtransform"]),
                }
            )
        if encoders.get("lidar") is not None: # 构建lidar的encoder
            self.encoders["lidar"] = nn.ModuleDict(
                {
                    "voxelize": Voxelization(**encoders["lidar"]["voxelize"]),
                    "backbone": build_backbone(encoders["lidar"]["backbone"]),
                }
            )
            self.voxelize_reduce = encoders["lidar"].get("voxelize_reduce", True) # True

        if fuser is not None:
            self.fuser = build_fuser(fuser) # 构建融合层
        else:
            self.fuser = None

        self.decoder = nn.ModuleDict(
            {
                "backbone": build_backbone(decoder["backbone"]),
                "neck": build_neck(decoder["neck"]),
            }
        ) # 构建lidar的decoder
        self.heads = nn.ModuleDict()
        for name in heads:
            if heads[name] is not None:
                self.heads[name] = build_head(heads[name]) # 构建head-->transfusion Head

        if "loss_scale" in kwargs:
            self.loss_scale = kwargs["loss_scale"]
        else:
            self.loss_scale = dict()
            for name in heads:
                if heads[name] is not None:
                    self.loss_scale[name] = 1.0

        self.init_weights() # 权重初始化

    def init_weights(self) -> None:
        if "camera" in self.encoders:
            self.encoders["camera"]["backbone"].init_weights()

    def extract_camera_features(
        self,
        x,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size() # (2, 6, 3, 256, 704)
        x = x.view(B * N, C, H, W) # 将batch和图片数量合并 --> # (12, 3, 256, 704)

        x = self.encoders["camera"]["backbone"](x) # (12, 192, 32, 88)和(12, 384, 16, 44)和(12, 768, 8, 22)
        x = self.encoders["camera"]["neck"](x) # ((12, 256, 32, 88)和(12, 256, 16, 44))

        if not isinstance(x, torch.Tensor):
            x = x[0] # (12, 256, 32, 88)

        BN, C, H, W = x.size() # 12, 256, 32, 88
        x = x.view(B, int(BN / B), C, H, W) # (2, 6, 256, 32, 88)

        x = self.encoders["camera"]["vtransform"](
            x,
            points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            img_aug_matrix,
            lidar_aug_matrix,
            img_metas,
        ) # (2, 80, 180, 180)
        return x # (2, 80, 180, 180)

    def extract_lidar_features(self, x) -> torch.Tensor:
        # x:List(tensor)
        # feats: (190773, 5)
        # coords: (190773, 4)
        # sizes:(190773,)
        feats, coords, sizes = self.voxelize(x)
        batch_size = coords[-1, 0] + 1 # 计算batch size:2
        x = self.encoders["lidar"]["backbone"](feats, coords, batch_size, sizes=sizes) # (2, 256, 180, 180)
        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        # 逐帧点云处理
        for k, res in enumerate(points):
            # f:(72517, 10, 5)
            # c:(72517, 3)
            # n:(72517,)
            f, c, n = self.encoders["lidar"]["voxelize"](res)
            feats.append(f) # (72517, 10, 5)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k)) # eg:(72517, 4)
            sizes.append(n) # (72517,)

        feats = torch.cat(feats, dim=0) # (190773, 10, 5)
        coords = torch.cat(coords, dim=0) # (190773, 4) batch id在第一维
        sizes = torch.cat(sizes, dim=0) # 190773

        if self.voxelize_reduce:
            feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1) # (190773, 5)
            feats = feats.contiguous() # (190773, 5)

        return feats, coords, sizes

    @auto_fp16(apply_to=("img", "points"))
    def forward(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        features = []
        # 特征提取
        for sensor in self.encoders:
            if sensor == "camera":
                feature = self.extract_camera_features(
                    img,
                    points,
                    camera2ego,
                    lidar2ego,
                    lidar2camera,
                    lidar2image,
                    camera_intrinsics,
                    img_aug_matrix,
                    lidar_aug_matrix,
                    metas,
                ) # (2, 80, 180, 180)
            elif sensor == "lidar":
                feature = self.extract_lidar_features(points) #  (2, 256, 180, 180)
            else:
                raise ValueError(f"unsupported sensor: {sensor}")
            features.append(feature)

        if self.fuser is not None:
            x = self.fuser(features) # (2, 256, 180, 180)
        else:
            assert len(features) == 1, features
            x = features[0]

        batch_size = x.shape[0] # 2

        x = self.decoder["backbone"](x) # (2, 128, 180, 180)和(2, 256, 90, 90)
        x = self.decoder["neck"](x) # [(2, 512, 180, 180)]

        if self.training:
            outputs = {}
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                elif type == "map":
                    losses = head(x, gt_masks_bev)
                else:
                    raise ValueError(f"unsupported head: {type}")
                for name, val in losses.items():
                    if val.requires_grad:
                        outputs[f"loss/{type}/{name}"] = val * self.loss_scale[type]
                    else:
                        outputs[f"stats/{type}/{name}"] = val
            return outputs
        else:
            outputs = [{} for _ in range(batch_size)]
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    bboxes = head.get_bboxes(pred_dict, metas)
                    for k, (boxes, scores, labels) in enumerate(bboxes):
                        outputs[k].update(
                            {
                                "boxes_3d": boxes.to("cpu"),
                                "scores_3d": scores.cpu(),
                                "labels_3d": labels.cpu(),
                            }
                        )
                elif type == "map":
                    logits = head(x)
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "masks_bev": logits[k].cpu(),
                                "gt_masks_bev": gt_masks_bev[k].cpu(),
                            }
                        )
                else:
                    raise ValueError(f"unsupported head: {type}")
            return outputs
