from typing import Tuple

import torch
from mmcv.runner import force_fp32
from torch import nn

from mmdet3d.ops import bev_pool

__all__ = ["BaseTransform", "BaseDepthTransform"]


def gen_dx_bx(xbound, ybound, zbound):
    # xbound: [low_bound, upper_bound, size]
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]]) # (0.3, 0.3, 20)
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]]) # (-53.85, -53.85, 0)
    nx = torch.LongTensor(
        [(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]]
    ) # (360, 360, 1)
    return dx, bx, nx


class BaseTransform(nn.Module):
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
    ) -> None:
        super().__init__()
        self.in_channels = in_channels # 256
        self.image_size = image_size # (256, 704)
        self.feature_size = feature_size # (32, 88)
        self.xbound = xbound # [-54.0, 54.0, 0.3]
        self.ybound = ybound # [-54.0, 54.0, 0.3]
        self.zbound = zbound # [-10.0, 10.0, 20.0]
        self.dbound = dbound # [1.0, 60.0, 0.5]
        
        # dx: (0.3, 0.3, 20)
        # bx: (-53.85, -53.85, 0)
        # nx: (360, 360, 1)
        dx, bx, nx = gen_dx_bx(self.xbound, self.ybound, self.zbound)
        self.dx = nn.Parameter(dx, requires_grad=False) # 注册Parameter
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.C = out_channels # 80
        self.frustum = self.create_frustum() # (118, 32, 88, 3)
        self.D = self.frustum.shape[0] # 118
        self.fp16_enabled = False

    @force_fp32()
    def create_frustum(self):
        iH, iW = self.image_size # (256, 784)
        fH, fW = self.feature_size # (32, 88)

        ds = (
            torch.arange(*self.dbound, dtype=torch.float)
            .view(-1, 1, 1)
            .expand(-1, fH, fW)
        ) # [1.0, 60.0, 0.5] --> (118, 1, 1) --> (118, 32, 88)
        D, _, _ = ds.shape # 118

        xs = (
            torch.linspace(0, iW - 1, fW, dtype=torch.float)
            .view(1, 1, fW)
            .expand(D, fH, fW)
        ) # (88) --> (1, 1, 88) --> (118, 32, 88)
        ys = (
            torch.linspace(0, iH - 1, fH, dtype=torch.float)
            .view(1, fH, 1)
            .expand(D, fH, fW)
        ) # (32) --> (1, 32, 1) --> (118, 32, 88)

        frustum = torch.stack((xs, ys, ds), -1) # (118, 32, 88, 3) 区分索引和内容(坐标)
        return nn.Parameter(frustum, requires_grad=False) # 注册frustum参数

    @force_fp32()
    def get_geometry(
        self,
        rots,
        trans,
        intrins,
        post_rots,
        post_trans,
        lidar2ego_rots,
        lidar2ego_trans,
        **kwargs,
    ):
        B, N, _ = trans.shape # 2, 6, 3
        #--------------------------------------
        # 1.undo post-transformation
        #--------------------------------------
        # B x N x D x H x W x 3
        # (118, 32, 88, 3) - (2, 6, 3)-->(2, 6, 1, 1, 1, 3) -->(2, 6, 118, 32, 88, 3)
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        # (2, 6, 3, 3)-->(2, 6, 1, 1, 1, 3, 3) * (2, 6, 118, 32, 88, 3, 1)
        points = (
            torch.inverse(post_rots)
            .view(B, N, 1, 1, 1, 3, 3)
            .matmul(points.unsqueeze(-1))
        ) # (2, 6, 118, 32, 88, 3, 1)
        #--------------------------------------
        # 2.cam_to_ego
        #--------------------------------------
        points = torch.cat(
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                points[:, :, :, :, :, 2:3],
            ),
            5,
        ) # 乘深度 (2, 6, 118, 32, 88, 3, 1)
        # (2, 6, 3, 3) *  (2, 6, 3, 3) --> (2, 6, 3, 3)
        combine = rots.matmul(torch.inverse(intrins)) # carmer到ego的旋转乘内参的逆
        # (2, 6, 1, 1, 1, 3, 3) * (2, 6, 118, 32, 88, 3, 1) --> (2, 6, 118, 32, 88, 3)
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        # (2, 6, 118, 32, 88, 3) + (2, 6, 1, 1, 1, 3) --> (2, 6, 118, 32, 88, 3)
        points += trans.view(B, N, 1, 1, 1, 3)
        #--------------------------------------
        # 3.ego_to_lidar
        #--------------------------------------
        # (2, 6, 118, 32, 88, 3) - (2, 3) --> (2, 1, 1, 1, 1, 3) --> (2, 6, 118, 32, 88, 3)
        points -= lidar2ego_trans.view(B, 1, 1, 1, 1, 3)
        # (2, 3, 3) --> (2, 1, 1, 1, 1, 3, 3) * (2, 6, 118, 32, 88, 3, 1) --> (2, 6, 118, 32, 88, 3)
        points = (
            torch.inverse(lidar2ego_rots)
            .view(B, 1, 1, 1, 1, 3, 3)
            .matmul(points.unsqueeze(-1))
            .squeeze(-1)
        )

        if "extra_rots" in kwargs:
            extra_rots = kwargs["extra_rots"]
            points = (
                extra_rots.view(B, 1, 1, 1, 1, 3, 3)
                .repeat(1, N, 1, 1, 1, 1, 1)
                .matmul(points.unsqueeze(-1))
                .squeeze(-1)
            )
        if "extra_trans" in kwargs:
            extra_trans = kwargs["extra_trans"]
            points += extra_trans.view(B, 1, 1, 1, 1, 3).repeat(1, N, 1, 1, 1, 1)

        return points # (2, 6, 118, 32, 88, 3)

    def get_cam_feats(self, x):
        raise NotImplementedError

    @force_fp32()
    def bev_pool(self, geom_feats, x):
        B, N, D, H, W, C = x.shape # (2, 6, 118, 32, 88, 80)
        Nprime = B * N * D * H * W # 3987456

        # flatten x
        x = x.reshape(Nprime, C) # (3987456, 80)

        # flatten indices
        # (2, 6, 118, 32, 88, 3) - (3,) --> (2, 6, 118, 32, 88, 3) 这里转换为voxle(pillar)坐标
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3) # (3987456, 3)
        batch_ix = torch.cat(
            [
                torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long)
                for ix in range(B)
            ]
        ) # (3987456, 1)
        geom_feats = torch.cat((geom_feats, batch_ix), 1) # (3987456, 4) --> (h, w, d, b)

        # filter out points that are outside box
        kept = (
            (geom_feats[:, 0] >= 0)
            & (geom_feats[:, 0] < self.nx[0])
            & (geom_feats[:, 1] >= 0)
            & (geom_feats[:, 1] < self.nx[1])
            & (geom_feats[:, 2] >= 0)
            & (geom_feats[:, 2] < self.nx[2])
        ) # (3987456,)
        x = x[kept] # (3666928, 80)
        geom_feats = geom_feats[kept] # (3666928, 4)
        # x: (3666928, 80)
        # geom_feats: (3666928, 4)
        # self.nx: (360, 360, 1)
        x = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1]) # (2, 80, 1, 360, 360)

        # collapse Z 
        # torch.unbind()移除指定维后，返回一个元组，包含了沿着指定维切片后的各个切片
        # 因为z的维度为1，所以tuple中只包含一个元素
        final = torch.cat(x.unbind(dim=2), 1) # (2, 80, 360, 360)

        return final

    @force_fp32()
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
        **kwargs,
    ):
        rots = camera2ego[..., :3, :3] # (2, 6, 4, 4) --> (2, 6, 3, 3) 相机到自车的旋转矩阵
        trans = camera2ego[..., :3, 3] # (2, 6, 4, 4) --> (2, 6, 3) 相机到自车的平移向量
        intrins = camera_intrinsics[..., :3, :3] # 相机内参 (2, 6, 3, 3)
        post_rots = img_aug_matrix[..., :3, :3] # 图像变换增强后的旋转矩阵 (2, 6, 3, 3)
        post_trans = img_aug_matrix[..., :3, 3] # 图像变换增强后的平移向量 (2, 6, 3)
        lidar2ego_rots = lidar2ego[..., :3, :3] # lidar到自车的旋转矩阵 (2, 6, 3, 3)
        lidar2ego_trans = lidar2ego[..., :3, 3] # lidar到自车的平移向量 (2, 6, 3)
        extra_rots = lidar_aug_matrix[..., :3, :3] # lidar变换后的旋转矩阵 (2, 3，3) 单位矩阵，并未使用
        extra_trans = lidar_aug_matrix[..., :3, 3] # lidar变换后的平移向量 (2, 3)

        geom = self.get_geometry(
            rots,
            trans,
            intrins,
            post_rots,
            post_trans,
            lidar2ego_rots,
            lidar2ego_trans,
            extra_rots=extra_rots,
            extra_trans=extra_trans
        )

        x = self.get_cam_feats(img)
        x = self.bev_pool(geom, x)
        return x


class BaseDepthTransform(BaseTransform):
    @force_fp32()
    def forward(
        self,
        img,
        points,
        sensor2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        cam_intrinsic,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        **kwargs,
    ):
        rots = camera2ego[..., :3, :3] # (2, 6, 4, 4) --> (2, 6, 3, 3) 相机到自车的旋转矩阵
        trans = camera2ego[..., :3, 3] # (2, 6, 4, 4) --> (2, 6, 3) 相机到自车的平移向量
        intrins = camera_intrinsics[..., :3, :3] # 相机内参 (2, 6, 3, 3)
        post_rots = img_aug_matrix[..., :3, :3] # 图像变换增强后的旋转矩阵 (2, 6, 3, 3)
        post_trans = img_aug_matrix[..., :3, 3] # 图像变换增强后的平移向量 (2, 6, 3)
        lidar2ego_rots = lidar2ego[..., :3, :3] # lidar到自车的旋转矩阵 (2, 6, 3, 3)
        lidar2ego_trans = lidar2ego[..., :3, 3] # lidar到自车的平移向量 (2, 6, 3)
        extra_rots = lidar_aug_matrix[..., :3, :3] # lidar变换后的旋转矩阵 (2, 3，3) 单位矩阵，并未使用
        extra_trans = lidar_aug_matrix[..., :3, 3] # lidar变换后的平移向量 (2, 3)

        batch_size = len(points) # 2
        depth = torch.zeros(batch_size, 6, 1, *self.image_size).to(points[0].device) # (2, 6, 1, 256, 704)

        # 逐帧处理
        for b in range(batch_size):
            cur_coords = points[b][:, :3].transpose(1, 0) # (3, 243617) lidar点云
            cur_img_aug_matrix = img_aug_matrix[b] # 提取图像增强矩阵(6, 4, 4)
            cur_lidar_aug_matrix = lidar_aug_matrix[b] # 提取lidar增强矩阵 (4, 4)
            cur_lidar2image = lidar2image[b] # (6, 4, 4)

            # lidar2image
            cur_coords = cur_lidar2image[:, :3, :3].matmul(cur_coords) # (6, 3, 243617)
            cur_coords += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1) # (6, 3, 243617)
            # get 2d coords
            dist = cur_coords[:, 2, :] # (6, 243617)
            cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5) # 深度截断 (6, 243617)
            cur_coords[:, :2, :] /= cur_coords[:, 2:3, :] # 转换图像坐标 (6, 2, 243617)

            # imgaug 乘图像增强变换矩阵
            cur_coords = cur_img_aug_matrix[:, :3, :3].matmul(cur_coords) # (6, 3, 243617)
            cur_coords += cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1) # (6, 3, 243617)
            cur_coords = cur_coords[:, :2, :].transpose(1, 2) # (6, 243617, 2)

            # normalize coords for grid sample
            cur_coords = cur_coords[..., [1, 0]]

            on_img = (
                (cur_coords[..., 0] < self.image_size[0])
                & (cur_coords[..., 0] >= 0)
                & (cur_coords[..., 1] < self.image_size[1])
                & (cur_coords[..., 1] >= 0)
            ) # 判断图像上的投影点 (6, 243617)
            # 逐个相机处理
            for c in range(6):
                masked_coords = cur_coords[c, on_img[c]].long() # eg:(27645, 2) lidar投影到该图像上的点坐标
                masked_dist = dist[c, on_img[c]] # lidar投影到该图像上的点深度 eg:(27645, ）
                # 在depth对应lidar投影位置处赋予lidar深度(巧妙)
                depth[b, c, 0, masked_coords[:, 0], masked_coords[:, 1]] = masked_dist # (2, 6, 1, 256, 704)
        
        # -------------------------------------------------
        # 1.计算6个cam的frustum产生的3维空间点在lidar系下的坐标
        # -------------------------------------------------
        geom = self.get_geometry(
            rots,
            trans,
            intrins,
            post_rots,
            post_trans,
            lidar2ego_rots,
            lidar2ego_trans,
        ) # (2, 6, 118, 32, 88, 3) 

        # -------------------------------------------------
        # 2.计算6个cam的frustum产生的3维空间点的上下文特征，通过隐深度表征 
        # 118, 32, 88是坐标索引，并不表示实际坐标
        # 这里相比于LSS，多了一个lidar投影的depth，有助于隐深度的恢复
        # -------------------------------------------------
        x = self.get_cam_feats(img, depth) # (2, 6, 118, 32, 88, 80) <-- (B, N, D, H, W, P)

        # -------------------------------------------------
        # 经过前两步已经计算出空间中每个点的实际坐标(3)和对应的特征(80隐含深度信息)
        # 3.计算pillar坐标，预定义BEV voxel，将点放入对应pillar，实现voxlelization
        # -------------------------------------------------
        x = self.bev_pool(geom, x) # (2, 80, 360, 360)
        return x # (2, 80, 360, 360)
