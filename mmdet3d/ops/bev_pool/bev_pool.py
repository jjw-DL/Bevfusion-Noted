import torch

from . import bev_pool_ext

__all__ = ["bev_pool"]


class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0) # 沿着行累加，对pillar内的特征累加
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool) # (3666928,) 第一个点一定保留
        kept[:-1] = ranks[1:] != ranks[:-1] # 找到每个voxel的位置(不同点) trick:坐标在同一个pillar内的点只保留一个 (3666927,)

        x, geom_feats = x[kept], geom_feats[kept] # 提取被保留的voxel特征和geo特征 (203770, 80)和(203770, 4)
        x = torch.cat((x[:1], x[1:] - x[:-1])) # 计算每个pillar内的特征，并与第一行拼接

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats 几何特征不求梯度
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        (kept,) = ctx.saved_tensors
        back = torch.cumsum(kept, 0) # 对于一维特征，沿着列累加
        back[kept] -= 1 # 减一变为索引

        val = gradx[back] # 提取上下文特征对应位置的梯度，几何特征不求梯度

        return val, None, None


class QuickCumsumCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks, B, D, H, W):
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool) # (3666928,) 第一个点一定保留
        kept[1:] = ranks[1:] != ranks[:-1] # 找到每个voxel的位置(不同点)
        interval_starts = torch.where(kept)[0].int() # (203770,)
        # 计算每个voxel包含的点的数量
        interval_lengths = torch.zeros_like(interval_starts) # (203770,)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1] # (203769,)
        interval_lengths[-1] = x.shape[0] - interval_starts[-1] # 单独处理最后一个点
        geom_feats = geom_feats.int() # (3666928, 4)

        out = bev_pool_ext.bev_pool_forward(
            x, # (3666928, 80)
            geom_feats, # (3666928, 4)
            interval_lengths, # (203770,)
            interval_starts, # (203770,)
            B, # 2
            D, # 1
            H, # 360
            W, # 360
        )

        ctx.save_for_backward(interval_starts, interval_lengths, geom_feats)
        ctx.saved_shapes = B, D, H, W
        return out

    @staticmethod
    def backward(ctx, out_grad):
        # interval_starts: (203770,)
        # interval_lengths: (203770,)
        # geom_feats: (3666928, 4)
        interval_starts, interval_lengths, geom_feats = ctx.saved_tensors
        B, D, H, W = ctx.saved_shapes # 2, 1, 360, 360

        # 每个特征的梯度是pytorch自动计算好的，我们需要做的是累加pillar内的梯度
        out_grad = out_grad.contiguous() # [b, d, h, w, c]-->(2, 118, 32, 88, 80)
        x_grad = bev_pool_ext.bev_pool_backward(
            out_grad, # (2, 118, 32, 88, 80)
            geom_feats, # (3666928, 4)
            interval_lengths, # (203770,)
            interval_starts, # (203770,)
            B, # 2
            D, # 1
            H, # 360
            W, # 360
        )

        return x_grad, None, None, None, None, None, None


def bev_pool(feats, coords, B, D, H, W):
    # feats: (3666928, 80)
    # coords: (3666928, 4)
    # B, D, H, W: 2, 1, 360, 360
    assert feats.shape[0] == coords.shape[0]
    # 计算坐标索引
    ranks = (
        coords[:, 0] * (W * D * B)
        + coords[:, 1] * (D * B)
        + coords[:, 2] * B
        + coords[:, 3]
    ) # (3666928,) 此时在同一个pillar内的点的ranks是相同的
    indices = ranks.argsort() # 对坐标索引排序
    # 按照排序索引，重新排列特征，坐标和索引 --> 将空间中相邻的排在一起
    feats, coords, ranks = feats[indices], coords[indices], ranks[indices]

    x = QuickCumsumCuda.apply(feats, coords, ranks, B, D, H, W) # (2, 1, 360, 360, 80)
    x = x.permute(0, 4, 1, 2, 3).contiguous() # (2, 80, 1, 360, 360)
    return x # (2, 80, 1, 360, 360)
