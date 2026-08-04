# -*- coding: utf-8 -*-
"""
FreeRegNet — 端到端自由坐标目标学习的自定义网络
====================================================

把 FreeReg 几何引擎的自由坐标映射
    F = π_D ∘ K ∘ (R,t) ∘ (I + W)
改为由一个轻量视觉主干直接**回归**, 用**自由坐标目标**(投影映射残差)
端到端训练 —— 即 "learning to register" (摊销几何优化).

架构
----
  image ROI ──► ViT backbone (timm, 轻量) ──► tokens ─┐
                                                      ├─► Pose head ─► (R,t)
  model points (canonical) ──► PointEncoder ──► ctx ──┘        │
                              (object conditioning)            ▼
                                          可微投影链 π∘K∘(R,t)∘(I+W)
                                                       │
                                                       ▼
                                          自由坐标残差 L_proj (自监督)
                                          + 位姿监督 L_pose (BOP GT)

可插拔主干
----------
- 默认: timm ViT-S/16 (DINOv2 式自监督视觉编码器, 轻量, 立即可训).
- 预留: Qwen2-VL-2B / VLoRA (需 transformers+peft; backbone='qwen_vl').

临床旁路
--------
本网络仅用于**刚体物体位姿前端** (BOP 等); 临床路径仍走中心线结构法
(已实测通用 NN 前端在内镜 OOD: FoundationPose TRE 104mm 崩溃).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# 6D 旋转表示 (Zhou et al., CVPR 2019) → 正交 R
# ----------------------------------------------------------------------
def rot6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """(B,6) → (B,3,3), 保证正交 (GS 正交化)."""
    a1, a2 = d6[..., 0:3], d6[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)  # 列为 b1,b2,b3


def geodesic_rot_err(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    """geodesic 旋转误差 (rad), (B,)."""
    R = R_pred @ R_gt.transpose(-1, -2)
    tr = (R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]).clamp(-1.0, 3.0)
    return torch.acos(((tr - 1.0) / 2.0).clamp(-1.0, 1.0))


# ----------------------------------------------------------------------
# 轻量 ViT 主干 (timm)
# ----------------------------------------------------------------------
class ViTBackbone(nn.Module):
    """timm ViT 图像主干 → patch tokens. 轻量 (~21M)."""

    def __init__(self, name: str = 'vit_small_patch16_224',
                 pretrained: bool = True, img_size: int = 224):
        super().__init__()
        import timm
        self.net = timm.create_model(
            name, pretrained=pretrained, num_classes=0,
            img_size=img_size)
        self.dim = self.net.num_features
        self.img_size = img_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 返回全局特征 (B, dim). timm num_classes=0 → pooled feature.
        return self.net(x)


class QwenVLBackbone(nn.Module):
    """预留: Qwen2-VL 视觉主干 (+VLoRA). 需 transformers+peft 与权重.
    当前为占位 — 未安装时给出清晰报错."""

    def __init__(self, *a, **k):
        super().__init__()
        try:
            import transformers  # noqa: F401
            import peft  # noqa: F401
        except Exception as e:  # pragma: no cover
            raise ImportError(
                'Qwen-VL 主干需 transformers+peft 与 Qwen2-VL 权重; '
                '请 `pip install transformers peft` 并下载权重, 或改用 '
                "backbone='vit' (timm 轻量 ViT).") from e
        self.dim = 384
        # TODO: 接入 Qwen2-VL vision tower + LoRA (权重下载后实现)
        raise NotImplementedError('Qwen-VL 主干待权重就绪后接入')


def make_backbone(kind: str = 'vit', **kw) -> nn.Module:
    if kind == 'vit':
        return ViTBackbone(**kw)
    if kind in ('qwen_vl', 'qwen', 'vlora'):
        return QwenVLBackbone(**kw)
    raise ValueError(f'unknown backbone {kind}')


# ----------------------------------------------------------------------
# 物体几何条件编码 (canonical model points → 物体身份嵌入)
# ----------------------------------------------------------------------
class PointEncoder(nn.Module):
    """把 canonical 模型点云编码为物体条件向量 (PointNet-lite)."""

    def __init__(self, feat_dim: int = 384, n_points: int = 256):
        super().__init__()
        self.n_points = n_points
        self.mlp = nn.Sequential(
            nn.Linear(3, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 256), nn.ReLU(inplace=True),
            nn.Linear(256, feat_dim))
        self.out = nn.Linear(feat_dim, feat_dim)

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        """pts (B,N,3) canonical → (B, feat_dim) 物体嵌入."""
        h = self.mlp(pts)              # (B,N,feat)
        h = h.max(dim=1).values        # global max pool (B,feat)
        return self.out(h)


# ----------------------------------------------------------------------
# FreeRegNet: 主干 + 物体条件 + 自由坐标位姿头
# ----------------------------------------------------------------------
class FreeRegNet(nn.Module):
    """端到端自由坐标位姿回归网络 (刚体; W 由后端 RTW 负责).

    输入: image ROI (B,3,H,W), canonical model points (B,N,3)
    输出: R (B,3,3), t (B,3)  (model→camera, 单位与训练数据一致 mm)
    """

    def __init__(self, backbone: str = 'vit', feat_dim: int = 384,
                 n_points: int = 256, **bkw):
        super().__init__()
        self.backbone = make_backbone(backbone, **bkw)
        fd = self.backbone.dim
        self.point_enc = PointEncoder(feat_dim=fd, n_points=n_points)
        self.fuse = nn.Sequential(
            nn.Linear(fd * 2, fd), nn.ReLU(inplace=True),
            nn.Linear(fd, fd), nn.ReLU(inplace=True))
        # 头: 6D 旋转 + log深度 (ROI 解耦位姿: 平移由裁剪中心+深度反投影)
        self.head_rot = nn.Linear(fd, 6)
        self.head_depth = nn.Linear(fd, 1)
        with torch.no_grad():
            self.head_rot.bias.zero_()
            self.head_rot.bias[0] = 1.0   # b1≈x
            self.head_rot.bias[4] = 1.0   # b2≈y
            self.head_depth.bias.fill_(math.log(500.0))  # ~500mm 初始深度

    def forward(self, image: torch.Tensor, model_pts: torch.Tensor):
        """返回 R (B,3,3), depth (B,) mm. 平移由 pose_from_roi 恢复."""
        feat_i = self.backbone(image)              # (B,fd)
        feat_m = self.point_enc(model_pts)         # (B,fd)
        h = self.fuse(torch.cat([feat_i, feat_m], dim=-1))
        R = rot6d_to_matrix(self.head_rot(h))      # (B,3,3)
        depth = torch.exp(self.head_depth(h).squeeze(-1))  # (B,) mm
        return R, depth


def pose_from_roi(R, depth, center, K):
    """ROI 解耦: 由预测 R + 深度 + 裁剪中心 + 内参 K 恢复完整平移.

    center (B,2): 物体在原图的图像位置 (cx,cy) px; K (B,3,3).
    t = depth * K^{-1} [cx, cy, 1]^T  (x,y 由观测位置定, z=depth).
    """
    B = R.shape[0]
    uv1 = torch.ones(B, 3, device=R.device, dtype=R.dtype)
    uv1[:, 0] = center[:, 0]
    uv1[:, 1] = center[:, 1]
    ray = torch.einsum('bij,bj->bi', torch.linalg.inv(K), uv1)  # (B,3)
    ray = ray / ray[:, 2:3].clamp_min(1e-6)                      # 归一化 z=1
    t = depth.unsqueeze(1) * ray                                 # (B,3)
    T = torch.eye(4, device=R.device, dtype=R.dtype).unsqueeze(0).repeat(B, 1, 1)
    T[:, :3, :3] = R
    T[:, :3, 3] = t
    return T


# ----------------------------------------------------------------------
# 自由坐标目标损失
# ----------------------------------------------------------------------
def free_coordinate_loss(R_pred, t_pred, model_pts, R_gt, t_gt,
                         w_rot=1.0, w_trans=1.0, w_proj=0.0):
    """组合监督位姿损失 + 可微投影残差 (自由坐标目标).

    L = w_rot*geodesic(R) + w_trans*|t-t_gt| + w_proj*深度一致性残差.
    深度一致性残差 = 预测位姿变换模型点的深度 vs GT 位姿变换点的深度,
    即自由坐标映射残差 π∘K∘(R,t)∘X 的深度分量 (自监督正则).
    """
    loss = R_pred.new_zeros(())
    r_err = geodesic_rot_err(R_pred, R_gt)                 # (B,)
    loss = loss + w_rot * r_err.mean()
    t_err = (t_pred - t_gt).norm(dim=-1)                   # (B,) mm
    loss = loss + w_trans * (t_err / 100.0).mean()
    if w_proj > 0:
        Xc = torch.einsum('bij,bnj->bni', R_pred, model_pts) \
            + t_pred.unsqueeze(1)
        Xg = torch.einsum('bij,bnj->bni', R_gt, model_pts) \
            + t_gt.unsqueeze(1)
        z_pred = Xc[..., 2].clamp_min(1e-3)
        z_gt = Xg[..., 2].clamp_min(1e-3)
        proj = (z_pred - z_gt).abs() / z_gt.clamp_min(1.0)
        loss = loss + w_proj * proj.mean()
    return loss, dict(rot=float(r_err.mean()), trans=float(t_err.mean()))


__all__ = [
    'FreeRegNet', 'ViTBackbone', 'QwenVLBackbone', 'PointEncoder',
    'make_backbone', 'rot6d_to_matrix', 'geodesic_rot_err',
    'pose_from_roi', 'free_coordinate_loss',
]
