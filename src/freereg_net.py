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


def adds_pointcloud_loss(R_pred, t_pred, R_gt, t_gt, pts):
    """ADD-S 可微点云损失 (对称感知).

    对每点取最近邻距离 — 对对称物体天然不变 (离散/连续对称均适用),
    避免对称歧义导致的旋转监督冲突 (如 ape 89° 误差).
    L = mean_i min_j || (R_pred x_i + t_pred) - (R_gt x_j + t_gt) ||
    """
    Xp = torch.einsum('bij,bnj->bni', R_pred, pts) + t_pred.unsqueeze(1)
    Xg = torch.einsum('bij,bnj->bni', R_gt, pts) + t_gt.unsqueeze(1)
    # 最近邻距离 (B,N): min_j ||Xp_i - Xg_j||
    d = torch.cdist(Xp, Xg)                    # (B,N,N)
    nn = d.min(dim=2).values                   # (B,N)
    return nn.mean()


def symmetric_rot_err(R_pred, R_gt, sym_mats=None):
    """对称感知旋转误差: 对离散对称 S 取 min geodesic(R_pred, R_gt S).
    sym_mats: list of (3,3) 对称旋转 (torch)."""
    if not sym_mats:
        return geodesic_rot_err(R_pred, R_gt)
    errs = [geodesic_rot_err(R_pred, R_gt @ S.to(R_gt.device, R_gt.dtype))
            for S in sym_mats]
    return torch.stack(errs, 0).min(0).values


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
    """Qwen2-VL 视觉主干 (+ 可选 VLoRA).

    用 Qwen2-VL 的 vision tower 提取图像特征. 需 transformers+peft 与权重
    (默认 Qwen/Qwen2-VL-2B-Instruct). 大模型, 建议 LoRA 微调而非全量.
    """

    def __init__(self, model_name: str = 'Qwen/Qwen2-VL-2B-Instruct',
                 lora: bool = True, lora_r: int = 8, **k):
        super().__init__()
        try:
            from transformers import Qwen2VLForConditionalGeneration
        except Exception as e:  # pragma: no cover
            raise ImportError(
                'Qwen-VL 主干需 transformers(+peft); '
                '请 `pip install transformers peft accelerate`.') from e
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float32)
        # 视觉 tower: Qwen2-VL 的 visual 模块
        self.visual = self.model.visual
        # temporal_patch=2, patch=14; 输出维=LLM hidden (merger 投影后)
        vcfg = self.model.config.vision_config
        self.patch = int(getattr(vcfg, 'patch_size', 14))
        self.temporal = int(getattr(vcfg, 'temporal_patch_size', 2))
        self.qwen_dim = int(getattr(self.model.config, 'hidden_size', 1536))
        # 投影到统一特征维度 (与位姿头对齐), 惰性初始化以适配实际输出维
        self.proj = None
        self.dim = 384
        if lora:
            try:
                from peft import LoraConfig, get_peft_model
                cfg = LoraConfig(r=lora_r, lora_alpha=16,
                                 target_modules=['qkv', 'proj'],
                                 lora_dropout=0.05)
                self.visual = get_peft_model(self.visual, cfg)
            except Exception:
                pass  # 无 peft 则全量/冻结
        else:
            for p in self.visual.parameters():
                p.requires_grad = False

    def _to_patches(self, x: torch.Tensor):
        """(B,3,H,W) → Qwen 格式: 时域复制 + 网格 patch 展平 + grid_thw.
        Qwen2-VL: temporal_patch=2, patch=14 → 每 patch 3*2*14*14=1176."""
        B, _, H, W = x.shape
        gh, gw = H // self.patch, W // self.patch
        x = x.unsqueeze(2).repeat(1, 1, self.temporal, 1, 1)  # (B,3,2,H,W)
        x = x.reshape(B, 3, self.temporal, gh, self.patch, gw, self.patch)
        x = x.permute(0, 3, 5, 1, 2, 4, 6)      # (B,gh,gw,3,2,p,p)
        x = x.reshape(B, gh * gw, 3 * self.temporal * self.patch * self.patch)
        # grid_thw: 单帧 t=1 (时域复制已在 1176 维内完成), [1, gh, gw]
        grid = torch.tensor([[1, gh, gw]] * B,
                            dtype=torch.long, device=x.device)
        return x.reshape(B * gh * gw, -1), grid, gh, gw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Qwen2-VL visual 前向 (patch+grid_thw), 池化 → 投影到 384 维.
        B = x.shape[0]
        pv, grid, gh, gw = self._to_patches(x)
        feats = self.visual(pv, grid_thw=grid)   # (B*tokens_merged, hidden)
        n = feats.shape[0] // B
        feats = feats.reshape(B, n, -1).mean(dim=1)  # (B, hidden)
        if self.proj is None:                    # 惰性初始化投影
            self.proj = nn.Linear(feats.shape[-1], self.dim).to(feats.device)
        return self.proj(feats)                  # (B, 384)


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
                         sym_mats=None, symmetric=False,
                         w_rot=1.0, w_trans=1.0, w_proj=0.0, w_adds=0.0):
    """自由坐标目标损失 (对称感知).

    L = w_rot*geodesic(R) [对称物体改 ADD-S] + w_trans*|Δt|
        + w_proj*深度一致性 + w_adds*ADD-S点云损失.
    symmetric=True 时旋转监督改用 ADD-S 点云损失 (避免对称歧义).
    """
    loss = R_pred.new_zeros(())
    if symmetric or w_adds > 0:
        adds = adds_pointcloud_loss(R_pred, t_pred, R_gt, t_gt, model_pts)
        loss = loss + max(w_adds, w_rot if symmetric else 0.0) * (adds / 10.0)
        r_err = symmetric_rot_err(R_pred, R_gt, sym_mats)
    else:
        r_err = geodesic_rot_err(R_pred, R_gt)
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
    'pose_from_roi', 'free_coordinate_loss', 'adds_pointcloud_loss',
    'symmetric_rot_err',
]
