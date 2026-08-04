# -*- coding: utf-8 -*-
"""
torch z-buffer 点泼溅光栅化器 (自包含, CUDA 加速)
=================================================

用途:
  1) AR 渲染: 把注册后的 3D 模型点渲染成 RGB + 深度图
  2) 深度感知合成: 与真实深度 z-buffer 比较 → 正确处理遮挡
  3) 合成数据生成: 已知位姿渲染深度 → 单元测试 / RTW 合成评测

与完整 3DGS 渲染器的关系:
  本模块是"点泼溅"(point splatting) —— 3DGS 光栅化的无高斯核简化版,
  接口语义一致 (输入点+特征+K+位姿, 输出 RGB+深度), M4 阶段可原位替换为
  官方 3DGS CUDA 光栅化器, 管线上游(注册/滤波)不受影响。

依赖: torch (CUDA 可选)
作者: ZCode · 2026-08-03
"""
import torch


def splat_render(points_cam: torch.Tensor,
                 feats: torch.Tensor,
                 K: torch.Tensor,
                 H: int, W: int,
                 radius: int = 0):
    """
    把相机系 3D 点泼溅渲染为特征图 + 深度图 (z-buffer 取最近)

    Args:
        points_cam: (N,3) float, 相机系点 (z>0 有效)
        feats:      (N,C) float, 每点特征 (C=3 即 RGB, 范围[0,1])
        K:          (3,3) float, 内参
        H, W:       输出图尺寸
        radius:     泼溅半径(像素). 0=单点, 1=3x3, 2=5x5 (填补稀疏空洞)

    Returns:
        dict:
          'feat':  (H,W,C) 渲染特征图 (背景=0)
          'depth': (H,W)   深度图 (背景=0)
          'mask':  (H,W)   bool, 有效渲染像素
    """
    device = points_cam.device
    dtype = points_cam.dtype
    C = feats.shape[1]

    z = points_cam[:, 2]
    u = K[0, 0] * points_cam[:, 0] / torch.clamp(z, min=1e-12) + K[0, 2]
    v = K[1, 1] * points_cam[:, 1] / torch.clamp(z, min=1e-12) + K[1, 2]
    ui = torch.round(u).long()
    vi = torch.round(v).long()

    inb = (z > 1e-6) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    ui, vi, z = ui[inb], vi[inb], z[inb]
    feats_in = feats[inb]

    # 泼溅偏移集合
    offsets = [(0, 0)] if radius <= 0 else [
        (dy, dx) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)
    ]

    # 第一遍: 全局 z-buffer (amin)
    zbuf = torch.full((H * W,), float('inf'), device=device, dtype=dtype)
    for dy, dx in offsets:
        u2 = ui + dx
        v2 = vi + dy
        ok = (u2 >= 0) & (u2 < W) & (v2 >= 0) & (v2 < H)
        lin = (v2[ok] * W + u2[ok]).long()
        zbuf.scatter_reduce_(0, lin, z[ok], reduce='amin')

    # 第二遍: 胜者点写入特征 (同深度多点取均值)
    fbuf = torch.zeros((H * W, C), device=device, dtype=dtype)
    cnt = torch.zeros((H * W, 1), device=device, dtype=dtype)
    for dy, dx in offsets:
        u2 = ui + dx
        v2 = vi + dy
        ok = (u2 >= 0) & (u2 < W) & (v2 >= 0) & (v2 < H)
        lin = (v2[ok] * W + u2[ok]).long()
        win = (z[ok] - zbuf[lin]).abs() < 1e-6
        lin_w = lin[win]
        fbuf.index_add_(0, lin_w, feats_in[ok][win])
        cnt.index_add_(0, lin_w, torch.ones((int(lin_w.shape[0]), 1), device=device, dtype=dtype))

    feat_img = (fbuf / cnt.clamp(min=1.0)).reshape(H, W, C)
    depth_img = zbuf.reshape(H, W)
    mask_img = torch.isfinite(depth_img)
    depth_img = torch.where(mask_img, depth_img, torch.zeros((), device=device, dtype=dtype))

    return {'feat': feat_img, 'depth': depth_img, 'mask': mask_img}


def composite_depth_aware(real_rgb: torch.Tensor,
                          real_depth: torch.Tensor,
                          virt_rgb: torch.Tensor,
                          virt_depth: torch.Tensor,
                          alpha: float = 0.85):
    """
    深度感知 AR 合成 (z-buffer 比较, 处理 真实↔虚拟 遮挡)

    Args:
        real_rgb:   (H,W,3) 真实图像 [0,1]
        real_depth: (H,W)   真实深度 (0=无深度 → 视为无穷远)
        virt_rgb:   (H,W,3) 虚拟渲染 [0,1]
        virt_depth: (H,W)   虚拟深度 (0=无渲染)
        alpha:      虚拟层不透明度

    Returns:
        composite: (H,W,3) 合成图
        occ_mask:  (H,W) bool, 虚拟物体可见(未被真实场景遮挡)的像素
    """
    real_z = torch.where(real_depth > 0, real_depth,
                         torch.full_like(real_depth, float('inf')))
    occ_mask = (virt_depth > 0) & (virt_depth < real_z)
    a = alpha * occ_mask.unsqueeze(-1).to(real_rgb.dtype)
    comp = real_rgb * (1 - a) + virt_rgb * a
    return comp, occ_mask
