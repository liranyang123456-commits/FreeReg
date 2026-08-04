# -*- coding: utf-8 -*-
"""
M5: 4D 动态场景渲染评测 (RTW 形变场驱动 3DGS)
=============================================

架构 (4D-GS 可解释轻量版):
    静态 3DGS (高斯集) + FFD 形变场 (锚点=控制点)
    → 每帧: μ(t) = μ₀ + W(μ₀)·δ(t), 协方差近似不变 (小形变)
    → 动态渲染

协议:
  1. 训练静态 3DGS (3 视角, duck)
  2. 施加两个已知 FFD 形变 (t1 轻度, t2 局部凹痕):
       GT:     稠密点泼溅(形变后) —— 真值动态帧
       本方案: 同一 δ 驱动高斯 μ  —— 动态渲染帧
       对照:   不形变(静态高斯)渲染 —— 显示建模形变的必要性
  3. 指标: PSNR (动态渲染 vs GT动态), 静态对照 vs GT动态

用法:
    conda activate py3d
    python scripts/run_4dgs_eval.py
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import sys
import time
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.geometry import make_T, exp_so3
from src.pipeline import load_mesh_points, render_model, lambert_colors
from src.nonrigid_rtw import FFDLattice
from src.gaussian_renderer import (GaussianModel, render_gaussians,
                                   optimize_gaussians, psnr)
import torch

OUT = Path('outputs/4dgs')
BOP_MODEL = r'D:\bop\lm\models\obj_000008.ply'
K0 = np.array([[572.4, 0, 325.3], [0, 573.6, 242.0], [0, 0, 1.0]])
H, W = 240, 320


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('=' * 66)
    print('M5: 4D 动态渲染 (RTW 形变场 × 3DGS)')
    print('=' * 66)
    K = K0.copy()
    K[:2] *= 0.5

    pts_dense, normals = load_mesh_points(BOP_MODEL, n_points=20000)
    mu = pts_dense.mean(axis=0)
    pts_dense -= mu
    rng = np.random.default_rng(0)
    sel = rng.choice(len(pts_dense), 4000, replace=False)
    pts_sparse = pts_dense[sel]

    views_def = [
        ('v1', exp_so3(np.array([0.15, -0.25, 0.10])), [0, 0, 900]),
        ('v2', exp_so3(np.array([-0.20, 0.15, -0.05])), [30, 10, 900]),
        ('v3', exp_so3(np.array([0.10, 0.30, -0.12])), [-25, -10, 900]),
    ]
    T_view = make_T(views_def[0][1], np.array(views_def[0][2], dtype=float))

    # ---------- 训练静态 3DGS ----------
    print('\n[1] 训练静态 3DGS (3视角, 150 iters)...')
    gts_static = {}
    for name, R, t in views_def:
        T = make_T(R, np.array(t, dtype=float))
        colors = lambert_colors(normals, R, base_rgb=(0.75, 0.55, 0.45))
        out = render_model(pts_dense, colors, T, K, H, W, radius=2,
                           device=device)
        gts_static[name] = out['feat'].cpu().numpy()
    colors_init = lambert_colors(normals[sel], np.eye(3),
                                 base_rgb=(0.75, 0.55, 0.45))
    model = GaussianModel.from_points(pts_sparse, colors=colors_init,
                                      device=device)
    train_views = [{'T_wc': make_T(R, np.array(t, dtype=float)),
                    'image': gts_static[name]} for name, R, t in views_def]
    optimize_gaussians(model, train_views, K, iters=150, device=device,
                       verbose=False)
    p_static = psnr(render_gaussians(model, T_view, K, H, W,
                                     device=device)['feat'].cpu().numpy(),
                    gts_static['v1'])
    print(f'    静态3DGS PSNR(v1) = {p_static:.2f} dB')

    # ---------- 构造 FFD 形变场 (锚点) ----------
    lat = FFDLattice(pts_dense.min(0), pts_dense.max(0), grid=(4, 4, 4),
                     device=device)
    W_dense = lat.weights(pts_dense)

    def warp(deltas, pts, W):
        d = torch.as_tensor(deltas, dtype=torch.float32, device=device)
        return pts + (W @ d).cpu().numpy() if isinstance(pts, np.ndarray) \
            else pts + W @ d

    deform_cases = []
    gx, gy, gz = lat.grid
    ii, jj, kk = np.meshgrid(np.arange(gx), np.arange(gy), np.arange(gz),
                             indexing='ij')
    C = lat.bbox_min + np.stack([ii, jj, kk], -1).reshape(-1, 3) * lat.step
    w_grad = (C[:, 0] - C[:, 0].mean()) / (np.ptp(C[:, 0]) + 1e-9)
    z_axis = T_view[:3, :3].T @ np.array([0, 0, 1.0])
    x_axis = T_view[:3, :3].T @ np.array([1.0, 0, 0])
    # t1: 全局平滑弯曲 (沿相机x横向渐变 12mm —— 大横向位移)
    delta1 = (12.0 * w_grad)[:, None] * x_axis[None, :]
    deform_cases.append(('t1_横弯12mm', delta1))
    # t2: 局部凹痕 (高斯, 幅度10mm, σ=15mm, 深度+横向复合方向)
    rng2 = np.random.default_rng(3)
    n_cam = normals @ T_view[:3, :3].T
    c_idx = int(np.argmax(-n_cam[:, 2] + rng2.normal(0, 0.1, len(n_cam))))
    c0 = pts_dense[c_idx]
    dent_dir = z_axis + 0.6 * x_axis
    dent_dir = dent_dir / np.linalg.norm(dent_dir)
    d2c = ((C - c0) ** 2).sum(axis=1)
    delta2 = (10.0 * np.exp(-d2c / (2 * 15.0 ** 2)))[:, None] * dent_dir[None, :]
    deform_cases.append(('t2_凹痕10mm', delta2))

    # ---------- 动态渲染对比 ----------
    print('\n[2] 动态帧渲染对比:')
    print(f'{"帧":>14} | {"PSNR 静态模型→动态GT":>22} | '
          f'{"PSNR 4D渲染→动态GT":>20}')
    rows_img = []
    mu0 = model.mu.detach().clone()
    for name, delta in deform_cases:
        # GT 动态帧: 稠密点形变后泼溅
        pts_def = warp(delta, torch.as_tensor(pts_dense, device=device),
                       W_dense).cpu().numpy()
        colors_d = lambert_colors(normals, T_view[:3, :3],
                                  base_rgb=(0.75, 0.55, 0.45))
        gt_dyn = render_model(pts_def, colors_d, T_view, K, H, W,
                              radius=2, device=device)['feat'].cpu().numpy()
        # 本方案: 同一 δ 驱动高斯均值 (锚点形变)
        W_gs = lat.weights(pts_sparse)
        d_t = torch.as_tensor(delta, dtype=torch.float32, device=device)
        model.mu.data = mu0 + W_gs @ d_t
        r_dyn = render_gaussians(model, T_view, K, H, W,
                                 device=device)['feat'].cpu().numpy()
        # 对照: 不形变
        model.mu.data = mu0
        r_static = render_gaussians(model, T_view, K, H, W,
                                    device=device)['feat'].cpu().numpy()
        # 物体区域 PSNR (全图被背景稀释, 掩盖真实动态误差)
        fg = (gt_dyn.max(-1) > 0.05) | (r_dyn.max(-1) > 0.05) \
            | (r_static.max(-1) > 0.05)

        def psnr_fg(a, b, m):
            mse = float(((a - b) ** 2)[m].mean())
            return 99.0 if mse < 1e-10 else float(-10 * np.log10(mse))
        p_dyn = psnr_fg(r_dyn, gt_dyn, fg)
        p_sta = psnr_fg(r_static, gt_dyn, fg)
        print(f'{name:>14} | {p_sta:22.2f} | {p_dyn:20.2f}')
        rows_img.append(np.concatenate([gt_dyn, r_static, r_dyn], axis=1))

    comp = (np.clip(np.concatenate(rows_img, axis=0), 0, 1) * 255
                ).astype(np.uint8)
    cv2.imwrite(str(OUT / 'dynamic.png'),
                cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
    print(f'\n对比图: {OUT / "dynamic.png"} (GT动态|静态渲染|4D渲染 × 2帧)')
    print('=' * 66)


if __name__ == '__main__':
    main()
