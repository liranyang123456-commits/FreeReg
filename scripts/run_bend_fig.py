# -*- coding: utf-8 -*-
"""
Fig.6 真实弯曲(bend)形变恢复实验
====================================================
替代原手绘示意图: 用 real_colon 真实结肠表面网格(含自然不对称),
施加已知整体弯曲 (bend), 在已知位姿下用 RTW 恢复形变, 报告真实
端到端形变恢复误差, 并重绘 Fig.6 投影点对比图.

协议 (聚焦形变恢复, 位姿取 GT 以避免对称翻转干扰):
  - GT: 对 base 表面施加 bend(amp), 渲染深度+噪声
  - 刚性: base 表面无形变 -> 与 GT bend 表面 NN 距离 (未补偿误差)
  - RTW: rtw_optimize_projective 恢复形变 -> 与 GT bend 表面 NN 距离
  - recovery error = NN 均值 (mm); 同时绘制投影点 rigid/GT/RTW

用法:
    conda activate py3d
    python scripts/run_bend_fig.py
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline import render_model
from src.nonrigid_rtw import rtw_optimize_projective
from src.pose_estimation import nearest_neighbors, multistart_icp_register
from src.geometry import make_T, exp_so3, transform
from src.metrics import pose_errors

RC = Path(r'E:\Diff_Rending_Re_3D\dataset\real_colon')
OUT = Path('outputs/bend_fig')
OUT.mkdir(parents=True, exist_ok=True)
H, W = 480, 640
K = np.array([[572.4, 0, 325.3], [0, 573.6, 242.0], [0, 0, 1.0]])
SCALE = 100.0  # real_colon 坐标 x100 -> extent ~146mm, 使 bend 20/35/50mm 合理


def subdivide_surface(nodes, tris, n_per=8):
    pts = []
    for a, b, c in tris:
        for i in range(n_per + 1):
            for j in range(n_per + 1 - i):
                u, v = i / n_per, j / n_per
                w = 1 - u - v
                pts.append(u * nodes[a] + v * nodes[b] + w * nodes[c])
    return np.array(pts)


def apply_bend(pts, amp_mm, u=None, bend_dir=None):
    """绕主轴 u 弯曲, 幅度 amp_mm (正弦剖面, 中部最大), 位移沿 bend_dir.
    bend_dir 取视线在模型系的方向 -> bend 改变深度, 单目可观测."""
    mean = pts.mean(0)
    pts = pts - mean
    cov = np.cov(pts.T)
    _, eigvec = np.linalg.eigh(cov)
    if u is None:
        u = eigvec[:, -1]           # 主轴 (管长方向)
    if bend_dir is None:
        bend_dir = eigvec[:, -2]
    bend_dir = bend_dir / np.linalg.norm(bend_dir)
    s = pts @ u
    s01 = (s - s.min()) / max(s.max() - s.min(), 1e-9)
    disp = (amp_mm * np.sin(np.pi * s01))[:, None] * bend_dir[None, :]
    return pts + disp + mean


def proj2d(pts_cam, K, H, W):
    z = pts_cam[:, 2]
    valid = z > 1.0
    uv = (K[:2, :] @ (pts_cam[valid].T / z[valid])).T
    m = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    return uv[m]


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gt = torch.load(RC / 'scene_0000' / 'gt.pt', map_location='cpu',
                    weights_only=False)
    nodes = gt['scene_spec']['nodes'].numpy().astype(np.float64) * SCALE
    tris = gt['scene_spec']['surface_tris'].numpy().astype(np.int64)
    base_surf = subdivide_surface(nodes, tris, n_per=8)
    base_surf -= base_surf.mean(0)
    extent = float(np.linalg.norm(nodes.max(0) - nodes.min(0)))
    print(f'colon scene_0000: {len(base_surf)} surf pts, extent={extent:.1f}mm')

    rng = np.random.default_rng(0)
    # 固定一个能看清弯曲的视角
    T_gt = make_T(exp_so3(np.array([0.05, 0.03, 0.0])),
                  np.array([0.0, 0.0, 320.0]))
    colors = np.tile(np.array([[0.85, 0.65, 0.60]]), (len(base_surf), 1))
    # bend 方向 = 视线(模型系)与主轴横向的混合: 既有可见弧形, 又有深度分量(可观测)
    _pts = base_surf - base_surf.mean(0)
    _ev, _vec = np.linalg.eigh(np.cov(_pts.T))
    U_AXIS = _vec[:, -1]
    VIEW_M = T_gt[:3, :3].T @ np.array([0, 0, 1.0])
    BEND_DIR = VIEW_M + _vec[:, -2]
    BEND_DIR = BEND_DIR / np.linalg.norm(BEND_DIR)

    results = {}
    for amp in [20, 35, 50]:
        bend_surf = apply_bend(base_surf, amp, u=U_AXIS, bend_dir=BEND_DIR)
        out = render_model(bend_surf, colors, T_gt, K, H, W, radius=1,
                           device=device)
        depth = out['depth'].cpu().numpy().astype(np.float64)
        fg = depth > 0
        depth[fg] += rng.normal(0, extent * 0.005, int(fg.sum()))

        # 刚性 (同 pose, 无形变) -> 未补偿误差
        P_gt = transform(T_gt, bend_surf)
        P_rig = transform(T_gt, base_surf)
        _, d_rig = nearest_neighbors(P_rig, P_gt, device=device)
        err_rig = float(d_rig.mean())

        # RTW 形变恢复 (已知 pose, 不回退位姿 -> 专注形变)
        rtw = rtw_optimize_projective(
            base_surf, depth, K, T_gt, grid=(6, 6, 6), iters=400, lr=3e-3,
            lambda_mag=0.5, lambda_sm=0.5, lambda_p2p=1.0, lambda_pose=1.0,
            inlier_hi=0.15 * extent, inlier_lo=0.03 * extent,
            device=device, seed=0)
        P_rtw = transform(rtw['T'], rtw['V_def'])
        _, d_rtw = nearest_neighbors(P_rtw, P_gt, device=device)
        err_rtw = float(d_rtw.mean())
        results[amp] = (err_rig, err_rtw)
        print(f'  bend {amp}mm: rigid={err_rig:.2f}mm  RTW={err_rtw:.2f}mm'
              f'  (恢复 {100*(1-err_rtw/err_rig):.0f}%)')

    # ---- 重绘 Fig.6: 投影点 rigid / GT / RTW 对比 ----
    make_figure(base_surf, T_gt, colors, results, extent, device)


def make_figure(base_surf, T_gt, colors, results, extent, device):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    amps = [20, 35, 50]
    fig, axes = plt.subplots(3, 3, figsize=(9.0, 7.2))
    cols = ['Rigid baseline', 'Ground-truth bend', 'RTW recovery']
    cc = ['#3773d2', '#e07023', '#23af55']
    for j, name in enumerate(cols):
        axes[0, j].set_title(name, fontsize=13, color=cc[j])
    rng = np.random.default_rng(1)
    _pts = base_surf - base_surf.mean(0)
    _ev, _vec = np.linalg.eigh(np.cov(_pts.T))
    U_AXIS = _vec[:, -1]
    VIEW_M = T_gt[:3, :3].T @ np.array([0, 0, 1.0])
    BEND_DIR = VIEW_M + _vec[:, -2]
    BEND_DIR = BEND_DIR / np.linalg.norm(BEND_DIR)
    for i, amp in enumerate(amps):
        bend_surf = apply_bend(base_surf, amp, u=U_AXIS, bend_dir=BEND_DIR)
        out = render_model(bend_surf, colors, T_gt, K, H, W, radius=1,
                           device=device)
        depth = out['depth'].cpu().numpy().astype(np.float64)
        fg = depth > 0
        depth[fg] += rng.normal(0, extent * 0.005, int(fg.sum()))
        rtw = rtw_optimize_projective(
            base_surf, depth, K, T_gt, grid=(6, 6, 6), iters=400, lr=3e-3,
            lambda_mag=0.5, lambda_sm=0.5, lambda_p2p=1.0, lambda_pose=1.0,
            inlier_hi=0.15 * extent, inlier_lo=0.03 * extent,
            device=device, seed=1)
        P = {'rigid': transform(T_gt, base_surf),
             'gt': transform(T_gt, bend_surf),
             'rtw': transform(rtw['T'], rtw['V_def'])}
        for j, mode in enumerate(['rigid', 'gt', 'rtw']):
            uv = proj2d(P[mode], K, H, W)
            ax = axes[i, j]
            ax.scatter(uv[:, 0], uv[:, 1], s=1, c=cc[j])
            ax.invert_yaxis(); ax.set_aspect('equal'); ax.axis('off')
            if j == 0:
                ax.set_ylabel(f'{amp} mm', fontsize=12)
            if mode == 'rtw':
                e_rig, e_rtw = results[amp]
                ax.set_xlabel(f'recovery err {e_rtw:.2f} vs rigid {e_rig:.2f} mm',
                              fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / 'fig6_bend_real.png', dpi=200)
    fig.savefig(Path('outputs/paper/figs') / 'fig6_dent.png', dpi=200)
    print('saved figure -> outputs/paper/figs/fig6_dent.png')


if __name__ == '__main__':
    main()
