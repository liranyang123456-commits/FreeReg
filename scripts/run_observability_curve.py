# -*- coding: utf-8 -*-
"""
Observability 梯度实验 —— 直接验证 Theorem (rank deficiency)
================================================================
把一个目标形变分解到单目深度残差算子的【可观测子空间】与【不可观测子空间】,
按 cosθ/sinθ 渐变二者比例 (θ=0 全可观测, θ=90° 全不可观测), 测 RTW 形变恢复率.
预期: 恢复率随不可观测比例上升而下降 → 把 rank-deficiency 定理变成可测曲线.

协议 (controlled): 真实结肠网格(real_colon scene), 已知 GT 位姿(聚焦形变恢复,
避开初值翻转干扰), 用修正后的 rtw_optimize_projective.

用法:
    conda activate py3d
    python scripts/run_observability_curve.py
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline import render_model
from src.nonrigid_rtw import (FFDLattice, rtw_optimize_projective,
                              rtw_optimize_projective_obs,
                              compute_observability_basis)
from src.pose_estimation import nearest_neighbors
from src.geometry import make_T, exp_so3, transform

RC = Path(r'E:\Diff_Rending_Re_3D\dataset\real_colon')
OUT = Path('outputs/observability')
OUT.mkdir(parents=True, exist_ok=True)
H, W = 480, 640
K = np.array([[572.4, 0, 325.3], [0, 573.6, 242.0], [0, 0, 1.0]])
SCALE = 100.0


def subdivide(nodes, tris, n_per=8):
    pts = []
    for a, b, c in tris:
        for i in range(n_per + 1):
            for j in range(n_per + 1 - i):
                u, v = i / n_per, j / n_per
                w = 1 - u - v
                pts.append(u * nodes[a] + v * nodes[b] + w * nodes[c])
    return np.array(pts)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gt = torch.load(RC / 'scene_0000' / 'gt.pt', map_location='cpu',
                    weights_only=False)
    nodes = gt['scene_spec']['nodes'].numpy().astype(np.float64) * SCALE
    tris = gt['scene_spec']['surface_tris'].numpy().astype(np.int64)
    base = subdivide(nodes, tris)
    base -= base.mean(0)
    extent = float(np.linalg.norm(nodes.max(0) - nodes.min(0)))
    print(f'scene_0000: {len(base)} pts, extent={extent:.1f}mm')

    T_gt = make_T(exp_so3(np.array([0.05, 0.03, 0.0])), np.array([0., 0., 320.]))
    R_gt = T_gt[:3, :3]
    colors = np.tile([[0.85, 0.65, 0.60]], (len(base), 1))

    # FFD lattice + 目标形变 (随机平滑 bend, 幅度 ~0.15*extent)
    lat = FFDLattice(base.min(0), base.max(0), grid=(6, 6, 6), device=device)
    rng = np.random.default_rng(3)
    delta_t = rng.normal(0, 1.0, (lat.n_ctrl, 3))
    # 平滑化
    edges = lat._edges.cpu().numpy()
    for _ in range(2):
        dn = delta_t.copy(); cnt = np.ones(len(delta_t))
        for a, b in edges:
            dn[a] += delta_t[b]; dn[b] += delta_t[a]; cnt[a] += 1; cnt[b] += 1
        delta_t = dn / cnt[:, None]
    delta_t *= (0.04 * extent) / np.linalg.norm(delta_t, axis=1).mean()
    Wd = lat.weights(base)                      # (N, n_ctrl)
    if torch.is_tensor(Wd):
        Wd = Wd.detach().cpu().numpy()

    # 可观测子空间 (基于 GT 位姿的深度残差算子)
    Wt = torch.as_tensor(Wd, dtype=torch.float32, device=device)
    Rt = torch.as_tensor(R_gt, dtype=torch.float32, device=device)
    V_obs, S, rank = compute_observability_basis(Wt, Rt)
    print(f'observability rank = {rank}/{3*lat.n_ctrl} ({100*rank/(3*lat.n_ctrl):.0f}%)')

    # 分解目标形变
    dv = torch.as_tensor(delta_t.reshape(-1), dtype=torch.float32, device=device)
    dv_obs = V_obs @ (V_obs.T @ dv)
    dv_unobs = dv - dv_obs
    n_obs = dv_obs.norm(); n_un = dv_unobs.norm()
    print(f'|obs|={n_obs:.2f}  |unobs|={n_un:.2f}  (unobs 占比 {100*n_un/(n_obs+n_un):.0f}%)')

    def apply_delta(dvec):
        d = dvec.reshape(lat.n_ctrl, 3).cpu().numpy()
        return base + (Wd @ d)

    def recover(md, seed=0):
        out = render_model(md, colors, T_gt, K, H, W, radius=2, device=device)
        depth = out['depth'].cpu().numpy().astype(np.float64)
        fg = depth > 0
        depth[fg] += rng.normal(0, extent * 0.005, int(fg.sum()))
        # 观测引导 RTW (论文核心): 约束在可观测子空间
        rtw = rtw_optimize_projective_obs(base, depth, K, T_gt,
                                          grid=(6, 6, 6), iters=900, lr=8e-3,
                                          lambda_mag=1e-5, lambda_sm=5e-4,
                                          lambda_unobs=1e-2,
                                          inlier_hi=0.2*extent,
                                          inlier_lo=0.05*extent,
                                          device=device, seed=seed)
        return rtw

    # 基线: rigid (无形变) 对每个 GT 形变的端到端误差
    angles = [0, 30, 45, 60, 75, 90]
    rows = []
    for th in angles:
        ct, st = np.cos(np.radians(th)), np.sin(np.radians(th))
        dv_th = n_obs * (ct * dv_obs / n_obs + st * dv_unobs / n_un)
        md = apply_delta(dv_th)
        P_gt = transform(T_gt, md)
        _, d_rig = nearest_neighbors(transform(T_gt, base), P_gt, device=device)
        err_rig = float(d_rig.mean())
        rtw = recover(md)
        _, d_rtw = nearest_neighbors(transform(rtw['T'], rtw['V_def']), P_gt,
                                     device=device)
        err_rtw = float(d_rtw.mean())
        rec_nn = 100 * (1 - err_rtw / max(err_rig, 1e-9))
        # 形变恢复率: 估计形变在 GT 形变方向的投影比例 (直接对应 observability)
        d_est = rtw['delta'].reshape(-1)
        if torch.is_tensor(d_est):
            d_est = d_est.detach().cpu().numpy()
        d_gt = dv_th.cpu().numpy()
        proj = float(np.dot(d_est, d_gt) / max(np.dot(d_gt, d_gt), 1e-9))
        rec_deform = 100 * proj
        rows.append((th, err_rig, err_rtw, rec_nn, rec_deform))
        print(f'  θ={th:3d}°: rigid={err_rig:.2f} RTW={err_rtw:.2f} '
              f'NN恢复={rec_nn:.0f}%  形变恢复率={rec_deform:.0f}%')

    import csv
    with open(OUT / 'observability_curve.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['theta_deg', 'rigid_err', 'rtw_err', 'rec_nn_pct',
                    'rec_deform_pct'])
        w.writerows(rows)
    make_figure(rows, rank, lat.n_ctrl)
    print('saved ->', OUT / 'observability_curve.csv')


def make_figure(rows, rank, n_ctrl):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    th = [r[0] for r in rows]
    rec_d = [r[4] for r in rows]   # 形变恢复率
    rec_nn = [r[3] for r in rows]  # 端到端 NN 恢复率
    fig, ax = plt.subplots(figsize=(4.8, 3.5))
    ax.plot(th, rec_d, 'o-', color='#23af55', lw=2, ms=6,
            label='deformation recovery (shape)')
    ax.plot(th, rec_nn, 's--', color='#3773d2', lw=1.6, ms=5,
            label='end-to-end surface error reduction')
    ax.axhline(0, color='#888', lw=0.8, ls=':')
    ax.set_xlabel('unobservable fraction of the deformation (deg)')
    ax.set_ylabel('recovery / error reduction (%)')
    ax.set_title(f'Recovery degrades with unobservability (rank {rank}/{3*n_ctrl})')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / 'observability_curve.png', dpi=200)
    fig.savefig(Path('outputs/paper/figs') / 'fig_observability_curve.png', dpi=200)
    print('saved figure')


if __name__ == '__main__':
    main()
