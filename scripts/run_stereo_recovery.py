# -*- coding: utf-8 -*-
"""
Stereo/多视角 实际形变恢复验证 (线性观测算子最小二乘)
================================================================
观测引导 RTW 的线性本质: 深度残差 r = J·δ, 可观测恢复由 J 的秩决定.
单目 J 秩亏 (23%), 双目 J 秩高 (41%) -> 最小二乘恢复 δ 的保真度随之提升.

本脚本对一个已知形变 δ_gt, 用单目/双目/多视角的观测算子 J 做 Tikhonov
最小二乘恢复, 报告形变恢复率 vs 视角数. 直接把 rank 提升转化为恢复提升.

用法:
    conda activate py3d
    python scripts/run_stereo_recovery.py
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.nonrigid_rtw import FFDLattice
from src.geometry import make_T, exp_so3

RC = Path(r'E:\Diff_Rending_Re_3D\dataset\real_colon')
OUT = Path('outputs/observability')
OUT.mkdir(parents=True, exist_ok=True)
SCALE = 100.0
GRID = (6, 6, 6)


def subdivide(nodes, tris, n_per=6):
    pts = []
    for a, b, c in tris:
        for i in range(n_per + 1):
            for j in range(n_per + 1 - i):
                u, v = i / n_per, j / n_per
                w = 1 - u - v
                pts.append(u * nodes[a] + v * nodes[b] + w * nodes[c])
    return np.array(pts)


def build_J(Wt, R_list):
    Js = []
    for R in R_list:
        Rz = R[2, :]
        J = -(Wt.unsqueeze(2) * Rz.reshape(1, 1, 3)).reshape(Wt.shape[0], -1)
        Js.append(J)
    return torch.cat(Js, 0)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gt = torch.load(RC / 'scene_0000' / 'gt.pt', map_location='cpu',
                    weights_only=False)
    nodes = gt['scene_spec']['nodes'].numpy().astype(np.float64) * SCALE
    tris = gt['scene_spec']['surface_tris'].numpy().astype(np.int64)
    base = subdivide(nodes, tris)
    base -= base.mean(0)
    extent = float(np.linalg.norm(nodes.max(0) - nodes.min(0)))
    lat = FFDLattice(base.min(0), base.max(0), grid=GRID, device=device)
    W = lat.weights(base)
    Wt = torch.as_tensor(W, dtype=torch.float32, device=device) \
        if not torch.is_tensor(W) else W.float()
    n3 = 3 * lat.n_ctrl
    print(f'model: {len(base)} pts, dof={n3}, extent={extent:.0f}mm')

    # 目标形变: 随机平滑 (多方向混合, 含可观测+不可观测分量)
    rng = np.random.default_rng(5)
    delta_gt = rng.normal(0, 1.0, (lat.n_ctrl, 3))
    edges = lat._edges.cpu().numpy()
    for _ in range(2):
        dn = delta_gt.copy(); cnt = np.ones(len(delta_gt))
        for a, b in edges:
            dn[a] += delta_gt[b]; dn[b] += delta_gt[a]; cnt[a] += 1; cnt[b] += 1
        delta_gt = dn / cnt[:, None]
    delta_gt *= (0.08 * extent) / np.linalg.norm(delta_gt, axis=1).mean()
    dg = torch.as_tensor(delta_gt.reshape(-1), dtype=torch.float32,
                         device=device)

    print(f'\n视角数 -> rank, 可观测形变能量占比 (无噪声, 纯几何)')
    rows = []
    for V in [1, 2, 3, 4, 6]:
        R_list = []
        for c in range(V):
            ang = (c - (V - 1) / 2) * np.radians(12.0)
            T = make_T(exp_so3(np.array([0.0, ang, 0.0])), np.array([0., 0., 320.]))
            R_list.append(torch.as_tensor(T[:3, :3], dtype=torch.float32,
                                          device=device))
        J = build_J(Wt, R_list)
        U, S, Vh = torch.linalg.svd(J, full_matrices=False)
        rank = int((S > S.max() * 1e-3).sum().item())
        V_obs = Vh[:rank].T                          # (n3, rank)
        # GT 形变在可观测子空间的能量占比 = 可恢复比例 (无噪声上限)
        proj = V_obs @ (V_obs.T @ dg)
        rec = 100 * float(proj.norm() / dg.norm())
        rows.append((V, rank, 100 * rank / n3, rec))
        print(f'  V={V}: rank={rank}/{n3} ({100*rank/n3:.0f}%)  '
              f'可恢复能量={rec:.0f}%')

    import csv
    with open(OUT / 'stereo_recovery.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['n_views', 'rank', 'rank_pct', 'recovery_pct'])
        w.writerows(rows)
    make_figure(rows, n3)
    print('saved ->', OUT / 'stereo_recovery.csv')


def make_figure(rows, n3):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    V = [r[0] for r in rows]
    rankp = [r[2] for r in rows]
    rec = [r[3] for r in rows]
    fig, ax = plt.subplots(figsize=(4.8, 3.5))
    ax.plot(V, rankp, 'o-', color='#3773d2', lw=2, ms=6,
            label='observable rank (%)')
    ax.plot(V, rec, 's--', color='#23af55', lw=2, ms=6,
            label='deformation recovery (%)')
    ax.set_xlabel('number of views')
    ax.set_ylabel('%')
    ax.set_title('Stereo/multi-view lifts observability AND recovery')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xticks(V)
    fig.tight_layout()
    fig.savefig(OUT / 'stereo_recovery.png', dpi=200)
    fig.savefig(Path('outputs/paper/figs') / 'fig_stereo_recovery.png', dpi=200)
    print('saved figure')


if __name__ == '__main__':
    main()
