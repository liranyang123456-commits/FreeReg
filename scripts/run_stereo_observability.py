# -*- coding: utf-8 -*-
"""
Stereo / 多视角 observability 提升分析
================================================================
单目深度残差算子 J = -W ⊗ R_z 只含单一视线方向 -> 秩亏 (23%).
多视角把各视角的 J_c 堆叠 J=[J_1;...;J_V], 每个视角贡献一个新的视线方向,
观测子空间 rank 随之提升. 本脚本定量测 rank 随 视角数/基线角度 的变化.

这是把"单目局限"转化为框架卖点的实验: FreeReg 支持任意输入,
stereo/多视角时可恢复的形变方向更多.

用法:
    conda activate py3d
    python scripts/run_stereo_observability.py
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


def obs_rank_multi(Wt, R_list, delta_gt=None, tau_ratio=1e-3):
    """多视角观测算子 J=[J_c] 的秩及目标形变可恢复能量。

    可恢复能量定义为 ||P_obs δ||² / ||δ||²，而不是投影向量范数比例。
    """
    Js = []
    for R in R_list:
        Rz = R[2, :]
        J = -(Wt.unsqueeze(2) * Rz.reshape(1, 1, 3)).reshape(Wt.shape[0], -1)
        Js.append(J)
    J = torch.cat(Js, 0)
    _, S, Vh = torch.linalg.svd(J, full_matrices=False)
    r = int((S > S.max() * tau_ratio).sum().item())
    energy_pct = float('nan')
    if delta_gt is not None and r > 0:
        V_obs = Vh[:r].T
        projected = V_obs @ (V_obs.T @ delta_gt)
        energy_pct = 100.0 * float(
            projected.square().sum() / delta_gt.square().sum().clamp(min=1e-12))
    return r, S, energy_pct


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gt = torch.load(RC / 'scene_0000' / 'gt.pt', map_location='cpu',
                    weights_only=False)
    nodes = gt['scene_spec']['nodes'].numpy().astype(np.float64) * SCALE
    tris = gt['scene_spec']['surface_tris'].numpy().astype(np.int64)
    base = subdivide(nodes, tris)
    base -= base.mean(0)
    lat = FFDLattice(base.min(0), base.max(0), grid=GRID, device=device)
    W = lat.weights(base)
    Wt = torch.as_tensor(W, dtype=torch.float32, device=device) \
        if not torch.is_tensor(W) else W.float()
    n_ctrl3 = 3 * lat.n_ctrl
    print(f'model: {len(base)} pts, FFD ctrl={lat.n_ctrl}, dof={n_ctrl3}')

    # 固定目标形变，用于“可恢复能量”而非仅报告秩。
    rng = np.random.default_rng(5)
    delta_np = rng.normal(0, 1.0, (lat.n_ctrl, 3))
    edges = lat._edges.detach().cpu().numpy()
    for _ in range(2):
        delta_next = delta_np.copy()
        count = np.ones(len(delta_np))
        for a, b in edges:
            delta_next[a] += delta_np[b]
            delta_next[b] += delta_np[a]
            count[a] += 1
            count[b] += 1
        delta_np = delta_next / count[:, None]
    delta_gt = torch.as_tensor(delta_np.reshape(-1), dtype=torch.float32,
                               device=device)

    rows = []
    # (1) rank vs 视角数 (固定基线间隔 12°)
    print('\n--- rank vs 视角数 (基线间隔 12°) ---')
    for V in [1, 2, 3, 4, 6]:
        R_list = []
        for c in range(V):
            ang = (c - (V - 1) / 2) * np.radians(12.0)
            T = make_T(exp_so3(np.array([0.0, ang, 0.0])),
                       np.array([0., 0., 320.]))
            R_list.append(torch.as_tensor(T[:3, :3], dtype=torch.float32,
                                          device=device))
        r, _, energy_pct = obs_rank_multi(Wt, R_list, delta_gt)
        pct = 100 * r / n_ctrl3
        rows.append(('views', V, 12.0, r, pct, energy_pct))
        print(f'  V={V}: rank={r}/{n_ctrl3} ({pct:.0f}%), '
              f'recoverable energy={energy_pct:.1f}%')

    # (2) rank vs 双目基线角度 (V=2)
    print('\n--- rank vs 双目基线角度 (V=2) ---')
    for sep_deg in [2, 5, 10, 20, 30, 45]:
        R_list = []
        for s in [-1, 1]:
            T = make_T(exp_so3(np.array([0.0, s * np.radians(sep_deg / 2), 0.0])),
                       np.array([0., 0., 320.]))
            R_list.append(torch.as_tensor(T[:3, :3], dtype=torch.float32,
                                          device=device))
        r, _, energy_pct = obs_rank_multi(Wt, R_list, delta_gt)
        pct = 100 * r / n_ctrl3
        rows.append(('baseline', 2, sep_deg, r, pct, energy_pct))
        print(f'  基线 ±{sep_deg/2:.0f}°: rank={r}/{n_ctrl3} ({pct:.0f}%), '
              f'recoverable energy={energy_pct:.1f}%')

    import csv
    with open(OUT / 'stereo_observability.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['mode', 'n_views', 'sep_deg', 'rank', 'rank_pct',
                    'recoverable_energy_pct'])
        w.writerows(rows)
    make_figure(rows, n_ctrl3)
    print('saved ->', OUT / 'stereo_observability.csv')


def make_figure(rows, n_ctrl3):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    vrows = [r for r in rows if r[0] == 'views']
    brows = [r for r in rows if r[0] == 'baseline']
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.4))
    a1.plot([r[1] for r in vrows], [r[4] for r in vrows], 'o-',
            color='#23af55', lw=2, ms=6)
    a1.set_xlabel('number of views')
    a1.set_ylabel('observable subspace (%)')
    a1.set_title('multi-view observability')
    a1.grid(alpha=0.3); a1.set_ylim(0, 100)
    a2.plot([r[2] for r in brows], [r[4] for r in brows], 's-',
            color='#3773d2', lw=2, ms=6)
    a2.set_xlabel('stereo baseline separation (deg)')
    a2.set_ylabel('observable subspace (%)')
    a2.set_title('stereo baseline effect (2 views)')
    a2.grid(alpha=0.3); a2.set_ylim(0, 100)
    fig.suptitle(f'Observability rank vs views / baseline (dof={n_ctrl3})')
    fig.tight_layout()
    fig.savefig(OUT / 'stereo_observability.png', dpi=200)
    print('saved figure')


if __name__ == '__main__':
    main()
