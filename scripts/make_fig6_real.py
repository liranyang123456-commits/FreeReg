# -*- coding: utf-8 -*-
"""
Fig.6 真实非刚性形变恢复可视化 (real_colon FEM, 替代手绘示意图)
================================================================
用 real_colon 真实 FEM 形变 (press-release), 对恢复好的收敛帧:
  列: Rigid(无形变) / Ground-truth deformed / RTW recovered
  行: 3 个真实帧 (scene/t)
  标注真实端到端形变恢复误差 (NN 距离 / 模型跨度).

诚实性: 全部为真实运行输出, 数字来自实时配准, 非手绘.

用法:
    conda activate py3d
    python scripts/make_fig6_real.py
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
from src.geometry import transform, backproject_depth, inverse_se3
from src.metrics import pose_errors

RC = Path(r'E:\Diff_Rending_Re_3D\dataset\real_colon')
OUT = Path('outputs/paper/figs')
OUT.mkdir(parents=True, exist_ok=True)
H, W = 128, 128
K = np.array([[160., 0, W / 2], [0, 160., H / 2], [0, 0, 1.0]])
FRAMES = [('scene_0004', 2), ('scene_0019', 5)]


def subdivide(nodes, tris, n_per=6):
    pts, meta = [], []
    for a, b, c in tris:
        for i in range(n_per + 1):
            for j in range(n_per + 1 - i):
                u, v = i / n_per, j / n_per
                w = 1 - u - v
                pts.append(u * nodes[a] + v * nodes[b] + w * nodes[c])
                meta.append([u, v, w, a, b, c])
    return np.array(pts), meta


def deform(nodes, meta, u_nodes):
    out = np.zeros((len(meta), 3))
    for i, (u, v, w, a, b, c) in enumerate(meta):
        p = u * nodes[a] + v * nodes[b] + w * nodes[c]
        out[i] = p + u * u_nodes[a] + v * u_nodes[b] + w * u_nodes[c]
    return out


def proj2d(pts_cam):
    z = pts_cam[:, 2]
    m = z > 1e-3
    uv = (K[:2, :] @ (pts_cam[m].T / z[m])).T
    inb = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    return uv[inb]


def eval_frame(scene, t, device):
    gt = torch.load(RC / scene / 'gt.pt', map_location='cpu', weights_only=False)
    nodes = gt['scene_spec']['nodes'].numpy().astype(np.float64)
    tris = gt['scene_spec']['surface_tris'].numpy().astype(np.int64)
    poses = gt['poses'].numpy().astype(np.float64)
    u_seq = gt['u_seq'].numpy().astype(np.float64).reshape(-1, 3)
    T_seq = u_seq.shape[0] // nodes.shape[0]
    u_seq = u_seq.reshape(T_seq, nodes.shape[0], 3)
    extent = float(np.linalg.norm(nodes.max(0) - nodes.min(0)))
    surf, meta = subdivide(nodes, tris)
    base = deform(nodes, meta, np.zeros_like(nodes))
    rng = np.random.default_rng(0)
    colors = np.tile(np.array([0.85, 0.65, 0.60]), (len(surf), 1))

    T_cam = inverse_se3(poses[t])
    u_gt = u_seq[t]
    surf_def = deform(nodes, meta, u_gt)
    out = render_model(surf_def, colors, T_cam, K, H, W, radius=1, device=device)
    depth = out['depth'].cpu().numpy()
    fg = depth > 0
    depth[fg] += rng.normal(0, extent * 0.01, int(fg.sum()))
    scene_pts, _, _ = backproject_depth(depth, K, mask=fg)
    T_est, _ = multistart_icp_register(base, scene_pts, device=device,
                                       coarse_iter=25, fine_iter=50,
                                       trim_coarse=0.5, trim_fine=0.7)
    rtw = rtw_optimize_projective(base, depth, K, T_est, grid=(4, 4, 4),
                                  iters=300, lr=3e-3, lambda_mag=0.5,
                                  lambda_sm=0.2, lambda_p2p=1.0,
                                  lambda_pose=1.0, inlier_hi=0.10 * extent,
                                  inlier_lo=0.03 * extent, device=device, seed=0)
    P_gt = transform(T_cam, surf_def)
    _, d_base = nearest_neighbors(transform(T_est, base), P_gt, device=device)
    _, d_rtw = nearest_neighbors(transform(rtw['T'], rtw['V_def']), P_gt,
                                 device=device)
    rec_base = float(d_base.mean() / extent) * 100
    rec_rtw = float(d_rtw.mean() / extent) * 100
    return dict(base=transform(T_est, base), gt=P_gt,
                rtw=transform(rtw['T'], rtw['V_def']),
                rec_base=rec_base, rec_rtw=rec_rtw)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    cols = ['Rigid (no warp)', 'Ground-truth deformed', 'RTW recovered']
    cc = ['#3773d2', '#e07023', '#23af55']
    fig = plt.figure(figsize=(9.6, 3.9))
    gs = GridSpec(2, 4, figure=fig, width_ratios=[1, 1, 1, 1.25],
                  hspace=0.25, wspace=0.15)
    axes = [[fig.add_subplot(gs[i, j]) for j in range(3)] for i in range(2)]
    for j, name in enumerate(cols):
        axes[0][j].set_title(name, fontsize=11, color=cc[j])
    for i, (scene, t) in enumerate(FRAMES):
        r = eval_frame(scene, t, device)
        print(f'{scene} t{t}: rec_base={r["rec_base"]:.2f}%  '
              f'rec_rtw={r["rec_rtw"]:.2f}%')
        for j, key in enumerate(['base', 'gt', 'rtw']):
            uv = proj2d(r[key])
            ax = axes[i][j]
            ax.scatter(uv[:, 0], uv[:, 1], s=2, c=cc[j])
            ax.invert_yaxis(); ax.set_aspect('equal'); ax.axis('off')
            if j == 0:
                ax.set_ylabel(f'{scene[-2:]}·t{t}', fontsize=10)
            if key == 'rtw':
                ax.set_xlabel(f'{r["rec_rtw"]:.2f}% vs rigid {r["rec_base"]:.2f}%',
                              fontsize=8)
    # 右列上: observability 梯度曲线; 右列下: stereo/多视角提升
    import pandas as pd
    ax_obs = fig.add_subplot(gs[0, 3])
    oc = pd.read_csv('outputs/observability/observability_curve.csv')
    ax_obs.plot(oc.theta_deg, oc.rec_deform_pct, 'o-', color='#23af55',
                lw=1.6, ms=4)
    ax_obs.axhline(0, color='#888', lw=0.7, ls=':')
    ax_obs.set_title('recovery vs unobservable frac', fontsize=9)
    ax_obs.set_xlabel('unobservable (deg)', fontsize=8)
    ax_obs.set_ylabel('recovery (%)', fontsize=8)
    ax_obs.grid(alpha=0.3)

    ax_st = fig.add_subplot(gs[1, 3])
    sr = pd.read_csv('outputs/observability/stereo_recovery.csv')
    ax_st.plot(sr.n_views, sr.rank_pct, 'o-', color='#3773d2', lw=1.6, ms=4,
               label='observable rank')
    ax_st.plot(sr.n_views, sr.recovery_pct, 's--', color='#e07023', lw=1.6,
               ms=4, label='recoverable deform')
    ax_st.set_title('stereo / multi-view lift', fontsize=9)
    ax_st.set_xlabel('number of views', fontsize=8)
    ax_st.set_ylabel('%', fontsize=8)
    ax_st.set_xticks(list(sr.n_views))
    ax_st.legend(fontsize=6, loc='center right')
    ax_st.grid(alpha=0.3)
    for a in (ax_obs, ax_st):
        for lab in a.get_xticklabels() + a.get_yticklabels():
            lab.set_fontsize(7)
    fig.suptitle('Non-rigid recovery + observability: single-view limit and '
                 'the stereo upgrade', fontsize=12)
    fig.savefig(OUT / 'fig6_dent.png', dpi=200, bbox_inches='tight')
    print('saved ->', OUT / 'fig6_dent.png')


if __name__ == '__main__':
    main()
