# -*- coding: utf-8 -*-
"""
全数据集结果图像生成器
========================

为每个实验数据集生成关键对比图 (输出 outputs/gallery/):
  G1 BOP 位姿叠加 (真实图 + 模型投影边界/坐标轴)
  G2 real_colon 形变恢复 (GT形变|刚性|RTW 深度三联)
  G3 凹痕 E2 终版 (obs|GT|刚性|E2深度 + 差分)

用法:
    conda activate py3d
    python scripts/run_results_gallery.py
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import sys
import json
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.geometry import make_T, exp_so3, transform, project_pose, inverse_se3
from src.pipeline import (RegConfig, load_mesh_points, register_rgbd,
                          render_model, lambert_colors)
from src.nonrigid_rtw import rtw_optimize_projective
from src.pose_estimation import nearest_neighbors
import torch

OUT = Path('outputs/gallery')
BOP = Path(r'D:\bop\lm')


def save(name, img):
    cv2.imwrite(str(OUT / name), img)
    print(f'  {OUT / name}')


def g1_bop_overlay(device):
    print('[G1] BOP 位姿叠加')
    sdir = BOP / 'test' / '000002'
    cams = json.load(open(sdir / 'scene_camera.json'))
    gts = json.load(open(sdir / 'scene_gt.json'))
    k = '243'
    K = np.array(cams[k]['cam_K']).reshape(3, 3)
    rgb = cv2.cvtColor(cv2.imread(str(sdir / 'rgb' / '000243.png')),
                       cv2.COLOR_BGR2RGB)
    depth = cv2.imread(str(sdir / 'depth' / '000243.png'),
                       cv2.IMREAD_UNCHANGED).astype(np.float64)
    g = gts[k][0]
    pts, normals = load_mesh_points(
        str(BOP / 'models' / f"obj_{g['obj_id']:06d}.ply"), n_points=15000)
    mv = sdir / 'mask_visib' / '000243_000000.png'
    mask = cv2.imread(str(mv), cv2.IMREAD_GRAYSCALE)
    res = register_rgbd(depth, K, pts, mask=mask,
                        cfg=RegConfig(device=device))
    T_est, T_gt = res['T'], np.eye(4)
    T_gt[:3, :3] = np.array(g['cam_R_m2c']).reshape(3, 3)
    T_gt[:3, 3] = np.array(g['cam_t_m2c']).reshape(3)

    vis = rgb.copy()
    # 模型投影边界 (估计位姿, 绿)
    r = render_model(pts, np.ones((len(pts), 3)), T_est, K,
                     rgb.shape[0], rgb.shape[1], radius=1, device=device)
    m = r['mask'].cpu().numpy()
    bnd = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_GRADIENT,
                           np.ones((3, 3), np.uint8)) > 0
    vis[bnd] = (0, 255, 0)
    # 坐标轴
    mu = pts.mean(0)
    diam = float(np.linalg.norm(pts.max(0) - pts.min(0)))
    axes = np.array([[0, 0, 0], [diam / 2, 0, 0], [0, diam / 2, 0],
                     [0, 0, diam / 2.0]], float)
    uv, _ = project_pose(K, T_est, axes)
    o = tuple(np.round(uv[0]).astype(int))
    for i, c in enumerate([(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
        p = tuple(np.round(uv[i + 1]).astype(int))
        cv2.line(vis, o, p, c, 2, cv2.LINE_AA)
    cv2.putText(vis, 'BOP LM scene2 f243: pose overlay (green=model proj, '
                     'axes=RGB/XYZ)', (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1)
    save('g1_bop_overlay.png', cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


def g2_real_colon(device):
    print('[G2] real_colon 形变恢复')
    s = Path(r'E:\Diff_Rending_Re_3D\dataset\real_colon\scene_0004')
    gt = torch.load(s / 'gt.pt', map_location='cpu', weights_only=False)
    nodes = gt['scene_spec']['nodes'].numpy().astype(np.float64)
    tris = gt['scene_spec']['surface_tris'].numpy().astype(np.int64)
    poses = gt['poses'].numpy()
    u_seq = gt['u_seq'].numpy().astype(np.float64).reshape(-1, 3)
    u_seq = u_seq.reshape(u_seq.shape[0] // nodes.shape[0], nodes.shape[0], 3)
    extent = float(np.linalg.norm(nodes.max(0) - nodes.min(0)))
    H = W = 192
    K = np.array([[200., 0, W / 2], [0, 200., H / 2], [0, 0, 1.0]])

    def subdiv(nodes, tris, n_per=8):
        pts, meta = [], []
        for (a, b, c) in tris:
            for i in range(n_per + 1):
                for j in range(n_per + 1 - i):
                    u, v = i / n_per, j / n_per
                    w = 1 - u - v
                    pts.append(u * nodes[a] + v * nodes[b] + w * nodes[c])
                    meta.append([u, v, w, a, b, c])
        return np.array(pts), meta

    def deform(nodes, meta, u):
        out = np.zeros((len(meta), 3))
        for i, (u_, v, w, a, b, c) in enumerate(meta):
            p = u_ * nodes[a] + v * nodes[b] + w * nodes[c]
            out[i] = p + u_ * u[a] + v * u[b] + w * u[c]
        return out

    surf, meta = subdiv(nodes, tris)
    base = deform(nodes, meta, np.zeros_like(nodes))
    t = 5
    u_gt = u_seq[t]
    T_cam = inverse_se3(poses[t])
    sdef = deform(nodes, meta, u_gt)
    colors = np.tile(np.array([0.85, 0.65, 0.6]), (len(sdef), 1))
    out = render_model(sdef, colors, T_cam, K, H, W, radius=1,
                       device=device)
    depth = out['depth'].cpu().numpy()
    rng = np.random.default_rng(0)
    fg = depth > 0
    depth[fg] += rng.normal(0, extent * 0.01, int(fg.sum()))

    from src.pose_estimation import multistart_icp_register
    from src.geometry import backproject_depth
    scene_pts, _, _ = backproject_depth(depth, K, mask=(depth > 0))
    T_est, _ = multistart_icp_register(base, scene_pts, device=device,
                                       coarse_iter=25, fine_iter=50,
                                       trim_coarse=0.5, trim_fine=0.7)
    rtw = rtw_optimize_projective(base, depth, K, T_est, grid=(4, 4, 4),
                                  iters=300, lr=3e-3, lambda_mag=0.5,
                                  lambda_sm=0.2, lambda_p2p=1.0,
                                  inlier_hi=0.10 * extent,
                                  inlier_lo=0.03 * extent, device=device,
                                  seed=0)

    def rd(m, T):
        return render_model(m, colors, T, K, H, W, radius=1,
                            device=device)['depth'].cpu().numpy()

    def nrm(d):
        m = d > 0
        o = np.zeros((*d.shape, 3), np.uint8)
        if m.any():
            dn = (d - d[m].min()) / max(d[m].max() - d[m].min(), 1e-9)
            g = (dn * 255).astype(np.uint8)
            o[m] = np.stack([g, g, g], -1)[m]
        return o

    def ddiff(a, b, cap=0.15):
        m = (a > 0) & (b > 0)
        o = np.zeros((*a.shape, 3), np.uint8)
        df = np.clip((b - a) / cap, -1, 1)
        o[m, 0] = (np.clip(-df[m], 0, 1) * 255).astype(np.uint8)
        o[m, 2] = (np.clip(df[m], 0, 1) * 255).astype(np.uint8)
        return o

    d_gt, d_rig, d_rtw = rd(sdef, T_cam), rd(base, T_est), rd(rtw['V_def'],
                                                              rtw['T'])
    row1 = np.concatenate([nrm(d_gt), nrm(d_rig), nrm(d_rtw)], axis=1)
    row2 = np.concatenate([ddiff(d_rig, d_gt), ddiff(d_rtw, d_gt),
                           np.zeros((*d_gt.shape, 3), np.uint8)], axis=1)
    comp = np.concatenate([row1, row2], axis=0)
    cv2.putText(comp, 'GT-deform | rigid | RTW', (5, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(comp, 'diff(blue=GT deeper): rig-vs-GT  rtw-vs-GT',
                (5, H + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1)
    save('g2_real_colon_recovery.png', comp)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('=' * 60)
    print('全数据集结果图像生成')
    print('=' * 60)
    g1_bop_overlay(device)
    g2_real_colon(device)
    # G3 复用已有 outputs/rtw_diag/dent_diag.png, 拷入 gallery
    import shutil
    src = Path('outputs/rtw_diag/dent_diag.png')
    if src.exists():
        shutil.copy(src, OUT / 'g3_dent_diag.png')
        print(f'  {OUT / "g3_dent_diag.png"} (复用)')
    print('=' * 60)


if __name__ == '__main__':
    main()
