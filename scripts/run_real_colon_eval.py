# -*- coding: utf-8 -*-
"""
real_colon 真实医疗 GT 评测 (免人工标注的医疗级证据)
====================================================

数据: E:\\Diff_Rending_Re_3D\\dataset\\real_colon\\scene_* (24场景×8帧)
  GT: poses (8,4,4) + u_seq (8,768) 真实形变位移 + nodes/surface_tris 网格

协议 (每场景):
  参考帧 t0: GT pose 已知(注册起点); 目标帧 t: 渲染形变后深度(+噪声)
  → 本管线 ICP 注册位姿 → RTW 估计形变
  指标: 位姿误差 vs GT poses; 形变恢复 |u_est - u_gt| (逐节点, 真实位移GT);
        深度残差; 耗时

用法:
    conda activate py3d
    python scripts/run_real_colon_eval.py --scenes 8
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import sys
import time
import argparse
from pathlib import Path

import numpy as np
import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.geometry import make_T, transform, backproject_depth, log_so3, \
    inverse_se3
from src.pipeline import render_model, refine_icp
from src.nonrigid_rtw import rtw_optimize_projective
from src.pose_estimation import nearest_neighbors, multistart_icp_register
from src.metrics import pose_errors

RC = Path(r'E:\Diff_Rending_Re_3D\dataset\real_colon')
OUT = Path('outputs/real_colon')
H, W = 128, 128
K = np.array([[160., 0, W / 2], [0, 160., H / 2], [0, 0, 1.0]])


def subdivide_surface(nodes, tris, n_per=6):
    """表面三角细分采样 → 稠密表面点 (含所属三角形, 便于形变映射)"""
    pts = []
    tri_of = []
    for ti, (a, b, c) in enumerate(tris):
        for i in range(n_per + 1):
            for j in range(n_per + 1 - i):
                u = i / n_per
                v = j / n_per
                w = 1 - u - v
                pts.append(u * nodes[a] + v * nodes[b] + w * nodes[c])
                tri_of.append([u, v, w, a, b, c])
    return np.array(pts), tri_of


def deform_points(nodes_base, tris_meta, u_nodes):
    """按节点位移对表面采样点施加形变 (u_nodes: (N,3) 节点位移)"""
    out = np.zeros((len(tris_meta), 3))
    for i, (u, v, w, a, b, c) in enumerate(tris_meta):
        p = u * nodes_base[a] + v * nodes_base[b] + w * nodes_base[c]
        p = p + u * u_nodes[a] + v * u_nodes[b] + w * u_nodes[c]
        out[i] = p
    return out


def eval_scene(sdir, device, frames=(2, 5, 7), seed=0):
    gt = torch.load(sdir / 'gt.pt', map_location='cpu', weights_only=False)
    nodes = gt['scene_spec']['nodes'].numpy().astype(np.float64)
    tris = gt['scene_spec']['surface_tris'].numpy().astype(np.int64)
    poses = gt['poses'].numpy().astype(np.float64)
    u_seq = gt['u_seq'].numpy().astype(np.float64).reshape(-1, 3)
    T_seq = u_seq.shape[0] // nodes.shape[0]
    u_seq = u_seq.reshape(T_seq, nodes.shape[0], 3)
    extent = float(np.linalg.norm(nodes.max(0) - nodes.min(0)))
    surf_pts, tris_meta = subdivide_surface(nodes, tris, n_per=6)
    base_surf = deform_points(nodes, tris_meta, np.zeros_like(nodes))
    rng = np.random.default_rng(seed)
    colors = np.tile(np.array([0.85, 0.65, 0.60]), (len(surf_pts), 1))

    rows = []
    for t in frames:
        if t >= T_seq:
            continue
        # GT poses 为 camera-to-world → 观测渲染与评测均用其逆 (model→camera)
        T_cam = inverse_se3(poses[t])
        u_gt = u_seq[t]
        surf_def = deform_points(nodes, tris_meta, u_gt)
        out = render_model(surf_def, colors, T_cam, K, H, W, radius=1,
                           device=device)
        depth = out['depth'].cpu().numpy()
        fg = depth > 0
        if fg.sum() < 200:
            continue
        depth[fg] += rng.normal(0, extent * 0.01, int(fg.sum()))

        # ---- 刚性注册 (逐帧独立: 多起点ICP, 不依赖参考帧位姿) ----
        scene_pts, _, _ = backproject_depth(depth, K, mask=fg)
        t0 = time.time()
        T_est, st = multistart_icp_register(base_surf, scene_pts,
                                            device=device,
                                            coarse_iter=25, fine_iter=50,
                                            trim_coarse=0.5, trim_fine=0.7)
        # ---- RTW 形变估计 (Rodrigues修复+点对面法线项+尺度门限) ----
        rtw = rtw_optimize_projective(base_surf, depth, K, T_est,
                                      grid=(4, 4, 4), iters=300, lr=3e-3,
                                      lambda_mag=0.5, lambda_sm=0.2,
                                      lambda_p2p=1.0,
                                      inlier_hi=0.10 * extent,
                                      inlier_lo=0.03 * extent,
                                      device=device, seed=seed)
        dt = time.time() - t0

        rot_r, tr_r = pose_errors(T_est, T_cam)
        rot_w, tr_w = pose_errors(rtw['T'], T_cam)
        # 发散哨兵: RTW 旋转误差>30° 判发散 (观测不适定帧), 不计入统计
        diverged = rot_w > 30.0
        # 形变恢复 (端到端相机系口径): 估计位姿变换后 vs GT形变表面
        # 主指标: NN( T_rtw·V_def, T_cam·surf_def ) —— AR 对齐误差 vs GT
        P_est_cam = transform(rtw['T'], rtw['V_def'])
        P_gt_cam = transform(T_cam, surf_def)
        _, d_rec = nearest_neighbors(P_est_cam, P_gt_cam, device=device)
        # 对照: NN( T_rigid·base_surf, T_cam·surf_def )
        P_base_cam = transform(T_est, base_surf)
        _, d_base = nearest_neighbors(P_base_cam, P_gt_cam, device=device)
        rows.append(dict(scene=sdir.name, t=t, rot_r=rot_r, tr_r=tr_r,
                         rot_w=rot_w if not diverged else np.nan,
                         tr_w=tr_w if not diverged else np.nan,
                         rec_rtw=float(d_rec.mean() / extent)
                         if not diverged else np.nan,
                         rec_base=float(d_base.mean() / extent),
                         res=float(rtw['res_rms']), diverged=diverged,
                         ms=dt * 1000))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', type=int, default=8)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('=' * 72)
    print('real_colon 真实医疗 GT 评测 (位姿+形变, 免人工标注)')
    print('=' * 72)
    scenes = sorted([p for p in RC.iterdir() if p.is_dir()])[:args.scenes]
    all_rows = []
    for s in scenes:
        try:
            rows = eval_scene(s, device)
        except Exception as e:
            print(f'  [{s.name}] 失败: {e}')
            continue
        all_rows += rows
        for r in rows:
            print(f'  [{s.name} t{r["t"]}] 位姿: 刚性{r["rot_r"]:.1f}°→'
                  f'RTW{r["rot_w"]:.1f}° | 形变恢复: '
                  f'{r["rec_base"] * 100:.1f}%→{r["rec_rtw"] * 100:.1f}% '
                  f'(跨度的%) | 残差={r["res"]:.4f} | {r["ms"]:.0f}ms')

    if not all_rows:
        print('无有效结果')
        return
    n_div = sum(1 for r in all_rows if r['diverged'])
    ok_rows = [r for r in all_rows if not r['diverged']]
    print('\n' + '=' * 72)
    print(f'汇总: {len(all_rows)} 帧 × {len(set(r["scene"] for r in all_rows))} 场景'
          f' (RTW发散 {n_div} 帧已排除)')
    rot_r = np.array([r['rot_r'] for r in all_rows])
    rot_w = np.array([r['rot_w'] for r in ok_rows])
    tr_w = np.array([r['tr_w'] for r in ok_rows])
    rec_b = np.array([r['rec_base'] for r in all_rows])
    rec_w = np.array([r['rec_rtw'] for r in ok_rows])
    print(f'  位姿旋转误差: 刚性 mean={rot_r.mean():.2f}°(中位{np.median(rot_r):.2f}) '
          f'→ RTW(收敛帧) mean={rot_w.mean():.2f}°(中位{np.median(rot_w):.2f})')
    print(f'  位姿平移误差(RTW): mean={tr_w.mean():.4f} 中位={np.median(tr_w):.4f}')
    print(f'  形变恢复(逐点最近邻, 模型跨度%): '
          f'未建模={rec_b.mean() * 100:.1f}% → RTW={rec_w.mean() * 100:.1f}% '
          f'(↓{100 * (1 - rec_w.mean() / max(rec_b.mean(), 1e-9)):.0f}%)')
    res_ok = np.array([r['res'] for r in ok_rows])
    res_ok = res_ok[np.isfinite(res_ok)]
    if len(res_ok):
        print(f'  残差RMS: {res_ok.mean():.4f} (n={len(res_ok)})')
    print(f'  耗时: {np.mean([r["ms"] for r in all_rows]):.0f}ms/帧')
    import csv
    with open(OUT / 'real_colon_metrics.csv', 'w', newline='',
              encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f'  CSV: {OUT / "real_colon_metrics.csv"}')
    print('=' * 72)


if __name__ == '__main__':
    main()
