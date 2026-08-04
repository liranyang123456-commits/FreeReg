# -*- coding: utf-8 -*-
"""
E3: 库基线对比实测 (open3d 官方 ICP vs 本管线 trimmed ICP)
=============================================================

同数据同初始化对比:
  baseline: open3d pipelines.registration_icp (point-to-point/point-to-plane)
  本管线:   trimmed ICP (截尾抗遮挡) + 多起点/FPFH 初始化

用法:
    conda activate py3d
    python scripts/run_open3d_baseline.py
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import sys
import json
import time
from pathlib import Path

import numpy as np
import cv2
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.geometry import backproject_depth, make_T
from src.pipeline import load_mesh_points, RegConfig, register_rgbd
from src.pose_estimation import icp
from src.metrics import add_s_error, pose_errors

BOP = Path(r'D:\bop\lm')
FRAMES = [15, 157, 323, 476, 607, 741, 914, 1028, 1137, 1231]


def o3d_icp_baseline(model_pts, scene_pts, T_init, method='p2p',
                     max_iter=50, thr=None):
    """open3d 官方 ICP (质心+同一初始化)"""
    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(model_pts))
    dst = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scene_pts))
    dst.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=max(thr, 5.0) * 2, max_nn=30))
    est = (o3d.pipelines.registration.TransformationEstimationPointToPlane()
           if method == 'p2plane' else
           o3d.pipelines.registration.TransformationEstimationPointToPoint())
    crit = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=max_iter)
    res = o3d.pipelines.registration.registration_icp(
        src, dst, thr, T_init, est, crit)
    return np.asarray(res.transformation), res.inlier_rmse


def main():
    device = 'cuda'
    sdir = BOP / 'test' / '000001'
    cams = json.load(open(sdir / 'scene_camera.json'))
    gts = json.load(open(sdir / 'scene_gt.json'))
    pts, _ = load_mesh_points(str(BOP / 'models' / 'obj_000001.ply'),
                              n_points=20000)
    model_icp = pts[::7]
    eval_pts = pts[::10]
    rng = np.random.default_rng(0)

    print('=' * 78)
    print('E3: open3d ICP (p2p/p2plane) vs 本管线 trimmed ICP (同初始化)')
    print('=' * 78)
    print(f'{"帧":>6} | {"o3d-p2p":>14} | {"o3d-p2plane":>14} | '
          f'{"ours(trim)":>14} | {"ours(完整管线)":>16}')
    agg = {'p2p': [], 'p2pl': [], 'ours': [], 'full': []}
    for fi in FRAMES:
        k = str(fi)
        K = np.array(cams[k]['cam_K']).reshape(3, 3)
        depth = cv2.imread(str(sdir / 'depth' / f'{fi:06d}.png'),
                           cv2.IMREAD_UNCHANGED).astype(np.float64)
        mv = sdir / 'mask_visib' / f'{fi:06d}_000000.png'
        mask = cv2.imread(str(mv), cv2.IMREAD_GRAYSCALE)
        g = gts[k][0]
        T_gt = make_T(np.array(g['cam_R_m2c']).reshape(3, 3),
                      np.array(g['cam_t_m2c']).reshape(3))
        scene, _, _ = backproject_depth(depth, K, mask=mask)
        sel = rng.choice(len(scene), min(15000, len(scene)), replace=False)
        scene_s = scene[sel]
        # 同一弱初始化: 质心对齐 (检验精化能力)
        T0 = make_T(np.eye(3), scene_s.mean(0) - model_icp.mean(0))
        extent = float(np.linalg.norm(pts.max(0) - pts.min(0)))

        row = {}
        t0 = time.time()
        T_a, _ = o3d_icp_baseline(model_icp, scene_s, T0, 'p2p',
                                  thr=0.2 * extent)
        row['p2p'] = add_s_error(eval_pts, T_a, T_gt)
        t_a = time.time() - t0
        t0 = time.time()
        try:
            T_b, _ = o3d_icp_baseline(model_icp, scene_s, T0, 'p2plane',
                                      thr=0.2 * extent)
            row['p2pl'] = add_s_error(eval_pts, T_b, T_gt)
        except Exception:
            row['p2pl'] = float('nan')
        t_b = time.time() - t0
        t0 = time.time()
        T_c, _ = icp(model_icp, scene_s, T_init=T0, max_iter=60,
                     trim_ratio=0.7, device=device)
        row['ours'] = add_s_error(eval_pts, T_c, T_gt)
        t_c = time.time() - t0
        t0 = time.time()
        res = register_rgbd(depth, K, pts, mask=mask,
                            cfg=RegConfig(device=device))
        row['full'] = add_s_error(eval_pts, res['T'], T_gt) \
            if res['T'] is not None else float('nan')
        t_d = time.time() - t0
        for kk, vv in [('p2p', row['p2p']), ('p2pl', row['p2pl']),
                       ('ours', row['ours']), ('full', row['full'])]:
            agg[kk].append(vv)
        print(f'{fi:>6} | {row["p2p"]:8.2f}mm    | {row["p2pl"]:8.2f}mm    |'
          f' {row["ours"]:8.2f}mm    | {row["full"]:8.2f}mm')

    def succ(a, th=10.0):
        a = np.array(a, dtype=float)
        return f'mean={np.nanmean(a):6.2f}mm  成功(<{th:g}mm)={np.nanmean(a < th):.0%}'
    print('-' * 78)
    print(f'o3d-p2point : {succ(agg["p2p"])}')
    print(f'o3d-p2plane : {succ(agg["p2pl"])}')
    print(f'ours-trimmed: {succ(agg["ours"])}')
    print(f'ours-完整管线: {succ(agg["full"])}')
    print('=' * 78)


if __name__ == '__main__':
    main()
