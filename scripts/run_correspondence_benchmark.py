# -*- coding: utf-8 -*-
"""
对应关系构建方法基准 (回答: 两个坐标系之间怎样构建有效对应?)
=============================================================

在同一真实数据 (BOP scene 000001) 上对比 5 类对应关系构建方法,
从对应层指标 (数量/内点率) 到位姿层指标 (旋转误差/耗时) 全链路量化:

  M1 人工/标记点 (oracle 上界): GT 对应 + 噪声 → Kabsch / PnP
  M2 SIFT 纹理特征 (2D-2D→深度提升→2D-3D): ROI-SIFT + PnP
  M3 FPFH 几何特征 (3D-3D): 全局 RANSAC 配准
  M4 渲染-比较 (隐式对应): 模板库相似度 → 假设 → ICP
  M5 深度提升 (2D→3D 直接对应): 反投影 → trimmed ICP

用法:
    conda activate py3d
    python scripts/run_correspondence_benchmark.py
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import sys
import json
import time
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.geometry import (make_T, transform, backproject_depth, project_pose)
from src.pipeline import load_mesh_points, RegConfig
from src.pose_estimation import (kabsch_umeyama, pnp_ransac,
                                 fpfh_ransac_register, icp)
from src.metrics import pose_errors
from src.appearance_engine import SIFTReferenceLibrary, register_sift
from src.render_compare_engine import PoseEngine

BOP = Path(r'D:\bop\lm')
FRAMES = [15, 157, 323, 476, 607, 741, 914, 1028, 1137, 1231]


def main():
    device = 'cuda'
    sdir = BOP / 'test' / '000001'
    cams = json.load(open(sdir / 'scene_camera.json'))
    gts = json.load(open(sdir / 'scene_gt.json'))
    pts, _ = load_mesh_points(str(BOP / 'models' / 'obj_000001.ply'),
                              n_points=20000)
    diameter = float(np.linalg.norm(pts.max(0) - pts.min(0)))
    mu = pts.mean(0)
    rng = np.random.default_rng(0)

    # ---------- M1: oracle 对应 (上界) ----------
    print('\n[M1] 人工标记点 (oracle): GT对应+噪声 → Kabsch / PnP')
    t0 = time.time()
    errs_k, errs_p = [], []
    for fi in FRAMES:
        k = str(fi)
        K = np.array(cams[k]['cam_K']).reshape(3, 3)
        g = gts[k][0]
        T_gt = make_T(np.array(g['cam_R_m2c']).reshape(3, 3),
                      np.array(g['cam_t_m2c']).reshape(3))
        # 模拟人工点穴: 模型上 12 个标记点, 3D 观测含 1mm 噪声
        sel = rng.choice(len(pts), 12, replace=False)
        Pm = pts[sel]
        Pc = transform(T_gt, Pm) + rng.normal(0, 1.0, Pm.shape)
        R, t, _ = kabsch_umeyama(Pm, Pc)
        rot_e, _ = pose_errors(make_T(R, t), T_gt)
        errs_k.append(rot_e)
        # 2D-3D: 标记点投影像素含 0.5px 噪声
        uv, _z = project_pose(K, T_gt, Pm)
        uv += rng.normal(0, 0.5, uv.shape)
        T_p, _ = pnp_ransac(Pm, uv, K)
        rot_p, _ = pose_errors(T_p, T_gt)
        errs_p.append(rot_p)
    print(f'    12对标记: Kabsch rot_err={np.mean(errs_k):.3f}° | '
          f'PnP rot_err={np.mean(errs_p):.3f}° | '
          f'{(time.time() - t0) * 1000 / len(FRAMES):.1f}ms/帧')

    # ---------- 准备: SIFT 参考库 + RC 引擎 ----------
    lib = SIFTReferenceLibrary()
    eval_set = {str(f) for f in FRAMES}
    ref_keys = [k for k in sorted(cams.keys(), key=int)
                if k not in eval_set and gts[k][0]['obj_id'] == 1]
    for rk in np.array(ref_keys)[np.linspace(0, len(ref_keys) - 1,
                                             40).astype(int)]:
        g = gts[rk][0]
        rgb_r = cv2.cvtColor(cv2.imread(str(sdir / 'rgb' / f'{int(rk):06d}.png')),
                             cv2.COLOR_BGR2RGB)
        depth_r = cv2.imread(str(sdir / 'depth' / f'{int(rk):06d}.png'),
                             cv2.IMREAD_UNCHANGED).astype(np.float64)
        K_r = np.array(cams[rk]['cam_K']).reshape(3, 3)
        mk_r = sdir / 'mask' / f'{int(rk):06d}_000000.png'
        mask_r = cv2.imread(str(mk_r), cv2.IMREAD_GRAYSCALE)
        T_r = make_T(np.array(g['cam_R_m2c']).reshape(3, 3),
                     np.array(g['cam_t_m2c']).reshape(3))
        lib.add(rgb_r, depth_r, K_r, T_r, mask=mask_r, frame_id=int(rk))
    engine = PoseEngine(pts, diameter, backend='rc', device=device)

    rows = []
    print('\n逐帧对比 (scene 000001, obj01 ape):')
    print(f'{"帧":>6} | {"M2 SIFT":>22} | {"M3 FPFH":>14} | '
          f'{"M4 渲染比较":>14} | {"M5 深度+ICP":>14}')
    for fi in FRAMES:
        k = str(fi)
        K = np.array(cams[k]['cam_K']).reshape(3, 3)
        depth = cv2.imread(str(sdir / 'depth' / f'{fi:06d}.png'),
                           cv2.IMREAD_UNCHANGED).astype(np.float64)
        rgb = cv2.cvtColor(cv2.imread(str(sdir / 'rgb' / f'{fi:06d}.png')),
                           cv2.COLOR_BGR2RGB)
        mv = sdir / 'mask_visib' / f'{fi:06d}_000000.png'
        mask = cv2.imread(str(mv), cv2.IMREAD_GRAYSCALE)
        g = gts[k][0]
        T_gt = make_T(np.array(g['cam_R_m2c']).reshape(3, 3),
                      np.array(g['cam_t_m2c']).reshape(3))
        row = {'frame': fi}

        # M2 SIFT
        t0 = time.time()
        T_s, st_s = register_sift(lib, rgb, K, mask_q=mask)
        dt_s = time.time() - t0
        if T_s is not None:
            rot_s, _ = pose_errors(T_s, T_gt)
            row['sift'] = (f'{st_s["n_good"]}对/{st_s["n_inliers"]}内/'
                           f'{rot_s:.0f}°/{dt_s * 1000:.0f}ms')
            row['sift_rot'] = rot_s
        else:
            row['sift'] = f'失败({st_s.get("error", "")[:12]})'
            row['sift_rot'] = np.nan

        # M3 FPFH
        scene_pts, _, _ = backproject_depth(depth, K, mask=mask)
        sel = rng.choice(len(scene_pts), min(20000, len(scene_pts)),
                         replace=False)
        t0 = time.time()
        T_f, fit, fr = fpfh_ransac_register(pts[::7], scene_pts[sel])
        T_f2, _ = icp(pts[::7], scene_pts[sel], T_init=T_f, max_iter=50,
                      trim_ratio=0.7, device=device)
        dt_f = time.time() - t0
        rot_f, _ = pose_errors(T_f2, T_gt)
        row['fpfh'] = f'{fit:.2f}/{rot_f:.0f}°/{dt_f * 1000:.0f}ms'
        row['fpfh_rot'] = rot_f

        # M4 渲染比较
        t0 = time.time()
        T_rc, st_rc = engine.register(depth, K, mask=mask, n_hypo=3)
        dt_rc = time.time() - t0
        rot_rc, _ = pose_errors(T_rc, T_gt)
        row['rc'] = f'{rot_rc:.0f}°/{dt_rc * 1000:.0f}ms'
        row['rc_rot'] = rot_rc

        # M5 深度提升+ICP (多起点)
        from src.pose_estimation import multistart_icp_register
        t0 = time.time()
        T_m, st_m = multistart_icp_register(pts[::7], scene_pts[sel],
                                            device=device)
        dt_m = time.time() - t0
        rot_m, _ = pose_errors(T_m, T_gt)
        row['ms'] = f'{rot_m:.0f}°/{dt_m * 1000:.0f}ms'
        row['ms_rot'] = rot_m

        rows.append(row)
        print(f'{fi:>6} | {row["sift"]:>22} | {row["fpfh"]:>14} | '
              f'{row["rc"]:>14} | {row["ms"]:>14}')

    # ---------- 汇总 ----------
    def stat(key):
        v = np.array([r[key] for r in rows], dtype=float)
        return np.nanmean(v), float(np.nanmean(v <= 10.0))
    print('\n' + '=' * 76)
    print('汇总 (10帧, rot≤10° 视为成功):')
    for name, key in [('M2 SIFT纹理对应', 'sift_rot'),
                      ('M3 FPFH几何特征', 'fpfh_rot'),
                      ('M4 渲染-比较模板', 'rc_rot'),
                      ('M5 深度提升+ICP', 'ms_rot')]:
        m, s = stat(key)
        print(f'  {name}: 平均rot_err={m:.1f}°  成功率={s:.0%}')
    print('=' * 76)


if __name__ == '__main__':
    main()
