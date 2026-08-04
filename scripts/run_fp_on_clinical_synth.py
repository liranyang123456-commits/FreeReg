# -*- coding: utf-8 -*-
"""
SOTA在Ours数据集: FoundationPose 官方 on 临床合成飞行视频 (TRE vs GT)
=======================================================================

与 run_ct_video_registration.py 完全同协议 (同轨迹/同渲染/同靶点/同初始化):
  FP register(K, rgb, depth, mask) vs 本管线 (Kalman预测→ICP)
指标: TRE-NN (100靶点), 位姿误差, 耗时

用法:
    conda activate py3d
    python scripts/run_fp_on_clinical_synth.py
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import sys
import time
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ---- FP 官方路径与 DLL ----
FP_DIR = r'E:\SOTA_Methods\FoundationPose'
os.add_dll_directory(r'C:\Users\lry\.conda\envs\py3d\Lib\site-packages\torch\lib')
os.add_dll_directory(
    r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin')
sys.path.insert(0, FP_DIR)
sys.path.insert(0, FP_DIR + r'\mycpp\build')

from src.geometry import make_T, exp_se3, transform, backproject_depth
from src.pipeline import render_model, refine_icp
from src.pose_estimation import nearest_neighbors
from src.metrics import pose_errors
from src.se3_kalman_filter import SE3KalmanFilter, SE3KalmanConfig
from src.ct_loader import make_flythrough_poses, airway_to_model, CTVolume
import torch

OUT = Path('outputs/ct/fp_synth')
N_FRAMES = 10
H, W = 240, 320
K = np.array([[200., 0, W / 2], [0, 200., H / 2], [0, 0, 1.0]])


class M: pass


def kalman_clinical():
    return SE3KalmanConfig(
        process_noise_pos=2.0, process_noise_rot=2e-5,
        process_noise_vel_pos=0.5, process_noise_vel_rot=2e-4,
        meas_noise_pos=2.0, meas_noise_rot=0.1,
        innovation_threshold=10.0, velocity_damping=0.98)


def build_fp_est(mesh, device):
    import nvdiffrast.torch as dr
    from estimater import FoundationPose
    glctx = dr.RasterizeCudaContext(device=device)
    est = FoundationPose(model_pts=mesh.vertices.copy(),
                         model_normals=mesh.vertex_normals.copy(),
                         symmetry_tfs=None, mesh=mesh, scorer=None,
                         refiner=None, glctx=glctx,
                         debug_dir=str(OUT / 'fp_debug'), debug=0)
    est.to_device(device)
    est.glctx = dr.RasterizeCudaContext(device=device)
    return est


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('=' * 66)
    print('FoundationPose on 临床合成飞行视频 (同协议 vs 本管线)')
    print('=' * 66)

    # ---------- 模型 (带面片, FP 需要 mesh) ----------
    z = np.load('outputs/ct/airway_cache.npz')
    pts, axis_pts, axis_dir = z['points'], z['axis_pts'], z['axis_dir']
    zm = np.load('outputs/ct/airway_mask.npz')
    vol = CTVolume(hu=zm['mask'].astype(np.int16), spacing=zm['spacing'],
                   origin=zm['origin'], orientation=np.eye(3))
    am = airway_to_model(vol, zm['mask'], max_pts=50000)
    import trimesh
    mesh = trimesh.Trimesh(vertices=am.points, faces=am.triangles,
                           process=False)
    print(f'气道模型: {len(mesh.vertices)}顶点 {len(mesh.faces)}面')

    m = M()
    m.points, m.axis_pts, m.axis_dir = pts, axis_pts, axis_dir
    poses_gt = make_flythrough_poses(m, n_frames=N_FRAMES, seed=0)
    rng = np.random.default_rng(1)
    colors = np.tile(np.array([0.85, 0.65, 0.60]), (len(pts), 1))
    import open3d as o3d
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    lm = np.asarray(pcd.farthest_point_down_sample(100).points)
    model_icp = pts[::max(1, len(pts) // 4000)]

    # ---------- FP ----------
    print('初始化 FoundationPose ...')
    est = build_fp_est(mesh, device)

    kf = SE3KalmanFilter(kalman_clinical())
    rows = []
    for i, T_gt in enumerate(poses_gt):
        out = render_model(pts, colors, T_gt, K, H, W, radius=1,
                           device=device)
        depth = out['depth'].cpu().numpy()
        fg = depth > 0
        depth[fg] += rng.normal(0, 0.8, int(fg.sum()))
        rgb = (out['feat'].cpu().numpy() * 255).astype(np.uint8)
        mask = (fg).astype(np.uint8) * 255

        # ---- FP register ----
        t0 = time.time()
        pose_fp = est.register(K, rgb, depth, mask, ob_id=1,
                               glctx=est.glctx, iteration=5)
        dt_fp = time.time() - t0
        T_fp = np.array(pose_fp, dtype=float).reshape(4, 4)
        T_fp[:3, 3] *= 1000.0    # FP 米制 → 毫米

        # ---- 本管线 (同初始化策略: 帧0扰动初始化, 后续预测跟踪) ----
        if i == 0:
            T_init = T_gt @ exp_se3(rng.normal(0, [5, 5, 5, 0.05, 0.05, 0.05]))
            n_it, trim = 50, 0.7
        else:
            T_init = kf.T if kf.initialized else rows[-1]['T_ours']
            n_it, trim = 25, 0.8
        scene, _, _ = backproject_depth(depth, K, mask=fg)
        t0 = time.time()
        if len(scene) < 800 and i > 0:
            T_ours = T_init
        else:
            T_ours, _ = refine_icp(model_icp, scene, T_init,
                                   max_iter=n_it, trim_ratio=trim,
                                   device=device)
        T_sm = kf.update(T_ours, dt=1.0)
        dt_ours = time.time() - t0

        # ---- TRE ----
        Pfp, Po, Pg = transform(T_fp, lm), transform(T_sm, lm), \
            transform(T_gt, lm)
        _, d_fp = nearest_neighbors(Pfp, Pg, device=device)
        _, d_ours = nearest_neighbors(Po, Pg, device=device)
        rot_fp, _ = pose_errors(T_fp, T_gt)
        rot_ours, _ = pose_errors(T_sm, T_gt)
        rows.append(dict(f=i, tre_fp=float(d_fp.mean()),
                         tre_ours=float(d_ours.mean()),
                         rot_fp=rot_fp, rot_ours=rot_ours,
                         ms_fp=dt_fp * 1000, ms_ours=dt_ours * 1000,
                         T_ours=T_sm))
        print(f'  f{i:02d}: FP TRE={d_fp.mean():6.2f}mm ({dt_fp:.1f}s) | '
              f'ours TRE={d_ours.mean():6.2f}mm ({dt_ours * 1000:.0f}ms)')

    # ---------- 汇总 ----------
    tre_fp = np.array([r['tre_fp'] for r in rows])
    tre_ours = np.array([r['tre_ours'] for r in rows])
    print('\n' + '=' * 66)
    print('汇总 (同协议同帧):')
    print(f'  FoundationPose: TRE-NN mean={tre_fp.mean():.2f}mm '
          f'中位={np.median(tre_fp):.2f}mm  '
          f'<10mm {float((tre_fp < 10).mean()):.0%}  '
          f'rot={np.mean([r["rot_fp"] for r in rows]):.1f}°  '
          f'耗时={np.mean([r["ms_fp"] for r in rows]) / 1000:.1f}s/帧')
    print(f'  本管线:         TRE-NN mean={tre_ours.mean():.2f}mm '
          f'中位={np.median(tre_ours):.2f}mm  '
          f'<10mm {float((tre_ours < 10).mean()):.0%}  '
          f'rot={np.mean([r["rot_ours"] for r in rows]):.1f}°  '
          f'耗时={np.mean([r["ms_ours"] for r in rows]):.0f}ms/帧')
    import csv
    with open(OUT / 'fp_vs_ours.csv', 'w', newline='',
              encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=[k for k in rows[0].keys()
                                          if k != 'T_ours'])
        w.writeheader()
        for r in rows:
            w.writerow({k: v for k, v in r.items() if k != 'T_ours'})
    print(f'  CSV: {OUT / "fp_vs_ours.csv"}')
    print('=' * 66)


if __name__ == '__main__':
    main()
