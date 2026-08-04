# -*- coding: utf-8 -*-
"""
E1: CT-视频注册完整评测 (省医患者 001, 合成支气管镜飞行视频)
==============================================================

链路:
  省医 CT → 气道分割/建模 → 合成飞行视频 (渲染深度+噪声, GT轨迹)
  → 帧0 临床式初始化 (GT+扰动, 模拟手动放置)
  → 逐帧: Kalman预测 → trimmed ICP 精化 (跟踪模式)
  → 性能指标: TRE(靶点配准误差, 含轴向/横向分解) / 位姿误差 /
              深度残差 / 抖动偏差 / 耗时

管状结构可观测性说明: 气道为近管状, 沿轴位置弱可观 → 指标分解报告
(横向误差为主指标, 轴向漂移为辅助指标)。

用法:
    conda activate py3d
    python scripts/run_ct_video_registration.py
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import sys
import time
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.geometry import make_T, exp_se3, log_so3, transform, \
    backproject_depth, project_pose
from src.ct_loader import load_ct_airway, make_flythrough_poses
from src.pipeline import RegConfig, render_model, refine_icp
from src.pose_estimation import nearest_neighbors
from src.metrics import pose_errors, pixel_jitter_deviation
from src.se3_kalman_filter import SE3KalmanFilter, SE3KalmanConfig
import torch

P1 = r'F:\省医-数据\省医-数据\ION患者整理\001王继深'
CACHE = Path('outputs/ct/airway_cache.npz')
OUT = Path('outputs/ct')
N_FRAMES = 40
H, W = 240, 320
K = np.array([[200., 0, W / 2], [0, 200., H / 2], [0, 0, 1.0]])


def load_model_cached():
    if CACHE.exists():
        z = np.load(CACHE)
        class M: pass
        m = M()
        m.points, m.axis_pts, m.axis_dir = z['points'], z['axis_pts'], \
            z['axis_dir']
        return m
    model = load_ct_airway(P1, max_slices=200, max_pts=50000)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, points=model.points, axis_pts=model.axis_pts,
             axis_dir=model.axis_dir)
    return model


def kalman_clinical():
    # 管状结构: ICP 绕轴旋转估计噪声大 (~5-10°) → meas_noise_rot 必须如实
    # 放大 (0.1rad), 否则滤波器把正常测量抖动误判为跟踪丢失 (实测频发)
    return SE3KalmanConfig(
        process_noise_pos=2.0, process_noise_rot=2e-5,
        process_noise_vel_pos=0.5, process_noise_vel_rot=2e-4,
        meas_noise_pos=2.0, meas_noise_rot=0.1,
        innovation_threshold=10.0, velocity_damping=0.98)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('=' * 66)
    print('E1: CT-视频注册评测 (省医001, 合成飞行视频40帧)')
    print('=' * 66)

    model = load_model_cached()
    pts = model.points
    print(f'气道: {len(pts)}点, 轴={model.axis_dir.round(2)}, '
          f'范围={np.round(pts.min(0), 0)}~{np.round(pts.max(0), 0)}mm')

    poses_gt = make_flythrough_poses(model, n_frames=N_FRAMES, seed=0)
    # 观测噪声模型: 深度σ=0.8mm (内窥镜级)
    rng = np.random.default_rng(1)
    colors = np.tile(np.array([0.85, 0.65, 0.60]), (len(pts), 1))

    # 靶点: FPS 采样 100 个表面点 (TRE 评估用)
    import open3d as o3d
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    lm = np.asarray(pcd.farthest_point_down_sample(100).points)
    model_icp = pts[::max(1, len(pts) // 4000)]

    kf = SE3KalmanFilter(kalman_clinical())
    cfg = RegConfig(device=device)
    meas_poses, sm_poses = [], []
    rows = []
    t_all = time.time()
    for i, T_gt in enumerate(poses_gt):
        out = render_model(pts, colors, T_gt, K, H, W, radius=1,
                           device=device)
        depth = out['depth'].cpu().numpy()
        fg = depth > 0
        depth[fg] += rng.normal(0, 0.8, int(fg.sum()))

        t0 = time.time()
        if i == 0:
            # 临床式初始化: GT + 5mm/3° 扰动 (模拟手动/粗配准)
            T_init = T_gt @ exp_se3(rng.normal(0, [5, 5, 5, 0.05, 0.05, 0.05]))
            n_iter, trim = 50, 0.7   # 帧0 充分收敛
        else:
            T_init = kf.T if kf.initialized else meas_poses[-1]
            n_iter, trim = 25, 0.8
        scene_pts, _, _ = backproject_depth(depth, K, mask=fg)
        if len(scene_pts) < 800 and i > 0:
            # 深部管腔塌陷: 观测不足 → 滑行 (创新门控的应用层体现)
            T_meas = T_init
            st = {'rmse': -1, 'skipped': True}
        else:
            T_meas, st = refine_icp(model_icp, scene_pts, T_init,
                                    max_iter=n_iter, trim_ratio=trim,
                                    device=device)
        T_sm = kf.update(T_meas, dt=1.0)
        dt = time.time() - t0
        meas_poses.append(T_meas)
        sm_poses.append(T_sm)

        # 指标
        rot_m, tr_m = pose_errors(T_meas, T_gt)
        rot_s, tr_s = pose_errors(T_sm, T_gt)
        Pm = transform(T_meas, lm)
        Pg = transform(T_gt, lm)
        _, d_meas = nearest_neighbors(Pm, Pg, device=device)
        Ps = transform(T_sm, lm)
        _, d_sm = nearest_neighbors(Ps, Pg, device=device)
        # 近场 TRE (临床靶点意义): 距相机原点 60mm 内的靶点
        # (Pg 已在相机系, 相机原点即 [0,0,0])
        near = np.linalg.norm(Pg, axis=1) < 60.0
        tre_near = float(np.linalg.norm(Pm[near] - Pg[near], axis=1).mean()) \
            if near.sum() >= 3 else float('nan')
        # 轴向/横向分解 (沿气道主轴)
        a = model.axis_dir
        err_vec = (Pm - Pg)
        ax_m = np.abs(err_vec @ a).mean()
        lat_m = np.sqrt(np.maximum(
            np.linalg.norm(err_vec, axis=1) ** 2 - (err_vec @ a) ** 2,
            0)).mean()
        rows.append(dict(f=i, rot_m=rot_m, tr_m=tr_m, rot_s=rot_s,
                         tr_s=tr_s, tre_m=float(d_meas.mean()),
                         tre_s=float(d_sm.mean()), tre_near=tre_near,
                         tre_ax=float(ax_m), tre_lat=float(lat_m),
                         ms=dt * 1000))
        if i % 10 == 0 or i == N_FRAMES - 1:
            print(f'  f{i:02d}: TRE={d_meas.mean():6.2f}mm(轴{ax_m:.1f}/'
                  f'横{lat_m:.1f}) rot={rot_m:.2f}° t={dt * 1000:.0f}ms')

    # ---------- 汇总 ----------
    tre_m = np.array([r['tre_m'] for r in rows])
    tre_s = np.array([r['tre_s'] for r in rows])
    rot_m = np.array([r['rot_m'] for r in rows])
    tr_m = np.array([r['tr_m'] for r in rows])
    rot_s = np.array([r['rot_s'] for r in rows])
    tr_s = np.array([r['tr_s'] for r in rows])
    ms = np.array([r['ms'] for r in rows])

    def uv_of(poses):
        out = []
        for T in poses:
            P = transform(T, lm[::10])
            z = P[:, 2]
            out.append(np.stack([K[0, 0] * P[:, 0] / z + K[0, 2],
                                 K[1, 1] * P[:, 1] / z + K[1, 2]], -1))
        return out
    ug, um, us = uv_of(poses_gt), uv_of(meas_poses), uv_of(sm_poses)
    nf = len(um)   # 实际帧数 (飞行轨迹可能少于 N_FRAMES)
    jm = np.mean([pixel_jitter_deviation(
        np.array([um[t][j] for t in range(nf)]),
        np.array([ug[t][j] for t in range(nf)]))
        for j in range(len(um[0]))])
    js = np.mean([pixel_jitter_deviation(
        np.array([us[t][j] for t in range(nf)]),
        np.array([ug[t][j] for t in range(nf)]))
        for j in range(len(us[0]))])

    print('\n' + '=' * 66)
    print('E1 指标汇总 (40帧合成飞行视频):')
    print(f'  TRE (100靶点):   测量={tre_m.mean():.2f}mm  '
          f'Kalman={tre_s.mean():.2f}mm  中位={np.median(tre_m):.2f}mm')
    print(f'  TRE成功率:       <10mm帧 {float((tre_m < 10).mean()):.0%}, '
          f'<15mm帧 {float((tre_m < 15).mean()):.0%}')
    tre_near_arr = np.array([r['tre_near'] for r in rows], dtype=float)
    print(f'  近场TRE(<60mm):  mean={np.nanmean(tre_near_arr):.2f}mm  '
          f'中位={np.nanmedian(tre_near_arr):.2f}mm  '
          f'<5mm帧 {float((tre_near_arr < 5).mean()):.0%}  ← 临床主指标')
    print(f'  TRE分解:         轴向={np.mean([r["tre_ax"] for r in rows]):.2f}mm '
          f'横向={np.mean([r["tre_lat"] for r in rows]):.2f}mm')
    print(f'  旋转误差:        测量={rot_m.mean():.2f}°(中位{np.median(rot_m):.2f})  '
          f'Kalman={rot_s.mean():.2f}°')
    print(f'  平移误差:        测量={tr_m.mean():.2f}mm(中位{np.median(tr_m):.2f})  '
          f'Kalman={tr_s.mean():.2f}mm')
    print(f'  抖动偏差:        测量={jm:.3f}px  Kalman={js:.3f}px  '
          f'↓{100 * (1 - js / max(jm, 1e-9)):.0f}%')
    print(f'  重定位/拒绝:     {kf.reloc_count}/{kf.reject_count}')
    print(f'  注册耗时:        mean={ms.mean():.0f}ms/帧 → {1000 / ms.mean():.0f} FPS')
    print(f'  总耗时:          {time.time() - t_all:.1f}s')

    # ---------- 输出三联帧 ----------
    panels = []
    n_poses = len(poses_gt)
    for i in sorted({n_poses // 6, n_poses // 2, n_poses - 2}):
        out_g = render_model(pts, colors, poses_gt[i], K, H, W, radius=1,
                             device=device)
        T_e = sm_poses[i]
        colors_e = np.tile(np.array([0.3, 0.9, 0.4]), (len(pts), 1))
        out_e = render_model(pts, colors_e, T_e, K, H, W, radius=1,
                             device=device)
        blend = np.clip(out_g['feat'].cpu().numpy()
                        + out_e['feat'].cpu().numpy() * 0.7, 0, 1)
        panels.append(blend)
    tri = (np.concatenate(panels, axis=1) * 255).astype(np.uint8)
    cv2.imwrite(str(OUT / 'ct_reg_overlay.png'),
                cv2.cvtColor(tri, cv2.COLOR_RGB2BGR))
    print(f'  叠加图: {OUT / "ct_reg_overlay.png"} (GT粉|估计绿×叠加)')

    # CSV
    import csv
    with open(OUT / 'ct_reg_metrics.csv', 'w', newline='',
              encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'  逐帧指标: {OUT / "ct_reg_metrics.csv"}')
    print('=' * 66)


if __name__ == '__main__':
    main()
