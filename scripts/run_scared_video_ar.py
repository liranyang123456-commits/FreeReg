# -*- coding: utf-8 -*-
"""
M2: SCARED 真实内窥镜视频 AR 注册与时序评测
============================================

真实数据链路:
  SCARED rgb.mp4 (197帧, 25fps) + frame_data 相机轨迹 (c2w)
  + point_cloud.obj (真实场景几何, 遮挡合成代理)
  → 虚拟模型锚定到组织表面 → 逐帧 AR 融合 (深度感知)
  → 输出 AR 视频 + 三联帧

时序评测:
  GT轨迹(平滑) vs 含噪轨迹(模拟VIO噪声+漂移) vs Kalman平滑
  → 多点像素抖动偏差 (jitter deviation)

单位注意: SCARED 位姿平移为米, 点云/深度为毫米 → 统一转毫米。
假设: point_cloud 位于关键帧(=帧0)相机系 (SCARED 常规约定)。

用法:
    conda activate py3d
    python scripts/run_scared_video_ar.py
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import sys
import json
import tarfile
import time
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.geometry import make_T, exp_se3, inverse_se3, transform
from src.pipeline import (load_mesh_points, render_model, composite_depth_aware,
                          lambert_colors)
from src.metrics import pixel_jitter_deviation
from src.geometry import project_pose
from src.se3_kalman_filter import SE3KalmanFilter, SE3KalmanConfig

import torch

SCARED = Path(r'E:\MIS_Datasets\SCARED\dataset_1\keyframe_1')
OUT = Path('outputs/scared')
N_FRAMES = 197
IMG_TOP_HALF = True   # rgb.mp4 上半为左目 (1280x1024)


def load_scared():
    # 标定
    fs = cv2.FileStorage(str(SCARED / 'endoscope_calibration.yaml'),
                         cv2.FILE_STORAGE_READ)
    K = fs.getNode('M1').mat()
    # 轨迹 (SCARED camera-pose 平移单位即为毫米, 实测帧间 |dt|≈0.3mm;
    # 切勿再乘 1000 —— 单位一致性是坐标系工程的第一纪律)
    poses = []
    with tarfile.open(str(SCARED / 'data' / 'frame_data.tar.gz'), 'r:gz') as tar:
        for name in sorted(tar.getnames()):
            d = json.loads(tar.extractfile(name).read().decode('utf-8'))
            T = np.array(d['camera-pose'], dtype=np.float64)
            poses.append(T)              # c2w (mm)
    return K, poses


def load_scene_points(max_pts=200000):
    """SCARED point_cloud.obj 为纯顶点列表 (v x y z), 手工解析 (mm)"""
    pts = []
    with open(SCARED / 'point_cloud.obj') as f:
        for line in f:
            if line.startswith('v '):
                p = line.split()
                pts.append((float(p[1]), float(p[2]), float(p[3])))
    pts = np.array(pts, dtype=np.float64)
    if len(pts) > max_pts:  # 渲染效率: 均匀降采样
        sel = np.linspace(0, len(pts) - 1, max_pts).astype(int)
        pts = pts[sel]
    return pts   # mm, 关键帧相机系


def frame_eye(frame):
    """从 1280x2048 帧取左目 (上半 1280x1024)"""
    h = frame.shape[0] // 2
    return frame[:h] if IMG_TOP_HALF else frame[h:]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('=' * 66)
    print('M2: SCARED 真实视频 AR 注册 (197帧真实轨迹)')
    print('=' * 66)

    K, poses_c2w = load_scared()
    scene_pts = load_scene_points()
    print(f'标定 fx={K[0, 0]:.1f} 轨迹帧数={len(poses_c2w)} '
          f'场景点={len(scene_pts)}')

    # ---------- 虚拟模型锚定 ----------
    # 模型: BOP duck (非对称), 缩放到 ~18mm, 置于场景表面质心处
    model, normals = load_mesh_points(
        r'D:\bop\lm\models\obj_000008.ply', n_points=20000)
    raw_extent = np.linalg.norm(model.max(0) - model.min(0))
    scale = 18.0 / raw_extent
    model = model * scale
    normals = normals  # 法线不受均匀缩放影响
    # 世界系 = 帧0相机系 (点云所在系) —— 锚定在点云质心上方
    c0 = scene_pts.mean(axis=0)
    # 把模型放到场景表面: 取深度较小(较近)四分位的表面点质心 (过滤NaN/inf)
    finite = np.isfinite(scene_pts).all(axis=1)
    zvals = scene_pts[finite, 2]
    z_quart = np.percentile(zvals, 25)
    surface = scene_pts[finite & (scene_pts[:, 2] < z_quart)]
    anchor = surface.mean(axis=0) if len(surface) > 100 \
        else scene_pts[finite].mean(axis=0)
    model_extent = np.linalg.norm(model.max(0) - model.min(0))
    T_world_obj = make_T(np.eye(3), anchor)
    print(f'锚点(场景表面质心) = {np.round(anchor, 1)}mm, '
          f'模型直径={model_extent:.1f}mm')

    # 位姿链: X_cam(t) = T_cw(t) · X_world,  T_cw = inv(c2w)
    poses_cw = [inverse_se3(T) for T in poses_c2w]

    # ---------- 渲染 AR 视频 (GT 轨迹) ----------
    colors = lambert_colors(normals, np.eye(3), base_rgb=(0.2, 0.9, 0.3))
    scene_colors = np.tile(np.array([0.85, 0.75, 0.65]),
                           (len(scene_pts), 1))
    cap = cv2.VideoCapture(str(SCARED / 'data' / 'rgb.mp4'))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    H = W = None
    writer = None
    t_render = time.time()
    n_done = 0
    for i in range(N_FRAMES):
        ok, fr = cap.read()
        if not ok:
            break
        left = frame_eye(fr)
        left_rgb = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
        if H is None:
            H, W = left_rgb.shape[:2]
            writer = cv2.VideoWriter(
                str(OUT / 'ar_gt.mp4'),
                cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
        T_cw = poses_cw[i]
        # 虚拟模型渲染 (render_model 内部完成 模型系→相机系 变换)
        r_virt = render_model(model, colors, T_cw @ T_world_obj,
                              K, H, W, radius=2, device=device)
        # 场景深度 (遮挡代理): 真实点云渲染 (已在世界系, 直接给位姿)
        r_scene = render_model(scene_pts, scene_colors, T_cw,
                               K, H, W, radius=1, device=device)
        import torch as _th
        comp, occ = composite_depth_aware(
            _th.as_tensor(left_rgb.astype(np.float64) / 255.0,
                          dtype=_th.float32, device=r_virt['feat'].device),
            _th.as_tensor(r_scene['depth'].cpu().numpy(),
                          dtype=_th.float32, device=r_virt['feat'].device),
            r_virt['feat'], r_virt['depth'], alpha=0.85)
        comp = comp.cpu().numpy()
        occ = occ.cpu().numpy()
        out_u8 = cv2.cvtColor((np.clip(comp, 0, 1) * 255).astype(np.uint8),
                              cv2.COLOR_RGB2BGR)
        writer.write(out_u8)
        n_done += 1
        if i in (30, 100, 160):
            tri = np.concatenate([
                left_rgb, (np.clip(comp, 0, 1) * 255).astype(np.uint8)],
                axis=1)
            cv2.imwrite(str(OUT / f'ar_frame_{i:03d}.png'),
                        cv2.cvtColor(tri, cv2.COLOR_RGB2BGR))
    cap.release()
    writer.release()
    print(f'AR视频: {n_done}帧 → {OUT / "ar_gt.mp4"} '
          f'({(time.time() - t_render):.1f}s, '
          f'{(time.time() - t_render) / max(n_done, 1) * 1000:.0f}ms/帧)')

    # ---------- 时序评测: 噪声轨迹 + Kalman ----------
    print('\n时序评测: GT轨迹 vs 含噪轨迹(VIO噪声+随机游走漂移) vs Kalman')
    rng = np.random.default_rng(0)
    noisy = []
    for i, T in enumerate(poses_cw):
        # 有界慢漂移 (模拟 VIO 漂移-回环校正循环, 振幅 2mm/0.6°)
        drift = np.array([2.0 * np.sin(i / 50.0), 1.5 * np.sin(i / 70.0),
                          1.0 * np.sin(i / 60.0), 0.01 * np.sin(i / 45.0),
                          0.008 * np.sin(i / 55.0), 0.006 * np.sin(i / 65.0)])
        eps = rng.normal(0, [0.5, 0.5, 0.5, 0.002, 0.002, 0.002])     # 逐帧噪声
        noisy.append(T @ exp_se3(eps + drift))
    cfg = SE3KalmanConfig(
        process_noise_pos=1.0, process_noise_rot=2e-6,
        process_noise_vel_pos=0.2, process_noise_vel_rot=1e-4,
        meas_noise_pos=1.0, meas_noise_rot=0.005,
        innovation_threshold=10.0, velocity_damping=0.98)
    kf = SE3KalmanFilter(cfg)
    smooth = [kf.update(T, dt=1.0) for T in noisy]

    probe = model[np.linspace(0, len(model) - 1, 9).astype(int)]
    probe_w = transform(T_world_obj, probe)

    def uv_seq(poses):
        out = []
        for T in poses:
            uvs, _ = None, None
            P = transform(T, probe_w)
            z = P[:, 2]
            u = K[0, 0] * P[:, 0] / z + K[0, 2]
            v = K[1, 1] * P[:, 1] / z + K[1, 2]
            out.append(np.stack([u, v], -1))
        return out

    devs_raw, devs_sm = [], []
    for j in range(len(probe_w)):
        uv_gt = np.array([uv_seq(poses_cw)[i][j] for i in range(n_done)])
        uv_raw = np.array([uv_seq(noisy)[i][j] for i in range(n_done)])
        uv_sm = np.array([uv_seq(smooth)[i][j] for i in range(n_done)])
        devs_raw.append(pixel_jitter_deviation(uv_raw, uv_gt))
        devs_sm.append(pixel_jitter_deviation(uv_sm, uv_gt))
    j_raw, j_sm = float(np.mean(devs_raw)), float(np.mean(devs_sm))
    print(f'  抖动偏差(9探针均值): 含噪={j_raw:.3f}px  '
          f'Kalman={j_sm:.3f}px  ↓{100 * (1 - j_sm / j_raw):.1f}%')
    print(f'  重定位={kf.reloc_count} 拒绝={kf.reject_count}')
    print('=' * 66)


if __name__ == '__main__':
    main()
