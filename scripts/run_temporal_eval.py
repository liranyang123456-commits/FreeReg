# -*- coding: utf-8 -*-
"""
视频时序一致性评测 (真实运动轨迹 + 真实 CAD 模型)
==================================================

Part A (受控轨迹): 平滑合成轨迹(真实运动剖面) + σ=5mm/3mrad 测量噪声
    → SE(3) Kalman → 抖动偏差(deviation jitter): 测量 vs 平滑

Part B (端到端虚拟视频): 真实 BOP 模型 + 平滑相机轨道轨迹
    → 光栅化器渲染深度(+噪声) = 虚拟相机视频
    → 跟踪管线: 帧0全局注册 → 逐帧 Kalman预测→ICP精化
    → 测量 vs Kalman平滑: 抖动偏差 / ADD-S / 重投影 / 耗时

设计依据: BOP LM test 为稀疏关键帧(帧间真实位移>100mm), 非视频;
          因此用渲染器生成"虚拟稠密视频", 可控验证时序链(docs/01 §6)。

用法:
    conda activate py3d
    python scripts/run_temporal_eval.py
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.geometry import exp_se3, make_T, transform, project_pose, \
    backproject_depth
from src.pipeline import (RegConfig, load_mesh_points, register_rgbd,
                          render_model, refine_icp)
from src.metrics import (add_s_error, pixel_jitter_deviation, reproj_rmse)
from src.se3_kalman_filter import SE3KalmanFilter, SE3KalmanConfig

import torch

# obj_000008 (duck): 非对称模型 —— 避免对称翻转导致的位姿歧义
# (对称物体的翻转是几何固有歧义, 已在 BOP 静态评测中用 ADD-S 指标正确评价;
#  时序实验需要无歧义轨迹, 故选非对称模型)
BOP_MODEL = Path(r'D:\bop\lm\models\obj_000008.ply')
K = np.array([[572.4, 0, 325.3], [0, 573.6, 242.0], [0, 0, 1.0]])
H, W = 480, 640


def kalman_mm(meas_pos=5.0, meas_rot=5e-3):
    """毫米单位、真实运动场景的 Kalman 配置

    dt 以【帧】为单位 (dt=1.0): Q_pos = process_noise_pos (mm²/帧)。
    关键: meas_noise_* 必须与位姿引擎的【真实噪声水平】匹配
    (欠规格的 R 会把引擎的正常抖动误判为跟踪丢失 → 频繁重定位)。
    """
    return SE3KalmanConfig(
        process_noise_pos=6.0,      # Q_pos = 6 mm²/帧
        process_noise_rot=6e-6,     # Q_rot = 6e-6 rad²/帧
        process_noise_vel_pos=0.5,  # 平移速度噪声 (std 0.7mm/帧)
        process_noise_vel_rot=2e-4, # 旋转速度噪声 (std 0.014rad/帧, 真实量级)
        meas_noise_pos=meas_pos,
        meas_noise_rot=meas_rot,
        innovation_threshold=10.0,  # 归一化(σ单位)
        velocity_damping=0.98)


def make_orbit_trajectory(n_frames=100, radius_orbit=60.0, z=900.0):
    """平滑相机轨道轨迹 (等价: 模型在相机系内做平滑反向运动)

    运动剖面: 椭圆平移 + 正弦俯仰/偏航, 含连续加速度 (非匀速),
    模拟手持 AR 相机的真实平滑运动。
    """
    poses = []
    for i in range(n_frames):
        a = 2 * np.pi * i / n_frames
        t = np.array([radius_orbit * np.sin(a),
                      30.0 * np.sin(2 * a),
                      z + 40.0 * np.cos(a)])
        rv = np.array([0.10 * np.sin(a), 0.15 * np.sin(a + 0.5),
                       0.08 * np.cos(a)])
        # 旋转部分: exp(rv)
        from src.geometry import exp_so3
        poses.append(make_T(exp_so3(rv), t))
    return poses


def centers_uv(poses, center):
    return np.array([project_pose(K, T, center.reshape(1, 3))[0][0]
                     for T in poses])


def multipoint_jitter_dev(est_poses, gt_poses, pts):
    """多点投影抖动偏差 (中心投影对过心旋转不敏感, 用表面多点更真实)"""
    pts = np.asarray(pts)
    devs = []
    for p in pts:
        uv_e = centers_uv(est_poses, p)
        uv_g = centers_uv(gt_poses, p)
        devs.append(pixel_jitter_deviation(uv_e, uv_g))
    return float(np.mean(devs))


def jitter_probe_points(model, n=9):
    """抖动探针点: 模型中心 + 8 个均匀散布表面点"""
    idx = np.linspace(0, len(model) - 1, n - 1).astype(int)
    return np.vstack([model.mean(axis=0, keepdims=True), model[idx]])


# ============================================================
# Part A: 受控轨迹 + 合成测量噪声
# ============================================================
def part_a(gt_poses, probe_pts, eval_pts):
    print('\n--- Part A: 平滑轨迹 + σ=5mm/3mrad 逐帧测量噪声 ---')
    rng = np.random.default_rng(0)
    scale = np.array([5, 5, 5, 0.003, 0.003, 0.003])
    meas = [T @ exp_se3(rng.normal(0, 1, 6) * scale) for T in gt_poses]

    kf = SE3KalmanFilter(kalman_mm())
    smooth = [kf.update(T, dt=1.0) for T in meas]  # dt 以帧为单位

    j_meas = multipoint_jitter_dev(meas, gt_poses, probe_pts)
    j_smooth = multipoint_jitter_dev(smooth, gt_poses, probe_pts)
    e_meas = np.mean([add_s_error(eval_pts, Tm, Tg)
                      for Tm, Tg in zip(meas, gt_poses)])
    e_smooth = np.mean([add_s_error(eval_pts, Ts, Tg)
                        for Ts, Tg in zip(smooth, gt_poses)])
    print(f'  抖动偏差RMS:   测量={j_meas:.3f}px  平滑={j_smooth:.3f}px  '
          f'降低={100 * (1 - j_smooth / j_meas):.1f}%')
    print(f'  ADD-S(相对GT): 测量={e_meas:.3f}mm  平滑={e_smooth:.3f}mm')
    print(f'  重定位次数: {kf.reloc_count}')
    return dict(j_meas=j_meas, j_smooth=j_smooth,
                e_meas=e_meas, e_smooth=e_smooth, reloc=kf.reloc_count)


# ============================================================
# Part B: 端到端虚拟视频跟踪
# ============================================================
def part_b(model, model_icp, eval_pts, probe_pts, device):
    print('\n--- Part B: 端到端虚拟视频 (渲染深度+噪声 → 逐帧跟踪) ---')
    gt_poses = make_orbit_trajectory(100)
    cfg = RegConfig(device=device)
    # ICP 引擎实测噪声: 平移波动 ~3mm, 旋转波动 ~1-2° (见 _debug_temporal)
    kf = SE3KalmanFilter(kalman_mm(meas_pos=3.0, meas_rot=0.02))
    rng = np.random.default_rng(1)

    white = np.ones((len(model), 3))
    meas_poses, smooth_poses = [], []
    times = []
    n_reloc = 0
    for i, T_gt in enumerate(gt_poses):
        # 虚拟相机: 渲染深度 + 噪声
        out = render_model(model, white, T_gt, K, H, W, radius=1,
                           device=device)
        depth = out['depth'].cpu().numpy()
        fg = depth > 0   # 仅对前景(物体)像素加噪声, 背景保持 0
        depth[fg] += rng.normal(0, 0.8, int(fg.sum()))

        t0 = time.time()
        if i == 0:
            res = register_rgbd(depth, K, model_icp, mask=(depth > 0),
                                cfg=cfg)
            T_meas = res['T']
        else:
            scene_pts, _, _ = backproject_depth(depth, K,
                                                mask=(depth > 0))
            T_pred = kf.T if kf.initialized else meas_poses[-1]
            # 虚拟视频场景只有前表面点云 → trim 0.6 抑制背面误对应
            T_meas, _ = refine_icp(model_icp, scene_pts, T_pred,
                                   max_iter=25, trim_ratio=0.6,
                                   device=device)
        T_smooth = kf.update(T_meas, dt=1.0)  # dt 以帧为单位
        times.append(time.time() - t0)
        meas_poses.append(T_meas)
        smooth_poses.append(T_smooth)

    j_meas = multipoint_jitter_dev(meas_poses, gt_poses, probe_pts)
    j_smooth = multipoint_jitter_dev(smooth_poses, gt_poses, probe_pts)
    e_meas = np.array([add_s_error(eval_pts, Tm, Tg)
                       for Tm, Tg in zip(meas_poses, gt_poses)])
    e_smooth = np.array([add_s_error(eval_pts, Ts, Tg)
                         for Ts, Tg in zip(smooth_poses, gt_poses)])
    r_meas = np.array([reproj_rmse(eval_pts, K, Tm, Tg)
                       for Tm, Tg in zip(meas_poses, gt_poses)])
    r_smooth = np.array([reproj_rmse(eval_pts, K, Ts, Tg)
                         for Ts, Tg in zip(smooth_poses, gt_poses)])
    print(f'  抖动偏差RMS:    测量={j_meas:.3f}px  平滑={j_smooth:.3f}px  '
          f'降低={100 * (1 - j_smooth / j_meas):.1f}%')
    print(f'  ADD-S mean:     测量={e_meas.mean():.3f}mm  '
          f'平滑={e_smooth.mean():.3f}mm')
    print(f'  重投影RMSE中位:  测量={np.median(r_meas):.3f}px  '
          f'平滑={np.median(r_smooth):.3f}px')
    print(f'  跟踪耗时: 帧0(全局)={times[0]:.2f}s  后续帧mean='
          f'{np.mean(times[1:]) * 1000:.0f}ms  '
          f'拒绝={kf.reject_count}次 重定位={kf.reloc_count}次')
    return dict(j_meas=j_meas, j_smooth=j_smooth,
                e_meas=float(e_meas.mean()), e_smooth=float(e_smooth.mean()),
                r_meas=float(np.median(r_meas)),
                r_smooth=float(np.median(r_smooth)),
                t_track=float(np.mean(times[1:])), reloc=kf.reloc_count,
                reject=kf.reject_count)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('=' * 66)
    print('视频时序一致性评测: SE(3) Kalman (真实运动剖面 + 真实模型)')
    print('=' * 66)

    model, _ = load_mesh_points(str(BOP_MODEL), n_points=20000)
    model = model - model.mean(axis=0)
    model_icp = model[::max(1, len(model) // 3000)]
    eval_pts = model[::max(1, len(model) // 2000)]
    probe_pts = jitter_probe_points(model)

    gt_poses = make_orbit_trajectory(100)
    ra = part_a(gt_poses, probe_pts, eval_pts)
    rb = part_b(model, model_icp, eval_pts, probe_pts, device)

    print('\n' + '=' * 66)
    print('汇总:')
    print(f'  Part A(受控):  抖动 {ra["j_meas"]:.2f}px → {ra["j_smooth"]:.2f}px '
          f'(↓{100 * (1 - ra["j_smooth"] / ra["j_meas"]):.0f}%), '
          f'ADD-S {ra["e_meas"]:.2f}mm → {ra["e_smooth"]:.2f}mm')
    print(f'  Part B(端到端): 抖动 {rb["j_meas"]:.2f}px → {rb["j_smooth"]:.2f}px '
          f'(↓{100 * (1 - rb["j_smooth"] / rb["j_meas"]):.0f}%), '
          f'ADD-S {rb["e_meas"]:.2f}mm → {rb["e_smooth"]:.2f}mm, '
          f'跟踪 {rb["t_track"] * 1000:.0f}ms/帧')
    print('=' * 66)


if __name__ == '__main__':
    main()
