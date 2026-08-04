# -*- coding: utf-8 -*-
"""
单元测试: 合成数据验证投影映射链条的每个环节
=============================================

T1  SE(3) exp/log 往返精度
T2  投影 → PnP+RANSAC 位姿恢复 (含噪声+外点)
T3  Kabsch/Umeyama 3D-3D 绝对定向恢复 (含尺度)
T4  多起点 ICP 无初值配准恢复
T5  SE(3) Kalman 像素级抖动降低
T6  光栅化器深度一致性 + 遮挡合成逻辑
T7  RTW: FFD 权重性质 + 非刚性恢复优于纯刚性

用法:
    conda activate py3d
    python scripts/run_unit_tests.py
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')  # Windows torch/numpy OpenMP 共存

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.geometry import (exp_se3, log_se3, exp_so3, log_so3, make_T,
                          transform, project_pose, backproject_depth)
from src.pose_estimation import (kabsch_umeyama, pnp_ransac,
                                 multistart_icp_register)
from src.metrics import pose_errors, reproj_rmse, pixel_jitter_rms
from src.se3_kalman_filter import SE3KalmanFilter

RESULTS = []


def report(name, ok, detail):
    RESULTS.append((name, ok, detail))
    tag = 'PASS' if ok else 'FAIL'
    print(f'  [{tag}] {name}: {detail}')


# ============================================================
# T1: SE(3) exp/log 往返
# ============================================================
def t1_se3_roundtrip():
    print('\n[T1] SE(3) 指数/对数映射往返精度')
    rng = np.random.default_rng(0)
    errs = []
    for _ in range(200):
        xi = rng.normal(0, 0.5, 6)
        err = np.linalg.norm(log_se3(exp_se3(xi)) - xi)
        errs.append(err)
    e = max(errs)
    report('exp∘log 往返', e < 1e-10, f'max_err={e:.2e}')


# ============================================================
# T2: 投影 → PnP 位姿恢复
# ============================================================
def t2_pnp_recovery():
    print('\n[T2] 合成 2D-3D 对应 → PnP+RANSAC 位姿恢复')
    rng = np.random.default_rng(1)
    K = np.array([[572.4, 0, 325.3], [0, 573.6, 242.0], [0, 0, 1.0]])
    N = 200
    X = rng.uniform(-60, 60, (N, 3))          # 模型点 (mm)
    X[:, 2] *= 0.3

    # 真实位姿: 旋转 ~20°, 平移 z≈900mm
    T_gt = make_T(exp_so3(np.array([0.1, -0.2, 0.15])),
                  np.array([-50, 30, 900.0]))
    uv, z = project_pose(K, T_gt, X)
    # 加 0.5px 噪声 + 15% 外点
    uv_n = uv + rng.normal(0, 0.5, uv.shape)
    n_out = int(0.15 * N)
    out_idx = rng.choice(N, n_out, replace=False)
    uv_n[out_idx] += rng.uniform(-40, 40, (n_out, 2))

    T_est, st = pnp_ransac(X, uv_n, K, reproj_thresh=2.0)
    assert T_est is not None, st
    rot_deg, trans_mm = pose_errors(T_est, T_gt)
    r_rmse = reproj_rmse(X, K, T_est, T_gt)
    ok = rot_deg < 1.0 and trans_mm < 20.0
    report('PnP 恢复', ok,
           f'rot_err={rot_deg:.3f}° trans_err={trans_mm:.2f}mm '
           f'reproj={r_rmse:.2f}px inl={st["inlier_ratio"]:.2f}')


# ============================================================
# T3: Kabsch/Umeyama 恢复 (含尺度)
# ============================================================
def t3_kabsch():
    print('\n[T3] Kabsch/Umeyama 3D-3D 绝对定向恢复')
    rng = np.random.default_rng(2)
    X = rng.normal(0, 50, (300, 3))
    R_gt = exp_so3(rng.normal(0, 0.6, 3))
    t_gt = rng.normal(0, 300, 3)
    s_gt = 2.5
    Y = s_gt * (X @ R_gt.T) + t_gt + rng.normal(0, 0.5, X.shape)

    R, t, s = kabsch_umeyama(X, Y, with_scale=True)
    rot_err = np.degrees(np.linalg.norm(log_so3(R @ R_gt.T)))
    s_err = abs(s - s_gt) / s_gt
    t_err = np.linalg.norm((t - t_gt)) / np.linalg.norm(t_gt)
    ok = rot_err < 0.5 and s_err < 0.01 and t_err < 0.02
    report('Umeyama(带尺度)', ok,
           f'rot_err={rot_err:.4f}° scale_err={s_err:.2%} trans_err={t_err:.2%}')


# ============================================================
# T4: 多起点 ICP 无初值配准
# ============================================================
def t4_icp_multistart():
    print('\n[T4] 多起点 ICP 无初值 3D-3D 配准恢复')
    from src.pipeline import load_mesh_points
    model_path = Path(r'D:\bop\lm\models\obj_000001.ply')
    device = 'cuda'
    if model_path.exists():
        pts, _ = load_mesh_points(str(model_path), n_points=3000)
        src_note = 'BOP obj_000001'
    else:
        rng = np.random.default_rng(3)
        pts = rng.normal(0, 50, (3000, 3))
        pts[:, 2] *= 0.4
        src_note = '随机点云(备用)'
    model = pts - pts.mean(axis=0)

    rng = np.random.default_rng(4)
    # 中等旋转 (~20°, 在 ICP 收敛域内; 大旋转需特征/学习法初始化, 见 docs/03 §6)
    T_gt = make_T(exp_so3(rng.normal(0, 0.2, 3)), np.array([120, -80, 950.0]))
    scene = transform(T_gt, model) + rng.normal(0, 1.0, model.shape)
    # 模拟遮挡: 随机去掉 25% 场景点
    keep = rng.random(scene.shape[0]) > 0.25
    scene = scene[keep]

    t0 = time.time()
    T_est, st = multistart_icp_register(model, scene, device=device,
                                        coarse_iter=30, trim_coarse=0.6,
                                        trim_fine=0.8)
    dt = time.time() - t0
    rot_deg, trans_mm = pose_errors(T_est, T_gt)
    ok = rot_deg < 5.0 and trans_mm < 60.0
    report(f'多起点ICP ({src_note})', ok,
           f'rot_err={rot_deg:.2f}° trans_err={trans_mm:.1f}mm '
           f'rmse={st["rmse"]:.2f}mm time={dt:.2f}s')


# ============================================================
# T5: SE(3) Kalman 像素抖动降低
# ============================================================
def t5_kalman_pixel_jitter():
    print('\n[T5] SE(3) Kalman 像素级抖动降低 (合成轨迹)')
    rng = np.random.default_rng(5)
    K = np.array([[572.4, 0, 325.3], [0, 573.6, 242.0], [0, 0, 1.0]])
    center = np.zeros(3)
    n = 100
    dt = 1 / 30.0
    # 注意: SE3KalmanFilter 默认噪声参数与重定位阈值为【米制】设计,
    # 因此轨迹使用米制单位 (单位一致性是坐标系工程的关键细节)

    # 真实轨迹: 平滑圆弧运动 (米)
    true_poses = []
    for i in range(n):
        a = 0.01 * i
        T = make_T(exp_so3(np.array([0.0, 0.0, 0.02 * np.sin(a * 3)])),
                   np.array([0.002 * i, 0.30, 0.90 + 0.05 * np.sin(a)]))
        true_poses.append(T)
    # 噪声测量 (模拟逐帧位姿估计): 平移 3mm + 旋转 2mrad
    meas = [T @ exp_se3(rng.normal(0, 1, 6) * np.array([0.003, 0.003, 0.003,
                                                        0.002, 0.002, 0.002]))
            for T in true_poses]

    kf = SE3KalmanFilter()
    smooth = [kf.update(T, dt) for T in meas]

    def centers(poses):
        return np.array([project_pose(K, T, center.reshape(1, 3))[0][0]
                         for T in poses])
    j_true = pixel_jitter_rms(centers(true_poses))
    j_meas = pixel_jitter_rms(centers(meas))
    j_smooth = pixel_jitter_rms(centers(smooth))
    red = 1 - (j_smooth - j_true) / max(j_meas - j_true, 1e-9)
    ok = j_smooth < j_meas * 0.5
    report('Kalman 像素抖动', ok,
           f'GT={j_true:.3f}px meas={j_meas:.3f}px smooth={j_smooth:.3f}px '
           f'抖动降低={100 * (1 - j_smooth / j_meas):.1f}%')


# ============================================================
# T6: 光栅化器一致性 + 遮挡逻辑
# ============================================================
def t6_rasterizer():
    print('\n[T6] 光栅化深度一致性 + 深度感知遮挡合成')
    import torch
    from src.rasterize import splat_render, composite_depth_aware
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    H, W = 120, 160
    K = torch.tensor([[100., 0, 80], [0, 100, 60], [0, 0, 1]], device=device)
    # 两个点: P1 近 (z=400) 与 P2 远 (z=800), 投影到同一像素
    pts = torch.tensor([[4., 6., 400.], [8., 12., 800.]], device=device)
    feat = torch.tensor([[1., 0., 0.], [0., 0., 1.]], device=device)
    out = splat_render(pts, feat, K, H, W, radius=0)
    # 按投影公式计算期望像素 (避免硬编码取整误差)
    u_exp = int(torch.round(100 * pts[0, 0] / pts[0, 2] + 80).item())
    v_exp = int(torch.round(100 * pts[0, 1] / pts[0, 2] + 60).item())
    d = out['depth'][v_exp, u_exp].item()
    rgb = out['feat'][v_exp, u_exp]
    ok_depth = abs(d - 400.0) < 1e-3
    ok_color = rgb[0].item() > 0.99  # 近点(红)胜出
    report('z-buffer 最近点胜出', ok_depth and ok_color,
           f'depth={d:.1f}(期望400) color=R{rgb[0]:.2f},G{rgb[1]:.2f},B{rgb[2]:.2f}')

    # 遮挡合成: 虚拟在真实之后 → 不可见; 之前 → 可见
    real_rgb = torch.full((4, 4, 3), 0.5, device=device)
    real_depth = torch.full((4, 4), 500.0, device=device)
    virt_rgb = torch.ones(4, 4, 3, device=device)
    virt_depth = torch.tensor([[600., 400., 0, 0]] * 4, device=device)
    comp, occ = composite_depth_aware(real_rgb, real_depth, virt_rgb, virt_depth)
    ok_occ = (not occ[0, 0].item()) and occ[0, 1].item()
    report('深度感知遮挡', ok_occ,
           f'远虚拟可见={bool(occ[0,0])}(期望False) 近虚拟可见={bool(occ[0,1])}(期望True)')


# ============================================================
# T7: RTW 非刚性恢复
# ============================================================
def t7_rtw():
    print('\n[T7] RTW: 非刚性形变恢复 (FFD 合成形变)')
    import torch
    from src.nonrigid_rtw import FFDLattice, rtw_optimize
    from src.pose_estimation import nearest_neighbors
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    rng = np.random.default_rng(7)
    # 模型: 椭球点云
    model = rng.normal(0, 40, (2500, 3))
    model[:, 2] *= 0.5
    model -= model.mean(axis=0)

    # GT: 已知 FFD 形变 (平滑 δ, 幅度 ~8mm)
    lat = FFDLattice(model.min(0), model.max(0), grid=(4, 4, 4), device=device)
    delta_gt = rng.normal(0, 6.0, (lat.n_ctrl, 3))
    # 平滑化 δ (相邻平均几次)
    for _ in range(3):
        e = lat._edges.cpu().numpy()
        d_new = delta_gt.copy()
        for a, b in e:
            d_new[a] = 0.5 * (d_new[a] + delta_gt[b])
        delta_gt = d_new
    W = lat.weights(model)
    model_def = model + (W @ torch.as_tensor(delta_gt, dtype=torch.float32,
                                             device=device)).cpu().numpy()

    T_gt = make_T(exp_so3(np.array([0.15, -0.1, 0.2])), np.array([30, -20, 900.0]))
    scene = transform(T_gt, model_def) + rng.normal(0, 0.5, model_def.shape)

    # 方法A: 纯刚性 ICP
    from src.pose_estimation import icp
    T_rigid, st_rigid = icp(model, scene, T_init=np.eye(4) @
                            make_T(np.eye(3), scene.mean(0) - model.mean(0)),
                            max_iter=80, trim_ratio=0.8, device=device)
    idx, d = nearest_neighbors(transform(T_rigid, model), scene, device=device)
    rmse_rigid = float(np.sqrt((d ** 2).mean()))

    # 方法B: RTW
    rtw = rtw_optimize(model, scene, T_rigid, grid=(4, 4, 4), iters=250,
                       device=device, seed=0)
    rmse_rtw = rtw['rmse']
    ok = rmse_rtw < rmse_rigid * 0.7
    report('RTW vs RT', ok,
           f'刚性RMSE={rmse_rigid:.3f}mm RTW-RMSE={rmse_rtw:.3f}mm '
           f'残差降低={100 * (1 - rmse_rtw / rmse_rigid):.1f}%')


if __name__ == '__main__':
    print('=' * 64)
    print('单元测试: 2D↔3D↔相机 投影映射链条')
    print('=' * 64)
    t0 = time.time()
    t1_se3_roundtrip()
    t2_pnp_recovery()
    t3_kabsch()
    t4_icp_multistart()
    t5_kalman_pixel_jitter()
    t6_rasterizer()
    t7_rtw()
    print('=' * 64)
    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    print(f'结果: {n_pass}/{len(RESULTS)} 通过, 总耗时 {time.time() - t0:.1f}s')
    print('=' * 64)
    sys.exit(0 if n_pass == len(RESULTS) else 1)
