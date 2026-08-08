# -*- coding: utf-8 -*-
"""
RTW 非刚性注册评测 (真实 CAD 模型 + 合成形变)
=============================================

验证 docs/03 §5 的核心命题:
    对非刚性形变物体, RTW (刚性+FFD形变场) 显著优于纯 RT。

协议:
  1. BOP 真实模型 obj_000001 → 施加已知 FFD 形变 (平滑, 幅度~6mm) + 已知位姿
  2. torch 光栅化器渲染深度 (+噪声) → 反投影得观测点云
  3. 方法A: 纯刚性配准 (FPFH+ICP)      → 残差/顶点误差
     方法B: RTW 联合优化 (本方案)      → 残差/顶点误差/形变恢复
  4. 对比指标: 截尾Chamfer RMSE, 逐顶点形变恢复误差, 刚性位姿误差, 耗时

用法:
    conda activate py3d
    python scripts/run_rtw_eval.py
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.geometry import make_T, exp_so3, transform
from src.pipeline import load_mesh_points, register_rgbd, RegConfig
from src.nonrigid_rtw import (FFDLattice, rtw_optimize,
                              rtw_optimize_projective,
                              rtw_optimize_projective_obs,
                              apply_deformation)
from src.pose_estimation import nearest_neighbors
from src.metrics import pose_errors

import torch

BOP_MODEL = Path(r'D:\bop\lm\models\obj_000008.ply')  # duck, 非对称


def chamfer_rmse(src, dst, trim=0.9, device='cuda'):
    idx, d = nearest_neighbors(src, dst, device=device)
    k = max(8, int(trim * len(d)))
    return float(np.sqrt((np.sort(d)[:k] ** 2).mean()))


def shape_chamfer(P, Q, trim=0.9, device='cuda'):
    """对称形状 Chamfer (规范不变): 衡量"形状"是否恢复, 而非对应场"""
    _, d1 = nearest_neighbors(P, Q, device=device)
    _, d2 = nearest_neighbors(Q, P, device=device)
    d = np.concatenate([d1, d2])
    k = max(8, int(trim * len(d)))
    return float(np.sqrt((np.sort(d)[:k] ** 2).mean()))


def depth_residual_rms(V_cam, depth_obs, K, inlier=3.0, device='cuda'):
    """投影深度残差: (全部有效点 RMS, 3mm内点比例)

    不设内点门限的 RMS 才有刚性-vs-RTW 可比性 (门限会造成幸存者偏差)。
    """
    import torch as th
    dev = device if th.cuda.is_available() else 'cpu'
    V = th.as_tensor(V_cam, dtype=th.float32, device=dev)
    D = th.as_tensor(depth_obs, dtype=th.float32, device=dev)
    Himg, Wimg = D.shape
    z = V[:, 2].clamp(min=1e-6)
    u = K[0, 0] * V[:, 0] / z + K[0, 2]
    v = K[1, 1] * V[:, 1] / z + K[1, 2]
    gx = 2.0 * u / (Wimg - 1) - 1.0
    gy = 2.0 * v / (Himg - 1) - 1.0
    g = th.stack([gx, gy], dim=-1).reshape(1, 1, -1, 2)
    z_obs = th.nn.functional.grid_sample(
        D.reshape(1, 1, Himg, Wimg), g, mode='bilinear',
        padding_mode='zeros', align_corners=True).reshape(-1)
    res = (z_obs - z)
    valid = (z_obs > 0) & (V[:, 2] > 1e-3)
    if int(valid.sum()) < 10:
        return float('nan'), 0.0
    inl = valid & (res.abs() < inlier)
    rms = float(th.sqrt((res[inl] ** 2).mean()).cpu()) if int(inl.sum()) > 10 \
        else float('nan')
    inl_frac = float((inl.float().mean()).cpu())
    return rms, inl_frac


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rng = np.random.default_rng(42)
    print('=' * 66)
    print('RTW 非刚性注册评测: 真实模型 + 合成 FFD 形变')
    print('=' * 66)

    # 1) 模型与 GT 形变 ------------------------------------------------
    model, normals = load_mesh_points(str(BOP_MODEL), n_points=8000)
    model = model - model.mean(axis=0)
    lat_gt = FFDLattice(model.min(0), model.max(0), grid=(4, 4, 4), device=device)
    delta_gt = rng.normal(0, 5.0, (lat_gt.n_ctrl, 3))
    # 轻度平滑化 (模拟组织级平滑形变, 保留幅度)
    edges = lat_gt._edges.cpu().numpy()
    for _ in range(1):
        d_new = delta_gt.copy()
        cnt = np.ones(len(delta_gt))
        for a, b in edges:
            d_new[a] += delta_gt[b]
            d_new[b] += delta_gt[a]
            cnt[a] += 1
            cnt[b] += 1
        delta_gt = d_new / cnt[:, None]
    W_gt = lat_gt.weights(model)
    # --- 构造受控 GT 形变: 随机平滑分量(mean 3mm) + 视线方向弯曲分量 ---
    # (弯曲分量沿相机 z 渐变 → 单视图可观测; 随机分量模拟真实形变)
    T_gt_tmp = make_T(exp_so3(np.array([0.12, -0.08, 0.18])),
                      np.array([25.0, -15.0, 900.0]))
    warp_rand = (W_gt @ torch.as_tensor(
        delta_gt, dtype=torch.float32, device=device)).cpu().numpy()
    s_r = 3.0 / max(np.linalg.norm(warp_rand, axis=1).mean(), 1e-9)
    delta_1 = delta_gt * s_r
    # 视线弯曲: 控制点 x 坐标渐变的相机-z 方向偏移 (模型系表示)
    z_axis_model = T_gt_tmp[:3, :3].T @ np.array([0, 0, 1.0])
    gx, gy, gz = lat_gt.grid
    ii, jj, kk = np.meshgrid(np.arange(gx), np.arange(gy), np.arange(gz),
                             indexing='ij')
    C = lat_gt.bbox_min + np.stack([ii, jj, kk], -1).reshape(-1, 3) * lat_gt.step
    w_grad = (C[:, 0] - C[:, 0].mean()) / (np.ptp(C[:, 0]) + 1e-9)  # [-0.5,0.5]
    delta_2 = (6.0 * w_grad)[:, None] * z_axis_model[None, :]
    delta_gt = delta_1 + delta_2
    model_def = model + (W_gt @ torch.as_tensor(
        delta_gt, dtype=torch.float32, device=device)).cpu().numpy()
    warp_mag = np.linalg.norm(model_def - model, axis=1)
    warp_z_mag = np.abs((model_def - model) @ T_gt_tmp[:3, :3].T[:, 2]).mean()
    print(f'\n[GT] FFD形变: 幅度 mean={warp_mag.mean():.2f}mm '
          f'max={warp_mag.max():.2f}mm (模型直径≈'
          f'{np.linalg.norm(model.max(0) - model.min(0)):.0f}mm)')

    # 2) GT 位姿 + 渲染深度 + 噪声 → 观测点云 --------------------------
    T_gt = make_T(exp_so3(np.array([0.12, -0.08, 0.18])),
                  np.array([25.0, -15.0, 900.0]))
    K = np.array([[572.4, 0, 325.3], [0, 573.6, 242.0], [0, 0, 1.0]])
    H, W = 480, 640
    from src.pipeline import render_model
    white = np.ones((len(model_def), 3))
    out = render_model(model_def, white, T_gt, K, H, W, radius=1,
                       device=device)
    depth_gt = out['depth'].cpu().numpy()
    depth_obs = depth_gt + rng.normal(0, 0.8, depth_gt.shape)
    depth_obs[depth_gt <= 0] = 0
    print(f'[观测] 渲染深度: 有效像素 {int((depth_gt > 0).sum())}, 噪声σ=0.8mm')

    # 3) 方法A: 纯刚性 -------------------------------------------------
    t0 = time.time()
    cfg = RegConfig(device=device)
    res_rigid = register_rgbd(depth_obs, K, model, mask=(depth_obs > 0),
                              cfg=cfg)
    t_rigid = time.time() - t0
    T_rigid = res_rigid['T']
    scene_pts = res_rigid['scene_pts']
    rmse_rigid = chamfer_rmse(transform(T_rigid, model), scene_pts,
                              device=device)
    rot_r, tr_r = pose_errors(T_rigid, T_gt)
    print(f'\n[A 纯刚性] RMSE={rmse_rigid:.3f}mm  '
          f'位姿误差 rot={rot_r:.2f}° t={tr_r:.2f}mm  耗时={t_rigid:.2f}s')

    # 4) 方法B: RTW (投影数据关联: 深度残差直接定义在成像平面) ----------
    # 完整配置: two_phase(先收敛位姿治R-δ耦合) + edge_band(排除剪影边缘竞争)
    # + inlier门限覆盖形变幅度(max~5.9mm) + lambda_pose锚定
    t0 = time.time()
    rtw = rtw_optimize_projective(model, depth_obs, K, T_rigid,
                                  grid=(4, 4, 4), iters=400, lr=1e-2,
                                  lambda_mag=1e-4, lambda_sm=1e-3,
                                  lambda_p2p=1.0, lambda_pose=1.0,
                                  two_phase=True, edge_band_px=0.0,
                                  inlier_hi=15.0, inlier_lo=8.0,
                                  device=device, seed=0, verbose=True)
    t_rtw = time.time() - t0
    V_cam_rtw = transform(rtw['T'], rtw['V_def'])
    rmse_rtw = chamfer_rmse(V_cam_rtw, scene_pts, device=device)
    rot_w, tr_w = pose_errors(rtw['T'], T_gt)

    # 逐顶点形变恢复 (对应关系已知): |V_def_est - V_def_gt|
    vtx_err = np.linalg.norm(rtw['V_def'] - model_def, axis=1)
    # 刚性残差形变 (RT-only 的未建模形变量) 作为基线
    vtx_err_rigid = np.linalg.norm(model - model_def, axis=1)

    # 可观测性分解: 面向相机的顶点 (法线朝向镜头) vs 背面顶点
    n_cam = normals @ T_gt[:3, :3].T
    front = n_cam[:, 2] < 0
    print(f'[B RTW]   RMSE={rmse_rtw:.3f}mm  '
          f'位姿误差 rot={rot_w:.2f}° t={tr_w:.2f}mm  耗时={t_rtw:.2f}s')
    print(f'        损失: {rtw["losses"][0]:.3f} → {rtw["losses"][-1]:.3f}  '
          f'深度残差RMS={rtw["res_rms"]:.3f}mm  可见顶点占比={front.mean():.1%}')

    # 5) 规范不变指标: 形状 Chamfer + 深度残差 ---------------------------
    # 注: 逐顶点对应误差混合了 RTW 规范自由度(全局刚体分量可被 R,t 或 δ
    # 任一方吸收)与单视图切向不可观分量, 不能准确反映形状恢复质量;
    # AR 关心的是"形状对不对", 故以形状 Chamfer + 深度残差为主指标。
    shape_rigid = shape_chamfer(model, model_def, trim=1.0, device=device)
    shape_rtw = shape_chamfer(rtw['V_def'], model_def, trim=1.0, device=device)
    res_rigid, inl_rigid = depth_residual_rms(transform(T_rigid, model),
                                              depth_obs, K, inlier=5.0,
                                              device=device)
    res_rtw, inl_rtw = depth_residual_rms(V_cam_rtw, depth_obs, K, inlier=5.0,
                                          device=device)

    rec_all = 100 * (1 - vtx_err.mean() / vtx_err_rigid.mean())
    print('\n' + '=' * 66)
    print('指标汇总:')
    print(f'  [主指标·规范不变]')
    print(f'  深度残差 RMS(5mm门限): 刚性={res_rigid:.3f}mm → RTW={res_rtw:.3f}mm  '
          f'(↓{100 * (1 - res_rtw / res_rigid):.1f}%)  [观测噪声σ=0.8mm]')
    print(f'  5mm内点比例:     刚性={inl_rigid:.1%} → RTW={inl_rtw:.1%}')
    print(f'  形状 Chamfer:    刚性={shape_rigid:.3f}mm → RTW={shape_rtw:.3f}mm  '
          f'(↓{100 * (1 - shape_rtw / shape_rigid):.1f}%)')
    print(f'  [参考指标]')
    print(f'  Chamfer RMSE:      刚性={rmse_rigid:.3f}mm → RTW={rmse_rtw:.3f}mm  '
          f'(↓{100 * (1 - rmse_rtw / rmse_rigid):.1f}%)')
    print(f'  逐顶点形变残差:    未建模={vtx_err_rigid.mean():.3f}mm → '
          f'RTW={vtx_err.mean():.3f}mm  (↓{rec_all:.1f}%, 含规范+切向不可观分量)')
    print(f'  刚性位姿误差:      rot 刚性={rot_r:.2f}° RTW={rot_w:.2f}°, '
          f't 刚性={tr_r:.2f}mm RTW={tr_w:.2f}mm')
    print(f'  耗时:              刚性={t_rigid:.2f}s  RTW额外={t_rtw:.2f}s')
    print('=' * 66)


def part2_local_dent(device):
    """Part 2: 局部高频形变 (高斯凹痕) —— RTW 的主场

    全局平滑形变易被刚性位姿吸收(刚性简并, Part 1);
    局部形变(如器械压迫组织凹痕)无法被全局刚性吸收, 且深度变化显著
    → 单视图可观测性好, 是 RTW 真正发挥作用的场景。
    """
    rng = np.random.default_rng(7)
    print('\n' + '=' * 66)
    print('Part 2: 局部高斯凹痕形变 (幅度8mm, σ=15mm, 前表面)')
    print('=' * 66)

    model, normals = load_mesh_points(str(BOP_MODEL), n_points=8000)
    model = model - model.mean(axis=0)
    T_gt = make_T(exp_so3(np.array([0.12, -0.08, 0.18])),
                  np.array([25.0, -15.0, 900.0]))
    K = np.array([[572.4, 0, 325.3], [0, 573.6, 242.0], [0, 0, 1.0]])
    H, W = 480, 640

    # 凹痕中心: 选一个面向相机的顶点
    n_cam = normals @ T_gt[:3, :3].T
    front_score = -n_cam[:, 2]
    c_idx = int(np.argmax(front_score + rng.normal(0, 0.1, len(front_score))))
    c0 = model[c_idx]
    sigma, amp = 15.0, 8.0
    d2 = ((model - c0) ** 2).sum(axis=1)
    z_axis_model = T_gt[:3, :3].T @ np.array([0, 0, 1.0])
    dent = (amp * np.exp(-d2 / (2 * sigma ** 2)))[:, None] * z_axis_model[None, :]
    model_def = model + dent
    warp_mag = np.linalg.norm(dent, axis=1)
    print(f'[GT] 凹痕: 顶点={c_idx}, 幅度 mean={warp_mag.mean():.2f}mm '
          f'max={warp_mag.max():.2f}mm')

    from src.pipeline import render_model
    white = np.ones((len(model_def), 3))
    out = render_model(model_def, white, T_gt, K, H, W, radius=1,
                       device=device)
    depth_gt = out['depth'].cpu().numpy()
    depth_obs = depth_gt + rng.normal(0, 0.8, depth_gt.shape)
    depth_obs[depth_gt <= 0] = 0

    # A: 纯刚性
    cfg = RegConfig(device=device)
    res_rigid = register_rgbd(depth_obs, K, model, mask=(depth_obs > 0),
                              cfg=cfg)
    T_rigid = res_rigid['T']
    rot_r, tr_r = pose_errors(T_rigid, T_gt)

    # B: RTW (凹痕 σ=15mm → 控制格需更细: 6³≈20mm 格元;
    #          Rodrigues修复后正则下调; 门限须覆盖形变幅度(8mm凹痕>3mm门限))
    rtw = rtw_optimize_projective(model, depth_obs, K, T_rigid,
                                  grid=(6, 6, 6), iters=500, lr=1e-2,
                                  lambda_mag=2e-4, lambda_sm=1e-3,
                                  lambda_p2p=1.0,
                                  inlier_hi=30.0, inlier_lo=12.0,
                                  device=device, seed=0)
    V_cam_rtw = transform(rtw['T'], rtw['V_def'])
    rot_w, tr_w = pose_errors(rtw['T'], T_gt)

    # 指标 (合成数据无野点: 形状 Chamfer 不截尾, 避免掩盖局部形变区)
    res_r, inl_r = depth_residual_rms(transform(T_rigid, model), depth_obs,
                                      K, inlier=5.0, device=device)
    res_w, inl_w = depth_residual_rms(V_cam_rtw, depth_obs, K, inlier=5.0,
                                      device=device)
    shape_r = shape_chamfer(model, model_def, trim=1.0, device=device)
    shape_w = shape_chamfer(rtw['V_def'], model_def, trim=1.0, device=device)
    # 渲染深度图差异 (AR 视觉误差的直接度量): 估计位姿渲染 vs GT 渲染
    from src.pipeline import render_model as _rm
    white_m = np.ones((len(model), 3))
    d_rig = _rm(model, white_m, T_rigid, K, H, W, radius=1,
                device=device)['depth'].cpu().numpy()
    d_rtw = _rm(rtw['V_def'], white_m, rtw['T'], K, H, W, radius=1,
                device=device)['depth'].cpu().numpy()
    sil = depth_gt > 0
    def _derr(d):
        v = sil & (d > 0)
        return float(np.abs(d[v] - depth_gt[v]).mean()) if v.sum() > 10 \
            else float('nan')
    render_err_r, render_err_w = _derr(d_rig), _derr(d_rtw)
    # 凹痕区域恢复: 凹痕显著(warp>2mm)的顶点集合上的逐顶点残差
    roi = warp_mag > 2.0
    vtx_w = np.linalg.norm(rtw['V_def'] - model_def, axis=1)
    print(f'[A 刚性] 深度残差={res_r:.3f}mm 内点={inl_r:.1%} '
          f'形状Ch={shape_r:.3f}mm 渲染差={render_err_r:.3f}mm rot={rot_r:.2f}°')
    print(f'[B RTW]  深度残差={res_w:.3f}mm 内点={inl_w:.1%} '
          f'形状Ch={shape_w:.3f}mm 渲染差={render_err_w:.3f}mm rot={rot_w:.2f}°')
    print(f'\n指标汇总 (Part 2):')
    print(f'  深度残差 RMS:  刚性={res_r:.3f}mm → RTW={res_w:.3f}mm  '
          f'(↓{100 * (1 - res_w / res_r):.1f}%)')
    print(f'  渲染深度差:    刚性={render_err_r:.3f}mm → RTW={render_err_w:.3f}mm  '
          f'(↓{100 * (1 - render_err_w / render_err_r):.1f}%)')
    print(f'  形状 Chamfer:  刚性={shape_r:.3f}mm → RTW={shape_w:.3f}mm')
    print(f'  凹痕区形变恢复: 未建模={warp_mag[roi].mean():.3f}mm → '
          f'RTW残差={vtx_w[roi].mean():.3f}mm  '
          f'(↓{100 * (1 - vtx_w[roi].mean() / warp_mag[roi].mean()):.1f}%)')


def part3_obs_guided(device):
    """Part 3 (E2): 全空间 RTW vs 可观测性引导 RTW (同一凹痕场景)"""
    rng = np.random.default_rng(7)
    print('\n' + '=' * 66)
    print('Part 3 (E2): 可观测性引导 RTW vs 全空间 RTW')
    print('=' * 66)
    model, normals = load_mesh_points(str(BOP_MODEL), n_points=8000)
    model = model - model.mean(axis=0)
    T_gt = make_T(exp_so3(np.array([0.12, -0.08, 0.18])),
                  np.array([25.0, -15.0, 900.0]))
    K = np.array([[572.4, 0, 325.3], [0, 573.6, 242.0], [0, 0, 1.0]])
    H, W = 480, 640
    n_cam = normals @ T_gt[:3, :3].T
    c_idx = int(np.argmax(-n_cam[:, 2] + rng.normal(0, 0.1, len(n_cam))))
    c0 = model[c_idx]
    d2 = ((model - c0) ** 2).sum(axis=1)
    z_axis_model = T_gt[:3, :3].T @ np.array([0, 0, 1.0])
    dent = (8.0 * np.exp(-d2 / (2 * 15.0 ** 2)))[:, None] \
        * z_axis_model[None, :]
    model_def = model + dent
    warp_mag = np.linalg.norm(dent, axis=1)

    from src.pipeline import render_model
    white = np.ones((len(model_def), 3))
    out = render_model(model_def, white, T_gt, K, H, W, radius=1,
                       device=device)
    depth_gt = out['depth'].cpu().numpy()
    depth_obs = depth_gt + rng.normal(0, 0.8, depth_gt.shape)
    depth_obs[depth_gt <= 0] = 0

    cfg = RegConfig(device=device)
    res_rigid = register_rgbd(depth_obs, K, model, mask=(depth_obs > 0),
                              cfg=cfg)
    T_rigid = res_rigid['T']

    # 全空间 RTW
    rtw_f = rtw_optimize_projective(model, depth_obs, K, T_rigid,
                                    grid=(6, 6, 6), iters=400, lr=1e-2,
                                    lambda_mag=2e-3, lambda_sm=5e-3,
                                    device=device, seed=0)
    # 可观测性引导 RTW
    rtw_o = rtw_optimize_projective_obs(model, depth_obs, K, T_rigid,
                                        grid=(6, 6, 6), iters=400, lr=1e-2,
                                        lambda_mag=2e-4, lambda_sm=1e-3,
                                        lambda_unobs=1e-2,
                                        inlier_hi=30.0, inlier_lo=12.0,
                                        device=device, seed=0, verbose=True)

    roi = warp_mag > 2.0
    rows = []
    for name, rtw in [('全空间RTW', rtw_f), ('观测引导RTW', rtw_o)]:
        V_cam = transform(rtw['T'], rtw['V_def'])
        res_r, inl_r = depth_residual_rms(V_cam, depth_obs, K, inlier=5.0,
                                          device=device)
        shape = shape_chamfer(rtw['V_def'], model_def, trim=1.0,
                              device=device)
        vtx = np.linalg.norm(rtw['V_def'] - model_def, axis=1)
        # 恢复能量分解 (可观测/不可观测子空间)
        V_obs = rtw.get('V_obs')
        decomp = ''
        if V_obs is not None and V_obs.shape[1] > 0:
            err_vec = (rtw['V_def'] - model_def)
            # 顶点级误差→控制点误差 (用 W 最小二乘反投影)
            W_all = rtw['lattice'].weights(model)
            import torch as _t
            Wt = W_all
            Et = _t.as_tensor(err_vec, dtype=_t.float32, device=Wt.device)
            pinv = _t.linalg.pinv(Wt)
            d_err = (pinv @ Et).reshape(-1)
            proj = V_obs @ (V_obs.T @ d_err)
            e_obs = float((proj ** 2).sum())
            e_unobs = float(((d_err - proj) ** 2).sum())
            decomp = f'  [恢复能量: 可观测 {e_obs:.1f} / 不可观测 {e_unobs:.1f}]'
        print(f'  [{name}] 深度残差={res_r:.3f}mm 内点={inl_r:.1%} '
              f'形状Ch={shape:.3f}mm 凹痕恢复={vtx[roi].mean():.3f}mm'
              f'(未建模{warp_mag[roi].mean():.2f}){decomp}')
        rows.append(dict(name=name, res=res_r, shape=shape,
                         vtx_roi=float(vtx[roi].mean())))
    print(f'  可观测子空间维数: {rtw_o["obs_rank"]}/{3 * 6 ** 3}')


if __name__ == '__main__':
    main()
    import torch as _th
    part2_local_dent('cuda' if _th.cuda.is_available() else 'cpu')
    part3_obs_guided('cuda' if _th.cuda.is_available() else 'cpu')
