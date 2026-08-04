# -*- coding: utf-8 -*-
"""
注册性能指标体系
================

几何精度:
  - ADD / ADD-S (BOP 标准, 模型点平均距离 / 最近邻距离)
  - AR recall (直径分数阈值下的召回率, BOP 官方协议)
  - 重投影 RMSE (像素)
  - 位姿误差 (旋转角 deg, 平移 mm)

时序稳定性:
  - 像素抖动 RMS (相邻帧投影中心位移的标准差)
  - 位姿增量抖动 RMS

渲染/合成:
  - 遮挡正确率 (深度合成决策与真值一致率)

依赖: numpy
作者: ZCode · 2026-08-03
"""
import numpy as np

from src.geometry import transform, project_pose, log_so3, inverse_se3, log_se3


# ============================================================
# 几何精度: ADD / ADD-S
# ============================================================

def add_error(model_pts: np.ndarray, T_est: np.ndarray, T_gt: np.ndarray) -> float:
    """ADD: 模型点在两位姿下的平均距离 (非对称物体)"""
    P_est = transform(T_est, model_pts)
    P_gt = transform(T_gt, model_pts)
    return float(np.linalg.norm(P_est - P_gt, axis=1).mean())


def add_s_error(model_pts: np.ndarray, T_est: np.ndarray, T_gt: np.ndarray,
                chunk: int = 4096) -> float:
    """ADD-S: 模型点变换后到 GT 变换点集的最近邻平均距离 (对称物体适用)"""
    P_est = transform(T_est, model_pts)
    P_gt = transform(T_gt, model_pts)
    N = P_est.shape[0]
    d_sum, d_cnt = 0.0, 0
    for i in range(0, N, chunk):
        d = np.linalg.norm(
            P_est[i:i + chunk, None, :] - P_gt[None, :, :], axis=2)
        d_sum += d.min(axis=1).sum()
        d_cnt += d.shape[0]
    return float(d_sum / d_cnt)


def ar_recall(errors: np.ndarray, diameter: float,
              thresholds=(0.05, 0.10, 0.15, 0.20)) -> dict:
    """BOP AR 协议: 误差 < threshold × 物体直径 的样本比例"""
    errors = np.asarray(errors, dtype=np.float64)
    out = {}
    for th in thresholds:
        out[f'AR@{int(th * 100)}%d'] = float((errors < th * diameter).mean())
    out['AR_mean'] = float(np.mean(list(out.values())))
    return out


def pose_errors(T_est: np.ndarray, T_gt: np.ndarray):
    """位姿误差: 旋转角(度), 平移差(与输入同单位)"""
    dR = T_est[:3, :3] @ T_gt[:3, :3].T
    rot_deg = float(np.degrees(np.linalg.norm(log_so3(dR))))
    trans = float(np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3]))
    return rot_deg, trans


# ============================================================
# 重投影误差 (像素)
# ============================================================

def reproj_rmse(model_pts: np.ndarray, K: np.ndarray,
                T_est: np.ndarray, T_gt: np.ndarray,
                dist: np.ndarray = None) -> float:
    """同一模型点在两位姿下投影的像素 RMS 差"""
    uv_e, z_e = project_pose(K, T_est, model_pts, dist)
    uv_g, z_g = project_pose(K, T_gt, model_pts, dist)
    vis = (z_e > 0) & (z_g > 0)
    if vis.sum() < 4:
        return float('nan')
    d = np.linalg.norm(uv_e[vis] - uv_g[vis], axis=1)
    return float(np.sqrt((d ** 2).mean()))


def project_center(K: np.ndarray, T: np.ndarray, center: np.ndarray):
    """模型中心投影到像素 (抖动度量用)"""
    c = np.asarray(center, dtype=np.float64).reshape(1, 3)
    uv, z = project_pose(K, T, c)
    return uv[0], float(z[0])


# ============================================================
# 时序稳定性
# ============================================================

def pixel_jitter_rms(centers_uv: np.ndarray) -> float:
    """像素抖动: 相邻帧投影中心位移的 RMS (理想静止=0)

    centers_uv: (T,2) 每帧模型中心像素
    适用: 场景近静止场景。运动场景请用 pixel_jitter_deviation。
    """
    centers_uv = np.asarray(centers_uv, dtype=np.float64)
    deltas = np.diff(centers_uv, axis=0)
    return float(np.sqrt((deltas ** 2).sum(axis=1).mean()))


def pixel_jitter_deviation(est_uv: np.ndarray, gt_uv: np.ndarray) -> float:
    """像素抖动偏差: 估计轨迹帧间位移与 GT 位移之差的 RMS (运动场景适用)

    jitter_dev = RMS_t( ||Δest(t) − Δgt(t)|| ),  Δp(t)=p(t+1)−p(t)
    物理意义: 叠加在真实运动上的高频噪声分量 (AR 漂浮/抖动的直接度量)
    """
    est_uv = np.asarray(est_uv, dtype=np.float64)
    gt_uv = np.asarray(gt_uv, dtype=np.float64)
    de = np.diff(est_uv, axis=0)
    dg = np.diff(gt_uv, axis=0)
    return float(np.sqrt(((de - dg) ** 2).sum(axis=1).mean()))


def pose_jitter_rms(poses) -> float:
    """位姿增量抖动: log(T(t+1)·T(t)⁻¹) 的标准差范数"""
    deltas = []
    for i in range(len(poses) - 1):
        deltas.append(log_se3(poses[i + 1] @ inverse_se3(poses[i])))
    return float(np.linalg.norm(np.std(np.array(deltas), axis=0)))


# ============================================================
# BOP 官方协议指标 (自实现版: MSSD / MSPD / VSD-lite)
# ============================================================

def _sym_variants(sym_rots, mu):
    from src.geometry import make_T
    if sym_rots is None:
        return None
    return [make_T(Rc, mu - Rc @ mu) for Rc in sym_rots]


def mssd(model_pts: np.ndarray, T_est: np.ndarray, T_gt: np.ndarray,
         diameter: float, sym_rots=None, n_sub: int = 2000,
         taus=None, chunk: int = 2048):
    """
    MSSD (Maximum Symmetry-aware Surface Distance, BOP 官方):
    e = min_{对称变体} max_x min_{x'} ||P_est(x) − P_gt'(x')||
    recall 阈值: 0.05-0.5 直径 (步长0.05), score = mean(recall)
    """
    from src.geometry import transform, make_T
    taus = np.linspace(0.05, 0.5, 10) if taus is None else taus
    mu = model_pts.mean(axis=0)
    if len(model_pts) > n_sub:
        sel = np.linspace(0, len(model_pts) - 1, n_sub).astype(int)
        V = model_pts[sel]
    else:
        V = model_pts
    P_est = transform(T_est, V)
    # 对称变体 (None 时仅恒等)
    variants = [np.eye(4)]
    if sym_rots is not None:
        variants = [make_T(Rc, mu - Rc @ mu) for Rc in sym_rots]
    best_e = np.inf
    for Tv in variants:
        P_g = transform(T_gt @ Tv, V)
        # max over est of min dist
        dmax = 0.0
        for i in range(0, len(P_est), chunk):
            d = np.linalg.norm(
                P_est[i:i + chunk, None, :] - P_g[None, :, :], axis=2)
            dmax = max(dmax, float(d.min(axis=1).max()))
        best_e = min(best_e, dmax)
    recall = {f'R@{t:.2f}d': float(best_e < t * diameter) for t in taus}
    return best_e, float(np.mean(list(recall.values()))), recall


def mspd(model_pts: np.ndarray, K: np.ndarray, T_est: np.ndarray,
         T_gt: np.ndarray, sym_rots=None, n_sub: int = 2000,
         taus=None, chunk: int = 2048):
    """
    MSPD (Maximum Symmetry-aware Projection Distance, BOP 官方):
    e = min_{对称变体} max_x min_{x'} ||π(P_est x) − π(P_gt' x')||  (像素)
    recall 阈值: 2.5-25px (步长2.5), score = mean(recall)
    """
    from src.geometry import transform, project, make_T
    taus = np.arange(2.5, 26.0, 2.5) if taus is None else taus
    mu = model_pts.mean(axis=0)
    if len(model_pts) > n_sub:
        sel = np.linspace(0, len(model_pts) - 1, n_sub).astype(int)
        V = model_pts[sel]
    else:
        V = model_pts
    P_est = transform(T_est, V)
    uv_e, z_e = project(K, P_est)
    variants = [np.eye(4)]
    if sym_rots is not None:
        variants = [make_T(Rc, mu - Rc @ mu) for Rc in sym_rots]
    best_e = np.inf
    for Tv in variants:
        P_g = transform(T_gt @ Tv, V)
        uv_g, z_g = project(K, P_g)
        vis = (z_g > 0)
        if vis.sum() < 4:
            continue
        dmax = 0.0
        for i in range(0, len(uv_e), chunk):
            d = np.linalg.norm(
                uv_e[i:i + chunk, None, :] - uv_g[None, :, :], axis=2)
            dmax = max(dmax, float(d.min(axis=1).max()))
        best_e = min(best_e, dmax)
    recall = {f'R@{t:.1f}px': float(best_e < t) for t in taus}
    return best_e, float(np.mean(list(recall.values()))), recall


def vsd_lite(model_pts: np.ndarray, K: np.ndarray, T_est: np.ndarray,
             T_gt: np.ndarray, mask_obs: np.ndarray, depth_obs: np.ndarray,
             device: str = 'cuda', tau: float = 20.0,
             thetas=(0.3, 0.4, 0.5, 0.6, 0.7), n_sub: int = 8000):
    """
    VSD-lite (Visible Surface Discrepancy, BOP 官方的自实现近似):
    双位姿渲染深度 → 可见性 → 深度差异 > τ 的像素比例
    score = mean_θ (1 − min(err/θ, 1))
    注: BOP 官方用网格全分辨率渲染, 此处为点泼溅近似 (同渲染器双盲对比,
    相对比较公平; 与官方绝对值存在系统性差异, 文档注明)。
    """
    from src.pipeline import render_model
    if len(model_pts) > n_sub:
        sel = np.linspace(0, len(model_pts) - 1, n_sub).astype(int)
        V = model_pts[sel]
    else:
        V = model_pts
    white = np.ones((len(V), 3))
    H, W = depth_obs.shape
    r_gt = render_model(V, white, T_gt, K, H, W, radius=1, device=device)
    r_est = render_model(V, white, T_est, K, H, W, radius=1, device=device)
    Dg = r_gt['depth'].cpu().numpy()
    De = r_est['depth'].cpu().numpy()
    M = mask_obs > 0
    vis_g = (Dg > 0) & M
    vis_e = (De > 0) & M
    union = vis_g | vis_e
    if union.sum() < 20:
        return float('nan'), float('nan')
    # 像素误差: 双方可见的深度差>τ, 或单方缺失
    dz = np.abs(De - Dg)
    err_px = ((vis_g & vis_e) & (dz > tau)) | (vis_g ^ vis_e)
    err = float(err_px[union].mean())
    score = float(np.mean([1.0 - min(err / th, 1.0) for th in thetas]))
    return err, score

def occlusion_accuracy(occ_pred: np.ndarray, occ_gt: np.ndarray) -> float:
    """遮挡决策正确率: 预测可见性 vs 真值可见性"""
    occ_pred = np.asarray(occ_pred).astype(bool)
    occ_gt = np.asarray(occ_gt).astype(bool)
    return float((occ_pred == occ_gt).mean())
