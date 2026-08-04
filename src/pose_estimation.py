# -*- coding: utf-8 -*-
"""
位姿求解器体系 (刚性 RT)
========================

对应关系 → 求解器 → 鲁棒估计 三段式的完整实现:

  3D↔3D : kabsch_umeyama (闭式绝对定向, 含尺度)
          icp (trimmed 迭代最近点, 抗遮挡/外点)
          multistart_icp_register (24 规范旋转多起点粗配准 + 精化)
  2D↔3D : pnp_ransac (EPnP + RANSAC + LM 精化)

依赖: numpy, opencv(cv2), torch(NN 加速, CUDA 可选)
作者: ZCode · 2026-08-03
"""
import numpy as np
import cv2
import torch

from src.geometry import (make_T, transform, inverse_se3, log_se3,
                          canonical_rotations_24)


# ============================================================
# 3D-3D: Kabsch / Umeyama 绝对定向 (闭式解)
# ============================================================

def kabsch_umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = False):
    """
    给定 N>=3 对 3D 对应点, 闭式求解 dst ≈ s·R·src + t

    Args:
        src, dst:  (N,3) 对应点 (src: 模型系, dst: 场景/相机系)
        with_scale: 是否估计相似变换尺度 (Umeyama)
    Returns:
        R: (3,3), t: (3,), s: float (with_scale=False 时恒为1)
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    assert src.shape == dst.shape and src.shape[0] >= 3

    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    X = src - mu_s
    Y = dst - mu_d

    H = (X.T @ Y) / src.shape[0]
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T

    if with_scale:
        var_s = (X ** 2).sum() / src.shape[0]
        s = float(np.trace(np.diag(S) @ D) / max(var_s, 1e-12))
    else:
        s = 1.0
    t = mu_d - s * (R @ mu_s)
    return R, t, s


# ============================================================
# 最近邻 (torch, 分块 cdist, 防显存爆炸)
# ============================================================

def nearest_neighbors(src: np.ndarray, dst: np.ndarray,
                      device: str = 'cuda', chunk: int = 8192):
    """
    对 src 每点找 dst 中最近邻
    Returns: idx (N,) int64, dist (N,) float
    """
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    src_t = torch.as_tensor(src, dtype=torch.float32, device=device)
    dst_t = torch.as_tensor(dst, dtype=torch.float32, device=device)
    N = src_t.shape[0]
    best_idx = torch.empty(N, dtype=torch.long, device=device)
    best_dist = torch.empty(N, dtype=torch.float32, device=device)
    for i in range(0, N, chunk):
        d = torch.cdist(src_t[i:i + chunk], dst_t)  # (c, M)
        dist, idx = d.min(dim=1)
        best_idx[i:i + chunk] = idx
        best_dist[i:i + chunk] = dist
    return best_idx.cpu().numpy(), best_dist.cpu().numpy()


# ============================================================
# 3D-3D: Trimmed ICP
# ============================================================

def icp(src: np.ndarray, dst: np.ndarray, T_init: np.ndarray = None,
        max_iter: int = 60, tol: float = 1e-7, trim_ratio: float = 0.7,
        device: str = 'cuda', verbose: bool = False):
    """
    Trimmed ICP: 交替 ①最近邻对应 ②trimmed Kabsch 位姿更新

    Args:
        src: (N,3) 模型点 (模型系)
        dst: (M,3) 场景点 (相机系)
        T_init: 初始位姿 (默认单位阵)
        trim_ratio: 保留残差最小比例 (抗遮挡, 0.5~0.9)
    Returns:
        T: (4,4) 模型→相机位姿
        stats: dict(rmse, n_inliers, iters, converged, rmse_hist)
    """
    T = np.eye(4) if T_init is None else T_init.copy()
    N = src.shape[0]
    k_keep = max(8, int(trim_ratio * N))
    rmse_hist = []
    converged = False

    for it in range(max_iter):
        src_t = transform(T, src)
        idx, dist = nearest_neighbors(src_t, dst, device=device)
        # trimmed: 仅保留残差最小的 k_keep 对
        thr = np.partition(dist, min(k_keep, N - 1))[min(k_keep, N - 1)]
        inl = dist <= thr
        if inl.sum() < 4:
            break
        R, t, _ = kabsch_umeyama(src_t[inl], dst[idx[inl]], with_scale=False)
        T_new = make_T(R, t) @ T
        delta = np.linalg.norm(log_se3(T_new @ inverse_se3(T)))
        T = T_new
        rmse = float(np.sqrt(np.mean(dist[inl] ** 2)))
        rmse_hist.append(rmse)
        if verbose:
            print(f'  [ICP it={it}] rmse={rmse:.4f} inl={inl.sum()} delta={delta:.2e}')
        if delta < tol:
            converged = True
            break

    idx, dist = nearest_neighbors(transform(T, src), dst, device=device)
    thr = np.partition(dist, min(k_keep, N - 1))[min(k_keep, N - 1)]
    inl = dist <= thr
    stats = {
        'rmse': float(np.sqrt(np.mean(dist[inl] ** 2))),
        'n_inliers': int(inl.sum()),
        'iters': len(rmse_hist),
        'converged': converged,
        'rmse_hist': rmse_hist,
    }
    return T, stats


# ============================================================
# 3D-3D: 多起点粗配准 + 精化 (无初值场景)
# ============================================================

def multistart_icp_register(model_pts: np.ndarray, scene_pts: np.ndarray,
                            device: str = 'cuda', coarse_iter: int = 20,
                            fine_iter: int = 60, trim_coarse: float = 0.5,
                            trim_fine: float = 0.7):
    """
    无初始位姿的模型→场景配准:
      质心对齐 + 24 个规范旋转(立方体对称群)多起点 ICP → 取 trimmed RMSE 最小者 → 精化

    对称物体注意: 多解是固有歧义, 需用 ADD-S 指标评价。
    """
    mu_m = model_pts.mean(axis=0)
    mu_s = scene_pts.mean(axis=0)
    base_t = mu_s - mu_m  # 质心对齐平移

    best = {'rmse': np.inf, 'T': None}
    rots = canonical_rotations_24()
    for i in range(rots.shape[0]):
        R0 = rots[i]
        T0 = make_T(R0, mu_s - R0 @ mu_m)
        T, st = icp(model_pts, scene_pts, T_init=T0, max_iter=coarse_iter,
                    trim_ratio=trim_coarse, device=device)
        if st['rmse'] < best['rmse']:
            best = {'rmse': st['rmse'], 'T': T}

    T_fine, st_fine = icp(model_pts, scene_pts, T_init=best['T'],
                          max_iter=fine_iter, trim_ratio=trim_fine, device=device)
    st_fine['coarse_rmse'] = best['rmse']
    return T_fine, st_fine


# ============================================================
# 3D-3D: FPFH 特征 + RANSAC 全局配准 (特征对应, 抗对称翻转)
# ============================================================

def fpfh_ransac_register(model_pts: np.ndarray, scene_pts: np.ndarray,
                         voxel: float = None, max_iter: int = 200000,
                         confidence: float = 0.999):
    """
    FPFH 特征匹配 + RANSAC 的全局 3D-3D 配准 (无需初值)

    与多起点旋转 ICP 互补: FPFH 利用局部几何特征签名建立对应,
    对近似对称物体的朝向歧义更鲁棒 (仍非完全免疫)。

    Args:
        model_pts, scene_pts: (N,3),(M,3) 点云
        voxel: 特征/降采样体素尺度 (None → 按场景点间距自动估计)
    Returns:
        T: (4,4) 初始位姿, fitness, inlier_rmse
    """
    import open3d as o3d

    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(model_pts)
    dst = o3d.geometry.PointCloud()
    dst.points = o3d.utility.Vector3dVector(scene_pts)

    if voxel is None:
        bbox = dst.get_max_bound() - dst.get_min_bound()
        voxel = float(np.linalg.norm(bbox)) / 40.0
        voxel = max(voxel, 1e-3)

    src = src.voxel_down_sample(voxel)
    dst = dst.voxel_down_sample(voxel)
    for pcd in (src, dst):
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30))
    fsrc = o3d.pipelines.registration.compute_fpfh_feature(
        src, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100))
    fdst = o3d.pipelines.registration.compute_fpfh_feature(
        dst, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100))

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src, dst, fsrc, fdst, mutual_filter=True,
        max_correspondence_distance=voxel * 1.5,
        estimation_method=o3d.pipelines.registration.
        TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel * 1.5),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            max_iter, confidence))
    return np.asarray(result.transformation), float(result.fitness), \
        float(result.inlier_rmse)


# ============================================================
# 2D-3D: PnP + RANSAC
# ============================================================

def pnp_ransac(obj_pts: np.ndarray, img_pts: np.ndarray, K: np.ndarray,
               dist: np.ndarray = None, reproj_thresh: float = 3.0,
               iterations: int = 2000, confidence: float = 0.999):
    """
    2D-3D 对应 → 位姿: EPnP+RANSAC 粗解 → LM 精化

    Args:
        obj_pts: (N,3) 模型系点
        img_pts: (N,2) 对应像素
        K: 3x3 内参
        dist: 畸变系数 (可选)
    Returns:
        T: (4,4) 位姿 或 None(失败)
        stats: dict(n_inliers, inlier_ratio, reproj_err)
    """
    obj = np.ascontiguousarray(obj_pts, dtype=np.float64).reshape(-1, 3)
    img = np.ascontiguousarray(img_pts, dtype=np.float64).reshape(-1, 2)
    if obj.shape[0] < 4:
        return None, {'error': 'need >= 4 correspondences'}

    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj, img, K, dist,
        iterationsCount=iterations, reprojectionError=float(reproj_thresh),
        confidence=confidence, flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers) < 4:
        return None, {'error': 'ransac failed', 'n_inliers': 0}

    inl = inliers.ravel()
    rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl], K, dist, rvec, tvec)

    R, _ = cv2.Rodrigues(rvec)
    T = make_T(R, tvec.reshape(3))

    proj, _ = cv2.projectPoints(obj[inl], rvec, tvec, K, dist)
    err = np.linalg.norm(proj.reshape(-1, 2) - img[inl], axis=1)
    stats = {
        'n_inliers': int(len(inl)),
        'inlier_ratio': float(len(inl) / obj.shape[0]),
        'reproj_err': float(err.mean()),
    }
    return T, stats
