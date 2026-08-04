# -*- coding: utf-8 -*-
"""
端到端注册 + AR 合成管线
========================

数据流 (对应 docs/01 §3.2 的精简可运行版):

    RGB-D 图像 ──反投影──▶ 场景点云 ──┐
                                      ├─多起点粗配准 + trimmed ICP──▶ T_obj^cam
    3D 模型 ──表面采样──▶ 模型点云 ──┘        (可选: RTW 非刚性精化)
                                      │
                                      ▼
                          点泼溅渲染 + 深度感知合成 ──▶ AR 融合图

接口与 FoundationPose 输出同构 (4x4 位姿), FP 权重就绪后可原位替换
"位姿引擎", 滤波/RTW/渲染/评测环节不动。

依赖: numpy, opencv, torch, open3d
作者: ZCode · 2026-08-03
"""
from dataclasses import dataclass, field
import numpy as np
import cv2
import torch

from src.geometry import backproject_depth, transform
from src.pose_estimation import (multistart_icp_register, icp,
                                 fpfh_ransac_register)
from src.rasterize import splat_render, composite_depth_aware
from src.nonrigid_rtw import rtw_optimize


# ============================================================
# 配置
# ============================================================

@dataclass
class RegConfig:
    device: str = 'cuda'
    n_model_icp: int = 3000       # ICP 使用的模型点 (子采样)
    n_scene_max: int = 20000      # 场景点上限
    trim_coarse: float = 0.5      # 粗配准截尾
    trim_fine: float = 0.7        # 精配准截尾
    coarse: str = 'best'          # 粗配准: 'fpfh' | 'multistart' | 'best'(双引擎取优)
    use_rtw: bool = False         # 是否启用非刚性 RTW 精化
    rtw_grid: tuple = (5, 5, 5)
    rtw_iters: int = 250
    rtw_lambda_mag: float = 1e-4
    rtw_lambda_sm: float = 1e-3
    seed: int = 0


@dataclass
class RenderConfig:
    radius: int = 1               # 泼溅半径
    alpha: float = 0.85           # 虚拟层不透明度
    base_rgb: tuple = (0.10, 0.85, 0.30)   # 模型基础色 (绿)
    light_dir: tuple = (0.4, 0.5, 1.0)     # 相机系方向光
    device: str = 'cuda'


# ============================================================
# 模型加载
# ============================================================

def load_mesh_points(path: str, n_points: int = 20000, seed: int = 0):
    """
    加载 3D 模型 (.obj/.ply/.stl) → 表面均匀采样点云 + 法线

    Returns:
        points:  (N,3) float64, 模型系
        normals: (N,3) float64 (单位向量)
    """
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(path)
    if len(mesh.triangles) > 0:
        pcd = mesh.sample_points_uniformly(number_of_points=n_points)
    else:  # 纯点云网格
        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices
        if len(pcd.points) > n_points:
            pcd = pcd.farthest_point_down_sample(n_points)
    if not pcd.has_normals():
        pcd.estimate_normals()
    pcd.normalize_normals()
    points = np.asarray(pcd.points, dtype=np.float64)
    normals = np.asarray(pcd.normals, dtype=np.float64)
    return points, normals


# ============================================================
# 核心: RGB-D 注册
# ============================================================

def register_rgbd(depth: np.ndarray, K: np.ndarray, model_pts: np.ndarray,
                  mask: np.ndarray = None, cfg: RegConfig = None):
    """
    单帧 RGB-D 注册: 深度反投影 → 多起点粗配准 → trimmed ICP 精化 → (可选) RTW

    Args:
        depth: (H,W) 深度 (单位与模型一致, 如均为 mm)
        K:     3x3 内参
        model_pts: (N,3) 模型点 (模型系)
        mask:  (H,W) 目标区域 (None=全图, 实际应用中由 SAM-2/检测器给出)
        cfg:   RegConfig
    Returns:
        dict: T(4,4), stats, scene_pts, rtw(可选)
    """
    cfg = cfg or RegConfig()
    rng = np.random.default_rng(cfg.seed)

    # 1) 深度反投影 → 场景点云 (3D-3D 化, docs/03 §3.1)
    scene_pts, _, _ = backproject_depth(depth, K, mask=mask)
    if scene_pts.shape[0] < 50:
        return {'T': None, 'stats': {'error': 'too few scene points',
                                     'n_scene': scene_pts.shape[0]}}
    if scene_pts.shape[0] > cfg.n_scene_max:
        sel = rng.choice(scene_pts.shape[0], cfg.n_scene_max, replace=False)
        scene_sub = scene_pts[sel]
    else:
        scene_sub = scene_pts

    # 2) 模型点子采样 (ICP 数据项)
    if model_pts.shape[0] > cfg.n_model_icp:
        sel_m = rng.choice(model_pts.shape[0], cfg.n_model_icp, replace=False)
        model_icp = model_pts[sel_m]
    else:
        model_icp = model_pts

    # 3) 粗配准 + trimmed ICP 精化
    def _fpfh_path():
        T0, fitness, fpfh_rmse = fpfh_ransac_register(model_icp, scene_sub)
        T_, st_ = icp(model_icp, scene_sub, T_init=T0,
                      max_iter=80, trim_ratio=cfg.trim_fine,
                      device=cfg.device)
        st_['coarse'] = 'fpfh'
        st_['fpfh_fitness'] = fitness
        st_['fpfh_rmse'] = fpfh_rmse
        return T_, st_

    def _multistart_path():
        T_, st_ = multistart_icp_register(
            model_icp, scene_sub, device=cfg.device,
            trim_coarse=cfg.trim_coarse, trim_fine=cfg.trim_fine)
        st_['coarse'] = 'multistart'
        return T_, st_

    if cfg.coarse == 'best':
        # 双引擎假设验证: FPFH 与多起点各自精化, 取 trimmed RMSE 更低者
        # (对称物体几何歧义的工程缓解; 根本解是 FoundationPose 外观感知)
        T_f, st_f = _fpfh_path()
        T_m, st_m = _multistart_path()
        if st_f['rmse'] <= st_m['rmse']:
            T, stats = T_f, st_f
        else:
            T, stats = T_m, st_m
        stats['dual_rmse'] = (st_f['rmse'], st_m['rmse'])
        stats['coarse'] = 'best:' + stats['coarse']
    elif cfg.coarse == 'fpfh':
        T, stats = _fpfh_path()
        if stats.get('fpfh_fitness', 1.0) < 0.05:
            T, stats = _multistart_path()
            stats['coarse'] = 'multistart(fallback)'
    else:
        T, stats = _multistart_path()

    out = {'T': T, 'stats': stats, 'scene_pts': scene_pts}

    # 4) 可选: RTW 非刚性精化 (docs/03 §5)
    if cfg.use_rtw:
        rtw = rtw_optimize(model_pts, scene_sub, T, grid=cfg.rtw_grid,
                           iters=cfg.rtw_iters,
                           lambda_mag=cfg.rtw_lambda_mag,
                           lambda_sm=cfg.rtw_lambda_sm,
                           device=cfg.device, seed=cfg.seed)
        out['rtw'] = rtw
        out['T'] = rtw['T']
    return out


def refine_icp(model_pts, scene_pts, T_init, max_iter=60, trim_ratio=0.7,
               device='cuda'):
    """对已初始化的位姿做 ICP 精化 (视频跟踪模式入口)"""
    return icp(model_pts, scene_pts, T_init=T_init, max_iter=max_iter,
               trim_ratio=trim_ratio, device=device)


# ============================================================
# 无渲染深度一致性评分 (多假设仲裁/对称翻转裁决)
# ============================================================

def depth_consistency_score(T: np.ndarray, model_pts: np.ndarray,
                            depth: np.ndarray, K: np.ndarray,
                            mask: np.ndarray, tol: float = 5.0):
    """
    位姿假设评分 (无渲染, ~5ms): 稠密模型点经 T 投影,
    返回 (内点率, mask覆盖率): 内点 = 落在mask内且|z - depth| < tol

    实测判别力: GT≈0.15-0.33 vs 对称翻转≈0.08-0.15 (BOP ape 6/6 正确裁决),
    远优于低分辨率渲染验证 —— 真实深度含全部细节信息。
    """
    P = transform(T, model_pts)
    z = P[:, 2]
    H, W = depth.shape
    u = np.round(K[0, 0] * P[:, 0] / np.maximum(z, 1e-9) + K[0, 2]).astype(int)
    v = np.round(K[1, 1] * P[:, 1] / np.maximum(z, 1e-9) + K[1, 2]).astype(int)
    ok = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z = u[ok], v[ok], z[ok]
    if len(z) < 10:
        return 0.0, 0.0
    in_mask = mask[v, u] > 0
    if in_mask.sum() < 10:
        return 0.0, 0.0
    dz = np.abs(depth[v[in_mask], u[in_mask]] - z[in_mask])
    return float((dz < tol).mean()), float(in_mask.mean())


def contour_chamfer_score(T: np.ndarray, model_pts: np.ndarray,
                          mask_obs: np.ndarray, K: np.ndarray,
                          device: str = 'cuda'):
    """
    轮廓形状评分: 渲染 mask 与观测 mask 的轮廓 chamfer (px) + IoU

    轮廓编码物体朝向语义 (如 ape 耳/面部突起方向), 对称翻转后轮廓必然
    不同 —— 实测判别力 9/9 (vs 深度一致性 5/9), 是对称翻转的最强经典信号。
    Returns: (chamfer_px, iou)
    """
    import cv2 as _cv2
    H, W = mask_obs.shape
    white = np.ones((len(model_pts), 3))
    out = render_model(model_pts, white, T, K, H, W, radius=1,
                       device=device)
    mh = out['mask'].cpu().numpy().astype(np.uint8)
    mo = (mask_obs > 0).astype(np.uint8)

    def _contour(m):
        e = _cv2.morphologyEx(m, _cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        return np.nonzero(e)
    ch, co = _contour(mh), _contour(mo)
    inter = np.logical_and(mh, mo).sum()
    union = np.logical_or(mh, mo).sum()
    iou = float(inter / max(union, 1))
    if len(ch[0]) < 10 or len(co[0]) < 10:
        return 1e3, iou
    dt_o = _cv2.distanceTransform((mo == 0).astype(np.uint8),
                                  _cv2.DIST_L2, 3)
    dt_h = _cv2.distanceTransform((mh == 0).astype(np.uint8),
                                  _cv2.DIST_L2, 3)
    chamfer = float((dt_o[ch].mean() + dt_h[co].mean()) / 2.0)
    return chamfer, iou


def arbitrate_poses(T_cands, model_pts, depth, K, mask, tol=5.0,
                    device='cuda', use_contour=True):
    """
    多候选位姿仲裁: 轮廓形状(主) + 深度一致性(辅)

    score = iou + max(0, 1 − chamfer/3px) + 0.2·深度内点率
    """
    best, best_s, scores = None, -1.0, []
    for T in T_cands:
        s, cov = depth_consistency_score(T, model_pts, depth, K, mask,
                                         tol=tol)
        cham, iou = (contour_chamfer_score(T, model_pts, mask, K,
                                           device=device)
                     if use_contour else (0.0, s))
        v = iou + max(0.0, 1.0 - cham / 3.0) + 0.2 * s
        scores.append({'score': v, 'depth': s, 'coverage': cov,
                       'chamfer': cham, 'iou': iou})
        if v > best_s:
            best_s, best = v, T
    return best, best_s, scores


# ============================================================
# 渲染与 AR 合成
# ============================================================

def lambert_colors(normals_obj: np.ndarray, R: np.ndarray,
                   base_rgb=(0.1, 0.85, 0.3), light_dir=(0.4, 0.5, 1.0)):
    """简单 Lambert 着色: 模型法线旋转到相机系, 与方向光点积"""
    n_cam = normals_obj @ R.T
    L = np.asarray(light_dir, dtype=np.float64)
    L = L / np.linalg.norm(L)
    inten = 0.35 + 0.65 * np.clip(n_cam @ L, 0.0, 1.0)
    base = np.asarray(base_rgb, dtype=np.float64).reshape(1, 3)
    return inten.reshape(-1, 1) * base


def render_model(model_pts: np.ndarray, colors: np.ndarray, T: np.ndarray,
                 K: np.ndarray, H: int, W: int, radius: int = 1,
                 device: str = 'cuda'):
    """模型点经位姿渲染为 RGB+深度 (torch 泼溅)"""
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    P = torch.as_tensor(transform(T, model_pts), dtype=torch.float32, device=device)
    Ft = torch.as_tensor(colors, dtype=torch.float32, device=device)
    Kt = torch.as_tensor(K, dtype=torch.float32, device=device)
    return splat_render(P, Ft, Kt, H, W, radius=radius)


def composite_ar(image_rgb: np.ndarray, depth: np.ndarray, render_out: dict,
                 alpha: float = 0.85):
    """
    深度感知 AR 合成
    Args:
        image_rgb: (H,W,3) float[0,1]
        depth:     (H,W) 真实深度 (0=无效)
        render_out: render_model 的输出
    Returns:
        composite: (H,W,3) float[0,1]
        occ_mask:  (H,W) bool 虚拟可见像素
    """
    device = render_out['feat'].device
    real = torch.as_tensor(image_rgb, dtype=torch.float32, device=device)
    rd = torch.as_tensor(depth, dtype=torch.float32, device=device)
    comp, occ = composite_depth_aware(real, rd, render_out['feat'],
                                      render_out['depth'], alpha=alpha)
    return comp.cpu().numpy(), occ.cpu().numpy()
