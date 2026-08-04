# -*- coding: utf-8 -*-
"""
中心线/管腔骨架结构对应 (E1.5) —— 内镜 CT-视频注册的结构化解法
==================================================================

动机 (docs/09 §3): 单目深度对应在内镜不可靠 → 改用**解剖结构对应**:
  CT 气道中心线 (3D 骨架)  ←→  视频帧管腔暗区骨架 (2D)

对应链:
  CT 气道掩码 → 3D 骨架化 → 中心线点 (mm)
  视频帧 → 暗区分割 → 形态学清理 → 2D 骨架化 → 骨架像素
  位姿优化: min_T Σ_i DT_skel( π(K·T·X_cl,i) )   (距离变换, 投影数据关联)
  → 投影中心线拟合观测骨架 → T (内窥镜视角下结构对齐)

依赖: numpy, cv2, skimage, torch
作者: ZCode · 2026-08-04
"""
import cv2
import numpy as np
import torch

from src.geometry import make_T
from src.nonrigid_rtw import rodrigues_torch


# ============================================================
# CT 气道中心线 (3D 骨架)
# ============================================================

def airway_centerline_3d(mask: np.ndarray, volume=None,
                         min_branch_pts: int = 20,
                         return_junctions: bool = False):
    """
    气道掩码 (Z,Y,X bool) → 3D 骨架点 (物理 mm, 若给 volume 则换算)
    skimage skeletonize(method='lee') 3D 骨架化

    return_junctions=True 时额外返回分叉点/端点在输出点集中的索引。
    """
    from skimage.morphology import skeletonize
    from scipy import ndimage
    skel = skeletonize(mask.astype(np.uint8), method='lee')
    ijk = np.argwhere(skel > 0)
    if len(ijk) < min_branch_pts:
        ijk = np.argwhere(mask)
        if len(ijk) == 0:
            return np.zeros((0, 3))
    if return_junctions:
        sk = (skel > 0).astype(np.uint8)
        ker = np.ones((3, 3, 3), np.uint8)
        ker[1, 1, 1] = 0
        deg = ndimage.convolve(sk, ker, mode='constant') * sk
        deg_map = {tuple(v): d for v, d in zip(map(tuple, ijk),
                                               deg[skel > 0].ravel())}
        junc_idx = np.array([i for i, v in enumerate(map(tuple, ijk))
                             if deg_map[v] >= 3], dtype=np.int64)
        end_idx = np.array([i for i, v in enumerate(map(tuple, ijk))
                            if deg_map[v] == 1], dtype=np.int64)
    pts = ijk.astype(np.float64)
    if volume is not None:
        pts = volume.origin + pts * volume.spacing
    if return_junctions:
        return pts, junc_idx, end_idx
    return pts


def centerline_graph(pts: np.ndarray, radius: float):
    """中心线点的邻域图 (备用: 分叉点/端点分析)"""
    from src.pose_estimation import nearest_neighbors
    _, d = nearest_neighbors(pts, pts, device='cpu')
    return d


# ============================================================
# 图像管腔骨架 (2D)
# ============================================================

def lumen_skeleton_2d(bgr: np.ndarray, dark_pct: float = 25,
                      min_area: int = 200, depth: np.ndarray = None,
                      fov: bool = True):
    """
    视频帧 → 管腔区域 → 2D 骨架

    两种管腔判据:
      depth 给定: 深部区域 (depth > 高分位) = 管腔深处 (合成/有深度)
      否则:       圆形视野内暗区 (真实内镜, 需排除暗角晕影)

    Returns:
        skel_mask: (H,W) bool 骨架像素
        dist_transform: (H,W) float 骨架距离变换 (注册损失用)
        dark_mask: (H,W) bool 管腔区域
    """
    from skimage.morphology import skeletonize
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    if depth is not None:
        valid = depth > 0
        if valid.sum() < 100:
            dark = np.zeros((H, W), np.uint8)
        else:
            thr = np.percentile(depth[valid], 100 - dark_pct)
            dark = ((depth >= thr) & valid).astype(np.uint8)
    else:
        if fov:  # 圆形视野 (排除内镜暗角)
            yy, xx = np.mgrid[0:H, 0:W]
            r = np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2)
            mask_fov = (r < min(H, W) / 2 * 0.9)
        else:
            mask_fov = np.ones((H, W), bool)
        vals = gray[mask_fov]
        thr = max(float(np.percentile(vals, dark_pct)), 8.0)
        dark = ((gray < thr) & mask_fov).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN,
                            np.ones((3, 3), np.uint8), iterations=2)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE,
                            np.ones((5, 5), np.uint8), iterations=1)
    # 面积过滤: 保留大连通域 (管腔主体)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    keep = np.zeros_like(dark)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[lab == i] = 1
    if keep.sum() < 50:
        keep = dark
    skel = skeletonize(keep > 0).astype(np.uint8)
    # 距离变换: 距骨架的距离 (注册损失用)
    dt = cv2.distanceTransform((skel == 0).astype(np.uint8),
                               cv2.DIST_L2, 5)
    return skel > 0, dt, keep > 0


# ============================================================
# 分叉点 (junction) 检测 —— 分支匹配的锚点
# ============================================================

def skeleton_junctions_2d(skel_mask: np.ndarray):
    """
    2D 骨架的分叉点/端点检测 (8邻域度数)
    Returns: junction_idx (J,2) 像素坐标(v,u), endpoint_idx (E,2)
    """
    sk = skel_mask.astype(np.uint8)
    ker = np.ones((3, 3), np.uint8)
    ker[1, 1] = 0
    deg = cv2.filter2D(sk, -1, ker) * sk
    jv, ju = np.nonzero(deg >= 3)
    ev, eu = np.nonzero((deg == 1) & (sk > 0))
    return np.stack([jv, ju], 1), np.stack([ev, eu], 1)


def centerline_junctions_3d(skel_mask: np.ndarray, volume=None):
    """
    3D 骨架分叉点/端点 (26邻域度数, 体素空间)
    Returns: (cl_idx 相对 airway_centerline_3d 输出的体素索引集合)
    """
    from scipy import ndimage
    sk = skel_mask.astype(np.uint8)
    ker = np.ones((3, 3, 3), np.uint8)
    ker[1, 1, 1] = 0
    deg = ndimage.convolve(sk, ker, mode='constant') * sk
    junc = np.argwhere(deg >= 3)
    ends = np.argwhere((deg == 1) & (sk > 0))
    return junc, ends


def crop_centerline_by_view(cl_pts: np.ndarray, T_init: np.ndarray,
                            K: np.ndarray, H: int, W: int,
                            margin_px: int = 30, z_range=(3.0, 400.0)):
    """
    视角裁剪: 仅保留在初始位姿下投影进入图像(含边距)且深度合理的
    中心线点 —— 消除长气道树投影自重叠 (Part B 关键升级)
    """
    from src.geometry import transform, project
    P = transform(T_init, cl_pts)
    uv, z = project(K, P)
    inb = (z > z_range[0]) & (z < z_range[1]) & \
          (uv[:, 0] > -margin_px) & (uv[:, 0] < W + margin_px) & \
          (uv[:, 1] > -margin_px) & (uv[:, 1] < H + margin_px)
    return cl_pts[inb], np.nonzero(inb)[0]


# ============================================================
# 骨架对应位姿优化 (距离变换, 可微)
# ============================================================

def register_centerline(cl_pts_3d: np.ndarray, dist_transform: np.ndarray,
                        K: np.ndarray, T_init: np.ndarray,
                        iters: int = 200, lr: float = 5e-3,
                        n_sub: int = 1500, device: str = 'cuda',
                        cl_junction_idx: np.ndarray = None,
                        lambda_branch: float = 3.0,
                        verbose: bool = False):
    """
    投影中心线 ↔ 观测骨架 的位姿优化 (支持分叉点加权)

    min_{R,t}  Σ_i DT( π(K·T·X_cl,i) ) + λ_branch·Σ_{j∈分叉} DT( π(...) )
    分叉点(分支锚点)的高权重约束分支结构对齐 (解决普通chamfer下
    分支互换/重叠的问题)。

    Args:
        cl_junction_idx: 分叉点在 cl_pts_3d 中的索引 (可选)
        lambda_branch: 分叉项权重倍数
    Returns:
        dict: T(4,4), losses, init_loss, final_loss
    """
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    rng = np.random.default_rng(0)
    if len(cl_pts_3d) > n_sub:
        sel = rng.choice(len(cl_pts_3d), n_sub, replace=False)
        V = cl_pts_3d[sel]
    else:
        V = cl_pts_3d
    Vt = torch.as_tensor(V, dtype=torch.float32, device=device)
    DT = torch.as_tensor(dist_transform, dtype=torch.float32, device=device)
    Kt = torch.as_tensor(K, dtype=torch.float32, device=device)
    H, W = DT.shape

    from src.geometry import log_so3
    rv = torch.tensor(log_so3(T_init[:3, :3]), dtype=torch.float32,
                      device=device, requires_grad=True)
    tv = torch.tensor(T_init[:3, 3].copy(), dtype=torch.float32,
                      device=device, requires_grad=True)
    opt = torch.optim.Adam([rv, tv], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters)

    # 分叉点子集 (在采样后的 V 中定位)
    jidx_t = None
    if cl_junction_idx is not None and len(cl_junction_idx) > 0:
        if len(cl_pts_3d) > n_sub:
            jset = set(cl_junction_idx.tolist())
            j_local = [i for i, g in enumerate(sel) if g in jset]
        else:
            j_local = [int(j) for j in cl_junction_idx]
        if j_local:
            jidx_t = torch.tensor(j_local, dtype=torch.long, device=device)

    def _loss():
        R = rodrigues_torch(rv)
        P = Vt @ R.T + tv
        z = P[:, 2].clamp(min=1e-3)
        u = Kt[0, 0] * P[:, 0] / z + Kt[0, 2]
        v = Kt[1, 1] * P[:, 1] / z + Kt[1, 2]
        gx = 2.0 * u / (W - 1) - 1.0
        gy = 2.0 * v / (H - 1) - 1.0
        grid = torch.stack([gx, gy], -1).reshape(1, 1, -1, 2)
        dt_s = torch.nn.functional.grid_sample(
            DT.reshape(1, 1, H, W), grid, mode='bilinear',
            padding_mode='border', align_corners=True).reshape(-1)
        inb = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (P[:, 2] > 1e-3)
        if int(inb.sum()) < 10:
            return dt_s.mean() * 10.0, 0
        loss = dt_s[inb].mean()
        if jidx_t is not None and len(jidx_t) > 0:
            loss = loss + lambda_branch * dt_s[jidx_t].mean()
        return loss, int(inb.sum())

    losses = []
    with torch.no_grad():
        l0, _ = _loss()
        init_loss = float(l0)
    for it in range(iters):
        opt.zero_grad()
        loss, _ = _loss()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(float(loss.detach().cpu()))
        if verbose and it % 50 == 0:
            print(f'  [CL it={it}] chamfer={losses[-1]:.2f}px')
    R = rodrigues_torch(rv.detach()).cpu().numpy()
    T = make_T(R, tv.detach().cpu().numpy())
    return {'T': T, 'losses': losses, 'init_loss': init_loss,
            'final_loss': losses[-1]}
