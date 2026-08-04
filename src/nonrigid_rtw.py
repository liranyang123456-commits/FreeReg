# -*- coding: utf-8 -*-
"""
RTW: 刚性位姿 + 非刚性形变场 联合估计 (CUDA)
============================================

数学模型 (docs/03 §5):

    T_RTW(p) = R·( p + W(p; δ) ) + t

    W(p; δ) = Σ_j w_j(p) · δ_j     FFD 自由形变: 三线性控制格插值

目标函数:

    min  Chamfer_trim( R·(V+W·δ)+t,  观测点云 )          ← 数据项
       + λ_mag · mean(|δ|²)                               ← 形变量正则
       + λ_sm  · Laplacian(δ)                             ← 形变平滑正则

求解: torch 自动微分 + Adam (ICP 式交替最近邻, 距离项可微)。

与 SOTA 的关系: 本模块是 Embedded Deformation (Sumner'07) / 4D-GS 形变场
的可解释轻量实现, 参数语义一致, 可原位升级为神经形变场。

依赖: torch (CUDA 可选), numpy
作者: ZCode · 2026-08-03
"""
import numpy as np
import torch

from src.geometry import log_so3, make_T


# ============================================================
# torch 版 Rodrigues
# ============================================================

def rodrigues_torch(rv: torch.Tensor) -> torch.Tensor:
    """旋转向量 (3,) → 旋转矩阵 (3,3), 可微

    注意: K 按叉乘构造即反对称, 切勿再 K-Kᵀ (会得 2K, 破坏正交性 —
    曾致 det(R)≈49 的灾难性缩放, 2026-08-04 定案修复)
    """
    theta = torch.norm(rv) + 1e-12
    k = rv / theta
    K = torch.zeros(3, 3, dtype=rv.dtype, device=rv.device)
    K[0, 1], K[0, 2] = -k[2], k[1]
    K[1, 0], K[1, 2] = k[2], -k[0]
    K[2, 0], K[2, 1] = -k[1], k[0]
    I = torch.eye(3, dtype=rv.dtype, device=rv.device)
    R = I + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)
    return R


# ============================================================
# FFD 自由形变控制格 (三线性插值)
# ============================================================

class FFDLattice:
    """自由形变控制格: 模型包围盒内的均匀控制点网格

    顶点形变 = 控制点偏移的三线性加权:  W(p) = Σ_j w_j(p)·δ_j
    权重 w 在形变前一次性计算 (顶点相对控制格位置固定 → 线性混合蒙皮)
    """

    def __init__(self, bbox_min, bbox_max, grid=(5, 5, 5), device='cuda',
                 margin=0.1):
        self.grid = tuple(grid)
        self.device = device
        bbox_min = np.asarray(bbox_min, dtype=np.float64)
        bbox_max = np.asarray(bbox_max, dtype=np.float64)
        span = bbox_max - bbox_min
        self.bbox_min = bbox_min - margin * span
        self.bbox_max = bbox_max + margin * span
        self.step = (self.bbox_max - self.bbox_min) / (np.array(self.grid) - 1)
        self.n_ctrl = int(np.prod(self.grid))
        # 控制点三维索引 → 邻居对 (平滑正则用)
        self._edges = self._build_edges()

    def _build_edges(self):
        gx, gy, gz = self.grid
        edges = []

        def cid(i, j, k):
            return (i * gy + j) * gz + k
        for i in range(gx):
            for j in range(gy):
                for k in range(gz):
                    if i + 1 < gx:
                        edges.append((cid(i, j, k), cid(i + 1, j, k)))
                    if j + 1 < gy:
                        edges.append((cid(i, j, k), cid(i, j + 1, k)))
                    if k + 1 < gz:
                        edges.append((cid(i, j, k), cid(i, j, k + 1)))
        return torch.tensor(edges, dtype=torch.long, device=self.device)

    def weights(self, V: np.ndarray) -> torch.Tensor:
        """顶点 (N,3) → 三线性权重 (N, M), 行和=1"""
        Vt = torch.as_tensor(V, dtype=torch.float32, device=self.device)
        bmin = torch.as_tensor(self.bbox_min, dtype=torch.float32, device=self.device)
        step = torch.as_tensor(self.step, dtype=torch.float32, device=self.device)
        grid = torch.tensor(self.grid, dtype=torch.long, device=self.device)

        g = (Vt - bmin) / step                       # (N,3) 控制格坐标
        g0 = torch.floor(g).long()
        frac = g - g0.float()
        g0 = torch.clamp(g0, torch.zeros_like(grid), grid - 2)
        g1 = g0 + 1

        N = Vt.shape[0]
        M = self.n_ctrl
        W = torch.zeros(N, M, dtype=torch.float32, device=self.device)
        gy, gz = self.grid[1], self.grid[2]
        for ci in range(8):
            ii = g0[:, 0] if (ci & 1) == 0 else g1[:, 0]
            jj = g0[:, 1] if (ci & 2) == 0 else g1[:, 1]
            kk = g0[:, 2] if (ci & 4) == 0 else g1[:, 2]
            wx = frac[:, 0] if ci & 1 else (1 - frac[:, 0])
            wy = frac[:, 1] if ci & 2 else (1 - frac[:, 1])
            wz = frac[:, 2] if ci & 4 else (1 - frac[:, 2])
            cid = (ii * gy + jj) * gz + kk
            W.scatter_add_(1, cid.unsqueeze(1), (wx * wy * wz).unsqueeze(1))
        # 显式归一化: 行和恒为1 (边界钳位下权重失真的保险,
        # real_colon 稀疏薄片边界效应定案的直接修复)
        W = W / W.sum(dim=1, keepdim=True).clamp(min=1e-9)
        return W  # (N, M)

    def laplacian(self, delta: torch.Tensor) -> torch.Tensor:
        """控制格相邻偏移差平方均值 (平滑正则)"""
        e = self._edges
        return ((delta[e[:, 0]] - delta[e[:, 1]]) ** 2).sum(-1).mean()


# ============================================================
# 分块最近邻 (索引不可微, 距离可微)
# ============================================================

def _nn_indices(src_t: torch.Tensor, dst_t: torch.Tensor, chunk=8192):
    """src 每点在 dst 中的最近邻索引 (no_grad)"""
    N = src_t.shape[0]
    idx = torch.empty(N, dtype=torch.long, device=src_t.device)
    with torch.no_grad():
        for i in range(0, N, chunk):
            d = torch.cdist(src_t[i:i + chunk], dst_t)
            idx[i:i + chunk] = d.argmin(dim=1)
    return idx


# ============================================================
# RTW 联合优化
# ============================================================

def rtw_optimize(model_pts: np.ndarray, scene_pts: np.ndarray,
                 T_init: np.ndarray, grid=(5, 5, 5), iters: int = 300,
                 lr: float = 1e-2, lambda_mag: float = 1e-4,
                 lambda_sm: float = 1e-3, trim_ratio: float = 0.7,
                 n_data: int = 3000, device: str = 'cuda', seed: int = 0,
                 verbose: bool = False):
    """
    刚性位姿 + FFD 形变场联合优化

    Args:
        model_pts: (N,3) 模型顶点 (模型系)
        scene_pts: (M,3) 观测场景点 (相机系)
        T_init:    初始刚性位姿 (建议由 ICP 提供)
        grid:      FFD 控制格分辨率
        iters/lr:  Adam 优化
        lambda_mag/lambda_sm: 形变量/平滑正则
        trim_ratio: 数据项截尾比例 (抗遮挡)
        n_data:    参与数据项的模型点子采样数
    Returns:
        dict: T(4,4), delta(M,3), lattice, losses, V_def_data, rmse
    """
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # 数据子采样
    if model_pts.shape[0] > n_data:
        sel = rng.choice(model_pts.shape[0], n_data, replace=False)
        V_data = model_pts[sel]
    else:
        V_data = model_pts
    V_all_t = torch.as_tensor(model_pts, dtype=torch.float32, device=device)
    Vd_t = torch.as_tensor(V_data, dtype=torch.float32, device=device)
    S_t = torch.as_tensor(scene_pts, dtype=torch.float32, device=device)

    # FFD 控制格 (建在模型包围盒上)
    lat = FFDLattice(model_pts.min(axis=0), model_pts.max(axis=0),
                     grid=grid, device=device)
    W_data = lat.weights(V_data)       # (n_data, M)

    # 优化变量
    rv = torch.tensor(log_so3(T_init[:3, :3]), dtype=torch.float32,
                      device=device, requires_grad=True)
    tv = torch.tensor(T_init[:3, 3].copy(), dtype=torch.float32,
                      device=device, requires_grad=True)
    delta = torch.zeros(lat.n_ctrl, 3, dtype=torch.float32,
                        device=device, requires_grad=True)

    opt = torch.optim.Adam([rv, tv, delta], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters)

    k_keep = max(8, int(trim_ratio * Vd_t.shape[0]))
    losses = []
    for it in range(iters):
        opt.zero_grad()
        R = rodrigues_torch(rv)
        V_def = Vd_t + W_data @ delta            # 模型系内形变
        V_cam = V_def @ R.T + tv                 # 刚体变换到相机系

        # 数据项: 截尾 Chamfer (pred→obs)
        idx = _nn_indices(V_cam, S_t)
        d = torch.norm(V_cam - S_t[idx], dim=1)
        d_trim = torch.topk(d, k_keep, largest=False).values.mean()

        # 正则项
        reg_mag = (delta ** 2).sum(-1).mean()
        reg_sm = lat.laplacian(delta)

        loss = d_trim + lambda_mag * reg_mag + lambda_sm * reg_sm
        loss.backward()
        opt.step()
        sched.step()
        losses.append(float(loss.detach().cpu()))
        if verbose and (it % 50 == 0 or it == iters - 1):
            print(f'  [RTW it={it}] loss={losses[-1]:.4f} data={float(d_trim):.4f} '
                  f'|δ|_mean={float(delta.detach().norm(dim=1).mean()):.4f}')

    # 输出
    R = rodrigues_torch(rv.detach()).cpu().numpy()
    T = make_T(R, tv.detach().cpu().numpy())
    delta_np = delta.detach().cpu().numpy()

    # 全顶点形变 (供渲染/评估)
    W_all = lat.weights(model_pts)
    V_def_all = (V_all_t + W_all @ delta.detach()).cpu().numpy()

    # 最终截尾 RMSE
    with torch.no_grad():
        Vc = torch.as_tensor(V_def_all, dtype=torch.float32, device=device) \
            @ torch.as_tensor(R, dtype=torch.float32, device=device).T \
            + torch.as_tensor(T[:3, 3], dtype=torch.float32, device=device)
        idx = _nn_indices(Vc, S_t)
        d = torch.norm(Vc - S_t[idx], dim=1)
        k2 = max(8, int(trim_ratio * d.shape[0]))
        rmse = float(torch.topk(d, k2, largest=False).values.mean().cpu())

    return {
        'T': T, 'delta': delta_np, 'lattice': lat, 'losses': losses,
        'V_def': V_def_all, 'rmse': rmse,
    }


def apply_deformation(V_new: np.ndarray, lattice: FFDLattice,
                      delta: np.ndarray) -> np.ndarray:
    """把学好的 FFD 形变场应用到新点集 (如稠密渲染点云)"""
    W = lattice.weights(V_new)
    d_t = torch.as_tensor(delta, dtype=torch.float32, device=W.device)
    V_t = torch.as_tensor(V_new, dtype=torch.float32, device=W.device)
    return (V_t + W @ d_t).cpu().numpy()


# ============================================================
# E2: 层次化控制格 + 区域聚焦 分层优化 (新算法族)
# ============================================================

def _residual_map(model_pts, T, lattice, delta, depth_img, K, device):
    """由当前 (T,δ) 计算逐像素残差图 (供区域聚焦)"""
    V = torch.as_tensor(model_pts, dtype=torch.float32, device=device)
    W = lattice.weights(model_pts)
    Vd = V + W @ torch.as_tensor(delta, dtype=torch.float32, device=device)
    R = torch.as_tensor(T[:3, :3], dtype=torch.float32, device=device)
    t = torch.as_tensor(T[:3, 3], dtype=torch.float32, device=device)
    Vc = Vd @ R.T + t
    z = Vc[:, 2].clamp(min=1e-6)
    Kt = torch.as_tensor(K, dtype=torch.float32, device=device)
    u = Kt[0, 0] * Vc[:, 0] / z + Kt[0, 2]
    v = Kt[1, 1] * Vc[:, 1] / z + Kt[1, 2]
    H, W = depth_img.shape
    gx = 2.0 * u / (W - 1) - 1.0
    gy = 2.0 * v / (H - 1) - 1.0
    grid_t = torch.stack([gx, gy], -1).reshape(1, 1, -1, 2)
    D = torch.as_tensor(depth_img, dtype=torch.float32, device=device)
    z_obs = torch.nn.functional.grid_sample(
        D.reshape(1, 1, H, W), grid_t, mode='bilinear',
        padding_mode='zeros', align_corners=True).reshape(-1)
    res = (z_obs - z).abs()
    ok = (z_obs > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    pix = (torch.round(v).long() * W + torch.round(u).long()).clamp(0, H * W - 1)
    rmap = torch.zeros(H * W, device=device)
    rmap.scatter_reduce_(0, pix[ok], res[ok], reduce='amax')
    return rmap.reshape(H, W)


def rtw_optimize_region_zoom(model_pts: np.ndarray, depth_img: np.ndarray,
                             K: np.ndarray, T_init: np.ndarray,
                             center_px, zoom: float = 3.0,
                             grid=(8, 8, 8), device: str = 'cuda',
                             verbose: bool = False, **kwargs):
    """
    E2 区域缩放重采样 (解决局部形变落样不足):

    以 center_px 为中心裁出 (W/zoom × H/zoom) 区域并上采样回原图尺寸,
    同步缩放内参 (深度值/mm 量纲不变) —— 局部特征获得 zoom× 像素与
    zoom× 模型落样, 梯度在凹痕尺度上良定义。
    位姿/形变参数为坐标系不变量, 可直接用于全图评测。
    """
    import cv2 as _cv2
    H, W = depth_img.shape
    u0, v0 = center_px
    w0, h0 = max(16, int(W / zoom)), max(16, int(H / zoom))
    x0 = int(np.clip(u0 - w0 // 2, 0, W - w0))
    y0 = int(np.clip(v0 - h0 // 2, 0, H - h0))
    crop = depth_img[y0:y0 + h0, x0:x0 + w0]
    s = min(H / h0, W / w0)
    D2 = _cv2.resize(crop, (W, H), interpolation=_cv2.INTER_NEAREST)
    # 缩放内参 (深度量纲不变, 像素映射一致)
    K2 = K.copy().astype(np.float64)
    K2[0, 0] = K[0, 0] * s
    K2[1, 1] = K[1, 1] * s
    K2[0, 2] = (K[0, 2] - x0) * s
    K2[1, 2] = (K[1, 2] - y0) * s
    if verbose:
        print(f'  [E2-zoom] 裁剪=({x0},{y0},{w0}x{h0}) 上采样×{s:.1f} →'
              f' fx={K2[0, 0]:.0f}')
    res = rtw_optimize_projective(model_pts, D2, K2, T_init, grid=grid,
                                  device=device, verbose=verbose, **kwargs)
    res['zoom'] = s
    res['crop'] = (x0, y0, w0, h0)
    return res


def rtw_optimize_hierarchical(model_pts: np.ndarray, depth_img: np.ndarray,
                              K: np.ndarray, T_init: np.ndarray,
                              grids=((4, 4, 4), (8, 8, 8)),
                              iters_per_level: int = 250,
                              focus_percentile: float = 70.0,
                              device: str = 'cuda', verbose: bool = False,
                              **kwargs):
    """
    E2 层次化规范分解优化 (新算法族):
      粗网格 (全局形变/位姿) → 层间残差图 → 区域聚焦 (高残差区)
      → 细网格 (局部特征, 如凹痕) 分层求解

    grids: 由粗到细的控制格序列
    focus_percentile: 区域聚焦的残差分位 (仅 top 区进入细层数据项)
    其余 kwargs 透传 rtw_optimize_projective (含 two_phase/edge_band/
    lambda_nrm/model_normals 等)
    """
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    T = T_init.copy()
    res = None
    for level, grid in enumerate(grids):
        focus_mask = None
        if level > 0 and res is not None:
            rmap = _residual_map(model_pts, res['T'], res['lattice'],
                                 res['delta'], depth_img, K, device)
            vals = rmap[rmap > 0]
            if len(vals) > 50:
                thr_focus = np.percentile(vals.cpu().numpy(),
                                          focus_percentile)
                focus_mask = (rmap > thr_focus).cpu().numpy()
                if verbose:
                    print(f'  [E2] level{level} grid={grid} 聚焦区='
                          f'{int(focus_mask.sum())}px (>{thr_focus:.3f})')
        res = rtw_optimize_projective(
            model_pts, depth_img, K, T, grid=grid,
            iters=iters_per_level, device=device, verbose=verbose,
            focus_mask=focus_mask, **kwargs)
        T = res['T']
        if verbose:
            print(f'  [E2] level{level} grid={grid} loss={res["losses"][-1]:.4f}')
    res['levels'] = len(grids)
    return res


# ============================================================
# 点对面 (point-to-plane) 观测面法线 —— 射线滑移的根本解
# ============================================================

def depth_to_points_normals(D: torch.Tensor, K: torch.Tensor):
    """
    深度图 → 点云与法线 (torch, 一次性预计算)

    N_obs(u,v) = normalize( cross( P(u+1,v)−P(u−1,v), P(u,v+1)−P(u,v−1) ) )
    朝向约定: n_z < 0 (面向相机, OpenCV 相机看 +z)
    Returns: P_obs (H,W,3), N_obs (H,W,3), valid (H,W) bool
    """
    H, W = D.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    vs, us = torch.meshgrid(torch.arange(H, device=D.device,
                                         dtype=torch.float32),
                            torch.arange(W, device=D.device,
                                         dtype=torch.float32),
                            indexing='ij')
    z = D
    x = (us - cx) * z / fx
    y = (vs - cy) * z / fy
    P = torch.stack([x, y, z], dim=-1)              # (H,W,3)
    # 中心差分切向
    Pu = torch.zeros_like(P)
    Pv = torch.zeros_like(P)
    Pu[:, 1:-1] = P[:, 2:] - P[:, :-2]
    Pv[1:-1, :] = P[2:, :] - P[:-2, :]
    N = torch.cross(Pu, Pv, dim=-1)
    n_norm = N.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    N = N / n_norm
    N = torch.where(N[..., 2:3] > 0, -N, N)         # 朝向相机 (n_z<0)
    valid = (D > 0) & (n_norm.squeeze(-1) > 1e-6)
    return P, N, valid


# ============================================================
# E2: 可观测性引导的 RTW (Observability-Guided RTW)
# ============================================================

def compute_observability_basis(W: torch.Tensor, R: torch.Tensor,
                                tau_ratio: float = 1e-3):
    """
    计算深度残差对 FFD 控制偏移的观测算子 J 的可观测子空间基

    深度残差 r_i = z_obs(u_i,v_i) − z_pred_i,
    ∂r_i/∂δ_jk = −W_ij · R_z,k   (R_z 为 R 的第 z 行, k∈{x,y,z})
    ⇒ J = −W ⊗ R_z ∈ R^{N×3M}

    Returns:
        V_obs:  (3M, r) 可观测方向基 (σ>τ)
        s:      (3M,) 全部奇异值 (分析用)
        r:      可观测子空间维数
    """
    N, M = W.shape
    Rz = R[2, :]                                   # (3,)
    # J[i, 3j+k] = -W[i,j] * Rz[k]
    J = -(W.unsqueeze(2) * Rz.reshape(1, 1, 3))    # (N,M,3)
    J = J.reshape(N, 3 * M)
    # SVD (N×3M, 对 3000×192 规模 GPU 足够)
    U, S, Vh = torch.linalg.svd(J, full_matrices=False)
    tau = S.max() * tau_ratio
    r = int((S > tau).sum().item())
    V_obs = Vh[:r].T if r > 0 else torch.zeros(3 * M, 0, device=W.device)
    return V_obs, S, r


def unobservable_penalty(delta_vec: torch.Tensor, V_obs: torch.Tensor):
    """不可观测分量惩罚: ||(I − V_obs V_obsᵀ) δ||² 的均值"""
    if V_obs.shape[1] == 0:
        return (delta_vec ** 2).mean()
    proj = V_obs @ (V_obs.T @ delta_vec)
    unobs = delta_vec - proj
    return (unobs ** 2).mean()


def rtw_optimize_projective_obs(model_pts: np.ndarray, depth_img: np.ndarray,
                                K: np.ndarray, T_init: np.ndarray,
                                grid=(4, 4, 4), iters: int = 400,
                                lr: float = 1e-2, lambda_mag: float = 1e-4,
                                lambda_sm: float = 1e-3,
                                lambda_unobs: float = 1e-2,
                                inlier_hi: float = 15.0,
                                inlier_lo: float = 3.0,
                                tau_ratio: float = 1e-3,
                                n_data: int = 3000, device: str = 'cuda',
                                seed: int = 0, verbose: bool = False):
    """
    可观测性引导的 RTW 优化 (docs/07 创新点2 实现)

    与 rtw_optimize_projective 的唯一区别:
    数据项不变, 额外对形变的**不可观测子空间分量**施加强收缩
    (λ_unobs·||δ_unobs||²), 抑制无数据支持的伪形变 (过拟合)。

    Returns: dict(T, delta, lattice, losses, V_def, res_rms, obs_rank,
                  obs_singular)
    """
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    if model_pts.shape[0] > n_data:
        sel = rng.choice(model_pts.shape[0], n_data, replace=False)
        V_data = model_pts[sel]
    else:
        V_data = model_pts
    Vd = torch.as_tensor(V_data, dtype=torch.float32, device=device)
    V_all = torch.as_tensor(model_pts, dtype=torch.float32, device=device)
    D = torch.as_tensor(depth_img, dtype=torch.float32, device=device)
    Kt = torch.as_tensor(K, dtype=torch.float32, device=device)
    Himg, Wimg = D.shape

    lat = FFDLattice(model_pts.min(axis=0), model_pts.max(axis=0),
                     grid=grid, device=device)
    W_data = lat.weights(V_data)

    rv = torch.tensor(log_so3(T_init[:3, :3]), dtype=torch.float32,
                      device=device, requires_grad=True)
    tv = torch.tensor(T_init[:3, 3].copy(), dtype=torch.float32,
                      device=device, requires_grad=True)
    delta = torch.zeros(lat.n_ctrl, 3, dtype=torch.float32,
                        device=device, requires_grad=True)

    # ---- 可观测性基 (初始位姿处计算, 优化中固定近似) ----
    with torch.no_grad():
        R0 = rodrigues_torch(rv.detach())
        V_obs, S, obs_rank = compute_observability_basis(W_data, R0,
                                                         tau_ratio)
    if verbose:
        print(f'  [E2] 可观测子空间维数: {obs_rank}/{3 * lat.n_ctrl} '
              f'(σmax={float(S.max()):.3f}, σmin>τ={float(S[obs_rank - 1]) if obs_rank > 0 else 0:.2e})')

    opt = torch.optim.Adam([rv, tv, delta], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters)

    losses = []
    for it in range(iters):
        opt.zero_grad()
        R = rodrigues_torch(rv)
        V_def = Vd + W_data @ delta
        V_cam = V_def @ R.T + tv
        z = V_cam[:, 2].clamp(min=1e-6)
        u = Kt[0, 0] * V_cam[:, 0] / z + Kt[0, 2]
        v = Kt[1, 1] * V_cam[:, 1] / z + Kt[1, 2]
        gx = (2.0 * u / (Wimg - 1) - 1.0)
        gy = (2.0 * v / (Himg - 1) - 1.0)
        grid_t = torch.stack([gx, gy], dim=-1).reshape(1, 1, -1, 2)
        z_obs = torch.nn.functional.grid_sample(
            D.reshape(1, 1, Himg, Wimg), grid_t, mode='bilinear',
            padding_mode='zeros', align_corners=True).reshape(-1)
        res = z_obs - z
        thr = inlier_hi + (inlier_lo - inlier_hi) * (it / max(iters - 1, 1))
        # Huber 鲁棒核 (替代硬门限): 内点二次, 外点线性(常数梯度) —
        # 防止形变垃圾点"掉出门限即隐身"导致的 δ 爆炸
        valid = (z_obs > 0) & (V_cam[:, 2] > 1e-3)
        ar = res.abs()
        huber = torch.where(ar <= thr, 0.5 * ar ** 2 / max(float(thr), 1e-9),
                            ar - 0.5 * thr)
        data = huber[valid].mean() if int(valid.sum()) >= 10 \
            else res.abs().mean()

        # 剪影约束: 落在图内但无观测深度(z_obs==0)的点 = 应在剪影外的错误点
        # → 按离剪影点数比例惩罚, 消除"无约束区域自由漂移" (调试实证根因)
        in_img = (u >= 0) & (u < Wimg) & (v >= 0) & (v < Himg) \
            & (V_cam[:, 2] > 1e-3)
        bg = in_img & (z_obs <= 0)
        n_bg = int(bg.sum())
        if n_bg > 0:
            frac = bg.float().mean()
            data = data + frac * thr

        reg_mag = (delta ** 2).sum(-1).mean()
        reg_sm = lat.laplacian(delta)
        reg_unobs = unobservable_penalty(delta.reshape(-1), V_obs)

        loss = data + lambda_mag * reg_mag + lambda_sm * reg_sm \
            + lambda_unobs * reg_unobs
        loss.backward()
        opt.step()
        sched.step()
        losses.append(float(loss.detach().cpu()))
        if verbose and (it % 100 == 0 or it == iters - 1):
            print(f'  [RTW-Obs it={it}] loss={losses[-1]:.4f} '
                  f'data={float(data):.3f} unobs={float(reg_unobs):.5f}')

    R = rodrigues_torch(rv.detach()).cpu().numpy()
    T = make_T(R, tv.detach().cpu().numpy())
    delta_np = delta.detach().cpu().numpy()
    W_all = lat.weights(model_pts)
    V_def_all = (V_all + W_all @ delta.detach()).cpu().numpy()

    # 深度残差 (与 rtw_optimize_projective 同口径)
    with torch.no_grad():
        Vc = torch.as_tensor(V_def_all, dtype=torch.float32, device=device) \
            @ torch.as_tensor(R, dtype=torch.float32, device=device).T \
            + torch.as_tensor(T[:3, 3], dtype=torch.float32, device=device)
        z = Vc[:, 2].clamp(min=1e-6)
        u = Kt[0, 0] * Vc[:, 0] / z + Kt[0, 2]
        v = Kt[1, 1] * Vc[:, 1] / z + Kt[1, 2]
        gx = (2.0 * u / (Wimg - 1) - 1.0)
        gy = (2.0 * v / (Himg - 1) - 1.0)
        grid_t = torch.stack([gx, gy], dim=-1).reshape(1, 1, -1, 2)
        z_obs = torch.nn.functional.grid_sample(
            D.reshape(1, 1, Himg, Wimg), grid_t, mode='bilinear',
            padding_mode='zeros', align_corners=True).reshape(-1)
        res = (z_obs - z)
        inl = (z_obs > 0) & (res.abs() < 3.0)
        res_rms = float(torch.sqrt((res[inl] ** 2).mean()).cpu()) \
            if inl.sum() > 10 else float('nan')

    return {'T': T, 'delta': delta_np, 'lattice': lat, 'losses': losses,
            'V_def': V_def_all, 'res_rms': res_rms,
            'obs_rank': obs_rank, 'obs_singular': S.cpu().numpy(),
            'V_obs': V_obs}


# ============================================================
# RTW 联合优化 —— 投影数据关联 (深度残差, 推荐)
# ============================================================

def rtw_optimize_projective(model_pts: np.ndarray, depth_img: np.ndarray,
                            K: np.ndarray, T_init: np.ndarray,
                            grid=(4, 4, 4), iters: int = 400, lr: float = 1e-2,
                            lambda_mag: float = 1e-4, lambda_sm: float = 1e-3,
                            lambda_p2p: float = 1.0, lambda_pose: float = 0.0,
                            focal_gamma: float = 0.0, lambda_nrm: float = 0.0,
                            two_phase: bool = False, edge_band_px: float = 0.0,
                            focus_mask: np.ndarray = None,
                            model_normals: np.ndarray = None,
                            n_data: int = 4000, inlier_hi: float = 15.0,
                            inlier_lo: float = 3.0, device: str = 'cuda',
                            seed: int = 0, verbose: bool = False):
    """
    刚性位姿 + FFD 形变联合优化 —— 投影数据关联版本

    数据关联: 模型点经 T_RTW 变换后【投影到观测深度图】采样深度,
              残差 = z_obs(u,v) − z_pred  (直接定义在成像平面上,
              与"2D像素↔3D模型"投影映射天然一致, 对应关系正确,
              优于最近邻 Chamfer —— 形变恢复的关键)

    内门限 coarse-to-fine: inlier_hi → inlier_lo 线性退火。

    Args:
        model_pts: (N,3) 模型顶点
        depth_img: (H,W) 观测深度 (0=无效)
        K:         3x3 内参
        T_init:    初始刚性位姿 (ICP 提供)
        inlier_hi/lo: 内点门限退火 (与深度同单位)
    Returns:
        dict: T, delta, lattice, losses, V_def, res_rms, inl_hist
    """
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    if model_pts.shape[0] > n_data:
        sel = rng.choice(model_pts.shape[0], n_data, replace=False)
        V_data = model_pts[sel]
    else:
        sel = np.arange(model_pts.shape[0])
        V_data = model_pts
    Vd = torch.as_tensor(V_data, dtype=torch.float32, device=device)
    # 法线随同一子采样对齐 (否则与数据点尺寸不匹配, 实测 8000vs4000 崩)
    if model_normals is not None:
        model_normals = model_normals[sel]
    V_all = torch.as_tensor(model_pts, dtype=torch.float32, device=device)
    D = torch.as_tensor(depth_img, dtype=torch.float32, device=device)
    Kt = torch.as_tensor(K, dtype=torch.float32, device=device)
    Himg, Wimg = D.shape
    # 点对面预计算: 观测点云与法线 (一次/观测)
    P_obs, N_obs, _ = depth_to_points_normals(D, Kt)

    # 边缘带排除掩码 (边缘竞争治理): 剪影边界 edge_band_px 带内像素
    # (位姿误差在此产生大残差, 竞争压制局部形变) —— 从数据项排除
    edge_excl = None
    if edge_band_px > 0:
        import cv2 as _cv2
        sil = (depth_img > 0).astype(np.uint8)
        bnd = _cv2.morphologyEx(sil, _cv2.MORPH_GRADIENT,
                                np.ones((3, 3), np.uint8))
        band = _cv2.dilate(bnd, np.ones((3, 3), np.uint8),
                           iterations=max(1, int(edge_band_px // 2)))
        edge_excl = torch.as_tensor(
            (sil.astype(bool) & (band > 0)), dtype=torch.bool,
            device=device)

    def _sample3(T3, grid_t):
        """3通道张量在 grid_t 处的双线性采样 → (N,3)"""
        o = torch.nn.functional.grid_sample(
            T3.permute(2, 0, 1).reshape(1, 3, Himg, Wimg), grid_t,
            mode='bilinear', padding_mode='zeros', align_corners=True)
        return o.reshape(3, -1).T

    lat = FFDLattice(model_pts.min(axis=0), model_pts.max(axis=0),
                     grid=grid, device=device)
    Wd = lat.weights(V_data)

    rv = torch.tensor(log_so3(T_init[:3, :3]), dtype=torch.float32,
                      device=device, requires_grad=True)
    tv = torch.tensor(T_init[:3, 3].copy(), dtype=torch.float32,
                      device=device, requires_grad=True)
    rv_ref = rv.detach().clone()
    tv_ref = tv.detach().clone()
    delta = torch.zeros(lat.n_ctrl, 3, dtype=torch.float32,
                        device=device, requires_grad=True)

    opt = torch.optim.Adam([rv, tv, delta], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters)

    def _proj_data(V_cam, thr):
        """前向: 投影+采样+鲁棒核+剪影/边缘排除 → (data, res, valid, n_inl)"""
        z = V_cam[:, 2].clamp(min=1e-6)
        u = Kt[0, 0] * V_cam[:, 0] / z + Kt[0, 2]
        v = Kt[1, 1] * V_cam[:, 1] / z + Kt[1, 2]
        gx = (2.0 * u / (Wimg - 1) - 1.0)
        gy = (2.0 * v / (Himg - 1) - 1.0)
        grid_t = torch.stack([gx, gy], dim=-1).reshape(1, 1, -1, 2)
        z_obs = torch.nn.functional.grid_sample(
            D.reshape(1, 1, Himg, Wimg), grid_t, mode='bilinear',
            padding_mode='zeros', align_corners=True).reshape(-1)
        res = z_obs - z
        valid = (z_obs > 0) & (V_cam[:, 2] > 1e-3)
        # 边缘带排除 (边缘竞争治理): 边界带内点从 valid 剔除
        if edge_excl is not None:
            eb = torch.nn.functional.grid_sample(
                edge_excl.float().reshape(1, 1, Himg, Wimg), grid_t,
                mode='nearest', padding_mode='zeros',
                align_corners=True).reshape(-1) > 0.5
            valid = valid & ~eb
        ar = res.abs()
        huber = torch.where(ar <= thr, 0.5 * ar ** 2 / max(float(thr), 1e-9),
                            ar - 0.5 * thr)
        n_inl = int(((ar < thr) & valid).sum().item())
        if n_inl < 20:
            data = res.abs().mean()
        else:
            if focal_gamma > 0:
                w = (ar / ar.max().clamp(min=1e-9)) ** focal_gamma
                data = (w * huber)[valid].sum() / w[valid].sum().clamp(min=1e-9)
            else:
                data = huber[valid].mean()
        # 剪影约束
        in_img = (u >= 0) & (u < Wimg) & (v >= 0) & (v < Himg) \
            & (V_cam[:, 2] > 1e-3)
        bg = in_img & (z_obs <= 0)
        n_bg = int(bg.sum())
        if n_bg > 0:
            data = data + bg.float().mean() * thr
        return data, n_inl, u, v, z, grid_t, valid, res

    # ---- Phase A: 仅优化 (rv,tv), δ 冻结 —— 位姿充分收敛, 治 R-δ 耦合 ----
    if two_phase:
        for it_a in range(max(60, iters // 3)):
            opt.zero_grad()
            R = rodrigues_torch(rv)
            V_cam = Vd @ R.T + tv
            thr = inlier_hi + (inlier_lo - inlier_hi) * \
                (it_a / max(iters - 1, 1))
            data_a, _, _, _, _, _, _, _ = _proj_data(V_cam, thr)
            data_a.backward()
            opt.step()
        rv.requires_grad_(False)
        tv.requires_grad_(False)
        if verbose:
            print(f'  [RTW-2P] Phase A done, 冻结 (rv,tv)')

    # 法线一致性预计算 (全法线一致-轻量版): 基底法线 (R 旋转近似)
    n_base_t = None
    if model_normals is not None and lambda_nrm > 0:
        n_base_t = torch.as_tensor(model_normals, dtype=torch.float32,
                                   device=device)

    # 区域聚焦掩码 (层次化/区域聚焦损失用): 数据项仅作用于掩码像素
    focus_t = None
    if focus_mask is not None:
        focus_t = torch.as_tensor(focus_mask, dtype=torch.float32,
                                  device=device)

    losses, inl_hist = [], []
    for it in range(iters):
        opt.zero_grad()
        R = rodrigues_torch(rv)
        V_def = Vd + Wd @ delta
        V_cam = V_def @ R.T + tv
        thr = inlier_hi + (inlier_lo - inlier_hi) * (it / max(iters - 1, 1))
        data, n_inl, u, v, z, grid_t, valid, res = _proj_data(V_cam, thr)
        inl_hist.append(n_inl)

        # 区域聚焦: 数据项收敛到高残差区域 (凹痕等局部特征)
        if focus_t is not None:
            fm = torch.nn.functional.grid_sample(
                focus_t.reshape(1, 1, Himg, Wimg), grid_t,
                mode='nearest', padding_mode='zeros',
                align_corners=True).reshape(-1) > 0.5
            v2 = valid & fm
            if int(v2.sum()) >= 20:
                data = data * 0.2 + 0.8 * (
                    (res.abs())[v2].mean())

        # 法线一致性 (轻量近似: 基底法线随 R 旋转 vs 观测法线)
        if n_base_t is not None:
            n_pred = n_base_t @ R.T
            n_o2 = _sample3(N_obs, grid_t)
            cos = (n_pred * n_o2).sum(dim=1).clamp(-1, 1)
            data = data + lambda_nrm * (1.0 - cos)[valid].mean()

        # 点对面法线项 (射线滑移根本解): res_p2p = (V_cam − P_obs)·N_obs
        # 沿视线滑动在曲面上产生法向残差, 深度残差不可见的部分被此项捕获
        p_o = _sample3(P_obs, grid_t)
        n_o = _sample3(N_obs, grid_t)
        res_p2p = ((V_cam - p_o) * n_o).sum(dim=1)
        ar2 = res_p2p.abs()
        huber2 = torch.where(ar2 <= thr, 0.5 * ar2 ** 2 / max(float(thr), 1e-9),
                             ar2 - 0.5 * thr)
        if n_inl >= 20:
            data = data + lambda_p2p * huber2[valid].mean()

        reg_mag = (delta ** 2).sum(-1).mean()
        reg_sm = lat.laplacian(delta)
        # 位姿先验 (R-δ 规范耦合治理): 软锚定 (rv,tv) 于刚性估计,
        # 迫使形变分量承担 dent, 防止 δ↔R,t 再平衡吃掉形变恢复
        reg_pose = ((rv - rv_ref) ** 2).sum() * 10.0 \
            + ((tv - tv_ref) ** 2).sum() / (lat.step.mean() ** 2 + 1e-9)
        # 雅可比约束: 相邻控制点偏移差 (局部剪切/剪切滑移的强抑制)
        # 射线滑移表现为高频局部形变 → 大 λ_sm 压制, 平滑真形变保留
        loss = data + lambda_mag * reg_mag + lambda_sm * reg_sm \
            + lambda_pose * reg_pose
        loss.backward()
        opt.step()
        sched.step()
        losses.append(float(loss.detach().cpu()))
        if verbose and (it % 50 == 0 or it == iters - 1):
            print(f'  [RTW-P it={it}] loss={losses[-1]:.3f} '
                  f'data={float(data):.3f} inl={n_inl} thr={thr:.1f}')

    R = rodrigues_torch(rv.detach()).cpu().numpy()
    T = make_T(R, tv.detach().cpu().numpy())
    delta_np = delta.detach().cpu().numpy()
    W_all = lat.weights(model_pts)
    V_def_all = (V_all + W_all @ delta.detach()).cpu().numpy()

    # 最终深度残差 RMS (内点)
    with torch.no_grad():
        Vc = torch.as_tensor(V_def_all, dtype=torch.float32, device=device) \
            @ torch.as_tensor(R, dtype=torch.float32, device=device).T \
            + torch.as_tensor(T[:3, 3], dtype=torch.float32, device=device)
        z = Vc[:, 2].clamp(min=1e-6)
        u = Kt[0, 0] * Vc[:, 0] / z + Kt[0, 2]
        v = Kt[1, 1] * Vc[:, 1] / z + Kt[1, 2]
        gx = (2.0 * u / (Wimg - 1) - 1.0)
        gy = (2.0 * v / (Himg - 1) - 1.0)
        grid_t = torch.stack([gx, gy], dim=-1).reshape(1, 1, -1, 2)
        z_obs = torch.nn.functional.grid_sample(
            D.reshape(1, 1, Himg, Wimg), grid_t, mode='bilinear',
            padding_mode='zeros', align_corners=True).reshape(-1)
        res = (z_obs - z)
        inl = (z_obs > 0) & (res.abs() < inlier_lo)
        res_rms = float(torch.sqrt((res[inl] ** 2).mean()).cpu()) \
            if inl.sum() > 10 else float('nan')

    return {'T': T, 'delta': delta_np, 'lattice': lat, 'losses': losses,
            'V_def': V_def_all, 'res_rms': res_rms, 'inl_hist': inl_hist}
