# -*- coding: utf-8 -*-
"""
渲染-比较 (Render-and-Compare) 位姿引擎
========================================

FoundationPose 范式的自包含实现 (几何版):
  离线: 环绕模型渲染多视角模板库 (测地线球面视点)
  在线: 观测深度 vs 模板库相似度打分 → top-K 位姿假设
        → ICP 精化 → 24 对称变体渲染验证 → 最终位姿

对应关系本质 (docs/05): "哪个渲染视角最像观测" = 隐式 2D-3D 对应,
是深度数据关联在假设空间上的推广。

解决对称 ADD 歧义: 几何配准阶段产生的多个对称假设,
通过渲染验证 (mask IoU + 深度一致性, 紧门限) 区分 —— 对称翻转在细节上
(如 ape 的耳朵) 产生不同的深度对齐, 可据此裁决; 真正旋转对称的物体
(如 bowl) 属协议内歧义, 用 ADD-S 评价。

FoundationPose 适配: 本引擎接口与其输出同构 (4x4 位姿), FP 安装后
可通过 PoseEngine(backend='foundationpose') 原位替换。

依赖: torch, numpy, open3d(仅测地球面)
作者: ZCode · 2026-08-03
"""
import hashlib
from pathlib import Path

import numpy as np
import torch

from src.geometry import make_T, transform, backproject_depth, exp_so3, \
    canonical_rotations_24
from src.pose_estimation import nearest_neighbors, icp


# ============================================================
# 测地线球面视点
# ============================================================

def geodesic_view_dirs(subdiv: int = 2) -> np.ndarray:
    """细分二十面体顶点方向 (subdiv=2 → 162 个近似均匀方向)"""
    import open3d as o3d
    mesh = o3d.geometry.TriangleMesh.create_icosahedron(radius=1.0)
    for _ in range(subdiv):
        mesh = mesh.subdivide_midpoint()
    V = np.asarray(mesh.vertices)
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    # 细分后顶点方向唯一化 (中点细分会产生重复方向)
    keys = np.round(V, 6)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return V[np.sort(idx)]


def look_at_T(eye: np.ndarray, up=(0, -1, 0)) -> np.ndarray:
    """相机位于 eye 看向原点 → T_cam_obj (OpenCV 约定: 视轴=+z_c, z>0 为前方)"""
    eye = np.asarray(eye, dtype=np.float64)
    z_c = -eye / np.linalg.norm(eye)         # 视轴指向场景中心
    up = np.asarray(up, dtype=np.float64)
    x_c = np.cross(z_c, up)
    if np.linalg.norm(x_c) < 1e-8:
        x_c = np.cross(z_c, np.array([0, 0, 1.0]))
    x_c = x_c / np.linalg.norm(x_c)
    y_c = np.cross(z_c, x_c)                 # 右手系: x×y=z
    R = np.stack([x_c, y_c, z_c], axis=0)    # 行=相机轴在世界系
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ eye
    return T


# ============================================================
# 模板库
# ============================================================

class TemplateLibrary:
    """模型的多视角渲染模板 (低分辨率深度 → 点云, 用于假设打分)"""

    def __init__(self, model_pts: np.ndarray, diameter: float,
                 n_subdiv: int = 2, img_size: int = 96, n_pts: int = 500,
                 device: str = 'cuda', cache_dir: str = None,
                 model_key: str = None):
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
        self.device = device
        self.img_size = img_size
        self.n_pts = n_pts
        self.model_pts = model_pts
        self.mu = model_pts.mean(axis=0)
        self.diameter = float(diameter)

        if model_key is None:
            h = hashlib.md5()
            h.update(model_pts.tobytes()[:4096])
            h.update(f'{diameter}{n_subdiv}{img_size}{n_pts}'.encode())
            key = h.hexdigest()[:12]
        else:
            key = model_key
        self.cache_path = Path(cache_dir) / f'tpl_{key}.npz' \
            if cache_dir else None
        if self.cache_path and self.cache_path.exists():
            z = np.load(self.cache_path)
            self.tpl_pts = z['tpl_pts']       # (V, n, 3)
            self.tpl_rots = z['tpl_rots']     # (V, 3, 3)
            self.tpl_zmed = z['tpl_zmed']     # (V,)
            return
        self._build(n_subdiv)
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(self.cache_path, tpl_pts=self.tpl_pts,
                     tpl_rots=self.tpl_rots, tpl_zmed=self.tpl_zmed)

    def _build(self, n_subdiv):
        from src.pipeline import render_model
        dirs = geodesic_view_dirs(n_subdiv)
        dist = 2.5 * self.diameter
        H = W = self.img_size
        fx = fy = self.img_size / 1.2          # 视场 ≈ 45°
        K = np.array([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1.0]])
        white = np.ones((len(self.model_pts), 3))
        rng = np.random.default_rng(0)

        tpl_pts, tpl_rots, tpl_zmed = [], [], []
        for d in dirs:
            T = look_at_T(d * dist)
            out = render_model(self.model_pts, white, T, K, H, W,
                               radius=1, device=self.device)
            depth = out['depth'].cpu().numpy()
            pts, _, z = backproject_depth(depth, K)
            if len(pts) < 50:
                pts = transform(T, self.model_pts[:self.n_pts])
            sel = rng.choice(len(pts), self.n_pts,
                             replace=(len(pts) < self.n_pts))
            tpl_pts.append(pts[sel])
            tpl_rots.append(T[:3, :3])
            tpl_zmed.append(float(np.median(z)))
        self.tpl_pts = np.array(tpl_pts)      # (V, n, 3)
        self.tpl_rots = np.array(tpl_rots)    # (V, 3, 3)
        self.tpl_zmed = np.array(tpl_zmed)    # (V,)

    @torch.no_grad()
    def score(self, obs_pts: np.ndarray, topk: int = 5, chunk: int = 16):
        """
        观测点云 (相机系, 模型尺度单位一致) → 模板相似度打分
        每个模板: 尺度对齐 + 质心对齐后双向 Chamfer
        Returns: top_idx (K,), scores (K,) [越小越好]
        """
        dev = self.device
        P = torch.as_tensor(obs_pts, dtype=torch.float32, device=dev)
        c_obs = P.mean(dim=0)
        T_all = torch.as_tensor(self.tpl_pts, dtype=torch.float32, device=dev)
        V = T_all.shape[0]
        scores = torch.empty(V, device=dev)
        for i in range(0, V, chunk):
            Q = T_all[i:i + chunk]                       # (c, n, 3)
            c_t = Q.mean(dim=1, keepdim=True)            # (c,1,3)
            # 纯平移对齐 (模型与观测同为毫米单位, 物体尺寸不可缩放;
            # 模板渲染距离差异仅影响弱透视, 质心对齐已足够)
            X = (Q - c_t) + c_obs.reshape(1, 1, 3)       # (c,n,3)
            # 双向 chamfer
            d1 = torch.cdist(P.unsqueeze(0), X).min(dim=2).values.mean(dim=1)
            d2 = torch.cdist(X, P.unsqueeze(0).expand(len(Q), -1, -1)
                             ).min(dim=2).values.mean(dim=1)
            scores[i:i + len(Q)] = d1 + d2
        top = torch.topk(scores, topk, largest=False)
        return top.indices.cpu().numpy(), top.values.cpu().numpy()

    def hypothesis_pose(self, tpl_idx: int, obs_pts: np.ndarray) -> np.ndarray:
        """由模板索引构造初始位姿: 模板旋转 + 模型质心→观测质心"""
        R = self.tpl_rots[tpl_idx]
        c_obs = obs_pts.mean(axis=0)
        t = c_obs - R @ self.mu
        return make_T(R, t)


# ============================================================
# 对称变体渲染验证
# ============================================================

@torch.no_grad()
def verify_pose_variants(T_cands, depth_obs: np.ndarray, mask_obs: np.ndarray,
                         K: np.ndarray, model_pts: np.ndarray,
                         mu: np.ndarray, device: str = 'cuda',
                         scale: float = 0.25):
    """
    对称假设裁决: 对每个候选位姿 + 其 24 规范旋转变体做低分辨率渲染,
    评分 = 0.5·mask IoU + 0.5·深度一致性(紧门限) → 返回最优

    对称翻转在几何细节上产生不同的深度对齐 (如 ape 耳部), 可据此区分。
    """
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    from src.pipeline import render_model

    H, W = depth_obs.shape
    Hs, Ws = int(H * scale), int(W * scale)
    Ks = K.copy()
    Ks[:2] *= scale
    D = torch.as_tensor(depth_obs, dtype=torch.float32, device=device)
    M_obs = torch.nn.functional.interpolate(
        torch.as_tensor(mask_obs > 0, dtype=torch.float32, device=device)
        .reshape(1, 1, H, W), size=(Hs, Ws), mode='nearest').reshape(Hs, Ws) > 0
    D_obs = torch.nn.functional.interpolate(
        D.reshape(1, 1, H, W), size=(Hs, Ws), mode='nearest').reshape(Hs, Ws)

    rots24 = canonical_rotations_24()
    white = np.ones((len(model_pts), 3))
    obs_valid = M_obs & (D_obs > 0)
    model_extent = float(np.linalg.norm(model_pts.max(axis=0)
                                        - model_pts.min(axis=0)))

    best = {'score': -1, 'T': None, 'variant': -1}
    for T in T_cands:
        for vi, Rc in enumerate(rots24):
            # 变体: T · (绕质心的规范旋转)
            Tv = T @ make_T(Rc, mu - Rc @ mu)
            out = render_model(model_pts, white, Tv, Ks, Hs, Ws,
                               radius=1, device=device)
            mh = out['mask']
            dh = out['depth']
            inter = mh & obs_valid
            union = mh | M_obs
            iou = (inter.sum().float() / union.sum().clamp(min=1).float())
            if inter.sum() < 20:
                depth_score = torch.zeros((), device=device)
            else:
                dz = (dh[inter] - D_obs[inter]).abs()
                # 紧门限: max(3mm, 3%模型直径) —— 对称翻转在细节深度上暴露
                thr = max(3.0, 0.03 * model_extent)
                depth_score = (dz < thr).float().mean()
            score = 0.4 * iou + 0.6 * depth_score
            if float(score) > best['score']:
                best = {'score': float(score), 'T': Tv, 'variant': vi}
    return best['T'], best['score']


# ============================================================
# 统一位姿引擎 (含 FoundationPose 适配点)
# ============================================================

class PoseEngine:
    """
    位姿引擎统一接口: register(depth, K, mask) → T_obj^cam (4x4)

    backend:
      'rc'            渲染-比较模板引擎 (默认, 自包含, 抗对称)
      'fpfh'/'multistart'/'best'  经典几何基线 (src/pipeline)
      'foundationpose'  官方 FoundationPose (需自行安装, 见异常提示)
    """

    def __init__(self, model_pts: np.ndarray, diameter: float,
                 backend: str = 'rc', device: str = 'cuda',
                 cache_dir: str = 'outputs/templates'):
        self.backend = backend
        self.device = device
        self.model_pts = model_pts
        self.mu = model_pts.mean(axis=0)
        self.diameter = diameter
        if backend == 'rc':
            self.lib = TemplateLibrary(model_pts, diameter, device=device,
                                       cache_dir=cache_dir)
        elif backend == 'foundationpose':
            raise RuntimeError(
                'FoundationPose 未安装。安装: git clone '
                'https://github.com/NVlabs/FoundationPose && '
                '按官方 README 配置权重; 或使用 backend="rc" '
                '(自包含渲染-比较引擎, 同构接口)。')

    def register(self, depth: np.ndarray, K: np.ndarray,
                 mask: np.ndarray = None, n_hypo: int = 4,
                 verify: bool = True):
        """渲染-比较假设 → ICP 精化 → 对称验证 → T (4x4)"""
        assert self.backend == 'rc'
        obs_pts, _, _ = backproject_depth(depth, K, mask=mask)
        if len(obs_pts) < 100:
            return None, {'error': 'too few obs points'}
        rng = np.random.default_rng(0)
        if len(obs_pts) > 800:
            obs_sub = obs_pts[rng.choice(len(obs_pts), 800, replace=False)]
        else:
            obs_sub = obs_pts

        idx, scores = self.lib.score(obs_sub, topk=n_hypo)
        cands, stats = [], []
        scene_sub = obs_pts[rng.choice(len(obs_pts),
                                       min(20000, len(obs_pts)), replace=False)]
        model_icp = self.model_pts[rng.choice(
            len(self.model_pts), min(3000, len(self.model_pts)),
            replace=False)]
        for ti in idx:
            T0 = self.lib.hypothesis_pose(int(ti), obs_pts)
            T, st = icp(model_icp, scene_sub, T_init=T0, max_iter=50,
                        trim_ratio=0.7, device=self.device)
            cands.append(T)
            stats.append({'tpl': int(ti), 'tpl_score': float(scores[len(stats)]),
                          'icp_rmse': st['rmse']})
        # 多假设仲裁: 无渲染深度一致性 (比低分辨率渲染验证更快更准)
        from src.pipeline import arbitrate_poses
        if verify:
            T_best, vscore, vscores = arbitrate_poses(
                cands, self.model_pts, depth, K, mask)
        else:
            order = np.argsort([s['icp_rmse'] for s in stats])
            T_best, vscore = cands[order[0]], 0.0
        return T_best, {'hypotheses': stats, 'verify_score': vscore}
