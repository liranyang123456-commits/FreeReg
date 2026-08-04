# -*- coding: utf-8 -*-
"""
3D Gaussian Splatting 渲染器 (纯 torch, 可微, EWA 泼溅)
=========================================================

Kerbl et al. SIGGRAPH 2023 的参考实现 (自包含, 免 CUDA 扩展编译):

  高斯 i: 均值 μ_i, 协方差 Σ_i = R_s S Sᵀ R_sᵀ, 不透明度 o_i, 颜色 c_i
  投影:   Σ_2d = J R Σ Rᵀ Jᵀ   (J = 透视投影雅可比, EWA)
  着色:   α(p) = o·exp(-½(p-μ_2d)ᵀ Σ_2d⁻¹ (p-μ_2d))
  合成:   前→后 alpha 混合  C = Σ c_i α_i Π_{j<i} (1-α_j)

与官方 CUDA 版关系: 数学完全一致; 本实现以正确性/可微性优先
(python 循环前到后), 速度低于官方 tile-based 内核 (实测见 docs/06),
接口语义一致, 可原位替换。

4D 扩展: 高斯均值/协方差由形变场 (FFDLattice, src/nonrigid_rtw) 驱动
→ 动态场景渲染 (锚点 = FFD 控制点), 见 scripts/run_4dgs_eval.py。

依赖: torch
作者: ZCode · 2026-08-03
"""
import numpy as np
import torch

from src.nonrigid_rtw import rodrigues_torch


# ============================================================
# 高斯模型
# ============================================================

class GaussianModel:
    """3DGS 高斯参数集 (全部可优化)"""

    def __init__(self, mu, log_scale, rotvec, color, logit_opacity,
                 device='cuda'):
        self.mu = torch.as_tensor(mu, dtype=torch.float32, device=device)
        self.log_scale = torch.as_tensor(log_scale, dtype=torch.float32,
                                         device=device)
        self.rotvec = torch.as_tensor(rotvec, dtype=torch.float32,
                                      device=device)
        self.color = torch.as_tensor(color, dtype=torch.float32,
                                     device=device)
        self.logit = torch.as_tensor(logit_opacity, dtype=torch.float32,
                                     device=device)

    @classmethod
    def from_points(cls, points, colors=None, init_scale=None,
                    scale_mult=2.0, device='cuda'):
        """由点云初始化: 各向同性小高斯, 不透明

        scale_mult: 间距倍数 (≥1.5 保证相邻高斯交叠覆盖, 否则斑点状)
        """
        N = len(points)
        if init_scale is None:
            # 平均最近邻间距 (k=1, 排除自身)
            P = torch.as_tensor(points[:min(N, 2000)], dtype=torch.float32,
                                device=device)
            d = torch.cdist(P, P)
            d.fill_diagonal_(float('inf'))
            init_scale = float(d.min(dim=1).values.mean().cpu())
        init_scale = max(init_scale * scale_mult, 1e-3)
        mu = torch.as_tensor(points, dtype=torch.float32, device=device)
        return cls(
            mu=mu,
            log_scale=torch.full((N, 3), np.log(init_scale), device=device),
            rotvec=torch.zeros(N, 3, device=device),
            color=torch.as_tensor(
                colors if colors is not None else np.full((N, 3), 0.5),
                dtype=torch.float32, device=device),
            logit_opacity=torch.full((N,), 2.0, device=device),  # sigmoid≈0.88
        )

    def params(self):
        return [self.mu, self.log_scale, self.rotvec, self.color, self.logit]

    def requires_grad_(self, flag=True):
        for p in self.params():
            p.requires_grad_(flag)
        return self

    @property
    def opacity(self):
        return torch.sigmoid(self.logit)

    @property
    def scales(self):
        return torch.exp(self.log_scale)


# ============================================================
# EWA 投影
# ============================================================

def project_gaussians(model: GaussianModel, T_wc: torch.Tensor,
                      fx, fy, cx, cy):
    """
    高斯投影到图像平面
    Returns: mu_2d (N,2), cov2d (N,2,2), mu_cam (N,3), valid (N,)
    """
    R_wc = T_wc[:3, :3]
    t_wc = T_wc[:3, 3]
    mu_cam = model.mu @ R_wc.T + t_wc              # (N,3)
    z = mu_cam[:, 2].clamp(min=1e-3)

    # 协方差: Σ = R_s S Sᵀ R_sᵀ,  Σ_cam = R_wc Σ R_wcᵀ
    N = model.mu.shape[0]
    Rs = rodrigues_torch_batch(model.rotvec)       # (N,3,3)
    S2 = (model.scales ** 2)
    Sigma = Rs * S2.unsqueeze(-1)                  # (N,3,3) Rs diag(s²)
    Sigma = Sigma @ Rs.transpose(1, 2)             # (N,3,3)
    Sigma_cam = R_wc.unsqueeze(0) @ Sigma @ R_wc.T.unsqueeze(0)

    # 透视雅可比 (2x3): x'=fx·x/z, y'=fy·y/z
    J = torch.zeros(N, 2, 3, device=model.mu.device)
    J[:, 0, 0] = fx / z
    J[:, 0, 2] = -fx * mu_cam[:, 0] / (z ** 2)
    J[:, 1, 1] = fy / z
    J[:, 1, 2] = -fy * mu_cam[:, 1] / (z ** 2)
    cov2d = J @ Sigma_cam @ J.transpose(1, 2)      # (N,2,2)

    mu_2d = torch.stack([fx * mu_cam[:, 0] / z + cx,
                         fy * mu_cam[:, 1] / z + cy], dim=1)
    valid = (mu_cam[:, 2] > 1e-3) & (model.opacity > 0.004)
    return mu_2d, cov2d, mu_cam, valid


def rodrigues_torch_batch(rv: torch.Tensor) -> torch.Tensor:
    """批量 Rodrigues: (N,3) → (N,3,3)"""
    N = rv.shape[0]
    theta = torch.norm(rv, dim=1, keepdim=True) + 1e-12  # (N,1)
    k = rv / theta                                        # (N,3)
    K = torch.zeros(N, 3, 3, device=rv.device)
    K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
    K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
    I = torch.eye(3, device=rv.device).unsqueeze(0).expand(N, -1, -1)
    st = torch.sin(theta).unsqueeze(-1)
    ct = torch.cos(theta).unsqueeze(-1)
    return I + st * K + (1 - ct) * (K @ K)


# ============================================================
# 渲染 (稀疏泼溅对 + 分段前→后 alpha 混合, 全向量化, 可微)
# ============================================================

def render_gaussians(model: GaussianModel, T_wc, K, H, W,
                     device='cuda', max_radius=12, bg=0.0,
                     track_depth=False):
    """
    EWA 高斯泼溅渲染 (前→后精确混合, 稀疏 (高斯,像素) 对 + 分段累积)

    实现: 枚举每个高斯的 3σ 泼溅窗口 → 计算每个 (高斯,像素) 对的 α
    → 按 (像素, 深度) 排序 → 分段 cumsum 计算每对的前向透过率 T_i
    → index_add 累加 T_i·α_i·c_i。全向量化 + 全可微 (训练/推理通用)。
    """
    if not torch.is_tensor(T_wc):
        T_wc = torch.as_tensor(T_wc, dtype=torch.float32, device=device)
    if not torch.is_tensor(K):
        K = torch.as_tensor(K, dtype=torch.float32, device=device)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    dev = model.mu.device
    N = model.mu.shape[0]

    mu_2d, cov2d, mu_cam, valid = project_gaussians(
        model, T_wc, fx, fy, cx, cy)
    z = mu_cam[:, 2]
    gid_all = torch.arange(N, device=dev)

    # ---- 1) 每高斯泼溅窗口 (3σ, 截断 max_radius) ----
    s00 = cov2d[:, 0, 0].clamp(min=1e-6)
    s01 = cov2d[:, 0, 1]
    s11 = cov2d[:, 1, 1].clamp(min=1e-6)
    rx = torch.ceil(3.0 * torch.sqrt(s00)).clamp(1, max_radius).long()
    ry = torch.ceil(3.0 * torch.sqrt(s11)).clamp(1, max_radius).long()
    sizes = ((2 * rx + 1) * (2 * ry + 1)).clamp(min=1)

    gid = torch.repeat_interleave(gid_all, sizes)
    offs = torch.cumsum(sizes, 0) - sizes
    local = torch.arange(gid.shape[0], device=dev) - offs[gid]
    pw = 2 * rx + 1
    lu = (local % pw[gid]) - rx[gid]
    lv = (local // pw[gid]) - ry[gid]
    u = torch.round(mu_2d[gid, 0]).long() + lu
    v = torch.round(mu_2d[gid, 1]).long() + lv
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H) & valid[gid]
    gid, u, v = gid[inb], u[inb], v[inb]
    lu_f, lv_f = lu[inb].float(), lv[inb].float()

    # ---- 2) 每对 α ----
    du = u.float() + 0.5 - mu_2d[gid, 0]
    dv = v.float() + 0.5 - mu_2d[gid, 1]
    det = (s00 * s11 - s01 ** 2).clamp(min=1e-9)
    inv_det = 1.0 / det
    power = (s11[gid] * du * du - 2 * s01[gid] * du * dv
             + s00[gid] * dv * dv) * inv_det[gid]
    alpha = (model.opacity[gid] * torch.exp(-0.5 * power)).clamp(max=0.99)

    # ---- 3) 按 (像素, 深度) 排序 → 分段前向透过率 ----
    pix = (v * W + u).long()
    zrank = torch.empty_like(gid_all)
    zrank[torch.argsort(z)] = gid_all        # 全局深度名次
    key = pix * N + zrank[gid]
    order = torch.argsort(key)
    pix_s, alpha_s, gid_s = pix[order], alpha[order], gid[order]
    z_s = z[gid_s]

    log_neg = torch.log((1.0 - alpha_s).clamp(min=1e-6))
    # fp64 cumsum: 全局 cumsum 幅度可达 ~1e6, fp32 下 "cum - cum_before"
    # 的段内差分被量化噪声吞掉 (实测透过率全 0), 必须双精度
    cum = torch.cumsum(log_neg.double(), 0)
    is_start = torch.cat([torch.ones(1, dtype=torch.bool, device=dev),
                          pix_s[1:] != pix_s[:-1]])
    seg_id = torch.cumsum(is_start.long(), 0) - 1
    starts = torch.nonzero(is_start).squeeze(1)
    # 每段起始处的 cum 前值 (首段为0); 索引映射: 段号 → 段首泼溅位置
    seg_start_cum = torch.zeros(starts.shape[0], dtype=cum.dtype, device=dev)
    seg_start_cum[1:] = cum[starts[1:] - 1]
    cum_before = seg_start_cum[seg_id]
    # 前向透过率: T_i = exp(Σ_{j<i} log(1-α_j)) = exp(cum - log_neg_i - cum_before)
    T_i = torch.exp((cum - log_neg.double() - cum_before).float())

    # ---- 4) 累加 (index_add, 可微) ----
    w = T_i * alpha_s
    C = torch.zeros(H * W, 3, device=dev)
    C.index_add_(0, pix_s, w.unsqueeze(1) * model.color[gid_s])
    Op = torch.zeros(H * W, device=dev)
    Op.index_add_(0, pix_s, w)
    if track_depth:
        D = torch.zeros(H * W, device=dev)
        D.index_add_(0, pix_s, w * z_s)

    # 背景: 每像素剩余透过率 = exp(该段总和) = exp(cum_end - cum_before_end)
    is_end = torch.cat([pix_s[1:] != pix_s[:-1],
                        torch.ones(1, dtype=torch.bool, device=dev)])
    T_end_pix = torch.exp((cum[is_end] - cum_before[is_end]).float())
    pix_end = pix_s[is_end]
    T_rem = torch.ones(H * W, device=dev)
    T_rem[pix_end] = T_end_pix
    C = C + T_rem.unsqueeze(1) * float(bg)
    Op = Op  # (不含背景)

    feat = C.reshape(H, W, 3)
    opacity_map = Op.reshape(H, W).clamp(max=1.0)
    if track_depth:
        depth = (D.reshape(H, W) / opacity_map.clamp(min=1e-6))
    else:
        depth = torch.zeros(H, W, device=dev)
    return {'feat': feat, 'depth': depth, 'opacity': opacity_map}


# ============================================================
# 训练 (多视角拟合, mini-3DGS)
# ============================================================

def optimize_gaussians(model: GaussianModel, views, K, iters=300,
                       lr=None, device='cuda', bg=0.0, verbose=True):
    """
    多视角图像拟合 (mini-3DGS 训练)
    views: list of dict(T_wc (4,4) numpy, image (H,W,3) float[0,1])
    """
    lr = lr or {'mu': 2e-3, 'log_scale': 5e-3, 'rotvec': 5e-3,
                'color': 1e-2, 'logit': 1e-2}
    model.requires_grad_(True)
    opt = torch.optim.Adam([
        {'params': [model.mu], 'lr': lr['mu']},
        {'params': [model.log_scale], 'lr': lr['log_scale']},
        {'params': [model.rotvec], 'lr': lr['rotvec']},
        {'params': [model.color], 'lr': lr['color']},
        {'params': [model.logit], 'lr': lr['logit']},
    ])
    Kt = torch.as_tensor(K, dtype=torch.float32, device=device)
    tgt = [torch.as_tensor(v['image'], dtype=torch.float32, device=device)
           for v in views]
    Ts = [torch.as_tensor(v['T_wc'], dtype=torch.float32, device=device)
          for v in views]
    H, W = tgt[0].shape[:2]

    hist = []
    for it in range(iters):
        opt.zero_grad()
        loss = 0.0
        for vi in range(len(views)):
            out = render_gaussians(model, Ts[vi], Kt, H, W, device=device,
                                   bg=bg)
            loss = loss + (out['feat'] - tgt[vi]).abs().mean() \
                + 0.2 * ((out['feat'] - tgt[vi]) ** 2).mean()
        loss.backward()
        opt.step()
        hist.append(float(loss.detach().cpu()) / len(views))
        if verbose and (it % 50 == 0 or it == iters - 1):
            print(f'  [3DGS it={it}] loss={hist[-1]:.5f}')
    model.requires_grad_(False)
    return hist


# ============================================================
# 图像质量指标
# ============================================================

def psnr(img, gt):
    mse = float(np.mean((np.asarray(img) - np.asarray(gt)) ** 2))
    return 99.0 if mse < 1e-10 else float(-10 * np.log10(mse))


def ssim_simple(img, gt):
    """简化 SSIM (全局统计, 免 skimage)"""
    x = np.asarray(img, dtype=np.float64)
    y = np.asarray(gt, dtype=np.float64)
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mux, muy = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = ((x - mux) * (y - muy)).mean()
    return float(((2 * mux * muy + C1) * (2 * cov + C2))
                 / ((mux ** 2 + muy ** 2 + C1) * (vx + vy + C2)))
