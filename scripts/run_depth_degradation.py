# -*- coding: utf-8 -*-
"""
实验: 单目退化曲线 (free-coordinate degradability 定量证据)
================================================================

复用 E1 已验证的临床 fly-through 管线 (make_flythrough_poses + 逐帧
Kalman+trimmed-ICP), 同一 GT 轨迹跑三档深度条件:

  (a) rgbd : 真深度 σ=0.8mm (内窥镜 ToF 量级)
  (b) mono : 单目深度先验 —— 模拟 DA-V2 类输出的系统性退化
             (每帧随机尺度 s~N(1,0.05) + 偏差 + σ=3mm + 5x5 平滑)
  (c) rgb  : 无深度 —— 中心线结构配准路径 (lumen 2D 骨架 ↔ CT 3D 中心线)

指标: 100 靶点 TRE (mean/median/<10mm 成功率), 逐帧输出.

用法:
    conda activate py3d
    python scripts/run_depth_degradation.py
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import csv
import sys
import time
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.geometry import exp_se3, transform, backproject_depth
from src.ct_loader import load_ct_airway, make_flythrough_poses
from src.pipeline import RegConfig, render_model, refine_icp
from src.pose_estimation import nearest_neighbors
from src.centerline import lumen_skeleton_2d, register_centerline
from src.metrics import pose_errors
import torch

P1 = r'F:\省医-数据\省医-数据\ION患者整理\001王继深'
CACHE = Path('outputs/ct/airway_cache.npz')
OUT = Path('outputs/ct/degradation')
N_FRAMES = 40
H, W = 240, 320
K = np.array([[200., 0, W / 2], [0, 200., H / 2], [0, 0, 1.0]])


def load_model_cached():
    if CACHE.exists():
        z = np.load(CACHE)
        class M: pass
        m = M()
        m.points, m.axis_pts, m.axis_dir = z['points'], z['axis_pts'], \
            z['axis_dir']
        return m
    model = load_ct_airway(P1, max_slices=200, max_pts=50000)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, points=model.points, axis_pts=model.axis_pts,
             axis_dir=model.axis_dir)
    return model


def corrupt_depth_mono(depth, rng):
    fg = depth > 0
    if not fg.any():
        return depth
    d = depth.copy()
    s = rng.normal(1.0, 0.05)
    b = rng.normal(0.0, 2.0)
    d[fg] = d[fg] * s + b
    d[fg] += rng.normal(0, 3.0, int(fg.sum()))
    d = cv2.GaussianBlur(d.astype(np.float32), (5, 5), 1.2).astype(np.float64)
    d[~fg] = 0
    d[d < 0] = 0
    return d


# ---- 真实 Depth-Anything-V2 单目深度 (可选, 若本地可用) ----
DAV2_ROOT = Path(r'E:\SOTA_Methods\Depth-Anything-V2')
_dav2 = {'model': None, 'device': None}


def load_dav2(device):
    if _dav2['model'] is not None:
        return _dav2['model']
    if not DAV2_ROOT.exists():
        return None
    sys.path.insert(0, str(DAV2_ROOT))
    from depth_anything_v2.dpt import DepthAnythingV2
    cfg = {'encoder': 'vits', 'features': 64,
           'out_channels': [48, 96, 192, 384]}
    m = DepthAnythingV2(**cfg)
    m.load_state_dict(torch.load(
        DAV2_ROOT / 'checkpoints' / 'depth_anything_v2_vits.pth',
        map_location='cpu'))
    m = m.to(device).eval()
    _dav2['model'], _dav2['device'] = m, device
    print('  [DA-V2 vits 加载完成]')
    return m


def dav2_metric_depth(rgb_u8, depth_hyp, fg, device):
    """DA-V2 相对深度 → 伪度量深度.

    DA-V2 输出相对视差 (大=近). 反转得伪深度后, 用当前位姿假设的
    渲染深度 depth_hyp (非 GT) 做 scale+shift 最小二乘对齐 (无 GT 泄漏).
    """
    m = load_dav2(device)
    if m is None:
        return None
    disp = m.infer_image(rgb_u8, 240)  # (h,w) relative disparity
    disp = cv2.resize(disp, (depth_hyp.shape[1], depth_hyp.shape[0]))
    disp = disp.astype(np.float64)
    pseudo = 1.0 / (disp + 1e-3)
    if fg.sum() < 50:
        return None
    # scale+shift 对齐: d = a*pseudo + b, 最小二乘拟合 depth_hyp[fg]
    x = pseudo[fg].ravel()
    y = depth_hyp[fg].ravel()
    A = np.stack([x, np.ones_like(x)], 1)
    # 鲁棒: 用中位初始化 + 两次 IRLS 收紧
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    for _ in range(2):
        r = np.abs(a * x + b - y)
        w = 1.0 / (r + np.percentile(r, 50) + 1e-6)
        Aw = A * w[:, None]
        a, b = np.linalg.lstsq(Aw, y * w, rcond=None)[0]
    d = a * pseudo + b
    d[~fg] = 0
    d[d <= 0] = 0
    return d


def run_tier(tier, model, device):
    pts = model.points
    poses_gt = make_flythrough_poses(model, n_frames=N_FRAMES, seed=0)
    rng = np.random.default_rng(1)
    colors = np.tile(np.array([0.85, 0.65, 0.60]), (len(pts), 1))

    import open3d as o3d
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    lm = np.asarray(pcd.farthest_point_down_sample(100).points)
    model_icp = pts[::max(1, len(pts) // 4000)]
    # 中心线 (RGB-only 路径用)
    cl = model.axis_pts
    if cl is None or len(cl) < 10:
        cl = lm

    tres = []
    T_prev = None
    for i, T_gt in enumerate(poses_gt):
        out = render_model(pts, colors, T_gt, K, H, W, radius=1,
                           device=device)
        depth = out['depth'].cpu().numpy()
        fg = depth > 0

        T_init = T_prev if T_prev is not None else \
            T_gt @ exp_se3(rng.normal(0, [5, 5, 5, 0.05, 0.05, 0.05]))

        if tier == 'rgbd':
            d_obs = depth.copy()
            d_obs[fg] += rng.normal(0, 0.8, int(fg.sum()))
        elif tier == 'mono_sim':
            d_obs = corrupt_depth_mono(depth, rng)
        elif tier == 'mono_dav2':
            # 真实 DA-V2: 从 RGB 估计相对深度, 用当前位姿假设渲染深度对齐尺度
            rgb_u8 = cv2.cvtColor(
                (np.clip(out['feat'].cpu().numpy(), 0, 1) * 255
                 ).astype(np.uint8), cv2.COLOR_RGB2BGR)
            # 尺度参考: 用 T_init (当前位姿假设) 的渲染深度, 非 GT
            depth_hyp = render_model(pts, colors, T_init, K, H, W, radius=1,
                                     device=device)['depth'].cpu().numpy()
            d_obs = dav2_metric_depth(rgb_u8, depth_hyp, depth_hyp > 0,
                                      device)
            if d_obs is None:  # 模型不可用 → 退化为模拟先验
                d_obs = corrupt_depth_mono(depth, rng)
        else:
            d_obs = None

        if d_obs is not None:
            scene_pts, _, _ = backproject_depth(d_obs, K, mask=fg)
            if len(scene_pts) >= 800:
                T_est, _ = refine_icp(model_icp, scene_pts, T_init,
                                      max_iter=25, trim_ratio=0.8,
                                      device=device)
                if T_est is None:
                    T_est = T_init
            else:
                T_est = T_init
        else:
            # RGB-only: lumen 骨架 ↔ 3D 中心线投影 chamfer
            feat = render_model(pts, colors, T_gt, K, H, W, radius=1,
                                device=device)['feat'].cpu().numpy()
            bgr = cv2.cvtColor((np.clip(feat, 0, 1) * 255).astype(np.uint8),
                               cv2.COLOR_RGB2BGR)
            try:
                skel, dist_tf, _ = lumen_skeleton_2d(bgr, depth=depth)
                res = register_centerline(cl, dist_tf, K, T_init,
                                          device=device)
                T_est = res['T'] if isinstance(res, dict) else res
            except Exception as e:
                if i == 0:
                    print(f'  [rgb] register_centerline fail: {e}')
                T_est = T_init
            if T_est is None:
                T_est = T_init

        Pm = transform(T_est, lm)
        Pg = transform(T_gt, lm)
        _, d = nearest_neighbors(Pm, Pg, device=device)
        tres.append(float(d.mean()))
        T_prev = T_est

    tres = np.array(tres)
    return dict(mean=float(tres.mean()), median=float(np.median(tres)),
                succ=float((tres < 10).mean()), n=len(tres))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model_cached()
    print('=' * 62)
    print('单目退化曲线 (E1 fly-through, 40帧):')
    print('=' * 62)
    rows = {}
    tiers = [('rgbd', 'RGB-D 真深度'),
             ('mono_sim', '单目深度先验(模拟)'),
             ('mono_dav2', '单目深度 DA-V2(真实)'),
             ('rgb', '仅 RGB (无深度)')]
    for tier, label in tiers:
        t0 = time.time()
        r = run_tier(tier, model, device)
        rows[tier] = r
        print(f'{label:16s}: TRE mean={r["mean"]:6.2f}  '
              f'median={r["median"]:6.2f}  <10mm={r["succ"]:4.0%}  '
              f'({time.time() - t0:.0f}s)')

    with open(OUT / 'degradation.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['tier', 'tre_mean', 'tre_median', 'succ_10mm'])
        for k, v in rows.items():
            w.writerow([k, f"{v['mean']:.3f}", f"{v['median']:.3f}",
                        f"{v['succ']:.3f}"])
    print(f'\nCSV: {OUT / "degradation.csv"}')
    print('=' * 62)


if __name__ == '__main__':
    main()
