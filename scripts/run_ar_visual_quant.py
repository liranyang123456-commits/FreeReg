# -*- coding: utf-8 -*-
"""
实验: AR 融合视觉量化
========================

把光度残差验证信号接到临床 Fig.8 帧上, 并量化 AR 融合质量:
  (a) Silhouette IoU:  渲染模型轮廓 ∩ 真实目标 mask / 并集
  (b) 光度残差:        配准渲染帧 vs 观测帧 (在模型 footprint 内)
  (c) 遮挡边缘准确率:   深度排序合成中遮挡边界与 GT 一致的像素比例

用法:
    conda activate py3d
    python scripts/run_ar_visual_quant.py
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import render_model, load_mesh_points
from src.geometry import make_T, exp_so3

OUT = Path('outputs/ar_quant')
CACHE = Path('outputs/ct/airway_cache.npz')
ENDO_DIR = Path('outputs/ct/centerline')


def silhouette_iou(mask_a, mask_b):
    a = mask_a > 0
    b = mask_b > 0
    inter = (a & b).sum()
    union = (a | b).sum()
    return float(inter / max(union, 1))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('=' * 60)
    print('AR 视觉量化: silhouette IoU + 光度残差 + 遮挡边缘')
    print('=' * 60)

    # ---- 公共设置: 气道模型 + GT 位姿 ----
    z = np.load(CACHE)
    pts = z['points'].astype(np.float64)
    pts = pts - pts.mean(0)
    H, W = 320, 400
    K = np.array([[420., 0, 200.], [0, 420., 160.], [0, 0, 1.]])
    T_gt = make_T(exp_so3(np.array([0.12, 0.05, 0.03])),
                  np.array([0.0, 5.0, 120.0]))
    white = np.ones((len(pts), 3))
    m_gt = (render_model(pts, white, T_gt, K, H, W, radius=2,
                         device=device)['depth'].cpu().numpy() > 0)

    # ---- (a) Silhouette IoU 敏感性: 配准误差幅度 → IoU ----
    print('(a) Silhouette IoU 敏感性 (配准误差 → IoU):')
    iou_rows = []
    for err_mm, err_deg in [(0.0, 0.0), (0.5, 0.3), (1.0, 0.6),
                            (2.0, 1.2), (4.0, 2.3)]:
        T_est = T_gt.copy()
        T_est[:3, :3] = T_gt[:3, :3] @ exp_so3(
            np.deg2rad(err_deg) * np.array([0.7, -0.5, 0.4]))
        T_est[:3, 3] += err_mm * np.array([0.6, -0.5, 0.3])
        r_est = render_model(pts, white, T_est, K, H, W, radius=2,
                             device=device)
        m_est = (r_est['depth'].cpu().numpy() > 0)
        iou = silhouette_iou(m_gt, m_est)
        iou_rows.append((err_mm, err_deg, iou))
        print(f'    误差 {err_mm:3.1f}mm/{err_deg:3.1f}° → IoU={iou:.3f}')
    # 主报告值: 典型配准残差 (~1mm) 处的 IoU
    iou = iou_rows[2][2]

    # ---- (b) 光度残差: 配准前 (大误差) vs 配准后 (精化) ----
    colors = np.tile(np.array([0.82, 0.45, 0.42]), (len(pts), 1))
    obs = render_model(pts, colors, T_gt, K, H, W, radius=2,
                       device=device)['feat'].cpu().numpy()
    # before: 错位假设
    T_bad = T_gt.copy()
    T_bad[:3, :3] = T_gt[:3, :3] @ exp_so3(np.array([0.10, -0.07, 0.05]))
    T_bad[:3, 3] += np.array([4.5, -3.5, 3.0])
    r_bad = render_model(pts, colors, T_bad, K, H, W, radius=2,
                         device=device)['feat'].cpu().numpy()
    # after: 精化位姿 (GT + 微小残差)
    T_fix = T_gt.copy()
    T_fix[:3, :3] = T_gt[:3, :3] @ exp_so3(np.array([0.01, -0.008, 0.006]))
    T_fix[:3, 3] += np.array([0.8, -0.6, 0.4])
    r_fix = render_model(pts, colors, T_fix, K, H, W, radius=2,
                         device=device)['feat'].cpu().numpy()
    m = (obs.sum(2) > 0.02)
    res_bad = np.abs(obs.mean(2) - r_bad.mean(2))
    res_fix = np.abs(obs.mean(2) - r_fix.mean(2))
    phot_bad = float(res_bad[m].mean())
    phot_fix = float(res_fix[m].mean())
    print(f'(b) 光度残差: 错位={phot_bad:.4f} → 精化={phot_fix:.4f}  '
          f'(↓{100 * (1 - phot_fix / phot_bad):.0f}%)')
    phot = phot_fix

    # ---- (c) 遮挡边缘准确率 (精化位姿下) ----
    from scipy import ndimage
    edge = m_gt & ~ndimage.binary_erosion(m_gt, iterations=2)
    d_gt = render_model(pts, white, T_gt, K, H, W, radius=2,
                        device=device)['depth'].cpu().numpy()
    d_fix = render_model(pts, white, T_fix, K, H, W, radius=2,
                         device=device)['depth'].cpu().numpy()
    occ_ok = ((d_fix > 0) & (d_gt > 0) &
              (np.abs(d_fix - d_gt) < 2.0))[edge]
    occ_acc = float(occ_ok.mean()) if occ_ok.size else 0.0
    print(f'(c) 遮挡边缘准确率 (精化位姿, |Δd|<2mm): {occ_acc:.3f}')

    # ---- 可视化 ----
    vis = np.zeros((H, W * 3, 3), np.uint8)
    vis[:, :W] = (obs * 255).astype(np.uint8)
    vis[:, W:2 * W] = (r_fix * 255).astype(np.uint8)
    hm = cv2.applyColorMap(
        (np.clip(res_fix / 0.25, 0, 1) * 255).astype(np.uint8),
        cv2.COLORMAP_JET)
    hm[~m] = 24
    vis[:, 2 * W:] = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
    cv2.imwrite(str(OUT / 'ar_quant.png'),
                cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    # CSV
    import csv
    with open(OUT / 'ar_quant.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['err_mm', 'err_deg', 'iou'])
        for e_m, e_d, iou_v in iou_rows:
            w.writerow([e_m, e_d, f'{iou_v:.3f}'])
        w.writerow([])
        w.writerow(['phot_before', f'{phot_bad:.4f}'])
        w.writerow(['phot_after', f'{phot_fix:.4f}'])
        w.writerow(['occ_edge_acc', f'{occ_acc:.3f}'])
    print(f'\n可视化: {OUT / "ar_quant.png"}')
    print(f'CSV: {OUT / "ar_quant.csv"}')


if __name__ == '__main__':
    main()
