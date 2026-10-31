# -*- coding: utf-8 -*-
"""
M4: 3DGS 渲染质量与速度评测
============================

协议:
  pseudo-GT: 稠密点泼溅 (20k点, radius=2, Lambert着色) 5 视角
  基线A:    稀疏硬泼溅 (4k点, radius=1, 同样着色)
  本方案B:  3DGS-torch (4k高斯初始化自同一稀疏点, 4视角训练, 第5视角测试)
  指标:     PSNR / SSIM(简化) / 渲染 FPS (640×480)
  对比:     点泼溅占位 vs 3DGS —— 验证 M4 升级收益

用法:
    conda activate py3d
    python scripts/run_3dgs_eval.py
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import sys
import time
import csv
import json
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.geometry import make_T, exp_so3
from src.pipeline import load_mesh_points, render_model, lambert_colors
from src.gaussian_renderer import (GaussianModel, render_gaussians,
                                   optimize_gaussians, psnr, ssim_simple)
import torch

OUT = Path('outputs/3dgs')
BOP_MODEL = r'D:\bop\lm\models\obj_000008.ply'
K0 = np.array([[572.4, 0, 325.3], [0, 573.6, 242.0], [0, 0, 1.0]])
H, W = 240, 320


def main():
    global OUT
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0,
                    help='高斯初始化子集 / torch 种子 (审计运行为 0)')
    ap.add_argument('--out', default=str(OUT))
    args = ap.parse_args()
    OUT = Path(args.out)
    seed = args.seed
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('=' * 66)
    print(f'M4: 3DGS 渲染评测 (pseudo-GT 稠密泼溅 vs 硬泼溅 vs 3DGS) seed={seed}')
    print('=' * 66)
    K = K0.copy()
    K[:2] *= 0.5   # 320×240 (半分辨率)

    pts_dense, normals = load_mesh_points(BOP_MODEL, n_points=20000)
    mu = pts_dense.mean(axis=0)
    pts_dense -= mu
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(pts_dense), 4000, replace=False)
    pts_sparse = pts_dense[sel]
    normals_sparse = normals[sel]

    # 5 视角 (训练4 + 测试1)
    views_def = [
        ('train1', exp_so3(np.array([0.15, -0.25, 0.10])), [0, 0, 900]),
        ('train2', exp_so3(np.array([-0.20, 0.15, -0.05])), [30, 10, 900]),
        ('train3', exp_so3(np.array([0.10, 0.30, -0.12])), [-25, -10, 900]),
        ('train4', exp_so3(np.array([-0.15, -0.10, 0.20])), [15, -20, 900]),
        ('test', exp_so3(np.array([0.05, 0.05, 0.30])), [-10, 15, 900]),
    ]

    # ---------- pseudo-GT (稠密泼溅) ----------
    print('\n渲染 pseudo-GT (20k点, radius=2)...')
    gts = {}
    for name, R, t in views_def:
        T = make_T(R, np.array(t, dtype=float))
        colors = lambert_colors(normals, R, base_rgb=(0.75, 0.55, 0.45))
        out = render_model(pts_dense, colors, T, K, H, W, radius=2,
                           device=device)
        gts[name] = out['feat'].cpu().numpy()

    # ---------- 基线A: 硬泼溅 ----------
    print('基线A: 硬泼溅 (4k点)...')
    base = {}
    for name, R, t in views_def:
        T = make_T(R, np.array(t, dtype=float))
        colors_s = lambert_colors(normals_sparse, R,
                                  base_rgb=(0.75, 0.55, 0.45))
        out = render_model(pts_sparse, colors_s, T, K, H, W, radius=1,
                           device=device)
        base[name] = out['feat'].cpu().numpy()

    # ---------- 本方案B: 3DGS ----------
    print('本方案B: 3DGS-torch (4k高斯, 训练4视角)...')
    colors_init = lambert_colors(normals_sparse, np.eye(3),
                                 base_rgb=(0.75, 0.55, 0.45))
    model = GaussianModel.from_points(pts_sparse, colors=colors_init,
                                      device=device)
    train_views = [{'T_wc': make_T(R, np.array(t, dtype=float)),
                    'image': gts[name]}
                   for name, R, t in views_def if name != 'test']
    t0 = time.time()
    optimize_gaussians(model, train_views, K, iters=300,
                       lr={'mu': 1e-3, 'log_scale': 2e-2, 'rotvec': 5e-3,
                           'color': 1e-2, 'logit': 1e-2},
                       device=device)
    t_train = time.time() - t0
    print(f'训练耗时: {t_train:.1f}s')

    out_test = {}
    for name, R, t in views_def:
        T = make_T(R, np.array(t, dtype=float))
        out_test[name] = render_gaussians(model, T, K, H, W,
                                          device=device)['feat'].cpu().numpy()

    # ---------- 指标 ----------
    print('\n' + '=' * 66)
    print('指标 (对 pseudo-GT):')
    print(f'{"视角":>8} | {"PSNR A(泼溅)":>12} | {"PSNR B(3DGS)":>12} | '
          f'{"SSIM A":>8} | {"SSIM B":>8}')
    pa, pb, sa, sb = [], [], [], []
    metric_rows = []
    for name, R, t in views_def:
        p_a, p_b = psnr(base[name], gts[name]), psnr(out_test[name], gts[name])
        s_a = ssim_simple(base[name], gts[name])
        s_b = ssim_simple(out_test[name], gts[name])
        pa.append(p_a); pb.append(p_b); sa.append(s_a); sb.append(s_b)
        metric_rows.append({
            'view': name, 'split': 'test' if name == 'test' else 'train',
            'psnr_splat_db': float(p_a), 'psnr_3dgs_db': float(p_b),
            'ssim_splat': float(s_a), 'ssim_3dgs': float(s_b),
        })
        tag = ' ←测试(未训练)' if name == 'test' else ''
        print(f'{name:>8} | {p_a:12.2f} | {p_b:12.2f} | {s_a:8.4f} | '
              f'{s_b:8.4f}{tag}')
    print(f'{"平均":>8} | {np.mean(pa):12.2f} | {np.mean(pb):12.2f} | '
          f'{np.mean(sa):8.4f} | {np.mean(sb):8.4f}')

    # ---------- FPS ----------
    H2, W2 = 480, 640
    K2 = K0.copy()
    n_warm = 3
    for _ in range(n_warm):
        render_gaussians(model, make_T(np.eye(3), np.array([0, 0, 900.0])),
                         K2, H2, W2, device=device)
    torch.cuda.synchronize() if device == 'cuda' else None
    t0 = time.time()
    n_fps = 10
    for _ in range(n_fps):
        render_gaussians(model, make_T(np.eye(3), np.array([0, 0, 900.0])),
                         K2, H2, W2, device=device)
    torch.cuda.synchronize() if device == 'cuda' else None
    dt = (time.time() - t0) / n_fps
    print(f'\n渲染速度 (640×480, {len(model.mu)}高斯, torch参考实现): '
          f'{dt * 1000:.0f}ms/帧 = {1 / dt:.1f} FPS')
    print('  [对照] CUDA tile 光栅化器速率见 scripts/time_3dgs_cuda_rasterizer.py')

    # 保存训练后的高斯参数, 供 CUDA tile 光栅化器 (gsplat) 同场景计时
    torch.save({
        'mu': model.mu.detach().cpu(),
        'log_scale': model.log_scale.detach().cpu(),
        'rotvec': model.rotvec.detach().cpu(),
        'color': model.color.detach().cpu(),
        'logit': model.logit.detach().cpu(),
        'K_timing': K0, 'H_timing': H2, 'W_timing': W2,
        'T_timing': make_T(np.eye(3), np.array([0, 0, 900.0])),
        'seed': seed,
    }, OUT / 'gaussians.pt')

    # ---------- 对比图 ----------
    panels = []
    for name in ['train1', 'test']:
        row = np.concatenate([gts[name], base[name], out_test[name]], axis=1)
        panels.append(row)
    comp = (np.clip(np.concatenate(panels, axis=0), 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(str(OUT / 'compare.png'), cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
    with open(OUT / 'metrics.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()))
        w.writeheader()
        w.writerows(metric_rows)
    with open(OUT / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump({
            'protocol': {
                'seed': seed, 'gaussians': int(len(model.mu)),
                'train_views': 4, 'held_out_views': 1,
                'training_iterations': 300,
                'quality_resolution': [H, W],
                'timing_resolution': [H2, W2],
                'timing_warmup_frames': n_warm,
                'timing_frames': n_fps,
            },
            'train_psnr_3dgs_mean_db': float(np.mean(pb[:4])),
            'heldout_psnr_3dgs_db': float(pb[-1]),
            'train_ssim_3dgs_mean': float(np.mean(sb[:4])),
            'heldout_ssim_3dgs': float(sb[-1]),
            'render_time_ms': float(dt * 1000),
            'render_fps': float(1 / dt),
            'training_time_s': float(t_train),
        }, f, indent=2)
    print(f'对比图: {OUT / "compare.png"} (GT|泼溅|3DGS × train1/test)')
    print(f'结构化结果: {OUT / "metrics.csv"}, {OUT / "summary.json"}')
    print('=' * 66)


if __name__ == '__main__':
    main()
