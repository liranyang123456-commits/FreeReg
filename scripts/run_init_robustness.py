# -*- coding: utf-8 -*-
"""
实验: 初始化鲁棒性 (收敛域对比)
================================

对 GT 位姿施加逐渐增大的扰动 (旋转 5/10/15/20° + 平移 5/10/20mm),
比较三种精化器的收敛率 (ADD-S < 10%d):

  (a) Vanilla ICP (无 trim): 标准点到点 ICP
  (b) FreeReg trimmed ICP:  本文修剪 ICP (trim_ratio=0.7)
  (c) FreeReg full engine:  多起点 + 轮廓仲裁 (register_rgbd)

在 BOP Linemod 真实帧上评测. 展示 FreeReg 引擎对初始化误差的鲁棒性.

用法:
    conda activate py3d
    python scripts/run_init_robustness.py --scenes 1 2 --frames 8
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.geometry import exp_se3, backproject_depth, transform
from src.pipeline import (RegConfig, load_mesh_points, register_rgbd,
                          refine_icp, render_model)
from src.pose_estimation import icp, nearest_neighbors
from src.metrics import add_s_error

BOP_ROOT = Path(r'D:\bop\lm')
OUT = Path('outputs/init_robust')
ROT_LEVELS = [10, 20, 30, 45]   # degrees
TR_LEVELS = [10, 20, 30, 40]    # mm (与旋转档一一对应)


def vanilla_icp(src, dst, T_init, device, max_iter=60):
    """无修剪 ICP (trim_ratio=1.0)."""
    return icp(src, dst, T_init=T_init, max_iter=max_iter, trim_ratio=0.999,
               device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', type=int, nargs='+', default=[1, 2])
    ap.add_argument('--frames', type=int, default=8)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = RegConfig(device=args.device)
    models_info = json.load(open(BOP_ROOT / 'models' / 'models_info.json'))
    cache = {}

    def get_model(oid):
        if oid not in cache:
            cache[oid] = load_mesh_points(
                str(BOP_ROOT / 'models' / f'obj_{oid:06d}.ply'),
                n_points=15000)[0]
        return cache[oid]

    rng = np.random.default_rng(0)
    # 收集样本
    samples = []
    for sid in args.scenes:
        sdir = BOP_ROOT / 'test' / f'{sid:06d}'
        cams = json.load(open(sdir / 'scene_camera.json'))
        keys = sorted(cams.keys(), key=int)
        step = max(1, len(keys) // args.frames)
        for fk in keys[::step][:args.frames]:
            samples.append((sid, fk))

    # 方法: (name, fn(T_init)->T_est or None)
    results = {m: {r: [] for r in ROT_LEVELS} for m in
               ['vanilla', 'trimmed', 'engine']}

    for sid, fk in samples:
        sdir = BOP_ROOT / 'test' / f'{sid:06d}'
        cams = json.load(open(sdir / 'scene_camera.json'))
        gts = json.load(open(sdir / 'scene_gt.json'))
        if fk not in cams:
            continue
        K = np.array(cams[fk]['cam_K']).reshape(3, 3)
        depth = cv2.imread(str(sdir / 'depth' / f'{int(fk):06d}.png'),
                           cv2.IMREAD_UNCHANGED).astype(np.float64)
        depth *= cams[fk].get('depth_scale', 1.0)
        g = gts[fk][0]
        oid = g['obj_id']
        pts = get_model(oid)
        diam = models_info[str(oid)]['diameter']
        mv = sdir / 'mask_visib' / f'{int(fk):06d}_000000.png'
        mask = cv2.imread(str(mv), cv2.IMREAD_GRAYSCALE) if mv.exists() \
            else None
        scene_pts, _, _ = backproject_depth(depth, K, mask=mask)
        if len(scene_pts) < 100:
            continue
        model_icp = pts[::max(1, len(pts) // 3000)]
        ev = pts[::max(1, len(pts) // 2000)]

        T_gt = np.eye(4)
        T_gt[:3, :3] = np.array(g['cam_R_m2c']).reshape(3, 3)
        T_gt[:3, 3] = np.array(g['cam_t_m2c']).reshape(3)

        for li, rot_deg in enumerate(ROT_LEVELS):
            # 扰动: 旋转 rot_deg + 平移取同档
            tr_mm = TR_LEVELS[li]
            xi = rng.normal(0, 1, 6)
            xi[:3] *= tr_mm / (np.linalg.norm(xi[:3]) + 1e-9)
            xi[3:] *= np.deg2rad(rot_deg) / (np.linalg.norm(xi[3:]) + 1e-9)
            T_init = T_gt @ exp_se3(xi)

            # (a) vanilla ICP
            T_v, _ = vanilla_icp(model_icp, scene_pts, T_init,
                                 args.device)
            adds_v = add_s_error(ev, T_v, T_gt) if T_v is not None else 1e9
            results['vanilla'][rot_deg].append(adds_v < 0.1 * diam)

            # (b) trimmed ICP
            T_t, _ = refine_icp(model_icp, scene_pts, T_init, max_iter=60,
                                trim_ratio=0.7, device=args.device)
            adds_t = add_s_error(ev, T_t, T_gt) if T_t is not None else 1e9
            results['trimmed'][rot_deg].append(adds_t < 0.1 * diam)

            # (c) full engine (contour arbitration) — 仅对最大扰动测一次
            if rot_deg == ROT_LEVELS[-1]:
                res = register_rgbd(depth, K, pts, mask=mask, cfg=cfg)
                adds_e = add_s_error(ev, res['T'], T_gt) \
                    if res['T'] is not None else 1e9
                results['engine'][rot_deg].append(adds_e < 0.1 * diam)

    print('\n' + '=' * 60)
    print('初始化鲁棒性: 收敛率 (ADD-S < 10%d)')
    print('=' * 60)
    header = f'{"方法":<12s}' + ''.join(f'{r:>8d}°' for r in ROT_LEVELS)
    print(header)
    for m in ['vanilla', 'trimmed', 'engine']:
        line = f'{m:<12s}'
        for r in ROT_LEVELS:
            vals = results[m][r]
            line += f'{np.mean(vals):>8.0%}' if vals else f'{"--":>8s}'
        print(line)

    with open(OUT / 'init_robust.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['method', 'rot_deg', 'converge_rate'])
        for m in results:
            for r in ROT_LEVELS:
                v = results[m][r]
                if v:
                    w.writerow([m, r, f'{np.mean(v):.3f}'])
    print('\nCSV: %s' % (OUT / "init_robust.csv"))

    # ---------- Part 2: 气道管状结构 (轴向滑动歧义, 初始化真正敏感) ----------
    print('\n' + '=' * 60)
    print('气道管状结构初始化鲁棒性 (轴向滑动歧义)')
    print('=' * 60)
    airway_csv = OUT / 'init_robust_airway.csv'
    cache = Path('outputs/ct/airway_cache.npz')
    if not cache.exists():
        print('  (无气道缓存, 跳过)')
        return
    from src.ct_loader import make_flythrough_poses
    z = np.load(cache)
    pts = z['points'].astype(np.float64)
    axis_dir = z['axis_dir']

    class M: pass
    model = M()
    model.points, model.axis_pts, model.axis_dir = pts, z['axis_pts'], axis_dir
    poses_gt = make_flythrough_poses(model, n_frames=40, seed=0)
    T_gt = poses_gt[10]  # 中段帧
    H2, W2 = 240, 320
    K2 = np.array([[200., 0, 160.], [0, 200., 120.], [0, 0, 1.]])
    colors = np.tile(np.array([0.85, 0.65, 0.60]), (len(pts), 1))
    out = render_model(pts, colors, T_gt, K2, H2, W2, radius=1,
                       device=args.device)
    depth = out['depth'].cpu().numpy()
    fg = depth > 0
    depth[fg] += np.random.default_rng(1).normal(0, 0.8, int(fg.sum()))
    scene_pts, _, _ = backproject_depth(depth, K2, mask=fg)
    model_icp = pts[::max(1, len(pts) // 3000)]
    ev = pts[::max(1, len(pts) // 2000)]

    # 沿气道主轴方向施加轴向偏移 (mm) —— 管状滑动是主要歧义
    AX_LEVELS = [0, 5, 10, 20, 30]
    ax_rows = {m: [] for m in ['vanilla', 'trimmed']}
    for ax in AX_LEVELS:
        T_init = T_gt.copy()
        T_init[:3, 3] += axis_dir * ax
        # 加少量横向扰动
        T_init[:3, 3] += np.random.default_rng(0).normal(0, 2.0, 3)
        for m, fn in [('vanilla', lambda: vanilla_icp(
                          model_icp, scene_pts, T_init, args.device)),
                      ('trimmed', lambda: refine_icp(
                          model_icp, scene_pts, T_init, max_iter=40,
                          trim_ratio=0.8, device=args.device))]:
            T_e, _ = fn()
            if T_e is None:
                ax_rows[m].append(999.0)
                continue
            Pe = transform(T_e, ev)
            Pg = transform(T_gt, ev)
            _, d = nearest_neighbors(Pe, Pg, device=args.device)
            ax_rows[m].append(float(d.mean()))
        print(f'  轴向偏移 {ax:2d}mm: vanilla={ax_rows["vanilla"][-1]:6.2f}mm  '
              f'trimmed={ax_rows["trimmed"][-1]:6.2f}mm')

    with open(airway_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['axial_mm', 'vanilla_tre', 'trimmed_tre'])
        for i, ax in enumerate(AX_LEVELS):
            w.writerow([ax, f'{ax_rows["vanilla"][i]:.3f}',
                        f'{ax_rows["trimmed"][i]:.3f}'])
    print(f'  CSV: {airway_csv}')


if __name__ == '__main__':
    main()
