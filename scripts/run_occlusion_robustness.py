# -*- coding: utf-8 -*-
"""
实验: 遮挡鲁棒性 (BOP Linemod 可见比例分箱)
==============================================

BOP 官方 scene_gt_info.json 提供每个目标实例的 visib_fract.
按可见比例分箱 [0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0], 分别统计:
  ADD-S AR@10%d, 成功率 (ADD-S < 10%d), 旋转误差, 重投影 RMSE.

验证 FreeReg 的轮廓仲裁+trimmed ICP 在遮挡加重时的退化曲线.

用法:
    conda activate py3d
    python scripts/run_occlusion_robustness.py --scenes 1 2 3 4 5
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

from src.pipeline import RegConfig, load_mesh_points, register_rgbd
from src.metrics import add_s_error, pose_errors

BOP_ROOT = Path(r'D:\bop\lm')
OUT = Path('outputs/occlusion')
BINS = [(0.0, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
# 合成遮挡: 随机遮挡块覆盖目标 mask 的比例 (Linemod 真实可见比偏高,
# 故用合成遮挡补足 20/40/60/80% 遮挡档位)
SYNTH_OCC = [0.0, 0.2, 0.4, 0.6]


def apply_synth_occlusion(depth, mask, occ_frac, rng):
    """在目标 mask 上随机放置矩形遮挡块, 直到覆盖 occ_frac 的可见像素."""
    if mask is None or occ_frac <= 0:
        return depth, mask
    d = depth.copy()
    m = mask.copy()
    ys, xs = np.where(m > 0)
    if len(xs) < 50:
        return d, m
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    target = int(occ_frac * len(xs))
    occ_mask = np.zeros_like(m)
    covered = 0
    tries = 0
    while covered < target and tries < 60:
        tries += 1
        bw = rng.integers((x1 - x0) // 5, (x1 - x0) // 2 + 1)
        bh = rng.integers((y1 - y0) // 5, (y1 - y0) // 2 + 1)
        bx = rng.integers(x0, max(x1 - bw, x0 + 1))
        by = rng.integers(y0, max(y1 - bh, y0 + 1))
        block = np.zeros_like(m)
        block[by:by + bh, bx:bx + bw] = 255
        new = (block > 0) & (m > 0) & (occ_mask == 0)
        covered += int(new.sum())
        occ_mask |= block
    # 遮挡块: 深度设为更近 (模拟前景遮挡), mask 中被遮挡部分移除
    d[occ_mask > 0] = np.maximum(d[occ_mask > 0] - 100, 50)
    m[occ_mask > 0] = 0
    return d, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    ap.add_argument('--max_per_bin', type=int, default=30)
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

    # 收集所有 (scene, frame, inst) 按可见比分箱
    bin_samples = {b: [] for b in range(len(BINS))}
    for sid in args.scenes:
        sdir = BOP_ROOT / 'test' / f'{sid:06d}'
        infos = json.load(open(sdir / 'scene_gt_info.json'))
        for fk, arr in infos.items():
            for inst, info in enumerate(arr):
                vf = info['visib_fract']
                for bi, (lo, hi) in enumerate(BINS):
                    if lo <= vf < hi:
                        bin_samples[bi].append((sid, fk, inst, vf))
                        break

    rows = []
    for bi, (lo, hi) in enumerate(BINS):
        samples = bin_samples[bi]
        rng = np.random.default_rng(0)
        if len(samples) > args.max_per_bin:
            idx = rng.choice(len(samples), args.max_per_bin, replace=False)
            samples = [samples[i] for i in sorted(idx)]
        print(f'\n=== 可见比 [{lo:.1f},{hi:.1f}): {len(samples)} 样本 ===')
        adds_list, rot_list, succ_list = [], [], []
        for sid, fk, inst, vf in samples:
            sdir = BOP_ROOT / 'test' / f'{sid:06d}'
            cams = json.load(open(sdir / 'scene_camera.json'))
            gts = json.load(open(sdir / 'scene_gt.json'))
            if fk not in cams:
                continue
            K = np.array(cams[fk]['cam_K']).reshape(3, 3)
            depth = cv2.imread(str(sdir / 'depth' / f'{int(fk):06d}.png'),
                               cv2.IMREAD_UNCHANGED).astype(np.float64)
            depth *= cams[fk].get('depth_scale', 1.0)
            g = gts[fk][inst]
            oid = g['obj_id']
            pts = get_model(oid)
            diam = models_info[str(oid)]['diameter']
            mv = sdir / 'mask_visib' / f'{int(fk):06d}_{inst:06d}.png'
            mask = cv2.imread(str(mv), cv2.IMREAD_GRAYSCALE) \
                if mv.exists() else None

            res = register_rgbd(depth, K, pts, mask=mask, cfg=cfg)
            if res['T'] is None:
                succ_list.append(0.0)
                continue
            T_est = res['T']
            T_gt = np.eye(4)
            T_gt[:3, :3] = np.array(g['cam_R_m2c']).reshape(3, 3)
            T_gt[:3, 3] = np.array(g['cam_t_m2c']).reshape(3)
            ev = pts[::max(1, len(pts) // 2000)]
            adds = add_s_error(ev, T_est, T_gt)
            rot, _ = pose_errors(T_est, T_gt)
            adds_list.append(adds)
            rot_list.append(rot)
            succ_list.append(1.0 if adds < 0.1 * diam else 0.0)

        if adds_list:
            adds_a = np.array(adds_list)
            ar = float((adds_a < 0.1).mean())  # placeholder norm
            # 用 ADD-S<10%d 成功率作为 AR
            row = dict(bin=f'[{lo:.1f},{hi:.1f})', n=len(adds_list),
                       succ=float(np.mean(succ_list)),
                       rot_med=float(np.median(rot_list)),
                       adds_med=float(np.median(adds_list)))
            rows.append(row)
            print(f'  n={row["n"]}  成功率={row["succ"]:.0%}  '
                  f'rot中位={row["rot_med"]:.2f}°  '
                  f'ADD-S中位={row["adds_med"]:.2f}mm')

    # ---------- Part 2: 合成遮挡扫档 (补足 Linemod 低遮挡缺失) ----------
    print('\n' + '=' * 60)
    print('合成遮挡扫档 (Linemod 帧, 人为遮挡块覆盖目标比例)')
    print('=' * 60)
    # 用高可见帧, 人为加遮挡
    flat = []
    for bi in range(len(BINS)):
        flat.extend(bin_samples[bi])
    rng2 = np.random.default_rng(3)
    if len(flat) > 60:
        flat = [flat[i] for i in sorted(rng2.choice(len(flat), 60,
                                                    replace=False))]
    occ_rows = []
    for occ in SYNTH_OCC:
        succ_l, adds_l, rot_l = [], [], []
        for sid, fk, inst, vf in flat:
            sdir = BOP_ROOT / 'test' / f'{sid:06d}'
            cams = json.load(open(sdir / 'scene_camera.json'))
            gts = json.load(open(sdir / 'scene_gt.json'))
            if fk not in cams:
                continue
            K = np.array(cams[fk]['cam_K']).reshape(3, 3)
            depth = cv2.imread(str(sdir / 'depth' / f'{int(fk):06d}.png'),
                               cv2.IMREAD_UNCHANGED).astype(np.float64)
            depth *= cams[fk].get('depth_scale', 1.0)
            g = gts[fk][inst]
            oid = g['obj_id']
            pts = get_model(oid)
            diam = models_info[str(oid)]['diameter']
            mv = sdir / 'mask_visib' / f'{int(fk):06d}_{inst:06d}.png'
            mask = cv2.imread(str(mv), cv2.IMREAD_GRAYSCALE) \
                if mv.exists() else None
            d_occ, m_occ = apply_synth_occlusion(depth, mask, occ, rng2)
            res = register_rgbd(d_occ, K, pts, mask=m_occ, cfg=cfg)
            if res['T'] is None:
                succ_l.append(0.0); continue
            T_est = res['T']
            T_gt = np.eye(4)
            T_gt[:3, :3] = np.array(g['cam_R_m2c']).reshape(3, 3)
            T_gt[:3, 3] = np.array(g['cam_t_m2c']).reshape(3)
            ev = pts[::max(1, len(pts) // 2000)]
            adds = add_s_error(ev, T_est, T_gt)
            rot, _ = pose_errors(T_est, T_gt)
            adds_l.append(adds); rot_l.append(rot)
            succ_l.append(1.0 if adds < 0.1 * diam else 0.0)
        if adds_l:
            occ_rows.append(dict(occ=f'{occ:.0%}', n=len(adds_l),
                                 succ=float(np.mean(succ_l)),
                                 rot_med=float(np.median(rot_l)),
                                 adds_med=float(np.median(adds_l))))
            print(f'  遮挡{occ:4.0%}: n={len(adds_l)}  '
                  f'成功率={np.mean(succ_l):.0%}  '
                  f'rot中位={np.median(rot_l):.2f}°  '
                  f'ADD-S中位={np.median(adds_l):.2f}mm')

    with open(OUT / 'occlusion_synth.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['occ', 'n', 'succ', 'rot_med',
                                          'adds_med'])
        w.writeheader()
        w.writerows(occ_rows)
    print(f'\nCSV: {OUT / "occlusion_synth.csv"}')


if __name__ == '__main__':
    main()
