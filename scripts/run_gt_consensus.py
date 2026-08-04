# -*- coding: utf-8 -*-
"""
真实视频人工 GT —— ③ 双人共识合并与评测
==========================================

合并 annotations_R1/R2.csv:
  逐帧逐标签配对 → |Δ| ≤ 15px 取均值作 GT, 否则标记仲裁
输出: gt_landmarks.csv + 一致性统计 + (可选) 注册位姿重投影误差
--demo: 合成双人标注 (提案点+N(0,5px)) 端到端验证流程

用法:
    python scripts/run_gt_consensus.py [--demo]
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import sys
import json
import argparse
import csv
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.geometry import transform

OUT = Path('outputs/gt_pkg')
AGREE_PX = 15.0


def load_annotations(reader):
    p = OUT / f'annotations_{reader}.csv'
    if not p.exists():
        return {}
    rows = {}
    with open(p, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows.setdefault(r['label'], []).append(
                (int(r['frame']), float(r['u']), float(r['v'])))
    return rows


def consensus(r1, r2):
    """逐标签配对 (同帧同标签), 返回 GT 与一致性统计"""
    gt, disa = [], []
    for label in sorted(set(r1) | set(r2)):
        a1 = r1.get(label, [])
        a2 = r2.get(label, [])
        used2 = set()
        for f1, u1, v1 in a1:
            best, bd = None, np.inf
            for j, (f2, u2, v2) in enumerate(a2):
                if f2 != f1 or j in used2:
                    continue
                d = float(np.hypot(u1 - u2, v1 - v2))
                if d < bd:
                    best, bd = j, d
            if best is not None and bd <= AGREE_PX:
                f2, u2, v2 = a2[best]
                used2.add(best)
                gt.append(dict(frame=f1, u=(u1 + u2) / 2, v=(v1 + v2) / 2,
                               label=label, disa=bd))
                disa.append(bd)
            else:
                gt.append(dict(frame=f1, u=u1, v=v1, label=label,
                               disa=np.nan))
    return gt, np.array(disa)


def eval_reprojection(gt):
    """用提案位姿把 CT 分叉投影 → 与 GT 地标最近距离 (重投影误差)"""
    junc = {}
    with open(OUT / 'ct_junctions.csv', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            junc[int(r['id'])] = np.array([float(r['x_mm']),
                                           float(r['y_mm']),
                                           float(r['z_mm'])])
    jpts = np.array(list(junc.values()))
    frames_meta = {}
    with open(OUT / 'frames.csv', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            frames_meta[int(r['frame'])] = json.loads(r['pose_json'])
    errs = []
    for g in gt:
        T = np.array(frames_meta.get(g['frame'], np.eye(4).tolist()))
        if T.shape != (4, 4) or g['frame'] not in frames_meta:
            continue
        # 视频内参: 与提案一致 (fx=W/2, 视频为 1920x2400? 按 frames 尺寸)
        bgr_shape = (2400, 1920)  # 视频原始 H,W (probe 所得)
        K = np.array([[1920 / 2., 0, 960], [0, 1920 / 2., 1200],
                      [0, 0, 1.0]])
        P = transform(T, jpts)
        z = P[:, 2]
        ok = z > 1e-3
        u = K[0, 0] * P[ok, 0] / z[ok] + K[0, 2]
        v = K[1, 1] * P[ok, 1] / z[ok] + K[1, 2]
        inb = (u >= 0) & (u < 1920) & (v >= 0) & (v < 2400)
        if inb.sum() == 0:
            continue
        d = np.sqrt((u[inb] - g['u']) ** 2 + (v[inb] - g['v']) ** 2).min()
        errs.append(float(d))
    return np.array(errs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--demo', action='store_true')
    args = ap.parse_args()

    if args.demo:
        # 合成双人标注: R1=提案投影点, R2=R1+N(0,5px)
        rng = np.random.default_rng(0)
        junc_rows = list(csv.DictReader(
            open(OUT / 'ct_junctions.csv', encoding='utf-8')))
        for reader, sig in [('R1', 0.0), ('R2', 5.0)]:
            with open(OUT / f'annotations_{reader}.csv', 'w',
                      newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(['frame', 'u', 'v', 'label'])
                for j, r in enumerate(junc_rows[:30]):
                    fi = 10000 + j
                    u = 960 + (j % 6 - 2.5) * 120 + rng.normal(0, sig)
                    v = 1200 + (j // 6 - 2) * 150 + rng.normal(0, sig)
                    lab = ['carina', 'right_main', 'left_main',
                           'right_lobar', 'left_lobar'][j % 5]
                    w.writerow([fi, round(u, 1), round(v, 1), lab])
        print('[demo] 已生成合成双人标注')

    r1 = load_annotations('R1')
    r2 = load_annotations('R2')
    if not r1 or not r2:
        print('缺少双人标注, 先各自运行 run_gt_annotate.py '
              '(或用 --demo 验证流程)')
        return
    gt, disa = consensus(r1, r2)
    n_agree = int(np.isfinite(disa).sum()) if len(disa) else 0
    print(f'\n共识: 总标记 {len(gt)}, 双人一致 {n_agree} '
          f'({100 * n_agree / max(len(gt), 1):.0f}%), '
          f'需仲裁 {len(gt) - n_agree}')
    if len(disa):
        print(f'  一致点双人偏差: mean={np.nanmean(disa):.1f}px '
              f'median={np.nanmedian(disa):.1f}px max={np.nanmax(disa):.1f}px')
    with open(OUT / 'gt_landmarks.csv', 'w', newline='',
              encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['frame', 'u', 'v', 'label',
                                          'disa'])
        w.writeheader()
        w.writerows(gt)
    print(f'  GT 地标: {OUT / "gt_landmarks.csv"}')

    errs = eval_reprojection(gt)
    if len(errs):
        print(f'\n注册重投影误差 (提案位姿 vs GT地标): '
              f'mean={errs.mean():.1f}px 中位={np.median(errs):.1f}px '
              f'(n={len(errs)})')
    print('=' * 66)


if __name__ == '__main__':
    main()
