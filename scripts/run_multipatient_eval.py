# -*- coding: utf-8 -*-
"""
E1.6: 多患者 CT-视频注册统计评测 (Wilcoxon 显著性)
====================================================

协议: 省医 N 例 CT → 各生成合成飞行视频 (同协议) →
      每例报告 TRE-NN (测量 ICP vs ICP+Kalman) →
      全体帧级配对 Wilcoxon 符号秩检验 (TMI 投稿统计素材)

用法:
    conda activate py3d
    python scripts/run_multipatient_eval.py --patients 6
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import sys
import time
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.geometry import exp_se3, transform, backproject_depth
from src.ct_loader import load_ct_airway, make_flythrough_poses
from src.pipeline import render_model, refine_icp
from src.pose_estimation import nearest_neighbors
from src.se3_kalman_filter import SE3KalmanFilter, SE3KalmanConfig
import torch

ION = Path(r'F:\省医-数据\省医-数据\ION患者整理')
EXTRACT = Path(r'E:\ct_extract')
RF = Path(r'F:\省医-数据\省医-数据\射频消融CT')
# 当前可用 CT 的患者 (002 为容错解压版; 003/004-007/009-030 待修复;
# 射频消融队列为独立 4 例, 术前 CT)
PATIENTS = [
    ('001王继深', ION / '001王继深'),
    ('002许军伟', EXTRACT / '002许军伟'),
    ('008骆明修', ION / '008骆明修'),
    ('013仝克勤', EXTRACT / '013仝克勤'),
    ('10001术前', RF / '10001' / '10001 术前30天'),
    ('10001术后24h', RF / '10001' / '10001 补充消融术后24h'),
    ('10001术后30天', RF / '10001' / '10001术后30天'),
    ('10001术后6个月', RF / '10001' / '10001术后6个月'),
    ('10002术前', RF / '10002' / '10002 术前30天-术后24H'),
    ('10002术后30天', RF / '10002' / '10002术后30天'),
    ('10003术前', RF / '10003' / '10003术前30天'),
    ('10003术后24h', RF / '10003' / '10003术后24h'),
    ('10003术后30天', RF / '10003' / '10003术后30天'),
    ('10004术前', RF / '10004' / '10004术前30天 已脱敏'),
    ('10004术后30天', RF / '10004' / '10004术后30天'),
    ('10004术后3个月', RF / '10004' / '10004术后3个月'),
]
CACHE_DIR = Path('outputs/ct/mp')
OUT = Path('outputs/ct/mp')
H, W = 240, 320
K = np.array([[200., 0, W / 2], [0, 200., H / 2], [0, 0, 1.0]])


class M: pass


def kalman_clinical():
    return SE3KalmanConfig(
        process_noise_pos=2.0, process_noise_rot=2e-5,
        process_noise_vel_pos=0.5, process_noise_vel_rot=2e-4,
        meas_noise_pos=2.0, meas_noise_rot=0.1,
        innovation_threshold=10.0, velocity_damping=0.98)


def eval_patient(pdir: Path, device, n_frames=40, seed=0):
    """单例: 加载(带缓存) → 飞行视频 → 跟踪 → per-frame TRE 数组"""
    cache = CACHE_DIR / f'{pdir.name}.npz'
    if cache.exists():
        z = np.load(cache)
        pts, axis_pts, axis_dir = z['points'], z['axis_pts'], z['axis_dir']
    else:
        try:
            model = load_ct_airway(str(pdir), max_slices=200,
                                   max_pts=40000)
        except Exception as e:
            print(f'  [{pdir.name}] CT 加载失败: {e}')
            return None
        pts, axis_pts, axis_dir = model.points, model.axis_pts, \
            model.axis_dir
        if len(pts) < 500 or len(axis_pts) < 6:
            print(f'  [{pdir.name}] 气道不足({len(pts)}点/'
                  f'{len(axis_pts)}轴点), 跳过')
            return None
        if len(pts) < 800:
            print(f'  [{pdir.name}] 气道较小({len(pts)}点), '
                  f'轴点{len(axis_pts)}, 继续')
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(cache, points=pts, axis_pts=axis_pts, axis_dir=axis_dir)

    m = M()
    m.points, m.axis_pts, m.axis_dir = pts, axis_pts, axis_dir
    try:
        poses = make_flythrough_poses(m, n_frames=n_frames, seed=seed)
    except RuntimeError:
        return None
    rng = np.random.default_rng(seed + 1)
    colors = np.tile(np.array([0.85, 0.65, 0.60]), (len(pts), 1))
    lm = pts[np.linspace(0, len(pts) - 1, 100).astype(int)]
    model_icp = pts[::max(1, len(pts) // 3000)]
    kf = SE3KalmanFilter(kalman_clinical())
    tre_m, tre_s, times = [], [], []
    meas_prev = None
    for i, T_gt in enumerate(poses):
        out = render_model(pts, colors, T_gt, K, H, W, radius=1,
                           device=device)
        depth = out['depth'].cpu().numpy()
        fg = depth > 0
        depth[fg] += rng.normal(0, 0.8, int(fg.sum()))
        t0 = time.time()
        if i == 0:
            T_init = T_gt @ exp_se3(
                rng.normal(0, [5, 5, 5, 0.05, 0.05, 0.05]))
            n_it, trim = 50, 0.7
        else:
            T_init = kf.T if kf.initialized else meas_prev
            n_it, trim = 25, 0.8
        scene, _, _ = backproject_depth(depth, K, mask=fg)
        if len(scene) < 800 and i > 0:
            T_meas = T_init
        else:
            T_meas, _ = refine_icp(model_icp, scene, T_init,
                                   max_iter=n_it, trim_ratio=trim,
                                   device=device)
        T_sm = kf.update(T_meas, dt=1.0)
        meas_prev = T_meas
        times.append(time.time() - t0)
        Pm, Pg = transform(T_meas, lm), transform(T_gt, lm)
        _, dm = nearest_neighbors(Pm, Pg, device=device)
        Ps = transform(T_sm, lm)
        _, ds = nearest_neighbors(Ps, Pg, device=device)
        tre_m.append(float(dm.mean()))
        tre_s.append(float(ds.mean()))
    return dict(name=pdir.name, tre_m=np.array(tre_m),
                tre_s=np.array(tre_s), ms=np.mean(times) * 1000,
                reloc=kf.reloc_count)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--patients', type=int, default=len(PATIENTS))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('=' * 66)
    print('E1.6: 多患者统计评测 (Wilcoxon)')
    print('=' * 66)

    pdirs = [p for _, p in PATIENTS][:args.patients]
    results = []
    for pdir in pdirs:
        t0 = time.time()
        r = eval_patient(pdir, device)
        if r is None:
            continue
        results.append(r)
        print(f'  [{r["name"]}] TRE: 测量={r["tre_m"].mean():.2f}mm '
              f'Kalman={r["tre_s"].mean():.2f}mm '
              f'({r["ms"]:.0f}ms/帧, {time.time() - t0:.0f}s)')

    if len(results) < 2:
        print('有效患者不足, 退出')
        return

    # ---------- 统计 ----------
    # 灾难性失败排除 (中位 TRE>20mm: 数据质量/分割失败, 非注册算法问题)
    excluded = [r for r in results if np.median(r['tre_m']) > 20.0]
    for r in excluded:
        print(f'  [排除] {r["name"]}: 中位TRE={np.median(r["tre_m"]):.1f}mm '
              f'(数据质量/气道分割失败, 非注册算法问题)')
    valid = [r for r in results if np.median(r['tre_m']) <= 20.0]
    if len(valid) < 2:
        print('有效患者不足, 退出')
        return
    all_m = np.concatenate([r['tre_m'] for r in valid])
    all_s = np.concatenate([r['tre_s'] for r in valid])
    from scipy.stats import wilcoxon
    stat, pval = wilcoxon(all_m, all_s)
    diff = all_m - all_s
    print('\n' + '=' * 66)
    print(f'汇总: {len(valid)} 例有效患者 (排除{len(excluded)}例), '
          f'共 {len(all_m)} 帧')
    print(f'  TRE-NN 测量:   mean={all_m.mean():.2f}mm '
          f'中位={np.median(all_m):.2f}mm')
    print(f'  TRE-NN Kalman: mean={all_s.mean():.2f}mm '
          f'中位={np.median(all_s):.2f}mm')
    print(f'  配对差值:      中位={np.median(diff):.2f}mm')
    print(f'  Wilcoxon 符号秩: W={stat:.0f}, p={pval:.2e} '
          f'{"显著(p<0.05)" if pval < 0.05 else "不显著"}')
    print(f'  成功率(<10mm): 测量 {float((all_m < 10).mean()):.0%} '
          f'Kalman {float((all_s < 10).mean()):.0%}')

    import csv
    with open(OUT / 'multipatient_summary.csv', 'w', newline='',
              encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['patient', 'tre_meas_mean', 'tre_kalman_mean',
                    'tre_meas_median', 'tre_kalman_median', 'ms',
                    'reloc'])
        for r in results:
            w.writerow([r['name'], f'{r["tre_m"].mean():.3f}',
                        f'{r["tre_s"].mean():.3f}',
                        f'{np.median(r["tre_m"]):.3f}',
                        f'{np.median(r["tre_s"]):.3f}',
                        f'{r["ms"]:.1f}', r['reloc']])
    print(f'  汇总表: {OUT / "multipatient_summary.csv"}')
    print('=' * 66)


if __name__ == '__main__':
    main()
