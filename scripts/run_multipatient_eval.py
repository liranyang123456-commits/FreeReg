# -*- coding: utf-8 -*-
"""
E1.6: 多轨迹 CT-视频注册统计评测
====================================================

协议: 省医 N 例 CT → 各生成合成飞行视频 (同协议) →
      每条轨迹报告 TRE-NN (测量 ICP vs ICP+Kalman) →
      探索性帧级配对统计 (帧内相关，不作独立样本推断)

用法:
    conda activate py3d
    python scripts/run_multipatient_eval.py --patients 6
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import sys
import time
import argparse
import csv
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.geometry import exp_se3, transform, backproject_depth
from src.ct_loader import load_ct_airway, make_flythrough_poses
from src.pipeline import render_model, refine_icp
from src.pose_estimation import nearest_neighbors
from src.se3_kalman_filter import SE3KalmanFilter, SE3KalmanConfig
import torch

ION = Path(os.environ.get(
    'FREEREG_ION_ROOT', r'F:\省医-数据\省医-数据\ION患者整理'))
EXTRACT = Path(os.environ.get('FREEREG_CT_EXTRACT_ROOT', r'E:\ct_extract'))
RF = Path(os.environ.get(
    'FREEREG_RF_ROOT', r'F:\省医-数据\省医-数据\射频消融CT'))


def resolve_directory(base: Path, pattern: str) -> Path:
    """Resolve a private directory without embedding participant names.

    When the private drive is not mounted the returned sentinel keeps the glob
    pattern so that the extracted-airway cache written by the audited run
    (outputs/ct/mp/<source-dir-name>.npz) can still be located by
    ``find_cached_airway``; the cached points and axis are what the evaluation
    consumes, so the result is unchanged.
    """
    matches = sorted(path for path in base.glob(pattern) if path.is_dir())
    return matches[0] if matches else base / f'__missing__{pattern}'


def find_cached_airway(pdir: Path, cache_dir: Path):
    """Return the cache file for a (possibly unavailable) source directory."""
    if not pdir.name.startswith('__missing__'):
        return None
    pattern = pdir.name[len('__missing__'):]
    prefix = pdir.parent.name if pdir.parent.name.isdigit() else ''
    hits = sorted(cache_dir.glob(f'{prefix}{pattern}.npz'))
    return hits[0] if hits else None


# Ordered private-data manifest.  Output identifiers are generated separately
# as trajectory_01, trajectory_02, ... and never contain source directory names.
PATIENTS = [
    resolve_directory(ION, '001*'),
    resolve_directory(EXTRACT, '002*'),
    resolve_directory(ION, '008*'),
    resolve_directory(EXTRACT, '013*'),
    resolve_directory(RF / '10001', '*术前30天'),
    resolve_directory(RF / '10001', '*补充消融术后24h'),
    resolve_directory(RF / '10001', '*术后30天'),
    resolve_directory(RF / '10001', '*术后6个月'),
    resolve_directory(RF / '10002', '*术前30天*'),
    resolve_directory(RF / '10002', '*术后30天'),
    resolve_directory(RF / '10003', '*术前30天'),
    resolve_directory(RF / '10003', '*术后24h'),
    resolve_directory(RF / '10003', '*术后30天'),
    resolve_directory(RF / '10004', '*术前30天*'),
    resolve_directory(RF / '10004', '*术后30天'),
    resolve_directory(RF / '10004', '*术后3个月'),
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


def eval_trajectory(pdir: Path, case_id: str, device, n_frames=40, seed=0):
    """单例: 加载(带缓存) → 飞行视频 → 跟踪 → per-frame TRE 数组"""
    cache = CACHE_DIR / f'{case_id}.npz'
    legacy_cache = CACHE_DIR / f'{pdir.name}.npz'
    if not cache.exists() and legacy_cache.exists():
        cache = legacy_cache
    if not cache.exists():
        hit = find_cached_airway(pdir, CACHE_DIR)
        if hit is not None:
            cache = hit
    if cache.exists():
        z = np.load(cache)
        pts, axis_pts, axis_dir = z['points'], z['axis_pts'], z['axis_dir']
    else:
        try:
            model = load_ct_airway(str(pdir), max_slices=200,
                                   max_pts=40000)
        except Exception as e:
            print(f'  [{case_id}] CT 加载失败: {type(e).__name__}')
            return None
        pts, axis_pts, axis_dir = model.points, model.axis_pts, \
            model.axis_dir
        if len(pts) < 500 or len(axis_pts) < 6:
            print(f'  [{case_id}] 气道不足({len(pts)}点/'
                  f'{len(axis_pts)}轴点), 跳过')
            return None
        if len(pts) < 800:
            print(f'  [{case_id}] 气道较小({len(pts)}点), '
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
    return dict(name=case_id, tre_m=np.array(tre_m),
                tre_s=np.array(tre_s), ms=np.mean(times) * 1000,
                reloc=kf.reloc_count)


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument('--patients', type=int, default=len(PATIENTS))
    ap.add_argument('--seed', type=int, default=0,
                    help='飞行轨迹生成 / 深度噪声 / 帧0初始化种子 (审计运行为 0)')
    ap.add_argument('--out', default=str(OUT),
                    help='输出目录 (CT 缓存仍读自 outputs/ct/mp)')
    args = ap.parse_args()
    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('=' * 66)
    print(f'E1.6: 多轨迹统计评测 seed={args.seed}')
    print('=' * 66)

    pdirs = [
        (f'trajectory_{index + 1:02d}', pdir)
        for index, pdir in enumerate(PATIENTS[:args.patients])
    ]
    results = []
    for case_id, pdir in pdirs:
        t0 = time.time()
        r = eval_trajectory(pdir, case_id, device, seed=args.seed)
        if r is None:
            continue
        results.append(r)
        print(f'  [{r["name"]}] TRE: 测量={r["tre_m"].mean():.2f}mm '
              f'Kalman={r["tre_s"].mean():.2f}mm '
              f'({r["ms"]:.0f}ms/帧, {time.time() - t0:.0f}s)')

    if len(results) < 2:
        print('有效轨迹不足, 退出')
        return

    # ---------- 统计 ----------
    # 指定的主分析集规则；不对超阈值原因作因果归因。
    excluded = [r for r in results if np.median(r['tre_m']) > 20.0]
    for r in excluded:
        print(f'  [排除] {r["name"]}: 中位TRE={np.median(r["tre_m"]):.1f}mm '
              f'(超过指定的20mm主分析阈值)')
    valid = [r for r in results if np.median(r['tre_m']) <= 20.0]
    if len(valid) < 2:
        print('有效轨迹不足, 退出')
        return
    all_m_evaluable = np.concatenate([r['tre_m'] for r in results])
    all_s_evaluable = np.concatenate([r['tre_s'] for r in results])
    all_m = np.concatenate([r['tre_m'] for r in valid])
    all_s = np.concatenate([r['tre_s'] for r in valid])
    from scipy.stats import wilcoxon
    stat, pval = wilcoxon(all_m, all_s)
    diff = all_m - all_s
    print('\n' + '=' * 66)
    print(f'汇总: 主分析集{len(valid)}条CT轨迹 '
          f'(按预声明阈值排除{len(excluded)}条), '
          f'共 {len(all_m)} 帧')
    print(f'  TRE-NN 测量:   mean={all_m.mean():.2f}mm '
          f'中位={np.median(all_m):.2f}mm')
    print(f'  TRE-NN Kalman: mean={all_s.mean():.2f}mm '
          f'中位={np.median(all_s):.2f}mm')
    print(f'  配对差值:      中位={np.median(diff):.2f}mm')
    print(f'  探索性逐帧 Wilcoxon: W={stat:.0f}, p={pval:.2e} '
          '(同一轨迹内帧相关，不作独立样本推断)')
    print(f'  成功率(<10mm): 测量 {float((all_m < 10).mean()):.0%} '
          f'Kalman {float((all_s < 10).mean()):.0%}')

    with open(OUT / 'multipatient_summary.csv', 'w', newline='',
              encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['trajectory_id', 'tre_measured_mean_mm', 'tre_filtered_mean_mm',
                    'tre_measured_median_mm', 'tre_filtered_median_mm',
                    'runtime_ms_per_frame',
                    'reloc'])
        for r in results:
            w.writerow([r['name'], f'{r["tre_m"].mean():.3f}',
                        f'{r["tre_s"].mean():.3f}',
                        f'{np.median(r["tre_m"]):.3f}',
                        f'{np.median(r["tre_s"]):.3f}',
                        f'{r["ms"]:.1f}', r['reloc']])
    with open(OUT / 'multipatient_frames.csv', 'w', newline='',
              encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['trajectory_id', 'frame', 'tre_measured_mm',
                    'tre_filtered_mm', 'analysis_status'])
        valid_names = {r['name'] for r in valid}
        for r in results:
            status = 'included' if r['name'] in valid_names else \
                    'excluded_analysis_rule_median_gt_20mm'
            for frame, (tre_m, tre_s) in enumerate(zip(r['tre_m'], r['tre_s'])):
                w.writerow([r['name'], frame, f'{tre_m:.9f}',
                            f'{tre_s:.9f}', status])
    summary = {
        'protocol': {
            'requested_ct_directories': args.patients,
            'frames_requested_per_trajectory': 40,
            'seed': args.seed,
            'exclusion_rule': (
                'exclude trajectory if median measured TRE > 20 mm'
            ),
            'gate': {
                'score': 'norm(translation)/sigma_p + '
                         'norm(rotation)/sigma_R',
                'threshold': 10.0,
                'max_prediction_only_rejections_before_reset': 5,
                'sigma_p_mm': 2.0,
                'sigma_R_rad': 0.1,
            },
        },
        'trajectories_evaluated': len(results),
        'trajectories_included': len(valid),
        'trajectories_excluded': len(excluded),
        'included_trajectory_ids': [r['name'] for r in valid],
        'excluded_trajectory_ids': [r['name'] for r in excluded],
        'frames_included': int(len(all_m)),
        'frames_all_evaluable': int(len(all_m_evaluable)),
        'tre_measured_mean_mm': float(all_m.mean()),
        'tre_measured_median_mm': float(np.median(all_m)),
        'tre_filtered_mean_mm': float(all_s.mean()),
        'tre_filtered_median_mm': float(np.median(all_s)),
        'success_measured_lt10': float((all_m < 10).mean()),
        'success_filtered_lt10': float((all_s < 10).mean()),
        'all_evaluable_tre_measured_mean_mm': float(
            all_m_evaluable.mean()
        ),
        'all_evaluable_tre_measured_median_mm': float(
            np.median(all_m_evaluable)
        ),
        'all_evaluable_tre_filtered_mean_mm': float(
            all_s_evaluable.mean()
        ),
        'all_evaluable_tre_filtered_median_mm': float(
            np.median(all_s_evaluable)
        ),
        'all_evaluable_success_measured_lt10': float(
            (all_m_evaluable < 10).mean()
        ),
        'all_evaluable_success_filtered_lt10': float(
            (all_s_evaluable < 10).mean()
        ),
        'runtime_ms_per_frame_weighted': float(
            sum(r['ms'] * len(r['tre_m']) for r in valid) / len(all_m)
        ),
        'runtime_fps_from_weighted_mean': float(
            1000.0 /
            (sum(r['ms'] * len(r['tre_m']) for r in valid) / len(all_m))
        ),
        'exploratory_frame_level_wilcoxon_statistic': float(stat),
        'exploratory_frame_level_wilcoxon_p': float(pval),
        'inference_note': (
            'Frames within a trajectory are correlated; the frame-level '
            'Wilcoxon result is descriptive and is not used as an '
            'independent-sample test.'
        ),
    }
    with open(OUT / 'multipatient_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f'  汇总表: {OUT / "multipatient_summary.csv"}')
    print(f'  逐帧表: {OUT / "multipatient_frames.csv"}')
    print(f'  结构化汇总: {OUT / "multipatient_summary.json"}')
    print('=' * 66)


if __name__ == '__main__':
    main()
