# -*- coding: utf-8 -*-
"""Compute bootstrap confidence intervals, RTW failure split, and excluded fly-through statistics."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / 'outputs'
MULTI = OUT_ROOT / 'multirun'
EV = OUT_ROOT / 'rerun'
OUT = OUT_ROOT / 'stats'
OUT.mkdir(parents=True, exist_ok=True)


def resolve_input(*candidates: Path) -> Path:
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def load_csv(path: Path) -> list[dict]:
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def fnum(row, key):
    v = row.get(key, '')
    if v in ('', None):
        return np.nan
    if isinstance(v, str) and v.lower() in ('true', 'false', 'nan'):
        if v.lower() == 'nan':
            return np.nan
    try:
        return float(v)
    except ValueError:
        return np.nan


def bootstrap_ci(values, stat=np.mean, n=10000, seed=0, alpha=0.05):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    rng = np.random.default_rng(seed)
    if len(x) == 0:
        return {'n': 0, 'point': None, 'lo': None, 'hi': None, 'sd': None}
    stats = np.empty(n)
    for i in range(n):
        stats[i] = stat(rng.choice(x, size=len(x), replace=True))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return {
        'n': int(len(x)),
        'point': float(stat(x)),
        'sd': float(x.std(ddof=1)) if len(x) > 1 else 0.0,
        'lo': float(lo),
        'hi': float(hi),
    }


def bootstrap_rate(flags, n=10000, seed=0, alpha=0.05):
    x = np.asarray(flags, dtype=float)
    return bootstrap_ci(x, stat=np.mean, n=n, seed=seed, alpha=alpha)


def cluster_bootstrap_mean(groups, n=10000, seed=0, alpha=0.05):
    keys = list(groups)
    rng = np.random.default_rng(seed)
    stats = np.empty(n)
    for i in range(n):
        chosen = rng.choice(keys, size=len(keys), replace=True)
        pool = np.concatenate([groups[k] for k in chosen])
        stats[i] = pool.mean()
    all_vals = np.concatenate([groups[k] for k in keys])
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return {
        'n_clusters': len(keys),
        'n_items': int(all_vals.size),
        'point': float(all_vals.mean()),
        'lo': float(lo),
        'hi': float(hi),
    }


def rtw_breakdown():
    rows = load_csv(resolve_input(
        OUT_ROOT / 'real_colon' / 'real_colon_metrics.csv',
        MULTI / 'real_colon' / 'seed0' / 'real_colon_metrics.csv',
        EV / 'real_colon' / 'real_colon_metrics.csv',
    ))
    for r in rows:
        r['diverged'] = str(r['diverged']).lower() == 'true'
        r['rot_r'] = fnum(r, 'rot_r')
        r['tr_r'] = fnum(r, 'tr_r')
        r['rec_rtw'] = fnum(r, 'rec_rtw')
        r['rec_base'] = fnum(r, 'rec_base')
        r['t'] = int(float(r['t']))

    def mode_of(r):
        if not r['diverged']:
            return 'converged'
        if r['rot_r'] >= 150:
            return 'near_pi_flip'
        if r['rot_r'] >= 30:
            return 'large_rotation_30_150'
        return 'other'

    for r in rows:
        r['mode'] = mode_of(r)

    counts = Counter(r['mode'] for r in rows)
    by_t = defaultdict(Counter)
    by_scene = defaultdict(Counter)
    for r in rows:
        by_t[r['t']][r['mode']] += 1
        by_scene[r['scene']][r['mode']] += 1

    ok = [r for r in rows if r['mode'] == 'converged']
    fail = [r for r in rows if r['mode'] != 'converged']
    rec_rtw = [r['rec_rtw'] * 100 for r in ok]
    rec_base_ok = [r['rec_base'] * 100 for r in ok]
    rec_base_fail = [r['rec_base'] * 100 for r in fail]
    rot_r_ok = [r['rot_r'] for r in ok]
    rot_r_fail = [r['rot_r'] for r in fail]

    all_fail_scenes = sorted(
        s for s, c in by_scene.items() if c['converged'] == 0)
    all_ok_scenes = sorted(
        s for s, c in by_scene.items() if sum(c.values()) == c['converged'])

    return {
        'n': len(rows),
        'counts': dict(counts),
        'by_frame_index': {str(k): dict(v) for k, v in sorted(by_t.items())},
        'all_fail_scenes': all_fail_scenes,
        'all_ok_scenes': all_ok_scenes,
        'n_scenes': len(by_scene),
        'rigid_rot_converged': bootstrap_ci(rot_r_ok),
        'rigid_rot_failed': bootstrap_ci(rot_r_fail),
        'rec_rtw_pct': bootstrap_ci(rec_rtw),
        'rec_base_converged_pct': bootstrap_ci(rec_base_ok),
        'rec_base_failed_pct': bootstrap_ci(rec_base_fail),
        'fail_rate': bootstrap_rate([r['mode'] != 'converged' for r in rows]),
        'note': (
            'Divergence sentinel is RTW rotation > 30 deg after optimization. '
            'All 37 failures also have rigid-init rotation > 30 deg.'
        ),
        'all_fail_have_rigid_gt_30': all(r['rot_r'] > 30 for r in fail),
        'any_success_with_rigid_gt_30': any(r['rot_r'] > 30 for r in ok),
    }


def clinical_breakdown():
    frames = load_csv(resolve_input(
        OUT_ROOT / 'ct' / 'mp' / 'multipatient_frames.csv',
        MULTI / 'clinical' / 'seed0' / 'multipatient_frames.csv',
        EV / 'clinical_cohort' / 'multipatient_frames.csv',
    ))
    summary = load_csv(resolve_input(
        OUT_ROOT / 'ct' / 'mp' / 'multipatient_summary.csv',
        MULTI / 'clinical' / 'seed0' / 'multipatient_summary.csv',
        EV / 'clinical_cohort' / 'multipatient_summary.csv',
    ))
    by_traj = defaultdict(list)
    status = {}
    for r in frames:
        tid = r['trajectory_id']
        status[tid] = r['analysis_status']
        by_traj[tid].append(fnum(r, 'tre_measured_mm'))

    included = [t for t, s in status.items() if s == 'included']
    excluded = [t for t, s in status.items() if s.startswith('excluded')]
    inc_means = [float(np.mean(by_traj[t])) for t in included]
    inc_medians = [float(np.median(by_traj[t])) for t in included]
    inc_groups = {t: np.asarray(by_traj[t], float) for t in included}

    excl_desc = {}
    for t in excluded:
        x = np.asarray(by_traj[t], float)
        excl_desc[t] = {
            'n': int(x.size),
            'first_mm': float(x[0]),
            'last_mm': float(x[-1]),
            'mean_mm': float(x.mean()),
            'median_mm': float(np.median(x)),
            'max_mm': float(x.max()),
            'n_over_20': int((x > 20).sum()),
        }

    t5 = np.asarray(by_traj.get('trajectory_05', []), float)
    t7 = np.asarray(by_traj.get('trajectory_07', []), float)
    identical = (
        t5.size and t7.size and t5.size == t7.size
        and np.allclose(t5, t7, rtol=0, atol=1e-9)
    )

    return {
        'n_evaluable_trajectories': len(by_traj),
        'n_included': len(included),
        'n_excluded': len(excluded),
        'included_ids': included,
        'excluded_ids': excluded,
        'included_frame_mean': bootstrap_ci(
            np.concatenate([by_traj[t] for t in included])),
        'included_trajectory_mean': bootstrap_ci(inc_means),
        'included_trajectory_median': bootstrap_ci(inc_medians),
        'included_cluster_bootstrap_frame_mean': cluster_bootstrap_mean(inc_groups),
        'excluded_profiles': excl_desc,
        'excluded_sequences_identical': bool(identical),
        'excluded_pattern': (
            'Both excluded trajectories start near 3 mm and lose lock along '
            'the fly-through, exceeding 80 mm in distal frames. Their '
            'per-frame TRE sequences are identical to 1e-9, consistent with '
            'serial CTs of one source that produced the same extracted airway '
            'axis under the same generator seed. Counted twice at directory '
            'level; one observed failure pattern.'
        ),
        'summary_rows': summary,
    }


def bop_ci():
    hybrid = load_csv(resolve_input(
        OUT_ROOT / 'bop_eval_hybrid_full.csv',
        MULTI / 'bop_on' / 'seed0' / 'bop_eval_hybrid.csv',
        EV / 'bop_hybrid' / 'bop_eval_hybrid_full.csv',
    ))
    classic = load_csv(resolve_input(
        OUT_ROOT / 'bop_eval_results.csv',
        EV / 'bop_classic' / 'bop_eval_results.csv',
    ))
    official = load_csv(resolve_input(
        OUT_ROOT / 'bop_official' / 'bop_official_metrics.csv',
        EV / 'bop_protocol' / 'bop_official_metrics.csv',
    ))

    def recall(rows, key, frac):
        flags = []
        for r in rows:
            if str(r.get('fail', 'False')).lower() == 'true':
                flags.append(0.0)
                continue
            flags.append(1.0 if fnum(r, key) < frac * fnum(r, 'diameter') else 0.0)
        return bootstrap_rate(flags)

    def rot_med(rows):
        rots = [fnum(r, 'rot_deg') for r in rows
                if str(r.get('fail', 'False')).lower() != 'true']
        return bootstrap_ci(rots, stat=np.median)

    return {
        'hybrid_adds10': recall(hybrid, 'add_s', 0.10),
        'hybrid_add20': recall(hybrid, 'add', 0.20),
        'hybrid_rot_median': rot_med(hybrid),
        'classic_adds10': recall(classic, 'add_s', 0.10),
        'classic_add20': recall(classic, 'add', 0.20),
        'classic_rot_median': rot_med(classic),
        'bop_ar': bootstrap_ci([fnum(r, 'ar') for r in official]),
        'n_hybrid': len(hybrid),
        'n_official': len(official),
    }


def tless_ci():
    """All-instance ADD-S recall with misses counted as failures."""
    path = resolve_input(
        OUT_ROOT / 'tless' / 'tless_metrics.csv',
        EV / 'tless' / 'tless_metrics.csv',
    )
    if not path.exists():
        return {}
    rows = load_csv(path)
    out = {'n_instances': len(rows),
           'n_missing': sum(1 for r in rows
                            if str(r.get('fail', 'False')).lower() == 'true')}
    for frac in (0.10, 0.15, 0.20):
        flags = []
        for r in rows:
            if str(r.get('fail', 'False')).lower() == 'true':
                flags.append(0.0)
            else:
                flags.append(1.0 if fnum(r, 'add_s') < frac * fnum(r, 'diameter')
                             else 0.0)
        out[f'adds{int(frac * 100)}'] = bootstrap_rate(flags)
    return out


def cascade_ci():
    """FreeRegNet / cascade / engine ADD-S@10%d recall on 300 instances."""
    path = resolve_input(
        OUT_ROOT / 'freereg_net' / 'cascade_metrics.csv',
        EV / 'freeregnet_cascade' / 'cascade_metrics.csv',
    )
    if not path.exists():
        return {}
    rows = load_csv(path)
    out = {'n_instances': len(rows)}
    for key, name in (('adds_nn', 'freeregnet'), ('adds_icp', 'cascade'),
                      ('adds_eng', 'geometric')):
        flags = [1.0 if fnum(r, key) < 0.10 * fnum(r, 'dia') else 0.0
                 for r in rows]
        out[name + '_adds10'] = bootstrap_rate(flags)
    flags = [1.0 if fnum(r, 'add_icp') < 0.10 * fnum(r, 'dia') else 0.0
             for r in rows]
    out['cascade_add10'] = bootstrap_rate(flags)
    return out


def main():
    report = {
        'rtw': rtw_breakdown(),
        'clinical': clinical_breakdown(),
        'bop': bop_ci(),
        'tless': tless_ci(),
        'cascade': cascade_ci(),
    }
    out_json = OUT / 'revision_stats.json'
    out_json.write_text(json.dumps(report, indent=2), encoding='utf-8')

    rtw = report['rtw']
    clin = report['clinical']
    bop = report['bop']
    lines = [
        '# Bootstrap confidence intervals and RTW failure breakdown',
        '',
        '## RTW failure modes (71 frames)',
        f"- counts: {rtw['counts']}",
        f"- by frame index t: {rtw['by_frame_index']}",
        f"- all-fail scenes: {rtw['all_fail_scenes']}",
        f"- all-ok scenes: {rtw['all_ok_scenes']}",
        f"- rec RTW % mean±SD [{rtw['rec_rtw_pct']['lo']:.3f}, {rtw['rec_rtw_pct']['hi']:.3f}]",
        f"- fail rate CI: {rtw['fail_rate']}",
        f"- all failures have rigid-init >30: {rtw['all_fail_have_rigid_gt_30']}",
        f"- any success with rigid-init >30: {rtw['any_success_with_rigid_gt_30']}",
        '',
        '## Clinical fly-through',
        f"- included traj mean TRE: {clin['included_trajectory_mean']}",
        f"- included frame mean TRE: {clin['included_frame_mean']}",
        f"- cluster bootstrap: {clin['included_cluster_bootstrap_frame_mean']}",
        f"- excluded: {clin['excluded_profiles']}",
        f"- identical 05/07: {clin['excluded_sequences_identical']}",
        f"- {clin['excluded_pattern']}",
        '',
        '## BOP bootstrap',
        f"- hybrid ADD-S@10: {bop['hybrid_adds10']}",
        f"- hybrid ADD@20: {bop['hybrid_add20']}",
        f"- BOP AR: {bop['bop_ar']}",
        '',
        '## T-LESS bootstrap (misses as failures)',
        f"- {report['tless']}",
        '',
        '## Cascade bootstrap (300 instances)',
        f"- {report['cascade']}",
        '',
    ]
    (OUT / 'revision_stats.md').write_text('\n'.join(lines), encoding='utf-8')
    print((OUT / 'revision_stats.md').read_text(encoding='utf-8'))
    print('wrote', out_json)


if __name__ == '__main__':
    main()
