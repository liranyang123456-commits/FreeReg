# -*- coding: utf-8 -*-
"""Aggregate the repeated-seed runs (scripts/run_vr_revision_multiseed.py)
into mean +/- SD tables, and compare them with the primary seed-0 runs.

Writes
    outputs/stats/multiseed_summary.json
    outputs/stats/multiseed_summary.md
"""
from __future__ import annotations

import csv
import json
from collections import Counter
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


def load_csv(path: Path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def seeds_of(task: str):
    d = MULTI / task
    if not d.exists():
        return []
    return sorted(int(p.name[4:]) for p in d.iterdir()
                  if p.is_dir() and p.name.startswith('seed'))


def msd(values):
    x = np.asarray([v for v in values if v is not None], dtype=float)
    if x.size == 0:
        return {'n': 0, 'mean': None, 'sd': None, 'values': []}
    return {'n': int(x.size), 'mean': float(x.mean()),
            'sd': float(x.std(ddof=1)) if x.size > 1 else 0.0,
            'values': [float(v) for v in x]}


# ---------------------------------------------------------------- BOP
def bop_metrics(rows):
    n = len(rows)
    ok = [r for r in rows if str(r.get('fail', 'False')).lower() != 'true']
    d = np.array([float(r['diameter']) for r in ok])
    a = np.array([float(r['add']) for r in ok])
    s = np.array([float(r['add_s']) for r in ok])
    rot = np.array([float(r['rot_deg']) for r in ok])
    return {
        'n': n, 'n_fail': n - len(ok),
        'adds10': float((s < 0.10 * d).sum() / n),
        'adds10_flags': {(int(r['scene']), int(r['frame']), int(r['obj'])):
                         float(float(r['add_s']) < 0.10 * float(r['diameter']))
                         for r in ok},
        'add20': float((a < 0.20 * d).sum() / n),
        'add20_flags': {(int(r['scene']), int(r['frame']), int(r['obj'])):
                        float(float(r['add']) < 0.20 * float(r['diameter']))
                        for r in ok},
        'rot_med': float(np.median(rot)),
        'time_mean_s': float(np.mean([float(r['time_s']) for r in ok])),
    }


def bop_block():
    res = {}
    per = {}
    for task in ('bop_on', 'bop_off'):
        per[task] = {}
        for s in seeds_of(task):
            p = MULTI / task / f'seed{s}' / 'bop_eval_hybrid.csv'
            if p.exists():
                per[task][s] = bop_metrics(load_csv(p))
        res[task] = {
            'seeds': sorted(per[task]),
            'adds10': msd([m['adds10'] for m in per[task].values()]),
            'add20': msd([m['add20'] for m in per[task].values()]),
            'rot_med': msd([m['rot_med'] for m in per[task].values()]),
            'n_fail': msd([m['n_fail'] for m in per[task].values()]),
            'per_seed': {s: {k: v for k, v in m.items() if not k.endswith('_flags')}
                         for s, m in per[task].items()},
        }
    # paired per-seed comparison on identical samples
    pairs = []
    for s in sorted(set(per['bop_on']) & set(per['bop_off'])):
        on, off = per['bop_on'][s], per['bop_off'][s]
        keys = sorted(set(on['add20_flags']) & set(off['add20_flags']))
        d20 = np.array([on['add20_flags'][k] - off['add20_flags'][k] for k in keys])
        d10 = np.array([on['adds10_flags'][k] - off['adds10_flags'][k] for k in keys])
        pairs.append({'seed': s, 'n_shared': len(keys),
                      'add20_on_minus_off': float(d20.mean()),
                      'add20_wins_on': int((d20 > 0).sum()),
                      'add20_wins_off': int((d20 < 0).sum()),
                      'adds10_on_minus_off': float(d10.mean()),
                      'adds10_wins_on': int((d10 > 0).sum()),
                      'adds10_wins_off': int((d10 < 0).sum())})
    res['paired_by_seed'] = pairs
    res['delta_add20'] = msd([p['add20_on_minus_off'] for p in pairs])
    res['delta_adds10'] = msd([p['adds10_on_minus_off'] for p in pairs])
    aud = resolve_input(
        OUT_ROOT / 'bop_eval_hybrid_full.csv',
        EV / 'bop_hybrid' / 'bop_eval_hybrid_full.csv',
    )
    if aud.exists():
        m = bop_metrics(load_csv(aud))
        res['primary_hybrid_seed0'] = {k: v for k, v in m.items() if not k.endswith('_flags')}
    return res


# ---------------------------------------------------------------- RTW
def rtw_metrics(rows):
    for r in rows:
        r['diverged'] = str(r['diverged']).lower() == 'true'
        r['rot_r'] = float(r['rot_r'])
        r['rec_rtw'] = float(r['rec_rtw']) if r['rec_rtw'] not in ('', 'nan') else np.nan
        r['rec_base'] = float(r['rec_base'])
    ok = [r for r in rows if not r['diverged']]
    fail = [r for r in rows if r['diverged']]
    modes = Counter('near_pi' if r['rot_r'] >= 150 else
                    ('large_rot' if r['rot_r'] >= 30 else 'other') for r in fail)
    return {
        'n': len(rows), 'n_converged': len(ok), 'n_failed': len(fail),
        'near_pi': modes.get('near_pi', 0), 'large_rot': modes.get('large_rot', 0),
        'other': modes.get('other', 0),
        'rec_base_conv_pct': float(np.mean([r['rec_base'] for r in ok]) * 100) if ok else None,
        'rec_rtw_conv_pct': float(np.nanmean([r['rec_rtw'] for r in ok]) * 100) if ok else None,
        'rot_w_med': float(np.median([float(r['rot_w']) for r in ok])) if ok else None,
        'all_fail_rigid_gt30': bool(all(r['rot_r'] > 30 for r in fail)),
        'any_ok_rigid_gt30': bool(any(r['rot_r'] > 30 for r in ok)),
    }


def rtw_block():
    per = {}
    for s in seeds_of('real_colon'):
        p = MULTI / 'real_colon' / f'seed{s}' / 'real_colon_metrics.csv'
        if p.exists():
            per[s] = rtw_metrics(load_csv(p))
    res = {'seeds': sorted(per), 'per_seed': per}
    for k in ('n_converged', 'near_pi', 'large_rot', 'rec_base_conv_pct',
              'rec_rtw_conv_pct', 'rot_w_med'):
        res[k] = msd([m[k] for m in per.values()])
    aud = resolve_input(
        OUT_ROOT / 'real_colon' / 'real_colon_metrics.csv',
        EV / 'real_colon' / 'real_colon_metrics.csv',
    )
    if aud.exists():
        res['primary_seed0'] = rtw_metrics(load_csv(aud))
    return res


# ---------------------------------------------------------------- temporal
def temporal_block():
    per = {}
    for s in seeds_of('temporal'):
        p = MULTI / 'temporal' / f'seed{s}' / 'summary.json'
        if p.exists():
            j = json.loads(p.read_text(encoding='utf-8'))
            per[s] = {
                'a_j_meas': j['part_a']['j_meas'], 'a_j_smooth': j['part_a']['j_smooth'],
                'a_red': 1 - j['part_a']['j_smooth'] / j['part_a']['j_meas'],
                'a_e_meas': j['part_a']['e_meas'], 'a_e_smooth': j['part_a']['e_smooth'],
                'b_j_meas': j['part_b']['j_meas'], 'b_j_smooth': j['part_b']['j_smooth'],
                'b_red': 1 - j['part_b']['j_smooth'] / j['part_b']['j_meas'],
                'b_e_meas': j['part_b']['e_meas'], 'b_e_smooth': j['part_b']['e_smooth'],
                'b_reloc': j['part_b']['reloc'], 'a_reloc': j['part_a']['reloc'],
            }
    res = {'seeds': sorted(per), 'per_seed': per}
    for k in ('a_j_meas', 'a_j_smooth', 'a_red', 'b_j_meas', 'b_j_smooth', 'b_red',
              'a_e_meas', 'a_e_smooth', 'b_e_meas', 'b_e_smooth'):
        res[k] = msd([m[k] for m in per.values()])
    return res


def scared_block():
    per = {}
    for s in seeds_of('scared'):
        p = MULTI / 'scared' / f'seed{s}' / 'summary.json'
        if p.exists():
            j = json.loads(p.read_text(encoding='utf-8'))
            per[s] = {'j_raw': j['jitter_raw_px'], 'j_sm': j['jitter_filtered_px'],
                      'red': j['jitter_reduction_fraction'], 'reloc': j['relocations'],
                      'frames': j['protocol']['completed_frames']}
    res = {'seeds': sorted(per), 'per_seed': per}
    for k in ('j_raw', 'j_sm', 'red'):
        res[k] = msd([m[k] for m in per.values()])
    return res


# ---------------------------------------------------------------- clinical
def clinical_block():
    per = {}
    for s in seeds_of('clinical'):
        p = MULTI / 'clinical' / f'seed{s}' / 'multipatient_summary.json'
        if p.exists():
            j = json.loads(p.read_text(encoding='utf-8'))
            fr = load_csv(MULTI / 'clinical' / f'seed{s}' / 'multipatient_frames.csv')
            traj = {}
            for r in fr:
                traj.setdefault((r['trajectory_id'], r['analysis_status']), []).append(
                    float(r['tre_measured_mm']))
            inc_means = [float(np.mean(v)) for (t, st), v in traj.items() if st == 'included']
            per[s] = {
                'n_included': j['trajectories_included'],
                'n_excluded': j['trajectories_excluded'],
                'excluded_ids': j['excluded_trajectory_ids'],
                'frames_included': j['frames_included'],
                'frames_all': j['frames_all_evaluable'],
                'tre_mean': j['tre_measured_mean_mm'],
                'tre_median': j['tre_measured_median_mm'],
                'succ10': j['success_measured_lt10'],
                'tre_filt_mean': j['tre_filtered_mean_mm'],
                'all_tre_mean': j['all_evaluable_tre_measured_mean_mm'],
                'all_tre_median': j['all_evaluable_tre_measured_median_mm'],
                'all_succ10': j['all_evaluable_success_measured_lt10'],
                'traj_mean_of_means': float(np.mean(inc_means)),
                'traj_sd_of_means': float(np.std(inc_means, ddof=1)),
                'runtime_ms': j['runtime_ms_per_frame_weighted'],
            }
    res = {'seeds': sorted(per), 'per_seed': per}
    for k in ('tre_mean', 'tre_median', 'succ10', 'tre_filt_mean', 'all_tre_mean',
              'all_tre_median', 'all_succ10', 'traj_mean_of_means', 'n_included',
              'frames_included'):
        res[k] = msd([m[k] for m in per.values()])
    return res


# ---------------------------------------------------------------- render
def render_block():
    per3, per4 = {}, {}
    for s in seeds_of('render_3dgs'):
        p = MULTI / 'render_3dgs' / f'seed{s}' / 'summary.json'
        if p.exists():
            j = json.loads(p.read_text(encoding='utf-8'))
            per3[s] = {'train_psnr': j['train_psnr_3dgs_mean_db'],
                       'heldout_psnr': j['heldout_psnr_3dgs_db'],
                       'train_ssim': j['train_ssim_3dgs_mean'],
                       'heldout_ssim': j['heldout_ssim_3dgs'],
                       'render_ms_shared_gpu': j['render_time_ms']}
    for s in seeds_of('render_4dgs'):
        p = MULTI / 'render_4dgs' / f'seed{s}' / 'summary.json'
        if p.exists():
            j = json.loads(p.read_text(encoding='utf-8'))
            gains = {c['case']: c['gain_db'] for c in j['cases']}
            per4[s] = {'static_psnr': j['static_3dgs_psnr_db'],
                       'mean_gain': j['mean_dynamic_gain_db'],
                       'gains': gains}
    res = {'3dgs': {'seeds': sorted(per3), 'per_seed': per3},
           '4dgs': {'seeds': sorted(per4), 'per_seed': per4}}
    for k in ('train_psnr', 'heldout_psnr', 'train_ssim', 'heldout_ssim'):
        res['3dgs'][k] = msd([m[k] for m in per3.values()])
    res['4dgs']['mean_gain'] = msd([m['mean_gain'] for m in per4.values()])
    cases = sorted({c for m in per4.values() for c in m['gains']})
    res['4dgs']['gain_by_case'] = {c: msd([m['gains'].get(c) for m in per4.values()])
                                   for c in cases}
    return res


def obs_block():
    """Observability penalty ablation; primary seed 17, repeats 18 and 19."""
    per = {}
    for s in seeds_of('obs_ablation'):
        p = MULTI / 'obs_ablation' / f'seed{s}' / 'observability_penalty_ablation.json'
        if p.exists():
            j = json.loads(p.read_text(encoding='utf-8'))
            per[s] = {str(r['lambda_unobs']): {
                'null_energy': r['estimated_unobservable_energy_fraction'],
                'surface_mm': r['surface_error_mm'],
                'obs_rel_err': r['observed_component_relative_error'],
                'rank': r['obs_rank']} for r in j['rows']}
    res = {'seeds': sorted(per), 'per_seed': per}
    for lam in ('0.0', '0.01'):
        for k in ('null_energy', 'surface_mm', 'obs_rel_err'):
            res[f'{k}_lam{lam}'] = msd([m[lam][k] for m in per.values() if lam in m])
    res['surface_delta_abs_max_mm'] = max(
        (abs(m['0.01']['surface_mm'] - m['0.0']['surface_mm']) for m in per.values()
         if '0.0' in m and '0.01' in m), default=None)
    return res


def fmt(m, scale=1.0, nd=3):
    if not m or m.get('n', 0) == 0:
        return 'n/a'
    if m['n'] == 1:
        return f"{m['mean'] * scale:.{nd}f} (1 run)"
    return f"{m['mean'] * scale:.{nd}f} ± {m['sd'] * scale:.{nd}f} (n={m['n']})"


def main():
    rep = {'bop': bop_block(), 'rtw': rtw_block(), 'temporal': temporal_block(),
           'scared': scared_block(), 'clinical': clinical_block(),
           'render': render_block(), 'obs_ablation': obs_block()}
    (OUT / 'multiseed_summary.json').write_text(
        json.dumps(rep, indent=2, default=str), encoding='utf-8')
    b, r, t, sc, c, rd = (rep['bop'], rep['rtw'], rep['temporal'], rep['scared'],
                          rep['clinical'], rep['render'])
    lines = ['# Repeated-seed summary (mean ± SD over seeds)', '']
    lines += ['## BOP hybrid (125 samples, n_refs 40)',
              f"- contour ON  seeds {b['bop_on']['seeds']}: ADD-S@10 {fmt(b['bop_on']['adds10'])}, "
              f"ADD@20 {fmt(b['bop_on']['add20'])}, rot med {fmt(b['bop_on']['rot_med'], nd=2)} deg",
              f"- contour OFF seeds {b['bop_off']['seeds']}: ADD-S@10 {fmt(b['bop_off']['adds10'])}, "
              f"ADD@20 {fmt(b['bop_off']['add20'])}, rot med {fmt(b['bop_off']['rot_med'], nd=2)} deg",
              f"- paired delta (on-off) ADD@20 {fmt(b['delta_add20'])}; ADD-S@10 {fmt(b['delta_adds10'])}",
              f"- per-seed pairs: {b['paired_by_seed']}",
              f"- primary seed0: {b.get('primary_hybrid_seed0')}", '']
    lines += ['## RTW (71 frames)',
              f"- converged {fmt(r['n_converged'], nd=1)}; near-pi {fmt(r['near_pi'], nd=1)}; "
              f"large-rot {fmt(r['large_rot'], nd=1)}",
              f"- rec base (conv) % {fmt(r['rec_base_conv_pct'], nd=2)}; rec RTW (conv) % {fmt(r['rec_rtw_conv_pct'], nd=2)}",
              f"- per seed: {r['per_seed']}", f"- primary: {r.get('primary_seed0')}", '']
    lines += ['## Temporal',
              f"- A jitter {fmt(t['a_j_meas'], nd=2)} -> {fmt(t['a_j_smooth'], nd=2)} px, reduction {fmt(t['a_red'], 100, 1)} %",
              f"- B jitter {fmt(t['b_j_meas'], nd=2)} -> {fmt(t['b_j_smooth'], nd=2)} px, reduction {fmt(t['b_red'], 100, 1)} %",
              f"- per seed: {t['per_seed']}", '']
    lines += ['## SCARED',
              f"- jitter {fmt(sc['j_raw'], nd=2)} -> {fmt(sc['j_sm'], nd=2)} px, reduction {fmt(sc['red'], 100, 1)} %",
              f"- per seed: {sc['per_seed']}", '']
    lines += ['## Clinical fly-through',
              f"- retained frame mean TRE {fmt(c['tre_mean'], nd=2)} mm; median {fmt(c['tre_median'], nd=2)}; "
              f"<10mm {fmt(c['succ10'], 100, 1)} %; included traj {fmt(c['n_included'], nd=1)}",
              f"- all-evaluable mean {fmt(c['all_tre_mean'], nd=2)} mm",
              f"- per seed: {c['per_seed']}", '']
    lines += ['## Rendering',
              f"- 3DGS train PSNR {fmt(rd['3dgs']['train_psnr'], nd=2)} dB; held-out {fmt(rd['3dgs']['heldout_psnr'], nd=2)} dB; "
              f"held-out SSIM {fmt(rd['3dgs']['heldout_ssim'], nd=4)}",
              f"- 4DGS mean gain {fmt(rd['4dgs']['mean_gain'], nd=2)} dB; by case {rd['4dgs']['gain_by_case']}",
              f"- per seed 3dgs: {rd['3dgs']['per_seed']}", '']
    ob = rep['obs_ablation']
    lines += ['## Observability penalty ablation (seeds 17/18/19)',
              f"- null energy lam=0: {fmt(ob.get('null_energy_lam0.0'))}; lam=0.01: {fmt(ob.get('null_energy_lam0.01'))}",
              f"- surface mm lam=0: {fmt(ob.get('surface_mm_lam0.0'))}; lam=0.01: {fmt(ob.get('surface_mm_lam0.01'))}; max |delta| {ob.get('surface_delta_abs_max_mm')}",
              f"- obs rel err lam=0.01: {fmt(ob.get('obs_rel_err_lam0.01'))}", '']
    (OUT / 'multiseed_summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
