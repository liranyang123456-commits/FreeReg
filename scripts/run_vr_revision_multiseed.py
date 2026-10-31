# -*- coding: utf-8 -*-
"""Repeated-seed runs for the stochastic evaluation protocols.

Runs the stochastic protocols several times with different seeds, under
deterministic environment controls, and records a manifest
(argv, return code, elapsed time, log and output hashes).  Outputs go to

    outputs/multirun/<task>/seed<k>/

Usage (py3d env):
    python scripts/run_vr_revision_multiseed.py            # everything
    python scripts/run_vr_revision_multiseed.py --tasks bop_on bop_off
    python scripts/run_vr_revision_multiseed.py --seeds 0 1 2
    python scripts/run_vr_revision_multiseed.py --out-root D:/scratch/multirun
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MULTI = ROOT / 'outputs' / 'multirun'
PY = sys.executable

ENV_CONTROLS = {
    'PYTHONHASHSEED': '0',
    'KMP_DUPLICATE_LIB_OK': 'TRUE',
    'PYTHONIOENCODING': 'utf-8',
    'PYTHONUNBUFFERED': '1',
    'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
}

# task -> (argv builder, expected outputs relative to the seed dir)
BOP_COMMON = ['scripts/run_bop_eval.py', '--engine', 'hybrid', '--n_refs', '40',
              '--scenes', '1', '2', '3', '4', '5', '--frames', '25']


def _bop(on: bool):
    def build(seed: int, out: Path):
        flag = '--use_contour' if on else '--no-use_contour'
        return BOP_COMMON + [flag, '--seed', str(seed),
                             '--out', str(out / 'bop_eval_hybrid.csv')]
    return build


TASKS = {
    'bop_on': (_bop(True), ['bop_eval_hybrid.csv']),
    'bop_off': (_bop(False), ['bop_eval_hybrid.csv']),
    'real_colon': (lambda s, o: ['scripts/run_real_colon_eval.py', '--scenes', '24',
                                 '--seed', str(s), '--out', str(o)],
                   ['real_colon_metrics.csv']),
    'temporal': (lambda s, o: ['scripts/run_temporal_eval.py', '--seed', str(s),
                               '--out', str(o)],
                 ['summary.json']),
    'scared': (lambda s, o: ['scripts/run_scared_video_ar.py', '--seed', str(s),
                             '--out', str(o)],
               ['summary.json']),
    'clinical': (lambda s, o: ['scripts/run_multipatient_eval.py', '--patients', '16',
                               '--seed', str(s), '--out', str(o)],
                 ['multipatient_summary.json', 'multipatient_frames.csv']),
    'render_3dgs': (lambda s, o: ['scripts/run_3dgs_eval.py', '--seed', str(s),
                                  '--out', str(o)],
                    ['summary.json', 'gaussians.pt']),
    'render_4dgs': (lambda s, o: ['scripts/run_4dgs_eval.py', '--seed', str(s),
                                  '--out', str(o)],
                    ['summary.json']),
    # audited seed is 17; repeats use 17, 18, 19 (pass --seeds 17 18 19)
    'obs_ablation': (lambda s, o: ['scripts/run_observability_ablation.py',
                                   '--seed', str(s), '--out', str(o)],
                     ['observability_penalty_ablation.json']),
}
DEFAULT_ORDER = ['bop_on', 'bop_off', 'real_colon', 'temporal', 'scared',
                 'clinical', 'render_3dgs', 'render_4dgs']


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', nargs='+', default=DEFAULT_ORDER)
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--skip_existing', action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument('--out-root', type=Path, default=MULTI,
                    help='directory for the runs and their manifest '
                         '(default: the archived revision/multirun)')
    args = ap.parse_args()

    root = args.out_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / 'multiseed_manifest.json'
    manifest = {'python': PY, 'environment_controls': ENV_CONTROLS, 'runs': []}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        except Exception:
            pass
    env = dict(os.environ)
    env.update(ENV_CONTROLS)

    for task in args.tasks:
        build, expected = TASKS[task]
        for seed in args.seeds:
            out = root / task / f'seed{seed}'
            if args.skip_existing and all((out / e).exists() for e in expected):
                print(f'[skip] {task} seed{seed} already complete')
                continue
            out.mkdir(parents=True, exist_ok=True)
            argv = [PY] + build(seed, out)
            log = out / 'run.log'
            print(f'[run ] {task} seed{seed}: {" ".join(argv[1:])}')
            t0 = time.time()
            with log.open('w', encoding='utf-8') as lf:
                rc = subprocess.call(argv, cwd=str(ROOT), env=env,
                                     stdout=lf, stderr=subprocess.STDOUT)
            elapsed = time.time() - t0
            outputs = {e: (sha256(out / e) if (out / e).exists() else None)
                       for e in expected}
            entry = {
                'task': task, 'seed': seed, 'argv': argv[1:], 'return_code': rc,
                'elapsed_s': round(elapsed, 1),
                'log': str(log.relative_to(ROOT) if log.is_relative_to(ROOT) else log),
                'outputs': outputs,
                'status': 'ok' if rc == 0 and all(outputs.values()) else 'failed',
                'ended_unix': time.time(),
            }
            manifest['runs'] = [r for r in manifest['runs']
                                if not (r['task'] == task and r['seed'] == seed)]
            manifest['runs'].append(entry)
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
            print(f'[done] {task} seed{seed}: rc={rc} {elapsed:.0f}s -> {entry["status"]}')


if __name__ == '__main__':
    main()
