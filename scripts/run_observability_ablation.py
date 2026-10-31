# -*- coding: utf-8 -*-
"""Controlled ablation of the unobservable-subspace penalty.

The geometry, target deformation, optimizer, initialization, and random seed are
identical across runs.  Only ``lambda_unobs`` changes.  Results are written as
both per-condition CSV and a JSON protocol summary.
"""
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.geometry import exp_so3, make_T, transform
from src.nonrigid_rtw import (
    FFDLattice,
    compute_observability_basis,
    rtw_optimize_projective_obs,
)
from src.pipeline import render_model
from src.pose_estimation import nearest_neighbors


RC = Path(r'E:\Diff_Rending_Re_3D\dataset\real_colon')
OUT = Path('outputs/observability')
GRID = (6, 6, 6)
SEED = 17
SCALE = 100.0
PENALTIES = (0.0, 1e-3, 1e-2, 1e-1)


def subdivide(nodes, tris, n_per=6):
    points = []
    for a, b, c in tris:
        for i in range(n_per + 1):
            for j in range(n_per + 1 - i):
                u, v = i / n_per, j / n_per
                points.append(u * nodes[a] + v * nodes[b] +
                              (1.0 - u - v) * nodes[c])
    return np.asarray(points)


def main():
    global OUT, SEED
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=SEED,
                    help='target-direction / depth-noise / optimizer seed '
                         '(audited run: 17)')
    ap.add_argument('--out', default=str(OUT))
    args = ap.parse_args()
    SEED = args.seed
    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gt = torch.load(RC / 'scene_0000' / 'gt.pt', map_location='cpu',
                    weights_only=False)
    nodes = gt['scene_spec']['nodes'].numpy().astype(np.float64) * SCALE
    tris = gt['scene_spec']['surface_tris'].numpy().astype(np.int64)
    base = subdivide(nodes, tris)
    base -= base.mean(axis=0)
    extent = float(np.linalg.norm(base.max(axis=0) - base.min(axis=0)))

    height, width = 480, 640
    intrinsic = np.array([
        [572.4, 0.0, 325.3],
        [0.0, 573.6, 242.0],
        [0.0, 0.0, 1.0],
    ])
    pose_gt = make_T(exp_so3(np.array([0.05, 0.03, 0.0])),
                     np.array([0.0, 0.0, 320.0]))

    lattice = FFDLattice(base.min(axis=0), base.max(axis=0),
                         grid=GRID, device=device)
    weights = lattice.weights(base)
    rotation = torch.as_tensor(pose_gt[:3, :3], dtype=torch.float32,
                               device=device)
    basis, singular, rank = compute_observability_basis(weights, rotation)

    rng = np.random.default_rng(SEED)
    raw = torch.as_tensor(rng.normal(size=3 * lattice.n_ctrl),
                          dtype=torch.float32, device=device)
    observed = basis @ (basis.T @ raw)
    unobserved = raw - observed
    observed = observed / observed.norm().clamp(min=1e-12)
    unobserved = unobserved / unobserved.norm().clamp(min=1e-12)
    # Known target: mostly observable, with a smaller deliberately
    # unrecoverable component.  Overall mean control displacement is 4% extent.
    target = 0.8 * observed + 0.2 * unobserved
    target = target * (
        0.04 * extent /
        target.reshape(-1, 3).norm(dim=1).mean().clamp(min=1e-12)
    )
    target_surface = base + (
        weights @ target.reshape(lattice.n_ctrl, 3)
    ).detach().cpu().numpy()

    colors = np.tile(np.array([0.85, 0.65, 0.60]), (len(base), 1))
    rendered = render_model(target_surface, colors, pose_gt, intrinsic,
                            height, width, radius=2, device=device)
    depth = rendered['depth'].cpu().numpy().astype(np.float64)
    foreground = depth > 0
    depth[foreground] += rng.normal(0.0, extent * 0.005,
                                    int(foreground.sum()))

    rows = []
    for penalty in PENALTIES:
        result = rtw_optimize_projective_obs(
            base, depth, intrinsic, pose_gt, grid=GRID,
            iters=700, lr=8e-3, lambda_mag=1e-5, lambda_sm=5e-4,
            lambda_unobs=penalty, inlier_hi=0.2 * extent,
            inlier_lo=0.05 * extent, device=device, seed=SEED,
        )
        estimated = torch.as_tensor(result['delta'].reshape(-1),
                                    dtype=torch.float32, device=device)
        estimated_observed = basis @ (basis.T @ estimated)
        estimated_unobserved = estimated - estimated_observed
        target_observed = basis @ (basis.T @ target)
        observed_error = (
            (estimated_observed - target_observed).norm() /
            target_observed.norm().clamp(min=1e-12)
        )
        unobserved_energy = (
            estimated_unobserved.square().sum() /
            estimated.square().sum().clamp(min=1e-12)
        )
        estimated_camera = transform(result['T'], result['V_def'])
        target_camera = transform(pose_gt, target_surface)
        _, distances = nearest_neighbors(estimated_camera, target_camera,
                                         device=device)
        rows.append({
            'lambda_unobs': penalty,
            'obs_rank': rank,
            'obs_rank_pct': 100.0 * rank / (3 * lattice.n_ctrl),
            'depth_residual_rms': float(result['res_rms']),
            'surface_error_mm': float(distances.mean()),
            'surface_error_pct_extent':
                100.0 * float(distances.mean()) / extent,
            'observed_component_relative_error': float(observed_error),
            'estimated_unobservable_energy_fraction':
                float(unobserved_energy),
        })
        print(rows[-1])

    with open(OUT / 'observability_penalty_ablation.csv', 'w', newline='',
              encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(OUT / 'observability_penalty_ablation.json', 'w',
              encoding='utf-8') as stream:
        json.dump({
            'protocol': {
                'dataset': 'real_colon/scene_0000 geometry',
                'controlled_synthetic_deformation': True,
                'seed': SEED,
                'grid': list(GRID),
                'dof': 3 * lattice.n_ctrl,
                'singular_threshold_ratio': 1e-3,
                'target_observed_weight': 0.8,
                'target_unobserved_weight': 0.2,
                'iterations': 700,
                'only_changed_parameter': 'lambda_unobs',
                'lambda_unobs_values': list(PENALTIES),
            },
            'rows': rows,
        }, stream, indent=2)
    print(f'Saved {OUT / "observability_penalty_ablation.csv"}')


if __name__ == '__main__':
    main()
