# -*- coding: utf-8 -*-
"""Time the same trained 4k-Gaussian scene with a CUDA tile rasterizer (gsplat)
and with the PyTorch EWA reference splatter.

Both renderers are timed in the same process, on the same
Gaussians, view, intrinsics and 640x480 canvas, back to back, so the ratio is
insensitive to other load on the GPU.

Run in an environment that has gsplat (here: conda env ``vggt``):
    python scripts/time_3dgs_cuda_rasterizer.py \
        --gaussians outputs/multirun/render_3dgs/seed0/gaussians.pt
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gaussian_renderer import GaussianModel, render_gaussians, psnr  # noqa: E402


def rotvec_to_quat_wxyz(rv: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.norm(rv, dim=1, keepdim=True)
    half = 0.5 * theta
    small = theta < 1e-8
    k = torch.where(small, torch.zeros_like(rv), rv / theta.clamp(min=1e-12))
    w = torch.cos(half)
    xyz = k * torch.sin(half)
    q = torch.cat([w, xyz], dim=1)
    return q / torch.linalg.norm(q, dim=1, keepdim=True)


def cuda_time(fn, n_warm: int, n: int):
    for _ in range(n_warm):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))  # ms
    t = np.asarray(times)
    return {'mean_ms': float(t.mean()), 'median_ms': float(np.median(t)),
            'std_ms': float(t.std(ddof=1)), 'min_ms': float(t.min()),
            'fps_from_median': float(1000.0 / np.median(t)), 'n': int(n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gaussians', required=True)
    ap.add_argument('--n_warm', type=int, default=10)
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--out', default=str(ROOT / 'outputs' / 'stats'
                                         / 'cuda_rasterizer_timing.json'))
    args = ap.parse_args()

    import gsplat
    from gsplat import rasterization

    dev = 'cuda'
    g = torch.load(args.gaussians, map_location='cpu', weights_only=False)
    model = GaussianModel(g['mu'], g['log_scale'], g['rotvec'], g['color'],
                          g['logit'], device=dev)
    K = np.asarray(g['K_timing'], dtype=np.float64)
    H, W = int(g['H_timing']), int(g['W_timing'])
    T = np.asarray(g['T_timing'], dtype=np.float64)
    N = int(model.mu.shape[0])

    # ---- gsplat inputs (same parameterisation) ----
    means = model.mu.detach()
    quats = rotvec_to_quat_wxyz(model.rotvec.detach())
    scales = model.scales.detach()
    opac = model.opacity.detach()
    colors = model.color.detach().clamp(0, 1)
    viewmat = torch.as_tensor(T, dtype=torch.float32, device=dev)[None]
    Kt = torch.as_tensor(K, dtype=torch.float32, device=dev)[None]

    def run_gsplat():
        img, alpha, _ = rasterization(
            means, quats, scales, opac, colors, viewmat, Kt, W, H,
            render_mode='RGB', packed=False, sh_degree=None,
            backgrounds=torch.zeros(1, 3, device=dev))
        return img

    def run_torch_ref():
        return render_gaussians(model, T, K, H, W, device=dev)['feat']

    with torch.no_grad():
        img_g = run_gsplat()[0].cpu().numpy()
        img_t = run_torch_ref().cpu().numpy()
        agree_psnr = float(psnr(img_g, img_t))
        # interleave to share the same background load
        t_ref_1 = cuda_time(run_torch_ref, args.n_warm, args.n)
        t_gs_1 = cuda_time(run_gsplat, args.n_warm, args.n)
        t_ref_2 = cuda_time(run_torch_ref, 0, args.n)
        t_gs_2 = cuda_time(run_gsplat, 0, args.n)

    def merge(a, b):
        return {k: (a[k] + b[k]) / 2 if k != 'n' else a[k] + b[k] for k in a}
    t_ref = merge(t_ref_1, t_ref_2)
    t_gs = merge(t_gs_1, t_gs_2)

    try:
        import pynvml  # noqa: F401
        util = None
    except Exception:
        util = None

    report = {
        'gaussians_file': str(Path(args.gaussians).resolve().relative_to(ROOT)),
        'n_gaussians': N,
        'resolution': [W, H],
        'view': 'T_timing from run_3dgs_eval.py (frontal, z=900 mm)',
        'torch_reference_ewa': t_ref,
        'gsplat_cuda_tile_rasterizer': t_gs,
        'speedup_median': float(t_ref['median_ms'] / t_gs['median_ms']),
        'image_agreement_psnr_db': agree_psnr,
        'gsplat_version': gsplat.__version__,
        'torch_version': torch.__version__,
        'cuda_version': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0),
        'platform': platform.platform(),
        'note': ('Both renderers timed back to back in one process on the same '
                 'Gaussians and view; the machine was shared with other GPU '
                 'jobs during the revision, so absolute times are conservative '
                 'and the ratio is the robust quantity.'),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
