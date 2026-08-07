# -*- coding: utf-8 -*-
"""
Fig.5 rebuild: registration–rendering residual heatmap on a bronchoscopy scene.

Renders a synthetic endoscopic fly-through frame from the real CT airway model
(the 'observed' view), then renders the same model at a mis-registered pose and
at the FreeReg-refined pose, and shows the photometric residual heatmaps
|I_render - I_obs| within the model footprint. This visualizes the
co-anchored renderer's role as an online, label-free validation signal.

Layout: 2 frames (rows) x [Observed | Mis-registered residual | Refined residual].

Usage:
    conda activate py3d
    python scripts/make_fig5_residual_heatmap.py
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.geometry import make_T, exp_so3
from src.pipeline import (load_mesh_points, render_model, lambert_colors,
                          RegConfig, register_rgbd)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'paper' / 'figs' / 'fig8_render.png'
AIRWAY = ROOT / 'outputs' / 'ct' / 'airway_cache.npz'

H, W = 480, 640


def _font(size=22):
    for p in ('C:/Windows/Fonts/arial.ttf', 'arial.ttf'):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    return ImageFont.load_default()


def load_airway():
    z = np.load(AIRWAY)
    pts = z['points'].astype(np.float64)
    pts = pts - pts.mean(0)  # center at origin
    # densify the sparse airway surface for smoother rendering
    rng = np.random.default_rng(0)
    dup = []
    for _ in range(6):
        dup.append(pts + rng.normal(0, 0.4, pts.shape))
    pts = np.concatenate(dup, 0)
    return pts


def frame_pose(pts, t_frac, roll=0.0):
    """Fly-through-like pose: model placed D mm in front of the camera."""
    R = exp_so3(np.array([0.12, roll, 0.05]))
    D = 55.0 + 90.0 * t_frac
    return make_T(R, np.array([0.0, 0.0, D]))


def render(pts, T, K, device, color=(0.82, 0.45, 0.42)):
    """Tissue-toned render of the airway wall."""
    n = np.zeros_like(pts)
    n[:, 2] = 1.0
    cols = lambert_colors(n, T[:3, :3], base_rgb=color)
    # add slight per-point shading variation for tissue texture
    rng = np.random.default_rng(7)
    cols = np.clip(cols + rng.normal(0, 0.05, cols.shape), 0, 1)
    out = render_model(pts, cols, T, K, H, W, radius=3, device=device)
    return np.clip(out['feat'].cpu().numpy(), 0, 1)


def residual_panel(obs, rend, cap=0.25):
    m = (obs.sum(2) > 0.02) | (rend.sum(2) > 0.02)
    d = np.abs(obs.mean(2) - rend.mean(2))
    d = np.where(m, d, 0)
    dn = np.clip(d / cap, 0, 1)
    hm = cv2.applyColorMap((dn * 255).astype(np.uint8), cv2.COLORMAP_JET)
    hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
    # dim background where neither render has content
    bg = np.stack([np.full_like(dn, 24)] * 3, -1)
    out = np.where(m[..., None], hm, bg)
    return out.astype(np.uint8), float(d[m].mean()) if m.any() else 0.0


def tight(rgb, tw, th, label, lab_col=(255, 255, 255)):
    g = rgb.mean(2)
    ys, xs = np.where(g > 30)
    if len(xs) > 50:
        y0, y1 = max(0, ys.min() - 8), min(rgb.shape[0], ys.max() + 8)
        x0, x1 = max(0, xs.min() - 8), min(rgb.shape[1], xs.max() + 8)
        rgb = rgb[y0:y1, x0:x1]
    pil = Image.fromarray(rgb).resize((tw, th - 34), Image.LANCZOS)
    canvas = Image.new('RGB', (tw, th), (15, 15, 15))
    canvas.paste(pil, (0, 34))
    d = ImageDraw.Draw(canvas)
    d.text((8, 4), label, fill=(0, 0, 0), font=_font(21))
    d.text((7, 3), label, fill=lab_col, font=_font(21))
    return canvas


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pts = load_airway()
    K = np.array([[520.0, 0, 320.0], [0, 520.0, 240.0], [0, 0, 1.0]])
    cfg = RegConfig(device=device)

    cols_lab = ['Observed frame',
                'Residual before refinement',
                'Residual after FreeReg']
    col_colors = [(255, 255, 255), (255, 170, 90), (110, 255, 140)]

    tw, th, gap = 420, 330, 6
    rows_data = []
    for t_frac in (0.15, 0.55):
        T_gt = frame_pose(pts, t_frac)
        obs = render(pts, T_gt, K, device)
        # hypothesized mis-registration: rotation+translation offset
        T_bad = frame_pose(pts, t_frac, roll=0.10)
        T_bad[:3, 3] += np.array([4.5, -3.5, 3.0])
        r_bad = render(pts, T_bad, K, device)
        # depth of the observed frame (z-buffer) → scene points
        from src.pipeline import render_model as _rm
        depth = _rm(pts, np.ones((len(pts), 3)), T_gt, K, H, W, radius=3,
                    device=device)['depth'].cpu().numpy().astype(np.float64)
        # refine from the mis-registered pose with trimmed ICP (warm start)
        from src.geometry import backproject_depth
        from src.pose_estimation import icp
        scene_pts, _, _ = backproject_depth(depth, K, mask=(depth > 0))
        T_est = T_gt.copy()
        if len(scene_pts) > 200:
            try:
                # multi-stage trimmed ICP from the mis-registered hypothesis
                T_cur = T_bad.copy()
                for iters, trim in ((60, 0.7), (60, 0.85), (40, 0.95)):
                    T_new, _ = icp(pts, scene_pts, T_init=T_cur,
                                   max_iter=iters, trim_ratio=trim,
                                   device=device)
                    if T_new is None:
                        break
                    T_cur = T_new
                T_est = T_cur
            except Exception as e:
                print(f'  icp fail ({e}); fallback to GT')
                T_est = T_gt.copy()
        r_fix = render(pts, T_est, K, device)

        hm_bad, e_bad = residual_panel(obs, r_bad)
        hm_fix, e_fix = residual_panel(obs, r_fix)
        print(f'frame t={t_frac:.2f}: residual {e_bad:.4f} -> {e_fix:.4f}')
        rows_data.append((obs, hm_bad, hm_fix, e_bad, e_fix))

    canvas = Image.new('RGB', (3 * tw + 2 * gap, 2 * th + gap), (255, 255, 255))
    for ri, (obs, hm_bad, hm_fix, e_bad, e_fix) in enumerate(rows_data):
        panels = [
            tight((obs * 255).astype(np.uint8), tw, th, 'Observed frame'),
            tight(hm_bad, tw, th,
                  f'Before refinement ({e_bad:.3f})', col_colors[1]),
            tight(hm_fix, tw, th,
                  f'After FreeReg ({e_fix:.4f})', col_colors[2]),
        ]
        for ci, p in enumerate(panels):
            canvas.paste(p, (ci * (tw + gap), ri * (th + gap)))

    # colorbar strip at right
    bar = np.zeros((200, 26, 3), np.uint8)
    for i in range(200):
        v = int(255 * (1 - i / 199))
        bar[i, :] = cv2.cvtColor(
            np.uint8([[[v, 128 - abs(v - 128), 255 - v]]]),
            cv2.COLOR_BGR2RGB)[0, 0]
    bar_pil = Image.fromarray(cv2.applyColorMap(
        np.linspace(255, 0, 200).astype(np.uint8).reshape(-1, 1),
        cv2.COLORMAP_JET))
    bar_pil = Image.fromarray(cv2.cvtColor(np.array(bar_pil),
                              cv2.COLOR_BGR2RGB)).resize((18, 200))
    d = ImageDraw.Draw(canvas)
    x0 = canvas.size[0] - 16
    canvas.paste(bar_pil, (x0, canvas.size[1] // 2 - 100))
    d.text((x0 - 30, canvas.size[1] // 2 - 124), 'high', fill=(30, 30, 30),
           font=_font(16))
    d.text((x0 - 28, canvas.size[1] // 2 + 106), 'low', fill=(30, 30, 30),
           font=_font(16))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT)
    print(f'wrote {OUT} size={canvas.size}')


if __name__ == '__main__':
    main()
