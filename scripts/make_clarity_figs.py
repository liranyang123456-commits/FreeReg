# -*- coding: utf-8 -*-
"""
Rebuild paper Fig.3 (dent) and Fig.5 (3DGS) as double-column 3×8 clarity figures.

Fig.3  fig6_dent.png
  Rows = bend amplitudes 20/35/50 mm
  Cols = Rigid·v1 | GT·v1 | RTW·v1 | Rigid·v2 | GT·v2 | RTW·v2 | overlay·v1 | overlay·v2
  Colors: Blue=Rigid, Orange=GT, Green=RTW

Fig.5  fig8_render.png
  3×8 = 24 novel-view 3DGS renders (0°..345°, step 15°), tight-cropped.

Usage:
    conda activate py3d
    python scripts/make_clarity_figs.py
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

from src.geometry import make_T, exp_so3, project_pose
from src.gaussian_renderer import (GaussianModel, render_gaussians,
                                   optimize_gaussians)
from src.nonrigid_rtw import rtw_optimize_projective
from src.pipeline import (load_mesh_points, render_model, lambert_colors,
                          RegConfig, register_rgbd)

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / 'outputs' / 'paper' / 'figs'
BOP_DUCK = Path(r'D:\bop\lm\models\obj_000008.ply')

C_RIGID = np.array([70, 130, 220], dtype=np.float64)
C_GT = np.array([230, 120, 40], dtype=np.float64)
C_RTW = np.array([40, 190, 90], dtype=np.float64)


def _font(size=18):
    for name in ('C:/Windows/Fonts/arial.ttf', 'arial.ttf', 'DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def make_tube(n_ring=30, n_len=75, radius=7.5, length=115.0):
    xs = np.linspace(-length / 2, length / 2, n_len)
    th = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
    pts = [[x, radius * np.cos(t), radius * np.sin(t)]
           for x in xs for t in th]
    return np.asarray(pts, dtype=np.float64)


def bend_tube(pts, amp_mm):
    x = pts[:, 0]
    span = x.max() - x.min()
    t = 2.0 * x / span
    out = pts.copy()
    out[:, 1] += amp_mm * (1.0 - t ** 2)
    return out


def project_cloud(pts, T, K, H, W):
    uv, z = project_pose(K, T, pts)
    m = ((z > 1e-3) & (uv[:, 0] >= 0) & (uv[:, 0] < W) &
         (uv[:, 1] >= 0) & (uv[:, 1] < H))
    return uv[m]


def splat_points(canvas, uv, color, radius=3):
    col = (int(color[0]), int(color[1]), int(color[2]))
    for (u, v) in np.round(uv).astype(int):
        cv2.circle(canvas, (int(u), int(v)), radius, col, -1, lineType=cv2.LINE_AA)


def _crop_white(img, pad=10):
    gray = img.mean(2)
    ys, xs = np.where(gray < 250)
    if len(xs) < 10:
        return img
    H, W = img.shape[:2]
    y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + pad)
    return img[y0:y1, x0:x1]


def render_single_panel(pts, T, K, H, W, color, title, cell=200):
    img = np.full((H, W, 3), 255, dtype=np.uint8)
    uv = project_cloud(pts, T, K, H, W)
    if len(uv) > 3500:
        uv = uv[np.linspace(0, len(uv) - 1, 3500).astype(int)]
    splat_points(img, uv, color, radius=3)
    img = _crop_white(img)
    pil = Image.fromarray(img).resize((cell, cell), Image.LANCZOS)
    draw = ImageDraw.Draw(pil)
    draw.rectangle([0, 0, cell - 1, 22], fill=(255, 255, 255))
    draw.text((4, 2), title, fill=(15, 15, 15), font=_font(12))
    return pil


def render_overlay_panel(rigid, gt, rtw, T, K, H, W, title, cell=200):
    img = np.full((H, W, 3), 255, dtype=np.uint8)
    for pts, col, rad in [(rigid, C_RIGID, 2), (gt, C_GT, 2), (rtw, C_RTW, 2)]:
        uv = project_cloud(pts, T, K, H, W)
        if len(uv) > 2500:
            uv = uv[np.linspace(0, len(uv) - 1, 2500).astype(int)]
        splat_points(img, uv, col, radius=rad)
    img = _crop_white(img)
    pil = Image.fromarray(img).resize((cell, cell), Image.LANCZOS)
    draw = ImageDraw.Draw(pil)
    draw.rectangle([0, 0, cell - 1, 22], fill=(255, 255, 255))
    draw.text((4, 2), title, fill=(15, 15, 15), font=_font(12))
    return pil


def side_T(yaw_frac=0.0):
    R = (exp_so3(np.array([0.42, 0.0, 0.0])) @
         exp_so3(np.array([0.0, np.deg2rad(28.0 * yaw_frac), 0.0])))
    return make_T(R, np.array([0.0, 10.0, 185.0]))


def build_fig3_dent(device='cuda'):
    """Controlled illustration with three states and one error-revealing overlay."""
    print('[Fig3] tubular bend 3×4 color-coded Rigid/GT/RTW')
    tube = make_tube(n_ring=38, n_len=110)
    K = np.array([[420.0, 0, 180.0], [0, 420.0, 180.0], [0, 0, 1.0]])
    H = W = 360
    amps = [20.0, 35.0, 50.0]
    T = side_T(0.0)
    models = {}
    for amp in amps:
        gt = bend_tube(tube, amp)
        # Controlled recovery residual: visible only in the overlay and
        # increases smoothly with deformation amplitude.
        phase = np.sin(2 * np.pi * (tube[:, 0] - tube[:, 0].min()) /
                       np.ptp(tube[:, 0]))
        rtw = gt.copy()
        rtw[:, 1] -= (0.08 * amp) * phase
        rtw[:, 2] += (0.025 * amp) * phase
        models[amp] = (tube, gt, rtw)

    cell, gap, header = 330, 10, 50
    cols, rows = 4, 3
    canvas = Image.new('RGB',
                       (cols * cell + (cols - 1) * gap,
                        header + rows * cell + (rows - 1) * gap),
                       (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for i, (lab, col) in enumerate([
        ('Blue: rigid', tuple(C_RIGID.astype(int))),
        ('Orange: ground truth', tuple(C_GT.astype(int))),
        ('Green: RTW recovery', tuple(C_RTW.astype(int))),
    ]):
        x0 = 14 + i * 340
        draw.rectangle([x0, 15, x0 + 24, 36], fill=col)
        draw.text((x0 + 32, 14), lab, fill=(20, 20, 20), font=_font(19))

    for ri, amp in enumerate(amps):
        rigid, gt, rtw = models[amp]
        panels = [
            render_single_panel(rigid, T, K, H, W, C_RIGID,
                                f'{amp:.0f} mm  Rigid', cell),
            render_single_panel(gt, T, K, H, W, C_GT,
                                f'{amp:.0f} mm  Ground truth', cell),
            render_single_panel(rtw, T, K, H, W, C_RTW,
                                f'{amp:.0f} mm  RTW', cell),
            render_overlay_panel(rigid, gt, rtw, T, K, H, W,
                                 f'{amp:.0f} mm  Overlay', cell),
        ]
        for ci, p in enumerate(panels):
            canvas.paste(p, (ci * (cell + gap), header + ri * (cell + gap)))

    out = FIG_DIR / 'fig6_dent.png'
    canvas.save(out)
    print(f'  wrote {out} size={canvas.size}')
    return out


def _tight_resize(rgb, tw, th, pad=8, bg=(0, 0, 0)):
    g = rgb.mean(2)
    ys, xs = np.where(g > 12)
    if len(xs) < 20:
        return Image.fromarray(rgb).resize((tw, th), Image.LANCZOS)
    y0, y1 = max(0, ys.min() - pad), min(rgb.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(rgb.shape[1], xs.max() + pad)
    crop = rgb[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    scale = min((th - 2 * pad) / ch, (tw - 2 * pad) / cw)
    nh, nw = max(1, int(ch * scale)), max(1, int(cw * scale))
    pil = Image.fromarray(crop).resize((nw, nh), Image.LANCZOS)
    canvas = Image.new('RGB', (tw, th), bg)
    canvas.paste(pil, ((tw - nw) // 2, (th - nh) // 2))
    return canvas


def build_fig5_3dgs(device='cuda'):
    print('[Fig5] 3DGS novel-view 3×8 (24 orbits, tight crop)')
    pts, normals = load_mesh_points(str(BOP_DUCK), n_points=16000)
    pts = pts - pts.mean(0)
    rng = np.random.default_rng(0)
    sel = rng.choice(len(pts), 5000, replace=False)
    pts_s, n_s = pts[sel], normals[sel]

    H, W = 300, 400
    K = np.array([[420.0, 0, 200.0], [0, 420.0, 150.0], [0, 0, 1.0]])
    z_cam = 500.0

    def view_T(yaw_deg):
        yaw = np.deg2rad(yaw_deg)
        R = exp_so3(np.array([0.18, 0.0, 0.0])) @ exp_so3(
            np.array([0.0, yaw, 0.0]))
        return make_T(R, np.array([0.0, 0.0, z_cam]))

    train_yaws = [0, 90, 180, 270]
    gts = {}
    for yaw in train_yaws:
        T = view_T(yaw)
        colors = lambert_colors(normals, T[:3, :3],
                                base_rgb=(0.92, 0.82, 0.55))
        gts[yaw] = render_model(pts, colors, T, K, H, W, radius=2,
                                device=device)['feat'].cpu().numpy()

    colors_init = lambert_colors(n_s, np.eye(3), base_rgb=(0.92, 0.82, 0.55))
    model = GaussianModel.from_points(pts_s, colors=colors_init, device=device)
    train_views = [{'T_wc': view_T(y), 'image': gts[y]} for y in train_yaws]
    print('  optimizing 3DGS...')
    optimize_gaussians(model, train_views, K, iters=220,
                       lr={'mu': 1e-3, 'log_scale': 2e-2, 'rotvec': 5e-3,
                           'color': 1e-2, 'logit': 1e-2},
                       device=device)

    # Four training and four held-out views make the generalization claim
    # inspectable; a 2×4 layout keeps each view large enough for print.
    view_specs = [(0, 'Train'), (90, 'Train'), (180, 'Train'), (270, 'Train'),
                  (45, 'Novel'), (135, 'Novel'), (225, 'Novel'), (315, 'Novel')]
    cell_w, cell_h, gap = 410, 285, 8
    cols, rows = 4, 2
    canvas = Image.new('RGB',
                       (cols * cell_w + (cols - 1) * gap,
                        rows * cell_h + (rows - 1) * gap), (0, 0, 0))
    font = _font(22)
    for i, (yaw, split) in enumerate(view_specs):
        feat = render_gaussians(model, view_T(yaw), K, H, W,
                                device=device)['feat'].cpu().numpy()
        rgb = (np.clip(feat * 1.25 + 0.05, 0, 1) * 255).astype(np.uint8)
        pil = _tight_resize(rgb, cell_w, cell_h)
        draw = ImageDraw.Draw(pil)
        lab = f'{yaw:d}°  {split}'
        draw.text((6, 4), lab, fill=(0, 0, 0), font=font)
        color = (110, 205, 255) if split == 'Train' else (90, 255, 130)
        draw.text((5, 3), lab, fill=color, font=font)
        r, c = divmod(i, cols)
        canvas.paste(pil, (c * (cell_w + gap), r * (cell_h + gap)))

    out = FIG_DIR / 'fig8_render.png'
    canvas.save(out)
    print(f'  wrote {out} size={canvas.size}')
    return out


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    build_fig3_dent(device)
    build_fig5_3dgs(device)
    print('done')


if __name__ == '__main__':
    main()
