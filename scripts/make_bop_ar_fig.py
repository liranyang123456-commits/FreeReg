# -*- coding: utf-8 -*-
"""
Rebuild Fig 7 (gal_ar.png): BOP Linemod AR overlays (green silhouette + axes).

Seven panels = four original scenes/objects + three additional viewpoints.
Tight crop, white gutters, no virtual-render black columns.

Usage:
    conda activate py3d
    python scripts/make_bop_ar_fig.py
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.geometry import project_pose
from src.metrics import add_s_error
from src.pipeline import RegConfig, load_mesh_points, register_rgbd, render_model

BOP_ROOT = Path(r'D:\bop\lm')
FIG_OUT = Path('outputs/paper/figs/gal_ar.png')
CELL_DIR = Path('outputs/paper/figs/_cells_ar')

# (scene_id, frame_idx) — diverse objects / viewpoints / occlusions
FRAMES = [
    (1, 15), (1, 635), (2, 243), (5, 215),   # original four
    (3, 50), (4, 100), (6, 80),              # three additional
]

PANEL_H = 360
GAP = 6  # white gutter between panels


def draw_axes(img_rgb, K, T, diameter):
    L = diameter / 2.0
    pts3d = np.array([[0, 0, 0], [L, 0, 0], [0, L, 0], [0, 0, L]],
                     dtype=np.float64)
    uv, z = project_pose(K, T, pts3d)
    if (z <= 0).any():
        return img_rgb
    o = tuple(np.round(uv[0]).astype(int))
    # RGB axes in RGB image space
    cols = [(255, 40, 40), (40, 220, 40), (40, 80, 255)]
    for i in range(1, 4):
        p = tuple(np.round(uv[i]).astype(int))
        cv2.line(img_rgb, o, p, cols[i - 1], 2, cv2.LINE_AA)
    return img_rgb


def overlay_silhouette(rgb, mask, alpha=0.42):
    """Green fill + hard contour on live RGB."""
    vis = rgb.copy()
    m = mask.astype(bool)
    if not m.any():
        return vis
    green = np.array([40, 220, 60], dtype=np.float64)
    vis[m] = (1.0 - alpha) * vis[m].astype(np.float64) + alpha * green
    vis = np.clip(vis, 0, 255).astype(np.uint8)
    bnd = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_GRADIENT,
                           np.ones((3, 3), np.uint8)) > 0
    # thicken contour slightly
    bnd = cv2.dilate(bnd.astype(np.uint8), np.ones((2, 2), np.uint8), 1) > 0
    vis[bnd] = (20, 255, 40)
    return vis


def content_crop(rgb, mask, margin=0.04):
    """Drop near-empty borders only; keep full scene context around the object."""
    H, W = rgb.shape[:2]
    # Prefer object+RGB activity so cluttered backgrounds stay
    gray = rgb.mean(2)
    active = (gray > 12) | (mask.astype(bool))
    ys, xs = np.where(active)
    if len(xs) < 50:
        return rgb
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    pad_y = int(margin * H)
    pad_x = int(margin * W)
    y0 = max(0, y0 - pad_y)
    y1 = min(H, y1 + pad_y)
    x0 = max(0, x0 - pad_x)
    x1 = min(W, x1 + pad_x)
    # Reject pathological skinny crops — fall back to full frame
    if (x1 - x0) < 0.6 * W or (y1 - y0) < 0.45 * H:
        return rgb
    return rgb[y0:y1, x0:x1]


def main():
    CELL_DIR.mkdir(parents=True, exist_ok=True)
    cfg = RegConfig(device='cuda')
    models_info = json.load(open(BOP_ROOT / 'models' / 'models_info.json'))
    cache = {}

    def get_model(obj_id):
        if obj_id not in cache:
            pts, normals = load_mesh_points(
                str(BOP_ROOT / 'models' / f'obj_{obj_id:06d}.ply'),
                n_points=20000)
            cache[obj_id] = (pts, normals)
        return cache[obj_id]

    cells = []
    for scene_id, fi in FRAMES:
        sdir = BOP_ROOT / 'test' / f'{scene_id:06d}'
        cams = json.load(open(sdir / 'scene_camera.json'))
        gts = json.load(open(sdir / 'scene_gt.json'))
        k = str(fi)
        # fallback: nearest existing key
        if k not in cams:
            keys = sorted(int(x) for x in cams.keys())
            nearest = min(keys, key=lambda x: abs(x - fi))
            print(f'[warn] scene{scene_id} f{fi} missing → use {nearest}')
            fi = nearest
            k = str(fi)
        K = np.array(cams[k]['cam_K']).reshape(3, 3)
        rgb = cv2.cvtColor(cv2.imread(str(sdir / 'rgb' / f'{fi:06d}.png')),
                           cv2.COLOR_BGR2RGB)
        depth = cv2.imread(str(sdir / 'depth' / f'{fi:06d}.png'),
                           cv2.IMREAD_UNCHANGED).astype(np.float64)
        depth *= cams[k].get('depth_scale', 1.0)
        H, W = depth.shape

        g = gts[k][0]
        obj_id = g['obj_id']
        pts, normals = get_model(obj_id)
        diameter = models_info[str(obj_id)]['diameter']
        mv = sdir / 'mask_visib' / f'{fi:06d}_000000.png'
        mask = cv2.imread(str(mv), cv2.IMREAD_GRAYSCALE) if mv.exists() else None

        res = register_rgbd(depth, K, pts, mask=mask, cfg=cfg)
        T_est = res['T']
        T_gt = np.eye(4)
        T_gt[:3, :3] = np.array(g['cam_R_m2c']).reshape(3, 3)
        T_gt[:3, 3] = np.array(g['cam_t_m2c']).reshape(3)
        eval_pts = pts[::max(1, len(pts) // 2000)]
        adds = add_s_error(eval_pts, T_est, T_gt)

        rend = render_model(pts, np.ones((len(pts), 3)), T_est, K, H, W,
                            radius=1, device=cfg.device)
        m = rend['mask'].cpu().numpy().astype(np.uint8)
        vis = overlay_silhouette(rgb, m)
        vis = draw_axes(vis, K, T_est, diameter)
        cell = content_crop(vis, m)

        cell_path = CELL_DIR / f'ar_s{scene_id:02d}_f{fi:04d}_obj{obj_id:02d}.png'
        Image.fromarray(cell).save(cell_path)
        cells.append(cell)
        print(f'[AR] scene{scene_id} f{fi} obj{obj_id:02d} '
              f'ADD-S={adds:.2f}mm → {cell_path.name} {cell.shape[1]}x{cell.shape[0]}')

    # compose 1×N with white gutters, uniform height
    resized = []
    for c in cells:
        h, w = c.shape[:2]
        nh = PANEL_H
        nw = max(1, int(round(w * nh / h)))
        resized.append(np.array(Image.fromarray(c).resize((nw, nh), Image.LANCZOS)))

    total_w = sum(c.shape[1] for c in resized) + GAP * (len(resized) - 1)
    canvas = np.full((PANEL_H, total_w, 3), 255, dtype=np.uint8)
    x = 0
    for c in resized:
        canvas[:, x:x + c.shape[1]] = c
        x += c.shape[1] + GAP

    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(FIG_OUT)
    print(f'wrote {FIG_OUT} size={canvas.shape[1]}x{canvas.shape[0]} n={len(resized)}')


if __name__ == '__main__':
    main()
