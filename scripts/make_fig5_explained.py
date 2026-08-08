# -*- coding: utf-8 -*-
"""Rebuild the clinical centerline figure (Fig: gal_clin_bronch) so the story is
self-explanatory: for each representative bronchoscopy frame we show the
registration pipeline as 4 stages with a shared legend.

Stages per frame (left to right):
  1. Live-View            raw endoscopic crop
  2. detected lumen       green 2D skeleton of the visible airway lumen
  3. before registration  red = CT centerline projected with the INITIAL pose
                          (misaligned -> red off the green)
  4. after FreeReg        red = CT centerline after centerline registration,
                          yellow = matched bifurcations (red hugs the green)

A bottom legend explains every colour. Output: figs/gal_clin_bronch.png
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
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for sibling scripts

from src.geometry import make_T, transform
from src.centerline import (airway_centerline_3d, lumen_skeleton_2d,
                            register_centerline, skeleton_junctions_2d,
                            crop_centerline_by_view)
from src.ct_loader import CTVolume
from make_centerline_endo_fig import extract_live_view, _align_rot, VIDEO, MASK_CACHE

OUT_DIR = Path('outputs/ct/centerline')
FIG_OUT = Path('outputs/paper/figs/gal_clin_bronch.png')

FRAMES = [14800]          # one representative frame -> one clear row
CELL = 340
GAP = 8
LEGEND_H = 52
TITLE_H = 34


def _font(size, bold=True):
    name = 'C:/Windows/Fonts/arialbd.ttf' if bold else 'C:/Windows/Fonts/arial.ttf'
    for p in (name, 'C:/Windows/Fonts/arial.ttf', 'arial.ttf'):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def project_points(pts, T, K, H, W):
    P = transform(T, pts)
    z = P[:, 2]
    ok = z > 1e-3
    u = np.round(K[0, 0] * P[ok, 0] / z[ok] + K[0, 2]).astype(int)
    v = np.round(K[1, 1] * P[ok, 1] / z[ok] + K[1, 2]).astype(int)
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return u[inb], v[inb]


def draw_red(bgr, u, v):
    vis = bgr.copy()
    for uu, vv in zip(u, v):
        cv2.circle(vis, (int(uu), int(vv)), 2, (0, 0, 255), -1)
    return vis


def stage_row(bgr, cl_pts, junc_idx, axis_dir, device):
    """Return (raw, lumen, before, after) BGR cells + chamfer meta."""
    H2, W2 = bgr.shape[:2]
    K2 = np.array([[W2 / 2., 0, W2 / 2.],
                   [0, W2 / 2., H2 / 2.], [0, 0, 1.0]], dtype=np.float64)
    skel, dt_img, _ = lumen_skeleton_2d(bgr, dark_pct=30)
    if int(skel.sum()) < 40:
        return None

    R0 = _align_rot(axis_dir)
    T_init = make_T(R0, np.array([0.0, 0.0, 60.0]) - R0 @ cl_pts.mean(0))
    cl_crop, crop_idx = crop_centerline_by_view(
        cl_pts, T_init, K2, H2, W2, margin_px=40, z_range=(3.0, 150.0))
    if len(cl_crop) < 50:
        cl_crop, crop_idx = cl_pts, np.arange(len(cl_pts))
    pos = {g: i for i, g in enumerate(crop_idx.tolist())}
    j_local = np.array([pos[j] for j in junc_idx if j in pos], dtype=np.int64)

    res = register_centerline(
        cl_crop, dt_img, K2, T_init, iters=250, lr=8e-3, device=device,
        cl_junction_idx=j_local, lambda_branch=3.0)

    # stage 1: raw
    c_raw = bgr.copy()
    # stage 2: lumen skeleton (green)
    c_lum = bgr.copy()
    c_lum[skel] = (0, 255, 0)
    # stage 3: before registration (green + red with INITIAL pose)
    c_bef = bgr.copy()
    c_bef[skel] = (0, 200, 0)
    ub, vb = project_points(cl_crop, T_init, K2, H2, W2)
    c_bef = draw_red(c_bef, ub, vb)
    # stage 4: after FreeReg (green + red aligned + yellow bifurcations)
    c_aft = bgr.copy()
    c_aft[skel] = (0, 255, 0)
    ua, va = project_points(cl_crop, res['T'], K2, H2, W2)
    c_aft = draw_red(c_aft, ua, va)
    j2d, _ = skeleton_junctions_2d(skel)
    if len(j_local) > 0 and len(j2d) > 0:
        Pj = transform(res['T'], cl_crop[j_local])
        zj = Pj[:, 2]
        okj = zj > 1e-3
        uj = np.round(K2[0, 0] * Pj[okj, 0] / zj[okj] + K2[0, 2]).astype(int)
        vj = np.round(K2[1, 1] * Pj[okj, 1] / zj[okj] + K2[1, 2]).astype(int)
        inj = (uj >= 0) & (uj < W2) & (vj >= 0) & (vj < H2)
        uj, vj = uj[inj], vj[inj]
        kept = []
        for uu, vv in zip(uj, vj):
            d = np.sqrt(((j2d - np.array([vv, uu])) ** 2).sum(1)).min()
            if d < 25:
                kept.append((int(uu), int(vv), d))
        kept = sorted(kept, key=lambda t: t[2])[:8]
        for uu, vv, _ in kept:
            cv2.circle(c_aft, (uu, vv), 6, (0, 255, 255), 2)
            cv2.circle(c_aft, (uu, vv), 2, (0, 0, 255), -1)

    cham0 = float(res['init_loss'])
    cham = float(res['final_loss'])
    return (c_raw, c_lum, c_bef, c_aft), cham0, cham


def chip(d, x, y, color_rgb, label):
    d.rectangle([x, y + 4, x + 26, y + 22], fill=color_rgb)
    d.text((x + 34, y), label, fill=(30, 30, 30), font=_font(24, bold=False))


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    z = np.load(MASK_CACHE)
    mask, spacing, origin = z['mask'], z['spacing'], z['origin']
    axis_dir = z['axis_dir']
    vol = CTVolume(hu=mask.astype(np.int16), spacing=spacing,
                   origin=origin, orientation=np.eye(3))
    cl_pts, junc_idx, end_idx = airway_centerline_3d(
        mask, volume=vol, return_junctions=True)
    print(f'centerline={len(cl_pts)} junctions={len(junc_idx)} device={device}')

    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        raise RuntimeError(f'cannot open {VIDEO}')

    rows = []
    for fi in FRAMES:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, bgr = cap.read()
        if not ok:
            print(f'f{fi} read fail')
            continue
        live = extract_live_view(bgr, out_size=512)
        if live is None:
            print(f'f{fi} no live-view')
            continue
        out = stage_row(live, cl_pts, junc_idx, axis_dir, device)
        if out is None:
            print(f'f{fi} stage fail')
            continue
        (c_raw, c_lum, c_bef, c_aft), cham0, cham = out
        rows.append(([c_raw, c_lum, c_bef, c_aft], cham0, cham, fi))
        print(f'f{fi}: chamfer {cham0:.1f} -> {cham:.2f}px')
    cap.release()
    if not rows:
        raise RuntimeError('no rows produced')

    nstage = 4
    titles = ['1. Live-View', '2. detected lumen',
              '3. before reg.', '4. after FreeReg']
    W = nstage * CELL + (nstage - 1) * GAP
    H = TITLE_H + len(rows) * CELL + (len(rows) - 1) * GAP + LEGEND_H
    canvas = Image.new('RGB', (W, H), (255, 255, 255))
    d = ImageDraw.Draw(canvas)

    # stage titles
    for c, ttl in enumerate(titles):
        tw = d.textlength(ttl, font=_font(26, bold=True))
        cx = c * (CELL + GAP) + CELL / 2
        d.text((cx - tw / 2, 2), ttl, fill=(20, 20, 20), font=_font(26, bold=True))

    y = TITLE_H
    for cells, cham0, cham, fi in rows:
        for c, cell in enumerate(cells):
            rgb = cv2.cvtColor(cell, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb).resize((CELL, CELL), Image.LANCZOS)
            canvas.paste(pil, (c * (CELL + GAP), y))
        # annotate chamfer drop on the last cell
        lab = f'centerline residual {cham0:.0f} -> {cham:.1f} px'
        tw = d.textlength(lab, font=_font(20, bold=False))
        d.text((W - tw - 8, y + 6), lab, fill=(255, 255, 255),
               font=_font(20, bold=False))
        y += CELL + GAP

    # legend
    ly = H - LEGEND_H + 12
    x = 10
    for rgb, lab in [((0, 200, 0), 'observed lumen skeleton (image)'),
                     ((220, 0, 0), 'projected CT centerline'),
                     ((230, 220, 0), 'matched bifurcation')]:
        chip(d, x, ly, rgb, lab)
        x += 34 + d.textlength(lab, font=_font(24, bold=False)) + 40

    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(FIG_OUT)
    print(f'wrote {FIG_OUT} size={canvas.size}')


if __name__ == '__main__':
    main()
