# -*- coding: utf-8 -*-
"""
Rebuild Fig 8 (gal_clin_bronch): centerline overlays on pure endoscopic ROIs.

1) Detect Ion Live-View diamond/square in screen-capture frames
2) Warp to upright square
3) Run FreeReg centerline registration on the crop
4) Overlay: green lumen skeleton, red projected CT centerline, yellow bifurcations
5) Write outputs/ct/centerline/centerline_overlay_endo.png and paper figs/gal_clin_bronch.png

Usage:
    conda activate py3d
    python scripts/make_centerline_endo_fig.py
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.geometry import make_T, transform
from src.centerline import (airway_centerline_3d, lumen_skeleton_2d,
                            register_centerline, skeleton_junctions_2d,
                            crop_centerline_by_view)
from src.ct_loader import CTVolume

VIDEO = (r'F:\省医-数据\省医-数据\ION患者整理\001王继深\术中视频'
         r'\Video_screenshot_Henan1\ion_video_2021-10-26_03-33-50_L.mp4')
MASK_CACHE = Path('outputs/ct/airway_mask.npz')
OUT_DIR = Path('outputs/ct/centerline')
FIG_OUT = Path('outputs/paper/figs/gal_clin_bronch.png')

N_PANELS = 20  # paper Fig 8: two rows × 10 columns
N_COLS = 10
N_ROWS = 2
GAP = 4  # light gutter between cells

# Prefer frames previously found to contain pink Live-View content
CANDIDATE_FRAMES = [
    11000, 11200, 11400, 11600, 11800,
    12000, 12200, 12400, 12600, 12800, 13000, 13200, 13400,
    13600, 13800, 14000, 14200, 14400, 14600, 14800, 15000,
    15200, 15400, 15600, 15800, 16000, 16200, 16400, 16600,
    16800, 17000, 17200, 17400, 17600, 18000,
    26000, 27000, 28000, 29000, 30000, 32000, 34000,
]


def _order_box_pts(pts):
    """Order 4 box points as tl, tr, br, bl."""
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)



def extract_live_view(bgr, out_size=512):
    """
    Crop Ion Live-View (rotated ~45 deg diamond) from a 1920x2400 screen
    capture, rotate upright, and trim to tissue-only square.
    Returns BGR square or None.
    """
    H, W = bgr.shape[:2]
    y_off = int(H * 0.42)
    bot = bgr[y_off:, :]
    rgb = cv2.cvtColor(bot, cv2.COLOR_BGR2RGB)

    pink = ((rgb[:, :, 0] > 70) &
            (rgb[:, :, 0] > rgb[:, :, 1] + 8) &
            (rgb.mean(2) < 220) &
            (rgb.mean(2) > 35))
    mask = pink.astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            np.ones((25, 25), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            np.ones((9, 9), np.uint8), iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[i, cv2.CC_STAT_AREA])
    if area < 25000:
        return None
    x, y, w, h = [int(v) for v in stats[i, :4]]
    pad = 40
    x0 = max(0, x - pad)
    y0 = max(0, y_off + y - pad)
    x1 = min(W, x + w + pad)
    y1 = min(H, y_off + y + h + pad)
    crop = bgr[y0:y1, x0:x1]
    if min(crop.shape[:2]) < 160:
        return None

    cy, cx = crop.shape[0] / 2.0, crop.shape[1] / 2.0
    best = None
    best_score = -1.0
    for angle in (45.0, -45.0, 40.0, -40.0, 50.0, -50.0):
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        nW = int(crop.shape[0] * sin + crop.shape[1] * cos)
        nH = int(crop.shape[0] * cos + crop.shape[1] * sin)
        M = M.copy()
        M[0, 2] += (nW / 2) - cx
        M[1, 2] += (nH / 2) - cy
        rot = cv2.warpAffine(crop, M, (nW, nH),
                             flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        gray = cv2.cvtColor(rot, cv2.COLOR_BGR2GRAY)
        ys, xs = np.where(gray > 20)
        if len(xs) < 500:
            continue
        yy0, yy1 = int(ys.min()), int(ys.max())
        xx0, xx1 = int(xs.min()), int(xs.max())
        region = rot[yy0:yy1 + 1, xx0:xx1 + 1]
        rh, rw = region.shape[:2]
        aspect = rw / max(rh, 1)
        if aspect < 0.75 or aspect > 1.35:
            continue
        # shrink while corner wedges are mostly black (rotation artifacts)
        sq0 = region.copy()
        sh, sw = sq0.shape[:2]
        side = min(sh, sw)
        cy2, cx2 = sh // 2, sw // 2
        for _ in range(30):
            sq = sq0[cy2 - side // 2:cy2 - side // 2 + side,
                     cx2 - side // 2:cx2 - side // 2 + side]
            g2 = cv2.cvtColor(sq, cv2.COLOR_BGR2GRAY)
            c = 28  # corner patch size
            corners = [g2[:c, :c], g2[:c, -c:], g2[-c:, :c], g2[-c:, -c:]]
            wedge = max(float((p < 18).mean()) for p in corners)
            if wedge < 0.12 or side <= 190:
                break
            side -= 8
        if side < 190:
            continue
        sq = cv2.resize(sq, (out_size, out_size), interpolation=cv2.INTER_AREA)
        # scrub the Ion 'xx mm' UI chip (dark plate with bright digits,
        # top-right region) at the source via inpainting
        h2, w2 = sq.shape[:2]
        chip = sq[0:int(0.16 * h2), int(0.66 * w2):]
        chip_gray = cv2.cvtColor(chip, cv2.COLOR_BGR2GRAY)
        if float((chip_gray < 70).mean()) > 0.10:
            mask_ui = np.zeros((h2, w2), np.uint8)
            m = (chip_gray < 70).astype(np.uint8) * 255
            m = cv2.dilate(m, np.ones((5, 5), np.uint8), 2)
            mask_ui[0:int(0.16 * h2), int(0.66 * w2):] = m
            sq = cv2.inpaint(sq, mask_ui, 6, cv2.INPAINT_TELEA)
        rgb_w = cv2.cvtColor(sq, cv2.COLOR_BGR2RGB)
        dark = float((rgb_w.mean(2) < 40).mean())
        pink_f = float(((rgb_w[:, :, 0] > 70) &
                        (rgb_w[:, :, 0] > rgb_w[:, :, 1] + 8)).mean())
        edges = cv2.Canny(cv2.cvtColor(sq, cv2.COLOR_BGR2GRAY), 40, 120)
        edge_f = float((edges > 0).mean())
        ui_black = float((rgb_w.mean(2) < 18).mean())
        if pink_f < 0.12 or dark < 0.03 or dark > 0.55 or edge_f < 0.008:
            continue
        if ui_black > 0.25:
            sq = sq[int(out_size * 0.08):int(out_size * 0.98),
                    int(out_size * 0.02):int(out_size * 0.82)]
            sq = cv2.resize(sq, (out_size, out_size), interpolation=cv2.INTER_AREA)
            rgb_w = cv2.cvtColor(sq, cv2.COLOR_BGR2RGB)
            if float((rgb_w.mean(2) < 18).mean()) > 0.25:
                continue
        score = pink_f * (0.05 + dark) * (0.01 + edge_f) * (1.1 - abs(1.0 - aspect))
        if score > best_score:
            best_score = score
            best = sq
    return best


def _align_rot(vec, dst=(0, 0, 1.0)):
    vec = vec / (np.linalg.norm(vec) + 1e-12)
    dst = np.asarray(dst, dtype=float)
    v = np.cross(vec, dst)
    s = np.linalg.norm(v)
    c = float(vec @ dst)
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    K_ = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K_ + K_ @ K_ * (1 - c) / (s * s)


def overlay_centerline(bgr, cl_pts, junc_idx, axis_dir, device):
    """Register + draw green skeleton / red CT centerline / yellow junctions."""
    H2, W2 = bgr.shape[:2]
    K2 = np.array([[W2 / 2., 0, W2 / 2.],
                   [0, W2 / 2., H2 / 2.],
                   [0, 0, 1.0]], dtype=np.float64)
    skel, dt_img, _ = lumen_skeleton_2d(bgr, dark_pct=30)
    if int(skel.sum()) < 40:
        return None, dict(reason='skel_too_small', skel=int(skel.sum()))

    R0 = _align_rot(axis_dir)
    T_init = make_T(R0, np.array([0.0, 0.0, 60.0]) - R0 @ cl_pts.mean(0))
    cl_crop, crop_idx = crop_centerline_by_view(
        cl_pts, T_init, K2, H2, W2, margin_px=40, z_range=(3.0, 150.0))
    if len(cl_crop) < 50:
        cl_crop, crop_idx = cl_pts, np.arange(len(cl_pts))
    jset = set(crop_idx.tolist())
    j_local = np.array([j for j in junc_idx if j in jset], dtype=np.int64)
    pos = {g: i for i, g in enumerate(crop_idx.tolist())}
    j_local = np.array([pos[j] for j in j_local], dtype=np.int64)

    res = register_centerline(
        cl_crop, dt_img, K2, T_init, iters=250, lr=8e-3, device=device,
        cl_junction_idx=j_local, lambda_branch=3.0)

    vis = bgr.copy()
    # green = image lumen skeleton
    vis[skel] = (0, 255, 0)
    # red = projected CT centerline
    P = transform(res['T'], cl_crop)
    z = P[:, 2]
    okm = z > 1e-3
    u = np.round(K2[0, 0] * P[okm, 0] / z[okm] + K2[0, 2]).astype(int)
    v = np.round(K2[1, 1] * P[okm, 1] / z[okm] + K2[1, 2]).astype(int)
    inb = (u >= 0) & (u < W2) & (v >= 0) & (v < H2)
    for uu, vv in zip(u[inb], v[inb]):
        cv2.circle(vis, (int(uu), int(vv)), 2, (0, 0, 255), -1)
    # yellow = only bifurcations near 2D skeleton junctions (avoid yellow blobs)
    j2d, _ = skeleton_junctions_2d(skel)
    if len(j_local) > 0 and len(j2d) > 0:
        Pj = transform(res['T'], cl_crop[j_local])
        zj = Pj[:, 2]
        okj = zj > 1e-3
        uj = np.round(K2[0, 0] * Pj[okj, 0] / zj[okj] + K2[0, 2]).astype(int)
        vj = np.round(K2[1, 1] * Pj[okj, 1] / zj[okj] + K2[1, 2]).astype(int)
        inj = (uj >= 0) & (uj < W2) & (vj >= 0) & (vj < H2)
        uj, vj = uj[inj], vj[inj]
        # keep projected junctions within 25px of a 2D skeleton junction
        kept = []
        for uu, vv in zip(uj, vj):
            d = np.sqrt(((j2d - np.array([vv, uu])) ** 2).sum(1)).min()
            if d < 25:
                kept.append((int(uu), int(vv), d))
        kept = sorted(kept, key=lambda t: t[2])[:8]
        for uu, vv, _ in kept:
            cv2.circle(vis, (uu, vv), 6, (0, 255, 255), 2)
            cv2.circle(vis, (uu, vv), 2, (0, 0, 255), -1)

    meta = dict(cham0=float(res['init_loss']), cham=float(res['final_loss']),
                skel=int(skel.sum()), n_cl=int(len(cl_crop)),
                n_j=int(len(j_local)))
    return vis, meta


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('=' * 66)
    print('Rebuild centerline Fig 8 on pure endoscopic Live-View crops')
    print('=' * 66)
    if not MASK_CACHE.exists():
        raise FileNotFoundError(MASK_CACHE)
    z = np.load(MASK_CACHE)
    mask, spacing, origin = z['mask'], z['spacing'], z['origin']
    axis_dir = z['axis_dir']
    vol = CTVolume(hu=mask.astype(np.int16), spacing=spacing,
                   origin=origin, orientation=np.eye(3))
    cl_pts, junc_idx, end_idx = airway_centerline_3d(
        mask, volume=vol, return_junctions=True)
    print(f'centerline={len(cl_pts)} junctions={len(junc_idx)} ends={len(end_idx)} '
          f'device={device}')

    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        raise RuntimeError(f'cannot open {VIDEO}')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panels = []
    used = []
    for fi in CANDIDATE_FRAMES:
        if len(panels) >= N_PANELS:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, bgr = cap.read()
        if not ok:
            print(f'  f{fi}: read fail')
            continue
        live = extract_live_view(bgr, out_size=512)
        if live is None:
            print(f'  f{fi}: no live-view crop')
            continue
        vis, meta = overlay_centerline(live, cl_pts, junc_idx, axis_dir, device)
        if vis is None:
            print(f'  f{fi}: overlay fail ({meta})')
            continue
        print(f'  f{fi}: OK skel={meta["skel"]} chamfer '
              f'{meta["cham0"]:.1f}->{meta["cham"]:.2f}px '
              f'cl={meta["n_cl"]} j={meta["n_j"]}')
        cv2.imwrite(str(OUT_DIR / f'endo_{fi:05d}.png'), vis)
        panels.append(vis)
        used.append(fi)
    cap.release()

    if len(panels) < N_PANELS:
        print(f'WARNING: only {len(panels)} panels; scanning more frames...')
        cap = cv2.VideoCapture(VIDEO)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for fi in range(10000, min(total, 55000), 250):
            if len(panels) >= N_PANELS:
                break
            if fi in used:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, bgr = cap.read()
            if not ok:
                continue
            live = extract_live_view(bgr, out_size=512)
            if live is None:
                continue
            vis, meta = overlay_centerline(live, cl_pts, junc_idx, axis_dir, device)
            if vis is None:
                continue
            print(f'  f{fi}: OK (scan) skel={meta["skel"]} '
                  f'chamfer {meta["cham0"]:.1f}->{meta["cham"]:.2f}px')
            cv2.imwrite(str(OUT_DIR / f'endo_{fi:05d}.png'), vis)
            panels.append(vis)
            used.append(fi)
        cap.release()

    if not panels:
        raise RuntimeError('no endoscopic panels produced')

    # strip collage
    strip = np.concatenate(panels[:N_PANELS], axis=1)
    cv2.imwrite(str(OUT_DIR / 'centerline_overlay_endo.png'), strip)
    print(f'wrote {OUT_DIR / "centerline_overlay_endo.png"}')

    # 2×10 paper figure with thin white gutters
    tw, th = 320, 320
    n = min(N_PANELS, len(panels))
    cols = N_COLS
    rows = int(np.ceil(n / cols))
    cw = tw * cols + GAP * max(0, cols - 1)
    ch = th * rows + GAP * max(0, rows - 1)
    canvas = Image.new('RGB', (cw, ch), (255, 255, 255))
    for k, vis in enumerate(panels[:n]):
        rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((tw, th), Image.LANCZOS)
        r, c = divmod(k, cols)
        canvas.paste(pil, (c * (tw + GAP), r * (th + GAP)))
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(FIG_OUT)
    print(f'wrote {FIG_OUT} size={canvas.size} frames={used[:n]}')
    print('=' * 66)


if __name__ == '__main__':
    main()
