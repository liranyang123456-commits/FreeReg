# -*- coding: utf-8 -*-
"""FreeReg teaser (Fig. 1) — multi-model / multi-scene overview (5 pairs).

Each group is  [full 3D model] -> [scene + FreeReg registration], all produced
by the real pipeline (register_rgbd / centerline pipeline / composite_depth_aware):

  A  BOP driller  (obj5)  -> BOP Linemod RGB-D scene   (green silhouette)
  B  BOP object   (scene2)-> BOP Linemod RGB-D scene   (green silhouette)
  C  CT airway tree       -> bronchoscopy Live-View f1 (red centerline)
  D  CT airway tree       -> bronchoscopy Live-View f2 (red centerline)
  E  virtual duck (obj8)  -> SCARED endoscopic video   (green silhouette)

Flat single-row layout, 5 groups. Output: outputs/paper/figs/concept_freereg.png
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.geometry import make_T, exp_so3, inverse_se3
from src.pipeline import (RegConfig, load_mesh_points, register_rgbd,
                          render_model, composite_depth_aware, lambert_colors)

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'outputs' / 'paper' / 'figs'
OUT = FIG / 'concept_freereg.png'
BOP_ROOT = Path(r'D:\bop\lm')
SCARED = Path(r'E:\MIS_Datasets\SCARED\dataset_1\keyframe_1')
AIRWAY = ROOT / 'outputs' / 'ct' / 'airway_cache.npz'
CENTERLINE_DIR = ROOT / 'outputs' / 'ct' / 'centerline'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def _font(size, bold=True):
    name = 'C:/Windows/Fonts/arialbd.ttf' if bold else 'C:/Windows/Fonts/arial.ttf'
    for p in (name, 'C:/Windows/Fonts/arial.ttf', 'arial.ttf'):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def square_pad(im, size, fill=255):
    h, w = im.shape[:2]
    s = max(h, w)
    sq = np.full((s, s, 3), fill, np.uint8)
    sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = im
    return np.array(Image.fromarray(sq).resize((size, size), Image.LANCZOS))


def render_mesh_solid(pts, color, R_euler, size, radius=2, dist_scale=1.05,
                      densify=3):
    """Render a point set as a clean solid on white, upright, filling the frame."""
    pts = pts.astype(np.float64) - pts.mean(0)
    if densify > 1:
        rng = np.random.default_rng(0)
        s = pts.std(0).mean() * 0.008
        pts = np.concatenate([pts + rng.normal(0, s, pts.shape)
                              for _ in range(densify)], 0)
    K = np.array([[size, 0, size / 2.], [0, size, size / 2.], [0, 0, 1.]])
    ext = np.linalg.norm(pts.max(0) - pts.min(0))
    dist = ext * dist_scale
    T = make_T(exp_so3(np.array(R_euler)), np.array([0., 0., dist]))
    cols = np.tile(np.array(color), (len(pts), 1))
    r = render_model(pts, cols, T, K, size, size, radius=radius, device=DEVICE)
    img = (np.clip(r['feat'].cpu().numpy(), 0, 1) * 255).astype(np.uint8)
    bg = np.full_like(img, 255)
    m = img.sum(2) > 15
    if not m.any():
        return Image.new('RGB', (size, size), (255, 255, 255))
    bg[m] = img[m]
    ys, xs = np.where(m)
    y0, y1 = max(0, ys.min() - 10), min(size, ys.max() + 10)
    x0, x1 = max(0, xs.min() - 10), min(size, xs.max() + 10)
    crop = bg[y0:y1, x0:x1]
    return Image.fromarray(square_pad(crop, size, fill=255))


def overlay_green(rgb, mask, alpha=0.45):
    vis = rgb.copy()
    mb = mask.astype(bool)
    if not mb.any():
        return vis
    green = np.array([40, 220, 60], dtype=np.float64)
    vis[mb] = (1 - alpha) * vis[mb].astype(np.float64) + alpha * green
    vis = np.clip(vis, 0, 255).astype(np.uint8)
    bnd = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT,
                           np.ones((3, 3), np.uint8)) > 0
    bnd = cv2.dilate(bnd.astype(np.uint8), np.ones((2, 2), np.uint8)) > 0
    vis[bnd] = (20, 255, 40)
    return vis


def crop_around(rgb, mask, fill, size, pad_ratio=1.0, min_half=120):
    if mask.astype(bool).any():
        ys, xs = np.where(mask.astype(bool))
        cy, cx = int(ys.mean()), int(xs.mean())
        half = int(max(ys.max() - ys.min(), xs.max() - xs.min()) * pad_ratio)
        half = max(half, min_half)
        H, W = rgb.shape[:2]
        y0, y1 = max(0, cy - half), min(H, cy + half)
        x0, x1 = max(0, cx - half), min(W, cx + half)
        rgb = rgb[y0:y1, x0:x1]
    return square_pad(rgb, size, fill=fill)


# ---------- BOP pair ----------

def bop_pair(scene_id, fi, size, model_color=(0.52, 0.58, 0.72)):
    cfg = RegConfig(device=DEVICE)
    sdir = BOP_ROOT / 'test' / f'{scene_id:06d}'
    cams = json.load(open(sdir / 'scene_camera.json'))
    gts = json.load(open(sdir / 'scene_gt.json'))
    k = str(fi)
    if k not in cams:
        fi = min((int(x) for x in cams), key=lambda x: abs(x - fi))
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
    pts, normals = load_mesh_points(
        str(BOP_ROOT / 'models' / f'obj_{obj_id:06d}.ply'), n_points=20000)
    mv = sdir / 'mask_visib' / f'{fi:06d}_000000.png'
    mask = cv2.imread(str(mv), cv2.IMREAD_GRAYSCALE) if mv.exists() else None
    res = register_rgbd(depth, K, pts, mask=mask, cfg=cfg)
    T = res['T']
    rend = render_model(pts, np.ones((len(pts), 3)), T, K, H, W, radius=1,
                        device=DEVICE)
    m = rend['mask'].cpu().numpy().astype(np.uint8)
    vis = overlay_green(rgb, m)
    scene = crop_around(vis, m, 255, size, pad_ratio=0.9)
    model = render_mesh_solid(pts, model_color, (0.2, 0.3, 0.0), size)
    print(f'[BOP] obj{obj_id} scene{scene_id} f{fi}')
    return model, Image.fromarray(scene), f'obj{obj_id}'


# ---------- CT airway model + bronch scene ----------

def airway_model(size):
    z = np.load(AIRWAY)
    pts = z['points'].astype(np.float64)
    rng = np.random.default_rng(0)
    pts = np.concatenate([pts + rng.normal(0, 0.3, pts.shape)
                          for _ in range(12)], 0)
    return render_mesh_solid(pts, (0.25, 0.5, 0.95), (0.2, 0.0, 0.0), size,
                             radius=3, dist_scale=1.15, densify=1)


def bronch_scene(frame_name, size):
    src = CENTERLINE_DIR / frame_name
    if not src.exists():
        return Image.new('RGB', (size, size), (30, 30, 30))
    im = cv2.imread(str(src))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    g = im.mean(2)
    ys, xs = np.where(g > 25)
    if len(xs) < 2000:
        return Image.new('RGB', (size, size), (30, 30, 30))
    im = im[ys.min():ys.max(), xs.min():xs.max()]
    # brighten dark endoscopic tissue for print legibility
    im = np.clip(im.astype(np.float64) * 1.5 + 12, 0, 255).astype(np.uint8)
    return Image.fromarray(square_pad(im, size, fill=20))


# ---------- SCARED duck pair ----------

def scared_pair(size, frame_idx=100):
    fs = cv2.FileStorage(str(SCARED / 'endoscope_calibration.yaml'),
                         cv2.FILE_STORAGE_READ)
    K = fs.getNode('M1').mat()
    import tarfile
    poses = []
    with tarfile.open(str(SCARED / 'data' / 'frame_data.tar.gz'), 'r:gz') as tar:
        for name in sorted(tar.getnames()):
            d = json.loads(tar.extractfile(name).read().decode('utf-8'))
            poses.append(np.array(d['camera-pose'], dtype=np.float64))
    pts = []
    with open(SCARED / 'point_cloud.obj') as f:
        for line in f:
            if line.startswith('v '):
                p = line.split()
                pts.append((float(p[1]), float(p[2]), float(p[3])))
    scene_pts = np.array(pts, dtype=np.float64)
    sel = np.linspace(0, len(scene_pts) - 1, 150000).astype(int)
    scene_pts = scene_pts[sel]
    model, normals = load_mesh_points(
        str(BOP_ROOT / 'models' / 'obj_000008.ply'), n_points=20000)
    scale = 18.0 / np.linalg.norm(model.max(0) - model.min(0))
    model = model * scale
    finite = np.isfinite(scene_pts).all(axis=1)
    zq = np.percentile(scene_pts[finite, 2], 25)
    anchor = scene_pts[finite & (scene_pts[:, 2] < zq)].mean(axis=0)
    T_world_obj = make_T(np.eye(3), anchor)
    cap = cv2.VideoCapture(str(SCARED / 'data' / 'rgb.mp4'))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, fr = cap.read()
    cap.release()
    left = fr[:fr.shape[0] // 2]
    left_rgb = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
    H, W = left_rgb.shape[:2]
    T_cw = inverse_se3(poses[frame_idx])
    colors = lambert_colors(normals, np.eye(3), base_rgb=(0.2, 0.9, 0.3))
    scene_colors = np.tile(np.array([0.85, 0.75, 0.65]), (len(scene_pts), 1))
    r_virt = render_model(model, colors, T_cw @ T_world_obj, K, H, W,
                          radius=2, device=DEVICE)
    r_scene = render_model(scene_pts, scene_colors, T_cw, K, H, W,
                           radius=1, device=DEVICE)
    comp, occ = composite_depth_aware(
        torch.as_tensor(left_rgb.astype(np.float64) / 255.0,
                        dtype=torch.float32, device=r_virt['feat'].device),
        torch.as_tensor(r_scene['depth'].cpu().numpy(),
                        dtype=torch.float32, device=r_virt['feat'].device),
        r_virt['feat'], r_virt['depth'], alpha=0.85)
    comp = (np.clip(comp.cpu().numpy(), 0, 1) * 255).astype(np.uint8)
    vm = r_virt['mask'].cpu().numpy().astype(np.uint8)
    scene = crop_around(comp, vm, 20, size, pad_ratio=1.1)
    model_img = render_mesh_solid(model, (0.85, 0.68, 0.15), (0.3, 0.4, 0.0),
                                  size, radius=2, dist_scale=1.0, densify=3)
    print('[SCARED] duck')
    return model_img, Image.fromarray(scene)


def arrow(d, x0, y0, x1, color=(60, 60, 60)):
    d.line([x0, y0, x1, y0], fill=color, width=5)
    d.polygon([(x1, y0), (x1 - 13, y0 - 8), (x1 - 13, y0 + 8)], fill=color)


def main():
    S = 300
    # Build the 5 model->scene pairs (model already loaded per pair).
    mA, sA, idA = bop_pair(5, 215, S)
    mB, sB, idB = bop_pair(2, 243, S)
    mC = airway_model(S)
    sC = bronch_scene('endo_14800.png', S)
    mD = airway_model(S)
    sD = bronch_scene('endo_13400.png', S)
    mE, sE = scared_pair(S, 100)

    groups = [
        (mA, sA, f'BOP Linemod ({idA})'),
        (mB, sB, f'BOP Linemod ({idB})'),
        (mE, sE, 'SCARED  (endoscopy)'),
        (mC, sC, 'CT airway  (bronch.)'),
        (mD, sD, 'CT airway  (bronch.)'),
    ]
    # two rows: 3 groups top, 2 groups bottom
    rows = [groups[:3], groups[3:]]

    pad = 24
    arrow_w = 58
    group_gap = 40
    top_lab = 44
    group_w = S + arrow_w + S
    row_h = S + top_lab + 54
    row_gap = 26
    ncols = max(len(r) for r in rows)
    W = pad * 2 + group_w * ncols + group_gap * (ncols - 1)
    H = pad + row_h * len(rows) + row_gap * (len(rows) - 1)
    canvas = Image.new('RGB', (W, H), (255, 255, 255))
    d = ImageDraw.Draw(canvas)

    y0 = pad
    for row in rows:
        # centre shorter rows
        row_w = group_w * len(row) + group_gap * (len(row) - 1)
        x = (W - row_w) // 2
        ymid = y0 + top_lab + S // 2
        for (mpim, spim, ttl) in row:
            canvas.paste(mpim, (x, y0 + top_lab))
            d.rectangle([x, y0 + top_lab, x + S, y0 + top_lab + S],
                        outline=(205, 205, 205), width=2)
            arrow(d, x + S + 5, ymid, x + S + arrow_w - 5)
            xs = x + S + arrow_w
            canvas.paste(spim, (xs, y0 + top_lab))
            d.rectangle([xs, y0 + top_lab, xs + S, y0 + top_lab + S],
                        outline=(205, 205, 205), width=2)
            tw = d.textlength(ttl, font=_font(30, bold=True))
            cxg = x + group_w / 2
            d.text((cxg - tw / 2, y0 + top_lab + S + 10), ttl,
                   fill=(25, 25, 25), font=_font(30, bold=True))
            mw = d.textlength('3D model', font=_font(20, bold=False))
            d.text((x + S / 2 - mw / 2, y0 + top_lab - 30), '3D model',
                   fill=(110, 110, 110), font=_font(20, bold=False))
            sw = d.textlength('registered overlay', font=_font(20, bold=False))
            d.text((xs + S / 2 - sw / 2, y0 + top_lab - 30),
                   'registered overlay', fill=(110, 110, 110),
                   font=_font(20, bold=False))
            x += group_w + group_gap
        y0 += row_h + row_gap

    OUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT)
    print(f'wrote {OUT} size={canvas.size} ratio={canvas.size[0]/canvas.size[1]:.2f}')


if __name__ == '__main__':
    main()
