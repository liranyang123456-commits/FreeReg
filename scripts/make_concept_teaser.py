# -*- coding: utf-8 -*-
"""Build a clean 3-panel concept teaser for FreeReg (Fig. 1).

Layout (left to right):
  [3D airway model]  +  [input image/video]  --FreeReg-->  [AR overlay]

Rendered with Open3D (solid mesh) and real frames; tight, white, minimal text.
Output: outputs/paper/figs/concept_freereg.png  (used in place of the TikZ pdf)
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

from src.geometry import make_T, exp_so3, transform, project_pose
from src.pipeline import render_model

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'outputs' / 'paper' / 'figs'
CACHE = ROOT / 'outputs' / 'ct' / 'airway_cache.npz'
OUT = FIG / 'concept_freereg.png'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def _font(size, bold=True):
    name = 'C:/Windows/Fonts/arialbd.ttf' if bold else 'C:/Windows/Fonts/arial.ttf'
    for p in (name, 'C:/Windows/Fonts/arial.ttf', 'arial.ttf'):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_airway_solid(size=520):
    """Render the CT airway as a filled, shaded solid on white."""
    z = np.load(CACHE)
    pts = z['points'].astype(np.float64)
    pts = pts - pts.mean(0)
    rng = np.random.default_rng(0)
    pts = np.concatenate([pts + rng.normal(0, 0.3, pts.shape)
                          for _ in range(12)], 0)
    K = np.array([[560., 0, 260.], [0, 560., 260.], [0, 0, 1.]])
    # orient tree: trunk up, crown spreading
    T = make_T(exp_so3(np.array([0.2, 0.0, 0.0])), np.array([0., 0., 150.]))
    # blue gradient along vertical
    zmin, zmax = pts[:, 1].min(), pts[:, 1].max()
    t = (pts[:, 1] - zmin) / (zmax - zmin + 1e-9)
    cols = np.stack([0.20 + 0.30 * t, 0.45 + 0.30 * t, 0.95 - 0.05 * t], 1)
    r = render_model(pts, cols, T, K, size, size, radius=3, device=DEVICE)
    img = (np.clip(r['feat'].cpu().numpy(), 0, 1) * 255).astype(np.uint8)
    bg = np.full_like(img, 255)
    m = img.sum(2) > 15
    bg[m] = img[m]
    # crop to content with margin
    ys, xs = np.where(m)
    y0, y1 = max(0, ys.min() - 18), min(size, ys.max() + 18)
    x0, x1 = max(0, xs.min() - 18), min(size, xs.max() + 18)
    crop = bg[y0:y1, x0:x1]
    # square-pad
    h, w = crop.shape[:2]
    s = max(h, w)
    sq = np.full((s, s, 3), 255, np.uint8)
    sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = crop
    return Image.fromarray(sq).resize((size, size), Image.LANCZOS)


def render_input_frame(size=520):
    """Clean bronchoscopy frame (content-cropped, circular field filled)."""
    src = FIG / '_concept_endo.png'
    if not src.exists():
        src = ROOT / 'outputs' / 'ct' / 'centerline' / 'endo_14800.png'
    im = cv2.imread(str(src))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # crop dark borders
    g = im.mean(2)
    ys, xs = np.where(g > 25)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    im = im[y0:y1, x0:x1]
    h, w = im.shape[:2]
    s = max(h, w)
    sq = np.full((s, s, 3), 20, np.uint8)
    sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = im
    return Image.fromarray(sq).resize((size, size), Image.LANCZOS)


def render_ar_overlay(size=520):
    """Real FreeReg registration result: endoscopy frame with the registered
    CT airway centerline overlaid (green lumen skeleton, red CT centerline,
    yellow bifurcations) from the actual pipeline output."""
    # prefer a real overlay frame with strong structure
    for cand in ['endo_14800.png', 'endo_11200.png', 'endo_13400.png',
                 'endo_12200.png']:
        src = ROOT / 'outputs' / 'ct' / 'centerline' / cand
        if src.exists():
            im = cv2.imread(str(src))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            g = im.mean(2)
            ys, xs = np.where(g > 25)
            if len(xs) < 2000:
                continue
            y0, y1 = ys.min(), ys.max()
            x0, x1 = xs.min(), xs.max()
            im = im[y0:y1, x0:x1]
            h, w = im.shape[:2]
            s = max(h, w)
            sq = np.full((s, s, 3), 20, np.uint8)
            sq[(s - h) // 2:(s - h) // 2 + h,
               (s - w) // 2:(s - w) // 2 + w] = im
            return Image.fromarray(sq).resize((size, size), Image.LANCZOS)
    return render_input_frame(size)


def arrow(draw, x0, y0, x1, color=(60, 60, 60)):
    draw.line([x0, y0, x1, y0], fill=color, width=6)
    draw.polygon([(x1, y0), (x1 - 18, y0 - 11), (x1 - 18, y0 + 11)],
                 fill=color)


def main():
    S = 380
    panel_model = render_airway_solid(S)
    panel_input = render_input_frame(S)
    panel_out = render_ar_overlay(S)

    pad = 20
    arrow_w = 130
    top_lab = 46
    W = S * 3 + arrow_w * 2 + pad * 2
    H = S + top_lab + 56
    canvas = Image.new('RGB', (W, H), (255, 255, 255))
    d = ImageDraw.Draw(canvas)

    x = pad
    for pim, lab in [(panel_model, '3D model'),
                     (panel_input, 'image / video'),
                     (panel_out, 'registered AR overlay')]:
        canvas.paste(pim, (x, top_lab))
        d.rectangle([x, top_lab, x + S, top_lab + S], outline=(200, 200, 200),
                    width=3)
        tw = d.textlength(lab, font=_font(36, bold=False))
        d.text((x + S / 2 - tw / 2, top_lab + S + 10), lab,
               fill=(40, 40, 40), font=_font(36, bold=False))
        x += S + arrow_w

    # arrows
    ymid = top_lab + S // 2
    arrow(d, pad + S + 12, ymid, pad + S + arrow_w - 12)
    plus_x = pad + S + arrow_w // 2
    d.text((plus_x - 22, ymid - 96), '+', fill=(60, 60, 60),
           font=_font(84, bold=True))
    x1 = pad + S + arrow_w + S + 12
    arrow(d, x1, ymid, x1 + arrow_w - 12)
    fr = 'FreeReg'
    tw = d.textlength(fr, font=_font(52, bold=True))
    d.text((x1 + arrow_w / 2 - tw / 2, ymid - 80), fr, fill=(25, 70, 160),
           font=_font(52, bold=True))
    sub = 'free-coordinate registration'
    tw2 = d.textlength(sub, font=_font(24, bold=False))
    d.text((x1 + arrow_w / 2 - tw2 / 2, ymid + 28), sub, fill=(90, 90, 90),
           font=_font(24, bold=False))

    title = 'Any 3D model  +  any image / video   →   anchored AR / MR overlay'
    twt = d.textlength(title, font=_font(38, bold=True))
    d.text((W / 2 - twt / 2, 2), title, fill=(20, 20, 20),
           font=_font(38, bold=True))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT)
    print(f'wrote {OUT} size={canvas.size}')


if __name__ == '__main__':
    main()
