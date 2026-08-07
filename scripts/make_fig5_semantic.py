"""Build a semantically explicit Fig. 5 for co-anchored rendering.

Rows:
  Top: train views for pseudo-GT | sparse splat | 3DGS.
  Bottom: held-out views for pseudo-GT | sparse splat | 3DGS.

This is larger and easier to interpret than a 24-panel orbit montage because it
directly shows both interpolation quality and the generalization claim.
"""
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.geometry import make_T, exp_so3
from src.pipeline import load_mesh_points, render_model, lambert_colors
from src.gaussian_renderer import GaussianModel, optimize_gaussians, render_gaussians

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "paper" / "figs" / "fig8_render.png"
MODEL = Path(r"D:\bop\lm\models\obj_000008.ply")


def font(size):
    for p in ("C:/Windows/Fonts/arial.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def view_T(yaw):
    R = exp_so3(np.array([0.18, 0.0, 0.0])) @ exp_so3(
        np.array([0.0, np.deg2rad(yaw), 0.0]))
    return make_T(R, np.array([0.0, 0.0, 390.0]))


def crop_tight(rgb, tw, th, label, color):
    g = rgb.mean(axis=2)
    ys, xs = np.where(g > 10)
    pad = 4
    y0, y1 = max(0, ys.min() - pad), min(rgb.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(rgb.shape[1], xs.max() + pad)
    crop = rgb[y0:y1, x0:x1]
    scale = min(tw / crop.shape[1], (th - 34) / crop.shape[0])
    nw, nh = int(crop.shape[1] * scale), int(crop.shape[0] * scale)
    panel = Image.fromarray(crop).resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (tw, th), (18, 18, 18))
    canvas.paste(panel, ((tw - nw) // 2, 34 + (th - 34 - nh) // 2))
    d = ImageDraw.Draw(canvas)
    d.text((10, 5), label, fill=(0, 0, 0), font=font(21))
    d.text((9, 4), label, fill=color, font=font(21))
    return canvas


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pts, normals = load_mesh_points(str(MODEL), n_points=16000)
    pts -= pts.mean(axis=0)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(pts), 5000, replace=False)
    pts_s, n_s = pts[idx], normals[idx]
    H, W = 300, 400
    K = np.array([[560.0, 0, 200.0], [0, 560.0, 150.0], [0, 0, 1.0]])
    train = [0, 90, 180, 270]
    heldout = [45, 135]

    dense_gt, sparse, gaussian = {}, {}, {}
    for yaw in train + heldout:
        T = view_T(yaw)
        c = lambert_colors(normals, T[:3, :3], base_rgb=(0.92, 0.82, 0.55))
        dense_gt[yaw] = render_model(pts, c, T, K, H, W, radius=2,
                                     device=device)["feat"].cpu().numpy()
        cs = lambert_colors(n_s, T[:3, :3], base_rgb=(0.92, 0.82, 0.55))
        sparse[yaw] = render_model(pts_s, cs, T, K, H, W, radius=1,
                                   device=device)["feat"].cpu().numpy()

    colors = lambert_colors(n_s, np.eye(3), base_rgb=(0.92, 0.82, 0.55))
    model = GaussianModel.from_points(pts_s, colors=colors, device=device)
    views = [{"T_wc": view_T(y), "image": dense_gt[y]} for y in train]
    optimize_gaussians(model, views, K, iters=220,
                       lr={"mu": 1e-3, "log_scale": 2e-2, "rotvec": 5e-3,
                           "color": 1e-2, "logit": 1e-2}, device=device)
    for yaw in train + heldout:
        gaussian[yaw] = render_gaussians(model, view_T(yaw), K, H, W,
                                         device=device)["feat"].cpu().numpy()

    tw, th, gap = 430, 330, 8
    canvas = Image.new("RGB", (3 * tw + 2 * gap, 2 * th + gap), "white")
    blue, green = (105, 190, 255), (95, 255, 145)
    labels = ["Pseudo-GT", "Sparse splat", "Co-anchored 3DGS"]
    sources = [dense_gt, sparse, gaussian]
    for c, (name, src) in enumerate(zip(labels, sources)):
        panel = crop_tight((np.clip(src[0], 0, 1) * 255).astype(np.uint8),
                           tw, th, f"{name}: train 0°", blue)
        canvas.paste(panel, (c * (tw + gap), 0))
        panel = crop_tight((np.clip(src[45], 0, 1) * 255).astype(np.uint8),
                           tw, th, f"{name}: novel 45°", green)
        canvas.paste(panel, (c * (tw + gap), th + gap))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT)
    print(f"Wrote {OUT}: {canvas.size}")


if __name__ == "__main__":
    main()
