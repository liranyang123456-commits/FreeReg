# -*- coding: utf-8 -*-
"""
生成「气道模型 CT 截面叠加图」供论文 Fig (TMI)
============================================

输入:
  outputs/ct/airway_mask.npz   -> mask (Z,Y,X) bool, spacing, origin
  outputs/ct/airway_cache.npz  -> points (N,3) mm, axis_pts, axis_dir

输出 (outputs/paper/figs/):
  fig_ct_overlay.png  -> 轴位 / 冠状位 / 矢状位 三视图截面,
                         CT 气道区(灰度底) + 配准气道模型轮廓(绿色) + 中心线(红)

说明:
  airway_mask.npz 只含气道二值掩码而无 HU 灰度体数据, 故底图用气道掩码的
  形态学距离场渲染为灰度(管腔亮、壁渐暗), 叠加配准模型截面轮廓与主轴中心线。
  若后续拿到原始 HU 体数据, 可将 base 替换为真实 CT 切片显示。

用法:
    conda activate py3d
    python scripts/make_ct_overlay_fig.py
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
from pathlib import Path
import cv2
import numpy as np
from scipy import ndimage
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
MASK_NPZ = ROOT / 'outputs/ct/airway_mask.npz'
CACHE_NPZ = ROOT / 'outputs/ct/airway_cache.npz'
OUT = ROOT / 'outputs/paper/figs/fig_ct_overlay.png'


def load():
    zm = np.load(MASK_NPZ)
    zc = np.load(CACHE_NPZ)
    mask = zm['mask'].astype(bool)            # (Z,Y,X)
    spacing = zm['spacing'].astype(float)     # (3,) z,y,x
    origin = zm['origin'].astype(float)       # (3,)
    pts = zc['points'].astype(float)          # (N,3) mm physical
    axis_pts = zm['axis_pts'].astype(float)   # (K,3) mm
    return mask, spacing, origin, pts, axis_pts


def xyz_to_ijk(xyz, origin, spacing):
    return (xyz - origin) / spacing           # -> (z,y,x) float


def slice_base(mask, idx, axis):
    """取某一正交截面的底图 (返回 m2 布尔面; 颜色在 overlay_slice 里上).
       无 HU 体数据, 用白底 + 气道管腔浅色填充, 保证轮廓清晰."""
    if axis == 0:    # 轴位: 固定 z -> (Y,X)
        m2 = mask[idx, :, :]
    elif axis == 1:  # 冠状位: 固定 y -> (Z,X)
        m2 = mask[:, idx, :]
    else:            # 矢状位: 固定 x -> (Z,Y)
        m2 = mask[:, :, idx]
    return m2


def overlay_slice(mask, pts_ijk, axis_ijk, idx, axis):
    """白底 + 气道管腔浅蓝填充 + 深蓝轮廓 + 橙模型点 + 红中心线."""
    m2 = slice_base(mask, idx, axis)
    H, W = m2.shape
    rgb = np.full((H, W, 3), 255, np.uint8)      # white background
    rgb[m2] = (200, 220, 240)                    # airway lumen: light blue fill
    # airway mask contour (dark blue outline)
    edge = m2 & ~ndimage.binary_erosion(m2)
    rgb[edge] = (30, 90, 200)

    # --- model points near this slice (surface points: allow a few-voxel band) ---
    tol = 3.0  # voxels
    if axis == 0:
        sel = np.abs(pts_ijk[:, 0] - idx) < tol
        uv = pts_ijk[sel][:, [2, 1]]  # x,y -> col,row
    elif axis == 1:
        sel = np.abs(pts_ijk[:, 1] - idx) < tol
        uv = pts_ijk[sel][:, [2, 0]]
    else:
        sel = np.abs(pts_ijk[:, 2] - idx) < tol
        uv = pts_ijk[sel][:, [1, 0]]
    for c, r in uv:
        ci, ri = int(round(c)), int(round(r))
        if 0 <= ri < H and 0 <= ci < W:
            rgb[max(0, ri-2):ri+3, max(0, ci-2):ci+3] = (255, 140, 0)  # model pts: orange

    # --- axis (centerline) points near slice ---
    if axis == 0:
        sa = np.abs(axis_ijk[:, 0] - idx) < tol
        auv = axis_ijk[sa][:, [2, 1]]
    elif axis == 1:
        sa = np.abs(axis_ijk[:, 1] - idx) < tol
        auv = axis_ijk[sa][:, [2, 0]]
    else:
        sa = np.abs(axis_ijk[:, 2] - idx) < tol
        auv = axis_ijk[sa][:, [1, 0]]
    for c, r in auv:
        ci, ri = int(round(c)), int(round(r))
        if 0 <= ri < H and 0 <= ci < W:
            rgb[max(0, ri-1):ri+2, max(0, ci-1):ci+2] = (255, 60, 60)
    return rgb


def crop_to_airway(rgb, m2, pad=18):
    """裁剪到气道 mask 的外接框(含边距), 让细小气道在画面中占满."""
    ys, xs = np.where(m2)
    if len(ys) == 0:
        return rgb
    y0, y1 = max(0, ys.min() - pad), min(m2.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(m2.shape[1], xs.max() + pad)
    return rgb[y0:y1, x0:x1]


def main():
    mask, spacing, origin, pts, axis_pts = load()
    pts_ijk = xyz_to_ijk(pts, origin, spacing)
    axis_ijk = xyz_to_ijk(axis_pts, origin, spacing)
    Z, Y, X = mask.shape
    # choose slices through the MASK centroid (robust to surface-vs-solid offset)
    mijk = np.argwhere(mask)
    cz, cy, cx = [int(round(v)) for v in mijk.mean(0)]
    views = [
        ('Axial',    0, min(max(cz, 0), Z - 1)),
        ('Coronal',  1, min(max(cy, 0), Y - 1)),
        ('Sagittal', 2, min(max(cx, 0), X - 1)),
    ]
    PANEL = 512
    panels = []
    for name, axis, idx in views:
        rgb = overlay_slice(mask, pts_ijk, axis_ijk, idx, axis)
        # crop to the airway ROI so the (thin) airway fills the panel
        if axis == 0:
            m2 = mask[idx, :, :]
        elif axis == 1:
            m2 = mask[:, idx, :]
        else:
            m2 = mask[:, :, idx]
        # crop rgb and m2 with the SAME box
        ys, xs = np.where(m2)
        cy0, cy1 = max(0, ys.min() - 14), min(m2.shape[0], ys.max() + 14)
        cx0, cx1 = max(0, xs.min() - 14), min(m2.shape[1], xs.max() + 14)
        rgb = rgb[cy0:cy1, cx0:cx1]
        m2 = m2[cy0:cy1, cx0:cx1]
        # square-pad (keep full resolution for the zoom inset)
        h, w = rgb.shape[:2]
        s = max(h, w)
        sq = np.full((s, s, 3), 255, np.uint8)
        y0 = (s - h) // 2; x0 = (s - w) // 2
        sq[y0:y0 + h, x0:x0 + w] = rgb

        pim = Image.fromarray(sq).resize((PANEL, PANEL), Image.LANCZOS)
        # ROI center: deepest point of the airway distance transform in this
        # slice = widest part of a branch; neighborhood shows centerline +
        # model points against the lumen wall at native resolution.
        m2sq = np.zeros((s, s), bool)
        m2sq[y0:y0 + h, x0:x0 + w] = m2
        dist = ndimage.distance_transform_edt(m2sq)
        # avoid border-hugging maxima
        dist[:s // 8, :] = 0; dist[-s // 8:, :] = 0
        dist[:, :s // 8] = 0; dist[:, -s // 8:] = 0
        ry, rx = np.unravel_index(np.argmax(dist), dist.shape)
        half = max(40, s // 7)
        ry = min(max(ry, half), s - half)
        rx = min(max(rx, half), s - half)
        roi = sq[ry - half:ry + half, rx - half:rx + half]
        zsize = 170
        zoom = cv2.resize(roi, (zsize, zsize), interpolation=cv2.INTER_LANCZOS4)
        scale = PANEL / s
        bx0 = int((rx - half) * scale); bx1 = int((rx + half) * scale)
        by0 = int((ry - half) * scale); by1 = int((ry + half) * scale)
        d = ImageDraw.Draw(pim)
        d.rectangle([bx0, by0, bx1, by1], outline=(50, 50, 50), width=2)
        ix0, iy0 = PANEL - zsize - 8, PANEL - zsize - 8
        pim.paste(Image.fromarray(zoom), (ix0, iy0))
        d.rectangle([ix0, iy0, ix0 + zsize, iy0 + zsize],
                    outline=(50, 50, 50), width=2)
        panels.append((name, pim))
    # annotate + concat 1x3
    tiles = []
    for name, pim in panels:
        d = ImageDraw.Draw(pim)
        d.rectangle([0, 0, 118, 30], fill=(30, 90, 200))
        d.text((7, 5), name, fill=(255, 255, 255))
        tiles.append(pim)
    W = sum(t.size[0] for t in tiles)
    H = tiles[0].size[1]
    canvas = Image.new('RGB', (W, H), (255, 255, 255))
    x = 0
    for t in tiles:
        canvas.paste(t, (x, 0)); x += t.size[0]
    canvas.save(OUT)
    print('saved', OUT, canvas.size)


if __name__ == '__main__':
    main()
