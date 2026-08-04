# -*- coding: utf-8 -*-
# temp: split hybrid trajectory jitter by flip frames (rot>30deg excluded)
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import sys
import json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.geometry import make_T, transform
from src.pipeline import load_mesh_points
from src.metrics import pose_errors, pixel_jitter_deviation, add_s_error

z = np.load('outputs/tracking_compare/ours_poses.npz', allow_pickle=True)
gts = json.load(open(r'D:\bop\lm\test\000001\scene_gt.json'))
pts, _ = load_mesh_points(r'D:\bop\lm\models\obj_000001.ply', n_points=20000)
eval_pts = pts[::10]
probe = pts[np.linspace(0, len(pts) - 1, 9).astype(int)]
K = np.array([[572.4, 0, 325.3], [0, 573.6, 242.0], [0, 0, 1.0]])

fs = sorted(int(k) for k in z.files)
good = []
for fi in fs:
    g = gts[str(fi)][0]
    T_gt = make_T(np.array(g['cam_R_m2c']).reshape(3, 3),
                  np.array(g['cam_t_m2c']).reshape(3))
    r, _ = pose_errors(z[str(fi)], T_gt)
    if r < 30:
        good.append(fi)
print('total', len(fs), 'flip(rot>30)', len(fs) - len(good),
      '({:.0f}%)'.format(100 * (len(fs) - len(good)) / len(fs)))


def jitter_of(sel):
    def uv(poses):
        out = []
        for T in poses:
            P = transform(T, probe)
            zz = P[:, 2]
            out.append(np.stack([K[0, 0] * P[:, 0] / zz + K[0, 2],
                                 K[1, 1] * P[:, 1] / zz + K[1, 2]], -1))
        return out
    gtl = []
    for fi in sel:
        g = gts[str(fi)][0]
        gtl.append(make_T(np.array(g['cam_R_m2c']).reshape(3, 3),
                          np.array(g['cam_t_m2c']).reshape(3)))
    ug, ue = uv(gtl), uv([z[str(f)] for f in sel])
    return float(np.mean([
        pixel_jitter_deviation(
            np.array([ue[t][j] for t in range(len(sel))]),
            np.array([ug[t][j] for t in range(len(sel))]))
        for j in range(len(probe))]))


def adds_of(sel):
    arr = []
    for fi in sel:
        g = gts[str(fi)][0]
        T_gt = make_T(np.array(g['cam_R_m2c']).reshape(3, 3),
                      np.array(g['cam_t_m2c']).reshape(3))
        arr.append(add_s_error(eval_pts, z[str(fi)], T_gt))
    return np.array(arr)


j_all, j_good = jitter_of(fs), jitter_of(good)
a_all, a_good = adds_of(fs), adds_of(good)
print('all: jitter={:.2f}px ADD-S={:.2f}mm'.format(j_all, a_all.mean()))
print('no-flip: jitter={:.2f}px ADD-S={:.2f}mm (n={})'.format(
    j_good, a_good.mean(), len(good)))
