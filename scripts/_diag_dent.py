# temp: 凹痕可视化诊断 (深度图对比+差分+凹痕区裁剪)
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import sys
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.geometry import make_T, exp_so3, transform
from src.pipeline import (load_mesh_points, render_model, register_rgbd,
                          RegConfig)
from src.nonrigid_rtw import rtw_optimize_projective
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
BOP_MODEL = Path(r'D:\bop\lm\models\obj_000008.ply')
rng = np.random.default_rng(7)
model, normals = load_mesh_points(str(BOP_MODEL), n_points=8000)
model = model - model.mean(axis=0)
T_gt = make_T(exp_so3(np.array([0.12, -0.08, 0.18])),
              np.array([25.0, -15.0, 900.0]))
K = np.array([[572.4, 0, 325.3], [0, 573.6, 242.0], [0, 0, 1.0]])
H, W = 480, 640
n_cam = normals @ T_gt[:3, :3].T
c_idx = int(np.argmax(-n_cam[:, 2] + rng.normal(0, 0.1, len(n_cam))))
c0 = model[c_idx]
d2 = ((model - c0) ** 2).sum(axis=1)
z_axis_model = T_gt[:3, :3].T @ np.array([0, 0, 1.0])
dent = (8.0 * np.exp(-d2 / (2 * 15.0 ** 2)))[:, None] * z_axis_model[None, :]
model_def = model + dent
white = np.ones((len(model_def), 3))
out = render_model(model_def, white, T_gt, K, H, W, radius=1, device=device)
depth_gt = out['depth'].cpu().numpy()
depth_obs = depth_gt + rng.normal(0, 0.8, depth_gt.shape)
depth_obs[depth_gt <= 0] = 0

cfg = RegConfig(device=device)
res_rigid = register_rgbd(depth_obs, K, model, mask=(depth_obs > 0), cfg=cfg)
T_rigid = res_rigid['T']

# 渲染: GT形变 / 刚性模型 / RTW形变 的深度
def render_d(m, T):
    return render_model(m, white, T, K, H, W, radius=1,
                        device=device)['depth'].cpu().numpy()

d_gt = render_d(model_def, T_gt)
d_rig = render_d(model, T_rigid)
rtw = rtw_optimize_projective(model, depth_obs, K, T_rigid, grid=(8, 8, 8),
                              iters=500, lr=1e-2, lambda_mag=1e-5,
                              lambda_sm=1e-4, lambda_p2p=1.0,
                              focal_gamma=2.0, inlier_hi=30.0,
                              inlier_lo=12.0, device=device, seed=0)
d_rtw = render_d(rtw['V_def'], rtw['T'])

# c0 投影像素 (GT)
Pc0 = transform(T_gt, c0.reshape(1, 3))[0]
u0 = int(K[0, 0] * Pc0[0] / Pc0[2] + K[0, 2])
v0 = int(K[1, 1] * Pc0[1] / Pc0[2] + K[1, 2])
print(f'c0 像素=({u0},{v0}) | GT深度={d_gt[v0, u0]:.1f} '
      f'刚性深度={d_rig[v0, u0]:.1f} RTW深度={d_rtw[v0, u0]:.1f} '
      f'(差: rig{depth_obs[v0, u0] - d_rig[v0, u0]:+.1f} '
      f'rtw{depth_obs[v0, u0] - d_rtw[v0, u0]:+.1f})')


def norm_img(d):
    m = d > 0
    out = np.zeros((*d.shape, 3), np.uint8)
    if m.any():
        dn = (d - d[m].min()) / max(d[m].max() - d[m].min(), 1e-9)
        g = (dn * 255).astype(np.uint8)
        out[m] = np.stack([g, g, g], axis=-1)[m]
    return out


def diff_img(a, b, cap=10.0):
    m = (a > 0) & (b > 0)
    out = np.zeros((*a.shape, 3), np.uint8)
    df = np.clip((b - a) / cap, -1, 1)
    out[m, 0] = (np.clip(-df[m], 0, 1) * 255).astype(np.uint8)
    out[m, 2] = (np.clip(df[m], 0, 1) * 255).astype(np.uint8)
    return out


panels = [
    np.concatenate([norm_img(depth_obs), norm_img(d_gt),
                    norm_img(d_rig), norm_img(d_rtw)], axis=1),
    np.concatenate([diff_img(d_rig, depth_obs), diff_img(d_rtw, depth_obs),
                    diff_img(d_rig, d_gt), diff_img(d_rtw, d_gt)], axis=1),
]
comp = np.concatenate(panels, axis=0)
for (x, y), lab in [((5, 20), 'obs GT-def rig rtw'),
                    ((5, H + 20), 'diff rig-vs-obs rtw-vs-obs rig-vs-gt rtw-vs-gt')]:
    cv2.putText(comp, lab, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1)
cv2.circle(comp, (u0, v0), 8, (0, 255, 255), 1)
cv2.circle(comp, (u0, v0 + H), 8, (0, 255, 255), 1)
Path('outputs/rtw_diag').mkdir(parents=True, exist_ok=True)
cv2.imwrite('outputs/rtw_diag/dent_diag.png', comp)
print('保存 outputs/rtw_diag/dent_diag.png (上排: obs|GT|rigid|rtw; '
      '下排: 差分图 蓝=观测更深)')
