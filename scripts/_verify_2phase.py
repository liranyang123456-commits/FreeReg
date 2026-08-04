# temp: 两阶段+边缘带 对凹痕恢复的效果验证
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.geometry import make_T, exp_so3, transform
from src.pipeline import (load_mesh_points, render_model, register_rgbd,
                          RegConfig)
from src.nonrigid_rtw import rtw_optimize_projective
from src.pose_estimation import nearest_neighbors
from src.metrics import pose_errors
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
warp_mag = np.linalg.norm(dent, axis=1)
white = np.ones((len(model_def), 3))
out = render_model(model_def, white, T_gt, K, H, W, radius=1, device=device)
depth_gt = out['depth'].cpu().numpy()
depth_obs = depth_gt + rng.normal(0, 0.8, depth_gt.shape)
depth_obs[depth_gt <= 0] = 0

cfg = RegConfig(device=device)
res_rigid = register_rgbd(depth_obs, K, model, mask=(depth_obs > 0), cfg=cfg)
T_rigid = res_rigid['T']
rot_r, _ = pose_errors(T_rigid, T_gt)
Pg = transform(T_gt, model_def)
_, db = nearest_neighbors(transform(T_rigid, model), Pg, device=device)
print(f'刚性: rot={rot_r:.2f}° NN-to-GT={db.mean():.3f}mm')
roi = warp_mag > 2.0
roi_d = np.linalg.norm((model - c0), axis=1) < 15.0

cases = [
    ('基线(无两阶段/无边缘带)', dict()),
    ('两阶段', dict(two_phase=True)),
    ('边缘带', dict(edge_band_px=4.0)),
    ('两阶段+边缘带', dict(two_phase=True, edge_band_px=4.0)),
    ('两阶段+边缘带+聚焦', dict(two_phase=True, edge_band_px=4.0,
                               focal_gamma=2.0)),
]
for name, kw in cases:
    rtw = rtw_optimize_projective(
        model, depth_obs, K, T_rigid, grid=(8, 8, 8), iters=400, lr=1e-2,
        lambda_mag=1e-5, lambda_sm=1e-4, lambda_p2p=1.0,
        inlier_hi=30.0, inlier_lo=12.0, device=device, seed=0, **kw)
    rot_w, _ = pose_errors(rtw['T'], T_gt)
    Pc = transform(rtw['T'], rtw['V_def'])
    _, dr = nearest_neighbors(Pc, Pg, device=device)
    vtx = np.linalg.norm(rtw['V_def'] - model_def, axis=1)
    Pd = transform(rtw['T'], rtw['V_def'])[roi_d]
    _, dd = nearest_neighbors(Pd, Pg[roi_d], device=device)
    dmax = np.linalg.norm(rtw['delta'], axis=1).max()
    print(f'{name}: rot={rot_w:.2f}° NN={dr.mean():.2f}mm '
          f'凹痕区NN={dd.mean():.2f}mm 恢复={vtx[roi].mean():.2f}'
          f'(未{warp_mag[roi].mean():.2f}) |δ|max={dmax:.1f}')
