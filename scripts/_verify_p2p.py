# temp: 鐐瑰闈慨澶嶉獙璇?(scene_0003 t2, 姝ゅ墠 rec=268%)
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.geometry import transform, backproject_depth, inverse_se3
from src.pipeline import render_model
from src.nonrigid_rtw import rtw_optimize_projective
from src.pose_estimation import nearest_neighbors, multistart_icp_register
from src.metrics import pose_errors

device = 'cuda' if torch.cuda.is_available() else 'cpu'
s = Path(r'E:\Diff_Rending_Re_3D\dataset\real_colon\scene_0003')
gt = torch.load(s / 'gt.pt', map_location='cpu', weights_only=False)
nodes = gt['scene_spec']['nodes'].numpy().astype(np.float64)
tris = gt['scene_spec']['surface_tris'].numpy().astype(np.int64)
poses = gt['poses'].numpy()
u_seq = gt['u_seq'].numpy().astype(np.float64).reshape(-1, 3)
u_seq = u_seq.reshape(u_seq.shape[0] // nodes.shape[0], nodes.shape[0], 3)
extent = float(np.linalg.norm(nodes.max(0) - nodes.min(0)))
H = W = 128
K = np.array([[160., 0, 64], [0, 160., 64], [0, 0, 1.0]])


def subdiv(nodes, tris, n_per=6):
    pts, meta = [], []
    for (a, b, c) in tris:
        for i in range(n_per + 1):
            for j in range(n_per + 1 - i):
                u, v = i / n_per, j / n_per
                w = 1 - u - v
                pts.append(u * nodes[a] + v * nodes[b] + w * nodes[c])
                meta.append([u, v, w, a, b, c])
    return np.array(pts), meta


def deform(nodes, meta, u):
    out = np.zeros((len(meta), 3))
    for i, (u_, v, w, a, b, c) in enumerate(meta):
        p = u_ * nodes[a] + v * nodes[b] + w * nodes[c]
        out[i] = p + u_ * u[a] + v * u[b] + w * u[c]
    return out


surf, meta = subdiv(nodes, tris)
base = deform(nodes, meta, np.zeros_like(nodes))
t = 2
u_gt = u_seq[t]
T_cam = inverse_se3(poses[t])
sdef = deform(nodes, meta, u_gt)
colors = np.tile(np.array([0.85, 0.65, 0.6]), (len(sdef), 1))
out = render_model(sdef, colors, T_cam, K, H, W, radius=1, device=device)
depth = out['depth'].cpu().numpy()
rng = np.random.default_rng(0)
fg = depth > 0
depth[fg] += rng.normal(0, extent * 0.01, int(fg.sum()))
scene_pts, _, _ = backproject_depth(depth, K, mask=(depth > 0))
T_est, st = multistart_icp_register(base, scene_pts, device=device,
                                    coarse_iter=25, fine_iter=50,
                                    trim_coarse=0.5, trim_fine=0.7)
rot_r, _ = pose_errors(T_est, T_cam)
print(f'鍒氭€?rot={rot_r:.2f}掳')

for lam in [0.0, 1.0, 3.0]:
    rtw = rtw_optimize_projective(base, depth, K, T_est, grid=(4, 4, 4),
                                  iters=300, lr=3e-3, lambda_mag=0.5,
                                  lambda_sm=0.2, lambda_p2p=lam,
                                  inlier_hi=0.10 * extent,
                                  inlier_lo=0.03 * extent,
                                  device=device, seed=0, verbose=True)
    rot_w, _ = pose_errors(rtw['T'], T_cam)
    d = np.linalg.norm(rtw['V_def'] - base, axis=1)
    Pc = transform(rtw['T'], rtw['V_def'])
    Pg = transform(T_cam, sdef)
    _, dr = nearest_neighbors(Pc, Pg, device=device)
    print(f'位_p2p={lam}: rot={rot_w:.2f}掳 |未|mean={d.mean():.4f} '
          f'rec={dr.mean() / extent * 100:.1f}% '
          f'loss={rtw["losses"][-1]:.4f}')
