# temp: 复算优化器初始化时的 n_inl (scene_0003 t2)
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.geometry import transform, backproject_depth, inverse_se3
from src.pipeline import render_model
from src.pose_estimation import multistart_icp_register

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
print('tris.shape =', tris.shape)


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
print('subdiv surf 点数 =', len(surf))
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

# 复算优化器 it=0 的采样 (T=T_est, delta=0)
Vd = torch.as_tensor(base, dtype=torch.float32, device=device)
R = torch.as_tensor(T_est[:3, :3], dtype=torch.float32, device=device)
tv = torch.as_tensor(T_est[:3, 3], dtype=torch.float32, device=device)
Kt = torch.as_tensor(K, dtype=torch.float32, device=device)
D = torch.as_tensor(depth, dtype=torch.float32, device=device)
V_cam = Vd @ R.T + tv
z = V_cam[:, 2].clamp(min=1e-6)
u = Kt[0, 0] * V_cam[:, 0] / z + Kt[0, 2]
v = Kt[1, 1] * V_cam[:, 1] / z + Kt[1, 2]
gx = 2.0 * u / (127) - 1.0
gy = 2.0 * v / (127) - 1.0
grid = torch.stack([gx, gy], -1).reshape(1, 1, -1, 2)
z_obs = torch.nn.functional.grid_sample(
    D.reshape(1, 1, 128, 128), grid, mode='bilinear',
    padding_mode='zeros', align_corners=True).reshape(-1)
res = z_obs - z
thr = 0.10 * extent
valid = (z_obs > 0) & (z > 1e-3)
n_inl = int(((res.abs() < thr) & valid).sum().item())
print(f'n_inl(it=0, 含噪声) = {n_inl} / {len(base)}')
print(f'res分布: mean={res.abs().mean():.4f} median={res.abs().median():.4f} '
      f'p25={res.abs().quantile(0.25):.4f} p75={res.abs().quantile(0.75):.4f}')
print(f'z_obs>0 比例={float((z_obs > 0).float().mean()):.3f}, '
      f'thr={thr:.4f}')
