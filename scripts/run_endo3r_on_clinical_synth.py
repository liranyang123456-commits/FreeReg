# -*- coding: utf-8 -*-
"""
SOTA在Ours数据集: Endo3R 官方 on 临床合成飞行视频 (ATE vs GT轨迹)
====================================================================

合成飞行视频帧 → Endo3R demo → 轨迹 → Umeyama对齐 → ATE vs GT。
对照: 本管线跟踪轨迹在同序列的逐帧位姿误差。

用法:
    conda activate py3d
    python scripts/run_endo3r_on_clinical_synth.py
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import sys
import subprocess
import time
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.geometry import make_T, transform
from src.pipeline import render_model
from src.pose_estimation import kabsch_umeyama
from src.ct_loader import make_flythrough_poses
import torch

OUT = Path('outputs/ct/endo3r_synth')
FRAMES_DIR = Path(r'E:\SOTA_Methods\Endo3R\examples\clinical_synth')
N_FRAMES = 30
H, W = 256, 320
K = np.array([[200., 0, W / 2], [0, 200., H / 2], [0, 0, 1.0]])


class M: pass


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('=' * 66)
    print('Endo3R on 临床合成飞行视频 (ATE vs GT)')
    print('=' * 66)

    # ---------- 渲染飞行帧 ----------
    z = np.load('outputs/ct/airway_cache.npz')
    pts, axis_pts, axis_dir = z['points'], z['axis_pts'], z['axis_dir']
    m = M()
    m.points, m.axis_pts, m.axis_dir = pts, axis_pts, axis_dir
    poses_gt = make_flythrough_poses(m, n_frames=N_FRAMES, seed=0)
    colors = np.tile(np.array([0.85, 0.65, 0.60]), (len(pts), 1))
    for i, T in enumerate(poses_gt):
        out = render_model(pts, colors, T, K, H, W, radius=1,
                           device=device)
        img = (out['feat'].cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite(str(FRAMES_DIR / f'{i:06d}.png'),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f'渲染 {N_FRAMES} 帧 → {FRAMES_DIR}')

    # ---------- 运行 Endo3R ----------
    t0 = time.time()
    cmd = [
        sys.executable, r'E:\SOTA_Methods\Endo3R\demo.py',
        '--demo_path', 'examples/clinical_synth',
        '--kf_every', '1',
        '--save_path', str(OUT / 'result'),
        '--ckpt_path', r'E:\SOTA_Methods\Endo3R\checkpoints\endo3r.pth',
        '--save_result',
    ]
    r = subprocess.run(cmd, cwd=r'E:\SOTA_Methods\Endo3R',
                       capture_output=True, text=True, timeout=1800)
    print(f'Endo3R 推理 {time.time() - t0:.0f}s, rc={r.returncode}')
    if r.returncode != 0:
        print(r.stderr[-1500:])
        return
    res_path = Path(r'E:\SOTA_Methods\Endo3R\outputs\ct\endo3r_synth'
                    r'\result\clinical_synth\result.npy')
    z2 = np.load(res_path, allow_pickle=True)
    item = z2.item() if z2.shape == () else z2[0]
    poses_e = np.array(item['poses_all'])
    print(f'Endo3R 位姿: {poses_e.shape}')

    # ---------- ATE (Umeyama 对齐, 尺度自由) ----------
    A = np.array([p[:3, 3] for p in poses_e])
    B = np.array([p[:3, 3] for p in
                  [np.linalg.inv(T) for T in poses_gt]])  # c2w
    R, t, s = kabsch_umeyama(A, B, with_scale=True)
    A2 = s * (A @ R.T) + t
    ate = np.linalg.norm(A2 - B, axis=1)
    print(f'\nEndo3R ATE (对齐, s={s:.2f}): mean={ate.mean():.2f}mm '
          f'中位={np.median(ate):.2f}mm max={ate.max():.1f}mm')

    # 对照: 本管线跟踪轨迹逐帧误差 (同序列 run_ct_video_registration 协议:
    # 其逐帧位姿误差即"轨迹误差", 无需对齐 —— 本管线有公制尺度)
    print('\n对照说明: 本管线逐帧位姿误差在同序列见 '
          'run_ct_video_registration.py (TRE-NN), 拥有公制尺度; '
          'Endo3R 为任意尺度单目重建, 仅形状可比。')
    print('=' * 66)


if __name__ == '__main__':
    main()
