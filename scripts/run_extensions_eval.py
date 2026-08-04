# -*- coding: utf-8 -*-
"""
扩展数据集紧凑评测: EndoNeRF 轨迹时序 + BlendedMVS 3DGS
==========================================================

E-EndoNeRF: LLFF 位姿轨迹 + 噪声 → SE(3) Kalman → 抖动偏差
E-BlendedMVS: 真实多视角场景 3DGS 训练 → PSNR (真实MVS数据)

用法:
    conda activate py3d
    python scripts/run_extensions_eval.py
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import sys
import time
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.geometry import make_T, exp_se3, transform
from src.se3_kalman_filter import SE3KalmanFilter, SE3KalmanConfig
from src.metrics import pixel_jitter_deviation
import torch

OUT = Path('outputs/extensions')


def e_endonerf(device):
    print('\n--- E-EndoNeRF: 真实形变轨迹时序评测 ---')
    pb = np.load(r'E:\MIS_Datasets\EndoNeRF\cutting_tissues_twice'
                 r'\poses_bounds.npy')
    # LLFF: 每行 [3x3 R | t(3,1) | hwf(3,1)] 展平的 3x5, 为 c2w
    n = pb.shape[0]
    poses = []
    for i in range(n):
        m = pb[i, :-2].reshape(3, 5)
        R = m[:, :3]
        t = m[:, 3]
        # LLFF c2w → 取其逆作 w2c? 轨迹本身一致即可, 用 c2w 的 t 轨迹
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        poses.append(T)
    # 抖动评测: 对轨迹加噪声, Kalman 平滑, 测轨迹中心抖动偏差
    rng = np.random.default_rng(0)
    scale = np.array([0.02, 0.02, 0.02, 0.001, 0.001, 0.001])
    noisy = [T @ exp_se3(rng.normal(0, 1, 6) * scale) for T in poses]
    kf = SE3KalmanFilter(SE3KalmanConfig(
        process_noise_pos=0.02, process_noise_rot=2e-5,
        process_noise_vel_pos=0.01, process_noise_vel_rot=1e-4,
        meas_noise_pos=0.02, meas_noise_rot=0.001,
        innovation_threshold=10.0, velocity_damping=0.98))
    smooth = [kf.update(T, dt=1.0) for T in noisy]
    # 轨迹中心抖动偏差 (用轨迹点序列)
    cen = np.array([T[:3, 3] for T in poses])
    cen_n = np.array([T[:3, 3] for T in noisy])
    cen_s = np.array([T[:3, 3] for T in smooth])
    j_n = float(np.sqrt(((np.diff(cen_n, axis=0)
                          - np.diff(cen, axis=0)) ** 2).sum(1).mean()))
    j_s = float(np.sqrt(((np.diff(cen_s, axis=0)
                          - np.diff(cen, axis=0)) ** 2).sum(1).mean()))
    print(f'  轨迹 {n}帧 | 中心抖动偏差: 噪声={j_n:.4f} → Kalman={j_s:.4f} '
          f'(↓{100 * (1 - j_s / j_n):.0f}%) | 重定位={kf.reloc_count} '
          f'拒绝={kf.reject_count}')
    return dict(frames=n, j_noisy=j_n, j_kalman=j_s, reloc=kf.reloc_count)


def e_blendedmvs(device):
    print('\n--- E-BlendedMVS: 真实多视角 3DGS ---')
    from src.gaussian_renderer import (GaussianModel, render_gaussians,
                                       optimize_gaussians, psnr, ssim_simple)
    from src.pipeline import lambert_colors
    root = Path(r'F:\SpecularReflectionRemoving\data\raw'
                r'\BlendedMVS_extracted\BlendedMVS')
    scenes = sorted([p for p in root.iterdir() if p.is_dir()])[:1]
    if not scenes:
        print('  BlendedMVS 未找到场景')
        return {}
    s = scenes[0]
    cam_dir = s / 'cams'
    img_dir = s / 'blended_images'
    cams = sorted(cam_dir.glob('*_cam.txt'))[:8]
    views = []
    for c in cams:
        lines = c.read_text().strip().split('\n')
        ex = np.array([[float(x) for x in lines[i].split()]
                       for i in range(1, 5)])
        intr = np.array([[float(x) for x in lines[i].split()]
                         for i in range(7, 10)])
        views.append((ex, intr))
    print(f'  场景 {s.name}: {len(views)} 视角')
    # 用位姿+内参渲染? 无3D点 — 3DGS 需点云初始化.
    # 用 cams 的位姿从图像特征生成稀疏点 (简化: 用深度? BlendedMVS 有
    # rendered_depth_maps. 此处用 VGGT 式: 以多视角三角化? 超范围.
    # 紧凑处理: 用 rendered_depth_maps 反投影一点云, 或跳过点云用
    # 以场景中心的均匀球初始化.
    imgs = []
    for i, c in enumerate(cams):
        im = cv2.imread(str(img_dir / f'{i:08d}.jpg'))
        if im is None:
            im = cv2.imread(str(img_dir / f'{i:08d}.png'))
        imgs.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    H, W = imgs[0].shape[:2]
    # 用 rendered_depth_maps 反投影初始化点云 (随机球初始化在真实纹理
    # 场景上必然失败 — 几何先验必需)
    from src.geometry import backproject_depth
    pts = None
    dm_dir = s / 'rendered_depth_maps'
    if dm_dir.exists():
        dpf = sorted(dm_dir.glob('*'))[0]
        dmap = cv2.imread(str(dpf), cv2.IMREAD_UNCHANGED)
        if dmap is not None:
            dmap = dmap.astype(np.float32)
            if dmap.max() > 20:
                dmap = dmap / 1000.0
            ex0, intr0 = views[0]
            sp, _, _ = backproject_depth(dmap, intr0,
                                         mask=(dmap > 0), stride=4)
            if len(sp) > 100:
                # 深度图点云在 view0 相机系 → 转到世界系再用于多视角训练
                from src.geometry import transform as _tf
                spw = _tf(ex0, sp)          # cam0 → world (ex 为 c2w)
                pts = spw[::max(1, len(spw) // 4000)]
                print(f'  点云初始化: {len(pts)}点 (深度图→世界系)')
    if pts is None:
        rng = np.random.default_rng(0)
        pts = rng.normal(0, 0.5, (3000, 3))
        print('  (深度图不可用, 回退随机球初始化)')
    model = GaussianModel.from_points(pts, device=device)
    train_views = [{'T_wc': np.linalg.inv(ex), 'image': im / 255.0}
                   for (ex, intr), im in zip(views[:-1], imgs[:-1])]
    Kv = views[0][1]
    t0 = time.time()
    optimize_gaussians(model, train_views, Kv, iters=120, device=device,
                       verbose=False)
    ex_t, intr_t = views[-1]
    out = render_gaussians(model, np.linalg.inv(ex_t), intr_t, H, W,
                           device=device)['feat'].cpu().numpy()
    p = psnr(out, imgs[-1] / 255.0)
    s_ = ssim_simple(out, imgs[-1] / 255.0)
    print(f'  3DGS 训练 {time.time() - t0:.0f}s | 测试视角 PSNR={p:.2f}dB '
          f'SSIM={s_:.4f}')
    save = (np.concatenate([imgs[-1] / 255.0, out], axis=1) * 255
            ).astype(np.uint8)
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / 'blendedmvs_3dgs.png'),
                cv2.cvtColor(save, cv2.COLOR_RGB2BGR)
                if save.max() > 1 else save)
    print(f'  图: {OUT / "blendedmvs_3dgs.png"}')
    return dict(psnr=p, ssim=s_)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('=' * 66)
    print('扩展数据集紧凑评测 (EndoNeRF + BlendedMVS)')
    print('=' * 66)
    r1 = e_endonerf(device)
    r2 = e_blendedmvs(device)
    print('\n' + '=' * 66)
    print('扩展汇总:')
    if r1:
        print(f'  EndoNeRF: {r1["frames"]}帧, 抖动 {r1["j_noisy"]:.4f}→'
              f'{r1["j_kalman"]:.4f}px')
    if r2:
        print(f'  BlendedMVS: PSNR={r2["psnr"]:.2f}dB SSIM={r2["ssim"]:.4f}')
    print('=' * 66)


if __name__ == '__main__':
    main()
