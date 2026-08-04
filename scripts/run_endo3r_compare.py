# -*- coding: utf-8 -*-
"""
SOTA 对比: Endo3R (DUSt3R家族, MICCAI'25) vs SCARED GT 轨迹
============================================================

Endo3R result.npy 解析 → 相机轨迹 → 与 SCARED frame_data GT 对比:
  轨迹对齐 (Umeyama 相似变换, 处理尺度歧义) → ATE/RPE + 深度对比

用法:
    conda activate py3d
    python scripts/run_endo3r_compare.py
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import sys
import json
import tarfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.geometry import inverse_se3
from src.pose_estimation import kabsch_umeyama
from src.metrics import pixel_jitter_deviation

RES = Path(r'E:\Free_coordinate_MR_Registration\outputs\endo3r'
           r'\scared_d1k1\result.npy')
GT_TAR = Path(r'E:\MIS_Datasets\SCARED\dataset_1\keyframe_1\data'
              r'\frame_data.tar.gz')
EVERY = 6   # 抽帧间隔 (与 _prep_endo3r_scared.py 一致)


def main():
    print('=' * 66)
    print('Endo3R vs SCARED GT 轨迹对比')
    print('=' * 66)
    z = np.load(RES, allow_pickle=True)
    print('result.npy 类型:', type(z), z.shape if hasattr(z, 'shape') else '')
    item = z.item() if z.shape == () else z[0]
    print('keys:', list(item.keys()) if isinstance(item, dict) else
          type(item))
    if isinstance(item, dict):
        for k, v in item.items():
            if isinstance(v, np.ndarray):
                print(f'  {k}: {v.shape} {v.dtype}')
            else:
                print(f'  {k}: {type(v).__name__}')

    # 提取位姿
    poses = None
    for cand in ['poses_all', 'poses', 'camera_poses', 'pose', 'c2w',
                 'traj']:
        if cand in item:
            poses = np.array(item[cand])
            print(f'\n位姿键: {cand} shape={poses.shape}')
            break
    if poses is None:
        first = item[list(item.keys())[0]]
        print('未知结构, 首元素:', type(first))
        return

    # 内参对比 (Endo3R 无标定估计 vs SCARED 标定, 按图像尺寸归一)
    if 'intrinsic' in item:
        K_est = item['intrinsic']
        fx_gt, fy_gt = 1035.3, 1035.1   # SCARED 标定 (1280×1024)
        # Endo3R 图像 224×224 ← 抽帧 320×256 ← 原图 1280×1024 的左半
        # 宽向缩放: 224/1280, 高向: 224/1024
        sx, sy = 224 / 1280, 224 / 1024
        fx_n, fy_n = K_est[0, 0] / sx, K_est[1, 1] / sy
        print(f'内参估计 (还原到原图尺度): fx={fx_n:.0f} fy={fy_n:.0f} '
              f'vs GT fx={fx_gt:.0f} fy={fy_gt:.0f} '
              f'(误差 {abs(fx_n - fx_gt) / fx_gt:.1%}/{abs(fy_n - fy_gt) / fy_gt:.1%})')

    n_endo = len(poses)
    # GT 轨迹 (c2w, mm), 取对应抽帧
    gt_all = []
    with tarfile.open(GT_TAR, 'r:gz') as tar:
        for name in sorted(tar.getnames()):
            d = json.loads(tar.extractfile(name).read().decode('utf-8'))
            T = np.array(d['camera-pose'], dtype=np.float64)
            gt_all.append(T)
    gt_sel = [gt_all[i] for i in range(0, len(gt_all), EVERY)][:n_endo]
    gt_sel = [inverse_se3(T) for T in gt_sel]      # w2c (与Endo3R惯例对齐检查)

    # Endo3R 位姿方向判别: 试 c2w 与 w2c 两种, 取对齐残差小者
    def align_ate(src_poses, dst_poses):
        A = np.array([p[:3, 3] for p in src_poses])
        B = np.array([p[:3, 3] for p in dst_poses])
        R, t, s = kabsch_umeyama(A, B, with_scale=True)
        A2 = s * (A @ R.T) + t
        ate = np.linalg.norm(A2 - B, axis=1)
        return ate, s
    gt_c2w = [inverse_se3(T) for T in gt_sel]
    ate1, s1 = align_ate(poses, gt_sel)
    ate2, s2 = align_ate(poses, gt_c2w)
    if ate1.mean() <= ate2.mean():
        ate, s, conv = ate1, s1, 'w2c'
    else:
        ate, s, conv = ate2, s2, 'c2w'
    print(f'\nEndo3R 位姿惯例判定: {conv} (对齐尺度 s={s:.2f})')
    print(f'ATE (轨迹对齐后): mean={ate.mean():.2f}mm '
          f'median={np.median(ate):.2f}mm max={ate.max():.1f}mm')

    # 相对运动 (RPE): 相邻帧位移幅值误差
    def rel_disp(P):
        return np.array([np.linalg.norm(
            P[i + 1][:3, 3] - P[i][:3, 3]) for i in range(len(P) - 1)])
    r_endo = rel_disp(poses)
    r_gt = rel_disp(gt_sel if conv == 'w2c' else gt_c2w)
    rpe = np.abs(r_endo * s - r_gt) / np.maximum(r_gt, 1e-6)
    print(f'RPE (相对位移相对误差): mean={np.nanmean(rpe):.1%} '
          f'median={np.nanmedian(rpe):.1%}')
    print('=' * 66)


if __name__ == '__main__':
    main()
