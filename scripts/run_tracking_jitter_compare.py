# -*- coding: utf-8 -*-
"""
FP 跟踪模式 vs 本管线(hybrid/Kalman) 同视频抖动对比 (docs/12 §5.4)
====================================================================

同一序列 (BOP scene1 全 200 帧):
  T_fp   = FoundationPose 官方逐帧结果 (linemod_res.yml)
  T_ours = 本管线 hybrid 逐帧注册
  T_kf   = T_ours + SE(3) Kalman 平滑
指标: 多点像素抖动偏差 (vs GT), ADD-S, 旋转误差

用法:
    conda activate py3d
    python scripts/run_tracking_jitter_compare.py
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import sys
import json
import time
from pathlib import Path

import numpy as np
import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.geometry import make_T, transform, backproject_depth
from src.pipeline import RegConfig, load_mesh_points, register_rgbd, \
    arbitrate_poses
from src.appearance_engine import SIFTReferenceLibrary, register_sift
from src.pose_estimation import icp as _icp
from src.metrics import add_s_error, pose_errors, pixel_jitter_deviation
from src.geometry import project_pose
from src.se3_kalman_filter import SE3KalmanFilter, SE3KalmanConfig
import torch

BOP = Path(r'D:\bop\lm')
FP_RES = Path(r'E:\SOTA_Methods\FoundationPose\debug\linemod_res.yml')
OUT = Path('outputs/tracking_compare')
SCENE, OBJ = 1, 1
K_KF = np.array([[572.4, 0, 325.3], [0, 573.6, 242.0], [0, 0, 1.0]])


def kalman_cfg():
    # hybrid 引擎噪声为厚尾(偶发对称翻转) → 门控拒绝滑行即可,
    # 绝不可"重置到(翻转)测量": max_consecutive_reject 调大,
    # 让滤波器滑过翻转帧, 好测量回来时自然接续 (docs/12 §5.4 实证)
    return SE3KalmanConfig(
        process_noise_pos=8.0, process_noise_rot=8e-6,
        process_noise_vel_pos=0.5, process_noise_vel_rot=2e-4,
        meas_noise_pos=8.0, meas_noise_rot=0.02,
        innovation_threshold=10.0, max_consecutive_reject=500,
        velocity_damping=0.98)


def load_fp_poses():
    d = yaml.safe_load(open(FP_RES, encoding='utf-8'))
    obj_dict = d.get(OBJ) or d.get(str(OBJ)) or d
    out = {}
    for fi_key, v in obj_dict.items():
        try:
            vv = v[str(OBJ)] if isinstance(v, dict) and str(OBJ) in v \
                else (v[OBJ] if isinstance(v, dict) and OBJ in v else v)
            arr = np.array(vv, dtype=float)
            if arr.shape == (4, 4):
                T = np.eye(4)
                T[:3, :3] = arr[:3, :3]
                T[:3, 3] = arr[:3, 3] * 1000.0   # FP 米制 → 毫米
                out[int(fi_key)] = T
        except Exception:
            continue
    return out


def jitter_of(poses_list, gt_list, probe_pts, K):
    def uv_seq(poses):
        seq = []
        for T in poses:
            P = transform(T, probe_pts)
            z = P[:, 2]
            seq.append(np.stack([K[0, 0] * P[:, 0] / z + K[0, 2],
                                 K[1, 1] * P[:, 1] / z + K[1, 2]], -1))
        return seq
    ug, ue = uv_seq(gt_list), uv_seq(poses_list)
    devs = []
    for j in range(len(probe_pts)):
        a = np.array([ug[t][j] for t in range(len(ug))])
        b = np.array([ue[t][j] for t in range(len(ue))])
        devs.append(pixel_jitter_deviation(b, a))
    return float(np.mean(devs))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('=' * 66)
    print('FP跟踪 vs 本管线Kalman 同视频抖动对比 (scene1 × 200帧)')
    print('=' * 66)

    # ---------- 数据与 GT ----------
    sdir = BOP / 'test' / f'{SCENE:06d}'
    cams = json.load(open(sdir / 'scene_camera.json'))
    gts = json.load(open(sdir / 'scene_gt.json'))
    keys = sorted(cams.keys(), key=int)
    pts, _ = load_mesh_points(str(BOP / 'models' / f'obj_{OBJ:06d}.ply'),
                              n_points=20000)
    mu = pts.mean(0)
    probe = pts[np.linspace(0, len(pts) - 1, 9).astype(int)]
    eval_pts = pts[::10]
    fp_poses = load_fp_poses()
    print(f'FP 位姿: {len(fp_poses)} 帧')

    # ---------- SIFT 参考库 ----------
    lib = SIFTReferenceLibrary()
    obj_id = OBJ
    ref_keys = [k for k in keys if any(g['obj_id'] == obj_id for g in gts[k])]
    for ri in np.linspace(0, len(ref_keys) - 1, 40).astype(int):
        rk = ref_keys[ri]
        gi = next(i for i, g in enumerate(gts[rk]) if g['obj_id'] == obj_id)
        g = gts[rk][gi]
        rgb_r = cv2.cvtColor(cv2.imread(
            str(sdir / 'rgb' / f'{int(rk):06d}.png')), cv2.COLOR_BGR2RGB)
        depth_r = cv2.imread(str(sdir / 'depth' / f'{int(rk):06d}.png'),
                             cv2.IMREAD_UNCHANGED).astype(np.float64)
        K_r = np.array(cams[rk]['cam_K']).reshape(3, 3)
        mk_r = sdir / 'mask' / f'{int(rk):06d}_{gi:06d}.png'
        mask_r = cv2.imread(str(mk_r), cv2.IMREAD_GRAYSCALE) \
            if mk_r.exists() else None
        T_r = make_T(np.array(g['cam_R_m2c']).reshape(3, 3),
                     np.array(g['cam_t_m2c']).reshape(3))
        lib.add(rgb_r, depth_r, K_r, T_r, mask=mask_r, frame_id=int(rk))

    # ---------- 本管线逐帧 (带缓存) ----------
    POSE_CACHE = OUT / 'ours_poses.npz'
    cfg = RegConfig(device=device)
    kf = SE3KalmanFilter(kalman_cfg())
    rows = []
    ours_poses, kf_poses = {}, {}
    t0_all = time.time()
    cache = dict(np.load(POSE_CACHE, allow_pickle=True)) \
        if POSE_CACHE.exists() else None
    for k in keys:
        fi = int(k)
        K = np.array(cams[k]['cam_K']).reshape(3, 3)
        depth = cv2.imread(str(sdir / 'depth' / f'{fi:06d}.png'),
                           cv2.IMREAD_UNCHANGED).astype(np.float64)
        g = gts[k][0]
        T_gt = make_T(np.array(g['cam_R_m2c']).reshape(3, 3),
                      np.array(g['cam_t_m2c']).reshape(3))
        mv = sdir / 'mask_visib' / f'{fi:06d}_000000.png'
        mask = cv2.imread(str(mv), cv2.IMREAD_GRAYSCALE) \
            if mv.exists() else None
        if cache is not None and str(fi) in cache:
            T_est = cache[str(fi)]
        else:
            rgb_q = cv2.cvtColor(cv2.imread(
                str(sdir / 'rgb' / f'{fi:06d}.png')), cv2.COLOR_BGR2RGB)
            scene_pts, _, _ = backproject_depth(depth, K, mask=mask)
            sel = np.random.default_rng(0).choice(
                len(scene_pts), min(20000, len(scene_pts)), replace=False)
            model_icp = pts[::max(1, len(pts) // 3000)]
            cands = []
            T_s, _ = register_sift(lib, rgb_q, K, mask_q=mask)
            if T_s is not None:
                T_r2, _ = _icp(model_icp, scene_pts[sel], T_init=T_s,
                               max_iter=50, trim_ratio=0.7, device=device)
                cands.append(T_r2)
            res_g = register_rgbd(depth, K, pts, mask=mask, cfg=cfg)
            if res_g['T'] is not None:
                cands.append(res_g['T'])
            if not cands:
                continue
            T_est, _, _ = arbitrate_poses(cands, pts, depth, K, mask)
        T_sm = kf.update(T_est, dt=1.0)
        ours_poses[fi] = T_est
        kf_poses[fi] = T_sm
        rows.append(dict(fi=fi, adds_fp=np.nan, adds_ours=np.nan,
                         adds_kf=np.nan))
    print(f'本管线逐帧完成: {len(ours_poses)}帧, '
          f'{(time.time() - t0_all):.0f}s')
    if cache is None:
        np.savez(POSE_CACHE,
                 **{str(fi): T for fi, T in ours_poses.items()})

    # ---------- 指标 ----------
    def eval_traj(pose_dict, name):
        fs = sorted(pose_dict.keys())
        adds, rots = [], []
        for fi in fs:
            if str(fi) not in gts:
                continue
            g = gts[str(fi)][0]
            T_gt = make_T(np.array(g['cam_R_m2c']).reshape(3, 3),
                          np.array(g['cam_t_m2c']).reshape(3))
            adds.append(add_s_error(eval_pts, pose_dict[fi], T_gt))
            rots.append(pose_errors(pose_dict[fi], T_gt)[0])
        return np.array(adds), np.array(rots)

    adds_fp, rot_fp = eval_traj(fp_poses, 'fp')
    adds_ours, rot_ours = eval_traj(ours_poses, 'ours')
    adds_kf, rot_kf = eval_traj(kf_poses, 'kf')

    fs_common = sorted(set(fp_poses.keys()) & set(ours_poses.keys()))
    probe_c = probe
    gt_list = []
    for fi in fs_common:
        g = gts[str(fi)][0]
        gt_list.append(make_T(np.array(g['cam_R_m2c']).reshape(3, 3),
                              np.array(g['cam_t_m2c']).reshape(3)))
    j_fp = jitter_of([fp_poses[f] for f in fs_common], gt_list,
                     probe_c, K_KF)
    j_ours = jitter_of([ours_poses[f] for f in fs_common], gt_list,
                       probe_c, K_KF)
    j_kf = jitter_of([kf_poses[f] for f in fs_common], gt_list,
                     probe_c, K_KF)

    print('\n' + '=' * 66)
    print('结果 (scene1, 公共帧 %d):' % len(fs_common))
    print(f'{"轨迹":>22} | {"抖动偏差px":>10} | {"ADD-S mean":>10} | '
          f'{"ADD-S<10mm":>10} | {"rot mean":>8}')
    def line(name, j, a, r):
        print(f'{name:>22} | {j:10.3f} | {a.mean():10.2f} | '
          f'{float((a < 10).mean()):10.0%} | {r.mean():8.2f}')
    line('FP官方(tracking)', j_fp, adds_fp, rot_fp)
    line('本管线(hybrid)', j_ours, adds_ours, rot_ours)
    line('本管线(hybrid+Kalman)', j_kf, adds_kf, rot_kf)
    print(f'Kalman 重定位/拒绝: {kf.reloc_count}/{kf.reject_count}')

    import csv
    with open(OUT / 'jitter_compare.csv', 'w', newline='',
              encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['traj', 'jitter_px', 'adds_mean', 'adds_lt10',
                    'rot_mean'])
        w.writerow(['FP', f'{j_fp:.3f}', f'{adds_fp.mean():.3f}',
                    f'{(adds_fp < 10).mean():.3f}', f'{rot_fp.mean():.3f}'])
        w.writerow(['ours', f'{j_ours:.3f}', f'{adds_ours.mean():.3f}',
                    f'{(adds_ours < 10).mean():.3f}',
                    f'{rot_ours.mean():.3f}'])
        w.writerow(['ours+KF', f'{j_kf:.3f}', f'{adds_kf.mean():.3f}',
                    f'{(adds_kf < 10).mean():.3f}', f'{rot_kf.mean():.3f}'])
    print(f'CSV: {OUT / "jitter_compare.csv"}')
    print('=' * 66)


if __name__ == '__main__':
    main()
