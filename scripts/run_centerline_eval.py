# -*- coding: utf-8 -*-
"""
E1.5 中心线骨架对应注册评测
============================

Part A (合成, 有GT): 飞行帧渲染 → 暗区骨架 → 中心线对应注册 (扰动初始化)
    → 位姿误差/chamfer 收敛, vs 深度ICP基线
Part B (真实视频): 省医001 帧 → 管腔骨架 → 中心线对应注册
    → chamfer 收敛 + 叠加可视化 (无GT, 结构对齐为主要证据)

用法:
    conda activate py3d
    python scripts/run_centerline_eval.py
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
from src.pipeline import render_model, refine_icp
from src.pose_estimation import nearest_neighbors
from src.metrics import pose_errors
from src.centerline import (airway_centerline_3d, lumen_skeleton_2d,
                            register_centerline, skeleton_junctions_2d,
                            crop_centerline_by_view)
from src.ct_loader import load_ct_airway, make_flythrough_poses, CTVolume
import torch

P1 = r'F:\省医-数据\省医-数据\ION患者整理\001王继深'
VIDEO = (r'F:\省医-数据\省医-数据\ION患者整理\001王继深\术中视频'
         r'\Video_screenshot_Henan1\ion_video_2021-10-26_03-33-50_L.mp4')
CACHE = Path('outputs/ct/airway_cache.npz')
MASK_CACHE = Path('outputs/ct/airway_mask.npz')
OUT = Path('outputs/ct/centerline')
H, W = 240, 320
K = np.array([[200., 0, W / 2], [0, 200., H / 2], [0, 0, 1.0]])


class M: pass


def load_model_and_mask():
    """加载气道模型 + 体数据掩码 (中心线需要掩码+spacing)"""
    if MASK_CACHE.exists():
        z = np.load(MASK_CACHE)
        return z['points'], z['mask'], z['spacing'], z['origin'], \
            z['axis_pts'], z['axis_dir']
    model = load_ct_airway(P1, max_slices=200, max_pts=50000)
    vol = model.volume
    MASK_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(MASK_CACHE, points=model.points, mask=model.mask,
             spacing=vol.spacing, origin=vol.origin,
             axis_pts=model.axis_pts, axis_dir=model.axis_dir)
    return model.points, model.mask, vol.spacing, vol.origin, \
        model.axis_pts, model.axis_dir


def part_a(pts, cl_pts, axis_pts, axis_dir, device):
    print('\n--- Part A: 合成飞行帧 (GT 验证) ---')
    model = M()
    model.points, model.axis_pts, model.axis_dir = pts, axis_pts, axis_dir
    poses = make_flythrough_poses(model, n_frames=20, seed=0)
    rng = np.random.default_rng(2)
    colors = np.tile(np.array([0.85, 0.65, 0.60]), (len(pts), 1))
    rows = []
    for i in [2, 8, 14]:
        T_gt = poses[i]
        out = render_model(pts, colors, T_gt, K, H, W, radius=1,
                           device=device)
        rgb = (out['feat'].cpu().numpy() * 255).astype(np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        depth_gt = out['depth'].cpu().numpy()
        skel, dt_img, dark = lumen_skeleton_2d(bgr, dark_pct=35,
                                               depth=depth_gt)
        if skel.sum() < 30:
            print(f'  f{i}: 骨架不足, 跳过')
            continue
        # 扰动初始化: 8mm/6° (临床粗配准量级)
        T_init = T_gt @ exp_se3(rng.normal(0, [8, 8, 8, 0.1, 0.1, 0.1]))
        rot0, tr0 = pose_errors(T_init, T_gt)
        # 方法1: 中心线对应
        t0 = time.time()
        res = register_centerline(cl_pts, dt_img, K, T_init, iters=200,
                                  device=device)
        t_cl = time.time() - t0
        rot_c, tr_c = pose_errors(res['T'], T_gt)
        # 方法2: 深度 ICP (基线)
        depth = out['depth'].cpu().numpy()
        scene, _, _ = __import__('src.geometry',
                                 fromlist=['backproject_depth']
                                 ).backproject_depth(depth, K,
                                                     mask=(depth > 0))
        T_icp, st = refine_icp(pts[::1], scene, T_init, max_iter=40,
                               trim_ratio=0.7, device=device)
        rot_i, tr_i = pose_errors(T_icp, T_gt)
        rows.append(dict(f=i, rot0=rot0, tr0=tr0,
                         cham0=res['init_loss'], cham=res['final_loss'],
                         rot_c=rot_c, tr_c=tr_c, rot_i=rot_i, tr_i=tr_i,
                         ms=t_cl * 1000))
        print(f'  f{i}: init({rot0:.1f}°/{tr0:.1f}mm) → '
              f'中心线 {rot_c:.1f}°/{tr_c:.1f}mm '
              f'(chamfer {res["init_loss"]:.1f}→{res["final_loss"]:.2f}px, '
              f'{t_cl * 1000:.0f}ms) | ICP {rot_i:.1f}°/{tr_i:.1f}mm')
    if rows:
        print('  均值: 中心线注册 |rot|=%.1f° |t|=%.1fmm; ICP |rot|=%.1f° '
              '|t|=%.1fmm' % (
                  np.mean([r['rot_c'] for r in rows]),
                  np.mean([r['tr_c'] for r in rows]),
                  np.mean([r['rot_i'] for r in rows]),
                  np.mean([r['tr_i'] for r in rows])))
    return rows


def _align_rot(vec, dst=(0, 0, 1.0)):
    """把单位向量 vec 对齐到 dst 的旋转矩阵"""
    vec = vec / np.linalg.norm(vec)
    dst = np.asarray(dst, dtype=float)
    v = np.cross(vec, dst)
    s = np.linalg.norm(v)
    c = float(vec @ dst)
    if s < 1e-9:
        return np.eye(3) if c > 0 else \
            np.diag([1.0, -1.0, -1.0])
    K_ = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K_ + K_ @ K_ * (1 - c) / (s * s)


def part_b(pts, cl_pts, junc_idx, axis_dir, device):
    print('\n--- Part B: 真实临床视频帧 (视角裁剪+分叉加权) ---')
    OUT.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(VIDEO)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(int(total * 0.15), int(total * 0.6), 4).astype(int)
    panels = []
    rows = []
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, bgr = cap.read()
        if not ok:
            continue
        H2, W2 = bgr.shape[:2]
        K2 = np.array([[W2 / 2., 0, W2 / 2], [0, W2 / 2., H2 / 2],
                       [0, 0, 1.0]])
        skel, dt_img, dark = lumen_skeleton_2d(bgr, dark_pct=25)
        if skel.sum() < 30:
            print(f'  f{fi}: 骨架不足({int(skel.sum())}px), 跳过')
            continue
        # 初始化: 气道主轴对齐视轴(支气管镜沿管腔推进), 中心置于镜前60mm
        R0 = _align_rot(axis_dir)
        T_init = make_T(R0, np.array([0, 0, 60.0]) - R0 @ cl_pts.mean(0))
        # 视角裁剪: 仅保留投影进入视野的中心线 (消除长树自重叠)
        cl_crop, crop_idx = crop_centerline_by_view(cl_pts, T_init, K2,
                                                    H2, W2, margin_px=60,
                                                    z_range=(3.0, 150.0))
        if len(cl_crop) < 50:
            cl_crop, crop_idx = cl_pts, np.arange(len(cl_pts))
        jset = set(crop_idx.tolist())
        j_local = np.array([j for j in junc_idx if j in jset],
                           dtype=np.int64)
        # 映射到裁剪后数组索引
        pos = {g: i for i, g in enumerate(crop_idx.tolist())}
        j_local = np.array([pos[j] for j in j_local], dtype=np.int64)
        t0 = time.time()
        res = register_centerline(cl_crop, dt_img, K2, T_init, iters=250,
                                  lr=8e-3, device=device,
                                  cl_junction_idx=j_local,
                                  lambda_branch=3.0)
        t_cl = time.time() - t0
        # 分叉对齐指标: 3D分叉投影 → 最近2D分叉距离
        j2d, e2d = skeleton_junctions_2d(skel)
        P = transform(res['T'], cl_crop[j_local]) if len(j_local) > 0 \
            else np.zeros((0, 3))
        jerr = float('nan')
        if len(P) > 0 and len(j2d) > 0:
            z = P[:, 2]
            okm = z > 1e-3
            u = np.round(K2[0, 0] * P[okm, 0] / z[okm] + K2[0, 2]).astype(int)
            v = np.round(K2[1, 1] * P[okm, 1] / z[okm] + K2[1, 2]).astype(int)
            inb = (u >= 0) & (u < W2) & (v >= 0) & (v < H2)
            if inb.sum() > 0:
                pu = np.stack([v[inb], u[inb]], 1).astype(float)
                dmin = []
                for p in pu:
                    dmin.append(np.sqrt(
                        ((j2d - p) ** 2).sum(1)).min())
                jerr = float(np.mean(dmin))
        rows.append(dict(f=fi, cham0=res['init_loss'],
                         cham=res['final_loss'], jerr=jerr,
                         ms=t_cl * 1000))
        print(f'  f{fi}: chamfer {res["init_loss"]:.1f} → '
              f'{res["final_loss"]:.2f}px | 分叉对齐={jerr:.1f}px '
              f'(裁剪{len(cl_crop)}点, 分叉{len(j_local)}) '
              f'({t_cl * 1000:.0f}ms)')

        # 叠加可视化: 投影中心线 (红) + 分叉 (黄) + 骨架 (绿)
        vis = bgr.copy()
        P = transform(res['T'], cl_crop)
        z = P[:, 2]
        okm = z > 1e-3
        u = np.round(K2[0, 0] * P[okm, 0] / z[okm] + K2[0, 2]).astype(int)
        v = np.round(K2[1, 1] * P[okm, 1] / z[okm] + K2[1, 2]).astype(int)
        inb = (u >= 0) & (u < W2) & (v >= 0) & (v < H2)
        vis[skel] = (0, 255, 0)
        for uu, vv in zip(u[inb], v[inb]):
            cv2.circle(vis, (uu, vv), 1, (0, 0, 255), -1)
        if len(P) > 0 and len(j_local) > 0:
            Pj = transform(res['T'], cl_crop[j_local])
            zj = Pj[:, 2]
            okj = zj > 1e-3
            uj = np.round(K2[0, 0] * Pj[okj, 0] / zj[okj]
                          + K2[0, 2]).astype(int)
            vj = np.round(K2[1, 1] * Pj[okj, 1] / zj[okj]
                          + K2[1, 2]).astype(int)
            inj = (uj >= 0) & (uj < W2) & (vj >= 0) & (vj < H2)
            for uu, vv in zip(uj[inj], vj[inj]):
                cv2.circle(vis, (uu, vv), 3, (0, 255, 255), 1)
        panels.append(vis)
    cap.release()
    if panels:
        comp = np.concatenate(panels[:4], axis=1)
        cv2.imwrite(str(OUT / 'centerline_overlay.png'), comp)
        print(f'  叠加图: {OUT / "centerline_overlay.png"} '
              f'(绿=图像骨架, 红=投影CT中心线)')
    if rows:
        print(f'  chamfer 均值: {np.mean([r["cham"] for r in rows]):.2f}px, '
              f'分叉对齐均值: {np.nanmean([r["jerr"] for r in rows]):.1f}px, '
              f'{np.mean([r["ms"] for r in rows]):.0f}ms/帧')
    return rows


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('=' * 66)
    print('E1.5: 中心线骨架对应注册评测')
    print('=' * 66)
    pts, mask, spacing, origin, axis_pts, axis_dir = load_model_and_mask()
    vol = CTVolume(hu=mask.astype(np.int16), spacing=spacing,
                   origin=origin, orientation=np.eye(3))
    cl_pts, junc_idx, end_idx = airway_centerline_3d(
        mask, volume=vol, return_junctions=True)
    print(f'气道: {len(pts)}点; 中心线: {len(cl_pts)}点 '
          f'(分叉{len(junc_idx)}, 端点{len(end_idx)})')
    part_a(pts, cl_pts, axis_pts, axis_dir, device)
    part_b(pts, cl_pts, junc_idx, axis_dir, device)
    print('=' * 66)


if __name__ == '__main__':
    main()
