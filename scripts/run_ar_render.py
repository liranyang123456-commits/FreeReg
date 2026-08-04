# -*- coding: utf-8 -*-
"""
AR/MR 融合渲染输出
==================

完整链路可视化: BOP 真实 RGB-D 帧 → 注册 → Lambert 着色渲染
→ 深度感知合成 (真实场景遮挡虚拟模型 / 虚拟模型遮挡背景)
→ 输出三联图 (原图 | 虚拟渲染 | AR 融合) + 模型坐标轴叠加。

用法:
    conda activate py3d
    python scripts/run_ar_render.py
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import sys
import json
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import (RegConfig, load_mesh_points, register_rgbd,
                          render_model, composite_ar, lambert_colors)
from src.geometry import project_pose
from src.metrics import add_s_error

BOP_ROOT = Path(r'D:\bop\lm')
OUT_DIR = Path('outputs/ar')

# (scene_id, frame_idx) —— 覆盖不同物体/视角/遮挡
FRAMES = [(1, 15), (1, 635), (2, 243), (5, 215)]


def draw_axes(img_bgr, K, T, diameter):
    """在图像上绘制模型坐标轴 (X红 Y绿 Z蓝), 长度=半径"""
    L = diameter / 2.0
    pts3d = np.array([[0, 0, 0], [L, 0, 0], [0, L, 0], [0, 0, L]],
                     dtype=np.float64)
    uv, z = project_pose(K, T, pts3d)
    if (z <= 0).any():
        return img_bgr
    o = tuple(np.round(uv[0]).astype(int))
    cols = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    for i in range(1, 4):
        p = tuple(np.round(uv[i]).astype(int))
        cv2.line(img_bgr, o, p, cols[i - 1], 2, cv2.LINE_AA)
    return img_bgr


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = RegConfig(device='cuda')
    models_info = json.load(open(BOP_ROOT / 'models' / 'models_info.json'))
    cache = {}

    def get_model(obj_id):
        if obj_id not in cache:
            pts, normals = load_mesh_points(
                str(BOP_ROOT / 'models' / f'obj_{obj_id:06d}.ply'),
                n_points=20000)
            cache[obj_id] = (pts, normals)
        return cache[obj_id]

    for scene_id, fi in FRAMES:
        sdir = BOP_ROOT / 'test' / f'{scene_id:06d}'
        cams = json.load(open(sdir / 'scene_camera.json'))
        gts = json.load(open(sdir / 'scene_gt.json'))
        k = str(fi)
        if k not in cams:
            print(f'[skip] scene {scene_id} frame {fi} 不存在')
            continue
        K = np.array(cams[k]['cam_K']).reshape(3, 3)
        rgb = cv2.cvtColor(cv2.imread(str(sdir / 'rgb' / f'{fi:06d}.png')),
                           cv2.COLOR_BGR2RGB)
        depth = cv2.imread(str(sdir / 'depth' / f'{fi:06d}.png'),
                           cv2.IMREAD_UNCHANGED).astype(np.float64)
        depth *= cams[k].get('depth_scale', 1.0)
        H, W = depth.shape

        g = gts[k][0]
        obj_id = g['obj_id']
        pts, normals = get_model(obj_id)
        diameter = models_info[str(obj_id)]['diameter']
        mv = sdir / 'mask_visib' / f'{fi:06d}_000000.png'
        mask = cv2.imread(str(mv), cv2.IMREAD_GRAYSCALE) if mv.exists() \
            else None

        # 注册
        res = register_rgbd(depth, K, pts, mask=mask, cfg=cfg)
        T_est = res['T']
        T_gt = np.eye(4)
        T_gt[:3, :3] = np.array(g['cam_R_m2c']).reshape(3, 3)
        T_gt[:3, 3] = np.array(g['cam_t_m2c']).reshape(3)
        eval_pts = pts[::max(1, len(pts) // 2000)]
        adds = add_s_error(eval_pts, T_est, T_gt)

        # 渲染 (Lambert 着色)
        colors = lambert_colors(normals, T_est[:3, :3])
        render = render_model(pts, colors, T_est, K, H, W, radius=1)
        rgb_f = rgb.astype(np.float64) / 255.0
        comp, occ = composite_ar(rgb_f, depth, render, alpha=0.8)
        n_virt = int((render['depth'] > 0).sum())
        n_vis = int(occ.sum())

        # 三联图
        render_rgb = (render['feat'].cpu().numpy() * 255).astype(np.uint8)
        comp_u8 = (np.clip(comp, 0, 1) * 255).astype(np.uint8)
        panels = [rgb, render_rgb, comp_u8]
        tri = np.concatenate(
            [cv2.cvtColor(p, cv2.COLOR_RGB2BGR) for p in panels], axis=1)
        # 在融合面板上画坐标轴
        x0 = 2 * W
        tri_axes = tri.copy()
        sub = tri_axes[:, x0:x0 + W]
        sub = draw_axes(sub, K, T_est, diameter)
        tri_axes[:, x0:x0 + W] = sub
        # 标注
        labels = ['Input RGB', 'Virtual render', 'AR composite (depth-aware)']
        for i, lab in enumerate(labels):
            cv2.putText(tri_axes, lab, (i * W + 8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        out_path = OUT_DIR / f'ar_s{scene_id:02d}_f{fi:04d}.png'
        cv2.imwrite(str(out_path), tri_axes)
        print(f'[AR] scene{scene_id} f{fi}: obj{obj_id:02d} '
              f'ADD-S={adds:.2f}mm 虚拟像素={n_virt} 可见={n_vis} '
              f'({100 * n_vis / max(n_virt, 1):.0f}% 未被遮挡) → {out_path}')

    print('\n全部输出完成。')


if __name__ == '__main__':
    main()
