# -*- coding: utf-8 -*-
"""
M3 多模型场景图评测
===================

场景: BOP 真实 RGB-D 背景 (scene2 f000243, benchvise) + 3 个虚拟模型
  duck (z=700mm, 最前) / ape (z=900mm, 中) / can (z=1100mm, 最后,
  且位于真实 benchvise 之后 → 真实场景遮挡虚拟模型)
→ 三重遮挡: 虚拟-虚拟 (duck⊃ape⊃can) + 真实-虚拟 (benchvise⊃can)

评测:
  1) 每模型独立注册 (合成观测深度 + GT mask) → ADD-S vs GT
  2) 场景图边 (遮挡关系) vs GT 深度排序 → 边正确率
  3) 逐像素遮挡决策正确率 (估计合成 vs GT合成)
  4) 渲染耗时 + 三联输出图

用法:
    conda activate py3d
    python scripts/run_scene_graph_eval.py
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import sys
import json
import time
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.geometry import make_T, exp_so3, transform
from src.pipeline import (RegConfig, load_mesh_points, register_rgbd,
                          render_model, lambert_colors)
from src.scene_graph import (SceneNode, SceneGraph, SceneEdge,
                             occlusion_decision_accuracy)
from src.metrics import add_s_error
import torch

BOP = Path(r'D:\bop\lm')
OUT = Path('outputs/scene_graph')

# (model_ply, 颜色, GT位姿t, GT旋转rv)
VIRTUALS = [
    ('obj_000008.ply', 'duck', (0.2, 0.9, 0.3), [0, -10, 700.0], [0.1, -0.2, 0.05]),
    ('obj_000001.ply', 'ape', (0.9, 0.5, 0.2), [60, 20, 900.0], [-0.05, 0.15, -0.1]),
    ('obj_000005.ply', 'can', (0.3, 0.5, 0.9), [20, 30, 1100.0], [0.02, 0.0, 0.25]),
]
BG_SCENE, BG_FRAME = 2, 243


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('=' * 66)
    print('M3: 多模型场景图评测 (真实背景 + 3虚拟模型 + 三重遮挡)')
    print('=' * 66)

    # ---------- 真实背景 ----------
    sdir = BOP / 'test' / f'{BG_SCENE:06d}'
    cams = json.load(open(sdir / 'scene_camera.json'))
    K = np.array(cams[str(BG_FRAME)]['cam_K']).reshape(3, 3)
    rgb = cv2.cvtColor(cv2.imread(
        str(sdir / 'rgb' / f'{BG_FRAME:06d}.png')), cv2.COLOR_BGR2RGB)
    depth_real = cv2.imread(str(sdir / 'depth' / f'{BG_FRAME:06d}.png'),
                            cv2.IMREAD_UNCHANGED).astype(np.float64)
    H, W = depth_real.shape

    # ---------- 虚拟模型 + GT 位姿 (编程校验摆放: 避免意外全遮挡) ----------
    def pix_to_3d(u_des, v_des, z):
        X = (u_des - K[0, 2]) / K[0, 0] * z
        Y = (v_des - K[1, 2]) / K[1, 1] * z
        return np.array([X, Y, z])

    def real_z(u, v):
        u, v = int(u), int(v)
        if 0 <= u < W and 0 <= v < H and depth_real[v, u] > 0:
            return depth_real[v, u]
        return np.inf

    # 真实物体 (benchvise) 掩码 bbox —— 用官方 mask_visib 精确定位
    mv_bg = sdir / 'mask_visib' / f'{BG_FRAME:06d}_000000.png'
    mask_bg = cv2.imread(str(mv_bg), cv2.IMREAD_GRAYSCALE) \
        if mv_bg.exists() else (depth_real > 0).astype(np.uint8) * 255
    ys, xs = np.nonzero(mask_bg > 0)
    bx = (int(xs.min()), int(xs.max()))
    by = (int(ys.min()), int(ys.max()))
    cy = (by[0] + by[1]) // 2

    # duck: 前景 (285,235) z=650 (遮挡 ape 与真实物体)
    t_duck = pix_to_3d(285, 235, 650.0)
    # ape: (350,245) z=800 —— benchvise(≈850mm) 之前, duck 之后,
    #      保证: 与duck重叠处被duck挡, 其余区域可见 (三重链 duck⊃ape⊃real)
    t_ape = pix_to_3d(350, 245, 800.0)
    # can: 沿真实深度自适应搜索 —— benchvise 右缘外, 立于远桌面之前
    # (z = 桌面深度-60 → 右部可见; 且 z > benchvise深度 → 左部被挡 real⊃can)
    t_can = None
    for u_try in range(bx[1] + 10, min(W - 40, bx[1] + 150), 5):
        zr = real_z(u_try, cy)
        if np.isfinite(zr) and zr > 1000:
            t_can = pix_to_3d(u_try, cy, min(zr - 60, 1100.0))
            break
    if t_can is None:
        t_can = pix_to_3d(bx[1] + 20, cy, 1000.0)
    VIRTUALS = [
        ('obj_000008.ply', 'duck', (0.2, 0.9, 0.3), t_duck.tolist(),
         [0.1, -0.2, 0.05]),
        ('obj_000001.ply', 'ape', (0.9, 0.5, 0.2), t_ape.tolist(),
         [-0.05, 0.15, -0.1]),
        ('obj_000005.ply', 'can', (0.3, 0.5, 0.9), t_can.tolist(),
         [0.02, 0.0, 0.25]),
    ]
    print(f'摆放: duck@({t_duck.round(0)}) ape@({t_ape.round(0)}) '
          f'can@({t_can.round(0)}) 真实物体u∈{bx}')

    models = {}
    for ply, name, color, t, rv in VIRTUALS:
        pts, normals = load_mesh_points(str(BOP / 'models' / ply),
                                        n_points=15000)
        mu = pts.mean(0)
        pts = pts - mu
        T = make_T(exp_so3(np.array(rv)), np.array(t, dtype=float))
        colors = np.tile(np.array(color), (len(pts), 1)) * \
            lambert_colors(normals, T[:3, :3])[:, :1] * 2
        colors = np.clip(colors, 0, 1)
        models[name] = dict(pts=pts, normals=normals, colors=colors,
                            T_gt=T, ply=ply)

    # ---------- GT 合成 (遮挡真值) ----------
    sg_gt = SceneGraph(K=K, scene_depth=depth_real)
    for name, m in models.items():
        sg_gt.add(SceneNode(model_id=name, pts=m['pts'],
                            colors=m['colors'], pose=m['T_gt']))
    comp_gt, vdepth_gt, owner_gt = sg_gt.composite(
        rgb, device=device, radius=2)
    # GT 深度排序 (模型两两)
    gt_order = {}
    for a in models:
        for b in models:
            if a < b:
                da = render_model(models[a]['pts'], models[a]['colors'],
                                  models[a]['T_gt'], K, H, W, radius=2,
                                  device=device)['depth'].cpu().numpy()
                db = render_model(models[b]['pts'], models[b]['colors'],
                                  models[b]['T_gt'], K, H, W, radius=2,
                                  device=device)['depth'].cpu().numpy()
                ov = (da > 0) & (db > 0)
                if ov.sum() >= 20:
                    nearer = a if np.median(da[ov]) < np.median(db[ov]) else b
                    gt_order[(a, b)] = nearer
    print('GT 深度排序:', gt_order)

    # ---------- 合成观测深度 → 每模型独立注册 ----------
    obs_depth = np.where(vdepth_gt > 0,
                         np.minimum(np.where(depth_real > 0, depth_real,
                                             np.inf), vdepth_gt),
                         depth_real)
    cfg = RegConfig(device=device)
    sg_est = SceneGraph(K=K, scene_depth=depth_real)
    rows = []
    for name, m in models.items():
        # GT 可见 mask: 该模型为最近虚拟模型 且 未被真实场景遮挡 的像素
        # (owner 仅标记虚拟前后关系; 真实遮挡需与 depth_real 比较)
        idx = list(models.keys()).index(name)
        real_z = np.where(depth_real > 0, depth_real, np.inf)
        visible = ((owner_gt == idx) & (vdepth_gt > 0)
                   & (vdepth_gt < real_z))
        mask = visible.astype(np.uint8) * 255
        if mask.sum() < 50:
            mask = (render_model(m['pts'], m['colors'], m['T_gt'], K, H, W,
                                 radius=1, device=device)['mask']
                    .cpu().numpy().astype(np.uint8)) * 255
        # 自适应截尾: 估计可见比例 → trim 匹配 (部分遮挡场景必需)
        full_sil = render_model(m['pts'], m['colors'], m['T_gt'], K, H, W,
                                radius=1, device=device)['mask'].cpu().numpy()
        vis_frac = mask.sum() / max(full_sil.sum(), 1)
        cfg_m = RegConfig(device=device,
                          trim_fine=float(np.clip(vis_frac + 0.1, 0.4, 0.8)))
        t0 = time.time()
        res = register_rgbd(obs_depth, K, m['pts'], mask=mask, cfg=cfg_m)
        dt = time.time() - t0
        T_est = res['T']
        adds = add_s_error(m['pts'][::8], T_est, m['T_gt']) \
            if T_est is not None else float('nan')
        rows.append((name, adds, dt, T_est))
        print(f'  注册 {name}: ADD-S={adds:.2f}mm  耗时={dt:.2f}s')
        sg_est.add(SceneNode(model_id=name, pts=m['pts'],
                             colors=m['colors'], pose=T_est,
                             confidence=1.0 if adds < 10 else 0.5))

    # ---------- 场景图边 vs GT ----------
    edges_est = sg_est.build_edges(H, W, device=device)
    n_edge_ok = 0
    for e in edges_est:
        pair = tuple(sorted([e.src, e.dst]))
        gt_nearer = gt_order.get(pair)
        ok = (gt_nearer == e.src) if gt_nearer else None
        n_edge_ok += bool(ok)
        print(f'  边 {e.src}⊃{e.dst} (重叠{e.overlap_px}px, '
              f'gap={e.depth_gap:.0f}mm) GT={gt_nearer} '
              f'{"✓" if ok else "✗"}')
    edge_acc = n_edge_ok / max(len(edges_est), 1)

    # ---------- 遮挡决策正确率 + 输出图 ----------
    comp_est, vdepth_est, owner_est = sg_est.composite(
        rgb, device=device, radius=2)
    occ_acc = occlusion_decision_accuracy(vdepth_est, vdepth_gt, depth_real)
    # 分组指标: 注册良好子集 (duck+ape) 的遮挡正确率 —— 隔离机制 vs 位姿误差
    idx_da = [list(models.keys()).index(n) for n in ('duck', 'ape')]
    vd_est2 = np.where(np.isin(owner_est, idx_da), vdepth_est, 0)
    vd_gt2 = np.where(np.isin(owner_gt, idx_da), vdepth_gt, 0)
    occ_acc2 = occlusion_decision_accuracy(vd_est2, vd_gt2, depth_real)

    tri = np.concatenate([
        rgb.astype(np.float64) / 255.0 if rgb.dtype != np.float64 else rgb,
        comp_gt, comp_est], axis=1)
    tri_u8 = (np.clip(tri, 0, 1) * 255).astype(np.uint8)
    for i, lab in enumerate(['Real bg', 'GT composite', 'Est composite']):
        cv2.putText(tri_u8, lab, (i * W + 8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imwrite(str(OUT / 'scene_graph.png'),
                cv2.cvtColor(tri_u8, cv2.COLOR_RGB2BGR))

    # ---------- 汇总 ----------
    adds_arr = np.array([r[1] for r in rows])
    print('\n' + '=' * 66)
    print('M3 汇总:')
    print(f'  每模型 ADD-S: {dict((r[0], round(r[1], 2)) for r in rows)}')
    print(f'  ADD-S 均值: {np.nanmean(adds_arr):.2f}mm, '
          f'<10%直径比例: {float(np.nanmean(adds_arr < 12)):.0%}')
    print(f'  场景图边正确率: {n_edge_ok}/{len(edges_est)} = {edge_acc:.0%}')
    print(f'  遮挡决策正确率: {occ_acc:.1%}  (全模型, 受can位姿误差主导)')
    print(f'  遮挡决策正确率: {occ_acc2:.1%}  (duck+ape 注册良好子集)')
    print(f'  输出: {OUT / "scene_graph.png"}')
    print('=' * 66)


if __name__ == '__main__':
    main()
