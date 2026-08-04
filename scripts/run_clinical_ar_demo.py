# -*- coding: utf-8 -*-
"""
E1-D: 省医真实支气管镜视频 AR 演示 (无标定单目)
================================================

真实临床链路 (无GT、无标定, "自由坐标系"的真实试炼):
  真实视频帧 → Depth-Anything-V2 单目深度 (相对深度)
  → 尺度搜索 + trimmed ICP (相对深度仿射歧义处理)
  → CT 气道模型注册到真实帧 → AR 融合输出
  → 无GT代理指标: 深度形状相关(Spearman) / 管腔轮廓IoU / 残差 / 耗时

诚实声明: 单目相对深度存在尺度-偏移歧义, 本演示展示的是
"自由坐标系"注册的可行性边界; 生产系统用双目/内镜测径或 VGGT。

用法:
    conda activate py3d
    python scripts/run_clinical_ar_demo.py
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import sys
import time
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.geometry import make_T, transform, backproject_depth
from src.pipeline import render_model, lambert_colors, composite_depth_aware
from src.pose_estimation import icp
from src.metrics import add_s_error
import torch

VIDEO = (r'F:\省医-数据\省医-数据\ION患者整理\001王继深\术中视频'
         r'\Video_screenshot_Henan1\ion_video_2021-10-26_03-33-50_L.mp4')
DA_REPO = r'E:\SOTA_Methods\Depth-Anything-V2'
DA_CKPT = r'E:\SOTA_Methods\Depth-Anything-V2\checkpoints\depth_anything_v2_vits.pth'
CACHE = Path('outputs/ct/airway_cache.npz')
OUT = Path('outputs/ct/clinical')
N_DEMO_FRAMES = 8
MODEL_SCALE_PRIOR = 0.05   # DA相对深度→mm 的初始尺度搜索中心


def load_da(device):
    """加载 Depth-Anything-V2 (本地仓库+权重)"""
    sys.path.insert(0, DA_REPO)
    from depth_anything_v2.dpt import DepthAnythingV2
    model = DepthAnythingV2(encoder='vits', features=64,
                            out_channels=[48, 96, 192, 384])
    model.load_state_dict(torch.load(DA_CKPT, map_location='cpu'))
    return model.to(device).eval()


def infer_depth(da, bgr, device):
    """DA-V2 推理 → 相对深度图 (越大越远, 已归一)"""
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        d = da.infer_image(img)
    d = (d - d.min()) / max(d.max() - d.min(), 1e-9)
    return d.astype(np.float32)


def spearman(a, b):
    """Spearman 秩相关 (深度形状一致性, 免 scipy.stats)"""
    from scipy.stats import rankdata
    ra, rb = rankdata(a), rankdata(b)
    ra = (ra - ra.mean()) / (ra.std() + 1e-12)
    rb = (rb - rb.mean()) / (rb.std() + 1e-12)
    return float((ra * rb).mean())


def register_scale_search(model_pts, obs_depth, K, device,
                          scales=None, max_scene=6000):
    """
    相对深度尺度搜索 + 质心对齐 + trimmed ICP (仿射歧义工程处理)
    Returns: T, scale, rmse
    """
    scales = scales if scales is not None else \
        np.geomspace(0.02, 1.0, 12)
    H, W = obs_depth.shape
    best = {'rmse': np.inf}
    rng = np.random.default_rng(0)
    model_icp = model_pts[rng.choice(len(model_pts),
                                     min(3000, len(model_pts)),
                                     replace=False)]
    for s in scales:
        scene, _, _ = backproject_depth(obs_depth * s * 1000.0, K,
                                        stride=2)
        if len(scene) < 200:
            continue
        scene = scene[rng.choice(len(scene),
                                 min(max_scene, len(scene)),
                                 replace=False)]
        T0 = make_T(np.eye(3), scene.mean(0) - model_icp.mean(0))
        T, st = icp(model_icp, scene, T_init=T0, max_iter=15,
                    trim_ratio=0.5, device=device)
        if st['rmse'] < best['rmse']:
            best = {'rmse': st['rmse'], 'T': T, 'scale': s,
                    'scene': scene}
    return best.get('T'), best.get('scale'), best.get('rmse'), \
        best.get('scene')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('=' * 66)
    print('E1-D: 省医001 真实支气管镜视频 AR 演示')
    print('=' * 66)

    # ---------- 模型 ----------
    z = np.load(CACHE)
    model_pts = z['points']
    mu = model_pts.mean(0)
    model_pts = model_pts - mu        # 中心化 (尺度/位姿从观测估计)
    print(f'气道模型: {len(model_pts)}点 (已中心化)')

    # ---------- DA ----------
    print('加载 Depth-Anything-V2 (vits)...')
    da = load_da(device)

    # ---------- 抽帧 ----------
    cap = cv2.VideoCapture(VIDEO)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    idxs = np.linspace(int(total * 0.1), int(total * 0.9),
                       N_DEMO_FRAMES).astype(int)
    print(f'视频: {total}帧 @{fps:.0f}fps, 抽取 {len(idxs)} 帧: {idxs}')

    rows, panels = [], []
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, bgr = cap.read()
        if not ok:
            continue
        H, W = bgr.shape[:2]
        K = np.array([[W / 2., 0, W / 2], [0, W / 2., H / 2], [0, 0, 1.0]])

        t0 = time.time()
        dep_rel = infer_depth(da, bgr, device)
        T, s, rmse, scene = register_scale_search(model_pts, dep_rel, K,
                                                  device)
        dt = time.time() - t0

        if T is None:
            print(f'  f{fi}: 注册失败')
            continue

        # 渲染 + 代理指标
        colors = lambert_colors(np.zeros_like(model_pts), np.eye(3),
                                base_rgb=(0.2, 0.9, 0.4))
        out = render_model(model_pts, colors, T, K, H, W, radius=2,
                           device=device)
        rd = out['depth'].cpu().numpy()
        rm = (rd > 0)
        # 深度形状相关 (重叠区, DA深度需同尺度对齐: 用估计尺度)
        ov = rm & (dep_rel > 0)
        rho = spearman((dep_rel[ov] * s).ravel(), rd[ov].ravel()) \
            if ov.sum() > 100 else float('nan')
        # 管腔轮廓 IoU: 暗区(管腔) vs 模型渲染mask
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        dark = gray < np.percentile(gray, 35)
        iou = float(((rm & dark).sum())
                    / max(((rm | dark).sum()), 1))
        rows.append(dict(frame=fi, scale=s, rmse=rmse, rho=rho, iou=iou,
                         ms=dt * 1000))
        print(f'  f{fi}: 尺度={s:.3f} rmse={rmse:.1f} '
              f'深度相关={rho:.2f} 轮廓IoU={iou:.2f} t={dt * 1000:.0f}ms')

        # AR 融合 (观测深度作场景深度代理)
        comp, occ = composite_depth_aware(
            torch.as_tensor(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                            .astype(np.float32) / 255.0, device=out['feat'].device),
            torch.as_tensor(dep_rel * s * 1000.0, dtype=torch.float32,
                            device=out['feat'].device),
            out['feat'], out['depth'], alpha=0.6)
        comp = comp.cpu().numpy()
        tri = np.concatenate([cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                              .astype(np.float64) / 255.0,
                              out['feat'].cpu().numpy(), comp], axis=1)
        panels.append(tri)

    cap.release()

    # ---------- 汇总 ----------
    if rows:
        rho_a = np.nanmean([r['rho'] for r in rows])
        iou_a = np.mean([r['iou'] for r in rows])
        ms_a = np.mean([r['ms'] for r in rows])
        print('\n' + '=' * 66)
        print('E1-D 汇总 (无GT代理指标):')
        print(f'  深度形状相关(Spearman): mean={rho_a:.2f} '
              f'({sum(1 for r in rows if r["rho"] > 0.5)}/{len(rows)}帧>0.5)')
        print(f'  管腔轮廓IoU:            mean={iou_a:.2f}')
        print(f'  注册+深度估计耗时:      mean={ms_a:.0f}ms/帧')
        comp = (np.clip(np.concatenate(panels[:4], axis=0), 0, 1) * 255
                ).astype(np.uint8)
        cv2.imwrite(str(OUT / 'clinical_ar.png'),
                    cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
        print(f'  输出: {OUT / "clinical_ar.png"} (原帧|模型渲染|AR融合 ×4帧)')
        import csv
        with open(OUT / 'clinical_metrics.csv', 'w', newline='',
                  encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f'  逐帧指标: {OUT / "clinical_metrics.csv"}')
    print('=' * 66)


if __name__ == '__main__':
    main()
