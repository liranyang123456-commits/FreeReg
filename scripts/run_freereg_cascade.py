# -*- coding: utf-8 -*-
"""
FreeRegNet 级联评测: NN 初位姿 → 几何 ICP 精修
====================================================

对比三种方式在同一样本上的 ADD/ADD-S 与耗时:
  (a) 纯 NN (FreeRegNet 前向, ~10ms)
  (b) NN + ICP 级联 (NN 初位姿 → trimmed ICP 精修)
  (c) 纯几何引擎 (register_rgbd, 基线)

用法:
    conda activate py3d
    python scripts/run_freereg_cascade.py --ckpt outputs/freereg_net/freereg_net_multi.pth --scenes 1 2
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.freereg_net import FreeRegNet, pose_from_roi
from src.pose_estimation import icp
from src.geometry import backproject_depth
from src.pipeline import load_mesh_points, RegConfig, register_rgbd
from src.metrics import add_error, add_s_error

LM = Path(r'D:\bop\lm')
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def roi_crop(img, mask, img_size=224):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        ys, xs = np.where(mask >= 0)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    base = max(x1 - x0, y1 - y0)
    m = int(0.25 * base) + 4
    H, W = mask.shape
    x0m, y0m = int(max(0, cx - base / 2 - m)), int(max(0, cy - base / 2 - m))
    x1m, y1m = int(min(W, cx + base / 2 + m)), int(min(H, cy + base / 2 + m))
    crop = cv2.resize(img[y0m:y1m, x0m:x1m], (img_size, img_size))
    crop = crop.astype(np.float32) / 255.0
    crop = ((crop - IMAGENET_MEAN) / IMAGENET_STD).astype(np.float32)
    return torch.from_numpy(crop.transpose(2, 0, 1)), (cx, cy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='outputs/freereg_net/freereg_net_multi.pth')
    ap.add_argument('--scenes', type=int, nargs='+', default=[1, 2])
    ap.add_argument('--max', type=int, default=80)
    ap.add_argument('--with_engine', action='store_true')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else 'cpu'
    # 兼容训练中的 ckpt (不存在则用 smoke)
    ckpt = args.ckpt if Path(args.ckpt).exists() \
        else 'outputs/freereg_net/smoke.pth'
    print(f'级联评测: ckpt={ckpt}')
    net = FreeRegNet(backbone='vit', n_points=256).to(device)
    net.load_state_dict(torch.load(ckpt, map_location=device))
    net.eval()
    registry_info = json.load(open(LM / 'models' / 'models_info.json'))
    cfg = RegConfig(device=device)
    cond_cache, pts_cache = {}, {}

    def get_obj(obj_id):
        if obj_id not in cond_cache:
            pts, _ = load_mesh_points(
                str(LM / 'models' / f'obj_{obj_id:06d}.ply'), n_points=20000)
            rng = np.random.default_rng(obj_id)
            idx = rng.choice(len(pts), 256, replace=len(pts) < 256)
            cond = pts[idx].astype(np.float32)
            cond -= cond.mean(axis=0, keepdims=True)
            cond_cache[obj_id] = torch.from_numpy(cond).to(device)
            pts_cache[obj_id] = dict(
                full=pts, eval=pts[::max(1, len(pts) // 2000)],
                diameter=registry_info[str(obj_id)]['diameter'])
        return cond_cache[obj_id], pts_cache[obj_id]

    rows = []
    for sid in args.scenes:
        sdir = LM / 'test' / f'{sid:06d}'
        if not sdir.exists():
            continue
        cams = json.load(open(sdir / 'scene_camera.json'))
        gts = json.load(open(sdir / 'scene_gt.json'))
        keys = sorted(cams.keys(), key=int)[:args.max]
        for k in keys:
            fi = int(k)
            K = np.array(cams[k]['cam_K']).reshape(3, 3)
            depth = cv2.imread(str(sdir / 'depth' / f'{fi:06d}.png'),
                               cv2.IMREAD_UNCHANGED).astype(np.float64)
            depth *= cams[k].get('depth_scale', 1.0)
            for gi, g in enumerate(gts[k]):
                obj = g['obj_id']
                mv = sdir / 'mask_visib' / f'{fi:06d}_{gi:06d}.png'
                mk = sdir / 'mask' / f'{fi:06d}_{gi:06d}.png'
                mp = mv if mv.exists() else mk
                if not mp.exists():
                    continue
                mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                img = cv2.cvtColor(cv2.imread(str(
                    sdir / 'rgb' / f'{fi:06d}.png')), cv2.COLOR_BGR2RGB)
                cond, prec = get_obj(obj)
                T_gt = np.eye(4)
                T_gt[:3, :3] = np.array(g['cam_R_m2c']).reshape(3, 3)
                T_gt[:3, 3] = np.array(g['cam_t_m2c']).reshape(3)
                # (a) 纯 NN
                crop, (cx, cy) = roi_crop(img, mask)
                t0 = time.time()
                with torch.no_grad():
                    R, dep = net(crop.unsqueeze(0).to(device),
                                 cond.unsqueeze(0))
                    Kt = torch.from_numpy(K.astype(np.float32)).to(device)
                    cen = torch.tensor([[cx, cy]], dtype=torch.float32).to(device)
                    T_nn = pose_from_roi(R, dep, cen,
                                         Kt.unsqueeze(0)).cpu().numpy()[0]
                t_nn = time.time() - t0
                # (b) NN + ICP 级联
                t0 = time.time()
                sp, _, _ = backproject_depth(depth, K, mask=(mask > 0), stride=2)
                T_icp = icp(prec['full'], sp, T_init=T_nn)[0] \
                    if len(sp) > 50 else T_nn
                t_icp = time.time() - t0
                # (c) 纯几何引擎 (可选)
                t_eng, T_eng = 0.0, None
                if args.with_engine:
                    t0 = time.time()
                    res = register_rgbd(depth, K, prec['full'], mask=mask, cfg=cfg)
                    T_eng = res['T']
                    t_eng = time.time() - t0
                rows.append(dict(obj=obj, dia=prec['diameter'],
                    adds_nn=add_s_error(prec['eval'], T_nn, T_gt),
                    adds_icp=add_s_error(prec['eval'], T_icp, T_gt),
                    add_icp=add_error(prec['eval'], T_icp, T_gt),
                    adds_eng=add_s_error(prec['eval'], T_eng, T_gt) if T_eng is not None else np.nan,
                    t_nn=t_nn, t_icp=t_icp + t_nn, t_eng=t_eng))

    d = np.array([r['dia'] for r in rows])
    def ar(key):
        v = np.array([r[key] for r in rows])
        return float((v < 0.10 * d).mean())
    print('\n' + '=' * 60)
    print(f'级联评测 ({len(rows)} 样本)')
    print('=' * 60)
    print(f'  (a) 纯 NN      : ADD-S@10%d={ar("adds_nn"):.3f}  '
          f't={np.mean([r["t_nn"] for r in rows]) * 1000:.0f}ms')
    print(f'  (b) NN+ICP级联 : ADD-S@10%d={ar("adds_icp"):.3f}  '
          f'ADD@10%d={ar("add_icp"):.3f}  t={np.mean([r["t_icp"] for r in rows]) * 1000:.0f}ms')
    if args.with_engine:
        print(f'  (c) 纯几何引擎 : ADD-S@10%d={ar("adds_eng"):.3f}  '
              f't={np.mean([r["t_eng"] for r in rows]) * 1000:.0f}ms')
    import csv
    with open('outputs/freereg_net/cascade_metrics.csv', 'w', newline='',
              encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print('  CSV: outputs/freereg_net/cascade_metrics.csv')


if __name__ == '__main__':
    main()
