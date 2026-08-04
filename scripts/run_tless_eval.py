# -*- coding: utf-8 -*-
"""
BOP T-LESS 评测 (无纹理/对称物体, 本管线 hybrid)
====================================================

数据: D:\\bop\\tless\\test_primesense (20场景) + models_cad
单位: depth_scale=0.1 (raw×0.1=mm), K fx≈1075
协议: 同 BOP LM 评测 (hybrid 引擎, ADD/ADD-S/重投影/耗时)

用法:
    conda activate py3d
    python scripts/run_tless_eval.py --scenes 1 2 3 4 5 --frames 15
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import RegConfig, load_mesh_points, register_rgbd
from src.metrics import (add_error, add_s_error, pose_errors, reproj_rmse)

TL = Path(r'D:\bop\tless')
OUT = Path('outputs/tless')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', type=int, nargs='+',
                    default=list(range(1, 21)))
    ap.add_argument('--frames', type=int, default=15)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device = args.device
    cfg = RegConfig(device=device)
    models_info = json.load(open(TL / 'models_cad' / 'models_info.json'))
    cache = {}

    def get_model(obj_id):
        if obj_id not in cache:
            pts, normals = load_mesh_points(
                str(TL / 'models_cad' / f'obj_{obj_id:06d}.ply'),
                n_points=20000)
            cache[obj_id] = {
                'pts': pts, 'eval': pts[::max(1, len(pts) // 2000)],
                'diameter': models_info[str(obj_id)]['diameter']}
        return cache[obj_id]

    rows = []
    t_all = time.time()
    for sid in args.scenes:
        sdir = TL / 'test_primesense' / f'{sid:06d}'
        if not sdir.exists():
            continue
        cams = json.load(open(sdir / 'scene_camera.json'))
        gts = json.load(open(sdir / 'scene_gt.json'))
        keys = sorted(cams.keys(), key=int)
        if 0 < args.frames < len(keys):
            sel = np.linspace(0, len(keys) - 1, args.frames).astype(int)
            keys = [keys[i] for i in sel]
        print(f'\n=== 场景 {sid:06d}: {len(keys)} 帧 ===')
        for k in keys:
            fi = int(k)
            K = np.array(cams[k]['cam_K']).reshape(3, 3)
            depth = cv2.imread(str(sdir / 'depth' / f'{fi:06d}.png'),
                               cv2.IMREAD_UNCHANGED).astype(np.float64)
            depth *= cams[k].get('depth_scale', 0.1)  # T-LESS: ×0.1 → mm
            for gi, g in enumerate(gts[k]):
                obj_id = g['obj_id']
                m = get_model(obj_id)
                mv = sdir / 'mask_visib' / f'{fi:06d}_{gi:06d}.png'
                mk = sdir / 'mask' / f'{fi:06d}_{gi:06d}.png'
                mask = cv2.imread(str(mv) if mv.exists() else str(mk),
                                  cv2.IMREAD_GRAYSCALE)
                T_gt = np.eye(4)
                T_gt[:3, :3] = np.array(g['cam_R_m2c']).reshape(3, 3)
                T_gt[:3, 3] = np.array(g['cam_t_m2c']).reshape(3)
                t0 = time.time()
                res = register_rgbd(depth, K, m['pts'], mask=mask, cfg=cfg)
                dt = time.time() - t0
                T_est = res['T']
                if T_est is None:
                    rows.append(dict(scene=sid, frame=fi, obj=obj_id,
                                     fail=True))
                    continue
                add = add_error(m['eval'], T_est, T_gt)
                adds = add_s_error(m['eval'], T_est, T_gt)
                rerr = reproj_rmse(m['eval'], K, T_est, T_gt)
                rot, tr = pose_errors(T_est, T_gt)
                rows.append(dict(scene=sid, frame=fi, obj=obj_id, fail=False,
                                 add=add, add_s=adds, reproj=rerr,
                                 rot_deg=rot, t_mm=tr, time_s=dt,
                                 diameter=m['diameter']))
        print(f'  场景{sid} 完成, 累计 {len(rows)} 样本')

    ok_rows = [r for r in rows if not r.get('fail')]
    print('\n' + '=' * 70)
    print(f'T-LESS 汇总: 有效 {len(ok_rows)} 样本, '
          f'失败 {len(rows) - len(ok_rows)}, 总耗时 {time.time() - t_all:.0f}s')
    print('=' * 70)
    if ok_rows:
        adds = np.array([r['add'] for r in ok_rows])
        addss = np.array([r['add_s'] for r in ok_rows])
        reprj = np.array([r['reproj'] for r in ok_rows])
        dias = np.array([r['diameter'] for r in ok_rows])
        times = np.array([r['time_s'] for r in ok_rows])
        for th in (0.05, 0.10, 0.15, 0.20):
            print(f'  AR@{int(th * 100):2d}%d : ADD={float((adds < th * dias).mean()):.3f}  '
          f'ADD-S={float((addss < th * dias).mean()):.3f}')
        print(f'  重投影RMSE: mean={reprj.mean():.2f}px  median={np.median(reprj):.2f}px')
        print(f'  旋转误差:   mean={np.array([r["rot_deg"] for r in ok_rows]).mean():.2f}°')
        print(f'  单帧耗时:   mean={times.mean():.2f}s')

    import csv
    if rows:
        with open(OUT / 'tless_metrics.csv', 'w', newline='',
                  encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f'  CSV: {OUT / "tless_metrics.csv"}')


if __name__ == '__main__':
    main()
