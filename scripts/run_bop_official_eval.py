# -*- coding: utf-8 -*-
"""
E3.1: BOP 官方协议评测 (MSSD / MSPD / VSD-lite, 自实现)
============================================================

hybrid 引擎逐帧注册 → 按 BOP 2019-2022 官方指标计算:
  MSSD (最大表面对称距离, 阈值0.05-0.5d)
  MSPD (最大对称投影距离, 阈值2.5-25px)
  VSD-lite (可见表面差异, τ=20mm, θ=0.3-0.7, 点泼溅近似)
  AR = mean(MSSD, MSPD, VSD) 分数

与 BOP 官方工具的差异 (诚实声明): 对称集按 NN-最大距离内蕴处理,
VSD 用点泼溅渲染近似 (非网格全分辨率), 绝对值与官方榜单存在
系统性差异, 相对比较公平; 文献值仅作对照标注。

用法:
    conda activate py3d
    python scripts/run_bop_official_eval.py --scenes 1 2 3 4 5 --frames 25
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
from src.geometry import make_T
from src.metrics import mssd, mspd, vsd_lite
from src.appearance_engine import SIFTReferenceLibrary, register_sift
from src.pose_estimation import icp as _icp
from src.geometry import backproject_depth
from src.pipeline import arbitrate_poses

BOP = Path(r'D:\bop\lm')
OUT = Path('outputs/bop_official')


def load_scene(scene_id):
    sdir = BOP / 'test' / f'{scene_id:06d}'
    cams = json.load(open(sdir / 'scene_camera.json'))
    gts = json.load(open(sdir / 'scene_gt.json'))
    return sdir, cams, gts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', type=int, nargs='+',
                    default=[1, 2, 3, 4, 5])
    ap.add_argument('--frames', type=int, default=25)
    ap.add_argument('--n_refs', type=int, default=40)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = args.device
    cfg = RegConfig(device=device)
    models_info = json.load(open(BOP / 'models' / 'models_info.json'))
    model_cache = {}

    def get_model(obj_id):
        if obj_id not in model_cache:
            pts, normals = load_mesh_points(
                str(BOP / 'models' / f'obj_{obj_id:06d}.ply'),
                n_points=20000)
            model_cache[obj_id] = {
                'pts': pts,
                'diameter': models_info[str(obj_id)]['diameter'],
            }
        return model_cache[obj_id]

    rows = []
    t_all = time.time()
    for scene_id in args.scenes:
        sdir, cams, gts = load_scene(scene_id)
        keys = sorted(cams.keys(), key=int)
        if 0 < args.frames < len(keys):
            sel = np.linspace(0, len(keys) - 1, args.frames).astype(int)
            keys = [keys[i] for i in sel]
        print(f'\n=== 场景 {scene_id:06d}: {len(keys)} 帧 ===')
        # hybrid SIFT 参考库
        eval_set = set(keys)
        sift_libs = {}
        obj_ids = {g['obj_id'] for k in keys for g in gts[k]}
        for obj_id in obj_ids:
            lib = SIFTReferenceLibrary()
            ref_keys = [k for k in sorted(cams.keys(), key=int)
                        if k not in eval_set
                        and any(g['obj_id'] == obj_id for g in gts[k])]
            for ri in np.linspace(0, len(ref_keys) - 1,
                                  min(args.n_refs, len(ref_keys))).astype(int):
                rk = ref_keys[ri]
                gi = next(i for i, g in enumerate(gts[rk])
                          if g['obj_id'] == obj_id)
                g = gts[rk][gi]
                rgb_r = cv2.cvtColor(cv2.imread(
                    str(sdir / 'rgb' / f'{int(rk):06d}.png')),
                    cv2.COLOR_BGR2RGB)
                depth_r = cv2.imread(
                    str(sdir / 'depth' / f'{int(rk):06d}.png'),
                    cv2.IMREAD_UNCHANGED).astype(np.float64)
                K_r = np.array(cams[rk]['cam_K']).reshape(3, 3)
                mk_r = sdir / 'mask' / f'{int(rk):06d}_{gi:06d}.png'
                mask_r = cv2.imread(str(mk_r), cv2.IMREAD_GRAYSCALE) \
                    if mk_r.exists() else None
                T_r = np.eye(4)
                T_r[:3, :3] = np.array(g['cam_R_m2c']).reshape(3, 3)
                T_r[:3, 3] = np.array(g['cam_t_m2c']).reshape(3)
                lib.add(rgb_r, depth_r, K_r, T_r, mask=mask_r,
                        frame_id=int(rk))
            sift_libs[obj_id] = lib

        for k in keys:
            fi = int(k)
            K = np.array(cams[k]['cam_K']).reshape(3, 3)
            depth = cv2.imread(str(sdir / 'depth' / f'{fi:06d}.png'),
                               cv2.IMREAD_UNCHANGED).astype(np.float64)
            for gi, g in enumerate(gts[k]):
                obj_id = g['obj_id']
                m = get_model(obj_id)
                mv = sdir / 'mask_visib' / f'{fi:06d}_{gi:06d}.png'
                mk = sdir / 'mask' / f'{fi:06d}_{gi:06d}.png'
                mask = cv2.imread(
                    str(mv) if mv.exists() else str(mk),
                    cv2.IMREAD_GRAYSCALE)
                T_gt = np.eye(4)
                T_gt[:3, :3] = np.array(g['cam_R_m2c']).reshape(3, 3)
                T_gt[:3, 3] = np.array(g['cam_t_m2c']).reshape(3)

                # ---- hybrid 注册 ----
                t0 = time.time()
                rgb_q = cv2.cvtColor(cv2.imread(
                    str(sdir / 'rgb' / f'{fi:06d}.png')),
                    cv2.COLOR_BGR2RGB)
                scene_pts, _, _ = backproject_depth(depth, K, mask=mask)
                sel = np.random.default_rng(0).choice(
                    len(scene_pts), min(20000, len(scene_pts)),
                    replace=False)
                model_icp = m['pts'][::max(1, len(m['pts']) // 3000)]
                cands = []
                T_s, _ = register_sift(sift_libs[obj_id], rgb_q, K,
                                       mask_q=mask)
                if T_s is not None:
                    T_r2, _ = _icp(model_icp, scene_pts[sel], T_init=T_s,
                                   max_iter=50, trim_ratio=0.7,
                                   device=device)
                    cands.append(T_r2)
                res_g = register_rgbd(depth, K, m['pts'], mask=mask,
                                      cfg=cfg)
                if res_g['T'] is not None:
                    cands.append(res_g['T'])
                if not cands:
                    continue
                T_est, _, _ = arbitrate_poses(cands, m['pts'], depth, K,
                                              mask)
                dt_reg = time.time() - t0

                # ---- BOP 指标 ----
                t0 = time.time()
                e_mssd, s_mssd, _ = mssd(m['pts'], T_est, T_gt,
                                         m['diameter'])
                e_mspd, s_mspd, _ = mspd(m['pts'], K, T_est, T_gt)
                e_vsd, s_vsd = vsd_lite(m['pts'], K, T_est, T_gt, mask,
                                        depth, device=device)
                dt_met = time.time() - t0
                rows.append(dict(scene=scene_id, frame=fi, obj=obj_id,
                                 mssd=float(e_mssd), mssd_s=s_mssd,
                                 mspd=float(e_mspd), mspd_s=s_mspd,
                                 vsd=float(e_vsd), vsd_s=s_vsd,
                                 ar=(s_mssd + s_mspd + s_vsd) / 3.0,
                                 reg_ms=dt_reg * 1000,
                                 met_ms=dt_met * 1000))
                if len(rows) % 10 == 0 or fi == keys[-1]:
                    print(f'  f{fi:04d} obj{obj_id:02d}: MSSD={s_mssd:.2f} '
                          f'MSPD={s_mspd:.2f} VSD={s_vsd:.2f} '
                          f'AR={rows[-1]["ar"]:.2f} '
                          f'(注册{dt_reg:.2f}s 指标{dt_met:.2f}s)')

    # ---------- 汇总 ----------
    rows = [r for r in rows if np.isfinite(r['ar'])]
    print('\n' + '=' * 70)
    print(f'BOP 官方协议汇总 (自实现): {len(rows)} 样本')
    print('=' * 70)
    for name, key in [('MSSD', 'mssd_s'), ('MSPD', 'mspd_s'),
                      ('VSD-lite', 'vsd_s'), ('AR(均值)', 'ar')]:
        v = np.array([r[key] for r in rows])
        print(f'  {name:>10}: mean={v.mean():.3f}  median={np.median(v):.3f}')
    v = np.array([r['ar'] for r in rows])
    print(f'  AR 分档: >0.5 {float((v > 0.5).mean()):.0%}, '
          f'>0.7 {float((v > 0.7).mean()):.0%}, '
          f'>0.9 {float((v > 0.9).mean()):.0%}')
    print(f'  注册耗时 mean={np.mean([r["reg_ms"] for r in rows]) / 1000:.2f}s '
          f'指标计算 mean={np.mean([r["met_ms"] for r in rows]) / 1000:.2f}s')

    import csv
    with open(OUT / 'bop_official_metrics.csv', 'w', newline='',
              encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'  逐样本: {OUT / "bop_official_metrics.csv"}')
    print(f'总耗时: {time.time() - t_all:.0f}s')


if __name__ == '__main__':
    main()
