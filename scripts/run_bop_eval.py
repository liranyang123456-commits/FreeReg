# -*- coding: utf-8 -*-
"""
BOP Linemod 真实数据评测
========================

在 BOP LM 基准 (D:\\bop\\lm) 上评测端到端 RGB-D 注册管线:

    深度反投影 → 多起点粗配准 → trimmed ICP 精化 → 与 GT 位姿比较

指标 (src/metrics.py):
  ADD / ADD-S (BOP 官方), AR@5/10/15/20%直径, 重投影RMSE(px),
  位姿误差(°, mm), 单帧耗时/FPS

说明:
  - 目标区域使用 BOP 官方 mask_visib (模拟 SAM-2 输出, 与评测协议无关)
  - 单位: 毫米 (BOP LM 深度/模型/GT 均为 mm, depth_scale=1.0)
  - 本管线为"经典几何基线", 与 FoundationPose 位姿引擎接口同构可替换

用法:
    conda activate py3d
    python scripts/run_bop_eval.py --scenes 1 2 3 4 5 --frames 25
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
from src.metrics import (add_error, add_s_error, ar_recall, pose_errors,
                         reproj_rmse)

BOP_ROOT = Path(r'D:\bop\lm')


def load_scene(scene_id: int):
    sdir = BOP_ROOT / 'test' / f'{scene_id:06d}'
    with open(sdir / 'scene_camera.json') as f:
        cams = json.load(f)
    with open(sdir / 'scene_gt.json') as f:
        gts = json.load(f)
    with open(sdir / 'scene_gt_info.json') as f:
        infos = json.load(f)
    return sdir, cams, gts, infos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    ap.add_argument('--frames', type=int, default=25,
                    help='每场景均匀采样帧数 (0=全部)')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--engine', default='classic',
                    choices=['classic', 'rc', 'hybrid'],
                    help='位姿引擎: classic=双引擎几何, rc=渲染-比较, '
                         'hybrid=SIFT纹理对应优先+几何兜底(抗对称)')
    ap.add_argument('--n_refs', type=int, default=8,
                    help='hybrid 引擎的 SIFT 参考帧数 (取评估集之外)')
    ap.add_argument('--out', default='outputs/bop_eval_results.csv')
    args = ap.parse_args()

    cfg = RegConfig(device=args.device)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 模型缓存 (obj_id → dict)
    with open(BOP_ROOT / 'models' / 'models_info.json') as f:
        models_info = json.load(f)
    model_cache = {}

    def get_model(obj_id):
        if obj_id not in model_cache:
            pts, normals = load_mesh_points(
                str(BOP_ROOT / 'models' / f'obj_{obj_id:06d}.ply'),
                n_points=20000)
            # ADD 评估用确定性子采样
            eval_pts = pts[::max(1, len(pts) // 2000)]
            entry = {
                'pts': pts, 'normals': normals, 'eval_pts': eval_pts,
                'diameter': models_info[str(obj_id)]['diameter'],
            }
            if args.engine == 'rc':
                from src.render_compare_engine import PoseEngine
                entry['engine'] = PoseEngine(
                    pts, entry['diameter'], backend='rc',
                    device=args.device)
            model_cache[obj_id] = entry
        return model_cache[obj_id]

    rows = []
    t_start = time.time()
    sift_stat = {'sift': 0, 'fallback': 0}
    for scene_id in args.scenes:
        sdir, cams, gts, infos = load_scene(scene_id)
        keys = sorted(cams.keys(), key=int)
        if args.frames > 0 and len(keys) > args.frames:
            sel = np.linspace(0, len(keys) - 1, args.frames).astype(int)
            keys = [keys[i] for i in sel]
        print(f'\n=== 场景 {scene_id:06d}: {len(keys)} 帧 ===')

        # hybrid: 构建 SIFT 参考库 (排除评估帧, 每物体一库)
        sift_libs = {}
        if args.engine == 'hybrid':
            from src.appearance_engine import (SIFTReferenceLibrary,
                                               register_sift)
            eval_set = set(keys)
            obj_ids = {g['obj_id'] for k in keys for g in gts[k]}
            for obj_id in obj_ids:
                m = get_model(obj_id)
                ref_keys = [k for k in sorted(cams.keys(), key=int)
                            if k not in eval_set
                            and any(g['obj_id'] == obj_id
                                    for g in gts[k])]
                sel_ref = np.linspace(0, len(ref_keys) - 1,
                                      min(args.n_refs, len(ref_keys))
                                      ).astype(int)
                lib = SIFTReferenceLibrary()
                n_added = 0
                for ri in sel_ref:
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
                    depth_r *= cams[rk].get('depth_scale', 1.0)
                    K_r = np.array(cams[rk]['cam_K']).reshape(3, 3)
                    mk_r = sdir / 'mask' / f'{int(rk):06d}_{gi:06d}.png'
                    mask_r = cv2.imread(str(mk_r), cv2.IMREAD_GRAYSCALE) \
                        if mk_r.exists() else None
                    T_r = np.eye(4)
                    T_r[:3, :3] = np.array(g['cam_R_m2c']).reshape(3, 3)
                    T_r[:3, 3] = np.array(g['cam_t_m2c']).reshape(3)
                    if lib.add(rgb_r, depth_r, K_r, T_r, mask=mask_r,
                               frame_id=int(rk)):
                        n_added += 1
                sift_libs[obj_id] = lib
                print(f'  [hybrid] obj{obj_id}: SIFT参考库 {n_added} 帧')

        for k in keys:
            fi = int(k)
            K = np.array(cams[k]['cam_K']).reshape(3, 3)
            depth = cv2.imread(str(sdir / 'depth' / f'{fi:06d}.png'),
                               cv2.IMREAD_UNCHANGED)
            if depth is None:
                continue
            depth = depth.astype(np.float64) * cams[k].get('depth_scale', 1.0)

            for gi, g in enumerate(gts[k]):
                obj_id = g['obj_id']
                m = get_model(obj_id)
                # 可见 mask (模拟 SAM-2 分割输出)
                mv_path = sdir / 'mask_visib' / f'{fi:06d}_{gi:06d}.png'
                m_path = sdir / 'mask' / f'{fi:06d}_{gi:06d}.png'
                if mv_path.exists():
                    mask = cv2.imread(str(mv_path), cv2.IMREAD_GRAYSCALE)
                elif m_path.exists():
                    mask = cv2.imread(str(m_path), cv2.IMREAD_GRAYSCALE)
                else:
                    mask = None

                T_gt = np.eye(4)
                T_gt[:3, :3] = np.array(g['cam_R_m2c']).reshape(3, 3)
                T_gt[:3, 3] = np.array(g['cam_t_m2c']).reshape(3)

                t0 = time.time()
                if args.engine == 'rc':
                    T_est, eng_stats = m['engine'].register(depth, K,
                                                            mask=mask)
                    res = {'T': T_est,
                           'stats': {'rmse': eng_stats.get(
                               'hypotheses', [{}])[0].get('icp_rmse', -1)}}
                elif args.engine == 'hybrid':
                    # 三引擎候选 + 深度一致性仲裁 (SIFT/FPFH/多起点)
                    from src.pipeline import arbitrate_poses
                    from src.geometry import backproject_depth
                    from src.pose_estimation import icp as _icp
                    rgb_q = cv2.cvtColor(cv2.imread(
                        str(sdir / 'rgb' / f'{fi:06d}.png')),
                        cv2.COLOR_BGR2RGB)
                    scene_pts, _, _ = backproject_depth(depth, K, mask=mask)
                    sel_s = np.random.default_rng(0).choice(
                        len(scene_pts), min(20000, len(scene_pts)),
                        replace=False)
                    model_icp = m['pts'][::max(1, len(m['pts']) // 3000)]
                    cands = []
                    T_s, st_s = register_sift(sift_libs[obj_id], rgb_q, K,
                                              mask_q=mask)
                    if T_s is not None:
                        T_r, _ = _icp(model_icp, scene_pts[sel_s],
                                      T_init=T_s, max_iter=50,
                                      trim_ratio=0.7, device=args.device)
                        cands.append(('sift', T_r))
                        sift_stat['sift'] += 1
                    res_g = register_rgbd(depth, K, m['pts'], mask=mask,
                                          cfg=cfg)
                    if res_g['T'] is not None:
                        cands.append(('geom', res_g['T']))
                        sift_stat['fallback'] += 1
                    if not cands:
                        res = {'T': None, 'stats': {'error': 'no candidates'}}
                    else:
                        T_est, bs, _sc = arbitrate_poses(
                            [c[1] for c in cands], m['pts'], depth, K, mask)
                        res = {'T': T_est,
                               'stats': {'rmse': bs,
                                         'winner': cands[int(np.argmax(
                                             _sc[i]['score']
                                             for i in range(len(_sc))))][0]}}
                else:
                    res = register_rgbd(depth, K, m['pts'], mask=mask,
                                        cfg=cfg)
                dt = time.time() - t0
                T_est = res['T']
                if T_est is None:
                    rows.append(dict(scene=scene_id, frame=fi, obj=obj_id,
                                     fail=True))
                    continue

                add = add_error(m['eval_pts'], T_est, T_gt)
                adds = add_s_error(m['eval_pts'], T_est, T_gt)
                rerr = reproj_rmse(m['eval_pts'], K, T_est, T_gt)
                rot_deg, t_mm = pose_errors(T_est, T_gt)
                rows.append(dict(
                    scene=scene_id, frame=fi, obj=obj_id, fail=False,
                    add=add, add_s=adds, reproj=rerr, rot_deg=rot_deg,
                    t_mm=t_mm, icp_rmse=res['stats']['rmse'],
                    time_s=dt, diameter=m['diameter'],
                ))
                print(f'  f{fi:04d} obj{obj_id:02d}: ADD={add:7.2f}mm '
                      f'ADD-S={adds:7.2f}mm reproj={rerr:6.2f}px '
                      f'rot={rot_deg:5.1f}° t={dt:.2f}s')

    # ================= 汇总 =================
    ok_rows = [r for r in rows if not r.get('fail')]
    n_fail = len(rows) - len(ok_rows)
    print('\n' + '=' * 70)
    print(f'BOP LM 评测汇总: 有效 {len(ok_rows)} 样本, 失败 {n_fail}'
          + (f' | SIFT成功 {sift_stat["sift"]}, 几何兜底 {sift_stat["fallback"]}'
             if args.engine == 'hybrid' else ''))
    print('=' * 70)
    if ok_rows:
        adds = np.array([r['add'] for r in ok_rows])
        addss = np.array([r['add_s'] for r in ok_rows])
        reprj = np.array([r['reproj'] for r in ok_rows])
        rots = np.array([r['rot_deg'] for r in ok_rows])
        times = np.array([r['time_s'] for r in ok_rows])
        dias = np.array([r['diameter'] for r in ok_rows])

        rec_add = ar_recall(adds, 1.0)  # 占位, 下面按样本直径算
        # 按各样本自身直径计算召回
        for th in (0.05, 0.10, 0.15, 0.20):
            r_add = float((adds < th * dias).mean())
            r_adds = float((addss < th * dias).mean())
            print(f'  AR@{int(th*100):2d}%d : ADD={r_add:.3f}  ADD-S={r_adds:.3f}')
        print(f'  重投影RMSE: mean={reprj.mean():.2f}px  median={np.median(reprj):.2f}px')
        print(f'  旋转误差:   mean={rots.mean():.2f}°  median={np.median(rots):.2f}°')
        print(f'  ADD:        mean={adds.mean():.2f}mm  median={np.median(adds):.2f}mm')
        print(f'  单帧耗时:   mean={times.mean():.2f}s  median={np.median(times):.2f}s '
              f'→ {1.0 / max(times.mean(), 1e-9):.1f} FPS')

    # 保存 CSV
    import csv
    if rows:
        keys_out = sorted({k2 for r in rows for k2 in r.keys()})
        with open(out_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=keys_out)
            w.writeheader()
            w.writerows(rows)
        print(f'\n逐样本结果已保存: {out_path}')
    print(f'总耗时: {time.time() - t_start:.1f}s')


if __name__ == '__main__':
    main()
