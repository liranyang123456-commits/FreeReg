# -*- coding: utf-8 -*-
"""
FP 官方结果对比 (FoundationPose vs 本管线, 同帧同 GT)
======================================================
解析 FoundationPose 官方 linemod_res.yml → ADD-S/位姿误差
vs 本管线 hybrid 同帧结果。
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import sys
import json
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline import load_mesh_points
from src.metrics import add_s_error, pose_errors
import yaml

BOP = Path(r'D:\bop\lm')
RES = Path(r'E:\SOTA_Methods\FoundationPose\debug\linemod_res.yml')
SCENE = 1
OBJ = 1


def main():
    d = yaml.safe_load(open(RES, encoding='utf-8'))
    # 结构: {frame_id: [score, pose_4x4] 或 {score:, estR:, estT:}}
    sample = d[list(d.keys())[0]]
    print('单条结构:', type(sample), str(sample)[:150])

    sdir = BOP / 'test' / f'{SCENE:06d}'
    gts = json.load(open(sdir / 'scene_gt.json'))
    pts, _ = load_mesh_points(str(BOP / 'models' / f'obj_{OBJ:06d}.ply'),
                              n_points=20000)
    eval_pts = pts[::10]
    rows = []
    raw_ts = []       # FP 原始平移 (用于尺度判定)
    gt_ts = []
    # 结构: {obj_id: {frame_str: {obj_id: 4x4}}}
    obj_dict = d.get(OBJ) or d.get(str(OBJ)) or d
    for fi_key, v in obj_dict.items():
        fi_str = str(int(fi_key))
        if fi_str not in gts:
            continue
        g = next((g for g in gts[fi_str] if g['obj_id'] == OBJ), None)
        if g is None:
            continue
        try:
            vv = v[str(OBJ)] if isinstance(v, dict) and str(OBJ) in v \
                else (v[OBJ] if isinstance(v, dict) and OBJ in v else v)
            arr = np.array(vv, dtype=float)
            if arr.shape != (4, 4):
                continue
        except Exception:
            continue
        R, t = arr[:3, :3], arr[:3, 3]
        raw_ts.append(t.copy())
        T_est = np.eye(4)
        T_est[:3, :3], T_est[:3, 3] = R, t
        T_gt = np.eye(4)
        T_gt[:3, :3] = np.array(g['cam_R_m2c']).reshape(3, 3)
        T_gt[:3, 3] = np.array(g['cam_t_m2c']).reshape(3)
        gt_ts.append(T_gt[:3, 3].copy())
        adds = add_s_error(eval_pts, T_est, T_gt)
        rot, tr = pose_errors(T_est, T_gt)
        rows.append(dict(frame=int(fi_str), adds=adds, rot=rot, tr=tr))
    if not rows:
        print('未解析到帧, 请检查 linemod_res.yml 结构')
        return
    # 尺度判定: FP 内部归一化单位 → GT(mm) 的全局尺度
    raw_ts = np.array(raw_ts)
    gt_ts = np.array(gt_ts)
    sc = np.median(np.linalg.norm(gt_ts, axis=1)
                   / np.maximum(np.linalg.norm(raw_ts, axis=1), 1e-9))
    print(f'FP→GT 平移尺度比: {sc:.2f} (应用后重评)')
    # 直接对 (R, t*s) 重算 ADD-S
    obj_dict = d.get(OBJ) or d.get(str(OBJ)) or d
    rows2 = []
    for fi_key, v in obj_dict.items():
        fi_str = str(int(fi_key))
        if fi_str not in gts:
            continue
        g = next((g for g in gts[fi_str] if g['obj_id'] == OBJ), None)
        if g is None:
            continue
        try:
            vv = v[str(OBJ)] if isinstance(v, dict) and str(OBJ) in v \
                else (v[OBJ] if isinstance(v, dict) and OBJ in v else v)
            arr = np.array(vv, dtype=float)
            if arr.shape != (4, 4):
                continue
        except Exception:
            continue
        T_est = np.eye(4)
        T_est[:3, :3] = arr[:3, :3]
        T_est[:3, 3] = arr[:3, 3] * sc
        T_gt = np.eye(4)
        T_gt[:3, :3] = np.array(g['cam_R_m2c']).reshape(3, 3)
        T_gt[:3, 3] = np.array(g['cam_t_m2c']).reshape(3)
        rows2.append(dict(adds=add_s_error(eval_pts, T_est, T_gt),
                          tr=np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3])))
    adds_arr = np.array([r['adds'] for r in rows2])
    rot_arr = np.array([r['rot'] for r in rows])
    tr_arr = np.array([r['tr'] for r in rows2])
    print(f'\nFoundationPose 官方 (scene{SCENE} obj{OBJ}, {len(rows2)}帧):')
    print(f'  ADD-S(尺度校正): mean={adds_arr.mean():.2f}mm  '
          f'中位={np.median(adds_arr):.2f}mm')
    print(f'  成功率: <5mm {float((adds_arr < 5).mean()):.0%}  '
          f'<10mm {float((adds_arr < 10).mean()):.0%}')
    print(f'  旋转误差: mean={rot_arr.mean():.2f}°')
    print(f'  平移误差: mean={tr_arr.mean():.2f}mm')
    # 本管线同场景对照 (此前 hybrid 实测 scene1×25: ADD-S AR@10%d=1.000,
    # ADD 中位 3.6mm) —— 见 docs/06/11
    print('\n本管线 hybrid 对照 (scene1, 25帧, docs/06):')
    print('  ADD-S AR@10%d=1.000, ADD 中位≈3.6mm')


if __name__ == '__main__':
    main()
