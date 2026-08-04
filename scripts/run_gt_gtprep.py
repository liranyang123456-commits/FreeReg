# -*- coding: utf-8 -*-
"""
真实视频人工 GT —— ① 提案包生成 (中心线提案+候选CT分叉)
==========================================================

产出 gt_pkg/:
  frames/     原始帧 PNG
  overlays/   帧+图像骨架(绿)+投影CT中心线(红)+分叉(黄) 叠加 PNG
  ct_junctions.csv   CT 分叉点库 (id,x,y,z,degree)
  frames.csv         帧清单 (含注册位姿)
  README.txt         标注说明

后续: 双人用 run_gt_annotate.py 各标一遍 → run_gt_consensus.py 合并。

用法:
    conda activate py3d
    python scripts/run_gt_gtprep.py --n 20
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

from src.geometry import make_T, transform
from src.centerline import (airway_centerline_3d, lumen_skeleton_2d,
                            register_centerline, crop_centerline_by_view,
                            skeleton_junctions_2d)
from src.ct_loader import CTVolume

VIDEO = (r'F:\省医-数据\省医-数据\ION患者整理\001王继深\术中视频'
         r'\Video_screenshot_Henan1\ion_video_2021-10-26_03-33-50_L.mp4')
MASK_CACHE = Path('outputs/ct/airway_mask.npz')
PTS_CACHE = Path('outputs/ct/airway_cache.npz')
OUT = Path('outputs/gt_pkg')


def _align_rot(vec, dst=(0, 0, 1.0)):
    vec = vec / np.linalg.norm(vec)
    dst = np.asarray(dst, dtype=float)
    v = np.cross(vec, dst)
    s = np.linalg.norm(v)
    c = float(vec @ dst)
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    K_ = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K_ + K_ @ K_ * (1 - c) / (s * s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--start', type=float, default=0.05,
                    help='视频起始比例 (默认0.05)')
    ap.add_argument('--end', type=float, default=0.90,
                    help='视频结束比例 (默认0.90)')
    args = ap.parse_args()
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    (OUT / 'frames').mkdir(parents=True, exist_ok=True)
    (OUT / 'overlays').mkdir(parents=True, exist_ok=True)

    # ---------- 气道中心线 + 分叉库 ----------
    zm = np.load(MASK_CACHE)
    z = np.load(PTS_CACHE)
    pts, axis_dir = z['points'], z['axis_dir']
    vol = CTVolume(hu=zm['mask'].astype(np.int16), spacing=zm['spacing'],
                   origin=zm['origin'], orientation=np.eye(3))
    cl_pts, junc_idx, end_idx = airway_centerline_3d(
        zm['mask'], volume=vol, return_junctions=True)
    print(f'中心线 {len(cl_pts)}点, 分叉 {len(junc_idx)}, 端点 {len(end_idx)}')
    with open(OUT / 'ct_junctions.csv', 'w', encoding='utf-8') as f:
        f.write('id,x_mm,y_mm,z_mm,type\n')
        for j in junc_idx:
            p = cl_pts[j]
            f.write(f'{j},{p[0]:.2f},{p[1]:.2f},{p[2]:.2f},junction\n')
        for e in end_idx:
            p = cl_pts[e]
            f.write(f'{e},{p[0]:.2f},{p[1]:.2f},{p[2]:.2f},endpoint\n')

    # ---------- 抽帧 + 提案注册 ----------
    cap = cv2.VideoCapture(VIDEO)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(int(total * args.start), int(total * args.end),
                       args.n).astype(int)
    R0 = _align_rot(axis_dir)
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
            print(f'  f{fi}: 骨架不足, 跳过')
            continue
        T_init = make_T(R0, np.array([0, 0, 60.0]) - R0 @ cl_pts.mean(0))
        cl_crop, crop_idx = crop_centerline_by_view(cl_pts, T_init, K2,
                                                    H2, W2, margin_px=60,
                                                    z_range=(3.0, 150.0))
        if len(cl_crop) < 50:
            cl_crop, crop_idx = cl_pts, np.arange(len(cl_pts))
        jset = set(crop_idx.tolist())
        pos = {g: i for i, g in enumerate(crop_idx.tolist())}
        j_local = np.array([pos[j] for j in junc_idx if j in jset],
                           dtype=np.int64)
        res = register_centerline(cl_crop, dt_img, K2, T_init, iters=250,
                                  lr=8e-3, device=device,
                                  cl_junction_idx=j_local,
                                  lambda_branch=3.0)
        T_est = res['T']

        # 叠加图
        vis = bgr.copy()
        vis[skel] = (0, 255, 0)
        P = transform(T_est, cl_crop)
        zz = P[:, 2]
        okm = zz > 1e-3
        u = np.round(K2[0, 0] * P[okm, 0] / zz[okm] + K2[0, 2]).astype(int)
        v = np.round(K2[1, 1] * P[okm, 1] / zz[okm] + K2[1, 2]).astype(int)
        inb = (u >= 0) & (u < W2) & (v >= 0) & (v < H2)
        for uu, vv in zip(u[inb], v[inb]):
            cv2.circle(vis, (uu, vv), 1, (0, 0, 255), -1)
        if len(j_local) > 0:
            Pj = transform(T_est, cl_crop[j_local])
            zj = Pj[:, 2]
            okj = zj > 1e-3
            uj = np.round(K2[0, 0] * Pj[okj, 0] / zj[okj] + K2[0, 2]
                          ).astype(int)
            vj = np.round(K2[1, 1] * Pj[okj, 1] / zj[okj] + K2[1, 2]
                          ).astype(int)
            inj = (uj >= 0) & (uj < W2) & (vj >= 0) & (vj < H2)
            for uu, vv in zip(uj[inj], vj[inj]):
                cv2.circle(vis, (uu, vv), 4, (0, 255, 255), 1)
        cv2.putText(vis, f'f{fi}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 255), 2)
        cv2.imwrite(str(OUT / 'frames' / f'f{fi:06d}.png'), bgr)
        cv2.imwrite(str(OUT / 'overlays' / f'f{fi:06d}.png'), vis)
        rows.append(dict(frame=fi, cham=res['final_loss'],
                         T=T_est.tolist()))
        print(f'  f{fi}: 提案 chamfer={res["final_loss"]:.1f}px')
    cap.release()

    with open(OUT / 'frames.csv', 'w', encoding='utf-8') as f:
        f.write('frame,proposal_chamfer_px,pose_json\n')
        for r in rows:
            f.write(f'{r["frame"]},{r["cham"]:.2f},'
                    f'"{json.dumps(r["T"])}"\n')
    with open(OUT / 'README.txt', 'w', encoding='utf-8') as f:
        f.write('真实视频人工 GT 标注包\n'
                '=====================\n'
                '1. 双人各自运行:\n'
                '   python scripts/run_gt_annotate.py --reader R1\n'
                '   python scripts/run_gt_annotate.py --reader R2\n'
                '2. 标注操作: 左键点击=管腔/分叉中心; n=下一张; b=上一张;\n'
                '   s=跳过; 1-5=解剖标签(1隆突 2右主 3左主 4右叶 5左叶); q=保存退出\n'
                '3. 合并共识:\n'
                '   python scripts/run_gt_consensus.py\n'
                f'本包含 {len(rows)} 帧提案 (overlay 中: 绿=图像骨架, '
                f'红=投影CT中心线, 黄=CT分叉点)\n')
    print(f'\n提案包完成: {OUT} ({len(rows)}帧)')


if __name__ == '__main__':
    main()
