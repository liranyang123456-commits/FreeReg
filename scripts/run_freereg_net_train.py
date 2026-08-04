# -*- coding: utf-8 -*-
"""
FreeRegNet 端到端自由坐标训练 (PoC: BOP LM 单物体)
====================================================

数据: D:\\bop\\lm\\test\\NNNNNN (rgb/depth/mask/scene_camera/scene_gt) +
      D:\\bop\\lm\\models\\obj_*.ply
流程: ROI 裁剪(224) → FreeRegNet → (R, depth) → pose_from_roi → T
      损失 = 自由坐标目标 (旋转 geodesic + 平移 + 深度一致性)
评测: ADD / ADD-S (BOP 协议, 相对物体直径)

临床旁路: 本网络仅刚体物体位姿前端; 临床仍走中心线结构法.

用法:
    conda activate py3d
    python scripts/run_freereg_net_train.py --obj 1 --scenes 1 2 --epochs 5
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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.freereg_net import (FreeRegNet, pose_from_roi, free_coordinate_loss,
                             geodesic_rot_err)
from src.pipeline import load_mesh_points
from src.metrics import add_error, add_s_error

LM = Path(r'D:\bop\lm')
OUT = Path('outputs/freereg_net')


class BopObjectDataset(Dataset):
    """BOP LM 单物体 ROI 数据集."""

    def __init__(self, scenes, obj_id, img_size=224, max_per_scene=200,
                 train=True, seed=0, model_pts=None):
        self.obj_id = obj_id
        self.img_size = img_size
        self.samples = []
        rng = np.random.default_rng(seed)
        for sid in scenes:
            sdir = LM / 'test' / f'{sid:06d}'
            if not sdir.exists():
                continue
            cams = json.load(open(sdir / 'scene_camera.json'))
            gts = json.load(open(sdir / 'scene_gt.json'))
            keys = sorted(cams.keys(), key=int)
            if 0 < max_per_scene < len(keys):
                sel = np.linspace(0, len(keys) - 1, max_per_scene).astype(int)
                keys = [keys[i] for i in sel]
            for k in keys:
                fi = int(k)
                for gi, g in enumerate(gts[k]):
                    if g['obj_id'] != obj_id:
                        continue
                    mv = sdir / 'mask_visib' / f'{fi:06d}_{gi:06d}.png'
                    mk = sdir / 'mask' / f'{fi:06d}_{gi:06d}.png'
                    mp = mv if mv.exists() else mk
                    if not mp.exists():
                        continue
                    T = np.eye(4)
                    T[:3, :3] = np.array(g['cam_R_m2c']).reshape(3, 3)
                    T[:3, 3] = np.array(g['cam_t_m2c']).reshape(3)
                    self.samples.append(dict(
                        scene=sid, frame=fi, gi=gi, K=np.array(
                            cams[k]['cam_K']).reshape(3, 3).astype(np.float32),
                        T=T.astype(np.float32), rgb=str(
                            sdir / 'rgb' / f'{fi:06d}.png'),
                        mask=str(mp)))
        # 打乱 + 训练/验证划分
        idx = rng.permutation(len(self.samples))
        self.samples = [self.samples[i] for i in idx]
        n_val = max(1, int(0.15 * len(self.samples)))
        self.samples = self.samples[n_val:] if train else self.samples[:n_val]
        self.train = train
        print(f'  {"train" if train else "val"}: {len(self.samples)} 样本')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        img = cv2.imread(s['rgb'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(s['mask'], cv2.IMREAD_GRAYSCALE)
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            ys, xs = np.where(mask >= 0)
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0   # 物体图像中心
        m = int(0.25 * max(x1 - x0, y1 - y0)) + 4
        H, W = mask.shape
        x0m, y0m = max(0, x0 - m), max(0, y0 - m)
        x1m, y1m = min(W, x1 + m), min(H, y1 + m)
        crop = img[y0m:y1m, x0m:x1m]
        crop = cv2.resize(crop, (self.img_size, self.img_size))
        crop = crop.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        crop = ((crop - mean) / std).astype(np.float32)
        crop = torch.from_numpy(crop.transpose(2, 0, 1))
        return dict(image=crop, center=torch.tensor([cx, cy], dtype=torch.float32),
                    K=torch.from_numpy(s['K']), T=torch.from_numpy(s['T']))


def evaluate(net, loader, model_pts_t, eval_pts_np, diameter, device):
    net.eval()
    adds, addss = [], []
    with torch.no_grad():
        for b in loader:
            img = b['image'].to(device)
            mp = model_pts_t.unsqueeze(0).repeat(img.shape[0], 1, 1)
            K = b['K'].to(device)
            cen = b['center'].to(device)
            R, depth = net(img, mp)
            T_pred = pose_from_roi(R, depth, cen, K).cpu().numpy()
            T_gt = b['T'].numpy()
            for j in range(T_pred.shape[0]):
                adds.append(add_error(eval_pts_np, T_pred[j], T_gt[j]))
                addss.append(add_s_error(eval_pts_np, T_pred[j], T_gt[j]))
    adds = np.array(adds)
    addss = np.array(addss)
    return dict(
        add_ar10=float((adds < 0.10 * diameter).mean()),
        adds_ar10=float((addss < 0.10 * diameter).mean()),
        add_mean=float(adds.mean()), adds_mean=float(addss.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--obj', type=int, default=1)
    ap.add_argument('--scenes', type=int, nargs='+', default=[1, 2, 5])
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--backbone', default='vit')
    ap.add_argument('--w_proj', type=float, default=0.2)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else 'cpu'
    print('=' * 66)
    print(f'FreeRegNet 训练 PoC: obj_{args.obj:06d}, scenes={args.scenes}')
    print('=' * 66)

    # 模型点云 (canonical) + 评测点 + 直径
    models_info = json.load(open(LM / 'models' / 'models_info.json'))
    diameter = models_info[str(args.obj)]['diameter']
    pts_full, _ = load_mesh_points(
        str(LM / 'models' / f'obj_{args.obj:06d}.ply'), n_points=20000)
    eval_pts = pts_full[::max(1, len(pts_full) // 2000)]
    # 网络条件点 (固定256点, 中心化)
    rng = np.random.default_rng(0)
    cond_idx = rng.choice(len(pts_full), 256, replace=False)
    cond_pts = pts_full[cond_idx].astype(np.float32)
    cond_pts = cond_pts - cond_pts.mean(axis=0, keepdims=True)
    model_pts_t = torch.from_numpy(cond_pts).to(device)

    tr = BopObjectDataset(args.scenes, args.obj, train=True)
    va = BopObjectDataset(args.scenes, args.obj, train=False)
    trl = DataLoader(tr, batch_size=args.batch, shuffle=True,
                     num_workers=0, drop_last=True)
    val = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=0)

    net = FreeRegNet(backbone=args.backbone, n_points=256).to(device)
    nparam = sum(p.numel() for p in net.parameters()) / 1e6
    print(f'  网络参数: {nparam:.1f}M | 直径 {diameter:.1f}mm')
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)

    for ep in range(args.epochs):
        net.train()
        t0 = time.time()
        eloss, erot, etrans, nb = 0.0, 0.0, 0.0, 0
        for b in trl:
            img = b['image'].to(device)
            B = img.shape[0]
            mp = model_pts_t.unsqueeze(0).repeat(B, 1, 1)
            K = b['K'].to(device)
            cen = b['center'].to(device)
            Tg = b['T'].to(device)
            R_gt, t_gt = Tg[:, :3, :3], Tg[:, :3, 3]
            R, depth = net(img, mp)
            T_pred = pose_from_roi(R, depth, cen, K)
            t_pred = T_pred[:, :3, 3]
            loss, parts = free_coordinate_loss(
                R, t_pred, mp, R_gt, t_gt, w_proj=args.w_proj)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            eloss += float(loss)
            erot += parts['rot']
            etrans += parts['trans']
            nb += 1
        # 评测
        m = evaluate(net, val, model_pts_t, eval_pts, diameter, device)
        print(f'  ep{ep + 1}/{args.epochs} loss={eloss / nb:.4f} '
          f'rot={erot / nb:.3f}rad trans={etrans / nb:.1f}mm '
          f'| val ADD-S@10%d={m["adds_ar10"]:.3f} ADD-S_mean={m["adds_mean"]:.1f}mm '
          f'({time.time() - t0:.0f}s)')

    # 保存
    torch.save(net.state_dict(), OUT / f'freereg_net_obj{args.obj:06d}.pth')
    print(f'\n  模型保存: {OUT / f"freereg_net_obj{args.obj:06d}.pth"}')
    m = evaluate(net, val, model_pts_t, eval_pts, diameter, device)
    print('  最终 val:', json.dumps(m, indent=2))


if __name__ == '__main__':
    main()
