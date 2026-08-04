# -*- coding: utf-8 -*-
"""
FreeRegNet 端到端自由坐标训练 (完整版: 多物体+增强+对称感知+长训练)
====================================================================

数据: 全部 13 物体 × 全场景, 数据增强 (crop抖动/光照/颜色/背景).
主干: timm 轻量 ViT (默认) 或 Qwen-VL (--backbone qwen_vl).
对称: 对称物体用 ADD-S 点云损失 (models_info symmetries_*), 非对称用 geodesic.
训练: 余弦退火 + 分层学习率 (主干低 lr, 头高 lr) + 主干分阶段微调.
级联: 训练后用 run_freereg_cascade.py 做 NN初位姿→ICP精修 评测.

用法:
    conda activate py3d
    python scripts/run_freereg_net_train.py --scenes 1 2 4 5 6 8 --epochs 50
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import sys
import json
import math
import time
import argparse
from pathlib import Path

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.freereg_net import (FreeRegNet, pose_from_roi, free_coordinate_loss,
                             geodesic_rot_err)
from src.pipeline import load_mesh_points
from src.metrics import add_error, add_s_error

LM = Path(r'D:\bop\lm')
OUT = Path('outputs/freereg_net')

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ----------------------------------------------------------------------
# 物体注册表: canonical 条件点 + 评测点 + 直径 + 对称信息
# ----------------------------------------------------------------------
class ObjectRegistry:
    def __init__(self, device):
        self.device = device
        self.info = json.load(open(LM / 'models' / 'models_info.json'))
        self.cache = {}

    def get(self, obj_id):
        if obj_id in self.cache:
            return self.cache[obj_id]
        pts, _ = load_mesh_points(
            str(LM / 'models' / f'obj_{obj_id:06d}.ply'), n_points=20000)
        eval_pts = pts_full_sample(pts, 2000)
        # 条件点 (固定256, 中心化)
        rng = np.random.default_rng(obj_id)
        idx = rng.choice(len(pts), 256, replace=len(pts) < 256)
        cond = pts[idx].astype(np.float32)
        cond = cond - cond.mean(axis=0, keepdims=True)
        cond_t = torch.from_numpy(cond).to(self.device)
        # 对称信息
        info = self.info[str(obj_id)]
        sym_mats = []
        for s in info.get('symmetries_discrete', []):
            R = np.array(s).reshape(4, 4)[:3, :3] if len(s) == 16 \
                else np.array(s).reshape(3, 3)
            sym_mats.append(torch.from_numpy(R.astype(np.float32)))
        symmetric = bool(info.get('symmetries_discrete')
                         or info.get('symmetries_continuous'))
        rec = dict(eval_pts=eval_pts, cond_t=cond_t,
                   diameter=info['diameter'], sym_mats=sym_mats,
                   symmetric=symmetric)
        self.cache[obj_id] = rec
        return rec


def pts_full_sample(pts, n):
    return pts[::max(1, len(pts) // n)]


# ----------------------------------------------------------------------
# 多物体数据集 + 数据增强
# ----------------------------------------------------------------------
class BopMultiObjectDataset(Dataset):
    def __init__(self, scenes, train=True, img_size=224, max_per_scene=300,
                 seed=0, augment=False):
        self.img_size = img_size
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.samples = []
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
                    mv = sdir / 'mask_visib' / f'{fi:06d}_{gi:06d}.png'
                    mk = sdir / 'mask' / f'{fi:06d}_{gi:06d}.png'
                    mp = mv if mv.exists() else mk
                    if not mp.exists():
                        continue
                    T = np.eye(4)
                    T[:3, :3] = np.array(g['cam_R_m2c']).reshape(3, 3)
                    T[:3, 3] = np.array(g['cam_t_m2c']).reshape(3)
                    self.samples.append(dict(
                        obj=g['obj_id'], K=np.array(
                            cams[k]['cam_K']).reshape(3, 3).astype(np.float32),
                        T=T.astype(np.float32),
                        rgb=str(sdir / 'rgb' / f'{fi:06d}.png'), mask=str(mp)))
        idx = self.rng.permutation(len(self.samples))
        self.samples = [self.samples[i] for i in idx]
        n_val = max(2, int(0.15 * len(self.samples)))
        self.samples = self.samples[n_val:] if train else self.samples[:n_val]
        print(f'  {"train" if train else "val"}: {len(self.samples)} 样本 '
          f'(多物体)')

    def __len__(self):
        return len(self.samples)

    def _augment(self, crop):
        # 光照/颜色抖动 (亮度/对比度/色调)
        if self.rng.random() < 0.8:
            crop = crop * self.rng.uniform(0.6, 1.4)  # 亮度
        if self.rng.random() < 0.8:
            mean = crop.mean(axis=(0, 1), keepdims=True)
            crop = (crop - mean) * self.rng.uniform(0.6, 1.4) + mean  # 对比度
        if self.rng.random() < 0.5:
            crop = crop + self.rng.normal(0, 0.05, crop.shape).astype(np.float32)
        if self.rng.random() < 0.3:  # 通道 shuffle (背景/光照扰动)
            perm = self.rng.permutation(3)
            crop = crop[..., perm]
        return np.clip(crop, 0, 1)

    def __getitem__(self, i):
        s = self.samples[i]
        img = cv2.cvtColor(cv2.imread(s['rgb']), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(s['mask'], cv2.IMREAD_GRAYSCALE)
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            ys, xs = np.where(mask >= 0)
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        H, W = mask.shape
        # 基础 margin
        base = max(x1 - x0, y1 - y0)
        m = int(0.25 * base) + 4
        # crop 抖动增强
        if self.augment:
            m += int(self.rng.uniform(0, 0.3 * base))
            cx += self.rng.uniform(-0.05, 0.05) * base
            cy += self.rng.uniform(-0.05, 0.05) * base
        x0m = int(max(0, cx - base / 2 - m))
        x1m = int(min(W, cx + base / 2 + m))
        y0m = int(max(0, cy - base / 2 - m))
        y1m = int(min(H, cy + base / 2 + m))
        crop = img[y0m:y1m, x0m:x1m]
        crop = cv2.resize(crop, (self.img_size, self.img_size))
        crop = crop.astype(np.float32) / 255.0
        if self.augment:
            crop = self._augment(crop)
        crop = ((crop - IMAGENET_MEAN) / IMAGENET_STD).astype(np.float32)
        crop = torch.from_numpy(crop.transpose(2, 0, 1))
        return dict(image=crop, obj=s['obj'],
                    center=torch.tensor([cx, cy], dtype=torch.float32),
                    K=torch.from_numpy(s['K']), T=torch.from_numpy(s['T']))


# ----------------------------------------------------------------------
# 评测
# ----------------------------------------------------------------------
def evaluate(net, loader, registry, device):
    net.eval()
    per_obj = {}
    with torch.no_grad():
        for b in loader:
            img = b['image'].to(device)
            K = b['K'].to(device)
            cen = b['center'].to(device)
            for j in range(img.shape[0]):
                obj = int(b['obj'][j])
                rec = registry.get(obj)
                mp = rec['cond_t'].unsqueeze(0)
                R, depth = net(img[j:j + 1], mp)
                T_pred = pose_from_roi(R, depth, cen[j:j + 1],
                                       K[j:j + 1]).cpu().numpy()[0]
                T_gt = b['T'][j].numpy()
                a = add_error(rec['eval_pts'], T_pred, T_gt)
                s_ = add_s_error(rec['eval_pts'], T_pred, T_gt)
                per_obj.setdefault(obj, []).append((a, s_, rec['diameter']))
    alladd, alladds = [], []
    for obj, lst in per_obj.items():
        for a, s_, d in lst:
            alladd.append(a < 0.10 * d)
            alladds.append(s_ < 0.10 * d)
    return dict(add_ar10=float(np.mean(alladd)),
                adds_ar10=float(np.mean(alladds)), n=len(alladd))


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', type=int, nargs='+',
                    default=[1, 2, 4, 5, 6, 8])
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--backbone_lr_mult', type=float, default=0.1)
    ap.add_argument('--freeze_epochs', type=int, default=3)
    ap.add_argument('--backbone', default='vit')
    ap.add_argument('--w_proj', type=float, default=0.2)
    ap.add_argument('--no_augment', action='store_true')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', default='freereg_net_multi.pth')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else 'cpu'
    print('=' * 66)
    print(f'FreeRegNet 完整训练: 多物体 scenes={args.scenes} '
          f'ep={args.epochs} backbone={args.backbone}')
    print('=' * 66)

    registry = ObjectRegistry(device)
    tr = BopMultiObjectDataset(args.scenes, train=True,
                               augment=not args.no_augment)
    va = BopMultiObjectDataset(args.scenes, train=False)
    trl = DataLoader(tr, batch_size=args.batch, shuffle=True,
                     num_workers=0, drop_last=True)
    val = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=0)

    net = FreeRegNet(backbone=args.backbone, n_points=256).to(device)
    nparam = sum(p.numel() for p in net.parameters()) / 1e6
    print(f'  网络参数: {nparam:.1f}M')
    # 分层学习率: 主干低 lr, 头高 lr
    bb = list(net.backbone.parameters())
    head = [p for n, p in net.named_parameters() if not n.startswith('backbone')]
    opt = torch.optim.AdamW([
        {'params': bb, 'lr': args.lr * args.backbone_lr_mult},
        {'params': head, 'lr': args.lr}], weight_decay=1e-4)
    # 余弦退火
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 1e-2)
    # 主干初始冻结
    for p in bb:
        p.requires_grad = False

    best = 0.0
    for ep in range(args.epochs):
        if ep == args.freeze_epochs:  # 解冻主干 (分阶段微调)
            for p in bb:
                p.requires_grad = True
        net.train()
        t0 = time.time()
        el, nb = 0.0, 0
        for b in trl:
            img = b['image'].to(device)
            B = img.shape[0]
            K = b['K'].to(device)
            cen = b['center'].to(device)
            Tg = b['T'].to(device)
            loss_b = img.new_zeros(())
            for j in range(B):
                obj = int(b['obj'][j])
                rec = registry.get(obj)
                mp = rec['cond_t'].unsqueeze(0)
                R, depth = net(img[j:j + 1], mp)
                T_pred = pose_from_roi(R, depth, cen[j:j + 1], K[j:j + 1])
                R_gt = Tg[j:j + 1, :3, :3]
                t_gt = Tg[j:j + 1, :3, 3]
                # 对称物体 → ADD-S 监督; 非对称 → geodesic
                eval_t = torch.from_numpy(
                    rec['eval_pts'].astype(np.float32)).to(device).unsqueeze(0)
                l, _ = free_coordinate_loss(
                    R, T_pred[:, :3, 3], eval_t, R_gt, t_gt,
                    sym_mats=rec['sym_mats'], symmetric=rec['symmetric'],
                    w_proj=args.w_proj, w_adds=1.0 if rec['symmetric'] else 0.0)
                loss_b = loss_b + l
            loss_b = loss_b / B
            opt.zero_grad()
            loss_b.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            el += float(loss_b)
            nb += 1
        sched.step()
        m = evaluate(net, val, registry, device)
        if m['adds_ar10'] > best:
            best = m['adds_ar10']
            torch.save(net.state_dict(), OUT / args.out)
        print(f'  ep{ep + 1}/{args.epochs} loss={el / nb:.4f} '
          f'| val ADD@10%d={m["add_ar10"]:.3f} ADD-S@10%d={m["adds_ar10"]:.3f} '
          f'(best {best:.3f}) lr={sched.get_last_lr()[-1]:.1e} '
          f'({time.time() - t0:.0f}s)', flush=True)

    print(f'\n  最佳 ADD-S@10%d = {best:.3f} | 模型: {OUT / args.out}')


if __name__ == '__main__':
    main()
