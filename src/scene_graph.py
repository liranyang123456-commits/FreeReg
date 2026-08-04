# -*- coding: utf-8 -*-
"""
多模型场景图 (Scene Graph) —— M3
==================================

数据结构设计 (docs/01 §5 的可运行实现):

  SceneNode: 一个已注册模型 (位姿/速度/置信度)
  SceneEdge: 模型间关系 ("occludes" 遮挡 / 深度差)
  SceneGraph: 节点集 + 边集 + 相机位姿 + 内参 + 真实场景深度

核心操作:
  build_edges:   逐对渲染深度图, 重叠区比较中位深度 → 遮挡关系边
  composite:     多模型深度排序 + 真实场景深度感知合成 (三层z-buffer:
                 real / virtual-near / virtual-far), 输出 AR 融合图

依赖: numpy, torch
作者: ZCode · 2026-08-04
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import torch

from src.geometry import transform
from src.pipeline import render_model, composite_depth_aware


@dataclass
class SceneNode:
    model_id: str
    pts: np.ndarray                # (N,3) 模型系点
    colors: np.ndarray             # (N,3) [0,1]
    pose: np.ndarray = None        # (4,4) T_obj^cam
    velocity: np.ndarray = None    # (6,) 可选
    confidence: float = 1.0


@dataclass
class SceneEdge:
    src: str                       # 前景模型 (遮挡者)
    dst: str                       # 被遮挡模型
    relation: str = 'occludes'
    overlap_px: int = 0            # 重叠像素数
    depth_gap: float = 0.0         # 重叠区中位深度差 (正=src更近)


@dataclass
class SceneGraph:
    nodes: Dict[str, SceneNode] = field(default_factory=dict)
    edges: List[SceneEdge] = field(default_factory=list)
    K: np.ndarray = None
    scene_depth: np.ndarray = None   # 真实场景深度 (遮挡合成用, 可None)
    camera_pose: np.ndarray = None   # T_cam^world (可选, 世界系管理时)

    # ---------------- 构建 ----------------
    def add(self, node: SceneNode):
        self.nodes[node.model_id] = node

    def render_each(self, H: int, W: int, device='cuda', radius=1):
        """逐模型渲染 → {model_id: render_out}"""
        outs = {}
        for mid, nd in self.nodes.items():
            if nd.pose is None:
                continue
            outs[mid] = render_model(nd.pts, nd.colors, nd.pose,
                                     self.K, H, W, radius=radius,
                                     device=device)
        return outs

    def build_edges(self, H: int, W: int, device='cuda',
                    min_overlap: int = 20):
        """逐对比较渲染深度, 建立遮挡关系边"""
        outs = self.render_each(H, W, device=device)
        ids = list(outs.keys())
        self.edges = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                da, db = outs[a]['depth'], outs[b]['depth']
                ov = (da > 0) & (db > 0)
                n_ov = int(ov.sum())
                if n_ov < min_overlap:
                    continue
                gap = float((da[ov] - db[ov]).median())
                if gap > 0:   # a 更远 → b 遮挡 a
                    self.edges.append(SceneEdge(
                        src=b, dst=a, overlap_px=n_ov, depth_gap=gap))
                else:
                    self.edges.append(SceneEdge(
                        src=a, dst=b, overlap_px=n_ov, depth_gap=-gap))
        return self.edges

    # ---------------- 合成 ----------------
    def composite(self, bg_rgb: np.ndarray, device='cuda',
                  radius=1, alpha=0.85, radius_scene=1):
        """
        多模型深度排序 + 真实场景深度感知合成
        bg_rgb: (H,W,3) float[0,1] 或 uint8
        Returns: composite (H,W,3) float, virt_depth (H,W) numpy, owner map
        """
        H, W = bg_rgb.shape[:2]
        if bg_rgb.dtype != np.float64:
            bg_rgb = bg_rgb.astype(np.float64) / 255.0
        outs = self.render_each(H, W, device=device, radius=radius)
        dev = 'cuda' if (device == 'cuda' and torch.cuda.is_available()) \
            else 'cpu'
        # 多模型 z-buffer: 逐像素取最近虚拟模型
        virt_depth = torch.full((H, W), float('inf'), device=dev)
        owner = torch.full((H, W), -1, dtype=torch.long, device=dev)
        virt_rgb = torch.zeros(H, W, 3, device=dev)
        for k, (mid, out) in enumerate(outs.items()):
            d = out['depth'].to(dev)
            upd = (d > 0) & (d < virt_depth)
            virt_depth = torch.where(upd, d, virt_depth)
            owner = torch.where(upd, torch.full_like(owner, k), owner)
            virt_rgb[upd] = out['feat'].to(dev)[upd]
        virt_depth = torch.where(torch.isfinite(virt_depth), virt_depth,
                                 torch.zeros((), device=dev))
        # 与真实场景深度合成
        real_rgb = torch.as_tensor(bg_rgb, dtype=torch.float32, device=dev)
        if self.scene_depth is not None:
            rd = torch.as_tensor(self.scene_depth, dtype=torch.float32,
                                 device=dev)
        else:
            rd = torch.zeros(H, W, device=dev)
        comp, occ = composite_depth_aware(real_rgb, rd, virt_rgb,
                                          virt_depth, alpha=alpha)
        owner_np = owner.cpu().numpy()
        return comp.cpu().numpy(), virt_depth.cpu().numpy(), owner_np


def occlusion_decision_accuracy(est_virt_depth: np.ndarray,
                                gt_virt_depth: np.ndarray,
                                scene_depth: np.ndarray) -> float:
    """遮挡决策正确率: 估计合成 vs GT合成 的逐像素可见性一致率"""
    sz = np.where(scene_depth > 0, scene_depth, np.inf)
    vis_est = (est_virt_depth > 0) & (est_virt_depth < sz)
    vis_gt = (gt_virt_depth > 0) & (gt_virt_depth < sz)
    region = (est_virt_depth > 0) | (gt_virt_depth > 0)
    if region.sum() < 10:
        return float('nan')
    return float((vis_est[region] == vis_gt[region]).mean())
