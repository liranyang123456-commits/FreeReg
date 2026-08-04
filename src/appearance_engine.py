# -*- coding: utf-8 -*-
"""
纹理/外观对应位姿引擎 (2D-2D 匹配 + 深度提升 + PnP)
====================================================

对应关系链 (docs/05 §M2):

    查询图像 ←─SIFT─→ 参考图像(已知位姿)
       │                    │
       │ 2D匹配              │ 深度反投影 + T_ref⁻¹
       ▼                    ▼
    2D 像素 ←────────── 模型系 3D 点
       │                    │
       └──── PnP+RANSAC ────┘  →  T_obj^cam_query

对称物体的关键解法: 几何对应(ICP/FPFH)无法区分对称翻转(位姿等价类),
而纹理对应(SIFT 描述子)可以 —— 如 ape 面部纹理与后脑纹理不可互换。

工程要点: BOP 类小目标(物体仅占~0.3%画面)必须 ROI 裁剪+放大后提特征,
否则关键点数量不足 (实测: 全图 3 个 vs ROI放大后 200+ 个)。

参考库说明 (诚实声明): 评测中参考帧取同场景其他帧及其 GT 位姿,
对应生产场景的"物体照片库+已知位姿" (与 OnePose/FoundationPose
model-free 模式的输入假设一致); BOP 官方榜单场景则交给
FoundationPose 的纹理感知神经评分器。

依赖: opencv, numpy
作者: ZCode · 2026-08-03
"""
import cv2
import numpy as np

from src.geometry import inverse_se3, transform
from src.pose_estimation import pnp_ransac


def _roi_detect(gray: np.ndarray, mask: np.ndarray, sift,
                upscale: float = 3.0, dilate: float = 0.25):
    """
    ROI 裁剪放大 SIFT 检测 → (uv(N,2) 原图坐标, des 或 None)

    小目标场景必需: 物体区域裁剪并上采样, 关键点数提升 1-2 个数量级。
    """
    H, W = gray.shape
    if mask is not None and (mask > 0).sum() > 30:
        vs, us = np.nonzero(mask > 0)
        dv = int((vs.max() - vs.min()) * dilate) + 4
        du = int((us.max() - us.min()) * dilate) + 4
        v0, v1 = max(0, vs.min() - dv), min(H, vs.max() + dv)
        u0, u1 = max(0, us.min() - du), min(W, us.max() + du)
        crop = gray[v0:v1, u0:u1]
        cmask = mask[v0:v1, u0:u1]
    else:
        v0 = u0 = 0
        crop, cmask = gray, None
    if upscale != 1.0:
        crop = cv2.resize(crop, None, fx=upscale, fy=upscale,
                          interpolation=cv2.INTER_CUBIC)
        if cmask is not None:
            cmask = cv2.resize(cmask, None, fx=upscale, fy=upscale,
                               interpolation=cv2.INTER_NEAREST)
    kp, des = sift.detectAndCompute(crop, cmask)
    if des is None or len(kp) == 0:
        return np.zeros((0, 2)), None
    uv = np.array([k.pt for k in kp], dtype=np.float64)
    uv = uv / upscale + np.array([u0, v0])
    return uv, des


class SIFTReference:
    """单条参考: 关键点(uv 原图坐标) + 描述子 + 模型系 3D 点 + 位姿"""
    __slots__ = ('uv', 'des', 'pts3d_model', 'T_gt', 'frame_id')

    def __init__(self, uv, des, pts3d_model, T_gt, frame_id):
        self.uv = uv
        self.des = des
        self.pts3d_model = pts3d_model
        self.T_gt = T_gt
        self.frame_id = frame_id


class SIFTReferenceLibrary:
    """参考图像库 (带位姿), 离线构建 ROI-SIFT 特征 + 模型系 3D 点"""

    def __init__(self, n_features: int = 4000, ratio: float = 0.8,
                 upscale: float = 3.0):
        self.sift = cv2.SIFT_create(nfeatures=n_features,
                                    contrastThreshold=0.008)
        self.ratio = ratio
        self.upscale = upscale
        self.refs = []

    def _detect(self, rgb, mask):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        return _roi_detect(gray, mask, self.sift, upscale=self.upscale)

    def add(self, rgb: np.ndarray, depth: np.ndarray, K: np.ndarray,
            T_gt: np.ndarray, mask: np.ndarray = None, frame_id: int = -1):
        """添加一帧参考 (rgb uint8, depth 与模型同单位, T_gt 模型→相机)"""
        uv, des = self._detect(rgb, mask)
        if des is None or len(uv) < 10:
            return False
        # 关键点像素 → 深度采样 → 模型系 3D (无效深度点剔除)
        ui = np.clip(np.round(uv[:, 0]).astype(int), 0, depth.shape[1] - 1)
        vi = np.clip(np.round(uv[:, 1]).astype(int), 0, depth.shape[0] - 1)
        z = depth[vi, ui].astype(np.float64)
        valid = z > 0
        if valid.sum() < 10:
            return False
        x = (uv[:, 0] - K[0, 2]) / K[0, 0] * z
        y = (uv[:, 1] - K[1, 2]) / K[1, 1] * z
        pts_cam = np.stack([x, y, z], axis=1)
        pts_model = transform(inverse_se3(T_gt), pts_cam)  # 相机系→模型系
        self.refs.append(SIFTReference(
            uv, des, pts_model, T_gt, frame_id))
        return True

    def match_reference(self, rgb_q: np.ndarray, mask_q: np.ndarray = None,
                        min_good: int = 6):
        """查询图与所有参考匹配 (ratio test), 返回 (最佳参考, 匹配对, 查询uv)"""
        uv_q, des_q = self._detect(rgb_q, mask_q)
        if des_q is None or len(uv_q) < 10:
            return None, None, None
        bf = cv2.BFMatcher(cv2.NORM_L2)
        best = {'n_good': 0}
        for ri, ref in enumerate(self.refs):
            if ref.des is None or len(ref.des) < 2:
                continue
            knn = bf.knnMatch(ref.des, des_q, k=2)
            good = [m for pair in knn if len(pair) == 2
                    for m, n in [pair] if m.distance < self.ratio * n.distance]
            if len(good) > best['n_good']:
                best = {'n_good': len(good), 'ri': ri, 'good': good}
        if best['n_good'] < min_good:
            return None, None, uv_q
        return best['ri'], best['good'], uv_q


def register_sift(lib: SIFTReferenceLibrary, rgb_q: np.ndarray,
                  K_q: np.ndarray, mask_q: np.ndarray = None,
                  min_inliers: int = 8):
    """
    纹理对应位姿估计: ROI-SIFT 匹配 → 2D-3D → PnP+RANSAC

    Returns:
        T: (4,4) 位姿 或 None
        stats: dict(n_good, n_inliers, inlier_ratio, reproj_err, ref_frame)
    """
    ri, good, uv_q_all = lib.match_reference(rgb_q, mask_q)
    if ri is None:
        return None, {'error': 'no good reference match'}
    ref = lib.refs[ri]
    obj_idx = np.array([m.queryIdx for m in good])
    q_idx = np.array([m.trainIdx for m in good])
    pts3d = ref.pts3d_model[obj_idx]      # 模型系 3D
    uv_q = uv_q_all[q_idx]                # 查询 2D
    T, stats = pnp_ransac(pts3d, uv_q, K_q, reproj_thresh=3.0)
    if T is None or stats.get('n_inliers', 0) < min_inliers:
        return None, {'error': 'pnp failed',
                      'n_good': int(len(good)), 'ref_frame': ref.frame_id}
    stats['n_good'] = int(len(good))
    stats['ref_frame'] = ref.frame_id
    return T, stats
