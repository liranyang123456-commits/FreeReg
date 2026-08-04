# -*- coding: utf-8 -*-
"""
坐标系与投影几何核心库
======================

实现 2D像素 ↔ 3D模型 ↔ 相机坐标系 投影映射链条的全部基础运算:

    模型系 {M} --T(SE3)--> 相机系 {C} --K,π,D--> 像素系 {P}

    x_p = π_D( K · T · X_m )
    X_c = z · K⁻¹ · [u, v, 1]ᵀ      (反投影)

内容:
  - SO(3):  exp_so3 / log_so3 (Rodrigues)
  - SE(3):  exp_se3 / log_se3 / inverse / compose / transform
  - 针孔投影: project (含 OpenCV 径向+切向畸变模型)
  - 反投影: backproject_depth (深度图 → 相机系点云)

约定:
  - 位姿 T ∈ SE(3) 为 4x4 齐次矩阵, 表示 "源坐标系 → 目标坐标系" 的变换
  - 李代数向量约定 xi = [v(3), w(3)] (前3平移, 后3旋转), 与 se3_kalman_filter.py 一致
  - 单位与输入一致 (BOP: 毫米; 其他: 米)

依赖: numpy
作者: ZCode · 2026-08-03
"""
import numpy as np


# ============================================================
# SO(3)
# ============================================================

def hat(w: np.ndarray) -> np.ndarray:
    """so(3) 向量 (3,) → 3x3 反对称矩阵"""
    return np.array([[0.0, -w[2], w[1]],
                     [w[2], 0.0, -w[0]],
                     [-w[1], w[0], 0.0]])


def exp_so3(w: np.ndarray) -> np.ndarray:
    """旋转向量 (3,) → 旋转矩阵 (3,3), Rodrigues 公式"""
    w = np.asarray(w, dtype=np.float64)
    theta = np.linalg.norm(w)
    if theta < 1e-12:
        return np.eye(3) + hat(w)  # 一阶近似, 避免除零
    K = hat(w / theta)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def log_so3(R: np.ndarray) -> np.ndarray:
    """旋转矩阵 (3,3) → 旋转向量 (3,)"""
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-12:
        return np.zeros(3)
    w = theta / (2.0 * np.sin(theta)) * np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ])
    return w


# ============================================================
# SE(3)
# ============================================================

def exp_se3(xi: np.ndarray) -> np.ndarray:
    """se(3) 向量 (6,)=[v,w] → SE(3) 4x4 (指数映射, 闭式解)"""
    xi = np.asarray(xi, dtype=np.float64)
    v, w = xi[:3], xi[3:]
    theta = np.linalg.norm(w)
    T = np.eye(4)
    if theta < 1e-12:
        T[:3, :3] = exp_so3(w)
        T[:3, 3] = v
        return T
    K = hat(w / theta)
    R = np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
    V = (np.eye(3)
         + (1.0 - np.cos(theta)) / theta**2 * K
         + (theta - np.sin(theta)) / theta**3 * (K @ K))
    T[:3, :3] = R
    T[:3, 3] = V @ v
    return T


def log_se3(T: np.ndarray) -> np.ndarray:
    """SE(3) 4x4 → se(3) 向量 (6,)=[v,w] (对数映射)"""
    R, t = T[:3, :3], T[:3, 3]
    w = log_so3(R)
    theta = np.linalg.norm(w)
    if theta < 1e-12:
        return np.concatenate([t, np.zeros(3)])
    K = hat(w / theta)
    V = (np.eye(3)
         + (1.0 - np.cos(theta)) / theta**2 * K
         + (theta - np.sin(theta)) / theta**3 * (K @ K))
    v = np.linalg.solve(V, t)
    return np.concatenate([v, w])


def inverse_se3(T: np.ndarray) -> np.ndarray:
    """SE(3) 求逆"""
    Ti = np.eye(4)
    Rt = T[:3, :3].T
    Ti[:3, :3] = Rt
    Ti[:3, 3] = -Rt @ T[:3, 3]
    return Ti


def compose_se3(*Ts) -> np.ndarray:
    """SE(3) 连乘: compose(A, B, C) = A @ B @ C"""
    out = np.eye(4)
    for T in Ts:
        out = out @ T
    return out


def transform(T: np.ndarray, X: np.ndarray) -> np.ndarray:
    """位姿 T 作用于点集 X (N,3) → (N,3)"""
    return X @ T[:3, :3].T + T[:3, 3]


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """由 R(3,3), t(3,) 组装 4x4"""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


# ============================================================
# 针孔投影 (透视 π + 畸变 D + 内参 K)
# ============================================================

def project(K: np.ndarray, X_c: np.ndarray, dist: np.ndarray = None):
    """
    相机系点 X_c (N,3) → 像素 uv (N,2)

    Args:
        K:    3x3 内参
        X_c:  (N,3) 相机坐标系点 (z>0 为可见)
        dist: OpenCV 畸变系数 [k1,k2,p1,p2,k3] (可选)
    Returns:
        uv: (N,2) 像素坐标
        z:  (N,)  深度 (供深度测试/可见性判断)
    """
    X_c = np.asarray(X_c, dtype=np.float64)
    z = X_c[:, 2].copy()
    z_safe = np.where(np.abs(z) < 1e-12, 1e-12, z)
    x = X_c[:, 0] / z_safe
    y = X_c[:, 1] / z_safe
    if dist is not None:
        x, y = _distort_norm(x, y, dist)
    u = K[0, 0] * x + K[0, 2]
    v = K[1, 1] * y + K[1, 2]
    return np.stack([u, v], axis=1), z


def project_pose(K: np.ndarray, T_cm: np.ndarray, X_m: np.ndarray,
                 dist: np.ndarray = None):
    """模型系点 X_m (N,3) 经位姿 T_cm 变换后投影 → uv (N,2), z (N,)"""
    return project(K, transform(T_cm, X_m), dist)


def _distort_norm(x, y, dist):
    """OpenCV 畸变模型 (归一化平面上): [k1,k2,p1,p2,k3]"""
    d = np.asarray(dist, dtype=np.float64).ravel()
    k1, k2, p1, p2 = d[0], d[1], d[2], d[3]
    k3 = d[4] if len(d) > 4 else 0.0
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return xd, yd


def backproject_depth(depth: np.ndarray, K: np.ndarray,
                      mask: np.ndarray = None, stride: int = 1):
    """
    深度图 → 相机系点云 (反投影)

    Args:
        depth:  (H,W) 深度图 (单位自定, 与输出点一致)
        K:      3x3 内参
        mask:   (H,W) bool/uint8, 仅取 mask>0 的像素 (可选)
        stride: 采样步长 (>1 则降采样)
    Returns:
        points: (M,3) 相机系 3D 点
        uv:     (M,2) 对应像素坐标
        z:      (M,)  深度值
    """
    H, W = depth.shape
    if mask is not None:
        valid = mask.astype(bool) & (depth > 0) & np.isfinite(depth)
    else:
        valid = (depth > 0) & np.isfinite(depth)
    if stride > 1:
        ds = np.zeros_like(valid)
        ds[::stride, ::stride] = True
        valid &= ds
    vs, us = np.nonzero(valid)
    z = depth[vs, us].astype(np.float64)
    x = (us - K[0, 2]) / K[0, 0] * z
    y = (vs - K[1, 2]) / K[1, 1] * z
    points = np.stack([x, y, z], axis=1)
    uv = np.stack([us.astype(np.float64), vs.astype(np.float64)], axis=1)
    return points, uv, z


# ============================================================
# 旋转工具
# ============================================================

def canonical_rotations_24() -> np.ndarray:
    """立方体对称群的 24 个旋转 (带符号置换矩阵, det=+1)

    用途: 多起点 ICP 的规范初始朝向集合。
    """
    import itertools
    rots = []
    for perm in itertools.permutations(range(3)):
        P = np.zeros((3, 3))
        for i, j in enumerate(perm):
            P[i, j] = 1.0
        for signs in itertools.product([1.0, -1.0], repeat=3):
            M = np.diag(signs) @ P
            if np.linalg.det(M) > 0:
                rots.append(M)
    return np.array(rots)  # (24,3,3)
