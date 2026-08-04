# -*- coding: utf-8 -*-
"""
CT DICOM 加载与气道建模 —— 医疗 CT-视频注册 (E1) 前端
======================================================

管线:
  DICOM 序列 → HU 体数据 (RescaleSlope/Intercept 校正)
  → 气道分割 (空气阈值 + 气管种子连通域, 形态学清理)
  → marching cubes 表面网格 → 点云
  → 气道主轴估计 (PCA) → 飞行轨迹采样 (合成支气管镜视频用)

单位: 毫米 (CT PixelSpacing/SliceThickness 均为 DICOM 标准 mm)
坐标系: 体素索引 → 物理坐标 (origin + ijk·spacing, LPS/RAS 按 DICOM
ImageOrientationPatient 处理, 默认取轴向扫描标准布局)

依赖: pydicom, numpy, scipy, skimage
作者: ZCode · 2026-08-04
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom
from scipy import ndimage


@dataclass
class CTVolume:
    hu: np.ndarray              # (Z,Y,X) int16 HU
    spacing: np.ndarray         # (3,) mm (z,y,x)
    origin: np.ndarray          # (3,) mm 物理原点
    orientation: np.ndarray     # (3,3) 体素→物理 旋转 (默认单位阵)

    def ijk_to_xyz(self, ijk: np.ndarray) -> np.ndarray:
        """体素索引 (N,3) → 物理坐标 mm"""
        return self.origin + ijk * self.spacing

    def xyz_to_ijk(self, xyz: np.ndarray) -> np.ndarray:
        return (xyz - self.origin) / self.spacing


@dataclass
class AirwayModel:
    points: np.ndarray          # (N,3) 气道表面点云 (mm, 物理系)
    triangles: np.ndarray       # (M,3) 三角面 (可空)
    mask: np.ndarray            # (Z,Y,X) bool 气道掩码 (体素)
    axis_pts: np.ndarray        # (K,3) 主轴采样点 (mm)
    axis_dir: np.ndarray        # (3,) 主轴方向 (单位向量)
    volume: CTVolume = None


def load_dicom_series(ct_dir: str, max_slices: int = 0) -> CTVolume:
    """
    递归读取目录下所有 DICOM 切片 → HU 体数据 (按 InstanceNumber/位置排序)
    max_slices>0 时取中间 N 层 (加速)
    """
    ct_dir = Path(ct_dir)
    files = [p for p in ct_dir.rglob('*') if p.is_file()
             and p.suffix == '' and p.name.upper().startswith('I')]
    if not files:
        files = [p for p in ct_dir.rglob('*') if p.is_file()
                 and p.name not in ('DICOMDIR', 'LOCKFILE', 'VERSION')]
    dsets = []
    for p in files:
        try:
            d = pydicom.dcmread(str(p), stop_before_pixels=False)
            _ = d.pixel_array  # 验证可读
            dsets.append(d)
        except Exception:
            continue
    if not dsets:
        raise RuntimeError(f'未找到可读 DICOM: {ct_dir}')

    def sort_key(d):
        return float(getattr(d, 'ImagePositionPatient',
                             [0, 0, getattr(d, 'InstanceNumber', 0)])[2])
    dsets.sort(key=sort_key)
    if max_slices > 0 and len(dsets) > max_slices:
        c = len(dsets) // 2
        dsets = dsets[c - max_slices // 2: c + max_slices // 2]

    imgs = [d.pixel_array.astype(np.float32) for d in dsets]
    slope = float(getattr(dsets[0], 'RescaleSlope', 1.0))
    inter = float(getattr(dsets[0], 'RescaleIntercept', 0.0))
    hu = np.stack(imgs, axis=0) * slope + inter   # (Z,Y,X)

    ps = getattr(dsets[0], 'PixelSpacing', [1.0, 1.0])
    sy, sx = float(ps[0]), float(ps[1])
    if len(dsets) > 1:
        p0 = np.array(getattr(dsets[0], 'ImagePositionPatient', [0, 0, 0]),
                      dtype=float)
        p1 = np.array(getattr(dsets[1], 'ImagePositionPatient', [0, 0, 1]),
                      dtype=float)
        sz = float(abs(p1[2] - p0[2])) or float(
            getattr(dsets[0], 'SliceThickness', 1.0))
    else:
        sz = float(getattr(dsets[0], 'SliceThickness', 1.0))
    origin = np.array(getattr(dsets[0], 'ImagePositionPatient', [0, 0, 0]),
                      dtype=float)
    spacing = np.array([sz, sy, sx])
    return CTVolume(hu=hu.astype(np.int16), spacing=spacing,
                    origin=origin, orientation=np.eye(3))


def segment_airway(vol: CTVolume, hu_lo=-1024, hu_hi=-400,
                   carina_ratio: float = 3.0) -> np.ndarray:
    """
    气道分割 (厚层 CT 稳健版): 空气阈值 → 身体约束 →
    顶部中央种子 (气管入口) 连通域 → 隆突截断 (保留气管+主支气管)

    实测教训 (001患者, 5mm 厚层):
      - 2x2x2 形态学开运算会摧毁薄壁气道 → 禁用
      - 气管经喉部连通体外空气, 不能用最大组件, 必须种子引导
      - 连通组件常含肺内空气 → 用横截面积突增定位隆突并截断

    Returns: (Z,Y,X) bool (气管+主支气管掩码)
    """
    air = (vol.hu > hu_lo) & (vol.hu < hu_hi)
    body = ndimage.binary_fill_holes(vol.hu > -500)
    aib = air & body
    if aib.sum() < 500:
        return np.zeros_like(air)
    # z 向闭运算桥接断片 (厚层 CT: 部分容积效应使气管在层间断连,
    # 实测 001 患者气管碎成 2-4 层一段的 7 段; (5,1,1) 结构只沿 z 桥接)
    aib_c = ndimage.binary_closing(aib, structure=np.ones((5, 1, 1)))
    lab, n = ndimage.label(aib_c)
    Z, Y, X = vol.hu.shape
    # 种子: 上 10% 层内距图像中心最近且组件足够大的空气体素 (气管入口)
    z_top = max(1, Z // 10)
    cand = np.argwhere(aib_c[:z_top])
    if len(cand) == 0:
        cand = np.argwhere(aib_c[:max(1, Z // 3)])
    if len(cand) == 0:
        return np.zeros_like(air)
    cy, cx = Y / 2, X / 2
    d = (cand[:, 1] - cy) ** 2 + (cand[:, 2] - cx) ** 2
    sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1)) if n else []
    for idx in np.argsort(d):
        s = cand[idx]
        lb_s = lab[s[0], s[1], s[2]]
        if lb_s > 0 and sizes[lb_s - 1] >= 200:
            break
    else:
        return np.zeros_like(air)
    airway = (lab == lb_s)
    # 隆突截断: 横截面积沿 z 的廓线, 从顶部往下找首次 > ratio×基线
    area = airway.sum(axis=(1, 2))
    nz = np.nonzero(area)[0]
    if len(nz) > 10:
        base = np.median(area[nz[:max(3, len(nz) // 4)]])
        cut = nz[-1] + 1
        for z in nz:
            if area[z] > max(base * carina_ratio, base + 40):
                cut = z
                break
        airway[cut:] = False
    return airway.astype(bool)


def airway_to_model(vol: CTVolume, mask: np.ndarray,
                    level: float = 0.5, max_pts: int = 60000) -> AirwayModel:
    """
    气道掩码 → marching cubes 网格 (物理坐标 mm) + 点云 + 主轴
    """
    from skimage import measure
    pad = 1
    m = np.pad(mask.astype(np.float32), pad)
    try:
        verts, faces, normals, _ = measure.marching_cubes(
            m, level=level, spacing=tuple(vol.spacing))
        verts_xyz = verts + vol.origin - vol.spacing * pad
        triangles = faces.astype(np.int32)
    except Exception:
        # marching cubes 失败 (退化) → 用边界体素点
        bnd = mask & ~ndimage.binary_erosion(mask)
        ijk = np.argwhere(bnd)
        verts_xyz = vol.origin + ijk * vol.spacing
        triangles = np.zeros((0, 3), dtype=np.int32)
    pts = verts_xyz
    if len(pts) > max_pts:
        sel = np.linspace(0, len(pts) - 1, max_pts).astype(int)
        pts = pts[sel]
    # 主轴: PCA 最大分量方向
    c = pts.mean(axis=0)
    cov = np.cov((pts - c).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis_dir = eigvecs[:, int(np.argmax(eigvals))]
    axis_dir = axis_dir / np.linalg.norm(axis_dir)
    # 主轴采样点: 沿主轴均匀取截面质心
    proj = (pts - c) @ axis_dir
    K = 40
    qs = np.linspace(np.percentile(proj, 2), np.percentile(proj, 98), K)
    axis_pts = []
    for i in range(K - 1):
        sel = (proj >= qs[i]) & (proj < qs[i + 1])
        if sel.sum() > 5:
            axis_pts.append(pts[sel].mean(axis=0))
    axis_pts = np.array(axis_pts) if axis_pts else c.reshape(1, 3)
    return AirwayModel(points=pts, triangles=triangles, mask=mask,
                       axis_pts=axis_pts, axis_dir=axis_dir, volume=vol)


def make_flythrough_poses(model: AirwayModel, n_frames: int = 60,
                          lateral_mm: float = 2.0, rot_deg: float = 4.0,
                          seed: int = 0):
    """
    沿气道主轴采样飞行相机位姿 (模拟支气管镜推进):
    相机位于轴点处, 视轴沿主轴切向, 附小幅横向摆动+旋转扰动 (真实运动剖面)
    Returns: list of (4,4) T_cam_airway (气道/模型系 → 相机系)
    """
    from src.geometry import make_T, exp_so3
    rng = np.random.default_rng(seed)
    pts = model.axis_pts
    if len(pts) < 2:
        raise RuntimeError('轴点不足, 无法飞行')
    # 取主气道中段 (15%~75% 轴点): 末端细支气管管腔塌陷不可观测
    # (临床支气管镜亦止于叶/段支气管, 厚层 CT 更远达不到终末)
    i0 = max(1, int(len(pts) * 0.15))
    i1 = max(i0 + 2, int(len(pts) * 0.75))
    seg = pts[i0:i1]
    n = min(n_frames, len(seg))
    sel = np.linspace(0, len(seg) - 1, n).astype(int)
    pts = seg
    sel = np.clip(sel, 0, len(pts) - 1)
    poses = []
    # 方向: 沿轴前进方向的切向 (平滑)
    tangents = np.gradient(pts, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-12
    up_ref = np.array([0, 0, 1.0])
    if abs(tangents.mean(axis=0)[2]) > 0.9:
        up_ref = np.array([0, 1.0, 0])
    for k in range(len(sel)):
        i = int(sel[k])
        eye = pts[i] + rng.normal(0, lateral_mm * 0.3, 3)
        f = tangents[i]
        # 摆动: 视轴绕切向小角度偏转
        f = exp_so3(rng.normal(0, np.radians(rot_deg) * 0.3, 3)) @ f
        f = f / np.linalg.norm(f)
        z_c = f                                  # OpenCV 视轴=+z
        x_c = np.cross(z_c, up_ref)
        if np.linalg.norm(x_c) < 1e-6:
            x_c = np.cross(z_c, np.array([1, 0, 0.]))
        x_c = x_c / np.linalg.norm(x_c)
        y_c = np.cross(z_c, x_c)
        R = np.stack([x_c, y_c, z_c], axis=0)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = -R @ eye
        poses.append(T)
    return poses


def find_best_series_dir(ct_root: str, min_slices: int = 50) -> Path:
    """在目录中定位切片最多的序列子目录 (多序列 CD 结构适用)"""
    ct_root = Path(ct_root)
    counts = {}
    for p in ct_root.rglob('*'):
        if p.is_file() and p.suffix == '' and p.name.upper().startswith('I') \
                and p.name[1:].isdigit():
            counts[p.parent] = counts.get(p.parent, 0) + 1
    if not counts:
        return ct_root
    best = max(counts.items(), key=lambda kv: kv[1])
    return best[0] if best[1] >= min_slices else ct_root


def load_ct_airway(patient_dir: str, max_slices: int = 250,
                   max_pts: int = 60000) -> AirwayModel:
    """一站式: 患者目录 → 气道模型 (自动定位最佳序列)"""
    patient_dir = Path(patient_dir)
    ct_dir = patient_dir / 'CT'
    if not ct_dir.exists():
        ct_dir = patient_dir
    ct_dir = find_best_series_dir(str(ct_dir))
    vol = load_dicom_series(str(ct_dir), max_slices=max_slices)
    mask = segment_airway(vol)
    return airway_to_model(vol, mask, max_pts=max_pts)
