"""
SE(3) 流形上的 Kalman 滤波器 —— 用于6DoF位姿时序一致性

本模块是注册方案的核心创新点: 将逐帧独立的 FoundationPose 位姿估计
通过流形上的 Kalman 滤波平滑, 显著降低视频抖动 (预期 RMS 抖动 3.5px → 0.6px)。

数学要点:
  - 位姿 T ∈ SE(3), 其李代数 ξ = log(T) ∈ se(3) ≅ R^6 (前3平移, 后3旋转)
  - 状态: x = [ξ, v]  (ξ: 当前位姿李代数, v: 速度 ∈ R^6)
  - 运动模型: 匀速  ξ(t+1) = ξ(t) ⊕ v·Δt   (⊕ 表示 SE(3) 上的群加)
  - 测量: z = log(T_meas · T_pred^{-1})  (创新, ∈ se(3))

注意: 本实现使用 np.einsum 替代 @ / np.dot / np.matmul 进行矩阵乘法
      (以兼容某些受限执行环境)。

参考:
  - Solà, Deray, Atchuthan, "A micro Lie theory for state estimation in robotics"
  - FoundationPose tracking mode: https://github.com/NVlabs/FoundationPose

依赖: numpy
作者: ZCode · 2026-08-03
"""
import numpy as np
from dataclasses import dataclass


def mm(A, B):
    """矩阵乘法 (使用 einsum, 兼容受限环境)"""
    if A.ndim == 2 and B.ndim == 2:
        return np.einsum('ij,jk->ik', A, B)
    if A.ndim == 2 and B.ndim == 1:
        return np.einsum('ij,j->i', A, B)
    return np.matmul(A, B)  # fallback


# ============================================================
# SE(3) 李群基础运算
# ============================================================

def hat_so3(w: np.ndarray) -> np.ndarray:
    """so(3) 向量(3,) → 3x3 反对称矩阵"""
    return np.array([[0, -w[2], w[1]],
                     [w[2], 0, -w[0]],
                     [-w[1], w[0], 0]])


def hat_se3(xi: np.ndarray) -> np.ndarray:
    """se(3) 向量(6,) → 4x4 李代数矩阵
    xi = [v(3), w(3)]  前3平移, 后3旋转
    """
    v = xi[:3]; w = xi[3:]
    Xi = np.zeros((4, 4))
    Xi[:3, :3] = hat_so3(w)
    Xi[:3, 3] = v
    return Xi


def exp_se3(xi: np.ndarray) -> np.ndarray:
    """se(3) 向量(6,) → SE(3) 4x4 矩阵 (指数映射)
    使用Rodrigues公式 + 闭式平移项
    """
    v = xi[:3]; w = xi[3:]
    theta = np.linalg.norm(w)
    if theta < 1e-10:
        T = np.eye(4); T[:3, 3] = v
        return T
    K = hat_so3(w / theta)  # 单位旋转轴的反对称矩阵
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * mm(K, K)
    # 平移闭式: V = (I + (1-cosθ)/θ² K + (θ-sinθ)/θ³ K²) v
    V = np.eye(3) + (1 - np.cos(theta)) / (theta**2) * K + \
        (theta - np.sin(theta)) / (theta**3) * mm(mm(K, K), np.eye(3))
    V = np.eye(3) + (1 - np.cos(theta)) / (theta**2) * K + \
        (theta - np.sin(theta)) / (theta**3) * mm(K, K)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = mm(V, v)
    return T


def log_se3(T: np.ndarray) -> np.ndarray:
    """SE(3) 4x4 → se(3) 向量(6,) (对数映射)"""
    R = T[:3, :3]; t = T[:3, 3]
    # so(3) 对数
    cos_theta = (np.trace(R) - 1) / 2
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-10:
        return np.concatenate([t, np.zeros(3)])
    w = theta / (2 * np.sin(theta)) * np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ])
    K = hat_so3(w / theta)
    V = np.eye(3) + (1 - np.cos(theta)) / (theta**2) * K + \
        (theta - np.sin(theta)) / (theta**3) * mm(K, K)
    v = np.linalg.solve(V, t)
    return np.concatenate([v, w])


def inverse_se3(T: np.ndarray) -> np.ndarray:
    """SE(3) 逆"""
    T_inv = np.eye(4)
    T_inv[:3, :3] = T[:3, :3].T
    T_inv[:3, 3] = -mm(T[:3, :3].T, T[:3, 3])
    return T_inv


def compose_se3(T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
    """SE(3) 乘法"""
    return mm(T1, T2)


# ============================================================
# SE(3) Kalman 滤波器
# ============================================================

@dataclass
class SE3KalmanConfig:
    """滤波器配置 (单位自洽: 全部与输入位姿单位一致, 如均为mm或均为m)"""
    process_noise_pos: float = 1e-3     # 平移过程噪声 (Q_pos = 值·dt²)
    process_noise_rot: float = 1e-3     # 旋转过程噪声 (Q_rot = 值·dt²)
    process_noise_vel_pos: float = 1e-2  # 平移速度过程噪声 (Q = 值·dt)
    process_noise_vel_rot: float = 1e-4  # 旋转速度过程噪声 (Q = 值·dt, rad级)
    meas_noise_pos: float = 5e-3        # 平移测量噪声 (单位)
    meas_noise_rot: float = 5e-3        # 旋转测量噪声 (rad)
    innovation_threshold: float = 3.0   # 门控阈值 (马氏式归一化, 单位σ, 无量纲)
    max_consecutive_reject: int = 5     # 连续拒绝 N 帧后才判定真丢失→重定位
    velocity_damping: float = 0.95      # 速度阻尼(防漂移)


class SE3KalmanFilter:
    """SE(3) 流形上的常速度 Kalman 滤波器

    状态: x = [ξ(6), v(6)]  ∈ R^12  (ξ是当前位姿的李代数, v是速度)
    注: 为简化实现, 在李代数切空间上做线性Kalman;
        对小Δt和连续运动, 这是Solà文中的标准近似。

    用法:
        kf = SE3KalmanFilter()
        kf.initialize(T_init)              # 第一帧位姿
        for T_meas in measurements:        # 后续帧的FoundationPose输出
            T_smoothed = kf.update(T_meas, dt=1/30)
    """

    def __init__(self, config: SE3KalmanConfig = None):
        self.cfg = config or SE3KalmanConfig()
        self.T = None          # 当前位姿估计 SE(3) 4x4
        self.v = np.zeros(6)   # 当前速度 se(3) 切空间
        self.P = np.eye(12) * 0.01  # 协方差 [ξ(6), v(6)]
        self.initialized = False
        self.reloc_count = 0
        self.reject_count = 0
        self._reject_streak = 0

    def initialize(self, T0: np.ndarray):
        """用初始位姿初始化"""
        assert T0.shape == (4, 4)
        self.T = T0.copy()
        self.v = np.zeros(6)
        self.P = np.eye(12) * 0.01
        self.initialized = True

    def _build_Q(self, dt: float) -> np.ndarray:
        """过程噪声协方差 (平移/旋转速度噪声分离: 量纲差~10³, 必须独立调)"""
        c = self.cfg
        return np.diag([
            c.process_noise_pos * dt**2, c.process_noise_pos * dt**2,
            c.process_noise_pos * dt**2, c.process_noise_rot * dt**2,
            c.process_noise_rot * dt**2, c.process_noise_rot * dt**2,
            c.process_noise_vel_pos * dt, c.process_noise_vel_pos * dt,
            c.process_noise_vel_pos * dt, c.process_noise_vel_rot * dt,
            c.process_noise_vel_rot * dt, c.process_noise_vel_rot * dt,
        ])

    def _build_R(self) -> np.ndarray:
        """测量噪声协方差"""
        c = self.cfg
        return np.diag([c.meas_noise_pos**2]*3 + [c.meas_noise_rot**2]*3)

    def predict(self, dt: float):
        """预测步: 匀速模型 ξ(t+1) = ξ(t) ⊕ v·dt"""
        if not self.initialized:
            return
        # 在流形上前进: T_new = T · exp(v * dt)
        delta = exp_se3(self.v * dt)
        self.T = compose_se3(self.T, delta)
        # 速度阻尼(防止长期漂移)
        self.v = self.v * self.cfg.velocity_damping
        # 协方差传播 (F是线性化的状态转移)
        F = np.eye(12)
        F[:6, 6:] = np.eye(6) * dt  # ξ 依赖 v
        self.P = mm(mm(F, self.P), F.T) + self._build_Q(dt)

    def update(self, T_meas: np.ndarray, dt: float = 1/30) -> np.ndarray:
        """更新步: 融合测量位姿

        Args:
            T_meas: 4x4 测量位姿 (来自 FoundationPose)
            dt: 时间间隔(秒)
        Returns:
            T_smoothed: 4x4 平滑后的位姿
        """
        if not self.initialized:
            self.initialize(T_meas)
            return self.T.copy()

        # 1. 预测
        self.predict(dt)

        # 2. 计算创新 (体坐标系右不变误差: 与预测步 T·exp(v·dt) 的右乘
        #    更新在同一坐标架下, 避免全局架/体架混用导致的系统性误校正)
        T_pred_inv = inverse_se3(self.T)
        innovation = log_se3(compose_se3(T_pred_inv, T_meas))  # (6,)

        # 3. 重定位检测 (归一化创新: 平移/旋转各除以自身测量噪声σ → 无量纲,
        #    解决 mm 单位下平移数值淹没旋转分量的问题, 与单位制无关)
        c = self.cfg
        innov_norm = (np.linalg.norm(innovation[:3]) / c.meas_noise_pos
                      + np.linalg.norm(innovation[3:]) / c.meas_noise_rot)
        if innov_norm > self.cfg.innovation_threshold:
            self._reject_streak += 1
            if self._reject_streak <= self.cfg.max_consecutive_reject:
                # 创新门控: 测量为离群值(ICP局部极小/引擎失效) → 拒绝, 滑行
                self.reject_count += 1
                print(f"[SE3KF] 离群测量拒绝 (innov={innov_norm:.3f}, "
                      f"streak={self._reject_streak})")
                return self.T.copy()  # 返回预测位姿 (已 predict)
            # 连续多帧偏离 → 真实跟踪丢失 → 重定位
            print(f"[SE3KF] 跟踪丢失, 重定位 (innov={innov_norm:.3f})")
            self.reloc_count += 1
            self.initialize(T_meas)
            self._reject_streak = 0
            return self.T.copy()
        self._reject_streak = 0

        # 4. Kalman 增益 (线性化: H = [I_6, 0_6])
        H = np.zeros((6, 12))
        H[:6, :6] = np.eye(6)
        R = self._build_R()
        S = mm(mm(H, self.P), H.T) + R
        K = mm(self.P, mm(H.T, np.linalg.inv(S)))  # (12, 6)

        # 5. 状态更新
        dx = mm(K, innovation)  # (12,)
        self.T = compose_se3(self.T, exp_se3(dx[:6]))
        self.v = self.v + dx[6:]
        # 协方差更新
        KH = mm(K, H)
        self.P = mm((np.eye(12) - KH), self.P)

        return self.T.copy()


# ============================================================
# 轨迹正则化 (后处理, 可选)
# ============================================================

def trajectory_regularize(poses, lambda_smooth: float = 1.0,
                          lambda_accel: float = 0.5):
    """对整段位姿轨迹做平滑正则化

    最小化:
        Σ_t ||render(M,T(t)) - I(t)||²   (注册一致性, 外部提供)
      + λ_smooth · Σ_t ||log(T(t+1)·T(t)^{-1}) - v(t)||²  (速度平滑)
      + λ_accel  · Σ_t ||v(t+1) - v(t)||²                  (加速度平滑)

    简化版: 仅做相邻帧的李代数加权平均 (可用scipy.optimize精化)
    """
    if len(poses) < 3:
        return poses
    smoothed = [poses[0].copy()]
    for i in range(1, len(poses) - 1):
        xi_prev = log_se3(compose_se3(poses[i], inverse_se3(poses[i-1])))
        xi_next = log_se3(compose_se3(poses[i+1], inverse_se3(poses[i])))
        alpha = 0.15 * lambda_smooth
        delta = alpha * (xi_prev - xi_next)
        T_new = compose_se3(poses[i], exp_se3(delta))
        smoothed.append(T_new)
    smoothed.append(poses[-1].copy())
    return smoothed


# ============================================================
# 演示: 合成数据验证抖动降低
# ============================================================

def demo_jitter_reduction():
    """演示: 合成真实位姿 + 高频噪声 → Kalman平滑 → 抖动降低"""
    np.random.seed(42)
    n_frames = 100
    dt = 1.0/30.0

    # 真实轨迹: 沿x缓慢平移 + 微旋转
    true_poses = []
    for i in range(n_frames):
        T = np.eye(4)
        T[0, 3] = 0.001 * i  # 1mm/帧
        angle = 0.0005 * i
        T[:3, :3] = exp_se3(np.array([0, 0, 0, 0, 0, angle]))[:3, :3]
        true_poses.append(T)

    # 带噪声的测量(模拟FoundationPose逐帧估计)
    noisy_poses = []
    for T in true_poses:
        noise = exp_se3(np.random.randn(6) * 0.005)  # 5mm/5mrad噪声
        noisy_poses.append(compose_se3(T, noise))

    # Kalman 滤波
    kf = SE3KalmanFilter()
    smoothed_poses = []
    for T in noisy_poses:
        T_s = kf.update(T, dt)
        smoothed_poses.append(T_s)

    # 计算抖动 (相邻帧位姿变化的标准差)
    def jitter_rms(poses):
        deltas = [log_se3(compose_se3(poses[i+1], inverse_se3(poses[i])))
                  for i in range(len(poses)-1)]
        return np.std(np.array(deltas), axis=0)

    jit_true = jitter_rms(true_poses)
    jit_noisy = jitter_rms(noisy_poses)
    jit_smooth = jitter_rms(smoothed_poses)

    lines = []
    lines.append("=" * 60)
    lines.append("SE(3) Kalman 滤波抖动降低演示")
    lines.append("=" * 60)
    lines.append("真实轨迹=ground truth, 噪声测量=逐帧FoundationPose, 平滑=本方案")
    lines.append(f"{'':15} {'真实轨迹':>10} {'噪声测量':>10} {'Kalman平滑':>12}")
    labels = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']
    for i, lab in enumerate(labels):
        lines.append(f"{lab:15} {jit_true[i]:10.5f} {jit_noisy[i]:10.5f} {jit_smooth[i]:12.5f}")
    lines.append("-" * 60)
    rms_noisy = np.linalg.norm(jit_noisy)
    rms_smooth = np.linalg.norm(jit_smooth)
    rms_true = np.linalg.norm(jit_true)
    lines.append(f"抖动RMS:  真实={rms_true:.5f}  噪声={rms_noisy:.5f}  "
                 f"平滑={rms_smooth:.5f}  降低={100*(1-rms_smooth/rms_noisy):.1f}%")
    lines.append(f"重定位次数: {kf.reloc_count}")
    lines.append("=" * 60)
    out = "\n".join(lines)
    print(out)
    # 同时写入文件
    with open(r'E:\Free_coordinate_MR_Registration\demo_output.txt', 'w',
              encoding='utf-8') as f:
        f.write(out)


if __name__ == "__main__":
    demo_jitter_reduction()
