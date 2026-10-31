# -*- coding: utf-8 -*-
"""Ego-motion-free registration vs filtering the camera pose directly.

A fixed model is observed by a moving camera. Per-frame PnP is delayed by
tau. Three render poses are scored at photon time:

  delayed     use the exposure-time model pose (the swimming overlay)
  coupled     extrapolate the mixed model-to-camera pose by tau
  ego-free    extrapolate only the camera, hold the world registration

Run from the repository root:

    python scripts/run_egomotion_free_reg.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.egomotion_free_registration import (  # noqa: E402
    EgoMotionFreeRegistration,
    predict_ego_motion,
)
from src.geometry import make_T, project_pose
from src.pose_estimation import pnp_ransac
from src.se3_kalman_filter import compose_se3, exp_se3, inverse_se3, log_se3


def _look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Camera-to-world pose. OpenCV camera looks along +Z."""
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    up = np.array([0.0, -1.0, 0.0])
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    R_wc = np.stack([right, down, forward], axis=1)
    return make_T(R_wc, eye)


def _model_cloud(rng: np.random.Generator) -> np.ndarray:
    """40 mm navigation target, in millimetres, model frame = world."""
    corners = np.array([
        [-20, -15, -10], [20, -15, -10], [-20, 15, -10], [20, 15, -10],
        [-20, -15, 10], [20, -15, 10], [-20, 15, 10], [20, 15, 10],
    ], dtype=np.float64)
    surface = rng.uniform(-20, 20, size=(32, 3))
    surface[:, 2] = rng.choice([-10, 10], size=32)
    return np.vstack([corners, surface])


def _room_cloud() -> np.ndarray:
    """Static background around the target, used only to observe ego-motion."""
    xs = np.linspace(-160, 160, 5)
    ys = np.linspace(-120, 120, 4)
    pts = [[x, y, z] for z in (-30.0, 30.0) for x in xs for y in ys]
    return np.asarray(pts, dtype=np.float64)


def _trajectory(n: int, dt: float, radius: float = 180.0):
    """Orbit plus one fast yaw burst. Returns T_cw and |omega| at each frame."""
    T_cw = []
    omega = []
    target = np.zeros(3)
    prev = None
    for i in range(n):
        t = i * dt
        yaw = 1.1 * t
        if 0.8 <= t <= 2.0:
            yaw += 3.5 * (t - 0.8)
        eye = np.array([
            radius * np.sin(yaw),
            40.0 * np.sin(0.7 * t),
            420.0 - 40.0 * (1 - np.cos(yaw)),
        ])
        # project_pose expects world/model -> camera, not camera -> world.
        T = inverse_se3(_look_at(eye, target))
        if prev is None:
            omega.append(0.0)
        else:
            xi = log_se3(compose_se3(inverse_se3(prev), T))
            omega.append(float(np.linalg.norm(xi[3:]) / dt))
        prev = T
        T_cw.append(T)
    return T_cw, np.asarray(omega)


def _pnp(obj: np.ndarray, T_cm: np.ndarray, K: np.ndarray,
         rng: np.random.Generator, sigma_px: float):
    uv, z = project_pose(K, T_cm, obj)
    keep = z > 1.0
    if keep.sum() < 6:
        return None
    noisy = uv[keep] + rng.normal(0.0, sigma_px, size=(int(keep.sum()), 2))
    T, stats = pnp_ransac(obj[keep], noisy, K, iterations=200, reproj_thresh=4.0)
    return T, stats


def _add_mm(model: np.ndarray, T_est: np.ndarray, T_gt: np.ndarray) -> float:
    err = (T_est[:3, :3] @ model.T).T + T_est[:3, 3] - (
        (T_gt[:3, :3] @ model.T).T + T_gt[:3, 3]
    )
    return float(np.linalg.norm(err, axis=1).mean())


def _reproj_px(model, T_est, T_gt, K) -> float:
    uv_hat, z_hat = project_pose(K, T_est, model)
    uv_gt, z_gt = project_pose(K, T_gt, model)
    ok = (z_hat > 1.0) & (z_gt > 1.0)
    if ok.sum() == 0:
        return float("nan")
    return float(np.linalg.norm(uv_hat[ok] - uv_gt[ok], axis=1).mean())


def _bin_summary(omega, values, edges):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (omega >= lo) & (omega < hi)
        if m.sum() == 0:
            continue
        out.append({
            "omega_rad_s": f"[{lo:.1f},{hi:.1f})",
            "n": int(m.sum()),
            "add_mm": round(float(np.mean(values[m])), 3),
        })
    return out


def run(n: int = 90, dt: float = 1.0 / 30.0, tau: float = 0.060,
        sigma_px: float = 1.2, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    K = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]])
    model = _model_cloud(rng)
    room = _room_cloud()
    T_wm = np.eye(4)
    T_wm[:3, 3] = np.array([0.0, 0.0, 40.0])
    T_cw, omega = _trajectory(n, dt)
    freereg = EgoMotionFreeRegistration(alpha=0.35)

    rows = []
    for i in range(1, n):
        T_cw_exp = T_cw[i]
        T_cm_exp = compose_se3(T_cw_exp, T_wm)
        T_cw_ph = predict_ego_motion(T_cw[i - 1], T_cw_exp, dt, tau)
        # Photon-time truth uses the same constant-twist model on the true
        # camera, so the score isolates registration rather than trajectory
        # interpolation.
        T_cm_ph = compose_se3(T_cw_ph, T_wm)

        bg = _pnp(room, T_cw_exp, K, rng, sigma_px)
        md = _pnp(model, T_cm_exp, K, rng, sigma_px)
        if bg is None or md is None or bg[0] is None or md[0] is None:
            continue
        T_cw_meas, T_cm_meas = bg[0], md[0]

        delayed = T_cm_meas
        # Coupled baseline extrapolates the mixed model-to-camera pose.
        if rows:
            coupled = predict_ego_motion(rows[-1]["T_cm_meas"], T_cm_meas, dt, tau)
        else:
            coupled = T_cm_meas
        freed = freereg.update(T_cm_meas, T_cw_meas, dt, tau)["T_cm_render"]

        rows.append({
            "T_cm_meas": T_cm_meas,
            "omega": float(omega[i]),
            "delayed_mm": _add_mm(model, delayed, T_cm_ph),
            "coupled_mm": _add_mm(model, coupled, T_cm_ph),
            "egofree_mm": _add_mm(model, freed, T_cm_ph),
            "delayed_px": _reproj_px(model, delayed, T_cm_ph, K),
            "coupled_px": _reproj_px(model, coupled, T_cm_ph, K),
            "egofree_px": _reproj_px(model, freed, T_cm_ph, K),
        })

    def col(name):
        return np.asarray([r[name] for r in rows], dtype=np.float64)

    w = col("omega")
    edges = [0.0, 0.8, 1.6, 10.0]
    summary = {
        "n_frames": len(rows),
        "dt_s": dt,
        "tau_s": tau,
        "pixel_noise_sigma": sigma_px,
        "mean_add_mm": {
            "delayed": round(float(col("delayed_mm").mean()), 3),
            "coupled_extrapolation": round(float(col("coupled_mm").mean()), 3),
            "egomotion_free": round(float(col("egofree_mm").mean()), 3),
        },
        "mean_reproj_px": {
            "delayed": round(float(np.nanmean(col("delayed_px"))), 3),
            "coupled_extrapolation": round(float(np.nanmean(col("coupled_px"))), 3),
            "egomotion_free": round(float(np.nanmean(col("egofree_px"))), 3),
        },
        "add_mm_by_yaw_rate": {
            "delayed": _bin_summary(w, col("delayed_mm"), edges),
            "coupled_extrapolation": _bin_summary(w, col("coupled_mm"), edges),
            "egomotion_free": _bin_summary(w, col("egofree_mm"), edges),
        },
    }
    return summary


def main() -> None:
    summary = run()
    out_dirs = [
        ROOT / "outputs" / "egomotion_free",
        Path(r"D:\ISMAR2026\egomotion-registration\outputs"),
    ]
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        path = d / "egomotion_free_registration.json"
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
