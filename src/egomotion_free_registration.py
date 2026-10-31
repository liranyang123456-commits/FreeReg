# -*- coding: utf-8 -*-
"""Ego-motion-free 3D-model registration for AR/MR navigation.

The measured model-to-camera pose mixes two motions:

    T_cm(t) = T_cw(t) @ T_wm

T_cw is the viewer/endoscope ego-motion. T_wm is the model-to-world
registration that navigation actually needs. Filtering T_cm directly
treats a head turn as a change of the model. This module keeps T_wm in
the world and predicts only T_cw forward by the display latency.

Render pose at photon time t+tau:

    T_cm(t+tau) = T_cw(t) @ Exp(xi_ego * tau / dt) @ T_wm
"""
from __future__ import annotations

import numpy as np

from src.se3_kalman_filter import compose_se3, exp_se3, inverse_se3, log_se3


def _as_T(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"expected 4x4 pose, got {T.shape}")
    return T


def world_registration(T_cm: np.ndarray, T_cw: np.ndarray) -> np.ndarray:
    """Remove ego-motion: T_wm = T_cw^{-1} @ T_cm."""
    return compose_se3(inverse_se3(_as_T(T_cw)), _as_T(T_cm))


def camera_pose_from_registration(T_cw: np.ndarray, T_wm: np.ndarray) -> np.ndarray:
    """Put the model back in the camera: T_cm = T_cw @ T_wm."""
    return compose_se3(_as_T(T_cw), _as_T(T_wm))


def predict_ego_motion(T_cw_prev: np.ndarray, T_cw: np.ndarray,
                       dt: float, tau: float) -> np.ndarray:
    """Constant-twist prediction of the camera pose at t+tau.

    xi_ego is the body twist from the previous camera pose to the current
    one. Only the viewer is extrapolated. The model registration is not.
    """
    if dt <= 0:
        raise ValueError("dt must be positive")
    xi = log_se3(compose_se3(inverse_se3(_as_T(T_cw_prev)), _as_T(T_cw)))
    return compose_se3(_as_T(T_cw), exp_se3(xi * (float(tau) / float(dt))))


class EgoMotionFreeRegistration:
    """Hold the model fixed in the world. Predict only the viewer.

    Args:
        alpha: SE(3) step toward each new world-registration measurement.
            1 keeps the raw measurement. Smaller values reject per-frame
            PnP noise without following ego-motion.
    """

    def __init__(self, alpha: float = 0.25):
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = float(alpha)
        self.T_wm = None
        self.T_cw_prev = None

    def reset(self) -> None:
        self.T_wm = None
        self.T_cw_prev = None

    def update(self, T_cm: np.ndarray, T_cw: np.ndarray,
               dt: float, tau: float) -> dict:
        """One frame.

        Returns the world registration and the camera pose that should be
        used to draw the model at photon time.
        """
        T_cw = _as_T(T_cw)
        T_wm_meas = world_registration(T_cm, T_cw)
        if self.T_wm is None:
            self.T_wm = T_wm_meas.copy()
        else:
            innov = log_se3(compose_se3(inverse_se3(self.T_wm), T_wm_meas))
            self.T_wm = compose_se3(self.T_wm, exp_se3(self.alpha * innov))

        if self.T_cw_prev is None or tau <= 0:
            T_cw_photon = T_cw.copy()
        else:
            T_cw_photon = predict_ego_motion(self.T_cw_prev, T_cw, dt, tau)
        self.T_cw_prev = T_cw.copy()

        T_render = camera_pose_from_registration(T_cw_photon, self.T_wm)
        return {
            "T_wm": self.T_wm.copy(),
            "T_cw_photon": T_cw_photon,
            "T_cm_render": T_render,
        }
