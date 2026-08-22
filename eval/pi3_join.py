"""Scale policies for joining metric Pi3X windows.

Pi3X's per-window depth scale puts each prediction in metres.  A free
Umeyama scale at every seam can then throw that metric information away and
compound a small bias over the whole walk.  This module keeps the policy
separate from the model runner so it can be tested without a GPU.
"""
from __future__ import annotations

import numpy as np

from umeyama import umeyama


JOIN_SCALE_MODES = ("free", "depth", "median")


def window_scales(scales, mode: str) -> np.ndarray:
    """Return the metric scale to apply to each raw window prediction.

    ``free`` and ``depth`` use each window's own LiDAR-derived scale.
    ``median`` deliberately replaces those values with one robust session
    statistic.  It does not mean taking a median of the join scales: those
    are seam fit parameters, not independent metric measurements, and
    multiplying one at every seam would still accumulate.
    """
    values = np.asarray(scales, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("at least one window scale is required")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("window scales must be finite and positive")
    if mode not in JOIN_SCALE_MODES:
        raise ValueError(f"unknown join scale mode: {mode}")
    if mode == "median":
        return np.full(values.shape, np.median(values), dtype=np.float64)
    return values.copy()


def fit_join(source, target, mode: str):
    """Fit one window join.

    Returns ``(rotation, applied_scale, translation, fitted_scale,
    median_residual)``.  The fitted scale is retained as a diagnostic in the
    rigid modes, while only ``applied_scale`` is used to build the trajectory.
    """
    if mode not in JOIN_SCALE_MODES:
        raise ValueError(f"unknown join scale mode: {mode}")
    rotation, fitted_scale, free_translation = umeyama(source, target)
    if mode == "free":
        applied_scale = fitted_scale
        translation = free_translation
    else:
        # Rotation from Umeyama is scale-independent.  Recompute only the
        # translation for the rigid transform, so the metric scale already
        # applied to the window is not discarded at the seam.
        applied_scale = 1.0
        source = np.asarray(source, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        translation = target.mean(axis=0) - rotation @ source.mean(axis=0)
    residual = float(np.median(
        np.linalg.norm(applied_scale * (source @ rotation.T)
                       + translation - target, axis=1)))
    return rotation, applied_scale, translation, fitted_scale, residual
