#!/usr/bin/env python3
"""CPU alignment of cached Pi3X windows with fixed metric scales.

Each observation is world-from-camera in a window-local coordinate frame.
Only window SE(3) transforms are optimized. Frame 0 here means window index 0,
not a capture frame ID. No reference poses enter this module.
Requires NumPy and SciPy; does not import PyTorch or the Pi3X model.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.sparse import lil_matrix

from pi3_join import fit_join


def prepare(frames, pred, scales):
    """Validate and metric-scale a rectangular cache without mutating it."""
    frames, pred, scales = np.asarray(frames), np.asarray(pred), np.asarray(scales)
    if frames.ndim != 2 or not all(frames.shape):
        raise ValueError("frames must be a nonempty windows x observations array")
    if not np.issubdtype(frames.dtype, np.integer):
        raise ValueError("frame IDs must be integers")
    if any(len(set(row)) != len(row) for row in frames):
        raise ValueError("a window must not repeat a frame ID")
    if pred.shape != (*frames.shape, 4, 4):
        raise ValueError("pred must have shape windows x observations x 4 x 4")
    if not np.isfinite(pred).all():
        raise ValueError("poses must be finite")
    if scales.shape != (len(frames),) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("one finite positive metric scale per window is required")
    if not np.allclose(pred[..., 3, :], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("poses must be homogeneous transforms")
    rot = pred[..., :3, :3]
    if (np.max(abs(rot @ rot.swapaxes(-1, -2)-np.eye(3))) > 1e-3
            or np.any(abs(np.linalg.det(rot)-1) > 1e-3)):
        raise ValueError("pose rotations must be proper orthogonal matrices")
    poses = pred.astype(np.float64).copy()
    # Project float32 round-off to SO(3), after rejecting malformed rotations.
    poses[..., :3, :3] = Rotation.from_matrix(rot.reshape(-1, 3, 3)).as_matrix().reshape(rot.shape)
    poses[..., :3, 3] *= scales[:, None, None]
    return frames.copy(), poses


def observations(frames):
    by_frame = {}
    for w, row in enumerate(frames):
        for k, f in enumerate(row):
            by_frame.setdefault(int(f), []).append((w, k))
    return by_frame


def overlap_pairs(frames):
    """All pairs sharing an ID; total weight of each shared frame is one."""
    rows, weights = [], []
    graph = [set() for _ in frames]
    for obs in observations(frames).values():
        pairs = list(combinations(obs, 2))
        for (a, i), (b, j) in pairs:
            rows.append((a, i, b, j))
            weights.append(1.0/len(pairs))
            graph[a].add(b)
            graph[b].add(a)
    seen, pending = {0}, [0]
    while pending:
        for other in graph[pending.pop()]:
            if other not in seen:
                seen.add(other)
                pending.append(other)
    if len(seen) != len(frames):
        raise ValueError("window overlap graph is disconnected")
    return np.asarray(rows, dtype=int).reshape(-1, 4), np.asarray(weights)


def mean_rotation(matrices):
    return Rotation.from_matrix(np.asarray(matrices)).mean().as_matrix()


def transform_poses(transform, poses):
    return transform @ poses


def sequential(frames, poses, *, rotation=False):
    """First-estimate-wins chain, positional depth join or orientation join.

    The orientation arm can initialize a connected graph even when window
    order is not topological. The position-only arm retains fit_join's refusal
    when translation does not provide sufficient scale/rotation information.
    """
    chain = dict(zip(frames[0].tolist(), poses[0].copy()))
    transforms = np.repeat(np.eye(4)[None], len(frames), axis=0)
    pending = set(range(1, len(frames)))
    while pending:
        progress = False
        for w in sorted(pending):
            shared = [(k, int(f)) for k, f in enumerate(frames[w]) if int(f) in chain]
            if not shared:
                continue
            src = np.array([poses[w, k, :3, 3] for k, _ in shared])
            dst = np.array([chain[f][:3, 3] for _, f in shared])
            if rotation:
                R = mean_rotation([chain[f][:3, :3] @ poses[w, k, :3, :3].T
                                   for k, f in shared])
                t = dst.mean(0)-R@src.mean(0)
            else:
                R, _, t, _, _ = fit_join(src, dst, "depth")
            transforms[w, :3, :3], transforms[w, :3, 3] = R, t
            for f, T in zip(frames[w], transform_poses(transforms[w], poses[w])):
                chain.setdefault(int(f), T)
            pending.remove(w)
            progress = True
        if not progress:
            raise ValueError("window overlap graph is disconnected")
    ids = np.array(sorted(chain), dtype=np.int64)
    return ids, np.array([chain[int(f)] for f in ids]), transforms


def merge_estimates(frames, poses, transforms):
    """Use every window estimate for each output frame, with equal weights."""
    transformed = transforms[:, None] @ poses
    by_frame = observations(frames)
    ids = np.array(sorted(by_frame), dtype=np.int64)
    result = np.repeat(np.eye(4)[None], len(ids), axis=0)
    for n, f in enumerate(ids):
        values = np.array([transformed[w, k] for w, k in by_frame[int(f)]])
        result[n, :3, 3] = values[:, :3, 3].mean(0)
        result[n, :3, :3] = mean_rotation(values[:, :3, :3])
    return ids, result


def robust_blocks(residual):
    """Embed block soft-L1 so ||output||^2 = 2*(sqrt(1+||r||^2)-1)."""
    squared = np.sum(residual**2, axis=-1, keepdims=True)
    return residual*np.sqrt(2.0/(np.sqrt(1.0+squared)+1.0))


def align_windows(frames, poses, *, position_sigma=0.05,
                  rotation_sigma_deg=5.0, robust=True, max_nfev=300):
    """Optimize all window transforms with the first gauge eliminated.

    ``poses`` must be validated, metric poses from prepare(). Scales remain
    fixed. Returns frame IDs, fused poses, window transforms, solver report.
    Sigma parameters are modelling choices, not calibrated covariances.
    """
    if not np.isfinite([position_sigma, rotation_sigma_deg]).all() or min(position_sigma, rotation_sigma_deg) <= 0:
        raise ValueError("residual scales must be finite and positive")
    if max_nfev < 1:
        raise ValueError("max_nfev must be positive")
    pairs, weight = overlap_pairs(frames)
    _, _, initial = sequential(frames, poses, rotation=True)
    if len(frames) == 1:
        ids, estimate = merge_estimates(frames, poses, initial)
        return ids, estimate, initial, {"success": True, "nfev": 0, "initial_cost": 0., "cost": 0.}
    a, i, b, j = pairs.T
    Q = poses[..., :3, :3]
    P = poses[..., :3, 3]
    sigma_r = np.radians(rotation_sigma_deg)

    def unpack(x):
        values = x.reshape(-1, 6)
        transforms = initial.copy()
        transforms[1:, :3, :3] = Rotation.from_rotvec(values[:, :3]).as_matrix() @ initial[1:, :3, :3]
        transforms[1:, :3, 3] += values[:, 3:]
        return transforms

    def residual(x):
        T = unpack(x)
        ra, rb = T[a, :3, :3], T[b, :3, :3]
        pa = np.einsum('nij,nj->ni', ra, P[a, i])+T[a, :3, 3]
        pb = np.einsum('nij,nj->ni', rb, P[b, j])+T[b, :3, 3]
        dp = (pa-pb)/position_sigma
        qa, qb = ra@Q[a, i], rb@Q[b, j]
        dr = Rotation.from_matrix(qa@qb.transpose(0, 2, 1)).as_rotvec()/sigma_r
        if robust:
            dp, dr = robust_blocks(dp), robust_blocks(dr)
        return (np.concatenate([dp, dr], axis=1)*np.sqrt(weight[:, None])).ravel()

    sparsity = lil_matrix((len(pairs)*6, (len(frames)-1)*6), dtype=int)
    for k, (wa, _, wb, _) in enumerate(pairs):
        for w in (wa, wb):
            if w:
                sparsity[6*k:6*k+6, 6*(w-1):6*w] = 1
    x0 = np.zeros((len(frames)-1)*6)
    before = residual(x0)
    solution = least_squares(residual, x0, jac_sparsity=sparsity.tocsr(),
                             max_nfev=max_nfev, ftol=1e-8, xtol=1e-8, gtol=1e-8,
                             tr_options={"atol": 1e-10, "btol": 1e-10})
    transforms = unpack(solution.x)
    ids, estimate = merge_estimates(frames, poses, transforms)
    report = {"success": bool(solution.success), "status": int(solution.status),
              "message": str(solution.message), "nfev": int(solution.nfev),
              "initial_cost": float(0.5*before@before), "cost": float(solution.cost),
              "optimality": float(solution.optimality), "overlap_pairs": len(pairs),
              "position_sigma_m": position_sigma, "rotation_sigma_deg": rotation_sigma_deg,
              "robust": robust, "metric_scales_optimized": False}
    return ids, estimate, transforms, report
