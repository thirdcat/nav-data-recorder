#!/usr/bin/env python3
"""Measurement-only translation information conditional on fixed camera rotations.

Landmarks are Schur-eliminated. Pose priors, IMU information and cheirality
penalties are deliberately absent. This is local model sensitivity, not a
calibrated confidence bound: rotation, calibration and correlated errors remain.
"""
import numpy as np

from local_ba_diagnostics import connectivity


def measurement_jacobian(point, pose, K):
    """Derivative of (u, v, z) with respect to a world landmark; camera is its negative."""
    camera = pose[:3, :3].T@(point-pose[:3, 3])
    x, y, z = camera
    if not np.isfinite(camera).all() or z <= .05:
        raise ValueError('point is not in front of camera')
    jac = np.array([[K[0, 0]/z, 0, -K[0, 0]*x/z**2],
                    [0, K[1, 1]/z, -K[1, 1]*y/z**2], [0, 0, 1.]])@pose[:3, :3].T
    value = np.array([K[0, 0]*x/z+K[0, 2], K[1, 1]*y/z+K[1, 2], z])
    return value, jac


def schur_information(poses, points, K, track, frame, uv, depth, heldout, *, use_depth=True):
    """Return the 3N by 3N PSD camera information before fixing translation gauges.

Points follow sorted training track IDs, as in local_rgbd_ba.solve(). Robust
IRLS weights correspond to that solver's block RGB and scalar depth soft-L1.
"""
    n = len(poses)
    train_ids = np.unique(track[~heldout])
    if points.shape != (len(train_ids), 3):
        raise ValueError('landmarks must follow sorted training track IDs')
    if any(np.unique(heldout[track == t]).size != 1 for t in np.unique(track)):
        raise ValueError('split entire tracks')
    H = np.zeros((3*n, 3*n))
    valid_observations = 0
    used = np.zeros(len(track), bool)
    for i, tid in enumerate(train_ids):
        blocks = np.zeros((n, 3, 3))
        for obs in np.flatnonzero((track == tid) & ~heldout):
            f = frame[obs]
            try:
                predicted, J = measurement_jacobian(points[i], poses[f], K[f])
            except ValueError:
                continue
            rgb_residual = (predicted[:2]-uv[obs])/1.5
            rgb_weight = 1./np.sqrt(1.+rgb_residual@rgb_residual)
            W = J[:2].T@J[:2]*rgb_weight/1.5**2
            if use_depth and np.isfinite(depth[obs]) and depth[obs] > .05:
                sigma = .03+.01*depth[obs]
                dz = (predicted[2]-depth[obs])/sigma
                W += np.outer(J[2], J[2])/(sigma**2*np.sqrt(1+dz**2))
            blocks[f] += W
            valid_observations += 1
            used[obs] = True
        landmark = blocks.sum(axis=0)
        cross = blocks.reshape(3*n, 3)
        H -= cross@np.linalg.pinv(landmark, rcond=1e-12)@cross.T
        for f in range(n):
            H[3*f:3*f+3, 3*f:3*f+3] += blocks[f]
    graph = connectivity(n, track, frame, ~used)
    return (H+H.T)/2, graph, valid_observations


def summarize_information(H, graph):
    n = len(H)//3
    eigenvalues = np.linalg.eigvalsh(H)
    tolerance = max(float(eigenvalues[-1])*1e-9, 1e-9)
    if eigenvalues[0] < -tolerance:
        raise ValueError('Schur information is not positive semidefinite')
    anchors = [min(c) for c in graph['components']]
    free = np.array([k for f in range(n) if f not in anchors for k in range(3*f, 3*f+3)])
    std = np.zeros(n)
    condition = None
    max_std = None
    if len(free):
        reduced = H[np.ix_(free, free)]
        vals, vectors = np.linalg.eigh(reduced)
        rank = int((vals > tolerance).sum())
        if rank == len(free):
            covariance = (vectors/vals)@vectors.T
            free_std = np.sqrt(np.maximum(np.diag(covariance), 0).reshape(-1, 3).sum(axis=1))
            std[[f for f in range(n) if f not in anchors]] = free_std
            max_std = float(std.max())
            condition = float(vals[-1]/vals[0])
        else:
            std[:] = np.nan
    else:
        rank = 0
        std[:] = np.nan
    return {'graph': graph, 'fixed_translation_anchors': anchors,
            'nullity_before_anchors': int((eigenvalues <= tolerance).sum()),
            'expected_component_translation_gauges': 3*len(anchors),
            'rank_after_anchors': rank, 'dimensions_after_anchors': len(free),
            'condition_after_anchors': condition, 'max_conditional_camera_std_m': max_std,
            'conditional_camera_std_m': [float(x) if np.isfinite(x) else None for x in std],
            'eigenvalues_before_anchors': eigenvalues.tolist(),
            'interpretation': 'local IRLS information, fixed rotations, one fixed camera per component; not calibrated uncertainty'}
