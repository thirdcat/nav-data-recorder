#!/usr/bin/env python3
"""Estimate pose from LiDAR depth alone, and score it against ARKit's.

The ultra-wide path needs a pose source that is not ARKit, and depth odometry is
the obvious candidate: it is metric by construction, so the scale problem that
dogs monocular SfM does not arise, and it does not care whether a wall has any
texture on it.

Whether it is *good enough* is a measurement, not an argument — and it can be
made right now, because every recorded session already carries LiDAR depth and
ARKit's own trajectory for the same frames. This runs point-to-plane ICP over
the depth maps and reports how far the result drifts from ARKit.

    python3 tools/depth_odometry.py ~/nav_data/20260807-140619-477750

What it cannot tell you: ARKit is a reference, not ground truth — it drifts
about 0.02 m/s itself. Agreement means the two make the same journey, not that
either is right. The point-to-plane residual is geometric consistency, not
accuracy. Disagreement is still decisive in the direction that matters, because
a depth-only track that cannot match a fused one will not beat it.

Requires numpy.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from read_session import Session  # noqa: E402


# ------------------------------------------------------------------ geometry

FLOOR_NORMAL_ANGLE_DEG = 15.0
FLOOR_NORMAL_COS = math.cos(math.radians(FLOOR_NORMAL_ANGLE_DEG))
FLOOR_MIN_POINTS = 100
# A twentieth-step is deliberately soft: the floor estimate removes a
# random-walk component without pretending that every floor is perfectly level
# or that a person never changes height. This is a stabiliser, not a hard pose
# constraint, and the small gain avoids re-inserting measurement jitter into
# the map.
FLOOR_LOCK_GAIN = 0.05
GRAVITY_LOCK_GAIN = 1.0
GRAVITY_ANCHOR_LAMBDA = 0.02
PHOTOMETRIC_WIDTH = 480
PHOTOMETRIC_HEIGHT = 360
PHOTOMETRIC_MIN_MATCHES = 100
# Match radii the probe reports, in metres. The first must be the default
# `--max-dist` so the probe reuses the association the solve already made.
#
# These tighten rather than widen, and that is forced by the data structure.
# `_voxel_matches` searches a point's own voxel and the 26 adjacent ones, so at
# a 0.03 m voxel no candidate it can return is further than about 0.10 m —
# `max_dist` at 0.15 m never rejects anything, which is *why* that parameter
# measured as inert. Widening it changes nothing; widening the stencil instead
# costs (2r+1)^3 lookups per point and is prohibitive past a voxel or two.
MATCH_RADIUS_PROBES = (0.15, 0.08, 0.04, 0.02)
# Fixed block scales, deliberately independent of the current residuals. The
# depth value is the measured order of LiDAR noise at room range; ten 8-bit
# levels is a conservative still/JPEG gradient-noise scale. They only convert
# metres and gray values to dimensionless normalised blocks. Huber still rejects
# individual outliers, but neither block's influence is tuned by its observed
# residual or by the trajectory being scored.
PHOTOMETRIC_DEPTH_SCALE = 0.01
PHOTOMETRIC_IMAGE_SCALE = 10.0
PHOTOMETRIC_HUBER_DELTA = 1.345
PHOTOMETRIC_OUTLIER_SIGMA = 5.0
PHOTOMETRIC_LM_INITIAL = 1e-2
PHOTOMETRIC_LM_MAX = 1e6
# Both block weights default to one because each block is divided by its fixed
# physical/noise scale and averaged over its own rows. A unit weight therefore
# means equal influence per normalised block, without making the observed
# residuals choose the result. The CLI exposes both weights for the ablation
# rather than hiding this choice in a calibration constant.

def backproject(depth: np.ndarray, fx: float, fy: float,
                cx: float, cy: float) -> np.ndarray:
    """Depth map to a 3-D point per pixel, in camera coordinates (X right, Y
    down, Z forward — the depth map's own convention, not ARKit's)."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float64),
                       np.arange(h, dtype=np.float64))
    z = depth.astype(np.float64)
    return np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=-1)


def _box_sum(a: np.ndarray, half: int) -> np.ndarray:
    """Sum over a square window, by integral image. Edges are clipped, not
    wrapped or padded, so a border pixel averages the neighbours it has."""
    h, w = a.shape
    ii = np.zeros((h + 1, w + 1))
    ii[1:, 1:] = a.cumsum(0).cumsum(1)
    y0, y1 = np.clip(np.arange(h) - half, 0, h), np.clip(np.arange(h) + half + 1, 0, h)
    x0, x1 = np.clip(np.arange(w) - half, 0, w), np.clip(np.arange(w) + half + 1, 0, w)
    return (ii[np.ix_(y1, x1)] - ii[np.ix_(y0, x1)]
            - ii[np.ix_(y1, x0)] + ii[np.ix_(y0, x0)])


def smooth_depth(depth: np.ndarray, valid: np.ndarray, half: int = 2,
                 edge: float = 0.05) -> np.ndarray:
    """Average each depth over its neighbours, but not across a discontinuity.

    Every real-time depth pipeline does this before anything else touches the
    map, and the reason is worth stating: normals come from differencing pixels
    two apart, which at two metres is a two-centimetre baseline. Per-pixel noise
    of about a centimetre therefore makes the *normal* nearly random, and
    point-to-plane ICP is built entirely on normals. Measured here, on a
    rendered walk with 0.5% depth noise, smoothing moved the drift over 2.2 m
    from 17 cm to half a centimetre — it is not a refinement, it is most of the
    result.

    Averaging across a depth edge would invent surface that is not there, so a
    pixel whose smoothed value moves more than `edge` keeps its own reading.
    """
    z = np.where(valid, depth, 0.0)
    total = _box_sum(z, half)
    count = _box_sum(valid.astype(np.float64), half)
    out = np.where(count > 0, total / np.maximum(count, 1e-9), depth)
    return np.where(np.abs(out - depth) > edge, depth, out)


def normals(points: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Surface normals from neighbouring pixels.

    Point-to-plane ICP converges far faster than point-to-point because it lets
    a point slide along the surface it landed on instead of pinning it to one
    correspondence — which matters here, where correspondences come from
    projection and are approximate by construction.
    """
    dx = np.zeros_like(points)
    dy = np.zeros_like(points)
    dx[:, 1:-1] = points[:, 2:] - points[:, :-2]
    dy[1:-1, :] = points[2:, :] - points[:-2, :]
    n = np.cross(dx, dy)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    ok = valid.copy()
    ok[:, :1] = ok[:, -1:] = False
    ok[:1, :] = ok[-1:, :] = False
    ok &= norm[..., 0] > 1e-9
    return np.divide(n, np.maximum(norm, 1e-12)), ok


def frame_points(depth: np.ndarray, K, smooth: bool = True,
                 near: float = 0.1, far: float = 5.0):
    """A depth map as points, normals and a validity mask — the one place that
    turns a frame into something ICP can consume, so the self-test and a real
    session go through identical preparation rather than similar-looking code."""
    valid = np.isfinite(depth) & (depth > near) & (depth < far)
    d = np.nan_to_num(np.asarray(depth, dtype=np.float64))
    if smooth:
        d = smooth_depth(d, valid)
    pts = backproject(d, *K)
    nrm, ok = normals(pts, valid)
    return pts, nrm, ok


def _unit_vector(value):
    """Return a finite unit 3-vector, or ``None`` for unusable input."""
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.size != 3 or not np.all(np.isfinite(vector)):
        return None
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else None


def _axis_angle_rotation(axis, angle):
    """Return the active rotation for ``angle`` radians about ``axis``."""
    axis = np.asarray(axis, dtype=np.float64)
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    x, y, z = axis
    skew = np.array([[0.0, -z, y],
                     [z, 0.0, -x],
                     [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def _minimal_rotation(source, target):
    """Return the shortest rotation that maps one unit vector onto another.

    The cross product fixes the correction axis in the plane of the two
    vectors. Consequently this correction contains no rotation about the
    vector being aligned — the unobservable yaw component is left alone.
    """
    source = _unit_vector(source)
    target = _unit_vector(target)
    if source is None or target is None:
        return np.eye(3), 0.0

    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    angle = math.atan2(sine, cosine)
    if sine > 1e-12:
        return _axis_angle_rotation(cross / sine, angle), angle
    if cosine >= 0.0:
        return np.eye(3), 0.0

    # Opposite vectors have infinitely many shortest axes. Pick the basis
    # vector least aligned with the source to make the fallback deterministic.
    basis = np.eye(3)[int(np.argmin(np.abs(source)))]
    axis = np.cross(source, basis)
    return _axis_angle_rotation(axis, math.pi), math.pi


def estimate_floor_height(points: np.ndarray, normals_: np.ndarray,
                          valid: np.ndarray, gravity: np.ndarray,
                          min_points: int = FLOOR_MIN_POINTS):
    """Estimate camera height above a floor, in the depth camera frame.

    ``gravity`` is a unit vector pointing down in the same (+Y down) frame as
    ``points``. A floor normal is parallel or anti-parallel to it, and a floor
    point has a positive projection along it. The 15-degree angular gate is a
    deliberately broad allowance for pixel normals and a slightly uneven
    surface; it is a modelling tolerance, not a measured property of a phone.
    The median rejects isolated horizontal furniture tops and returns the
    number of selected points so callers can report when the estimate was made
    from a thin sliver of floor.
    """
    g = np.asarray(gravity, dtype=np.float64).reshape(-1)
    if g.size != 3 or not np.all(np.isfinite(g)):
        return None, 0
    norm = float(np.linalg.norm(g))
    if norm <= 1e-12:
        return None, 0
    g = g / norm

    p = np.asarray(points, dtype=np.float64)
    n = np.asarray(normals_, dtype=np.float64)
    use = np.asarray(valid, dtype=bool).copy()
    use &= np.all(np.isfinite(p), axis=-1)
    use &= np.all(np.isfinite(n), axis=-1)
    normal_alignment = np.abs(np.einsum("...i,i->...", n, g))
    use &= normal_alignment >= FLOOR_NORMAL_COS
    height = np.einsum("...i,i->...", p, g)
    use &= height > 0.0
    selected = height[use]
    count = int(selected.size)
    if count < min_points:
        return None, count
    return float(np.median(selected)), count


def frame_conditioning(points: np.ndarray, normals_: np.ndarray,
                       valid: np.ndarray, stride: int = 4):
    """Conditioning supplied by one depth frame, without registration.

    This is deliberately separate from the Hessian assembled inside ``icp``:
    its rows come from the current frame only, rather than from projective or
    map correspondences.  The four-pixel grid is the real-time sampling used
    by the app; changing it changes the samples, but not the quantity being
    measured.  The weak eigenvector is sign-canonicalised because an
    eigenvector and its negative describe the same axis.
    """
    if stride < 1:
        raise ValueError("stride must be positive")
    sampled = np.zeros(valid.shape, dtype=bool)
    sampled[::stride, ::stride] = True
    use = valid & sampled
    p = points[use]
    n = normals_[use]
    if len(p) == 0:
        return 0.0, np.zeros(6, dtype=np.float64)

    A = np.hstack([np.cross(p, n), n])
    H = A.T @ A / len(A)
    try:
        ev, evec = np.linalg.eigh(H)
    except np.linalg.LinAlgError:
        return 0.0, np.zeros(6, dtype=np.float64)
    largest = float(ev[-1])
    if not np.isfinite(largest) or largest <= 1e-12:
        return 0.0, np.zeros(6, dtype=np.float64)

    cond = max(0.0, float(ev[0] / largest))
    weak = np.asarray(evec[:, 0], dtype=np.float64)
    pivot = int(np.argmax(np.abs(weak)))
    if weak[pivot] < 0:
        weak = -weak
    return cond, weak


def robust_weights(residual: np.ndarray, scale: float,
                   delta: float = PHOTOMETRIC_HUBER_DELTA,
                   outlier_sigma: float = PHOTOMETRIC_OUTLIER_SIGMA,
                   centre: float | None = None) -> np.ndarray:
    """Huber weights with a hard tail rejection for either residual term."""
    values = np.asarray(residual, dtype=np.float64)
    if centre is None:
        centre = float(np.median(values)) if len(values) else 0.0
    z = np.abs(values - centre) / max(float(scale), 1e-12)
    weights = np.ones(len(values), dtype=np.float64)
    huber = z > delta
    weights[huber] = delta / np.maximum(z[huber], 1e-12)
    weights[z > outlier_sigma] = 0.0
    return weights


def robust_cost(residual: np.ndarray, scale: float,
                centre: float = 0.0,
                delta: float = PHOTOMETRIC_HUBER_DELTA) -> float:
    """Huber objective used by the LM accept/reject test."""
    z = np.abs((np.asarray(residual, dtype=np.float64) - centre)
               / max(float(scale), 1e-12))
    cost = np.where(z <= delta, 0.5 * z * z,
                    delta * (z - 0.5 * delta))
    return float(np.mean(cost)) if len(cost) else math.inf


def _bilinear_sample(image: np.ndarray, u: np.ndarray,
                     v: np.ndarray) -> np.ndarray:
    """Sample a grayscale image at floating-point coordinates."""
    x0 = np.floor(u).astype(np.int64)
    y0 = np.floor(v).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = u - x0
    wy = v - y0
    return ((1.0 - wx) * (1.0 - wy) * image[y0, x0]
            + wx * (1.0 - wy) * image[y0, x1]
            + (1.0 - wx) * wy * image[y1, x0]
            + wx * wy * image[y1, x1])


def _project_image(q: np.ndarray, K):
    fx, fy, cx, cy = K
    z = q[:, 2]
    u = fx * q[:, 0] / z + cx
    v = fy * q[:, 1] / z + cy
    J = np.zeros((len(q), 2, 3), dtype=np.float64)
    z2 = z * z
    J[:, 0, 0] = fx / z
    J[:, 0, 2] = -fx * q[:, 0] / z2
    J[:, 1, 1] = fy / z
    J[:, 1, 2] = -fy * q[:, 1] / z2
    return u, v, J


def photometric_terms(points: np.ndarray, previous_image: np.ndarray,
                      current_image: np.ndarray, previous_K, current_K,
                      T: np.ndarray, previous_pose: np.ndarray,
                      base_pose: np.ndarray, mode: str):
    """Build image residuals and translation-only Jacobians for one solve.

    ``T`` is the transform being solved. In ``frame_to_frame`` mode it maps
    the previous depth camera into the current one, so current points use
    ``inv(T)`` to reach the previous image. In ``forward`` mode it maps current
    points into the previous depth/predicted frame. ``base_pose`` is the
    world-from-current-prediction pose in the map case, or the last depth pose
    in the frame-to-frame case. The same equation is used for an image that is
    several depth frames old; ``previous_pose`` keeps that exact image pose in
    the chain.

    The returned Jacobian has three translation columns when the caller fixes
    rotation and six columns otherwise. Rotation columns are intentionally
    zero: the measured complementarity is for the IMU-rotation 3-DoF problem.
    """
    if previous_image is None or current_image is None:
        return None
    p = np.asarray(points, dtype=np.float64)
    if len(p) < PHOTOMETRIC_MIN_MATCHES:
        return None
    if len(p) > 8000:
        p = p[np.random.default_rng(0).choice(len(p), 8000, replace=False)]

    T = np.asarray(T, dtype=np.float64)
    if mode == "frame_to_frame":
        # T maps previous-depth -> current. Current-depth -> previous-image is
        # inv(previous_image_pose) * base_pose * inv(T).
        R = T[:3, :3]
        q_depth = (p - T[:3, 3]) @ R
        dq_dt = -R.T
        bridge = np.linalg.inv(previous_pose) @ base_pose
    elif mode == "forward":
        # T maps current-depth -> base_pose (the predicted or previous pose).
        q_depth = p @ T[:3, :3].T + T[:3, 3]
        dq_dt = np.eye(3)
        bridge = np.linalg.inv(previous_pose) @ base_pose
    else:
        raise ValueError(f"unknown photometric transform mode: {mode}")

    q = q_depth @ bridge[:3, :3].T + bridge[:3, 3]
    u_prev, v_prev, J_proj = _project_image(q, previous_K)
    u_cur, v_cur, _ = _project_image(p, current_K)
    h, w = previous_image.shape
    hc, wc = current_image.shape
    good = ((q[:, 2] > 0.1)
            & (u_prev >= 1.0) & (u_prev < w - 2.0)
            & (v_prev >= 1.0) & (v_prev < h - 2.0)
            & (u_cur >= 1.0) & (u_cur < wc - 2.0)
            & (v_cur >= 1.0) & (v_cur < hc - 2.0))
    if good.sum() < PHOTOMETRIC_MIN_MATCHES:
        return None

    u_prev, v_prev = u_prev[good], v_prev[good]
    u_cur, v_cur = u_cur[good], v_cur[good]
    q = q[good]
    J_proj = J_proj[good]
    previous_image = np.asarray(previous_image, dtype=np.float64)
    current_image = np.asarray(current_image, dtype=np.float64)
    grad_v, grad_u = np.gradient(previous_image)
    sampled_previous = _bilinear_sample(previous_image, u_prev, v_prev)
    sampled_current = _bilinear_sample(current_image, u_cur, v_cur)
    gradient = np.stack([
        _bilinear_sample(grad_u, u_prev, v_prev),
        _bilinear_sample(grad_v, u_prev, v_prev),
    ], axis=1)

    dq_image_dt = bridge[:3, :3] @ dq_dt
    du_dt = np.einsum("nij,jk->nik", J_proj, dq_image_dt)
    J_translation = np.einsum("ni,nij->nj", gradient, du_dt)
    residual = sampled_previous - sampled_current
    finite = (np.all(np.isfinite(J_translation), axis=1)
              & np.isfinite(residual))
    if finite.sum() < PHOTOMETRIC_MIN_MATCHES:
        return None
    return J_translation[finite], residual[finite]


def _se3_log(T):
    """Small-motion twist of a transform, as (rotation, translation).

    Only ever applied to a frame-to-frame correction, which is degrees and
    centimetres, so the small-angle reading of the rotation is exact enough and
    avoids a matrix logarithm for no gain.
    """
    R = T[:3, :3]
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2.0
    return np.concatenate([w, T[:3, 3]])


def _se3_exp(x):
    w, t = x[:3], x[3:]
    dR = np.array([[1, -w[2], w[1]], [w[2], 1, -w[0]], [-w[1], w[0], 1]])
    u, _, vt = np.linalg.svd(dR)
    T = np.eye(4)
    T[:3, :3] = u @ vt
    T[:3, 3] = t
    return T


def anchor_to_prior(prior, solution, H, lam):
    """Pull the ICP solution back toward the prior, per direction, by evidence.

    This is the whole of the fusion. Damping a point-to-plane solve toward
    *zero* — the obvious reading of Tikhonov — asks the estimator to prefer "the
    camera did not move", and turning it up far enough to suppress the bad
    directions suppresses the good ones with it: measured on a real session, the
    error only reached the truth's magnitude at the point where the answer had
    become the identity. Damping toward the prior instead asks it to prefer
    "whatever ARKit said", which is a far better default and costs nothing on the
    axes depth actually observes.

    Each eigen-direction of the point-to-plane Hessian keeps the fraction
    `ev / (ev + lam)` of ICP's deviation from the prior. A direction the geometry
    pins hard moves freely; one it barely sees stays where ARKit put it. `lam` is
    in the Hessian's own units, scaled by its largest eigenvalue, so it means
    "how much better than the weakest useful direction the evidence has to be".
    """
    d = _se3_log(np.linalg.inv(prior) @ solution)
    ev, V = np.linalg.eigh(H)
    scale = lam * max(ev[-1], 1e-12)
    keep = ev / (ev + scale)
    return prior @ _se3_exp(V @ (keep * (V.T @ d)))


def _projective_matches(src_pts, src_ok, dst_pts, dst_normals, dst_ok, K, T,
                        max_dist):
    """Return the exact projective matches used by point-to-plane ICP.

    Keeping this association in one place matters: a residual computed with a
    different projection or distance gate would not be comparable to ICP's
    solve. The returned points are already in the target camera frame.
    """
    fx, fy, cx, cy = K
    h, w = dst_ok.shape
    p = src_pts[src_ok]
    if len(p) < 100:
        return None

    # Match icp()'s deterministic cap so residuals use the same population.
    if len(p) > 8000:
        p = p[np.random.default_rng(0).choice(len(p), 8000, replace=False)]

    q = p @ T[:3, :3].T + T[:3, 3]
    z = q[:, 2]
    ok = z > 0.1
    u = np.full(len(q), -1.0)
    v = np.full(len(q), -1.0)
    u[ok] = fx * q[ok, 0] / z[ok] + cx
    v[ok] = fy * q[ok, 1] / z[ok] + cy
    ui = np.round(u).astype(np.int64)
    vi = np.round(v).astype(np.int64)
    ok &= (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    if ok.sum() < 100:
        return None
    ui, vi = np.clip(ui, 0, w - 1), np.clip(vi, 0, h - 1)
    ok &= dst_ok[vi, ui]

    target = dst_pts[vi, ui]
    normal = dst_normals[vi, ui]
    diff = target - q
    ok &= np.linalg.norm(diff, axis=-1) < max_dist
    if ok.sum() < 100:
        return None
    return q[ok], target[ok], normal[ok], float(ok.mean())


_VOXEL_KEY_DTYPE = np.dtype([("x", "<i8"), ("y", "<i8"), ("z", "<i8")])
_VOXEL_NEIGHBOURS = np.array(
    [(dx, dy, dz) for dx in (-1, 0, 1)
     for dy in (-1, 0, 1) for dz in (-1, 0, 1)], dtype=np.int64)


def _voxel_matches(src_pts, src_ok, local_map, predicted, T, max_dist):
    """Find nearest fused map points in the predicted pose's voxel stencil.

    The source is transformed into the predicted camera frame by ``T`` and
    then into world coordinates. Each source point searches its voxel and the
    26 adjacent voxels; the nearest fused point in that stencil supplies both
    the target and its fused normal. Targets and normals are returned in the
    predicted camera frame, which is the frame ``T`` maps into.
    """
    p = src_pts[src_ok]
    if len(p) < 100 or not len(local_map):
        return None

    # Match icp()'s deterministic cap so the map and projective associations
    # use the same bounded population.
    if len(p) > 8000:
        p = p[np.random.default_rng(0).choice(len(p), 8000, replace=False)]

    q = p @ T[:3, :3].T + T[:3, 3]
    world = q @ predicted[:3, :3].T + predicted[:3, 3]
    map_points = local_map.points
    map_normals = local_map.normals
    key_records = np.ascontiguousarray(local_map.keys).view(
        _VOXEL_KEY_DTYPE).reshape(-1)
    base_keys = np.floor(world / local_map.voxel).astype(np.int64)

    query = base_keys[None, :, :] + _VOXEL_NEIGHBOURS[:, None, :]
    query_records = np.ascontiguousarray(query).view(
        _VOXEL_KEY_DTYPE).reshape(-1)
    index = np.searchsorted(key_records, query_records).reshape(
        len(_VOXEL_NEIGHBOURS), len(p))
    in_bounds = index < len(map_points)
    safe = np.minimum(index, len(map_points) - 1)
    same = np.all(local_map.keys[safe] == query, axis=2)
    valid = in_bounds & same
    delta = map_points[safe] - world[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", delta, delta)
    d2[~valid] = np.inf
    nearest_slot = np.argmin(d2, axis=0)
    columns = np.arange(len(p))
    nearest = safe[nearest_slot, columns]
    nearest_d2 = d2[nearest_slot, columns]
    matched = nearest_d2 < max_dist * max_dist
    if matched.sum() < 100:
        return None

    q = q[matched]
    indices = nearest[matched]
    target_world = map_points[indices]
    target = (target_world - predicted[:3, 3]) @ predicted[:3, :3]
    normal = map_normals[indices] @ predicted[:3, :3]
    return q, target, normal, float(matched.mean())


def _icp_with_photometric(src_pts, src_ok, dst_pts, dst_normals, dst_ok, K,
                          iters, max_dist, prior, rcond, anchor,
                          association, fixed_rotation, photometric,
                          depth_weight, photometric_weight,
                          return_diagnostics):
    """Multi-scale LM solve for the optional photometric block.

    The old depth-only branch below is kept byte-for-byte in spirit and is
    still used when this function is not entered. Here the image objective is
    solved coarse-to-fine. Each level uses the fixed image scale and Huber
    objective, and an LM step is accepted only when the same level's combined
    robust cost decreases. No block scale is estimated from the residuals.
    """
    if not np.isfinite(depth_weight) or depth_weight < 0.0:
        raise ValueError("depth_weight must be non-negative")
    if not np.isfinite(photometric_weight) or photometric_weight <= 0.0:
        raise ValueError("photometric_weight must be positive for photometric ICP")
    translation_only = fixed_rotation is not None
    if translation_only:
        fixed_rotation = np.asarray(fixed_rotation, dtype=np.float64)
        if fixed_rotation.shape != (3, 3) or not np.all(np.isfinite(fixed_rotation)):
            raise ValueError("fixed_rotation must be a finite 3x3 matrix")
        T = np.eye(4) if prior is None else prior.copy()
        T[:3, :3] = fixed_rotation
    else:
        T = np.eye(4) if prior is None else prior.copy()

    levels = tuple(getattr(photometric, "levels", (None,)))
    level_shapes = tuple(getattr(photometric, "level_shapes", ()))
    level_scales = tuple(getattr(photometric, "_scales", ()))
    diagnostics = {
        "photometric_used": False,
        "photometric_evaluations": 0,
        "photometric_accepted_steps": 0,
        "photometric_matches": 0,
        "depth_scale": None,
        "photometric_scale": None,
        "photometric_scales": [],
        "photometric_level_shapes": level_shapes,
        "photometric_level_scales": level_scales,
        "photometric_level_count": len(levels),
        "converged": False,
        "iterations": 0,
        "failure_reason": None,
    }
    inlier_frac = 0.0
    conditioning = 0.0
    last_H = None
    photo_seen = False

    def get_depth_matches(candidate):
        if depth_weight <= 0.0:
            return None
        if association is None:
            return _projective_matches(src_pts, src_ok, dst_pts,
                                       dst_normals, dst_ok, K, candidate,
                                       max_dist)
        return association(src_pts, src_ok, candidate, max_dist)

    def get_photo_matches(candidate):
        return photometric(candidate)

    def block_cost(depth_matches, photo_matches, depth_scale,
                   depth_centre, photo_scale, photo_centre):
        value = 0.0
        blocks = 0
        if depth_matches is not None and depth_weight > 0.0:
            qa, target, na, _ = depth_matches
            b_depth = np.einsum("ij,ij->i", target - qa, na)
            value += depth_weight * robust_cost(b_depth, depth_scale,
                                                depth_centre)
            blocks += 1
        if photo_matches is not None:
            value += photometric_weight * robust_cost(
                photo_matches[1], photo_scale, photo_centre)
            blocks += 1
        return value / max(blocks, 1)

    final_level = levels[-1] if levels else None
    for level in levels:
        if hasattr(photometric, "begin_level"):
            photometric.begin_level(level)
        diagnostics["failure_reason"] = None
        seed_photo = get_photo_matches(T)
        if seed_photo is None:
            continue
        photo_seen = True
        # Do not subtract the pair's median brightness offset: that would be
        # an implicit exposure correction and would make the objective depend
        # on the current residual population. The fixed scale is the only
        # cross-block normalisation.
        photo_centre = 0.0
        photo_scale = PHOTOMETRIC_IMAGE_SCALE
        diagnostics["photometric_scale"] = photo_scale
        diagnostics["photometric_scales"].append(photo_scale)
        level_converged = False
        lm = PHOTOMETRIC_LM_INITIAL

        for iteration in range(iters):
            diagnostics["iterations"] += 1
            depth_matches = get_depth_matches(T)
            photo_matches = get_photo_matches(T)
            if depth_matches is None and photo_matches is None:
                diagnostics["failure_reason"] = "no matches"
                break
            if photo_matches is not None:
                diagnostics["photometric_used"] = True
                diagnostics["photometric_evaluations"] += 1
                diagnostics["photometric_matches"] = len(photo_matches[1])
                inlier_frac = (1.0 if depth_matches is None
                               else depth_matches[3])

            dimension = 3 if translation_only else 6
            H = np.zeros((dimension, dimension), dtype=np.float64)
            g = np.zeros(dimension, dtype=np.float64)
            depth_scale = None
            depth_centre = 0.0

            if depth_matches is not None and depth_weight > 0.0:
                qa, target, na, inlier_frac = depth_matches
                A_depth = (na if translation_only
                           else np.hstack([np.cross(qa, na), na]))
                b_depth = np.einsum("ij,ij->i", target - qa, na)
                depth_scale = PHOTOMETRIC_DEPTH_SCALE
                diagnostics["depth_scale"] = depth_scale
                weights = robust_weights(b_depth, depth_scale,
                                         centre=depth_centre)
                A_scaled = A_depth / depth_scale
                b_scaled = b_depth / depth_scale
                H += depth_weight * (A_scaled.T
                                     @ (weights[:, None] * A_scaled)
                                     / len(A_scaled))
                g += depth_weight * (A_scaled.T
                                     @ (weights * b_scaled)
                                     / len(A_scaled))

            if photo_matches is not None:
                A_photo, b_photo = photo_matches
                weights = robust_weights(
                    b_photo, photo_scale, centre=photo_centre)
                if not translation_only:
                    A_photo = np.hstack([
                        np.zeros((len(A_photo), 3), dtype=np.float64),
                        A_photo,
                    ])
                A_scaled = A_photo / photo_scale
                b_scaled = b_photo / photo_scale
                H += photometric_weight * (A_scaled.T
                                           @ (weights[:, None] * A_scaled)
                                           / len(A_scaled))
                # photometric_terms returns the derivative of r_p itself.
                g -= photometric_weight * (A_scaled.T
                                           @ (weights * b_scaled)
                                           / len(A_scaled))

            if not np.all(np.isfinite(H)) or not np.all(np.isfinite(g)):
                diagnostics["failure_reason"] = "non-finite normal equation"
                break
            ev, evec = np.linalg.eigh(H)
            last_H = H
            conditioning = (max(0.0, float(ev[0] / ev[-1]))
                            if ev[-1] > 1e-12 else 0.0)
            current_cost = block_cost(
                depth_matches, photo_matches,
                depth_scale or PHOTOMETRIC_DEPTH_SCALE,
                depth_centre, photo_scale, photo_centre)

            accepted = False
            while not accepted:
                diagonal = np.maximum(np.diag(H), 1e-6)
                damped = H + lm * np.diag(diagonal)
                try:
                    damp_ev, damp_vec = np.linalg.eigh(damped)
                except np.linalg.LinAlgError:
                    diagnostics["failure_reason"] = "LM eigensolve"
                    break
                usable = damp_ev > max(damp_ev[-1] * rcond, 1e-12)
                if not usable.any():
                    diagnostics["failure_reason"] = "singular LM Hessian"
                    break
                V = damp_vec[:, usable]
                x = V @ ((V.T @ g) / damp_ev[usable])
                translation = x if translation_only else x[3:]
                if (not np.all(np.isfinite(x))
                        or np.linalg.norm(translation) > max_dist * 3):
                    candidate = None
                else:
                    candidate = _apply_icp_increment(
                        T, x, translation_only)
                if candidate is not None and not _is_valid_rigid_pose(candidate):
                    candidate = None
                if candidate is not None:
                    candidate_depth = get_depth_matches(candidate)
                    candidate_photo = get_photo_matches(candidate)
                    candidate_cost = block_cost(
                        candidate_depth, candidate_photo,
                        depth_scale or PHOTOMETRIC_DEPTH_SCALE,
                        depth_centre, photo_scale, photo_centre)
                else:
                    candidate_cost = math.inf
                candidate_has_photo = (candidate_photo is not None
                                       if candidate is not None else False)
                photo_required = photo_matches is not None
                if (np.isfinite(candidate_cost)
                        and candidate_cost <= current_cost
                        and (not photo_required or candidate_has_photo)):
                    accepted = True
                    T = candidate
                    lm = max(lm / 3.0, 1e-7)
                    diagnostics["photometric_used"] |= candidate_photo is not None
                    if candidate_photo is not None:
                        diagnostics["photometric_accepted_steps"] += 1
                        diagnostics["photometric_matches"] = len(candidate_photo[1])
                    if (np.linalg.norm(translation) < 2e-4
                            and (translation_only
                                 or np.linalg.norm(x[:3]) < 2e-4)):
                        level_converged = True
                else:
                    lm *= 10.0
                    if lm > PHOTOMETRIC_LM_MAX:
                        diagnostics["failure_reason"] = "LM rejected steps"
                        break
            if diagnostics["failure_reason"] in (
                    "LM eigensolve", "singular LM Hessian", "LM rejected steps"):
                break
            if level_converged:
                break

        if level == final_level:
            diagnostics["converged"] = level_converged

    if not photo_seen:
        # A failed image association must not silently turn a normal depth run
        # into the new damped solver. This preserves the depth-only baseline.
        return icp(src_pts, src_ok, dst_pts, dst_normals, dst_ok, K,
                   iters=iters, max_dist=max_dist, prior=prior, rcond=rcond,
                   anchor=anchor, association=association,
                   fixed_rotation=fixed_rotation, depth_weight=depth_weight,
                   photometric_weight=photometric_weight,
                   return_diagnostics=return_diagnostics)

    if (not translation_only and anchor is not None and prior is not None
            and last_H is not None):
        T = anchor_to_prior(prior, T, last_H, anchor)
    if return_diagnostics:
        return T, inlier_frac, conditioning, diagnostics
    return T, inlier_frac, conditioning


def icp(src_pts, src_ok, dst_pts, dst_normals, dst_ok, K, iters=20,
        max_dist=0.15, prior=None, rcond=1e-3, anchor=None,
        association=None, fixed_rotation=None, photometric=None,
        depth_weight=1.0, photometric_weight=1.0, return_diagnostics=False):
    """Point-to-plane ICP with a pluggable association.

    With no callback, correspondences come from projecting a transformed source
    point into the target's image and taking whatever pixel it lands on. That
    is the KinectFusion trick: it is O(1) per point with no spatial index, and
    on depth maps — which are already an image — it is what the data is shaped
    for. A callback can instead supply another correspondence set while the
    point-to-plane solve remains the same.

    Returns the 4x4 transform taking source camera coordinates into target
    camera coordinates, the fraction of points that found a match, and how well
    the geometry constrains the answer. With a free rotation this is the
    smallest eigenvalue of the 6x6 point-to-plane Hessian as a fraction of the
    largest; with ``fixed_rotation`` it is the corresponding quantity from the
    3x3 translation-only Hessian.
    """
    if photometric is not None and photometric_weight > 0.0:
        return _icp_with_photometric(
            src_pts, src_ok, dst_pts, dst_normals, dst_ok, K, iters,
            max_dist, prior, rcond, anchor, association, fixed_rotation,
            photometric, depth_weight, photometric_weight,
            return_diagnostics)
    if not np.isfinite(depth_weight) or depth_weight < 0.0:
        raise ValueError("depth_weight must be non-negative")
    if not np.isfinite(photometric_weight) or photometric_weight < 0.0:
        raise ValueError("photometric_weight must be non-negative")
    translation_only = fixed_rotation is not None
    if translation_only:
        fixed_rotation = np.asarray(fixed_rotation, dtype=np.float64)
        if fixed_rotation.shape != (3, 3) or not np.all(np.isfinite(fixed_rotation)):
            raise ValueError("fixed_rotation must be a finite 3x3 matrix")
        T = np.eye(4) if prior is None else prior.copy()
        T[:3, :3] = fixed_rotation
    else:
        T = np.eye(4) if prior is None else prior.copy()

    inlier_frac = 0.0
    conditioning = 0.0
    last_H = None
    photo_active = False
    diagnostics = {
        "photometric_used": False,
        "photometric_matches": 0,
        "depth_scale": None,
        "photometric_scale": None,
        "converged": False,
        "iterations": 0,
        "failure_reason": None,
    }
    for iteration in range(iters):
        diagnostics["iterations"] = iteration + 1
        matches = None
        if depth_weight > 0.0:
            if association is None:
                matches = _projective_matches(src_pts, src_ok, dst_pts,
                                              dst_normals, dst_ok, K, T,
                                              max_dist)
            else:
                matches = association(src_pts, src_ok, T, max_dist)

        photo_matches = None
        if photometric is not None and photometric_weight > 0.0:
            photo_matches = photometric(T)
            if photo_matches is not None:
                photo_active = True
                diagnostics["photometric_used"] = True
                diagnostics["photometric_matches"] = len(photo_matches[1])
                if matches is None:
                    # There is no depth population to define the usual ICP
                    # inlier fraction in a photo-only ablation.
                    inlier_frac = 1.0

        if matches is None and photo_matches is None:
            diagnostics["failure_reason"] = "no matches"
            break

        # The no-photo branch is intentionally the old solve. In particular,
        # `--imu-rotation` alone must remain a depth-only regression baseline;
        # robust scaling belongs to the new combined experiment, not to the
        # existing estimator it is being compared against.
        legacy_depth = (matches is not None and not photo_active
                        and (photometric is None or photo_matches is None)
                        and depth_weight == 1.0)
        if legacy_depth:
            qa, target, na, inlier_frac = matches
            # Linearised point-to-plane residual: for a small rotation w and
            # translation t, minimise sum over points of
            # ((q + w x q + t - target) . n)^2. The Jacobian row is [q x n, n].
            # If the IMU supplied the rotation, it is already in T and only the
            # translation remains unknown; retaining A = n makes the conditioning
            # report describe that 3-DoF problem rather than the abandoned axes.
            A = na if translation_only else np.hstack([np.cross(qa, na), na])
            b = np.einsum("ij,ij->i", (target - qa), na)
            try:
                H = A.T @ A / len(A)
                g = A.T @ b / len(A)
                ev, evec = np.linalg.eigh(H)
            except np.linalg.LinAlgError:
                diagnostics["failure_reason"] = "depth eigensolve"
                break
        else:
            dimension = 3 if translation_only else 6
            H = np.zeros((dimension, dimension), dtype=np.float64)
            g = np.zeros(dimension, dtype=np.float64)

            if matches is not None and depth_weight > 0.0:
                qa, target, na, inlier_frac = matches
                A_depth = (na if translation_only
                           else np.hstack([np.cross(qa, na), na]))
                b_depth = np.einsum("ij,ij->i", (target - qa), na)
                depth_scale = PHOTOMETRIC_DEPTH_SCALE
                weights = robust_weights(b_depth, depth_scale, centre=0.0)
                diagnostics["depth_scale"] = depth_scale
                if np.any(weights > 0.0):
                    A_scaled = A_depth / depth_scale
                    b_scaled = b_depth / depth_scale
                    H += depth_weight * (A_scaled.T
                                         @ (weights[:, None] * A_scaled)
                                         / len(A_scaled))
                    g += depth_weight * (A_scaled.T
                                         @ (weights * b_scaled)
                                         / len(A_scaled))

            if photo_matches is not None and photometric_weight > 0.0:
                A_photo, b_photo = photo_matches
                photo_scale = PHOTOMETRIC_IMAGE_SCALE
                weights = robust_weights(b_photo, photo_scale, centre=0.0)
                diagnostics["photometric_scale"] = photo_scale
                if not translation_only:
                    A_photo = np.hstack([
                        np.zeros((len(A_photo), 3), dtype=np.float64),
                        A_photo,
                    ])
                A_scaled = A_photo / photo_scale
                b_scaled = b_photo / photo_scale
                H += photometric_weight * (A_scaled.T
                                           @ (weights[:, None] * A_scaled)
                                           / len(A_scaled))
                # Unlike the point-to-plane block, the returned Jacobian is
                # the literal derivative of r_p = I_prev(warp(p)) - I_cur(p).
                # Gauss-Newton therefore uses -J^T r for its update; using the
                # depth block's +A^T(target-q) sign here walks the warp away
                # from the image instead of toward it.
                g -= photometric_weight * (A_scaled.T
                                           @ (weights * b_scaled)
                                           / len(A_scaled))

            try:
                ev, evec = np.linalg.eigh(H)
            except np.linalg.LinAlgError:
                diagnostics["failure_reason"] = "combined eigensolve"
                break

        # Clamped at zero: a singular Hessian comes back with a faintly negative
        # smallest eigenvalue, and a negative "fraction" reads as a bug rather
        # than as the blindness it is.
        last_H = H
        conditioning = (max(0.0, float(ev[0] / ev[-1]))
                        if ev[-1] > 1e-12 else 0.0)
        usable = ev > ev[-1] * rcond
        if not usable.any():
            diagnostics["failure_reason"] = "singular Hessian"
            break
        V = evec[:, usable]
        x = V @ ((V.T @ g) / ev[usable])
        # An implausible step is still rejected rather than applied: the cut
        # above removes directions that are hopeless, and this catches the ones
        # that are merely bad.
        translation = x if translation_only else x[3:]
        if (not np.all(np.isfinite(x))
                or np.linalg.norm(translation) > max_dist * 3):
            diagnostics["failure_reason"] = "implausible step"
            break
        if translation_only:
            T[:3, 3] += x
        else:
            wx, wy, wz = x[:3]
            dR = np.array([[1, -wz, wy], [wz, 1, -wx], [-wy, wx, 1]])
            # Re-orthonormalise: the small-angle matrix above is not a
            # rotation, and composing dozens of them without this walks the
            # estimate off SO(3).
            u_, _, vt_ = np.linalg.svd(dR)
            dR = u_ @ vt_
            step = np.eye(4)
            step[:3, :3] = dR
            step[:3, 3] = x[3:]
            T = step @ T
        if np.linalg.norm(translation) < 1e-6 and (translation_only or
                                                   np.linalg.norm(x[:3]) < 1e-6):
            diagnostics["converged"] = True
            break
    # Anchoring happens once, on the converged answer, rather than inside the
    # loop: damping every iteration would also slow convergence along the
    # directions that are well observed, which is the opposite of the intent.
    if (not translation_only and anchor is not None and prior is not None
            and last_H is not None):
        T = anchor_to_prior(prior, T, last_H, anchor)
    if return_diagnostics:
        return T, inlier_frac, conditioning, diagnostics
    return T, inlier_frac, conditioning



# ------------------------------------------------------------- the local map

def render_map(pts, nrm, P, K, shape, footprint=0.0, max_splat=3):
    """Draw the accumulated map as the depth frame this pose would have seen.

    Registering against a rendered map rather than the previous frame is what
    separates odometry that drifts from odometry that does not. Frame-to-frame
    error is a random walk — every frame's noise is added and never revisited —
    while a map is an average over many observations, so its geometry is
    quieter than any single frame, and it still contains the corner you walked
    past two seconds ago that the current view has lost.

    Rendering rather than nearest-neighbour search is deliberate: it puts the
    map back into the shape the projective ICP above already consumes, so the
    registration code is unchanged and only its target differs.

    `footprint` is the world-space size of one map sample — the voxel edge. Each
    sample is drawn as the block of pixels it actually subtends at its own depth
    rather than as a single pixel, which is not cosmetic: a 3 cm voxel covers
    about three pixels across at two metres, so one-pixel splats leave eight of
    every nine pixels empty. ICP then finds a target for one source point in
    seven, and a registration running on a seventh of the evidence is where the
    accumulated track quietly went wrong.
    """
    fx, fy, cx, cy = K
    h, w = shape
    empty = (np.zeros((h, w, 3)), np.zeros((h, w, 3)), np.zeros((h, w), bool))
    R, t = P[:3, :3], P[:3, 3]
    q = (pts - t) @ R                       # world -> camera
    z = q[:, 2]
    keep = np.nonzero(z > 0.1)[0]
    if not len(keep):
        return empty
    qk, zk = q[keep], z[keep]
    ui = np.round(fx * qk[:, 0] / zk + cx).astype(np.int64)
    vi = np.round(fy * qk[:, 1] / zk + cy).astype(np.int64)

    # Half-width of each sample's footprint, in pixels at its own depth.
    if footprint > 0:
        rad = np.floor(0.5 * footprint * fx / zk).astype(np.int64)
        np.clip(rad, 0, max_splat, out=rad)
    else:
        rad = np.zeros(len(keep), np.int64)

    reach = int(rad.max())
    pix, src = [], []
    for du in range(-reach, reach + 1):
        for dv in range(-reach, reach + 1):
            sel = np.nonzero(rad >= max(abs(du), abs(dv)))[0]
            if not len(sel):
                continue
            uu, vv = ui[sel] + du, vi[sel] + dv
            good = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
            if not good.any():
                continue
            pix.append(vv[good] * w + uu[good])
            src.append(sel[good])
    if not pix:
        return empty
    flat = np.concatenate(pix)
    src = np.concatenate(src)
    if len(flat) < 10:
        return empty

    qq, nn, zz = qk[src], nrm[keep][src] @ R, zk[src]
    # Z-buffer: sort by pixel then depth so the nearest surface wins instead of
    # whatever happened to be written last.
    order = np.lexsort((zz, flat))
    fo, qo, no, zo = flat[order], qq[order], nn[order], zz[order]
    first = np.ones(len(fo), bool)
    first[1:] = fo[1:] != fo[:-1]
    run = np.cumsum(first) - 1
    pix = fo[first]

    # Taking only the nearest sample would bias the whole map towards the
    # camera. Sensor noise spreads a surface into a slab a couple of centimetres
    # thick, and per-pixel nearest-wins then picks the *near tail* of that
    # spread every time — measured at roughly -2 cm on 0.5% depth noise, a
    # systematic error that each keyframe writes back into the map and the next
    # frame registers against. Averaging everything within a slab's depth of the
    # nearest sample recovers the surface instead of its leading edge, while
    # still letting a genuine occluder — which sits further than a slab in front
    # of what it hides — win outright.
    window = max(footprint, 0.02)
    nrun = int(run[-1]) + 1

    def average(take):
        idx = run[take]
        psum = np.zeros((nrun, 3))
        nsum = np.zeros((nrun, 3))
        np.add.at(psum, idx, qo[take])
        np.add.at(nsum, idx, no[take])
        cnt = np.maximum(np.bincount(idx, minlength=nrun), 1)[:, None]
        return psum / cnt, nsum / cnt

    # First pass: everything within a slab's depth of the nearest sample. That
    # window is one-sided, so it still clips the far half of the spread and
    # lands short. Re-centring it on the mean it just produced and averaging
    # again is symmetric, and takes the residual bias to a few millimetres.
    centre = zo[first][run]
    for _ in range(2):
        psum, nsum = average(np.abs(zo - centre) <= window)
        centre = psum[run, 2]
    nsum /= np.maximum(np.linalg.norm(nsum, axis=1, keepdims=True), 1e-12)

    out_p = np.zeros((h * w, 3))
    out_n = np.zeros((h * w, 3))
    ok = np.zeros(h * w, bool)
    out_p[pix] = psum
    out_n[pix] = nsum
    ok[pix] = True
    return out_p.reshape(h, w, 3), out_n.reshape(h, w, 3), ok.reshape(h, w)


def voxel_downsample(pts, nrm, size):
    """One point per voxel, by picking a representative.

    Kept for the frame-to-frame path and for callers that only want the map
    bounded. It does *not* make the surface quieter — it picks an arbitrary
    member of each voxel, so the noise of whichever observation won is the noise
    that survives. `LocalMap` averages instead, which is the difference between
    a map that sharpens with more looks and one that merely stays small.
    """
    keys = np.floor(pts / size).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return pts[idx], nrm[idx]


class LocalMap:
    """A local map that *fuses* observations per voxel rather than stacking them.

    Concatenating each frame's points into the map is what made accumulation
    diverge where frame-to-frame did not: every frame is inserted at its
    *estimated* pose, so the pose error is baked into the geometry, the surface
    thickens inside the voxel, and ICP registers happily against the smear it
    just created. Averaging inverts that — a voxel seen ten times is ten times
    quieter, so more looks sharpen the surface instead of fattening it.

    Weight is capped so an old voxel cannot outvote the present indefinitely;
    without that the map stops responding to the scene long before the session
    ends. This is KinectFusion's running-average TSDF update, minus the signed
    distance field: the map here has to be a point cloud because `render_map`
    consumes one.
    """

    def __init__(self, voxel=0.03, range_m=6.0, max_weight=20.0):
        self.voxel = voxel
        self.range = range_m
        self.max_weight = max_weight
        self.keys = np.zeros((0, 3), np.int64)
        self.psum = np.zeros((0, 3))
        self.nsum = np.zeros((0, 3))
        self.w = np.zeros(0)

    def __len__(self):
        return len(self.w)

    @property
    def points(self):
        return self.psum / np.maximum(self.w, 1e-9)[:, None]

    @property
    def normals(self):
        n = self.nsum / np.maximum(self.w, 1e-9)[:, None]
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)

    def integrate(self, pts, nrm, pose):
        """Fold one frame's points, placed at `pose`, into the map."""
        world = pts @ pose[:3, :3].T + pose[:3, 3]
        wnrm = nrm @ pose[:3, :3].T
        keys = np.floor(world / self.voxel).astype(np.int64)

        # Collapse *this frame* to one vote per voxel first. A near surface
        # covers hundreds of pixels and a far one covers three, and without this
        # the map would be weighted by pixel count — which is a fact about
        # perspective, not about how well the geometry is known.
        uk, inv = np.unique(keys, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        ps = np.zeros((len(uk), 3))
        ns = np.zeros((len(uk), 3))
        np.add.at(ps, inv, world)
        np.add.at(ns, inv, wnrm)
        cnt = np.bincount(inv, minlength=len(uk)).astype(np.float64)[:, None]
        ps /= cnt
        ns /= cnt

        allk = np.vstack([self.keys, uk])
        u2, inv2 = np.unique(allk, axis=0, return_inverse=True)
        inv2 = inv2.reshape(-1)
        psum = np.zeros((len(u2), 3))
        nsum = np.zeros((len(u2), 3))
        wsum = np.zeros(len(u2))
        np.add.at(psum, inv2, np.vstack([self.psum, ps]))
        np.add.at(nsum, inv2, np.vstack([self.nsum, ns]))
        np.add.at(wsum, inv2, np.concatenate([self.w, np.ones(len(uk))]))

        over = wsum > self.max_weight
        if over.any():
            scale = self.max_weight / wsum[over]
            psum[over] *= scale[:, None]
            nsum[over] *= scale[:, None]
            wsum[over] = self.max_weight

        self.keys, self.psum, self.nsum, self.w = u2, psum, nsum, wsum

    def trim(self, centre):
        """Drop everything outside the local window around `centre`."""
        if not len(self.w):
            return
        near = np.linalg.norm(self.points - centre, axis=1) < self.range
        self.keys = self.keys[near]
        self.psum = self.psum[near]
        self.nsum = self.nsum[near]
        self.w = self.w[near]

    def render(self, pose, K, shape):
        if not len(self.w):
            h, w = shape
            return (np.zeros((h, w, 3)), np.zeros((h, w, 3)),
                    np.zeros((h, w), bool))
        return render_map(self.points, self.normals, pose, K, shape,
                          footprint=self.voxel)


class Tracker:
    """Accumulates a trajectory from depth frames, one `step` at a time.

    Split out of the session loop so the accumulation can be tested against
    rendered frames with known poses. The single-step accuracy was never the
    problem — drift only appears over dozens of frames, which is exactly what a
    per-pair test cannot see.
    """

    def __init__(self, frame_to_frame=False, max_dist=0.15, voxel=0.03,
                 map_range=6.0, keyframe_dist=0.05, keyframe_angle=5.0,
                 keyframe_fill=0.6, min_conditioning=1e-3,
                 projective_association=False, imu=None, floor_lock=False,
                 floor_lock_gain=FLOOR_LOCK_GAIN, gravity_lock=False,
                 gravity_lock_gain=GRAVITY_LOCK_GAIN, gravity_anchor=False,
                 gravity_anchor_lambda=GRAVITY_ANCHOR_LAMBDA,
                 imu_rotation=False, depth_weight=1.0,
                 photometric_weight=1.0, photometric_reference="keyframe",
                 keyframe_on_image=False, keyframe_min_inliers=0.0,
                 match_radius_probe=False):
        self.frame_to_frame = frame_to_frame
        self.projective_association = projective_association
        self.imu = imu
        self.floor_lock = floor_lock
        self.floor_lock_gain = floor_lock_gain
        self.gravity_lock = gravity_lock
        if not np.isfinite(gravity_lock_gain) or not 0.0 <= gravity_lock_gain <= 1.0:
            raise ValueError("gravity_lock_gain must be between 0 and 1")
        self.gravity_lock_gain = gravity_lock_gain
        self.gravity_anchor = gravity_anchor
        if not np.isfinite(gravity_anchor_lambda) or gravity_anchor_lambda <= 0.0:
            raise ValueError("gravity_anchor_lambda must be positive")
        self.gravity_anchor_lambda = gravity_anchor_lambda
        self.imu_rotation = imu_rotation
        if not np.isfinite(depth_weight) or depth_weight < 0.0:
            raise ValueError("depth_weight must be non-negative")
        if not np.isfinite(photometric_weight) or photometric_weight < 0.0:
            raise ValueError("photometric_weight must be non-negative")
        if photometric_reference not in ("previous", "keyframe"):
            raise ValueError("photometric_reference must be 'previous' or 'keyframe'")
        self.depth_weight = depth_weight
        self.photometric_weight = photometric_weight
        self.photometric_reference = photometric_reference
        self.keyframe_on_image = keyframe_on_image
        if not np.isfinite(keyframe_min_inliers) or not 0.0 <= keyframe_min_inliers <= 1.0:
            raise ValueError("keyframe_min_inliers must be between 0 and 1")
        # Compared with `>=`, so the default of 0.0 admits every frame the
        # existing thresholds admitted, including one that matched nothing.
        self.keyframe_min_inliers = keyframe_min_inliers
        self.keyframe_flags = []
        self.keyframes_refused_inliers = 0
        self.match_radius_probe = match_radius_probe
        self.match_radius_fractions = []
        self.max_dist = max_dist
        self.min_conditioning = min_conditioning
        self.keyframe_dist = keyframe_dist
        self.keyframe_angle = math.radians(keyframe_angle)
        self.keyframe_fill = keyframe_fill
        self.map = LocalMap(voxel=voxel, range_m=map_range)
        self.poses = [np.eye(4)]
        self.inliers = []
        self.conditioning = []
        self.keyframes = 0
        self._prev = None
        self._last_kf = None
        self.image_keyframes = 0
        self._velocity = np.eye(4)
        self._velocity_dt = None
        self._last_timestamp = None
        self.imu_uses = 0
        self.imu_fallbacks = 0
        self.floor_heights = []
        self.floor_point_counts = []
        self.floor_corrections = 0.0
        self._floor_frames = 0
        self._floor_plane = None
        self._floor_gravity_world = None
        self._gravity_world_ref = None
        self._imu_world_rotation_ref = None
        self.imu_rotation_uses = 0
        self.imu_rotation_fallbacks = 0
        self.gravity_errors = []
        self.gravity_pre_errors = []
        self.gravity_corrections = 0.0
        self.photometric_uses = 0
        self.photometric_attempts = 0
        self.photometric_evaluations = 0
        self.photometric_accepted_steps = 0
        self.photometric_convergence_failures = 0
        self.photometric_matches = []
        self.photometric_scales = []
        self.depth_scales = []
        self.photometric_level_shapes = ()
        self.photometric_level_scales = ()
        self.photometric_level_count = 0
        self.photometric_used_history = []
        self._photo_ref = None
        self.photometric_reference_distances = []
        self.photometric_reference_angles = []

    @property
    def photo_reference(self):
        """Return the latest image that was actually integrated as a keyframe."""
        return self._photo_ref

    def _save_photo_reference(self, photo, pose):
        if photo is None or self.photometric_reference != "keyframe":
            return
        self._photo_ref = {
            "image": photo["image"],
            "K": photo["K"],
            "pose": pose.copy(),
            "frame": photo.get("frame"),
        }

    def _record_photo_reference_distance(self, pose, reference_pose):
        if reference_pose is None:
            return
        relative = relative_camera_transform(reference_pose, pose)
        self.photometric_reference_distances.append(
            float(np.linalg.norm(relative[:3, 3])))
        angle = math.acos(max(-1.0, min(1.0,
                              (np.trace(relative[:3, :3]) - 1) / 2)))
        self.photometric_reference_angles.append(math.degrees(angle))

    def _floor_observation(self, pts, nrm, ok, gravity):
        if not self.floor_lock or gravity is None:
            return None, 0
        height, count = estimate_floor_height(pts, nrm, ok, gravity)
        if height is not None:
            self.floor_heights.append(height)
            self.floor_point_counts.append(count)
            self._floor_frames += 1
        return height, count

    def _apply_floor_lock(self, pose, height, gravity):
        """Softly align pose height with the first observed floor plane."""
        if not self.floor_lock or height is None or gravity is None:
            return pose
        g = np.asarray(gravity, dtype=np.float64).reshape(-1)
        if g.size != 3 or not np.all(np.isfinite(g)):
            return pose
        g /= max(float(np.linalg.norm(g)), 1e-12)
        if self._floor_plane is None:
            # Floor points satisfy dot(point_camera, g) == height. Transform
            # that plane into the tracker world at the first usable frame.
            self._floor_gravity_world = pose[:3, :3] @ g
            self._floor_plane = float(
                np.dot(pose[:3, 3], self._floor_gravity_world) + height)
            return pose

        # Use the first-frame world normal as the vertical axis. Recomputing
        # it from the drifting ICP rotation would rotate the correction into
        # horizontal motion, precisely where the floor lock is meant to stay
        # out of the way.
        world_g = self._floor_gravity_world
        target = self._floor_plane - height
        current = float(np.dot(pose[:3, 3], world_g))
        error = target - current
        correction = self.floor_lock_gain * error
        if not np.isfinite(correction):
            return pose
        corrected = pose.copy()
        corrected[:3, 3] += correction * world_g
        self.floor_corrections += abs(float(correction))
        return corrected

    def floor_report(self, frame_count):
        """Summarise floor observations for the command-line report."""
        heights = np.asarray(self.floor_heights, dtype=np.float64)
        if not len(heights):
            return {"visible": 0, "frames": frame_count, "median": None,
                    "variation": None, "correction": self.floor_corrections}
        median = float(np.median(heights))
        variation = float(1.4826 * np.median(np.abs(heights - median)))
        return {"visible": self._floor_frames, "frames": frame_count,
                "median": median, "variation": variation,
                "correction": self.floor_corrections}

    @staticmethod
    def _gravity_angle(a, b):
        a = _unit_vector(a)
        b = _unit_vector(b)
        if a is None or b is None:
            return None
        return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(a, b))))))

    def _apply_gravity_lock(self, pose, gravity):
        """Restore roll and pitch while preserving the current yaw.

        ``pose`` is world-from-camera. The estimated camera gravity is thus
        ``pose.R @ gravity``. The correction maps that vector to the first
        frame's world gravity and is multiplied on the *left*: it is a world
        frame correction, not a camera-local yaw/roll operation. Translation
        is deliberately untouched.
        """
        g_cam = _unit_vector(gravity)
        if g_cam is None:
            return pose
        if self._gravity_world_ref is None:
            self._gravity_world_ref = pose[:3, :3] @ g_cam
            self.gravity_errors.append(0.0)
            self.gravity_pre_errors.append(0.0)
            return pose

        g_est = pose[:3, :3] @ g_cam
        pre_error = self._gravity_angle(g_est, self._gravity_world_ref)
        if pre_error is None:
            return pose
        corrected = pose
        if self.gravity_lock:
            _, angle = _minimal_rotation(g_est, self._gravity_world_ref)
            applied = angle * self.gravity_lock_gain
            if applied > 1e-12:
                axis = np.cross(g_est, self._gravity_world_ref)
                axis_norm = float(np.linalg.norm(axis))
                if axis_norm <= 1e-12 and angle > 0.0:
                    basis = np.eye(3)[int(np.argmin(np.abs(g_est)))]
                    axis = np.cross(g_est, basis)
                axis /= max(float(np.linalg.norm(axis)), 1e-12)
                corrected = pose.copy()
                # Gravity gives two axes. Do not rotate the pose around the
                # gravity axis, and do not move its camera centre.
                corrected[:3, :3] = (_axis_angle_rotation(axis, applied)
                                     @ pose[:3, :3])
                self.gravity_corrections += math.degrees(applied)
        post_error = self._gravity_angle(
            corrected[:3, :3] @ g_cam, self._gravity_world_ref)
        self.gravity_pre_errors.append(pre_error)
        self.gravity_errors.append(0.0 if post_error is None else post_error)
        return corrected

    def _gravity_prior(self, predicted, gravity):
        """Return a gravity-corrected predicted pose and ICP-relative prior.

        The map ICP target is expressed in the camera frame attached to
        ``predicted``. Its transform therefore is not the world-from-camera
        pose itself: it is the relative correction from that predicted frame.
        Keeping the predicted translation in ``corrected`` and taking
        ``inv(predicted) @ corrected`` leaves gravity silent about position
        while putting its two rotation constraints in ICP's coordinates.
        """
        if not self.gravity_anchor:
            return predicted, None
        g_cam = _unit_vector(gravity)
        if g_cam is None:
            return predicted, None
        if self._gravity_world_ref is None:
            self._gravity_world_ref = predicted[:3, :3] @ g_cam

        g_est = predicted[:3, :3] @ g_cam
        correction, _ = _minimal_rotation(g_est, self._gravity_world_ref)
        corrected = predicted.copy()
        corrected[:3, :3] = correction @ predicted[:3, :3]
        return corrected, np.linalg.inv(predicted) @ corrected

    def _aligned_imu_rotation(self, absolute_rotation):
        """Align an IMU world-from-depth rotation to the first tracker frame."""
        if not self.imu_rotation:
            return None
        if absolute_rotation is None:
            self.imu_rotation_fallbacks += 1
            return None
        rotation = np.asarray(absolute_rotation, dtype=np.float64)
        if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
            self.imu_rotation_fallbacks += 1
            return None
        if self._imu_world_rotation_ref is None:
            self._imu_world_rotation_ref = rotation.copy()
        self.imu_rotation_uses += 1
        return self._imu_world_rotation_ref.T @ rotation

    def gravity_report(self):
        """Summarise roll/pitch drift against the first-frame gravity vector."""
        if not self.gravity_errors:
            return None
        return {"start": self.gravity_errors[0],
                "end": self.gravity_errors[-1],
                "maximum": max(self.gravity_errors),
                "pre_end": self.gravity_pre_errors[-1],
                "pre_maximum": max(self.gravity_pre_errors)}

    def _keyframe_due(self, pose, fill):
        """Insert on motion, or when the view has outrun what the map covers.

        Distance and angle are the usual test. The fill term is what stops a
        walk down a corridor from tracking against a map of the room behind it:
        new ground is new ground whether or not the phone moved far to reach it.
        """
        if self._last_kf is None:
            return True
        d = np.linalg.inv(self._last_kf) @ pose
        angle = math.acos(max(-1.0, min(1.0, (np.trace(d[:3, :3]) - 1) / 2)))
        return (np.linalg.norm(d[:3, 3]) > self.keyframe_dist
                or angle > self.keyframe_angle
                or fill < self.keyframe_fill)

    def _photometric_callback(self, pts, ok, pair, previous_pose,
                              base_pose, mode):
        if pair is None or previous_pose is None:
            return None
        points = pts[ok]
        if len(points) < PHOTOMETRIC_MIN_MATCHES:
            return None
        return PhotometricPyramid(
            points, pair["previous_image"], pair["current_image"],
            pair["previous_K"], pair["current_K"], previous_pose,
            base_pose, mode)

    def _record_icp_diagnostics(self, diagnostics, photometric_attempt=False):
        self.photometric_used_history.append(
            bool(diagnostics.get("photometric_used")))
        if photometric_attempt:
            self.photometric_attempts += 1
            if (not diagnostics.get("photometric_used")
                    or not diagnostics.get("converged")):
                self.photometric_convergence_failures += 1
        if diagnostics.get("photometric_used"):
            self.photometric_uses += 1
            self.photometric_evaluations += int(
                diagnostics.get("photometric_evaluations", 0))
            self.photometric_accepted_steps += int(
                diagnostics.get("photometric_accepted_steps", 0))
            self.photometric_matches.append(diagnostics["photometric_matches"])
            if diagnostics["photometric_scale"] is not None:
                self.photometric_scales.append(diagnostics["photometric_scale"])
            if diagnostics["depth_scale"] is not None:
                self.depth_scales.append(diagnostics["depth_scale"])
            if diagnostics.get("photometric_level_shapes"):
                self.photometric_level_shapes = tuple(
                    diagnostics["photometric_level_shapes"])
                self.photometric_level_scales = tuple(
                    diagnostics.get("photometric_level_scales", ()))
                self.photometric_level_count = int(
                    diagnostics.get("photometric_level_count", 0))

    def step(self, pts, nrm, ok, K, rotation_prior=None, timestamp=None,
             gravity=None, imu_rotation=None, photometric_pair=None,
             previous_image_pose=None, photometric_current=None,
             photometric_reference_pose=None):
        """Register one frame and return its world-from-camera pose."""
        imu_R = self._aligned_imu_rotation(imu_rotation)
        floor_height, _ = self._floor_observation(pts, nrm, ok, gravity)
        if self._prev is None:
            self._prev = (pts, nrm, ok)
            self._last_timestamp = timestamp
            if self.depth_weight > 0.0:
                self.map.integrate(pts[ok], nrm[ok], self.poses[0])
                if photometric_current is not None:
                    self.image_keyframes += 1
            self._last_kf = self.poses[0]
            self.keyframes = 1
            self.poses[0] = self._apply_gravity_lock(self.poses[0], gravity)
            self.poses[0] = self._apply_floor_lock(
                self.poses[0], floor_height, gravity)
            if self.depth_weight > 0.0:
                self._save_photo_reference(photometric_current, self.poses[0])
            return self.poses[0]

        imu_motion = None
        if self.imu is not None and self._last_timestamp is not None:
            imu_motion = self.imu.relative_motion(self._last_timestamp, timestamp)

        made_keyframe = False
        if self.frame_to_frame:
            prior = None
            fixed_rotation = None
            if imu_R is not None:
                # ICP maps previous-camera coordinates into the current camera
                # frame, so the known relative rotation is current^T * previous.
                fixed_rotation = imu_R.T @ self.poses[-1][:3, :3]
                prior = np.eye(4)
                prior[:3, :3] = fixed_rotation
            elif imu_motion is not None and imu_motion["rotation"] is not None:
                # IMU motion is previous-camera-from-current-camera, while ICP
                # maps the previous frame into the current frame.
                prior = np.eye(4)
                prior[:3, :3] = imu_motion["rotation"].T
                self.imu_uses += 1
            elif self.imu is not None:
                self.imu_fallbacks += 1
                if rotation_prior is not None:
                    prior = np.eye(4)
                    prior[:3, :3] = rotation_prior
            elif rotation_prior is not None:
                prior = np.eye(4)
                prior[:3, :3] = rotation_prior
            anchor = None
            if self.gravity_anchor and fixed_rotation is None:
                predicted = self.poses[-1] @ (
                    np.linalg.inv(prior) if prior is not None else self._velocity)
                gravity_pose, gravity_prior = self._gravity_prior(predicted, gravity)
                if gravity_prior is not None:
                    prior = np.linalg.inv(gravity_pose) @ self.poses[-1]
                    anchor = self.gravity_anchor_lambda
            prev_pts, prev_nrm, prev_ok = self._prev
            photo_fn = self._photometric_callback(
                pts, ok, photometric_pair, previous_image_pose,
                self.poses[-1], "frame_to_frame")
            # ICP solves target-from-source; the trajectory needs its inverse.
            T, frac, cond, diagnostics = icp(
                prev_pts, prev_ok, pts, nrm, ok, K,
                max_dist=self.max_dist, prior=prior, anchor=anchor,
                fixed_rotation=fixed_rotation, photometric=photo_fn,
                depth_weight=self.depth_weight,
                photometric_weight=self.photometric_weight,
                return_diagnostics=True)
            self._record_icp_diagnostics(
                diagnostics, photometric_attempt=(photometric_pair is not None
                                                  and previous_image_pose is not None))
            pose = self.poses[-1] @ np.linalg.inv(T)
            pose = self._apply_gravity_lock(pose, gravity)
            pose = self._apply_floor_lock(pose, floor_height, gravity)
            self._record_photo_reference_distance(
                pose, photometric_reference_pose)
        else:
            # Predict where we are and let ICP supply the correction. A
            # constant-velocity guess is the fallback; with an IMU stream the
            # same prediction also carries the measured relative rotation and
            # acceleration over this depth-frame interval.
            predicted = self.poses[-1] @ self._velocity
            if imu_motion is not None and imu_motion["acceleration"] is not None:
                dt = timestamp - self._last_timestamp
                if (np.isfinite(dt) and dt > 0 and self._velocity_dt is not None
                        and self._velocity_dt > 0):
                    v_prev = self._velocity[:3, 3] / self._velocity_dt
                    relative = np.eye(4)
                    relative[:3, :3] = imu_motion["rotation"]
                    relative[:3, 3] = (v_prev * dt
                                       + 0.5 * imu_motion["acceleration"] * dt * dt)
                    predicted = self.poses[-1] @ relative
                    self.imu_uses += 1
                else:
                    self.imu_fallbacks += 1
            elif self.imu is not None:
                self.imu_fallbacks += 1
            if imu_R is not None:
                # The IMU owns orientation in this mode. Translation remains
                # the existing constant-velocity/acceleration prediction.
                predicted[:3, :3] = imu_R
            gravity_prior = None
            gravity_pose = predicted
            if self.gravity_anchor and imu_R is None:
                gravity_pose, gravity_prior = self._gravity_prior(predicted, gravity)
            anchor = self.gravity_anchor_lambda if gravity_prior is not None else None
            map_fixed_rotation = np.eye(3) if imu_R is not None else None
            if self.depth_weight <= 0.0:
                # A photometric-only run has no map to associate against. It
                # still uses the same current->previous transform convention
                # as the map-out-of-view fallback below.
                fill = 0.0
                initial = None
            elif self.projective_association:
                # Kept as a switch for comparisons and easy rollback. The
                # default map association below never renders the map.
                dst_p, dst_n, dst_o = self.map.render(predicted, K, ok.shape)
                fill = float(dst_o.mean())
            else:
                initial = _voxel_matches(
                    pts, ok, self.map, predicted, np.eye(4), self.max_dist)
                fill = 0.0 if initial is None else initial[3]
                # Observation only: the same map at the same predicted pose,
                # asked at tighter match radii. A frame that matches poorly
                # because the map has no points there loses nothing further as
                # the radius tightens — what it did match was already close. A
                # frame that matches poorly because its predicted pose is wrong
                # is matching at a distance, so its fraction falls away faster
                # than everyone else's. The gate in `--keyframe-min-inliers`
                # could not ask this, because it changes the map it measures.
                if self.match_radius_probe:
                    self.match_radius_fractions.append([
                        (0.0 if probe is None else probe[3])
                        for probe in (
                            initial if radius == self.max_dist else
                            _voxel_matches(pts, ok, self.map, predicted,
                                           np.eye(4), radius)
                            for radius in MATCH_RADIUS_PROBES)])
            if fill < 0.02:
                # Nothing of the map is in view; fall back rather than invent.
                # Source is the *current* frame here, as it is against the map
                # below — not the previous one as in the frame-to-frame branch
                # above. So T is already previous-from-current and composes
                # directly; inverting it as well stepped backwards, moving the
                # pose by twice the motion in the wrong direction.
                prev_pts, prev_nrm, prev_ok = self._prev
                fallback_prior = (np.linalg.inv(self.poses[-1]) @ gravity_pose
                                  if gravity_prior is not None
                                  else (np.linalg.inv(self.poses[-1]) @ predicted
                                        if self.depth_weight <= 0.0 else None))
                photo_fn = self._photometric_callback(
                    pts, ok, photometric_pair, previous_image_pose,
                    self.poses[-1], "forward")
                T, frac, cond, diagnostics = icp(
                    pts, ok, prev_pts, prev_nrm, prev_ok, K,
                    max_dist=self.max_dist, prior=fallback_prior,
                    anchor=anchor,
                    fixed_rotation=(self.poses[-1][:3, :3].T
                                    @ predicted[:3, :3]
                                    if imu_R is not None else None),
                    photometric=photo_fn, depth_weight=self.depth_weight,
                    photometric_weight=self.photometric_weight,
                    return_diagnostics=True)
                self._record_icp_diagnostics(
                    diagnostics, photometric_attempt=(photometric_pair is not None
                                                      and previous_image_pose is not None))
                pose = self.poses[-1] @ T
            else:
                photo_fn = self._photometric_callback(
                    pts, ok, photometric_pair, previous_image_pose,
                    predicted, "forward")
                if self.projective_association:
                    T, frac, cond, diagnostics = icp(
                        pts, ok, dst_p, dst_n, dst_o, K,
                        max_dist=self.max_dist, prior=gravity_prior,
                        anchor=anchor, fixed_rotation=map_fixed_rotation,
                        photometric=photo_fn, depth_weight=self.depth_weight,
                        photometric_weight=self.photometric_weight,
                        return_diagnostics=True)
                    self._record_icp_diagnostics(
                        diagnostics, photometric_attempt=(photometric_pair is not None
                                                          and previous_image_pose is not None))
                else:
                    source, target, normal, initial_frac = initial

                    # The map lookup is made once at the predicted pose. ICP
                    # iterations refine those matches without rasterising or
                    # searching the map again.
                    def association(_source_pts, _source_ok, T, _max_dist):
                        q = source @ T[:3, :3].T + T[:3, 3]
                        return q, target, normal, initial_frac

                    T, frac, cond, diagnostics = icp(
                        pts, ok, pts, nrm, ok, K,
                        max_dist=self.max_dist, prior=gravity_prior,
                        anchor=anchor, association=association,
                        fixed_rotation=map_fixed_rotation,
                        photometric=photo_fn, depth_weight=self.depth_weight,
                        photometric_weight=self.photometric_weight,
                        return_diagnostics=True)
                    self._record_icp_diagnostics(
                        diagnostics, photometric_attempt=(photometric_pair is not None
                                                          and previous_image_pose is not None))
                    fill = frac
                # T maps current-camera coords into the predicted frame, so the
                # actual pose is the prediction composed with it.
                pose = predicted @ T
            pose = self._apply_gravity_lock(pose, gravity)
            pose = self._apply_floor_lock(pose, floor_height, gravity)
            previous_pose = self.poses[-1]
            if not _is_valid_rigid_pose(pose):
                # A photometric-only run can lose its basin on a textureless
                # or repeated-texture pair. Keep the last valid pose and make
                # the next frame start from a stationary prediction instead
                # of allowing a singular transform to cascade through the
                # trajectory and the final scoring inverses.
                pose = (previous_pose.copy()
                        if _is_valid_rigid_pose(previous_pose) else np.eye(4))
                self._velocity = np.eye(4)
            else:
                self._velocity = np.linalg.inv(previous_pose) @ pose
            if timestamp is not None and self._last_timestamp is not None:
                dt = timestamp - self._last_timestamp
                self._velocity_dt = dt if np.isfinite(dt) and dt > 0 else None
            else:
                self._velocity_dt = None
            # Normal keyframes are never made from a pose the geometry could
            # not pin down: that frame's position along the free axis is a
            # guess, and folding it in writes the guess into the map for every
            # later frame to register against. Image-forced keyframes are the
            # explicit exception: their purpose is to put the photo reference
            # on the same map keyframe set, even when the normal thresholds did
            # not request a keyframe.
            image_keyframe = (self.keyframe_on_image
                              and photometric_current is not None)
            # The registration-quality gate sits outside the or-group on
            # purpose: a frame that registered badly should stay out of the map
            # whether or not it carries an image. Conditioning asks whether the
            # geometry *could* pin the pose down; this asks whether it did.
            wanted = (image_keyframe
                      or (cond > self.min_conditioning
                          and self._keyframe_due(pose, fill)))
            registered = frac >= self.keyframe_min_inliers
            if wanted and not registered:
                self.keyframes_refused_inliers += 1
            made_keyframe = self.depth_weight > 0.0 and wanted and registered
            if made_keyframe:
                self.map.integrate(pts[ok], nrm[ok], pose)
                self.map.trim(pose[:3, 3])
                self._last_kf = pose
                self.keyframes += 1
                if photometric_current is not None:
                    self.image_keyframes += 1
                self._save_photo_reference(photometric_current, pose)

            self._record_photo_reference_distance(
                pose, photometric_reference_pose)

        self.poses.append(pose)
        self.inliers.append(frac)
        self.conditioning.append(cond)
        self.keyframe_flags.append(made_keyframe)
        self._prev = (pts, nrm, ok)
        self._last_timestamp = timestamp
        return pose


def trajectory_point_to_plane_residual(trajectory, frames, max_dist=0.15):
    """Measure geometric consistency of a trajectory against depth frames.

    `trajectory` contains world-from-camera matrices. `frames` contains one
    `(points, normals, valid, K)` tuple per trajectory pose. Each current frame
    is moved into the previous frame's camera coordinates and matched with the
    same projective association used by `icp()`. The returned median is the
    median of the per-frame point-to-plane medians, so a single noisy frame
    cannot dominate the session summary.

    A `None` entry means that fewer than 100 points survived the same matching
    gates as ICP. Keeping those entries lets callers compare two trajectories
    on exactly the same frame pairs.
    """
    if len(trajectory) != len(frames):
        raise ValueError("trajectory and frames must have the same length")

    per_frame = [None]
    for i in range(1, len(trajectory)):
        # world_from_camera gives current-camera -> previous-camera here.
        T = relative_camera_transform(trajectory[i], trajectory[i - 1])
        src_pts, _, src_ok, _ = frames[i]
        dst_pts, dst_normals, dst_ok, K = frames[i - 1]
        matches = _projective_matches(src_pts, src_ok, dst_pts, dst_normals,
                                      dst_ok, K, T, max_dist)
        if matches is None:
            per_frame.append(None)
            continue
        q, target, normal, _ = matches
        residual = np.abs(np.einsum("ij,ij->i", target - q, normal))
        per_frame.append(float(np.median(residual)))

    valid = [value for value in per_frame if value is not None]
    return {
        "median": float(np.median(valid)) if valid else None,
        "per_frame": per_frame,
        "frames": len(valid),
    }


# ------------------------------------------------------------------ scoring

def quat_to_matrix(x, y, z, w):
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def relative_camera_transform(previous_world_from_camera,
                              current_world_from_camera):
    """Return current-camera <- previous-camera from two ``T_wc`` poses.

    Session poses are ``T_wc`` because ``X_world = R @ X_camera + t``. Thus
    the transform that maps a point in the previous camera into the current
    camera is ``inv(T_wc_current) @ T_wc_previous``. Keeping this operation in
    one named function prevents a world-frame rotation from being mistaken for
    a camera-frame relative rotation in the photometric and ICP paths.
    """
    previous = np.asarray(previous_world_from_camera, dtype=np.float64)
    current = np.asarray(current_world_from_camera, dtype=np.float64)
    if previous.shape != (4, 4) or current.shape != (4, 4):
        raise ValueError("relative poses must be 4x4")
    return np.linalg.inv(current) @ previous


def _relative_pose_self_test():
    """Check the T_wc composition direction with a non-commuting pose pair."""
    previous = np.eye(4)
    previous[:3, :3] = _axis_angle_rotation(np.array([0.2, 0.7, -0.1]), 0.31)
    previous[:3, 3] = [0.4, -0.2, 1.1]
    current = np.eye(4)
    current[:3, :3] = _axis_angle_rotation(np.array([-0.4, 0.3, 0.8]), -0.22)
    current[:3, 3] = [-0.3, 0.5, 1.7]
    forward = relative_camera_transform(previous, current)
    backward = relative_camera_transform(current, previous)
    return bool(np.allclose(backward @ forward, np.eye(4), atol=1e-10))


def _image_gradient_self_test():
    """Guard the row/column order used by the photometric Jacobian."""
    probe = np.tile(np.arange(8, dtype=np.float64), (6, 1))
    grad_row, grad_col = np.gradient(probe)
    return bool(np.allclose(grad_row, 0.0)
                and np.allclose(grad_col, 1.0))


def load_photometric_image(path, cache=None):
    """Load an image as raw grayscale and reduce it to the measured 480x360."""
    if cache is not None and path in cache:
        return cache[path]
    try:
        import cv2
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise OSError(f"cannot decode image: {path}")
        image = cv2.resize(image, (PHOTOMETRIC_WIDTH, PHOTOMETRIC_HEIGHT),
                           interpolation=cv2.INTER_AREA)
    except ImportError:
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "--photometric needs opencv-python or Pillow") from exc
        with Image.open(path) as source:
            image = source.convert("L").resize(
                (PHOTOMETRIC_WIDTH, PHOTOMETRIC_HEIGHT), Image.Resampling.BOX)
        image = np.asarray(image)
    image = np.asarray(image, dtype=np.float64)
    if cache is not None:
        cache[path] = image
    return image


def photometric_image_intrinsics(pose, image_entry):
    """Scale full-resolution pose intrinsics to the fixed photo image size."""
    sx = PHOTOMETRIC_WIDTH / float(image_entry["width"])
    sy = PHOTOMETRIC_HEIGHT / float(image_entry["height"])
    return (pose["fx"] * sx, pose["fy"] * sy,
            pose["cx"] * sx, pose["cy"] * sy)


def _downsample_gray(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Area-reduce one grayscale pyramid level without changing its value scale."""
    try:
        import cv2
        return np.asarray(cv2.resize(image, (width, height),
                                     interpolation=cv2.INTER_AREA),
                          dtype=np.float64)
    except ImportError:  # pragma: no cover - cv2 is the normal image backend
        from PIL import Image
        source = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="L")
        return np.asarray(source.resize((width, height), Image.Resampling.BOX),
                          dtype=np.float64)


class PhotometricPyramid:
    """Five-level 480x360 grayscale image pyramid.

    The exact valid ladder is 32x24, 64x48, 120x90, 240x180, and 480x360.
    These sizes follow exact scaling relative to the 480x360 working image;
    the image remains raw 8-bit grayscale, with no exposure or sRGB
    correction.
    """

    def __init__(self, points, previous_image, current_image, previous_K,
                 current_K, previous_pose, base_pose, mode):
        self.points = np.asarray(points, dtype=np.float64)
        self.previous_pose = previous_pose
        self.base_pose = base_pose
        self.mode = mode
        self.levels = tuple(range(5))  # coarse -> fine
        self._scales = (1.0 / 15.0, 1.0 / 7.5, 1.0 / 4.0,
                        1.0 / 2.0, 1.0)
        self._sizes = ((32, 24), (64, 48), (120, 90),
                       (240, 180), (480, 360))
        self._previous = {}
        self._current = {}
        for level, (width, height) in enumerate(self._sizes):
            self._previous[level] = _downsample_gray(
                previous_image, width, height)
            self._current[level] = _downsample_gray(
                current_image, width, height)
        self._previous_K = previous_K
        self._current_K = current_K
        self._level = 0
        self.level_shapes = tuple(
            (int(self._current[level].shape[1]),
             int(self._current[level].shape[0]))
            for level in self.levels)

    def begin_level(self, level: int):
        self._level = int(level)

    def __call__(self, T):
        level = self._level
        scale = self._scales[level]
        previous_K = tuple(float(value) * scale for value in self._previous_K)
        current_K = tuple(float(value) * scale for value in self._current_K)
        return photometric_terms(
            self.points, self._previous[level], self._current[level],
            previous_K, current_K, T, self.previous_pose,
            self.base_pose, self.mode)


def _apply_icp_increment(T, x, translation_only):
    """Apply one already-accepted ICP/LM increment to ``T``."""
    updated = T.copy()
    if translation_only:
        updated[:3, 3] += x
        return updated
    wx, wy, wz = x[:3]
    dR = np.array([[1, -wz, wy], [wz, 1, -wx], [-wy, wx, 1]])
    u_, _, vt_ = np.linalg.svd(dR)
    dR = u_ @ vt_
    step = np.eye(4)
    step[:3, :3] = dR
    step[:3, 3] = x[3:]
    return step @ updated


def _is_valid_rigid_pose(T):
    """Reject non-finite or singular poses before they poison the next step."""
    value = np.asarray(T, dtype=np.float64)
    if value.shape != (4, 4) or not np.all(np.isfinite(value)):
        return False
    if not np.allclose(value[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        return False
    R = value[:3, :3]
    return bool(abs(float(np.linalg.det(R))) > 1e-8
                and np.allclose(R.T @ R, np.eye(3), atol=1e-5))


def photometric_synthetic_truth_test() -> bool:
    """Verify the five-level optimizer on two known 10/18 cm translations."""
    h, w = PHOTOMETRIC_HEIGHT, PHOTOMETRIC_WIDTH
    x, y = np.meshgrid(np.arange(w, dtype=np.float64),
                       np.arange(h, dtype=np.float64))
    previous = (90.0 + 35.0 * np.sin(x / 13.0)
                + 28.0 * np.sin((x + y) / 27.0)
                + 12.0 * np.cos(y / 11.0))
    K = (200.0, 200.0, w / 2.0, h / 2.0)
    px, py = np.meshgrid(np.linspace(-0.7, 0.7, 64),
                         np.linspace(-0.45, 0.45, 48))
    points = np.c_[px.ravel(), py.ravel(), np.full(px.size, 2.0)]
    identity = np.eye(4)
    ok = np.ones(len(points), dtype=bool)
    for translation in (0.10, 0.18):
        shift = int(round(K[0] * translation / points[0, 2]))
        current = np.zeros_like(previous)
        current[:, :w - shift] = previous[:, shift:]
        pyramid = PhotometricPyramid(
            points, previous, current, K, K, identity, identity, "forward")
        estimate, _, _, diagnostics = icp(
            points, ok, points, np.zeros_like(points), ok, K,
            max_dist=0.30, fixed_rotation=np.eye(3), depth_weight=0.0,
            photometric=pyramid, return_diagnostics=True)
        if (not diagnostics["converged"]
                or abs(float(estimate[0, 3]) - translation) > 0.002
                or abs(float(estimate[1, 3])) > 0.002
                or abs(float(estimate[2, 3])) > 0.002):
            return False
    return True


# Device-frame vectors and rotations first use this fixed physical-frame
# transform into the ARKit camera frame. It is the exact landscape/portrait
# axis permutation measured for the capture device.
R_CAM_FROM_DEV = np.array([[0.0, -1.0, 0.0],
                           [1.0,  0.0, 0.0],
                           [0.0,  0.0, 1.0]])
IMU_ACCELERATION_MPS2 = 9.80665


# ARKit's camera looks down -Z with +Y up; the depth map's own frame is +Z
# forward, +Y down. Columns are the depth axes in ARKit camera coordinates.
ARKIT_TO_DEPTH = np.diag([1.0, -1.0, -1.0])


def arkit_gravity_to_depth(gravity):
    """Convert pose.jsonl gravity from ARKit camera to depth coordinates."""
    g = _unit_vector(gravity)
    return None if g is None else _unit_vector(ARKIT_TO_DEPTH.T @ g)


def motion_gravity_to_depth(gravity):
    """Convert motion.jsonl device gravity into the depth-camera frame.

    The ultra-wide path has no ARKit pose. Its ``gx/gy/gz`` vector is in the
    CoreMotion device frame, so it takes the measured device-to-ARKit-camera
    rotation first and the same ARKit-camera-to-depth basis change second.
    The two sources agree to 0.094 degrees in the relative rotation frame.
    """
    g = _unit_vector(gravity)
    if g is None:
        return None
    return _unit_vector(ARKIT_TO_DEPTH.T @ R_CAM_FROM_DEV @ g)


class IMUStream:
    """Nearest-timestamp IMU joins for consecutive depth frames.

    CoreMotion already provides fused attitudes, so the quaternion at each
    endpoint is used directly for relative rotation. User acceleration is
    averaged only over the interval and converted from G/device axes to
    m/s^2/depth axes. A missing or stale endpoint join makes the whole motion
    unavailable, allowing the tracker to keep its constant-velocity fallback.
    """

    def __init__(self, samples, max_join_gap=0.05):
        rows = sorted((row for row in samples if np.isfinite(row.get("t", np.nan))),
                      key=lambda row: row["t"])
        self.samples = rows
        self.times = np.asarray([row["t"] for row in rows], dtype=np.float64)
        self.max_join_gap = max_join_gap

    def _nearest(self, timestamp):
        if not self.samples or timestamp is None or not np.isfinite(timestamp):
            return None
        i = int(np.searchsorted(self.times, timestamp))
        candidates = []
        if i < len(self.samples):
            candidates.append(i)
        if i > 0:
            candidates.append(i - 1)
        index = min(candidates, key=lambda j: abs(self.times[j] - timestamp))
        if abs(self.times[index] - timestamp) > self.max_join_gap:
            return None
        return self.samples[index]

    def gravity(self, timestamp):
        """Return the nearest CoreMotion gravity sample in depth coordinates."""
        sample = self._nearest(timestamp)
        if sample is None:
            return None
        try:
            value = [sample["gx"], sample["gy"], sample["gz"]]
        except (KeyError, TypeError):
            return None
        return motion_gravity_to_depth(value)

    def absolute_rotation(self, timestamp):
        """Return CoreMotion's world-from-depth rotation at ``timestamp``.

        CoreMotion's quaternion is world-from-device. The depth camera first
        changes device axes into the ARKit camera axes, then applies the
        ARKit-to-depth basis change. This is an absolute pose conversion, not
        the conjugation used below for a relative rotation.
        """
        sample = self._nearest(timestamp)
        if sample is None:
            return None
        try:
            R_world_from_device = quat_to_matrix(
                sample["qx"], sample["qy"], sample["qz"], sample["qw"])
        except (KeyError, ValueError, ZeroDivisionError):
            return None
        return R_world_from_device @ R_CAM_FROM_DEV.T @ ARKIT_TO_DEPTH

    def relative_motion(self, t0, t1):
        """Return previous-camera-from-current-camera IMU motion, or None."""
        if t0 is None or t1 is None or not (np.isfinite(t0) and np.isfinite(t1)):
            return None
        if t1 <= t0:
            return None
        start = self._nearest(t0)
        end = self._nearest(t1)
        if start is None or end is None:
            return None
        try:
            R0 = quat_to_matrix(start["qx"], start["qy"],
                                start["qz"], start["qw"])
            R1 = quat_to_matrix(end["qx"], end["qy"],
                                end["qz"], end["qw"])
        except (KeyError, ValueError, ZeroDivisionError):
            return None
        R_rel_dev = R0.T @ R1
        R_rel_cam = R_CAM_FROM_DEV @ R_rel_dev @ R_CAM_FROM_DEV.T
        R_rel_depth = ARKIT_TO_DEPTH.T @ R_rel_cam @ ARKIT_TO_DEPTH

        interval = [row for row in self.samples
                    if t0 <= row["t"] <= t1]
        acceleration = None
        if interval:
            try:
                values = np.asarray([[row["ax"], row["ay"], row["az"]]
                                     for row in interval], dtype=np.float64)
            except (KeyError, TypeError, ValueError):
                values = np.empty((0, 3))
            if len(values) and np.all(np.isfinite(values)):
                a_cam = (R_CAM_FROM_DEV @ values.mean(axis=0)
                         * IMU_ACCELERATION_MPS2)
                acceleration = ARKIT_TO_DEPTH.T @ a_cam
        return {"rotation": R_rel_depth, "acceleration": acceleration}


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", nargs="?")
    ap.add_argument("--stride", type=int, default=1,
                    help="use every Nth depth frame; larger means bigger motion "
                         "between frames, which is where ICP breaks")
    ap.add_argument("--limit", type=int, default=100000,
                    help="cap on frames used. The default no longer truncates: "
                         "a low cap silently scored only the start of a session, "
                         "which is its worst part — the phone is being raised "
                         "and tracking has not settled.")
    ap.add_argument("--min-confidence", type=int, default=1,
                    help="drop depth pixels below this ARConfidenceLevel "
                         "(0 low, 1 medium, 2 high). ARKit's depth is guided by "
                         "the colour image, so a dim room produces a full depth "
                         "map that is almost entirely level 0 — geometry the "
                         "sensor is telling you not to trust.")
    ap.add_argument("--max-dist", type=float, default=0.15,
                    help="metres; correspondences further apart are rejected")
    ap.add_argument("--frame-to-frame", action="store_true",
                    help="register against the previous frame instead of the "
                         "accumulated map. Drifts by construction; kept because "
                         "it is what the map has to beat.")
    ap.add_argument("--projective-association", action="store_true",
                    help="use the old rendered-map association instead of "
                         "voxel nearest-neighbour matching")
    ap.add_argument("--voxel", type=float, default=0.03,
                    help="metres; map resolution")
    ap.add_argument("--map-range", type=float, default=6.0,
                    help="metres; map points further than this from the current "
                         "pose are dropped, keeping this a local map")
    ap.add_argument("--keyframe-dist", type=float, default=0.05,
                    help="metres of motion before a frame is folded into the "
                         "map; 0 folds in every frame, which is what diverged")
    ap.add_argument("--keyframe-angle", type=float, default=5.0,
                    help="degrees of rotation before a frame is folded in")
    keyframe_image = ap.add_mutually_exclusive_group()
    keyframe_image.add_argument("--keyframe-on-image", dest="keyframe_on_image",
                                action="store_true", default=False,
                                help="force every frame with an image to be a "
                                     "map keyframe")
    keyframe_image.add_argument("--no-keyframe-on-image",
                                dest="keyframe_on_image", action="store_false",
                                help="only create image keyframes when the normal "
                                     "keyframe thresholds are met")
    ap.add_argument("--keyframe-min-inliers", type=float, default=0.0,
                    help="ICP inlier fraction a frame must reach before it is "
                         "folded into the map. Kept at 0 (off): this gate "
                         "measures the frame against the map the gate itself "
                         "maintains, so refusing frames starves the map, which "
                         "lowers the next frame's inlier fraction, which "
                         "refuses more. At 0.92 the median goes from 96-99% to "
                         "50-83% and loop error is 4.6x worse. See docs/POSE.md")
    ap.add_argument("--match-radius-probe", action="store_true",
                    help="also report the map match fraction at wider radii, "
                         "against the same map at the same predicted pose. "
                         "Separates a frame the map does not cover from a frame "
                         "whose predicted pose is wrong. Observation only, and "
                         "it triples the association cost")
    ap.add_argument("--raw-depth", action="store_true",
                    help="skip the depth smoothing. Kept because it is a large "
                         "effect and should be visible rather than assumed")
    ap.add_argument("--rotation-prior", action="store_true",
                    help="seed ICP with ARKit's frame-to-frame rotation and let "
                         "it solve translation from there — the architecture a "
                         "real system uses, where the gyro carries rotation")
    ap.add_argument("--imu", action="store_true",
                    help="use CoreMotion attitude and acceleration for the "
                         "motion prediction; without it, retain constant velocity")
    ap.add_argument("--imu-rotation", action="store_true",
                    help="take world-from-camera rotation from CoreMotion and "
                         "solve only translation in ICP")
    ap.add_argument("--photometric", action="store_true",
                    help="add grayscale image alignment between consecutive "
                         "written image frames")
    ap.add_argument("--photometric-reference", choices=("keyframe", "previous"),
                    default="keyframe",
                    help="image reference for the photometric term: the latest "
                         "map keyframe (default) or the previous image frame")
    ap.add_argument("--photometric-stride", type=int, default=1,
                    help="use every Nth available image pair for the "
                         "photometric term (default 1)")
    ap.add_argument("--photometric-self-test", action="store_true",
                    help="run the synthetic 10/18 cm photometric truth test "
                         "and exit")
    ap.add_argument("--no-depth", "--photometric-only", action="store_true",
                    dest="no_depth",
                    help="disable the depth term (useful for the photometric-only "
                         "ablation)")
    ap.add_argument("--depth-weight", type=float, default=1.0,
                    help="weight of the robustly normalised depth term "
                         "(default 1.0)")
    ap.add_argument("--photometric-weight", type=float, default=1.0,
                    help="weight of the robustly normalised photometric term "
                         "(default 1.0)")
    ap.add_argument("--floor-lock", action="store_true",
                    help="estimate the floor from depth and softly constrain "
                         "camera height using pose gravity")
    ap.add_argument("--gravity-lock", action="store_true",
                    help="restore roll and pitch from the first frame's gravity")
    ap.add_argument("--gravity-lock-gain", "--gravity-gain", type=float,
                    default=GRAVITY_LOCK_GAIN, dest="gravity_lock_gain",
                    help="fraction of the gravity correction applied per frame "
                         "(0..1; default 1.0)")
    ap.add_argument("--gravity-anchor", action="store_true",
                    help="anchor the ICP solve toward a gravity-corrected "
                         "motion prediction")
    ap.add_argument("--gravity-anchor-lambda", "--gravity-lambda", type=float,
                    default=GRAVITY_ANCHOR_LAMBDA,
                    dest="gravity_anchor_lambda",
                    help="eigen-direction damping toward the gravity prior "
                         "(default 0.02)")
    ap.add_argument("--dump-poses", metavar="PATH",
                    help="write the estimated and reference trajectories to an "
                         "npz, so another estimator can be scored against the "
                         "same frames through the same conventions. Anything "
                         "that rebuilds the ARKit reference itself is deciding "
                         "the axis convention a second time, which this "
                         "repository has already got wrong once")
    ap.add_argument("--include-unconverged", action="store_true",
                    help="keep frames whose ARKit tracking had not converged. "
                         "They are dropped by default, as tools/export_episodes.py "
                         "already drops them: their pose is parked near the origin "
                         "and scoring against it measures the reference, not ICP")
    args = ap.parse_args(argv)
    if args.photometric_stride < 1:
        ap.error("--photometric-stride must be a positive integer")

    if args.photometric_self_test or args.photometric:
        if not photometric_synthetic_truth_test():
            print("! photometric synthetic truth self-test failed",
                  file=sys.stderr)
            return 2
        if args.photometric_self_test:
            print("photometric synthetic truth self-test: PASS")
            return 0
    if args.session is None:
        ap.error("the following arguments are required: session")
    if args.photometric and not args.imu_rotation:
        print("  ! --photometric without --imu-rotation uses a translation-only "
              "image Jacobian inside the 6-DoF path; rotation is not constrained "
              "by this term")
    if args.photometric and not _relative_pose_self_test():
        print("! relative T_wc composition self-test failed", file=sys.stderr)
        return 2
    if args.photometric and not _image_gradient_self_test():
        print("! image-gradient axis self-test failed", file=sys.stderr)
        return 2

    session = Session(args.session)
    motion = session.motion() if (
        args.imu or args.imu_rotation or args.floor_lock or args.gravity_lock
        or args.gravity_anchor) else []
    motion_stream = IMUStream(motion) if motion else None
    imu = motion_stream if args.imu else None
    index = session.depth_index()
    if not index:
        print("! no depth in this session", file=sys.stderr)
        return 1

    poses = {p["frame"]: p for p in session.poses()}
    candidates = index[::args.stride]
    matched = [e for e in candidates if e.get("frame") in poses]
    # ARKit parks the pose near the origin until tracking converges, so an
    # unconverged frame contributes a fictitious jump to the reference
    # trajectory — inflating the distance travelled and the per-frame motion
    # ICP is being scored against. `tools/export_episodes.py` already drops
    # these; scoring kept them, which made the first seconds of every session
    # look like an estimator failure rather than a reference that had not
    # started yet.
    if args.include_unconverged:
        entries, unconverged = matched, 0
    else:
        entries = [e for e in matched if poses[e["frame"]]["tracking"] == "normal"]
        unconverged = len(matched) - len(entries)
    truncated = len(entries) > args.limit
    entries = entries[:args.limit]
    if len(entries) < 5:
        print("! too few depth frames with matching poses", file=sys.stderr)
        return 1

    span = index[-1]["t"] - index[0]["t"]
    rate = (len(index) - 1) / span if span > 0 else 0.0
    duration = entries[-1]["t"] - entries[0]["t"]
    # The rate ICP actually saw, which is not the rate the depth arrived at:
    # --stride decimates it, an unposed or unconverged frame removes it, and
    # every number below is a function of the spacing that survives. Reporting
    # only the arrival rate made a 5 Hz measurement indistinguishable from a
    # 30 Hz one in the output, which is how a result gets written up at the
    # wrong rate.
    scored = (len(entries) - 1) / duration if duration > 0 else 0.0
    print(f"{session.id}: {len(entries)} depth frames, stride {args.stride}")
    m = session.manifest
    cfg = m.get("config", {})
    print(f"  app {m.get('appVersion','?')} ({m.get('appBuild','?')})   "
          f"configured stills {cfg.get('stillsHz','?')} Hz, depth {cfg.get('depthHz','?')} Hz")
    print(f"  depth arrived at {rate:.1f} Hz, scored at {scored:.1f} Hz")
    probe = index[len(index) // 2]
    if probe.get("confidenceOffset") is not None:
        c = np.bincount(np.asarray(session.confidence_frame(probe)).ravel(), minlength=3)[:3]
        frac = c / max(c.sum(), 1)
        print(f"  depth confidence: low {frac[0] * 100:.0f}%  "
              f"medium {frac[1] * 100:.0f}%  high {frac[2] * 100:.0f}%")
        if frac[0] > 0.8:
            print("  ! ARKit rates almost all of this depth as low confidence. Its "
                  "depth is guided by the colour image, so a dim room yields a "
                  "full-looking map the sensor does not stand behind — odometry "
                  "scored on it measures the lighting, not the method. Re-record "
                  "in good light.")
    else:
        print("  depth confidence: not recorded "
              "(Settings -> Depth confidence map, and worth having)")
    if unconverged:
        print(f"  dropped {unconverged} frame(s) whose ARKit tracking had not "
              f"converged — their pose is not a reference to score against "
              f"(--include-unconverged keeps them)")
    unposed = len(candidates) - len(matched)
    if unposed:
        print(f"  ! {unposed} of {len(candidates)} depth frames carry no pose "
              f"— the frame join is failing, and what is left is a decimation "
              f"of the capture rather than the capture")
    if truncated:
        print(f"  ! using {len(entries)} of {len(candidates)} frames "
              f"— --limit is truncating, and the start of a session is its "
              f"worst part")
    # Dropping frames leaves holes, and ICP has to cross each one in a single
    # step. A hole several frame-intervals wide is a harder registration than
    # anything the rate above suggests, so it is said out loud.
    if len(entries) > 2:
        gaps = np.diff([e["t"] for e in entries])
        if scored > 0 and gaps.max() > 4.0 / scored:
            print(f"  ! largest gap between scored frames is {gaps.max():.2f} s "
                  f"({gaps.max() * scored:.0f}x the median spacing) — ICP has "
                  f"to cross that in one step")
    thermal = [e for e in session.events() if e["kind"].startswith("thermal")]
    if thermal:
        print(f"  ! {len(thermal)} thermal event(s) — capture may have been "
              f"paused mid-session")
    if scored < 12:
        print(f"  ! that is 5 Hz-class spacing. ICP converges on small motion, "
              f"so this is the wrong data to judge it on — re-record at 30 Hz, "
              f"and check --stride is not decimating it away.")

    frame_rows = session.frames()
    image_rows = {row["frame"]: row for row in frame_rows}
    depth_width = int(entries[0]["width"])
    depth_height = int(entries[0]["height"])
    if frame_rows:
        colour_width = int(frame_rows[0]["width"])
        colour_height = int(frame_rows[0]["height"])
        intrinsics_source = "frames.jsonl"
    else:
        # Video sessions have no frames.jsonl. Recover only the colour width
        # from the quoted principal point, rounded to the capture formats'
        # 16-pixel granularity; preserve the depth aspect ratio for height.
        first_pose = poses[entries[0]["frame"]]
        colour_width = max(16, int(round((2.0 * first_pose["cx"]) / 16.0) * 16))
        colour_height = max(1, int(round(
            colour_width * depth_height / max(depth_width, 1))))
        intrinsics_source = "video fallback from 2*cx rounded to 16"
    sx = depth_width / float(colour_width)
    sy = depth_height / float(colour_height)
    print(f"  depth intrinsics: {intrinsics_source}, colour "
          f"{colour_width}x{colour_height}, depth {depth_width}x{depth_height}, "
          f"scale sx={sx:.9f} sy={sy:.9f}")

    tracker = Tracker(frame_to_frame=args.frame_to_frame,
                      max_dist=args.max_dist, voxel=args.voxel,
                      map_range=args.map_range,
                      keyframe_dist=args.keyframe_dist,
                      keyframe_angle=args.keyframe_angle,
                      projective_association=args.projective_association,
                      imu=imu, floor_lock=args.floor_lock,
                      gravity_lock=args.gravity_lock,
                      gravity_lock_gain=args.gravity_lock_gain,
                      gravity_anchor=args.gravity_anchor,
                      gravity_anchor_lambda=args.gravity_anchor_lambda,
                      imu_rotation=args.imu_rotation,
                      depth_weight=(0.0 if args.no_depth else args.depth_weight),
                      photometric_weight=args.photometric_weight,
                      photometric_reference=args.photometric_reference,
                      keyframe_on_image=args.keyframe_on_image,
                      keyframe_min_inliers=args.keyframe_min_inliers,
                      match_radius_probe=args.match_radius_probe)
    ref = []                    # ARKit, same frames, in the depth convention
    frame_data = []             # prepared depth frames, reused for consistency
    frame_conditioning_values = []
    rel_err_t, rel_err_r, moved = [], [], []
    image_cache = {}
    last_photo = None
    photo_requested = 0
    photo_pairs_available = 0
    photo_pair_index = 0
    photo_image_frames = 0
    photo_reference_frames = 0
    photo_reference_skips = 0
    photo_moved = []
    image_error_reported = False

    for i, entry in enumerate(entries):
        depth = np.asarray(session.depth_frame(entry), dtype=np.float64)
        if args.min_confidence > 0 and entry.get("confidenceOffset") is not None:
            conf = np.asarray(session.confidence_frame(entry))
            depth = np.where(conf >= args.min_confidence, depth, np.nan)
        p = poses[entry["frame"]]
        # Intrinsics are quoted for the full-resolution colour frame. The
        # depth map is a uniformly resized view, so use the data dimensions,
        # never the imperfect principal point, to derive both axes' scale.
        frame_sx = depth.shape[1] / float(colour_width)
        frame_sy = depth.shape[0] / float(colour_height)
        K = (p["fx"] * frame_sx, p["fy"] * frame_sy,
             p["cx"] * frame_sx, p["cy"] * frame_sy)

        pts, nrm, ok = frame_points(depth, K, smooth=not args.raw_depth)
        frame_data.append((pts, nrm, ok, K))
        frame_cond, _ = frame_conditioning(pts, nrm, ok)
        frame_conditioning_values.append(frame_cond)

        R = quat_to_matrix(p["qx"], p["qy"], p["qz"], p["qw"]) @ ARKIT_TO_DEPTH
        A = np.eye(4)
        A[:3, :3] = R
        A[:3, 3] = [p["tx"], p["ty"], p["tz"]]
        ref.append(A)

        prior = None
        if args.rotation_prior and len(ref) > 1:
            # Rotation only. Over 0.2 s a gyro is essentially drift-free, so
            # taking it from ARKit stands in for an IMU without smuggling in
            # the translation that is the thing under test.
            # Transposed: the relative reference pose is previous-from-current,
            # and ICP works in current-from-previous. Seeding it the other way
            # round starts the solve at twice the wrong rotation, which is worse
            # than starting at identity — and looked like evidence against the
            # method rather than a bug in the harness.
            prior = relative_camera_transform(ref[-2], ref[-1])[:3, :3]
        # pose.jsonl stores gravity in the ARKit camera frame (-Z forward, +Y
        # up). The tracker consumes depth coordinates (+Z forward, +Y down),
        # so apply the same measured basis change as the pose. On the
        # ultra-wide path ARKit is absent; motion.jsonl's device gravity is
        # transformed through R_CAM_FROM_DEV and ARKIT_TO_DEPTH instead.
        gravity = arkit_gravity_to_depth(
            [p.get("gravX"), p.get("gravY"), p.get("gravZ")])
        if gravity is None and motion_stream is not None:
            gravity = motion_stream.gravity(entry["t"])
        imu_rotation = (motion_stream.absolute_rotation(entry["t"])
                        if args.imu_rotation and motion_stream is not None
                        else None)
        photometric_pair = None
        previous_image_pose = None
        photometric_current = None
        photometric_reference_pose = None
        image_entry = image_rows.get(entry["frame"])
        current_image = None
        current_image_K = None
        if ((args.photometric or args.keyframe_on_image)
                and image_entry is not None):
            try:
                current_image = load_photometric_image(
                    session.frame_path(image_entry), image_cache)
                current_image_K = photometric_image_intrinsics(p, image_entry)
                photo_image_frames += 1
                if args.photometric_reference == "keyframe":
                    reference_photo = tracker.photo_reference
                else:
                    reference_photo = last_photo
                # --keyframe-on-image also needs the image, but only to know
                # which frames are image frames. Building a pair here without
                # --photometric would switch the image term on for a run that
                # never asked for it, and the photometric report is printed
                # under `if args.photometric`, so it would run unannounced.
                if not args.photometric:
                    reference_photo = None
                if len(ref) > 1 and reference_photo is not None:
                    photo_pairs_available += 1
                    photo_reference_frames += 1
                    photometric_reference_pose = reference_photo["pose"]
                    if args.photometric_reference == "keyframe":
                        photo_moved.append(float(np.linalg.norm(
                            relative_camera_transform(
                                ref[-1], photometric_reference_pose)[:3, 3])))
                    else:
                        photo_moved.append(float(np.linalg.norm(
                            relative_camera_transform(
                                ref[-1], reference_photo["ref_pose"])[:3, 3])))
                    if photo_pair_index % args.photometric_stride == 0:
                        photo_requested += 1
                        photometric_pair = {
                            "previous_image": reference_photo["image"],
                            "current_image": current_image,
                            "previous_K": reference_photo["K"],
                            "current_K": current_image_K,
                        }
                        previous_image_pose = photometric_reference_pose
                    photo_pair_index += 1
                elif len(ref) > 1:
                    photo_reference_skips += 1
                photometric_current = {
                    "image": current_image,
                    "K": current_image_K,
                    "frame": entry["frame"],
                }
            except (OSError, RuntimeError, ValueError) as exc:
                if not image_error_reported:
                    print(f"  ! photometric image unavailable: {exc}")
                    image_error_reported = True

        tracker.step(pts, nrm, ok, K, rotation_prior=prior,
                     timestamp=entry["t"], gravity=gravity,
                     imu_rotation=imu_rotation,
                     photometric_pair=photometric_pair,
                     previous_image_pose=previous_image_pose,
                     photometric_current=photometric_current,
                     photometric_reference_pose=photometric_reference_pose)

        if (current_image is not None
                and args.photometric_reference == "previous"):
            last_photo = {"image": current_image, "K": current_image_K,
                          "pose": tracker.poses[-1].copy(),
                          "ref_pose": ref[-1].copy(),
                          "frame": entry["frame"]}

        if len(ref) > 1:
            truth = relative_camera_transform(ref[-1], ref[-2])
            got = relative_camera_transform(tracker.poses[-1],
                                            tracker.poses[-2])
            rel_err_t.append(float(np.linalg.norm(truth[:3, 3] - got[:3, 3])))
            moved.append(float(np.linalg.norm(truth[:3, 3])))
            dR = truth[:3, :3].T @ got[:3, :3]
            rel_err_r.append(math.degrees(
                math.acos(max(-1.0, min(1.0, (np.trace(dR) - 1) / 2)))))

    est = tracker.poses
    inliers = tracker.inliers
    est_p = np.array([T[:3, 3] for T in est])
    ref_p = np.array([T[:3, 3] for T in ref])
    if args.dump_poses:
        # Both trajectories as 4x4 world-from-camera in the depth convention
        # (+Z forward, +Y down), already through ARKIT_TO_DEPTH, alongside the
        # frame indices and timestamps they belong to. A second estimator scored
        # against this file inherits the convention instead of choosing one.
        np.savez(args.dump_poses,
                 estimate=np.asarray(est, dtype=np.float64),
                 reference=np.asarray(ref, dtype=np.float64),
                 frame=np.asarray([e["frame"] for e in entries], dtype=np.int64),
                 t=np.asarray([e["t"] for e in entries], dtype=np.float64),
                 inliers=np.asarray(inliers, dtype=np.float64),
                 conditioning=np.asarray(tracker.conditioning, dtype=np.float64),
                 convention="world_from_camera, +Z forward +Y down (depth frame)")
        print(f"  wrote {len(est)} estimated and {len(ref)} reference poses to "
              f"{args.dump_poses}")
    travelled = float(np.linalg.norm(np.diff(ref_p, axis=0), axis=1).sum())

    print(f"  ARKit travelled {travelled:.2f} m over {duration:.1f} s")
    print(f"  ICP inliers: median {np.median(inliers) * 100:.0f}% "
          f"(min {min(inliers) * 100:.0f}%)")
    if not args.frame_to_frame:
        print(f"  keyframe on image: {'on' if args.keyframe_on_image else 'off'}")
        print(f"  map: {tracker.keyframes} keyframes of {len(entries)} frames, "
              f"{tracker.image_keyframes} with images "
              f"({tracker.image_keyframes / max(len(entries), 1) * 100:.1f}%), "
              f"{len(tracker.map)} voxels")
    if args.imu:
        mode = ("frame-to-frame rotation prior" if args.frame_to_frame
                else "motion prediction")
        print(f"  IMU {mode}: used {tracker.imu_uses} frame(s), "
              f"constant-velocity fallback {tracker.imu_fallbacks} frame(s)")
    if args.imu_rotation:
        print(f"  IMU absolute rotation: used {tracker.imu_rotation_uses} frame(s), "
              f"fallback {tracker.imu_rotation_fallbacks} frame(s)")
        if tracker.conditioning:
            translation_cond = np.asarray(tracker.conditioning, dtype=np.float64)
            print(f"  translation-only ICP conditioning (3x3): median "
                  f"{np.median(translation_cond):.3g}  "
                  f"p10 {np.percentile(translation_cond, 10):.3g}  "
                  f"degenerate {(translation_cond <= tracker.min_conditioning).sum()}"
                  f"/{len(translation_cond)}")
    # Report on the term whenever it actually ran, not only when it was asked
    # for. `--keyframe-on-image` once switched it on without `--photometric`,
    # and because this block was gated on the flag rather than on the tracker,
    # the run said nothing at all — the arm meant to isolate map density was
    # silently the same computation as the arm it was the control for.
    if args.photometric or tracker.photometric_uses:
        if not args.photometric:
            print("  ! photometric term ran without --photometric")
        depth_scale = (float(np.median(tracker.depth_scales))
                       if tracker.depth_scales else None)
        photo_scale = (float(np.median(tracker.photometric_scales))
                       if tracker.photometric_scales else None)
        scale_ratio = (depth_scale / photo_scale
                       if depth_scale is not None and photo_scale is not None
                       and photo_scale > 0.0 else None)
        depth_text = "n/a" if depth_scale is None else f"{depth_scale:.4g} m"
        photo_text = "n/a" if photo_scale is None else f"{photo_scale:.4g} gray"
        ratio_text = "n/a" if scale_ratio is None else f"{scale_ratio:.4g}"
        level_text = ("n/a" if not tracker.photometric_level_shapes else
                      " -> ".join(
                          f"{width}x{height}@{scale:.8g}"
                          for (width, height), scale in zip(
                              tracker.photometric_level_shapes,
                              tracker.photometric_level_scales)))
        print(f"  photometric: evaluated {tracker.photometric_uses}/{len(entries)} "
              f"frames ({tracker.photometric_uses / max(len(entries), 1) * 100:.0f}%), "
              f"evaluations {tracker.photometric_evaluations}, "
              f"accepted LM steps {tracker.photometric_accepted_steps}, "
              f"gray {PHOTOMETRIC_WIDTH}x{PHOTOMETRIC_HEIGHT}, "
              f"image frames {photo_image_frames}/{len(entries)}, "
              f"pairs {photo_pairs_available} available/{photo_requested} selected/"
              f"{tracker.photometric_uses} processed, stride {args.photometric_stride}, "
              f"convergence failures {tracker.photometric_convergence_failures}/"
              f"{tracker.photometric_attempts}, "
              f"levels {tracker.photometric_level_count} ({level_text}), "
              f"residual scales depth/photo {depth_text}/{photo_text} "
              f"(ratio {ratio_text})")
        reference_label = ("keyframe" if args.photometric_reference == "keyframe"
                           else "previous-image")
        print(f"  photometric reference: {reference_label} "
              f"{photo_reference_frames} frames, no reference "
              f"{photo_reference_skips} skipped, total {len(entries)}")
        if tracker.photometric_reference_distances:
            print(f"  photometric reference distance: median "
                  f"{np.median(tracker.photometric_reference_distances):.3f} m  "
                  f"angle median "
                  f"{np.median(tracker.photometric_reference_angles):.2f}°")
        else:
            print("  photometric reference distance: n/a  angle median n/a")
    if args.floor_lock:
        floor = tracker.floor_report(len(entries))
        if floor["median"] is None:
            print(f"  floor lock: visible 0/{floor['frames']} frames, "
                  f"correction {floor['correction']:.3f} m")
        else:
            print(f"  floor lock: visible {floor['visible']}/{floor['frames']} "
                  f"frames ({floor['visible'] / max(floor['frames'], 1) * 100:.0f}%), "
                  f"height median {floor['median']:.2f} m, "
                  f"variation {floor['variation']:.2f} m, "
                  f"correction {floor['correction']:.3f} m")
    gravity_report = tracker.gravity_report()
    if gravity_report is not None:
        gravity_mode = ("locked" if args.gravity_lock
                        else "anchored" if args.gravity_anchor else "unlocked")
        print(f"  gravity drift ({gravity_mode}): "
              f"end {gravity_report['end']:.2f}°  "
              f"max {gravity_report['maximum']:.2f}°"
              + (f"  gain {args.gravity_lock_gain:.2f}" if args.gravity_lock else "")
              + (f"  lambda {args.gravity_anchor_lambda:.3g}"
                 if args.gravity_anchor else ""))
    # How often the view left an axis unmeasured. This is the failure mode the
    # method actually has indoors — a corridor, or a wall at arm's length — and
    # without it a bad number looks like bad code rather than bad geometry.
    weak = sum(1 for c in tracker.conditioning if c <= tracker.min_conditioning)
    if weak:
        print(f"  ! {weak} of {len(tracker.conditioning)} frames were "
              f"under-constrained — a face of the room out of view leaves an "
              f"axis unobservable, and those frames coast on the prediction")
    frame_cond = np.asarray(frame_conditioning_values, dtype=np.float64)
    frame_weak = frame_cond <= tracker.min_conditioning
    print(f"  frame-only conditioning  median {np.median(frame_cond):.3g}  "
          f"p10 {np.percentile(frame_cond, 10):.3g}  "
          f"degenerate {frame_weak.sum()}/{len(frame_cond)} "
          f"({frame_weak.mean() * 100:.0f}%)")

    estimate_residual = trajectory_point_to_plane_residual(
        est, frame_data, max_dist=args.max_dist)
    arkit_residual = trajectory_point_to_plane_residual(
        ref, frame_data, max_dist=args.max_dist)
    common = [i for i, (estimate, arkit) in enumerate(
        zip(estimate_residual["per_frame"], arkit_residual["per_frame"]))
              if i > 0 and estimate is not None and arkit is not None]
    if common:
        estimate_mm = float(np.median(
            [estimate_residual["per_frame"][i] for i in common])) * 1000
        arkit_mm = float(np.median(
            [arkit_residual["per_frame"][i] for i in common])) * 1000
        weak_common = sum(
            tracker.conditioning[i - 1] <= tracker.min_conditioning
            for i in common)
        print(f"  point-to-plane residual   estimate {estimate_mm:.1f} mm   "
              f"arkit {arkit_mm:.1f} mm   frames {len(common)}")
        print(f"  weak conditioning         {weak_common}/{len(common)} frames "
              f"({weak_common / len(common) * 100:.0f}%)")
        if len(common) < len(entries) - 1:
            print(f"  ! residual comparison used {len(common)} of "
                  f"{len(entries) - 1} frame pairs — both trajectories "
                  f"had to have the same valid matches")
    else:
        print("  point-to-plane residual   unavailable — no common frame pairs "
              "with valid matches")
    print()
    print("relative pose error, frame to frame — the honest measure for "
          "odometry")
    print(f"  translation  median {np.median(rel_err_t) * 100:.1f} cm   "
          f"p90 {np.percentile(rel_err_t, 90) * 100:.1f} cm")
    print(f"  rotation     median {np.median(rel_err_r):.2f}°   "
          f"p90 {np.percentile(rel_err_r, 90):.2f}°")
    conditioning = np.asarray(tracker.conditioning, dtype=np.float64)
    translation_errors = np.asarray(rel_err_t, dtype=np.float64)
    rotation_errors = np.asarray(rel_err_r, dtype=np.float64)
    if len(tracker.photometric_used_history) == len(translation_errors):
        photo_applied = np.asarray(tracker.photometric_used_history, dtype=bool)
    else:
        photo_applied = np.zeros(len(translation_errors), dtype=bool)
    for label, selected in (
            ("cond < 0.01", conditioning < 0.01),
            ("0.01 <= cond < 0.05", ((conditioning >= 0.01)
                                     & (conditioning < 0.05))),
            ("cond >= 0.05", conditioning >= 0.05)):
        if not selected.any():
            print(f"  {label:12} 0 frames")
            continue
        print(f"  {label:12} {selected.sum()} frames  translation median "
              f"{np.median(translation_errors[selected]) * 100:.1f} cm "
              f"p90 {np.percentile(translation_errors[selected], 90) * 100:.1f} cm  "
              f"rotation median {np.median(rotation_errors[selected]):.2f}°")
    # Registration quality is a detector rather than a predictor, and the two
    # want different statistics. The inlier fraction saturates near 1 — most of
    # the mass sits in a flat band — so a correlation taken over the whole range
    # is dominated by the part that carries no signal, and reports "inert" for a
    # gate that is in fact selective. What answers the question is the
    # conditional error either side of a cut, and whether the ratio grows as the
    # cut tightens. That is the opposite failure mode from `max_dist` and the
    # conditioning gate, which bite often and mean nothing.
    inlier_fractions = np.asarray(tracker.inliers, dtype=np.float64)
    if len(inlier_fractions) == len(translation_errors):
        print(f"  ICP inliers: median {np.median(inlier_fractions) * 100:.1f}%  "
              f"p10 {np.percentile(inlier_fractions, 10) * 100:.1f}%  "
              f"min {inlier_fractions.min() * 100:.1f}%")
        for cut in (0.98, 0.95, 0.92):
            below = inlier_fractions < cut
            if not below.any():
                print(f"  inliers < {cut:.2f}   0 frames")
                continue
            low = float(np.median(translation_errors[below])) * 100
            high = (float(np.median(translation_errors[~below])) * 100
                    if (~below).any() else float("nan"))
            ratio = low / high if high > 0 else float("nan")
            print(f"  inliers < {cut:.2f}   {below.sum():4d} frames  "
                  f"translation median {low:5.2f} cm   "
                  f"vs {high:5.2f} cm at or above   ({ratio:.2f}x)")
    # Whether keyframe selection filters anything at all, asked without picking
    # a threshold: if the frames folded into the map look like the frames in
    # general, the selection is not selecting.
    keyframe_flags = np.asarray(tracker.keyframe_flags, dtype=bool)
    if (len(keyframe_flags) == len(inlier_fractions)
            and keyframe_flags.any() and not keyframe_flags.all()):
        print(f"  folded into the map      {keyframe_flags.sum()} frames  "
              f"inliers median {np.median(inlier_fractions[keyframe_flags]) * 100:.1f}%  "
              f"conditioning median {np.median(conditioning[keyframe_flags]):.3g}")
        print(f"  every scored frame       {len(keyframe_flags)} frames  "
              f"inliers median {np.median(inlier_fractions) * 100:.1f}%  "
              f"conditioning median {np.median(conditioning):.3g}")
        if tracker.keyframe_min_inliers > 0.0:
            print(f"  refused by --keyframe-min-inliers "
                  f"{tracker.keyframe_min_inliers:.2f}: "
                  f"{tracker.keyframes_refused_inliers} frames")
        radii = np.asarray(tracker.match_radius_fractions, dtype=np.float64)
        if len(radii) == len(keyframe_flags):
            print("  match fraction at tighter radii, same map and pose — a "
                  "deficit from absent map points holds steady, one from a "
                  "wrong pose widens")
            for label, selected in (("keyframes", keyframe_flags),
                                    ("other frames", ~keyframe_flags)):
                if not selected.any():
                    continue
                text = "  ".join(
                    f"{radius:.2f} m {np.median(radii[selected, i]) * 100:5.1f}%"
                    for i, radius in enumerate(MATCH_RADIUS_PROBES))
                print(f"  {label:12} {text}")
            gaps = " ".join(
                f"{radius:.2f} m {(np.median(radii[~keyframe_flags, i]) - np.median(radii[keyframe_flags, i])) * 100:+5.2f}"
                for i, radius in enumerate(MATCH_RADIUS_PROBES))
            print(f"  keyframe deficit, percentage points:  {gaps}")
    if args.photometric or tracker.photometric_uses:
        for label, selected in (("photometric applied", photo_applied),
                                ("photometric not applied", ~photo_applied)):
            if not selected.any():
                print(f"  {label:24} 0 frames")
                continue
            print(f"  {label:24} {selected.sum()} frames  translation median "
                  f"{np.median(translation_errors[selected]) * 100:.1f} cm "
                  f"p90 {np.percentile(translation_errors[selected], 90) * 100:.1f} cm")
    # Without this the translation error is a number rather than a verdict:
    # reporting "did not move" scores exactly the distance actually moved, so
    # anything above that line is worse than no estimate at all.
    baseline = float(np.median(moved))
    print(f"  the frames are {baseline * 100:.1f} cm apart (median), so "
          f"'assume no motion' scores {baseline * 100:.1f} cm")
    speed = travelled / duration if duration > 0 else 0.0
    if args.photometric or tracker.photometric_uses:
        image_motion = np.asarray(photo_moved, dtype=np.float64)
        motion_text = ("n/a" if not len(image_motion) else
                       f"{np.median(image_motion) * 100:.1f} cm median, "
                       f"{np.max(image_motion) * 100:.1f} cm max")
        print(f"  photometric scope: {speed:.2f} m/s average walk, "
              f"{motion_text} per image pair; this eight-session set is "
              f"slow-walk only and does not establish fast-walk performance")
    if speed < 0.15:
        # Below walking pace the frames barely differ, so per-frame error is
        # sensor noise rather than tracking failure and the baseline comparison
        # says nothing. Reporting it as a verdict here would condemn a working
        # estimator for the crime of being measured while standing still.
        print(f"  (only {speed:.2f} m/s — too close to stationary for that "
              f"comparison to mean anything; walk a few metres to test this)")
    elif np.median(rel_err_t) > baseline:
        print("  ! ICP is worse than assuming no motion — it is diverging, "
              "not merely imprecise")

    # Absolute drift, after putting both trajectories in the same frame. Rigid
    # alignment only — no scale term, because both are metric and a scale fit
    # would hide exactly the failure worth seeing.
    ec, rc = est_p - est_p.mean(0), ref_p - ref_p.mean(0)
    U, _, Vt = np.linalg.svd(ec.T @ rc)
    d = np.sign(np.linalg.det(U @ Vt))
    Ropt = U @ np.diag([1, 1, d]) @ Vt
    ate = np.linalg.norm(ec @ Ropt - rc, axis=1)
    print()
    print("absolute trajectory error after rigid alignment")
    print(f"  rms {ate.mean() * 100:.1f} cm   max {ate.max() * 100:.1f} cm")
    # The keyframes have a lower inlier fraction than frames in general, in
    # every session. That has two readings which produce the same number: they
    # overlap the map less because they are taken after the camera has moved
    # (harmless — it is what keyframing is), or they are genuinely worse poses
    # (fatal to any design that anchors a reference to them). Pose error
    # separates the two, and keyframes are interleaved along the whole
    # trajectory, so accumulated drift is common to both groups.
    if len(keyframe_flags) == len(ate) - 1 or len(keyframe_flags) == len(ate):
        flags = keyframe_flags[:len(ate)] if len(keyframe_flags) >= len(ate) \
            else np.concatenate([[False], keyframe_flags])
        if flags.any() and not flags.all():
            print(f"  keyframes    {flags.sum():4d} frames  "
                  f"absolute median {np.median(ate[flags]) * 100:5.1f} cm")
            print(f"  other frames {(~flags).sum():4d} frames  "
                  f"absolute median {np.median(ate[~flags]) * 100:5.1f} cm")
    estimate_loop = float(np.linalg.norm(est_p[-1] - est_p[0]))
    reference_loop = float(np.linalg.norm(ref_p[-1] - ref_p[0]))
    print(f"  end-start distance {estimate_loop:.3f} m "
          f"(ARKit {reference_loop:.3f} m; score)")
    # Per second rather than per metre, so it is comparable to the ~0.02 m/s
    # that published benchmarks measure for ARKit itself. Percentage-of-distance
    # is meaningless on a near-stationary clip and invites reading 130% as a
    # catastrophe when the distance was half a metre.
    print(f"  drift {ate.mean() / max(duration, 1e-6) * 100:.1f} cm/s "
          f"against ARKit, over {duration:.1f} s and {travelled:.2f} m "
          f"(ARKit's own published drift is ~2 cm/s)")

    if np.median(inliers) < 0.3:
        print("\n! ICP is barely associating points. Likely the scene is beyond "
              "the sensor's ~5 m range, or motion between frames is too large — "
              "try --stride 1 or a slower capture.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
