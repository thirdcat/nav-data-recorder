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


def icp(src_pts, src_ok, dst_pts, dst_normals, dst_ok, K, iters=20,
        max_dist=0.15, prior=None, rcond=1e-3, anchor=None):
    """Point-to-plane ICP with projective association.

    Correspondences come from projecting a transformed source point into the
    target's image and taking whatever pixel it lands on. That is the
    KinectFusion trick: it is O(1) per point with no spatial index, and on
    depth maps — which are already an image — it is what the data is shaped for.

    Returns the 4x4 transform taking source camera coordinates into target
    camera coordinates, the fraction of points that found a match, and how well
    the geometry constrains the answer — the smallest eigenvalue of the
    point-to-plane Hessian as a fraction of the largest, so 1 is a view that
    pins all six degrees of freedom and 0 is one that leaves an axis free.
    """
    T = np.eye(4) if prior is None else prior.copy()

    inlier_frac = 0.0
    conditioning = 0.0
    last_H = None
    for _ in range(iters):
        matches = _projective_matches(src_pts, src_ok, dst_pts, dst_normals,
                                      dst_ok, K, T, max_dist)
        if matches is None:
            break
        qa, target, na, inlier_frac = matches
        # Linearised point-to-plane residual: for a small rotation w and
        # translation t, minimise sum over points of
        # ((q + w x q + t - target) . n)^2. The Jacobian row is [q x n, n].
        A = np.hstack([np.cross(qa, na), na])
        b = np.einsum("ij,ij->i", (target - qa), na)

        # Solve the normal equations through their eigendecomposition rather
        # than by least squares, so the *shape* of the solution is visible and
        # not just its value. A view of two walls with the floor and ceiling out
        # of frame leaves vertical motion completely unobserved; `lstsq` will
        # still answer, confidently and enormously, and one such frame takes the
        # whole trajectory with it. Directions the geometry does not constrain
        # are dropped from the update instead, which leaves them at whatever the
        # motion prediction said — declining to invent motion nothing was seen
        # to support. It is a 6x6 problem, so this costs nothing.
        try:
            H = A.T @ A / len(A)
            g = A.T @ b / len(A)
            ev, evec = np.linalg.eigh(H)
        except np.linalg.LinAlgError:
            break
        last_H = H
        # Clamped at zero: a singular Hessian comes back with a faintly negative
        # smallest eigenvalue, and a negative "fraction" reads as a bug rather
        # than as the blindness it is.
        conditioning = max(0.0, float(ev[0] / ev[-1])) if ev[-1] > 1e-12 else 0.0
        usable = ev > ev[-1] * rcond
        if not usable.any():
            break
        V = evec[:, usable]
        x = V @ ((V.T @ g) / ev[usable])
        # An implausible step is still rejected rather than applied: the cut
        # above removes directions that are hopeless, and this catches the ones
        # that are merely bad.
        if not np.all(np.isfinite(x)) or np.linalg.norm(x[3:]) > max_dist * 3:
            break
        wx, wy, wz = x[:3]
        dR = np.array([[1, -wz, wy], [wz, 1, -wx], [-wy, wx, 1]])
        # Re-orthonormalise: the small-angle matrix above is not a rotation, and
        # composing dozens of them without this walks the estimate off SO(3).
        u_, _, vt_ = np.linalg.svd(dR)
        dR = u_ @ vt_
        step = np.eye(4)
        step[:3, :3] = dR
        step[:3, 3] = x[3:]
        T = step @ T
        if np.linalg.norm(x[:3]) < 1e-6 and np.linalg.norm(x[3:]) < 1e-6:
            break
    # Anchoring happens once, on the converged answer, rather than inside the
    # loop: damping every iteration would also slow convergence along the
    # directions that are well observed, which is the opposite of the intent.
    if anchor is not None and prior is not None and last_H is not None:
        T = anchor_to_prior(prior, T, last_H, anchor)
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
                 keyframe_fill=0.6, min_conditioning=1e-3):
        self.frame_to_frame = frame_to_frame
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
        self._velocity = np.eye(4)

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

    def step(self, pts, nrm, ok, K, rotation_prior=None):
        """Register one frame and return its world-from-camera pose."""
        if self._prev is None:
            self._prev = (pts, nrm, ok)
            self.map.integrate(pts[ok], nrm[ok], self.poses[0])
            self._last_kf = self.poses[0]
            self.keyframes = 1
            return self.poses[0]

        if self.frame_to_frame:
            prior = None
            if rotation_prior is not None:
                prior = np.eye(4)
                prior[:3, :3] = rotation_prior
            prev_pts, prev_nrm, prev_ok = self._prev
            # ICP solves target-from-source; the trajectory needs its inverse.
            T, frac, cond = icp(prev_pts, prev_ok, pts, nrm, ok, K,
                                max_dist=self.max_dist, prior=prior)
            pose = self.poses[-1] @ np.linalg.inv(T)
        else:
            # Predict where we are, render the map from there, and let ICP
            # supply the correction. A constant-velocity guess costs nothing and
            # keeps the correction inside the basin ICP converges from.
            predicted = self.poses[-1] @ self._velocity
            dst_p, dst_n, dst_o = self.map.render(predicted, K, ok.shape)
            fill = float(dst_o.mean())
            if fill < 0.02:
                # Nothing of the map is in view; fall back rather than invent.
                # Source is the *current* frame here, as it is against the map
                # below — not the previous one as in the frame-to-frame branch
                # above. So T is already previous-from-current and composes
                # directly; inverting it as well stepped backwards, moving the
                # pose by twice the motion in the wrong direction.
                prev_pts, prev_nrm, prev_ok = self._prev
                T, frac, cond = icp(pts, ok, prev_pts, prev_nrm, prev_ok, K,
                                    max_dist=self.max_dist)
                pose = self.poses[-1] @ T
            else:
                T, frac, cond = icp(pts, ok, dst_p, dst_n, dst_o, K,
                                    max_dist=self.max_dist)
                # T maps current-camera coords into the predicted frame, so the
                # actual pose is the prediction composed with it.
                pose = predicted @ T
            self._velocity = np.linalg.inv(self.poses[-1]) @ pose
            # Keyframes only, and never from a pose the geometry could not pin
            # down: that frame's position along the free axis is a guess, and
            # folding it in writes the guess into the map for every later frame
            # to register against. Folding in every frame re-inserts the same
            # surface at a slightly different estimated pose dozens of times a
            # second, which thickens it faster than averaging can sharpen it.
            if cond > self.min_conditioning and self._keyframe_due(pose, fill):
                self.map.integrate(pts[ok], nrm[ok], pose)
                self.map.trim(pose[:3, 3])
                self._last_kf = pose
                self.keyframes += 1

        self.poses.append(pose)
        self.inliers.append(frac)
        self.conditioning.append(cond)
        self._prev = (pts, nrm, ok)
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
        T = np.linalg.inv(trajectory[i - 1]) @ trajectory[i]
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


# ARKit's camera looks down -Z with +Y up; the depth map's own frame is +Z
# forward, +Y down. Columns are the depth axes in ARKit camera coordinates.
ARKIT_TO_DEPTH = np.diag([1.0, -1.0, -1.0])


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
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
    ap.add_argument("--raw-depth", action="store_true",
                    help="skip the depth smoothing. Kept because it is a large "
                         "effect and should be visible rather than assumed")
    ap.add_argument("--rotation-prior", action="store_true",
                    help="seed ICP with ARKit's frame-to-frame rotation and let "
                         "it solve translation from there — the architecture a "
                         "real system uses, where the gyro carries rotation")
    ap.add_argument("--include-unconverged", action="store_true",
                    help="keep frames whose ARKit tracking had not converged. "
                         "They are dropped by default, as tools/export_episodes.py "
                         "already drops them: their pose is parked near the origin "
                         "and scoring against it measures the reference, not ICP")
    args = ap.parse_args(argv)

    session = Session(args.session)
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

    tracker = Tracker(frame_to_frame=args.frame_to_frame,
                      max_dist=args.max_dist, voxel=args.voxel,
                      map_range=args.map_range,
                      keyframe_dist=args.keyframe_dist,
                      keyframe_angle=args.keyframe_angle)
    ref = []                    # ARKit, same frames, in the depth convention
    frame_data = []             # prepared depth frames, reused for consistency
    rel_err_t, rel_err_r, moved = [], [], []

    for i, entry in enumerate(entries):
        depth = np.asarray(session.depth_frame(entry), dtype=np.float64)
        if args.min_confidence > 0 and entry.get("confidenceOffset") is not None:
            conf = np.asarray(session.confidence_frame(entry))
            depth = np.where(conf >= args.min_confidence, depth, np.nan)
        p = poses[entry["frame"]]
        # Intrinsics are quoted for the full-resolution colour frame; the depth
        # map is a fraction of that size and shares the optical axis. `cx` sits
        # within a pixel or two of half the colour width, which is a more robust
        # way to recover the ratio than joining another stream for it.
        s = depth.shape[1] / (2.0 * p["cx"])
        K = (p["fx"] * s, p["fy"] * s, p["cx"] * s, p["cy"] * s)

        pts, nrm, ok = frame_points(depth, K, smooth=not args.raw_depth)
        frame_data.append((pts, nrm, ok, K))

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
            prior = (np.linalg.inv(ref[-2]) @ ref[-1])[:3, :3].T
        tracker.step(pts, nrm, ok, K, rotation_prior=prior)

        if len(ref) > 1:
            truth = np.linalg.inv(ref[-2]) @ ref[-1]
            got = np.linalg.inv(tracker.poses[-2]) @ tracker.poses[-1]
            rel_err_t.append(float(np.linalg.norm(truth[:3, 3] - got[:3, 3])))
            moved.append(float(np.linalg.norm(truth[:3, 3])))
            dR = truth[:3, :3].T @ got[:3, :3]
            rel_err_r.append(math.degrees(
                math.acos(max(-1.0, min(1.0, (np.trace(dR) - 1) / 2)))))

    est = tracker.poses
    inliers = tracker.inliers
    est_p = np.array([T[:3, 3] for T in est])
    ref_p = np.array([T[:3, 3] for T in ref])
    travelled = float(np.linalg.norm(np.diff(ref_p, axis=0), axis=1).sum())

    print(f"  ARKit travelled {travelled:.2f} m over {duration:.1f} s")
    print(f"  ICP inliers: median {np.median(inliers) * 100:.0f}% "
          f"(min {min(inliers) * 100:.0f}%)")
    if not args.frame_to_frame:
        print(f"  map: {tracker.keyframes} keyframes of {len(entries)} frames, "
              f"{len(tracker.map)} voxels")
    # How often the view left an axis unmeasured. This is the failure mode the
    # method actually has indoors — a corridor, or a wall at arm's length — and
    # without it a bad number looks like bad code rather than bad geometry.
    weak = sum(1 for c in tracker.conditioning if c <= tracker.min_conditioning)
    if weak:
        print(f"  ! {weak} of {len(tracker.conditioning)} frames were "
              f"under-constrained — a face of the room out of view leaves an "
              f"axis unobservable, and those frames coast on the prediction")

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
    # Without this the translation error is a number rather than a verdict:
    # reporting "did not move" scores exactly the distance actually moved, so
    # anything above that line is worse than no estimate at all.
    baseline = float(np.median(moved))
    print(f"  the frames are {baseline * 100:.1f} cm apart (median), so "
          f"'assume no motion' scores {baseline * 100:.1f} cm")
    speed = travelled / duration if duration > 0 else 0.0
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
