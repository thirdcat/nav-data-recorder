#!/usr/bin/env python3
"""Export a session as a Gaussian-splatting training set.

Structure-from-motion is the step every 3DGS pipeline starts with, and it is
the step this recorder does not need: ARKit already hands back a pose per frame
and the LiDAR already hands back metres. So this writes the COLMAP model that a
trainer expects to *find*, rather than the images it would run SfM on — poses
converted, not solved, and the initial point cloud back-projected from depth
instead of triangulated.

    python3 tools/export_3dgs.py ~/nav_data/20260813-162849-2994fa /tmp/gs

Three properties of this capture drive the choices here, and each is a place
where the obvious export would be wrong:

**Autofocus is on.** `fx` drifts by up to 11% within one session and `cx` walks
across the image centre, so the export writes one COLMAP camera *per image*
rather than one per session. A shared camera model silently absorbs that drift
into the geometry.

**Depth is registered to colour but is not the same size.** 256x192 is exactly
1/7.5 of 1920x1440, so depth intrinsics come from the dimension ratio. They are
never derived from `2*cx`: the principal point is a real optical quantity that
moves with focus, and using it as a width proxy injects that motion into the
back-projection.

**The world origin can move mid-session.** ARKit re-origins after an
interruption, so frames on either side of `ar.interruptionEnded` are not in one
coordinate system. They are exported as separate segments rather than being
concatenated into a reconstruction that cannot converge.

A `MultiCamRecorder` session has two lenses and no tracker, and it takes a
different path through this file — see `export_rig` below. `--poses` is required
there, because nothing in such a session knows where the camera was.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from read_session import Session  # noqa: E402

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None


# ARKit's camera looks down -Z with +Y up. COLMAP's — and the depth map's — is
# +Z forward with +Y down. The same basis change `depth_odometry.py` uses, and
# deliberately the same constant, so the two never drift apart.
ARKIT_TO_COLMAP = np.diag([1.0, -1.0, -1.0])

# Depth outside this band is not evidence. The near limit is where the LiDAR
# stops returning; the far limit is where its noise exceeds the voxel size the
# init cloud is built at, measured in docs/POSE.md.
DEPTH_NEAR_M = 0.15
DEPTH_FAR_M = 5.0


def quat_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n == 0.0:
        raise ValueError("zero quaternion")
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def matrix_to_quat(R: np.ndarray) -> tuple[float, float, float, float]:
    """Rotation matrix to (qw, qx, qy, qz) — COLMAP's scalar-first order."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw, qx = 0.25 * s, (R[2, 1] - R[1, 2]) / s
        qy, qz = (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw, qx = (R[2, 1] - R[1, 2]) / s, 0.25 * s
        qy, qz = (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw, qx = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s
        qy, qz = 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw, qx = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s
        qy, qz = (R[1, 2] + R[2, 1]) / s, 0.25 * s
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    return qw / n, qx / n, qy / n, qz / n


def camera_to_world(pose: dict[str, Any]) -> np.ndarray:
    """The stored ARKit pose as a 4x4 world-from-camera in COLMAP camera axes."""
    R = quat_to_matrix(pose["qx"], pose["qy"], pose["qz"], pose["qw"]) @ ARKIT_TO_COLMAP
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = (pose["tx"], pose["ty"], pose["tz"])
    return T


def world_to_camera(pose: dict[str, Any]) -> tuple[tuple[float, ...], np.ndarray]:
    """COLMAP's `images.txt` pair: (qw, qx, qy, qz) and the translation."""
    c2w = camera_to_world(pose)
    R = c2w[:3, :3].T
    t = -R @ c2w[:3, 3]
    return matrix_to_quat(R), t


def apply_guard_band(rows: list[dict[str, Any]], holdout: list[int], *,
                     radius_m: float, angle_deg: float
                     ) -> tuple[list[dict[str, Any]], list[int], int]:
    """Drop training frames that sit almost on top of a held-out one.

    Every eighth frame of a 5 Hz walk is 12 cm from the frames on either side of
    it, so the standard split asks a model to interpolate between two views it
    was given rather than to reconstruct anything. Measured on one session: 6 of
    23 held-out views had a training camera within 10 cm *and* 10 degrees.

    A guard band removes those neighbours from training. The held-out views stay
    spread through the walk — a contiguous block would instead ask the model
    about a part of the room nobody visited, which is a different and much
    harder question — but they stop having a near-duplicate to copy.

    Returns the surviving rows, the held-out indices renumbered into them, and
    how many training frames the band cost.
    """
    if radius_m <= 0 or not holdout:
        return rows, holdout, 0
    held = set(holdout)
    centres = np.array([camera_to_world(r["pose"])[:3, 3] for r in rows])
    forwards = np.array([camera_to_world(r["pose"])[:3, 2] for r in rows])
    cos_limit = math.cos(math.radians(angle_deg))

    keep = []
    for i, row in enumerate(rows):
        if i in held:
            keep.append(i)
            continue
        near = np.linalg.norm(centres[list(held)] - centres[i], axis=1) < radius_m
        if near.any():
            aligned = forwards[list(held)][near] @ forwards[i] > cos_limit
            if aligned.any():
                continue
        keep.append(i)

    renumber = {old: new for new, old in enumerate(keep)}
    return ([rows[i] for i in keep],
            sorted(renumber[i] for i in holdout),
            len(rows) - len(keep))


def rebase_pose(pose: dict[str, Any], transform: np.ndarray) -> dict[str, Any]:
    """The same camera, expressed in another session's world.

    Rewriting the pose is what lets everything downstream stay single-session:
    the exporter, the COLMAP writer and the back-projection all read `pose` and
    none of them has to learn that several worlds were involved. The transform
    acts on the ARKit world, so it composes on the left of the ARKit pose — and
    the quaternion goes back in ARKit's scalar-last order, which is the order
    the rest of the file expects to read.
    """
    R = quat_to_matrix(pose["qx"], pose["qy"], pose["qz"], pose["qw"])
    t = np.array([pose["tx"], pose["ty"], pose["tz"]])
    R_new = transform[:3, :3] @ R
    t_new = transform[:3, :3] @ t + transform[:3, 3]
    qw, qx, qy, qz = matrix_to_quat(R_new)
    moved = dict(pose)
    moved.update({"qx": qx, "qy": qy, "qz": qz, "qw": qw,
                  "tx": float(t_new[0]), "ty": float(t_new[1]), "tz": float(t_new[2])})
    return moved


def substitute_poses(rows: list[dict[str, Any]], npz_path: str,
                     key: str = "estimate") -> tuple[list[dict[str, Any]], float]:
    """Re-pose the session from a `--dump-poses` trajectory, keeping everything else.

    The export takes ARKit's pose because it is there. It is not the only
    trajectory this repository has: `tools/depth_odometry.py --dump-poses`
    writes an independent depth-ICP estimate alongside ARKit's own, and
    `docs/3DGS.md` records that the two disagree by 20-30 cm over a session
    without anyone having asked which of them builds a better splat. This is
    what lets that be a two-arm comparison instead of an opinion.

    **The two conventions are the same one.** `ARKIT_TO_DEPTH` in
    `depth_odometry` and `ARKIT_TO_COLMAP` here are both `diag(1, -1, -1)`, and
    the dump applies it exactly where `camera_to_world` does, over the same
    ARKit world. So the npz's 4x4 *is* this file's `camera_to_world`, and
    getting back to a stored pose is the same permutation again.

    That claim is not taken on inspection. The npz carries ARKit's own poses
    under `reference`, so every call re-derives those and compares them against
    what the session stored: the returned figure is the worst disagreement in
    metres, and anything above a millimetre raises instead of exporting. A
    conversion that is wrong in the same way in both directions would pass a
    round trip through itself and fail this.

    Intrinsics are untouched. They are ARKit's per-frame `fx/fy/cx/cy`, a
    property of the camera and not of whoever solved for its pose.

    Returns the worst disagreement in metres, or `None` when the file carries no
    `reference` to check against — which is the case for anything written in
    another estimator's world frame, `eval/fuse_rate.py` among them. Such a
    trajectory is not refused here, but nothing has established that this
    conversion describes it, and the caller has to say so.
    """
    data = np.load(npz_path)
    if key not in data:
        raise SystemExit(f"{npz_path} has no '{key}' — keys are {list(data)}")
    by_frame = {int(f): T for f, T in zip(data["frame"], data[key])}

    # `None` and `0.0` are different answers and the caller prints them
    # differently: a file with no `reference` has not been checked, and saying
    # it agrees to zero would be inventing the check.
    worst = None
    if "reference" in data:
        worst = 0.0
        check = {int(f): T for f, T in zip(data["frame"], data["reference"])}
        for row in rows:
            T = check.get(int(row["frame"]))
            if T is None:
                continue
            worst = max(worst, float(np.abs(
                camera_to_world(row["pose"]) - T).max()))
        if worst > 1e-3:
            raise SystemExit(
                f"{npz_path}'s own ARKit poses differ from the session's by "
                f"{worst:.4f} — the conversion here does not describe that file, "
                f"and the substituted trajectory would inherit the error")

    out = []
    for row in rows:
        T = by_frame.get(int(row["frame"]))
        if T is None:
            continue
        row = dict(row)
        row["pose"] = pose_with_matrix(row["pose"], T)
        out.append(row)
    if not out:
        raise SystemExit(f"no frame of this session appears in {npz_path}")
    return out, worst


def pose_with_matrix(pose: dict[str, Any], T: np.ndarray) -> dict[str, Any]:
    """The same camera intrinsics carried under a new world-from-camera matrix.

    `T` is `camera_to_world`'s output — world-from-camera with +Z forward and +Y
    down — and the stored pose wants ARKit's quaternion, so the basis change goes
    back on. Both the trajectory substitution and the multi-camera rig arrive at
    a 4x4 and need it stored as a pose, and they go through this one function so
    that a convention fixed in one place cannot be wrong in the other: the two
    are checked against each other by the identity gate in
    `docs/RIG_EXPORT_PREREG.md`, which only means anything if they share the
    conversion.
    """
    qw, qx, qy, qz = matrix_to_quat(T[:3, :3] @ ARKIT_TO_COLMAP)
    moved = dict(pose)
    moved.update({"qx": qx, "qy": qy, "qz": qz, "qw": qw,
                  "tx": float(T[0, 3]), "ty": float(T[1, 3]),
                  "tz": float(T[2, 3])})
    return moved


# --- the multi-camera rig ----------------------------------------------------
#
# Widest image-to-depth gap accepted when pairing a lens against the LiDAR. The
# wide arm comes off the depth device itself and matches to 0.000 ms; the
# ultra-wide is a second device and lands 9-12 ms away at the median.
RIG_MAX_DEPTH_DT_S = 0.05

# How far a trajectory's own timestamp may sit from the session's for the same
# frame number. These files carry no `reference` for `substitute_poses` to
# check, so this is the only thing standing between an export and the wrong
# `pi3traj/*.npz`: the four dual-lens sessions were recorded within minutes of
# each other and number their frames 0..N alike, and so do the *two lenses of
# one session*, whose shutters sit 46-65 ms apart. A millisecond is three orders
# below that gap and above nothing the right file produces, which is zero.
RIG_TIME_TOLERANCE_S = 1e-3

# What the calibration says the derived lens's optical centre is offset by, in
# metres. Checked at export time rather than trusted: `calib/` stores the
# translation in millimetres and everything else here is metres, so a missed
# conversion puts the second lens 19 m away with every matrix still orthonormal
# and every file still well-formed.
RIG_BASELINE_TOLERANCE_M = 1e-4


def _poseless():
    """`eval/pi3_poseless.py`, imported rather than copied.

    That file already answers the two geometric questions this path needs, and
    answers them with measurements written down beside them: which intrinsics a
    lens's *active* format has (`intrinsics_for`, `active_hfov` — the factory
    numbers are for 4032x3024 and no capture path runs that format), and how to
    move LiDAR depth between the two lenses' cones (`depth_in_wide`,
    `depth_in_ultrawide` — stretching a 72.6 deg map onto a 106.2 deg frame
    covers the whole image with measurements that exist for 41 % of it).
    Re-deriving any of it here would give this repository two answers to one
    question, and the copy is the one that goes stale.

    The import is deferred because it pulls OpenCV, which the ARKit path has
    never needed and which `tools/` otherwise does without.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "eval"))
    import pi3_poseless  # noqa: PLC0415
    return pi3_poseless


def calib_path(lens: str) -> Path:
    """This device's factory calibration for one lens, as `calib/` records it."""
    return Path(__file__).resolve().parent.parent / "calib" / f"iphone17-1_{lens}.json"


def rig_extrinsic(path: Path, identity: bool = False) -> np.ndarray:
    """The 4x4 taking a point in the derived lens's frame to the wide camera's.

    `calib/README.md` settles two things this depends on. The wide camera is the
    extrinsic *reference*, so its own extrinsic is exactly identity and the
    LiDAR depth is already in its frame — nothing has to be composed. And the
    direction is `R x + t` rather than its inverse, decided by correlating a
    textured patch across the probe's own stereo pair, where the two conventions
    predict shifts of opposite sign.

    Translation is millimetres in the file and metres everywhere else here.

    `identity` is the gate's setting rather than a user's: a derivation applied
    with a transform that changes nothing has to reproduce the export that never
    went through it, which is the only way to show the plumbing is not adding
    something of its own. See `docs/RIG_EXPORT_PREREG.md` criterion 2.
    """
    T = np.eye(4)
    if identity:
        return T
    data = json.loads(path.read_text())
    cols = np.asarray(data["extrinsic_matrix_columns"], dtype=np.float64)
    T[:3, :3] = cols[:3].T          # three rotation columns, column-major
    T[:3, 3] = cols[3] / 1000.0     # the file is in millimetres
    off = float(np.abs(T[:3, :3] @ T[:3, :3].T - np.eye(3)).max())
    if off > 1e-6 or abs(np.linalg.det(T[:3, :3]) - 1.0) > 1e-6:
        raise SystemExit(f"{path}: the extrinsic rotation is not a rotation "
                         f"(orthonormal to {off:.1e}, det "
                         f"{np.linalg.det(T[:3, :3]):.6f})")
    return T


def slerp(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:
    """Shortest-arc interpolation between two scalar-first quaternions.

    The sign flip matters: `q` and `-q` are the same rotation, so without it a
    pair that happens to be stored with opposite signs interpolates the long way
    round and the camera spins through most of a turn between two frames 200 ms
    apart.
    """
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1, d = -q1, -d
    if d > 0.9995:
        # Nearly parallel: the sines below both go to zero and the ratio is
        # numerically worthless, while a straight line is within a rounding of
        # the arc.
        q = q0 + u * (q1 - q0)
    else:
        theta = math.acos(max(-1.0, min(1.0, d)))
        s = math.sin(theta)
        q = (math.sin((1.0 - u) * theta) / s) * q0 + (math.sin(u * theta) / s) * q1
    return q / np.linalg.norm(q)


def interpolate_camera_pose(times: np.ndarray, mats: np.ndarray,
                            t: float) -> np.ndarray | None:
    """Where the posed camera was at time `t`, or None if that is outside the walk.

    The reason this exists rather than "take the nearest pose" is measured:
    `MultiCamRecorder` throttles each lens to 5 Hz independently, so the two
    shutters land 46-65 ms apart at the median and up to 110 ms apart, and over
    that gap the camera travels 13-27 mm and turns 1.6-2.0 degrees. The rig
    transform being applied is 19.272 mm and 0.462 degrees. **The timing offset
    is the same size as the translation it is modelling and four times the
    rotation**, so a nearest-pose construction would ship an error larger than
    the transform and would look exactly as correct.

    Nothing is extrapolated. An ultra-wide frame recorded before the first wide
    frame or after the last has no pair to interpolate between, and continuing
    the walk past its end is inventing; those frames are dropped and counted.

    An exact sample returns that sample untouched — not the interpolant
    evaluated at zero — so that the identity gate compares two paths that
    genuinely computed the same thing rather than two that agreed to nine
    decimals. The interpolation itself is checked against a known answer in
    `tools/test_export_3dgs.py`, on a trajectory whose true pose at the midpoint
    is derivable by hand.
    """
    if t < times[0] or t > times[-1]:
        return None
    j = int(np.searchsorted(times, t, side="left"))
    if j < len(times) and times[j] == t:
        return mats[j]
    if j == 0 or j >= len(times):
        return None
    i = j - 1
    u = float((t - times[i]) / (times[j] - times[i]))
    q = slerp(np.array(matrix_to_quat(mats[i][:3, :3])),
              np.array(matrix_to_quat(mats[j][:3, :3])), u)
    T = np.eye(4)
    T[:3, :3] = quat_to_matrix(q[1], q[2], q[3], q[0])
    T[:3, 3] = (1.0 - u) * mats[i][:3, 3] + u * mats[j][:3, 3]
    return T


def load_trajectory(npz_path: str, key: str, frame_time: dict[int, float], *,
                    allow_broken: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A `pi3_poseless` trajectory, checked against the session that must own it.

    Two refusals, both of which have a way of happening on this corpus.

    **The wrong file.** `substitute_poses` catches a foreign trajectory by
    re-deriving the session's own ARKit poses from the file's `reference`. There
    is no `reference` here and there cannot be — the whole point of a poseless
    session is that no tracker ran — so that check is unavailable exactly where
    the risk is highest: four dual-lens sessions recorded within minutes, all
    numbering their frames from zero, plus a second trajectory per session for
    the *other lens* with the same frame numbers again. What the files do carry
    is `t`, taken from the same stream entries this session reads, so the right
    file agrees to zero and the other lens's disagrees by the 46-65 ms that
    separates the two shutters.

    **A trajectory that already said it was broken.** `broken_from` is set when
    a window join disagreed with the chain about the shared path, and everything
    after it inherits that. `d67f0a` carries it on both arms. Exporting past it
    produces a training set whose second half is in a drifted frame, which no
    downstream check would attribute to the trajectory.
    """
    data = np.load(npz_path)
    if key not in data:
        raise SystemExit(f"{npz_path} has no '{key}' — keys are {list(data)}")
    if "t" not in data:
        raise SystemExit(f"{npz_path} carries no 't', so nothing can establish "
                         f"that it belongs to this session; refusing rather "
                         f"than exporting an unowned trajectory")
    frames = np.asarray(data["frame"], dtype=np.int64)
    times = np.asarray(data["t"], dtype=np.float64)
    mats = np.asarray(data[key], dtype=np.float64)

    shared = [(f, t) for f, t in zip(frames, times) if int(f) in frame_time]
    if len(shared) < 4:
        raise SystemExit(f"{npz_path} shares only {len(shared)} frame numbers "
                         f"with this stream — wrong session, or wrong lens")
    worst = max(abs(t - frame_time[int(f)]) for f, t in shared)
    if worst > RIG_TIME_TOLERANCE_S:
        raise SystemExit(
            f"{npz_path} disagrees with this session's own timestamps by "
            f"{worst * 1000:.1f} ms on frames it claims to describe, against a "
            f"tolerance of {RIG_TIME_TOLERANCE_S * 1000:g} ms. Tens to hundreds "
            f"of milliseconds is the *other lens* of this session: the two arms "
            f"number their frames from zero independently, so the numbers match "
            f"and only the clock tells them apart. Minutes is another session")

    broken = int(data["broken_from"]) if "broken_from" in data else -1
    if broken >= 0 and not allow_broken:
        raise SystemExit(
            f"{npz_path} flagged itself broken from frame {broken}: a window "
            f"join disagreed with the chain about the shared path and every "
            f"pose after it inherits that. Pass --allow-broken-trajectory to "
            f"export it anyway, and do not read the result as one frame")
    return frames, times, mats, worst, broken


def rig_rows(session: Session, stream: str, K: np.ndarray, max_depth_dt: float
             ) -> list[dict[str, Any]]:
    """One row per image of a lens, with the depth frame nearest it in time.

    In an ARKit session the image and its depth come out of one `ARFrame` and
    share a frame number. Here they are independent outputs of a multi-cam
    session with their own counters, so time is the only honest key — the same
    join `eval/pi3_poseless.py` makes, made by the same function so the exporter
    and the poser cannot pair frames differently.

    The pose carries this lens's intrinsics and a placeholder identity; the
    caller fills the geometry in. `tracking` is set to `normal` because there is
    no tracker to ask — a multi-cam session leaves ARKit to reach the second
    lens. The equivalent signal is the trajectory's own `broken_from`, and
    `load_trajectory` refuses on it.
    """
    rows = _poseless().pair_by_time(session, max_depth_dt, stream)
    for row in rows:
        row["path"] = session.frame_path(row)
        row["_session"] = session
        row["pose"] = {"tx": 0.0, "ty": 0.0, "tz": 0.0,
                       "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
                       "fx": float(K[0, 0]), "fy": float(K[1, 1]),
                       "cx": float(K[0, 2]), "cy": float(K[1, 2]),
                       "tracking": "normal"}
    return rows


def rig_depth_loader(session: Session, row: dict[str, Any], K: np.ndarray,
                     depth_calib: dict, extrinsic: np.ndarray | None):
    """Return a callable giving this row's depth on its own image grid.

    Three things are wrong with handing the raw depth map to `depth_points` in a
    multi-cam session, and each is silent.

    The grid is **320x240 against a 640x480 or 3840x2160 image**, so the
    dimension-ratio rule the ARKit path uses would scale the *image's* focal
    length onto the depth grid — right only while the two describe one cone, and
    `calib/README.md` measures the depth camera at 70.38 deg against the wide
    video's 69.53. The depth camera reports its own intrinsics for the format it
    ran, and this uses them.

    The ultra-wide **does not see the depth camera's cone at all**: 106.2 deg
    against 72.6, so the LiDAR covers about 40 % of that frame's area, and a
    resize would fill the rest with measurements that do not exist.

    And the depth sits **19.272 mm and 0.462 deg away** from the ultra-wide's
    optical centre, which at 2 m displaces a point by 19.3 mm and 16.1 mm
    respectively — not negligible against the agreement a renderer wants.

    So each lens gets the depth put where that lens can use it, by the functions
    in `eval/pi3_poseless.py` that already do it, and the result comes back on
    the image grid — where `depth_points` reads the image's own intrinsics
    unscaled, because the ratio is one.

    Deferred rather than computed now: an ultra-wide depth grid is 8.3 M floats,
    and holding one per frame would cost gigabytes for arrays used once.
    """
    pl = _poseless()

    def load():
        if extrinsic is None:
            depth = pl.depth_in_wide(session, row, K, depth_calib)
        else:
            depth = pl.depth_in_ultrawide(session, row, K, depth_calib, extrinsic)
        # A multi-cam session carries no confidence channel — `AVDepthData`'s
        # filtering is off and nothing writes `confidenceOffset` — so this is a
        # map of "the LiDAR returned here", not a collapse of ARKit's 0/1/2.
        # It has to exist: `write_depth_mask` reads a missing confidence as
        # *everything valid*, which on an ultra-wide frame would tell the
        # trainer that the 60 % of the image the LiDAR never reached is a
        # measured zero.
        conf = np.where(depth > 0.0, 2, 0).astype(np.uint8)
        return depth, conf

    return load


def export_rig(session_dir: str, out_dir: str, *, poses: str,
               pose_key: str = "estimate", arms: str = "both",
               derived_lens: str = "ultrawide", extrinsic: str = "calib",
               derived_poses: str | None = None,
               pose_time: str = "interpolate",
               max_depth_dt: float = RIG_MAX_DEPTH_DT_S,
               allow_broken: bool = False, derived_init_cloud: bool = False,
               hfov_source: float | None = None,
               hfov_derived: float | None = None,
               min_baseline_m: float = 0.0, sharp_ratio: float = 0.0,
               conf_min: int = 2, holdout_every: int = 8,
               **dataset) -> dict[str, Any]:
    """Both lenses of one walk into one training set, in one coordinate frame.

    The problem this solves is not that the second lens is unposed. It is that
    posing it *independently* does not put it in the first lens's frame:
    `eval/pi3_poseless.py` on the two arms of the same session gives
    trajectories whose best-fit-aligned residuals run 2.4 to 33.8 cm at the
    median with scale ratios 0.946-1.058, against a rig baseline of 1.93 cm.
    Two such trajectories cannot be glued, and no amount of care in the export
    would glue them.

    They do not have to be. The rig is calibrated, so **posing one arm poses the
    other**: the wide trajectory supplies the geometry, and every ultra-wide
    pose is the wide pose at that instant composed with a fixed transform read
    from `calib/`. One solve, one frame, no relocalisation.

    Two decisions here are load-bearing and both are measured rather than
    assumed:

    **The derived pose is interpolated to the derived frame's own timestamp.**
    The lenses do not shoot together — 46-65 ms apart at the median — and the
    camera moves 13-27 mm over that gap, more than the 19.272 mm baseline being
    applied. `interpolate_camera_pose` carries the argument. `--rig-pose-time
    nearest` keeps the naive construction so the two can be compared.

    **The derived arm contributes images, not points.** Its depth *is* the wide
    arm's depth, forward-scattered into a 27-times-larger grid and dilated to
    close the lattice that leaves. Back-projecting that would re-enter the same
    76 800 LiDAR returns as millions of resampled copies, and the voxel average
    would then be dominated by the resampling rather than by the measurement.
    `--derived-init-cloud` turns it on, and the identity gate needs it: with a
    transform that changes nothing, the two paths must produce the same cloud
    as well as the same cameras.

    The holdout is chosen on the source arm and **carried to the derived frames
    paired with it**. A held-out wide view whose ultra-wide twin — 60 ms and
    19 mm away — sat in training would be scored against a near-duplicate it was
    effectively given, which is the leak `--guard-m` exists to close elsewhere.
    """
    if Image is None:
        raise SystemExit("Pillow is required: pip install pillow")
    session = Session(session_dir)
    manifest = session.manifest
    if manifest.get("kind") != "multicam":
        raise SystemExit(f"{session_dir} is not a multicam session "
                         f"(kind={manifest.get('kind')!r})")
    if conf_min < 1:
        raise SystemExit(
            "--conf-min 0 cannot be honoured on a multi-camera session. There "
            "is no confidence channel to relax; the mask records where the "
            "LiDAR returned at all, and admitting the rest would mark the part "
            "of the frame the sensor never reached as valid depth")
    if any(d.get("confidenceOffset") is not None for d in session.depth_index()):
        raise SystemExit(
            "this session carries a per-pixel confidence channel and the rig "
            "path has no way to move it into either lens's grid — the depth is "
            "reprojected, so a confidence map resampled by size would mask the "
            "wrong pixels and nothing renders a mask")

    pl = _poseless()
    lenses = {"source": "wide", "derived": derived_lens}
    streams = {"wide": "frames_wide", "ultrawide": "frames"}
    hfov = {"source": hfov_source, "derived": hfov_derived}
    intrinsics, notes = {}, []
    for role, lens in lenses.items():
        calib = pl.load_calibration(calib_path(lens))
        logged = hfov[role] if hfov[role] is not None else pl.active_hfov(manifest, lens)
        rows0 = next(iter(session.stream(streams[lens])), None)
        if rows0 is None:
            raise SystemExit(f"{session_dir} has no {streams[lens]}.jsonl — "
                             f"this session did not record the {lens} lens")
        K = pl.intrinsics_for(calib, rows0["width"], rows0["height"], logged)
        intrinsics[role] = K
        fov = 2 * math.degrees(math.atan(rows0["width"] / (2 * K[0, 0])))
        if logged is None:
            notes.append(f"the {lens} lens logged no active field of view; its "
                         f"focal length is the factory calibration scaled from "
                         f"4032x3024, which is an assumption and not a "
                         f"measurement — pass --hfov-{role} if you have one")
        print(f"  {role} lens {lens}: {rows0['width']}x{rows0['height']} at "
              f"{fov:.4f} deg across, fx {K[0, 0]:.1f} "
              f"({'logged' if logged else 'FACTORY, ASSUMED'})")

    depth_calib, depth_source = pl.depth_calibration(manifest)
    print(f"  depth grid from {depth_source}")

    source_rows = rig_rows(session, streams["wide"], intrinsics["source"], max_depth_dt)
    frame_time = {int(r["frame"]): float(r["t"]) for r in source_rows}
    frames, times, mats, worst_t, broken = load_trajectory(
        poses, pose_key, frame_time, allow_broken=allow_broken)
    print(f"  poses from {poses}:{pose_key} — {len(frames)} of "
          f"{len(source_rows)} wide frames, timestamps agree to "
          f"{worst_t * 1000:.3f} ms" + (f"; BROKEN FROM {broken}, exported anyway"
                                        if broken >= 0 else ""))

    by_frame = {int(f): T for f, T in zip(frames, mats)}
    source_rows = [r for r in source_rows if int(r["frame"]) in by_frame]
    for row in source_rows:
        row["pose"] = pose_with_matrix(row["pose"], by_frame[int(row["frame"])])
    source_rows, source_dropped = select_frames(
        session, source_rows, min_baseline_m=min_baseline_m,
        sharp_ratio=sharp_ratio, require_exact_depth=False)

    E = rig_extrinsic(calib_path(derived_lens), identity=(extrinsic == "identity"))
    baseline_m = float(np.linalg.norm(E[:3, 3]))
    order = np.argsort(times)
    times, mats = times[order], mats[order]

    measured = None
    supplied_offsets: list[float] = []
    if derived_poses:
        data = np.load(derived_poses)
        measured = {int(f): T for f, T in zip(data["frame"], data["estimate"])}
        print(f"  derived lens posed from {derived_poses}: "
              f"{len(measured)} frames supplied, rig derivation bypassed")

    derived_rows, no_bracket = [], 0
    if arms in ("both", "derived"):
        for row in rig_rows(session, streams[derived_lens], intrinsics["derived"],
                            max_depth_dt):
            # `nearest` drops the same frames `interpolate` cannot bracket, even
            # though clamping to an end pose would give it an answer. Otherwise
            # the two constructions would be compared on different frame sets
            # and the comparison would carry a second variable.
            T = None
            if times[0] <= row["t"] <= times[-1]:
                T = (mats[int(np.argmin(np.abs(times - row["t"])))]
                     if pose_time == "nearest"
                     else interpolate_camera_pose(times, mats, float(row["t"])))
            if T is None:
                no_bracket += 1
                continue
            if measured is not None:
                # The derived lens is being *given* its poses rather than having
                # them composed from the rig. `eval/uw_selfcal.py` measures that
                # the chain disagrees with its own photographs by 0.8-4.4 deg
                # per consecutive frame pair, and emits a corrected pose per
                # frame; this is the door those come in through.
                #
                # The baseline check below cannot run on them — that check asks
                # whether the *composition* placed the centre exactly the
                # calibrated distance away, and a measured pose is under no such
                # obligation. What is reported instead is how far each supplied
                # pose sits from where the rig would have put it, because a
                # correction that moves the lens as a group is a re-placement
                # rather than a correction and the number has to be visible.
                given = measured.get(int(row["frame"]))
                if given is None:
                    no_bracket += 1
                    continue
                supplied_offsets.append(
                    float(np.linalg.norm(given[:3, 3] - (T @ E)[:3, 3])))
                row["pose"] = pose_with_matrix(row["pose"], given)
                row["_derived_from"] = T
                row["_init_cloud"] = derived_init_cloud
                derived_rows.append(row)
                continue
            derived = T @ E
            # Criterion 3, run on every step rather than sampled afterwards: the
            # composition has to put the derived optical centre exactly the
            # calibrated baseline from the pose it was derived from. This is
            # what catches the millimetre-for-metre reading of `calib/`, a
            # rotation dropped from the translation, and a composition applied
            # on the wrong side — each of which leaves every file well-formed.
            got = float(np.linalg.norm(derived[:3, 3] - T[:3, 3]))
            if abs(got - baseline_m) > RIG_BASELINE_TOLERANCE_M:
                raise SystemExit(
                    f"frame {row['frame']}: the derived camera centre came out "
                    f"{got * 1000:.4f} mm from the pose it was derived from, "
                    f"against the calibrated {baseline_m * 1000:.4f} mm")
            row["pose"] = pose_with_matrix(row["pose"], derived)
            row["_derived_from"] = T
            row["_init_cloud"] = derived_init_cloud
            derived_rows.append(row)
        if no_bracket:
            print(f"  {no_bracket} {derived_lens} frames fell outside the posed "
                  f"walk and were dropped rather than extrapolated")
        if supplied_offsets:
            off = np.asarray(supplied_offsets)
            print(f"  supplied poses sit {off.mean()*100:.2f} cm from the rig "
                  f"derivation on average (median {np.median(off)*100:.2f}, "
                  f"max {off.max()*100:.2f}) — a correction moves frames apart, "
                  f"a re-placement moves them together")
    derived_rows, derived_dropped = select_frames(
        session, derived_rows, min_baseline_m=min_baseline_m,
        sharp_ratio=sharp_ratio, require_exact_depth=False)

    if arms == "derived":
        source_rows = []
    for role, rows in (("source", source_rows), ("derived", derived_rows)):
        # Which transport a row needs is a property of its *lens*, not of which
        # arm it is: the LiDAR is the wide camera plus a scanner, so a wide
        # frame's depth is already in its own frame and only has to be resampled
        # through K, while an ultra-wide frame's has to cross the rig.
        for row in rows:
            row["_lens"] = lenses[role]
            row["_depth_fn"] = rig_depth_loader(
                session, row, intrinsics[role], depth_calib,
                E if lenses[role] == "ultrawide" else None)

    # The holdout is the source arm's, carried onto whichever derived frames are
    # paired with it. Left to index arithmetic on the concatenated list it would
    # scatter across both lenses and leave every held-out wide view with its own
    # ultra-wide twin, 60 ms and 19 mm away, sitting in training.
    rows = source_rows + derived_rows
    if source_rows:
        chosen = holdout_split(len(source_rows), holdout_every)
        source_t = np.array([r["t"] for r in source_rows])
        holdout = list(chosen)
        for j, row in enumerate(derived_rows, start=len(source_rows)):
            k = int(np.argmin(np.abs(source_t - row["t"])))
            if k in chosen:
                holdout.append(j)
    else:
        holdout = holdout_split(len(derived_rows), holdout_every)

    gaps = []
    if source_rows and derived_rows:
        source_t = np.array([r["t"] for r in source_rows])
        gaps = [float(np.abs(source_t - r["t"]).min()) for r in derived_rows]

    report = write_dataset(out_dir, rows, holdout_indices=sorted(holdout),
                           conf_min=conf_min, **dataset)
    report.update({
        "session": session.id,
        "kind": "multicam-rig",
        "arms": arms,
        "source_lens": lenses["source"], "derived_lens": lenses["derived"],
        "source_images": len(source_rows), "derived_images": len(derived_rows),
        "poses": poses, "pose_key": pose_key,
        "trajectory_time_agreement_ms": round(worst_t * 1000, 4),
        "trajectory_broken_from": broken,
        "rig_extrinsic": extrinsic,
        "rig_baseline_mm": round(baseline_m * 1000, 4),
        "rig_pose_time": pose_time,
        "derived_without_bracket": no_bracket,
        "derived_in_init_cloud": derived_init_cloud,
        "lens_gap_ms": (round(float(np.median(gaps)) * 1000, 1) if gaps else None),
        "lens_gap_max_ms": (round(float(np.max(gaps)) * 1000, 1) if gaps else None),
        "dropped": {"source": source_dropped, "derived": derived_dropped},
        "path_len_m": round(float(sum(
            np.linalg.norm(camera_to_world(b["pose"])[:3, 3]
                           - camera_to_world(a["pose"])[:3, 3])
            for a, b in zip(source_rows, source_rows[1:]))), 3),
        "convention": "COLMAP: world_from_camera converted to world-to-camera, "
                      "+Z forward +Y down; world is the wide arm's Pi3X frame, "
                      "metric; the derived lens is that pose composed with "
                      "calib/'s R x + t",
        "note": "path_len_m is the source arm's walk; the two lenses walk it "
                "twice and summing the concatenated list would double it",
    })
    if notes:
        report["assumptions"] = notes
    with open(os.path.join(out_dir, "export.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    return report


def sharpness(path: str, long_side: int = 960) -> float:
    """Variance of a Laplacian — small on a blurred frame, small in the dark.

    Deliberately not thresholded here. On a night session the whole population
    shifts down by two orders of magnitude, so an absolute cut would reject the
    session rather than its worst frames; the caller compares against the
    session's own median instead.
    """
    if Image is None:
        return float("nan")
    with Image.open(path) as im:
        im = im.convert("L")
        scale = long_side / max(im.size)
        if scale < 1.0:
            im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))))
        a = np.asarray(im, dtype=np.float64)
    lap = (a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:] - 4.0 * a[1:-1, 1:-1])
    return float(lap.var())


def segment_frames(session: Session, rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Split on world-origin resets. Poses across one are not comparable."""
    breaks = sorted(e["t"] for e in session.events()
                    if e.get("kind") == "ar.interruptionEnded")
    if not breaks:
        return [rows]
    segments: list[list[dict[str, Any]]] = [[]]
    it = iter(breaks)
    nxt = next(it, None)
    for row in rows:
        while nxt is not None and row["t"] >= nxt:
            segments.append([])
            nxt = next(it, None)
        segments[-1].append(row)
    return [s for s in segments if s]


def select_frames(session: Session, rows: list[dict[str, Any]], *,
                  min_baseline_m: float, sharp_ratio: float,
                  require_exact_depth: bool) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Drop the frames a trainer cannot use, and say how many went to each cause."""
    dropped = {"tracking": 0, "no_depth": 0, "stale_depth": 0, "blurry": 0, "too_close": 0}

    kept: list[dict[str, Any]] = []
    for row in rows:
        if row["pose"].get("tracking") != "normal":
            dropped["tracking"] += 1
            continue
        if row.get("depth") is None:
            dropped["no_depth"] += 1
            continue
        if require_exact_depth and row.get("depth_dt") != 0.0:
            dropped["stale_depth"] += 1
            continue
        kept.append(row)

    if sharp_ratio > 0.0 and Image is not None and kept:
        values = np.array([sharpness(r["path"]) for r in kept])
        for row, value in zip(kept, values):
            row["_sharpness"] = float(value)
        floor = float(np.median(values)) * sharp_ratio
        sharp = [r for r in kept if r["_sharpness"] >= floor]
        dropped["blurry"] = len(kept) - len(sharp)
        kept = sharp

    if min_baseline_m > 0.0 and kept:
        thinned = [kept[0]]
        last = camera_to_world(kept[0]["pose"])[:3, 3]
        for row in kept[1:]:
            here = camera_to_world(row["pose"])[:3, 3]
            if float(np.linalg.norm(here - last)) >= min_baseline_m:
                thinned.append(row)
                last = here
        dropped["too_close"] = len(kept) - len(thinned)
        kept = thinned

    return kept, dropped


def depth_points(session: Session, row: dict[str, Any], *,
                 conf_min: int, near: float, far: float, pix_stride: int,
                 depth: np.ndarray | None = None,
                 conf: np.ndarray | None = None
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Back-project one frame's depth into the world. Returns points, normals, uv."""
    entry = row["depth"]
    z = (np.asarray(session.depth_frame(entry), dtype=np.float32)
         if depth is None else depth)
    if conf is None:
        conf = session.confidence_frame(entry)
    h, w = z.shape

    pose = row["pose"]
    sx, sy = w / row["width"], h / row["height"]
    fx, fy = pose["fx"] * sx, pose["fy"] * sy
    cx, cy = pose["cx"] * sx, pose["cy"] * sy

    vv, uu = np.mgrid[0:h, 0:w]
    ok = np.isfinite(z) & (z > near) & (z < far)
    if conf is not None and conf_min > 0:
        ok &= np.asarray(conf) >= conf_min

    # Normals come from the neighbourhood before subsampling, so a stride does
    # not widen the cross product's footprint and round off every edge.
    px = (uu - cx) * z / fx
    py = (vv - cy) * z / fy
    cam = np.stack([px, py, z], axis=-1)
    du = np.zeros_like(cam)
    dv = np.zeros_like(cam)
    du[:, 1:-1] = cam[:, 2:] - cam[:, :-2]
    dv[1:-1, :] = cam[2:, :] - cam[:-2, :]
    nrm = np.cross(du, dv)
    norm = np.linalg.norm(nrm, axis=-1, keepdims=True)
    nrm = np.divide(nrm, norm, out=np.zeros_like(nrm), where=norm > 1e-12)

    if pix_stride > 1:
        sel = np.zeros_like(ok)
        sel[::pix_stride, ::pix_stride] = True
        ok &= sel

    if not ok.any():
        empty = np.zeros((0, 3))
        return empty, empty, np.zeros((0, 2), dtype=np.int64)

    c2w = camera_to_world(pose)
    pts = cam[ok] @ c2w[:3, :3].T + c2w[:3, 3]
    nrm = nrm[ok] @ c2w[:3, :3].T
    uv = np.stack([uu[ok], vv[ok]], axis=1).astype(np.int64)
    return pts, nrm, uv


def sample_colours(path: str, uv: np.ndarray, depth_shape: tuple[int, int]) -> np.ndarray:
    """Colour for each depth pixel, read at the matching place in the JPEG."""
    if Image is None or uv.size == 0:
        return np.full((len(uv), 3), 128, dtype=np.uint8)
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"))
    h, w = depth_shape
    ratio_y, ratio_x = rgb.shape[0] / h, rgb.shape[1] / w
    ys = np.clip(((uv[:, 1] + 0.5) * ratio_y).astype(np.int64), 0, rgb.shape[0] - 1)
    xs = np.clip(((uv[:, 0] + 0.5) * ratio_x).astype(np.int64), 0, rgb.shape[1] - 1)
    return rgb[ys, xs]


def voxel_average(points: np.ndarray, normals: np.ndarray, colours: np.ndarray,
                  voxel: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fuse to one point per voxel by averaging, not by picking a survivor.

    Picking is cheaper and is what a downsample usually does, but it keeps one
    frame's noise instead of cancelling it across the frames that saw the same
    surface. The count comes back too: it is how many observations a point has,
    which is the only honest confidence available for the init cloud.
    """
    if len(points) == 0:
        return points, normals, colours, np.zeros(0, dtype=np.int64)
    keys = np.floor(points / voxel).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    n = len(counts)

    def accumulate(values: np.ndarray) -> np.ndarray:
        out = np.zeros((n, values.shape[1]), dtype=np.float64)
        for axis in range(values.shape[1]):
            out[:, axis] = np.bincount(inverse, weights=values[:, axis], minlength=n)
        return out / counts[:, None]

    pts = accumulate(points)
    nrm = accumulate(normals)
    norm = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = np.divide(nrm, norm, out=np.zeros_like(nrm), where=norm > 1e-12)
    col = np.clip(accumulate(colours.astype(np.float64)), 0, 255).astype(np.uint8)
    return pts, nrm, col, counts


def write_ply(path: str, points: np.ndarray, normals: np.ndarray,
              colours: np.ndarray) -> None:
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        for p, n, c in zip(points, normals, colours):
            fh.write(struct.pack("<6f3B", p[0], p[1], p[2], n[0], n[1], n[2],
                                 int(c[0]), int(c[1]), int(c[2])))


def write_colmap(out: str, rows: list[dict[str, Any]], names: list[str],
                 sizes: list[tuple[int, int]], scales: list[float],
                 points: np.ndarray, colours: np.ndarray) -> None:
    sparse = os.path.join(out, "sparse", "0")
    os.makedirs(sparse, exist_ok=True)

    with open(os.path.join(sparse, "cameras.txt"), "w") as fh:
        fh.write("# Camera list with one line of data per camera:\n"
                 "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
                 "# One camera per image: autofocus moves fx and cx during a session.\n")
        for i, (row, (w, h), s) in enumerate(zip(rows, sizes, scales), start=1):
            p = row["pose"]
            fh.write(f"{i} PINHOLE {w} {h} "
                     f"{p['fx'] * s:.6f} {p['fy'] * s:.6f} "
                     f"{p['cx'] * s:.6f} {p['cy'] * s:.6f}\n")

    with open(os.path.join(sparse, "images.txt"), "w") as fh:
        fh.write("# Image list with two lines of data per image:\n"
                 "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
                 "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
                 "# Poses are ARKit's, converted — not solved. The second line is\n"
                 "# empty because there are no tracks: the points come from LiDAR.\n")
        for i, (row, name) in enumerate(zip(rows, names), start=1):
            (qw, qx, qy, qz), t = world_to_camera(row["pose"])
            fh.write(f"{i} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} "
                     f"{t[0]:.9f} {t[1]:.9f} {t[2]:.9f} {i} {name}\n\n")

    with open(os.path.join(sparse, "points3D.txt"), "w") as fh:
        fh.write("# 3D point list with one line of data per point:\n"
                 "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        for i, (p, c) in enumerate(zip(points, colours), start=1):
            fh.write(f"{i} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                     f"{int(c[0])} {int(c[1])} {int(c[2])} 0\n")


def holdout_split(count: int, every: int) -> list[int]:
    """Indices reserved for evaluation, or none when `every` is zero.

    Every n-th frame, which on a walk means a view the trainer has never seen
    from a place it has nearly been. That is the honest test for this capture:
    a random split would leave neighbours 12 cm away in the training set, and
    interpolating between two views 12 cm apart is not the question being asked.

    The phase is `i % every == 0` and not something better centred, because that
    is what `gsplat`'s and the reference 3DGS loader's `test_every` means. A
    split that disagrees with the trainer's is worse than no split at all here:
    the frames this export keeps out of the initial point cloud would then be
    training views, and the frames actually being scored would be ones whose
    geometry was handed over at initialisation.
    """
    if every <= 1:
        return []
    return list(range(0, count, every))


def write_transforms(out: str, rows: list[dict[str, Any]], names: list[str],
                     sizes: list[tuple[int, int]], scales: list[float],
                     depth_names: list[str] | None, depth_dir: str = "depths",
                     depth_scale: float = 0.001,
                     holdout: list[int] | None = None,
                     lenses: list[str | None] | None = None,
                     stamps: list[float] | None = None) -> None:
    """Nerfstudio's `transforms.json`.

    Its `transform_matrix` is camera-to-world in the OpenGL convention — X
    right, Y up, Z back — which is ARKit's camera frame exactly. So this one
    writes the pose *without* the basis change that COLMAP needs; the two files
    describing the same cameras differ, and that is correct.
    """
    frames = []
    for i, (row, name, (w, h), s) in enumerate(zip(rows, names, sizes, scales)):
        p = row["pose"]
        R = quat_to_matrix(p["qx"], p["qy"], p["qz"], p["qw"])
        m = np.eye(4)
        m[:3, :3] = R
        m[:3, 3] = (p["tx"], p["ty"], p["tz"])
        frame = {
            "file_path": f"images/{name}",
            "transform_matrix": [[float(v) for v in r] for r in m],
            "fl_x": p["fx"] * s, "fl_y": p["fy"] * s,
            "cx": p["cx"] * s, "cy": p["cy"] * s,
            "w": w, "h": h,
            "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0,
        }
        if depth_names is not None:
            frame["depth_file_path"] = f"{depth_dir}/{depth_names[i]}"
        # Which lens took this photograph. Nerfstudio ignores keys it does not
        # know, and without it a multi-camera export is a directory of images
        # whose two populations can only be told apart by their pixel
        # dimensions — which is exactly the kind of inference that goes wrong
        # the first time both lenses are downscaled to the same size.
        if lenses is not None and lenses[i] is not None:
            frame["lens"] = lenses[i]
        # The capture time, written only where two lenses share a file. It is
        # what lets a reader pair them for itself instead of taking the
        # exporter's word for which frames go together — and the pairing is the
        # thing worth checking, since the two shutters are 46-65 ms apart.
        if stamps is not None:
            frame["t"] = stamps[i]
        frames.append(frame)

    doc = {
        "camera_model": "PINHOLE",
        "ply_file_path": "sparse_pc.ply",
        "frames": frames,
    }
    if depth_names is not None:
        # Metres per stored unit: 0.001 for millimetre PNGs, 1.0 for metre .npy.
        doc["depth_unit_scale_factor"] = depth_scale
    if holdout:
        held = set(holdout)
        doc["train_filenames"] = [f"images/{n}" for i, n in enumerate(names) if i not in held]
        doc["val_filenames"] = [f"images/{n}" for i, n in enumerate(names) if i in held]
        doc["test_filenames"] = list(doc["val_filenames"])
    with open(os.path.join(out, "transforms.json"), "w") as fh:
        json.dump(doc, fh, indent=1)


def write_image(src: str, dst: str, downscale: int, mode: str) -> tuple[int, int]:
    """Place one training image and report the size it ended up as."""
    if downscale == 1 and mode != "resize":
        with Image.open(src) as im:
            size = im.size
        if mode == "symlink":
            if os.path.lexists(dst):
                os.unlink(dst)
            os.symlink(os.path.abspath(src), dst)
        else:
            shutil.copy2(src, dst)
        return size

    with Image.open(src) as im:
        target = (max(1, im.width // downscale), max(1, im.height // downscale))
        im = im.convert("RGB").resize(target, Image.LANCZOS)
        im.save(dst, quality=95)
    return target


def resample_nearest(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Depth to the training image's size, nearest-neighbour, never bilinear.

    An interpolated depth halfway between a foreground edge and the wall behind
    it is a measurement of nothing, and it lands exactly where a depth loss is
    most confident and most wrong.
    """
    w, h = size
    dh, dw = frame.shape
    ys = np.clip((np.arange(h) + 0.5) * dh / h, 0, dh - 1).astype(np.int64)
    xs = np.clip((np.arange(w) + 0.5) * dw / w, 0, dw - 1).astype(np.int64)
    return frame[np.ix_(ys, xs)]


def write_depth(path: str, entry_depth: np.ndarray, conf: np.ndarray | None,
                size: tuple[int, int], fmt: str, conf_min: int) -> None:
    """Per-image depth in whichever form the trainer reads.

    `png16` holds millimetres in 16 bits, which is compact and is what most
    loaders assume. `npy` holds float32 metres, which is what DN-Splatter's
    parser wants with `--depth-unit-scale-factor 1.0` — and which costs about
    11 MB per full-resolution frame, so it is not the default.
    """
    grid = resample_nearest(entry_depth, size)
    if conf is not None:
        grid = np.where(resample_nearest(np.asarray(conf), size) >= conf_min,
                        grid, 0.0)
    grid = np.nan_to_num(grid, nan=0.0, posinf=0.0, neginf=0.0)
    if fmt == "npy":
        np.save(path, grid.astype(np.float32))
    else:
        Image.fromarray(np.clip(grid * 1000.0, 0, 65535).astype(np.uint16)).save(path)


def write_depth_mask(path: str, conf: np.ndarray | None, size: tuple[int, int],
                     conf_min: int) -> None:
    """DN-Splatter's confidence mask, in DN-Splatter's inverted convention.

    **0 means valid and 255 means invalid.** The loader computes `1 - mask/255`
    and keeps pixels where that is positive, so writing the intuitive polarity
    would train on exactly the pixels the LiDAR said not to trust — and would
    look fine, because the mask is never rendered.

    The mask is binary because that is all the loader can use: it thresholds
    rather than weights, so the ARKit levels are collapsed here rather than
    somewhere the collapse is invisible. Continuous weighting needs a patch to
    the regulariser, not a different file.
    """
    w, h = size
    if conf is None:
        Image.fromarray(np.zeros((h, w), np.uint8)).save(path, quality=95)
        return
    valid = resample_nearest(np.asarray(conf), size) >= conf_min
    Image.fromarray(np.where(valid, 0, 255).astype(np.uint8)).save(path, quality=95)


def export(session_dir: str, out_dir: str, *, downscale: int = 1,
           conf_min: int = 2, voxel: float = 0.02, max_points: int = 1_000_000,
           min_baseline_m: float = 0.0, sharp_ratio: float = 0.0,
           pix_stride: int = 1, image_mode: str = "symlink",
           with_depth: bool = True, require_exact_depth: bool = True,
           segment: int | None = None, holdout_every: int = 8,
           guard_m: float = 0.0, depth_format: str = "png16",
           poses: str | None = None, pose_key: str = "estimate",
           rig: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    session = Session(session_dir)
    if Image is None:
        raise SystemExit("Pillow is required: pip install pillow")

    # A multi-camera session has two lenses and no tracker, so neither the pose
    # join nor the interruption split below applies to it — `posed_images` finds
    # no `pose.jsonl` and returns nothing, which would surface as "too few
    # frames" rather than as the real reason. Dispatch on what the recorder
    # wrote rather than on a flag, so that pointing this at such a session
    # cannot quietly produce an empty export.
    if session.manifest.get("kind") == "multicam":
        if not poses:
            raise SystemExit(
                f"{session_dir} is a multicam session: it left ARKit to reach "
                f"the second lens, so it carries no poses at all. Give it a "
                f"wide-arm trajectory with --poses "
                f"pi3traj/<id>_wide_local.npz")
        if segment is not None:
            raise SystemExit("--segment is an ARKit re-origin split; a multicam "
                             "session has no ARKit world to re-origin")
        return [export_rig(
            session_dir, out_dir, poses=poses, pose_key=pose_key,
            min_baseline_m=min_baseline_m, sharp_ratio=sharp_ratio,
            conf_min=conf_min, holdout_every=holdout_every,
            downscale=downscale, voxel=voxel, max_points=max_points,
            pix_stride=pix_stride, image_mode=image_mode, with_depth=with_depth,
            guard_m=guard_m, depth_format=depth_format, **(rig or {}))]

    # A rig flag on a single-lens session would otherwise do nothing at all, and
    # the export would look like it had honoured it.
    stray = [k for k, v in (rig or {}).items()
             if v not in (None, False, "both", "ultrawide", "calib",
                          "interpolate", RIG_MAX_DEPTH_DT_S)]
    if stray:
        print(f"  WARNING {', '.join(sorted(stray))} apply only to a multicam "
              f"session and are being ignored here")

    all_rows = session.posed_images()
    if poses:
        before = len(all_rows)
        all_rows, worst = substitute_poses(all_rows, poses, pose_key)
        checked = (f"its own ARKit poses agree with the session to {worst:.2e} m"
                   if worst is not None else
                   "**it carries no `reference`, so the convention is UNCHECKED** "
                   "— a trajectory in another estimator's world frame will be "
                   "silently misplaced here")
        print(f"  poses from {poses}:{pose_key} — {len(all_rows)} of {before} "
              f"frames matched; {checked}")
    for row in all_rows:
        row["_session"] = session
    segments = segment_frames(session, all_rows)
    if segment is not None:
        if not 0 <= segment < len(segments):
            raise SystemExit(f"segment {segment} of {len(segments)}")
        segments = [segments[segment]]

    reports = []
    for index, seg_rows in enumerate(segments):
        suffix = "" if len(segments) == 1 else f"_seg{index}"
        out = out_dir + suffix
        rows, dropped = select_frames(
            session, seg_rows, min_baseline_m=min_baseline_m,
            sharp_ratio=sharp_ratio, require_exact_depth=require_exact_depth)
        if len(rows) < 2:
            reports.append({"out": out, "images": len(rows), "skipped": "too few frames",
                            "dropped": dropped})
            continue
        report = write_dataset(out, rows, downscale=downscale, conf_min=conf_min,
                               voxel=voxel, max_points=max_points,
                               pix_stride=pix_stride, image_mode=image_mode,
                               with_depth=with_depth, holdout_every=holdout_every,
                               guard_m=guard_m, depth_format=depth_format)
        report.update({"session": session.id, "segment": index, "dropped": dropped})
        with open(os.path.join(out, "export.json"), "w") as fh:
            json.dump(report, fh, indent=1)
        reports.append(report)
    return reports


def write_dataset(out: str, rows: list[dict[str, Any]], *, downscale: int = 1,
                  conf_min: int = 2, voxel: float = 0.02,
                  max_points: int = 1_000_000, pix_stride: int = 1,
                  image_mode: str = "symlink", with_depth: bool = True,
                  holdout_every: int = 8, holdout_indices: list[int] | None = None,
                  guard_m: float = 0.0, guard_deg: float = 10.0,
                  depth_format: str = "png16") -> dict[str, Any]:
    """Write one training set from a list of rows, whatever sessions they came from.

    Every row carries its own `_session`, so a merged list of several walks goes
    through exactly the path a single walk does. That is the point of rebasing
    the poses upstream rather than teaching this function about frames.
    """
    if True:
        # DN-Splatter reads `depth/` and finds the confidence folder by sorting
        # its contents rather than by a key in the JSON, so the names have to
        # sort into frame order — which `%06d` does.
        depth_dir = "depth" if depth_format == "npy" else "depths"
        os.makedirs(os.path.join(out, "images"), exist_ok=True)
        if with_depth:
            os.makedirs(os.path.join(out, depth_dir), exist_ok=True)
            os.makedirs(os.path.join(out, "depth_normals_mask"), exist_ok=True)

        # A caller merging several sessions dictates the split, so that the
        # single-session arm and the merged arm are scored on the very same
        # photographs. Left to itself the rule would pick every eighth row of a
        # longer list, which is a different set of frames and not a comparison.
        chosen = (holdout_indices if holdout_indices is not None
                  else holdout_split(len(rows), holdout_every))
        rows, chosen, guarded = apply_guard_band(rows, chosen, radius_m=guard_m,
                                                 angle_deg=guard_deg)
        holdout = set(chosen)

        names, sizes, scales, depth_names = [], [], [], []
        pts_all, nrm_all, col_all = [], [], []
        for i, row in enumerate(rows):
            name = f"{i:06d}.jpg"
            size = write_image(row["path"], os.path.join(out, "images", name),
                               downscale, image_mode)
            names.append(name)
            sizes.append(size)
            scales.append(size[0] / row["width"])

            owner = row["_session"]
            # A row may carry its own depth: the multi-camera path hands over a
            # map already moved onto this lens's image grid, because a
            # multi-cam session's depth is neither the size nor the cone of
            # either lens's image and the dimension-ratio rule below would be
            # answering a question about a camera that did not run.
            if "_depth_fn" in row:
                depth, conf = row["_depth_fn"]()
            else:
                depth = np.asarray(owner.depth_frame(row["depth"]), dtype=np.float32)
                conf = owner.confidence_frame(row["depth"])
            # An evaluation frame contributes its image and its depth for
            # scoring, but not its geometry to the initial cloud. Otherwise the
            # trainer starts already holding the answer to the question it is
            # about to be asked, and every held-out number is flattered.
            # `_init_cloud` says the same of a frame whose depth is a resampled
            # copy of another frame's — see `export_rig`.
            pts, nrm, uv = ((np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 2), int))
                            if i in holdout or not row.get("_init_cloud", True) else
                            depth_points(owner, row, conf_min=conf_min,
                                         near=DEPTH_NEAR_M, far=DEPTH_FAR_M,
                                         pix_stride=pix_stride, depth=depth, conf=conf))
            if len(pts):
                pts_all.append(pts)
                nrm_all.append(nrm)
                col_all.append(sample_colours(row["path"], uv, depth.shape))

            if with_depth:
                dname = f"{i:06d}.npy" if depth_format == "npy" else f"{i:06d}.png"
                write_depth(os.path.join(out, depth_dir, dname), depth, conf, size,
                            depth_format, conf_min)
                write_depth_mask(os.path.join(out, "depth_normals_mask", f"{i:06d}.jpg"),
                                 conf, size, conf_min)
                depth_names.append(dname)

        points = np.concatenate(pts_all) if pts_all else np.zeros((0, 3))
        normals = np.concatenate(nrm_all) if nrm_all else np.zeros((0, 3))
        colours = (np.concatenate(col_all) if col_all
                   else np.zeros((0, 3), dtype=np.uint8))
        raw_count = len(points)
        points, normals, colours, counts = voxel_average(points, normals, colours, voxel)

        # Thin by observation count, not at random: a point seen once is the
        # one most likely to be a depth-edge artefact.
        if max_points and len(points) > max_points:
            keep = np.argsort(-counts)[:max_points]
            keep.sort()
            points, normals, colours, counts = (points[keep], normals[keep],
                                                colours[keep], counts[keep])

        held = sorted(holdout)
        write_colmap(out, rows, names, sizes, scales, points, colours)
        # The same cloud under both names each lineage looks for: Nerfstudio
        # reads the path named in transforms.json, and the reference 3DGS loader
        # looks for sparse/0/points3D.ply and only falls back to parsing the
        # text model if it is missing.
        write_ply(os.path.join(out, "sparse_pc.ply"), points, normals, colours)
        write_ply(os.path.join(out, "sparse", "0", "points3D.ply"),
                  points, normals, colours)
        lenses = [row.get("_lens") for row in rows]
        write_transforms(out, rows, names, sizes, scales,
                         depth_names if with_depth else None, depth_dir,
                         1.0 if depth_format == "npy" else 0.001, held,
                         lenses if any(lenses) else None,
                         [float(r["t"]) for r in rows] if any(lenses) else None)
        if held:
            with open(os.path.join(out, "holdout.txt"), "w") as fh:
                fh.write("# image names reserved for evaluation, one per line.\n"
                         "# Their depth is scored against but never fused into\n"
                         "# sparse_pc.ply, so the init cloud does not leak.\n")
                for i in held:
                    fh.write(names[i] + "\n")

        report = {
            "out": out,
            "images": len(rows),
            "holdout": len(held),
            "guard_band_m": guard_m,
            "guarded_out": guarded,
            "points_raw": raw_count,
            "points": len(points),
            "voxel_m": voxel,
            "conf_min": conf_min,
            "downscale": downscale,
            "depth_format": depth_format if with_depth else None,
            "duration_s": round(rows[-1]["t"] - rows[0]["t"], 2),
            "path_len_m": round(float(sum(
                np.linalg.norm(camera_to_world(b["pose"])[:3, 3]
                               - camera_to_world(a["pose"])[:3, 3])
                for a, b in zip(rows, rows[1:]))), 3),
            "convention": "COLMAP: world_from_camera converted to world-to-camera, "
                          "+Z forward +Y down; world is ARKit's, +Y up, metric",
        }
        return report


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("session")
    ap.add_argument("out")
    ap.add_argument("--downscale", type=int, default=1,
                    help="integer image downscale; intrinsics scale with it")
    ap.add_argument("--conf-min", type=int, default=2,
                    help="lowest ARKit depth confidence used for the init cloud (0/1/2)")
    ap.add_argument("--voxel", type=float, default=0.02,
                    help="init-cloud fusion voxel in metres")
    ap.add_argument("--max-points", type=int, default=1_000_000)
    ap.add_argument("--min-baseline", type=float, default=0.0,
                    help="drop frames closer than this to the last kept one, in metres")
    ap.add_argument("--sharp-ratio", type=float, default=0.0,
                    help="drop frames below this fraction of the session's median sharpness")
    ap.add_argument("--pix-stride", type=int, default=1,
                    help="subsample depth pixels when building the init cloud")
    ap.add_argument("--images", choices=("symlink", "copy", "resize"), default="symlink")
    ap.add_argument("--no-depth", action="store_true",
                    help="skip the per-image depth and its confidence mask")
    ap.add_argument("--depth-format", choices=("png16", "npy"), default="png16",
                    help="png16 stores millimetres compactly; npy stores float metres, "
                         "which is what DN-Splatter reads (about 11 MB per full-size frame)")
    ap.add_argument("--allow-stale-depth", action="store_true",
                    help="accept nearest-in-time depth on pre-unified-gate sessions")
    ap.add_argument("--segment", type=int, default=None)
    ap.add_argument("--holdout-every", type=int, default=8,
                    help="reserve every n-th frame for evaluation; 0 or 1 keeps all")
    ap.add_argument("--guard-m", type=float, default=0.0,
                    help="drop training frames within this distance and 10 degrees "
                         "of a held-out one, so the evaluation is not scored on "
                         "views that have a near-duplicate in training")
    ap.add_argument("--poses", metavar="TRAJ.npz",
                    help="re-pose the export from a tools/depth_odometry.py "
                         "--dump-poses trajectory instead of ARKit's own")
    ap.add_argument("--pose-key", default="estimate",
                    help="which trajectory in that file; 'reference' is ARKit "
                         "and must reproduce the default export")

    rig = ap.add_argument_group(
        "multi-camera sessions",
        "A MultiCamRecorder session records two lenses on one walk and carries "
        "no poses. --poses supplies the wide arm's trajectory and the other "
        "lens is derived from it through calib/'s rig transform, so both land "
        "in one frame without a second solve. See docs/RIG_EXPORT_PREREG.md.")
    rig.add_argument("--derived-poses", metavar="POSES.npz",
                     help="pose the derived lens from this file instead of "
                          "composing it through the rig extrinsic; for "
                          "eval/uw_selfcal.py's corrected poses")
    rig.add_argument("--arms", choices=("both", "source", "derived"), default="both",
                     help="which arms reach the training set: the pose source "
                          "(wide), the derived lens, or both")
    rig.add_argument("--derived-lens", choices=("ultrawide", "wide"),
                     default="ultrawide",
                     help="which lens is derived through the rig transform")
    rig.add_argument("--rig-extrinsic", choices=("calib", "identity"), default="calib",
                     help="'identity' is the gate: the derivation must then "
                          "reproduce the export that never went through it")
    rig.add_argument("--rig-pose-time", choices=("interpolate", "nearest"),
                     default="interpolate",
                     help="the two lenses shoot 46-65 ms apart and the camera "
                          "moves further than the rig baseline in that time; "
                          "'interpolate' poses the derived frame at its own "
                          "timestamp, 'nearest' is the naive construction kept "
                          "for comparison")
    rig.add_argument("--max-depth-dt", type=float, default=RIG_MAX_DEPTH_DT_S,
                     help="widest image-to-depth gap accepted, seconds")
    rig.add_argument("--allow-broken-trajectory", action="store_true",
                     help="export past a trajectory's own broken_from flag")
    rig.add_argument("--derived-init-cloud", action="store_true",
                     help="let the derived arm's resampled depth into the "
                          "initial cloud; off because it is the same LiDAR "
                          "returns re-entered as resampled copies")
    rig.add_argument("--hfov-source", type=float, default=None,
                     help="active horizontal FOV of the wide lens, degrees, "
                          "for sessions recorded before it was logged")
    rig.add_argument("--hfov-derived", type=float, default=None,
                     help="active horizontal FOV of the derived lens, degrees")
    a = ap.parse_args(argv)

    reports = export(
        a.session, a.out, downscale=a.downscale, conf_min=a.conf_min, voxel=a.voxel,
        max_points=a.max_points, min_baseline_m=a.min_baseline,
        sharp_ratio=a.sharp_ratio, pix_stride=a.pix_stride,
        image_mode=("resize" if a.downscale > 1 else a.images),
        with_depth=not a.no_depth, require_exact_depth=not a.allow_stale_depth,
        segment=a.segment, holdout_every=a.holdout_every, guard_m=a.guard_m,
        depth_format=a.depth_format, poses=a.poses, pose_key=a.pose_key,
        rig={"arms": a.arms, "derived_lens": a.derived_lens,
             "extrinsic": a.rig_extrinsic, "pose_time": a.rig_pose_time,
             "max_depth_dt": a.max_depth_dt,
             "allow_broken": a.allow_broken_trajectory,
             "derived_init_cloud": a.derived_init_cloud,
             "hfov_source": a.hfov_source, "hfov_derived": a.hfov_derived,
             "derived_poses": a.derived_poses})
    for report in reports:
        print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


def export_merged(specs: list[tuple[str, np.ndarray]], out_dir: str, *,
                  holdout_every: int = 8, min_baseline_m: float = 0.0,
                  sharp_ratio: float = 0.0, require_exact_depth: bool = True,
                  guard_m: float = 0.0, **dataset) -> dict[str, Any]:
    """One training set out of several walks, in the first one's frame.

    The transforms come from `align_set.py`. Each session's poses are rewritten
    into the reference frame before anything else happens, so the rest of the
    pipeline never learns that more than one recording was involved.

    **The reference session's rows come first, and only they are held out.**
    That is what makes the merged arm comparable to a single-session arm: both
    are scored on the same photographs, and the question "did adding a second
    walk help" has a fixed thing to be measured on. Holding out every eighth row
    of the concatenated list would sample a different set of frames in each arm
    and answer nothing.
    """
    if Image is None:
        raise SystemExit("Pillow is required: pip install pillow")

    merged: list[dict[str, Any]] = []
    per_session = []
    for order, (session_dir, transform) in enumerate(specs):
        session = Session(session_dir)
        rows = session.posed_images()
        for row in rows:
            row["_session"] = session
        # An interruption re-origins the world, and a transform solved against
        # the whole session does not describe both halves. Take the longest run.
        segments = segment_frames(session, rows)
        rows = max(segments, key=len)
        rows, dropped = select_frames(session, rows, min_baseline_m=min_baseline_m,
                                      sharp_ratio=sharp_ratio,
                                      require_exact_depth=require_exact_depth)
        if order > 0 or transform is not None:
            for row in rows:
                row["pose"] = rebase_pose(row["pose"], transform)
        merged.extend(rows)
        per_session.append({"session": session.id, "images": len(rows),
                            "dropped": dropped,
                            "segments": len(segments)})

    reference_count = per_session[0]["images"]
    report = write_dataset(out_dir, merged, guard_m=guard_m,
                           holdout_indices=holdout_split(reference_count, holdout_every),
                           **dataset)
    report.update({
        "merged_from": per_session,
        "reference": per_session[0]["session"],
        "reference_images": reference_count,
        "holdout_note": "held out from the reference session only, so a "
                        "single-session arm scores the same frames",
    })
    with open(os.path.join(out_dir, "export.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    return report
