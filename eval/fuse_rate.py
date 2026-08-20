#!/usr/bin/env python3
"""Fuse a 30 Hz ICP trajectory with sparse Pi3X pose anchors.

Both input estimators use their own world frame.  The fusion is built in the
Pi3X world frame: at an anchor it is exactly Pi3X, and between anchors it
follows ICP's camera-to-camera relative motion.  The endpoint disagreement is
interpolated as one smooth rigid correction over the interval instead of being
left as a jump at the next anchor.

Only rigid alignment is used for scoring.  The trajectories are metric and no
scale is fitted here or in the reported ATE.

**`--out` writes ARKit's world, and it is gated.**  It used to write the Pi3X
frame with `reference=traj["reference"]` copied in beside it — ARKit's poses,
in ARKit's world, next to an estimate in a different one.  `export_3dgs.py`'s
`substitute_poses` finds that key, checks it against the session, and prints
*"its own ARKit poses agree with the session to 0.00e+00 m"*, which is true of
the copied array and says nothing whatever about the trajectory being
substituted.  A check that runs, passes, and tests nothing is worse than the
`UNCHECKED` path it bypasses, because it reads as a clean bill of health.

So the fusion is rebased into ARKit's world before it is written, the two
arrays now share a frame, and `convention_gate` below refuses the file when the
two things the exporter genuinely cannot survive are not established.  See
`docs/FUSED_POSE_PREREG.md` for the criteria and for the controls each probe
was calibrated against.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


DEFAULT_TRAJ_DIR = Path(os.environ.get("TRAJ", Path(__file__).resolve().parent.parent / "traj"))
DEFAULT_PI3_DIR = Path(os.environ.get("PI3TRAJ", Path(__file__).resolve().parent.parent / "pi3traj"))

# Gate thresholds, fixed in docs/FUSED_POSE_PREREG.md before any arm was judged.
# The camera-axis bar is both absolute and relative to what the same probe reads
# on the session's own ICP trajectory, whose axis error is zero by construction:
# a probe floor set by one session's rotation drift is not a property of the arm.
AXIS_LIMIT_DEG = 5.0
AXIS_FLOOR_MULTIPLE = 2.0
AXIS_FLOOR_MIN_DEG = 1.0
# Baselines short enough that per-step noise inflates the ratio, and long enough
# that drift dominates it, both read wrong on a trajectory that is right. 0.3-1.0
# m is where the known-correct ICP trajectory reads 1.0027 across 18 sessions.
SCALE_BAND_M = (0.3, 1.0)
SCALE_TOLERANCE = 0.05


def _as_pose_array(value, name):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 3 or value.shape[1:] != (4, 4):
        raise ValueError(f"{name} must have shape (N, 4, 4), got {value.shape}")
    if len(value) < 1:
        raise ValueError(f"{name} is empty")
    return value


def _load(path, label):
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        required = {"frame", "t", "estimate", "reference"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"{path}: missing fields: {', '.join(missing)}")
        frames = np.asarray(data["frame"], dtype=np.int64)
        times = np.asarray(data["t"], dtype=np.float64)
        estimate = _as_pose_array(data["estimate"], f"{label}.estimate")
        reference = _as_pose_array(data["reference"], f"{label}.reference")
    if not (len(frames) == len(times) == len(estimate) == len(reference)):
        raise ValueError(
            f"{path}: frame/t/estimate/reference lengths differ: "
            f"{len(frames)}/{len(times)}/{len(estimate)}/{len(reference)}"
        )
    if len(np.unique(frames)) != len(frames):
        raise ValueError(f"{path}: frame ids are not unique")
    if np.any(np.diff(times) <= 0):
        raise ValueError(f"{path}: timestamps must be strictly increasing")
    return {
        "path": path,
        "frame": frames,
        "t": times,
        "estimate": estimate,
        "reference": reference,
    }


def _resolve(value, directory, label):
    """Resolve either an explicit NPZ path or a short session id."""
    candidate = Path(value)
    if candidate.is_file():
        return candidate
    if candidate.suffix == ".npz":
        candidate = candidate.with_suffix("")
    path = Path(directory) / f"{candidate.name}.npz"
    if not path.is_file():
        raise SystemExit(
            f"cannot find {label} trajectory {value!r}; tried {candidate} and {path}"
        )
    return path


def _inverse(T):
    """Inverse of a rigid 4x4 pose, retaining the input dtype/shape contract."""
    R = T[:3, :3]
    p = T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ p
    return out


def _project_rotation(R):
    """Project small numerical rotation drift back to SO(3)."""
    U, _, Vt = np.linalg.svd(R)
    out = U @ Vt
    if np.linalg.det(out) < 0:
        U[:, -1] *= -1
        out = U @ Vt
    return out


def _matrix_to_quaternion(R):
    """Return a unit quaternion in (w, x, y, z) order."""
    R = _project_rotation(R)
    trace = float(np.trace(R))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        q = np.array(
            [0.25 * s, (R[2, 1] - R[1, 2]) / s,
             (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s],
            dtype=np.float64,
        )
    else:
        diagonal = np.diag(R)
        i = int(np.argmax(diagonal))
        if i == 0:
            s = np.sqrt(max(1.0 + R[0, 0] - R[1, 1] - R[2, 2], 0.0)) * 2.0
            q = np.array(
                [(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                 (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s],
                dtype=np.float64,
            )
        elif i == 1:
            s = np.sqrt(max(1.0 - R[0, 0] + R[1, 1] - R[2, 2], 0.0)) * 2.0
            q = np.array(
                [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                 0.25 * s, (R[1, 2] + R[2, 1]) / s],
                dtype=np.float64,
            )
        else:
            s = np.sqrt(max(1.0 - R[0, 0] - R[1, 1] + R[2, 2], 0.0)) * 2.0
            q = np.array(
                [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                 (R[1, 2] + R[2, 1]) / s, 0.25 * s],
                dtype=np.float64,
            )
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("could not normalize correction rotation")
    return q / norm


def _quaternion_to_matrix(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),
         2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z),
         2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w),
         1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def _interpolate_rigid(correction, alpha):
    """Interpolate identity -> correction with linear translation and slerp."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha <= 0.0:
        return np.eye(4, dtype=np.float64)
    if alpha >= 1.0:
        return correction.copy()
    q1 = _matrix_to_quaternion(correction[:3, :3])
    q0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    if np.dot(q0, q1) < 0.0:
        q1 = -q1
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if dot > 1.0 - 1e-8:
        q = q0 + alpha * (q1 - q0)
        q /= np.linalg.norm(q)
    else:
        theta = np.arccos(dot)
        q = (np.sin((1.0 - alpha) * theta) * q0
             + np.sin(alpha * theta) * q1) / np.sin(theta)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = _quaternion_to_matrix(q)
    out[:3, 3] = alpha * correction[:3, 3]
    return out


def _kabsch_rotation(estimate, reference):
    """The rotation carrying centred `estimate` onto centred `reference`.

    Returned right-multiplied, which is how `_rigid_alignment` has always
    applied it, so the world-gauge fit and the alignment every ATE is scored
    through are the same arithmetic rather than two spellings of it.

    Positions only, and deliberately no scale.  `scale_ratio` below exists to
    find a metric error, and a gauge fit free to absorb one would leave the file
    certifying itself.
    """
    ec = estimate - estimate.mean(0)
    rc = reference - reference.mean(0)
    U, _, Vt = np.linalg.svd(ec.T @ rc)
    return U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt


def _rigid_alignment(estimate, reference):
    """Fit rotation and translation only, matching build_vis/pi3_chain."""
    estimate = np.asarray(estimate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if estimate.shape != reference.shape or estimate.ndim != 2 or estimate.shape[1] != 3:
        raise ValueError(f"alignment arrays must both have shape (N, 3), got "
                         f"{estimate.shape} and {reference.shape}")
    if len(estimate) == 0:
        raise ValueError("cannot align an empty trajectory")
    R = _kabsch_rotation(estimate, reference)
    aligned = (estimate - estimate.mean(0)) @ R + reference.mean(0)
    return aligned


def _ate(estimate, reference):
    aligned = _rigid_alignment(estimate, reference)
    return float(np.linalg.norm(aligned - reference, axis=1).mean())


def _world_gauge(estimate, reference):
    """The rigid transform putting `estimate`'s world on top of ARKit's.

    Fitted on positions at every frame.  This is a relabelling, not a
    correction: the exporter builds its initial point cloud from the same poses
    it writes into `images.txt`, so a left-multiplied rigid transform moves the
    cameras and the cloud together and the reconstruction is unchanged.  The ICP
    arm is the proof that this was never the hazard — its `estimate[0]` is the
    identity, so `traj/` is written in camera 0's frame rather than ARKit's, and
    it exported and trained to 21.760 dB regardless.

    It is done anyway because the file carries ARKit's poses under `reference`.
    Two arrays in one file describing two different worlds is what let
    `substitute_poses` report perfect agreement about a trajectory it had not
    looked at.
    """
    R = _kabsch_rotation(estimate, reference)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.T
    T[:3, 3] = reference.mean(0) - R.T @ estimate.mean(0)
    return T


def axis_offset(estimate, reference):
    """Mean camera-axis rotation between a trajectory and ARKit, in degrees.

    The difference the exporter cannot survive is a constant *right*-multiplier
    on the 3x3.  `camera_to_world` means +Z forward and +Y down, and a file
    written about any other axes is misplaced at every frame while looking
    perfectly well-formed.

    So the world gauge is fitted away on positions alone — which touches no
    rotation — and what is left, `D_i = R_ark(i)^T G R_est(i)`, is averaged over
    the trajectory.  A constant axis error makes every `D_i` equal to it, so it
    survives the average; independent per-frame rotation error does not.

    **The number is unreadable without its floor.**  Run on `traj/<id>.npz`'s
    ICP estimate, whose axes are `diag(1, -1, -1)` applied by the same code that
    writes `reference` and are therefore right by construction, this reads
    0.933 deg on cb4586 and a median of 3.06 across the corpus against a truth
    of zero.  That is accumulated rotation drift leaking into the average, and
    it is why the gate compares against the session's own ICP reading rather
    than against zero.  Injected errors are recovered: 2.104 for 2 deg, 5.042
    for 5, 179.999 for `diag(1, -1, -1)`.
    """
    estimate = np.asarray(estimate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    G = _world_gauge(estimate[:, :3, 3], reference[:, :3, 3])[:3, :3]
    total = np.zeros((3, 3), dtype=np.float64)
    for est, ref in zip(estimate, reference):
        total += ref[:3, :3].T @ (G @ est[:3, :3])
    mean = _project_rotation(total)
    return float(np.degrees(np.arccos(
        np.clip((np.trace(mean) - 1.0) / 2.0, -1.0, 1.0))))


def scale_ratio(estimate, reference, band=SCALE_BAND_M):
    """Metric scale against ARKit, from world-frame distances alone.

    The second thing the exporter cannot survive: the depth maps are in metres
    and the poses have to be in the same ones.  Nothing else in this file can
    see it — every ATE here is computed after a rigid alignment, which is blind
    to a global scale by construction.

    Distances between world positions use neither estimator's rotation, and that
    is the whole design.  The first attempt compared *camera-frame* relative
    translations; run on ICP, whose convention is correct, it returned axis
    offsets up to 135 degrees and scales up to 1.39, because a camera-frame
    translation is expressed through the estimator's own drifting rotation and
    that drift is indistinguishable there from a real error.  It was rejected
    rather than tuned.

    The band matters as much as the statistic.  Below it, per-step noise
    inflates the ratio — the known-correct ICP trajectory reads 1.082 at
    0.02-0.10 m across the corpus.  Above it, drift dominates and it scatters.
    At 0.3-1.0 m that same trajectory reads 1.0027, and 13 of the 18 sessions
    land inside the +/- 0.05 the gate asks for.
    """
    est = np.asarray(estimate, dtype=np.float64)[:, :3, 3]
    ref = np.asarray(reference, dtype=np.float64)[:, :3, 3]
    low, high = band
    # Geometric spacing over the frame lag: every lag would be quadratic in the
    # trajectory length and the extra pairs are near-duplicates of their
    # neighbours, so they would buy precision the drift floor cannot use.
    lags = np.unique(np.geomspace(1, max(len(est) - 1, 1), 60).astype(np.int64))
    found = []
    for lag in lags:
        if lag >= len(est):
            continue
        walked = np.linalg.norm(ref[lag:] - ref[:-lag], axis=1)
        keep = (walked >= low) & (walked <= high)
        if keep.any():
            found.append(np.linalg.norm(est[lag:] - est[:-lag], axis=1)[keep]
                         / walked[keep])
    if not found:
        return None, 0
    pooled = np.concatenate(found)
    return float(np.median(pooled)), int(len(pooled))


def convention_gate(fused, traj):
    """Decide whether `export_3dgs.py --poses` can be told to read this file.

    The question is narrow and is not about accuracy: does the exporter's
    `camera_to_world` describe these 4x4s?  Two ways it can fail to, and this
    checks both against the one trajectory in the same file whose answer is
    already known.

    `traj/<id>.npz`'s ICP estimate is that trajectory.  Its axes and its metre
    are right by construction, so what the probes read on it is their floor on
    *this* session — set by this session's drift, not by the arm being judged.
    Comparing the fusion against zero instead would fail sessions where ICP
    drifts and pass sessions where it does not, which measures the weather.

    What this does **not** establish: that the fused arm builds a better or
    worse splat than ARKit or ICP.  That needs a training run.  Nothing here
    licenses a word about it.
    """
    reference = traj["reference"]
    axis = axis_offset(fused, reference)
    floor = axis_offset(traj["estimate"], reference)
    limit = max(AXIS_FLOOR_MULTIPLE * floor, AXIS_FLOOR_MIN_DEG)
    scale, pairs = scale_ratio(fused, reference)
    scale_floor, _ = scale_ratio(traj["estimate"], reference)

    axis_ok = bool(axis < AXIS_LIMIT_DEG and axis <= limit)
    scale_ok = bool(scale is not None and abs(scale - 1.0) <= SCALE_TOLERANCE)
    return {
        "axis_offset_deg": round(axis, 4),
        "axis_floor_deg": round(floor, 4),
        "axis_limit_deg": round(min(AXIS_LIMIT_DEG, limit), 4),
        "axis_ok": axis_ok,
        "scale_ratio": None if scale is None else round(scale, 5),
        "scale_floor_ratio": None if scale_floor is None else round(scale_floor, 5),
        "scale_band_m": list(SCALE_BAND_M),
        "scale_pairs": pairs,
        "scale_ok": scale_ok,
        "passed": bool(axis_ok and scale_ok),
        "floor_source": "traj estimate (ICP): axes and metre correct by construction",
        "not_established": "whether this arm builds a better splat; that needs training",
    }


def fuse(traj, pi3):
    """Return a full-rate fused pose array and masks used by the report."""
    traj_frames = traj["frame"]
    pi3_frames = pi3["frame"]
    traj_frame_set = set(int(frame) for frame in traj_frames)
    missing = [int(frame) for frame in pi3_frames if int(frame) not in traj_frame_set]
    if missing:
        raise ValueError(f"Pi3X frames are not a subset of traj frames: {missing[:8]}")

    traj_by_frame = {int(frame): i for i, frame in enumerate(traj_frames)}
    anchors = np.asarray([traj_by_frame[int(frame)] for frame in pi3_frames], dtype=np.int64)
    if np.any(np.diff(anchors) <= 0):
        raise ValueError("Pi3X anchor frames must be in the same order as traj")

    icp = traj["estimate"]
    pi3_estimate = pi3["estimate"]
    fused = np.empty_like(icp, dtype=np.float64)

    # The output world frame is Pi3X's frame.  Prefix and suffix have no second
    # anchor, so they inherit ICP's relative motion from the nearest anchor.
    first, last = int(anchors[0]), int(anchors[-1])
    for k in range(first - 1, -1, -1):
        fused[k] = pi3_estimate[0] @ _inverse(icp[first]) @ icp[k]
    for k in range(last + 1, len(icp)):
        fused[k] = pi3_estimate[-1] @ _inverse(icp[last]) @ icp[k]

    # At each anchor interval, propagate ICP from the exact Pi3X start anchor,
    # then spread the endpoint correction over the interval.
    for anchor_number, (left, right) in enumerate(zip(anchors[:-1], anchors[1:])):
        left, right = int(left), int(right)
        start_pose = pi3_estimate[anchor_number]
        end_pose = pi3_estimate[anchor_number + 1]
        predicted_end = start_pose @ _inverse(icp[left]) @ icp[right]
        correction = end_pose @ _inverse(predicted_end)
        for k in range(left, right + 1):
            predicted = start_pose @ _inverse(icp[left]) @ icp[k]
            alpha = ((traj["t"][k] - traj["t"][left])
                     / (traj["t"][right] - traj["t"][left]))
            fused[k] = _interpolate_rigid(correction, alpha) @ predicted
        fused[left] = start_pose
        fused[right] = end_pose

    # There is one anchor even for a very short/synthetic trajectory.  The
    # loops above intentionally leave its exact pose untouched.
    for pi_index, traj_index in enumerate(anchors):
        fused[int(traj_index)] = pi3_estimate[pi_index]

    anchor_mask = np.zeros(len(traj_frames), dtype=bool)
    anchor_mask[anchors] = True
    between_mask = (~anchor_mask
                    & (np.arange(len(traj_frames)) > first)
                    & (np.arange(len(traj_frames)) < last))
    outer_mask = ~anchor_mask & ~between_mask
    return fused, {
        "anchor_indices": anchors,
        "anchor_mask": anchor_mask,
        "between_mask": between_mask,
        "outer_mask": outer_mask,
    }


def _score(fused, traj, pi3, masks):
    reference = traj["reference"][:, :3, 3]
    icp = traj["estimate"][:, :3, 3]
    pi3_positions = pi3["estimate"][:, :3, 3]
    anchor_mask = masks["anchor_mask"]
    between_mask = masks["between_mask"]
    # The anchor rows are the same source samples, so the fused and Pi3X
    # anchor scores use exactly the same correspondence and alignment fit.
    anchor_fused = fused[anchor_mask, :3, 3]
    anchor_ref = reference[anchor_mask]
    between_fused = fused[between_mask, :3, 3]
    between_ref = reference[between_mask]
    between_icp = icp[between_mask]

    # Score both sources against the exact same ARKit rows.  The two NPZs carry
    # duplicate references, but using the full-rate reference here makes the
    # equality at an anchor an explicit property of the comparison.
    anchor_pi3_ate = _ate(pi3_positions, anchor_ref)
    anchor_fused_ate = _ate(anchor_fused, anchor_ref)
    if len(between_fused):
        between_fused_ate = _ate(between_fused, between_ref)
        between_icp_ate = _ate(between_icp, between_ref)
    else:
        between_fused_ate = None
        between_icp_ate = None

    # A second, explicitly labelled non-anchor score includes the short prefix
    # and suffix when the first/last Pi3X frame is not a depth frame boundary.
    non_anchor = ~anchor_mask
    non_anchor_fused_ate = _ate(fused[non_anchor, :3, 3], reference[non_anchor])
    non_anchor_icp_ate = _ate(icp[non_anchor], reference[non_anchor])
    return {
        "anchors": {
            "frames": int(anchor_mask.sum()),
            "fused_ate_cm": anchor_fused_ate * 100.0,
            "pi3x_ate_cm": anchor_pi3_ate * 100.0,
            "delta_cm": (anchor_fused_ate - anchor_pi3_ate) * 100.0,
            "not_worse": bool(anchor_fused_ate <= anchor_pi3_ate + 1e-10),
        },
        "between": {
            "frames": int(between_mask.sum()),
            "fused_ate_cm": (None if between_fused_ate is None else between_fused_ate * 100.0),
            "icp_ate_cm": (None if between_icp_ate is None else between_icp_ate * 100.0),
            "delta_cm": (None if between_fused_ate is None
                         else (between_fused_ate - between_icp_ate) * 100.0),
            "better_than_icp": (None if between_fused_ate is None
                                else bool(between_fused_ate < between_icp_ate)),
        },
        "non_anchor_including_tails": {
            "frames": int(non_anchor.sum()),
            "fused_ate_cm": non_anchor_fused_ate * 100.0,
            "icp_ate_cm": non_anchor_icp_ate * 100.0,
            "delta_cm": (non_anchor_fused_ate - non_anchor_icp_ate) * 100.0,
        },
    }


def _report(fused, traj, pi3, masks, session):
    scores = _score(fused, traj, pi3, masks)
    n = len(traj["frame"])
    report = {
        "session": session,
        "traj_frames": n,
        "pi3x_anchors": len(pi3["frame"]),
        "fused_frames": len(fused),
        "frame_count_matches": bool(len(fused) == n),
        "first_anchor_frame": int(pi3["frame"][0]),
        "last_anchor_frame": int(pi3["frame"][-1]),
        "prefix_frames": int((masks["outer_mask"] &
                               (np.arange(n) < masks["anchor_indices"][0])).sum()),
        "suffix_frames": int((masks["outer_mask"] &
                               (np.arange(n) > masks["anchor_indices"][-1])).sum()),
        "scores": scores,
        "distribution": (
            "For each adjacent Pi3X anchor pair, propagate ICP's relative "
            "camera motion from the left anchor and slerp/linearly interpolate "
            "the rigid endpoint correction across the interval; use the nearest "
            "anchor plus ICP relative motion outside the anchored span."
        ),
        "scale_fitted": False,
    }
    return report


def _format_gate(gate):
    axis = (f"  camera axes: {gate['axis_offset_deg']:.3f} deg against a "
            f"{gate['axis_limit_deg']:.3f} bar "
            f"(ICP reads {gate['axis_floor_deg']:.3f} where the truth is 0), "
            f"{'PASS' if gate['axis_ok'] else 'FAIL'}")
    if gate["scale_ratio"] is None:
        return [axis, "  metric scale: no frame pair in the band, UNMEASURED"]
    return [axis,
            f"  metric scale: {gate['scale_ratio']:.4f} against ARKit over "
            f"{gate['scale_pairs']} pairs at "
            f"{gate['scale_band_m'][0]:g}-{gate['scale_band_m'][1]:g} m, "
            f"tolerance {SCALE_TOLERANCE:g} "
            f"(ICP reads {gate['scale_floor_ratio']:.4f} where the truth is 1), "
            f"{'PASS' if gate['scale_ok'] else 'FAIL'}"]


def rebase_to_arkit(fused, traj):
    """The same fusion, expressed in ARKit's world instead of Pi3X's.

    A left-multiplied rigid transform, so nothing about the trajectory changes
    except which origin it is quoted against — the shape, the metre and the
    camera axes are all untouched, and every ATE in the report is computed after
    a rigid alignment and so cannot move.  `--report-out` is compared
    bit-for-bit before and after in `docs/FUSED_POSE_PREREG.md` criterion 5;
    if a number moves, the transform was not rigid.

    Written so that `estimate` and `reference` in the output describe one world.
    They did not before, and that is what let `substitute_poses` check the
    copied `reference`, find it perfect, and report agreement about poses it had
    never examined.
    """
    G = _world_gauge(fused[:, :3, 3], traj["reference"][:, :3, 3])
    return np.einsum("ij,njk->nik", G, fused), G


def _write_output(path, fused, traj, masks, gate):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        frame=traj["frame"],
        t=traj["t"],
        estimate=fused,
        reference=traj["reference"],
        anchor=masks["anchor_mask"],
        convention="world_from_camera, +Z forward +Y down (depth frame); "
                   "world frame is ARKit's, rebased from Pi3X's by a rigid "
                   "transform fitted on positions — no scale",
        # Recorded in the file because the reader of an npz cannot re-run the
        # gate: `reference` alone would tell them nothing, which is the defect
        # this whole path exists to close.
        gate=json.dumps(gate),
    )


def _format_report(report):
    scores = report["scores"]
    anchors = scores["anchors"]
    between = scores["between"]
    between_ate = "n/a" if between["fused_ate_cm"] is None else (
        f"fused {between['fused_ate_cm']:.2f} cm, ICP {between['icp_ate_cm']:.2f} cm"
    )
    return "\n".join([
        f"{report['session']}: {report['fused_frames']} fused frames "
        f"(traj {report['traj_frames']}, Pi3X anchors {report['pi3x_anchors']})",
        f"  anchors ({anchors['frames']}): fused {anchors['fused_ate_cm']:.2f} cm, "
        f"Pi3X {anchors['pi3x_ate_cm']:.2f} cm, "
        f"delta {anchors['delta_cm']:+.2f} cm, "
        f"{'PASS' if anchors['not_worse'] else 'FAIL'}",
        f"  between anchors ({between['frames']}): {between_ate}, "
        f"{'PASS' if between['better_than_icp'] else 'FAIL' if between['better_than_icp'] is not None else 'n/a'}",
        f"  tails: prefix {report['prefix_frames']}, suffix {report['suffix_frames']}; "
        "scale fit no",
    ])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "session",
        help="short session id, or an explicit traj NPZ path",
    )
    parser.add_argument(
        "pi3_path",
        nargs="?",
        help="explicit pi3traj NPZ path when session is a traj path",
    )
    parser.add_argument("--traj-dir", type=Path, default=DEFAULT_TRAJ_DIR)
    parser.add_argument("--pi3-dir", type=Path, default=DEFAULT_PI3_DIR)
    parser.add_argument("--out", "--output", type=Path,
                        help="write the 30 Hz fused trajectory NPZ")
    parser.add_argument("--report-out", type=Path,
                        help="write the JSON score report")
    args = parser.parse_args(argv)

    if args.pi3_path is None:
        traj_path = _resolve(args.session, args.traj_dir, "traj")
        pi3_path = _resolve(args.session, args.pi3_dir, "pi3traj")
        session = Path(args.session).stem
    else:
        traj_path = _resolve(args.session, args.traj_dir, "traj")
        pi3_path = _resolve(args.pi3_path, args.pi3_dir, "pi3traj")
        session = traj_path.stem

    traj = _load(traj_path, "traj")
    pi3 = _load(pi3_path, "pi3traj")
    fused, masks = fuse(traj, pi3)
    # Rebased before scoring rather than on the way out, so that the report and
    # the file describe one trajectory in one frame. Criterion 5 of the
    # pre-registration is that this cannot move a number; see the comparison
    # there rather than trusting the claim.
    fused, _gauge = rebase_to_arkit(fused, traj)
    report = _report(fused, traj, pi3, masks, session)
    gate = convention_gate(fused, traj)
    report["convention_gate"] = gate
    if not report["frame_count_matches"]:
        raise SystemExit("internal error: fused trajectory is not full-rate")
    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(_format_report(report))
    print("\n".join(_format_gate(gate)))
    if args.out:
        # Refused rather than written-with-a-warning. A file on disk gets picked
        # up later by someone who did not watch it being made, and `--poses`
        # gives them no way to ask whether the conversion describes it.
        if not gate["passed"]:
            raise SystemExit(
                f"refusing to write {args.out}: the exporter's pose conversion "
                f"is not established for this fusion. See the two lines above "
                f"and docs/FUSED_POSE_PREREG.md")
        _write_output(args.out, fused, traj, masks, gate)
        print(f"  wrote {args.out} — export_3dgs.py --poses may read it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
