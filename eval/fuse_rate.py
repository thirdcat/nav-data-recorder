#!/usr/bin/env python3
"""Fuse a 30 Hz ICP trajectory with sparse Pi3X pose anchors.

Both input estimators use their own world frame.  The fused trajectory is
written in the Pi3X world frame: at an anchor it is exactly Pi3X, and between
anchors it follows ICP's camera-to-camera relative motion.  The endpoint
disagreement is interpolated as one smooth rigid correction over the interval
instead of being left as a jump at the next anchor.

Only rigid alignment is used for scoring.  The trajectories are metric and no
scale is fitted here or in the reported ATE.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


DEFAULT_TRAJ_DIR = Path(os.environ.get("TRAJ", Path(__file__).resolve().parent.parent / "traj"))
DEFAULT_PI3_DIR = Path(os.environ.get("PI3TRAJ", Path(__file__).resolve().parent.parent / "pi3traj"))


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


def _rigid_alignment(estimate, reference):
    """Fit rotation and translation only, matching build_vis/pi3_chain."""
    estimate = np.asarray(estimate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if estimate.shape != reference.shape or estimate.ndim != 2 or estimate.shape[1] != 3:
        raise ValueError(f"alignment arrays must both have shape (N, 3), got "
                         f"{estimate.shape} and {reference.shape}")
    if len(estimate) == 0:
        raise ValueError("cannot align an empty trajectory")
    ec = estimate - estimate.mean(0)
    rc = reference - reference.mean(0)
    U, _, Vt = np.linalg.svd(ec.T @ rc)
    R = U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    aligned = ec @ R + reference.mean(0)
    return aligned


def _ate(estimate, reference):
    aligned = _rigid_alignment(estimate, reference)
    return float(np.linalg.norm(aligned - reference, axis=1).mean())


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


def _write_output(path, fused, traj, masks):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        frame=traj["frame"],
        t=traj["t"],
        estimate=fused,
        reference=traj["reference"],
        anchor=masks["anchor_mask"],
        convention="world_from_camera, +Z forward +Y down (depth frame); world frame is Pi3X",
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
    report = _report(fused, traj, pi3, masks, session)
    if not report["frame_count_matches"]:
        raise SystemExit("internal error: fused trajectory is not full-rate")
    if args.out:
        _write_output(args.out, fused, traj, masks)
    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(_format_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
