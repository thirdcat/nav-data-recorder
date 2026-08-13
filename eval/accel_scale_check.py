#!/usr/bin/env python3
"""Check a trajectory's metric scale against the accelerometer.

Scaling a trajectory by s scales its acceleration by s, and CoreMotion reports
acceleration directly, having seen neither the camera nor the depth sensor. That
makes it the only independent witness available for translation — the rotation
already has one in CoreMotion's attitude, and GPS turned out to have nothing to
say: across twenty sessions its accuracy is 1.6x the distance walked and the
position never moved at all in thirteen of them.

It is a scale check and nothing more. The acceleration is never integrated,
because integrating it is what does not work at these speeds.

Run it on ARKit's own trajectory and it measures the noise floor rather than
independence, since ARKit fuses the IMU. Run it on the depth-only estimate, or
on a learned one, and it is a real test. Both are in the npz that
`tools/depth_odometry.py --dump-poses` writes, so this needs no GPU.

    python3 eval/accel_scale_check.py traj/1696fa.npz ~/nav_data/2026...-1696fa
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

G = 9.80665
CUTOFF_HZ = 2.0

# Measured over 11,329 samples from seven sessions and recorded in docs/POSE.md:
# 90 degrees about Z, taking the portrait-natural device frame to the camera's
# native landscape frame.
R_CAM_FROM_DEV = np.array([[0.0, -1.0, 0.0],
                           [1.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0]])
# The dumped poses are already in the depth convention (+Z forward, +Y down),
# which is ARKit's camera frame through diag(1, -1, -1).
ARKIT_TO_DEPTH = np.diag([1.0, -1.0, -1.0])


def lowpass(x, dt, cutoff=CUTOFF_HZ):
    """Zero-phase single-pole filter, applied forward and back.

    Differentiating twice amplifies noise brutally, so both signals are
    band-limited identically before differencing rather than after — which is
    what keeps the second difference finite.
    """
    alpha = 1.0 / (1.0 + 1.0 / (2.0 * math.pi * cutoff * dt))
    out = np.array(x, dtype=np.float64, copy=True)
    for _ in range(2):
        for i in range(1, len(out)):
            out[i] = out[i - 1] + alpha * (out[i] - out[i - 1])
        out = out[::-1].copy()
    return out


def check(poses, times, motion_path):
    """Scale that best explains a trajectory's acceleration by the IMU's."""
    motion = [json.loads(l) for l in open(motion_path) if l.strip()]
    if len(poses) < 60 or len(motion) < 100:
        return None
    P = poses[:, :3, 3]
    dt = float(np.median(np.diff(times)))
    grid = np.arange(times[0], times[-1], dt)
    Pg = np.stack([np.interp(grid, times, P[:, i]) for i in range(3)], 1)
    acc_world = np.diff(lowpass(Pg, dt), n=2, axis=0) / (dt * dt)

    # Into the frame CoreMotion reports in, using the trajectory's own attitude,
    # so no unknown yaw is left between the two signals. Mixing the frames leaves
    # magnitude intact and destroys the per-axis correlation, which is a precise
    # enough signature to recognise: it shows up as a plausible ratio next to a
    # correlation of zero.
    idx = np.clip(np.searchsorted(times, grid), 0, len(poses) - 1)
    acc_dev = np.empty_like(acc_world)
    for i in range(len(acc_world)):
        # Dumped poses are world-from-camera in the depth frame; ARKit's camera
        # frame is one basis change back, and the device frame one more.
        R_wc = poses[idx[i + 1], :3, :3] @ ARKIT_TO_DEPTH.T
        acc_dev[i] = R_CAM_FROM_DEV.T @ (R_wc.T @ acc_world[i])

    # `userAcceleration` has gravity removed already, is in g, and carries the
    # accelerometer's sign convention — it reports the reaction, so it is the
    # negative of the coordinate acceleration.
    tm = np.array([m["t"] for m in motion])
    A = -np.array([[m["ax"], m["ay"], m["az"]] for m in motion]) * G
    acc_imu = lowpass(np.stack([np.interp(grid, tm, A[:, i]) for i in range(3)], 1),
                      dt)[1:-1]

    n = min(len(acc_dev), len(acc_imu))
    at, ai = acc_dev[:n], acc_imu[:n]
    denom = float((ai * ai).sum())
    if denom < 1e-9:
        return None
    # A least-squares fit, not a ratio of magnitudes. Differentiation noise adds
    # in quadrature to a magnitude but not to a cross term, so an RMS ratio is
    # biased upward while this is not.
    return {"fit": float((at * ai).sum() / denom),
            "corr": float(np.corrcoef(at.ravel(), ai.ravel())[0, 1]),
            "samples": n}


def main(argv):
    if len(argv) < 2:
        print(__doc__.strip().splitlines()[-1])
        return 2
    dump, session = Path(argv[0]), Path(argv[1])
    z = np.load(dump)
    times = z["t"].astype(np.float64)
    motion = session / "motion.jsonl"
    rows = []
    for name, key in (("ARKit (fuses the IMU — a noise floor, not a test)", "reference"),
                      ("depth ICP (never sees the accelerometer)", "estimate")):
        got = check(z[key].astype(np.float64), times, motion)
        rows.append((name, got))
    print(f"{session.name}")
    for name, got in rows:
        if got is None:
            print(f"  {name:52} unavailable")
            continue
        print(f"  {name:52} scale {got['fit']:.3f}  corr {got['corr']:+.3f}"
              f"  n={got['samples']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
