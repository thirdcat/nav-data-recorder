#!/usr/bin/env python3
"""Self-calibrate the ultra-wide by carrying LiDAR-anchored points out of the cone.

`docs/3DGS.md` measures that adding the raw ultra-wide arm to a training set
costs the *same wide photographs* 0.7 dB, and nobody can say whether that is
pose or camera model.  Both existing routes to the ultra-wide's camera model are
blind, in complementary places: the lens-to-lens similarity fit sees only the
inner 57 % of the radius (that is all the wide arm overlaps), and the plumb-line
fit reaches the corner but the rooms do not offer straight edges long enough to
resolve it.  The factory table is measured on a format nothing records at and is
not an input here, in either direction.

The opening this exploits is that **depth does not have to be observed at the
instant it is used**.  LiDAR covers only the central cone, so projecting it into
the ultra-wide constrains only the region already known to be fine — but a point
whose 3D position is fixed by LiDAR *while it sits inside the cone* can be
tracked outward and used to constrain the model *where there is no depth at
all*.  On `d06152`, 903 of 1125 cone-anchored tracks reach past the cone and the
furthest reaches 2211 px against a 2203 px corner.

    # the measurement
    python3 eval/uw_selfcal.py ~/nav_data/20260820-211848-d06152 \
        --poses pi3traj/d06152_wide_depth.npz --variants --bootstrap 40

    # the gate: warp the frames by a known radial map and demand it back
    python3 eval/uw_selfcal.py ~/nav_data/20260820-211848-d06152 \
        --poses pi3traj/d06152_wide_depth.npz --inject-sweep -80,-40,-20,20,40,80

    # the null control: the same pipeline on the wide arm, which has depth over
    # most of its frame and which three instruments agree is nearly rectilinear
    python3 eval/uw_selfcal.py ~/nav_data/20260820-211848-d06152 \
        --poses pi3traj/d06152_wide_depth.npz --lens wide --variants

    # criterion 4: undistort with the fitted model, then run the profile tool
    python3 eval/uw_selfcal.py ~/nav_data/20260820-211848-d06152 \
        --poses pi3traj/d06152_wide_depth.npz --write-undistorted /tmp/undist

Only numpy, scipy and OpenCV.  No GPU, no torch, no weights, no training.

**The confound is the whole point.**  Trajectory error also produces
reprojection error, so the pose is given six free degrees of freedom per frame
and the camera model is credited only with what survives that.  The two are
separated by shape, and the shape argument is in `docs/UW_SELFCAL_PREREG.md`:
every pose degree of freedom except forward translation averages to zero when
the radial residual component is averaged over azimuth, and forward translation
survives only at order rho^1 weighted by 1/Z, where the lens term is rho^3 and
rho^5 and depth-independent.

**One radial term, not two.**  `--order` defaults to 1.  `D_corner` is evaluated
at `rho = 1` and the tracks reach 0.95-0.98, so the corner is always a short
extrapolation; with two coefficients trading along `rho^2` against `rho^4` that
extrapolation is unstable, and the two-term fit **fails the injection gate on
the wide arm** — the arm whose answer three other instruments already agree on —
with a response slope of 1.539 where the one-term fit gives 1.081.

Thresholds, gates and the decision rule were written to
`docs/UW_SELFCAL_PREREG.md` before this file existed.  Two of them did not
survive contact with a control and the page says which and why: the `model/pose`
residual ratio was retracted because it fails on a synthetic input whose answer
is known, and the radial order was cut from two terms to one.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - the CLI reports this clearly.
    cv2 = None

try:
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix
except ImportError:  # pragma: no cover
    least_squares = None
    lil_matrix = None

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "eval"), str(REPO / "tools")]

from read_session import Session  # noqa: E402
import pi3_poseless as PL  # noqa: E402
import export_3dgs as EX  # noqa: E402
# The radial model, the injection warp and the exact composition of two radial
# maps already exist in the plumb-line instrument and are unit-tested there.
# Re-deriving any of them here would give the repository two answers to one
# question, and the copy is the one that goes stale.
from fit_uw_distortion import (  # noqa: E402
    compose_expected, corner_displacement, half_diagonal, inject_distortion,
    is_monotone, rectify_points, rms_displacement,
)

# `d06152` logs `wide fov 69.52744`; `57f29e` and `d0f44f` were recorded before
# the recorder was made symmetric and log nothing for the wide lens.  Same
# device, same 640x480 wide format, and `docs/POSE.md` measures the factory
# 72.58 to be wrong by 3.1 degrees, so the logged value is carried across rather
# than falling back to a number already known to be wrong.  This is an
# assumption and the report says so.
WIDE_HFOV_FALLBACK_DEG = 69.52744

# Where the wide arm's cone ends in the ultra-wide frame: the wide half-diagonal
# is 40.94 deg at 640x480 / 69.52744 deg, so `f_uw * tan(40.94)` = 1249 px at
# 3840x2160 against a 2203 px half-diagonal.  Reported, never used as a filter —
# the filter is "the LiDAR returned here", which is the physical cone.
WIDE_HALF_DIAGONAL_DEG = math.degrees(math.atan(
    (0.5 * math.hypot(640.0, 480.0))
    / ((640.0 / 2.0) / math.tan(math.radians(WIDE_HFOV_FALLBACK_DEG) / 2.0))))

MAX_DEPTH_DT_S = 0.05

# Bound on each radial coefficient.  See `solve` for why it exists.
COEFFICIENT_LIMIT = 0.35


# ---------------------------------------------------------------------------
# The radial model.  `rectify_points` (raw -> pinhole) comes from
# fit_uw_distortion; projection needs the inverse, which does not exist there.
# ---------------------------------------------------------------------------

def invert_radius(rho_rect: np.ndarray, coefficients: Sequence[float],
                  iterations: int = 12) -> np.ndarray:
    """Raw normalised radius whose rectified radius is `rho_rect`.

    Solves `x (1 + a1 x^2 + a2 x^4 + ...) = rho_rect` by Newton from
    `x = rho_rect`.  The map is monotone over the frame wherever `is_monotone`
    holds, so Newton converges to float tolerance in a few steps;
    `test_uw_selfcal.py` checks the round trip rather than trusting that.
    """
    x = np.asarray(rho_rect, dtype=np.float64).copy()
    for _ in range(iterations):
        x2 = x * x
        gain = np.ones_like(x)
        derivative = np.ones_like(x)
        term = np.ones_like(x)
        for power, coefficient in enumerate(coefficients, start=1):
            term = term * x2
            gain = gain + coefficient * term
            derivative = derivative + (2 * power + 1) * coefficient * term
        x = x - (x * gain - rho_rect) / np.where(np.abs(derivative) < 1e-6,
                                                 1e-6, derivative)
    return x


def distort_points(points: np.ndarray, width: int, height: int,
                   coefficients: Sequence[float]) -> np.ndarray:
    """Pinhole pixels -> recorded pixels.  The inverse of `rectify_points`."""
    centre = np.array([0.5 * width, 0.5 * height], dtype=np.float64)
    radius = half_diagonal(width, height)
    delta = np.asarray(points, dtype=np.float64) - centre
    rho_rect = np.hypot(delta[..., 0], delta[..., 1]) / radius
    rho_raw = invert_radius(rho_rect, coefficients)
    ratio = np.where(rho_rect > 1e-12, rho_raw / np.where(rho_rect > 1e-12,
                                                          rho_rect, 1.0), 1.0)
    return centre + delta * ratio[..., None]


def warp_map(width: int, height: int, coefficients: Sequence[float]
             ) -> tuple[np.ndarray, np.ndarray]:
    """The remap `inject_distortion` applies, so the depth can take the same one.

    An injected frame is a simulation of a lens with more distortion, and that
    lens would put its registered LiDAR depth at the distorted positions too.
    Warping the image without warping the depth would inject a depth-to-image
    registration error instead of a distortion, which is a different experiment.
    `test_uw_selfcal.py` asserts this map reproduces `inject_distortion` on the
    image path, so there is one map and not two.
    """
    xs = np.arange(width, dtype=np.float64)
    ys = np.arange(height, dtype=np.float64)
    u, v = np.meshgrid(xs, ys)
    source = rectify_points(np.stack([u, v], axis=-1), width, height, coefficients)
    return (source[..., 0].astype(np.float32), source[..., 1].astype(np.float32))


def undistort_map(width: int, height: int, coefficients: Sequence[float]
                  ) -> tuple[np.ndarray, np.ndarray]:
    """The remap that straightens a recorded frame onto the same canvas."""
    xs = np.arange(width, dtype=np.float64)
    ys = np.arange(height, dtype=np.float64)
    u, v = np.meshgrid(xs, ys)
    source = distort_points(np.stack([u, v], axis=-1), width, height, coefficients)
    return (source[..., 0].astype(np.float32), source[..., 1].astype(np.float32))


def rodrigues(omega: np.ndarray) -> np.ndarray:
    """(N,3) rotation vectors -> (N,3,3) rotation matrices."""
    omega = np.atleast_2d(np.asarray(omega, dtype=np.float64))
    theta = np.linalg.norm(omega, axis=1)
    out = np.tile(np.eye(3), (len(omega), 1, 1))
    big = theta > 1e-12
    if big.any():
        axis = omega[big] / theta[big, None]
        c = np.cos(theta[big])[:, None, None]
        s = np.sin(theta[big])[:, None, None]
        K = np.zeros((int(big.sum()), 3, 3))
        K[:, 0, 1], K[:, 0, 2] = -axis[:, 2], axis[:, 1]
        K[:, 1, 0], K[:, 1, 2] = axis[:, 2], -axis[:, 0]
        K[:, 2, 0], K[:, 2, 1] = -axis[:, 1], axis[:, 0]
        out[big] = (c * np.eye(3) + s * K
                    + (1 - c) * (axis[:, :, None] * axis[:, None, :]))
    return out


# ---------------------------------------------------------------------------
# Session plumbing
# ---------------------------------------------------------------------------

class Arm:
    """One lens of one walk: frames, poses, intrinsics, depth.

    The two arms differ in exactly three places and they are all here rather
    than sprinkled through the fit: which image stream is read, whether the
    trajectory has to be interpolated and composed with the rig extrinsic, and
    which of `pi3_poseless`'s two depth reprojections applies.

    **The shutters do not fire together.**  `docs/3DGS.md` measures the two
    lenses 56-65 ms apart at the median and up to 103 ms, over which the camera
    travels 34-39 mm — more than the 19.272 mm rig baseline being applied.  So
    the wide trajectory is SLERP-interpolated to the ultra-wide frame's own
    timestamp, using `tools/export_3dgs.py`'s `interpolate_camera_pose`, which
    `docs/RIG_EXPORT_PREREG.md` shows passing a gate the nearest-pose
    construction fails.  Importing it rather than re-deriving it means
    `tools/test_export_3dgs.py` already guards this composition.
    """

    def __init__(self, session_dir: str, poses_npz: str, lens: str,
                 scale: float, frames: int, start: int | None,
                 pose_key: str = "estimate", hfov: float | None = None):
        if cv2 is None:
            raise SystemExit("OpenCV is required: pip install opencv-python")
        self.session = Session(session_dir)
        self.dir = Path(session_dir)
        self.lens = lens
        manifest_path = self.dir / "manifest.json"
        self.manifest = (json.loads(manifest_path.read_text())
                         if manifest_path.exists() else {})
        if self.manifest.get("kind") != "multicam":
            raise SystemExit(f"{session_dir} is not a multicam session")

        stream = "frames" if lens == "ultrawide" else "frames_wide"
        rows = list(self.session.stream(stream))
        if not rows:
            raise SystemExit(f"{session_dir} has no {stream}.jsonl")

        # Intrinsics for the recorded format, never the factory table scaled.
        calib = PL.load_calibration(REPO / "calib" / f"iphone17-1_{lens}.json")
        logged = hfov if hfov is not None else PL.active_hfov(self.manifest, lens)
        self.hfov_assumed = False
        if logged is None and lens == "wide":
            logged = WIDE_HFOV_FALLBACK_DEG
            self.hfov_assumed = True
        if logged is None:
            raise SystemExit(f"no active field of view for the {lens} lens; "
                             f"pass --hfov")
        self.hfov = float(logged)
        self.native_w, self.native_h = int(rows[0]["width"]), int(rows[0]["height"])
        self.scale = float(scale)
        self.w = int(round(self.native_w * self.scale))
        self.h = int(round(self.native_h * self.scale))
        self.K = PL.intrinsics_for(calib, self.w, self.h, self.hfov)
        self.f = float(self.K[0, 0])
        self.R_half = half_diagonal(self.w, self.h)
        self.native_R_half = half_diagonal(self.native_w, self.native_h)

        self.depth_calib, self.depth_source = PL.depth_calibration(self.manifest)

        traj = np.load(poses_npz, allow_pickle=True)
        mats = np.asarray(traj[pose_key], dtype=np.float64)
        times = np.asarray(traj["t"], dtype=np.float64)
        order = np.argsort(times)
        self.traj_t, self.traj_T = times[order], mats[order]
        self.join_scale_mode = (str(traj["join_scale_mode"])
                                if "join_scale_mode" in traj.files
                                else "free (predates the flag)")
        self.broken_from = (int(traj["broken_from"])
                            if "broken_from" in traj.files else -1)
        # `interpolate_camera_pose` returns an exact sample untouched, so the
        # wide arm — whose own frames own the trajectory — goes through the
        # identical code path with no interpolation actually happening, and the
        # rig extrinsic is identity.
        self.E = (EX.rig_extrinsic(EX.calib_path("ultrawide"))
                  if lens == "ultrawide" else np.eye(4))

        depth_rows = sorted(self.session.depth_index(), key=lambda d: d["t"])
        if not depth_rows:
            raise SystemExit("this session has no depth index")
        dts = np.array([d["t"] for d in depth_rows])

        usable, no_bracket, no_depth = [], 0, 0
        for row in rows:
            T = EX.interpolate_camera_pose(self.traj_t, self.traj_T, float(row["t"]))
            if T is None:
                no_bracket += 1
                continue
            k = int(np.argmin(np.abs(dts - row["t"])))
            if abs(dts[k] - row["t"]) > MAX_DEPTH_DT_S:
                no_depth += 1
                continue
            usable.append({"frame": int(row["frame"]), "t": float(row["t"]),
                           "file": row["file"], "pose": T @ self.E,
                           "depth": depth_rows[k],
                           "depth_dt": float(row["t"] - dts[k])})
        if len(usable) < 8:
            raise SystemExit(f"only {len(usable)} {lens} frames are both posed "
                             f"and paired with depth")
        self.no_bracket, self.no_depth = no_bracket, no_depth

        # A fixed, contiguous, centred block, so the choice of frames cannot be
        # tuned after seeing an answer.
        n = min(frames, len(usable))
        if start is None:
            start = max(0, (len(usable) - n) // 2)
        self.rows = usable[start:start + n]
        self.n_usable = len(usable)
        self.start = start

        self.images: list[np.ndarray] | None = None
        self._depth: list[np.ndarray] | None = None

    # -- images and depth --------------------------------------------------

    def load_images(self) -> list[np.ndarray]:
        if self.images is None:
            out = []
            for row in self.rows:
                img = cv2.imread(str(self.dir / row["file"]), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise SystemExit(f"cannot read {row['file']}")
                if (img.shape[1], img.shape[0]) != (self.w, self.h):
                    img = cv2.resize(img, (self.w, self.h),
                                     interpolation=cv2.INTER_AREA)
                out.append(img)
            self.images = out
        return self.images

    def depth_maps(self) -> list[np.ndarray]:
        """LiDAR depth on each frame's grid, as z in *this lens's* camera frame.

        Cached as float16, which is what `depth.bin` stores anyway, because the
        anchoring pass runs once per fit and there are several fits per session.
        """
        if self._depth is None:
            out = []
            for i in range(len(self.rows)):
                row = {"depth": self.rows[i]["depth"],
                       "width": self.w, "height": self.h}
                if self.lens == "ultrawide":
                    z = PL.depth_in_ultrawide(self.session, row, self.K,
                                              self.depth_calib, self.E)
                else:
                    z = PL.depth_in_wide(self.session, row, self.K, self.depth_calib)
                out.append(z.astype(np.float16))
            self._depth = out
        return self._depth

    # -- reporting ---------------------------------------------------------

    def cone_radius(self) -> float:
        """Where the wide arm's cone ends, in this arm's working pixels."""
        if self.lens != "ultrawide":
            return self.R_half
        return self.f * math.tan(math.radians(WIDE_HALF_DIAGONAL_DEG))

    def describe(self) -> dict[str, Any]:
        gaps = np.array([abs(r["depth_dt"]) for r in self.rows])
        return {
            "session": self.dir.name, "lens": self.lens,
            "native": [self.native_w, self.native_h],
            "working": [self.w, self.h],
            "hfov_deg": self.hfov, "hfov_assumed": self.hfov_assumed,
            "f_working_px": self.f, "half_diagonal_working_px": self.R_half,
            "half_diagonal_native_px": self.native_R_half,
            "cone_radius_native_px": (self.cone_radius() / self.R_half
                                      * self.native_R_half),
            "frames": len(self.rows), "frames_available": self.n_usable,
            "frames_outside_the_posed_walk": self.no_bracket,
            "frames_without_depth": self.no_depth,
            "start": self.start,
            "join_scale_mode": self.join_scale_mode,
            "broken_from": self.broken_from,
            "depth_source": self.depth_source,
            "depth_gap_median_ms": float(np.median(gaps) * 1000),
            "depth_gap_worst_ms": float(gaps.max() * 1000),
            "rig_baseline_mm": float(np.linalg.norm(self.E[:3, 3]) * 1000),
            "shutter_offset_median_ms": self.shutter_offset(),
        }

    def shutter_offset(self) -> float | None:
        """Median gap to the nearest wide frame — the reason for the SLERP."""
        if self.lens != "ultrawide":
            return 0.0
        wide = [r["t"] for r in self.session.stream("frames_wide")]
        if not wide:
            return None
        wt = np.array(wide)
        gaps = [abs(wt - r["t"]).min() for r in self.rows]
        return float(np.median(gaps) * 1000)


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------

def track(images: Sequence[np.ndarray], *, max_corners: int = 4000,
          quality: float = 0.01, min_distance: int = 10,
          win: int = 31, levels: int = 5, fb_tolerance: float = 1.5,
          min_length: int = 3, valid: np.ndarray | None = None
          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pyramidal Lucas-Kanade tracks, forward-backward checked.

    Returns `(obs_track, obs_frame, obs_uv)`: one row per observation, tracks
    numbered densely from zero.  Features are re-seeded on every frame away from
    live tracks, so a track that dies does not leave the population thinning
    towards the end of the block.

    **The window and the pyramid are sized from the measured motion, not from
    habit.**  On `d06152` the camera turns 2-12 degrees between consecutive
    frames at 5 Hz, which at the working focal length of 721 px is 25-250 px of
    image displacement, and the displacement is largest exactly where this
    instrument needs observations — at large radius.  A 21 px window over four
    levels reaches about 168 px and loses them: it returns 8 135 observations of
    which 298 sit past `rho = 0.8`, against 30 154 and 1 422 for the settings
    here.  That was chosen on the *census*, before any coefficient was fitted
    from it.

    The forward-backward check is what keeps this honest at large radius.  A
    drifting track is a systematic displacement that grows with the number of
    frames tracked, which correlates with radius, which is exactly the signature
    being measured.
    """
    lk = dict(winSize=(win, win), maxLevel=levels,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    n = len(images)
    live: dict[int, np.ndarray] = {}
    obs: list[tuple[int, int, float, float]] = []
    next_id = 0

    for i in range(n):
        if i > 0 and live:
            ids = list(live)
            p0 = np.float32([live[k] for k in ids]).reshape(-1, 1, 2)
            p1, st, _ = cv2.calcOpticalFlowPyrLK(images[i - 1], images[i],
                                                 p0, None, **lk)
            p0b, st_b, _ = cv2.calcOpticalFlowPyrLK(images[i], images[i - 1],
                                                    p1, None, **lk)
            fb = np.linalg.norm(p0.reshape(-1, 2) - p0b.reshape(-1, 2), axis=1)
            ok = (st.reshape(-1) == 1) & (st_b.reshape(-1) == 1) & (fb < fb_tolerance)
            pts = p1.reshape(-1, 2)
            h, w = images[i].shape[:2]
            ok &= ((pts[:, 0] > 2) & (pts[:, 0] < w - 3)
                   & (pts[:, 1] > 2) & (pts[:, 1] < h - 3))
            if valid is not None:
                # A warped frame has a hard black boundary where the source ran
                # off the sensor, and LK locks onto it: that edge moves with the
                # *frame*, not with the scene, so it produces long confident
                # tracks that report zero parallax at large radius.  On the wide
                # arm it took a +15 px injection to +101.8 px recovered.  A
                # track that touches the invalid region is dropped, not trimmed.
                inside = valid[np.clip(pts[:, 1].astype(int), 0, h - 1),
                               np.clip(pts[:, 0].astype(int), 0, w - 1)]
                ok &= inside
            live = {k: pts[j] for j, k in enumerate(ids) if ok[j]}
        for k, p in live.items():
            obs.append((k, i, float(p[0]), float(p[1])))

        budget = max_corners - len(live)
        if budget > 0:
            mask = (np.full(images[i].shape, 255, np.uint8) if valid is None
                    else (valid.astype(np.uint8) * 255))
            for p in live.values():
                cv2.circle(mask, (int(round(p[0])), int(round(p[1]))),
                           min_distance, 0, -1)
            fresh = cv2.goodFeaturesToTrack(images[i], maxCorners=budget,
                                            qualityLevel=quality,
                                            minDistance=min_distance,
                                            blockSize=7, mask=mask)
            if fresh is not None:
                for p in fresh.reshape(-1, 2):
                    live[next_id] = p
                    obs.append((next_id, i, float(p[0]), float(p[1])))
                    next_id += 1

    if not obs:
        return (np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, 2)))
    arr = np.array(obs, dtype=np.float64)
    tid = arr[:, 0].astype(np.int64)
    counts = np.bincount(tid, minlength=next_id)
    keep = counts[tid] >= min_length
    arr, tid = arr[keep], tid[keep]
    _, tid = np.unique(tid, return_inverse=True)
    return tid, arr[:, 1].astype(np.int64), arr[:, 2:4]


# ---------------------------------------------------------------------------
# Anchoring
# ---------------------------------------------------------------------------

def anchor(arm: Arm, obs_track: np.ndarray, obs_frame: np.ndarray,
           obs_uv: np.ndarray, *, depth_warp: Sequence[float] | None = None,
           consistency: float = 0.05, patch: int = 2,
           min_valid: float = 0.6) -> dict[str, Any]:
    """Fix each track's 3D position from LiDAR while it is inside the cone.

    Only `Z` is read from the depth map.  The ray direction comes from the
    camera model being fitted, which makes the anchor first-order insensitive to
    an error in the *depth camera's* cone — unlogged on two of the three
    sessions here, and known from `docs/POSE.md` not to be a constant of the
    device at all.  Unprojecting the depth pixel instead would import that error
    as a ray direction, which is the quantity under measurement.

    `depth_warp` moves the pinhole-scattered depth to where a lens with those
    coefficients would put it.  That is how the second pass re-reads `Z` under
    the fitted model rather than under a pinhole, and how an injected run keeps
    its depth registered to its warped image.

    A depth pixel covers several image pixels, so `depth_in_*` dilates to close
    the lattice it leaves; near a discontinuity that dilation can carry a
    background return onto a foreground pixel.  An anchor is therefore accepted
    only where a small neighbourhood agrees to `consistency` and is at least
    `min_valid` filled.
    """
    n_tracks = int(obs_track.max()) + 1 if len(obs_track) else 0
    best_r = np.full(n_tracks, np.inf)
    best = np.full(n_tracks, -1, dtype=np.int64)
    depth_at = np.zeros(n_tracks)

    cx, cy = arm.w / 2.0, arm.h / 2.0
    radius = np.hypot(obs_uv[:, 0] - cx, obs_uv[:, 1] - cy)
    remap = warp_map(arm.w, arm.h, depth_warp) if depth_warp is not None else None
    maps = arm.depth_maps()

    order = np.argsort(obs_frame, kind="stable")
    bounds = np.searchsorted(obs_frame[order], np.arange(len(arm.rows) + 1))
    for i in range(len(arm.rows)):
        sel = order[bounds[i]:bounds[i + 1]]
        if not len(sel):
            continue
        z = maps[i].astype(np.float32)
        if remap is not None:
            z = cv2.remap(z, remap[0], remap[1], cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        iu = np.clip(np.round(obs_uv[sel, 0]).astype(int), patch, arm.w - patch - 1)
        iv = np.clip(np.round(obs_uv[sel, 1]).astype(int), patch, arm.h - patch - 1)
        win = np.stack([z[iv + dy, iu + dx]
                        for dy in range(-patch, patch + 1)
                        for dx in range(-patch, patch + 1)], axis=1)
        valid = win > 0.05
        frac = valid.mean(axis=1)
        centre = z[iv, iu].astype(np.float64)
        lo = np.where(valid, win, np.inf).min(axis=1)
        hi = np.where(valid, win, -np.inf).max(axis=1)
        with warnings.catch_warnings():
            # An all-empty neighbourhood is a hole, which `frac` already
            # rejects; nanmedian says so loudly and there is nothing to fix.
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(np.where(valid, win, np.nan), axis=1)
        good = ((centre > 0.05) & (frac >= min_valid) & np.isfinite(med)
                & ((hi - lo) <= consistency * np.maximum(med, 1e-6)))
        cand = np.flatnonzero(good)
        for j in cand:
            o = sel[j]
            t = obs_track[o]
            if radius[o] < best_r[t]:
                best_r[t] = radius[o]
                best[t] = o
                depth_at[t] = centre[j]

    return {"anchor_obs": best, "anchored": best >= 0,
            "anchor_depth": depth_at, "radius": radius,
            "anchor_radius": best_r}


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------

class Problem:
    """Observations, anchors and chain poses, ready to be minimised over.

    Parameters, in one vector:

        [a1 .. a_order]  the radial model
        [xi_1 .. xi_N-1] a 6-vector SE(3) correction per frame, frame 0 held
                         fixed, which is the whole gauge; scale is pinned by the
                         LiDAR depths themselves
        [s]              one global multiplier on the chain's positions about
                         frame 0, frozen after a pre-alignment (see `align`)
    """

    def __init__(self, arm: Arm, obs_track, obs_frame, obs_uv, anchors,
                 order: int = 2):
        keep_track = anchors["anchored"]
        use = keep_track[obs_track]
        tids = np.unique(obs_track[use])
        remap = -np.ones(int(obs_track.max()) + 1, dtype=np.int64)
        remap[tids] = np.arange(len(tids))

        self.arm = arm
        self.order = order
        self.obs_track = remap[obs_track[use]]
        self.obs_frame = obs_frame[use]
        self.obs_uv = obs_uv[use]
        anchor_obs = anchors["anchor_obs"][tids]
        self.trk_uv = obs_uv[anchor_obs]
        self.trk_frame = obs_frame[anchor_obs]
        self.trk_z = anchors["anchor_depth"][tids]
        self.trk_radius = anchors["anchor_radius"][tids]
        self.n_tracks = len(tids)
        self.n_frames = len(arm.rows)
        self.n_params = order + 6 * (self.n_frames - 1) + 1
        self.T_chain = np.stack([r["pose"] for r in arm.rows])

        # The anchor observation carries no information — it is where the point
        # was defined and its residual is zero by construction — so it is
        # excluded, and the residual count is the number of *carried*
        # observations.
        self.is_anchor = np.zeros(len(self.obs_track), bool)
        position = {int(o): j for j, o in enumerate(np.flatnonzero(use))}
        for o in anchor_obs:
            j = position.get(int(o))
            if j is not None:
                self.is_anchor[j] = True
        self.active = ~self.is_anchor
        self.weight = np.ones(self.n_tracks)

        self.centre = np.array([arm.w / 2.0, arm.h / 2.0])
        self.obs_radius = np.hypot(obs_uv[use, 0] - self.centre[0],
                                   obs_uv[use, 1] - self.centre[1])

    # -- parameter packing -------------------------------------------------

    def unpack(self, params: np.ndarray):
        a = params[:self.order]
        xi = np.zeros((self.n_frames, 6))
        xi[1:] = params[self.order:-1].reshape(self.n_frames - 1, 6)
        return a, xi, params[-1]

    def zero(self) -> np.ndarray:
        x = np.zeros(self.n_params)
        x[-1] = 1.0
        return x

    # -- geometry ----------------------------------------------------------

    def poses(self, xi: np.ndarray, scale: float):
        Rc = rodrigues(xi[:, :3])
        Rw = self.T_chain[:, :3, :3] @ Rc
        origin = self.T_chain[0, :3, 3]
        base = origin + scale * (self.T_chain[:, :3, 3] - origin)
        tw = base + np.einsum("nij,nj->ni", self.T_chain[:, :3, :3], xi[:, 3:])
        return Rw, tw

    def points(self, a, Rw, tw):
        """Each track's 3D point: the anchor ray through the model, at LiDAR range."""
        rect = rectify_points(self.trk_uv, self.arm.w, self.arm.h, a)
        ray = np.stack([(rect[:, 0] - self.centre[0]) / self.arm.f,
                        (rect[:, 1] - self.centre[1]) / self.arm.f,
                        np.ones(self.n_tracks)], axis=1)
        cam = self.trk_z[:, None] * ray
        j = self.trk_frame
        return np.einsum("nij,nj->ni", Rw[j], cam) + tw[j]

    def predict(self, params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a, xi, scale = self.unpack(params)
        Rw, tw = self.poses(xi, scale)
        P = self.points(a, Rw, tw)
        j = self.obs_frame
        rel = P[self.obs_track] - tw[j]
        X = np.einsum("nji,nj->ni", Rw[j], rel)          # R^T rel
        z = X[:, 2]
        ahead = z > 0.05
        zz = np.where(ahead, z, 1.0)
        pin = np.stack([self.arm.f * X[:, 0] / zz + self.centre[0],
                        self.arm.f * X[:, 1] / zz + self.centre[1]], axis=1)
        return distort_points(pin, self.arm.w, self.arm.h, a), ahead

    def live(self, ahead: np.ndarray) -> np.ndarray:
        return ahead & self.active

    def residual(self, params: np.ndarray) -> np.ndarray:
        raw, ahead = self.predict(params)
        e = raw - self.obs_uv
        w = self.weight[self.obs_track]
        e = e * np.sqrt(np.maximum(w, 0.0))[:, None]
        e = np.where(self.live(ahead)[:, None], e, 0.0)
        return np.nan_to_num(e, nan=0.0, posinf=0.0, neginf=0.0).ravel()

    def sparsity(self, free: np.ndarray):
        rows = np.arange(len(self.obs_track))
        cols: list[np.ndarray] = []
        rws: list[np.ndarray] = []
        for c in range(self.order):
            cols.append(np.full(len(rows), c))
            rws.append(rows)
        cols.append(np.full(len(rows), self.n_params - 1))
        rws.append(rows)
        for arr in (self.obs_frame, self.trk_frame[self.obs_track]):
            alive = arr > 0
            base = self.order + 6 * (arr[alive] - 1)
            for k in range(6):
                cols.append(base + k)
                rws.append(rows[alive])
        r = np.concatenate(rws)
        c = np.concatenate(cols)
        index = -np.ones(self.n_params, dtype=np.int64)
        index[free] = np.arange(free.sum())
        keep = index[c] >= 0
        r, c = r[keep], index[c[keep]]
        S = lil_matrix((2 * len(self.obs_track), int(free.sum())), dtype=np.int8)
        S[2 * r, c] = 1
        S[2 * r + 1, c] = 1
        return S


def align(problem: Problem, lo: float = 0.35, hi: float = 3.0,
          steps: int = 48) -> float:
    """One global multiplier on the chain's positions, by grid then refinement.

    `57f29e` and `d0f44f` only have the historical `free` chains, whose join
    scales compound to 1.08 and 1.79; a `free` chain's local metric scale can be
    tens of per cent from the LiDAR's.  An initialisation that far out leaves a
    2 px robust loss with no gradient at all, so the bulk of it is taken out
    first.  This is one number for the whole block, it is frozen before any
    camera model is fitted, and it cannot represent a radial field — the report
    prints it so a reader can see how far the chain had to move.
    """
    def cost(s: float) -> float:
        x = problem.zero()
        x[-1] = s
        raw, ahead = problem.predict(x)
        live = problem.live(ahead)
        if not live.any():
            return np.inf
        d = np.linalg.norm(raw[live] - problem.obs_uv[live], axis=1)
        return float(np.median(d))

    grid = np.exp(np.linspace(math.log(lo), math.log(hi), steps))
    values = [cost(s) for s in grid]
    best = int(np.argmin(values))
    a = grid[max(best - 1, 0)]
    b = grid[min(best + 1, steps - 1)]
    fine = np.linspace(a, b, 21)
    return float(fine[int(np.argmin([cost(s) for s in fine]))])


def solve(problem: Problem, *, fit_model: bool, fit_pose: bool,
          init: np.ndarray | None = None, f_scale: float = 2.0,
          stages: Sequence[float] = (40.0, 8.0), max_nfev: int = 50
          ) -> dict[str, Any]:
    """Minimise reprojection error over the camera model and the pose, or either.

    Staged robustly: a `free`-chain initialisation can start tens of pixels out
    even after `align`, where a 2 px `soft_l1` has no gradient, so the loss width
    is walked down rather than started at its final value.  Fixed parameters are
    removed from the vector rather than bounded to zero, so the trust region
    never has to be conditioned on a direction that cannot move.
    """
    if least_squares is None:
        raise SystemExit("scipy is required: pip install scipy")
    x = problem.zero() if init is None else init.copy()
    if not fit_model:
        x[:problem.order] = 0.0
    if not fit_pose:
        x[problem.order:-1] = 0.0

    free = np.zeros(problem.n_params, bool)
    free[:problem.order] = fit_model
    free[problem.order:-1] = fit_pose
    if not free.any():
        # The `none` variant: nothing to solve, only to evaluate.  Reported
        # rather than skipped, because it is the floor the other three are read
        # against.
        a, xi, scale = problem.unpack(x)
        return {"x": x, "coefficients": [float(v) for v in a],
                "trajectory_scale": float(scale), "xi": xi, "at_bound": False,
                "cost": float(0.5 * (problem.residual(x) ** 2).sum())}
    S = problem.sparsity(free)

    def embed(p: np.ndarray) -> np.ndarray:
        full = x.copy()
        full[free] = p
        return full

    # The radial coefficients are bounded well outside any physical answer.
    # `COEFFICIENT_LIMIT` at the ultra-wide's 2203 px half-diagonal is a 770 px
    # corner displacement, twenty times the largest number anyone has proposed
    # for this lens; the bound exists only so a badly conditioned fit reports a
    # bound rather than a folded map with a corner displacement of 2 688 px,
    # which is what the wide arm's +30 px injection produced without it.
    lo = np.full(problem.n_params, -np.inf)
    hi = np.full(problem.n_params, np.inf)
    lo[:problem.order] = -COEFFICIENT_LIMIT
    hi[:problem.order] = COEFFICIENT_LIMIT

    # scipy's `lsmr` trust-region step solves a 2-D subproblem and indexes
    # `B[0, 1]`, so a *single* free parameter — the `model`-only variant at
    # `--order 1` — raises IndexError inside `solve_trust_region_2d`.  With one
    # parameter the Jacobian is a single column and there is nothing for the
    # sparsity pattern to save, so fall back to the exact dense solver.
    single = int(free.sum()) < 2
    extra: dict[str, Any] = ({"tr_solver": "exact"} if single
                             else {"jac_sparsity": S})

    result = None
    p = x[free].copy()
    for width in list(stages) + [f_scale]:
        result = least_squares(lambda q: problem.residual(embed(q)), p,
                               loss="soft_l1", f_scale=float(width),
                               method="trf", bounds=(lo[free], hi[free]),
                               x_scale="jac", max_nfev=max_nfev, **extra)
        p = result.x
    p = np.clip(p, lo[free], hi[free])
    x = embed(p)
    a, xi, scale = problem.unpack(x)
    return {"x": x, "coefficients": [float(v) for v in a],
            "trajectory_scale": float(scale), "xi": xi,
            "at_bound": bool(np.any(np.abs(a) >= COEFFICIENT_LIMIT - 1e-9)),
            "cost": float(result.cost) if result is not None else float("nan")}


def residual_stats(problem: Problem, params: np.ndarray) -> dict[str, Any]:
    raw, ahead = problem.predict(params)
    live = problem.live(ahead)
    d = np.linalg.norm(raw[live] - problem.obs_uv[live], axis=1)
    if not len(d):
        return {"n": 0, "rms_px": float("nan"), "median_px": float("nan"),
                "p90_px": float("nan")}
    return {"n": int(live.sum()),
            "rms_px": float(np.sqrt((d ** 2).mean())),
            "median_px": float(np.median(d)),
            "p90_px": float(np.percentile(d, 90))}


def prune(problem: Problem, params: np.ndarray, k: float = 5.0) -> int:
    """Drop observations the first pass cannot explain, by MAD."""
    raw, ahead = problem.predict(params)
    live = problem.live(ahead)
    if not live.any():
        return 0
    d = np.linalg.norm(raw - problem.obs_uv, axis=1)
    med = float(np.median(d[live]))
    mad = float(np.median(np.abs(d[live] - med))) or 1e-6
    drop = live & (d > med + k * 1.4826 * mad)
    problem.active &= ~drop
    return int(drop.sum())


# ---------------------------------------------------------------------------
# Radial versus pose: the shape decomposition
# ---------------------------------------------------------------------------

def shape_decomposition(problem: Problem, params: np.ndarray,
                        bins: int = 10) -> dict[str, Any]:
    """Split the residual field into radial and tangential parts, by radius.

    `docs/UW_SELFCAL_PREREG.md` derives why this separates the confound: every
    pose degree of freedom except forward translation has zero azimuthal mean in
    its radial component, and forward translation survives only at order rho^1
    weighted by 1/Z, where a radial distortion is rho^3 and rho^5 and
    depth-independent.
    """
    raw, ahead = problem.predict(params)
    live = problem.live(ahead)
    e = (problem.obs_uv - raw)[live]                 # observed minus predicted
    d = problem.obs_uv[live] - problem.centre
    r = np.hypot(d[:, 0], d[:, 1])
    good = r > 1e-6
    e, d, r = e[good], d[good], r[good]
    rhat = d / r[:, None]
    that = np.stack([-rhat[:, 1], rhat[:, 0]], axis=1)
    er = np.einsum("ni,ni->n", e, rhat)
    et = np.einsum("ni,ni->n", e, that)
    rho = r / problem.arm.R_half

    edges = np.linspace(0.0, max(float(rho.max()), 1e-6), bins + 1)
    idx = np.clip(np.digitize(rho, edges) - 1, 0, bins - 1)
    profile = []
    for b in range(bins):
        sel = idx == b
        if sel.sum() < 20:
            continue
        profile.append({
            "rho": float(rho[sel].mean()), "n": int(sel.sum()),
            "radial_px": float(er[sel].mean()),
            "radial_sem_px": float(er[sel].std(ddof=1) / math.sqrt(int(sel.sum()))),
            "tangential_px": float(et[sel].mean()),
            "tangential_sem_px": float(et[sel].std(ddof=1) / math.sqrt(int(sel.sum()))),
        })

    out: dict[str, Any] = {"profile": profile}
    if len(profile) >= 4:
        x = np.array([p["rho"] for p in profile])
        y = np.array([p["radial_px"] for p in profile])
        w = 1.0 / np.maximum([p["radial_sem_px"] for p in profile], 1e-6)
        design = np.stack([x, x ** 3, x ** 5], axis=1)
        sol, *_ = np.linalg.lstsq(design * w[:, None], y * w, rcond=None)
        out["radial_terms_px"] = {"b1": float(sol[0]), "b3": float(sol[1]),
                                  "b5": float(sol[2])}
        # `observed - predicted` pointing inward where the model under-distorts
        # means a1 = -b3 / R, a2 = -b5 / R.  The injection run checks this sign
        # rather than the reader having to trust it.
        implied = [float(-sol[1] / problem.arm.R_half),
                   float(-sol[2] / problem.arm.R_half)]
        out["implied_coefficients"] = implied
        out["implied_corner_px_native"] = float(
            problem.arm.native_R_half * sum(implied))
    return out


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------

def _fit_once(arm: Arm, obs, depth_warp, order: int):
    anchors = anchor(arm, *obs, depth_warp=depth_warp)
    problem = Problem(arm, *obs, anchors, order=order)
    if problem.n_tracks < 20:
        raise SystemExit(f"only {problem.n_tracks} tracks could be anchored")
    scale = align(problem)
    seed = problem.zero()
    seed[-1] = scale
    fit = solve(problem, fit_model=True, fit_pose=True, init=seed)
    dropped = prune(problem, fit["x"])
    fit = solve(problem, fit_model=True, fit_pose=True, init=fit["x"], stages=())
    return problem, fit, dropped


def run(arm: Arm, *, inject: Sequence[float] | None = None,
        variants: bool = False, bootstrap: int = 0, order: int = 2,
        seed: int = 42, quiet: bool = False,
        tracking: dict[str, Any] | None = None) -> dict[str, Any]:
    """Track, anchor, fit.  `inject` warps every frame first — image and depth."""
    t0 = time.time()
    images = arm.load_images()
    valid = None
    if inject is not None:
        mx, my = warp_map(arm.w, arm.h, inject)
        images = [cv2.remap(im, mx, my, cv2.INTER_LANCZOS4,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                  for im in images]
        # An expanding injection samples from beyond the sensor, so the warped
        # frame has a black margin.  `inject_distortion` returns this mask for
        # exactly that reason; ignoring it is what let the tracker follow the
        # synthetic edge instead of the scene.
        _, valid = inject_distortion(np.zeros((arm.h, arm.w), np.uint8), inject)

    obs = track(images, valid=valid, **(tracking or {}))
    obs_track, obs_frame, obs_uv = obs

    # Pass one: the depth is where a pinhole (plus any injection) puts it.
    warp0 = list(inject) if inject is not None else None
    problem, fit, dropped = _fit_once(arm, obs, warp0, order)
    first = list(fit["coefficients"])

    # Pass two: re-scatter the depth through the fitted model, so `Z` is read
    # where the fitted lens says the depth landed rather than where a pinhole
    # does.  Composition order matters — the injection is applied to an image
    # that already carries the native distortion.
    warp1 = (compose_expected(first, list(inject), order) if inject is not None
             else first)
    problem, fit, dropped2 = _fit_once(arm, obs, warp1, order)

    rho_obs = problem.obs_radius / arm.R_half
    census = {
        "tracks": int(obs_track.max()) + 1 if len(obs_track) else 0,
        "observations": int(len(obs_track)),
        "fitted_tracks": problem.n_tracks,
        "carried_observations": int(problem.active.sum()),
        "pruned_observations": dropped + dropped2,
        "observations_beyond_cone": int((problem.obs_radius > arm.cone_radius()).sum()),
        "observations_rho_ge_0.8": int((rho_obs >= 0.8).sum()),
        "max_rho": float(rho_obs.max()) if len(rho_obs) else 0.0,
        "max_radius_native_px": (float(rho_obs.max() * arm.native_R_half)
                                 if len(rho_obs) else 0.0),
        "anchor_radius_median_native_px": float(
            np.median(problem.trk_radius) / arm.R_half * arm.native_R_half),
        "anchor_depth_median_m": float(np.median(problem.trk_z)),
    }
    if not quiet:
        print(f"  {census['tracks']} tracks, {census['fitted_tracks']} anchored "
              f"and fitted, {census['carried_observations']} carried "
              f"observations, {census['observations_rho_ge_0.8']} at rho >= 0.8, "
              f"max rho {census['max_rho']:.3f} "
              f"({census['max_radius_native_px']:.0f} px native)")

    out: dict[str, Any] = {"census": census, "arm": arm.describe()}
    if inject is not None:
        out["injected"] = [float(v) for v in inject]
        out["injected_corner_px_native"] = corner_displacement(
            arm.native_w, arm.native_h, inject)

    out["fit"] = {
        "coefficients": fit["coefficients"],
        "trajectory_scale": fit["trajectory_scale"],
        "corner_px_native": corner_displacement(arm.native_w, arm.native_h,
                                                fit["coefficients"]),
        "rms_px_at_quarter_native": rms_displacement(
            arm.native_w // 4, arm.native_h // 4, fit["coefficients"]),
        "monotone": is_monotone(fit["coefficients"]),
        "at_bound": fit["at_bound"],
        "residual": residual_stats(problem, fit["x"]),
        "pose_correction": {
            "rotation_deg_median": float(np.degrees(np.median(
                np.linalg.norm(fit["xi"][:, :3], axis=1)))),
            "translation_cm_median": float(100 * np.median(
                np.linalg.norm(fit["xi"][:, 3:], axis=1))),
            # Per frame, so a reader can tell a smooth drift the chain
            # accumulated from frame-to-frame noise the optimiser invented.
            # Frame 0 is the gauge and is exactly zero by construction.
            "rotation_deg_per_frame": [
                float(v) for v in np.degrees(
                    np.linalg.norm(fit["xi"][:, :3], axis=1))],
            "translation_cm_per_frame": [
                float(v) for v in 100 * np.linalg.norm(fit["xi"][:, 3:], axis=1)],
        },
    }
    out["first_pass_coefficients"] = first
    out["first_pass_corner_px_native"] = corner_displacement(
        arm.native_w, arm.native_h, first)

    if variants:
        out["variants"] = {}
        base = problem.zero()
        base[-1] = fit["trajectory_scale"]
        for name, (fm, fp) in (("none", (False, False)), ("pose", (False, True)),
                               ("model", (True, False)), ("both", (True, True))):
            if name == "both":
                res, params = fit, fit["x"]
            else:
                res = solve(problem, fit_model=fm, fit_pose=fp, init=base)
                params = res["x"]
            out["variants"][name] = {
                "coefficients": res["coefficients"],
                "corner_px_native": corner_displacement(
                    arm.native_w, arm.native_h, res["coefficients"]),
                "residual": residual_stats(problem, params),
            }
            # The shape argument in the pre-registration assumes the residual is
            # `pose field + lens field` with neither modelled, so the place to
            # run it is `none`.  Run on the pose-only fit it is a *lower bound*:
            # the synthetic control in `test_uw_selfcal.py` shows a free
            # per-frame pose absorbing about 85 % of a known radial term, which
            # is exactly why the joint fit and not this is the estimator.
            if name in ("none", "pose"):
                out["shape_" + name] = shape_decomposition(problem, params)
        out["shape_both"] = shape_decomposition(problem, fit["x"])

    if bootstrap:
        rng = np.random.default_rng(seed)
        draws = []
        for _ in range(bootstrap):
            problem.weight = np.bincount(
                rng.integers(0, problem.n_tracks, problem.n_tracks),
                minlength=problem.n_tracks).astype(float)
            try:
                b = solve(problem, fit_model=True, fit_pose=True,
                          init=fit["x"], stages=(), max_nfev=15)
                draws.append(corner_displacement(arm.native_w, arm.native_h,
                                                 b["coefficients"]))
            except Exception:                      # pragma: no cover
                pass
        problem.weight = np.ones(problem.n_tracks)
        if draws:
            p05, p50, p95 = (float(np.percentile(draws, q)) for q in (5, 50, 95))
            out["bootstrap"] = {"n": len(draws), "corner_px_native_p05": p05,
                                "corner_px_native_p50": p50,
                                "corner_px_native_p95": p95,
                                "spans_zero": bool(p05 < 0 < p95)}

    out["seconds"] = round(time.time() - t0, 1)
    out["_problem"] = problem
    return out


# ---------------------------------------------------------------------------
# Criterion 4: hand the fitted model to an instrument that shares no input
# ---------------------------------------------------------------------------

def write_undistorted(arm: Arm, coefficients: Sequence[float], out_dir: str,
                      profile_frames: int = 40) -> str:
    """A session whose ultra-wide frames have been straightened, nothing else.

    `eval/uw_radial_profile.py` then runs on it **unmodified, as a subprocess**.
    It is not edited and not imported; it does not know it is being tested, and
    its own control — a frame against itself downscaled by a known 0.3200 —
    still runs first.
    """
    src, dst = arm.dir, Path(out_dir)
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "frames").mkdir(parents=True)
    for name in ("frames.jsonl", "frames_wide.jsonl", "depth.jsonl", "depth.bin",
                 "manifest.json", "motion.jsonl", "events.jsonl", "frames_wide"):
        if (src / name).exists():
            os.symlink(src / name, dst / name)

    rows = [json.loads(l) for l in (src / "frames.jsonl").read_text().splitlines()
            if l.strip()]
    # Only the frames the profile tool can reach need straightening; the rest are
    # symlinked so nothing is missing if its selection ever changes.
    import wide_fov_probe as W
    picked = {int(r[0]["frame"]) for r in W.pair_frames(src, profile_frames, 1)}
    picked.add(int(rows[len(rows) // 2]["frame"]))    # the tool's control frame

    mx, my = undistort_map(arm.native_w, arm.native_h, coefficients)
    made = 0
    for row in rows:
        target = dst / row["file"]
        if int(row["frame"]) not in picked:
            os.symlink(src / row["file"], target)
            continue
        img = cv2.imread(str(src / row["file"]), cv2.IMREAD_COLOR)
        out = cv2.remap(img, mx, my, cv2.INTER_LANCZOS4,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        cv2.imwrite(str(target), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
        made += 1
    print(f"  straightened {made} of {len(rows)} ultra-wide frames into {dst}")
    return str(dst)


def run_profile(session_dir: str) -> str:
    proc = subprocess.run(
        [sys.executable, str(REPO / "eval" / "uw_radial_profile.py"), session_dir],
        capture_output=True, text=True)
    return proc.stdout + proc.stderr


def profile_spread(text: str) -> dict[str, Any]:
    """Peak-to-trough of the annulus scales in a profile report, per block."""
    out: dict[str, Any] = {}
    block, values = None, []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("CONTROL"):
            if block and values:
                out[block] = {"values": values,
                              "spread_pct": 100 * (max(values) - min(values)) / min(values)}
            block, values = "control", []
        elif stripped.startswith("QUESTION"):
            if block and values:
                out[block] = {"values": values,
                              "spread_pct": 100 * (max(values) - min(values)) / min(values)}
            block, values = "lenses", []
        elif " px   s " in stripped:
            values.append(float(stripped.split(" s ")[1].split()[0]))
    if block and values:
        out[block] = {"values": values,
                      "spread_pct": 100 * (max(values) - min(values)) / min(values)}
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report_run(result: dict[str, Any]) -> None:
    arm, fit = result["arm"], result["fit"]
    print(f"\n  coefficients " + "  ".join(
        f"a{i + 1} {v:+.6f}" for i, v in enumerate(fit["coefficients"])))
    print(f"  corner displacement at {arm['native'][0]}x{arm['native'][1]}: "
          f"{fit['corner_px_native']:+.1f} px"
          + ("" if fit["monotone"] else "   [NOT MONOTONE — the map folds over]"))
    print(f"    first pass, depth scattered through a pinhole: "
          f"{result['first_pass_corner_px_native']:+.1f} px "
          f"(moved {fit['corner_px_native'] - result['first_pass_corner_px_native']:+.1f})")
    print(f"  D_rms at {arm['native'][0] // 4}x{arm['native'][1] // 4}: "
          f"{fit['rms_px_at_quarter_native']:.2f} px "
          f"(0.7 dB needs 4.4 px; under 1.0 px refutes it)")
    r = fit["residual"]
    print(f"  residual {r['rms_px']:.2f} px rms, {r['median_px']:.2f} median, "
          f"{r['p90_px']:.2f} p90, over {r['n']} carried observations")
    print(f"  pose correction {fit['pose_correction']['rotation_deg_median']:.3f} deg "
          f"and {fit['pose_correction']['translation_cm_median']:.2f} cm median; "
          f"chain positions scaled x{fit['trajectory_scale']:.4f}")
    if "bootstrap" in result:
        b = result["bootstrap"]
        print(f"  bootstrap over {b['n']} track resamples: corner "
              f"[{b['corner_px_native_p05']:+.1f}, {b['corner_px_native_p95']:+.1f}] px"
              + ("   ** SPANS ZERO — no coefficient claimed" if b["spans_zero"] else ""))
    if "variants" in result:
        print("\n  what each freedom explains, on the same observations:")
        print("    fit     rms px   corner px")
        for name in ("none", "pose", "model", "both"):
            v = result["variants"][name]
            print(f"    {name:<6} {v['residual']['rms_px']:7.3f}   "
                  f"{v['corner_px_native']:+9.1f}")
        n0 = result["variants"]["none"]["residual"]["rms_px"]
        npose = result["variants"]["pose"]["residual"]["rms_px"]
        nboth = result["variants"]["both"]["residual"]["rms_px"]
        pose_share, model_share = n0 - npose, npose - nboth
        print(f"    pose explains {pose_share:.3f} px of rms; the camera model "
              f"explains a further {model_share:.3f} px")
        if pose_share > 0:
            print(f"    model/pose = {model_share / pose_share:.4f} — read this "
                  f"as a description, not a test.\n"
                  f"    `docs/UW_SELFCAL_PREREG.md` pre-registered 'under 0.10 "
                  f"means not separable'\n"
                  f"    and RETRACTED it: on synthetic data with a known "
                  f"+24.2 px lens and\n"
                  f"    wrong poses the same ratio is 0.0024 while the joint fit "
                  f"returns +24.17.\n"
                  f"    The model bites at large radius, where few observations "
                  f"live, so it barely\n"
                  f"    moves a whole-frame rms even when it is perfectly "
                  f"identified. The gate is\n"
                  f"    injection recovery.")
    for key, title in (("shape_none", "with the chain pose and a pinhole, "
                                      "nothing modelled"),
                       ("shape_pose", "after a pose-only fit — a lower bound, "
                                      "the pose has eaten some"),
                       ("shape_both", "after the joint fit — this one should be "
                                      "flat if the model is right")):
        s = result.get(key)
        if not s:
            continue
        print(f"\n  azimuthally averaged residual, {title}.\n"
              "  A pose error cannot produce a non-zero radial mean except at "
              "rho^1:")
        print("      rho      n      radial px          tangential px")
        for p in s["profile"]:
            print(f"    {p['rho']:5.3f} {p['n']:7d}   "
                  f"{p['radial_px']:+7.3f} +/- {p['radial_sem_px']:.3f}   "
                  f"{p['tangential_px']:+7.3f} +/- {p['tangential_sem_px']:.3f}")
        if "radial_terms_px" in s:
            t = s["radial_terms_px"]
            print(f"    b1 (forward translation / focal) {t['b1']:+.3f} px, "
                  f"b3 {t['b3']:+.3f} px, b5 {t['b5']:+.3f} px")
            print(f"    -> implied corner displacement "
                  f"{s['implied_corner_px_native']:+.1f} px at native")


def strip(result: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in result.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("session")
    ap.add_argument("--poses", required=True,
                    help="npz from eval/pi3_poseless.py on the WIDE arm")
    ap.add_argument("--pose-key", default="estimate")
    ap.add_argument("--lens", choices=("ultrawide", "wide"), default="ultrawide")
    ap.add_argument("--scale", type=float, default=None,
                    help="working resolution as a fraction of native "
                         "(default 0.5 ultra-wide, 1.0 wide)")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--start", type=int, default=None,
                    help="first frame of the block (default: centred)")
    ap.add_argument("--hfov", type=float, default=None)
    ap.add_argument("--order", type=int, default=1, choices=(1, 2),
                    help="radial terms: 1 fits a1 only (default — 2 fails the "
                         "injection gate on the wide arm, see "
                         "docs/UW_SELFCAL_PREREG.md), 2 fits a1 and a2")
    ap.add_argument("--variants", action="store_true",
                    help="also fit pose-only, model-only and neither")
    ap.add_argument("--bootstrap", type=int, default=0)
    ap.add_argument("--inject", type=float, default=None,
                    help="corner displacement in native px to warp the frames by")
    ap.add_argument("--inject-sweep", default=None,
                    help="comma-separated corner displacements, native px")
    ap.add_argument("--write-undistorted", default=None,
                    help="write a session straightened by the fit and run "
                         "eval/uw_radial_profile.py on it")
    ap.add_argument("--min-distance", type=int, default=None,
                    help="minimum spacing between tracked features, working px")
    ap.add_argument("--quality", type=float, default=None,
                    help="goodFeaturesToTrack quality level")
    ap.add_argument("--out", default=None, help="JSON report")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)

    scale = a.scale if a.scale is not None else (0.5 if a.lens == "ultrawide" else 1.0)
    arm = Arm(a.session, a.poses, a.lens, scale, a.frames, a.start,
              pose_key=a.pose_key, hfov=a.hfov)
    d = arm.describe()
    print(f"{d['session']} {d['lens']}: {d['frames']} of {d['frames_available']} "
          f"usable frames from index {d['start']}, "
          f"{d['native'][0]}x{d['native'][1]} worked at "
          f"{d['working'][0]}x{d['working'][1]}")
    print(f"  {d['hfov_deg']:.5f} deg across"
          + ("  [ASSUMED: this session logged none for this lens]"
             if d["hfov_assumed"] else "  [logged by the recorder]")
          + f", f {d['f_working_px']:.1f} px at the working size")
    print(f"  poses {a.poses}:{a.pose_key}, join mode {d['join_scale_mode']}, "
          f"rig baseline {d['rig_baseline_mm']:.2f} mm, shutter offset "
          f"{d['shutter_offset_median_ms']:.1f} ms median")
    print(f"  depth from {d['depth_source']}; image-to-depth gap "
          f"{d['depth_gap_median_ms']:.1f} ms median, "
          f"{d['depth_gap_worst_ms']:.1f} worst")
    print(f"  the wide arm's cone ends at {d['cone_radius_native_px']:.0f} px "
          f"native, against a {d['half_diagonal_native_px']:.0f} px half-diagonal")

    report: dict[str, Any] = {"argv": argv}
    tracking = {k: v for k, v in (("min_distance", a.min_distance),
                                  ("quality", a.quality)) if v is not None}
    if tracking:
        print(f"  tracking overrides {tracking}")
    baseline = run(arm, variants=a.variants, bootstrap=a.bootstrap,
                   order=a.order, seed=a.seed, tracking=tracking)
    report_run(baseline)
    report["baseline"] = strip(baseline)

    if a.inject is not None or a.inject_sweep:
        sizes = ([a.inject] if a.inject is not None
                 else [float(v) for v in a.inject_sweep.split(",")])
        native = baseline["fit"]["coefficients"]
        rows = []
        print("\n  INJECTION RECOVERY — the gate.  Corner px at native; "
              "'expected' is the\n  exact composition of the injection with the "
              "baseline fit, not their sum.")
        print("    injected   expected  recovered     error   bar")
        for size in sizes:
            coefficients = [size / arm.native_R_half] + [0.0] * (a.order - 1)
            got = run(arm, inject=coefficients, order=a.order, quiet=True,
                      tracking=tracking)
            expected = corner_displacement(
                arm.native_w, arm.native_h,
                compose_expected(native, coefficients, a.order))
            recovered = got["fit"]["corner_px_native"]
            bar = max(5.0, 0.15 * abs(size))
            rows.append({"injected_px": size, "expected_px": expected,
                         "recovered_px": recovered,
                         "error_px": recovered - expected, "bar_px": bar,
                         "passes": bool(abs(recovered - expected) <= bar),
                         "coefficients": got["fit"]["coefficients"],
                         "residual": got["fit"]["residual"],
                         "census": got["census"]})
            print(f"    {size:+8.1f}   {expected:+8.1f}  {recovered:+9.1f}   "
                  f"{recovered - expected:+7.1f}   {bar:4.1f}"
                  + ("" if rows[-1]["passes"] else "   FAILS"))
        xs = np.array([r["injected_px"] for r in rows])
        ys = np.array([r["recovered_px"] - baseline["fit"]["corner_px_native"]
                       for r in rows])
        slope = float(np.polyfit(xs, ys, 1)[0]) if len(xs) >= 2 else float("nan")
        ok = bool(0.85 <= slope <= 1.15)
        print(f"    response slope {slope:.3f} (pre-registered band 0.85-1.15)"
              + ("" if ok else "   FAILS"))
        report["injection"] = {
            "rows": rows, "slope": slope, "slope_passes": ok,
            "all_pass": bool(all(r["passes"] for r in rows) and ok),
            "baseline_corner_px": baseline["fit"]["corner_px_native"]}

    if a.write_undistorted:
        print("\n  PROFILE TEST — criterion 4")
        before = run_profile(str(arm.dir))
        out_dir = write_undistorted(arm, baseline["fit"]["coefficients"],
                                    a.write_undistorted)
        after = run_profile(out_dir)
        sb, sa = profile_spread(before), profile_spread(after)
        for label, s in (("before", sb), ("after ", sa)):
            for block in ("control", "lenses"):
                if block in s:
                    print(f"    {label} {block:<8} "
                          + " ".join(f"{v:.4f}" for v in s[block]["values"])
                          + f"   peak-to-trough {s[block]['spread_pct']:.2f} %")
        report["profile"] = {"before_text": before, "after_text": after,
                             "before": sb, "after": sa}

    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1, default=float))
        print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
