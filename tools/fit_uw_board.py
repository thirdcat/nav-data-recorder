#!/usr/bin/env python3
"""Fit a lens's radial distortion from checkerboard rows and columns.

``tools/fit_uw_distortion.py`` was refused on the ultra-wide because indoor
rooms do not offer long enough straight edges: at the median chain length those
captures yield (0.16 of the frame diagonal) a 110 px corner displacement bends a
line by 0.12 px against a 1.44 px straightness floor, and three sessions
disagreed by four times their mean.  ``docs/UW_DISTORTION_PREREG.md`` closed by
naming the capture that would settle it -- straight edges spanning 40 % of the
frame diagonal, held still enough to put the floor under a pixel.  A
checkerboard does that.

**The board runs off the frame edge**, so the full interior grid is almost never
seen and detections come back as sub-grids of every shape from 3x3 to 8x6.  A
sub-grid does not say which block of the board it is, so the object points are
unknown up to an integer offset.  That breaks nothing here: every detected row
and every detected column is a set of collinear world points whatever sub-block
it came from, so each becomes one plumb-line chain and the fit is
``tools/fit_uw_distortion.py``'s, imported unchanged.

    # the measurement
    python3 tools/fit_uw_board.py fit --lens ultrawide

    # null control: the wide arm should come back near zero
    python3 tools/fit_uw_board.py fit --lens wide

    # known-answer control: warp every frame and demand the coefficient back
    python3 tools/fit_uw_board.py inject --lens ultrawide

    # one lens cannot have six distortions
    python3 tools/fit_uw_board.py takes --lens ultrawide

    # secondary, correspondence-based: focal length, with its own control
    python3 tools/fit_uw_board.py focal --lens ultrawide

    # synthetic controls only, no capture needed
    python3 tools/fit_uw_board.py selftest

The pre-registered acceptance criteria and thresholds are in
``docs/UW_BOARD_PREREG.md``.  Requires numpy, scipy and opencv.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - the CLI reports this clearly.
    cv2 = None

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fit_uw_distortion import (  # noqa: E402
    ChainSet,
    _cost,
    _huber,
    bootstrap,
    compose_expected,
    corner_displacement,
    fit_coefficients,
    half_diagonal,
    identifiability,
    inject_distortion,
    opencv_k1,
    radial_gain,
    rms_displacement,
    sufficiency,
)


# The six takes of the 60 mm 8x6-square board through the multi-cam path.
BOARD_TAKES = (
    "20260823-003245-1e87b0",
    "20260823-003258-098cbe",
    "20260823-003306-161205",
    "20260823-003312-e52acc",
    "20260823-003318-0b4627",
    "20260823-003324-899c75",
)

NAV_DATA = Path("~/nav_data").expanduser()

# Detection settings, frozen in docs/UW_BOARD_PREREG.md before any fit ran.
SB_FLAGS_NAME = "EXHAUSTIVE|ACCURACY|LARGER"
WORK_SCALES = {"ultrawide": (1920, 1280, 960), "wide": (640, 480)}
MIN_SPACING_FRACTION = 0.01     # of the frame diagonal; kills the dock icon
GRID_GATE = 0.12                # max homography residual / median spacing
MIN_GRID = 3                    # peeling stops here
SUBPIX_WINDOW_DIVISOR = 12
SUBPIX_WINDOW_RANGE = (5, 21)
SUBPIX_REVERT_FRACTION = 0.5

# The pre-registered injection sweeps, as corner displacements in native px.
UW_INJECTIONS = (-80.0, -40.0, -20.0, 20.0, 40.0, 80.0)
WIDE_INJECTIONS = (-30.0, -15.0, 15.0, 30.0)


# ---------------------------------------------------------------------------
# Board detection
# ---------------------------------------------------------------------------

def _sb_flags() -> int:
    return (cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
            | cv2.CALIB_CB_LARGER)


def grid_spacing(grid: np.ndarray) -> float:
    """Median distance between 4-neighbour corners of a detected sub-grid."""
    steps = []
    if grid.shape[1] > 1:
        steps.append(np.linalg.norm(np.diff(grid, axis=1), axis=-1).ravel())
    if grid.shape[0] > 1:
        steps.append(np.linalg.norm(np.diff(grid, axis=0), axis=-1).ravel())
    if not steps:
        return 0.0
    return float(np.median(np.concatenate(steps)))


def _lattice(rows: int, cols: int) -> np.ndarray:
    i, j = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    return np.stack([j.ravel(), i.ravel()], axis=-1).astype(np.float64)


def homography_residual(grid: np.ndarray) -> float:
    """max |corner - H(lattice)| / spacing, the grid-consistency measure.

    A planar board seen through *any* projective camera maps its integer
    lattice to the image by a homography exactly; radial distortion perturbs
    that, but only slightly -- ``docs/UW_BOARD_PREREG.md`` records the synthetic
    sweep showing a true ``a1 = 0.08`` (a 176 px corner) reaches 0.0593 at the
    95th percentile against the 0.12 gate.  A detector that ran a row onto the
    monitor bezel, or laid a grid across the ceiling, lands far above it.
    """
    rows, cols = grid.shape[:2]
    lattice = _lattice(rows, cols)
    points = grid.reshape(-1, 2)
    matrix, _ = cv2.findHomography(lattice, points, 0)
    if matrix is None:
        return float("inf")
    projected = cv2.perspectiveTransform(lattice.reshape(-1, 1, 2),
                                         matrix).reshape(-1, 2)
    spacing = grid_spacing(grid)
    if spacing <= 0.0:
        return float("inf")
    return float(np.max(np.linalg.norm(projected - points, axis=1)) / spacing)


def peel_grid(grid: np.ndarray, gate: float = GRID_GATE
              ) -> tuple[np.ndarray | None, float, int]:
    """Drop boundary rows/columns until the grid gate passes, or give up."""
    peeled = 0
    while True:
        residual = homography_residual(grid)
        if residual <= gate:
            return grid, residual, peeled
        rows, cols = grid.shape[:2]
        candidates = []
        if rows > MIN_GRID:
            candidates += [grid[1:, :], grid[:-1, :]]
        if cols > MIN_GRID:
            candidates += [grid[:, 1:], grid[:, :-1]]
        if not candidates:
            return None, residual, peeled
        scored = sorted(((homography_residual(c), n)
                         for n, c in enumerate(candidates)), key=lambda t: t[0])
        grid = candidates[scored[0][1]]
        peeled += 1


def detect_board(gray: np.ndarray, scales: Sequence[int]
                 ) -> dict[str, Any] | None:
    """One gated, subpixel-refined sub-grid in native pixel coordinates."""
    height, width = gray.shape
    diagonal = math.hypot(width, height)
    min_spacing = MIN_SPACING_FRACTION * diagonal
    best = None
    for long_side in scales:
        if long_side >= width:
            work, scale = gray, 1.0
        else:
            scale = long_side / width
            work = cv2.resize(gray, (long_side, int(round(height * scale))),
                              interpolation=cv2.INTER_AREA)
        found, corners, meta = cv2.findChessboardCornersSBWithMeta(
            work, (MIN_GRID, MIN_GRID), _sb_flags())
        if not found:
            continue
        grid = corners.reshape(meta.shape[0], meta.shape[1], 2)
        grid = (grid.astype(np.float64) + 0.5) / scale - 0.5
        if grid_spacing(grid) < min_spacing:
            continue                       # the desktop dock's thumbnail icon
        grid, residual, peeled = peel_grid(grid)
        if grid is None:
            continue
        if best is None or grid.size > best[0].size:
            best = (grid, residual, peeled, long_side)
    if best is None:
        return None

    grid, residual, peeled, long_side = best
    spacing = grid_spacing(grid)
    window = int(np.clip(round(spacing / SUBPIX_WINDOW_DIVISOR),
                         *SUBPIX_WINDOW_RANGE))
    seed = grid.reshape(-1, 2).copy()
    seed[:, 0] = np.clip(seed[:, 0], window + 2, width - 3 - window)
    seed[:, 1] = np.clip(seed[:, 1], window + 2, height - 3 - window)
    points = seed.reshape(-1, 1, 2).astype(np.float32)
    cv2.cornerSubPix(gray, points, (window, window), (-1, -1),
                     (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
                      40, 0.01))
    refined = points.reshape(-1, 2).astype(np.float64)
    runaway = (np.linalg.norm(refined - seed, axis=1)
               > SUBPIX_REVERT_FRACTION * window)
    refined[runaway] = seed[runaway]
    return {
        "shape": (int(grid.shape[0]), int(grid.shape[1])),
        "seed": grid.reshape(-1, 2),
        "corners": refined,
        "size": (int(width), int(height)),
        "grid_residual": float(residual),
        "peeled": int(peeled),
        "reverted": int(runaway.sum()),
        "spacing": float(spacing),
        "scale": int(long_side),
    }


def grid_chains(detection: dict[str, Any], *, refined: bool = True,
                mask: np.ndarray | None = None) -> list[np.ndarray]:
    """Rows and columns of the sub-grid, each a chain of collinear points."""
    rows, cols = detection["shape"]
    key = "corners" if refined else "seed"
    grid = detection[key].reshape(rows, cols, 2)
    valid = None
    if mask is not None:
        u = np.clip(np.round(grid[..., 0]).astype(np.int64), 0,
                    mask.shape[1] - 1)
        v = np.clip(np.round(grid[..., 1]).astype(np.int64), 0,
                    mask.shape[0] - 1)
        valid = mask[v, u]
    chains = []
    if cols >= MIN_GRID:
        for i in range(rows):
            if valid is not None and not valid[i, :].all():
                continue
            chains.append(grid[i, :, :].copy())
    if rows >= MIN_GRID:
        for j in range(cols):
            if valid is not None and not valid[:, j].all():
                continue
            chains.append(grid[:, j, :].copy())
    return chains


# ---------------------------------------------------------------------------
# Session plumbing
# ---------------------------------------------------------------------------

def take_paths(take: str, lens: str) -> list[Path]:
    folder = "frames" if lens == "ultrawide" else "frames_wide"
    listing = NAV_DATA / take / (
        "frames.jsonl" if lens == "ultrawide" else "frames_wide.jsonl")
    root = NAV_DATA / take
    files = []
    with listing.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                files.append(root / json.loads(line)["file"])
    if not files:  # pragma: no cover - defensive
        files = sorted((root / folder).glob("*.jpg"))
    return files


def _detect_one(job: tuple[str, str, tuple[int, ...], tuple[float, ...] | None]
                ) -> tuple[str, dict[str, Any] | None]:
    path, _take, scales, inject = job
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return path, None
    mask = None
    if inject is not None:
        gray, mask = inject_distortion(gray, list(inject))
    detection = detect_board(gray, scales)
    if detection is None:
        return path, None
    if mask is not None:
        detection["mask"] = np.packbits(mask)
        detection["mask_shape"] = mask.shape
    detection["take"] = _take
    detection["path"] = path
    detection["sharpness"] = float(cv2.Laplacian(
        cv2.resize(gray, None, fx=0.25, fy=0.25,
                   interpolation=cv2.INTER_AREA), cv2.CV_64F).var())
    return path, detection


def detect_take_set(takes: Sequence[str], lens: str, *,
                    inject: Sequence[float] | None = None,
                    workers: int = 0) -> list[dict[str, Any]]:
    scales = WORK_SCALES[lens]
    jobs = []
    for take in takes:
        for path in take_paths(take, lens):
            jobs.append((str(path), take, scales,
                         tuple(inject) if inject is not None else None))
    workers = workers or min(20, os.cpu_count() or 1)
    results: list[dict[str, Any]] = []
    frames_read = 0
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            for _path, detection in pool.map(_detect_one, jobs, chunksize=2):
                frames_read += 1
                if detection is not None:
                    results.append(detection)
    else:  # pragma: no cover - serial path is for debugging
        for job in jobs:
            _path, detection = _detect_one(job)
            frames_read += 1
            if detection is not None:
                results.append(detection)
    for detection in results:
        detection["frames_read"] = frames_read
    if results:
        results[0]["_frames_read"] = frames_read
    return results


def build_chainset(detections: Sequence[dict[str, Any]], *,
                   refined: bool = True, min_chord_fraction: float = 0.0
                   ) -> tuple[ChainSet | None, dict[str, Any]]:
    chains: list[np.ndarray] = []
    frames: list[int] = []
    width = height = 0
    contributing = 0
    peeled = reverted = 0
    for index, detection in enumerate(detections):
        width, height = detection["size"]
        mask = None
        if "mask" in detection:
            mask = np.unpackbits(detection["mask"]).astype(bool)
            mask = mask[: int(np.prod(detection["mask_shape"]))]
            mask = mask.reshape(detection["mask_shape"])
        found = grid_chains(detection, refined=refined, mask=mask)
        if min_chord_fraction > 0.0:
            diagonal = math.hypot(width, height)
            found = [c for c in found
                     if np.hypot(*(c[-1] - c[0])) >= min_chord_fraction * diagonal]
        if found:
            contributing += 1
        chains.extend(found)
        frames.extend([index] * len(found))
        peeled += detection.get("peeled", 0)
        reverted += detection.get("reverted", 0)
    census = {
        "frames_read": detections[0]["frames_read"] if detections else 0,
        "detections": len(detections),
        "frames_contributing": contributing,
        "chains": len(chains),
        "width": width, "height": height,
        "peeled_rows": peeled, "subpix_reverted": reverted,
    }
    if not chains:
        return None, census
    return ChainSet(chains, frames, width, height), census


# ---------------------------------------------------------------------------
# One measurement
# ---------------------------------------------------------------------------

def cost_drop(chains: ChainSet, coefficients: Sequence[float]) -> float:
    order = len(coefficients)
    delta = max(float(np.median(chains.residual_px([0.0] * order))), 1e-6)
    zero = _cost([0.0] * order, chains, delta)
    best = _cost(list(coefficients), chains, delta)
    return float(1.0 - best / zero) if zero > 0 else 0.0


def measure(detections: Sequence[dict[str, Any]], *, order: int = 1,
            draws: int = 200, refined: bool = True,
            min_chord_fraction: float = 0.0,
            label: str = "") -> dict[str, Any]:
    chains, census = build_chainset(detections, refined=refined,
                                    min_chord_fraction=min_chord_fraction)
    report: dict[str, Any] = {"label": label, "census": census}
    if chains is None:
        report["sufficiency"] = {"passed": False, "failed": ["chains"]}
        return report
    report["sufficiency"] = sufficiency(chains, census)
    coefficients = fit_coefficients(chains, order)
    width, height = chains.width, chains.height
    diagonal = math.hypot(width, height)
    to_960 = 960.0 / width

    before = chains.residual_px([0.0] * order)
    after = chains.residual_px(coefficients)
    report.update({
        "order": order,
        "coefficients": coefficients,
        "corner_px": corner_displacement(width, height, coefficients),
        "rms_px_native": rms_displacement(width, height, coefficients),
        "rms_px_960x540": rms_displacement(width, height, coefficients) * 960.0 / 3840.0
        if width == 3840 else None,
        "k1_opencv_1466": opencv_k1(width, height, 1466.0, coefficients)
        if width == 3840 else None,
        "straightness": {
            "median_residual_px_before": float(np.median(before)),
            "median_residual_px_after": float(np.median(after)),
            "mean_residual_px_before": float(np.mean(before)),
            "mean_residual_px_after": float(np.mean(after)),
            "p90_residual_px_before": float(np.percentile(before, 90)),
            "p90_residual_px_after": float(np.percentile(after, 90)),
            "median_residual_960_before": float(np.median(before)) * to_960,
            "median_residual_960_after": float(np.median(after)) * to_960,
        },
        "cost_drop": cost_drop(chains, coefficients),
        "identifiability": identifiability(chains, coefficients),
        "median_chord_fraction": float(np.median(chains.chords_px([0.0] * order))
                                       / diagonal),
        "median_rho": float(np.median(chains.rho_mean)),
        "outer_chains_rho60": int(np.sum(chains.rho_mean >= 0.6)),
    })
    if draws:
        report["bootstrap"] = bootstrap(chains, coefficients, draws=draws)
    return report


# ---------------------------------------------------------------------------
# Secondary: focal length by Zhang, legal on sub-grids because the in-plane
# offset is a gauge absorbed by the extrinsic.  See docs/UW_BOARD_PREREG.md.
# ---------------------------------------------------------------------------

def handedness(detections: Sequence[dict[str, Any]]) -> list[int]:
    signs = []
    for detection in detections:
        rows, cols = detection["shape"]
        grid = detection["corners"].reshape(rows, cols, 2)
        along_col = grid[0, 1] - grid[0, 0]
        along_row = grid[1, 0] - grid[0, 0]
        cross = along_col[0] * along_row[1] - along_col[1] * along_row[0]
        signs.append(int(np.sign(cross)))
    return signs


def calibrate(detections: Sequence[dict[str, Any]], *,
              square: float = 0.06) -> dict[str, Any]:
    object_points, image_points = [], []
    width = height = 0
    for detection in detections:
        rows, cols = detection["shape"]
        width, height = detection["size"]
        i, j = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
        obj = np.stack([j.ravel() * square, i.ravel() * square,
                        np.zeros(rows * cols)], axis=-1).astype(np.float32)
        object_points.append(obj)
        image_points.append(detection["corners"]
                            .reshape(-1, 1, 2).astype(np.float32))
    flags = (cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST)
    rms, camera, dist, _, _ = cv2.calibrateCamera(
        object_points, image_points, (width, height), None, None, flags=flags)
    k1 = float(dist.ravel()[0])
    focal = 0.5 * (float(camera[0, 0]) + float(camera[1, 1]))
    return {
        "views": len(object_points),
        "reprojection_rms_px": float(rms),
        "fx": float(camera[0, 0]), "fy": float(camera[1, 1]),
        "cx": float(camera[0, 2]), "cy": float(camera[1, 2]),
        "k1": k1,
        "focal_mean": focal,
        "corner_px": opencv_corner_displacement(width, height, focal, k1),
        "size": (width, height),
    }


def opencv_corner_displacement(width: int, height: int, focal: float,
                               k1: float) -> float:
    """This page's D_corner from an OPENCV k1: solve R = r_u (1 + k1 (r_u/f)^2)."""
    radius = half_diagonal(width, height)
    r_u = radius
    for _ in range(200):
        f_value = r_u * (1.0 + k1 * (r_u / focal) ** 2) - radius
        derivative = 1.0 + 3.0 * k1 * (r_u / focal) ** 2
        if abs(derivative) < 1e-12:
            break
        step = f_value / derivative
        r_u -= step
        if abs(step) < 1e-9:
            break
    return float(r_u - radius)


# ---------------------------------------------------------------------------
# Synthetic controls
# ---------------------------------------------------------------------------

def synthetic_grid_gate(seed: int = 0, trials: int = 300) -> dict[str, Any]:
    """The number that justifies the 0.12 grid gate, recomputed on demand."""
    width, height = 3840, 2160
    radius = half_diagonal(width, height)
    centre = np.array([width / 2, height / 2])
    rng = np.random.default_rng(seed)
    out = {}
    for a1 in (0.0, 0.01, 0.02, 0.03, 0.05, 0.08):
        values = []
        for _ in range(trials):
            rows, cols = int(rng.integers(3, 9)), int(rng.integers(3, 9))
            rotation = cv2.Rodrigues(rng.normal(0, 0.5, 3))[0]
            depth = rng.uniform(0.6, 2.0)
            square = rng.uniform(0.05, 0.30)
            i, j = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
            obj = np.stack([j * square, i * square, np.zeros_like(i, float)],
                           -1).reshape(-1, 3)
            obj = obj - obj.mean(axis=0)
            points = (rotation @ obj.T).T + np.array(
                [rng.normal(0, 0.2), rng.normal(0, 0.2), depth])
            if (points[:, 2] <= 0.2).any():
                continue
            image = centre + 1466.0 * points[:, :2] / points[:, 2:3]
            rho = np.linalg.norm(image - centre, axis=1) / radius
            if rho.max() > 1.0:
                continue
            span = np.linalg.norm(image.max(0) - image.min(0)) / math.hypot(
                width, height)
            if span < 0.15:
                continue
            delta = image - centre
            rho2 = (delta ** 2).sum(-1) / radius ** 2
            distorted = centre + delta * radial_gain(rho2, [a1])[:, None]
            values.append(homography_residual(distorted.reshape(rows, cols, 2)))
        values = np.asarray(values)
        out[f"a1={a1:+.2f}"] = {
            "corner_px": a1 * radius, "trials": len(values),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "max": float(values.max()),
        }
    return out


def synthetic_subgrid_calibration(seed: int = 1, views: int = 40,
                                  focal: float = 1466.0,
                                  k1: float = -0.02) -> dict[str, Any]:
    """Does Zhang survive unknown in-plane offsets and 90-degree relabels?"""
    width, height = 3840, 2160
    rng = np.random.default_rng(seed)
    camera = np.array([[focal, 0, width / 2], [0, focal, height / 2],
                       [0, 0, 1]], dtype=np.float64)
    dist = np.array([k1, 0.0, 0.0, 0.0, 0.0])
    square = 0.06
    object_points, image_points = [], []
    full_rows, full_cols = 5, 7
    while len(object_points) < views:
        rows = int(rng.integers(3, full_rows + 1))
        cols = int(rng.integers(3, full_cols + 1))
        i0 = int(rng.integers(0, full_rows - rows + 1))
        j0 = int(rng.integers(0, full_cols - cols + 1))
        i, j = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
        board = np.stack([(j + j0) * square, (i + i0) * square,
                          np.zeros_like(i, float)], -1)
        if rng.random() < 0.5:                    # a 90-degree relabel
            board = np.transpose(board, (1, 0, 2))[::-1, :, :]
        flat = board.reshape(-1, 3)
        rvec = rng.normal(0, 0.35, 3)
        tvec = np.array([rng.normal(0, 0.15), rng.normal(0, 0.1),
                         rng.uniform(0.5, 1.4)])
        projected, _ = cv2.projectPoints(flat, rvec, tvec, camera, dist)
        projected = projected.reshape(-1, 2)
        if (projected[:, 0] < 0).any() or (projected[:, 0] > width).any():
            continue
        if (projected[:, 1] < 0).any() or (projected[:, 1] > height).any():
            continue
        rows_used, cols_used = board.shape[:2]
        i2, j2 = np.meshgrid(np.arange(rows_used), np.arange(cols_used),
                             indexing="ij")
        # the fitter is told an arbitrary origin, exactly as the real data forces
        obj = np.stack([j2.ravel() * square, i2.ravel() * square,
                        np.zeros(rows_used * cols_used)], -1).astype(np.float32)
        object_points.append(obj)
        image_points.append((projected
                             + rng.normal(0, 0.2, projected.shape)
                             ).reshape(-1, 1, 2).astype(np.float32))
    flags = (cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST)
    rms, fitted, fitted_dist, _, _ = cv2.calibrateCamera(
        object_points, image_points, (width, height), None, None, flags=flags)
    focal_fit = 0.5 * (fitted[0, 0] + fitted[1, 1])
    truth = opencv_corner_displacement(width, height, focal, k1)
    got = opencv_corner_displacement(width, height, focal_fit,
                                     float(fitted_dist.ravel()[0]))
    return {
        "views": len(object_points), "reprojection_rms_px": float(rms),
        "focal_true": focal, "focal_fit": float(focal_fit),
        "focal_error_pct": 100.0 * (float(focal_fit) - focal) / focal,
        "k1_true": k1, "k1_fit": float(fitted_dist.ravel()[0]),
        "corner_true_px": truth, "corner_fit_px": got,
        "corner_error_px": got - truth,
        "passes": abs(100.0 * (float(focal_fit) - focal) / focal) <= 1.0
        and abs(got - truth) <= 5.0,
    }


def synthetic_chain_recovery(seed: int = 2, a1: float = -0.02,
                             noise: float = 0.25) -> dict[str, Any]:
    """Board rows/columns of the real shape, warped by a known a1, refitted."""
    width, height = 3840, 2160
    radius = half_diagonal(width, height)
    centre = np.array([width / 2, height / 2])
    rng = np.random.default_rng(seed)
    chains, frames = [], []
    for frame in range(120):
        rows, cols = int(rng.integers(3, 9)), int(rng.integers(3, 9))
        rotation = cv2.Rodrigues(rng.normal(0, 0.5, 3))[0]
        square = rng.uniform(0.06, 0.30)
        i, j = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
        obj = np.stack([j * square, i * square, np.zeros_like(i, float)],
                       -1).reshape(-1, 3)
        obj = obj - obj.mean(axis=0)
        points = (rotation @ obj.T).T + np.array(
            [rng.normal(0, 0.25), rng.normal(0, 0.2), rng.uniform(0.6, 1.6)])
        if (points[:, 2] <= 0.2).any():
            continue
        image = centre + 1466.0 * points[:, :2] / points[:, 2:3]
        delta = image - centre
        rho2 = (delta ** 2).sum(-1) / radius ** 2
        if np.sqrt(rho2).max() > 1.0:
            continue
        # invert the rectification so that fitting a1 straightens it
        distorted = centre + delta / radial_gain(rho2, [a1])[:, None]
        distorted = distorted + rng.normal(0, noise, distorted.shape)
        grid = distorted.reshape(rows, cols, 2)
        detection = {"shape": (rows, cols), "corners": grid.reshape(-1, 2),
                     "seed": grid.reshape(-1, 2), "size": (width, height)}
        found = grid_chains(detection)
        chains.extend(found)
        frames.extend([frame] * len(found))
    chainset = ChainSet(chains, frames, width, height)
    fitted = fit_coefficients(chainset, 1)
    return {
        "chains": len(chains), "a1_true": a1, "a1_fit": fitted[0],
        "corner_true_px": a1 * radius,
        "corner_fit_px": corner_displacement(width, height, fitted),
        "corner_error_px": corner_displacement(width, height, fitted) - a1 * radius,
    }


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def _fmt(report: dict[str, Any]) -> str:
    if "corner_px" not in report:
        return f"{report.get('label','')}: no chains"
    band = report["identifiability"]
    boot = report.get("bootstrap", {})
    straight = report["straightness"]
    return (f"{report['label']:<28} n={report['census']['chains']:5d} "
            f"det={report['census']['detections']:4d} "
            f"a1={report['coefficients'][0]:+.5f} "
            f"corner={report['corner_px']:+8.2f} px  "
            f"band=[{band['corner_px_band_low']:+8.1f},"
            f"{band['corner_px_band_high']:+8.1f}]  "
            f"boot=[{boot.get('corner_px_p05', float('nan')):+7.1f},"
            f"{boot.get('corner_px_p95', float('nan')):+7.1f}]  "
            f"cost drop={100*report['cost_drop']:5.2f}%  "
            f"resid {straight['median_residual_px_before']:.3f}"
            f" -> {straight['median_residual_px_after']:.3f}")


def mode_fit(args) -> dict[str, Any]:
    takes = args.takes or list(BOARD_TAKES)
    detections = detect_take_set(takes, args.lens, workers=args.workers)
    out = {"lens": args.lens, "takes": takes}
    report = measure(detections, order=args.order, draws=args.draws,
                     refined=not args.seeds, label=f"pooled {args.lens}")
    out["pooled"] = report
    print(_fmt(report))
    if args.sensitivity:
        out["sensitivity"] = {}
        for name, kwargs in (
                ("subpix corners", {"refined": True}),
                ("seed corners", {"refined": False}),
                ("chord>=0.20", {"min_chord_fraction": 0.20}),
                ("chord>=0.30", {"min_chord_fraction": 0.30}),
                ("order 2", {"order": 2}),
                ("order 2, draws", {"order": 2, "draws": args.draws})):
            sub = measure(detections, label=name,
                          **{**{"order": args.order, "draws": 0,
                                "refined": not args.seeds}, **kwargs})
            out["sensitivity"][name] = sub
            print(_fmt(sub))
    return out


def mode_takes(args) -> dict[str, Any]:
    out = {"lens": args.lens, "per_take": {}}
    values = []
    for take in (args.takes or list(BOARD_TAKES)):
        detections = detect_take_set([take], args.lens, workers=args.workers)
        report = measure(detections, order=args.order, draws=args.draws,
                         refined=not args.seeds, label=take[-6:])
        out["per_take"][take] = report
        print(_fmt(report))
        if "corner_px" in report:
            values.append((take, report["corner_px"],
                           report["sufficiency"]["passed"]))
    if values:
        corners = np.array([v[1] for v in values])
        gated = np.array([v[1] for v in values if v[2]])
        out["spread"] = {
            "n": len(corners), "mean": float(corners.mean()),
            "spread": float(corners.max() - corners.min()),
            "spread_over_abs_mean": float((corners.max() - corners.min())
                                          / abs(corners.mean()))
            if corners.mean() != 0 else float("inf"),
            "same_sign": bool(np.all(np.sign(corners) == np.sign(corners[0]))),
            "gated_n": int(len(gated)),
            "gated_mean": float(gated.mean()) if len(gated) else None,
            "gated_spread": float(gated.max() - gated.min()) if len(gated) else None,
        }
        print(json.dumps(out["spread"], indent=1))
    return out


def mode_inject(args) -> dict[str, Any]:
    takes = args.takes or list(BOARD_TAKES)
    radius = half_diagonal(3840, 2160) if args.lens == "ultrawide" \
        else half_diagonal(640, 480)
    injections = list(args.injections) if args.injections else list(
        UW_INJECTIONS if args.lens == "ultrawide" else WIDE_INJECTIONS)

    # Each arm/injection is detected once; every variant below is measured from
    # the same detections, so the gate grades the *estimator*, not the detector.
    variants = [(f"order{order} {'subpix' if refined else 'seeds'}",
                 order, refined)
                for order in ({1, 2} if args.both_orders else {args.order})
                for refined in ((True, False) if args.both_variants else
                                (not args.seeds,))]

    base_detections = detect_take_set(takes, args.lens, workers=args.workers)
    baselines = {name: measure(base_detections, order=order, draws=0,
                               refined=refined, label=f"baseline {name}")
                 for name, order, refined in variants}
    for report in baselines.values():
        print(_fmt(report))
    out: dict[str, Any] = {"lens": args.lens,
                           "baseline": baselines, "variants": {}}
    rows: dict[str, list] = {name: [] for name, _, _ in variants}
    for displacement in injections:
        a1 = displacement / radius
        detections = detect_take_set(takes, args.lens, inject=[a1],
                                     workers=args.workers)
        for name, order, refined in variants:
            report = measure(detections, order=order, draws=0, refined=refined,
                             label=f"{name} inject {displacement:+.0f} px")
            print(_fmt(report))
            baseline = baselines[name]
            expected = compose_expected(baseline["coefficients"], [a1], order)
            width, height = report["census"]["width"], report["census"]["height"]
            expect_px = corner_displacement(width, height, expected)
            got_px = report.get("corner_px", float("nan"))
            bar = max(5.0, 0.15 * abs(displacement))
            row = {
                "injected_px": displacement, "a1": a1,
                "expected_px": expect_px, "recovered_px": got_px,
                "difference_px": got_px - baseline["corner_px"],
                "error_px": got_px - expect_px, "bar_px": bar,
                "passes": bool(abs(got_px - expect_px) <= bar),
                "sign_ok": bool(np.sign(got_px - baseline["corner_px"])
                                == np.sign(displacement)),
                "chains": report["census"]["chains"],
                "cost_drop": report.get("cost_drop"),
            }
            rows[name].append(row)
            print(f"    {name:16s} injected {displacement:+7.1f}"
                  f"  expected {expect_px:+8.2f}  recovered {got_px:+8.2f}"
                  f"  error {got_px-expect_px:+7.2f}  bar {bar:5.2f}"
                  f"  {'pass' if row['passes'] else 'FAIL'}")
    for name, _, _ in variants:
        entry: dict[str, Any] = {"rows": rows[name]}
        if len(rows[name]) >= 2:
            x = np.array([r["injected_px"] for r in rows[name]])
            y = np.array([r["difference_px"] for r in rows[name]])
            slope, intercept = np.polyfit(x, y, 1)
            entry["slope"] = {
                "slope": float(slope), "intercept_px": float(intercept),
                "passes": bool(0.85 <= float(slope) <= 1.15),
                "all_pass": bool(all(r["passes"] for r in rows[name])),
                "all_sign_ok": bool(all(r["sign_ok"] for r in rows[name])),
            }
            print(name, json.dumps(entry["slope"]))
        out["variants"][name] = entry
    return out


def mode_focal(args) -> dict[str, Any]:
    control = synthetic_subgrid_calibration()
    print("synthetic sub-grid control:", json.dumps(control, indent=1))
    takes = args.takes or list(BOARD_TAKES)
    detections = detect_take_set(takes, args.lens, workers=args.workers)
    signs = handedness(detections)
    constant = len(set(signs)) == 1
    print(f"handedness constant: {constant}  ({sorted(set(signs))}, "
          f"n={len(signs)})")
    out = {"control": control, "handedness_constant": constant,
           "handedness_counts": {str(s): signs.count(s) for s in set(signs)}}
    if not (control["passes"] and constant):
        out["quoted"] = False
        print("control failed -- no focal length quoted")
        return out
    calibration = calibrate(detections)
    out["calibration"] = calibration
    # Added after the ultra-wide came back at a 41.5 px reprojection RMS: a
    # planar board through a pinhole-plus-k1 camera reprojects to well under a
    # pixel, so a fit that cannot is not describing this camera and its focal
    # length is not a measurement.  Declared in docs/UW_BOARD_PREREG.md; it can
    # only ever refuse a number, never manufacture one.
    out["quoted"] = bool(calibration["reprojection_rms_px"] <= 2.0)
    if not out["quoted"]:
        print(f"reprojection RMS {calibration['reprojection_rms_px']:.2f} px "
              f"exceeds 2 px -- the model does not describe these frames, "
              f"no focal length quoted")
    print(json.dumps(calibration, indent=1))
    return out


def mode_selftest(args) -> dict[str, Any]:
    out = {
        "grid_gate": synthetic_grid_gate(trials=args.trials),
        "subgrid_calibration": synthetic_subgrid_calibration(),
        "chain_recovery": [synthetic_chain_recovery(a1=a)
                           for a in (-0.05, -0.02, 0.0, 0.02, 0.05)],
    }
    print(json.dumps(out, indent=1))
    return out


def main(argv: list[str]) -> int:
    if cv2 is None:
        print("this tool needs opencv:  pip install opencv-python-headless",
              file=sys.stderr)
        return 2
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("fit", "takes", "inject", "focal",
                                         "selftest"))
    parser.add_argument("--lens", choices=("ultrawide", "wide"),
                        default="ultrawide")
    parser.add_argument("--takes", nargs="*", default=None)
    parser.add_argument("--order", type=int, default=1, choices=(1, 2))
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--injections", type=float, nargs="*", default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--sensitivity", action="store_true")
    parser.add_argument("--seeds", action="store_true",
                        help="use the detector's own subpixel corners and skip "
                             "the native-resolution cornerSubPix refinement")
    parser.add_argument("--both-variants", action="store_true",
                        help="grade subpix and seeds side by side")
    parser.add_argument("--both-orders", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    handler = {"fit": mode_fit, "takes": mode_takes, "inject": mode_inject,
               "focal": mode_focal, "selftest": mode_selftest}[args.mode]
    report = handler(args)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1, default=float)
                                  + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
