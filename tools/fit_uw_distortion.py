#!/usr/bin/env python3
"""Measure a lens's radial distortion from recorded frames, by plumb-line fit.

The factory calibration under ``calib/`` is measured at 4032x3024 and no
capture path in this repo runs at that format; different formats read out
different parts of this sensor, so the factory table does not describe the
recorded pixels (``calib/README.md``, ``docs/POSE.md``).  This tool therefore
reads nothing but the recorded frames.

A straight edge in the world must image as a straight line under a correct
camera model.  The residual curvature is the distortion.  Indoor scenes supply
the edges: wall-ceiling junctions, door and wardrobe frames, shelf edges, floor
plank seams.

    # null control: the wide arm should come back near zero
    python3 tools/fit_uw_distortion.py ~/nav_data/20260820-211848-d06152 --lens wide

    # known-answer control: warp the wide arm and demand the coefficient back
    python3 tools/fit_uw_distortion.py ~/nav_data/20260820-211848-d06152 \
        --lens wide --inject -0.05

    # the measurement
    python3 tools/fit_uw_distortion.py ~/nav_data/20260820-211848-d06152 \
        --lens ultrawide

The model is a radial rectification about the frame centre, in a radius
normalised so that ``rho = 1`` at the frame corner::

    p_rect = c + (p - c) * (1 + a1 rho^2 + a2 rho^4)

so ``a1 + a2`` is directly the corner displacement as a fraction of the frame
half-diagonal.  Focal length is not identifiable from straightness and is not
fitted: a uniform scale takes straight lines to straight lines.  The objective
is therefore built to be gauge-invariant -- each chain's rectified residual is
divided back through the local Jacobian of the radial map, which puts it in raw
image pixels, where the edge-localisation noise lives.  See ``ChainSet`` for
why the two obvious alternatives are both exploitable.

Read the identifiability band in the report before the coefficient.  Chains
that point at the image centre carry no distortion information, and a fit whose
cost is flat is reported flat rather than silently resolved.

The pre-registered acceptance criteria and thresholds are in
``docs/UW_DISTORTION_PREREG.md``.  Requires numpy, scipy and opencv.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - the CLI reports this clearly.
    cv2 = None

try:
    from scipy.optimize import minimize
except ImportError:  # pragma: no cover
    minimize = None


# ---------------------------------------------------------------------------
# The radial model.  Everything downstream speaks in these terms.
# ---------------------------------------------------------------------------

def half_diagonal(width: int, height: int) -> float:
    """The radius normalisation: rho == 1 exactly at the frame corner."""
    return 0.5 * math.hypot(float(width), float(height))


def radial_gain(rho2: np.ndarray, coefficients: Sequence[float]) -> np.ndarray:
    """1 + a1 rho^2 + a2 rho^4 + ... evaluated from rho squared."""
    gain = np.ones_like(rho2)
    term = np.ones_like(rho2)
    for coefficient in coefficients:
        term = term * rho2
        gain = gain + coefficient * term
    return gain


def rectify_points(points: np.ndarray, width: int, height: int,
                   coefficients: Sequence[float]) -> np.ndarray:
    """Map raw pixel positions to the straightened frame."""
    centre = np.array([0.5 * width, 0.5 * height], dtype=np.float64)
    radius = half_diagonal(width, height)
    delta = np.asarray(points, dtype=np.float64) - centre
    rho2 = (delta[..., 0] ** 2 + delta[..., 1] ** 2) / (radius * radius)
    return centre + delta * radial_gain(rho2, coefficients)[..., None]


def is_monotone(coefficients: Sequence[float], rho_max: float = 1.05) -> bool:
    """Reject a fold-over: d/dr of r*(1 + a1 rho^2 + ...) must stay positive."""
    rho = np.linspace(0.0, rho_max, 256)
    rho2 = rho * rho
    derivative = np.ones_like(rho)
    term = np.ones_like(rho)
    for power, coefficient in enumerate(coefficients, start=1):
        term = term * rho2
        derivative = derivative + (2 * power + 1) * coefficient * term
    return bool(np.all(derivative > 0.0))


def corner_displacement(width: int, height: int,
                        coefficients: Sequence[float]) -> float:
    """Signed pixel displacement the model applies at the frame corner."""
    return half_diagonal(width, height) * float(sum(coefficients))


def rms_displacement(width: int, height: int,
                     coefficients: Sequence[float]) -> float:
    """RMS displacement over the frame -- the whole-image average PSNR sees."""
    xs = np.linspace(0.5, width - 0.5, 129)
    ys = np.linspace(0.5, height - 0.5, 129)
    u, v = np.meshgrid(xs, ys)
    centre = (0.5 * width, 0.5 * height)
    radius = half_diagonal(width, height)
    dx, dy = u - centre[0], v - centre[1]
    r = np.hypot(dx, dy)
    rho2 = (r / radius) ** 2
    displacement = r * (radial_gain(rho2, coefficients) - 1.0)
    return float(np.sqrt(np.mean(displacement ** 2)))


def opencv_k1(width: int, height: int, focal: float,
              coefficients: Sequence[float]) -> float:
    """First-order translation of a1 into an OPENCV-model k1, for handoff."""
    if focal <= 0.0:
        return float("nan")
    return -float(coefficients[0]) * (half_diagonal(width, height) / focal) ** 2


# ---------------------------------------------------------------------------
# Edge chains.  A line-segment detector is the wrong instrument: it assumes
# straightness and would chop a bowed line into straight pieces, deleting the
# very signal being measured.  So: Canny, chain tracing, subpixel refinement,
# and splitting only at corners.
# ---------------------------------------------------------------------------

_NEIGHBOUR_ORDER = ((-1, 0), (1, 0), (0, -1), (0, 1),
                    (-1, -1), (1, -1), (-1, 1), (1, 1))


WORK_LONG_SIDE = 960


EDGE_APERTURE = 5           # Canny's own Sobel aperture, matched below
EDGE_FLOOR = 640.0          # 40 on a 3x3 Sobel scale; a 5x5 runs ~16x larger


def _edges(gray: np.ndarray, percentile: float = 97.0,
           floor: float = EDGE_FLOOR
           ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Canny with thresholds taken from the frame's own gradient distribution.

    The usual median-*intensity* auto-Canny keeps almost nothing on these dim,
    blurry indoor frames -- it returned 10k edge pixels out of 8.3M on the
    ultra-wide.  Setting the high threshold at a percentile of the gradient
    magnitude makes the yield a property of the edge content instead of the
    exposure.  CLAHE first, because several of these rooms are lit from one
    window and half the frame is in shadow.  The hysteresis ratio is 0.2 and
    the smoothing is deliberately heavy: on soft, motion-blurred edges a light
    filter makes the gradient direction unstable and Canny's non-maximum
    suppression flip-flops between adjacent pixels, laying down a *braided*
    pair of curves whose constant micro-junctions defeat any tracer.  A 5x5
    Sobel aperture stabilises the direction; ``cv2.Canny`` must be told to use
    the same one, or the thresholds are computed on a scale about sixteen times
    off from the gradient it actually applies them to.
    """
    equalised = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    blurred = cv2.GaussianBlur(equalised, (5, 5), 1.0)
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=EDGE_APERTURE)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=EDGE_APERTURE)
    magnitude = np.hypot(gx, gy)
    high = max(float(np.percentile(magnitude, percentile)), floor)
    edge = cv2.Canny(blurred, 0.2 * high, high,
                     apertureSize=EDGE_APERTURE, L2gradient=True)
    return edge, gx, gy, magnitude


def _trace(edge: np.ndarray, min_points: int) -> list[np.ndarray]:
    """8-connected chains that continue *through* junctions.

    Deleting junction pixels -- the obvious way to keep chains unambiguous --
    cuts a floor-plank seam wherever a chair leg crosses it, and at 1920x1440
    that left 1.7 usable chains per frame.  So the walk instead prefers the
    neighbour that best continues the current tangent, and a junction pixel may
    be entered more than once.  Where the continuation is not smooth,
    ``_split_at_corners`` cuts it afterwards on a geometric test.
    """
    kernel = np.ones((3, 3), np.uint8)
    kernel[1, 1] = 0
    occupied = (edge > 0).astype(np.uint8)
    degree = cv2.filter2D(occupied, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    degree = np.where(occupied > 0, degree, 0)

    padded = np.pad(occupied, 1).astype(bool)
    stride = padded.shape[1]
    flat = padded.reshape(-1)
    offsets = [dy * stride + dx for dx, dy in _NEIGHBOUR_ORDER]
    steps = {offset: (float(dx), float(dy))
             for offset, (dx, dy) in zip(offsets, _NEIGHBOUR_ORDER)}

    budget = np.where(flat, 1, 0).astype(np.int8)
    padded_degree = np.pad(degree, 1).reshape(-1)
    junctions = flat & (padded_degree > 2)
    budget[junctions] = np.minimum(padded_degree[junctions] - 1, 3)

    endpoints = flat & (padded_degree == 1)
    starts = list(np.flatnonzero(endpoints)) + list(np.flatnonzero(flat))

    chains: list[np.ndarray] = []
    for start in starts:
        if budget[start] <= 0:
            continue
        budget[start] -= 1
        y0, x0 = divmod(int(start), stride)
        chain = [(x0, y0)]
        used = {start}
        current = start
        tangent: tuple[float, float] | None = None
        while len(chain) < 40000:
            best, best_score, best_step = -1, -2.0, None
            for offset in offsets:
                candidate = current + offset
                if budget[candidate] <= 0 or candidate in used:
                    continue
                dx, dy = steps[offset]
                norm = math.hypot(dx, dy)
                if tangent is None:
                    score = 1.0 / norm          # prefer 4-connected first
                else:
                    score = (tangent[0] * dx + tangent[1] * dy) / norm
                    if score <= 0.0:            # never double back
                        continue
                if score > best_score:
                    best, best_score, best_step = candidate, score, (dx, dy)
            if best < 0:
                break
            budget[best] -= 1
            used.add(best)
            step = steps[best - current]
            chain.append((chain[-1][0] + step[0], chain[-1][1] + step[1]))
            current = best
            if len(chain) >= 4:
                tangent = (chain[-1][0] - chain[-4][0],
                           chain[-1][1] - chain[-4][1])
            else:
                tangent = best_step
        if len(chain) >= min_points:
            points = np.asarray(chain, dtype=np.float64) - 1.0
            chains.append(points)
    return chains


def _sample_bilinear(image: np.ndarray, x: np.ndarray,
                     y: np.ndarray) -> np.ndarray:
    height, width = image.shape
    x = np.clip(x, 0.0, width - 1.001)
    y = np.clip(y, 0.0, height - 1.001)
    x0 = x.astype(np.int64)
    y0 = y.astype(np.int64)
    fx = x - x0
    fy = y - y0
    top = image[y0, x0] * (1 - fx) + image[y0, x0 + 1] * fx
    bottom = image[y0 + 1, x0] * (1 - fx) + image[y0 + 1, x0 + 1] * fx
    return top * (1 - fy) + bottom * fy


def _refine_subpixel(chain: np.ndarray, gx: np.ndarray, gy: np.ndarray,
                     magnitude: np.ndarray) -> np.ndarray:
    """Devernay-style refinement: parabola on |grad| along the edge normal."""
    xs = chain[:, 0].astype(np.int64)
    ys = chain[:, 1].astype(np.int64)
    nx = gx[ys, xs]
    ny = gy[ys, xs]
    norm = np.hypot(nx, ny)
    good = norm > 1e-6
    nx = np.where(good, nx / np.where(good, norm, 1.0), 0.0)
    ny = np.where(good, ny / np.where(good, norm, 1.0), 0.0)
    centre = _sample_bilinear(magnitude, chain[:, 0], chain[:, 1])
    before = _sample_bilinear(magnitude, chain[:, 0] - nx, chain[:, 1] - ny)
    after = _sample_bilinear(magnitude, chain[:, 0] + nx, chain[:, 1] + ny)
    denominator = before - 2.0 * centre + after
    step = np.where(np.abs(denominator) > 1e-9,
                    0.5 * (before - after) / np.where(np.abs(denominator) > 1e-9,
                                                      denominator, 1.0),
                    0.0)
    step = np.clip(step, -0.5, 0.5)
    return np.stack([chain[:, 0] + step * nx, chain[:, 1] + step * ny], axis=1)


def _split_at_corners(chain: np.ndarray, window: int = 5,
                      turn_degrees: float = 30.0) -> list[np.ndarray]:
    """Cut only where the chain genuinely turns; a distortion bow does not."""
    n = len(chain)
    if n < 2 * window + 3:
        return [chain]
    before = chain[window:-window] - chain[:-2 * window]
    after = chain[2 * window:] - chain[window:-window]
    dot = (before * after).sum(axis=1)
    cross = before[:, 0] * after[:, 1] - before[:, 1] * after[:, 0]
    angle = np.degrees(np.abs(np.arctan2(cross, dot)))
    corners = np.flatnonzero(angle > turn_degrees) + window

    pieces = []
    start = 0
    for corner in corners:
        stop = corner - window
        if stop - start >= 2:
            pieces.append(chain[start:stop])
        start = corner + window
    if n - start >= 2:
        pieces.append(chain[start:])
    return pieces


def _line_residual(points: np.ndarray) -> np.ndarray:
    """Signed perpendicular distance to the total-least-squares line."""
    centred = points - points.mean(axis=0)
    scatter = centred.T @ centred
    _, vectors = np.linalg.eigh(scatter)
    normal = vectors[:, 0]                    # smallest eigenvalue direction
    return centred @ normal


def _is_smooth(points: np.ndarray, max_straight_px: float,
               min_r_squared: float) -> bool:
    """Keep a smooth bow, drop a zigzag; keep an already-straight chain."""
    residual = _line_residual(points)
    rms = float(np.sqrt(np.mean(residual ** 2)))
    if rms <= max_straight_px:
        return True
    direction = points[-1] - points[0]
    length = float(np.hypot(*direction))
    if length < 1e-6:
        return False
    t = (points - points[0]) @ (direction / length)
    design = np.stack([np.ones_like(t), t, t * t], axis=1)
    coefficients, *_ = np.linalg.lstsq(design, residual, rcond=None)
    predicted = design @ coefficients
    total = float(np.sum((residual - residual.mean()) ** 2))
    if total <= 0.0:
        return False
    r_squared = 1.0 - float(np.sum((residual - predicted) ** 2)) / total
    return r_squared >= min_r_squared


def _resample(points: np.ndarray, limit: int) -> np.ndarray:
    if len(points) <= limit:
        return points
    index = np.linspace(0, len(points) - 1, limit).round().astype(np.int64)
    return points[index]


def extract_chains(gray: np.ndarray, *, min_chord_fraction: float = 0.12,
                   min_points: int = 40, max_bow_ratio: float = 0.15,
                   max_straight_px: float = 1.0, min_r_squared: float = 0.5,
                   points_per_chain: int = 120, percentile: float = 97.0,
                   work_long_side: int = WORK_LONG_SIDE,
                   mask: np.ndarray | None = None) -> list[np.ndarray]:
    """Long, smooth, subpixel edge chains that are candidates for straight.

    Detection runs at a canonical working scale of at most ``work_long_side``
    pixels on the long side and the accepted chains are mapped back to the
    frame's own pixel grid before returning, so the caller always sees native
    coordinates.  This is not a shortcut: at full 3840x2160 the braiding
    described in ``_edges`` cut every chain far below the length gate -- five
    ultra-wide frames yielded three usable chains -- while the same frames at
    960 on the long side yield sixty.  960 is also exactly the resolution the
    ``--downscale 4`` training run in ``docs/3DGS.md`` saw, so the measurement
    is made where the 0.7 dB was lost.  The fitted coefficient is normalised by
    the half-diagonal and is therefore the same number at either scale.
    """
    scale = min(1.0, float(work_long_side) / max(gray.shape))
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_AREA)
        if mask is not None:
            mask = cv2.resize(mask.astype(np.uint8), (gray.shape[1],
                                                      gray.shape[0]),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
    height, width = gray.shape
    diagonal = math.hypot(width, height)
    min_chord = min_chord_fraction * diagonal

    edge, gx, gy, magnitude = _edges(gray, percentile)
    if mask is not None:
        edge = np.where(mask, edge, 0)

    accepted: list[np.ndarray] = []
    for chain in _trace(edge, min_points):
        refined = _refine_subpixel(chain, gx, gy, magnitude)
        for piece in _split_at_corners(refined):
            if len(piece) < min_points:
                continue
            chord = float(np.hypot(*(piece[-1] - piece[0])))
            if chord < min_chord:
                continue
            residual = _line_residual(piece)
            if float(np.max(np.abs(residual))) / chord > max_bow_ratio:
                continue
            if not _is_smooth(piece, max_straight_px, min_r_squared):
                continue
            accepted.append(_resample(piece, points_per_chain))
    if scale < 1.0:
        accepted = [(chain + 0.5) / scale - 0.5 for chain in accepted]
    return accepted


# ---------------------------------------------------------------------------
# The fit.  Cost is a robust sum of per-chain bow/chord, which is invariant to
# a uniform rescale of the image and so cannot be driven to zero by shrinking.
# ---------------------------------------------------------------------------

class ChainSet:
    """Chains from many frames, flattened for a vectorised objective."""

    def __init__(self, chains: Sequence[np.ndarray],
                 frames: Sequence[int], width: int, height: int):
        if not chains:
            raise ValueError("no chains")
        self.width = int(width)
        self.height = int(height)
        self.count = len(chains)
        self.frames = np.asarray(frames, dtype=np.int64)
        self.sizes = np.asarray([len(c) for c in chains], dtype=np.int64)
        self.id = np.repeat(np.arange(self.count), self.sizes)
        points = np.concatenate(chains, axis=0)
        centre = np.array([0.5 * width, 0.5 * height])
        self.delta = points - centre
        radius = half_diagonal(width, height)
        self.rho2 = (self.delta[:, 0] ** 2
                     + self.delta[:, 1] ** 2) / (radius * radius)
        self.rho_mean = np.bincount(self.id, np.sqrt(self.rho2)) / self.sizes
        norm = np.maximum(np.hypot(self.delta[:, 0], self.delta[:, 1]), 1e-9)
        self.unit = self.delta / norm[:, None]      # radial direction per point
        ends = np.cumsum(self.sizes) - 1
        self.first = np.concatenate([[0], ends[:-1] + 1])
        self.last = ends
        self.weight = self.chords_px([0.0])
        self.weight = self.weight / max(float(np.median(self.weight)), 1e-9)

    def _straightness(self, coefficients: Sequence[float]
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-chain (raw-pixel residual, rectified residual, chord).

        The residual is divided back through the local Jacobian of the radial
        map, which puts it in **raw image pixels** -- the domain where the edge
        localisation noise actually lives, and the only domain where the
        objective is a proper least-squares.

        Without that division the objective is exploitable in both directions.
        A chain pointing at the image centre carries no distortion information
        at all, but a larger ``a1`` stretches it lengthwise, so dividing by its
        chord buys cost for nothing and drags the fit positive; conversely a
        raw sum of rectified residuals is minimised by shrinking the outer
        field, which drags it negative.  Dividing by the Jacobian removes both:
        a uniform rescale multiplies residual and Jacobian alike.
        """
        gain = radial_gain(self.rho2, coefficients)             # tangential
        x = self.delta[:, 0] * gain
        y = self.delta[:, 1] * gain
        n = self.sizes.astype(np.float64)
        dx = x - (np.bincount(self.id, x) / n)[self.id]
        dy = y - (np.bincount(self.id, y) / n)[self.id]
        sxx = np.bincount(self.id, dx * dx)
        syy = np.bincount(self.id, dy * dy)
        sxy = np.bincount(self.id, dx * dy)
        trace = sxx + syy
        discriminant = np.maximum(trace * trace - 4.0 * (sxx * syy - sxy * sxy),
                                  0.0)
        smallest = 0.5 * (trace - np.sqrt(discriminant))

        # Eigenvector of the smaller eigenvalue: the fitted line's normal.
        ax, ay = sxy, smallest - sxx
        bx, by = smallest - syy, sxy
        take = (ax * ax + ay * ay) >= (bx * bx + by * by)
        nx = np.where(take, ax, bx)
        ny = np.where(take, ay, by)
        length = np.hypot(nx, ny)
        degenerate = length < 1e-12
        nx = np.where(degenerate, 1.0, nx / np.where(degenerate, 1.0, length))
        ny = np.where(degenerate, 0.0, ny / np.where(degenerate, 1.0, length))

        signed = dx * nx[self.id] + dy * ny[self.id]
        radial = 1.0 + sum((2 * power + 1) * coefficient * self.rho2 ** power
                           for power, coefficient
                           in enumerate(coefficients, start=1))
        cosine = nx[self.id] * self.unit[:, 0] + ny[self.id] * self.unit[:, 1]
        cosine2 = cosine * cosine
        jacobian = np.sqrt(radial * radial * cosine2
                           + gain * gain * (1.0 - cosine2))
        raw = np.sqrt(np.bincount(self.id, (signed / jacobian) ** 2) / n)

        rectified = np.sqrt(np.maximum(smallest, 0.0) / n)
        chord = np.hypot(x[self.last] - x[self.first],
                         y[self.last] - y[self.first])
        return raw, rectified, chord

    def residual_px(self, coefficients: Sequence[float]) -> np.ndarray:
        """Per-chain RMS straightness residual, in raw image pixels."""
        return self._straightness(coefficients)[0]

    def bows(self, coefficients: Sequence[float]) -> np.ndarray:
        """Per-chain residual divided by chord: a dimensionless curvature."""
        raw, _, chord = self._straightness(coefficients)
        return raw / np.maximum(chord, 1e-9)

    def sagitta_px(self, coefficients: Sequence[float]) -> np.ndarray:
        return self._straightness(coefficients)[0]

    def chords_px(self, coefficients: Sequence[float]) -> np.ndarray:
        return self._straightness(coefficients)[2]


def _huber(values: np.ndarray, delta: float) -> np.ndarray:
    absolute = np.abs(values)
    quadratic = np.minimum(absolute, delta)
    return 0.5 * quadratic ** 2 + delta * (absolute - quadratic)


def _cost(coefficients: Sequence[float], chains: ChainSet, delta: float,
          weights: np.ndarray | None = None) -> float:
    if not is_monotone(coefficients):
        return float("inf")
    penalty = chains.weight * _huber(chains.residual_px(coefficients), delta)
    if weights is not None:
        return float(np.sum(weights * penalty))
    return float(np.sum(penalty))


def fit_coefficients(chains: ChainSet, order: int = 1, *,
                     weights: np.ndarray | None = None,
                     grid: float = 0.4, steps: int = 161,
                     start: Sequence[float] | None = None) -> list[float]:
    """Coarse grid on a1, then Nelder-Mead over the requested order."""
    delta = float(np.median(chains.residual_px([0.0] * order)))
    delta = max(delta, 1e-6)

    if start is None:
        best, best_cost = 0.0, float("inf")
        for a1 in np.linspace(-grid, grid, steps):
            value = _cost([a1] + [0.0] * (order - 1), chains, delta, weights)
            if value < best_cost:
                best, best_cost = float(a1), value
        start = [best] + [0.0] * (order - 1)

    if minimize is None:  # pragma: no cover - scipy is a hard requirement
        return list(start)
    result = minimize(lambda a: _cost(a, chains, delta, weights),
                      np.asarray(start, dtype=np.float64),
                      method="Nelder-Mead",
                      options={"xatol": 1e-6, "fatol": 1e-12, "maxiter": 2000})
    return [float(v) for v in result.x]


def identifiability(chains: ChainSet, coefficients: Sequence[float],
                    tolerance: float = 0.02) -> dict[str, float]:
    """The a1 band over which the cost barely moves.

    A plumb-line fit is unidentifiable from chains that point at the image
    centre: they carry no distortion information, and their cost is flat in
    a1, so an optimiser lands somewhere arbitrary and a bootstrap over frames
    agrees with itself about it.  Scanning the cost exposes that directly --
    a wide band means the data does not contain the answer.
    """
    delta = max(float(np.median(chains.residual_px([0.0] * len(coefficients)))),
                1e-6)
    best = _cost(coefficients, chains, delta)
    zero = _cost([0.0] * len(coefficients), chains, delta)
    tail = list(coefficients[1:])
    grid = np.linspace(-0.4, 0.4, 801)
    inside = [float(a1) for a1 in grid
              if _cost([a1] + tail, chains, delta) <= best * (1.0 + tolerance)]
    radius = half_diagonal(chains.width, chains.height)
    return {
        "cost_at_zero": zero,
        "cost_at_fit": best,
        "cost_reduction": float(1.0 - best / zero) if zero > 0 else 0.0,
        "a1_band_low": min(inside) if inside else float("nan"),
        "a1_band_high": max(inside) if inside else float("nan"),
        "corner_px_band_low": radius * (min(inside) + sum(tail))
        if inside else float("nan"),
        "corner_px_band_high": radius * (max(inside) + sum(tail))
        if inside else float("nan"),
    }


def bootstrap(chains: ChainSet, coefficients: Sequence[float], *,
              draws: int = 200, seed: int = 42) -> dict[str, float]:
    """Resample frames with replacement; the spread is the honest error bar."""
    rng = np.random.default_rng(seed)
    unique = np.unique(chains.frames)
    order = len(coefficients)
    samples = []
    slot = np.searchsorted(unique, chains.frames)
    for _ in range(draws):
        drawn = rng.choice(unique, size=len(unique), replace=True)
        weights = np.bincount(np.searchsorted(unique, drawn),
                              minlength=len(unique)).astype(np.float64)
        per_chain = weights[slot]
        if per_chain.sum() <= 0:
            continue
        samples.append(fit_coefficients(chains, order, weights=per_chain,
                                        start=list(coefficients)))
    if not samples:  # pragma: no cover
        return {}
    array = np.asarray(samples)
    displacement = half_diagonal(chains.width, chains.height) * array.sum(axis=1)
    low, high = np.percentile(displacement, [5.0, 95.0])
    return {
        "draws": len(samples),
        "a1_std": float(np.std(array[:, 0])),
        "corner_px_p05": float(low),
        "corner_px_p95": float(high),
        "corner_px_halfwidth": float(0.5 * (high - low)),
    }


# ---------------------------------------------------------------------------
# Injection: build an image whose correct rectification coefficient is known.
# ---------------------------------------------------------------------------

def inject_distortion(image: np.ndarray, coefficients: Sequence[float]
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Warp so that ``rectify_points(.., coefficients)`` straightens the result.

    The output pixel ``p`` samples the source at ``rectify(p)``, so the exact
    coefficient that straightens the output is the one passed in -- no
    polynomial inversion sits between the truth and the answer.  The second
    return value is the valid-data mask, eroded, for the positive coefficients
    that pull samples in from outside the source frame.
    """
    height, width = image.shape[:2]
    xs = np.arange(width, dtype=np.float64)
    ys = np.arange(height, dtype=np.float64)
    u, v = np.meshgrid(xs, ys)
    grid = np.stack([u, v], axis=-1)
    source = rectify_points(grid, width, height, coefficients)
    map_x = source[..., 0].astype(np.float32)
    map_y = source[..., 1].astype(np.float32)
    warped = cv2.remap(image, map_x, map_y, cv2.INTER_LANCZOS4,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    inside = ((map_x >= 0) & (map_x <= width - 1)
              & (map_y >= 0) & (map_y <= height - 1)).astype(np.uint8)
    inside = cv2.erode(inside, np.ones((9, 9), np.uint8))
    return warped, inside.astype(bool)


def compose_expected(native: Sequence[float], injected: Sequence[float],
                     order: int) -> list[float]:
    """The coefficient that straightens an injected frame with native error.

    Rectifying the warped image means applying the native rectification to the
    already-injected position, so the truth is the composition, not the sum.
    Fitted numerically on rho in [0, 1] so the comparison is exact rather than
    first order.
    """
    rho = np.linspace(0.0, 1.0, 512)
    rho2 = rho * rho
    intermediate = rho * radial_gain(rho2, injected)
    composed = intermediate * radial_gain(intermediate ** 2, native)
    design = np.stack([rho * rho2 ** power for power in range(1, order + 1)],
                      axis=1)
    solution, *_ = np.linalg.lstsq(design, composed - rho, rcond=None)
    return [float(v) for v in solution]


# ---------------------------------------------------------------------------
# Session plumbing
# ---------------------------------------------------------------------------

def _frame_records(session: Path, lens: str) -> tuple[Path, list[str]]:
    """Frame directory and file list for one lens of one session."""
    if lens == "wide" and (session / "frames_wide.jsonl").exists():
        listing = session / "frames_wide.jsonl"
    else:
        listing = session / "frames.jsonl"
    files = []
    with listing.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                files.append(json.loads(line)["file"])
    return session, files


def select_frames(session: Path, lens: str, count: int) -> list[Path]:
    """Sharper half of the session by Laplacian variance, evenly spaced.

    Fixed rule, chosen before any fit ran: it selects on sharpness, which is a
    property of the capture, and never on the answer.
    """
    root, files = _frame_records(session, lens)
    paths = [root / name for name in files]
    sharpness = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            sharpness.append(-1.0)
            continue
        if max(image.shape) > 1280:
            image = cv2.resize(image, None, fx=0.5, fy=0.5,
                               interpolation=cv2.INTER_AREA)
        sharpness.append(float(cv2.Laplacian(image, cv2.CV_64F).var()))
    sharpness = np.asarray(sharpness)
    usable = np.flatnonzero(sharpness >= max(np.median(sharpness), 0.0))
    if len(usable) == 0:  # pragma: no cover
        usable = np.arange(len(paths))
    if len(usable) > count:
        usable = usable[np.linspace(0, len(usable) - 1, count)
                        .round().astype(np.int64)]
    return [paths[i] for i in usable]


def gather(paths: Iterable[Path], *, inject: Sequence[float] | None = None,
           progress: bool = False, **chain_options
           ) -> tuple[ChainSet | None, dict[str, Any]]:
    """Run chain extraction over frames and flatten the result."""
    chains: list[np.ndarray] = []
    frames: list[int] = []
    width = height = 0
    contributing = 0
    read = 0
    for index, path in enumerate(paths):
        read = index + 1
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        mask = None
        if inject is not None:
            image, mask = inject_distortion(image, inject)
        height, width = image.shape
        found = extract_chains(image, mask=mask, **chain_options)
        if found:
            contributing += 1
        chains.extend(found)
        frames.extend([index] * len(found))
        if progress:
            print(f"  frame {index + 1}: {len(found)} chains "
                  f"({len(chains)} total)", file=sys.stderr)
    census = {"frames_read": read,
              "frames_contributing": contributing,
              "chains": len(chains),
              "width": width, "height": height}
    if not chains:
        return None, census
    return ChainSet(chains, frames, width, height), census


PREREG = {
    "min_frames_contributing": 30,
    "min_chains": 300,
    "min_outer_chains": 80,
    "min_outer_rho": 0.6,
    "min_median_chord_fraction": 0.12,
    "max_bootstrap_halfwidth_fraction": 0.25,
}


def sufficiency(chains: ChainSet, census: dict[str, Any]) -> dict[str, Any]:
    """The pre-registered gate: refuse to quote a coefficient without data."""
    diagonal = math.hypot(chains.width, chains.height)
    chords = chains.chords_px([0.0])
    outer = int(np.sum(chains.rho_mean >= PREREG["min_outer_rho"]))
    median_chord_fraction = float(np.median(chords) / diagonal)
    checks = {
        "frames_contributing": (census["frames_contributing"],
                                PREREG["min_frames_contributing"]),
        "chains": (chains.count, PREREG["min_chains"]),
        "outer_chains": (outer, PREREG["min_outer_chains"]),
        "median_chord_fraction": (round(median_chord_fraction, 4),
                                  PREREG["min_median_chord_fraction"]),
    }
    failures = [name for name, (value, floor) in checks.items() if value < floor]
    return {"checks": {k: {"value": v, "required": f}
                       for k, (v, f) in checks.items()},
            "passed": not failures, "failed": failures}


def measure(paths: Sequence[Path], *, order: int = 1,
            inject: Sequence[float] | None = None, draws: int = 200,
            progress: bool = False, **chain_options) -> dict[str, Any]:
    """One complete measurement: chains, gate, fit, bootstrap, residuals."""
    chains, census = gather(paths, inject=inject, progress=progress,
                            **chain_options)
    report: dict[str, Any] = {"census": census,
                              "inject": list(inject) if inject else None}
    if chains is None:
        report["sufficiency"] = {"passed": False, "failed": ["chains"]}
        return report

    report["sufficiency"] = sufficiency(chains, census)
    coefficients = fit_coefficients(chains, order)
    width, height = chains.width, chains.height

    before = chains.bows([0.0] * order)
    after = chains.bows(coefficients)
    before_px = chains.residual_px([0.0] * order)
    after_px = chains.residual_px(coefficients)

    report.update({
        "order": order,
        "coefficients": coefficients,
        "corner_px": corner_displacement(width, height, coefficients),
        "rms_px": rms_displacement(width, height, coefficients),
        "straightness": {
            "median_bow_before": float(np.median(before)),
            "median_bow_after": float(np.median(after)),
            "median_residual_px_before": float(np.median(before_px)),
            "median_residual_px_after": float(np.median(after_px)),
            "mean_residual_px_before": float(np.mean(before_px)),
            "mean_residual_px_after": float(np.mean(after_px)),
            "p90_residual_px_before": float(np.percentile(before_px, 90)),
            "p90_residual_px_after": float(np.percentile(after_px, 90)),
        },
        "identifiability": identifiability(chains, coefficients),
        "median_chord_px": float(np.median(chains.chords_px([0.0] * order))),
        "median_rho": float(np.median(chains.rho_mean)),
    })
    if draws:
        report["bootstrap"] = bootstrap(chains, coefficients, draws=draws)
    return report


def main(argv: list[str]) -> int:
    if cv2 is None:
        print("this tool needs opencv:  pip install opencv-python-headless",
              file=sys.stderr)
        return 2
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("session", help="a recorded session directory")
    parser.add_argument("--lens", choices=("ultrawide", "wide"),
                        default="ultrawide",
                        help="which arm to read; wide uses frames_wide/ when "
                             "the session has one")
    parser.add_argument("--frames", type=int, default=60,
                        help="how many frames to read, from the sharper half")
    parser.add_argument("--order", type=int, default=1, choices=(1, 2),
                        help="1 fits a1 only; 2 adds a4")
    parser.add_argument("--inject", type=float, default=None,
                        help="known-answer control: warp each frame so this a1 "
                             "is the correct answer, then fit")
    parser.add_argument("--draws", type=int, default=200,
                        help="bootstrap resamples over frames; 0 disables")
    parser.add_argument("--min-chord-fraction", type=float, default=0.12)
    parser.add_argument("--max-straight-px", type=float, default=1.0)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--out", default=None, help="write the report JSON here")
    args = parser.parse_args(argv)

    session = Path(args.session).expanduser().resolve()
    paths = select_frames(session, args.lens, args.frames)
    report = measure(paths, order=args.order,
                     inject=[args.inject] if args.inject is not None else None,
                     draws=args.draws, progress=args.progress,
                     min_chord_fraction=args.min_chord_fraction,
                     max_straight_px=args.max_straight_px)
    report["session"] = str(session)
    report["lens"] = args.lens
    report["frames_requested"] = args.frames
    print(json.dumps(report, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
