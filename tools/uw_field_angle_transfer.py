#!/usr/bin/env python3
"""Transfer Apple's radial lookup to the format a session actually recorded at.

The factory calibration under ``calib/`` is quoted at 4032x3024.  Nothing
records at that format, and two earlier attempts to give the raw ultra-wide a
camera model were blamed on the mismatch and abandoned.  That blame is only
half right.

**Distortion is a property of the lens as a function of field angle.**  Two
readouts of one sensor are the same physical image plane sampled at different
pitches, so at equal field angle their raw pixel radii differ by exactly one
scalar ``k = pitch_calibrated / pitch_active``.  Transferring the table is a
change of the *radius axis* by ``k`` and nothing else -- the magnification
values are untouched.  ``rectify_3dgs_export.py --hfov 102.5`` instead matched
the horizontal *extent*, which is the wrong invariant and stretches the table
by the wrong factor.

The trap this tool exists to avoid, recorded in ``docs/POSE.md``:
``videoFieldOfView`` -- the 106.2007 in the manifest -- is a **whole-frame
quantity that already contains the distortion**, so it is a field angle at the
frame's horizontal edge and ``(W/2)/tan(hfov/2)`` is **not** a paraxial focal
length.  Here ``videoFieldOfView`` only ever enters as the argument of a
tangent inside ``R_cal``; the paraxial focal length of the active format is
``fx * k``, and that is the number used everywhere a focal length is needed.

    python3 tools/uw_field_angle_transfer.py report
    python3 tools/uw_field_angle_transfer.py roundtrip
    python3 tools/uw_field_angle_transfer.py wide-null ~/nav_data/20260820-211848-d06152
    python3 tools/uw_field_angle_transfer.py profile ~/nav_data/20260820-211848-d06152

The pre-registered acceptance criteria are in ``docs/UW_TRANSFER_PREREG.md``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - the CLI reports this clearly.
    cv2 = None

REPO = Path(__file__).resolve().parent.parent
CALIB = {
    "ultrawide": REPO / "calib" / "iphone17-1_ultrawide.json",
    "wide": REPO / "calib" / "iphone17-1_wide.json",
}

# The two active formats this device records at, as the manifests log them.
# These are inputs, not results: `videoFieldOfView` per lens, and the frame size.
ACTIVE = {
    "ultrawide": {"width": 3840, "height": 2160, "hfov": 106.2007},
    "wide": {"width": 640, "height": 480, "hfov": 69.52744},
}

# The 3DGS page's own anchor: ten pixels of texture displacement at f = 653 px
# is 1.6 dB, and the unattributed penalty is 0.7 dB = 4.4 px.
ANCHOR_FOCAL_PX = 653.0
ANCHOR_PX_PER_DB = 10.0 / 1.6
UNATTRIBUTED_PX = 4.4


# ---------------------------------------------------------------------------
# The transfer
# ---------------------------------------------------------------------------

def _lookup(radius: np.ndarray, table: np.ndarray, r_max: float) -> np.ndarray:
    """Apple's linear interpolation, clamped past the last knot as they do."""
    n = len(table)
    radius = np.asarray(radius, dtype=np.float64)
    position = radius * (n - 1) / r_max
    index = np.clip(position.astype(np.int64), 0, n - 2)
    fraction = np.clip(position - index, 0.0, 1.0)
    interpolated = (1.0 - fraction) * table[index] + fraction * table[index + 1]
    return np.where(radius < r_max, interpolated, table[-1])


MODES = ("field-angle", "paraxial", "width", "identity", "focal")


class Transfer:
    """Apple's radial table re-indexed onto an active format's pixel grid.

    ``mode`` selects how the radius axis is scaled.  ``field-angle`` is the
    claim under test; the rest are the pre-registered ablations.

    ``field-angle``  k = (W/2) / R_cal(hfov/2), the pixel-pitch ratio.
    ``paraxial``     k = ((W/2)/tan(hfov/2)) / fx -- the trap, run deliberately.
    ``width``        k = W_active / W_calibrated, a pure vertical crop.  This is
                     also the radius axis ``rectify_3dgs_export.py --hfov 102.5``
                     uses: it treats the active frame as a resample of the
                     calibration frame and scales the source coordinates by the
                     image size, so the two ablations coincide here.
    ``identity``     no distortion at all; the resampling control.
    """

    def __init__(self, calibration: dict[str, Any], width: int, height: int,
                 hfov: float, mode: str = "field-angle",
                 f_paraxial: float | None = None):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode
        self.calibration = calibration
        self.width, self.height = int(width), int(height)
        self.hfov = float(hfov)

        reference = calibration["reference_dimensions"]
        self.ref_w, self.ref_h = float(reference[0]), float(reference[1])
        intrinsics = calibration["intrinsics"]
        self.fx = float(intrinsics["fx"])
        self.fy = float(intrinsics["fy"])
        self.cx_cal = float(intrinsics["cx"])
        self.cy_cal = float(intrinsics["cy"])
        self.pixel_size_mm = float(calibration.get("pixel_size_mm") or 0.0)
        centre = calibration.get("lens_distortion_center") or [self.cx_cal,
                                                               self.cy_cal]
        self.centre_cal = np.asarray(centre, dtype=np.float64)
        self.r_max_cal = float(math.hypot(
            max(self.centre_cal[0], self.ref_w - self.centre_cal[0]),
            max(self.centre_cal[1], self.ref_h - self.centre_cal[1])))

        self.forward_table = np.asarray(
            calibration.get("lens_distortion_lookup_table") or [0.0, 0.0],
            dtype=np.float64)
        self.shipped_inverse_table = np.asarray(
            calibration.get("inverse_lens_distortion_lookup_table")
            or [0.0, 0.0], dtype=np.float64)

        # --- the one scalar -------------------------------------------------
        # `videoFieldOfView` enters here and only here, as a field angle.
        self.theta_edge = math.radians(self.hfov / 2.0)
        self.r_ideal_edge_cal = self.fx * math.tan(self.theta_edge)
        self.r_raw_edge_cal = float(
            self.r_ideal_edge_cal
            * (1.0 + _lookup(self.r_ideal_edge_cal, self.forward_table,
                             self.r_max_cal)))
        if mode == "field-angle":
            self.k = (self.width / 2.0) / self.r_raw_edge_cal
        elif mode == "paraxial":
            self.k = ((self.width / 2.0) / math.tan(self.theta_edge)) / self.fx
        elif mode == "width":
            self.k = self.width / self.ref_w
        elif mode == "focal":
            # A device that hands over a paraxial focal length directly -- an
            # ARKit session's `fx` -- fixes k without going through a field of
            # view at all.  Same transfer, one fewer assumption.
            if not f_paraxial:
                raise ValueError("mode 'focal' needs f_paraxial")
            self.k = float(f_paraxial) / self.fx
        else:                                   # identity
            self.k = (self.width / 2.0) / self.r_raw_edge_cal
            self.forward_table = np.zeros_like(self.forward_table)
            self.shipped_inverse_table = np.zeros_like(
                self.shipped_inverse_table)

        self.f_paraxial = self.fx * self.k
        self.r_max = self.k * self.r_max_cal
        self.pitch_active_mm = (self.pixel_size_mm / self.k
                                if self.k > 0 else float("nan"))

        # The active readout is assumed centred on the same optical axis, so
        # the offsets of the principal point and of the distortion centre from
        # the frame centre carry over scaled by k.  They are a few pixels.
        image_centre_cal = np.array([self.ref_w / 2.0, self.ref_h / 2.0])
        image_centre = np.array([self.width / 2.0, self.height / 2.0])
        self.centre = image_centre + self.k * (self.centre_cal
                                               - image_centre_cal)
        self.principal = image_centre + self.k * (
            np.array([self.cx_cal, self.cy_cal]) - image_centre_cal)

        self.r_corner = float(np.linalg.norm(
            np.maximum(self.centre,
                       np.array([self.width, self.height]) - self.centre)))
        self._build_inverse()

    # -- the radial maps -----------------------------------------------------

    def magnification_forward(self, r_ideal: np.ndarray) -> np.ndarray:
        """ideal -> raw magnification at an ideal radius in ACTIVE pixels."""
        return _lookup(np.asarray(r_ideal, float) / self.k,
                       self.forward_table, self.r_max_cal)

    def magnification_shipped_inverse(self, r_raw: np.ndarray) -> np.ndarray:
        """raw -> ideal magnification from Apple's second table."""
        return _lookup(np.asarray(r_raw, float) / self.k,
                       self.shipped_inverse_table, self.r_max_cal)

    def ideal_to_raw_radius(self, r_ideal: np.ndarray) -> np.ndarray:
        r_ideal = np.asarray(r_ideal, dtype=np.float64)
        return r_ideal * (1.0 + self.magnification_forward(r_ideal))

    def _build_inverse(self, samples: int = 200001) -> None:
        """Numerically invert the forward map, so the round trip is exact.

        Apple ships both directions and they agree to about 0.2 px at the
        calibration scale -- 42 knots over a 2527 px radius is 62 px between
        samples, so that residual is interpolation, not a misread convention.
        It is still 0.2 px, and a model whose two directions disagree by that
        much cannot be checked to "well under a pixel".  The model therefore
        carries one map and its own numerical inverse; the shipped inverse is
        kept only to report the disagreement.
        """
        reach = 1.35 * max(self.r_corner, 1.0)
        ideal = np.linspace(0.0, reach, samples)
        raw = self.ideal_to_raw_radius(ideal)
        self._inverse_raw = raw
        self._inverse_ideal = ideal
        self.monotone = bool(np.all(np.diff(raw) > 0.0))

    def raw_to_ideal_radius(self, r_raw: np.ndarray) -> np.ndarray:
        """The numerical inverse of ``ideal_to_raw_radius``."""
        r_raw = np.asarray(r_raw, dtype=np.float64)
        guess = np.interp(r_raw, self._inverse_raw, self._inverse_ideal)
        # Two Newton steps on f(R) = R*(1+m(R)) - r, differentiated by a
        # central difference because m is piecewise linear in R.
        for _ in range(2):
            step = max(self.r_max_cal * self.k * 1e-4, 1e-3)
            value = self.ideal_to_raw_radius(guess) - r_raw
            slope = ((self.ideal_to_raw_radius(guess + step)
                      - self.ideal_to_raw_radius(np.maximum(guess - step, 0.0)))
                     / (step + np.minimum(guess, step)))
            slope = np.where(np.abs(slope) < 1e-9, 1.0, slope)
            guess = np.maximum(guess - value / slope, 0.0)
        return guess

    def raw_to_ideal(self, points: np.ndarray) -> np.ndarray:
        delta = np.asarray(points, dtype=np.float64) - self.centre
        r = np.linalg.norm(delta, axis=-1)
        gain = np.where(r > 1e-9, self.raw_to_ideal_radius(r) / np.maximum(r, 1e-9), 1.0)
        return self.centre + delta * gain[..., None]

    def ideal_to_raw(self, points: np.ndarray) -> np.ndarray:
        delta = np.asarray(points, dtype=np.float64) - self.centre
        r = np.linalg.norm(delta, axis=-1)
        gain = np.where(r > 1e-9, self.ideal_to_raw_radius(r) / np.maximum(r, 1e-9), 1.0)
        return self.centre + delta * gain[..., None]

    # -- what the caller wants to know --------------------------------------

    def field_angle(self, r_raw: float) -> float:
        """Field angle, degrees, at a raw radius in active pixels."""
        return math.degrees(math.atan(
            float(self.raw_to_ideal_radius(np.array([r_raw]))[0])
            / self.f_paraxial))

    def corner_displacement(self) -> float:
        """Raw -> ideal displacement at the frame corner, in active pixels."""
        r = self.r_corner
        return float(self.raw_to_ideal_radius(np.array([r]))[0] - r)

    def rms_displacement(self, samples: int = 257) -> float:
        xs = np.linspace(0.5, self.width - 0.5, samples)
        ys = np.linspace(0.5, self.height - 0.5,
                         max(3, int(round(samples * self.height / self.width))))
        u, v = np.meshgrid(xs, ys)
        r = np.hypot(u - self.centre[0], v - self.centre[1])
        displacement = self.raw_to_ideal_radius(r) - r
        return float(np.sqrt(np.mean(displacement ** 2)))

    def polynomial(self, order: int = 5) -> list[float]:
        """The raw->ideal map as ``fit_uw_distortion``'s a1..aN in rho.

        ``rho`` is the radius normalised so that rho == 1 at the frame corner,
        which is the basis ``tools/fit_uw_distortion.py`` fits in, so the two
        instruments can be compared without either being re-derived.
        """
        half = 0.5 * math.hypot(self.width, self.height)
        rho = np.linspace(1e-6, 1.0, 2048)
        r = rho * half
        gain = self.raw_to_ideal_radius(r) / r - 1.0
        design = np.stack([rho ** (2 * power)
                           for power in range(1, order + 1)], axis=1)
        solution, *_ = np.linalg.lstsq(design, gain, rcond=None)
        return [float(value) for value in solution]

    def report(self) -> dict[str, Any]:
        half = 0.5 * math.hypot(self.width, self.height)
        return {
            "mode": self.mode,
            "active": [self.width, self.height],
            "video_field_of_view_deg": self.hfov,
            "k": self.k,
            "f_paraxial_px": self.f_paraxial,
            "f_naive_paraxial_px": (self.width / 2.0) / math.tan(self.theta_edge),
            "f_size_scaled_px": self.fx * self.width / self.ref_w,
            "focal_mm": self.fx * self.pixel_size_mm,
            "pitch_active_um": self.pitch_active_mm * 1000.0,
            "r_max_active_px": self.r_max,
            "corner_radius_px": self.r_corner,
            "corner_radius_in_calibration_px": self.r_corner / self.k,
            "corner_fraction_of_r_max": self.r_corner / self.k / self.r_max_cal,
            "corner_field_angle_deg": self.field_angle(self.r_corner),
            "edge_field_angle_deg": self.field_angle(self.width / 2.0),
            "corner_displacement_px": self.corner_displacement(),
            # The same corner read off Apple's *other* table instead of by
            # inverting the forward one.  The two compose to the identity to
            # 0.2 px, but at the rim the knots are 62 calibration pixels apart
            # and the magnification is climbing 1.4 points per knot, so their
            # pointwise values there are not the same number.  That gap is the
            # model's own uncertainty at the corner and is reported, not hidden.
            "corner_displacement_shipped_inverse_px": float(
                self.r_corner * self.magnification_shipped_inverse(
                    np.array([self.r_corner]))[0]),
            "rms_displacement_px": self.rms_displacement(),
            "half_diagonal_px": half,
            "monotone": self.monotone,
        }

    # -- the warp ------------------------------------------------------------

    def image_maps(self, img_w: int, img_h: int, f_out: float | None = None
                   ) -> tuple[np.ndarray, np.ndarray]:
        """``cv2.remap`` coordinates for an image of this size.

        The output is a pinhole of focal ``f_out``, defaulting to the active
        format's own paraxial focal length rescaled to the image -- so the
        default output is "the same camera with the lens model removed", at the
        same centre scale, and a radius in the output means the same thing it
        meant in the input to first order.  The calibration is quoted against
        the *active* frame here, not against ``reference_dimensions``, because
        the transfer has already moved it.
        """
        scale_x, scale_y = img_w / self.width, img_h / self.height
        f_out = float(f_out if f_out else self.f_paraxial * scale_x)
        u, v = np.meshgrid(np.arange(img_w, dtype=np.float64),
                           np.arange(img_h, dtype=np.float64))
        # Output pixel -> bearing -> ideal position on the active frame's plane.
        x = (u + 0.5 - img_w / 2.0) / f_out * self.f_paraxial + self.principal[0]
        y = (v + 0.5 - img_h / 2.0) / f_out * self.f_paraxial + self.principal[1]
        raw = self.ideal_to_raw(np.stack([x, y], axis=-1))
        return ((raw[..., 0] * scale_x).astype(np.float32),
                (raw[..., 1] * scale_y).astype(np.float32))

    def undistort(self, image: np.ndarray, f_out: float | None = None,
                  maps: tuple[np.ndarray, np.ndarray] | None = None
                  ) -> np.ndarray:
        if cv2 is None:
            raise RuntimeError("this needs opencv-python-headless")
        height, width = image.shape[:2]
        if maps is None:
            maps = self.image_maps(width, height, f_out)
        return cv2.remap(image, maps[0], maps[1], cv2.INTER_LANCZOS4,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def load(lens: str, mode: str = "field-angle",
         width: int | None = None, height: int | None = None,
         hfov: float | None = None, calibration: str | None = None,
         f_paraxial: float | None = None) -> Transfer:
    path = Path(calibration) if calibration else CALIB[lens]
    data = json.loads(Path(path).read_text())
    active = ACTIVE[lens]
    return Transfer(data,
                    width if width else active["width"],
                    height if height else active["height"],
                    hfov if hfov else active["hfov"], mode=mode,
                    f_paraxial=f_paraxial)


# ---------------------------------------------------------------------------
# Criterion 1 -- round trip
# ---------------------------------------------------------------------------

def round_trip(transfer: Transfer, samples: int = 20001) -> dict[str, Any]:
    """Distort then undistort, across the full radius including the corner."""
    r = np.linspace(0.0, transfer.r_corner, samples)
    raw = transfer.ideal_to_raw_radius(r)
    back = transfer.raw_to_ideal_radius(raw)
    model = np.abs(back - r)

    # And the same thing through Apple's shipped pair, which is a different
    # question: whether the two tables they ship agree with each other.
    shipped = np.abs(raw * (1.0 + transfer.magnification_shipped_inverse(raw)) - r)

    # A two-dimensional check as well, so an error in the centre handling shows.
    grid_x = np.linspace(0.5, transfer.width - 0.5, 401)
    grid_y = np.linspace(0.5, transfer.height - 0.5, 401)
    u, v = np.meshgrid(grid_x, grid_y)
    points = np.stack([u, v], axis=-1)
    trip = transfer.raw_to_ideal(transfer.ideal_to_raw(points))
    planar = np.linalg.norm(trip - points, axis=-1)
    return {
        "radial_max_px": float(model.max()),
        "radial_max_px_at_corner": float(model[-1]),
        "planar_max_px": float(planar.max()),
        "shipped_pair_max_px": float(shipped.max()),
        "shipped_pair_at_corner_px": float(shipped[-1]),
        "monotone": transfer.monotone,
        "corner_radius_px": transfer.r_corner,
    }


# ---------------------------------------------------------------------------
# Criterion 3 -- the annulus profile
#
# `eval/uw_radial_profile.py` is a script, not a module: importing it runs the
# whole published analysis at import time.  It is also on this task's
# do-not-edit list.  So its source is exec'd up to the point where the script
# body begins, which gives byte-identical `RINGS` and `annulus_scales` without
# touching the file or re-deriving the instrument here.
# ---------------------------------------------------------------------------

PROFILE_SCRIPT = REPO / "eval" / "uw_radial_profile.py"
_SCRIPT_BODY_MARKER = 'print("CONTROL'


def load_profile_module():
    source = PROFILE_SCRIPT.read_text()
    cut = source.find(_SCRIPT_BODY_MARKER)
    if cut < 0:
        raise SystemExit(
            f"{PROFILE_SCRIPT} no longer starts its script body with "
            f"{_SCRIPT_BODY_MARKER!r}; refusing to guess where the definitions "
            "end")
    namespace: dict[str, Any] = {"__name__": "uw_radial_profile_defs",
                                 "__file__": str(PROFILE_SCRIPT)}
    exec(compile(source[:cut], str(PROFILE_SCRIPT), "exec"), namespace)
    for required in ("RINGS", "annulus_scales"):
        if required not in namespace:
            raise SystemExit(f"{PROFILE_SCRIPT} no longer defines {required}")
    return namespace


def annulus_profile(session: Path, transfer: Transfer | None, module,
                    pairs: int = 40, equalize: bool = True,
                    progress: bool = False,
                    wide_transfer: Transfer | None = None) -> dict[str, Any]:
    """The published annulus fit, optionally on undistorted frames.

    ``wide_transfer`` undistorts the *other* arm too.  The brief's version of
    this test corrects only the ultra-wide, which is right only if the wide arm
    is already a pinhole -- the profile is a ratio of the two lenses' maps, so
    whatever the wide arm does is still in it.  Correcting both is the
    physically complete statement: if both tables are right, the profile is
    exactly flat.
    """
    probe = sys.modules["wide_fov_probe"]
    rings = module["RINGS"]
    scales = {ring: [] for ring in rings}
    inliers = {ring: 0 for ring in rings}
    used = 0
    maps = None
    wide_maps = None
    for uw, wide, dt in probe.pair_frames(session, pairs, 1):
        if abs(dt) > probe.MAX_DT_MS:
            continue
        a = cv2.imread(str(session / uw["file"]), cv2.IMREAD_GRAYSCALE)
        b = cv2.imread(str(session / wide["file"]), cv2.IMREAD_GRAYSCALE)
        if a is None or b is None:
            continue
        if transfer is not None:
            if maps is None:
                maps = transfer.image_maps(a.shape[1], a.shape[0])
            a = transfer.undistort(a, maps=maps)
        if wide_transfer is not None:
            if wide_maps is None:
                wide_maps = wide_transfer.image_maps(b.shape[1], b.shape[0])
            b = wide_transfer.undistort(b, maps=wide_maps)
        used += 1
        for ring, (scale, count) in module["annulus_scales"](
                a, b, equalize=equalize).items():
            if scale is not None:
                scales[ring].append(scale)
                inliers[ring] += count
        if progress:
            print(f"    pair {used}", file=sys.stderr)
    values = []
    for ring in rings:
        if scales[ring]:
            values.append({"ring": list(ring),
                           "s": float(np.median(scales[ring])),
                           "frames": len(scales[ring]),
                           "inliers": inliers[ring]})
        else:
            values.append({"ring": list(ring), "s": None, "frames": 0,
                           "inliers": 0})
    good = [item["s"] for item in values if item["s"] is not None]
    result = {"pairs": used, "annuli": values}
    if len(good) >= 2:
        result["peak_to_trough_pct"] = 100.0 * (max(good) - min(good)) / min(good)
        result["first_to_last_pct"] = 100.0 * (good[-1] - good[0]) / good[0]
        result["median_s"] = float(np.median(good))
    return result


def self_downscale_control(session: Path, transfer: Transfer | None, module,
                           factor: float = 0.3200) -> dict[str, Any]:
    """One frame against itself downscaled: no lens difference, must be flat."""
    probe = sys.modules["wide_fov_probe"]
    rows = probe.rows(session / "frames.jsonl")
    image = cv2.imread(str(session / rows[len(rows) // 2]["file"]),
                       cv2.IMREAD_GRAYSCALE)
    if transfer is not None:
        image = transfer.undistort(image)
    shrunk = cv2.resize(image, None, fx=factor, fy=factor,
                        interpolation=cv2.INTER_AREA)
    out = []
    for ring, (scale, count) in module["annulus_scales"](image, shrunk).items():
        out.append({"ring": list(ring), "s": scale, "inliers": count})
    good = [item["s"] for item in out if item["s"] is not None]
    result = {"truth": factor, "annuli": out}
    if len(good) >= 2:
        result["peak_to_trough_pct"] = 100.0 * (max(good) - min(good)) / min(good)
        result["median_s"] = float(np.median(good))
    return result


# ---------------------------------------------------------------------------
# Criterion 2 -- the wide-arm null control
# ---------------------------------------------------------------------------

def _arkit_focal(session: Path) -> float | None:
    """The device's own paraxial focal length, if this session logged poses."""
    path = session / "pose.jsonl"
    if not path.exists():
        return None
    values = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "fx" in record:
                values.append(float(record["fx"]))
    return float(np.median(values)) if values else None


def wide_null(session: Path, frames: int = 60, order: int = 5,
              draws: int = 0, hfov: float | None = None) -> dict[str, Any]:
    """What the transfer predicts for the wide arm, against what it measures."""
    sys.path.insert(0, str(REPO / "tools"))
    import fit_uw_distortion as fit                       # noqa: E402

    paths = fit.select_frames(session, "wide", frames)
    chains, census = fit.gather(paths)
    if chains is None:
        return {"census": census, "error": "no chains"}

    measured = fit.fit_coefficients(chains, 1)
    band = fit.identifiability(chains, measured)
    width, height = chains.width, chains.height
    half = fit.half_diagonal(width, height)

    # An ARKit session hands over `fx` per frame, which is a paraxial focal
    # length and so fixes k with no field-of-view reading at all.  A multi-cam
    # session logs `videoFieldOfView` instead and goes through the field angle.
    arkit_fx = _arkit_focal(session)
    if arkit_fx:
        transfer = load("wide", "focal", width=width, height=height,
                        f_paraxial=arkit_fx)
    else:
        transfer = load("wide", "field-angle", width=width, height=height,
                        hfov=hfov)
    poly = transfer.polynomial(order)
    # Same thing without the field-angle step: the factory table read straight
    # onto this frame by image size, which is what "+15.6 px" comes from.
    naive = load("wide", "width", width=width, height=height)
    naive_poly = naive.polynomial(order)

    def residuals(coefficients: Sequence[float]) -> dict[str, float]:
        values = chains.residual_px(list(coefficients))
        return {"median_px": float(np.median(values)),
                "mean_px": float(np.mean(values)),
                "p90_px": float(np.percentile(values, 90))}

    zero = [0.0] * order
    out = {
        "session": str(session),
        "frame": [width, height],
        "census": census,
        "measured": {
            "coefficients": measured,
            "corner_px": fit.corner_displacement(width, height, measured),
            "identifiability": band,
        },
        "arkit_fx_px": arkit_fx,
        "transferred": {
            "mode": transfer.mode,
            "k": transfer.k,
            "f_paraxial_px": transfer.f_paraxial,
            "corner_px": transfer.corner_displacement(),
            "rms_px": transfer.rms_displacement(),
            "polynomial": poly,
            "monotone_for_chainset": fit.is_monotone(poly),
        },
        "size_scaled_factory": {
            "k": naive.k,
            "corner_px": naive.corner_displacement(),
            "polynomial": naive_poly,
        },
        "straightness": {
            "uncorrected": residuals(zero),
            "measured_fit": residuals(list(measured) + [0.0] * (order - 1)),
            "transferred": residuals(poly),
            "size_scaled_factory": residuals(naive_poly),
            # The sign test.  A radial map that is mostly a uniform scale
            # cannot change straightness at all, so a small improvement is only
            # evidence if reversing the model hurts by a comparable amount.
            "transferred_negated": residuals([-value for value in poly]),
            "transferred_half": residuals([0.5 * value for value in poly]),
            "transferred_double": residuals([2.0 * value for value in poly]),
        },
    }
    if draws:
        out["measured"]["bootstrap"] = fit.bootstrap(chains, measured,
                                                     draws=draws)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_report(lens: str, modes: Sequence[str]) -> None:
    for mode in modes:
        transfer = load(lens, mode)
        report = transfer.report()
        print(f"--- {lens} / {mode}")
        for key, value in report.items():
            if isinstance(value, float):
                print(f"    {key:34s} {value:.6f}")
            else:
                print(f"    {key:34s} {value}")


def main(argv: list[str]) -> int:
    if cv2 is None:
        print("this tool needs opencv:  pip install opencv-python-headless",
              file=sys.stderr)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("report", help="the transfer, per lens and mode")
    p.add_argument("--lens", default="both", choices=("both", "ultrawide", "wide"))
    p.add_argument("--modes", default="field-angle,paraxial,width,identity")

    p = sub.add_parser("roundtrip", help="criterion 1")
    p.add_argument("--lens", default="both", choices=("both", "ultrawide", "wide"))
    p.add_argument("--modes", default="field-angle")

    p = sub.add_parser("wide-null", help="criterion 2")
    p.add_argument("session")
    p.add_argument("--frames", type=int, default=60)
    p.add_argument("--order", type=int, default=5)
    p.add_argument("--draws", type=int, default=0)
    p.add_argument("--hfov", type=float, default=None,
                   help="override the wide arm's logged videoFieldOfView")
    p.add_argument("--out", default=None)

    p = sub.add_parser("profile", help="criterion 3")
    p.add_argument("sessions", nargs="+")
    p.add_argument("--pairs", type=int, default=40)
    p.add_argument("--modes", default="none,field-angle")
    p.add_argument("--no-equalize", action="store_true")
    p.add_argument("--wide-too", action="store_true",
                   help="undistort the wide arm as well; if both tables are "
                        "right the profile is then exactly flat")
    p.add_argument("--control", action="store_true",
                   help="also run the self-downscale control")
    p.add_argument("--out", default=None)

    args = parser.parse_args(argv)
    lenses = (["ultrawide", "wide"] if getattr(args, "lens", None) == "both"
              else [getattr(args, "lens", "ultrawide")])

    if args.command == "report":
        for lens in lenses:
            _print_report(lens, args.modes.split(","))
        return 0

    if args.command == "roundtrip":
        for lens in lenses:
            for mode in args.modes.split(","):
                result = round_trip(load(lens, mode))
                print(f"--- {lens} / {mode}")
                print(json.dumps(result, indent=1))
        return 0

    if args.command == "wide-null":
        report = wide_null(Path(args.session).expanduser().resolve(),
                           frames=args.frames, order=args.order,
                           draws=args.draws, hfov=args.hfov)
        print(json.dumps(report, indent=1))
        if args.out:
            Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
        return 0

    if args.command == "profile":
        sys.path[:0] = [str(REPO / "eval"), str(REPO / "tools")]
        import wide_fov_probe                              # noqa: F401,E402
        module = load_profile_module()
        report: dict[str, Any] = {"sessions": {}}
        for name in args.sessions:
            session = Path(name).expanduser().resolve()
            report["sessions"][session.name] = {}
            for mode in args.modes.split(","):
                transfer = None if mode == "none" else load("ultrawide", mode)
                wide = (load("wide", mode)
                        if args.wide_too and mode != "none" else None)
                result = annulus_profile(session, transfer, module,
                                         pairs=args.pairs,
                                         equalize=not args.no_equalize,
                                         wide_transfer=wide)
                report["sessions"][session.name][mode] = result
                line = "  ".join(
                    f"{item['s']:.4f}" if item["s"] is not None else "  --  "
                    for item in result["annuli"])
                spread = result.get("peak_to_trough_pct")
                print(f"{session.name[-6:]:>8s} {mode:<12s} {line}   "
                      f"peak-to-trough "
                      + (f"{spread:.2f} %" if spread is not None else "n/a")
                      + f"   ({result['pairs']} pairs)")
                sys.stdout.flush()
            if args.control:
                for mode in args.modes.split(","):
                    transfer = None if mode == "none" else load("ultrawide", mode)
                    control = self_downscale_control(session, transfer, module)
                    report["sessions"][session.name].setdefault("control", {})
                    report["sessions"][session.name]["control"][mode] = control
                    line = "  ".join(
                        f"{item['s']:.4f}" if item["s"] is not None else "  --  "
                        for item in control["annuli"])
                    print(f"{session.name[-6:]:>8s} CONTROL {mode:<12s} {line}"
                          f"   truth 0.3200")
                    sys.stdout.flush()
        if args.out:
            Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
