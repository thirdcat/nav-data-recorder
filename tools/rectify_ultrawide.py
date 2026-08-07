#!/usr/bin/env python3
"""Rectify ultra-wide frames to a pinhole camera of a chosen field of view.

The ultra-wide is the only lens on the phone wide enough to cover the OAK-1-W's
96.3°, but it is wide enough that a pinhole model no longer describes it — an
image position stops mapping to a fixed bearing, which is the one property the
capture exists to provide. Apple ships the correction as a radial magnification
lookup table on `AVCameraCalibrationData`; this applies it, and then reprojects
onto whatever pinhole is asked for.

    python3 tools/rectify_ultrawide.py ~/uw_probe/0001 -o ./rectified
    python3 tools/rectify_ultrawide.py ~/uw_probe/0001 -o ./rectified --check-lines

Input is a capture directory from the app's ultra-wide probe: JPEGs plus a
`calibration.json` carrying the intrinsics and the distortion tables. See
docs/ULTRAWIDE.md.

Requires numpy and opencv-python.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    print("this tool needs opencv:  pip install opencv-python-headless", file=sys.stderr)
    raise SystemExit(2)


# ------------------------------------------------------------------ the model

def magnification(radii: np.ndarray, table: np.ndarray, r_max: float) -> np.ndarray:
    """Interpolate Apple's radial magnification table at the given radii.

    The table holds relative radial magnification at `len(table)` linearly
    spaced radii, the first at r=0 and the last at the largest radius the image
    contains — which is the distance from the distortion centre to the farthest
    corner, not half the diagonal, because the centre is not the image centre.
    Past the end Apple's own sample clamps to the final entry rather than
    extrapolating, and so does this.
    """
    n = len(table)
    pos = radii * (n - 1) / r_max
    idx = np.clip(pos.astype(np.int64), 0, n - 2)
    frac = np.clip(pos - idx, 0.0, 1.0)
    inside = radii < r_max
    interpolated = (1.0 - frac) * table[idx] + frac * table[idx + 1]
    return np.where(inside, interpolated, table[-1])


def apply_table(points: np.ndarray, table: np.ndarray, centre: np.ndarray,
                size: tuple[float, float]) -> np.ndarray:
    """Move points radially by the table's magnification, about `centre`.

    With `lensDistortionLookupTable` this takes distorted points to rectified
    ones; with `inverseLensDistortionLookupTable` it goes the other way. Which
    direction you want depends on which end you are iterating over — a warp
    iterates over output pixels and needs the inverse.
    """
    r_max = max_radius(centre, size)
    v = points - centre
    r = np.linalg.norm(v, axis=-1)
    mag = magnification(r, table, r_max)
    return centre + v * (1.0 + mag)[..., None]


def max_radius(centre: np.ndarray, size: tuple[float, float]) -> float:
    dx = max(centre[0], size[0] - centre[0])
    dy = max(centre[1], size[1] - centre[1])
    return float(math.hypot(dx, dy))


# ------------------------------------------------------------------ the warp

class Rectifier:
    """Maps an output pinhole frame back onto the captured ultra-wide frame."""

    def __init__(self, calib: dict, hfov: float, width: int, height: int):
        self.ref_w, self.ref_h = calib["reference_dimensions"]
        k = calib["intrinsics"]
        self.fx, self.fy = float(k["fx"]), float(k["fy"])
        self.cx, self.cy = float(k["cx"]), float(k["cy"])
        self.centre = np.array(calib["lens_distortion_center"], dtype=np.float64)
        self.inverse = np.asarray(calib["inverse_lens_distortion_lookup_table"],
                                  dtype=np.float64)
        self.forward = np.asarray(calib.get("lens_distortion_lookup_table") or [],
                                  dtype=np.float64)

        self.hfov, self.out_w, self.out_h = hfov, width, height
        self.f_out = (width / 2.0) / math.tan(math.radians(hfov) / 2.0)

        # Output pixel -> bearing -> where that bearing lands on the rectified
        # ultra-wide plane -> where the lens actually put it.
        u, v = np.meshgrid(np.arange(width, dtype=np.float64),
                           np.arange(height, dtype=np.float64))
        rays_x = (u + 0.5 - width / 2.0) / self.f_out
        rays_y = (v + 0.5 - height / 2.0) / self.f_out
        rect = np.stack([rays_x * self.fx + self.cx,
                         rays_y * self.fy + self.cy], axis=-1)
        self.rectified_px = rect
        self.source_px = apply_table(rect, self.inverse, self.centre,
                                     (self.ref_w, self.ref_h))

    # -- what the caller needs to know before trusting the result -------------

    @property
    def source_fov(self) -> tuple[float, float]:
        """Field of view of the whole rectified source frame, degrees."""
        h = 2 * math.degrees(math.atan((self.ref_w / 2) / self.fx))
        v = 2 * math.degrees(math.atan((self.ref_h / 2) / self.fy))
        return h, v

    @property
    def required_diagonal(self) -> float:
        """Diagonal FOV the output demands of the source, degrees.

        The horizontal figure understates it: a 96.3° × 64.6° pinhole reaches
        much further into the corners than either axis suggests, and the corners
        are where a source frame runs out first.
        """
        tx = (self.out_w / 2.0) / self.f_out
        ty = (self.out_h / 2.0) / self.f_out
        return 2 * math.degrees(math.atan(math.hypot(tx, ty)))

    @property
    def source_diagonal(self) -> float:
        tx = (self.ref_w / 2.0) / self.fx
        ty = (self.ref_h / 2.0) / self.fy
        return 2 * math.degrees(math.atan(math.hypot(tx, ty)))

    def coverage(self) -> dict:
        """Whether every output pixel has a source pixel behind it."""
        x, y = self.source_px[..., 0], self.source_px[..., 1]
        outside = (x < 0) | (y < 0) | (x > self.ref_w - 1) | (y > self.ref_h - 1)
        return {
            "outside": int(outside.sum()),
            "total": int(outside.size),
            "fraction": float(outside.mean()),
        }

    def distortion_magnitude(self) -> float:
        """Largest distance the correction moves a pixel, in source pixels.

        If this is small the whole exercise is unnecessary; if it is large the
        rectification is doing real work and is worth checking against imagery.
        """
        return float(np.linalg.norm(self.source_px - self.rectified_px, axis=-1).max())

    def round_trip_error(self) -> float | None:
        """Forward table applied to the inverse table's output, in pixels.

        Apple ships both directions; they should compose to the identity. Any
        residual here is either their tables disagreeing or this interpolation
        being wrong, and both are worth knowing before trusting an image.
        """
        if self.forward.size == 0:
            return None
        back = apply_table(self.source_px, self.forward, self.centre,
                           (self.ref_w, self.ref_h))
        return float(np.linalg.norm(back - self.rectified_px, axis=-1).max())

    # -- the warp itself ------------------------------------------------------

    def maps(self, img_w: int, img_h: int) -> tuple[np.ndarray, np.ndarray]:
        """`cv2.remap` coordinates for an image of this size.

        The calibration is quoted against `reference_dimensions`, which need not
        be the size of the frame actually captured, so everything scales here
        rather than at every use.
        """
        sx, sy = img_w / self.ref_w, img_h / self.ref_h
        map_x = (self.source_px[..., 0] * sx).astype(np.float32)
        map_y = (self.source_px[..., 1] * sy).astype(np.float32)
        return map_x, map_y

    def rectify(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        map_x, map_y = self.maps(w, h)
        return cv2.remap(img, map_x, map_y, cv2.INTER_LANCZOS4,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)


# ------------------------------------------------------- straightness check

def straightness(img: np.ndarray, min_length: int = 120) -> dict:
    """How far edge chains in the image bow away from straight, in pixels.

    The honest test of a rectification is imagery, not arithmetic: real straight
    edges — a doorframe, a skirting board, the join between wall and ceiling —
    are straight in the world, so whatever bow survives is the model's error.

    Edge chains are used rather than a line detector, because a line detector
    only finds things that are already straight and would report the curved
    edges as several short straight ones.
    """
    grey = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(grey, (5, 5), 0), 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    residuals = []
    for c in contours:
        pts = c.reshape(-1, 2).astype(np.float64)
        if len(pts) < min_length:
            continue
        centred = pts - pts.mean(axis=0)
        # Direction of greatest spread; the perpendicular offset about it is
        # the bow. A chain that wanders in two directions (a corner, a blob)
        # scores badly and should — it is not a straight edge and the filter
        # below drops it.
        _, s, vt = np.linalg.svd(centred, full_matrices=False)
        if s[0] < 4 * s[1]:
            continue
        along = centred @ vt[0]
        perp = centred @ vt[1]

        # Averaged in bins along the chain, not pointwise. An edge has width,
        # and a contour traces up one side of it and back down the other, so a
        # pointwise residual has a floor of half the edge width whatever the
        # geometry does. Binning cancels the two sides against each other and
        # leaves the bow, which is the thing being measured.
        bins = 24
        edges_ = np.linspace(along.min(), along.max() + 1e-9, bins + 1)
        which = np.clip(np.digitize(along, edges_) - 1, 0, bins - 1)
        counts = np.bincount(which, minlength=bins)
        sums = np.bincount(which, weights=perp, minlength=bins)
        occupied = counts > 0
        if occupied.sum() < bins // 2:
            continue
        means = sums[occupied] / counts[occupied]
        residuals.append(float(np.sqrt((means ** 2).mean())))

    if not residuals:
        return {"chains": 0, "median": None, "p90": None}
    r = np.array(residuals)
    return {
        "chains": len(residuals),
        "median": float(np.median(r)),
        "p90": float(np.percentile(r, 90)),
    }


# ------------------------------------------------------------------ driver

def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture", help="probe capture directory (images + calibration.json)")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--hfov", type=float, default=96.3,
                    help="horizontal field of view of the output, degrees")
    ap.add_argument("--width", type=int, default=848)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--check-lines", action="store_true",
                    help="measure edge-chain straightness before and after")
    args = ap.parse_args(argv)

    path = os.path.join(args.capture, "calibration.json")
    if not os.path.exists(path):
        print(f"! no calibration.json in {args.capture}\n"
              f"  The probe writes one per lens directory after every shot. If "
              f"there are JPEGs here but no calibration.json, they came from a "
              f"build before that was true — re-run the probe.", file=sys.stderr)
        return 2
    with open(path) as f:
        calib = json.load(f)

    if calib.get("calibration") == "unavailable" or "intrinsics" not in calib:
        print(f"! this capture carries no lens calibration, so it cannot be "
              f"rectified.\n  notes: {calib.get('notes')}", file=sys.stderr)
        return 2

    r = Rectifier(calib, args.hfov, args.width, args.height)
    sh, sv = r.source_fov

    print(f"source     {r.ref_w}x{r.ref_h}  fx {r.fx:.1f} fy {r.fy:.1f}  "
          f"H {sh:.1f} V {sv:.1f}  diagonal {r.source_diagonal:.1f}")
    print(f"output     {args.width}x{args.height}  f {r.f_out:.1f}  "
          f"H {args.hfov:.1f} V {2 * math.degrees(math.atan((args.height / 2) / r.f_out)):.1f}  "
          f"diagonal {r.required_diagonal:.1f}")

    margin = r.source_diagonal - r.required_diagonal
    cover = r.coverage()
    if cover["outside"]:
        print(f"! {cover['fraction'] * 100:.2f}% of output pixels fall outside the "
              f"source frame — they will be black. The lens does not reach "
              f"{args.hfov:.1f} degrees at this aspect "
              f"(needs {r.required_diagonal:.1f} diagonal, has {r.source_diagonal:.1f}).")
    else:
        print(f"coverage   every output pixel has source behind it "
              f"({margin:+.1f} degrees of diagonal margin)")

    print(f"distortion largest correction {r.distortion_magnitude():.1f} source px")
    rt = r.round_trip_error()
    if rt is None:
        print("round trip (no forward table shipped — cannot self-check)")
    elif rt > 1.0:
        print(f"! round trip {rt:.2f} px — the two tables do not compose to the "
              f"identity, so one of them is being read wrong")
    else:
        print(f"round trip {rt:.3f} px (forward o inverse = identity)")

    names = sorted(n for n in os.listdir(args.capture)
                   if n.lower().endswith((".jpg", ".jpeg", ".png")))
    if not names:
        print("! no images in the capture directory")
        return 1

    os.makedirs(args.out, exist_ok=True)
    before, after = [], []
    for index, name in enumerate(names):
        img = cv2.imread(os.path.join(args.capture, name))
        if img is None:
            print(f"! could not read {name}")
            continue
        out = r.rectify(img)
        cv2.imwrite(os.path.join(args.out, f"{index:06d}.jpg"), out,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        if args.check_lines:
            before.append(straightness(img))
            after.append(straightness(out))

    print(f"\n{len(names)} frame(s) -> {args.out}")

    if args.check_lines:
        def summarise(rows):
            vals = [x["median"] for x in rows if x["median"] is not None]
            chains = sum(x["chains"] for x in rows)
            return (float(np.median(vals)) if vals else None), chains

        b, bc = summarise(before)
        a, ac = summarise(after)
        print("\nedge-chain straightness (median bow, px; lower is straighter)")
        print(f"  captured   {b if b is None else f'{b:.3f}'}   ({bc} chains)")
        print(f"  rectified  {a if a is None else f'{a:.3f}'}   ({ac} chains)")
        if b is not None and a is not None:
            if a < b:
                print(f"  -> straightened by {(1 - a / b) * 100:.0f}%")
            else:
                print("  ! not straighter after rectification. Either the scene has "
                      "no long straight edges to measure, or the correction is "
                      "being applied wrongly — check a frame by eye before "
                      "believing either number.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
