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

    def __init__(self, calib: dict, hfov: float, width: int, height: int,
                 table: str = "lens_distortion_lookup_table"):
        self.ref_w, self.ref_h = calib["reference_dimensions"]
        k = calib["intrinsics"]
        self.fx, self.fy = float(k["fx"]), float(k["fy"])
        self.cx, self.cy = float(k["cx"]), float(k["cy"])
        self.centre = np.array(calib.get("lens_distortion_center")
                               or [self.cx, self.cy], dtype=np.float64)
        # Which table warps rectified -> distorted, the direction remap needs.
        # `lens_distortion_lookup_table` is the lens's own distortion, so that
        # is the default; the other is its inverse. They are near-exact mutual
        # inverses, so a round-trip check cannot tell them apart — only imagery
        # can, which is what `--check-lines` settles by trying both.
        other = ("inverse_lens_distortion_lookup_table"
                 if table == "lens_distortion_lookup_table"
                 else "lens_distortion_lookup_table")
        self.table_name = table
        self.inverse = np.asarray(calib.get(table) or [0.0, 0.0], dtype=np.float64)
        self.forward = np.asarray(calib.get(other) or [], dtype=np.float64)
        # A capture shot with geometric distortion correction on arrives already
        # rectified, so there is no distortion left to undo and Apple ships no
        # tables for it. That is not a broken capture — it is a pinhole already,
        # and everything below still applies with an identity correction. The
        # reprojection onto the requested field of view is the part that matters
        # either way.
        self.already_rectified = not calib.get(table)

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

    def maps(self, img_w: int, img_h: int,
             correct: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """`cv2.remap` coordinates for an image of this size.

        The calibration is quoted against `reference_dimensions`, which need not
        be the size of the frame actually captured, so everything scales here
        rather than at every use.
        """
        px = self.source_px if correct else self.rectified_px
        sx, sy = img_w / self.ref_w, img_h / self.ref_h
        map_x = (px[..., 0] * sx).astype(np.float32)
        map_y = (px[..., 1] * sy).astype(np.float32)
        return map_x, map_y

    def rectify(self, img: np.ndarray, correct: bool = True) -> np.ndarray:
        """Warp to the output pinhole, optionally *without* the correction.

        `correct=False` samples at the ideal pinhole positions instead of the
        distorted ones — the same crop, the same scale, the same output size,
        with the lens model switched off. That is the control the straightness
        check needs: comparing the rectified frame against the original
        4032×3024 capture compares two different resolutions, and bow measured
        in pixels scales with the image, so the numbers are not commensurable.
        """
        h, w = img.shape[:2]
        map_x, map_y = self.maps(w, h, correct=correct)
        return cv2.remap(img, map_x, map_y, cv2.INTER_LANCZOS4,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)


# ------------------------------------------------------- straightness check

def straightness(img: np.ndarray, min_length: int | None = None) -> dict:
    """How far edge chains in the image bow away from straight, in pixels.

    The honest test of a rectification is imagery, not arithmetic: real straight
    edges — a doorframe, a skirting board, the join between wall and ceiling —
    are straight in the world, so whatever bow survives is the model's error.

    Edge chains are used rather than a line detector, because a line detector
    only finds things that are already straight and would report the curved
    edges as several short straight ones.
    """
    grey = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if min_length is None:
        # Scaled to the frame, not fixed. A fixed 120 points keeps thousands of
        # chains in a 12 MP capture and ten in an 848x480 export, so the two
        # would not be measuring the same population of edges.
        min_length = max(30, int(0.08 * max(grey.shape)))
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
        print("! this capture carries no lens calibration, so it cannot be "
              "rectified.", file=sys.stderr)
        print("  Without intrinsics there is no way to know what bearing a pixel "
              "means, which is the whole point. Recover them with "
              "tools/estimate_intrinsics.py if the capture has poses, or "
              "re-shoot in the mode this device delivers calibration in.",
              file=sys.stderr)
        for note in calib.get("notes") or []:
            print(f"  {note}", file=sys.stderr)
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

    if r.already_rectified:
        print("distortion no tables shipped — the capture was corrected in the "
              "camera pipeline, so this is a reprojection onto the requested "
              "pinhole and nothing more")
    else:
        print(f"distortion largest correction {r.distortion_magnitude():.1f} source px")
    rt = r.round_trip_error()
    if r.already_rectified:
        pass
    elif rt is None:
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
    control_dir = args.out.rstrip("/") + "_uncorrected"
    check = r
    if args.check_lines:
        os.makedirs(control_dir, exist_ok=True)
        # Same field of view, sized so one output pixel is one source pixel at
        # the centre — the export's own scale would hide most of the effect.
        native_w = int(round(2 * r.fx * math.tan(math.radians(args.hfov) / 2)))
        native_h = int(round(native_w * args.height / args.width))
        check = Rectifier(calib, args.hfov, native_w, native_h)
        check_alt = Rectifier(calib, args.hfov, native_w, native_h,
                              table="inverse_lens_distortion_lookup_table")
        print(f"line check at {native_w}x{native_h} (native crop scale)")
    before, after, swapped = [], [], []
    for index, name in enumerate(names):
        # IGNORE_ORIENTATION, deliberately. The JPEG carries an EXIF rotation
        # tag and OpenCV honours it by default, which hands back a frame
        # transposed relative to the sensor. The calibration is quoted in sensor
        # coordinates, so an auto-rotated frame gets the two scale factors
        # swapped and comes out stretched rather than rectified.
        img = cv2.imread(os.path.join(args.capture, name),
                         cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
        if img is None:
            print(f"! could not read {name}")
            continue
        ih, iw = img.shape[:2]
        if abs((iw / ih) - (r.ref_w / r.ref_h)) > 0.01:
            print(f"! {name} is {iw}x{ih} but the calibration describes "
                  f"{r.ref_w}x{r.ref_h}. "
                  + ("The frame is transposed — it was stored rotated and the "
                     "rotation was applied on load. Rectifying it would stretch "
                     "rather than correct."
                     if abs((ih / iw) - (r.ref_w / r.ref_h)) < 0.01
                     else "Aspect ratios do not match at all.")
                  + " Skipping.")
            continue
        out = r.rectify(img)
        cv2.imwrite(os.path.join(args.out, f"{index:06d}.jpg"), out,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        if args.check_lines:
            # Measured at the crop's native resolution, not at the export size.
            # This lens bends a pixel by ~27 source px across the exported
            # field; the downscale to 848 wide shrinks that to about 6, which is
            # near the floor of what edge chains can resolve. At native scale
            # the bow is at full magnitude and the comparison has room to say
            # something.
            #
            # The control is the same crop with the lens model switched off, so
            # both sides share size, framing and edge population — the
            # correction is the only difference between them.
            control = check.rectify(img, correct=False)
            cv2.imwrite(os.path.join(control_dir, f"{index:06d}.jpg"), control,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            before.append(straightness(control))
            after.append(straightness(check.rectify(img, correct=True)))
            swapped.append(straightness(check_alt.rectify(img, correct=True)))

    print(f"\n{len(names)} frame(s) -> {args.out}")

    if args.check_lines:
        def summarise(rows):
            vals = [x["median"] for x in rows if x["median"] is not None]
            chains = sum(x["chains"] for x in rows)
            return (float(np.median(vals)) if vals else None), chains

        b, bc = summarise(before)
        a, ac = summarise(after)
        s, sc = summarise(swapped)
        print(f"\nedge-chain straightness at {check.out_w}x{check.out_h} "
              f"(median bow, px; lower is straighter)")
        print(f"  uncorrected      {b if b is None else f'{b:.3f}'}   ({bc} chains)"
              f"   -> {control_dir}")
        print(f"  rectified        {a if a is None else f'{a:.3f}'}   ({ac} chains)")
        print(f"  other table      {s if s is None else f'{s:.3f}'}   ({sc} chains)")
        # The two tables are mutual inverses, so no amount of arithmetic
        # distinguishes them — whichever straightens real edges is the one that
        # maps rectified to distorted, and this is where that gets decided.
        if a is not None and s is not None and s < a:
            print(f"  ! the other table is straighter. Re-run with the tables "
                  f"swapped — the warp is currently applying the correction "
                  f"backwards, which doubles the distortion instead of removing "
                  f"it.")
        if b is None or a is None:
            print("  ! no long straight edges found. This scene cannot answer the "
                  "question — re-shoot something with a doorframe in it.")
        elif min(bc, ac) < 5 or max(bc, ac) > 4 * max(1, min(bc, ac)):
            print(f"  ! the two sides found very different numbers of chains "
                  f"({bc} vs {ac}), so the medians are not comparable. Treat this "
                  f"as no result and look at the two directories by eye.")
        elif a < b:
            print(f"  -> straightened by {(1 - a / b) * 100:.0f}%")
        else:
            print("  ! not straighter after correction. Either this lens is "
                  "already near-pinhole over the exported field, or the "
                  "correction is being applied wrongly — compare the two "
                  "directories by eye before believing the number.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
