#!/usr/bin/env python3
"""Self-test for the ultra-wide rectifier.

Builds a lens with a known distortion polynomial, expresses it as Apple's
magnification lookup table, renders a scene of straight lines through it, and
checks that rectification puts them back where the pinhole model says they
belong. That exercises the parts that can be quietly wrong — the table's radius
convention, which direction each table maps, and the reprojection onto a
different field of view.

    python3 tools/test_rectify_ultrawide.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rectify_ultrawide import (Rectifier, apply_table, max_radius,  # noqa: E402
                               straightness)

try:
    import cv2
except ImportError:  # pragma: no cover
    print("this test needs opencv:  pip install opencv-python-headless")
    raise SystemExit(2)


# A plausible ultra-wide: 4224x2376, 106.2 degrees horizontal.
REF_W, REF_H = 4224, 2376
FX = FY = (REF_W / 2) / math.tan(math.radians(106.2) / 2)
CX, CY = REF_W / 2, REF_H / 2
CENTRE = np.array([CX, CY])
K1, K2 = -0.085, 0.012          # barrel, as a wide lens has
TABLE_N = 128


def distort_radius(r: np.ndarray) -> np.ndarray:
    """True lens model: rectified radius -> distorted radius."""
    u = r / FX
    return r * (1.0 + K1 * u ** 2 + K2 * u ** 4)


def build_tables():
    """The same lens as two Apple-style magnification tables.

    `rect_to_dist` is Apple's lensDistortionLookupTable, the direction a warp
    needs; `dist_to_rect` is its inverse. Apple ships both, and the rectifier
    composes them as a self-check, so the test has to produce a genuine pair
    rather than one table used twice.
    """
    r_max = max_radius(CENTRE, (REF_W, REF_H))
    radii = np.linspace(0.0, r_max, TABLE_N)

    # rectified -> distorted: Apple's lensDistortionLookupTable.
    rect_to_dist = np.where(radii > 0, distort_radius(radii) / np.maximum(radii, 1e-9) - 1.0, 0.0)

    # forward: at each *distorted* radius, the magnification back to rectified.
    # Inverting the polynomial numerically rather than analytically keeps the
    # test honest about the model instead of encoding an algebraic shortcut.
    dense = np.linspace(0.0, r_max * 1.6, 20000)
    dense_distorted = distort_radius(dense)
    order = np.argsort(dense_distorted)
    rectified_at = np.interp(radii, dense_distorted[order], dense[order])
    dist_to_rect = np.where(radii > 0, rectified_at / np.maximum(radii, 1e-9) - 1.0, 0.0)
    return rect_to_dist, dist_to_rect


RECT_TO_DIST, DIST_TO_RECT = build_tables()

CALIB = {
    "reference_dimensions": [REF_W, REF_H],
    "intrinsics": {"fx": FX, "fy": FY, "cx": CX, "cy": CY},
    "lens_distortion_center": [CX, CY],
    # Apple's naming, which is the opposite of what this test first assumed.
    # `lensDistortionLookupTable` is the lens's own distortion — rectified to
    # distorted, the direction a remap needs — and the inverse table undoes it.
    # Confirmed against a real capture by sign: the real inverse table is
    # positive at the rim (pushing a barrel image outward, which is the
    # correcting direction) and the forward table negative.
    "lens_distortion_lookup_table": RECT_TO_DIST.tolist(),
    "inverse_lens_distortion_lookup_table": DIST_TO_RECT.tolist(),
}


def render_distorted(width: int, height: int, spacing: int = 160) -> np.ndarray:
    """A grid of world-straight lines, photographed through the lens above.

    Drawn by inverse mapping: for each pixel of the distorted frame, find where
    it came from on the rectified plane and ask whether a grid line is there.
    Drawing the grid rectified and warping it forward would blur the lines and
    make the straightness measurement about the resampler instead of the model.
    """
    sx, sy = width / REF_W, height / REF_H
    u, v = np.meshgrid(np.arange(width, dtype=np.float64),
                       np.arange(height, dtype=np.float64))
    pts = np.stack([u / sx, v / sy], axis=-1)
    rect = apply_table(pts, DIST_TO_RECT, CENTRE, (REF_W, REF_H))

    gx = np.abs((rect[..., 0] % spacing) - spacing / 2) > (spacing / 2 - 3)
    gy = np.abs((rect[..., 1] % spacing) - spacing / 2) > (spacing / 2 - 3)
    img = np.where(gx | gy, 255, 30).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def check(name: str, ok: bool, detail: str) -> bool:
    print(f"  {name:36} {detail:34} {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    print("rectifying a synthetic ultra-wide with known distortion")
    results = []

    r = Rectifier(CALIB, hfov=96.3, width=848, height=480)

    # 1. The two tables must compose to the identity.
    rt = r.round_trip_error()
    results.append(check("table round trip", rt < 0.5, f"{rt:.4f} px"))

    # 2. The correction must be doing real work, or the rest proves nothing.
    mag = r.distortion_magnitude()
    results.append(check("distortion is non-trivial", mag > 20, f"{mag:.1f} px"))

    # 3. Every output pixel must have source behind it at 96.3 degrees.
    cover = r.coverage()
    results.append(check("coverage at 96.3 degrees", cover["outside"] == 0,
                         f"{cover['outside']} pixels outside"))

    # 4. A bearing must land where the requested pinhole says it does. This is
    #    the property the whole exercise is for: the ultra-wide's own image
    #    plane does not have it, and the rectified frame must.
    for angle in (10.0, 30.0, 45.0):
        ray = np.array([math.tan(math.radians(angle)), 0.0])
        expected_u = 848 / 2 + ray[0] * r.f_out
        rect_pt = np.array([[ray[0] * FX + CX, CY]])
        src = apply_table(rect_pt, RECT_TO_DIST, CENTRE, (REF_W, REF_H))
        # Find which output pixel maps to that same source location.
        d = np.linalg.norm(r.source_px - src[0], axis=-1)
        got_v, got_u = np.unravel_index(np.argmin(d), d.shape)
        err = abs(got_u + 0.5 - expected_u)
        results.append(check(f"bearing {angle:.0f} deg lands right", err < 1.5,
                             f"{err:.2f} px"))

    # 5. End to end on imagery: straight lines must come back straight.
    distorted = render_distorted(2112, 1188)
    rectified = r.rectify(distorted)
    before = straightness(distorted)
    after = straightness(rectified, min_length=60)
    ok = (before["median"] is not None and after["median"] is not None
          and after["median"] < 0.5 and after["median"] < before["median"] / 3)
    results.append(check("straight lines come back straight",
                         ok, f"{before['median']:.2f} -> {after['median']:.2f} px"))

    # 6. Asking for more than the lens has must be caught, not silently cropped.
    wide = Rectifier(CALIB, hfov=140.0, width=848, height=480)
    results.append(check("over-wide request reports gaps",
                         wide.coverage()["outside"] > 0,
                         f"{wide.coverage()['fraction'] * 100:.1f}% outside"))

    # 7. A capture that arrives already rectified ships no tables. The
    #    reprojection onto the requested pinhole still has to be exact — that
    #    is the whole job in that mode.
    bare = {k: v for k, v in CALIB.items() if "lookup_table" not in k}
    p = Rectifier(bare, hfov=96.3, width=848, height=480)
    results.append(check("no tables: recognised", p.already_rectified, "flagged"))
    results.append(check("no tables: correction is identity",
                         p.distortion_magnitude() < 1e-6,
                         f"{p.distortion_magnitude():.2e} px"))
    errs = []
    for angle in (10.0, 30.0, 45.0):
        want_u = 848 / 2 + math.tan(math.radians(angle)) * p.f_out
        target = np.array([math.tan(math.radians(angle)) * FX + CX, CY])
        d = np.linalg.norm(p.source_px - target, axis=-1)
        gv, gu = np.unravel_index(np.argmin(d), d.shape)
        errs.append(abs(gu + 0.5 - want_u))
    results.append(check("no tables: bearings still land right",
                         max(errs) < 1.5, f"max {max(errs):.2f} px"))

    print()
    if all(results):
        print("all passed")
        return 0
    print("FAILURES — the rectification is wrong")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
