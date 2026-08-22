#!/usr/bin/env python3
"""Self-test for the field-angle transfer.

The claim the transfer makes is falsifiable without any capture: build one
synthetic lens, express it as Apple-style tables at two *different* readouts of
the same sensor, and demand that the transfer carries the first onto the second
exactly.  A transfer that only ever sees one format cannot be checked at all,
which is how the horizontal-extent version survived two attempts.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uw_field_angle_transfer import Transfer  # noqa: E402


# One lens: a mustache profile, so the test cannot be passed by a monotone
# approximation that happens to agree at the ends.  The profile folds over past
# about 63 degrees, so every inversion below samples inside MAX_FIELD_DEG.
MAX_FIELD_DEG = 62.0

def raw_radius_mm(theta: np.ndarray, focal_mm: float = 2.2) -> np.ndarray:
    ideal = focal_mm * np.tan(theta)
    rho = ideal / (focal_mm * math.tan(math.radians(58.0)))
    return ideal * (1.0 - 0.030 * rho ** 2 + 0.055 * rho ** 4 - 0.060 * rho ** 6)


def build_calibration(width: int, height: int, pitch_mm: float,
                      focal_mm: float = 2.2, knots: int = 42) -> dict:
    """Apple-style tables for this lens read out at one particular format."""
    fx = focal_mm / pitch_mm
    centre = [width / 2.0, height / 2.0]
    r_max = math.hypot(max(centre[0], width - centre[0]),
                       max(centre[1], height - centre[1]))
    # lens_distortion_lookup_table: ideal -> raw, indexed by the ideal radius.
    ideal = np.linspace(0.0, r_max, knots)
    theta = np.arctan(ideal * pitch_mm / focal_mm)
    raw = raw_radius_mm(theta, focal_mm) / pitch_mm
    forward = np.divide(raw, np.maximum(ideal, 1e-12)) - 1.0
    forward[0] = 0.0
    # inverse: raw -> ideal, indexed by the raw radius.
    sample = np.linspace(0.0, math.radians(MAX_FIELD_DEG), 20001)
    sample_raw = raw_radius_mm(sample, focal_mm) / pitch_mm
    sample_ideal = focal_mm * np.tan(sample) / pitch_mm
    grid = np.linspace(0.0, r_max, knots)
    inverse = np.interp(grid, sample_raw, sample_ideal)
    inverse = np.divide(inverse, np.maximum(grid, 1e-12)) - 1.0
    inverse[0] = 0.0
    return {
        "reference_dimensions": [width, height],
        "intrinsics": {"fx": fx, "fy": fx, "cx": centre[0], "cy": centre[1]},
        "lens_distortion_center": centre,
        "lens_distortion_lookup_table": [float(v) for v in forward],
        "inverse_lens_distortion_lookup_table": [float(v) for v in inverse],
        "pixel_size_mm": pitch_mm,
    }


def whole_frame_hfov(width: int, pitch_mm: float, focal_mm: float = 2.2
                     ) -> float:
    """What the device would report: the true field angle at the frame edge."""
    half_mm = width / 2.0 * pitch_mm
    sample = np.linspace(0.0, math.radians(MAX_FIELD_DEG), 200001)
    raw = raw_radius_mm(sample)
    if not np.all(np.diff(raw) > 0) or half_mm > raw[-1]:
        raise SystemExit("the synthetic lens folds over inside the test range")
    return 2.0 * math.degrees(float(np.interp(half_mm, raw, sample)))


def main() -> int:
    results = []

    def check(name, value, limit, unit="px"):
        ok = value <= limit
        results.append(ok)
        print(f"  {name:44s} {value:10.6f} {unit:3s} "
              f"(<= {limit:g})   {'PASS' if ok else 'FAIL'}")

    # The calibrated readout, and a second readout of the same sensor: 16:9,
    # wider physically and at a coarser pitch, exactly the situation the phone
    # is in.
    pitch_cal = 0.0014
    calibration = build_calibration(4032, 3024, pitch_cal)
    active_w, active_h = 3840, 2160
    pitch_active = 0.0014 * 4032 / 3840 * 1.05
    hfov = whole_frame_hfov(active_w, pitch_active)

    transfer = Transfer(calibration, active_w, active_h, hfov,
                        mode="field-angle")
    truth_k = pitch_cal / pitch_active

    print("field-angle transfer self-test")
    check("k recovered from videoFieldOfView", abs(transfer.k - truth_k),
          2e-4, "")
    check("paraxial focal recovered",
          abs(transfer.f_paraxial - 2.2 / pitch_active), 0.5)

    # The trap, stated as a number: reading the same figure as paraxial is
    # wrong by much more than the tolerance above.
    naive = (active_w / 2.0) / math.tan(math.radians(hfov / 2.0))
    print(f"  (the paraxial reading would give {naive:.1f} px against a true "
          f"{2.2 / pitch_active:.1f})")
    results.append(abs(naive - 2.2 / pitch_active) > 5.0)

    # The transferred table must reproduce the lens at every field angle, not
    # only at the edge it was pinned to.
    theta = np.radians(np.linspace(0.0, 57.0, 400))
    expected = raw_radius_mm(theta) / pitch_active
    ideal_active = (2.2 / pitch_active) * np.tan(theta)
    got = transfer.ideal_to_raw_radius(ideal_active)
    check("raw radius at every field angle",
          float(np.max(np.abs(got - expected))), 1.5)

    trip = transfer.raw_to_ideal_radius(transfer.ideal_to_raw_radius(
        np.linspace(0.0, transfer.r_corner, 5000)))
    check("round trip over the full radius",
          float(np.max(np.abs(trip - np.linspace(0.0, transfer.r_corner, 5000)))),
          0.01)

    points = np.stack(np.meshgrid(np.linspace(0.5, active_w - 0.5, 201),
                                  np.linspace(0.5, active_h - 0.5, 201)),
                      axis=-1)
    planar = np.linalg.norm(transfer.raw_to_ideal(transfer.ideal_to_raw(points))
                            - points, axis=-1)
    check("round trip over the whole frame", float(np.max(planar)), 0.01)

    identity = Transfer(calibration, active_w, active_h, hfov, mode="identity")
    check("identity mode moves nothing",
          float(np.max(np.abs(identity.ideal_to_raw_radius(
              np.linspace(0, 2000, 100)) - np.linspace(0, 2000, 100)))),
          1e-9)

    # A pure resample must be a no-op for the transfer: calibrating the same
    # lens at half the resolution and transferring back has to return the
    # original table's behaviour.
    half = build_calibration(2016, 1512, pitch_cal * 2.0)
    from_half = Transfer(half, active_w, active_h, hfov, mode="field-angle")
    check("same answer from a half-resolution calibration",
          float(np.max(np.abs(from_half.ideal_to_raw_radius(ideal_active)
                              - got))), 2.0)

    # The polynomial handoff has to describe the same map.
    poly = transfer.polynomial(5)
    rho = np.linspace(1e-6, 1.0, 500)
    r = rho * 0.5 * math.hypot(active_w, active_h)
    gain = np.ones_like(rho)
    term = np.ones_like(rho)
    for coefficient in poly:
        term = term * rho ** 2
        gain = gain + coefficient * term
    check("polynomial handoff matches the table",
          float(np.max(np.abs(r * gain - transfer.raw_to_ideal_radius(r)))),
          1.0)

    print(f"\n{'all passed' if all(results) else 'FAILURES'}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
