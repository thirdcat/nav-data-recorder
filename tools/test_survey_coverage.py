#!/usr/bin/env python3
"""Self-tests for the coverage survey.

The survey exists to say one thing — a walk past a surface is worth less than a
walk around it — and a measurement that cannot tell those two apart would say it
anyway, quietly, on every session. So the tests build both arrangements with
known geometry and demand the difference.

    python3 tools/test_survey_coverage.py
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import survey_coverage as sc  # noqa: E402


FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILURES.append(name)


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def test_bins_resolve_at_the_stated_angle() -> None:
    print("direction bins separate at about the stated angle")
    rng = np.random.default_rng(3)
    base = unit(rng.normal(size=(400, 3)))
    b0 = sc.direction_bins(base, 15.0)

    def fraction_split(angle_deg: float) -> float:
        """How often two views this far apart land in different cells."""
        # Rotate each vector by a fixed angle about an axis perpendicular to it.
        axis = unit(np.cross(base, rng.normal(size=(400, 3))))
        a = math.radians(angle_deg)
        turned = (base * math.cos(a) + np.cross(axis, base) * math.sin(a)
                  + axis * (axis * base).sum(1, keepdims=True) * (1 - math.cos(a)))
        return float((sc.direction_bins(unit(turned), 15.0) != b0).mean())

    check("identical views are never split", fraction_split(0.0) == 0.0)
    small, large = fraction_split(2.0), fraction_split(60.0)
    check("a 2-degree change rarely splits", small < 0.25, f"{small:.2f}")
    check("a 60-degree change almost always splits", large > 0.95, f"{large:.2f}")
    check("resolution is monotone", small < fraction_split(15.0) < large,
          f"{small:.2f} {fraction_split(15.0):.2f} {large:.2f}")

    # The equal-area claim: straight down is where a phone aimed at the floor
    # lives, and a naive lat/long grid shatters that neighbourhood.
    down = np.tile(np.array([0.0, 0.0, -1.0]), (200, 1))
    jitter = unit(down + rng.normal(scale=0.02, size=(200, 3)))
    spread = len(np.unique(sc.direction_bins(jitter, 15.0)))
    check("near-identical downward views do not shatter", spread <= 3, f"{spread} cells")


def test_bin_count_matches_the_angle_actually_subtended() -> None:
    """The instrument must report the angle that is there, not a proxy for it.

    Both arrangements have a span that can be worked out on paper, so the test
    is against the arithmetic rather than against the other arrangement. An
    inequality between the two would also pass if the survey were counting
    cameras, or distance travelled, or nothing at all.
    """
    print("bin count tracks the angle a surface is actually seen across")
    target = np.zeros(3)

    def bins_for(centres: np.ndarray) -> int:
        return len(np.unique(sc.direction_bins(unit(centres - target), 15.0)))

    # A surface 2 m away, passed along a 1.5 m stretch of walk: the geometry of
    # this capture. Subtends 2*atan(0.75/2) = 41 degrees.
    half, dist = 0.75, 2.0
    line = np.stack([np.linspace(-half, half, 60), np.zeros(60),
                     np.full(60, dist)], axis=1)
    expected_line = math.degrees(2 * math.atan(half / dist)) / 15.0
    got_line = bins_for(line)
    check("a walk past reports its 41 degrees",
          abs(got_line - expected_line) <= 2.0,
          f"{got_line} cells, arithmetic says about {expected_line:.1f}")

    # The same sixty frames taken around the surface instead: 180 degrees.
    angles = np.linspace(0, math.pi, 60)
    arc = np.stack([dist * np.sin(angles), np.zeros(60), dist * np.cos(angles)], axis=1)
    got_arc = bins_for(arc)
    check("an orbit reports its 180 degrees", abs(got_arc - 180.0 / 15.0) <= 2.0,
          f"{got_arc} cells, arithmetic says 12")

    check("so the orbit is worth several times the walk", got_arc >= 3 * got_line,
          f"walk {got_line}, orbit {got_arc}")

    # Frame count is not what moved: both used sixty cameras.
    dense = np.stack([np.linspace(-half, half, 600), np.zeros(600),
                      np.full(600, dist)], axis=1)
    check("ten times the frames on the same line buys nothing",
          bins_for(dense) - got_line <= 1, f"{got_line} -> {bins_for(dense)}")


def test_which_second_pass_actually_adds_directions() -> None:
    """The result that decides what "walk it twice" has to mean.

    A surface is only ever seen through a window of half the field of view
    either side of the optical axis, and that window sits around the same axis
    however far away the camera is. So a second pass at a different standoff
    retraces the first pass's arc. A second pass at a different *height* tilts
    the arc out of that plane, and every cell it visits is new.
    """
    print("which second pass adds viewing directions")
    fov = math.radians(70.6)

    def sweep(distance: float, height: float, length: float = 2.0, n: int = 40) -> set:
        reach = min(length / 2, distance * math.tan(fov / 2))
        x = np.linspace(-reach, reach, n)
        centres = np.stack([x, np.full(n, height), np.full(n, distance)], axis=1)
        return set(sc.direction_bins(unit(centres), 15.0).tolist())

    base = sweep(1.5, 0.0)
    check("a single pass gets a handful of cells", 4 <= len(base) <= 7, f"{len(base)}")
    check("walking the same line again adds nothing",
          not (sweep(1.5, 0.0) - base))
    check("a pass further from the wall adds nothing",
          not (sweep(2.5, 0.0) - base), f"{sorted(sweep(2.5, 0.0) - base)}")
    check("a pass nearer the wall adds nothing either",
          not (sweep(0.8, 0.0) - base), f"{sorted(sweep(0.8, 0.0) - base)}")

    higher = sweep(1.5, 0.8)
    check("a pass at a different height adds all of its cells",
          len(higher - base) == len(higher), f"{len(higher - base)} of {len(higher)}")
    lower = sweep(1.5, -0.8)
    check("and so does one below", len(lower - base) == len(lower),
          f"{len(lower - base)} of {len(lower)}")
    check("the three together beat any one of them",
          len(base | higher | lower) >= 3 * len(base) - 1,
          f"{len(base | higher | lower)} vs {len(base)}")


def test_path_shape() -> None:
    print("path shape names what the walk was")
    t = np.linspace(0, 1, 50)
    line = np.stack([t * 5, np.zeros(50), np.zeros(50)], axis=1)
    shape = sc.path_shape(line)
    check("a straight walk is very linear", shape["linearity"] > 1e6,
          f"{shape['linearity']}")
    check("and its spread is the length it covered",
          abs(shape["spread_m"] - 5.0) < 1e-6, f"{shape['spread_m']}")

    a = np.linspace(0, 2 * math.pi, 50)
    circle = np.stack([np.cos(a), np.zeros(50), np.sin(a)], axis=1)
    shape = sc.path_shape(circle)
    check("a circle is not", shape["linearity"] < 2.0, f"{shape['linearity']}")
    check("and is flat", shape["flatness"] > 1e6, f"{shape['flatness']}")

    rng = np.random.default_rng(1)
    shape = sc.path_shape(rng.normal(size=(200, 3)))
    check("a blob is neither", shape["linearity"] < 2.0 and shape["flatness"] < 2.0,
          f"{shape['linearity']} {shape['flatness']}")
    check("too few points is reported, not guessed",
          sc.path_shape(np.zeros((2, 3)))["linearity"] is None)

    # The failure this pairing exists to prevent: shape without size. A capture
    # that wandered 40 cm and one that wandered 4 m score the same linearity,
    # and only spread tells them apart.
    small = sc.path_shape(rng.normal(size=(200, 3)) * 0.05)
    big = sc.path_shape(rng.normal(size=(200, 3)) * 0.5)
    check("a tiny wander and a large one have the same shape",
          abs(small["linearity"] - big["linearity"]) < 0.6,
          f"{small['linearity']} vs {big['linearity']}")
    check("but very different spread", big["spread_m"] > 5 * small["spread_m"],
          f"{small['spread_m']} vs {big['spread_m']}")
    check("spread is a distance, not a standard deviation",
          small["spread_m"] > max(small["sigma_m"]),
          f"{small['spread_m']} vs {small['sigma_m']}")


def test_vertical_only_excludes_the_easy_surface() -> None:
    print("--vertical-only drops horizontal surfaces")
    up = sc.WORLD_UP
    check("world up is ARKit's +Y", tuple(up) == (0.0, 1.0, 0.0), f"{up}")

    # The filter as the survey applies it, stated where it can fail.
    floor = np.array([0.0, 1.0, 0.0])
    ceiling = np.array([0.0, -1.0, 0.0])
    wall = np.array([1.0, 0.0, 0.0])
    tilted = np.array([0.0, math.cos(math.radians(50)), math.sin(math.radians(50))])
    horizontal = math.cos(math.radians(45.0))
    keep = lambda n: bool(abs(float(n @ up)) < horizontal)  # noqa: E731
    check("a floor is dropped", not keep(floor))
    check("a ceiling is dropped too", not keep(ceiling))
    check("a wall is kept", keep(wall))
    check("a surface 50 degrees off horizontal is kept", keep(tilted))
    steep = np.array([0.0, math.cos(math.radians(40)), math.sin(math.radians(40))])
    check("one 40 degrees off horizontal is not", not keep(steep))


def test_runs_on_a_generated_session() -> None:
    print("end to end on a synthetic session")
    tools = os.path.dirname(os.path.abspath(__file__))
    with tempfile.TemporaryDirectory() as tmp:
        made = subprocess.run([sys.executable, os.path.join(tools, "make_test_session.py"), tmp],
                              capture_output=True, text=True)
        if made.returncode != 0:
            check("fixture generated", False, made.stderr.strip()[:200])
            return
        sessions = [os.path.join(tmp, d) for d in os.listdir(tmp)
                    if os.path.isdir(os.path.join(tmp, d))]
        check("fixture generated", len(sessions) == 1, f"{sessions}")
        if not sessions:
            return
        result = sc.survey(sessions[0], pix_stride=4)
        if "error" in result:
            check("survey produced numbers", False, result["error"])
            return
        check("survey produced numbers", result["surface_voxels"] > 0,
              f"{result}")
        check("fractions are fractions",
              all(0.0 <= v <= 1.0 for v in result["frac_at_least"].values()),
              f"{result['frac_at_least']}")
        check("median is at least one view",
              result["views_per_voxel"]["median"] >= 1.0)
        check("the table renders", "session" in sc.format_table([result]))
        check("an error row renders too",
              "boom" in sc.format_table([{"id": "abcdef", "error": "boom"}]))


def test_plan_view_summarises_rather_than_recounts() -> None:
    """The mistake `plot_coverage.py` exists not to make.

    Counting directions in the plan view directly would merge a wall's top and
    bottom into one cell and add their viewing directions together — inflating
    every session in proportion to how tall its geometry is, while still looking
    like a picture of the right thing.
    """
    print("the plan view takes a median, it does not re-count")
    import plot_coverage as pc

    # One column, three heights, seen from 1, 5 and 3 directions.
    cells = np.array([[4, 0, 7], [4, 1, 7], [4, 2, 7]])
    counts = np.array([1, 5, 3])
    columns, medians = pc.column_medians(cells, counts)
    check("three voxels collapse to one column", len(columns) == 1, f"{len(columns)}")
    check("the column takes the median, not the sum",
          medians[0] == 3.0, f"{medians[0]} (sum would be 9)")

    # Two columns, kept apart and each summarised on its own.
    cells = np.array([[0, 0, 0], [0, 5, 0], [9, 0, 9]])
    counts = np.array([2, 4, 7])
    columns, medians = pc.column_medians(cells, counts)
    order = np.lexsort((columns[:, 1], columns[:, 0]))
    columns, medians = columns[order], medians[order]
    check("distinct columns stay distinct", len(columns) == 2, f"{len(columns)}")
    check("each summarised separately", list(medians) == [3.0, 7.0], f"{list(medians)}")
    check("column keys are x and z, not x and y",
          columns[1].tolist() == [9, 9], f"{columns[1].tolist()}")


def main() -> int:
    test_bins_resolve_at_the_stated_angle()
    test_bin_count_matches_the_angle_actually_subtended()
    test_which_second_pass_actually_adds_directions()
    test_path_shape()
    test_vertical_only_excludes_the_easy_surface()
    test_plan_view_summarises_rather_than_recounts()
    test_runs_on_a_generated_session()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} failed: {', '.join(FAILURES)}")
        return 1
    print("all passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
