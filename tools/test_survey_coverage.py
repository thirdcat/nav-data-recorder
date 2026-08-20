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
import shutil
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


def test_merging_a_session_with_itself_adds_nothing() -> None:
    """The invariant that keeps merging honest.

    Two recordings of one room contribute two sets of votes into one grid. If
    the fold failed to collapse a direction that both saw, every merge would
    score better than it should — and it would look like a result rather than a
    bug, because merging is *supposed* to raise the number.

    A session merged with an exact copy of itself is the case where the right
    answer is known: not one extra viewing direction anywhere.
    """
    print("a session merged with itself gains nothing")
    tools = os.path.dirname(os.path.abspath(__file__))
    with tempfile.TemporaryDirectory() as tmp:
        made = subprocess.run([sys.executable, os.path.join(tools, "make_test_session.py"), tmp],
                              capture_output=True, text=True)
        sessions = [os.path.join(tmp, d) for d in os.listdir(tmp)
                    if os.path.isdir(os.path.join(tmp, d))]
        if made.returncode != 0 or not sessions:
            check("fixture generated", False, made.stderr.strip()[:200])
            return
        session = sessions[0]

        alone_cells, alone_counts, _, _ = sc.voxel_view_counts(session, pix_stride=6)
        doubled_cells, doubled_counts, _, _ = sc.merged_view_counts(
            [session, session], {}, pix_stride=6)

        check("the same voxels come back", len(alone_counts) == len(doubled_counts),
              f"{len(alone_counts)} vs {len(doubled_counts)}")
        check("and not one extra viewing direction",
              float(doubled_counts.mean()) == float(alone_counts.mean()),
              f"{alone_counts.mean():.4f} -> {doubled_counts.mean():.4f}")
        check("three-view coverage is unchanged",
              float((doubled_counts >= 3).mean()) == float((alone_counts >= 3).mean()),
              f"{(alone_counts >= 3).mean():.4f} -> {(doubled_counts >= 3).mean():.4f}")

        # And a copy moved somewhere the original never was must add surface
        # without inventing angles on the surface that was already there. The
        # transform list, rather than a dict, is what lets one copy stay put:
        # both entries are the same path and a dict could only place them alike.
        #
        # The tolerances are not slack, they are the honest precision available.
        # **A voxel partition is not translation-invariant in floating point**,
        # even by a whole number of cells: adding 64 to a coordinate near 1 m
        # costs six bits of its mantissa, and the points that were sitting on a
        # cell boundary land on the other side of it. Measured here at about
        # 5 % of voxels, and at 7 % for a 40 m shift on a 5 cm grid where the
        # cell size is not a binary fraction either.
        shifted = np.eye(4)
        shifted[0, 3] = 64.0
        far_cells, far_counts, _, _ = sc.merged_view_counts(
            [session, session], [None, shifted], voxel=0.0625, pix_stride=6)
        base_cells, base_counts, _, _ = sc.voxel_view_counts(
            session, voxel=0.0625, pix_stride=6)
        ratio = len(far_counts) / len(base_counts)
        check("a copy placed elsewhere roughly doubles the surface",
              1.9 < ratio < 2.15, f"{len(base_counts)} -> {len(far_counts)} = {ratio:.2f}x")
        drift = abs(float(far_counts.mean()) - float(base_counts.mean()))
        check("and leaves views per voxel where they were",
              drift < 0.1 * float(base_counts.mean()),
              f"{base_counts.mean():.4f} -> {far_counts.mean():.4f}")

        # The failure this whole test exists for: if the fold stopped collapsing
        # a direction two copies shared, views per voxel would head for double.
        check("nowhere near double, which is what a broken fold would give",
              float(far_counts.mean()) < 1.5 * float(base_counts.mean()),
              f"{far_counts.mean():.4f} against {base_counts.mean():.4f}")


def _fixture(tmp: str) -> str | None:
    """A synthetic session on disk, or None if the generator failed."""
    tools = os.path.dirname(os.path.abspath(__file__))
    made = subprocess.run([sys.executable, os.path.join(tools, "make_test_session.py"), tmp],
                          capture_output=True, text=True)
    sessions = [os.path.join(tmp, d) for d in os.listdir(tmp)
                if os.path.isdir(os.path.join(tmp, d))]
    if made.returncode != 0 or len(sessions) != 1:
        check("fixture generated", False, made.stderr.strip()[:200])
        return None
    return sessions[0]


def _trajectory(session: str, path: str, *, transform=None, scale=1.0,
                reference: bool = True) -> None:
    """Write a `--poses` npz holding this session's own trajectory, moved.

    With no transform and no scale it is the identity case: the file says
    exactly what the session already says, so the survey must not move.
    """
    from export_3dgs import camera_to_world
    from read_session import Session

    rows = [r for r in Session(session).posed_images() if r.get("depth")]
    frames = np.array([r["frame"] for r in rows], dtype=np.int64)
    ref = np.array([camera_to_world(r["pose"]) for r in rows])
    est = ref.copy()
    if transform is not None:
        est = transform @ est
    if scale != 1.0:
        est[:, :3, 3] *= scale
    out = {"estimate": est, "frame": frames}
    if reference:
        out["reference"] = ref
    np.savez(path, **out)


def test_poses_npz_round_trips() -> None:
    """The identity case, which is the only one whose answer is known.

    `docs/3DGS.md` records an ARKit-agreement check that hid a solver defect, so
    agreement on its own is not evidence. What makes this worth running is that
    the expected answer is *exact*: a file holding the session's own poses must
    reproduce the session's own survey to the last voxel. Anything short of that
    is the basis change being wrong somewhere and cancelling somewhere else.
    """
    print("a trajectory npz holding the session's own poses changes nothing")
    with tempfile.TemporaryDirectory() as tmp:
        session = _fixture(tmp)
        if session is None:
            return
        npz = os.path.join(tmp, "same.npz")
        _trajectory(session, npz)

        base = sc.survey(session, pix_stride=4)
        same = sc.survey(session, pix_stride=4, poses=npz)
        check("the reference check ran and passed",
              same.get("pose_check_m") is not None
              and same["pose_check_m"] < 1e-9, f"{same.get('pose_check_m')}")
        check("the same frames survive", base["frames"] == same["frames"],
              f"{base['frames']} vs {same['frames']}")
        check("the same surface comes back",
              base["surface_voxels"] == same["surface_voxels"],
              f"{base['surface_voxels']} vs {same['surface_voxels']}")
        check("every coverage fraction is identical",
              base["frac_at_least"] == same["frac_at_least"],
              f"{base['frac_at_least']} vs {same['frac_at_least']}")
        check("and so is the range", base["median_range_m"] == same["median_range_m"],
              f"{base['median_range_m']} vs {same['median_range_m']}")


def _exact_counts(session: str, **kwargs) -> np.ndarray:
    """Distinct directions per voxel, folded on the voxel index itself.

    `fold_votes` folds on a hash of that index instead, which is what makes it
    cheap. This is the same count without the hash, and the only reason it is
    written twice is to measure the difference — see the test below.
    """
    cells, bins, _, _ = sc.gather_votes(session, **kwargs)
    pairs = np.unique(np.concatenate([cells, bins[:, None]], axis=1), axis=0)
    _, counts = np.unique(pairs[:, :3], axis=0, return_counts=True)
    return counts


def test_poses_npz_is_rigid_invariant_but_not_scale_invariant() -> None:
    """Coverage is a property of the geometry, not of the frame it is in.

    The pair matters more than either half. Invariance alone would also be what
    a substitution that silently ignored the file returned, so the scaled arm is
    run beside it: the trajectory has to be able to move the numbers before its
    failing to move them means anything.

    **Why the tolerance is a pp and not zero, which is a fact about the survey
    and not about the substitution.** `fold_votes` identifies a voxel by
    `(x·73856093) ^ (y·19349663) ^ (z·83492791)`, an XOR with no mixing step, so
    two different voxels can share a key — and when they do their direction sets
    are unioned, which can only push coverage *up*. Moving the cloud changes
    which cells collide. Folding on the voxel index itself instead:

        fixture      189 898 voxels, 8.11 % at >=2      hashed: 166 212, 19.00 %
        cb4586 vert   24 108 voxels, 20.57 % at >=3     hashed:  24 033, 20.76 %
        5bd1ed vert   47 401 voxels, 13.64 %            hashed:  46 658, 14.55 %
        1868dd vert   52 243 voxels,  9.55 %            hashed:  51 896,  9.84 %

    On real sessions it is 0.2 to 0.9 pp, always upward, and does not move the
    published table at its printed precision. On this fixture it is 2.3x,
    because a synthetic room of exactly planar surfaces makes voxel indices
    regular and an unmixed XOR of linear multiples collide systematically. The
    fixture is the worst case, not the typical one. Both arms are asserted here
    so a regression cannot hide in either.
    """
    print("moving the whole world leaves coverage alone; rescaling it does not")
    with tempfile.TemporaryDirectory() as tmp:
        session = _fixture(tmp)
        if session is None:
            return
        angle = math.radians(37.0)
        moved = np.eye(4)
        moved[:3, :3] = np.array([[math.cos(angle), 0.0, math.sin(angle)],
                                  [0.0, 1.0, 0.0],
                                  [-math.sin(angle), 0.0, math.cos(angle)]])
        moved[:3, 3] = (0.05, -0.10, 0.15)

        rigid = os.path.join(tmp, "rigid.npz")
        scaled = os.path.join(tmp, "scaled.npz")
        _trajectory(session, rigid, transform=moved)
        _trajectory(session, scaled, scale=1.30)

        base = sc.survey(session, pix_stride=4)
        turned = sc.survey(session, pix_stride=4, poses=rigid)
        bigger = sc.survey(session, pix_stride=4, poses=scaled)

        drift = max(abs(base["frac_at_least"][k] - turned["frac_at_least"][k])
                    for k in base["frac_at_least"])
        check("a rotated, translated world scores the same to within a point",
              drift < 0.01, f"{drift:.4f}: {base['frac_at_least']} vs "
              f"{turned['frac_at_least']}")

        # The residual above is the hash's, so on the unhashed fold it vanishes.
        exact_base = _exact_counts(session, pix_stride=4)
        exact_turned = _exact_counts(session, pix_stride=4, transform=moved)
        exact_drift = max(abs(float((exact_base >= k).mean())
                              - float((exact_turned >= k).mean()))
                          for k in (2, 3, 5))
        check("and to a thousandth once the voxel index is not hashed",
              exact_drift < 0.001, f"{exact_drift:.5f}")
        check("the hash is the one that inflates, never deflates",
              (base["frac_at_least"]["3"]
               >= float((exact_base >= 3).mean()) - 1e-9),
              f"hashed {base['frac_at_least']['3']} vs exact "
              f"{float((exact_base >= 3).mean()):.4f}")

        check("the path is the same size in it",
              abs(base["spread_m"] - turned["spread_m"]) < 1e-6,
              f"{base['spread_m']} vs {turned['spread_m']}")
        check("a 1.3x trajectory does move the numbers",
              bigger["frac_at_least"] != base["frac_at_least"],
              f"{base['frac_at_least']} vs {bigger['frac_at_least']}")
        check("and it moves the path's size by about 1.3x",
              abs(bigger["spread_m"] / base["spread_m"] - 1.30) < 0.05,
              f"{bigger['spread_m'] / base['spread_m']:.3f}")


def test_vertical_only_refuses_a_trajectory_with_no_up() -> None:
    """The filter needs a world whose +Y is up, and only `reference` says so.

    `pi3traj/<id>_*_local.npz` carries none — its convention string says "no
    reference — this session has no tracker". Run `--vertical-only` against one
    and the wall/floor split happens about whatever axis that solver's gauge
    happened to land on, and prints a percentage either way.
    """
    print("--vertical-only refuses a trajectory whose up is not established")
    with tempfile.TemporaryDirectory() as tmp:
        session = _fixture(tmp)
        if session is None:
            return
        blind = os.path.join(tmp, "no_reference.npz")
        _trajectory(session, blind, reference=False)

        allowed = sc.survey(session, pix_stride=4, poses=blind)
        check("without --vertical-only it runs", "error" not in allowed,
              f"{allowed.get('error')}")
        check("and says the conversion was never checked",
              allowed.get("pose_check_m", "missing") is None,
              f"{allowed.get('pose_check_m', 'missing')}")

        refused = sc.survey(session, pix_stride=4, poses=blind, vertical_only=True)
        check("with it, the run is refused rather than filtered",
              "error" in refused and "reference" in refused["error"],
              f"{refused.get('error', refused)}")


def test_the_survey_is_blind_to_which_lens_took_the_picture() -> None:
    """The lever test. It is written to pass, and its passing is the finding.

    A measurement of what a wider lens buys must be a function of the lens. This
    one is not: `_gather` opens no photograph, and the only thing it takes from
    the camera is `fx * depth_width / image_width` — the depth grid's angular
    scale. Declare the same camera at another resolution and nothing moves.

    The second half is what happens if the mismatch is papered over instead. On
    a multi-cam session there is one depth stream, registered to the wide
    camera; hand its grid the ultra-wide's intrinsics and the back-projection
    spreads every ray by f_depth / f_declared = 226.91 / 120.13 = 1.89, which
    does not add coverage, it inflates the room. A wider frame with no wider
    depth behind it is not more of the scene, it is the same scene drawn bigger.

    A doubled lens is asserted here rather than 1.89x because the fixture makes
    the arithmetic checkable: median *range* grows by 1.35, not by 2, since a
    ray at angle theta has range z/cos(theta) and the rays near the axis barely
    move. The pre-registered form of this check demanded 1.4x and was simply
    wrong about that; the surviving claims are the ones the geometry forces —
    the range grows, the surface spreads over half again as many voxels, and
    views per voxel falls.
    """
    print("the survey reads the depth camera, not the lens")
    with tempfile.TemporaryDirectory() as tmp:
        session = _fixture(tmp)
        if session is None:
            return
        import json as _json
        from read_session import Session

        def variant(name: str, width_scale: float, focal_scale: float) -> str:
            """The same session, re-declaring the camera that took the images."""
            out = os.path.join(tmp, name)
            shutil.copytree(session, out)
            for stream, keys in (("frames", ("width", "height")),
                                 ("pose", ("fx", "fy", "cx", "cy"))):
                path = os.path.join(out, f"{stream}.jsonl")
                scale = width_scale if stream == "frames" else focal_scale
                rows = [_json.loads(l) for l in open(path) if l.strip()]
                for r in rows:
                    for k in keys:
                        if k in r:
                            r[k] = type(r[k])(r[k] * scale)
                with open(path, "w") as fh:
                    for r in rows:
                        fh.write(_json.dumps(r, sort_keys=True) + "\n")
            return out

        base = sc.survey(session, pix_stride=4)
        # Same camera, twice the pixels: fx and the image width scale together.
        same_lens = sc.survey(variant("same_lens", 2.0, 2.0), pix_stride=4)
        # A genuinely wider lens declared over the same depth grid.
        wider = sc.survey(variant("wider_lens", 2.0, 1.0), pix_stride=4)

        check("the same camera at another resolution scores identically",
              base["frac_at_least"] == same_lens["frac_at_least"]
              and base["surface_voxels"] == same_lens["surface_voxels"],
              f"{base['frac_at_least']} vs {same_lens['frac_at_least']}")
        check("so the lens's field of view is not an input at all",
              base["median_range_m"] == same_lens["median_range_m"],
              f"{base['median_range_m']} vs {same_lens['median_range_m']}")
        check("declaring a wider lens over the same depth inflates the room",
              wider["median_range_m"] > 1.25 * base["median_range_m"],
              f"{base['median_range_m']} -> {wider['median_range_m']}")
        check("which spreads the surface rather than covering more of it",
              wider["surface_voxels"] > 1.4 * base["surface_voxels"]
              and wider["views_per_voxel"]["mean"] < base["views_per_voxel"]["mean"],
              f"{base['surface_voxels']} -> {wider['surface_voxels']}, "
              f"{base['views_per_voxel']['mean']} -> "
              f"{wider['views_per_voxel']['mean']}")
        check("and coverage falls, so a wider declaration never buys views",
              wider["frac_at_least"]["3"] < base["frac_at_least"]["3"],
              f"{base['frac_at_least']['3']} -> {wider['frac_at_least']['3']}")


def main() -> int:
    test_bins_resolve_at_the_stated_angle()
    test_bin_count_matches_the_angle_actually_subtended()
    test_which_second_pass_actually_adds_directions()
    test_path_shape()
    test_vertical_only_excludes_the_easy_surface()
    test_plan_view_summarises_rather_than_recounts()
    test_runs_on_a_generated_session()
    test_merging_a_session_with_itself_adds_nothing()
    test_poses_npz_round_trips()
    test_poses_npz_is_rigid_invariant_but_not_scale_invariant()
    test_vertical_only_refuses_a_trajectory_with_no_up()
    test_the_survey_is_blind_to_which_lens_took_the_picture()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} failed: {', '.join(FAILURES)}")
        return 1
    print("all passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
