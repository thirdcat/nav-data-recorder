#!/usr/bin/env python3
"""Self-tests for map-to-map alignment.

A registration that is wrong still returns a transform, and the transform still
looks like a transform. The only way to know is to build a room, move it by a
number you chose, and demand that number back — so that is what these do, and
they do it for rotations large enough that a local method would never find them.

    python3 tools/test_align_sessions.py
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import align_sessions as al  # noqa: E402


FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILURES.append(name)


def room(seed: int = 0, width: float = 4.0, depth: float = 6.0,
         height: float = 2.5, spacing: float = 0.04) -> tuple[np.ndarray, np.ndarray]:
    """Four walls, a floor and one interior partition, with outward normals.

    The partition matters: two bare rectangles are symmetric under a 180-degree
    yaw, and a test that cannot tell 0 from 180 would pass on a coin flip.
    """
    rng = np.random.default_rng(seed)
    points, normals = [], []

    def wall(p0, p1, n):
        length = float(np.linalg.norm(np.array(p1) - np.array(p0)))
        steps = max(2, int(length / spacing))
        for u in np.linspace(0, 1, steps):
            base = np.array(p0) + u * (np.array(p1) - np.array(p0))
            for y in np.arange(0.05, height, spacing):
                points.append([base[0], y, base[1]])
                normals.append(n)

    wall((0, 0), (width, 0), (0.0, 0.0, 1.0))
    wall((0, depth), (width, depth), (0.0, 0.0, -1.0))
    wall((0, 0), (0, depth), (1.0, 0.0, 0.0))
    wall((width, 0), (width, depth), (-1.0, 0.0, 0.0))
    wall((1.0, 1.5), (1.0, 3.5), (1.0, 0.0, 0.0))          # the asymmetry

    floor_x, floor_z = np.meshgrid(np.arange(0, width, spacing * 3),
                                   np.arange(0, depth, spacing * 3))
    for x, z in zip(floor_x.ravel(), floor_z.ravel()):
        points.append([x, 0.0, z])
        normals.append([0.0, 1.0, 0.0])

    p = np.asarray(points, dtype=float)
    n = np.asarray(normals, dtype=float)
    p += rng.normal(scale=0.002, size=p.shape)              # sensor noise
    return p, n


def as_cloud(points: np.ndarray, normals: np.ndarray, name: str) -> dict:
    upright = np.abs(normals @ al.WORLD_UP) < math.cos(math.radians(45.0))
    return {"id": name, "points": points, "normals": normals,
            "vertical": upright, "counts": np.ones(len(points)),
            "centres": np.zeros((1, 3))}


def transform_of(yaw_deg: float, translation) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = al.yaw_matrix(math.radians(yaw_deg))
    T[:3, 3] = translation
    return T


def apply(T: np.ndarray, points: np.ndarray, normals: np.ndarray):
    return points @ T[:3, :3].T + T[:3, 3], normals @ T[:3, :3].T


def test_nearest_voxel_against_brute_force() -> None:
    print("the voxel index finds what a brute-force scan finds")
    rng = np.random.default_rng(1)
    target = rng.uniform(-2, 2, size=(4000, 3))
    query = rng.uniform(-2, 2, size=(300, 3))
    index = al.NearestVoxel(target, 0.10)
    idx, dist = index.query(query)

    brute = np.linalg.norm(target[None, :, :] - query[:, None, :], axis=2)
    true_dist = brute.min(axis=1)

    # The grid only guarantees the true nearest inside one cell; beyond that it
    # may miss, so the claim under test is exactly that guarantee.
    close = true_dist < 0.10
    agree = np.isclose(dist[close], true_dist[close], atol=1e-9)
    check("exact within one cell", bool(agree.all()),
          f"{int((~agree).sum())} of {int(close.sum())} disagree")
    check("never reports closer than the truth",
          bool(np.all(dist[idx >= 0] >= true_dist[idx >= 0] - 1e-9)))
    far = ~close
    check("misses are reported as misses, not as wrong answers",
          bool(np.all(np.isinf(dist[far]) | (dist[far] >= true_dist[far] - 1e-9))))


def test_nearest_voxel_when_cells_hold_many_points() -> None:
    """The case the test above cannot see, and the one the real clouds are in.

    4 000 points across 4 m at a 10 cm voxel leaves 0.06 points per cell, so
    almost every occupied cell holds exactly one point and a lookup that scans
    one point per cell is indistinguishable from a correct one. Every real call
    is the opposite: `icp` queries a 5 cm grid against clouds fused at 1-2 cm,
    which is tens of points per cell. Pin the density so the difference shows.
    """
    print("the voxel index scans every point in a cell, not one of them")
    rng = np.random.default_rng(7)
    target = rng.uniform(0, 1, size=(20000, 3))       # 20 000 points in a metre
    index = al.NearestVoxel(target, 0.10)             # ~20 per cell
    check("the fixture really is dense", index.max_per_cell > 5,
          f"max {index.max_per_cell} points per cell")

    # The sharpest form: a point's distance to itself is zero, and no grid size
    # may change that. The old lookup returned 4.3 cm here.
    _, self_dist = index.query(target[::17])
    check("a point finds itself at zero distance",
          bool(np.nanmax(self_dist) < 1e-9), f"worst {np.nanmax(self_dist):.4f} m")

    # And a known displacement comes back as itself, not as the cell size.
    for shift in (0.002, 0.02):
        _, moved = index.query(target[::17] + np.array([shift, 0.0, 0.0]))
        moved = moved[np.isfinite(moved)]
        check(f"a {shift * 100:.1f} cm shift reads as at most {shift * 100:.1f} cm",
              bool(np.percentile(moved, 90) <= shift + 1e-9),
              f"p90 {np.percentile(moved, 90) * 100:.2f} cm")

    query = rng.uniform(0, 1, size=(400, 3))
    idx, dist = index.query(query)
    brute = np.linalg.norm(target[None, :, :] - query[:, None, :], axis=2)
    truth = brute.min(axis=1)
    close = truth < 0.10
    check("exact against brute force when cells are crowded",
          bool(np.allclose(dist[close], truth[close], atol=1e-9)),
          f"worst gap {np.abs(dist[close] - truth[close]).max():.2e} m")


def test_floor_height() -> None:
    print("the floor is found as the busiest slab")
    rng = np.random.default_rng(2)
    floor = np.stack([rng.uniform(0, 4, 5000), np.full(5000, 1.25),
                      rng.uniform(0, 4, 5000)], axis=1)
    walls = np.stack([rng.uniform(0, 4, 500), rng.uniform(1.25, 4.0, 500),
                      rng.uniform(0, 4, 500)], axis=1)
    got = al.floor_height(np.concatenate([floor, walls]))
    check("within one bin of the truth", abs(got - 1.25) < 0.06, f"{got:.3f}")


def test_recovers_a_known_transform() -> None:
    print("a room moved by a known amount comes back")
    points, normals = room()
    fixed = as_cloud(points, normals, "fixed")

    for yaw, translation in ((23.0, (1.2, 0.0, -0.7)),
                             (135.0, (-2.0, 0.0, 3.1)),
                             (270.0, (0.5, 0.0, 0.5))):
        T = transform_of(yaw, translation)
        # The *source* is the room seen in another frame, so it carries the
        # inverse: aligning it should reproduce T.
        inverse = np.linalg.inv(T)
        moved_p, moved_n = apply(inverse, points, normals)
        moving = as_cloud(moved_p, moved_n, "moving")

        result = al.align_clouds(fixed, moving)
        got = result["transform"]
        centre_error = float(np.linalg.norm(got[:3, 3] - T[:3, 3]))
        angle = math.degrees(math.acos(max(-1.0, min(1.0,
            (np.trace(got[:3, :3].T @ T[:3, :3]) - 1) / 2))))
        check(f"yaw {yaw:.0f}° recovered to under 1°", angle < 1.0,
              f"off by {angle:.2f}°")
        check(f"yaw {yaw:.0f}° translation to under 5 cm", centre_error < 0.05,
              f"off by {100 * centre_error:.1f} cm")
        check(f"yaw {yaw:.0f}° fitness above 90% at 5 cm",
              result["fitness"]["5cm"] > 0.90, f"{result['fitness']['5cm']:.2f}")


def test_a_different_room_does_not_pass() -> None:
    """The control arm, on geometry where the right answer is known to be none."""
    print("an unrelated room fails to align")
    points, normals = room(seed=3, width=4.0, depth=6.0)
    fixed = as_cloud(points, normals, "fixed")
    other_p, other_n = room(seed=4, width=9.0, depth=2.0, height=3.2)
    other = as_cloud(other_p, other_n, "other")

    same = al.align_clouds(fixed, as_cloud(*apply(
        np.linalg.inv(transform_of(40.0, (1.0, 0.0, 1.0))), points, normals), "same"))
    wrong = al.align_clouds(fixed, other)

    check("the true pair reaches high fitness", same["fitness"]["5cm"] > 0.90,
          f"{same['fitness']['5cm']:.2f}")
    check("the wrong pair does not", wrong["fitness"]["5cm"] < same["fitness"]["5cm"] / 2,
          f"true {same['fitness']['5cm']:.2f} vs wrong {wrong['fitness']['5cm']:.2f}")


def test_four_degrees_of_freedom_holds_the_vertical() -> None:
    print("4-DOF leaves gravity alone")
    points, normals = room(seed=5)
    fixed = as_cloud(points, normals, "fixed")

    # Tip the source slightly: 6-DOF should take the tilt out, 4-DOF must not.
    tilt = np.eye(4)
    tilt[:3, :3] = al.small_rotation(np.array([math.radians(3.0), 0.0, 0.0]))
    moved_p, moved_n = apply(tilt, points, normals)
    moving = as_cloud(moved_p, moved_n, "moving")

    six = al.align_clouds(fixed, moving, dof=6)
    four = al.align_clouds(fixed, moving, dof=4)
    check("6-DOF removes most of a 3° tilt", six["tilt_deg"] > 1.0,
          f"{six['tilt_deg']:.2f}° of correction")
    check("4-DOF applies no tilt at all", four["tilt_deg"] < 1e-6,
          f"{four['tilt_deg']:.4f}°")
    check("and therefore fits worse when the tilt is real",
          four["fitness"]["5cm"] <= six["fitness"]["5cm"] + 1e-9,
          f"4-DOF {four['fitness']['5cm']:.2f} vs 6-DOF {six['fitness']['5cm']:.2f}")


def test_fitness_counts_every_source_point() -> None:
    """The property that made fitness the criterion instead of the residual."""
    print("fitness charges for points that found no home")
    points, normals = room(seed=6)
    fixed = as_cloud(points, normals, "fixed")

    # Half the source is the room; half is somewhere else entirely. A residual
    # conditioned on matching would ignore the half that cannot match.
    stray = points.copy()
    stray[:, 0] += 50.0
    mixed = as_cloud(np.concatenate([points, stray]),
                     np.concatenate([normals, normals]), "mixed")
    result = al.align_clouds(fixed, mixed)
    check("fitness lands near one half", 0.35 < result["fitness"]["5cm"] < 0.65,
          f"{result['fitness']['5cm']:.2f}")
    check("while the residual stays small on the half that matched",
          result["point_to_plane_m"] < 0.02,
          f"{100 * result['point_to_plane_m']:.1f} cm")


def test_coverage_gain_is_not_a_validity_test() -> None:
    """The trap this repository keeps walking into, pinned so it stays walked out of.

    Coverage gain on the shared surface is the right question to ask about a
    merge — and it is a terrible test of whether the merge is real. Forced into
    one frame, two unrelated rooms still land some points in the same voxels,
    and those voxels then collect views from two arbitrary directions. Measured
    on real data: a subway concourse aligned against an apartment scored +45
    percentage points, higher than any genuine pair in the set.

    So the test here is not that the number is small for a wrong pair. It is
    that the number is **useless** for telling the two apart, which is why
    `align_sessions.py` refuses to print it until fitness has cleared a control.
    """
    print("coverage gain cannot tell a real merge from a wrong one")
    points, normals = room(seed=8, width=4.0, depth=6.0)
    fixed = as_cloud(points, normals, "fixed")

    T = transform_of(35.0, (1.0, 0.0, -0.5))
    true_pair = as_cloud(*apply(np.linalg.inv(T), points, normals), "true")
    other_p, other_n = room(seed=9, width=8.0, depth=2.5, height=3.0)
    wrong_pair = as_cloud(other_p, other_n, "wrong")

    good = al.align_clouds(fixed, true_pair)
    bad = al.align_clouds(fixed, wrong_pair)

    check("fitness separates them cleanly",
          good["fitness"]["5cm"] > 3 * bad["fitness"]["5cm"],
          f"{good['fitness']['5cm']:.2f} against {bad['fitness']['5cm']:.2f}")

    # And the thing that must not be used as the gate: how many voxels the wrong
    # alignment still manages to share. Any at all is enough to compute a
    # flattering "gain" from, which is the whole hazard.
    index = al.NearestVoxel(fixed["points"], 0.05)
    moved = wrong_pair["points"] @ bad["transform"][:3, :3].T + bad["transform"][:3, 3]
    _, dist = index.query(moved)
    accidental = int((np.where(np.isfinite(dist), dist, np.inf) < 0.05).sum())
    check("a wrong alignment still shares voxels to compute a gain from",
          accidental > 100, f"{accidental} points land within 5 cm anyway")


def main() -> int:
    test_nearest_voxel_against_brute_force()
    test_nearest_voxel_when_cells_hold_many_points()
    test_floor_height()
    test_recovers_a_known_transform()
    test_a_different_room_does_not_pass()
    test_four_degrees_of_freedom_holds_the_vertical()
    test_fitness_counts_every_source_point()
    test_coverage_gain_is_not_a_validity_test()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} failed: {', '.join(FAILURES)}")
        return 1
    print("all passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
