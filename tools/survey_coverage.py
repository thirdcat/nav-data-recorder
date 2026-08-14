#!/usr/bin/env python3
"""How many different directions does each piece of surface get seen from?

Image count is the number people quote and it is the wrong one. Two hundred
frames of the same wall from the same doorway is one observation of that wall,
and a Gaussian splat fitted to it has nothing constraining how far away the wall
is — only that its colours land in the right pixels. The quantity that decides
whether a reconstruction is determined is *angular* coverage, and everything
needed to compute it is already recorded.

    python3 tools/survey_coverage.py ~/nav_data/*/

Method: back-project each image-paired depth map with its ARKit pose, drop the
points into a voxel grid, and count for each voxel how many coarsely distinct
directions it was viewed from. Coarse is the point — standing still for ninety
frames should count once, not ninety times — so directions are binned at
`--bin-deg` before being counted.

The basis change and the depth back-projection are imported from
`export_3dgs.py` rather than repeated. A second copy of that conversion is a
second place for it to be wrong, and the two would then disagree about what the
same session contains.

Read the output as: `>=3 views` is the fraction of surface a splat has enough
evidence to place, and `range` is the strongest thing that predicts it — across
23 recorded sessions the median distance to the surface correlates with wall
coverage at Spearman -0.63, while path `linearity` correlates at -0.03. Walking
*past* something close converts metres of travel into viewing angle; walking
*toward* something far does not, however the path is shaped.

Use `--vertical-only` for the number that matters. Floors and ceilings are
covered almost for free by any walk, and they can dominate the aggregate: one
session here reads 18.6 % of surface at three or more views, and 1.4 % once
horizontal surfaces are excluded.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_3dgs import (DEPTH_FAR_M, DEPTH_NEAR_M, camera_to_world,  # noqa: E402
                         depth_points)
from read_session import Session  # noqa: E402


def direction_bins(vectors: np.ndarray, bin_deg: float) -> np.ndarray:
    """Label unit vectors by roughly equal-area cells about `bin_deg` across.

    Equal-area matters: a plain latitude/longitude grid makes cells near the
    poles vanishingly narrow, so two nearly identical downward views — which is
    what a phone aimed at the floor produces all session — would land in
    different cells and be counted as coverage that is not there.
    """
    n_theta = max(1, int(round(180.0 / bin_deg)))
    theta = np.arccos(np.clip(vectors[:, 2], -1.0, 1.0))
    phi = np.arctan2(vectors[:, 1], vectors[:, 0]) + math.pi
    band = np.clip((theta / math.pi * n_theta).astype(np.int64), 0, n_theta - 1)
    per_band = np.maximum(1, (2 * n_theta * np.sin((band + 0.5) / n_theta * math.pi)
                              ).astype(np.int64))
    return band * (2 * n_theta + 1) + (phi / (2 * math.pi) * per_band).astype(np.int64) % per_band


def path_shape(centres: np.ndarray) -> dict[str, Any]:
    """Is the camera path a line, a ribbon, or a volume — and how big?

    `spread_m` is the largest distance between any two camera positions, which
    is the quantity that becomes viewing angle. `linearity` is a *shape* ratio
    and carries no size at all: a capture that wandered 40 cm and one that
    wandered 4 m both read 1.1 if they wandered equally in every direction.

    They are reported together because reading the ratio alone made a
    near-stationary object scan look like the best-shaped capture in the set.
    The singular values are also kept, under a name that says what they are —
    they are standard deviations, not extents, and calling them extents is how
    a 41 cm spread came to be described as an 11 cm one.
    """
    if len(centres) < 3:
        return {"spread_m": None, "sigma_m": None, "linearity": None, "flatness": None}
    centred = centres - centres.mean(axis=0)
    sv = np.linalg.svd(centred, compute_uv=False) / math.sqrt(len(centred))
    if len(centres) <= 512:
        spread = float(np.linalg.norm(centres[:, None, :] - centres[None, :, :],
                                      axis=2).max())
    else:  # bounding-box diagonal: an upper bound, and O(n) instead of O(n^2)
        spread = float(np.linalg.norm(centres.max(0) - centres.min(0)))
    return {
        "spread_m": round(spread, 3),
        "sigma_m": [round(float(x), 3) for x in sv],
        "linearity": round(float(sv[0] / max(sv[1], 1e-9)), 1),
        "flatness": round(float(sv[1] / max(sv[2], 1e-9)), 1),
    }


# ARKit runs `worldAlignment = .gravity`, so world +Y is up without any fitting.
WORLD_UP = np.array([0.0, 1.0, 0.0])


def _gather(session_dir: str, *, voxel: float = 0.05, conf_min: int = 2,
            bin_deg: float = 15.0, pix_stride: int = 2,
            vertical_only: bool = False, vertical_deg: float = 45.0,
            transform: np.ndarray | None = None,
            near: float = DEPTH_NEAR_M, far: float = DEPTH_FAR_M):
    """Raw (voxel cell, direction cell) votes for one session.

    Returns `(cells, counts, centres, ranges)` — the integer voxel index of each
    distinct occupied voxel, how many direction cells saw it, the camera centres
    along the path, and every point's range from the camera that measured it.
    `survey()` summarises this and `plot_coverage.py` draws it; neither counts
    anything itself, so the picture and the table cannot disagree.
    """
    session = Session(session_dir)
    rows = [r for r in session.posed_images()
            if r.get("depth") is not None and r["pose"].get("tracking") == "normal"]
    if len(rows) < 2:
        raise ValueError(f"only {len(rows)} usable frames")

    horizontal = math.cos(math.radians(vertical_deg))
    cells, bins, centres, ranges = [], [], [], []
    for row in rows:
        points, normals, _ = depth_points(session, row, conf_min=conf_min, near=near,
                                          far=far, pix_stride=pix_stride)
        if len(points) and vertical_only:
            upright = np.abs(normals @ WORLD_UP) < horizontal
            points, normals = points[upright], normals[upright]
        if not len(points):
            continue
        centre = camera_to_world(row["pose"])[:3, 3]
        if transform is not None:
            # A session aligned into another's frame keeps its own geometry; only
            # where it sits changes. Both the surface and the eye that saw it
            # have to move, or the directions come out measured from the wrong
            # place — which is a silent way to invent coverage.
            points = points @ transform[:3, :3].T + transform[:3, 3]
            centre = transform[:3, :3] @ centre + transform[:3, 3]
        centres.append(centre)
        ranges.append(np.linalg.norm(points - centre, axis=1))

        cells.append(np.floor(points / voxel).astype(np.int64))
        view = centre - points
        view /= np.maximum(np.linalg.norm(view, axis=1, keepdims=True), 1e-9)
        bins.append(direction_bins(view, bin_deg))

    if not cells:
        raise ValueError("no depth passed the confidence and range gates")
    return (np.concatenate(cells), np.concatenate(bins),
            np.asarray(centres), np.concatenate(ranges))


def voxel_view_counts(session_dir: str, **kwargs):
    """Per-voxel distinct-direction counts. The one place this is computed.

    Returns `(cells, counts, centres, ranges)` — the integer voxel index of each
    distinct occupied voxel, how many direction cells saw it, the camera centres
    along the path, and every point's range from the camera that measured it.
    `survey()` summarises this and `plot_coverage.py` draws it; neither counts
    anything itself, so the picture and the table cannot disagree.
    """
    return fold_votes(*_gather(session_dir, **kwargs))


def fold_votes(cells: np.ndarray, bins: np.ndarray, centres: np.ndarray,
               ranges: np.ndarray):
    """Collapse raw (voxel, direction) votes into a count per voxel.

    Kept separate from the gathering so several sessions can pool their votes
    *before* the fold. Folding each session first and adding the results would
    count a voxel twice for one direction two sessions happened to share, which
    is the one thing this measurement must never do.
    """
    key = ((cells[:, 0] * 73856093) ^ (cells[:, 1] * 19349663)
           ^ (cells[:, 2] * 83492791))

    # One vote per (voxel, direction cell), so repeated frames from one spot
    # collapse before anything is counted.
    pairs = np.unique(np.stack([key, bins], axis=1), axis=0)
    uniq, counts = np.unique(pairs[:, 0], return_counts=True)

    # One representative row per distinct voxel, aligned with `counts`.
    order = np.argsort(key)
    sorted_key, sorted_cells = key[order], cells[order]
    first = np.ones(len(sorted_key), dtype=bool)
    first[1:] = sorted_key[1:] != sorted_key[:-1]
    return (sorted_cells[first], counts[np.searchsorted(uniq, sorted_key[first])],
            centres, ranges)


def gather_votes(session_dir: str, **kwargs):
    """The raw votes a session casts, before any folding."""
    return _gather(session_dir, **kwargs)


def merged_view_counts(session_dirs: list[str], transforms,
                       return_provenance: bool = False, **kwargs):
    """The same counting over several aligned sessions, as one space.

    Nothing here merges point clouds. Each session casts its own votes in the
    shared frame and they land in the same voxel grid, which is what makes the
    answer meaningful: a voxel reaches three views because three genuinely
    different places saw it, not because two clouds overlapped.
    """
    # `transforms` is either keyed by session id, which is what `align_set.py`
    # writes, or a plain list aligned with `session_dirs` — the second form
    # exists because the same directory can legitimately appear twice, and a
    # dictionary cannot give two copies of one path two different placements.
    def placement(index: int, session_dir: str):
        if isinstance(transforms, (list, tuple)):
            return transforms[index]
        return transforms.get(os.path.basename(os.path.normpath(session_dir)))

    cells, bins, centres, ranges = [], [], [], []
    for index, session_dir in enumerate(session_dirs):
        c, b, ce, r = _gather(session_dir, transform=placement(index, session_dir),
                              **kwargs)
        cells.append(c)
        bins.append(b)
        centres.append(ce)
        ranges.append(r)
    if not cells:
        raise ValueError("no session produced any votes")
    all_cells = np.concatenate(cells)
    folded = fold_votes(all_cells, np.concatenate(bins),
                        np.concatenate(centres), np.concatenate(ranges))
    if not return_provenance:
        return folded

    # How many *sessions* saw each voxel, which is the difference between
    # "merging helped here" and "only one walk ever came this way". Without it
    # the aggregate reads as a disappointment: on 5bd1ed + cb4586 the merged map
    # scores 20 % at three views, between the two sessions' own 15 % and 21 %,
    # because most of the merged area is surface a single session saw and
    # merging cannot add angles to a place the other walk never reached.
    owner = np.concatenate([np.full(len(c), i) for i, c in enumerate(cells)])
    key = ((all_cells[:, 0] * 73856093) ^ (all_cells[:, 1] * 19349663)
           ^ (all_cells[:, 2] * 83492791))
    pairs = np.unique(np.stack([key, owner], axis=1), axis=0)
    uniq, seen_by = np.unique(pairs[:, 0], return_counts=True)

    kept_cells = folded[0]
    kept_key = ((kept_cells[:, 0] * 73856093) ^ (kept_cells[:, 1] * 19349663)
                ^ (kept_cells[:, 2] * 83492791))
    return folded + (seen_by[np.searchsorted(uniq, kept_key)],)


def survey(session_dir: str, *, voxel: float = 0.05, conf_min: int = 2,
           bin_deg: float = 15.0, pix_stride: int = 2, vertical_only: bool = False,
           vertical_deg: float = 45.0,
           near: float = DEPTH_NEAR_M, far: float = DEPTH_FAR_M) -> dict[str, Any]:
    try:
        _, counts, centres, ranges = voxel_view_counts(
            session_dir, voxel=voxel, conf_min=conf_min, bin_deg=bin_deg,
            pix_stride=pix_stride, vertical_only=vertical_only,
            vertical_deg=vertical_deg, near=near, far=far)
    except ValueError as exc:
        return {"id": os.path.basename(os.path.normpath(session_dir)), "error": str(exc)}

    result = {
        "id": Session(session_dir).id,
        "frames": len(centres),
        "surface_voxels": int(counts.size),
        "voxel_m": voxel,
        "bin_deg": bin_deg,
        "conf_min": conf_min,
        "vertical_only": vertical_only,
        "median_range_m": round(float(np.median(ranges)), 2),
        "views_per_voxel": {
            "median": float(np.median(counts)),
            "mean": round(float(counts.mean()), 2),
            "p90": float(np.percentile(counts, 90)),
            "max": int(counts.max()),
        },
        "frac_at_least": {str(k): round(float((counts >= k).mean()), 4)
                          for k in (2, 3, 5, 10)},
    }
    result.update(path_shape(centres))
    return result


def format_table(results: list[dict[str, Any]]) -> str:
    head = (f"{'session':>12} {'frames':>6} {'voxels':>7} {'median':>6} "
            f"{'>=2':>6} {'>=3':>6} {'>=5':>6} {'range':>7} {'spread':>8} {'lin':>5}")
    lines = [head, "-" * len(head)]
    for r in results:
        if "error" in r:
            lines.append(f"{r['id'][-6:]:>12}  {r['error']}")
            continue
        f = r["frac_at_least"]
        lines.append(
            f"{r['id'][-12:]:>12} {r['frames']:>6} {r['surface_voxels']:>7} "
            f"{r['views_per_voxel']['median']:>6.0f} "
            f"{100 * f['2']:>5.0f}% {100 * f['3']:>5.0f}% {100 * f['5']:>5.0f}% "
            f"{r['median_range_m']:>6.2f}m {r['spread_m'] or 0:>7.2f}m "
            f"{r['linearity'] if r['linearity'] is not None else 0:>5.1f}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--voxel", type=float, default=0.05, help="surface cell size in metres")
    ap.add_argument("--bin-deg", type=float, default=15.0,
                    help="how far apart two views must be to count separately")
    ap.add_argument("--conf-min", type=int, default=2,
                    help="lowest ARKit depth confidence counted (0/1/2)")
    ap.add_argument("--pix-stride", type=int, default=2)
    ap.add_argument("--vertical-only", action="store_true",
                    help="count only surfaces more than 45 degrees from horizontal — "
                         "floors and ceilings are covered by any walk and hide the rest")
    ap.add_argument("--transforms", default=None,
                    help="a JSON from align_set.py; the sessions are then scored "
                         "as ONE space rather than one row each")
    ap.add_argument("--json", action="store_true", help="emit the full records")
    a = ap.parse_args(argv)

    if a.transforms:
        with open(a.transforms) as fh:
            doc = json.load(fh)
        transforms = {k: np.asarray(v, dtype=float)
                      for k, v in doc["transforms"].items()}
        cells, counts, centres, ranges, seen_by = merged_view_counts(
            a.sessions, transforms, return_provenance=True, voxel=a.voxel,
            conf_min=a.conf_min, bin_deg=a.bin_deg, pix_stride=a.pix_stride,
            vertical_only=a.vertical_only)
        merged = {
            "id": f"merged({len(a.sessions)})",
            "frames": len(centres),
            "surface_voxels": int(counts.size),
            "voxel_m": a.voxel, "bin_deg": a.bin_deg, "conf_min": a.conf_min,
            "vertical_only": a.vertical_only,
            "median_range_m": round(float(np.median(ranges)), 2),
            "views_per_voxel": {"median": float(np.median(counts)),
                                "mean": round(float(counts.mean()), 2),
                                "p90": float(np.percentile(counts, 90)),
                                "max": int(counts.max())},
            "frac_at_least": {str(k): round(float((counts >= k).mean()), 4)
                              for k in (2, 3, 5, 10)},
        }
        merged.update(path_shape(centres))
        print(format_table([merged]))

        print("\nbroken out by how many sessions reached the surface —")
        print("merging can only add angles where more than one walk went")
        print(f"  {'sessions':>9} {'voxels':>8} {'share':>7} {'median':>7} "
              f"{'mean':>6} {'>=3':>6} {'>=5':>6}")
        for k in range(1, len(a.sessions) + 1):
            sel = seen_by == k
            if sel.sum() < 20:
                continue
            c = counts[sel]
            print(f"  {k:>9} {int(sel.sum()):>8} {100 * sel.mean():>6.0f}% "
                  f"{np.median(c):>7.0f} {c.mean():>6.2f} "
                  f"{100 * (c >= 3).mean():>5.0f}% {100 * (c >= 5).mean():>5.0f}%")
        return 0

    results = []
    for path in a.sessions:
        try:
            results.append(survey(path, voxel=a.voxel, conf_min=a.conf_min,
                                  bin_deg=a.bin_deg, pix_stride=a.pix_stride,
                                  vertical_only=a.vertical_only))
        except Exception as exc:  # one broken session must not stop a sweep
            results.append({"id": os.path.basename(os.path.normpath(path)),
                            "error": f"{type(exc).__name__}: {exc}"})

    if a.json:
        print(json.dumps(results, indent=1))
    else:
        print(format_table(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
