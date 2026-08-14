#!/usr/bin/env python3
"""Put two sessions of the same space into one coordinate frame.

Thermal pressure caps a session at about 35 seconds — `HANDOVER.md` §2 measured
it — so a room cannot be captured in one recording. Everything downstream of
that fact needs several fragments merged, and merging them needs the transform
between two ARKit worlds, which share nothing: each session picks its own origin
and its own yaw.

    python3 tools/align_sessions.py ~/nav_data/<a> ~/nav_data/<b>
    python3 tools/align_sessions.py ~/nav_data/<a> ~/nav_data/<b> --ply merged.ply

**The two worlds share gravity.** `ARRecorder.swift:224` runs
`worldAlignment = .gravity`, so both are +Y up and the unknown is a yaw and a
translation — four degrees of freedom, not six. That is what makes a global
search cheap enough to do exhaustively rather than hope an initial guess lands
in the right basin: the yaw is swept in full, and for each yaw the translation
falls out of a cross-correlation.

Three stages, and the last one is the one that matters:

1. **Coarse.** Vertical surfaces only, flattened to a floor plan. Sweep yaw,
   cross-correlate, take the peak. Floors and ceilings are excluded because a
   floor correlates with any other floor and drowns the walls that carry the
   shape of the room.
2. **Refine.** Point-to-plane ICP, six degrees of freedom this time. Letting
   roll and pitch move is deliberate: they should come back near zero, and how
   near is a measurement of whether the two gravity estimates agreed.
3. **Verify, against a control that can win.** The same alignment is run
   against an unrelated session. A residual is only evidence if the wrong pair
   produces a worse one, and the wrong pair is the only thing that can show
   that a plausible-looking transform is a coincidence of two flat walls.
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
                         depth_points, voxel_average, write_ply)
from read_session import Session  # noqa: E402

WORLD_UP = np.array([0.0, 1.0, 0.0])
# Voxel indices are packed into one integer so a lookup is a sorted search
# rather than a dictionary probe. 4096 cells a side at 5 cm covers 200 m, which
# is more room than any session here walks.
_GRID = 4096
_HALF = _GRID // 2


def voxel_keys(cells: np.ndarray) -> np.ndarray:
    shifted = cells + _HALF
    if shifted.min() < 0 or shifted.max() >= _GRID:
        raise ValueError("point cloud is larger than the packing grid")
    return ((shifted[:, 0] * _GRID + shifted[:, 1]) * _GRID + shifted[:, 2]).astype(np.int64)


class NearestVoxel:
    """Nearest-neighbour lookup on a voxel grid, vectorised over queries.

    SciPy's KD-tree would be the obvious tool and `tools/` deliberately does not
    have SciPy — `AGENTS.md` keeps this directory on numpy so a session can be
    inspected anywhere. A voxel grid is the honest substitute here because the
    clouds are already fused at a voxel size: within one cell there is one
    point, so scanning the 27 surrounding cells finds the true nearest neighbour
    for any distance below the cell size, which is the only range ICP cares
    about once it is close.
    """

    def __init__(self, points: np.ndarray, voxel: float):
        self.voxel = voxel
        self.points = points
        cells = np.floor(points / voxel).astype(np.int64)
        self.keys = voxel_keys(cells)
        order = np.argsort(self.keys)
        self.keys = self.keys[order]
        self.index = order
        self._offsets = np.array([(dx, dy, dz)
                                  for dx in (-1, 0, 1)
                                  for dy in (-1, 0, 1)
                                  for dz in (-1, 0, 1)], dtype=np.int64)

    def query(self, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Nearest target index per query point, and the distance. -1 if none."""
        base = np.floor(query / self.voxel).astype(np.int64)
        best_dist = np.full(len(query), np.inf)
        best_idx = np.full(len(query), -1, dtype=np.int64)
        for offset in self._offsets:
            keys = voxel_keys(base + offset)
            slot = np.searchsorted(self.keys, keys)
            slot = np.clip(slot, 0, len(self.keys) - 1)
            hit = self.keys[slot] == keys
            if not hit.any():
                continue
            candidate = self.index[slot[hit]]
            delta = self.points[candidate] - query[hit]
            dist = np.einsum("ij,ij->i", delta, delta)
            where = np.flatnonzero(hit)
            better = dist < best_dist[where]
            best_dist[where[better]] = dist[better]
            best_idx[where[better]] = candidate[better]
        return best_idx, np.sqrt(best_dist)


# --- building a cloud from a session -----------------------------------------

def session_cloud(session_dir: str, *, voxel: float = 0.05, conf_min: int = 2,
                  pix_stride: int = 2) -> dict[str, Any]:
    """The fused metric cloud a session implies, in its own ARKit world."""
    session = Session(session_dir)
    rows = [r for r in session.posed_images()
            if r.get("depth") is not None and r["pose"].get("tracking") == "normal"]
    if len(rows) < 5:
        raise ValueError(f"{session.id}: only {len(rows)} usable frames")

    pts, nrm, centres = [], [], []
    for row in rows:
        points, normals, _ = depth_points(session, row, conf_min=conf_min,
                                          near=DEPTH_NEAR_M, far=DEPTH_FAR_M,
                                          pix_stride=pix_stride)
        if not len(points):
            continue
        pts.append(points)
        nrm.append(normals)
        centres.append(camera_to_world(row["pose"])[:3, 3])

    points = np.concatenate(pts)
    normals = np.concatenate(nrm)
    colours = np.zeros((len(points), 3), dtype=np.uint8)
    points, normals, _, counts = voxel_average(points, normals, colours, voxel)
    upright = np.abs(normals @ WORLD_UP) < math.cos(math.radians(45.0))
    return {
        "id": session.id,
        "points": points,
        "normals": normals,
        "counts": counts,
        "vertical": upright,
        "centres": np.asarray(centres),
    }


def floor_height(points: np.ndarray, bin_m: float = 0.05) -> float:
    """The most populated horizontal slab, which indoors is the floor."""
    y = points[:, 1]
    lo, hi = float(y.min()), float(y.max())
    if hi - lo < bin_m:
        return lo
    bins = np.arange(lo, hi + bin_m, bin_m)
    counts, edges = np.histogram(y, bins=bins)
    return float(edges[int(np.argmax(counts))] + bin_m / 2)


# --- coarse: yaw sweep with a correlated floor plan ---------------------------

def floor_plan(points: np.ndarray, cell: float, origin: np.ndarray,
               shape: tuple[int, int]) -> np.ndarray:
    """Occupancy in the gravity-horizontal plane, one cell per occupied column."""
    ix = np.floor((points[:, 0] - origin[0]) / cell).astype(np.int64)
    iz = np.floor((points[:, 2] - origin[1]) / cell).astype(np.int64)
    keep = (ix >= 0) & (ix < shape[0]) & (iz >= 0) & (iz < shape[1])
    grid = np.zeros(shape, dtype=np.float64)
    if keep.any():
        grid[ix[keep], iz[keep]] = 1.0
    return grid


def yaw_matrix(radians: float) -> np.ndarray:
    c, s = math.cos(radians), math.sin(radians)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def coarse_align(target: dict[str, Any], source: dict[str, Any], *,
                 cell: float = 0.10, yaw_step_deg: float = 3.0) -> dict[str, Any]:
    """Sweep yaw; for each, take the translation from a cross-correlation.

    Only vertical surfaces enter the plan. A floor correlates with any other
    floor at every offset, which buries the walls that actually say where the
    room is — an early version of this matched two corridors by their floors and
    was confidently wrong by four metres.
    """
    fixed = target["points"][target["vertical"]]
    moving = source["points"][source["vertical"]]
    if len(fixed) < 100 or len(moving) < 100:
        raise ValueError("too few vertical surface points to align on")

    # Gravity gives the vertical directly: match the floors and the remaining
    # search is two-dimensional.
    dy = floor_height(target["points"]) - floor_height(source["points"])

    span = np.concatenate([fixed[:, [0, 2]], moving[:, [0, 2]]])
    extent = span.max(axis=0) - span.min(axis=0)
    size = int(np.ceil(max(extent) / cell)) * 2 + 8
    shape = (size, size)

    origin_fixed = fixed[:, [0, 2]].min(axis=0) - cell * size / 4
    grid_fixed = floor_plan(fixed, cell, origin_fixed, shape)
    spectrum_fixed = np.fft.rfft2(grid_fixed)

    best = None
    for yaw_deg in np.arange(0.0, 360.0, yaw_step_deg):
        rotated = moving @ yaw_matrix(math.radians(yaw_deg)).T
        origin_moving = rotated[:, [0, 2]].min(axis=0) - cell * size / 4
        grid_moving = floor_plan(rotated, cell, origin_moving, shape)
        # Correlation via FFT: the peak is the shift that lines the plans up.
        correlation = np.fft.irfft2(spectrum_fixed * np.conj(np.fft.rfft2(grid_moving)),
                                    s=shape)
        flat = int(np.argmax(correlation))
        peak = np.unravel_index(flat, shape)
        score = float(correlation[peak])
        shift = np.array([peak[0], peak[1]], dtype=np.float64)
        # A peak past the halfway point is a negative shift wrapped around.
        shift[shift > shape[0] / 2] -= shape[0]
        offset = shift * cell + (origin_fixed - origin_moving)
        if best is None or score > best["score"]:
            best = {"score": score, "yaw_deg": float(yaw_deg),
                    "translation": np.array([offset[0], dy, offset[1]])}

    transform = np.eye(4)
    transform[:3, :3] = yaw_matrix(math.radians(best["yaw_deg"]))
    transform[:3, 3] = best["translation"]
    best["transform"] = transform
    return best


# --- refine: point-to-plane ICP ----------------------------------------------

def icp(target: dict[str, Any], source: dict[str, Any], initial: np.ndarray, *,
        voxel: float = 0.05, iterations: int = 30, max_pairs: int = 60_000,
        reject: float = 0.20, dof: int = 6) -> dict[str, Any]:
    """Point-to-plane ICP from a coarse start.

    Six degrees of freedom by default, which is more than the geometry needs:
    the sessions are supposed to share gravity, so roll and pitch should come
    back near zero, and how near is the only measurement available of whether
    that assumption held. `dof=4` holds the vertical fixed instead and is the
    arm to compare against — if four does as well, the freedom was spent
    absorbing noise.
    """
    fixed_points = target["points"]
    fixed_normals = target["normals"]
    index = NearestVoxel(fixed_points, voxel)

    moving = source["points"]
    if len(moving) > max_pairs:
        pick = np.linspace(0, len(moving) - 1, max_pairs).astype(np.int64)
        moving = moving[pick]

    transform = initial.copy()
    history = []
    for _ in range(iterations):
        placed = moving @ transform[:3, :3].T + transform[:3, 3]
        idx, dist = index.query(placed)
        ok = (idx >= 0) & (dist < reject)
        if ok.sum() < 100:
            break
        p = placed[ok]
        q = fixed_points[idx[ok]]
        n = fixed_normals[idx[ok]]

        # Linearised point-to-plane: unknowns are a small rotation and a shift.
        cross = np.cross(p, n)
        b = -np.einsum("ij,ij->i", p - q, n)
        if dof == 4:
            # Yaw only: the vertical is gravity's and is not ours to move.
            A = np.concatenate([cross[:, [1]], n], axis=1)
            solution, *_ = np.linalg.lstsq(A, b, rcond=None)
            w = np.array([0.0, solution[0], 0.0])
            t = solution[1:]
        else:
            A = np.concatenate([cross, n], axis=1)
            solution, *_ = np.linalg.lstsq(A, b, rcond=None)
            w, t = solution[:3], solution[3:]

        step = np.eye(4)
        step[:3, :3] = small_rotation(w)
        step[:3, 3] = t
        transform = step @ transform
        history.append(float(np.sqrt(np.mean(b ** 2))))
        if len(history) > 2 and abs(history[-2] - history[-1]) < 1e-5:
            break

    placed = moving @ transform[:3, :3].T + transform[:3, 3]
    idx, dist = index.query(placed)
    ok = (idx >= 0) & (dist < reject)
    residual = float("nan")
    if ok.any():
        n = fixed_normals[idx[ok]]
        residual = float(np.median(np.abs(
            np.einsum("ij,ij->i", placed[ok] - fixed_points[idx[ok]], n))))

    # Fitness over *every* source point, not the ones that matched.
    #
    # The residual cannot be the criterion and `HANDOVER.md` §6 already says
    # why: it is the quantity ICP minimises, so it grades itself. Worse here,
    # it is conditioned on matching — a registration that finds a home for only
    # a fifth of its points is scored on that easiest fifth and comes out
    # looking tighter than one that placed three quarters. Measured: a control
    # against an unrelated room scored 1.4 cm against the true pair's 2.0 cm.
    #
    # Fitness has no such escape. It is the fraction of the source that landed
    # within a fixed distance of any surface, so failing to place a point costs
    # exactly as much as placing it wrongly.
    finite = np.where(idx >= 0, dist, np.inf)
    fitness = {f"{int(t * 100)}cm": float((finite < t).mean())
               for t in (0.02, 0.05, 0.10, 0.20)}
    return {
        "transform": transform,
        "fitness": fitness,
        "overlap": float(ok.mean()),
        "point_to_plane_m": residual,
        "point_to_point_m": float(np.median(dist[ok])) if ok.any() else float("nan"),
        "iterations": len(history),
    }


def small_rotation(w: np.ndarray) -> np.ndarray:
    """Rodrigues for a small axis-angle, kept exact so large steps stay valid."""
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        return np.eye(3)
    k = w / theta
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)


def tilt_degrees(rotation: np.ndarray) -> float:
    """How far the recovered rotation tips the vertical — gravity's disagreement."""
    up = rotation @ WORLD_UP
    return math.degrees(math.acos(max(-1.0, min(1.0, float(up @ WORLD_UP)))))


def align(target_dir: str, source_dir: str, **kwargs) -> dict[str, Any]:
    target = session_cloud(target_dir, **kwargs)
    source = session_cloud(source_dir, **kwargs)
    return align_clouds(target, source)


def align_clouds(target: dict[str, Any], source: dict[str, Any],
                 dof: int = 6) -> dict[str, Any]:
    coarse = coarse_align(target, source)
    refined = icp(target, source, coarse["transform"], dof=dof)
    refined["coarse_yaw_deg"] = coarse["yaw_deg"]
    refined["tilt_deg"] = tilt_degrees(refined["transform"][:3, :3])
    refined["target"] = target["id"]
    refined["source"] = source["id"]
    refined["target_points"] = int(len(target["points"]))
    refined["source_points"] = int(len(source["points"]))
    refined["dof"] = dof
    return refined


def align_set(session_dirs: list[str], *, reference: int = 0,
              min_fitness: float = 0.20, min_ratio: float = 2.0,
              **build) -> dict[str, Any]:
    """Put a whole set of sessions in one frame, via a maximum spanning tree.

    Two fragments of a room may not overlap each other at all while both overlap
    a third, so aligning everything directly to one reference throws away the
    only path that exists. The tree lets a session reach the reference through
    whichever neighbour it actually shares surface with.

    It is a tree and not a pose graph: there is no loop closure, so error
    accumulates along a chain. That is why every session is also scored
    *directly against the reference cloud* after composition — the edge fitness
    says the hop was good, and only the composed fitness says the session ended
    up in the right place.

    **An edge is admitted on a ratio, not on an absolute score.** The first
    version used a fixed cut at 0.35 and refused a genuine pair at 0.32 — a pair
    that passes both independent checks, beating an unrelated-room control 2.5x
    and adding 20 points of three-view coverage when merged. Absolute fitness
    measures how much of a session found a home, so it falls with how little
    area the two happen to share: that pair overlapped on 2 364 voxels where an
    accepted one overlapped on 11 851. Nothing was wrong with the alignment.

    So each edge is compared against the other candidates in its own row, which
    are the controls the matrix already contains — an unrelated session is what
    most entries in a row are. An edge must beat the median of its row's others
    by `min_ratio`, and clear a low absolute floor so that a row of uniform
    rubbish cannot elect a winner.
    """
    # A sweep over a folder must not die on one unusable recording: several
    # sessions here are three seconds long and never reach normal tracking.
    clouds, rejected = [], []
    for d in session_dirs:
        try:
            clouds.append(session_cloud(d, **build))
        except ValueError as exc:
            rejected.append(f"{os.path.basename(os.path.normpath(d))[-6:]}: {exc}")
    n = len(clouds)
    if n < 2:
        raise ValueError("fewer than two usable sessions: " + "; ".join(rejected))

    edges: dict[tuple[int, int], dict[str, Any]] = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            result = align_clouds(clouds[i], clouds[j])
            edges[(i, j)] = result

    def score(i: int, j: int) -> float:
        return edges[(i, j)]["fitness"]["5cm"]

    def ratio(i: int, j: int) -> float:
        """How far this edge stands out from the rest of its own row."""
        others = [score(i, k) for k in range(n) if k != i and k != j]
        if not others:
            return float("inf")          # a pair has no in-matrix control
        return score(i, j) / max(float(np.median(others)), 1e-6)

    # Prim's, taking the strongest admissible edge each time.
    placed = {reference: np.eye(4)}
    tree: list[dict[str, Any]] = []
    while len(placed) < n:
        best = None
        for parent in placed:
            for child in range(n):
                if child in placed:
                    continue
                if score(parent, child) < min_fitness:
                    continue
                if ratio(parent, child) < min_ratio:
                    continue
                if best is None or score(parent, child) > score(*best[:2]):
                    best = (parent, child)
        if best is None:
            break
        parent, child = best
        placed[child] = placed[parent] @ edges[(parent, child)]["transform"]
        tree.append({"parent": clouds[parent]["id"], "child": clouds[child]["id"],
                     "edge_fitness": round(score(parent, child), 4),
                     "edge_ratio": round(ratio(parent, child), 2)})

    # The honest check: how well does the composed transform place each session
    # against the reference itself, rather than against its parent?
    index = NearestVoxel(clouds[reference]["points"], 0.05)
    report = []
    for i, transform in sorted(placed.items()):
        if i == reference:
            continue
        moved = clouds[i]["points"] @ transform[:3, :3].T + transform[:3, 3]
        _, dist = index.query(moved)
        composed = float((np.where(np.isfinite(dist), dist, np.inf) < 0.05).mean())
        direct = edges[(reference, i)]["fitness"]["5cm"]
        report.append({
            "id": clouds[i]["id"],
            "composed_fitness_5cm": round(composed, 4),
            "direct_fitness_5cm": round(direct, 4),
            "transform": transform,
        })

    # Sessions the reference could not reach are not failures — most often they
    # are a different room. Grouping them says so, and turns a folder of
    # recordings into an answer to "which of these are the same place".
    parent_of = list(range(n))

    def find(x: int) -> int:
        while parent_of[x] != x:
            parent_of[x] = parent_of[parent_of[x]]
            x = parent_of[x]
        return x

    for i in range(n):
        for j in range(n):
            if i != j and score(i, j) >= min_fitness and ratio(i, j) >= min_ratio:
                a_root, b_root = find(i), find(j)
                if a_root != b_root:
                    parent_of[b_root] = a_root
    groups: dict[int, list[str]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(clouds[i]["id"])

    return {
        "reference": clouds[reference]["id"],
        "placed": len(placed),
        "of": n,
        "groups": sorted(groups.values(), key=len, reverse=True),
        "rejected": rejected,
        "unplaced": [clouds[i]["id"] for i in range(n) if i not in placed],
        "tree": tree,
        "sessions": report,
        "pairwise": {f"{clouds[i]['id'][-6:]}->{clouds[j]['id'][-6:]}":
                     round(edges[(i, j)]["fitness"]["5cm"], 3)
                     for (i, j) in edges},
        "ratios": {f"{clouds[i]['id'][-6:]}->{clouds[j]['id'][-6:]}":
                   round(ratio(i, j), 2) for (i, j) in edges},
        "min_fitness": min_fitness,
        "min_ratio": min_ratio,
        "transforms": {clouds[i]["id"]: placed[i] for i in placed},
    }


def shared_coverage(target_dir: str, source_dir: str, transform: np.ndarray, *,
                    voxel: float = 0.05, bin_deg: float = 15.0,
                    conf_min: int = 2, pix_stride: int = 3) -> dict[str, Any]:
    """Does merging add viewing directions, or only surface?

    The question the thermal ceiling forces is not whether two fragments can be
    put in one frame — that is stage one — but whether doing so makes the room
    *reconstructable*. Merging always adds area, and area proves nothing. So the
    comparison is restricted to the surface both sessions touched: if the second
    session only extended the map, the shared columns will not move.
    """
    from survey_coverage import direction_bins  # local: only this path needs it

    def votes(session_dir: str, move: np.ndarray | None):
        session = Session(session_dir)
        rows = [r for r in session.posed_images()
                if r.get("depth") is not None and r["pose"].get("tracking") == "normal"]
        keys, bins = [], []
        for row in rows:
            points, normals, _ = depth_points(session, row, conf_min=conf_min,
                                              near=DEPTH_NEAR_M, far=DEPTH_FAR_M,
                                              pix_stride=pix_stride)
            if not len(points):
                continue
            upright = np.abs(normals @ WORLD_UP) < math.cos(math.radians(45.0))
            points = points[upright]
            if not len(points):
                continue
            centre = camera_to_world(row["pose"])[:3, 3]
            if move is not None:
                points = points @ move[:3, :3].T + move[:3, 3]
                centre = move[:3, :3] @ centre + move[:3, 3]
            cell = np.floor(points / voxel).astype(np.int64)
            keys.append((cell[:, 0] * 73856093) ^ (cell[:, 1] * 19349663)
                        ^ (cell[:, 2] * 83492791))
            view = centre - points
            view /= np.maximum(np.linalg.norm(view, axis=1, keepdims=True), 1e-9)
            bins.append(direction_bins(view, bin_deg))
        return np.concatenate(keys), np.concatenate(bins)

    def counts(keys, bins, restrict=None):
        pairs = np.unique(np.stack([keys, bins], axis=1), axis=0)
        uniq, n = np.unique(pairs[:, 0], return_counts=True)
        if restrict is not None:
            keep = np.isin(uniq, restrict)
            uniq, n = uniq[keep], n[keep]
        return uniq, n

    ka, da = votes(target_dir, None)
    kb, db = votes(source_dir, transform)
    ua, _ = counts(ka, da)
    ub, _ = counts(kb, db)
    overlap = np.intersect1d(ua, ub)
    if len(overlap) < 200:
        return {"shared_voxels": int(len(overlap)),
                "note": "too little shared surface to compare"}

    _, alone = counts(ka, da, overlap)
    _, merged = counts(np.concatenate([ka, kb]), np.concatenate([da, db]), overlap)
    return {
        "shared_voxels": int(len(overlap)),
        "alone_mean": round(float(alone.mean()), 2),
        "alone_ge3": round(float((alone >= 3).mean()), 4),
        "merged_mean": round(float(merged.mean()), 2),
        "merged_ge3": round(float((merged >= 3).mean()), 4),
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("target", help="the session whose frame everything lands in")
    ap.add_argument("source", help="the session to move")
    ap.add_argument("--control", default=None,
                    help="a session of a DIFFERENT space; without one the "
                         "residual has nothing to be better than")
    ap.add_argument("--dof", type=int, choices=(4, 6), default=6,
                    help="4 holds the vertical fixed, which gravity already gave us")
    ap.add_argument("--voxel", type=float, default=0.05)
    ap.add_argument("--pix-stride", type=int, default=2)
    ap.add_argument("--conf-min", type=int, default=2)
    ap.add_argument("--coverage", action="store_true",
                    help="the criterion that matters: does merging add viewing "
                         "directions on the surface both sessions touched, or "
                         "only more surface")
    ap.add_argument("--ply", default=None, help="write the merged cloud here")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    build = dict(voxel=a.voxel, pix_stride=a.pix_stride, conf_min=a.conf_min)
    target = session_cloud(a.target, **build)
    source = session_cloud(a.source, **build)
    result = align_clouds(target, source, dof=a.dof)

    control = None
    if a.control:
        control_cloud = session_cloud(a.control, **build)
        control = align_clouds(target, control_cloud, dof=a.dof)

    if a.json:
        printable = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                     for k, v in result.items()}
        if control:
            printable["control"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                    for k, v in control.items()}
        print(json.dumps(printable, indent=1))
    else:
        print(f"{result['source']} -> {result['target']}")
        print(f"  points          {result['source_points']} onto {result['target_points']}")
        print(f"  coarse yaw      {result['coarse_yaw_deg']:.0f}°")
        print(f"  ICP iterations  {result['iterations']}")
        print(f"  fitness         " + "  ".join(
            f"{k} {100 * v:.0f}%" for k, v in result["fitness"].items())
            + "   <- fraction of ALL source points that close")
        print(f"  point-to-plane  {100 * result['point_to_plane_m']:.1f} cm median "
              f"over the {100 * result['overlap']:.0f}% that matched — "
              f"self-scored, not evidence")
        print(f"  residual tilt   {result['tilt_deg']:.2f}°  "
              f"(gravity is shared, so this should be near zero)")
        if control:
            print(f"\ncontrol: {control['source']} — a different space")
            print(f"  fitness         " + "  ".join(
                f"{k} {100 * v:.0f}%" for k, v in control["fitness"].items()))
            print(f"  point-to-plane  {100 * control['point_to_plane_m']:.1f} cm median "
                  f"(lower than the true pair is expected and means nothing)")
            true5 = result["fitness"]["5cm"]
            ctrl5 = control["fitness"]["5cm"]
            ratio = true5 / max(ctrl5, 1e-9)
            print(f"\n  fitness at 5 cm: {100 * true5:.0f}% vs {100 * ctrl5:.0f}% "
                  f"= {ratio:.1f}x")
            print(f"{'PASS' if ratio >= 2.0 else 'FAIL'}: the true pair "
                  f"{'clears' if ratio >= 2.0 else 'does not clear'} 2x the control")
        else:
            print("\nno --control given: this fitness has nothing to be better than")

    if a.coverage and not a.json:
        cov = shared_coverage(a.target, a.source, result["transform"],
                              voxel=a.voxel, conf_min=a.conf_min)
        print("\nshared-surface coverage (vertical surfaces both sessions saw)")
        if "note" in cov:
            print(f"  {cov['shared_voxels']} shared voxels — {cov['note']}")
        else:
            print(f"  shared voxels   {cov['shared_voxels']}")
            print(f"  target alone    {cov['alone_mean']:.2f} views mean, "
                  f"{100 * cov['alone_ge3']:.0f}% at 3+")
            print(f"  merged          {cov['merged_mean']:.2f} views mean, "
                  f"{100 * cov['merged_ge3']:.0f}% at 3+")
            gain = 100 * (cov["merged_ge3"] - cov["alone_ge3"])
            print(f"  gain            {gain:+.0f} percentage points")
            print(f"\n{'PASS' if gain > 3 else 'NO GAIN'}: merging "
                  f"{'adds viewing directions' if gain > 3 else 'only added surface'}")

    if a.ply:
        moved = source["points"] @ result["transform"][:3, :3].T + result["transform"][:3, 3]
        moved_n = source["normals"] @ result["transform"][:3, :3].T
        points = np.concatenate([target["points"], moved])
        normals = np.concatenate([target["normals"], moved_n])
        colours = np.concatenate([
            np.tile(np.array([[70, 130, 200]], np.uint8), (len(target["points"]), 1)),
            np.tile(np.array([[220, 120, 60]], np.uint8), (len(moved), 1))])
        write_ply(a.ply, points, normals, colours)
        print(f"\nwrote {a.ply} — target in blue, source in orange")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
