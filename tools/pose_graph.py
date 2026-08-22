#!/usr/bin/env python3
"""Close the loops `align_set.py`'s spanning tree throws away.

`tools/align_set.py` aligns every ordered pair of sessions and then keeps a
maximum spanning tree. Every session reaches the reference by exactly one path,
so every measurement off that path is discarded and any disagreement between two
paths is absorbed silently by whichever path the tree happened to pick. That is
the first item on `docs/3DGS.md`'s "Not done" list: *a tree, not a pose graph*.

This is the pose graph. Nodes are sessions, edges are measured relative
transforms, and the estimate is the one that disagrees least with all of them at
once rather than the one that obeys N-1 of them exactly and ignores the rest.

    python3 tools/pose_graph.py graph.json --out optimised.json

`docs/POSE_GRAPH_PREREG.md` fixed every constant below before this file existed.
Three of them decide the shape of the answer:

**Edges are weighted by the photometric score and never by geometric fitness.**
Measured across the corpus tree, fitness runs 0.51-0.85 on real edges and
0.21-0.63 on false ones, and in the verified grouping two *cut* edges score 0.73
against a *kept* one at 0.33 — the ordering is close to reversed. The
photometric score is the only column that separates (real 0.04-0.15, false
0.35-1.03) and the only one that does not touch the geometry being judged.
`information()` therefore takes a score and refuses a fitness.

**The information matrix is anisotropic.** On four pairs the correction the
photographs demanded was 3.6 to 7.8 times larger horizontally than vertically,
while geometric ICP sits at its floor (0.80 cm cross-session against 0.77 cm of
sensor repeatability). An isotropic σ would contradict four measurements. The
split is horizontal/vertical in the **gravity** frame, which is why the residual
below is expressed in the world frame and not in the measurement's.

**An edge with no photometric score is refused, not defaulted.** Three tree
edges of eleven cannot be judged at all — two overlapping frames is not enough —
and `eval/verify_groups.py` says so rather than guessing. `information()` raises
`UnjudgeableEdge`, in the shape `eval/umeyama.py` established: a fallback that
looks like a result is worse than a refusal.

numpy only, per `AGENTS.md`. No SciPy, and nothing here imports from `eval/`.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

# --- the constants docs/POSE_GRAPH_PREREG.md pinned -------------------------

# Median vertical correction over the three genuine refinement pairs
# {0.5, 1.9, 3.5} cm, which agrees with "height is pinned to about a centimetre".
SIGMA_V = 0.020
# 5.21 x that, the geometric mean of the per-pair horizontal/vertical ratios on
# the same three pairs {3.6, 7.4, 5.3}. The false pair's 7.8x is excluded
# because it is not a pair.
SIGMA_H = 0.104
# Residual tilt measured on genuine pairs, 0.18-0.59 degrees. Gravity is shared,
# so this is small and the graph should not be allowed to spend it.
SIGMA_TILT = math.radians(0.5)
# NOT MEASURED, and labelled as such wherever it appears. sigma_h over a 2.5 m
# lever arm: the camera centres of the twelve placed sessions span a median
# radius of 1.05 m about their own centroid and depth reaches 0.2-5.0 m beyond
# them. The pre-registration's A8 measures how much the answer depends on it.
SIGMA_YAW = math.radians(2.4)

# The photometric score of a good pair, used as the gauge of the weight map.
# Only ratios of weights reach the estimate, so this constant cancels.
REF_SCORE = 0.05
# w = (REF_SCORE / score) ** WEIGHT_EXPONENT. Two means the residual sigma is
# proportional to the photometric score, which is what the three genuine
# refinement rows suggest (0.0435 -> 1.7 cm, 0.1234 -> 14.2, 0.1485 -> 18.7).
# Three points and two parameters: an ordering, not a law.
WEIGHT_EXPONENT = 2.0


class UnjudgeableEdge(ValueError):
    """No photometric score, so there is no honest weight for this edge.

    The shape `eval/umeyama.py` established for `ScaleNotObservable`: the
    failure this repository keeps meeting is not a wrong answer that looks
    wrong, it is a fallback that looks like a result. An edge whose photographs
    could not be compared is not an edge with average confidence.
    """


class GraphNotConnected(ValueError):
    """Some node cannot reach the reference through any admitted edge."""


# --- SO(3) ------------------------------------------------------------------

def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def so3_exp(w: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(w))
    if theta < 1e-12:
        W = skew(w)
        return np.eye(3) + W + 0.5 * (W @ W)
    K = skew(np.asarray(w, dtype=float) / theta)
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def so3_log(R: np.ndarray) -> np.ndarray:
    """Axis-angle of a rotation, stable at 0 and at pi."""
    R = np.asarray(R, dtype=float)
    trace = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    theta = math.acos(trace)
    if theta < 1e-8:
        # First order is enough here and avoids 0/0.
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2.0
    if math.pi - theta < 1e-6:
        # Near pi the antisymmetric part vanishes; take the axis from R + I.
        A = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(A), 0.0, None))
        k = int(np.argmax(axis))
        if axis[k] < 1e-12:
            return np.zeros(3)
        axis = A[:, k] / axis[k]
        axis = axis / np.linalg.norm(axis)
        # Sign is ambiguous at exactly pi; pick the one the antisymmetric part
        # agrees with when it still carries any information.
        anti = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        if float(anti @ axis) < 0:
            axis = -axis
        return axis * theta
    return (theta / (2.0 * math.sin(theta))
            * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]))


def left_jacobian_inv(phi: np.ndarray) -> np.ndarray:
    """Inverse left Jacobian of SO(3): Log(Exp(a) Exp(phi)) ~= phi + Jli(phi) a."""
    theta = float(np.linalg.norm(phi))
    P = skew(phi)
    if theta < 1e-6:
        return np.eye(3) - 0.5 * P + (P @ P) / 12.0
    half = theta / 2.0
    coefficient = (1.0 - theta * math.cos(half) / (2.0 * math.sin(half))) / theta ** 2
    return np.eye(3) - 0.5 * P + coefficient * (P @ P)


# --- edges ------------------------------------------------------------------

def information(score: float | None, *, sigma_h: float = SIGMA_H,
                sigma_v: float = SIGMA_V, sigma_tilt: float = SIGMA_TILT,
                sigma_yaw: float = SIGMA_YAW, ref_score: float = REF_SCORE,
                exponent: float = WEIGHT_EXPONENT) -> np.ndarray:
    """The 6x6 information matrix of an edge, from its **photometric** score.

    Ordering is (rotation, translation) with the rotation expressed in the
    gravity-aligned world frame, so index 1 of each triple is the vertical:
    yaw for the rotation block, height for the translation block.

    Geometric fitness is deliberately not a parameter. Across the corpus tree it
    does not separate real edges from false ones, and against the verified
    grouping its ordering is close to reversed.
    """
    if score is None or not np.isfinite(score):
        raise UnjudgeableEdge(
            "this edge has no photometric score, so it has no weight. Two "
            "overlapping frames is not enough to judge a pairing, and an "
            "unjudged edge is not an edge of average confidence — it is one "
            "the photographs were never asked about.")
    if score <= 0:
        raise UnjudgeableEdge(f"photometric score {score} is not positive")
    weight = (ref_score / float(score)) ** exponent
    diag = np.array([1.0 / sigma_tilt ** 2, 1.0 / sigma_yaw ** 2, 1.0 / sigma_tilt ** 2,
                     1.0 / sigma_h ** 2, 1.0 / sigma_v ** 2, 1.0 / sigma_h ** 2])
    return np.diag(weight * diag)


@dataclass
class Edge:
    """A measured relative pose, plus what is known about how much to trust it.

    `measurement` is world-from-source in the parent's frame: the 4x4 that takes
    a point of `child`'s ARKit world into `parent`'s, which is exactly what
    `align_sessions.align_clouds(target=parent, source=child)` returns.
    """
    parent: str
    child: str
    measurement: np.ndarray
    information: np.ndarray
    score: float | None = None
    fitness: float | None = None
    note: str = ""

    def __post_init__(self) -> None:
        self.measurement = np.asarray(self.measurement, dtype=float).reshape(4, 4)
        self.information = np.asarray(self.information, dtype=float).reshape(6, 6)


def edge_from_score(parent: str, child: str, measurement: np.ndarray,
                    score: float | None, *, fitness: float | None = None,
                    **weights) -> Edge:
    """Build an edge, refusing rather than defaulting when it cannot be judged."""
    return Edge(parent, child, measurement, information(score, **weights),
                score=score, fitness=fitness)


# --- graph ------------------------------------------------------------------

def compose(edges: Sequence[Edge], reference: str,
            seed: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    """Breadth-first composition from the reference — what a tree gives you.

    Used as the initial guess and as the arm every result is compared against.
    On a graph with a loop this depends on the traversal order, which is exactly
    the problem: two paths to the same node do not agree and one of them is
    silently chosen.

    `seed` is what keeps the comparison honest. Handed the whole edge set at
    once, the traversal will happily reach a node down a *chord* and call the
    result "the tree composition" — which is then compared against itself. Pass
    the tree-only composition as the seed and the chords extend it instead of
    replacing it.
    """
    neighbours: dict[str, list[tuple[str, np.ndarray]]] = {}
    for e in edges:
        neighbours.setdefault(e.parent, []).append((e.child, e.measurement))
        neighbours.setdefault(e.child, []).append(
            (e.parent, np.linalg.inv(e.measurement)))
    poses = {k: np.asarray(v, dtype=float) for k, v in (seed or {}).items()}
    poses.setdefault(reference, np.eye(4))
    queue = list(poses)
    while queue:
        node = queue.pop(0)
        for other, relative in neighbours.get(node, ()):
            if other in poses:
                continue
            poses[other] = poses[node] @ relative
            queue.append(other)
    return poses


def residual(poses: dict[str, np.ndarray], edge: Edge) -> np.ndarray:
    """(rotation, translation) disagreement of one edge, in the world frame.

    Both halves are expressed in gravity-aligned world axes rather than in the
    measurement's own frame, so that index 1 means "vertical" in both — which is
    what makes the anisotropic information matrix mean what the measurements
    that set it meant.
    """
    Xi, Xj = poses[edge.parent], poses[edge.child]
    Ri, ti = Xi[:3, :3], Xi[:3, 3]
    Rj, tj = Xj[:3, :3], Xj[:3, 3]
    Rz, tz = edge.measurement[:3, :3], edge.measurement[:3, 3]
    E = Rj @ (Ri @ Rz).T
    return np.concatenate([so3_log(E), tj - (ti + Ri @ tz)])


def residuals(poses: dict[str, np.ndarray], edges: Sequence[Edge]) -> np.ndarray:
    return np.array([residual(poses, e) for e in edges]) if edges else np.zeros((0, 6))


def chi2(poses: dict[str, np.ndarray], edges: Sequence[Edge]) -> float:
    total = 0.0
    for e in edges:
        r = residual(poses, e)
        total += float(r @ e.information @ r)
    return total


def edge_jacobians(poses: dict[str, np.ndarray], edge: Edge
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Residual and its derivative w.r.t. (omega, u) of parent and child.

    The perturbation is left on rotation and additive on translation, both in
    the world frame:  R <- Exp(omega) R,  t <- t + u.
    """
    Xi, Xj = poses[edge.parent], poses[edge.child]
    Ri, ti = Xi[:3, :3], Xi[:3, 3]
    Rj, tj = Xj[:3, :3], Xj[:3, 3]
    Rz, tz = edge.measurement[:3, :3], edge.measurement[:3, 3]

    E = Rj @ (Ri @ Rz).T
    phi = so3_log(E)
    lever = Ri @ tz
    r = np.concatenate([phi, tj - (ti + lever)])

    Jli = left_jacobian_inv(phi)
    Ji = np.zeros((6, 6))
    Jj = np.zeros((6, 6))
    Ji[0:3, 0:3] = -Jli @ E
    Ji[3:6, 0:3] = skew(lever)
    Ji[3:6, 3:6] = -np.eye(3)
    Jj[0:3, 0:3] = Jli
    Jj[3:6, 3:6] = np.eye(3)
    return r, Ji, Jj


def reachable(edges: Sequence[Edge], reference: str, names: Iterable[str]) -> set[str]:
    neighbours: dict[str, set[str]] = {}
    for e in edges:
        neighbours.setdefault(e.parent, set()).add(e.child)
        neighbours.setdefault(e.child, set()).add(e.parent)
    seen, queue = {reference}, [reference]
    while queue:
        node = queue.pop()
        for other in neighbours.get(node, ()):
            if other not in seen:
                seen.add(other)
                queue.append(other)
    return seen & set(names)


@dataclass
class Result:
    poses: dict[str, np.ndarray]
    initial: dict[str, np.ndarray]
    reference: str
    iterations: int
    chi2_before: float
    chi2_after: float
    converged: bool
    unplaced: list[str] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)

    def displacement_m(self) -> dict[str, np.ndarray]:
        return {k: self.poses[k][:3, 3] - self.initial[k][:3, 3] for k in self.poses}


def optimise(edges: Sequence[Edge], reference: str, *,
             initial: dict[str, np.ndarray] | None = None,
             iterations: int = 100, tolerance: float = 1e-13,
             damping: float = 1e-9, strict: bool = True) -> Result:
    """Levenberg-damped Gauss-Newton over the whole graph at once.

    The reference's six variables are **eliminated** rather than softly pinned.
    A weak prior on the reference is the usual trick and it is the wrong one
    here: it lets the whole map slide by an amount nobody chose, and the frame
    `align_set.py` promised its callers is the reference session's exactly.

    With no loop this returns the tree composition — the residual can be driven
    to zero on every edge at once, and that solution is the composition. That is
    a property to check rather than to assume, and
    `tools/test_pose_graph.py` checks it.
    """
    # A node named in `initial` but touched by no admitted edge has to appear
    # here, or it is silently dropped instead of being reported unplaced — which
    # is exactly the case this graph exists to make visible: every edge that
    # could have reached it was cut by the photographs.
    names = sorted({e.parent for e in edges} | {e.child for e in edges}
                   | {reference} | set(initial or ()))
    if reference not in names:
        raise GraphNotConnected(f"reference {reference} is in no edge")
    live = reachable(edges, reference, names)
    unplaced = sorted(set(names) - live)
    if unplaced and strict:
        raise GraphNotConnected(
            f"{len(unplaced)} node(s) cannot reach the reference through an "
            f"admitted edge: {', '.join(unplaced)}. They are not placed at what "
            f"the tree said; they are not placed.")
    edges = [e for e in edges if e.parent in live and e.child in live]

    poses = {k: np.array(v, dtype=float) for k, v in (initial or compose(edges, reference)).items()}
    for name in live:
        if name not in poses:
            raise GraphNotConnected(f"no initial pose for {name}")
    poses = {k: v for k, v in poses.items() if k in live}

    free = [n for n in sorted(live) if n != reference]
    slot = {n: 6 * k for k, n in enumerate(free)}
    size = 6 * len(free)
    start = chi2(poses, edges)

    if size == 0 or not edges:
        return Result(poses, {k: v.copy() for k, v in poses.items()}, reference,
                      0, start, start, True, unplaced)

    initial_poses = {k: v.copy() for k, v in poses.items()}
    lam = damping
    current = start
    history: list[dict[str, Any]] = []
    converged = False
    used = 0
    for step in range(iterations):
        H = np.zeros((size, size))
        b = np.zeros(size)
        for e in edges:
            r, Ji, Jj = edge_jacobians(poses, e)
            ends = [(J, node) for J, node in ((Ji, e.parent), (Jj, e.child))
                    if node != reference]
            for Ja, na in ends:
                sa = slot[na]
                weighted = Ja.T @ e.information
                b[sa:sa + 6] += weighted @ r
                for Jb, nb in ends:
                    sb = slot[nb]
                    H[sa:sa + 6, sb:sb + 6] += weighted @ Jb

        scale = np.clip(np.diag(H), 1e-12, None)
        for _ in range(12):
            try:
                delta = np.linalg.solve(H + lam * np.diag(scale), -b)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            trial = {k: v.copy() for k, v in poses.items()}
            for node in free:
                s = slot[node]
                omega, u = delta[s:s + 3], delta[s + 3:s + 6]
                trial[node][:3, :3] = so3_exp(omega) @ trial[node][:3, :3]
                trial[node][:3, 3] = trial[node][:3, 3] + u
            candidate = chi2(trial, edges)
            if candidate <= current:
                poses = trial
                current = candidate
                lam = max(lam / 10.0, 1e-14)
                break
            lam *= 10.0
        else:
            break

        used = step + 1
        history.append({"iteration": used, "chi2": current,
                        "max_step": float(np.abs(delta).max()), "lambda": lam})
        if float(np.abs(delta).max()) < tolerance:
            converged = True
            break

    return Result(poses, initial_poses, reference, used, start, current,
                  converged, unplaced, history)


# --- CLI --------------------------------------------------------------------

def load_graph(path: str) -> tuple[str, list[Edge], dict[str, np.ndarray] | None,
                                   list[tuple[str, str]] | None]:
    data = json.loads(Path(path).read_text())
    reference = data["reference"]
    edges = []
    for row in data["edges"]:
        edges.append(edge_from_score(row["parent"], row["child"],
                                     np.asarray(row["measurement"], dtype=float),
                                     row.get("score"), fitness=row.get("fitness")))
    initial = None
    if data.get("transforms"):
        initial = {k: np.asarray(v, dtype=float) for k, v in data["transforms"].items()}
    tree = data.get("tree")
    if tree:
        tree = [(e["parent"], e["child"]) if isinstance(e, dict) else tuple(e)
                for e in tree]
    return reference, edges, initial, tree


def baseline(edges: Sequence[Edge], reference: str,
             tree: list[tuple[str, str]] | None) -> tuple[dict[str, np.ndarray], bool]:
    """The arm the optimisation is compared against: the tree's own composition.

    Without knowing which edges the tree used, a traversal will reach a node
    down a chord and the comparison then measures the optimiser against itself.
    Returns the composition and whether the tree was actually known.
    """
    if not tree:
        return compose(edges, reference), False
    on_tree = set(tree) | {(b, a) for a, b in tree}
    only = [e for e in edges if (e.parent, e.child) in on_tree]
    return compose(edges, reference, seed=compose(only, reference)), True


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("graph", help="JSON with reference, edges [parent, child, "
                                  "measurement, score], optional transforms and "
                                  "tree (which edges the spanning tree used)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--sigma-yaw-deg", type=float, default=math.degrees(SIGMA_YAW))
    ap.add_argument("--weight-exponent", type=float, default=WEIGHT_EXPONENT)
    a = ap.parse_args(argv)

    reference, edges, initial, tree_pairs = load_graph(a.graph)
    if a.sigma_yaw_deg != math.degrees(SIGMA_YAW) or a.weight_exponent != WEIGHT_EXPONENT:
        edges = [edge_from_score(e.parent, e.child, e.measurement, e.score,
                                 fitness=e.fitness,
                                 sigma_yaw=math.radians(a.sigma_yaw_deg),
                                 exponent=a.weight_exponent) for e in edges]

    tree, known = baseline(edges, reference, tree_pairs)
    result = optimise(edges, reference, initial=initial or tree)

    print(f"reference {reference[-6:]}, {len(result.poses)} nodes, {len(edges)} edges")
    print(f"chi2 {result.chi2_before:.3f} -> {result.chi2_after:.3f} "
          f"in {result.iterations} iterations"
          f"{'' if result.converged else '  (did not converge)'}")
    if not known:
        print("\n! this graph carries no `tree`, so the baseline below is a "
              "traversal that may\n  itself have reached a node down a chord. "
              "\"moved\" is then measured against\n  something the spanning tree "
              "never produced, and understates it. Pass the\n  tree if you have "
              "it — eval/close_loops.py does.")
    print(f"\n{'node':>8} {'moved':>9} {'horizontal':>11} {'vertical':>9}")
    for name in sorted(result.poses):
        d = result.poses[name][:3, 3] - tree.get(name, result.poses[name])[:3, 3]
        print(f"{name[-6:]:>8} {np.linalg.norm(d) * 100:>7.2f}cm "
              f"{np.linalg.norm(d[[0, 2]]) * 100:>9.2f}cm {abs(d[1]) * 100:>7.2f}cm")

    print(f"\n{'edge':>20} {'score':>8} {'|rot|':>8} {'|trans|':>9}   (residual after)")
    for e in edges:
        r = residual(result.poses, e)
        print(f"{e.parent[-6:]}<-{e.child[-6:]:>8} "
              f"{(e.score if e.score is not None else float('nan')):>8.4f} "
              f"{math.degrees(np.linalg.norm(r[:3])):>7.3f}d "
              f"{np.linalg.norm(r[3:]) * 100:>7.2f}cm")

    if a.out:
        Path(a.out).write_text(json.dumps({
            "reference": reference,
            "convention": "world_from_session, reference session's world",
            "transforms": {k: v.tolist() for k, v in result.poses.items()},
            "tree_composition": {k: v.tolist() for k, v in tree.items()},
            "chi2_before": result.chi2_before, "chi2_after": result.chi2_after,
            "iterations": result.iterations, "converged": result.converged,
            "unplaced": result.unplaced,
        }, indent=1))
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
