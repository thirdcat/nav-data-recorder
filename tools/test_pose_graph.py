#!/usr/bin/env python3
"""Acceptance for `tools/pose_graph.py`, pre-registered in docs/POSE_GRAPH_PREREG.md.

An optimiser that has never been run on a known answer has not been shown to
work, so every case here builds a graph whose true poses were chosen, corrupts
the measurements, and demands the truth back — or, where noise makes the truth
unrecoverable in principle, demands a pre-registered improvement over the
spanning tree the optimiser is supposed to replace.

    python3 tools/test_pose_graph.py

Exits nonzero on failure. numpy only.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pose_graph as pg  # noqa: E402

FAILURES: list[str] = []
NOTES: list[str] = []


def check(name: str, ok: bool, detail: str) -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}")
    if not ok:
        FAILURES.append(f"{name}: {detail}")


# --- synthetic graphs -------------------------------------------------------

def pose(yaw: float, tilt: np.ndarray, t: np.ndarray) -> np.ndarray:
    """A node pose: a yaw about gravity plus an explicit tilt, and a position."""
    X = np.eye(4)
    X[:3, :3] = pg.so3_exp(tilt) @ pg.so3_exp(np.array([0.0, yaw, 0.0]))
    X[:3, 3] = t
    return X


def truth(rng: np.random.Generator, n: int, *, box=(6.0, 0.4, 6.0),
          tilt_deg: float = 0.5) -> list[np.ndarray]:
    out = []
    for _ in range(n):
        t = (rng.random(3) - 0.5) * np.asarray(box)
        tilt = np.array([rng.normal(), 0.0, rng.normal()]) * math.radians(tilt_deg)
        out.append(pose(rng.uniform(0, 2 * math.pi), tilt, t))
    return out


def measure(true_poses, i: int, j: int, eps: np.ndarray) -> np.ndarray:
    """The measurement Z_ij whose world-frame residual is exactly `eps`.

    Inverting the residual definition rather than perturbing Z directly is what
    makes the generative noise and the information matrix the same model, which
    is what an efficiency claim needs.
    """
    Xi, Xj = true_poses[i], true_poses[j]
    Ri, ti = Xi[:3, :3], Xi[:3, 3]
    Rj, tj = Xj[:3, :3], Xj[:3, 3]
    Rz = Ri.T @ pg.so3_exp(-eps[:3]) @ Rj
    tz = Ri.T @ (tj - ti - eps[3:])
    Z = np.eye(4)
    Z[:3, :3], Z[:3, 3] = Rz, tz
    return Z


PINNED = dict(sigma_h=pg.SIGMA_H, sigma_v=pg.SIGMA_V,
              sigma_tilt=pg.SIGMA_TILT, sigma_yaw=pg.SIGMA_YAW)
ISOTROPIC = dict(sigma_h=0.05, sigma_v=0.05,
                 sigma_tilt=math.radians(1.0), sigma_yaw=math.radians(1.0))


def draw(rng, sigmas) -> np.ndarray:
    return np.array([rng.normal(0, sigmas["sigma_tilt"]),
                     rng.normal(0, sigmas["sigma_yaw"]),
                     rng.normal(0, sigmas["sigma_tilt"]),
                     rng.normal(0, sigmas["sigma_h"]),
                     rng.normal(0, sigmas["sigma_v"]),
                     rng.normal(0, sigmas["sigma_h"])])


RING = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
CHORDS = [(5, 0), (0, 3), (1, 4)]


def build(rng, *, noise=None, score: float = 0.05, chords=True, weights=None):
    """Six nodes, a ring plus two chords: five tree edges, three loops."""
    true_poses = truth(rng, 6)
    names = [f"s{k}" for k in range(6)]
    pairs = RING + (CHORDS if chords else [])
    edges, tree_edges = [], []
    for i, j in pairs:
        eps = np.zeros(6) if noise is None else draw(rng, noise)
        e = pg.edge_from_score(names[i], names[j], measure(true_poses, i, j, eps),
                               score, **(weights or {}))
        edges.append(e)
        if (i, j) in RING:
            tree_edges.append(e)
    return true_poses, names, edges, tree_edges


def relative_truth(true_poses, reference: int = 0) -> list[np.ndarray]:
    inv = np.linalg.inv(true_poses[reference])
    return [inv @ X for X in true_poses]


def position_rms(poses: dict, names, target) -> float:
    err = [np.linalg.norm(poses[n][:3, 3] - target[k][:3, 3])
           for k, n in enumerate(names)]
    return float(np.sqrt(np.mean(np.square(err))))


def jostle(poses: dict, reference: str, rng, *, metres=0.05, degrees=2.0) -> dict:
    """Move every non-reference node off the answer, so nothing is handed over."""
    out = {}
    for name, X in poses.items():
        Y = X.copy()
        if name != reference:
            axis = rng.normal(size=3)
            Y[:3, :3] = pg.so3_exp(axis / np.linalg.norm(axis)
                                   * math.radians(degrees)) @ Y[:3, :3]
            step = rng.normal(size=3)
            Y[:3, 3] = Y[:3, 3] + step / np.linalg.norm(step) * metres
        out[name] = Y
    return out


# --- A1 ---------------------------------------------------------------------

def a1_exact_recovery() -> None:
    print("\nA1  exact recovery on a known answer, noiseless, with three loops")
    worst_t, worst_r, worst_iter, ok_all = 0.0, 0.0, 0, True
    for seed in range(20):
        rng = np.random.default_rng(seed)
        true_poses, names, edges, tree_edges = build(rng)
        start = jostle(pg.compose(tree_edges, "s0"), "s0", rng)
        result = pg.optimise(edges, "s0", initial=start, iterations=50)
        target = relative_truth(true_poses)
        for k, n in enumerate(names):
            dt = np.linalg.norm(result.poses[n][:3, 3] - target[k][:3, 3])
            dr = np.linalg.norm(pg.so3_log(result.poses[n][:3, :3] @ target[k][:3, :3].T))
            worst_t, worst_r = max(worst_t, dt), max(worst_r, dr)
        worst_iter = max(worst_iter, result.iterations)
        ok_all &= result.converged
    check("A1", worst_t <= 1e-9 and worst_r <= 1e-9 and ok_all and worst_iter <= 50,
          f"worst position error {worst_t:.2e} m, rotation {worst_r:.2e} rad, "
          f"{worst_iter} iterations max, converged={ok_all} (bar 1e-9, 50)")


# --- A2 ---------------------------------------------------------------------

def a2_loop_closure() -> None:
    print("\nA2  a loop that does not close under the tree, and closes after")
    for label, sigmas in (("pinned", PINNED), ("isotropic-misspecified", ISOTROPIC)):
        tree_gap, improved, better, rms_tree, rms_opt, ref_moved = [], 0, 0, [], [], 0.0
        for seed in range(200):
            rng = np.random.default_rng(10_000 + seed)
            true_poses, names, edges, tree_edges = build(rng, noise=sigmas)
            tree = pg.compose(tree_edges, "s0")
            chords = [e for e in edges if e not in tree_edges]
            tree_gap.append(max(float(np.linalg.norm(pg.residual(tree, c)[3:]))
                                for c in chords))
            result = pg.optimise(edges, "s0", initial=tree, iterations=100)
            worst_before = float(np.abs(pg.residuals(tree, edges)[:, 3:]).max())
            worst_after = float(np.abs(pg.residuals(result.poses, edges)[:, 3:]).max())
            improved += worst_after <= worst_before
            target = relative_truth(true_poses)
            rt = position_rms(tree, names, target)
            ro = position_rms(result.poses, names, target)
            rms_tree.append(rt)
            rms_opt.append(ro)
            better += ro < rt
            ref_moved = max(ref_moved, float(np.abs(result.poses["s0"] - np.eye(4)).max()))

        gap = float(np.median(tree_gap))
        gain = 1.0 - float(np.median(rms_opt)) / float(np.median(rms_tree))
        worse = 1.0 - better / 200
        if label == "pinned":
            check("A2a", gap >= 0.03,
                  f"median worst chord disagreement under the tree {gap * 100:.1f} cm "
                  f"(bar 3.0 cm)")
        check(f"A2b [{label}]", improved / 200 >= 0.95,
              f"worst edge residual not increased in {improved}/200 seeds (bar 95%)")
        check(f"A2c [{label}]", gain >= 0.20 and worse <= 0.10,
              f"median RMS position error {np.median(rms_tree) * 100:.2f} cm -> "
              f"{np.median(rms_opt) * 100:.2f} cm = {gain * 100:.1f}% better; "
              f"worse than the tree in {worse * 100:.1f}% of seeds "
              f"(bars 20%, 10%)")
        check(f"A2d [{label}]", ref_moved == 0.0,
              f"reference moved by {ref_moved:.1e} (bar exactly 0)")


# --- A3 ---------------------------------------------------------------------

def a3_chain_untouched() -> None:
    print("\nA3  a chain with no loop must come back to the tree composition")
    settings = [("pinned", PINNED)]
    for deg in (0.6, 1.2, 2.4, 4.8):
        settings.append((f"yaw{deg}", dict(PINNED, sigma_yaw=math.radians(deg))))
    for ratio in (1.0, 5.21, 10.0):
        settings.append((f"aniso{ratio}", dict(PINNED, sigma_v=pg.SIGMA_H / ratio)))
    worst_t = worst_r = 0.0
    for label, weights in settings:
        for seed in range(200):
            rng = np.random.default_rng(20_000 + seed)
            _, names, edges, tree_edges = build(rng, noise=PINNED, chords=False,
                                                weights=weights)
            tree = pg.compose(tree_edges, "s0")
            start = jostle(tree, "s0", rng)
            result = pg.optimise(edges, "s0", initial=start, iterations=100)
            for n in names:
                worst_t = max(worst_t, float(np.linalg.norm(
                    result.poses[n][:3, 3] - tree[n][:3, 3])))
                worst_r = max(worst_r, float(np.linalg.norm(pg.so3_log(
                    result.poses[n][:3, :3] @ tree[n][:3, :3].T))))
    check("A3", worst_t <= 1e-9 and worst_r <= 1e-9,
          f"over {len(settings)} sigma settings x 200 seeds, worst departure from "
          f"the tree composition {worst_t:.2e} m, {worst_r:.2e} rad (bar 1e-9)")


# --- A4 ---------------------------------------------------------------------

def numeric_minimiser(edges, reference, names, initial, *, iterations=400):
    """A second implementation: finite-difference LM on a flat parameter vector.

    Shares the residual definition — that is the objective and there is only one
    of it — and shares nothing else. No analytic Jacobian, no block assembly, no
    sparsity, no gauge elimination beyond dropping the reference's columns.
    """
    free = [n for n in names if n != reference]
    poses = {k: v.copy() for k, v in initial.items()}

    def apply(base, x):
        out = {k: v.copy() for k, v in base.items()}
        for k, n in enumerate(free):
            out[n][:3, :3] = pg.so3_exp(x[6 * k:6 * k + 3]) @ out[n][:3, :3]
            out[n][:3, 3] = out[n][:3, 3] + x[6 * k + 3:6 * k + 6]
        return out

    def stacked(x):
        p = apply(poses, x)
        rows = []
        for e in edges:
            L = np.linalg.cholesky(e.information)
            rows.append(L.T @ pg.residual(p, e))
        return np.concatenate(rows)

    lam = 1e-6
    for _ in range(iterations):
        x0 = np.zeros(6 * len(free))
        r0 = stacked(x0)
        J = np.zeros((len(r0), len(x0)))
        for c in range(len(x0)):
            h = 1e-7
            xp, xm = x0.copy(), x0.copy()
            xp[c] += h
            xm[c] -= h
            J[:, c] = (stacked(xp) - stacked(xm)) / (2 * h)
        A = J.T @ J
        g = J.T @ r0
        step = np.linalg.solve(A + lam * np.diag(np.clip(np.diag(A), 1e-12, None)), -g)
        trial = apply(poses, step)
        if float(np.sum(np.square(stacked(step)))) <= float(np.sum(np.square(r0))):
            poses = trial
            lam = max(lam / 3, 1e-14)
        else:
            lam *= 5
        if np.abs(step).max() < 1e-13:
            break
    return poses


def a4_independent() -> None:
    print("\nA4  agreement with a second, independently written minimiser")
    worst = 0.0
    for seed in range(6):
        rng = np.random.default_rng(30_000 + seed)
        _, names, edges, tree_edges = build(rng, noise=PINNED)
        tree = pg.compose(tree_edges, "s0")
        mine = pg.optimise(edges, "s0", initial=tree, iterations=200).poses
        theirs = numeric_minimiser(edges, "s0", names, tree)
        for n in names:
            worst = max(worst, float(np.linalg.norm(
                mine[n][:3, 3] - theirs[n][:3, 3])))
    check("A4", worst <= 1e-9,
          f"worst position disagreement {worst:.2e} m over 6 graphs (bar 1e-9)")

    # A4b. The pre-registration asked for a closed-form translation-only check.
    # It cannot be run as written and the reason is a finding, not a nuisance:
    # rotation and translation do NOT decouple. d(e_t)/d(omega_i) = skew(R_i t_z)
    # is exactly the lever arm that lets a node move sideways by turning, so the
    # minimiser of the stated objective is not the translation-only solution.
    # What can be checked is the limit: clamp rotation with a tiny sigma and the
    # answer must approach the closed form.
    clamp = dict(PINNED, sigma_tilt=1e-5, sigma_yaw=1e-5)
    rng = np.random.default_rng(31_000)
    names = [f"s{k}" for k in range(6)]
    true_t = [(rng.random(3) - 0.5) * np.array([6.0, 0.4, 6.0]) for _ in range(6)]
    edges = []
    for i, j in RING + CHORDS:
        Z = np.eye(4)
        Z[:3, 3] = true_t[j] - true_t[i] + rng.normal(0, 0.05, 3)
        edges.append(pg.edge_from_score(names[i], names[j], Z, 0.05, **clamp))
    got = pg.optimise(edges, "s0", iterations=400).poses

    # Closed form: weighted linear least squares on the incidence matrix.
    W = np.diag([1 / pg.SIGMA_H, 1 / pg.SIGMA_V, 1 / pg.SIGMA_H])
    rows, rhs = [], []
    for e in edges:
        A = np.zeros((3, 3 * 5))
        for node, sign in ((e.child, 1.0), (e.parent, -1.0)):
            if node == "s0":
                continue
            c = 3 * (names.index(node) - 1)
            A[:, c:c + 3] = sign * np.eye(3)
        rows.append(W @ A)
        rhs.append(W @ e.measurement[:3, 3])
    x, *_ = np.linalg.lstsq(np.concatenate(rows), np.concatenate(rhs), rcond=None)
    closed = {"s0": np.zeros(3)}
    for k, n in enumerate(names[1:]):
        closed[n] = x[3 * k:3 * k + 3]
    gap = max(float(np.linalg.norm(got[n][:3, 3] - closed[n])) for n in names)
    check("A4b", gap <= 1e-4,
          f"with rotation clamped at 1e-5 rad the solver reaches the closed-form "
          f"translation least squares to {gap:.2e} m (bar 1e-4; the "
          f"pre-registered 1e-9 was unreachable and why is recorded in the code)")
    NOTES.append(
        "A4 was substituted: the pre-registered closed-form check assumed "
        "translation and rotation decouple. They do not — d(e_t)/d(omega) = "
        "skew(R t_z) is the lever arm that trades yaw against horizontal "
        "position, which is the same coupling that makes sigma_yaw matter. A4 "
        "is now a second, independently written minimiser; A4b keeps the closed "
        "form as a clamped limit.")


# --- A5 ---------------------------------------------------------------------

def a5_jacobian() -> None:
    print("\nA5  the analytic Jacobian is the derivative it claims to be")
    rng = np.random.default_rng(40_000)
    _, names, edges, tree_edges = build(rng, noise=PINNED)
    poses = pg.compose(tree_edges, "s0")
    poses = jostle(poses, "zzz", rng, metres=0.3, degrees=10.0)
    worst = 0.0
    for e in edges:
        r0, Ji, Jj = pg.edge_jacobians(poses, e)
        for J, node in ((Ji, e.parent), (Jj, e.child)):
            numeric = np.zeros((6, 6))
            for c in range(6):
                h = 1e-6
                plus, minus = {}, {}
                for sign, store in ((+1, plus), (-1, minus)):
                    store.update({k: v.copy() for k, v in poses.items()})
                    d = np.zeros(6)
                    d[c] = sign * h
                    store[node][:3, :3] = pg.so3_exp(d[:3]) @ store[node][:3, :3]
                    store[node][:3, 3] = store[node][:3, 3] + d[3:]
                numeric[:, c] = (pg.residual(plus, e) - pg.residual(minus, e)) / (2 * h)
            denom = max(float(np.abs(numeric).max()), 1e-9)
            worst = max(worst, float(np.abs(numeric - J).max()) / denom)
    check("A5", worst <= 1e-6,
          f"worst analytic-vs-finite-difference error {worst:.2e}, relative to the "
          f"column scale (bar 1e-6)")


# --- A6 ---------------------------------------------------------------------

def _sweep_displacement(builder, variants) -> tuple[float, dict, float]:
    reference = None
    worst = worst_h = 0.0
    per = {}
    for label, weights in variants:
        rng = np.random.default_rng(50_000)
        _, names, edges, tree_edges = builder(rng, weights)
        got = pg.optimise(edges, "s0", initial=pg.compose(tree_edges, "s0"),
                          iterations=200).poses
        if reference is None:
            reference = got
            per[label] = 0.0
            continue
        step = [got[n][:3, 3] - reference[n][:3, 3] for n in names]
        per[label] = max(float(np.linalg.norm(d)) for d in step)
        worst = max(worst, per[label])
        worst_h = max(worst_h, max(float(np.linalg.norm(d[[0, 2]])) for d in step))
    return worst, per, worst_h


def a6_anisotropy() -> None:
    print("\nA6  P1 and P2, the predictions about the anisotropic information matrix")

    def flat(rng, weights):
        """Gravity-aligned: no tilt on any node, no tilt in any measurement."""
        true_poses = [pose(rng.uniform(0, 2 * math.pi), np.zeros(3),
                           (rng.random(3) - 0.5) * np.array([6.0, 0.4, 6.0]))
                      for _ in range(6)]
        names = [f"s{k}" for k in range(6)]
        edges, tree = [], []
        for i, j in RING + CHORDS:
            eps = draw(rng, PINNED)
            eps[0] = eps[2] = 0.0
            e = pg.edge_from_score(names[i], names[j], measure(true_poses, i, j, eps),
                                   0.05, **weights)
            edges.append(e)
            if (i, j) in RING:
                tree.append(e)
        return true_poses, names, edges, tree

    def tilted(rng, weights):
        true_poses = truth(rng, 6, tilt_deg=5.0)
        names = [f"s{k}" for k in range(6)]
        edges, tree = [], []
        for i, j in RING + CHORDS:
            eps = draw(rng, dict(PINNED, sigma_tilt=math.radians(5.0)))
            e = pg.edge_from_score(names[i], names[j], measure(true_poses, i, j, eps),
                                   0.05, **weights)
            edges.append(e)
            if (i, j) in RING:
                tree.append(e)
        return true_poses, names, edges, tree

    variants = [("aniso 5.21", PINNED), ("iso 1.0", dict(PINNED, sigma_v=pg.SIGMA_H))]
    flat_move, _, flat_h = _sweep_displacement(flat, variants)
    tilt_move, _, _ = _sweep_displacement(tilted, variants)
    check("A6/P1", flat_move <= 1e-3,
          f"gravity-aligned graph: dropping sigma_h/sigma_v from 5.21 to 1.0 moves "
          f"the worst node {flat_move * 1000:.3f} mm (bar 1 mm)")
    # The bar is not moved. What the split says is that the derivation behind P1
    # was right about the plane it named and blind to the one it did not: sigma_v
    # also sets the price at which *tilt* is recruited to absorb a vertical
    # residual, and tilt moves heights over a lever arm of metres.
    print(f"        of that {flat_move * 1000:.3f} mm, {flat_h * 1000:.3f} mm is "
          f"horizontal — P1 holds in the plane it was derived for and fails "
          f"vertically")
    check("A6/P2", tilt_move > 1e-3,
          f"5-degree-tilted graph: the same change moves the worst node "
          f"{tilt_move * 1000:.3f} mm (bar >1 mm, i.e. the anisotropy is not inert)")


# --- A7, A8 -----------------------------------------------------------------

def a7_a8_sensitivity() -> None:
    print("\nA7/A8  how much the answer depends on the two uncalibrated choices")

    def graph(rng, weights):
        return build(rng, noise=PINNED, weights=weights)

    # A7 needs edges of *differing* score, or the exponent cancels exactly.
    scores = [0.03, 0.05, 0.08, 0.11, 0.14, 0.20, 0.06, 0.17]

    def scored(rng, weights):
        true_poses = truth(rng, 6)
        names = [f"s{k}" for k in range(6)]
        edges, tree = [], []
        for k, (i, j) in enumerate(RING + CHORDS):
            e = pg.edge_from_score(names[i], names[j],
                                   measure(true_poses, i, j, draw(rng, PINNED)),
                                   scores[k], **weights)
            edges.append(e)
            if (i, j) in RING:
                tree.append(e)
        return true_poses, names, edges, tree

    p_variants = [(f"p={p}", dict(PINNED, exponent=float(p))) for p in (2, 0, 1, 3)]
    p_move, p_per, _ = _sweep_displacement(scored, p_variants)
    print(f"        exponent sweep, displacement from p=2: "
          + ", ".join(f"{k} {v * 100:.2f} cm" for k, v in p_per.items()))
    check("A7 (synthetic, reported not barred)", True,
          f"worst node displacement across p in 0..3 is {p_move * 100:.2f} cm")

    y_variants = [(f"yaw={d}", dict(PINNED, sigma_yaw=math.radians(d)))
                  for d in (2.4, 0.6, 1.2, 4.8)]
    y_move, y_per, _ = _sweep_displacement(graph, y_variants)
    print(f"        sigma_yaw sweep, displacement from 2.4 deg: "
          + ", ".join(f"{k} {v * 100:.2f} cm" for k, v in y_per.items()))
    check("A8 (synthetic, reported not barred)", True,
          f"worst node displacement across sigma_yaw 0.6-4.8 deg is "
          f"{y_move * 100:.2f} cm")


# --- A10 --------------------------------------------------------------------

def a10_unjudgeable() -> None:
    print("\nA10  an edge nobody could judge is refused, not given a default weight")
    raised = []
    for bad in (None, float("nan"), 0.0, -0.1):
        try:
            pg.information(bad)
            raised.append(f"{bad!r} was accepted")
        except pg.UnjudgeableEdge:
            pass
    check("A10a", not raised, "score None/nan/0/negative all raise UnjudgeableEdge"
          if not raised else "; ".join(raised))

    rng = np.random.default_rng(60_000)
    true_poses = truth(rng, 3)
    names = ["s0", "s1", "s2"]
    good = pg.edge_from_score("s0", "s1", measure(true_poses, 0, 1, np.zeros(6)), 0.05)
    try:
        pg.optimise([good], "s0", initial={"s0": np.eye(4),
                                           "s1": np.eye(4), "s2": np.eye(4)})
        ok, detail = False, "s2 was placed although no admitted edge reaches it"
    except pg.GraphNotConnected as exc:
        ok, detail = "s2" in str(exc), str(exc).split(":")[0]
    check("A10b", ok, detail)

    result = pg.optimise([good], "s0", strict=False,
                         initial={"s0": np.eye(4), "s1": np.eye(4), "s2": np.eye(4)})
    check("A10c", result.unplaced == ["s2"] and "s2" not in result.poses,
          f"non-strict run reports unplaced={result.unplaced} and does not emit a "
          f"pose for it")


def main() -> int:
    print("acceptance for tools/pose_graph.py "
          "(bars fixed in docs/POSE_GRAPH_PREREG.md before the solver existed)")
    a1_exact_recovery()
    a2_loop_closure()
    a3_chain_untouched()
    a4_independent()
    a5_jacobian()
    a6_anisotropy()
    a7_a8_sensitivity()
    a10_unjudgeable()
    for note in NOTES:
        print(f"\nnote: {note}")
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED")
        for f in FAILURES:
            print(f"  {f}")
        return 1
    print("all acceptance criteria passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
