#!/usr/bin/env python3
"""Turn `align_set.py`'s spanning tree into a pose graph, and optimise it.

`tools/align_set.py` keeps a maximum spanning tree, so every session reaches the
reference by one path and every measurement off that path is thrown away. This
puts the discarded measurements back and solves for all the poses at once.

    python3 eval/close_loops.py transforms.json --out closed.json

Three things happen, and the order is the point.

1. **Every tree edge is scored photometrically**, by the same `score_edge` that
   `eval/verify_groups.py` uses — the same function, not a copy — so the verdict
   this tool starts from is bit-identical to the published one. Four of eleven
   survive on the corpus; that is a fact this tool must reproduce and not
   quietly move.

2. **Chords are proposed by geometry and admitted by photographs.** A chord is a
   pair the tree never used. Aligning every one is far too slow, so the
   co-observation count `align_set.py` already wrote — poses only, no images —
   proposes them, `align_sessions.align_clouds` measures them, and the
   photometric score decides. Geometry never decides.

3. **The graph is optimised per connected component.** A component is what the
   photographs established; a chord that joins two of them is a change to the
   published grouping and is printed as one.

It lives in `eval/` because it reads photographs, and `tools/` is kept on numpy
alone. The solver itself is `tools/pose_graph.py` and has no photographs in it.

No GPU. Nothing here trains or renders, so nothing here demonstrates a PSNR or a
coverage improvement, and the report says so.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_groups as vg  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "tools"))
import align_sessions as al  # noqa: E402
import pose_graph as pgraph  # noqa: E402


def resolve(root: Path, name: str) -> str | None:
    if (root / name).is_dir():
        return str(root / name)
    hit = glob.glob(str(root / f"*{name[-6:]}"))
    return hit[0] if hit else None


def load_cache(path: str | None) -> dict:
    if path and Path(path).exists():
        return json.loads(Path(path).read_text())
    return {}


def save_cache(path: str | None, cache: dict) -> None:
    if path:
        Path(path).write_text(json.dumps(cache, indent=1))


def measure_chord(clouds: dict, cache: dict, a: str, b: str, dirs: dict,
                  build: dict) -> np.ndarray | None:
    """The relative transform of a pair the tree never used, measured directly.

    Composing it through the tree would be circular: the loop would close by
    construction and there would be nothing to optimise.
    """
    key = f"{a}->{b}"
    if key in cache:
        return np.asarray(cache[key]["transform"], dtype=float)
    for name in (a, b):
        if name not in clouds:
            clouds[name] = al.session_cloud(dirs[name], **build)
    try:
        got = al.align_clouds(clouds[a], clouds[b])
    except ValueError as exc:
        cache[key] = {"transform": None, "error": str(exc)}
        return None
    cache[key] = {"transform": np.asarray(got["transform"]).tolist(),
                  "fitness_5cm": got["fitness"]["5cm"],
                  "tilt_deg": got["tilt_deg"]}
    return np.asarray(got["transform"], dtype=float)


def components(names, pairs):
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    out: dict[str, list[str]] = {}
    for n in names:
        out.setdefault(find(n), []).append(n)
    return sorted(out.values(), key=len, reverse=True)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("transforms", help="the JSON align_set.py wrote")
    ap.add_argument("--sessions-root", default=str(Path.home() / "nav_data"))
    ap.add_argument("--min-co-observing", type=int, default=50,
                    help="a chord is only proposed if this many frames of one "
                         "session sat near a frame of the other facing the same "
                         "way. Poses only, so it costs nothing, and it is a "
                         "proposal — the photographs still decide")
    ap.add_argument("--pairs", type=int, default=24,
                    help="frame pairs per photometric score; the same default "
                         "eval/verify_groups.py uses")
    ap.add_argument("--voxel", type=float, default=0.05)
    ap.add_argument("--pix-stride", type=int, default=2)
    ap.add_argument("--conf-min", type=int, default=2)
    ap.add_argument("--sigma-yaw-deg", type=float, default=math.degrees(pgraph.SIGMA_YAW),
                    help="NOT MEASURED anywhere; see docs/POSE_GRAPH_PREREG.md")
    ap.add_argument("--sigma-ratio", type=float, default=pgraph.SIGMA_H / pgraph.SIGMA_V,
                    help="horizontal sigma over vertical sigma; 5.21 is the "
                         "geometric mean of the three genuine refinement pairs")
    ap.add_argument("--weight-exponent", type=float, default=pgraph.WEIGHT_EXPONENT)
    ap.add_argument("--cache", default=None,
                    help="reuse geometric chord alignments across runs")
    ap.add_argument("--no-chords", action="store_true",
                    help="tree edges only: the arm that must reproduce "
                         "eval/verify_groups.py exactly")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    data = json.loads(Path(a.transforms).read_text())
    root = Path(a.sessions_root)
    placed = list(data["transforms"])
    T = {k: np.asarray(v, dtype=float) for k, v in data["transforms"].items()}
    tree = data.get("tree") or []
    if not tree:
        print("this JSON carries no tree; re-run align_set.py")
        return 1
    dirs = {n: resolve(root, n) for n in placed}
    missing = [n for n, d in dirs.items() if not d]
    if missing:
        print(f"session directory not found for: {', '.join(m[-6:] for m in missing)}")
        return 1

    weights = dict(sigma_yaw=math.radians(a.sigma_yaw_deg),
                   sigma_h=pgraph.SIGMA_H,
                   sigma_v=pgraph.SIGMA_H / a.sigma_ratio,
                   exponent=a.weight_exponent)

    # --- 1. the tree edges, scored by the same code as the published verdict ---
    print(f"{len(tree)} tree edges, scored by eval/verify_groups.score_edge — the "
          f"same function, so this must reproduce the published verdict\n")
    print(f"{'edge':>20} {'fitness':>8} {'pairs':>6} {'photometric':>12}  verdict")
    scored: list[dict] = []
    for edge in tree:
        p, c = edge["parent"], edge["child"]
        relative = np.linalg.inv(T[p]) @ T[c]
        got = vg.score_edge(dirs[p], dirs[c], relative, limit=a.pairs)
        s = "-" if got["score"] is None else f"{got['score']:.4f}"
        print(f"{p[-6:]}<-{c[-6:]:>8} {edge['edge_fitness']:>8.2f} "
              f"{got['pairs']:>6} {s:>12}  {got['verdict']}")
        scored.append({"parent": p, "child": c, "kind": "tree",
                       "fitness": edge["edge_fitness"],
                       "measurement": relative, **got})
    kept_tree = [e for e in scored if e["verdict"] == "keep"]
    print(f"\n{len(kept_tree)} of {len(tree)} tree edges survive")

    # --- 2. chords: geometry proposes, photographs decide ---------------------
    chords: list[dict] = []
    if not a.no_chords:
        co = data.get("co_observing", {})
        best: dict[tuple[str, str], int] = {}
        on_tree = {(e["parent"], e["child"]) for e in tree}
        on_tree |= {(b, x) for x, b in on_tree}
        for key, n in co.items():
            x, y = key.split("->")
            if x not in T or y not in T or (x, y) in on_tree or x == y:
                continue
            pair = (x, y) if x < y else (y, x)
            if n > best.get(pair, -1):
                best[pair] = n
        candidates = sorted(((n, p) for p, n in best.items()
                             if n >= a.min_co_observing), reverse=True)
        print(f"\n{len(candidates)} chords proposed at co-observation >= "
              f"{a.min_co_observing} (poses only, no images)\n")
        print(f"{'chord':>20} {'co-obs':>7} {'fitness':>8} {'pairs':>6} "
              f"{'photometric':>12}  verdict")
        cache = load_cache(a.cache)
        clouds: dict[str, dict] = {}
        build = dict(voxel=a.voxel, pix_stride=a.pix_stride, conf_min=a.conf_min)
        for n_obs, (x, y) in candidates:
            t0 = time.time()
            Z = measure_chord(clouds, cache, x, y, dirs, build)
            save_cache(a.cache, cache)
            if Z is None:
                print(f"{x[-6:]}<-{y[-6:]:>8} {n_obs:>7} "
                      f"{'unalignable — no edge exists':>42}")
                continue
            fitness = cache[f"{x}->{y}"].get("fitness_5cm", float("nan"))
            got = vg.score_edge(dirs[x], dirs[y], Z, limit=a.pairs)
            s = "-" if got["score"] is None else f"{got['score']:.4f}"
            print(f"{x[-6:]}<-{y[-6:]:>8} {n_obs:>7} {fitness:>8.2f} "
                  f"{got['pairs']:>6} {s:>12}  {got['verdict']}"
                  f"   ({time.time() - t0:.0f}s)")
            chords.append({"parent": x, "child": y, "kind": "chord",
                           "fitness": fitness, "co_observing": n_obs,
                           "measurement": Z, **got})

    kept_chords = [e for e in chords if e["verdict"] == "keep"]
    admitted = kept_tree + kept_chords

    # --- 3. what the admitted edges say about the grouping --------------------
    tree_groups = components(placed, [(e["parent"], e["child"]) for e in kept_tree])
    all_groups = components(placed, [(e["parent"], e["child"]) for e in admitted])
    print(f"\ntree edges alone: {len(tree_groups)} components  "
          f"{[len(g) for g in tree_groups]}")
    print(f"with chords:      {len(all_groups)} components  "
          f"{[len(g) for g in all_groups]}")
    if len(all_groups) != len(tree_groups):
        print("\n  ** THE PUBLISHED GROUPING HAS CHANGED. A chord the tree never "
              "used has\n     joined components the verified tree left apart. "
              "docs/3DGS.md says this\n     is legitimate — \"an edge the tree "
              "never used could rejoin two\" — but it\n     is a change and it is "
              "reported, not absorbed:")
        for e in kept_chords:
            before = [g for g in tree_groups if e["parent"] in g][0]
            if e["child"] not in before:
                print(f"     {e['parent'][-6:]} <- {e['child'][-6:]}  "
                      f"photometric {e['score']:.4f}, fitness {e['fitness']:.2f}, "
                      f"{e['co_observing']} co-observing frames")

    # --- 4. optimise each component ------------------------------------------
    report = {"components": [], "edges": [
        {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in e.items()}
        for e in scored + chords]}
    out_transforms: dict[str, list] = {}
    print()
    for group in all_groups:
        if len(group) < 2:
            continue
        inside = [e for e in admitted if e["parent"] in group and e["child"] in group]
        chord_count = sum(1 for e in inside if e["kind"] == "chord")
        loops = len(inside) - (len(group) - 1)
        root = next(n for n in placed if n in group)
        # Tree edges first, so the composition BFS prefers the path align_set
        # actually used and the "before" arm is genuinely the tree's answer.
        ordered = ([e for e in inside if e["kind"] == "tree"]
                   + [e for e in inside if e["kind"] == "chord"])
        edges = [pgraph.edge_from_score(e["parent"], e["child"], e["measurement"],
                                        e["score"], fitness=e["fitness"], **weights)
                 for e in ordered]
        # The "before" arm has to be the tree's answer and nothing else. Handed
        # every edge at once the traversal reaches a node down a chord, and the
        # comparison then measures the optimiser against itself.
        tree_only = [k for k, e in zip(edges, ordered) if e["kind"] == "tree"]
        before = pgraph.compose(edges, root, seed=pgraph.compose(tree_only, root))
        drift = max(float(np.linalg.norm(
            before[n][:3, 3] - (np.linalg.inv(T[root]) @ T[n])[:3, 3]))
            for n in group if n in T)
        if drift > 1e-9:
            print(f"  ! the reconstructed tree composition differs from what "
                  f"align_set.py wrote by {drift * 100:.3f} cm — the baseline is "
                  f"not the published one")
        result = pgraph.optimise(edges, root, initial=before, iterations=200)

        print(f"component rooted at {root[-6:]}: {len(group)} sessions, "
              f"{len(inside)} edges ({chord_count} chord), "
              f"{loops} independent loop{'s' if loops != 1 else ''}")
        if loops <= 0:
            print("  no loop here: the optimum is the tree composition and the "
                  "solver returns it unchanged.")
        print(f"  chi2 {result.chi2_before:.4f} -> {result.chi2_after:.4f} "
              f"in {result.iterations} iterations"
              f"{'' if result.converged else '  (did NOT converge)'}")
        print(f"  {'session':>8} {'moved':>10} {'horizontal':>12} {'vertical':>10}")
        moves = []
        for name in group:
            d = result.poses[name][:3, 3] - before[name][:3, 3]
            moves.append(float(np.linalg.norm(d)))
            print(f"  {name[-6:]:>8} {np.linalg.norm(d) * 100:>8.2f} cm "
                  f"{np.linalg.norm(d[[0, 2]]) * 100:>10.2f} cm "
                  f"{abs(d[1]) * 100:>8.2f} cm")
        print(f"  {'edge':>20} {'score':>8} {'|rot| before':>13} {'after':>8} "
              f"{'|trans| before':>15} {'after':>9}")
        for e, edge in zip(ordered, edges):
            rb = pgraph.residual(before, edge)
            ra = pgraph.residual(result.poses, edge)
            print(f"  {e['parent'][-6:]}<-{e['child'][-6:]:>8} {e['score']:>8.4f} "
                  f"{math.degrees(np.linalg.norm(rb[:3])):>11.3f} d "
                  f"{math.degrees(np.linalg.norm(ra[:3])):>6.3f} d "
                  f"{np.linalg.norm(rb[3:]) * 100:>12.2f} cm "
                  f"{np.linalg.norm(ra[3:]) * 100:>6.2f} cm")
        report["components"].append({
            "root": root, "sessions": group, "edges": len(inside),
            "chords": chord_count, "loops": loops,
            "chi2_before": result.chi2_before, "chi2_after": result.chi2_after,
            "converged": result.converged,
            "moved_m": {n: float(np.linalg.norm(
                result.poses[n][:3, 3] - before[n][:3, 3])) for n in group},
            "max_move_m": max(moves),
        })
        for name in group:
            out_transforms[name] = result.poses[name].tolist()
        print()

    alone = [g[0] for g in all_groups if len(g) == 1]
    if alone:
        print(f"{len(alone)} session(s) reach nothing: "
              f"{', '.join(x[-6:] for x in alone)}")
        print("  not placed at what the tree said — not placed. Every edge that "
              "could\n  have reached them was cut by the photographs or could not "
              "be judged.\n")

    print("A coverage or PSNR improvement has NOT been demonstrated by this run. "
          "Nothing\nhere trains or renders. The numbers above are metres of "
          "disagreement removed\nbetween measurements, and loop closure removes "
          "inconsistency between paths,\nnot a bias every edge shares — and "
          "docs/3DGS.md measures the horizontal error\nof this estimator to be "
          "largely the latter.")

    if a.out:
        Path(a.out).write_text(json.dumps({
            "reference": data.get("reference"),
            "source": str(a.transforms),
            "convention": "world_from_session, each component in its own root's "
                          "world; components are NOT in one frame",
            "coherent_threshold": vg.COHERENT,
            "sigma": {"h": weights["sigma_h"], "v": weights["sigma_v"],
                      "yaw_deg": a.sigma_yaw_deg,
                      "tilt_deg": math.degrees(pgraph.SIGMA_TILT),
                      "weight_exponent": a.weight_exponent},
            "transforms": out_transforms,
            "unplaced": alone,
            **report,
        }, indent=1))
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
