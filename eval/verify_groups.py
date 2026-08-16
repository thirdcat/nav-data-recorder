#!/usr/bin/env python3
"""Cut the spanning tree where the photographs disagree, and report what is left.

`tools/align_set.py` groups sessions by which edges its geometric gate admits.
The corpus sweep in `docs/3DGS.md` shows that gate cannot referee itself: over
eleven tree edges, fitness ranges 0.51-0.85 on the real ones and 0.21-0.63 on
the false, and the ratio criterion is worse still — the default rejects two real
pairs of four while admitting five false ones. Only the photometric score
separates them, 0.04-0.15 against 0.35-1.03, and it is the one signal that never
touches the geometry being judged.

Scoring every admitted edge is too slow. The **tree** is what actually placed
each session, so scoring its N-1 edges and cutting the false ones is enough to
answer the question that matters: which of these walks are really the same room.

    python3 eval/verify_groups.py transforms.json

The result is conservative by construction. Cutting a tree edge separates two
components that some *unverified* edge might legitimately rejoin, so a split
here means "not established", not "proven different".
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import photo_align as pa  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "tools"))
from export_3dgs import camera_to_world  # noqa: E402
from read_session import Session  # noqa: E402

# Measured on the corpus: real pairs 0.0435-0.1485, false 0.3458-1.0270. The
# threshold sits in the empty middle rather than beside either population.
COHERENT = 0.25


def score_edge(a_dir: str, b_dir: str, transform: np.ndarray, limit: int = 24) -> dict:
    a_session, b_session = Session(a_dir), Session(b_dir)
    a_rows = [r for r in a_session.posed_images() if r.get("depth") is not None]
    b_rows = [r for r in b_session.posed_images() if r.get("depth") is not None]
    lite = lambda rows: [{"pose": camera_to_world(r["pose"])} for r in rows]
    pairs = pa.overlapping_pairs(lite(a_rows), lite(b_rows), transform, limit=limit)
    if not pairs:
        return {"pairs": 0, "score": None, "verdict": "no-overlap"}

    a_b, b_b = {}, {}
    for i, j in pairs:
        a_b.setdefault(i, pa.frame_bundle(a_session, a_rows[i]))
        b_b.setdefault(j, pa.frame_bundle(b_session, b_rows[j]))
    pairs = [(i, j) for i, j in pairs if a_b.get(i) is not None and b_b.get(j) is not None]
    if len(pairs) < 3:
        return {"pairs": len(pairs), "score": None, "verdict": "too-few-usable"}

    got = [pa.reproject_score(a_b[i], b_b[j], transform)[0] for i, j in pairs]
    got = [v for v in got if np.isfinite(v)]
    if not got:
        return {"pairs": len(pairs), "score": None, "verdict": "unscorable"}
    score = float(np.median(got))
    return {"pairs": len(pairs), "score": score,
            "verdict": "keep" if score <= COHERENT else "cut"}


def components(names: list[str], edges: list[tuple[str, str]]) -> list[list[str]]:
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    out: dict[str, list[str]] = {}
    for n in names:
        out.setdefault(find(n), []).append(n)
    return sorted(out.values(), key=len, reverse=True)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("transforms")
    ap.add_argument("--sessions-root", default=str(Path.home() / "nav_data"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    data = json.loads(Path(a.transforms).read_text())
    tree = data.get("tree")
    if not tree:
        print("this JSON carries no tree; re-run align_set.py after d33a64b")
        return 1
    root = Path(a.sessions_root)
    resolve = lambda name: (str(root / name) if (root / name).is_dir()
                            else (glob.glob(str(root / f"*{name[-6:]}")) or [None])[0])

    placed = list(data["transforms"])
    transforms = {k: np.asarray(v, dtype=float) for k, v in data["transforms"].items()}

    print(f"{len(tree)} tree edges to check "
          f"(scoring every admitted edge would be far slower and answers the "
          f"same question)\n")
    print(f"{'edge':>20} {'fitness':>8} {'pairs':>6} {'photometric':>12}  verdict")
    kept, cut = [], []
    for edge in tree:
        parent_id, child_id = edge["parent"], edge["child"]
        pa_dir, ch_dir = resolve(parent_id), resolve(child_id)
        if not pa_dir or not ch_dir:
            print(f"{parent_id[-6:]}<-{child_id[-6:]:>8}  session directory not found")
            continue
        # the child's placement expressed in the parent's frame
        relative = np.linalg.inv(transforms[parent_id]) @ transforms[child_id]
        got = score_edge(pa_dir, ch_dir, relative)
        s = "-" if got["score"] is None else f"{got['score']:.4f}"
        print(f"{parent_id[-6:]}<-{child_id[-6:]:>8} {edge['edge_fitness']:>8.2f} "
              f"{got['pairs']:>6} {s:>12}  {got['verdict']}")
        (kept if got["verdict"] == "keep" else cut).append((parent_id, child_id, got))

    groups = components(placed, [(p, c) for p, c, _ in kept])
    print(f"\n{len(kept)} of {len(tree)} edges survive; the tree falls into "
          f"{len(groups)} components")
    for k, g in enumerate(groups):
        print(f"  group {k}: " + ", ".join(x[-6:] for x in g))
    print("\n  a split means the pairing is not established, not that the rooms")
    print("  differ: an edge the tree never used could legitimately rejoin two.")

    if a.out:
        Path(a.out).write_text(json.dumps({
            "source": str(a.transforms),
            "threshold": COHERENT,
            "kept": [{"parent": p, "child": c, **g} for p, c, g in kept],
            "cut": [{"parent": p, "child": c, **g} for p, c, g in cut],
            "groups": groups,
        }, indent=1))
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
