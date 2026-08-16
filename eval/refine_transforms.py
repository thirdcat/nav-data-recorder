#!/usr/bin/env python3
"""Refine a geometric alignment with the photographs, as a pipeline stage.

`tools/align_set.py` puts several walks in one frame with point-to-plane ICP and
writes the transforms. That alignment is good in the direction the floor
constrains and weak in the two it does not: measured on four pairs, the
correction the photographs then demand is **3.6 to 7.8 times larger horizontally
than vertically**, and on three of the four it is 14 to 29 cm. `docs/3DGS.md`
carries the numbers.

So this reads that JSON, refines every placed session against the reference by
descending the photometric score, and writes the same schema back. It lives in
`eval/` rather than `tools/` because it needs to read photographs, and `tools/`
is kept on numpy alone.

    python3 eval/refine_transforms.py transforms.json --out refined.json

The score also separates a real pairing from a false one by an order of
magnitude — genuine pairs land at 0.02-0.08, deliberately wrong ones at
0.49-0.70 — so a session that will not come down is reported rather than
quietly written out with a plausible transform.
"""
from __future__ import annotations

import argparse
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

# Above this, the photographs of the two sessions do not agree at any alignment.
# The gap in the corpus is wide: 0.082 for the worst genuine pair against 0.486
# for the best deliberately wrong one.
INCOHERENT = 0.30


def refine_one(reference_dir: str, source_dir: str, transform: np.ndarray, *,
               span: float, passes: int, pairs: int) -> dict:
    ref_session, src_session = Session(reference_dir), Session(source_dir)
    ref_rows = [r for r in ref_session.posed_images() if r.get("depth") is not None]
    src_rows = [r for r in src_session.posed_images() if r.get("depth") is not None]
    lite = lambda rows: [{"pose": camera_to_world(r["pose"])} for r in rows]
    chosen = pa.overlapping_pairs(lite(ref_rows), lite(src_rows), transform, limit=pairs)
    if not chosen:
        return {"status": "no-overlap"}

    ref_bundles, src_bundles = {}, {}
    for i, j in chosen:
        ref_bundles.setdefault(i, pa.frame_bundle(ref_session, ref_rows[i]))
        src_bundles.setdefault(j, pa.frame_bundle(src_session, src_rows[j]))
    chosen = [(i, j) for i, j in chosen
              if ref_bundles.get(i) is not None and src_bundles.get(j) is not None]
    if len(chosen) < 3:
        return {"status": "too-few-usable-pairs", "pairs": len(chosen)}

    start = [pa.reproject_score(ref_bundles[i], src_bundles[j], transform)[0]
             for i, j in chosen]
    start = [v for v in start if np.isfinite(v)]
    before = float(np.median(start)) if start else float("nan")

    refined, history = pa.refine(ref_bundles, src_bundles, chosen, transform,
                                 span=span, passes=passes)
    after = history[-1]["score"] if history else float("nan")
    shift = refined[:3, 3] - transform[:3, 3]
    reach = span * (2 - 2.0 ** (1 - passes))
    return {
        "status": "incoherent" if after > INCOHERENT else "refined",
        "pairs": len(chosen),
        "score_before": before,
        "score_after": after,
        "shift_m": shift.tolist(),
        "horizontal_cm": float(np.linalg.norm(shift[[0, 2]]) * 100),
        "vertical_cm": float(abs(shift[1]) * 100),
        # A descent that spent its whole budget on an axis reported where it
        # stopped, not where the minimum is.
        "at_bound": [n for n, v in zip("XYZ", shift) if abs(v) > 0.9 * reach],
        "transform": refined.tolist(),
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("transforms", help="the JSON align_set.py wrote")
    ap.add_argument("--sessions-root", default=str(Path.home() / "nav_data"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--span", type=float, default=0.20,
                    help="half-width of the first coordinate-descent bracket; "
                         "0.04 was too narrow and clipped three pairs of four")
    ap.add_argument("--passes", type=int, default=7)
    ap.add_argument("--pairs", type=int, default=24)
    ap.add_argument("--keep-incoherent", action="store_true",
                    help="write a refined transform even for a session whose "
                         "photographs never agree; off by default because that "
                         "is what a false pairing looks like")
    a = ap.parse_args(argv)

    data = json.loads(Path(a.transforms).read_text())
    root = Path(a.sessions_root)
    reference = data["reference"]
    ref_dir = str(root / reference)

    out = dict(data)
    out["transforms"] = dict(data["transforms"])
    out["photometric"] = {}
    print(f"reference {reference[-6:]}\n")
    print(f"{'session':>8} {'pairs':>6} {'before':>8} {'after':>8} "
          f"{'horizontal':>11} {'vertical':>9} {'ratio':>6}")
    for name, matrix in data["transforms"].items():
        T = np.asarray(matrix, dtype=float)
        if name == reference or np.allclose(T, np.eye(4)):
            continue
        report = refine_one(ref_dir, str(root / name), T,
                            span=a.span, passes=a.passes, pairs=a.pairs)
        out["photometric"][name] = report
        if report["status"] not in ("refined", "incoherent"):
            print(f"{name[-6:]:>8} {report['status']:>50}")
            continue
        ratio = report["horizontal_cm"] / max(report["vertical_cm"], 1e-9)
        print(f"{name[-6:]:>8} {report['pairs']:>6} {report['score_before']:>8.4f} "
              f"{report['score_after']:>8.4f} {report['horizontal_cm']:>9.1f}cm "
              f"{report['vertical_cm']:>7.1f}cm {ratio:>6.1f}x")
        if report["at_bound"]:
            print(f"{'':>8} ** ran to the edge of the search on "
                  f"{', '.join(report['at_bound'])} — widen --span")
        if report["status"] == "incoherent":
            print(f"{'':>8} ** the photographs never agree ({report['score_after']:.3f} "
                  f"against {INCOHERENT} for a real pair). This is what a false "
                  f"pairing looks like;")
            print(f"{'':>8}    the geometric gate admitted it anyway. Not written "
                  f"unless --keep-incoherent.")
            if not a.keep_incoherent:
                continue
        out["transforms"][name] = report["transform"]

    if a.out:
        out["convention"] = (data.get("convention", "")
                             + " (refined photometrically by eval/refine_transforms.py)")
        Path(a.out).write_text(json.dumps(out, indent=1))
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
