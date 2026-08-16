#!/usr/bin/env python3
"""Put a set of sessions into one frame, and write the transforms out.

The pairwise tool answers whether two fragments can be joined. This one answers
the question the thermal ceiling actually asks: a room takes several recordings,
and they have to become one map together.

    python3 tools/align_set.py ~/nav_data/a ~/nav_data/b ~/nav_data/c
    python3 tools/align_set.py ~/nav_data/*/ --reference 2 --out transforms.json

The output is a JSON file of 4x4 matrices keyed by session id, in the reference
session's frame — which `export_3dgs.py --merge` reads to build one training set
out of several walks.

Read the two fitness columns as a pair. `direct` is how well a session aligns to
the reference on its own; `composed` is how well it lands after being carried
through the tree. A session with low direct and high composed reached the
reference through a neighbour, which is the whole point of the tree. A session
where composed is much *worse* than direct is a warning: the chain drifted, and
that session would be better re-rooted.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from align_sessions import align_set  # noqa: E402


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--reference", type=int, default=0,
                    help="index into the session list; its frame is the output frame")
    ap.add_argument("--min-fitness", type=float, default=0.20,
                    help="absolute floor, so a row of uniform rubbish cannot "
                         "elect a winner")
    ap.add_argument("--min-ratio", type=float, default=2.0,
                    help="how far an edge must stand out from the other "
                         "candidates in its own row, which are its controls")
    ap.add_argument("--voxel", type=float, default=0.05)
    ap.add_argument("--pix-stride", type=int, default=2)
    ap.add_argument("--conf-min", type=int, default=2)
    ap.add_argument("--min-co-observing", type=int, default=4,
                    help="an edge needs this many frames of one session sitting "
                         "near a frame of the other and facing the same way. "
                         "Costs nothing — poses only — and is a rejection, not "
                         "an admission: real pairs run 28-184, but false ones "
                         "reach 44, so only the near-zero end is decisive")
    ap.add_argument("--out", default=None, help="write the transforms here")
    a = ap.parse_args(argv)

    sessions = [d.rstrip("/") for d in a.sessions]
    result = align_set(sessions, reference=a.reference, min_fitness=a.min_fitness,
                       min_ratio=a.min_ratio, voxel=a.voxel,
                       pix_stride=a.pix_stride, conf_min=a.conf_min,
                       min_co_observing=a.min_co_observing)

    print(f"reference {result['reference'][-6:]}, "
          f"{result['placed']} of {result['of']} sessions placed")
    for note in result.get("rejected", []):
        print(f"  unusable: {note}")
    bad = result.get("unalignable", [])
    if bad:
        print(f"  {len(bad)} ordered pairs could not be aligned at all "
              f"(no edge, not a low score); first few:")
        for note in bad[:3]:
            print(f"    {note}")
    if result.get("over_connected"):
        print(f"\n  ! {result['admitted_edges']} edges admitted among "
              f"{result['of']} sessions, and one group holds "
              f"{len(result['groups'][0])} of them. That is over-connection, not "
              f"a large room.\n    The thresholds are calibrated at the default "
              f"--pix-stride; a coarser one moves the whole fitness scale.\n"
              f"    Re-establish them against a pair you know before trusting "
              f"these groups.")
    if len(result["groups"]) > 1:
        print("\nthese sessions are not all of one space:")
        for k, group in enumerate(result["groups"]):
            note = "  <- the reference is here" if result["reference"] in group else ""
            print(f"  group {k}: {', '.join(s[-6:] for s in group)}{note}")
        print("  a group is a set connected by admissible edges. Align each "
              "separately,\n  or pass --reference into the group you want.")
    elif result["unplaced"]:
        print(f"  unplaced: {', '.join(s[-6:] for s in result['unplaced'])}")

    print("\npairwise fitness at 5 cm — the separation is the evidence")
    ids = sorted({k.split('->')[0] for k in result["pairwise"]})
    print("        " + "".join(f"{i:>8}" for i in ids))
    for row in ids:
        cells = []
        for col in ids:
            if row == col:
                cells.append(f"{'-':>8}")
            else:
                cells.append(f"{result['pairwise'].get(f'{row}->{col}', 0):>8.2f}")
        print(f"{row:>8}" + "".join(cells))

    print("\nratio to the rest of the row — the row is its own control set")
    print("        " + "".join(f"{i:>8}" for i in ids))
    for row in ids:
        cells = [f"{'-':>8}" if row == col
                 else f"{result['ratios'].get(f'{row}->{col}', 0):>8.1f}"
                 for col in ids]
        print(f"{row:>8}" + "".join(cells))

    print("\ntree")
    for edge in result["tree"]:
        print(f"  {edge['parent'][-6:]} -> {edge['child'][-6:]}  "
              f"fitness {edge['edge_fitness']:.2f}, {edge['edge_ratio']:.1f}x its row")

    print("\nplacement against the reference cloud itself")
    print(f"  {'session':>8} {'direct':>8} {'composed':>9}")
    for session in result["sessions"]:
        flag = ""
        if session["composed_fitness_5cm"] < session["direct_fitness_5cm"] - 0.05:
            flag = "   <- the chain lost accuracy; consider re-rooting"
        print(f"  {session['id'][-6:]:>8} {session['direct_fitness_5cm']:>8.2f} "
              f"{session['composed_fitness_5cm']:>9.2f}{flag}")

    if a.out:
        payload = {
            "reference": result["reference"],
            "min_fitness": result["min_fitness"],
            "convention": "world_from_session: multiply a session's ARKit world "
                          "point by this to land in the reference session's world",
            "transforms": {k: v.tolist() for k, v in result["transforms"].items()},
            # The tree is what actually placed each session, so a verifier that
            # wants to test the edges rather than the composition needs it here
            # and not only in the printed report.
            "tree": result["tree"],
            "groups": result["groups"],
            "co_observing": {f"{a}->{b}": v for (a, b), v in
                             result.get("co_observing_pairs", {}).items()},
        }
        with open(a.out, "w") as fh:
            json.dump(payload, fh, indent=1)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
