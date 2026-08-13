#!/usr/bin/env python3
"""Triage a directory of sessions: which are worth analysing, and which are not.

Captures accumulate faster than anyone can look at them — a 30-second walk is
about 200 MB, so a week of collection is tens of gigabytes. Most questions about
a session are answerable in a second from its index files, without touching the
depth blob or a single JPEG, and the answers decide whether the expensive tools
should be pointed at it at all.

    python3 tools/survey_sessions.py ~/nav_data
    python3 tools/survey_sessions.py ~/nav_data --usable      # only the good ones

The column that has cost the most time is `conf`. ARKit's depth is guided by the
colour image, so a dim room returns a full, convincing depth map that is almost
entirely low-confidence; one 30 m session measured 98.4% low and every number
derived from it turned out to be measuring the lighting. Anything under about
50% cannot answer a question about depth.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from read_session import Session  # noqa: E402


def summarise(path: str) -> dict | None:
    try:
        session = Session(path)
    except (FileNotFoundError, ValueError):
        return None

    poses = session.poses()
    if not poses:
        return None
    converged = [p for p in poses if p.get("tracking") == "normal"]

    xyz = np.array([[p["tx"], p["ty"], p["tz"]] for p in converged]) if converged else None
    travelled = (float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())
                 if xyz is not None and len(xyz) > 1 else 0.0)
    duration = poses[-1]["t"] - poses[0]["t"]

    index = session.depth_index()
    span = (index[-1]["t"] - index[0]["t"]) if len(index) > 1 else 0.0
    depth_hz = (len(index) - 1) / span if span > 0 else 0.0

    # Sampled, not exhaustive: the point is triage, and a handful of frames
    # settles a percentage that only ever gets read as "usable or not".
    usable = None
    probes = [e for e in index[::max(1, len(index) // 8)] if e.get("confidenceOffset") is not None]
    if probes:
        counts = np.zeros(3, np.int64)
        for entry in probes[:8]:
            counts += np.bincount(np.asarray(session.confidence_frame(entry)).ravel(),
                                  minlength=3)[:3]
        usable = float((counts[1] + counts[2]) / max(counts.sum(), 1))

    events = session.events()
    focus = session.focus_settle()
    # `.complete` is written by the recorder and deliberately *not* uploaded —
    # `SessionStore.payloadFiles` skips hidden files because the marker belongs
    # to the phone, not to the dataset. So every session that arrives over
    # Wi-Fi lacks it, and judging truncation on the marker alone condemned all
    # of them. The manifest's own `counts` are the check that survives the
    # transfer: they are written after the streams close, so agreeing with them
    # means nothing was lost either on the phone or on the way here.
    counts = session.manifest.get("counts", {}) if session.manifest else {}
    counts_agree = bool(counts) and all(
        sum(1 for _ in session.stream(name)) == expected
        for name, expected in counts.items()
        if name in ("pose", "motion", "location", "heading", "planes", "events"))

    return {
        "id": session.id,
        "complete": session.is_complete,
        "counts_agree": counts_agree,
        "duration": duration,
        "travelled": travelled,
        "speed": travelled / duration if duration > 0 else 0.0,
        "depth_hz": depth_hz,
        "depth_frames": len(index),
        "usable": usable,
        "unconverged": len(poses) - len(converged),
        "focus_settle": (focus["settled_after"] if focus is not None
                          else None),
        "thermal": sum(1 for e in events if e["kind"].startswith("thermal")),
        "breaks": sum(1 for e in events if e["kind"] == "ar.interruptionEnded"),
        "bytes": sum(os.path.getsize(os.path.join(r, f))
                     for r, _, fs in os.walk(path) for f in fs),
    }


def verdict(row: dict) -> str:
    """Why a session is or is not worth the expensive tools."""
    if row["usable"] is not None and row["usable"] < 0.5:
        return "too dark"
    if row["depth_hz"] < 12:
        return "depth too slow"
    if row["travelled"] < 3.0:
        return "barely moved"
    if not row["complete"] and not row["counts_agree"]:
        return "truncated"
    if row["thermal"]:
        return "thermal pause"
    return "ok"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="+", help="directory of sessions, or sessions")
    ap.add_argument("--usable", action="store_true", help="list only sessions that pass")
    args = ap.parse_args(argv)

    candidates: list[str] = []
    for root in args.root:
        root = root.rstrip("/")
        if os.path.exists(os.path.join(root, "manifest.json")):
            candidates.append(root)
            continue
        candidates.extend(os.path.join(root, n) for n in sorted(os.listdir(root))
                          if os.path.isdir(os.path.join(root, n)))

    rows = [r for r in (summarise(p) for p in candidates) if r]
    if not rows:
        print("no sessions found", file=sys.stderr)
        return 1

    print(f"{'session':26} {'dur':>6} {'moved':>7} {'m/s':>5} {'depth':>7} "
          f"{'focus':>7} {'conf':>6} {'size':>7}  verdict")
    shown = 0
    for row in rows:
        note = verdict(row)
        if args.usable and note != "ok":
            continue
        shown += 1
        conf = "  n/a" if row["usable"] is None else f"{row['usable'] * 100:4.0f}%"
        focus = "  n/a" if row["focus_settle"] is None else f"{row['focus_settle']:5.1f}s"
        print(f"{row['id']:26} {row['duration']:5.0f}s {row['travelled']:6.1f}m "
              f"{row['speed']:5.2f} {row['depth_hz']:5.1f}Hz {focus:>7} {conf:>6} "
              f"{row['bytes'] / 1e6:6.0f}MB  {note}")

    if not args.usable:
        good = sum(1 for r in rows if verdict(r) == "ok")
        total = sum(r["bytes"] for r in rows)
        print(f"\n{good} of {len(rows)} usable, {total / 1e9:.1f} GB total")
        if good < len(rows):
            print("rerun with --usable to list only those, and point the "
                  "expensive tools at them")
    elif shown == 0:
        print("\nnothing passed. The usual cause is lighting — check the "
              "`conf` column without --usable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
