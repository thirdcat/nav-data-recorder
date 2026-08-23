#!/usr/bin/env python3
"""Self-tests for session triage, mostly about one thing: not offering to delete
something that matters.

The detectors are the easy half and are tested here for completeness. The half
worth the file is the guards. The first version of `find_references` searched a
directory tree containing virtualenvs and model checkpoints, took longer than
its own timeout, returned nothing, and reported four sessions as removable that
`docs/3DGS.md` has published tables about — because "the scan found nothing" and
"the scan did not run" produced identical output. So the contract under test is:

  * a session named anywhere under a reference root is never offered
  * a session younger than the age bar is never offered
  * a reference scan that fails protects **everything**, loudly
  * being small is not a fault and never makes a session a candidate

    python3 tools/test_triage_sessions.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import triage_sessions as ts  # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool) -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}   {name}")
    if not ok:
        FAILURES.append(name)


def write_session(root: str, sid: str, *, kind: str = "arkit", hz: float = 5.0,
                  duration: float = 20.0, frames: int | None = None,
                  tracking_normal: float = 1.0, ended: bool = True,
                  age_days: float = 30.0, extra: dict | None = None) -> str:
    """A session directory complete enough for the reader, and no more."""
    path = os.path.join(root, sid)
    os.makedirs(path, exist_ok=True)
    if frames is None:
        frames = int(hz * duration)

    if kind == "multicam":
        manifest = {"id": sid, "kind": "multicam", "images": frames,
                    "wide_images": frames, "stills_hz": hz}
    else:
        manifest = {
            "id": sid, "startedAt": 1000.0, "config": {"stillsHz": hz},
            "counts": {"frames": frames},
        }
        if ended:
            manifest["endedAt"] = 1000.0 + duration
        rows = 60 * duration
        with open(os.path.join(path, "pose.jsonl"), "w") as fh:
            for i in range(int(rows)):
                state = "normal" if i < rows * tracking_normal else "limited:relocalizing"
                fh.write(json.dumps({"frame": i, "tracking": state}) + "\n")
    manifest.update(extra or {})
    with open(os.path.join(path, "manifest.json"), "w") as fh:
        json.dump(manifest, fh)

    when = time.time() - age_days * 86400
    os.utime(path, (when, when))
    return path


def triage(root: str, refs: list[str]) -> dict[str, ts.Session]:
    sessions = []
    for name in sorted(os.listdir(root)):
        s = ts.Session(os.path.join(root, name))
        s.read()
        sessions.append(s)
    if not ts.find_references(sessions, refs):
        for s in sessions:
            s.references_unknown = True
    return {s.short: s for s in sessions}


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "nav_data")
        refs_dir = os.path.join(tmp, "docs")
        os.makedirs(root)
        os.makedirs(refs_dir)

        write_session(root, "20260101-000000-aaaaaa")
        write_session(root, "20260101-000001-bbbbbb", frames=20)          # 20 % of rate
        write_session(root, "20260101-000002-cccccc", tracking_normal=0.3)
        write_session(root, "20260101-000003-dddddd", ended=False)
        write_session(root, "20260101-000004-eeeeee", duration=4.0, frames=20)
        write_session(root, "20260101-000005-ffffff", frames=20)          # referenced
        write_session(root, "20260101-000006-999999", frames=20, age_days=1.0)
        write_session(root, "20260101-000007-777777", kind="multicam", frames=100,
                      extra={"images_dropped_late": 3, "wide_images_dropped_late": 0})
        write_session(root, "20260101-000008-888888",
                      extra={"camera": {"exposureLock": "locked",
                                        "stillAdjustingWhenLocked": True}})

        # One reference of each kind the scanner is supposed to find.
        with open(os.path.join(refs_dir, "notes.md"), "w") as fh:
            fh.write("the ffffff walk is the one the table above is measured on\n")
        os.makedirs(os.path.join(refs_dir, "exports", "walk_888888"))

        print("detectors")
        s = triage(root, [refs_dir])
        check("a healthy session is ok", s["aaaaaa"].verdict == "ok")
        check("a frame deficit is suspect",
              s["bbbbbb"].verdict == "suspect"
              and "20%" in s["bbbbbb"].suspect[0])
        check("poor tracking is suspect",
              s["cccccc"].verdict == "suspect"
              and "30%" in s["cccccc"].suspect[0])
        check("a session with no endedAt is broken",
              s["dddddd"].verdict == "broken")
        check("a short session is thin, not broken", s["eeeeee"].verdict == "thin")
        check("late drops are suspect on a multi-cam session",
              s["777777"].verdict == "suspect" and "late" in s["777777"].suspect[0])
        check("a lock taken mid-adjustment is suspect",
              s["888888"].verdict == "suspect")

        print("guards")
        check("a session named in a document is never offered",
              s["ffffff"].verdict == "suspect"
              and s["ffffff"].removable(7.0) == (False, "referenced in docs"))
        check("a session named only by a directory is never offered",
              bool(s["888888"].referenced_in))
        check("a recent session is never offered",
              s["999999"].removable(7.0)[0] is False
              and "days ago" in s["999999"].removable(7.0)[1])
        check("thin alone is never offered", s["eeeeee"].removable(7.0)[0] is False)
        check("an old unreferenced suspect is offered",
              s["bbbbbb"].removable(7.0) == (True, ""))

        print("fail-closed")
        # The exact shape of the original bug: a reference root that cannot be
        # searched must protect everything rather than protect nothing.
        s = triage(root, [os.path.join(tmp, "does-not-exist")])
        check("a reference scan that cannot run returns False",
              ts.find_references([], [os.path.join(tmp, "does-not-exist")]) is False)
        check("no session is offered when references are unknown",
              all(not v.removable(7.0)[0] for v in s.values()))
        check("and it says why",
              s["bbbbbb"].removable(7.0)[1] == "the reference scan did not run")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} failed: " + "; ".join(FAILURES))
        return 1
    print("all passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
