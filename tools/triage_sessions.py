#!/usr/bin/env python3
"""Say which recorded sessions are unusable, and which are safe to remove.

A capture corpus accumulates two kinds of rubbish: recordings that went wrong at
the time, and recordings that were only ever a test. Both look identical to `ls`.
This reads each session's own manifest and streams and says which is which.

    python3 tools/triage_sessions.py                    # report, change nothing
    python3 tools/triage_sessions.py --trash ~/nav_trash --apply

**Nothing is deleted by default and the destructive path is not the easy one.**
`--apply` alone moves candidates to `--trash`; erasing them outright needs
`--delete --apply` and a typed confirmation. A walk cannot be re-recorded: the
room has changed, the light has changed, and the session that gets deleted by
accident is the one some measurement turns out to have rested on.

Two guards apply before any candidate is offered, and neither can be switched
off:

  **Referenced sessions are never candidates.** If a session's short id appears
  anywhere under a reference root — `docs/`, and the analysis workspace beside
  this repo — some result was measured on it, and deleting it turns a table in a
  document into a claim nobody can re-check. Nineteen of seventy-eight sessions
  were unreferenced when this was written; the other fifty-nine are evidence.

  **Recent sessions are never candidates.** Whatever was recorded this week is
  the experiment in progress, and it has not had time to be referenced by
  anything yet. `--min-age-days` sets the line.

The thresholds are measured rather than chosen. Across the 54 ARKit sessions
recorded up to 2026-08-24:

  delivered frames / (stills Hz x duration)   0.93 - 1.00, with two outliers at
                                              0.49 and 0.23
  pose rows with tracking == "normal"         84 % - 99 %, with outliers at
                                              75 %, 65 %, 43 % and 11 %

So the bars below sit in the gaps rather than inside either distribution, and a
session that trips one is unlike every healthy session in the corpus rather than
merely at the bad end of it.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

# Below these a session is unlike every healthy one measured. See the module
# docstring for the distributions they were read off.
MIN_FRAME_RATIO = 0.85
MIN_TRACKING_NORMAL = 0.80
# Not a defect — a judgement. A short walk inside a small area is the best
# angular coverage in the corpus (`docs/3DGS.md`, stage 4) and the worst thing
# to register against anything else, so "thin" is reported and never counted as
# a reason to remove on its own.
THIN_FRAMES = 40
THIN_SECONDS = 8.0


class Session:
    def __init__(self, path: str) -> None:
        self.path = path.rstrip("/")
        self.id = os.path.basename(self.path)
        self.short = self.id.rsplit("-", 1)[-1]
        self.manifest: dict | None = None
        self.broken: list[str] = []
        self.suspect: list[str] = []
        self.thin: list[str] = []
        self.kind = "?"
        self.preset = None
        self.duration = 0.0
        self.frames = 0
        self.frame_ratio: float | None = None
        self.tracking_normal: float | None = None
        self.referenced_in: list[str] = []
        #: True when the reference scan could not run. Protects unconditionally.
        self.references_unknown = False
        self.age_days = 0.0
        self.bytes = 0

    # -- reading -----------------------------------------------------------

    def read(self) -> None:
        self.age_days = (time.time() - os.path.getmtime(self.path)) / 86400.0
        self.bytes = _tree_bytes(self.path)
        mp = os.path.join(self.path, "manifest.json")
        if not os.path.exists(mp):
            self.broken.append("no manifest.json")
            return
        try:
            with open(mp) as fh:
                self.manifest = json.load(fh)
        except (OSError, ValueError) as exc:
            self.broken.append(f"manifest unreadable: {exc}")
            return
        m = self.manifest
        self.kind = m.get("kind", "arkit")
        self.preset = m.get("preset")
        if m.get("terminationReason") == "outOfStorage" or m.get("termination") == "outOfStorage":
            self.broken.append("ended out of storage")
        if self.kind == "multicam":
            self._read_multicam(m)
        else:
            self._read_arkit(m)

    def _read_arkit(self, m: dict) -> None:
        started, ended = m.get("startedAt"), m.get("endedAt")
        if not ended:
            # The manifest is rewritten on stop. Without `endedAt` the recorder
            # never got there, so counts and the video are both unfinished.
            self.broken.append("never stopped cleanly (no endedAt)")
            return
        self.duration = float(ended) - float(started or ended)
        counts = m.get("counts", {})
        self.frames = counts.get("frames", 0)
        if not self.frames:
            self.broken.append("no image frames")

        video = m.get("video")
        if video and video.get("file"):
            vp = os.path.join(self.path, video["file"])
            if not os.path.exists(vp):
                self.broken.append(f"manifest declares {video['file']} and it is missing")
            elif os.path.getsize(vp) == 0:
                self.broken.append(f"{video['file']} is empty — not finalised")

        hz = (m.get("config") or {}).get("stillsHz") or 0
        if hz and self.duration > 0 and self.frames:
            expected = hz * self.duration
            self.frame_ratio = self.frames / expected
            if self.frame_ratio < MIN_FRAME_RATIO:
                self.suspect.append(
                    f"delivered {self.frame_ratio:.0%} of the frames its rate implies "
                    f"({self.frames} of ~{expected:.0f})")

        normal, total = _tracking_fractions(os.path.join(self.path, "pose.jsonl"))
        if total:
            self.tracking_normal = normal / total
            if self.tracking_normal < MIN_TRACKING_NORMAL:
                self.suspect.append(
                    f"ARKit tracking was normal for only {self.tracking_normal:.0%} of frames")

        self._read_camera(m.get("camera") or {})
        if self.frames and self.frames < THIN_FRAMES:
            self.thin.append(f"{self.frames} frames")
        if 0 < self.duration < THIN_SECONDS:
            self.thin.append(f"{self.duration:.1f} s")

    def _read_multicam(self, m: dict) -> None:
        self.frames = m.get("images", 0)
        wide = m.get("wide_images", 0)
        if not self.frames:
            self.broken.append("no ultra-wide frames")
        hz = m.get("stills_hz") or 0
        if hz and self.frames:
            # No wall clock in this manifest; the rate and the count give the
            # walk's length well enough to judge whether it is a test.
            self.duration = self.frames / hz
        # `docs/DATA_FORMAT.md` calls these the load number and says they should
        # be zero on a healthy session. They are not the rate gate.
        late = m.get("images_dropped_late", 0)
        late_wide = m.get("wide_images_dropped_late", 0)
        if late or late_wide:
            self.suspect.append(
                f"dropped {late} ultra-wide and {late_wide} wide frames late — "
                "the capture queue could not keep up")
        if wide and self.frames and abs(wide - self.frames) > 0.2 * self.frames:
            self.suspect.append(f"lenses disagree: {self.frames} ultra-wide against {wide} wide")
        lock = m.get("exposure_lock", "")
        if "still adjusting" in _notes_text(self.path):
            self.suspect.append("exposure locked while the device was still adjusting")
        elif lock.startswith("!"):
            self.suspect.append(lock)
        if self.frames and self.frames < THIN_FRAMES:
            self.thin.append(f"{self.frames} frames")

    def _read_camera(self, cam: dict) -> None:
        if cam.get("stillAdjustingWhenLocked"):
            self.suspect.append(
                "exposure locked while the device was still adjusting — the whole "
                "session sits at whatever it caught")
        lock = cam.get("exposureLock") or ""
        if lock.startswith("!"):
            self.suspect.append(lock.lstrip("! "))

    # -- verdict -----------------------------------------------------------

    @property
    def verdict(self) -> str:
        if self.broken:
            return "broken"
        if self.suspect:
            return "suspect"
        if self.thin:
            return "thin"
        return "ok"

    @property
    def reasons(self) -> list[str]:
        return self.broken + self.suspect + self.thin

    def removable(self, min_age_days: float) -> tuple[bool, str]:
        """Candidate for removal, or the reason it is protected.

        `thin` alone never qualifies: a short close walk is the best-covered
        session in this corpus, not a mistake.
        """
        if self.references_unknown:
            return False, "the reference scan did not run"
        if self.referenced_in:
            return False, f"referenced in {', '.join(self.referenced_in[:3])}"
        if self.age_days < min_age_days:
            return False, f"recorded {self.age_days:.1f} days ago"
        if self.broken or self.suspect:
            return True, ""
        return False, "nothing wrong with it"


def _tree_bytes(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def _tracking_fractions(pose_path: str) -> tuple[int, int]:
    if not os.path.exists(pose_path):
        return 0, 0
    normal = total = 0
    with open(pose_path) as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            total += 1
            if row.get("tracking") == "normal":
                normal += 1
    return normal, total


def _notes_text(path: str) -> str:
    """The multi-camera recorder writes its findings as events, not manifest keys."""
    ep = os.path.join(path, "events.jsonl")
    if not os.path.exists(ep):
        return ""
    try:
        with open(ep) as fh:
            return fh.read()
    except OSError:
        return ""


# Directories under a reference root that cannot contain a reference and can
# contain gigabytes: virtualenvs, checkpoints, exported datasets. The first
# version of this searched them, took over two minutes, hit its timeout, and
# came back empty — which read as "nothing is referenced" and offered four
# sessions that `docs/3DGS.md` has tables about. That failure is the reason for
# both the excludes below and the fail-closed contract in `find_references`.
SKIP_DIRS = ("venvs", "runs", "data", "build", "artifacts", "vendor", "node_modules",
             ".git", ".venv", "__pycache__", "DerivedData", "logs", "frames",
             "frames_wide", "episodes")
# A reference is a mention in prose, code or configuration. Nothing else can
# carry one, and everything else is where the bytes are.
REF_SUFFIXES = (".md", ".json", ".py", ".sh", ".txt", ".yml", ".yaml", ".swift", ".log")
# The name pass is cheap, so it skips far less — only what cannot hold a
# reference at all. Depth-bounded instead, since an exported dataset's leaves
# are thousands of images.
NAME_SKIP_DIRS = (".git", "venvs", ".venv", "__pycache__", "node_modules")
NAME_SCAN_DEPTH = 3

# A coincidental match keeps a session that could have gone; a missed match
# offers one that documents depend on. The first costs disk and the second costs
# evidence, so every ambiguity here is resolved towards "referenced" — no
# word-boundary cleverness, no attempt to reject a six-hex-digit id that happens
# to fall inside a coordinate.


def find_references(sessions: list[Session], roots: list[str]) -> bool:
    """Mark every session whose short id appears under a reference root.

    Returns whether the scan actually ran. **A failed scan is not an empty
    result**: callers must treat `False` as "every session may be referenced",
    because the alternative is deleting the corpus a document was measured on
    because a `grep` timed out. That is not hypothetical — see `SKIP_DIRS`.

    Two things count as a reference. A session id in the *contents* of a
    document or a script is the obvious one. A session id in a *path* is the
    other: `gs3d/data/guard_merged_5bd1ed` says that session was exported and
    trained on without any file mentioning it.
    """
    roots = [r for r in roots if os.path.exists(r)]
    if not roots:
        # Nothing to search is a real answer only if the caller meant it. It is
        # reported as a failure so `--apply` refuses rather than assuming.
        return False
    by_short = {s.short: s for s in sessions}
    pattern = "|".join(sorted(by_short))
    hits: set[str] = set()

    cmd = ["grep", "-rhoIE", pattern]
    for name in SKIP_DIRS:
        cmd.append(f"--exclude-dir={name}")
    for suffix in REF_SUFFIXES:
        cmd.append(f"--include=*{suffix}")
    cmd.extend(roots)
    try:
        # grep exits 1 when it matches nothing, which is not an error here.
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if done.returncode > 1:
            return False
        hits.update(done.stdout.split())
    except (OSError, subprocess.SubprocessError):
        return False

    # Names, separately and with a different skip set. `gs3d/data/walk_87bc2c`
    # is a reference that no file contains: the session was exported and trained
    # on, and only the directory's name says so. Enumerating names is cheap even
    # where reading contents is not, so this descends into the directories the
    # content pass skips — bounded by depth, because the leaves of an exported
    # dataset are thousands of JPEGs and none of them is a reference.
    for root in roots:
        base_depth = root.rstrip("/").count(os.sep)
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in NAME_SKIP_DIRS]
            if dirpath.count(os.sep) - base_depth >= NAME_SCAN_DEPTH:
                dirnames[:] = []
                filenames = []
            for name in dirnames + filenames:
                for short in by_short:
                    if short in name:
                        hits.add(short)

    labels = [os.path.basename(r.rstrip("/")) or r for r in roots]
    for hit in hits:
        session = by_short.get(hit)
        if session is not None and not session.referenced_in:
            session.referenced_in = labels
    return True


def main(argv: list[str] | None = None) -> int:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory", nargs="?",
                    default=os.path.expanduser("~/nav_data"),
                    help="directory of session folders (default ~/nav_data)")
    ap.add_argument("--refs", nargs="*", default=None,
                    help="roots to search for references to a session id "
                         "(default: this repo's docs/ and ../gs3d)")
    ap.add_argument("--min-age-days", type=float, default=7.0,
                    help="never offer a session younger than this (default 7)")
    ap.add_argument("--trash", help="move removable sessions here instead of deleting")
    ap.add_argument("--delete", action="store_true",
                    help="erase instead of moving. Requires --apply and a typed confirmation")
    ap.add_argument("--apply", action="store_true",
                    help="actually move or delete. Without it this only reports")
    ap.add_argument("--all", action="store_true", help="list healthy sessions too")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.directory):
        print(f"no such directory: {args.directory}", file=sys.stderr)
        return 2

    entries = sorted(
        os.path.join(args.directory, name)
        for name in os.listdir(args.directory)
        if os.path.isdir(os.path.join(args.directory, name)))
    if not entries:
        print(f"no sessions in {args.directory}")
        return 0

    sessions = []
    for path in entries:
        s = Session(path)
        s.read()
        sessions.append(s)

    refs = args.refs
    if refs is None:
        refs = [os.path.join(here, "docs"), os.path.join(os.path.dirname(here), "gs3d")]
    if not find_references(sessions, refs):
        for s in sessions:
            s.references_unknown = True
        print("! could not scan for references in "
              f"{', '.join(refs) or '(no roots given)'} — "
              "every session is being treated as referenced, and nothing will be "
              "offered for removal.", file=sys.stderr)

    if args.json:
        print(json.dumps([{
            "id": s.id, "kind": s.kind, "preset": s.preset, "verdict": s.verdict,
            "reasons": s.reasons, "referenced": bool(s.referenced_in),
            "age_days": round(s.age_days, 2), "bytes": s.bytes,
            "frames": s.frames, "frame_ratio": s.frame_ratio,
            "tracking_normal": s.tracking_normal,
            "removable": s.removable(args.min_age_days)[0],
        } for s in sessions], indent=1))
        return 0

    _report(sessions, args)
    candidates = [s for s in sessions if s.removable(args.min_age_days)[0]]
    return _act(candidates, args)


def _report(sessions: list[Session], args) -> None:
    order = {"broken": 0, "suspect": 1, "thin": 2, "ok": 3}
    shown = [s for s in sessions if args.all or s.verdict != "ok"]
    shown.sort(key=lambda s: (order[s.verdict], s.id))

    print(f"{len(sessions)} sessions in {args.directory}")
    tally = {}
    for s in sessions:
        tally[s.verdict] = tally.get(s.verdict, 0) + 1
    print("  " + ", ".join(f"{tally[k]} {k}" for k in
                           ("broken", "suspect", "thin", "ok") if k in tally))
    print()
    for s in shown:
        keep = s.removable(args.min_age_days)[1]
        mark = "KEEP" if keep else "----"
        print(f"[{s.verdict:7}] {s.id}  {s.kind}/{s.preset or '?'}  "
              f"{_human(s.bytes)}  {s.age_days:.0f}d  {mark}")
        for reason in s.reasons:
            print(f"           - {reason}")
        if keep:
            print(f"           kept: {keep}")


def _act(candidates: list[Session], args) -> int:
    if not candidates:
        print("\nnothing removable — every flagged session is referenced, recent, or both.")
        return 0

    total = sum(s.bytes for s in candidates)
    verb = "delete" if args.delete else f"move to {args.trash}" if args.trash else "move"
    print(f"\nremovable: {len(candidates)} sessions, {_human(total)}")
    for s in candidates:
        print(f"  {s.id}  {_human(s.bytes)}  {s.reasons[0]}")

    if not args.apply:
        print(f"\nreporting only. Add --apply to {verb}.")
        return 0
    if not args.delete and not args.trash:
        print("\n--apply needs somewhere to put them: pass --trash DIR, or --delete "
              "to erase.", file=sys.stderr)
        return 2
    if args.delete:
        # Typed confirmation rather than a flag. A flag can be pasted from a
        # previous command; the word has to be meant.
        print(f"\nThis erases {len(candidates)} recordings permanently. They cannot be "
              "re-recorded.")
        try:
            if input('Type "delete" to confirm: ').strip() != "delete":
                print("cancelled")
                return 1
        except EOFError:
            print("cancelled — no terminal to confirm on", file=sys.stderr)
            return 1

    if args.trash:
        os.makedirs(args.trash, exist_ok=True)
    for s in candidates:
        if args.delete:
            shutil.rmtree(s.path)
            print(f"deleted {s.id}")
        else:
            dest = os.path.join(args.trash, s.id)
            if os.path.exists(dest):
                print(f"skipped {s.id} — already in the trash directory")
                continue
            shutil.move(s.path, dest)
            print(f"moved {s.id}")
    return 0


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


if __name__ == "__main__":
    raise SystemExit(main())
