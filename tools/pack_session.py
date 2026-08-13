#!/usr/bin/env python3
"""Pack a recorded session into something small enough to hand over.

A 30-second capture at 30 Hz is a couple of hundred megabytes, almost all of it
JPEGs that the pose and depth tooling never opens. Committing one to git costs
that on every clone, for good. This strips a session to the streams an analysis
actually reads and writes a tarball.

    python3 tools/pack_session.py ~/nav_data/20260808-101500-ab12cd
    python3 tools/pack_session.py ~/nav_data/2026* --frames 300 --with-images

Attach the result to a GitHub release rather than committing it — release assets
live outside git history and download fine from a sandbox:

    gh release upload dev-latest 20260808-101500-ab12cd.tar.gz

Requires numpy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from read_session import Session  # noqa: E402


# Everything an analysis reads. `frames/` and `video.mov` are excluded on
# purpose: the pose and depth tools never open them, and they are the bulk.
KEEP = ["manifest.json", "pose.jsonl", "motion.jsonl", "events.jsonl",
        "location.jsonl", "heading.jsonl", "planes.jsonl", "frames.jsonl",
        ".complete"]


def pack(src: str, out_dir: str, limit: int | None, with_images: bool) -> str:
    session = Session(src)
    name = session.id
    staging = os.path.join(out_dir, name)
    os.makedirs(staging, exist_ok=True)

    index = session.depth_index()
    kept = index[:limit] if limit else index

    # Depth and confidence are rewritten rather than copied, because the index
    # carries byte offsets into one concatenated blob — truncating the list
    # without rebuilding the file would leave every offset pointing at the wrong
    # frame, which reads as a working capture and is not one.
    rows = []
    depth_path = os.path.join(staging, "depth.bin")
    conf_path = os.path.join(staging, "confidence.bin")
    d_off = c_off = 0
    with open(depth_path, "wb") as df, open(conf_path, "wb") as cf:
        for entry in kept:
            row = dict(entry)
            raw = np.asarray(session.depth_frame(entry), dtype=np.float16).tobytes()
            df.write(raw)
            row["offset"], row["length"] = d_off, len(raw)
            d_off += len(raw)
            if entry.get("confidenceOffset") is not None:
                conf = np.asarray(session.confidence_frame(entry), dtype=np.uint8).tobytes()
                cf.write(conf)
                row["confidenceOffset"], row["confidenceLength"] = c_off, len(conf)
                c_off += len(conf)
            rows.append(row)

    if c_off == 0:
        os.remove(conf_path)
    with open(os.path.join(staging, "depth.jsonl"), "w") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows))

    frames_kept = {r["frame"] for r in rows}
    for filename in KEEP:
        source = os.path.join(src, filename)
        if not os.path.exists(source):
            continue
        if filename == "pose.jsonl" and limit:
            # Poses are cheap, but keeping the whole stream against a truncated
            # depth list makes the two disagree about how long the session was.
            keep_to = max(frames_kept) if frames_kept else 0
            with open(source) as fin, open(os.path.join(staging, filename), "w") as fout:
                for line in fin:
                    try:
                        if json.loads(line).get("frame", 0) <= keep_to:
                            fout.write(line)
                    except json.JSONDecodeError:
                        pass
            continue
        with open(source, "rb") as fin, open(os.path.join(staging, filename), "wb") as fout:
            fout.write(fin.read())

    if with_images:
        src_frames = os.path.join(src, "frames")
        if os.path.isdir(src_frames):
            dst_frames = os.path.join(staging, "frames")
            os.makedirs(dst_frames, exist_ok=True)
            for entry in session.stream("frames"):
                if limit and entry["frame"] not in frames_kept:
                    continue
                base = os.path.basename(entry["file"])
                source = os.path.join(src_frames, base)
                if os.path.exists(source):
                    with open(source, "rb") as a, open(os.path.join(dst_frames, base), "wb") as b:
                        b.write(a.read())

    archive = os.path.join(out_dir, f"{name}.tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(staging, arcname=name)
    return archive


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("-o", "--out", default=".")
    ap.add_argument("--frames", type=int, default=None,
                    help="keep only the first N depth frames")
    ap.add_argument("--with-images", action="store_true",
                    help="include the JPEGs. Only needed for export or "
                         "intrinsics work — pose and depth analysis never "
                         "opens them, and they are most of the bytes.")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    for path in args.sessions:
        path = path.rstrip("/")
        try:
            with tempfile.TemporaryDirectory() as staging:
                archive = pack(path, args.out, args.frames, args.with_images)
                del staging
        except FileNotFoundError as exc:
            print(f"! {path}: {exc}", file=sys.stderr)
            continue
        before = sum(os.path.getsize(os.path.join(root, f))
                     for root, _, files in os.walk(path) for f in files)
        after = os.path.getsize(archive)
        print(f"{archive}  {after / 1e6:.1f} MB "
              f"(from {before / 1e6:.0f} MB, {before / max(after, 1):.0f}x smaller)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
