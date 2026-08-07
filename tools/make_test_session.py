#!/usr/bin/env python3
"""Generate a synthetic session in the on-disk format.

Lets the reader and any downstream pipeline be exercised without a device —
useful because the iOS app itself can only be built and run on real hardware.
The numbers are fake but the layout, timestamps and byte offsets are exactly
what the app writes.

    python3 tools/make_test_session.py /tmp/fixture
    python3 tools/read_session.py /tmp/fixture/20260807-014530-fixture
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys

SESSION_ID = "20260807-014530-fixture"
CLOCK_ANCHOR = 1_754_531_130.0  # unix = t + anchor
START_CLOCK = 12_345.0
DURATION = 20.0

DEPTH_W, DEPTH_H = 256, 192


def write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def build(out_dir: str) -> str:
    path = os.path.join(out_dir, SESSION_ID)
    os.makedirs(path, exist_ok=True)

    # GPS at 1 Hz, walking a straight line east at ~14 m/s.
    locations = []
    for i in range(int(DURATION)):
        t = START_CLOCK + i
        locations.append({
            "t": t,
            "wall": t + CLOCK_ANCHOR,
            "lat": 37.5665,
            "lon": 126.9780 + i * 0.00016,
            "alt": 38.0,
            "altEllipsoidal": 62.0,
            "hAcc": 4.0,
            "vAcc": 6.0,
            "speed": 14.0,
            "speedAcc": 0.5,
            "course": 90.0,
            "courseAcc": 2.0,
        })
    write_jsonl(os.path.join(path, "location.jsonl"), locations)

    # IMU at 100 Hz.
    motion = []
    for i in range(int(DURATION * 100)):
        t = START_CLOCK + i / 100.0
        motion.append({
            "t": t,
            "ax": 0.02 * math.sin(i / 20.0), "ay": 0.01, "az": -0.03,
            "gx": 0.0, "gy": 0.0, "gz": -1.0,
            "rx": 0.001, "ry": 0.002, "rz": 0.0005,
            "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
            "mx": 12.0, "my": -3.0, "mz": 44.0,
            "magAcc": 2,
        })
    write_jsonl(os.path.join(path, "motion.jsonl"), motion)

    # Poses at 30 Hz.
    poses = []
    for i in range(int(DURATION * 30)):
        t = START_CLOCK + i / 30.0
        poses.append({
            "t": t, "frame": i,
            "tx": i * 14.0 / 30.0, "ty": 0.0, "tz": 0.0,
            "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
            "fx": 1590.0, "fy": 1590.0, "cx": 960.0, "cy": 720.0,
            "tracking": "normal" if i > 20 else "limited:initializing",
            "exposure": 0.008,
        })
    write_jsonl(os.path.join(path, "pose.jsonl"), poses)

    # Depth at 5 Hz: a floor plane receding to the horizon, as float16 metres.
    index = []
    with open(os.path.join(path, "depth.bin"), "wb") as depth_file:
        offset = 0
        for k in range(int(DURATION * 5)):
            frame_index = k * 6  # 30 fps video / 5 Hz depth
            values = []
            for y in range(DEPTH_H):
                # Rows near the top look further away; above the horizon there
                # is no return at all, which is what infinity encodes.
                depth = float("inf") if y < DEPTH_H * 0.35 else 1.5 + 40.0 / (y - DEPTH_H * 0.3)
                values.extend([depth] * DEPTH_W)
            payload = struct.pack(f"<{len(values)}e", *values)
            depth_file.write(payload)
            index.append({
                "t": START_CLOCK + k / 5.0,
                "frame": frame_index,
                "offset": offset,
                "length": len(payload),
                "width": DEPTH_W,
                "height": DEPTH_H,
                "format": "float16",
                "confidenceOffset": None,
                "confidenceLength": None,
            })
            offset += len(payload)
    write_jsonl(os.path.join(path, "depth.jsonl"), index)

    write_jsonl(os.path.join(path, "heading.jsonl"), [
        {"t": START_CLOCK + i, "trueHeading": 90.0, "magneticHeading": 82.0, "accuracy": 5.0}
        for i in range(int(DURATION))
    ])

    events = [
        {"t": START_CLOCK, "kind": "session.start", "detail": f"id={SESSION_ID} battery=0.87"},
        {"t": START_CLOCK + 0.4, "kind": "ar.started", "detail": "format=1920x1440@60 depth=true"},
        {"t": START_CLOCK + 1.2, "kind": "ar.tracking", "detail": "normal"},
        {"t": START_CLOCK + 11.0, "kind": "thermal", "detail": "fair"},
        {"t": START_CLOCK + DURATION, "kind": "session.stop", "detail": "user"},
    ]
    write_jsonl(os.path.join(path, "events.jsonl"), events)

    manifest = {
        "schemaVersion": 1,
        "id": SESSION_ID,
        "startedAt": START_CLOCK + CLOCK_ANCHOR,
        "endedAt": START_CLOCK + CLOCK_ANCHOR + DURATION,
        "clockAnchor": CLOCK_ANCHOR,
        "startClock": START_CLOCK,
        "device": {
            "model": "iPhone17,1",
            "systemVersion": "18.5",
            "name": "test fixture",
            "hasLiDAR": True,
            "attitudeReferenceFrame": "xArbitraryCorrectedZVertical",
        },
        "config": {
            "recordVideo": True, "recordDepth": True, "recordConfidence": False,
            "videoFPS": 30, "depthHz": 5, "motionHz": 100,
            "videoBitrate": 12000000, "degradeOnThermalPressure": True,
        },
        "video": {
            "file": "video.mov", "width": 1920, "height": 1440, "codec": "hevc",
            "nominalFPS": 30, "bitrate": 12000000, "firstFramePTS": START_CLOCK,
        },
        "counts": {
            "location": len(locations), "heading": int(DURATION),
            "motion": len(motion), "pose": len(poses),
            "depth": len(index), "events": len(events),
        },
        "appVersion": "0.1.0",
        "appBuild": "1",
        "terminationReason": None,
    }
    with open(os.path.join(path, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    # No real video is produced — encoding is the one thing that genuinely
    # needs the device. Readers must tolerate its absence.
    open(os.path.join(path, ".complete"), "w").close()
    return path


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", nargs="?", default="./fixture",
                        help="directory to create the session in")
    args = parser.parse_args(argv)
    path = build(args.out)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
