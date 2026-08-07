#!/usr/bin/env python3
"""Generate a synthetic session in the on-disk format.

Lets the reader and any downstream pipeline be exercised without a device —
useful because the iOS app itself can only be built and run on real hardware.
The numbers are fake but the layout, timestamps and byte offsets are exactly
what the app writes.

The scenario is an indoor one: someone walking a slow loop around a room,
holding the phone. GPS is present but poor, as it is inside a building; the
camera pose comes from visual-inertial odometry and is what actually localises.

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

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

SESSION_ID = "20260807-014530-fixture"
CLOCK_ANCHOR = 1_754_531_130.0  # unix = t + anchor
START_CLOCK = 12_345.0
DURATION = 20.0

DEPTH_W, DEPTH_H = 256, 192
DEPTH_HZ = 5
STILLS_HZ = 5
AR_FPS = 60

# A 4 m x 3 m room, walked in a loop of radius 1.2 m.
ROOM_W, ROOM_D, ROOM_H = 4.0, 3.0, 2.4
WALK_RADIUS = 1.2
# Matches the Unitree G1 head camera the data is collected for.
CAMERA_HEIGHT = 1.15


def write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


IMG_W, IMG_H = 1920, 1440

# Held pitched down, matching the reference episodes (24.6-31.5 deg) rather than
# the geometric optimum.
PITCH_DOWN_DEG = 25.0


def _matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def camera_rotation(yaw: float, pitch_down: float, roll: float):
    """ARKit world_from_camera for a camera yawed, pitched down and rolled.

    Built as one matrix so that the quaternion and the gravity vector written
    into the fixture are derived from the same rotation. Writing them
    independently — as this generator used to — lets them disagree, and then a
    reader bug and a fixture bug look identical.
    """
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(-pitch_down), math.sin(-pitch_down)
    cr, sr = math.cos(roll), math.sin(roll)
    ry = [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]]
    rx = [[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]]
    rz = [[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]]
    return _matmul(_matmul(ry, rx), rz)


def matrix_to_quat(R):
    """(x, y, z, w), scalar last."""
    tr = R[0][0] + R[1][1] + R[2][2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return ((R[2][1] - R[1][2]) / s, (R[0][2] - R[2][0]) / s,
                (R[1][0] - R[0][1]) / s, 0.25 * s)
    if R[0][0] > R[1][1] and R[0][0] > R[2][2]:
        s = math.sqrt(1.0 + R[0][0] - R[1][1] - R[2][2]) * 2
        return (0.25 * s, (R[0][1] + R[1][0]) / s, (R[0][2] + R[2][0]) / s,
                (R[2][1] - R[1][2]) / s)
    if R[1][1] > R[2][2]:
        s = math.sqrt(1.0 + R[1][1] - R[0][0] - R[2][2]) * 2
        return ((R[0][1] + R[1][0]) / s, 0.25 * s, (R[1][2] + R[2][1]) / s,
                (R[0][2] - R[2][0]) / s)
    s = math.sqrt(1.0 + R[2][2] - R[0][0] - R[1][1]) * 2
    return ((R[0][2] + R[2][0]) / s, (R[1][2] + R[2][1]) / s, 0.25 * s,
            (R[1][0] - R[0][1]) / s)


def synthetic_frame(index: int):
    """A frame with enough structure to see a crop or a rotation by eye.

    Real pixels rather than a stub, so the export path — which crops, resizes
    and re-encodes — can actually be exercised.
    """
    img = Image.new("RGB", (IMG_W, IMG_H))
    px = img.load()
    # Coarse blocks: cheap to generate and obvious under a centre crop.
    for by in range(0, IMG_H, 80):
        for bx in range(0, IMG_W, 80):
            shade = ((bx // 80) * 13 + (by // 80) * 29 + index * 7) % 256
            for y in range(by, min(by + 80, IMG_H)):
                for x in range(bx, min(bx + 80, IMG_W)):
                    px[x, y] = (shade, (shade * 3) % 256, 255 - shade)
    return img


def build(out_dir: str, portrait: bool = False, interruption: bool = False) -> str:
    path = os.path.join(out_dir, SESSION_ID)
    os.makedirs(path, exist_ok=True)

    # GPS at 1 Hz. Indoors this wanders by tens of metres and the accuracy
    # figures say so — it is here to identify the building, not the room.
    locations = []
    for i in range(int(DURATION)):
        t = START_CLOCK + i
        locations.append({
            "t": t,
            "wall": t + CLOCK_ANCHOR,
            "lat": 37.5665 + math.sin(i / 3.0) * 0.00012,
            "lon": 126.9780 + math.cos(i / 3.0) * 0.00012,
            "alt": 38.0,
            "altEllipsoidal": 62.0,
            "hAcc": 35.0,
            "vAcc": 48.0,
            "speed": -1.0,      # negative means invalid
            "speedAcc": -1.0,
            "course": -1.0,
            "courseAcc": -1.0,
        })
    write_jsonl(os.path.join(path, "location.jsonl"), locations)

    # IMU at 100 Hz: handheld sway plus walking cadence.
    motion = []
    for i in range(int(DURATION * 100)):
        t = START_CLOCK + i / 100.0
        step = math.sin(i / 50.0 * math.pi)
        motion.append({
            "t": t,
            "ax": 0.06 * step, "ay": 0.03 * math.sin(i / 17.0), "az": 0.08 * step,
            "gx": 0.0, "gy": 0.0, "gz": -1.0,
            "rx": 0.02 * math.sin(i / 31.0), "ry": 0.11, "rz": 0.01,
            "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
            "mx": 12.0, "my": -3.0, "mz": 44.0,
            "magAcc": 1,
        })
    write_jsonl(os.path.join(path, "motion.jsonl"), motion)

    # Poses at the ARKit frame rate — one per frame, whether or not that frame
    # was written as an image.
    # Position and yaw are chosen together so the camera actually faces the way
    # it is walking: with yaw psi the ARKit forward axis is (-sin psi, 0,
    # -cos psi), and (R cos psi, h, -R sin psi) is the path whose tangent
    # matches it.
    poses = []
    for i in range(int(DURATION * AR_FPS)):
        t = START_CLOCK + i / AR_FPS
        theta = 2 * math.pi * i / (DURATION * AR_FPS)
        roll = math.radians((90.0 if portrait else 0.0) + 3.0 * math.sin(i / 40.0))
        R = camera_rotation(theta, math.radians(PITCH_DOWN_DEG), roll)
        qx, qy, qz, qw = matrix_to_quat(R)
        poses.append({
            "t": t, "frame": i,
            "tx": WALK_RADIUS * math.cos(theta),
            "ty": CAMERA_HEIGHT,
            "tz": -WALK_RADIUS * math.sin(theta),
            "qx": qx, "qy": qy, "qz": qz, "qw": qw,
            "fx": 1295.0, "fy": 1295.0, "cx": 960.0, "cy": 720.0,
            "tracking": "normal" if i > 40 else "limited:initializing",
            "exposure": 0.016,
            # Gravity in camera coordinates, from the same rotation as the
            # quaternion above: world down (0,-1,0) expressed in camera axes.
            "gravX": -R[1][0], "gravY": -R[1][1], "gravZ": -R[1][2],
        })
    write_jsonl(os.path.join(path, "pose.jsonl"), poses)

    # Depth: a wall a couple of metres ahead, floor below. Everything is well
    # inside LiDAR's ~5 m range, which is the point of using it indoors.
    index = []
    with open(os.path.join(path, "depth.bin"), "wb") as depth_file:
        offset = 0
        for k in range(int(DURATION * DEPTH_HZ)):
            frame_index = k * (AR_FPS // DEPTH_HZ)
            values = []
            for y in range(DEPTH_H):
                if y < DEPTH_H * 0.55:
                    depth = 2.4                          # wall ahead
                else:
                    # Floor receding towards the wall.
                    depth = 1.2 + 2.0 * (DEPTH_H - y) / (DEPTH_H * 0.45)
                values.extend([depth] * DEPTH_W)
            payload = struct.pack(f"<{len(values)}e", *values)
            depth_file.write(payload)
            index.append({
                "t": START_CLOCK + k / DEPTH_HZ,
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

    # Planes: floor, two walls and a table, discovered as the loop progresses
    # and then re-reported once as they grow.
    planes = []

    def plane(t, pid, event, alignment, classification, pos, size, yaw=0.0):
        planes.append({
            "t": t, "id": pid, "event": event,
            "alignment": alignment, "classification": classification,
            "tx": pos[0], "ty": pos[1], "tz": pos[2],
            "qx": 0.0, "qy": math.sin(yaw / 2), "qz": 0.0, "qw": math.cos(yaw / 2),
            "cx": 0.0, "cy": 0.0, "cz": 0.0,
            "width": size[0], "height": size[1], "rotationOnYAxis": 0.0,
        })

    plane(START_CLOCK + 1.5, "F0000000-0000-0000-0000-000000000001", "added",
          "horizontal", "floor", (0.0, 0.0, 0.0), (2.0, 1.5))
    plane(START_CLOCK + 4.0, "W0000000-0000-0000-0000-000000000002", "added",
          "vertical", "wall", (0.0, 1.2, -ROOM_D / 2), (3.0, ROOM_H))
    plane(START_CLOCK + 9.0, "T0000000-0000-0000-0000-000000000003", "added",
          "horizontal", "table", (1.0, 0.75, 0.4), (1.2, 0.8))
    plane(START_CLOCK + 12.0, "W0000000-0000-0000-0000-000000000004", "added",
          "vertical", "wall", (-ROOM_W / 2, 1.2, 0.0), (ROOM_D, ROOM_H), yaw=math.pi / 2)
    # The floor grows as more of the room is seen.
    plane(START_CLOCK + 14.0, "F0000000-0000-0000-0000-000000000001", "updated",
          "horizontal", "floor", (0.0, 0.0, 0.0), (ROOM_W, ROOM_D))
    # A spurious plane that ARKit later merges away.
    plane(START_CLOCK + 6.0, "X0000000-0000-0000-0000-000000000005", "added",
          "horizontal", "none:undetermined", (0.5, 0.02, 0.5), (0.4, 0.4))
    plane(START_CLOCK + 15.0, "X0000000-0000-0000-0000-000000000005", "removed",
          "horizontal", "none:undetermined", (0.5, 0.02, 0.5), (0.4, 0.4))
    planes.sort(key=lambda row: row["t"])
    write_jsonl(os.path.join(path, "planes.jsonl"), planes)

    # Stills: one JPEG per captured frame, sharing `frame` with the pose stream.
    # The image bytes are placeholders — encoding needs the device — but the
    # index and the join key are exactly what the app writes.
    os.makedirs(os.path.join(path, "frames"), exist_ok=True)
    frames = []
    for k in range(int(DURATION * STILLS_HZ)):
        frame_index = k * (AR_FPS // STILLS_HZ)
        name = f"{frame_index:06d}.jpg"
        dst = os.path.join(path, "frames", name)
        if Image is not None:
            synthetic_frame(k).save(dst, "JPEG", quality=80)
            size = os.path.getsize(dst)
        else:
            payload = b"\xff\xd8\xff\xe0" + b"\x00" * 512
            with open(dst, "wb") as f:
                f.write(payload)
            size = len(payload)
        frames.append({
            "t": START_CLOCK + k / STILLS_HZ,
            "frame": frame_index,
            "file": f"frames/{name}",
            "width": IMG_W, "height": IMG_H,
            "bytes": size,
        })
    write_jsonl(os.path.join(path, "frames.jsonl"), frames)

    write_jsonl(os.path.join(path, "heading.jsonl"), [
        {"t": START_CLOCK + i, "trueHeading": -1.0,
         "magneticHeading": 82.0 + 25 * math.sin(i / 2.0), "accuracy": -1.0}
        for i in range(int(DURATION))
    ])

    events = [
        {"t": START_CLOCK, "kind": "session.start", "detail": f"id={SESSION_ID} battery=0.87"},
        {"t": START_CLOCK + 0.4, "kind": "ar.started", "detail": "format=1920x1440@60 depth=true"},
        {"t": START_CLOCK + 1.2, "kind": "ar.tracking", "detail": "normal"},
        {"t": START_CLOCK + 8.5, "kind": "ar.tracking", "detail": "limited:excessiveMotion"},
        {"t": START_CLOCK + 9.4, "kind": "ar.tracking", "detail": "normal"},
        {"t": START_CLOCK + DURATION, "kind": "session.stop", "detail": "user"},
    ]
    if interruption:
        # ARKit resets the world origin here, so an exporter must cut rather
        # than concatenate across it.
        events.append({"t": START_CLOCK + 10.0, "kind": "ar.interrupted",
                       "detail": "camera capture suspended"})
        events.append({"t": START_CLOCK + 10.5, "kind": "ar.interruptionEnded",
                       "detail": "world origin reset; pose frame discontinuity"})
        events.sort(key=lambda e: e["t"])
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
            "attitudeReferenceFrame": "xArbitraryZVertical",
        },
        "config": {
            "captureMode": "stills", "stillsHz": STILLS_HZ, "stillQuality": 0.85,
            "recordDepth": True, "recordConfidence": False,
            "detectPlanes": True,
            "videoFPS": 30, "depthHz": DEPTH_HZ, "motionHz": 100,
            "videoBitrate": 12000000,
            "useMagnetometerCorrection": False,
            "degradeOnThermalPressure": True,
        },
        "video": None,
        "counts": {
            "location": len(locations), "heading": int(DURATION),
            "motion": len(motion), "pose": len(poses), "planes": len(planes),
            "frames": len(frames), "depth": len(index), "events": len(events),
        },
        "appVersion": "0.1.0",
        "appBuild": "1",
        "terminationReason": None,
    }
    with open(os.path.join(path, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    # The JPEGs are placeholders, not real images — encoding is the one thing
    # that genuinely needs the device.
    open(os.path.join(path, ".complete"), "w").close()
    return path


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", nargs="?", default="./fixture",
                        help="directory to create the session in")
    parser.add_argument("--portrait", action="store_true",
                        help="simulate the phone held portrait rather than landscape")
    parser.add_argument("--interruption", action="store_true",
                        help="include an ar.interruptionEnded, which must split the export")
    args = parser.parse_args(argv)
    path = build(args.out, portrait=args.portrait, interruption=args.interruption)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
