#!/usr/bin/env python3
"""Self-test the exporter orientation gate and 180-degree correction.

The fixture is generated once, then its poses are copied and rolled 180
degrees. This checks the same gravity-based detection and ARKit-camera-side
correction used for a real recording without needing a phone.

    python3 tools/test_export_episodes.py
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_test_session  # noqa: E402
from export_episodes import (  # noqa: E402
    ARKIT_TO_FLU,
    FLIP_180,
    export_segment,
    export_session,
    matrix_to_quat,
    quat_to_matrix,
    world_basis,
)
from read_session import Session  # noqa: E402


def export_args(out: str, poses_only: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        out=out,
        force=False,
        keep_limited=False,
        hz=None,
        poses_only=poses_only,
        width=848,
        height=480,
        fit="crop",
        hfov=None,
        uniform_timestamps=None,
        min_frames=10,
    )


def flip_session(src: str, dst: str) -> None:
    shutil.copytree(src, dst)
    pose_path = os.path.join(dst, "pose.jsonl")
    with open(pose_path) as f:
        rows = [json.loads(line) for line in f]

    with open(pose_path, "w") as f:
        for row in rows:
            R = quat_to_matrix(row["qx"], row["qy"], row["qz"], row["qw"]) @ FLIP_180
            row["qx"], row["qy"], row["qz"], row["qw"] = matrix_to_quat(R)
            row["gravX"] = -R[1, 0]
            row["gravY"] = -R[1, 1]
            row["gravZ"] = -R[1, 2]
            f.write(json.dumps(row, sort_keys=True) + "\n")


def read_pose_rows(path: str) -> list[list[float]]:
    with open(os.path.join(path, "poses_tum.txt")) as f:
        return [[float(value) for value in line.split()]
                for line in f if not line.startswith("#")]


def check_upside_down_detected(session_path: str) -> bool:
    orientation = Session(session_path).shot_orientation()
    ok = (orientation is not None
          and orientation["dominant"] == "landscape (upside down)"
          and orientation["upside_down"])
    print(f"  {'upside-down orientation is detected':42} {'PASS' if ok else 'FAIL'}")
    return ok


def check_flipped_export(session_path: str, out: str) -> bool:
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        summaries = export_session(session_path, export_args(out))
    rows = read_pose_rows(os.path.join(out, os.path.basename(session_path)))
    up = []
    for row in rows:
        qx, qy, qz, qw = row[4:8]
        up.append(quat_to_matrix(qx, qy, qz, qw)[2, 2])
    said = "rotating images and poses by 180 degrees" in captured.getvalue()
    ok = bool(summaries) and said and float(np.mean(up)) > 0
    print(f"  {'upside-down export restores camera up':42} {'PASS' if ok else 'FAIL'}")
    return ok


def check_normal_export_unchanged(session_path: str, out: str) -> bool:
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        export_session(session_path, export_args(out))
    actual = read_pose_rows(os.path.join(out, os.path.basename(session_path)))

    rows = [row for row in Session(session_path).posed_images()
            if row["tracking"] == "normal"]
    first = rows[0]["pose"]
    R0 = quat_to_matrix(first["qx"], first["qy"], first["qz"], first["qw"])
    N, _ = world_basis(R0)
    p0 = np.array([first["tx"], first["ty"], first["tz"]])
    expected = []
    for row in rows:
        pose = row["pose"]
        R = N.T @ quat_to_matrix(pose["qx"], pose["qy"], pose["qz"], pose["qw"]) @ ARKIT_TO_FLU
        t = N.T @ (np.array([pose["tx"], pose["ty"], pose["tz"]]) - p0)
        expected.append([row["t"] - rows[0]["t"], *t, *matrix_to_quat(R)])
    ok = (len(actual) == len(expected)
          and np.allclose(np.asarray(actual), np.asarray(expected), atol=5e-8))
    print(f"  {'normal export keeps the existing pose transform':42} {'PASS' if ok else 'FAIL'}")
    return ok


def check_image_rotation(session_path: str, out: str) -> bool:
    """Exercise the geometry=None copy path, which must re-encode on rotation."""
    from PIL import Image, ImageChops, ImageStat

    row = Session(session_path).posed_images()[0]
    os.makedirs(out, exist_ok=True)
    export_segment(Session(session_path), [row], out, None, None, True,
                   rotate_180=True)
    with Image.open(row["path"]) as source, Image.open(os.path.join(out, "images", "000000.jpg")) as got:
        source = source.convert("RGB")
        got = got.convert("RGB")
        expected = source.transpose(Image.ROTATE_180)
        expected_error = ImageStat.Stat(ImageChops.difference(got, expected)).mean[0]
        original_error = ImageStat.Stat(ImageChops.difference(got, source)).mean[0]
    ok = expected_error < original_error / 3
    print(f"  {'image is rotated before the copy path':42} {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        normal = make_test_session.build(os.path.join(tmp, "normal"))
        flipped = os.path.join(tmp, "flipped", os.path.basename(normal))
        os.makedirs(os.path.dirname(flipped), exist_ok=True)
        flip_session(normal, flipped)

        results = [
            check_upside_down_detected(flipped),
            check_flipped_export(flipped, os.path.join(tmp, "flip-export")),
            check_normal_export_unchanged(normal, os.path.join(tmp, "normal-export")),
            check_image_rotation(flipped, os.path.join(tmp, "image-export")),
        ]

    print()
    if all(results):
        print("all passed")
        return 0
    print("FAILURES — the exporter orientation correction is wrong")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
