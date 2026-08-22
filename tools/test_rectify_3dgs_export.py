#!/usr/bin/env python3
"""Self-tests for the exported ultra-wide rectification utility."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rectify_3dgs_export import rectify_export, rectified_intrinsics  # noqa: E402


def _write_fixture(root: Path) -> None:
    (root / "images").mkdir(parents=True)
    (root / "depths").mkdir()
    (root / "depth_normals_mask").mkdir()
    (root / "sparse" / "0").mkdir(parents=True)

    for name, shape in (("000000.jpg", (8, 16)), ("000001.jpg", (18, 32))):
        image = np.zeros((*shape, 3), dtype=np.uint8)
        image[..., 0] = np.arange(shape[1], dtype=np.uint8)[None, :]
        assert cv2.imwrite(str(root / "images" / name), image)
    depth = np.full((18, 32), 2000, dtype=np.uint16)
    assert cv2.imwrite(str(root / "depths" / "000001.png"), depth)
    mask = np.zeros((18, 32), dtype=np.uint8)
    assert cv2.imwrite(str(root / "depth_normals_mask" / "000001.jpg"), mask)

    transforms = {
        "camera_model": "PINHOLE",
        "frames": [
            {"file_path": "images/000000.jpg", "transform_matrix": np.eye(4).tolist(),
             "fl_x": 8.0, "fl_y": 8.0, "cx": 8.0, "cy": 4.0,
             "w": 16, "h": 8, "lens": "wide",
             "depth_file_path": "depths/000000.png"},
            {"file_path": "images/000001.jpg", "transform_matrix": np.eye(4).tolist(),
             "fl_x": 16.0, "fl_y": 16.0, "cx": 16.0, "cy": 9.0,
             "w": 32, "h": 18, "lens": "ultrawide",
             "depth_file_path": "depths/000001.png"},
        ],
    }
    (root / "transforms.json").write_text(json.dumps(transforms))
    (root / "export.json").write_text(json.dumps({"images": 2}))
    (root / "sparse" / "0" / "cameras.txt").write_text(
        "1 PINHOLE 16 8 8 8 8 4\n"
        "2 PINHOLE 32 18 16 16 16 9\n")
    (root / "sparse" / "0" / "images.txt").write_text(
        "1 1 0 0 0 0 0 0 1 000000.jpg\n\n"
        "2 1 0 0 0 0 0 0 2 000001.jpg\n\n")


def test_rectifies_only_the_ultrawide() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "source"
        output = Path(tmp) / "output"
        _write_fixture(source)
        result = rectify_export(str(source), str(output), hfov=102.5)
        assert result["ultrawide_frames"] == 1

        data = json.loads((output / "transforms.json").read_text())
        wide, ultra = data["frames"]
        assert "ultrawide_rectified" in data
        assert wide["fl_x"] == 8.0
        expected = rectified_intrinsics(32, 18, 102.5)
        for key, value in expected.items():
            assert abs(ultra[key] - value) < 1e-9

        cameras = (output / "sparse" / "0" / "cameras.txt").read_text().splitlines()
        assert cameras[0].split()[4:8] == ["8", "8", "8", "4"]
        assert all(abs(float(a) - float(b)) < 1e-6
                   for a, b in zip(cameras[1].split()[4:8],
                                   [str(expected[key]) for key in
                                    ("fl_x", "fl_y", "cx", "cy")]))
        assert cv2.imread(str(output / "images" / "000001.jpg")) is not None
        assert cv2.imread(str(output / "depths" / "000001.png"),
                          cv2.IMREAD_UNCHANGED).shape == (18, 32)


def main() -> int:
    test_rectifies_only_the_ultrawide()
    print("rectify_3dgs_export self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
