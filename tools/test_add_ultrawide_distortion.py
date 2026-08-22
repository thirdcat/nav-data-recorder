#!/usr/bin/env python3
"""Self-tests for the Apple-lookup -> OpenCV export conversion."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from add_ultrawide_distortion import (  # noqa: E402
    COEFFICIENT_NAMES,
    add_distortion,
    fit_opencv_coefficients,
)


def _calibration() -> dict:
    ref_w, ref_h = 400, 300
    fx = fy = 160.0
    cx, cy = 200.0, 150.0
    radius_max = float(np.hypot(200.0, 150.0))
    radii = np.linspace(0.0, radius_max, 42)
    # A modest known radial barrel model, expressed as Apple's
    # ideal->distorted magnification table.
    r = radii / fx
    table = 0.04 * r * r - 0.015 * r ** 4
    return {
        "reference_dimensions": [ref_w, ref_h],
        "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        "lens_distortion_center": [cx, cy],
        "inverse_lens_distortion_lookup_table": table.tolist(),
    }


def _write_export(root: Path, calibration_path: Path) -> None:
    (root / "images").mkdir(parents=True)
    (root / "sparse" / "0").mkdir(parents=True)
    (root / "images" / "wide.jpg").write_bytes(b"wide")
    (root / "images" / "ultra.jpg").write_bytes(b"ultra")
    frames = [
        {"file_path": "images/wide.jpg", "lens": "wide", "w": 64, "h": 48,
         "fl_x": 40.0, "fl_y": 40.0, "cx": 32.0, "cy": 24.0,
         "transform_matrix": np.eye(4).tolist()},
        {"file_path": "images/ultra.jpg", "lens": "ultrawide", "w": 200, "h": 120,
         "fl_x": 80.0, "fl_y": 80.0, "cx": 100.0, "cy": 60.0,
         "transform_matrix": np.eye(4).tolist()},
    ]
    (root / "transforms.json").write_text(json.dumps({
        "camera_model": "PINHOLE", "frames": frames,
    }))
    (root / "sparse" / "0" / "cameras.txt").write_text(
        "1 PINHOLE 64 48 40 40 32 24\n"
        "2 PINHOLE 200 120 80 80 100 60\n")
    (root / "sparse" / "0" / "images.txt").write_text(
        "1 1 0 0 0 0 0 0 1 wide.jpg\n\n"
        "2 1 0 0 0 0 0 0 2 ultra.jpg\n\n")


def main() -> int:
    calibration = _calibration()
    frame = {"w": 200, "h": 120, "fl_x": 80.0, "fl_y": 80.0,
             "cx": 100.0, "cy": 60.0}
    fitted = fit_opencv_coefficients(calibration, frame)
    assert fitted["k1"] > 0.0
    assert fitted["k2"] < 0.0
    assert fitted["fit_p95_px"] < 0.1, fitted

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        calibration_path = root / "calibration.json"
        calibration_path.write_text(json.dumps(calibration))
        source = root / "source"
        output = root / "output"
        _write_export(source, calibration_path)
        report = add_distortion(str(source), str(output),
                                calibration=str(calibration_path))
        assert report["ultrawide_frames"] == 1
        data = json.loads((output / "transforms.json").read_text())
        assert data["camera_model"] == "OPENCV"
        wide, ultra = data["frames"]
        assert all(wide[name] == 0.0 for name in COEFFICIENT_NAMES)
        assert ultra["k1"] > 0.0 and ultra["k2"] < 0.0
        assert json.loads((source / "transforms.json").read_text())["camera_model"] == "PINHOLE"
        camera_lines = (output / "sparse" / "0" / "cameras.txt").read_text().splitlines()
        assert camera_lines[0].split()[1] == "OPENCV"
        assert len(camera_lines[0].split()) == 12
        assert camera_lines[1].split()[1] == "OPENCV"

    print("add_ultrawide_distortion self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
