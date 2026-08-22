#!/usr/bin/env python3
"""Add an OpenCV camera model to the raw ultra-wide arm of an export.

The multi-camera exporter keeps the raw ultra-wide image, but historically
declared every frame as a zero-distortion ``PINHOLE`` camera.  Apple gives us a
radial lookup table rather than OpenCV coefficients.  This tool fits that
lookup, in normalized coordinates, to Nerfstudio's OpenCV model and writes a
new export.  DN-Splatter then applies the inverse distortion while generating
rays, so the image is not resampled and the wider field of view is retained.

    python3 tools/add_ultrawide_distortion.py /tmp/gs_raw /tmp/gs_opencv

Only frames marked ``lens=ultrawide`` receive non-zero coefficients.  The
global camera model is ``OPENCV`` because Nerfstudio uses one camera type for
the dataset; the wide arm receives six zero coefficients.  The per-frame
``k1..k4,p1,p2`` values in ``transforms.json`` are the source of truth.  The
COLMAP text model stores the compatible four-coefficient subset.

The input is never modified and the output directory must not already exist.
Requires NumPy.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uw_field_angle_transfer import Transfer  # noqa: E402


DEFAULT_CALIBRATION = (Path(__file__).resolve().parent.parent
                       / "calib" / "iphone17-1_ultrawide.json")
COEFFICIENT_NAMES = ("k1", "k2", "k3", "k4", "p1", "p2")


def _max_radius(centre: np.ndarray, width: float, height: float) -> float:
    dx = max(float(centre[0]), float(width) - float(centre[0]))
    dy = max(float(centre[1]), float(height) - float(centre[1]))
    return math.hypot(dx, dy)


def _apply_lookup(points: np.ndarray, calibration: dict[str, Any],
                  table_name: str) -> np.ndarray:
    """Map calibration-plane pixels with one of Apple's radial tables."""
    reference_width, reference_height = calibration["reference_dimensions"]
    intrinsics = calibration["intrinsics"]
    centre = np.asarray(calibration.get("lens_distortion_center") or [
        intrinsics["cx"], intrinsics["cy"]
    ], dtype=np.float64)
    table = np.asarray(calibration.get(table_name)
                       or [0.0, 0.0], dtype=np.float64)
    if table.size < 2:
        return points.copy()

    radius_max = _max_radius(centre, reference_width, reference_height)
    radius = np.linalg.norm(points - centre, axis=-1)
    position = radius * (table.size - 1) / radius_max
    index = np.clip(position.astype(np.int64), 0, table.size - 2)
    fraction = np.clip(position - index, 0.0, 1.0)
    interpolated = ((1.0 - fraction) * table[index]
                    + fraction * table[index + 1])
    magnification = np.where(radius < radius_max, interpolated, table[-1])
    return centre + (points - centre) * (1.0 + magnification)[..., None]


def _sample_grid(width: int, height: int) -> np.ndarray:
    """Pixel-centre grid with enough samples to fit the fourth radial term."""
    nx = min(129, max(33, width))
    ny = min(129, max(33, int(round(nx * height / width))))
    xs = np.linspace(0.5, width - 0.5, nx, dtype=np.float64)
    ys = np.linspace(0.5, height - 0.5, ny, dtype=np.float64)
    u, v = np.meshgrid(xs, ys)
    return np.stack([u, v], axis=-1)


def _design_matrix(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r2 = x * x + y * y
    radial_x = [x * r2 ** power for power in range(1, 5)]
    radial_y = [y * r2 ** power for power in range(1, 5)]
    # OpenCV/Nerfstudio order is k1,k2,k3,k4,p1,p2.
    x_matrix = np.stack(radial_x + [2.0 * x * y, r2 + 2.0 * x * x], axis=-1)
    y_matrix = np.stack(radial_y + [r2 + 2.0 * y * y, 2.0 * x * y], axis=-1)
    return x_matrix.reshape(-1, 6), y_matrix.reshape(-1, 6)


def fit_opencv_coefficients(
        calibration: dict[str, Any], frame: dict[str, Any],
        table_name: str = "inverse_lens_distortion_lookup_table",
        active_hfov: float | None = None,
        ) -> dict[str, Any]:
    """Fit Apple's ideal->raw map to normalized OpenCV coefficients.

    The lookup is quoted in a factory reference pixel grid.  Converting the
    active frame to that grid through its own ``fl_x/fl_y/cx/cy`` preserves the
    ray angle even when the recorder downscaled or selected a different aspect
    ratio.  The returned coefficients are dimensionless and therefore remain
    valid after the trainer rescales the output resolution.
    """
    width, height = int(frame["w"]), int(frame["h"])
    fx, fy = float(frame["fl_x"]), float(frame["fl_y"])
    cx, cy = float(frame["cx"]), float(frame["cy"])
    if width < 1 or height < 1 or min(fx, fy) <= 0.0:
        raise ValueError("frame has invalid dimensions or focal lengths")

    if active_hfov:
        # Opt-in: index the table by field angle rather than by pretending the
        # frame is a resample of the calibrated one.  The conversion between
        # the two pixel grids is then a single scalar and the frame's own
        # `fl_x` -- which the caller has already replaced with the transferred
        # paraxial focal length -- never enters the geometry.
        # See docs/UW_TRANSFER_PREREG.md.
        transfer = Transfer(calibration, width, height, float(active_hfov),
                            mode="field-angle")
        ideal = _sample_grid(width, height)
        raw = (transfer.ideal_to_raw(ideal) if table_name
               == "lens_distortion_lookup_table"
               else transfer.raw_to_ideal(ideal))
        x = (ideal[..., 0] - cx) / fx
        y = (ideal[..., 1] - cy) / fy
        distorted_x = (raw[..., 0] - cx) / fx
        distorted_y = (raw[..., 1] - cy) / fy
        design_x, design_y = _design_matrix(x, y)
        coefficients = np.linalg.lstsq(
            np.vstack([design_x, design_y]),
            np.concatenate([(distorted_x - x).reshape(-1),
                            (distorted_y - y).reshape(-1)]),
            rcond=None)[0]
        predicted_x = x.reshape(-1) + design_x @ coefficients
        predicted_y = y.reshape(-1) + design_y @ coefficients
        residual_px = np.hypot(predicted_x - distorted_x.reshape(-1),
                               predicted_y - distorted_y.reshape(-1))
        pixel_error = residual_px * max(fx, fy)
        values = {name: float(value)
                  for name, value in zip(COEFFICIENT_NAMES, coefficients)}
        values.update({
            "fit_rmse_px": float(np.sqrt(np.mean(pixel_error ** 2))),
            "fit_median_px": float(np.median(pixel_error)),
            "fit_p95_px": float(np.percentile(pixel_error, 95)),
            "fit_max_px": float(np.max(pixel_error)),
            "fit_samples": int(ideal.shape[0] * ideal.shape[1]),
            "fit_reference_dimensions": [width, height],
            "fit_table": table_name,
            "fit_transfer": "field-angle",
            "fit_transfer_k": transfer.k,
            "fit_f_paraxial_px": transfer.f_paraxial,
        })
        return values

    factory_width, factory_height = calibration["reference_dimensions"]
    factory_intrinsics = calibration["intrinsics"]
    factory_fx, factory_fy = (float(factory_intrinsics["fx"]),
                              float(factory_intrinsics["fy"]))
    factory_centre = np.asarray(
        calibration.get("lens_distortion_center") or [
            factory_intrinsics["cx"], factory_intrinsics["cy"]
        ], dtype=np.float64)

    ideal = _sample_grid(width, height)
    ideal_factory = factory_centre + np.stack([
        (ideal[..., 0] - cx) * factory_fx / fx,
        (ideal[..., 1] - cy) * factory_fy / fy,
    ], axis=-1)
    raw_factory = _apply_lookup(ideal_factory, calibration, table_name)
    raw = np.stack([
        cx + (raw_factory[..., 0] - factory_centre[0]) * fx / factory_fx,
        cy + (raw_factory[..., 1] - factory_centre[1]) * fy / factory_fy,
    ], axis=-1)

    x = (ideal[..., 0] - cx) / fx
    y = (ideal[..., 1] - cy) / fy
    distorted_x = (raw[..., 0] - cx) / fx
    distorted_y = (raw[..., 1] - cy) / fy
    design_x, design_y = _design_matrix(x, y)
    coefficients = np.linalg.lstsq(
        np.vstack([design_x, design_y]),
        np.concatenate([(distorted_x - x).reshape(-1),
                        (distorted_y - y).reshape(-1)]),
        rcond=None,
    )[0]

    predicted_x = x.reshape(-1) + design_x @ coefficients
    predicted_y = y.reshape(-1) + design_y @ coefficients
    residual_px = np.hypot(predicted_x - distorted_x.reshape(-1),
                           predicted_y - distorted_y.reshape(-1))
    # Use the larger focal length for a conservative pixel-equivalent report.
    pixel_error = residual_px * max(fx, fy)
    values = {name: float(value)
              for name, value in zip(COEFFICIENT_NAMES, coefficients)}
    values.update({
        "fit_rmse_px": float(np.sqrt(np.mean(pixel_error ** 2))),
        "fit_median_px": float(np.median(pixel_error)),
        "fit_p95_px": float(np.percentile(pixel_error, 95)),
        "fit_max_px": float(np.max(pixel_error)),
        "fit_samples": int(ideal.shape[0] * ideal.shape[1]),
        "fit_reference_dimensions": [int(factory_width), int(factory_height)],
        "fit_table": table_name,
    })
    return values


def _zero_coefficients() -> dict[str, float]:
    return {name: 0.0 for name in COEFFICIENT_NAMES}


def _camera_ids_by_name(path: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    with path.open() as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.split()
            if len(fields) >= 10:
                result[fields[9]] = int(fields[8])
    return result


def _update_cameras(path: Path, updates: dict[int, dict[str, float]]) -> None:
    lines = path.read_text().splitlines(keepends=True)
    output = []
    for line in lines:
        if line.startswith("#") or not line.strip():
            output.append(line)
            continue
        fields = line.split()
        camera_id = int(fields[0])
        if camera_id not in updates:
            output.append(line)
            continue
        values = updates[camera_id]
        params = [values["fl_x"], values["fl_y"], values["cx"], values["cy"],
                  values["k1"], values["k2"], values["p1"], values["p2"]]
        newline = "\n" if line.endswith("\n") else ""
        output.append(" ".join([
            fields[0], "OPENCV", fields[2], fields[3],
            *(f"{value:.9f}" for value in params)
        ]) + newline)
    path.write_text("".join(output))


def add_distortion(model: str, out: str, *, calibration: str | None = None,
                   lens: str = "ultrawide",
                   table_name: str = "inverse_lens_distortion_lookup_table",
                   active_hfov: float | None = None
                   ) -> dict[str, Any]:
    source = Path(model).resolve()
    destination = Path(out).resolve()
    if source == destination:
        raise SystemExit("input and output export directories must differ")
    if not source.is_dir():
        raise SystemExit(f"export directory not found: {source}")
    if destination.exists():
        raise SystemExit(f"output already exists: {destination}")

    transforms_path = source / "transforms.json"
    cameras_path = source / "sparse" / "0" / "cameras.txt"
    images_path = source / "sparse" / "0" / "images.txt"
    if not transforms_path.exists() or not cameras_path.exists() or not images_path.exists():
        raise SystemExit("not a complete 3DGS export: transforms/sparse files missing")
    transforms = json.loads(transforms_path.read_text())
    frames = transforms.get("frames", [])
    targets = [frame for frame in frames if frame.get("lens") == lens]
    if not targets:
        raise SystemExit(f"the export has no frames marked lens={lens}")

    calibration_path = (Path(calibration).resolve()
                        if calibration else DEFAULT_CALIBRATION)
    if not calibration_path.exists():
        raise SystemExit(f"calibration not found: {calibration_path}")
    calibration_data = json.loads(calibration_path.read_text())

    # Never preserve symlinks: writing metadata into a copied export must not
    # mutate the source capture when the input used --images symlink.
    shutil.copytree(source, destination, symlinks=False)
    camera_ids = _camera_ids_by_name(destination / "sparse" / "0" / "images.txt")
    if len(camera_ids) < len(frames):
        raise SystemExit("sparse/0/images.txt does not describe every frame")

    updates: dict[int, dict[str, float]] = {}
    reports = []
    for frame in frames:
        if frame.get("lens") == lens and active_hfov:
            # The frame's declared focal length came from
            # (W/2)/tan(videoFieldOfView/2), which is not a paraxial focal
            # length because `videoFieldOfView` already contains the
            # distortion.  A distortion model normalised by the wrong focal
            # length is not the same model, so it is replaced here.
            transfer = Transfer(calibration_data, int(frame["w"]),
                                int(frame["h"]), float(active_hfov),
                                mode="field-angle")
            frame["fl_x"] = frame["fl_y"] = float(transfer.f_paraxial)
        coeffs = (fit_opencv_coefficients(calibration_data, frame, table_name,
                                          active_hfov)
                  if frame.get("lens") == lens else _zero_coefficients())
        for name in COEFFICIENT_NAMES:
            frame[name] = coeffs[name]
        camera_name = Path(frame["file_path"]).name
        if camera_name not in camera_ids:
            raise SystemExit(f"no COLMAP image entry for {camera_name}")
        updates[camera_ids[camera_name]] = {
            "fl_x": float(frame["fl_x"]), "fl_y": float(frame["fl_y"]),
            "cx": float(frame["cx"]), "cy": float(frame["cy"]),
            **{name: coeffs[name] for name in COEFFICIENT_NAMES},
        }
        if frame.get("lens") == lens:
            reports.append({"file_path": frame["file_path"], **coeffs})

    transforms["camera_model"] = "OPENCV"
    if active_hfov:
        transforms["ultrawide_distortion_transfer"] = "field-angle"
        transforms["ultrawide_active_hfov"] = float(active_hfov)
    transforms["ultrawide_distortion_model"] = f"apple_{table_name}_fit"
    transforms["ultrawide_distortion_calibration"] = str(calibration_path)
    transforms["ultrawide_distortion_table"] = table_name
    transforms["ultrawide_distortion_coefficients"] = {
        "order": list(COEFFICIENT_NAMES),
        "frames": len(reports),
        "max_fit_p95_px": max(item["fit_p95_px"] for item in reports),
        "max_fit_max_px": max(item["fit_max_px"] for item in reports),
    }
    (destination / "transforms.json").write_text(
        json.dumps(transforms, indent=1) + "\n")
    _update_cameras(destination / "sparse" / "0" / "cameras.txt", updates)

    export_report_path = destination / "export.json"
    if export_report_path.exists():
        report = json.loads(export_report_path.read_text())
        report["camera_model"] = "OPENCV"
        report["ultrawide_distortion_model"] = f"apple_{table_name}_fit"
        report["ultrawide_distortion_calibration"] = str(calibration_path)
        report["ultrawide_distortion_table"] = table_name
        export_report_path.write_text(json.dumps(report, indent=1) + "\n")

    return {
        "input": str(source), "output": str(destination),
        "camera_model": "OPENCV", "ultrawide_frames": len(reports),
        "calibration": str(calibration_path),
        "table": table_name,
        "max_fit_p95_px": transforms["ultrawide_distortion_coefficients"]["max_fit_p95_px"],
        "max_fit_max_px": transforms["ultrawide_distortion_coefficients"]["max_fit_max_px"],
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", help="existing export_3dgs.py directory")
    parser.add_argument("out", help="new export directory; must not already exist")
    parser.add_argument("--calibration", default=None,
                        help="ultra-wide calibration JSON; defaults to calib/")
    parser.add_argument("--lens", default="ultrawide",
                        help="frame lens receiving the fitted coefficients")
    parser.add_argument("--table", dest="table_name",
                        choices=("inverse_lens_distortion_lookup_table",
                                 "lens_distortion_lookup_table"),
                        default="inverse_lens_distortion_lookup_table",
                        help="Apple table used for ideal-to-raw fit; inverse is "
                             "the documented direction, the other is an ablation")
    parser.add_argument("--field-angle-hfov", type=float, default=None,
                        help="the ACTIVE format's videoFieldOfView, e.g. "
                             "106.2007. Opt-in: index the factory table by "
                             "field angle and replace the frame's fl_x/fl_y "
                             "with the transferred paraxial focal length. "
                             "See docs/UW_TRANSFER_PREREG.md")
    args = parser.parse_args(argv)
    print(json.dumps(add_distortion(args.model, args.out,
                                    calibration=args.calibration,
                                    lens=args.lens,
                                    table_name=args.table_name,
                                    active_hfov=args.field_angle_hfov),
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
