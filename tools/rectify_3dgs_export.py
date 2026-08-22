#!/usr/bin/env python3
"""Rectify the ultra-wide arm of an exported 3DGS dataset.

The multi-camera exporter deliberately keeps the raw ultra-wide image.  That
is useful for pose estimation, but the 3DGS export currently declares every
frame as a pinhole camera with zero distortion.  This tool makes that
declaration true for the ultra-wide arm while leaving the wide-arm poses and
the shared point cloud unchanged.

    python3 tools/rectify_3dgs_export.py /tmp/gs_d06152_depth_x4 \
        /tmp/gs_d06152_depth_x4_rectified --hfov 102.5

The output is a new export directory; the input is never modified.  The
calibration lookup table is the factory calibration copied under ``calib/``.
The target FOV must not request pixels outside that calibrated source cone.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - the CLI reports this clearly.
    cv2 = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rectify_ultrawide import Rectifier  # noqa: E402
from uw_field_angle_transfer import Transfer  # noqa: E402


DEFAULT_CALIBRATION = (Path(__file__).resolve().parent.parent
                       / "calib" / "iphone17-1_ultrawide.json")


class _FieldAngleRectifier:
    """`Rectifier`'s interface, backed by the field-angle transfer.

    The default path reprojects onto a requested `--hfov` and reads the factory
    table at calibration radii, which assumes the recorded frame is a resample
    of the calibrated one.  This one instead re-indexes the table onto the
    active format's own pixel grid through the field angle -- see
    `docs/UW_TRANSFER_PREREG.md` -- and, unless a target `hfov` is asked for,
    leaves the framing alone: the output is the same camera with the lens model
    removed, at its own paraxial focal length.
    """

    def __init__(self, calib: dict, active_hfov: float, width: int, height: int,
                 hfov: float | None = None):
        self.transfer = Transfer(calib, width, height, active_hfov,
                                 mode="field-angle")
        self.hfov = hfov
        self.out_w, self.out_h = width, height
        self.f_out = (self.transfer.f_paraxial if hfov is None
                      else (width / 2.0) / np.tan(np.radians(hfov) / 2.0))
        self._maps = self.transfer.image_maps(width, height, f_out=self.f_out)

    def maps(self, img_w: int, img_h: int, correct: bool = True):
        if (img_w, img_h) != (self.out_w, self.out_h):
            transfer = Transfer(self.transfer.calibration, img_w, img_h,
                                self.transfer.hfov, mode="field-angle")
            f_out = (transfer.f_paraxial if self.hfov is None
                     else (img_w / 2.0) / np.tan(np.radians(self.hfov) / 2.0))
            return transfer.image_maps(img_w, img_h, f_out=f_out)
        return self._maps

    def coverage(self) -> dict:
        # Pixel-centre coordinates, so the sampleable range is [-0.5, n-0.5];
        # the source here is the recorded frame itself rather than a much
        # larger calibration grid, so the half pixel at each edge is the
        # difference between "covered" and a spurious rim of failures.
        map_x, map_y = self._maps
        outside = ((map_x < -0.5) | (map_y < -0.5)
                   | (map_x > self.out_w - 0.5) | (map_y > self.out_h - 0.5))
        return {"outside": int(outside.sum()), "total": int(outside.size),
                "fraction": float(outside.mean())}

    def intrinsics(self) -> dict[str, float]:
        return {"fl_x": float(self.f_out), "fl_y": float(self.f_out),
                "cx": float(self.out_w / 2.0), "cy": float(self.out_h / 2.0)}


def rectified_intrinsics(width: int, height: int, hfov: float) -> dict[str, float]:
    """Return the square-pixel pinhole intrinsics for the output grid."""
    if width < 1 or height < 1 or not np.isfinite(hfov) or not 0.0 < hfov < 180.0:
        raise ValueError("width, height and hfov must describe a real camera")
    f = (width / 2.0) / np.tan(np.radians(hfov) / 2.0)
    return {"fl_x": float(f), "fl_y": float(f),
            "cx": float(width / 2.0), "cy": float(height / 2.0)}


def _remap(array: np.ndarray, rectifier: Rectifier, *, interpolation: int
           ) -> np.ndarray:
    """Apply the rectifier's output-pixel -> captured-pixel map."""
    if cv2 is None:
        raise RuntimeError("this tool needs opencv-python-headless")
    h, w = array.shape[:2]
    map_x, map_y = rectifier.maps(w, h, correct=True)
    return cv2.remap(array, map_x, map_y, interpolation,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _rectify_image(path: Path, rectifier: Rectifier) -> None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"cannot read image {path}")
    cv2.imwrite(str(path), _remap(image, rectifier, interpolation=cv2.INTER_LANCZOS4),
                [cv2.IMWRITE_JPEG_QUALITY, 95])


def _rectify_depth(path: Path, rectifier: Rectifier) -> None:
    if path.suffix.lower() == ".npy":
        raw = np.load(path)
        had_channel = raw.ndim == 3 and raw.shape[-1] == 1
        if had_channel:
            raw = raw[..., 0]
        if raw.ndim != 2:
            raise SystemExit(f"depth array must be 2-D or HxWx1: {path}")
        out = _remap(raw, rectifier, interpolation=cv2.INTER_NEAREST)
        np.save(path, out[..., None] if had_channel else out)
        return

    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise SystemExit(f"cannot read depth {path}")
    out = _remap(depth, rectifier, interpolation=cv2.INTER_NEAREST)
    if not cv2.imwrite(str(path), out):
        raise SystemExit(f"cannot write depth {path}")


def _rectify_mask(path: Path, rectifier: Rectifier) -> None:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise SystemExit(f"cannot read depth mask {path}")
    out = _remap(mask, rectifier, interpolation=cv2.INTER_NEAREST)
    if not cv2.imwrite(str(path), out):
        raise SystemExit(f"cannot write depth mask {path}")


def _camera_ids_by_name(path: Path) -> dict[str, int]:
    result = {}
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
    out = []
    for line in lines:
        if line.startswith("#") or not line.strip():
            out.append(line)
            continue
        fields = line.split()
        camera_id = int(fields[0])
        if camera_id in updates:
            values = updates[camera_id]
            fields[4:8] = [f"{values[key]:.9f}"
                           for key in ("fl_x", "fl_y", "cx", "cy")]
            newline = "\n" if line.endswith("\n") else ""
            out.append(" ".join(fields) + newline)
        else:
            out.append(line)
    path.write_text("".join(out))


def rectify_export(model: str, out: str, *, hfov: float | None = None,
                   calibration: str | None = None,
                   field_angle_hfov: float | None = None) -> dict[str, object]:
    """Copy and rectify one export, returning a small provenance report."""
    if cv2 is None:
        raise SystemExit("this tool needs opencv-python-headless")
    source = Path(model).resolve()
    destination = Path(out).resolve()
    if source == destination:
        raise SystemExit("input and output export directories must differ")
    if not source.is_dir():
        raise SystemExit(f"export directory not found: {source}")
    if destination.exists():
        raise SystemExit(f"output already exists: {destination}")

    transforms_path = source / "transforms.json"
    if not transforms_path.exists():
        raise SystemExit(f"not a 3DGS export: {transforms_path} is missing")
    transforms = json.loads(transforms_path.read_text())
    frames = transforms.get("frames", [])
    target_frames = [frame for frame in frames
                     if frame.get("lens") == "ultrawide"]
    if not target_frames:
        raise SystemExit("the export has no frames marked lens=ultrawide")

    calib_path = Path(calibration).resolve() if calibration else DEFAULT_CALIBRATION
    if not calib_path.exists():
        raise SystemExit(f"calibration not found: {calib_path}")
    calib = json.loads(calib_path.read_text())

    # Copy files rather than preserving symlinks. An export made with
    # `--images symlink` must not let an in-place remap of the copy overwrite
    # the source capture.
    shutil.copytree(source, destination, symlinks=False)

    camera_updates: dict[int, dict[str, float]] = {}
    images_txt = destination / "sparse" / "0" / "images.txt"
    camera_ids = _camera_ids_by_name(images_txt)
    if len(camera_ids) < len(frames):
        raise SystemExit("sparse/0/images.txt does not describe every frame")

    changed = 0
    for frame in frames:
        if frame.get("lens") != "ultrawide":
            continue
        image_rel = Path(frame["file_path"])
        image_path = destination / image_rel
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"cannot read image {image_path}")
        height, width = image.shape[:2]
        rectifier = (_FieldAngleRectifier(calib, field_angle_hfov, width,
                                          height, hfov)
                     if field_angle_hfov else Rectifier(calib, hfov, width,
                                                        height))
        coverage = rectifier.coverage()
        if coverage["outside"]:
            raise SystemExit(
                f"{frame['file_path']}: target hfov "
                f"{hfov if hfov else rectifier.f_out:g} requests "
                f"{coverage['outside']} of {coverage['total']} pixels outside "
                "the calibrated source cone")
        _rectify_image(image_path, rectifier)

        depth_rel = frame.get("depth_file_path")
        if depth_rel:
            depth_path = destination / depth_rel
            if depth_path.exists():
                depth_image = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
                if depth_path.suffix.lower() == ".npy":
                    depth_array = np.load(depth_path)
                    depth_shape = depth_array.shape[:2]
                elif depth_image is not None:
                    depth_shape = depth_image.shape[:2]
                else:
                    raise SystemExit(f"cannot read depth {depth_path}")
                depth_rectifier = (
                    _FieldAngleRectifier(calib, field_angle_hfov,
                                         depth_shape[1], depth_shape[0], hfov)
                    if field_angle_hfov
                    else Rectifier(calib, hfov, depth_shape[1], depth_shape[0]))
                _rectify_depth(depth_path, depth_rectifier)
                mask_path = destination / "depth_normals_mask" / f"{depth_path.stem}.jpg"
                if mask_path.exists():
                    _rectify_mask(mask_path, depth_rectifier)

        values = (rectifier.intrinsics() if field_angle_hfov
                  else rectified_intrinsics(width, height, hfov))
        frame.update(values)
        camera_updates[camera_ids[Path(frame["file_path"]).name]] = values
        changed += 1

    transforms["ultrawide_rectified"] = True
    if hfov is not None:
        transforms["ultrawide_rectification_hfov"] = float(hfov)
    if field_angle_hfov:
        transforms["ultrawide_rectification_transfer"] = "field-angle"
        transforms["ultrawide_active_hfov"] = float(field_angle_hfov)
    transforms["ultrawide_rectification_calibration"] = str(calib_path)
    transforms_path = destination / "transforms.json"
    transforms_path.write_text(json.dumps(transforms, indent=1) + "\n")
    _update_cameras(destination / "sparse" / "0" / "cameras.txt", camera_updates)

    export_report = destination / "export.json"
    if export_report.exists():
        report = json.loads(export_report.read_text())
        report["ultrawide_rectified"] = True
        if hfov is not None:
            report["ultrawide_rectification_hfov"] = float(hfov)
        if field_angle_hfov:
            report["ultrawide_rectification_transfer"] = "field-angle"
            report["ultrawide_active_hfov"] = float(field_angle_hfov)
        report["ultrawide_rectification_calibration"] = str(calib_path)
        export_report.write_text(json.dumps(report, indent=1) + "\n")

    return {"input": str(source), "output": str(destination),
            "ultrawide_frames": changed,
            "hfov": float(hfov) if hfov is not None else None,
            "field_angle_hfov": (float(field_angle_hfov)
                                 if field_angle_hfov else None),
            "calibration": str(calib_path)}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", help="existing export_3dgs.py directory")
    ap.add_argument("out", help="new export directory; must not already exist")
    ap.add_argument("--hfov", type=float, default=None,
                    help="target pinhole horizontal FOV in degrees; required "
                         "unless --field-angle-hfov is given, in which case it "
                         "is optional and omitting it keeps the framing")
    ap.add_argument("--field-angle-hfov", type=float, default=None,
                    help="the ACTIVE format's videoFieldOfView, e.g. 106.2007. "
                         "Opt-in: re-indexes the factory table onto the active "
                         "format's pixel grid through the field angle instead "
                         "of assuming the frame is a resample of the "
                         "calibrated one. See docs/UW_TRANSFER_PREREG.md")
    ap.add_argument("--calibration", default=None,
                    help="ultra-wide calibration JSON; defaults to calib/")
    args = ap.parse_args(argv)
    if args.hfov is None and args.field_angle_hfov is None:
        ap.error("one of --hfov or --field-angle-hfov is required")
    print(json.dumps(rectify_export(args.model, args.out, hfov=args.hfov,
                                    calibration=args.calibration,
                                    field_angle_hfov=args.field_angle_hfov),
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
