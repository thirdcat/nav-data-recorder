#!/usr/bin/env python3
"""Export a session as a Gaussian-splatting training set.

Structure-from-motion is the step every 3DGS pipeline starts with, and it is
the step this recorder does not need: ARKit already hands back a pose per frame
and the LiDAR already hands back metres. So this writes the COLMAP model that a
trainer expects to *find*, rather than the images it would run SfM on — poses
converted, not solved, and the initial point cloud back-projected from depth
instead of triangulated.

    python3 tools/export_3dgs.py ~/nav_data/20260813-162849-2994fa /tmp/gs

Three properties of this capture drive the choices here, and each is a place
where the obvious export would be wrong:

**Autofocus is on.** `fx` drifts by up to 11% within one session and `cx` walks
across the image centre, so the export writes one COLMAP camera *per image*
rather than one per session. A shared camera model silently absorbs that drift
into the geometry.

**Depth is registered to colour but is not the same size.** 256x192 is exactly
1/7.5 of 1920x1440, so depth intrinsics come from the dimension ratio. They are
never derived from `2*cx`: the principal point is a real optical quantity that
moves with focus, and using it as a width proxy injects that motion into the
back-projection.

**The world origin can move mid-session.** ARKit re-origins after an
interruption, so frames on either side of `ar.interruptionEnded` are not in one
coordinate system. They are exported as separate segments rather than being
concatenated into a reconstruction that cannot converge.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import struct
import sys
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from read_session import Session  # noqa: E402

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None


# ARKit's camera looks down -Z with +Y up. COLMAP's — and the depth map's — is
# +Z forward with +Y down. The same basis change `depth_odometry.py` uses, and
# deliberately the same constant, so the two never drift apart.
ARKIT_TO_COLMAP = np.diag([1.0, -1.0, -1.0])

# Depth outside this band is not evidence. The near limit is where the LiDAR
# stops returning; the far limit is where its noise exceeds the voxel size the
# init cloud is built at, measured in docs/POSE.md.
DEPTH_NEAR_M = 0.15
DEPTH_FAR_M = 5.0


def quat_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n == 0.0:
        raise ValueError("zero quaternion")
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def matrix_to_quat(R: np.ndarray) -> tuple[float, float, float, float]:
    """Rotation matrix to (qw, qx, qy, qz) — COLMAP's scalar-first order."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw, qx = 0.25 * s, (R[2, 1] - R[1, 2]) / s
        qy, qz = (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw, qx = (R[2, 1] - R[1, 2]) / s, 0.25 * s
        qy, qz = (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw, qx = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s
        qy, qz = 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw, qx = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s
        qy, qz = (R[1, 2] + R[2, 1]) / s, 0.25 * s
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    return qw / n, qx / n, qy / n, qz / n


def camera_to_world(pose: dict[str, Any]) -> np.ndarray:
    """The stored ARKit pose as a 4x4 world-from-camera in COLMAP camera axes."""
    R = quat_to_matrix(pose["qx"], pose["qy"], pose["qz"], pose["qw"]) @ ARKIT_TO_COLMAP
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = (pose["tx"], pose["ty"], pose["tz"])
    return T


def world_to_camera(pose: dict[str, Any]) -> tuple[tuple[float, ...], np.ndarray]:
    """COLMAP's `images.txt` pair: (qw, qx, qy, qz) and the translation."""
    c2w = camera_to_world(pose)
    R = c2w[:3, :3].T
    t = -R @ c2w[:3, 3]
    return matrix_to_quat(R), t


def apply_guard_band(rows: list[dict[str, Any]], holdout: list[int], *,
                     radius_m: float, angle_deg: float
                     ) -> tuple[list[dict[str, Any]], list[int], int]:
    """Drop training frames that sit almost on top of a held-out one.

    Every eighth frame of a 5 Hz walk is 12 cm from the frames on either side of
    it, so the standard split asks a model to interpolate between two views it
    was given rather than to reconstruct anything. Measured on one session: 6 of
    23 held-out views had a training camera within 10 cm *and* 10 degrees.

    A guard band removes those neighbours from training. The held-out views stay
    spread through the walk — a contiguous block would instead ask the model
    about a part of the room nobody visited, which is a different and much
    harder question — but they stop having a near-duplicate to copy.

    Returns the surviving rows, the held-out indices renumbered into them, and
    how many training frames the band cost.
    """
    if radius_m <= 0 or not holdout:
        return rows, holdout, 0
    held = set(holdout)
    centres = np.array([camera_to_world(r["pose"])[:3, 3] for r in rows])
    forwards = np.array([camera_to_world(r["pose"])[:3, 2] for r in rows])
    cos_limit = math.cos(math.radians(angle_deg))

    keep = []
    for i, row in enumerate(rows):
        if i in held:
            keep.append(i)
            continue
        near = np.linalg.norm(centres[list(held)] - centres[i], axis=1) < radius_m
        if near.any():
            aligned = forwards[list(held)][near] @ forwards[i] > cos_limit
            if aligned.any():
                continue
        keep.append(i)

    renumber = {old: new for new, old in enumerate(keep)}
    return ([rows[i] for i in keep],
            sorted(renumber[i] for i in holdout),
            len(rows) - len(keep))


def rebase_pose(pose: dict[str, Any], transform: np.ndarray) -> dict[str, Any]:
    """The same camera, expressed in another session's world.

    Rewriting the pose is what lets everything downstream stay single-session:
    the exporter, the COLMAP writer and the back-projection all read `pose` and
    none of them has to learn that several worlds were involved. The transform
    acts on the ARKit world, so it composes on the left of the ARKit pose — and
    the quaternion goes back in ARKit's scalar-last order, which is the order
    the rest of the file expects to read.
    """
    R = quat_to_matrix(pose["qx"], pose["qy"], pose["qz"], pose["qw"])
    t = np.array([pose["tx"], pose["ty"], pose["tz"]])
    R_new = transform[:3, :3] @ R
    t_new = transform[:3, :3] @ t + transform[:3, 3]
    qw, qx, qy, qz = matrix_to_quat(R_new)
    moved = dict(pose)
    moved.update({"qx": qx, "qy": qy, "qz": qz, "qw": qw,
                  "tx": float(t_new[0]), "ty": float(t_new[1]), "tz": float(t_new[2])})
    return moved


def sharpness(path: str, long_side: int = 960) -> float:
    """Variance of a Laplacian — small on a blurred frame, small in the dark.

    Deliberately not thresholded here. On a night session the whole population
    shifts down by two orders of magnitude, so an absolute cut would reject the
    session rather than its worst frames; the caller compares against the
    session's own median instead.
    """
    if Image is None:
        return float("nan")
    with Image.open(path) as im:
        im = im.convert("L")
        scale = long_side / max(im.size)
        if scale < 1.0:
            im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))))
        a = np.asarray(im, dtype=np.float64)
    lap = (a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:] - 4.0 * a[1:-1, 1:-1])
    return float(lap.var())


def segment_frames(session: Session, rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Split on world-origin resets. Poses across one are not comparable."""
    breaks = sorted(e["t"] for e in session.events()
                    if e.get("kind") == "ar.interruptionEnded")
    if not breaks:
        return [rows]
    segments: list[list[dict[str, Any]]] = [[]]
    it = iter(breaks)
    nxt = next(it, None)
    for row in rows:
        while nxt is not None and row["t"] >= nxt:
            segments.append([])
            nxt = next(it, None)
        segments[-1].append(row)
    return [s for s in segments if s]


def select_frames(session: Session, rows: list[dict[str, Any]], *,
                  min_baseline_m: float, sharp_ratio: float,
                  require_exact_depth: bool) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Drop the frames a trainer cannot use, and say how many went to each cause."""
    dropped = {"tracking": 0, "no_depth": 0, "stale_depth": 0, "blurry": 0, "too_close": 0}

    kept: list[dict[str, Any]] = []
    for row in rows:
        if row["pose"].get("tracking") != "normal":
            dropped["tracking"] += 1
            continue
        if row.get("depth") is None:
            dropped["no_depth"] += 1
            continue
        if require_exact_depth and row.get("depth_dt") != 0.0:
            dropped["stale_depth"] += 1
            continue
        kept.append(row)

    if sharp_ratio > 0.0 and Image is not None and kept:
        values = np.array([sharpness(r["path"]) for r in kept])
        for row, value in zip(kept, values):
            row["_sharpness"] = float(value)
        floor = float(np.median(values)) * sharp_ratio
        sharp = [r for r in kept if r["_sharpness"] >= floor]
        dropped["blurry"] = len(kept) - len(sharp)
        kept = sharp

    if min_baseline_m > 0.0 and kept:
        thinned = [kept[0]]
        last = camera_to_world(kept[0]["pose"])[:3, 3]
        for row in kept[1:]:
            here = camera_to_world(row["pose"])[:3, 3]
            if float(np.linalg.norm(here - last)) >= min_baseline_m:
                thinned.append(row)
                last = here
        dropped["too_close"] = len(kept) - len(thinned)
        kept = thinned

    return kept, dropped


def depth_points(session: Session, row: dict[str, Any], *,
                 conf_min: int, near: float, far: float, pix_stride: int,
                 depth: np.ndarray | None = None,
                 conf: np.ndarray | None = None
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Back-project one frame's depth into the world. Returns points, normals, uv."""
    entry = row["depth"]
    z = (np.asarray(session.depth_frame(entry), dtype=np.float32)
         if depth is None else depth)
    if conf is None:
        conf = session.confidence_frame(entry)
    h, w = z.shape

    pose = row["pose"]
    sx, sy = w / row["width"], h / row["height"]
    fx, fy = pose["fx"] * sx, pose["fy"] * sy
    cx, cy = pose["cx"] * sx, pose["cy"] * sy

    vv, uu = np.mgrid[0:h, 0:w]
    ok = np.isfinite(z) & (z > near) & (z < far)
    if conf is not None and conf_min > 0:
        ok &= np.asarray(conf) >= conf_min

    # Normals come from the neighbourhood before subsampling, so a stride does
    # not widen the cross product's footprint and round off every edge.
    px = (uu - cx) * z / fx
    py = (vv - cy) * z / fy
    cam = np.stack([px, py, z], axis=-1)
    du = np.zeros_like(cam)
    dv = np.zeros_like(cam)
    du[:, 1:-1] = cam[:, 2:] - cam[:, :-2]
    dv[1:-1, :] = cam[2:, :] - cam[:-2, :]
    nrm = np.cross(du, dv)
    norm = np.linalg.norm(nrm, axis=-1, keepdims=True)
    nrm = np.divide(nrm, norm, out=np.zeros_like(nrm), where=norm > 1e-12)

    if pix_stride > 1:
        sel = np.zeros_like(ok)
        sel[::pix_stride, ::pix_stride] = True
        ok &= sel

    if not ok.any():
        empty = np.zeros((0, 3))
        return empty, empty, np.zeros((0, 2), dtype=np.int64)

    c2w = camera_to_world(pose)
    pts = cam[ok] @ c2w[:3, :3].T + c2w[:3, 3]
    nrm = nrm[ok] @ c2w[:3, :3].T
    uv = np.stack([uu[ok], vv[ok]], axis=1).astype(np.int64)
    return pts, nrm, uv


def sample_colours(path: str, uv: np.ndarray, depth_shape: tuple[int, int]) -> np.ndarray:
    """Colour for each depth pixel, read at the matching place in the JPEG."""
    if Image is None or uv.size == 0:
        return np.full((len(uv), 3), 128, dtype=np.uint8)
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"))
    h, w = depth_shape
    ratio_y, ratio_x = rgb.shape[0] / h, rgb.shape[1] / w
    ys = np.clip(((uv[:, 1] + 0.5) * ratio_y).astype(np.int64), 0, rgb.shape[0] - 1)
    xs = np.clip(((uv[:, 0] + 0.5) * ratio_x).astype(np.int64), 0, rgb.shape[1] - 1)
    return rgb[ys, xs]


def voxel_average(points: np.ndarray, normals: np.ndarray, colours: np.ndarray,
                  voxel: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fuse to one point per voxel by averaging, not by picking a survivor.

    Picking is cheaper and is what a downsample usually does, but it keeps one
    frame's noise instead of cancelling it across the frames that saw the same
    surface. The count comes back too: it is how many observations a point has,
    which is the only honest confidence available for the init cloud.
    """
    if len(points) == 0:
        return points, normals, colours, np.zeros(0, dtype=np.int64)
    keys = np.floor(points / voxel).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    n = len(counts)

    def accumulate(values: np.ndarray) -> np.ndarray:
        out = np.zeros((n, values.shape[1]), dtype=np.float64)
        for axis in range(values.shape[1]):
            out[:, axis] = np.bincount(inverse, weights=values[:, axis], minlength=n)
        return out / counts[:, None]

    pts = accumulate(points)
    nrm = accumulate(normals)
    norm = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = np.divide(nrm, norm, out=np.zeros_like(nrm), where=norm > 1e-12)
    col = np.clip(accumulate(colours.astype(np.float64)), 0, 255).astype(np.uint8)
    return pts, nrm, col, counts


def write_ply(path: str, points: np.ndarray, normals: np.ndarray,
              colours: np.ndarray) -> None:
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        for p, n, c in zip(points, normals, colours):
            fh.write(struct.pack("<6f3B", p[0], p[1], p[2], n[0], n[1], n[2],
                                 int(c[0]), int(c[1]), int(c[2])))


def write_colmap(out: str, rows: list[dict[str, Any]], names: list[str],
                 sizes: list[tuple[int, int]], scales: list[float],
                 points: np.ndarray, colours: np.ndarray) -> None:
    sparse = os.path.join(out, "sparse", "0")
    os.makedirs(sparse, exist_ok=True)

    with open(os.path.join(sparse, "cameras.txt"), "w") as fh:
        fh.write("# Camera list with one line of data per camera:\n"
                 "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
                 "# One camera per image: autofocus moves fx and cx during a session.\n")
        for i, (row, (w, h), s) in enumerate(zip(rows, sizes, scales), start=1):
            p = row["pose"]
            fh.write(f"{i} PINHOLE {w} {h} "
                     f"{p['fx'] * s:.6f} {p['fy'] * s:.6f} "
                     f"{p['cx'] * s:.6f} {p['cy'] * s:.6f}\n")

    with open(os.path.join(sparse, "images.txt"), "w") as fh:
        fh.write("# Image list with two lines of data per image:\n"
                 "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
                 "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
                 "# Poses are ARKit's, converted — not solved. The second line is\n"
                 "# empty because there are no tracks: the points come from LiDAR.\n")
        for i, (row, name) in enumerate(zip(rows, names), start=1):
            (qw, qx, qy, qz), t = world_to_camera(row["pose"])
            fh.write(f"{i} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} "
                     f"{t[0]:.9f} {t[1]:.9f} {t[2]:.9f} {i} {name}\n\n")

    with open(os.path.join(sparse, "points3D.txt"), "w") as fh:
        fh.write("# 3D point list with one line of data per point:\n"
                 "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        for i, (p, c) in enumerate(zip(points, colours), start=1):
            fh.write(f"{i} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                     f"{int(c[0])} {int(c[1])} {int(c[2])} 0\n")


def holdout_split(count: int, every: int) -> list[int]:
    """Indices reserved for evaluation, or none when `every` is zero.

    Every n-th frame, which on a walk means a view the trainer has never seen
    from a place it has nearly been. That is the honest test for this capture:
    a random split would leave neighbours 12 cm away in the training set, and
    interpolating between two views 12 cm apart is not the question being asked.

    The phase is `i % every == 0` and not something better centred, because that
    is what `gsplat`'s and the reference 3DGS loader's `test_every` means. A
    split that disagrees with the trainer's is worse than no split at all here:
    the frames this export keeps out of the initial point cloud would then be
    training views, and the frames actually being scored would be ones whose
    geometry was handed over at initialisation.
    """
    if every <= 1:
        return []
    return list(range(0, count, every))


def write_transforms(out: str, rows: list[dict[str, Any]], names: list[str],
                     sizes: list[tuple[int, int]], scales: list[float],
                     depth_names: list[str] | None, depth_dir: str = "depths",
                     depth_scale: float = 0.001,
                     holdout: list[int] | None = None) -> None:
    """Nerfstudio's `transforms.json`.

    Its `transform_matrix` is camera-to-world in the OpenGL convention — X
    right, Y up, Z back — which is ARKit's camera frame exactly. So this one
    writes the pose *without* the basis change that COLMAP needs; the two files
    describing the same cameras differ, and that is correct.
    """
    frames = []
    for i, (row, name, (w, h), s) in enumerate(zip(rows, names, sizes, scales)):
        p = row["pose"]
        R = quat_to_matrix(p["qx"], p["qy"], p["qz"], p["qw"])
        m = np.eye(4)
        m[:3, :3] = R
        m[:3, 3] = (p["tx"], p["ty"], p["tz"])
        frame = {
            "file_path": f"images/{name}",
            "transform_matrix": [[float(v) for v in r] for r in m],
            "fl_x": p["fx"] * s, "fl_y": p["fy"] * s,
            "cx": p["cx"] * s, "cy": p["cy"] * s,
            "w": w, "h": h,
            "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0,
        }
        if depth_names is not None:
            frame["depth_file_path"] = f"{depth_dir}/{depth_names[i]}"
        frames.append(frame)

    doc = {
        "camera_model": "PINHOLE",
        "ply_file_path": "sparse_pc.ply",
        "frames": frames,
    }
    if depth_names is not None:
        # Metres per stored unit: 0.001 for millimetre PNGs, 1.0 for metre .npy.
        doc["depth_unit_scale_factor"] = depth_scale
    if holdout:
        held = set(holdout)
        doc["train_filenames"] = [f"images/{n}" for i, n in enumerate(names) if i not in held]
        doc["val_filenames"] = [f"images/{n}" for i, n in enumerate(names) if i in held]
        doc["test_filenames"] = list(doc["val_filenames"])
    with open(os.path.join(out, "transforms.json"), "w") as fh:
        json.dump(doc, fh, indent=1)


def write_image(src: str, dst: str, downscale: int, mode: str) -> tuple[int, int]:
    """Place one training image and report the size it ended up as."""
    if downscale == 1 and mode != "resize":
        with Image.open(src) as im:
            size = im.size
        if mode == "symlink":
            if os.path.lexists(dst):
                os.unlink(dst)
            os.symlink(os.path.abspath(src), dst)
        else:
            shutil.copy2(src, dst)
        return size

    with Image.open(src) as im:
        target = (max(1, im.width // downscale), max(1, im.height // downscale))
        im = im.convert("RGB").resize(target, Image.LANCZOS)
        im.save(dst, quality=95)
    return target


def resample_nearest(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Depth to the training image's size, nearest-neighbour, never bilinear.

    An interpolated depth halfway between a foreground edge and the wall behind
    it is a measurement of nothing, and it lands exactly where a depth loss is
    most confident and most wrong.
    """
    w, h = size
    dh, dw = frame.shape
    ys = np.clip((np.arange(h) + 0.5) * dh / h, 0, dh - 1).astype(np.int64)
    xs = np.clip((np.arange(w) + 0.5) * dw / w, 0, dw - 1).astype(np.int64)
    return frame[np.ix_(ys, xs)]


def write_depth(path: str, entry_depth: np.ndarray, conf: np.ndarray | None,
                size: tuple[int, int], fmt: str, conf_min: int) -> None:
    """Per-image depth in whichever form the trainer reads.

    `png16` holds millimetres in 16 bits, which is compact and is what most
    loaders assume. `npy` holds float32 metres, which is what DN-Splatter's
    parser wants with `--depth-unit-scale-factor 1.0` — and which costs about
    11 MB per full-resolution frame, so it is not the default.
    """
    grid = resample_nearest(entry_depth, size)
    if conf is not None:
        grid = np.where(resample_nearest(np.asarray(conf), size) >= conf_min,
                        grid, 0.0)
    grid = np.nan_to_num(grid, nan=0.0, posinf=0.0, neginf=0.0)
    if fmt == "npy":
        np.save(path, grid.astype(np.float32))
    else:
        Image.fromarray(np.clip(grid * 1000.0, 0, 65535).astype(np.uint16)).save(path)


def write_depth_mask(path: str, conf: np.ndarray | None, size: tuple[int, int],
                     conf_min: int) -> None:
    """DN-Splatter's confidence mask, in DN-Splatter's inverted convention.

    **0 means valid and 255 means invalid.** The loader computes `1 - mask/255`
    and keeps pixels where that is positive, so writing the intuitive polarity
    would train on exactly the pixels the LiDAR said not to trust — and would
    look fine, because the mask is never rendered.

    The mask is binary because that is all the loader can use: it thresholds
    rather than weights, so the ARKit levels are collapsed here rather than
    somewhere the collapse is invisible. Continuous weighting needs a patch to
    the regulariser, not a different file.
    """
    w, h = size
    if conf is None:
        Image.fromarray(np.zeros((h, w), np.uint8)).save(path, quality=95)
        return
    valid = resample_nearest(np.asarray(conf), size) >= conf_min
    Image.fromarray(np.where(valid, 0, 255).astype(np.uint8)).save(path, quality=95)


def export(session_dir: str, out_dir: str, *, downscale: int = 1,
           conf_min: int = 2, voxel: float = 0.02, max_points: int = 1_000_000,
           min_baseline_m: float = 0.0, sharp_ratio: float = 0.0,
           pix_stride: int = 1, image_mode: str = "symlink",
           with_depth: bool = True, require_exact_depth: bool = True,
           segment: int | None = None, holdout_every: int = 8,
           guard_m: float = 0.0, depth_format: str = "png16") -> list[dict[str, Any]]:
    session = Session(session_dir)
    if Image is None:
        raise SystemExit("Pillow is required: pip install pillow")

    all_rows = session.posed_images()
    for row in all_rows:
        row["_session"] = session
    segments = segment_frames(session, all_rows)
    if segment is not None:
        if not 0 <= segment < len(segments):
            raise SystemExit(f"segment {segment} of {len(segments)}")
        segments = [segments[segment]]

    reports = []
    for index, seg_rows in enumerate(segments):
        suffix = "" if len(segments) == 1 else f"_seg{index}"
        out = out_dir + suffix
        rows, dropped = select_frames(
            session, seg_rows, min_baseline_m=min_baseline_m,
            sharp_ratio=sharp_ratio, require_exact_depth=require_exact_depth)
        if len(rows) < 2:
            reports.append({"out": out, "images": len(rows), "skipped": "too few frames",
                            "dropped": dropped})
            continue
        report = write_dataset(out, rows, downscale=downscale, conf_min=conf_min,
                               voxel=voxel, max_points=max_points,
                               pix_stride=pix_stride, image_mode=image_mode,
                               with_depth=with_depth, holdout_every=holdout_every,
                               guard_m=guard_m, depth_format=depth_format)
        report.update({"session": session.id, "segment": index, "dropped": dropped})
        with open(os.path.join(out, "export.json"), "w") as fh:
            json.dump(report, fh, indent=1)
        reports.append(report)
    return reports


def write_dataset(out: str, rows: list[dict[str, Any]], *, downscale: int = 1,
                  conf_min: int = 2, voxel: float = 0.02,
                  max_points: int = 1_000_000, pix_stride: int = 1,
                  image_mode: str = "symlink", with_depth: bool = True,
                  holdout_every: int = 8, holdout_indices: list[int] | None = None,
                  guard_m: float = 0.0, guard_deg: float = 10.0,
                  depth_format: str = "png16") -> dict[str, Any]:
    """Write one training set from a list of rows, whatever sessions they came from.

    Every row carries its own `_session`, so a merged list of several walks goes
    through exactly the path a single walk does. That is the point of rebasing
    the poses upstream rather than teaching this function about frames.
    """
    if True:
        # DN-Splatter reads `depth/` and finds the confidence folder by sorting
        # its contents rather than by a key in the JSON, so the names have to
        # sort into frame order — which `%06d` does.
        depth_dir = "depth" if depth_format == "npy" else "depths"
        os.makedirs(os.path.join(out, "images"), exist_ok=True)
        if with_depth:
            os.makedirs(os.path.join(out, depth_dir), exist_ok=True)
            os.makedirs(os.path.join(out, "depth_normals_mask"), exist_ok=True)

        # A caller merging several sessions dictates the split, so that the
        # single-session arm and the merged arm are scored on the very same
        # photographs. Left to itself the rule would pick every eighth row of a
        # longer list, which is a different set of frames and not a comparison.
        chosen = (holdout_indices if holdout_indices is not None
                  else holdout_split(len(rows), holdout_every))
        rows, chosen, guarded = apply_guard_band(rows, chosen, radius_m=guard_m,
                                                 angle_deg=guard_deg)
        holdout = set(chosen)

        names, sizes, scales, depth_names = [], [], [], []
        pts_all, nrm_all, col_all = [], [], []
        for i, row in enumerate(rows):
            name = f"{i:06d}.jpg"
            size = write_image(row["path"], os.path.join(out, "images", name),
                               downscale, image_mode)
            names.append(name)
            sizes.append(size)
            scales.append(size[0] / row["width"])

            owner = row["_session"]
            depth = np.asarray(owner.depth_frame(row["depth"]), dtype=np.float32)
            conf = owner.confidence_frame(row["depth"])
            # An evaluation frame contributes its image and its depth for
            # scoring, but not its geometry to the initial cloud. Otherwise the
            # trainer starts already holding the answer to the question it is
            # about to be asked, and every held-out number is flattered.
            pts, nrm, uv = ((np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 2), int))
                            if i in holdout else
                            depth_points(owner, row, conf_min=conf_min,
                                         near=DEPTH_NEAR_M, far=DEPTH_FAR_M,
                                         pix_stride=pix_stride, depth=depth, conf=conf))
            if len(pts):
                pts_all.append(pts)
                nrm_all.append(nrm)
                col_all.append(sample_colours(row["path"], uv, depth.shape))

            if with_depth:
                dname = f"{i:06d}.npy" if depth_format == "npy" else f"{i:06d}.png"
                write_depth(os.path.join(out, depth_dir, dname), depth, conf, size,
                            depth_format, conf_min)
                write_depth_mask(os.path.join(out, "depth_normals_mask", f"{i:06d}.jpg"),
                                 conf, size, conf_min)
                depth_names.append(dname)

        points = np.concatenate(pts_all) if pts_all else np.zeros((0, 3))
        normals = np.concatenate(nrm_all) if nrm_all else np.zeros((0, 3))
        colours = (np.concatenate(col_all) if col_all
                   else np.zeros((0, 3), dtype=np.uint8))
        raw_count = len(points)
        points, normals, colours, counts = voxel_average(points, normals, colours, voxel)

        # Thin by observation count, not at random: a point seen once is the
        # one most likely to be a depth-edge artefact.
        if max_points and len(points) > max_points:
            keep = np.argsort(-counts)[:max_points]
            keep.sort()
            points, normals, colours, counts = (points[keep], normals[keep],
                                                colours[keep], counts[keep])

        held = sorted(holdout)
        write_colmap(out, rows, names, sizes, scales, points, colours)
        # The same cloud under both names each lineage looks for: Nerfstudio
        # reads the path named in transforms.json, and the reference 3DGS loader
        # looks for sparse/0/points3D.ply and only falls back to parsing the
        # text model if it is missing.
        write_ply(os.path.join(out, "sparse_pc.ply"), points, normals, colours)
        write_ply(os.path.join(out, "sparse", "0", "points3D.ply"),
                  points, normals, colours)
        write_transforms(out, rows, names, sizes, scales,
                         depth_names if with_depth else None, depth_dir,
                         1.0 if depth_format == "npy" else 0.001, held)
        if held:
            with open(os.path.join(out, "holdout.txt"), "w") as fh:
                fh.write("# image names reserved for evaluation, one per line.\n"
                         "# Their depth is scored against but never fused into\n"
                         "# sparse_pc.ply, so the init cloud does not leak.\n")
                for i in held:
                    fh.write(names[i] + "\n")

        report = {
            "out": out,
            "images": len(rows),
            "holdout": len(held),
            "guard_band_m": guard_m,
            "guarded_out": guarded,
            "points_raw": raw_count,
            "points": len(points),
            "voxel_m": voxel,
            "conf_min": conf_min,
            "downscale": downscale,
            "depth_format": depth_format if with_depth else None,
            "duration_s": round(rows[-1]["t"] - rows[0]["t"], 2),
            "path_len_m": round(float(sum(
                np.linalg.norm(camera_to_world(b["pose"])[:3, 3]
                               - camera_to_world(a["pose"])[:3, 3])
                for a, b in zip(rows, rows[1:]))), 3),
            "convention": "COLMAP: world_from_camera converted to world-to-camera, "
                          "+Z forward +Y down; world is ARKit's, +Y up, metric",
        }
        return report


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("session")
    ap.add_argument("out")
    ap.add_argument("--downscale", type=int, default=1,
                    help="integer image downscale; intrinsics scale with it")
    ap.add_argument("--conf-min", type=int, default=2,
                    help="lowest ARKit depth confidence used for the init cloud (0/1/2)")
    ap.add_argument("--voxel", type=float, default=0.02,
                    help="init-cloud fusion voxel in metres")
    ap.add_argument("--max-points", type=int, default=1_000_000)
    ap.add_argument("--min-baseline", type=float, default=0.0,
                    help="drop frames closer than this to the last kept one, in metres")
    ap.add_argument("--sharp-ratio", type=float, default=0.0,
                    help="drop frames below this fraction of the session's median sharpness")
    ap.add_argument("--pix-stride", type=int, default=1,
                    help="subsample depth pixels when building the init cloud")
    ap.add_argument("--images", choices=("symlink", "copy", "resize"), default="symlink")
    ap.add_argument("--no-depth", action="store_true",
                    help="skip the per-image depth and its confidence mask")
    ap.add_argument("--depth-format", choices=("png16", "npy"), default="png16",
                    help="png16 stores millimetres compactly; npy stores float metres, "
                         "which is what DN-Splatter reads (about 11 MB per full-size frame)")
    ap.add_argument("--allow-stale-depth", action="store_true",
                    help="accept nearest-in-time depth on pre-unified-gate sessions")
    ap.add_argument("--segment", type=int, default=None)
    ap.add_argument("--holdout-every", type=int, default=8,
                    help="reserve every n-th frame for evaluation; 0 or 1 keeps all")
    ap.add_argument("--guard-m", type=float, default=0.0,
                    help="drop training frames within this distance and 10 degrees "
                         "of a held-out one, so the evaluation is not scored on "
                         "views that have a near-duplicate in training")
    a = ap.parse_args(argv)

    reports = export(
        a.session, a.out, downscale=a.downscale, conf_min=a.conf_min, voxel=a.voxel,
        max_points=a.max_points, min_baseline_m=a.min_baseline,
        sharp_ratio=a.sharp_ratio, pix_stride=a.pix_stride,
        image_mode=("resize" if a.downscale > 1 else a.images),
        with_depth=not a.no_depth, require_exact_depth=not a.allow_stale_depth,
        segment=a.segment, holdout_every=a.holdout_every, guard_m=a.guard_m,
        depth_format=a.depth_format)
    for report in reports:
        print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


def export_merged(specs: list[tuple[str, np.ndarray]], out_dir: str, *,
                  holdout_every: int = 8, min_baseline_m: float = 0.0,
                  sharp_ratio: float = 0.0, require_exact_depth: bool = True,
                  guard_m: float = 0.0, **dataset) -> dict[str, Any]:
    """One training set out of several walks, in the first one's frame.

    The transforms come from `align_set.py`. Each session's poses are rewritten
    into the reference frame before anything else happens, so the rest of the
    pipeline never learns that more than one recording was involved.

    **The reference session's rows come first, and only they are held out.**
    That is what makes the merged arm comparable to a single-session arm: both
    are scored on the same photographs, and the question "did adding a second
    walk help" has a fixed thing to be measured on. Holding out every eighth row
    of the concatenated list would sample a different set of frames in each arm
    and answer nothing.
    """
    if Image is None:
        raise SystemExit("Pillow is required: pip install pillow")

    merged: list[dict[str, Any]] = []
    per_session = []
    for order, (session_dir, transform) in enumerate(specs):
        session = Session(session_dir)
        rows = session.posed_images()
        for row in rows:
            row["_session"] = session
        # An interruption re-origins the world, and a transform solved against
        # the whole session does not describe both halves. Take the longest run.
        segments = segment_frames(session, rows)
        rows = max(segments, key=len)
        rows, dropped = select_frames(session, rows, min_baseline_m=min_baseline_m,
                                      sharp_ratio=sharp_ratio,
                                      require_exact_depth=require_exact_depth)
        if order > 0 or transform is not None:
            for row in rows:
                row["pose"] = rebase_pose(row["pose"], transform)
        merged.extend(rows)
        per_session.append({"session": session.id, "images": len(rows),
                            "dropped": dropped,
                            "segments": len(segments)})

    reference_count = per_session[0]["images"]
    report = write_dataset(out_dir, merged, guard_m=guard_m,
                           holdout_indices=holdout_split(reference_count, holdout_every),
                           **dataset)
    report.update({
        "merged_from": per_session,
        "reference": per_session[0]["session"],
        "reference_images": reference_count,
        "holdout_note": "held out from the reference session only, so a "
                        "single-session arm scores the same frames",
    })
    with open(os.path.join(out_dir, "export.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    return report
