#!/usr/bin/env python3
"""Self-tests for the 3DGS export, mostly about one thing: the basis change.

This repository has shipped a transposed rotation once and spent a day finding
it. A conversion that is wrong by a transpose or a sign still produces a
plausible-looking file — poses in the right neighbourhood, a point cloud with
the right extent — and the failure only appears as a reconstruction that will
not converge. So the tests here do not check that the export runs. They put a
known geometry in, and demand the known answer back.

    python3 tools/test_export_3dgs.py
"""
from __future__ import annotations

import math
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import export_3dgs as ex  # noqa: E402


FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILURES.append(name)


def close(a, b, tol=1e-6) -> bool:
    return bool(np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), atol=tol))


def pose(tx=0.0, ty=0.0, tz=0.0, quat=(0.0, 0.0, 0.0, 1.0),
         fx=1360.0, fy=1360.0, cx=960.0, cy=720.0) -> dict:
    qx, qy, qz, qw = quat
    return {"tx": tx, "ty": ty, "tz": tz, "qx": qx, "qy": qy, "qz": qz, "qw": qw,
            "fx": fx, "fy": fy, "cx": cx, "cy": cy, "tracking": "normal"}


def project(p: dict, world_point: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Pixel of a world point, going only through what the exporter writes."""
    (qw, qx, qy, qz), t = ex.world_to_camera(p)
    R = ex.quat_to_matrix(qx, qy, qz, qw)
    cam = R @ np.asarray(world_point, dtype=float) + t
    fx, fy = p["fx"] * scale, p["fy"] * scale
    cx, cy = p["cx"] * scale, p["cy"] * scale
    return np.array([fx * cam[0] / cam[2] + cx, fy * cam[1] / cam[2] + cy, cam[2]])


# --- quaternion algebra ------------------------------------------------------

def test_quaternion_roundtrip() -> None:
    print("quaternion round trip")
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(200):
        v = rng.normal(size=4)
        v /= np.linalg.norm(v)
        R = ex.quat_to_matrix(v[0], v[1], v[2], v[3])
        qw, qx, qy, qz = ex.matrix_to_quat(R)
        back = ex.quat_to_matrix(qx, qy, qz, qw)
        worst = max(worst, float(np.abs(R - back).max()))
    check("matrix -> quat -> matrix", worst < 1e-9, f"max error {worst:.2e}")

    # Every branch of the trace decomposition, not just the generic one.
    for axis in range(3):
        angle = math.pi * 0.999
        v = np.zeros(3)
        v[axis] = math.sin(angle / 2)
        R = ex.quat_to_matrix(v[0], v[1], v[2], math.cos(angle / 2))
        qw, qx, qy, qz = ex.matrix_to_quat(R)
        check(f"near-180 rotation about axis {axis}",
              close(R, ex.quat_to_matrix(qx, qy, qz, qw), 1e-9))


def test_pose_inverse() -> None:
    print("world_to_camera inverts camera_to_world")
    p = pose(tx=1.5, ty=-0.3, tz=2.0, quat=(0.1, 0.5, -0.2, 0.8))
    c2w = ex.camera_to_world(p)
    (qw, qx, qy, qz), t = ex.world_to_camera(p)
    w2c = np.eye(4)
    w2c[:3, :3] = ex.quat_to_matrix(qx, qy, qz, qw)
    w2c[:3, 3] = t
    check("w2c @ c2w == I", close(w2c @ c2w, np.eye(4)))
    check("camera centre preserved", close(c2w[:3, 3], (1.5, -0.3, 2.0)))
    check("rotation is proper", abs(np.linalg.det(c2w[:3, :3]) - 1.0) < 1e-9,
          f"det {np.linalg.det(c2w[:3, :3]):.6f}")


# --- the basis change, stated as pictures ------------------------------------

def test_axis_semantics() -> None:
    print("ARKit -> COLMAP axis semantics")
    p = pose(tx=0.0, ty=0.0, tz=0.0)

    # ARKit's camera looks down -Z. A point 2 m along -Z is dead ahead and must
    # land on the principal point at positive depth.
    uvz = project(p, np.array([0.0, 0.0, -2.0]))
    check("straight ahead -> principal point", close(uvz[:2], (960.0, 720.0), 1e-6))
    check("straight ahead -> positive depth", uvz[2] > 0, f"z={uvz[2]:.3f}")

    # Behind the camera must come out negative, or a whole hemisphere of the
    # world silently projects into the image.
    check("behind camera -> negative depth", project(p, np.array([0.0, 0.0, 2.0]))[2] < 0)

    # COLMAP's image y grows downward, so something physically above the camera
    # belongs in an earlier row.
    up = project(p, np.array([0.0, 1.0, -2.0]))
    check("world up -> smaller v", up[1] < 720.0, f"v={up[1]:.1f}")
    right = project(p, np.array([1.0, 0.0, -2.0]))
    check("world right -> larger u", right[0] > 960.0, f"u={right[0]:.1f}")

    # Scale is metric and untouched: double the distance, halve the offset.
    near = project(p, np.array([1.0, 0.0, -2.0]))[0] - 960.0
    far = project(p, np.array([1.0, 0.0, -4.0]))[0] - 960.0
    check("perspective is metric", abs(near - 2.0 * far) < 1e-6,
          f"{near:.3f} vs {far:.3f}")


def test_rotated_camera() -> None:
    print("a rotated camera still points where it should")
    # Yaw 90 degrees about ARKit's +Y (up): the camera now looks down world -X.
    a = math.pi / 4
    p = pose(quat=(0.0, math.sin(a), 0.0, math.cos(a)))
    ahead = project(p, np.array([-2.0, 0.0, 0.0]))
    check("yawed camera sees along world -X", close(ahead[:2], (960.0, 720.0), 1e-4))
    check("yawed camera depth positive", abs(ahead[2] - 2.0) < 1e-6, f"z={ahead[2]:.4f}")
    behind = project(p, np.array([2.0, 0.0, 0.0]))
    check("and not along +X", behind[2] < 0)


# --- back-projection ---------------------------------------------------------

class FakeSession:
    """Just enough of `Session` for `depth_points`."""

    def __init__(self, depth: np.ndarray, conf: np.ndarray | None):
        self._depth, self._conf = depth, conf

    def depth_frame(self, entry):
        return self._depth

    def confidence_frame(self, entry):
        return self._conf


def synthetic_row(p: dict, depth: np.ndarray, width=1920, height=1440) -> dict:
    return {"pose": p, "depth": {}, "width": width, "height": height,
            "path": "", "t": 0.0}


def test_backprojection_is_the_inverse_of_projection() -> None:
    print("back-projection inverts projection")
    h, w = 192, 256
    p = pose(tx=0.7, ty=-0.2, tz=1.3, quat=(0.05, 0.3, -0.1, 0.947))
    depth = np.full((h, w), 2.5, dtype=np.float32)
    session = FakeSession(depth, np.full((h, w), 2, dtype=np.uint8))
    pts, nrm, uv = ex.depth_points(session, synthetic_row(p, depth),
                                   conf_min=2, near=0.1, far=5.0, pix_stride=1)
    check("every pixel back-projected", len(pts) == h * w, f"{len(pts)} of {h * w}")

    # Round trip each point through the exported COLMAP pose and demand the
    # pixel it came from, at the depth it was written with.
    scale = w / 1920
    errs = []
    for idx in (0, len(pts) // 3, len(pts) // 2, len(pts) - 1):
        got = project(p, pts[idx], scale)
        errs.append(abs(got[0] - (uv[idx, 0] + 0.0)))
        errs.append(abs(got[1] - (uv[idx, 1] + 0.0)))
        errs.append(abs(got[2] - 2.5))
    check("pixel and depth recovered", max(errs) < 1e-3, f"max error {max(errs):.2e}")


def test_two_cameras_see_one_plane() -> None:
    """The test that catches a sign error the single-camera test cannot.

    One camera can be self-consistent under a flipped translation, because the
    same flip is applied when the point goes out and when it comes back. Two
    cameras looking at the same wall from different places cannot: a wrong sign
    puts the second camera's copy of the wall twice the baseline away from the
    first, and the walls stop coinciding.
    """
    print("two cameras, one wall")
    h, w = 96, 128
    baseline = 0.6

    def wall_depth(cam_x: float) -> np.ndarray:
        """Depth of the plane world-Z = -3 seen from (cam_x, 0, 0), no rotation."""
        p = pose(tx=cam_x)
        sx = w / 1920
        fx, cx = 1360.0 * sx, 960.0 * sx
        fy, cy = 1360.0 * sx, 720.0 * sx
        vv, uu = np.mgrid[0:h, 0:w]
        # The plane is at distance 3 along the camera's viewing direction, and
        # the camera looks down world -Z, so depth is constant across it.
        del p, fx, cx, fy, cy, uu, vv
        return np.full((h, w), 3.0, dtype=np.float32)

    clouds = []
    for cam_x in (0.0, baseline):
        p = pose(tx=cam_x)
        depth = wall_depth(cam_x)
        session = FakeSession(depth, np.full((h, w), 2, dtype=np.uint8))
        pts, _, _ = ex.depth_points(session, synthetic_row(p, depth, 1920, 1440),
                                    conf_min=2, near=0.1, far=5.0, pix_stride=1)
        clouds.append(pts)

    z0, z1 = clouds[0][:, 2], clouds[1][:, 2]
    check("both walls at world z = -3",
          abs(z0.mean() + 3.0) < 1e-5 and abs(z1.mean() + 3.0) < 1e-5,
          f"{z0.mean():.5f} and {z1.mean():.5f}")

    # The second camera moved +x by the baseline, so its wall patch must be
    # offset by exactly that — not zero (translation dropped), not double
    # (translation applied with the wrong sign).
    dx = float(clouds[1][:, 0].mean() - clouds[0][:, 0].mean())
    check("baseline recovered exactly once", abs(dx - baseline) < 1e-5,
          f"got {dx:.5f}, wanted {baseline}")

    ax = float(np.abs(clouds[0][:, 1]).max())
    check("no vertical drift on an unrotated camera", ax < 3.0, f"spread {ax:.3f}")


def test_confidence_and_range_gates() -> None:
    print("depth gates")
    h, w = 16, 16
    depth = np.full((h, w), 2.0, dtype=np.float32)
    depth[0, :] = 0.05          # nearer than the sensor returns
    depth[1, :] = 9.0           # past the usable range
    depth[2, :] = np.nan
    conf = np.full((h, w), 2, dtype=np.uint8)
    conf[3, :] = 1
    conf[4, :] = 0
    session = FakeSession(depth, conf)
    pts, _, _ = ex.depth_points(session, synthetic_row(pose(), depth),
                                conf_min=2, near=ex.DEPTH_NEAR_M, far=ex.DEPTH_FAR_M,
                                pix_stride=1)
    check("near, far, nan and low confidence all dropped",
          len(pts) == (h - 5) * w, f"{len(pts)} of {(h - 5) * w}")

    pts_any, _, _ = ex.depth_points(session, synthetic_row(pose(), depth),
                                    conf_min=0, near=ex.DEPTH_NEAR_M,
                                    far=ex.DEPTH_FAR_M, pix_stride=1)
    check("conf_min=0 keeps the low-confidence rows",
          len(pts_any) == (h - 3) * w, f"{len(pts_any)} of {(h - 3) * w}")


def test_voxel_average() -> None:
    print("voxel fusion averages rather than picks")
    pts = np.array([[0.001, 0.0, 0.0], [0.009, 0.0, 0.0], [0.5, 0.0, 0.0]])
    nrm = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    col = np.array([[0, 0, 0], [100, 100, 100], [7, 8, 9]], dtype=np.uint8)
    p, n, c, counts = ex.voxel_average(pts, nrm, col, 0.05)
    check("two voxels", len(p) == 2, f"{len(p)}")
    order = np.argsort(p[:, 0])
    p, n, c, counts = p[order], n[order], c[order], counts[order]
    check("position averaged", abs(p[0, 0] - 0.005) < 1e-9, f"{p[0, 0]}")
    check("colour averaged", int(c[0, 0]) == 50, f"{c[0, 0]}")
    check("count reported", list(counts) == [2, 1], f"{list(counts)}")
    check("normals stay unit", close(np.linalg.norm(n, axis=1), [1.0, 1.0]))
    empty = ex.voxel_average(np.zeros((0, 3)), np.zeros((0, 3)),
                             np.zeros((0, 3), dtype=np.uint8), 0.05)
    check("empty input survives", len(empty[0]) == 0)


def test_segment_split() -> None:
    print("world-origin resets split the export")

    class Ev:
        def __init__(self, events):
            self._events = events

        def events(self):
            return self._events

    rows = [{"t": float(i)} for i in range(10)]
    check("no events -> one segment",
          len(ex.segment_frames(Ev([]), rows)) == 1)
    segs = ex.segment_frames(Ev([{"kind": "ar.interruptionEnded", "t": 4.5}]), rows)
    check("one reset -> two segments", len(segs) == 2, f"{len(segs)}")
    check("split at the right place",
          [r["t"] for r in segs[0]] == [0, 1, 2, 3, 4], f"{[r['t'] for r in segs[0]]}")
    segs = ex.segment_frames(
        Ev([{"kind": "ar.interruptionEnded", "t": 2.5},
            {"kind": "ar.interruptionEnded", "t": 7.5},
            {"kind": "thermal", "t": 5.0}]), rows)
    check("two resets -> three segments, other events ignored",
          len(segs) == 3, f"{len(segs)}")


def test_colmap_text_reads_back() -> None:
    print("the written COLMAP model parses back to the same pose")
    p = pose(tx=0.4, ty=1.1, tz=-2.2, quat=(0.2, -0.4, 0.1, 0.887))
    rows = [{"pose": p}, {"pose": pose(tx=1.0)}]
    with tempfile.TemporaryDirectory() as tmp:
        ex.write_colmap(tmp, rows, ["000000.jpg", "000001.jpg"],
                        [(1920, 1440), (1920, 1440)], [1.0, 1.0],
                        np.array([[0.0, 0.0, 1.0]]), np.array([[10, 20, 30]], dtype=np.uint8))
        lines = [ln for ln in open(os.path.join(tmp, "sparse", "0", "images.txt"))
                 if ln.strip() and not ln.startswith("#")]
        parts = lines[0].split()
        qw, qx, qy, qz = (float(v) for v in parts[1:5])
        t = np.array([float(v) for v in parts[5:8]])
        R = ex.quat_to_matrix(qx, qy, qz, qw)
        centre = -R.T @ t
        check("camera centre survives the file",
              close(centre, (0.4, 1.1, -2.2), 1e-6), f"{centre}")
        check("name and camera id written",
              parts[8] == "1" and parts[9] == "000000.jpg", f"{parts[8:10]}")

        cam = [ln for ln in open(os.path.join(tmp, "sparse", "0", "cameras.txt"))
               if ln.strip() and not ln.startswith("#")]
        check("one camera per image", len(cam) == 2, f"{len(cam)}")
        check("PINHOLE with four parameters",
              cam[0].split()[1] == "PINHOLE" and len(cam[0].split()) == 8,
              cam[0].strip())

        pts = [ln for ln in open(os.path.join(tmp, "sparse", "0", "points3D.txt"))
               if ln.strip() and not ln.startswith("#")]
        check("point written with colour", pts[0].split()[4:7] == ["10", "20", "30"],
              pts[0].strip())


def test_intrinsics_scale_with_downscale() -> None:
    print("downscaling moves the intrinsics with the pixels")
    p = pose()
    world = np.array([0.6, 0.0, -2.0])
    full = project(p, world, 1.0)
    half = project(p, world, 0.5)
    check("pixel coordinates halve", close(half[:2], full[:2] * 0.5, 1e-6),
          f"{half[:2]} vs {full[:2] * 0.5}")
    check("depth is unchanged by downscaling", abs(half[2] - full[2]) < 1e-9)


def test_holdout_matches_the_trainers_convention() -> None:
    print("evaluation frames are the ones a trainer will also hold out")
    held = ex.holdout_split(80, 8)
    check("one in eight", len(held) == 10, f"{len(held)}")
    check("evenly spaced", set(np.diff(held)) == {8}, f"{np.diff(held)}")
    check("disabled by 0", ex.holdout_split(80, 0) == [])
    check("disabled by 1", ex.holdout_split(80, 1) == [])
    check("short sequence stays in range", max(ex.holdout_split(5, 8), default=-1) < 5)

    # gsplat and the reference 3DGS loader both select test views as
    # `index % test_every == 0`. Ours has to be the same set, or the frames kept
    # out of the point cloud are not the frames being scored.
    for every in (2, 4, 8, 16):
        ours = ex.holdout_split(97, every)
        theirs = [i for i in range(97) if i % every == 0]
        check(f"identical to `i % {every} == 0`", ours == theirs,
              f"{ours[:5]} vs {theirs[:5]}")


def test_depth_sidecars() -> None:
    """The confidence mask polarity, which is inverted and silent when wrong.

    DN-Splatter computes `1 - mask/255` and keeps what is positive, so writing
    the intuitive polarity trains on exactly the pixels the LiDAR said not to
    trust. Nothing renders the mask, so the mistake would never be seen.
    """
    print("depth and confidence sidecars")
    from PIL import Image as PILImage

    depth = np.full((6, 8), 2.5, np.float32)
    depth[0, :] = 7.0
    conf = np.full((6, 8), 2, np.uint8)
    conf[1, :] = 0

    with tempfile.TemporaryDirectory() as tmp:
        mask_path = os.path.join(tmp, "m.jpg")
        ex.write_depth_mask(mask_path, conf, (8, 6), 2)
        with PILImage.open(mask_path) as im:
            mask = np.asarray(im.convert("L"))
        check("high confidence writes 0, meaning valid", mask[2, 0] < 40, f"{mask[2, 0]}")
        check("low confidence writes 255, meaning invalid", mask[1, 0] > 215,
              f"{mask[1, 0]}")
        check("DN's own decoding keeps the confident row",
              (1 - mask[2, 0] / 255.0) > 0)
        check("and drops the unconfident one", (1 - mask[1, 0] / 255.0) <= 0.2,
              f"{1 - mask[1, 0] / 255.0:.3f}")

        npy_path = os.path.join(tmp, "d.npy")
        ex.write_depth(npy_path, depth, conf, (8, 6), "npy", 2)
        got = np.load(npy_path)
        check("npy is float metres", got.dtype == np.float32, f"{got.dtype}")
        check("npy keeps the value in metres", abs(float(got[2, 0]) - 2.5) < 1e-6,
              f"{got[2, 0]}")
        check("npy zeroes the unconfident row", float(got[1, 0]) == 0.0,
              f"{got[1, 0]}")
        check("npy is at the requested size", got.shape == (6, 8), f"{got.shape}")

        png_path = os.path.join(tmp, "d.png")
        ex.write_depth(png_path, depth, conf, (8, 6), "png16", 2)
        with PILImage.open(png_path) as im:
            mm = np.asarray(im)
        check("png16 is millimetres", mm.dtype == np.uint16 and mm[2, 0] == 2500,
              f"{mm.dtype} {mm[2, 0]}")
        check("png16 zeroes the unconfident row", mm[1, 0] == 0, f"{mm[1, 0]}")
        # The two formats have to describe the same depth, or the scale factor
        # written into transforms.json is right for one of them and wrong for
        # the other.
        check("the two formats agree once scaled",
              abs(float(mm[2, 0]) * 0.001 - float(got[2, 0]) * 1.0) < 1e-3)


def test_nearest_resample_does_not_invent_depth() -> None:
    print("depth resampling stays nearest-neighbour")
    frame = np.zeros((4, 4), np.float32)
    frame[:, :2] = 1.0          # a hard edge: 1 m in front of 5 m
    frame[:, 2:] = 5.0
    up = ex.resample_nearest(frame, (16, 16))
    check("only the two original values survive",
          set(np.unique(up).tolist()) == {1.0, 5.0}, f"{np.unique(up)}")
    check("no value lands between the surfaces",
          not ((up > 1.001) & (up < 4.999)).any())
    down = ex.resample_nearest(frame, (2, 2))
    check("downsampling too", set(np.unique(down).tolist()) == {1.0, 5.0},
          f"{np.unique(down)}")


def test_ply_header() -> None:
    print("PLY is readable")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "p.ply")
        ex.write_ply(path, np.array([[1.0, 2.0, 3.0]]), np.array([[0.0, 0.0, 1.0]]),
                     np.array([[255, 128, 0]], dtype=np.uint8))
        with open(path, "rb") as fh:
            blob = fh.read()
        head, body = blob.split(b"end_header\n", 1)
        check("vertex count in header", b"element vertex 1" in head)
        check("body is 27 bytes", len(body) == 27, f"{len(body)}")
        xyz = np.frombuffer(body[:12], dtype="<f4")
        check("coordinates survive", close(xyz, (1.0, 2.0, 3.0), 1e-6), f"{xyz}")
        check("colour survives", tuple(body[24:27]) == (255, 128, 0), f"{tuple(body[24:27])}")


def main() -> int:
    test_quaternion_roundtrip()
    test_pose_inverse()
    test_axis_semantics()
    test_rotated_camera()
    test_backprojection_is_the_inverse_of_projection()
    test_two_cameras_see_one_plane()
    test_confidence_and_range_gates()
    test_voxel_average()
    test_segment_split()
    test_colmap_text_reads_back()
    test_intrinsics_scale_with_downscale()
    test_holdout_matches_the_trainers_convention()
    test_depth_sidecars()
    test_nearest_resample_does_not_invent_depth()
    test_ply_header()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} failed: {', '.join(FAILURES)}")
        return 1
    print("all passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
