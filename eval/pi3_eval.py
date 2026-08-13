#!/usr/bin/env python3
"""Score Pi3X's windowed relative pose against ARKit, beside ICP's.

The question is not per-pair accuracy. At 30 Hz ICP already gets 1 cm on 2 cm of
motion even where the geometry is degenerate; what it cannot do is accumulate.
So this measures a *window* — the transform from the first image frame of a run
of N to the last — which is where drift lives and where a model that sees all N
frames at once should have an advantage if it has one.

Scale is the other half. Pi3X is conditioned on the session's own metric LiDAR
depth, so its translations should already be in metres. That is tested rather
than assumed: the scale ratio against ARKit is reported and never fitted, since
fitting it to the reference would be assuming the answer.

Conventions are inherited, not re-derived. `ARKIT_TO_DEPTH` is diag(1, -1, -1),
which takes ARKit camera axes to +X right, +Y down, +Z forward — exactly the
OpenCV Right-Down-Forward frame Pi3X documents for `poses` and `camera_poses`.
The reference trajectory comes from `--dump-poses`, already in that frame.
"""
import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

# Portable: the session reader comes from tools/ — the one direction the
# dependency is allowed to run — and the Pi3 checkout is wherever PI3_ROOT says.
# Nothing here assumes a particular machine.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
sys.path.insert(0, os.environ.get("PI3_ROOT", str(Path(__file__).resolve().parent / "models" / "pi3")))

from read_session import Session  # noqa: E402
from pi3.models.pi3x import Pi3X  # noqa: E402
from pi3.utils.basic import load_multimodal_data  # noqa: E402

ARKIT_TO_DEPTH = np.diag([1.0, -1.0, -1.0])


def quat_to_matrix(x, y, z, w):
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n == 0:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def relative(a, b):
    """Transform taking camera `b` coordinates into camera `a` coordinates."""
    return np.linalg.inv(a) @ b


def rotation_angle(R):
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2))))


def load_window(session, image_rows, poses, frames, tmpdir):
    """Copy one window's images out and build the matching condition arrays."""
    images = tmpdir / "images"
    images.mkdir(parents=True, exist_ok=True)
    depths, intrinsics, reference = [], [], []
    for order, frame in enumerate(frames):
        row = image_rows[frame]
        shutil.copyfile(session.frame_path(row), images / f"{order:04d}.jpg")

        entry = depth_by_frame[frame]
        depth = np.asarray(session.depth_frame(entry), dtype=np.float32)
        depth = np.where(np.isfinite(depth), depth, 0.0)
        # The loader wants depth at the colour resolution; the depth map is a
        # uniformly resized view of the same frustum, so nearest-neighbour back
        # up to 1920x1440 is exactly inverting that resize.
        import cv2
        depths.append(cv2.resize(depth, (row["width"], row["height"]),
                                 interpolation=cv2.INTER_NEAREST))

        p = poses[frame]
        K = np.array([[p["fx"], 0.0, p["cx"]],
                      [0.0, p["fy"], p["cy"]],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
        intrinsics.append(K)

        R = quat_to_matrix(p["qx"], p["qy"], p["qz"], p["qw"]) @ ARKIT_TO_DEPTH
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [p["tx"], p["ty"], p["tz"]]
        reference.append(T)

    return (str(images), np.stack(depths), np.stack(intrinsics),
            np.stack(reference))


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--dump", required=True, help="npz from --dump-poses")
    ap.add_argument("--window", type=int, default=24,
                    help="image frames per window (5 Hz, so 24 is ~4.8 s)")
    ap.add_argument("--stride", type=int, default=24)
    ap.add_argument("--windows", type=int, default=6)
    ap.add_argument("--condition", default="depth+intrinsics",
                    choices=("none", "intrinsics", "depth+intrinsics",
                             "depth+intrinsics+pose"))
    ap.add_argument("--out")
    args = ap.parse_args(argv)

    session = Session(args.session)
    global depth_by_frame
    rows = [json.loads(l) for l in
            Path(args.session, "frames.jsonl").read_text().splitlines() if l.strip()]
    image_rows = {r["frame"]: r for r in rows}
    depth_rows = [json.loads(l) for l in
                  Path(args.session, "depth.jsonl").read_text().splitlines() if l.strip()]
    depth_by_frame = {d["frame"]: d for d in depth_rows}
    pose_rows = [json.loads(l) for l in
                 Path(args.session, "pose.jsonl").read_text().splitlines() if l.strip()]
    poses = {p["frame"]: p for p in pose_rows}

    dump = np.load(args.dump)
    est_by_frame = {int(f): T for f, T in zip(dump["frame"], dump["estimate"])}
    ref_by_frame = {int(f): T for f, T in zip(dump["frame"], dump["reference"])}

    usable = [f for f in sorted(image_rows)
              if f in depth_by_frame and f in poses and f in ref_by_frame]
    print(f"{Path(args.session).name}: {len(usable)} image frames with depth, "
          f"pose and a scored reference")

    model = Pi3X.from_pretrained("yyfz233/Pi3X").eval().to("cuda")
    if args.condition == "none":
        model.disable_multimodal()
    dtype = (torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8
             else torch.float16)

    results = []
    for w in range(args.windows):
        start = w * args.stride
        frames = usable[start:start + args.window]
        if len(frames) < args.window:
            break
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            path, depths, Ks, reference = load_window(
                session, image_rows, poses, frames, tmpdir)
            conditions = {
                "intrinsics": Ks if "intrinsics" in args.condition else None,
                "depths": depths,
                "poses": reference if "pose" in args.condition else None,
            }
            imgs, conditions = load_multimodal_data(
                path, conditions, interval=1, verbose=(w == 0), device="cuda")
            measured_depth = conditions["depths"]
            if "depth" not in args.condition:
                conditions["depths"] = None
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=dtype):
                    res = model(imgs=imgs, **conditions)
            pred = res["camera_poses"][0].float().cpu().numpy()
            # Scale from the session's own depth, never from the reference.
            # Pi3X builds `points` by unprojecting `local_points` through
            # `camera_poses`, so a scale fitted on the point map is the scale
            # the translations carry — that is the assumption under test here,
            # and it is what makes this a metric estimate rather than a fitted
            # one. Over-determined: thousands of pixels, one unknown.
            local = res["local_points"][0].float().cpu().numpy()
            conf = torch.sigmoid(res["conf"][0, ..., 0]).float().cpu().numpy()
            meas = measured_depth[0].float().cpu().numpy()
            pz = local[..., 2]
            good = (meas > 0.05) & np.isfinite(meas) & (conf > 0.5) & (pz > 0.05)
            s_depth = (float((pz[good] * meas[good]).sum()
                             / max((pz[good] * pz[good]).sum(), 1e-9))
                       if good.sum() > 1000 else float("nan"))
            n_good = int(good.sum())

        # First-to-last of the window, for all three trajectories.
        ark = relative(reference[0], reference[-1])
        pi3 = relative(pred[0], pred[-1])
        icp = relative(est_by_frame[frames[0]], est_by_frame[frames[-1]])

        travelled = float(np.linalg.norm(np.diff(
            np.stack([T[:3, 3] for T in reference]), axis=0), axis=1).sum())
        row = {
            "window": w, "first": frames[0], "last": frames[-1],
            "travelled_m": travelled,
            "arkit_m": float(np.linalg.norm(ark[:3, 3])),
            "pi3_m": float(np.linalg.norm(pi3[:3, 3])),
            "icp_m": float(np.linalg.norm(icp[:3, 3])),
            "pi3_err_m": float(np.linalg.norm(pi3[:3, 3] - ark[:3, 3])),
            "icp_err_m": float(np.linalg.norm(icp[:3, 3] - ark[:3, 3])),
            "pi3_rot_deg": rotation_angle(ark[:3, :3].T @ pi3[:3, :3]),
            "icp_rot_deg": rotation_angle(ark[:3, :3].T @ icp[:3, :3]),
        }
        # Never fitted, only reported: if depth conditioning delivers metric
        # scale this is 1, and if it does not the errors above are meaningless
        # and this number is the finding.
        row["pi3_scale"] = (row["pi3_m"] / row["arkit_m"]
                            if row["arkit_m"] > 1e-6 else float("nan"))
        # The oracle scale above needs ARKit; this one needs only the LiDAR.
        row["scale_from_depth"] = s_depth
        row["scale_pixels"] = n_good
        corrected = pi3[:3, 3] * s_depth
        row["pi3_fixed_err_m"] = float(np.linalg.norm(corrected - ark[:3, 3]))
        row["pi3_fixed_m"] = float(np.linalg.norm(corrected))
        results.append(row)
        print(f"  window {w}: {travelled:.2f} m walked   "
              f"ARKit {row['arkit_m']:.3f}  Pi3X {row['pi3_m']:.3f} "
              f"(scale {row['pi3_scale']:.3f})  ICP {row['icp_m']:.3f}   "
              f"err Pi3X {row['pi3_err_m'] * 100:.1f} cm  "
              f"ICP {row['icp_err_m'] * 100:.1f} cm   "
              f"depth-fit scale {s_depth:.3f} -> "
              f"{row['pi3_fixed_err_m'] * 100:.1f} cm", flush=True)

    if results:
        pe = np.array([r["pi3_err_m"] for r in results])
        ie = np.array([r["icp_err_m"] for r in results])
        sc = np.array([r["pi3_scale"] for r in results])
        print(f"\n  median window error   Pi3X {np.median(pe) * 100:.1f} cm   "
              f"ICP {np.median(ie) * 100:.1f} cm")
        print(f"  median scale ratio    {np.median(sc):.3f}  "
              f"(1.000 means depth conditioning gave metric translations)")
        print(f"  Pi3X beats ICP on {int((pe < ie).sum())}/{len(pe)} windows")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
