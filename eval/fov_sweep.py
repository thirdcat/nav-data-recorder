#!/usr/bin/env python3
"""Does the learned pose source care about field of view?

The pose source was validated on the wide camera, ~72 degrees horizontal. The
ultra-wide path would feed it a 96.3 degree rectified pinhole instead, and that
field-of-view change is the only untested difference between what was measured and
what would be deployed. A wider view cannot be synthesised from a narrower one, so
this measures the gradient on the side we can reach: crop the wide frames to
narrower pinholes and see whether accuracy moves at all, and in which direction.

A narrower pinhole from a pinhole is exactly a centre crop -- fx and fy are
unchanged in pixels, only the principal point shifts -- so no resampling is
involved and the geometry stays exact. Every arm still downsamples to the loader's
pixel budget, so no arm is upsampled and resolution is not confounded with field
of view.

Reuses pi3_chain's window loop, LiDAR scale fit, Umeyama join and scoring
unchanged; the only difference is the images and intrinsics handed in.
"""
import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tools"))
sys.path.insert(0, os.environ.get("PI3_ROOT", str(HERE / "vendor" / "Pi3")))
import pi3_eval as E  # noqa: E402
import pi3_chain as C  # noqa: E402

from pi3.models.pi3x import Pi3X  # noqa: E402
from pi3.utils.basic import load_multimodal_data  # noqa: E402


def crop_for_hfov(width, fx, hfov_deg):
    """Crop width that turns a pinhole of focal fx into one of this field of view."""
    want = 2.0 * fx * math.tan(math.radians(hfov_deg) / 2.0)
    return int(min(width, max(64, round(want))))


def load_window_cropped(session, image_rows, poses, frames, tmpdir, hfov_deg):
    """load_window, but every image and depth centre-cropped to `hfov_deg`.

    hfov_deg of None means leave the frame alone, which reproduces the baseline
    byte for byte through the same code path.
    """
    images = tmpdir / "images"
    images.mkdir(parents=True, exist_ok=True)
    depths, intrinsics, reference, applied = [], [], [], []

    # The crop *size* is fixed once per window, from the window's median focal
    # length. Autofocus moves fx by about 1.7% within a session, so sizing each
    # frame separately produces frames of different shapes that cannot be stacked
    # -- and would also vary the field of view frame to frame, which is the very
    # thing being held as the independent variable. The crop *position* still
    # follows each frame's own principal point, which is what keeps the result a
    # pinhole with that frame's own fx.
    crop_wh = None
    if hfov_deg is not None:
        fx_med = float(np.median([poses[f]["fx"] for f in frames]))
        w0, h0 = image_rows[frames[0]]["width"], image_rows[frames[0]]["height"]
        cw = crop_for_hfov(w0, fx_med, hfov_deg)
        crop_wh = (cw, int(round(cw * h0 / w0)))

    for order, frame in enumerate(frames):
        row = image_rows[frame]
        p = poses[frame]
        src = session.frame_path(row)

        entry = E.depth_by_frame[frame]
        depth = np.asarray(session.depth_frame(entry), dtype=np.float32)
        depth = np.where(np.isfinite(depth), depth, 0.0)
        depth = cv2.resize(depth, (row["width"], row["height"]),
                           interpolation=cv2.INTER_NEAREST)

        fx, fy, cx, cy = p["fx"], p["fy"], p["cx"], p["cy"]
        if hfov_deg is None:
            shutil.copyfile(src, images / f"{order:04d}.jpg")
        else:
            img = cv2.imread(str(src), cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            cw, ch = crop_wh
            # Crop about the principal point, not the image centre: that is what
            # keeps the result a pinhole with the same fx and a shifted cx.
            x0 = int(round(min(max(cx - cw / 2.0, 0), w - cw)))
            y0 = int(round(min(max(cy - ch / 2.0, 0), h - ch)))
            cv2.imwrite(str(images / f"{order:04d}.jpg"),
                        img[y0:y0 + ch, x0:x0 + cw],
                        [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            depth = depth[y0:y0 + ch, x0:x0 + cw]
            cx, cy = cx - x0, cy - y0
            applied.append(2.0 * math.degrees(math.atan(cw / (2.0 * fx))))

        depths.append(depth)
        intrinsics.append(np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                                   dtype=np.float64))
        R = E.quat_to_matrix(p["qx"], p["qy"], p["qz"], p["qw"]) @ E.ARKIT_TO_DEPTH
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [p["tx"], p["ty"], p["tz"]]
        reference.append(T)

    return (str(images), np.stack(depths), np.stack(intrinsics),
            np.stack(reference),
            float(np.median(applied)) if applied else None)


def run(session_path, dump, hfov_deg, window, overlap, condition, device="cuda"):
    root = Path(session_path)
    session = E.Session(session_path)
    rows = [json.loads(l) for l in (root / "frames.jsonl").read_text().splitlines() if l.strip()]
    image_rows = {r["frame"]: r for r in rows}
    E.depth_by_frame = {d["frame"]: d for d in
                        (json.loads(l) for l in (root / "depth.jsonl").read_text().splitlines() if l.strip())}
    poses = {p["frame"]: p for p in
             (json.loads(l) for l in (root / "pose.jsonl").read_text().splitlines() if l.strip())}

    d = np.load(dump)
    est_by_frame = {int(f): T for f, T in zip(d["frame"], d["estimate"])}
    ref_by_frame = {int(f): T for f, T in zip(d["frame"], d["reference"])}
    usable = [f for f in sorted(image_rows)
              if f in E.depth_by_frame and f in poses and f in ref_by_frame]

    model = Pi3X.from_pretrained("yyfz233/Pi3X").eval().to(device)
    import chunked_conv
    chunked_conv.apply(model)
    dtype = (torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8
             else torch.float16)

    win = min(window, len(usable))
    ov = min(overlap, max(win - 1, 1))
    step = max(win - ov, 1)

    # Window starts are enumerated up front, with a final window pinned to the end
    # of the session. The obvious `while start + win <= n` loop drops whatever tail
    # does not fill a whole window -- 60 of 260 frames on the longest session --
    # and it drops them silently, so the scored trajectory would simply be shorter
    # than the walk without anything in the output saying so.
    starts = list(range(0, max(len(usable) - win, 0) + 1, step))
    if starts[-1] + win < len(usable):
        starts.append(len(usable) - win)

    chained, joins, actual = {}, [], None
    for start in starts:
        frames = usable[start:start + win]
        with tempfile.TemporaryDirectory() as tmp:
            path, depths, Ks, _ref, got = load_window_cropped(
                session, image_rows, poses, frames, Path(tmp), hfov_deg)
            actual = got if got is not None else actual
            conditions = {"intrinsics": Ks if "intrinsics" in condition else None,
                          "depths": depths, "poses": None}
            imgs, conditions = load_multimodal_data(
                path, conditions, interval=1, verbose=False, device=device)
            measured = conditions["depths"]
            if "depth" not in condition:
                conditions["depths"] = None
            with torch.no_grad():
                with torch.amp.autocast(device, dtype=dtype):
                    res = model(imgs=imgs, **conditions)
            pred = res["camera_poses"][0].float().cpu().numpy()
            local = res["local_points"][0].float().cpu().numpy()
            conf = torch.sigmoid(res["conf"][0, ..., 0]).float().cpu().numpy()
            meas = measured[0].float().cpu().numpy()

        pz = local[..., 2]
        good = (meas > 0.05) & np.isfinite(meas) & (conf > 0.5) & (pz > 0.05)
        s = (float((pz[good] * meas[good]).sum()
                   / max((pz[good] * pz[good]).sum(), 1e-9))
             if good.sum() > 1000 else 1.0)
        scaled = pred.copy()
        scaled[:, :3, 3] *= s

        if not chained:
            for f, T in zip(frames, scaled):
                chained[f] = T
        else:
            shared = [f for f in frames if f in chained]
            if len(shared) < 3:
                break
            src_p = np.stack([scaled[frames.index(f)][:3, 3] for f in shared])
            dst_p = np.stack([chained[f][:3, 3] for f in shared])
            R, js, t = C.umeyama(src_p, dst_p)
            joins.append(js)
            for f, T in zip(frames, scaled):
                if f in chained:
                    continue
                W = np.eye(4)
                W[:3, :3] = R @ T[:3, :3]
                W[:3, 3] = js * (R @ T[:3, 3]) + t
                chained[f] = W

    if len(chained) != len(usable):
        raise RuntimeError(
            f"windowing covered {len(chained)} of {len(usable)} usable frames; "
            "a partial trajectory would score as if it were the whole walk")

    frames = sorted(chained)
    est = np.stack([chained[f][:3, 3] for f in frames])
    ref = np.stack([ref_by_frame[f][:3, 3] for f in frames])
    icp = np.stack([est_by_frame[f][:3, 3] for f in frames])
    travelled = float(np.linalg.norm(np.diff(ref, axis=0), axis=1).sum())

    def loop(p):
        return float(np.linalg.norm(p[-1] - p[0]))

    def ate(p):
        pc, rc = p - p.mean(0), ref - ref.mean(0)
        U, _, Vt = np.linalg.svd(pc.T @ rc)
        dd = np.sign(np.linalg.det(U @ Vt))
        R = U @ np.diag([1, 1, dd]) @ Vt
        return float(np.linalg.norm(pc @ R - rc, axis=1).mean())

    return {
        "session": root.name, "hfov_requested": hfov_deg, "hfov_actual": actual,
        "frames": len(frames), "windows": len(joins) + 1,
        "travelled_m": travelled,
        "loop_arkit_pct": loop(ref) / travelled * 100,
        "loop_pi3_pct": loop(est) / travelled * 100,
        "loop_icp_pct": loop(icp) / travelled * 100,
        "ate_pi3_cm": ate(est) * 100, "ate_icp_cm": ate(icp) * 100,
    }


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--dump", required=True)
    ap.add_argument("--hfov", type=float, nargs="+", default=[None],
                    help="horizontal fields of view to crop to; 0 means native")
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--overlap", type=int, default=24)
    ap.add_argument("--condition", default="intrinsics")
    ap.add_argument("--out")
    args = ap.parse_args(argv)

    results = []
    for h in args.hfov:
        hv = None if (h is None or h <= 0) else float(h)
        r = run(args.session, args.dump, hv, args.window, args.overlap, args.condition)
        results.append(r)
        tag = "native" if hv is None else f"{r['hfov_actual']:.1f}deg"
        print(f"{r['session'][-6:]}  {tag:>9}  win {r['windows']}  "
              f"loop ARKit {r['loop_arkit_pct']:.2f}%  Pi3X {r['loop_pi3_pct']:.2f}%  "
              f"ICP {r['loop_icp_pct']:.2f}%   ATE Pi3X {r['ate_pi3_cm']:.1f}cm  "
              f"ICP {r['ate_icp_cm']:.1f}cm", flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
