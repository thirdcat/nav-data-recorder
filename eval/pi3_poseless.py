#!/usr/bin/env python3
"""Pose a session that has no tracker: the ultra-wide walks.

`MultiCamRecorder` reaches the ultra-wide lens by leaving ARKit, so its sessions
carry images, depth and IMU but **no `pose.jsonl`**. Everything else in `eval/`
assumes ARKit's pose is there — for the intrinsics, and as the reference to
score against — so this supplies the first from `calib/` and does without the
second.

    python3 eval/pi3_poseless.py ~/nav_data/<id> --out pi3traj/<id>.npz

Two joins have to be made by hand and both are worth stating.

**Images to depth, by time.** In an ARKit session the image and its depth come
out of one `ARFrame` and share a frame number. Here they are two independent
outputs of a multi-cam session with their own counters, so the only honest key
is the presentation timestamp, which both take from the same clock. Each image
takes the nearest depth frame and the gap is recorded; a pairing wider than
`--max-dt` is dropped rather than used.

**Pixels to bearings, from the calibration.** `calib/` carries the ultra-wide's
factory intrinsics at 4032x3024, and video frames are smaller, so `fx, fy, cx,
cy` scale by the image size.

The frames are **not rectified**, and that is a measurement rather than an
omission. `docs/POSE.md` records three arms on the wide camera: correcting its
lens costs Pi3X +3.19 cm of ATE in four sessions of four, while resampling
through the identity costs nothing (-0.31 cm, better in three of four). The
cost is the correction itself — the model has already learned this distortion —
and the ultra-wide is three to five times *flatter* than the wide lens out to
three quarters of the corner radius. Correcting it would buy less and cost the
same. `--rectified` remains for frames that have been through
`tools/rectify_ultrawide.py` anyway, so the choice stays checkable.

There is no reference trajectory here and there cannot be. That is the point of
the mode, and it means the usual ATE-against-ARKit is unavailable: what this
writes has to be judged by the checks that need no reference.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "tools"))
from read_session import Session  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def load_calibration(path: Path) -> dict:
    data = json.loads(path.read_text())
    w, h = data["reference_dimensions"]
    k = data["intrinsics"]
    return {"fx": k["fx"], "fy": k["fy"], "cx": k["cx"], "cy": k["cy"],
            "width": float(w), "height": float(h)}


def intrinsics_for(calib: dict, width: int, height: int,
                   active_hfov_deg: float | None = None) -> np.ndarray:
    """Intrinsics for the frames as recorded.

    Two things make this less obvious than scaling the factory numbers, and the
    first `--dry-run` on a real capture caught both.

    **The focal length is one number, not two.** The pixels are square, so `fx`
    and `fy` are equal; scaling each axis by its own ratio produced 769.8 and
    577.4 on a 1920x1080 frame, which describes a camera that does not exist. A
    16:9 format is a *crop* of the 4:3 sensor, and a crop changes the extent,
    never the focal length.

    **The active format is not the calibrated one.** `calib/` is measured at
    4032x3024 and implies 102.5 degrees across; the format this recorder
    actually selected reports **106.2**. Different formats read out different
    parts of the sensor, so scaling the calibration to the frame size answers a
    question about a format that was not used. When the recorder wrote down what
    the device said about the format it chose, that is the number to trust —
    `active_hfov_deg` is it.

    The principal point stays at the frame centre. It is a few pixels off centre
    on the calibrated format (-5.7, -5.9 of 4032x3024) and there is no way to
    know how a different format's readout moves it, so pretending otherwise
    would be inventing precision.
    """
    if active_hfov_deg:
        f = (width / 2.0) / np.tan(np.radians(active_hfov_deg) / 2.0)
    else:
        f = calib["fx"] * (width / calib["width"])
    return np.array([[f, 0.0, width / 2.0],
                     [0.0, f, height / 2.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def active_hfov(manifest: dict) -> float | None:
    """The horizontal field of view the recorder wrote down for this session."""
    for note in manifest.get("notes", []):
        if note.startswith("ultra-wide fov "):
            try:
                return float(note.split()[-1])
            except ValueError:
                return None
    return None


def depth_in_ultrawide(session: Session, row: dict, K: np.ndarray,
                       wide_calib: dict, extrinsic: np.ndarray) -> np.ndarray:
    """Put the LiDAR's depth where the ultra-wide can use it, and nowhere else.

    The two do not see the same cone. The depth comes from the LiDAR device,
    which is the wide camera plus a scanner, so it is a 4:3 map of **72.6 x 57.7
    degrees**; the ultra-wide frame is 16:9 and **106.2 x 73.7**. Stretching one
    onto the other covers 100 % of the image with measurements that exist for
    41 % of it, and the error is systematic — the first run's per-window scales
    piled up between 1.07 and 1.18 rather than scattering about 1.

    So each depth pixel is unprojected in the wide camera's frame, moved by the
    extrinsic from `calib/` — 19.272 mm and 0.462 degrees, its direction checked
    against the probe's own stereo pair — and projected into the ultra-wide.
    Everything outside the LiDAR's cone stays zero, which is what the model's
    depth conditioning reads as "no measurement" rather than "far away".
    """
    z = np.asarray(session.depth_frame(row["depth"]), dtype=np.float32)
    z = np.where(np.isfinite(z), z, 0.0)
    dh, dw = z.shape

    # The depth map is a uniformly resized view of the wide camera's frame, so
    # its intrinsics are the wide camera's scaled to this grid.
    sx, sy = dw / wide_calib["width"], dh / wide_calib["height"]
    fx, fy = wide_calib["fx"] * sx, wide_calib["fy"] * sy
    cx, cy = wide_calib["cx"] * sx, wide_calib["cy"] * sy

    v, u = np.mgrid[0:dh, 0:dw]
    good = z > 0.05
    if not good.any():
        return np.zeros((row["height"], row["width"]), dtype=np.float32)
    zz = z[good]
    pts = np.stack([(u[good] - cx) * zz / fx, (v[good] - cy) * zz / fy, zz], axis=1)

    # `calib/` gives the ultra-wide's pose relative to the wide camera as
    # `R x + t`, verified against the probe's stereo pair. The depth is in the
    # wide camera's frame, so it needs the inverse to land in the ultra-wide's.
    R, t = extrinsic[:3, :3], extrinsic[:3, 3]
    moved = (pts - t) @ R

    ahead = moved[:, 2] > 0.05
    moved = moved[ahead]
    if not len(moved):
        return np.zeros((row["height"], row["width"]), dtype=np.float32)
    su = moved[:, 0] / moved[:, 2] * K[0, 0] + K[0, 2]
    sv = moved[:, 1] / moved[:, 2] * K[1, 1] + K[1, 2]
    iu, iv = np.round(su).astype(np.int64), np.round(sv).astype(np.int64)
    inside = ((iu >= 0) & (iu < row["width"]) & (iv >= 0) & (iv < row["height"]))

    out = np.zeros((row["height"], row["width"]), dtype=np.float32)
    # Nearest surface wins where several depth pixels land on one image pixel,
    # so a background point cannot overwrite the thing in front of it.
    order = np.argsort(-moved[inside, 2])
    out[iv[inside][order], iu[inside][order]] = moved[inside, 2][order]

    # One depth pixel covers about six image pixels across, so the projection
    # leaves a lattice of gaps that are not missing data. Close them with a
    # small dilation, which never invents a value where no depth pixel landed
    # nearby.
    filled = cv2.dilate(out, np.ones((7, 7), np.uint8))
    return np.where(out > 0, out, np.where(filled > 0, filled, 0.0)).astype(np.float32)


def pair_by_time(session: Session, max_dt: float) -> list[dict]:
    """One row per image with the nearest depth frame, and how far off it was."""
    depth_rows = sorted(session.depth_index(), key=lambda d: d["t"])
    if not depth_rows:
        raise SystemExit("this session has no depth index")
    times = np.array([d["t"] for d in depth_rows])
    rows, dropped = [], 0
    for entry in session.stream("frames"):
        k = int(np.searchsorted(times, entry["t"]))
        best = min([i for i in (k - 1, k) if 0 <= i < len(depth_rows)],
                   key=lambda i: abs(times[i] - entry["t"]), default=None)
        if best is None:
            dropped += 1
            continue
        dt = float(entry["t"] - times[best])
        if abs(dt) > max_dt:
            dropped += 1
            continue
        rows.append({**entry, "depth": depth_rows[best], "depth_dt": dt})
    if dropped:
        print(f"  {dropped} images had no depth within {max_dt * 1000:.0f} ms and were dropped")
    return rows


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("session")
    ap.add_argument("--out", required=True, help="npz for the chained trajectory")
    ap.add_argument("--calibration",
                    default=str(REPO / "calib" / "iphone17-1_ultrawide.json"))
    ap.add_argument("--rectified", default=None,
                    help="JSON written by tools/rectify_ultrawide.py, when the "
                         "frames have already been reprojected to a pinhole")
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--overlap", type=int, default=12)
    ap.add_argument("--max-dt", type=float, default=0.05,
                    help="widest image-to-depth gap to accept, seconds")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the pairing and the intrinsics, load no model")
    a = ap.parse_args(argv)

    session = Session(a.session)
    manifest_path = Path(a.session) / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if manifest.get("kind") != "multicam":
        print(f"  note: manifest kind is {manifest.get('kind')!r}, not 'multicam' — "
              f"if this session has poses, eval/pi3_chain.py is the right tool")

    rows = pair_by_time(session, a.max_dt)
    if len(rows) < 4:
        raise SystemExit(f"only {len(rows)} images paired with depth")
    gaps = np.array([abs(r["depth_dt"]) for r in rows])
    print(f"{len(rows)} images paired with depth; "
          f"gap median {np.median(gaps) * 1000:.1f} ms, worst {gaps.max() * 1000:.1f} ms")

    calib = load_calibration(Path(a.calibration))
    if a.rectified:
        rect = json.loads(Path(a.rectified).read_text())
        calib = {"fx": rect["fx"], "fy": rect.get("fy", rect["fx"]),
                 "cx": rect["cx"], "cy": rect["cy"],
                 "width": float(rect["width"]), "height": float(rect["height"])}
        print(f"  intrinsics from the rectifier: f {calib['fx']:.1f} at "
              f"{calib['width']:.0f}x{calib['height']:.0f}")
    hfov = active_hfov(manifest)
    if hfov:
        print(f"  the recorder logged the active format at {hfov:.1f} deg across; "
              f"using it rather than scaling the calibration, which is measured "
              f"on a different format")
    K = intrinsics_for(calib, rows[0]["width"], rows[0]["height"], hfov)
    print(f"  intrinsics at {rows[0]['width']}x{rows[0]['height']}: "
          f"fx {K[0, 0]:.1f} fy {K[1, 1]:.1f} cx {K[0, 2]:.1f} cy {K[1, 2]:.1f}")
    fov = 2 * np.degrees(np.arctan(rows[0]["width"] / (2 * K[0, 0])))
    print(f"  implied horizontal field of view {fov:.1f} deg")

    if a.dry_run:
        print("\n  dry run: the pairing and intrinsics above are what the model "
              "would be handed.")
        return 0

    # Imported here so --dry-run works on a machine with no GPU and no torch,
    # which is where the pairing is usually checked first.
    import shutil
    import tempfile
    import torch
    from pi3.models.pi3x import Pi3X
    from pi3.utils.basic import load_multimodal_data
    from umeyama import umeyama, ScaleNotObservable

    wide_calib = load_calibration(REPO / "calib" / "iphone17-1_wide.json")
    uw = json.loads((REPO / "calib" / "iphone17-1_ultrawide.json").read_text())
    cols = np.asarray(uw["extrinsic_matrix_columns"], dtype=np.float64)
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = cols[:3].T
    extrinsic[:3, 3] = cols[3] / 1000.0        # the file is in millimetres
    print(f"  depth reprojected into the ultra-wide through the "
          f"{np.linalg.norm(extrinsic[:3, 3]) * 1000:.2f} mm baseline")

    model = Pi3X.from_pretrained("yyfz233/Pi3X").eval().to("cuda")
    dtype = (torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8
             else torch.float16)

    window = min(a.window, len(rows))
    overlap = min(a.overlap, max(window - 1, 1))
    step = max(window - overlap, 1)
    starts = list(range(0, max(len(rows) - window, 0) + 1, step))
    if starts and starts[-1] + window < len(rows):
        starts.append(len(rows) - window)

    chained: dict[int, np.ndarray] = {}
    joins, residuals, scales = [], [], []
    broken_from = None

    for start in starts:
        block = rows[start:start + window]
        with tempfile.TemporaryDirectory() as tmp:
            images = Path(tmp) / "images"
            images.mkdir(parents=True, exist_ok=True)
            depths = []
            for order, row in enumerate(block):
                shutil.copyfile(session.frame_path(row), images / f"{order:04d}.jpg")
                depths.append(depth_in_ultrawide(session, row, K, wide_calib,
                                                 extrinsic))
            Ks = np.stack([K] * len(block))
            conditions = {"intrinsics": Ks, "depths": np.stack(depths), "poses": None}
            imgs, conditions = load_multimodal_data(
                str(images), conditions, interval=1, PIXEL_LIMIT=255000,
                verbose=False, device="cuda")
            measured = conditions["depths"]
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
                res = model(imgs=imgs, **conditions)
            pred = res["camera_poses"][0].float().cpu().numpy()
            local = res["local_points"][0].float().cpu().numpy()
            conf = torch.sigmoid(res["conf"][0, ..., 0]).float().cpu().numpy()
            meas = measured[0].float().cpu().numpy()

        # Metric scale from this window's own depth. There is no reference here
        # to borrow it from, which is the whole point of the mode.
        pz = local[..., 2]
        good = (meas > 0.05) & np.isfinite(meas) & (conf > 0.5) & (pz > 0.05)
        s = (float((pz[good] * meas[good]).sum()
                   / max((pz[good] * pz[good]).sum(), 1e-9))
             if good.sum() > 1000 else 1.0)
        scales.append(s)
        scaled = pred.copy()
        scaled[:, :3, 3] *= s
        frames = [int(r["frame"]) for r in block]

        if not chained:
            for f, T in zip(frames, scaled):
                chained[f] = T
            continue
        shared = [f for f in frames if f in chained]
        if len(shared) < 3:
            print(f"  ! only {len(shared)} shared frames at {frames[0]}, stopping")
            break
        src = np.stack([scaled[frames.index(f)][:3, 3] for f in shared])
        dst = np.stack([chained[f][:3, 3] for f in shared])
        try:
            R, js, t = umeyama(src, dst)
        except ScaleNotObservable as exc:
            print(f"  ! join at {shared[0]} refused: {exc}")
            break
        joins.append(js)
        # The guard `pi3_chain` grew for the same reason: on 2994fa one join sat
        # at 8.6 cm against a typical 0.4 and the chain reported nothing while
        # the trajectory went 2.45 m out. Without a reference here, this is the
        # only thing that can say the answer stopped being trustworthy.
        residual = float(np.median(np.linalg.norm(js * (src @ R.T) + t - dst, axis=1)))
        residuals.append(residual)
        if len(residuals) > 2:
            typical = float(np.median(residuals[:-1]))
            if residual > max(6.0 * typical, 0.03) and broken_from is None:
                broken_from = int(shared[0])
                print(f"  ! join at frame {shared[0]}: residual {residual*100:.1f} cm "
                      f"against a typical {typical*100:.1f} cm — the windows "
                      f"disagree about the shared path, and everything after "
                      f"this inherits it")
        for f, T in zip(frames, scaled):
            if f in chained:
                continue
            W = np.eye(4)
            W[:3, :3] = R @ T[:3, :3]
            W[:3, 3] = js * (R @ T[:3, 3]) + t
            chained[f] = W

    order = sorted(chained)
    estimate = np.stack([chained[f] for f in order])
    stamp = {int(r["frame"]): float(r["t"]) for r in rows}
    np.savez(a.out, estimate=estimate,
             frame=np.asarray(order, dtype=np.int64),
             t=np.asarray([stamp[f] for f in order]),
             join_scales=np.asarray(joins),
             join_residual_m=np.asarray(residuals),
             window_scale=np.asarray(scales),
             broken_from=np.asarray(-1 if broken_from is None else broken_from),
             convention="world_from_camera, metres; no reference — this session "
                        "has no tracker")
    walked = float(np.linalg.norm(np.diff(estimate[:, :3, 3], axis=0), axis=1).sum())
    print(f"\n{len(order)} poses over {len(starts)} windows, {walked:.2f} m walked")
    if joins:
        print(f"  join scales: " + " ".join(f"{j:.3f}" for j in joins[:10]))
        print(f"  join residuals cm: " + " ".join(f"{r*100:.1f}" for r in residuals[:10]))
    print(f"  per-window depth scale: " + " ".join(f"{s:.3f}" for s in scales[:10]))
    if broken_from is not None:
        print(f"  ** not trustworthy after frame {broken_from}")
    print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
