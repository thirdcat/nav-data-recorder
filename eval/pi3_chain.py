#!/usr/bin/env python3
"""Chain overlapping Pi3X windows into one trajectory, and score the loop.

Each window is internally consistent and independently scaled. Joining them is
the step the windowed numbers do not cover, and it is where an approach that
wins per window can still lose a trajectory — the same shape as a photometric
term that won pair-wise and lost the walk.

Consecutive windows share `--overlap` frames. The join is the similarity
transform (rotation, translation, one scale) that best maps the newer window's
poses onto the running trajectory over exactly those shared frames, by Umeyama.
Scale is part of the join because each window carries its own, and forcing them
equal would bake the first window's scale into everything after it.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, os.environ.get("PI3_ROOT", ""))
import pi3_eval as E  # noqa: E402
from umeyama import umeyama  # noqa: E402

from pi3.models.pi3x import Pi3X  # noqa: E402
from pi3.utils.basic import load_multimodal_data  # noqa: E402



def single_instance(name):
    """Refuse to start when another copy of this job already holds the card.

    Free memory is the wrong question to ask: three copies of the same sweep each
    saw room at launch, and then took the card from one another. What matters is
    whether one is already running.
    """
    import atexit
    import fcntl
    import tempfile
    path = Path(tempfile.gettempdir()) / f"{name}.lock"
    handle = open(path, "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit(
            f"another {name} is already running ({path}) — "
            "GPU jobs here run one at a time")
    atexit.register(handle.close)
    return handle


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--dump", required=True)
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--overlap", type=int, default=12)
    ap.add_argument("--condition", default="intrinsics")
    ap.add_argument("--out")
    ap.add_argument("--save-trajectory",
                    help="npz of the chained trajectory on the same frames as "
                         "--dump, so it can be checked against signals that need "
                         "no reference — the accelerometer above all, which is "
                         "the only independent witness translation has")
    ap.add_argument("--save-windows",
                    help="npz of the raw per-window predictions, so the join "
                         "can be re-examined without paying for the GPU again")
    args = ap.parse_args(argv)
    single_instance("pi3_chain")

    root = Path(args.session)
    session = E.Session(args.session)
    rows = [json.loads(l) for l in (root / "frames.jsonl").read_text().splitlines() if l.strip()]
    image_rows = {r["frame"]: r for r in rows}
    E.depth_by_frame = {d["frame"]: d for d in
                        (json.loads(l) for l in (root / "depth.jsonl").read_text().splitlines() if l.strip())}
    poses = {p["frame"]: p for p in
             (json.loads(l) for l in (root / "pose.jsonl").read_text().splitlines() if l.strip())}

    dump = np.load(args.dump)
    est_by_frame = {int(f): T for f, T in zip(dump["frame"], dump["estimate"])}
    ref_by_frame = {int(f): T for f, T in zip(dump["frame"], dump["reference"])}
    usable = [f for f in sorted(image_rows)
              if f in E.depth_by_frame and f in poses and f in ref_by_frame]

    model = Pi3X.from_pretrained("yyfz233/Pi3X").eval().to("cuda")
    dtype = (torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8
             else torch.float16)

    # A session shorter than the window is one window, not none — and one
    # window is the goal, because every joining problem recorded here came from
    # a seam. `--window 400` reads as "a single pass if it fits".
    window = min(args.window, len(usable))
    overlap = min(args.overlap, max(window - 1, 1))
    step = max(window - overlap, 1)
    chained = {}            # frame -> 4x4 world-from-camera, one common frame
    joins = []
    join_residuals: list[float] = []
    broken_join = None
    saved = {"frames": [], "pred": [], "scale": []}
    # Anchor a final window to the end of the session. Stepping until the next
    # window would overrun leaves a tail uncovered — 1868dd came out at 240 of
    # its 260 frames — and the loop score is the distance from the last frame to
    # the first, so dropping the tail does not degrade that number, it replaces
    # it with a different one. The count printed below was honest and still told
    # nobody, because only a reader who knew the session length could see it.
    starts = list(range(0, max(len(usable) - window, 0) + 1, step))
    if starts and starts[-1] + window < len(usable):
        starts.append(len(usable) - window)
    for start in starts:
        frames = usable[start:start + window]
        with tempfile.TemporaryDirectory() as tmp:
            path, depths, Ks, reference = E.load_window(
                session, image_rows, poses, frames, Path(tmp))
            conditions = {"intrinsics": Ks if "intrinsics" in args.condition else None,
                          "depths": depths,
                          "poses": None}
            imgs, conditions = load_multimodal_data(
                path, conditions, interval=1, PIXEL_LIMIT=E.PIXEL_LIMIT,
                verbose=False, device="cuda")
            measured = conditions["depths"]
            if "depth" not in args.condition:
                conditions["depths"] = None
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=dtype):
                    res = model(imgs=imgs, **conditions)
            pred = res["camera_poses"][0].float().cpu().numpy()
            local = res["local_points"][0].float().cpu().numpy()
            conf = torch.sigmoid(res["conf"][0, ..., 0]).float().cpu().numpy()
            meas = measured[0].float().cpu().numpy()

        # Metric scale from this window's own LiDAR, never from the reference.
        pz = local[..., 2]
        good = (meas > 0.05) & np.isfinite(meas) & (conf > 0.5) & (pz > 0.05)
        s = (float((pz[good] * meas[good]).sum()
                   / max((pz[good] * pz[good]).sum(), 1e-9))
             if good.sum() > 1000 else 1.0)
        scaled = pred.copy()
        scaled[:, :3, 3] *= s
        saved["frames"].append(np.asarray(frames, dtype=np.int64))
        saved["pred"].append(pred)
        saved["scale"].append(s)

        if not chained:
            for f, T in zip(frames, scaled):
                chained[f] = T
        else:
            shared = [f for f in frames if f in chained]
            if len(shared) < 3:
                print(f"  ! only {len(shared)} shared frames, cannot join")
                break
            src = np.stack([scaled[frames.index(f)][:3, 3] for f in shared])
            dst = np.stack([chained[f][:3, 3] for f in shared])
            R, js, t = umeyama(src, dst)
            joins.append(js)

            # How far apart the two windows still are once the best similarity
            # has been applied. This is the number that mattered on 2994fa and
            # nothing was watching it: every join in that session sat at
            # 0.3-1.4 cm except one, which was 9.8 cm median and 43.3 cm worst,
            # and the chain went on to report a trajectory 2.45 m out.
            #
            # The scale is a symptom, not the disease. Two windows that disagree
            # about the *shape* of the shared path cannot be reconciled by any
            # similarity, so the fit spends its freedom hiding the disagreement
            # — in the scale if it has one (js fell to 0.73 there, which read as
            # a scale collapse), in the orientation if it does not (holding
            # js = 1 leaves the ATE unchanged at 246 cm and bends the chain
            # instead). Refusing a bad join is the only thing that helps.
            residual = np.linalg.norm(js * (src @ R.T) + t - dst, axis=1)
            join_residuals.append(float(np.median(residual)))
            if len(join_residuals) > 2:
                typical = float(np.median(join_residuals[:-1]))
                if join_residuals[-1] > max(6.0 * typical, 0.03):
                    print(f"  ! join at frame {shared[0]}: residual "
                          f"{join_residuals[-1]*100:.1f} cm against a typical "
                          f"{typical*100:.1f} cm — the windows disagree about "
                          f"the shared path, and everything after this point "
                          f"inherits the disagreement")
                    broken_join = int(shared[0])
            for f, T in zip(frames, scaled):
                if f in chained:
                    continue
                W = np.eye(4)
                W[:3, :3] = R @ T[:3, :3]
                W[:3, 3] = js * (R @ T[:3, 3]) + t
                chained[f] = W

    if len(chained) != len(usable):
        raise SystemExit(
            f"windowing covered {len(chained)} of {len(usable)} frames — "
            "refusing to score a trajectory that silently dropped some")
    frames = sorted(chained)
    if len(frames) < 2:
        print("  not enough frames chained")
        return 1
    est = np.stack([chained[f][:3, 3] for f in frames])
    ref = np.stack([ref_by_frame[f][:3, 3] for f in frames])
    icp = np.stack([est_by_frame[f][:3, 3] for f in frames])

    def loop(p):
        return float(np.linalg.norm(p[-1] - p[0]))

    travelled = float(np.linalg.norm(np.diff(ref, axis=0), axis=1).sum())

    def ate(p):
        pc, rc = p - p.mean(0), ref - ref.mean(0)
        U, _, Vt = np.linalg.svd(pc.T @ rc)
        d = np.sign(np.linalg.det(U @ Vt))
        R = U @ np.diag([1, 1, d]) @ Vt
        return float(np.linalg.norm(pc @ R - rc, axis=1).mean())

    out = {
        "session": root.name, "frames": len(frames), "windows": len(joins) + 1,
        "travelled_m": travelled,
        "join_scales": joins,
        "join_residual_m": join_residuals,
        "broken_join_frame": broken_join,
        "loop_arkit": loop(ref), "loop_pi3": loop(est), "loop_icp": loop(icp),
        "ate_pi3_cm": ate(est) * 100, "ate_icp_cm": ate(icp) * 100,
    }
    print(f"{root.name}: {len(frames)} frames, {len(joins)+1} windows, "
          f"{travelled:.1f} m walked")
    if broken_join is not None:
        print(f"  ** this trajectory is not trustworthy after frame {broken_join}")
    print(f"  join residuals cm: "
          + " ".join(f"{r*100:.1f}" for r in join_residuals[:10])
          + (" ..." if len(join_residuals) > 10 else ""))
    print(f"  join scales: "
          + " ".join(f"{j:.3f}" for j in joins[:10])
          + (" ..." if len(joins) > 10 else ""))
    print(f"  loop   ARKit {out['loop_arkit']:.3f}   Pi3X {out['loop_pi3']:.3f}"
          f"   ICP {out['loop_icp']:.3f}   (m)")
    print(f"  loop % of path   ARKit {out['loop_arkit']/travelled*100:.1f}%"
          f"   Pi3X {out['loop_pi3']/travelled*100:.1f}%"
          f"   ICP {out['loop_icp']/travelled*100:.1f}%")
    print(f"  ATE    Pi3X {out['ate_pi3_cm']:.1f} cm   ICP {out['ate_icp_cm']:.1f} cm")
    if args.save_trajectory:
        # Positions only where the chain has them, with the timestamps and the
        # ARKit reference for the same frames, so a check can be run either side
        # by side or on its own once ARKit is gone.
        keep = np.asarray(frames, dtype=np.int64)
        stamp = {int(f): float(s) for f, s in zip(dump["frame"], dump["t"])}
        np.savez(args.save_trajectory,
                 frame=keep,
                 t=np.asarray([stamp[int(f)] for f in keep], dtype=np.float64),
                 estimate=np.stack([chained[int(f)] for f in keep]),
                 reference=np.stack([ref_by_frame[int(f)] for f in keep]),
                 convention="world_from_camera, +Z forward +Y down (depth frame)")
    if args.save_windows:
        np.savez(args.save_windows,
                 frames=np.stack(saved["frames"]),
                 pred=np.stack(saved["pred"]),
                 scale=np.asarray(saved["scale"], dtype=np.float64))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
