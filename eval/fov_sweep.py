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
import subprocess
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
import chunked_conv  # noqa: E402

from pi3.models.pi3x import Pi3X  # noqa: E402
from pi3.utils.basic import load_multimodal_data  # noqa: E402


RECTIFIER_CACHE = {}


def rectifier_for(calibration_path, hfov_deg, width, height):
    """A `Rectifier` for this output size, built once and reused.

    The wide lens is not the pinhole this repo has been treating it as: its
    factory table magnifies by +1.1 % at a quarter of the corner radius and
    +3.9 % at the corner, and every pose number here was measured on frames
    with that left in. The ultra-wide, measured the same way, is *flatter* over
    most of its field — 0.4 to 0.6 % out to three quarters of the radius —
    which inverts the assumption the tooling was built on.

    So this arm asks the question that follows: does correcting the wide frames
    change what the pose model does with them? A win is free accuracy on the
    pipeline that already exists, and a loss prices what the ultra-wide path
    would have to pay for its own correction.
    """
    key = (calibration_path, hfov_deg, width, height)
    if key not in RECTIFIER_CACHE:
        import json
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
        from rectify_ultrawide import Rectifier
        with open(calibration_path) as fh:
            calib = json.load(fh)
        RECTIFIER_CACHE[key] = Rectifier(calib, hfov_deg, width, height)
    return RECTIFIER_CACHE[key]


def crop_for_hfov(width, fx, hfov_deg):
    """Crop width that turns a pinhole of focal fx into one of this field of view."""
    want = 2.0 * fx * math.tan(math.radians(hfov_deg) / 2.0)
    return int(min(width, max(64, round(want))))


def load_window_cropped(session, image_rows, poses, frames, tmpdir, hfov_deg,
                        rectify_calibration=None, mask_outside_deg=None,
                        resample_only=False):
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
        if mask_outside_deg is not None:
            # Keep the frame's geometry and blank the periphery, which separates
            # two explanations the crop arm cannot: does a wider view help
            # because the *bearings* spread further, or because there is simply
            # more scene in the frame? A crop removes both at once. This removes
            # the content and leaves the geometry — same image size, same fx,
            # same principal point, black outside the requested field.
            img = cv2.imread(str(src), cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            keep = crop_for_hfov(w, fx, mask_outside_deg)
            kh = int(round(keep * h / w))
            x0 = int(round(min(max(cx - keep / 2.0, 0), w - keep)))
            y0 = int(round(min(max(cy - kh / 2.0, 0), h - kh)))
            blanked = np.zeros_like(img)
            blanked[y0:y0 + kh, x0:x0 + keep] = img[y0:y0 + kh, x0:x0 + keep]
            cv2.imwrite(str(images / f"{order:04d}.jpg"), blanked,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            masked = np.zeros_like(depth)
            masked[y0:y0 + kh, x0:x0 + keep] = depth[y0:y0 + kh, x0:x0 + keep]
            depth = masked
            applied.append(mask_outside_deg)
        elif resample_only:
            # The other half of the rectification question. Rectifying does two
            # things at once — it corrects the lens and it resamples every pixel
            # with Lanczos — and correcting made Pi3X worse in all four sessions
            # tried. This arm resamples the same way and corrects nothing, by
            # remapping through the identity. If it loses too, the cost is the
            # resample; if it does not, the cost is removing a distortion the
            # model was trained to expect.
            img = cv2.imread(str(src), cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            u, v = np.meshgrid(np.arange(w, dtype=np.float32),
                               np.arange(h, dtype=np.float32))
            cv2.imwrite(str(images / f"{order:04d}.jpg"),
                        cv2.remap(img, u, v, cv2.INTER_LANCZOS4,
                                  borderMode=cv2.BORDER_REPLICATE),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        elif rectify_calibration is not None:
            # Undo the lens and reproject onto a pinhole of the same field, so
            # the only thing that moves against the baseline is the distortion.
            img = cv2.imread(str(src), cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            hf = hfov_deg if hfov_deg is not None else 2.0 * math.degrees(
                math.atan(w / (2.0 * fx)))
            rect = rectifier_for(rectify_calibration, hf, w, h)
            cv2.imwrite(str(images / f"{order:04d}.jpg"), rect.rectify(img),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            depth = rect.rectify(depth, correct=True)
            fx = fy = rect.f_out
            cx, cy = w / 2.0, h / 2.0
            applied.append(hf)
        elif hfov_deg is None:
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


# One model for the whole sweep. Building it per arm leaves the caching
# allocator holding each arm's peak, so a four-arm run reserves far more of the
# card than any single arm needs — which is how this collided with another
# session's job and killed it. The weights do not depend on the field of view.
_MODEL = None


def _model(device):
    global _MODEL
    if _MODEL is None:
        _MODEL = Pi3X.from_pretrained("yyfz233/Pi3X").eval().to(device)
        chunked_conv.apply(_MODEL)
    return _MODEL


def release():
    """Hand the card back between sessions, so a queued job can start."""
    global _MODEL
    _MODEL = None
    torch.cuda.empty_cache()


def run(session_path, dump, hfov_deg, window, overlap, condition,
        device="cuda", save_trajectory=None, rectify_calibration=None,
        mask_outside_deg=None, resample_only=False):
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

    model = _model(device)
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
                session, image_rows, poses, frames, Path(tmp), hfov_deg,
                rectify_calibration=rectify_calibration,
                mask_outside_deg=mask_outside_deg,
                resample_only=resample_only)
            actual = got if got is not None else actual
            conditions = {"intrinsics": Ks if "intrinsics" in condition else None,
                          "depths": depths, "poses": None}
            imgs, conditions = load_multimodal_data(
                path, conditions, interval=1, PIXEL_LIMIT=E.PIXEL_LIMIT,
                verbose=False, device=device)
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

    if save_trajectory:
        # 각 시야각 arm 의 궤적. 여기 ATE 는 이미 알고 있으므로, 참조 없이
        # 도는 검사(가속도계 상관)가 그 ATE 를 예측하는지 시험할 수 있는
        # 재료가 된다 — 지금 그 검사가 잡는 두 세션은 이미 알던 것들이라,
        # 모르던 실패를 잡는지가 아직 미확인이다.
        stamp = {int(f): float(s) for f, s in zip(d["frame"], d["t"])}
        np.savez(save_trajectory,
                 frame=np.asarray(frames, dtype=np.int64),
                 t=np.asarray([stamp[int(f)] for f in frames], dtype=np.float64),
                 estimate=np.stack([chained[int(f)] for f in frames]),
                 reference=np.stack([ref_by_frame[int(f)] for f in frames]),
                 hfov=float("nan") if actual is None else float(actual),
                 convention="world_from_camera, +Z forward +Y down (depth frame)")
    return {
        "session": root.name, "hfov_requested": hfov_deg, "hfov_actual": actual,
        "frames": len(frames), "windows": len(joins) + 1,
        "travelled_m": travelled,
        "loop_arkit_pct": loop(ref) / travelled * 100,
        "loop_pi3_pct": loop(est) / travelled * 100,
        "loop_icp_pct": loop(icp) / travelled * 100,
        "ate_pi3_cm": ate(est) * 100, "ate_icp_cm": ate(icp) * 100,
    }



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
    ap.add_argument("--in-process", action="store_true",
                    help="run this arm here instead of forking. The parent uses "
                         "it for the children; a measurement taken any other way "
                         "shares a process with its neighbours")
    ap.add_argument("--hfov", type=float, nargs="+", default=[None],
                    help="horizontal fields of view to crop to; 0 means native")
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--overlap", type=int, default=24)
    ap.add_argument("--condition", default="intrinsics")
    ap.add_argument("--rectify-calibration", default=None,
                    help="undo the lens with this calibration.json and reproject "
                         "onto a pinhole before the model sees the frame. The "
                         "wide lens magnifies by 3.9%% at the corner and every "
                         "pose number in this repo was measured with that left "
                         "in, so this arm prices correcting it")
    ap.add_argument("--mask-outside", type=float, default=None,
                    help="blank everything beyond this field of view but keep "
                         "the frame's size and intrinsics. A crop removes the "
                         "peripheral content AND the spread of bearings; this "
                         "removes only the content, so the two explanations for "
                         "why a wider lens helps can be told apart")
    ap.add_argument("--resample-only", action="store_true",
                    help="remap through the identity: resamples exactly as "
                         "rectification does and corrects nothing, which is the "
                         "control that says whether the cost was the correction "
                         "or the resample")
    ap.add_argument("--out")
    ap.add_argument("--save-trajectory-dir",
                    help="arm 마다 궤적을 이 디렉토리에 <세션>_<시야각>.npz 로 저장")
    args = ap.parse_args(argv)
    # The parent holds the card for the whole sweep; its children are the
    # sweep, so they must not queue behind it.
    if not args.in_process:
        single_instance("fov_sweep")

    results = []
    # One measurement per process. Running the same forward twice inside a
    # single process does not reproduce — 0.5 to 1.4% on the fitted scale,
    # measured on two different cards — while a fresh process reproduces to
    # printed precision. The suspected cause is Pi3's attention backend being
    # offered as a list, which lets the runtime choose per call, but that is not
    # confirmed; the rule follows the measurement rather than the diagnosis.
    #
    # So each arm is run in its own child, and the parent only collects. The
    # arms of a sweep are the thing being compared, and comparing measurements
    # that share a process compares them through whatever the runtime picked.
    for h in args.hfov:
        hv = None if (h is None or h <= 0) else float(h)
        if args.in_process:
            saved = None
            if args.save_trajectory_dir:
                Path(args.save_trajectory_dir).mkdir(parents=True, exist_ok=True)
                tag = "native" if hv is None else f"{hv:.0f}deg"
                saved = str(Path(args.save_trajectory_dir)
                            / f"{Path(args.session).name[-6:]}_{tag}.npz")
            r = run(args.session, args.dump, hv, args.window, args.overlap,
                    args.condition, save_trajectory=saved,
                    rectify_calibration=args.rectify_calibration,
                    mask_outside_deg=args.mask_outside,
                    resample_only=args.resample_only)
        else:
            child = subprocess.run(
                [sys.executable, __file__, args.session, "--dump", args.dump,
                 "--window", str(args.window), "--overlap", str(args.overlap),
                 "--condition", args.condition, "--in-process"]
                + (["--rectify-calibration", args.rectify_calibration]
                   if args.rectify_calibration else [])
                + (["--mask-outside", str(args.mask_outside)]
                   if args.mask_outside else [])
                + (["--resample-only"] if args.resample_only else [])
                + (["--save-trajectory-dir", args.save_trajectory_dir]
                   if args.save_trajectory_dir else [])
                + [
                 "--hfov", str(h if h is not None else 0), "--out", "-"],
                capture_output=True, text=True)
            if child.returncode != 0:
                print(child.stderr[-400:], file=sys.stderr)
                raise SystemExit(f"arm hfov={h} failed")
            r = json.loads(child.stdout.strip().splitlines()[-1])[0]
        tag = "native" if hv is None else f"{r['hfov_actual']:.1f}deg"
        r["arm"] = tag
        results.append(r)
        print(f"  {tag:>10}  ATE {r['ate_pi3_cm']:.1f} cm   "
              f"loop {r['loop_pi3_pct']:.2f}%", flush=True)

    if args.out == "-":
        print(json.dumps(results))
    elif args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
