#!/usr/bin/env python3
"""Does a two-lens export actually put both lenses in one frame?

`tools/export_3dgs.py` on a `MultiCamRecorder` session poses the wide arm from a
trajectory and *derives* every ultra-wide pose from it through the rig transform
in `calib/`. Nothing in that construction can fail loudly. A dropped rotation, a
transform composed on the wrong side, a millimetre read as a metre, an
interpolation evaluated at the wrong clock — each produces a complete, valid,
plausible training set whose second lens is in the wrong place, and the only
symptom is a reconstruction that will not sharpen.

So this reads **only the exported files** and asks two questions of them.

    python3 tools/check_rig_export.py /tmp/gs_57f29e

**Is the baseline there?** The derived camera centre has to sit the calibrated
19.272 mm from the wide pose at the same instant. This tool re-derives that wide
pose by interpolating the *exported* wide cameras at the derived frame's own
timestamp, so it is not repeating the exporter's arithmetic with the exporter's
intermediate values.

**Does the rig transform predict the other lens's photograph?** Back-project a
wide frame's depth through its exported pose, project the points into the paired
ultra-wide frame through *its* exported pose, and compare the colour the point
carries with the colour the ultra-wide image has where it lands. That number
alone means nothing — a dim or flat scene scores well against anything — so it
comes with a **control**: the same points projected into an unrelated ultra-wide
frame from the same session. The ratio is the result, and `docs/3DGS.md` records
3.3x and 7.7x as what this style of check produced on the single-lens export.

It also comes with a **self-check**, run first: the same points projected back
into the wide frame they came from. That has a known answer — the colour must
come back almost exactly — and if it does not, the algebra here is wrong and
neither of the other two columns means anything.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_3dgs import interpolate_camera_pose, quat_to_matrix  # noqa: E402

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

# Nerfstudio's `transform_matrix` is camera-to-world in OpenGL axes — X right, Y
# up, Z back — while a depth map and a projection are +Z forward, +Y down. The
# same constant `export_3dgs.py` writes with, applied in the opposite direction.
GL_TO_CV = np.diag([1.0, -1.0, -1.0])

# What `calib/iphone17-1_ultrawide.json` says the rig is, in metres and degrees.
CALIB_BASELINE_M = 0.019272
CALIB_ROTATION_DEG = 0.4620


def load_export(path: str) -> tuple[dict, list[dict]]:
    with open(os.path.join(path, "transforms.json")) as fh:
        doc = json.load(fh)
    frames = doc["frames"]
    if not any("lens" in f for f in frames):
        raise SystemExit(f"{path} is not a multi-camera export: its frames carry "
                         f"no `lens`, so there is no second arm to check")
    if not all("t" in f for f in frames):
        raise SystemExit(f"{path} carries no per-frame `t`; re-export with a "
                         f"current tools/export_3dgs.py")
    return doc, frames


def camera_to_world(frame: dict) -> np.ndarray:
    """The frame's pose with +Z forward and +Y down, which is what projects."""
    T = np.asarray(frame["transform_matrix"], dtype=np.float64)
    out = np.eye(4)
    out[:3, :3] = T[:3, :3] @ GL_TO_CV
    out[:3, 3] = T[:3, 3]
    return out


def angle_between(A: np.ndarray, B: np.ndarray) -> float:
    """Degrees of rotation separating two orientations."""
    c = (np.trace(A.T @ B) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def check_baseline(frames: list[dict], tolerance_mm: float) -> int:
    """Criterion 3: the derived centre sits the calibrated baseline away.

    Reported against two references, because the pre-registration says they read
    differently and only one of them is defensible.

    *The wide pose at the same instant*, recovered here by interpolating the
    exported wide cameras. This is the construction's own claim and it has to
    hold to a tenth of a millimetre.

    *The nearest wide frame's own centre*, which is the criterion as it was
    written down. It only equals the first when the two lenses fire together,
    and they do not — 46-65 ms apart, over which the camera travels further than
    the baseline being applied. The gap between the two columns below is that
    motion, measured, and it is the reason the exporter interpolates.
    """
    source = sorted([f for f in frames if f["lens"] == "wide"], key=lambda f: f["t"])
    derived = sorted([f for f in frames if f["lens"] != "wide"], key=lambda f: f["t"])
    if not source or not derived:
        print("  no pair of arms in this export — nothing to check")
        return 0
    times = np.array([f["t"] for f in source])
    mats = np.stack([camera_to_world(f) for f in source])

    same_instant, nearest, rotations, gaps = [], [], [], []
    for f in derived:
        T = interpolate_camera_pose(times, mats, float(f["t"]))
        if T is None:
            continue
        C = camera_to_world(f)
        same_instant.append(float(np.linalg.norm(C[:3, 3] - T[:3, 3])))
        rotations.append(angle_between(T[:3, :3], C[:3, :3]))
        k = int(np.argmin(np.abs(times - f["t"])))
        nearest.append(float(np.linalg.norm(C[:3, 3] - mats[k][:3, 3])))
        gaps.append(abs(float(f["t"] - times[k])))

    same_instant = np.array(same_instant)
    nearest = np.array(nearest)
    worst = float(np.abs(same_instant - CALIB_BASELINE_M).max()) * 1000.0
    print(f"  {len(same_instant)} derived frames, lens gap median "
          f"{np.median(gaps) * 1000:.1f} ms, max {np.max(gaps) * 1000:.1f} ms")
    print(f"  vs the wide pose at the same instant: "
          f"{same_instant.mean() * 1000:.4f} mm mean, worst deviation from the "
          f"calibrated {CALIB_BASELINE_M * 1000:.3f} mm is {worst:.5f} mm")
    print(f"  rotation between the two: {np.median(rotations):.4f} deg median "
          f"against calib's {CALIB_ROTATION_DEG:.4f}")
    print(f"  vs the nearest wide frame's own centre: "
          f"{nearest.mean() * 1000:.2f} mm mean, "
          f"{np.median(nearest) * 1000:.2f} median, "
          f"{nearest.max() * 1000:.2f} worst")

    failures = 0
    if worst > tolerance_mm:
        print(f"  FAIL the derived centre is not the calibrated baseline from "
              f"the pose it was derived from ({worst:.5f} mm > {tolerance_mm})")
        failures += 1
    else:
        print(f"  ok   every derived centre is {CALIB_BASELINE_M * 1000:.3f} mm "
              f"from the wide pose at its instant, to {tolerance_mm} mm")
    literal = float(np.abs(nearest - CALIB_BASELINE_M).max()) * 1000.0
    if literal > tolerance_mm:
        print(f"  FAIL as literally pre-registered — against the nearest wide "
              f"*frame* the worst deviation is {literal:.2f} mm, which is the "
              f"camera moving between the two shutters, not the transform")
        failures += 1
    else:
        print(f"  ok   and the same holds against the nearest wide frame")
    return failures


def read_depth(export: str, frame: dict, scale: float) -> np.ndarray | None:
    path = os.path.join(export, frame.get("depth_file_path", ""))
    if not frame.get("depth_file_path") or not os.path.exists(path):
        return None
    if path.endswith(".npy"):
        return np.load(path).astype(np.float32) * scale
    with Image.open(path) as im:
        return np.asarray(im, dtype=np.float32) * scale


def colours_at(path: str, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"))
    return rgb[v, u].astype(np.float64)


def project(frame: dict, world: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pixels of world points in one exported camera, and which ones land in it."""
    C = camera_to_world(frame)
    cam = (world - C[:3, 3]) @ C[:3, :3]
    ahead = cam[:, 2] > 0.05
    u = np.full(len(world), -1.0)
    v = np.full(len(world), -1.0)
    u[ahead] = cam[ahead, 0] / cam[ahead, 2] * frame["fl_x"] + frame["cx"]
    v[ahead] = cam[ahead, 1] / cam[ahead, 2] * frame["fl_y"] + frame["cy"]
    iu, iv = np.round(u).astype(np.int64), np.round(v).astype(np.int64)
    inside = ahead & (iu >= 0) & (iu < frame["w"]) & (iv >= 0) & (iv < frame["h"])
    return iu, iv, inside


def check_reprojection(export: str, doc: dict, frames: list[dict], *,
                       stride: int, every: int, bar: float,
                       control_min_m: float) -> int:
    """Criterion 4: the rig transform has to predict the other lens's photograph.

    The control is picked mechanically — the derived frame half a sequence away —
    rather than chosen, because a control selected for being far away is a
    control selected to make the ratio look good.
    """
    scale = float(doc.get("depth_unit_scale_factor", 0.001))
    source = sorted([f for f in frames if f["lens"] == "wide"], key=lambda f: f["t"])
    derived = sorted([f for f in frames if f["lens"] != "wide"], key=lambda f: f["t"])
    if not source or not derived:
        print("  no pair of arms in this export — nothing to check")
        return 0
    times = np.array([f["t"] for f in source])

    rows, unscorable = [], 0
    for n, f in enumerate(derived[::every]):
        index = n * every
        wide = source[int(np.argmin(np.abs(times - f["t"])))]
        z = read_depth(export, wide, scale)
        if z is None:
            raise SystemExit(f"{export} has no depth for {wide['file_path']}; "
                             f"re-export without --no-depth, since this check "
                             f"is exactly the geometry the depth carries")
        z = z[::stride, ::stride]
        vv, uu = np.mgrid[0:z.shape[0], 0:z.shape[1]]
        uu, vv = uu * stride, vv * stride
        ok = (z > 0.15) & (z < 5.0)
        if ok.sum() < 500:
            continue
        u0, v0, z0 = uu[ok], vv[ok], z[ok]

        C = camera_to_world(wide)
        cam = np.stack([(u0 - wide["cx"]) * z0 / wide["fl_x"],
                        (v0 - wide["cy"]) * z0 / wide["fl_y"], z0], axis=1)
        world = cam @ C[:3, :3].T + C[:3, 3]
        carried = colours_at(os.path.join(export, wide["file_path"]), u0, v0)

        # Two controls, because one of them is not enough on its own.
        #
        # *Mechanical* is the frame half a sequence away — what `docs/3DGS.md`
        # used, chosen by counting rather than by looking. On a walk this short
        # the points usually do not reach it at all, so it scores on a handful
        # of frames and those are the ones where it happened to overlap.
        #
        # *Hardest* fixes that: among the frames at least `--control-min-m`
        # from the paired camera, the one the points reach most. It is chosen on
        # geometry alone and never on colour, so it cannot be selected for
        # flattering the answer, and being the most-overlapping unrelated frame
        # makes it the strictest control available rather than the kindest.
        mechanical = derived[(index + len(derived) // 2) % len(derived)]
        here = camera_to_world(f)[:3, 3]
        far = [g for g in derived
               if np.linalg.norm(camera_to_world(g)[:3, 3] - here) >= control_min_m]
        hardest = max(far, key=lambda g: int(project(g, world)[2].sum()), default=None)

        # The self column first: these points came out of this camera, so they
        # must go back into it. Anything but a near-zero here means the two
        # conventions in this file disagree and the other columns are noise.
        scored = {}
        targets = [("self", wide), ("pair", f), ("mechanical", mechanical)]
        if hardest is not None:
            targets.append(("hardest", hardest))
        for name, target in targets:
            iu, iv, inside = project(target, world)
            if inside.sum() < 200:
                continue
            got = colours_at(os.path.join(export, target["file_path"]),
                             iu[inside], iv[inside])
            want = carried[inside]
            scored[name] = (float(np.abs(got - want).mean()),
                            # The two lenses are separate sensors running their
                            # own auto-exposure, so some of the residual is a
                            # constant brightness difference and not a geometric
                            # disagreement at all. Removing the per-frame median
                            # offset says how much. It is applied to the control
                            # identically, so it cannot flatter the ratio.
                            float(np.abs((got - np.median(got - want, axis=0))
                                         - want).mean()),
                            int(inside.sum()))
        if "pair" not in scored:
            unscorable += 1
            continue
        scored["separation"] = {
            name: float(np.linalg.norm(camera_to_world(t)[:3, 3] - here))
            for name, t in targets[2:]}
        rows.append(scored)

    if not rows:
        raise SystemExit("no frame had a scorable paired projection; nothing "
                         "can be concluded")
    print(f"  {len(rows)} paired frames scored at stride {stride}, "
          f"{int(np.median([r['pair'][2] for r in rows]))} points landing in "
          f"the paired frame; {unscorable} frames had too few to score")
    self_mean = float(np.mean([r["self"][0] for r in rows]))
    print(f"  {'':36}   raw   exposure-matched  frames")
    print(f"  {'wide -> its own frame (self-check)':<36} {self_mean:6.2f}")

    failures = 0
    if self_mean > 12.0:
        print(f"  FAIL the self-check is {self_mean:.2f}/255 — points do not "
              f"return to the camera they came from, so this instrument is "
              f"measuring its own algebra and nothing below is readable")
        failures += 1
    else:
        print(f"  ok   points return to their own camera at {self_mean:.2f}/255")

    # Every control is scored against the paired arm **on the frames where that
    # control could be scored at all**. Averaging the paired arm over 89 frames
    # and a control over the 11 it happened to reach would compare two different
    # questions, and would quietly drop the control's own failure to overlap —
    # which is the thing that makes it a control.
    ratios = {}
    for name in ("mechanical", "hardest"):
        common = [r for r in rows if name in r]
        if not common:
            continue
        pair_mean = float(np.mean([r["pair"][0] for r in common]))
        ctrl_mean = float(np.mean([r[name][0] for r in common]))
        pair_flat = float(np.mean([r["pair"][1] for r in common]))
        ctrl_flat = float(np.mean([r[name][1] for r in common]))
        gap = float(np.median([r["separation"][name] for r in common]))
        ratios[name] = (ctrl_mean / pair_mean if pair_mean > 0 else float("inf"),
                        ctrl_flat / pair_flat if pair_flat > 0 else float("inf"))
        print(f"  {'wide -> the paired ultra-wide':<36} {pair_mean:6.2f}"
              f"        {pair_flat:6.2f}     {len(common)}")
        print(f"  {'wide -> control, ' + name:<36} {ctrl_mean:6.2f}"
              f"        {ctrl_flat:6.2f}     {len(common)}"
              f"   ({gap:.2f} m away)")
        print(f"    ratio {ratios[name][0]:.2f}x   "
              f"(exposure-matched {ratios[name][1]:.2f}x)")

    if not ratios:
        print("  FAIL no control was scorable; the paired column alone means "
              "nothing and no ratio can be reported")
        return failures + 1
    # The weakest of the controls decides. Reporting the best one would be
    # choosing the control after seeing the answer.
    worst_name = min(ratios, key=lambda k: ratios[k][0])
    ratio = ratios[worst_name][0]
    print(f"\n  ratio {ratio:.2f}x on the weakest control ({worst_name})")
    if ratio < bar:
        print(f"  FAIL {ratio:.2f}x is under the pre-registered {bar}x. A ratio "
              f"near 1.0 means the rig transform cannot tell the paired frame "
              f"from an unrelated one and is not working")
        failures += 1
    else:
        print(f"  ok   {ratio:.2f}x clears the pre-registered {bar}x")
    return failures


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("export", help="a directory written by export_3dgs.py")
    ap.add_argument("--stride", type=int, default=4,
                    help="subsample the wide depth by this before projecting")
    ap.add_argument("--every", type=int, default=1,
                    help="score every n-th derived frame")
    ap.add_argument("--ratio-bar", type=float, default=1.5,
                    help="lowest reprojection-versus-control ratio that passes")
    ap.add_argument("--control-min-m", type=float, default=1.0,
                    help="how far a control camera must sit from the paired "
                         "one to count as an unrelated view")
    ap.add_argument("--baseline-tolerance-mm", type=float, default=0.1)
    ap.add_argument("--skip-photometric", action="store_true",
                    help="geometry only; the photometric arm reads every image")
    a = ap.parse_args(argv)
    if Image is None:
        raise SystemExit("Pillow is required: pip install pillow")

    doc, frames = load_export(a.export)
    print(f"{a.export}: {len(frames)} frames, "
          + ", ".join(f"{sum(1 for f in frames if f.get('lens') == lens)} {lens}"
                      for lens in sorted({f.get("lens") for f in frames})))
    print("\ncriterion 3 — the calibrated baseline")
    failures = check_baseline(frames, a.baseline_tolerance_mm)
    if not a.skip_photometric:
        print("\ncriterion 4 — does the rig predict the other lens's photograph")
        failures += check_reprojection(a.export, doc, frames, stride=a.stride,
                                       every=a.every, bar=a.ratio_bar,
                                       control_min_m=a.control_min_m)
    print()
    print("all checks passed" if failures == 0 else f"{failures} check(s) FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
