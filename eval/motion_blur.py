#!/usr/bin/env python3
"""How far a pixel travels during one exposure, and whether the photograph agrees.

The streak is computable from what is already on disk. Back-project a frame's
depth with its own pose, project the points through the pose interpolated at
exposure start and at exposure end, and read the displacement. Nothing here
looks at a pixel, so the image is left free to be an independent witness.

Three modes, in the order the argument has to be made:

    python3 eval/motion_blur.py extent    ~/nav_data/<id>          how big
    python3 eval/motion_blur.py inventory ~/nav_data/*/            who has it
    python3 eval/motion_blur.py witness   ~/nav_data/<id>          is it real
    python3 eval/motion_blur.py windows   ~/nav_data/<id>          what the window choice costs
    python3 eval/motion_blur.py --self-test

`extent` splits the streak into a rotation-only and a translation-only arm,
because they want different kernels: rotation moves every pixel by nearly the
same vector, translation moves each by an amount proportional to 1/Z. The
`spread` column — the streak's own p90 minus p10 within one frame — is the
number that decides whether a single per-frame kernel can do the job.

`witness` is the part that cannot be argued with. Motion blur suppresses image
gradients ALONG the smear and leaves them across it, so

    R = gradient energy along the predicted angle / energy across it

must come back under 1. Room content has its own strong vertical and horizontal
grain, which would produce exactly that by accident, so every frame is also
scored with ANOTHER frame's predicted angle: only the gap between the two
columns is the result. See `docs/3DGS.md` § *Motion blur is 6-12 px*.
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from read_session import Session  # noqa: E402

try:
    import cv2
except ImportError:  # only `witness` needs it
    cv2 = None

ARKIT_TO_COLMAP = np.diag([1.0, -1.0, -1.0])

# The three readings of `ARFrame.timestamp` that Apple's documentation leaves
# open, as (start, end) in units of the exposure. `eval/frame_time.py` measures
# which one is right; it is `centred`, and that is this file's default.
WINDOWS = {"lead": (-1.0, 0.0), "centred": (-0.5, 0.5), "trail": (0.0, 1.0)}


def quat_to_matrix(qx, qy, qz, qw):
    n = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw) or 1.0
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)],
    ])


def slerp(q0, q1, u):
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + u*(q1 - q0)
    else:
        th = math.acos(max(-1.0, min(1.0, d)))
        q = (math.sin((1-u)*th)*q0 + math.sin(u*th)*q1) / math.sin(th)
    return q / np.linalg.norm(q)


def pose_at(poses, times, t, max_gap=0.1):
    """Interpolated (R_c2w in COLMAP axes, C) at monotonic time `t`.

    Returns None outside the stream, and None across a gap wider than
    `max_gap` — a tracking dropout must not be interpolated over as if the
    camera had travelled smoothly through it.
    """
    i = int(np.searchsorted(times, t))
    if i <= 0 or i >= len(times):
        return None
    a, b = poses[i-1], poses[i]
    span = times[i] - times[i-1]
    if span <= 0 or span > max_gap:
        return None
    u = (t - times[i-1]) / span
    q = slerp([a["qx"], a["qy"], a["qz"], a["qw"]],
              [b["qx"], b["qy"], b["qz"], b["qw"]], u)
    R = quat_to_matrix(*q) @ ARKIT_TO_COLMAP
    C = (np.array([a["tx"], a["ty"], a["tz"]])*(1-u)
         + np.array([b["tx"], b["ty"], b["tz"]])*u)
    return R, C


def project(R, C, K, pts):
    """Pixels for world points through a camera at (R_c2w, C), K = (fx,fy,cx,cy)."""
    cam = (pts - C) @ R
    z = cam[:, 2]
    ok = z > 1e-3
    uv = np.full((len(pts), 2), np.nan)
    uv[ok, 0] = K[0]*cam[ok, 0]/z[ok] + K[2]
    uv[ok, 1] = K[1]*cam[ok, 1]/z[ok] + K[3]
    return uv


def frame_cloud(s, row, stride=8, conf_min=2, near=0.15, far=8.0):
    """World points from one frame's depth, and the depth pixel rows they came from.

    The rows come back because `eval/frame_time.py` needs to bin by them; a
    rolling shutter is a per-row question and this is the only place the
    association exists.
    """
    p, d = row["pose"], row["depth"]
    z = np.asarray(s.depth_frame(d), np.float32)
    conf = s.confidence_frame(d)
    conf = np.asarray(conf) if conf is not None else None
    dh, dw = z.shape
    sx, sy = dw/row["width"], dh/row["height"]
    Kd = (p["fx"]*sx, p["fy"]*sy, p["cx"]*sx, p["cy"]*sy)
    vv, uu = np.mgrid[0:dh, 0:dw]
    ok = np.isfinite(z) & (z > near) & (z < far)
    if conf is not None and conf_min > 0:
        ok &= conf >= conf_min
    sel = np.zeros_like(ok)
    sel[::stride, ::stride] = True
    ok &= sel
    if ok.sum() < 50:
        return None
    cam = np.stack([(uu[ok]-Kd[2])*z[ok]/Kd[0],
                    (vv[ok]-Kd[3])*z[ok]/Kd[1], z[ok]], axis=-1)
    return cam, vv[ok], dh


def usable_frames(s):
    """Image rows that carry a pose at normal tracking, a depth map and an exposure."""
    out = []
    for row in s.posed_images():
        p, d = row.get("pose"), row.get("depth")
        if not p or not d:
            continue
        if not str(p.get("tracking", "")).startswith("normal"):
            continue
        if not p.get("exposure"):
            continue
        out.append(row)
    return out


def streaks(s, row, poses, times, window="centred", stride=8):
    """Per-point exposure displacement in full-resolution pixels, and its parts.

    Returns (total, rotation only, translation only, depth rows, median depth).
    The rotation arm holds the camera centre at mid-exposure and the translation
    arm holds its orientation, so the two are the same motion split rather than
    two different experiments.
    """
    p = row["pose"]
    exp = p["exposure"]
    t = p["t"]
    lo, hi = WINDOWS[window]
    A = pose_at(poses, times, t + lo*exp)
    B = pose_at(poses, times, t + hi*exp)
    M = pose_at(poses, times, t)
    if A is None or B is None or M is None:
        return None
    cloud = frame_cloud(s, row, stride=stride)
    if cloud is None:
        return None
    cam, rows_v, dh = cloud
    Rm, Cm = M
    pts = cam @ Rm.T + Cm
    Kf = (p["fx"], p["fy"], p["cx"], p["cy"])
    Ra, Ca = A
    Rb, Cb = B
    uva, uvb = project(Ra, Ca, Kf, pts), project(Rb, Cb, Kf, pts)
    total = np.linalg.norm(uvb - uva, axis=1)
    rot = np.linalg.norm(project(Rb, Cm, Kf, pts) - project(Ra, Cm, Kf, pts), axis=1)
    tr = np.linalg.norm(project(Rm, Cb, Kf, pts) - project(Rm, Ca, Kf, pts), axis=1)
    g = np.isfinite(total) & np.isfinite(rot) & np.isfinite(tr)
    if g.sum() < 50:
        return None
    return total[g], rot[g], tr[g], rows_v[g], float(np.median(cam[g, 2])), uvb[g] - uva[g]


def per_session(path, stride=8, window="centred", limit=None):
    s = Session(path)
    poses = s.poses()
    times = np.array([p["t"] for p in poses])
    rows = usable_frames(s)
    out = []
    for row in (rows[:limit] if limit else rows):
        r = streaks(s, row, poses, times, window=window, stride=stride)
        if r is None:
            continue
        total, rot, tr, _, zmed, _ = r
        out.append((float(np.median(total)), float(np.percentile(total, 90)),
                    float(np.median(rot)), float(np.median(tr)),
                    float(np.percentile(total, 90) - np.percentile(total, 10)),
                    row["pose"]["exposure"]*1000.0, zmed))
    return np.array(out) if out else None


def cmd_extent(args):
    print(f"{'session':>8} {'frames':>6} {'blur_med':>8} {'blur_p90':>8} "
          f"{'rot':>6} {'trans':>6} {'spread':>6} {'exp_ms':>6} {'depth_m':>7}")
    for path in args.sessions:
        r = per_session(path, stride=args.stride, window=args.window)
        name = os.path.basename(path.rstrip("/"))[-6:]
        if r is None:
            print(f"{name:>8}      -")
            continue
        print(f"{name:>8} {len(r):>6} {np.median(r[:,0]):>8.1f} {np.median(r[:,1]):>8.1f} "
              f"{np.median(r[:,2]):>6.1f} {np.median(r[:,3]):>6.1f} {np.median(r[:,4]):>6.1f} "
              f"{np.median(r[:,5]):>6.1f} {np.median(r[:,6]):>7.2f}")


def cmd_inventory(args):
    """Every session ranked by blur, which is what picks the arms of an experiment.

    `conf%` is here because a session with no confident depth has no depth
    supervision to offer and cannot be an arm whatever its blur says.
    """
    rows = []
    for path in args.sessions:
        name = os.path.basename(path.rstrip("/"))[-6:]
        try:
            s = Session(path)
        except (FileNotFoundError, ValueError):
            continue
        try:
            r = per_session(path, stride=8, window=args.window, limit=args.limit)
        except Exception as exc:                       # a torn session must not stop the sweep
            print(f"{name:>8}   skipped: {type(exc).__name__}", file=sys.stderr)
            continue
        if r is None:
            continue
        confs = []
        for d in s.depth_index()[::20]:
            c = s.confidence_frame(d)
            if c is not None:
                confs.append(float((np.asarray(c) >= 2).mean()))
        rows.append((name, len(s.frames()), len(r), float(np.median(r[:, 5])),
                     float(np.median(r[:, 0])), float(np.median(r[:, 1])),
                     100*float(np.median(confs)) if confs else float("nan")))
    rows.sort(key=lambda r: -r[4])
    print(f"{'session':>8} {'imgs':>5} {'scored':>7} {'exp_ms':>7} {'blur':>6} {'blur90':>7} {'conf%':>6}")
    for r in rows:
        print(f"{r[0]:>8} {r[1]:>5} {r[2]:>7} {r[3]:>7.1f} {r[4]:>6.1f} {r[5]:>7.1f} {r[6]:>6.0f}")
    print(f"\n{len(rows)} sessions scored")


def cmd_witness(args):
    """Does the photograph agree, in magnitude and in direction.

    The shuffle column is the whole instrument. Without it a room full of
    vertical edges returns R < 1 for any angle that is not vertical, and the
    result would be the furniture.
    """
    if cv2 is None:
        raise SystemExit("witness needs opencv (cv2)")
    from scipy.stats import spearmanr
    rng = np.random.default_rng(args.seed)
    print(f"{'session':>8} {'n':>4} " + " ".join(f"{w:>7}" for w in WINDOWS)
          + f" {'R_all':>7} {'ctrl':>7} {'R_hi':>7} {'ctrl':>7} {'r(m,R)':>7} {'ctrl':>7}")
    for path in args.sessions:
        s = Session(path)
        poses = s.poses()
        times = np.array([p["t"] for p in poses])
        rec = []
        for row in usable_frames(s):
            per_window = {}
            for wname in WINDOWS:
                r = streaks(s, row, poses, times, window=wname, stride=6)
                if r is None:
                    per_window = None
                    break
                v = r[5]
                mag = np.linalg.norm(v, axis=1)
                # circular mean of the axis (mod 180), weighted by how far it moved
                a2 = np.arctan2(v[:, 1], v[:, 0]) * 2.0
                th = 0.5*np.arctan2(np.sum(mag*np.sin(a2)), np.sum(mag*np.cos(a2)))
                per_window[wname] = (float(np.median(mag)), float(th))
            if per_window is None:
                continue
            img = cv2.imread(s.frame_path(row), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.GaussianBlur(img, (0, 0), 1.0)
            gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
            # E(theta) = Jxx cos^2 + 2 Jxy cos sin + Jyy sin^2, so three numbers
            # hold the whole directional energy curve and no image is retained.
            rec.append({"w": per_window,
                        "Jxx": float(np.mean(gx*gx)), "Jyy": float(np.mean(gy*gy)),
                        "Jxy": float(np.mean(gx*gy)),
                        "lap": float(cv2.Laplacian(img, cv2.CV_64F).var())})
        name = os.path.basename(path.rstrip("/"))[-6:]
        if len(rec) < 30:
            print(f"{name:>8} {len(rec):>4}   too few")
            continue
        lap = np.array([r["lap"] for r in rec])
        keep = lap > 0
        line = [f"{name:>8} {len(rec):>4}"]
        for wname in WINDOWS:
            mag = np.array([r["w"][wname][0] for r in rec])
            line.append(f"{spearmanr(mag[keep], lap[keep])[0]:>7.3f}")
        R_true, R_ctrl = [], []
        perm = rng.permutation(len(rec))
        for i, r in enumerate(rec):
            R_true.append(energy_ratio(r, r["w"]["centred"][1]))
            R_ctrl.append(energy_ratio(r, rec[perm[i]]["w"]["centred"][1]))
        R_true, R_ctrl = np.array(R_true), np.array(R_ctrl)
        mag = np.array([r["w"]["centred"][0] for r in rec])
        hi = mag >= np.percentile(mag, 66)
        line += [f"{np.median(R_true):>7.3f}", f"{np.median(R_ctrl):>7.3f}",
                 f"{np.median(R_true[hi]):>7.3f}", f"{np.median(R_ctrl[hi]):>7.3f}",
                 f"{spearmanr(mag, R_true)[0]:>7.3f}", f"{spearmanr(mag, R_ctrl)[0]:>7.3f}"]
        print(" ".join(line))


def cmd_windows(args):
    """What the three readings of `ARFrame.timestamp` actually disagree about.

    The streak LENGTH barely moves between them, which is why neither the
    magnitude nor the direction witness can choose one. What moves is the
    kernel's CENTRE, and by the size of the blur itself — so the choice is not
    cosmetic, it is just not answerable from the blur. `eval/frame_time.py
    window` is the instrument that answers it, and it says `centred`.
    """
    print(f"{'session':>8} {'frames':>6} {'len(lead-trail)':>16} {'centre(lead-trail)':>19}")
    for path in args.sessions:
        s = Session(path)
        poses = s.poses()
        times = np.array([p["t"] for p in poses])
        dlen, dctr = [], []
        for row in usable_frames(s):
            got = {}
            for wname in ("lead", "trail"):
                r = streaks(s, row, poses, times, window=wname, stride=args.stride)
                if r is None:
                    got = None
                    break
                p = row["pose"]
                exp, t = p["exposure"], p["t"]
                lo, hi = WINDOWS[wname]
                A = pose_at(poses, times, t + lo*exp)
                B = pose_at(poses, times, t + hi*exp)
                M = pose_at(poses, times, t)
                cloud = frame_cloud(s, row, stride=args.stride)
                if A is None or B is None or M is None or cloud is None:
                    got = None
                    break
                cam, _, _ = cloud
                pts = cam @ M[0].T + M[1]
                Kf = (p["fx"], p["fy"], p["cx"], p["cy"])
                a, b = project(*A, Kf, pts), project(*B, Kf, pts)
                got[wname] = (np.linalg.norm(b-a, axis=1), (a+b)/2)
            if got is None:
                continue
            g = np.isfinite(got["lead"][0]) & np.isfinite(got["trail"][0])
            if g.sum() < 50:
                continue
            dlen.append(float(np.median(np.abs(got["lead"][0][g] - got["trail"][0][g]))))
            dctr.append(float(np.median(np.linalg.norm(
                got["lead"][1][g] - got["trail"][1][g], axis=1))))
        name = os.path.basename(path.rstrip("/"))[-6:]
        if not dlen:
            print(f"{name:>8}      -")
            continue
        print(f"{name:>8} {len(dlen):>6} {np.median(dlen):>15.2f}p {np.median(dctr):>18.2f}p")
    print("\n  The length column is why the blur cannot choose the window;")
    print("  the centre column is why the choice still matters.")


def energy_ratio(r, theta):
    c, s = np.cos(theta), np.sin(theta)
    along = r["Jxx"]*c*c + 2*r["Jxy"]*c*s + r["Jyy"]*s*s
    across = r["Jxx"]*s*s - 2*r["Jxy"]*c*s + r["Jyy"]*c*c
    return along/across if across > 0 else np.nan


def self_test():
    """A known camera motion in, the closed-form streak demanded back.

    A camera translating by `v*dt` perpendicular to its axis smears a point at
    depth Z by `f*v*dt/Z` pixels, and one rotating by `w*dt` smears every point
    by `f*w*dt` whatever its depth. Both are exact, and the depth independence
    of the second is the part worth pinning: it is the reason the `rot` and
    `trans` columns are reported apart.
    """
    f, cx, cy = 1400.0, 960.0, 720.0
    K = (f, f, cx, cy)
    pts = np.array([[0.0, 0.0, 1.5], [0.3, -0.2, 3.0], [-0.1, 0.15, 0.8]])
    I = np.eye(3)

    dx = 0.010                                   # 10 mm sideways
    got = np.linalg.norm(project(I, np.array([dx, 0, 0]), K, pts)
                         - project(I, np.zeros(3), K, pts), axis=1)
    want = f*dx/pts[:, 2]
    assert np.allclose(got, want, rtol=2e-3), (got, want)

    ang = math.radians(0.35)                     # 0.35 deg of yaw
    R = np.array([[math.cos(ang), 0, math.sin(ang)], [0, 1, 0],
                  [-math.sin(ang), 0, math.cos(ang)]])
    got = np.linalg.norm(project(R, np.zeros(3), K, pts)
                         - project(I, np.zeros(3), K, pts), axis=1)
    assert got.std() / got.mean() < 0.02, f"rotation streak must not depend on depth: {got}"
    assert abs(got.mean() - f*ang) / (f*ang) < 0.05, (got.mean(), f*ang)

    # slerp and the axis convention: a quaternion round trip must return itself
    q = np.array([0.1, -0.2, 0.05, 0.97]); q /= np.linalg.norm(q)
    assert np.allclose(slerp(q, q, 0.37), q, atol=1e-12)
    assert np.allclose(quat_to_matrix(*q) @ quat_to_matrix(*q).T, np.eye(3), atol=1e-12)

    # the directional energy ratio: a field of purely horizontal gradients must
    # read 1 across the vertical and blow up along the horizontal
    r = {"Jxx": 4.0, "Jyy": 0.0, "Jxy": 0.0}
    assert abs(energy_ratio(r, math.pi/2)) < 1e-12
    assert not np.isfinite(energy_ratio(r, 0.0)) or energy_ratio(r, 0.0) > 1e6
    print("motion_blur self-test: ok")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    for name in ("extent", "inventory", "witness", "windows"):
        p = sub.add_parser(name)
        p.add_argument("sessions", nargs="+")
        p.add_argument("--window", default="centred", choices=list(WINDOWS))
        p.add_argument("--stride", type=int, default=8)
        p.add_argument("--limit", type=int, default=None)
        p.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if args.cmd == "extent":
        cmd_extent(args)
    elif args.cmd == "inventory":
        cmd_inventory(args)
    elif args.cmd == "witness":
        cmd_witness(args)
    elif args.cmd == "windows":
        cmd_windows(args)
    else:
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
