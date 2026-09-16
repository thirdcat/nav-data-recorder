#!/usr/bin/env python3
"""When a photograph happened, and how well the pose stream describes the inside of it.

Three questions a blur-aware trainer has to have answered before it is worth
building, all answerable from sessions already on disk:

    python3 eval/frame_time.py interp  ~/nav_data/<id>     can sub-exposure poses be supplied
    python3 eval/frame_time.py window  ~/nav_data/<id>     is the timestamp start, middle or end
    python3 eval/frame_time.py readout ~/nav_data/<id>     is there a rolling shutter

`interp` is leave-one-out: interpolate a pose from its neighbours and compare
with the one actually recorded, in pixels. The smallest span leave-one-out can
reach is two pose intervals, twice what a 16.7 ms exposure needs, so the span is
swept and read down rather than assumed — the block length is exactly the kind
of unchosen constant `HANDOVER.md` §8 keeps finding.

`window` is the one that cannot be done by looking at blur. Back-project frame
A's depth into frame B with both poses read at `t + tau`, and sweep tau. A
common tau does **not** cancel: A and B sit at different points of the
trajectory, so only the true offset puts the two photographs on top of each
other. Two controls run with it — an unrelated frame, which must show no
minimum, and `--inject`, which shifts every pose timestamp by a known amount
and demands the recovered minimum move by exactly that.

`readout` asks `window` again per image row. A global shutter returns one tau
everywhere; a rolling shutter returns a ramp whose full-frame extent is the
readout time. On the corpus as it stands this comes back **undecided** — see
`docs/3DGS.md` § *Motion blur is 6-12 px* — and the tool prints the per-band
curve depth so the reason is visible rather than inferred.
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from read_session import Session  # noqa: E402
from motion_blur import (frame_cloud, pose_at, project, quat_to_matrix,  # noqa: E402
                         slerp, usable_frames)

try:
    from PIL import Image
except ImportError:
    Image = None

TAUS = np.arange(-24, 24.1, 3.0) / 1000.0


# -- interp -----------------------------------------------------------------

def session_depth(s, every=20):
    """One median depth for the session, used to price a translation in pixels."""
    zs = []
    for d in s.depth_index()[::every]:
        z = np.asarray(s.depth_frame(d), np.float32)
        z = z[np.isfinite(z) & (z > 0.15) & (z < 8.0)]
        if z.size:
            zs.append(float(np.median(z)))
    return float(np.median(zs)) if zs else 1.5


def interp_error(poses, k, Z):
    """Median pixel error of interpolating the midpoint of a span of 2k intervals."""
    out, spans = [], []
    for i in range(k, len(poses)-k):
        a, m, b = poses[i-k], poses[i], poses[i+k]
        span = b["t"] - a["t"]
        if span <= 0 or span > 0.020*2*k*1.5:
            continue
        u = (m["t"] - a["t"]) / span
        q = slerp([a["qx"], a["qy"], a["qz"], a["qw"]],
                  [b["qx"], b["qy"], b["qz"], b["qw"]], u)
        dR = quat_to_matrix(*q).T @ quat_to_matrix(m["qx"], m["qy"], m["qz"], m["qw"])
        ang = math.acos(max(-1.0, min(1.0, (np.trace(dR)-1)/2)))
        Ci = (np.array([a["tx"], a["ty"], a["tz"]])*(1-u)
              + np.array([b["tx"], b["ty"], b["tz"]])*u)
        f = m.get("fx", 1400.0)
        out.append(ang*f + np.linalg.norm(Ci - np.array([m["tx"], m["ty"], m["tz"]]))*f/Z)
        spans.append(span)
    if not out:
        return float("nan"), float("nan")
    return float(np.median(out)), float(np.median(spans))*1000.0


def cmd_interp(args):
    ks = [1, 2, 3, 4, 6]
    print(f"{'session':>8}  " + "  ".join(f"{'k=%d' % k:>9}" for k in ks)
          + "   slope   -> 17ms")
    for path in args.sessions:
        s = Session(path)
        poses = [p for p in s.poses() if str(p.get("tracking", "")).startswith("normal")]
        if len(poses) < 40:
            print(f"{os.path.basename(path.rstrip('/'))[-6:]:>8}   too few poses")
            continue
        Z = session_depth(s)
        res = [interp_error(poses, k, Z) for k in ks]
        es = np.array([r[0] for r in res])
        sp = np.array([r[1] for r in res])
        g = np.isfinite(es) & np.isfinite(sp)
        slope, icpt = np.polyfit(np.log(sp[g]), np.log(es[g]), 1)
        at17 = math.exp(icpt + slope*math.log(16.7))
        print(f"{os.path.basename(path.rstrip('/'))[-6:]:>8}  "
              + "  ".join(f"{e:>9.2f}" for e in es)
              + f"   {slope:5.2f}   {at17:5.2f} px")
    print("\n  k = half-width in pose samples; the span is 2k intervals, so k=1 is 33 ms.")
    print("  The 17 ms column is read down the fitted slope, not measured.")


# -- window and readout -----------------------------------------------------

def load_image(s, row, downscale=4):
    if Image is None:
        raise SystemExit("window/readout need Pillow")
    with Image.open(s.frame_path(row)) as im:
        im = im.convert("RGB")
        im = im.resize((im.width//downscale, im.height//downscale), Image.BILINEAR)
        return np.asarray(im).astype(np.float32)


def _sweep(s, poses, times, rows, step, max_pairs, nband, rng, inject=0.0):
    """Residual against tau, summed over frame pairs; optionally split by source row.

    `inject` is added to every pose timestamp before the lookup, so the whole
    curve must translate by exactly that. It is the only control that pins the
    sign, which is what decides whether a negative tau means the content is
    early or the pose is.
    """
    times = times + inject
    band_acc = np.zeros((nband, len(TAUS)))
    band_cnt = np.zeros((nband, len(TAUS)))
    ctrl = np.zeros(len(TAUS))
    n = 0
    self_check = []
    stride = max(1, len(rows)//max_pairs)
    for i in range(0, len(rows)-step, stride):
        A, B = rows[i], rows[i+step]
        C = rows[int(rng.integers(0, len(rows)))]
        pa, pb = A["pose"], B["pose"]
        cloud = frame_cloud(s, A, stride=2, near=0.3, far=6.0)
        if cloud is None:
            continue
        cam, rows_v, dh = cloud
        band = np.clip(rows_v*nband//dh, 0, nband-1)
        ia, ib, ic = load_image(s, A), load_image(s, B), load_image(s, C)
        h, w = ia.shape[:2]
        sc = w / A["width"]
        ka = (pa["fx"]*sc, pa["fy"]*sc, pa["cx"]*sc, pa["cy"]*sc)
        kb = (pb["fx"]*sc, pb["fy"]*sc, pb["cx"]*sc, pb["cy"]*sc)
        # where each depth point sits in its own downscaled image
        ua = np.stack([cam[:, 0]/cam[:, 2]*ka[0] + ka[2],
                       cam[:, 1]/cam[:, 2]*ka[1] + ka[3]], 1)
        gs = ((ua[:, 0] >= 0) & (ua[:, 0] < w-1) & (ua[:, 1] >= 0) & (ua[:, 1] < h-1))
        if gs.sum() < 300:
            continue
        src = ia[ua[gs, 1].astype(int), ua[gs, 0].astype(int)]
        cam_g, band_g = cam[gs], band[gs]
        for j, tau in enumerate(TAUS):
            # the lookup time is NOT shifted: `times` already carries the
            # injection, and shifting both would cancel it silently
            PA = pose_at(poses, times, pa["t"]+tau)
            PB = pose_at(poses, times, pb["t"]+tau)
            PC = pose_at(poses, times, C["pose"]["t"]+tau)
            if PA is None or PB is None or PC is None:
                continue
            pts = cam_g @ PA[0].T + PA[1]
            uv = project(PB[0], PB[1], kb, pts)
            m = np.isfinite(uv).all(1)
            m &= (uv[:, 0] >= 0) & (uv[:, 0] < w-1) & (uv[:, 1] >= 0) & (uv[:, 1] < h-1)
            if m.sum() >= 100:
                err = np.mean(np.abs(ib[uv[m, 1].astype(int), uv[m, 0].astype(int)] - src[m]), axis=1)
                bm = band_g[m]
                for b in range(nband):
                    q = bm == b
                    if q.sum() >= 20:
                        band_acc[b, j] += float(err[q].mean())
                        band_cnt[b, j] += 1
            uvc = project(PC[0], PC[1], kb, pts)
            mc = np.isfinite(uvc).all(1)
            mc &= (uvc[:, 0] >= 0) & (uvc[:, 0] < w-1) & (uvc[:, 1] >= 0) & (uvc[:, 1] < h-1)
            if mc.sum() >= 100:
                ctrl[j] += float(np.mean(np.abs(
                    ic[uvc[mc, 1].astype(int), uvc[mc, 0].astype(int)] - src[mc])))
        # self-check: project straight back into the frame the colour came from
        P0 = pose_at(poses, times, pa["t"])
        if P0 is not None:
            uv0 = project(P0[0], P0[1], ka, cam_g @ P0[0].T + P0[1])
            m0 = np.isfinite(uv0).all(1)
            m0 &= (uv0[:, 0] >= 0) & (uv0[:, 0] < w-1) & (uv0[:, 1] >= 0) & (uv0[:, 1] < h-1)
            if m0.sum() >= 100:
                self_check.append(float(np.mean(np.abs(
                    ia[uv0[m0, 1].astype(int), uv0[m0, 0].astype(int)] - src[m0]))))
        n += 1
    return band_acc, band_cnt, ctrl/max(n, 1), n, (
        float(np.median(self_check)) if self_check else float("nan"))


def peak(curve):
    """Sub-step minimum, by the parabola through it and its two neighbours."""
    if not np.isfinite(curve).any():
        return float("nan")
    i = int(np.nanargmin(curve))
    t = TAUS[i]*1000
    if 0 < i < len(curve)-1:
        y0, y1, y2 = curve[i-1], curve[i], curve[i+1]
        d = y0 - 2*y1 + y2
        if d:
            t = (TAUS[i] - 0.5*(y2-y0)/d*(TAUS[1]-TAUS[0]))*1000
    return float(t)


def cmd_window(args):
    rng = np.random.default_rng(args.seed)
    for path in args.sessions:
        s = Session(path)
        poses = s.poses()
        times = np.array([p["t"] for p in poses])
        rows = usable_frames(s)
        acc, cnt, ctrl, n, sc = _sweep(s, poses, times, rows, args.step,
                                       args.max_pairs, 1, rng)
        name = os.path.basename(path.rstrip("/"))[-6:]
        if n == 0:
            print(f"{name}: no pairs")
            continue
        c = np.divide(acc[0], cnt[0], out=np.full(len(TAUS), np.nan), where=cnt[0] > 0)
        print(f"{name} n={n} self-check {sc:.2f}/255   tau = {peak(c):+.1f} ms")
        print("   true " + " ".join(f"{t*1000:+.0f}:{v:.2f}" for t, v in zip(TAUS, c)))
        print("   ctrl " + " ".join(f"{t*1000:+.0f}:{v:.2f}" for t, v in zip(TAUS, ctrl)))
        if args.inject:
            a2, c2, _, _, _ = _sweep(s, poses, times, rows, args.step,
                                     args.max_pairs, 1, np.random.default_rng(args.seed),
                                     inject=args.inject)
            cc = np.divide(a2[0], c2[0], out=np.full(len(TAUS), np.nan), where=c2[0] > 0)
            moved = peak(cc) - peak(c)
            print(f"   inject {args.inject*1000:+.1f} ms -> tau {peak(cc):+.1f}, "
                  f"moved {moved:+.1f}")
        print("   candidates: centred 0.0, lead -exposure/2, trail +exposure/2 ms")


def cmd_readout(args):
    rng = np.random.default_rng(args.seed)
    for path in args.sessions:
        s = Session(path)
        poses = s.poses()
        times = np.array([p["t"] for p in poses])
        rows = usable_frames(s)
        acc, cnt, _, n, _ = _sweep(s, poses, times, rows, args.step,
                                   args.max_pairs, args.bands, rng)
        c = np.divide(acc, cnt, out=np.full_like(acc, np.nan), where=cnt > 0)
        ts = [peak(c[b]) for b in range(args.bands)]
        depths = [float(np.nanmax(c[b]) - np.nanmin(c[b])) for b in range(args.bands)]
        span = ts[-1] - ts[0]
        full = abs(span)*args.bands/(args.bands-1)
        print(f"{os.path.basename(path.rstrip('/'))[-6:]:>8}  tau top->bottom: "
              + "  ".join(f"{t:+.2f}" for t in ts)
              + f"  ms  | full-frame readout {full:.1f} ms"
              + "  | curve depth " + " ".join(f"{d:.2f}" for d in depths))
    print("\n  A flat row is a global shutter; a monotone ramp is a rolling one.")
    print("  A shallow curve depth means that band's minimum is not to be trusted.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd")
    for name in ("interp", "window", "readout"):
        p = sub.add_parser(name)
        p.add_argument("sessions", nargs="+")
        p.add_argument("--step", type=int, default=3, help="frames between the pair")
        p.add_argument("--max-pairs", type=int, default=90)
        p.add_argument("--bands", type=int, default=4)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--inject", type=float, default=0.0,
                       help="seconds added to every pose timestamp, as a sign control")
    args = ap.parse_args(argv)
    if args.cmd == "interp":
        cmd_interp(args)
    elif args.cmd == "window":
        cmd_window(args)
    elif args.cmd == "readout":
        cmd_readout(args)
    else:
        ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
