#!/usr/bin/env python3
"""What focal length does the wide arm of a dual-lens capture actually have?

The recorder writes the ultra-wide's active format FOV into the manifest
("ultra-wide fov 106.2007") and writes nothing for the wide arm. So code reading
the wide arm has to fall back to scaling the factory calibration by the image
size — which is the exact mistake `intrinsics_for()` exists to avoid on the
ultra-wide side, where the active format turned out not to be the calibrated one.
Nobody has checked whether 640x480 is the calibrated 4:3 readout scaled down or a
different format entirely.

The two lenses shoot the same scene at the same instant, 19.272 mm apart. At room
distances that baseline is small, so for far content the wide frame is close to a
scaled central crop of the ultra-wide frame, and a similarity fit between the two
recovers

    s = r_wide / r_ultrawide = f_wide / f_ultrawide

with f_ultrawide known from the logged format. That gives f_wide in pixels, and
from it the wide arm's true horizontal field of view.

Two things make a naive fit lie, and both are answered here rather than assumed.
Both lenses are **raw** — `geometric_distortion_correction: false` — so a fit
spanning a wide radius is biased by barrel distortion; the fit is therefore run
twice, over two central radii, and disagreement between them is reported as a
bias warning instead of being averaged away. And an instrument that cannot
recover a scale it was handed proves nothing about a scale it was asked for, so
the same pipeline is first run on an ultra-wide frame against itself downscaled
by known factors. The decision rule was written to a file before any fit ran.

numpy and cv2 only: no GPU, no torch, no weights. It reads sessions and nothing
else, and writes only what --out asks for.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

# The session format is defined in tools/, and eval/ is allowed to import
# from tools/ — never the other way round. Re-deriving the depth blob's
# layout here would be a second definition of it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from read_session import Session  # noqa: E402

# The logged active format of the ultra-wide, and the factory calibration of the
# wide, both from the capture side. These are inputs, not results.
UW_HFOV_DEG = 106.2007          # manifest note, the format the recorder pinned
UW_WIDTH = 3840.0
WIDE_CAL_FX = 2745.4            # calib/iphone17-1_wide.json, at 4032x3024
WIDE_CAL_WIDTH = 4032.0

PRESCALE = 0.30                 # a round number, not the hypothesis: the
                                # ultra-wide is shrunk by this before matching so
                                # both images carry features at similar scales,
                                # and the fitted scale is multiplied back by it
RATIO = 0.75                    # Lowe
MIN_INLIERS = 25                # pre-registered floor; below it the pair is a
                                # reported failure, never a silent drop
MAX_DT_MS = 40.0
RANSAC_PX = 3.0


def f_ultrawide() -> float:
    return (UW_WIDTH / 2.0) / math.tan(math.radians(UW_HFOV_DEG / 2.0))


def hfov_from_scale(s: float, wide_width: float) -> float:
    """Implied wide h-FOV in degrees for a measured match scale."""
    f_w = s * f_ultrawide()
    return 2.0 * math.degrees(math.atan((wide_width / 2.0) / f_w))


def rows(path: Path) -> list:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def pair_frames(session: Path, count: int, min_separation: int = 8) -> list:
    """The `count` tightest-in-time pairs, kept apart along the walk.

    The first version of this took evenly spaced frames and then found each one's
    nearest wide frame, which produced a median offset of 60 ms and threw away
    four fifths of the session on the 40 ms cap. That was backwards: the offset
    is what decides whether a pair is usable at all, so the pairs are chosen *by*
    it. `min_separation` keeps the survivors from clustering into one moment of
    the walk, since five views of one wall is one measurement, not five.

    Why the offset matters more than it looks: the phone is walking, so between
    the two exposures it advances. Forward motion scales the image, which is the
    very quantity being measured — 60 ms at 1 m/s is 6 cm, and 6 cm at 2 m is a
    3 % scale change, the same size as the effect under test.
    """
    uw = rows(session / "frames.jsonl")
    wide = rows(session / "frames_wide.jsonl")
    wt = np.array([r["t"] for r in wide])
    cand = []
    for i, u in enumerate(uw):
        j = int(np.argmin(np.abs(wt - u["t"])))
        cand.append((abs((wide[j]["t"] - u["t"]) * 1000.0), i, j))
    cand.sort()
    picked, out = [], []
    for adt, i, j in cand:
        if adt > MAX_DT_MS or len(out) >= count:
            break
        if any(abs(i - k) < min_separation for k in picked):
            continue
        picked.append(i)
        out.append((uw[i], wide[j], (wide[j]["t"] - uw[i]["t"]) * 1000.0))
    return sorted(out, key=lambda r: r[0]["frame"])


def fit_scale(img_uw, img_wide, radius_px, prescale=PRESCALE):
    """Similarity scale taking ultra-wide pixels to wide pixels.

    Matches are kept only where the ultra-wide point lies within `radius_px` of
    the ultra-wide image centre, measured in the *original* ultra-wide pixels, so
    the restriction means the same angle regardless of the prescale.
    """
    small = cv2.resize(img_uw, None, fx=prescale, fy=prescale,
                       interpolation=cv2.INTER_AREA)
    sift = cv2.SIFT_create()
    k1, d1 = sift.detectAndCompute(small, None)
    k2, d2 = sift.detectAndCompute(img_wide, None)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return {"ok": False, "why": "too few features",
                "matches": 0, "inliers": 0}

    pairs = cv2.BFMatcher().knnMatch(d1, d2, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2)
            if m.distance < RATIO * n.distance]

    cx, cy = small.shape[1] / 2.0, small.shape[0] / 2.0
    src, dst = [], []
    for m in good:
        p = k1[m.queryIdx].pt
        if math.hypot(p[0] - cx, p[1] - cy) / prescale > radius_px:
            continue
        src.append(p)
        dst.append(k2[m.trainIdx].pt)
    if len(src) < MIN_INLIERS:
        return {"ok": False, "why": f"only {len(src)} central matches",
                "matches": len(src), "inliers": 0}

    M, mask = cv2.estimateAffinePartial2D(
        np.float32(src).reshape(-1, 1, 2), np.float32(dst).reshape(-1, 1, 2),
        method=cv2.RANSAC, ransacReprojThreshold=RANSAC_PX)
    if M is None:
        return {"ok": False, "why": "no similarity found",
                "matches": len(src), "inliers": 0}
    inliers = int(mask.sum())
    if inliers < MIN_INLIERS:
        return {"ok": False, "why": f"only {inliers} inliers",
                "matches": len(src), "inliers": inliers}

    scale = math.hypot(M[0, 0], M[0, 1]) * prescale
    rot = math.degrees(math.atan2(M[1, 0], M[0, 0]))
    return {"ok": True, "scale": scale, "rot_deg": rot,
            "matches": len(src), "inliers": inliers}


def synthetic_control(session: Path, factors, radius_px):
    """Hand the pipeline a scale it already knows and see if it comes back.

    Two factors, one of them away from the hypothesis, so a pipeline that somehow
    favours 0.3023 cannot pass this by accident.
    """
    uw = rows(session / "frames.jsonl")
    img = cv2.imread(str(session / uw[len(uw) // 2]["file"]), cv2.IMREAD_GRAYSCALE)
    out = []
    for k in factors:
        shrunk = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        r = fit_scale(img, shrunk, radius_px)
        r["truth"] = k
        r["error_pct"] = (100.0 * (r["scale"] - k) / k) if r["ok"] else float("nan")
        out.append(r)
    return out


def radius_sweep(sessions, pairs_per, separation, radii):
    """Is the answer distortion? Fit the same pairs over shrinking radii.

    Both lenses are raw, and barrel distortion pulls the periphery inward, which
    inflates a scale measured as r_wide / r_ultrawide. If that is what produces
    the discrepancy, the fitted scale must fall as the fit is confined nearer the
    centre, and extrapolate to the calibrated value at radius zero. If it flattens
    out somewhere else, distortion is not the explanation. This is the difference
    between arguing about distortion and measuring it.
    """
    per_pair = {}
    for sess in sessions:
        for uw, wide, dt_ms in pair_frames(sess, pairs_per, separation):
            img_uw = cv2.imread(str(sess / uw["file"]), cv2.IMREAD_GRAYSCALE)
            img_w = cv2.imread(str(sess / wide["file"]), cv2.IMREAD_GRAYSCALE)
            if img_uw is None or img_w is None:
                continue
            got = {}
            for r in radii:
                fit = fit_scale(img_uw, img_w, r)
                if fit["ok"]:
                    got[r] = fit["scale"]
            per_pair[(sess.name[-6:], uw["frame"])] = got
    return per_pair


# ---------------------------------------------------------------------------
# Stage two: parallax, which breaks the ratio's degeneracy.
#
# s = f_w / f_uw is a ratio, so it cannot say which of the two focal lengths is
# wrong. The baseline can: it was measured at the factory in millimetres and the
# LiDAR reports metres, so the disparity between the two lenses at a known depth
# fixes an absolute focal length. After the best similarity is subtracted, what
# is left is proportional to 1/Z with slope -f_w * t_x.
# ---------------------------------------------------------------------------

T_X_M = -19.233642578125e-3        # calib, ultra-wide relative to wide, metres
T_Z_M = -1.209808349609375e-3
WIDE_HFOV_H0 = 72.581              # implied by the calibration at 640 px wide
DEPTH_HFOV_DOC = 74.6              # what calib/README.md says the LiDAR reports


def depth_index(session: Path) -> dict:
    return {round(r["t"], 6): r for r in rows(session / "depth.jsonl")}


def sample_depth(depth, u, v, img_w, img_h, radius_ratio=1.0,
                 patch=1, edge_tol=0.05):
    """Depth at a wide-image pixel, nearest neighbour, holes rejected.

    `radius_ratio` is how much of the image's normalised radius one unit of the
    depth grid's normalised radius covers. It is 1.0 when the grid spans exactly
    the image's field, and tan(36.3)/tan(37.3) = 0.9632 if the grid really spans
    the 74.6 degrees the documentation claims while the image spans 72.6.
    """
    dh, dw = depth.shape
    un = (u - img_w / 2.0) / (img_w / 2.0) * radius_ratio
    vn = (v - img_h / 2.0) / (img_h / 2.0) * radius_ratio
    du = int(round(dw / 2.0 + un * dw / 2.0))
    dv = int(round(dh / 2.0 + vn * dh / 2.0))
    if not (0 <= du < dw and 0 <= dv < dh):
        return None
    if patch <= 1:
        z = float(depth[dv, du])
        if not np.isfinite(z) or z < 0.15 or z > 8.0:
            return None
        return z
    # A regressor measured with noise drags a slope toward zero, and 1/Z is the
    # regressor here: a match that lands on a depth edge is assigned the wrong
    # surface entirely. So take the median of a small patch and refuse the point
    # when the patch does not agree with itself.
    r = patch // 2
    win = depth[max(dv - r, 0):dv + r + 1, max(du - r, 0):du + r + 1]
    win = win[np.isfinite(win) & (win > 0.15) & (win < 8.0)]
    if win.size < max(3, patch):
        return None
    z = float(np.median(win))
    if float(win.std()) / max(z, 1e-6) > edge_tol:
        return None
    return z


def parallax_pair(img_uw, img_w, depth, radius_px, radius_ratio=1.0,
                  prescale=PRESCALE, shuffle_depth=False):
    """Recover f_w in pixels from one pair, by regressing residual on 1/Z."""
    small = cv2.resize(img_uw, None, fx=prescale, fy=prescale,
                       interpolation=cv2.INTER_AREA)
    sift = cv2.SIFT_create()
    k1, d1 = sift.detectAndCompute(small, None)
    k2, d2 = sift.detectAndCompute(img_w, None)
    if d1 is None or d2 is None:
        return {"ok": False, "why": "no features"}
    pairs = cv2.BFMatcher().knnMatch(d1, d2, k=2)
    good = [m for m, n in (q for q in pairs if len(q) == 2)
            if m.distance < RATIO * n.distance]
    cx, cy = small.shape[1] / 2.0, small.shape[0] / 2.0
    src, dst = [], []
    for m in good:
        p = k1[m.queryIdx].pt
        if math.hypot(p[0] - cx, p[1] - cy) / prescale > radius_px:
            continue
        src.append(p)
        dst.append(k2[m.trainIdx].pt)
    if len(src) < MIN_INLIERS:
        return {"ok": False, "why": f"only {len(src)} central matches"}

    src_a = np.float32(src).reshape(-1, 1, 2)
    dst_a = np.float32(dst).reshape(-1, 1, 2)
    M, mask = cv2.estimateAffinePartial2D(src_a, dst_a, method=cv2.RANSAC,
                                          ransacReprojThreshold=RANSAC_PX)
    if M is None:
        return {"ok": False, "why": "no similarity"}
    keep = mask.ravel().astype(bool)
    if keep.sum() < MIN_INLIERS:
        return {"ok": False, "why": f"only {int(keep.sum())} inliers"}

    pred = cv2.transform(src_a, M).reshape(-1, 2)[keep]
    obs = dst_a.reshape(-1, 2)[keep]
    resid_u = obs[:, 0] - pred[:, 0]

    h, w = img_w.shape[:2]
    inv_z, res = [], []
    for (u, v), r in zip(obs, resid_u):
        z = sample_depth(depth, u, v, w, h, radius_ratio)
        if z is None:
            continue
        inv_z.append(1.0 / z)
        res.append(float(r))
    if len(inv_z) < MIN_INLIERS:
        return {"ok": False, "why": f"only {len(inv_z)} points with depth"}
    inv_z = np.array(inv_z)
    res = np.array(res)
    span = float(inv_z.max() - inv_z.min())
    if span < 0.6:
        return {"ok": False, "why": f"1/Z span only {span:.2f}, no leverage"}
    if shuffle_depth:                      # permutation null
        inv_z = np.random.default_rng(0).permutation(inv_z)

    A = np.stack([inv_z, np.ones_like(inv_z)], 1)
    coef, *_ = np.linalg.lstsq(A, res, rcond=None)
    slope, intercept = float(coef[0]), float(coef[1])
    fit = A @ coef
    dof = max(len(res) - 2, 1)
    sigma2 = float(((res - fit) ** 2).sum() / dof)
    var = sigma2 * float(np.linalg.inv(A.T @ A)[0, 0])
    se = math.sqrt(max(var, 0.0))
    scale = math.hypot(M[0, 0], M[0, 1]) * prescale
    return {"ok": True, "slope": slope, "se": se, "intercept": intercept,
            "n": len(res), "span": span, "scale": scale,
            "f_w": slope / abs(T_X_M), "inliers": int(keep.sum())}


def synth_wide(img_uw, depth, f_uw, f_w, t_x, out_size=(640, 480)):
    """Build a wide frame from the ultra-wide, at a chosen focal pair.

    The synthetic frame is what the wide camera *would* have seen if the two
    focal lengths were the injected ones. Handing that to the same pipeline is
    the only way to know the pipeline can tell 1441.5 from 1372.2 at all.
    """
    w, h = out_size
    # Holes are filled and the grid is upsampled smoothly before warping. A
    # speckled synthetic frame is a bad control for the wrong reason: it fails on
    # feature count rather than on geometry, which says nothing about whether the
    # geometry is right.
    d = depth.astype(np.float32).copy()
    bad_d = ~np.isfinite(d) | (d < 0.15) | (d > 8.0)
    if bad_d.any():
        d[bad_d] = 0.0
        d = cv2.inpaint((d * 30.0).clip(0, 255).astype(np.uint8),
                        bad_d.astype(np.uint8), 3, cv2.INPAINT_TELEA
                        ).astype(np.float32) / 30.0
    z = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    x = (uu - w / 2.0) / f_w * z
    y = (vv - h / 2.0) / f_w * z
    zu = z + T_Z_M
    map_x = (f_uw * (x + t_x) / np.maximum(zu, 1e-6)
             + img_uw.shape[1] / 2.0).astype(np.float32)
    map_y = (f_uw * y / np.maximum(zu, 1e-6)
             + img_uw.shape[0] / 2.0).astype(np.float32)
    bad = ~np.isfinite(z) | (z < 0.15) | (z > 8.0)
    map_x[bad] = -1
    map_y[bad] = -1
    return cv2.remap(img_uw, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def joint_pair(img_uw, img_w, depth, radius_px, radius_ratio=1.0,
               prescale=PRESCALE, patch=1, edge_tol=0.05):
    """Fit the similarity and the parallax term together, and say if it is identifiable.

    Subtracting a similarity first and regressing the leftovers on 1/Z looked
    right and is not: for a planar surface the disparity -f_w t_x / Z is itself
    affine in image coordinates, so the similarity absorbs it and the leftovers
    are noise. These are indoor walls and floors, so that is the normal case, not
    a corner case. Fitting

        u_w = A + B u_uw + C v_uw + G (1/Z),      G = -f_w t_x

    keeps both terms in one design, and the variance inflation factor of the 1/Z
    column then says whether the scene separated them at all. A VIF of 50 means
    the answer is 7x noisier than the point count suggests; a VIF in the hundreds
    means the scene was a plane and no amount of averaging will help.
    """
    small = cv2.resize(img_uw, None, fx=prescale, fy=prescale,
                       interpolation=cv2.INTER_AREA)
    sift = cv2.SIFT_create()
    k1, d1 = sift.detectAndCompute(small, None)
    k2, d2 = sift.detectAndCompute(img_w, None)
    if d1 is None or d2 is None:
        return {"ok": False, "why": "no features"}
    knn = cv2.BFMatcher().knnMatch(d1, d2, k=2)
    good = [m for m, n in (q for q in knn if len(q) == 2)
            if m.distance < RATIO * n.distance]
    cx, cy = small.shape[1] / 2.0, small.shape[0] / 2.0
    src, dst = [], []
    for m in good:
        p = k1[m.queryIdx].pt
        if math.hypot(p[0] - cx, p[1] - cy) / prescale > radius_px:
            continue
        src.append((p[0] / prescale, p[1] / prescale))
        dst.append(k2[m.trainIdx].pt)
    if len(src) < MIN_INLIERS:
        return {"ok": False, "why": f"only {len(src)} central matches"}

    M, mask = cv2.estimateAffinePartial2D(
        np.float32([(u * prescale, v * prescale) for u, v in src]).reshape(-1, 1, 2),
        np.float32(dst).reshape(-1, 1, 2),
        method=cv2.RANSAC, ransacReprojThreshold=RANSAC_PX)
    if M is None:
        return {"ok": False, "why": "no similarity"}
    keep = mask.ravel().astype(bool)
    scale = math.hypot(M[0, 0], M[0, 1]) * prescale

    h, w = img_w.shape[:2]
    U, V, Uw, INV = [], [], [], []
    for (su, sv), (du, dv), k in zip(src, dst, keep):
        if not k:
            continue
        z = sample_depth(depth, du, dv, w, h, radius_ratio, patch, edge_tol)
        if z is None:
            continue
        U.append(su); V.append(sv); Uw.append(du); INV.append(1.0 / z)
    if len(U) < MIN_INLIERS:
        return {"ok": False, "why": f"only {len(U)} points with depth"}

    A = np.stack([np.ones(len(U)), np.array(U), np.array(V), np.array(INV)], 1)
    y = np.array(Uw)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    dof = max(len(y) - 4, 1)
    sigma2 = float((resid ** 2).sum() / dof)
    cov = sigma2 * np.linalg.pinv(A.T @ A)
    g, g_se = float(coef[3]), math.sqrt(max(float(cov[3, 3]), 0.0))

    # VIF of the 1/Z column against the similarity's own columns.
    X = A[:, :3]
    beta, *_ = np.linalg.lstsq(X, A[:, 3], rcond=None)
    r = A[:, 3] - X @ beta
    ss_tot = float(((A[:, 3] - A[:, 3].mean()) ** 2).sum())
    ss_res = float((r ** 2).sum())
    vif = float("inf") if ss_res <= 0 else ss_tot / ss_res
    return {"ok": True, "g": g, "g_se": g_se, "vif": vif, "n": len(y),
            "scale": scale, "rms_px": math.sqrt(sigma2),
            "f_w": abs(g) / abs(T_X_M), "span": float(max(INV) - min(INV))}


def run_parallax(sessions, args, s_med):
    """Stage two, with its controls run before its result is believed."""
    ratio = 1.0
    if args.depth_hfov:
        ratio = (math.tan(math.radians(WIDE_HFOV_H0 / 2.0))
                 / math.tan(math.radians(args.depth_hfov / 2.0)))
    print(f"\n--- parallax: absolute focal from the {abs(T_X_M)*1e3:.3f} mm "
          f"baseline (depth grid radius ratio {ratio:.4f})")
    f_uw_a = f_ultrawide()
    f_w_a = s_med * f_uw_a
    f_w_b = WIDE_CAL_FX * 640.0 / WIDE_CAL_WIDTH
    f_uw_b = f_w_b / s_med
    print(f"  (a) log stands: f_uw {f_uw_a:.1f}, f_w {f_w_a:.1f}, "
          f"slope {abs(T_X_M)*f_w_a:.3f} px per 1/m")
    print(f"  (b) log is low: f_uw {f_uw_b:.1f}, f_w {f_w_b:.1f}, "
          f"slope {abs(T_X_M)*f_w_b:.3f} px per 1/m")

    # Gather the pairs that have an exact-timestamp depth frame.
    work = []
    for sess in sessions:
        didx = depth_index(sess)
        reader = Session(str(sess))
        for uw, wide, dt_ms in pair_frames(sess, args.pairs, args.separation):
            entry = didx.get(round(wide["t"], 6))
            if entry is None:
                continue
            work.append((sess, reader, uw, wide, dt_ms, entry))
    print(f"  {len(work)} pairs have a depth frame at the wide frame's exact "
          f"timestamp")
    if not work:
        print("  no pair to work with")
        return

    # Control 1: can the pipeline tell the two hypotheses apart at all?
    sess, reader, uw, wide, _dt, entry = work[0]
    img_uw = cv2.imread(str(sess / uw["file"]), cv2.IMREAD_GRAYSCALE)
    depth = np.asarray(reader.depth_frame(entry), dtype=np.float32)
    print("  control 1 — inject a known focal pair, ask for it back")
    inject_ok = []
    for label, f_uw_t, f_w_t in (("a", f_uw_a, f_w_a), ("b", f_uw_b, f_w_b)):
        fake = synth_wide(img_uw, depth, f_uw_t, f_w_t, T_X_M)
        got = parallax_pair(img_uw, fake, depth, args.radius, ratio)
        if not got["ok"]:
            print(f"    inject {label}: FAILED — {got['why']}")
            inject_ok.append(False)
            continue
        f_uw_hat = got["f_w"] / got["scale"]
        err = 100.0 * (f_uw_hat - f_uw_t) / f_uw_t
        print(f"    inject {label}: f_uw {f_uw_t:7.1f} -> {f_uw_hat:7.1f} "
              f"({err:+.1f} %)  f_w {got['f_w']:6.1f}  n {got['n']}")
        inject_ok.append(abs(err) <= 2.0)

    # Control 2: with no baseline there must be no slope.
    flat = synth_wide(img_uw, depth, f_uw_a, f_w_a, 0.0)
    null = parallax_pair(img_uw, flat, depth, args.radius, ratio)
    if null["ok"]:
        z = abs(null["slope"]) / max(null["se"], 1e-9)
        print(f"    zero baseline: slope {null['slope']:+.3f} +/- {null['se']:.3f}"
              f"  ({z:.1f} sigma)  — must be consistent with zero")
        null_ok = z < 3.0
    else:
        print(f"    zero baseline: FAILED — {null['why']}")
        null_ok = False

    # The real pairs.
    print(f"\n  {'session':<8}{'frame':>7}{'n':>5}{'1/Z span':>10}{'slope':>9}"
          f"{'+/-':>7}{'f_w':>8}{'f_uw':>9}")
    got_rows, fails, shuffled = [], [], []
    for sess, reader, uw, wide, dt_ms, entry in work:
        img_uw = cv2.imread(str(sess / uw["file"]), cv2.IMREAD_GRAYSCALE)
        img_w = cv2.imread(str(sess / wide["file"]), cv2.IMREAD_GRAYSCALE)
        depth = np.asarray(reader.depth_frame(entry), dtype=np.float32)
        r = parallax_pair(img_uw, img_w, depth, args.radius, ratio)
        tag = sess.name[-6:]
        if not r["ok"]:
            fails.append((tag, uw["frame"], r["why"]))
            continue
        f_uw_hat = r["f_w"] / r["scale"]
        got_rows.append({"session": tag, "frame": uw["frame"], **r,
                         "f_uw": f_uw_hat})
        print(f"  {tag:<8}{uw['frame']:>7}{r['n']:>5}{r['span']:>10.2f}"
              f"{r['slope']:>9.3f}{r['se']:>7.3f}{r['f_w']:>8.1f}{f_uw_hat:>9.1f}")
        sh = parallax_pair(img_uw, img_w, depth, args.radius, ratio,
                           shuffle_depth=True)
        if sh["ok"]:
            shuffled.append(abs(sh["slope"]) / max(sh["se"], 1e-9))
    for tag, frame, why in fails:
        print(f"    failure: {tag} frame {frame}: {why}")
    if not got_rows:
        print("  no pair produced a parallax fit")
        return

    slopes = np.array([r["slope"] for r in got_rows])
    f_uws = np.array([r["f_uw"] for r in got_rows])
    med_slope = float(np.median(slopes))
    med_f_uw = float(np.median(f_uws))
    rse = float(np.std(f_uws, ddof=1) / math.sqrt(len(f_uws)) / med_f_uw * 100.0)
    print(f"\n  {len(got_rows)} pairs, {len(fails)} failed")
    print(f"  slope median {med_slope:.3f} px per 1/m   "
          f"f_w {med_slope/abs(T_X_M):.1f}   f_uw {med_f_uw:.1f}")
    print(f"  f_uw spread {f_uws.min():.0f}..{f_uws.max():.0f}, "
          f"relative SE of the median {rse:.1f} %")
    if shuffled:
        print(f"  control 3 — depth shuffled: median {np.median(shuffled):.1f} "
              f"sigma (must be small)")

    # Secondary and NOT pre-registered: the joint fit, run because the
    # pre-registered one failed and the reason mattered more than the number.
    print("\n  secondary (not pre-registered): similarity and parallax fitted "
          "together, with the identifiability of the 1/Z column")
    print(f"  {'session':<8}{'frame':>7}{'n':>5}{'VIF':>9}{'g':>9}{'+/-':>8}"
          f"{'f_w':>8}{'rms px':>8}")
    joint = []
    for sess, reader, uw, wide, dt_ms, entry in work:
        img_uw = cv2.imread(str(sess / uw["file"]), cv2.IMREAD_GRAYSCALE)
        img_w = cv2.imread(str(sess / wide["file"]), cv2.IMREAD_GRAYSCALE)
        depth = np.asarray(reader.depth_frame(entry), dtype=np.float32)
        j = joint_pair(img_uw, img_w, depth, args.radius, ratio)
        if not j["ok"]:
            continue
        joint.append(j)
        print(f"  {sess.name[-6:]:<8}{uw['frame']:>7}{j['n']:>5}{j['vif']:>9.1f}"
              f"{j['g']:>9.3f}{j['g_se']:>8.3f}{j['f_w']:>8.1f}{j['rms_px']:>8.2f}")
    if joint:
        vifs = np.array([j["vif"] for j in joint])
        usable = [j for j in joint if j["vif"] < 20]
        print(f"  VIF median {np.median(vifs):.1f}, min {vifs.min():.1f}; "
              f"{len(usable)} of {len(joint)} pairs below 20")
        if usable:
            gg = np.array([j["g"] for j in usable])
            print(f"  g median {np.median(gg):+.3f} px per 1/m  -> f_w "
                  f"{abs(np.median(gg))/abs(T_X_M):.0f} px "
                  f"(a wants {abs(T_X_M)*f_w_a*1:.2f}/{f_w_a:.0f}, "
                  f"b wants {abs(T_X_M)*f_w_b:.2f}/{f_w_b:.0f})")

    ok = all(inject_ok) and null_ok
    if not ok:
        verdict = "VOID — a control failed, so no focal length is reported"
    elif rse > 2.5:
        verdict = f"UNDECIDED — relative SE {rse:.1f} % exceeds the 2.5 % floor"
    elif 1412 <= med_f_uw <= 1471:
        verdict = ("(a) the logged 106.2007 stands; the discrepancy is on the "
                   "wide side")
    elif 1345 <= med_f_uw <= 1400:
        verdict = ("(b) the logged ultra-wide FOV is low; the wide arm is the "
                   "calibrated format")
    else:
        verdict = (f"UNDECIDED — f_uw {med_f_uw:.0f} falls in neither band")
    print(f"  PARALLAX VERDICT: {verdict}")


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="+", help="session directories")
    ap.add_argument("--pairs", type=int, default=5, help="pairs per session")
    ap.add_argument("--radius", type=float, default=600.0,
                    help="central radius in ultra-wide px for the primary fit")
    ap.add_argument("--radius-check", type=float, default=400.0,
                    help="second, tighter radius; disagreement means distortion")
    ap.add_argument("--separation", type=int, default=8,
                    help="minimum ultra-wide frames between chosen pairs")
    ap.add_argument("--parallax", action="store_true",
                    help="stage two: recover absolute focal lengths from the "
                         "metric baseline and LiDAR depth")
    ap.add_argument("--depth-hfov", type=float, default=None,
                    help="assume the depth grid spans this h-FOV while the image "
                         "spans %.3f (default: the same field)" % WIDE_HFOV_H0)
    ap.add_argument("--sweep", action="store_true",
                    help="also fit over a ladder of radii, to see whether the "
                         "answer is lens distortion")
    ap.add_argument("--out")
    args = ap.parse_args(argv)

    f_uw = f_ultrawide()
    s_h0 = (WIDE_CAL_FX * 640.0 / WIDE_CAL_WIDTH) / f_uw
    print(f"f_ultrawide {f_uw:.2f} px at {UW_HFOV_DEG:.4f} deg across {UW_WIDTH:.0f}")
    print(f"H0: wide is the calibrated readout downscaled -> s = {s_h0:.4f}, "
          f"{hfov_from_scale(s_h0, 640.0):.2f} deg\n")

    sessions = [Path(s) for s in args.sessions]

    print("synthetic control — the same pipeline on a known downscale")
    control = synthetic_control(sessions[0], (0.3023, 0.25), args.radius)
    for c in control:
        if c["ok"]:
            print(f"  truth {c['truth']:.4f}  recovered {c['scale']:.4f}  "
                  f"error {c['error_pct']:+.2f} %  inliers {c['inliers']}")
        else:
            print(f"  truth {c['truth']:.4f}  FAILED: {c['why']}")
    control_ok = all(c["ok"] and abs(c["error_pct"]) <= 1.0 for c in control)
    print(f"  control {'passes' if control_ok else 'FAILS'} "
          f"(needs every factor within 1 %)\n")

    print(f"{'session':<8}{'uw frame':>9}{'dt ms':>8}{'inliers':>9}"
          f"{'s @' + str(int(args.radius)):>10}{'s @' + str(int(args.radius_check)):>10}"
          f"{'hFOV deg':>10}{'rot deg':>9}")
    measured, failures, results = [], [], []
    for sess in sessions:
        for uw, wide, dt_ms in pair_frames(sess, args.pairs, args.separation):
            tag = sess.name[-6:]
            if abs(dt_ms) > MAX_DT_MS:
                failures.append((tag, uw["frame"], f"|dt| {dt_ms:.0f} ms"))
                print(f"{tag:<8}{uw['frame']:>9}{dt_ms:>8.0f}"
                      f"{'--':>9}{'skipped, dt':>31}")
                continue
            img_uw = cv2.imread(str(sess / uw["file"]), cv2.IMREAD_GRAYSCALE)
            img_w = cv2.imread(str(sess / wide["file"]), cv2.IMREAD_GRAYSCALE)
            if img_uw is None or img_w is None:
                failures.append((tag, uw["frame"], "unreadable frame"))
                continue
            a = fit_scale(img_uw, img_w, args.radius)
            b = fit_scale(img_uw, img_w, args.radius_check)
            if not a["ok"]:
                failures.append((tag, uw["frame"], a["why"]))
                print(f"{tag:<8}{uw['frame']:>9}{dt_ms:>8.0f}"
                      f"{a['inliers']:>9}{'FAILED: ' + a['why']:>31}")
                continue
            fov = hfov_from_scale(a["scale"], float(img_w.shape[1]))
            measured.append(a["scale"])
            results.append({"session": sess.name, "frame": uw["frame"],
                            "dt_ms": dt_ms, "inliers": a["inliers"],
                            "s": a["scale"],
                            "s_tight": b["scale"] if b["ok"] else None,
                            "hfov_deg": fov, "rot_deg": a["rot_deg"]})
            tight = f"{b['scale']:.4f}" if b["ok"] else "--"
            print(f"{tag:<8}{uw['frame']:>9}{dt_ms:>8.0f}{a['inliers']:>9}"
                  f"{a['scale']:>10.4f}{tight:>10}{fov:>10.2f}{a['rot_deg']:>9.2f}")

    if not measured:
        print("\nno pair produced a fit — no verdict")
        return 1

    arr = np.array(measured)
    med = float(np.median(arr))
    iqr = float(np.percentile(arr, 75) - np.percentile(arr, 25))
    tight = np.array([r["s_tight"] for r in results if r["s_tight"] is not None])
    bias = abs(float(np.median(tight)) - med) if tight.size else float("nan")

    print(f"\n{len(measured)} pairs fitted, {len(failures)} failed")
    print(f"s median {med:.4f}   IQR {iqr:.4f}   min {arr.min():.4f}   "
          f"max {arr.max():.4f}")
    print(f"implied wide hFOV {hfov_from_scale(med, 640.0):.2f} deg   "
          f"(H0 says {hfov_from_scale(s_h0, 640.0):.2f})")
    print(f"radius sensitivity |s({int(args.radius)}) - s({int(args.radius_check)})| "
          f"= {bias:.4f}" + ("   BIAS WARNING" if bias > 0.005 else ""))
    # Walking between the two exposures scales the image, so if the residual
    # offset were driving the answer it would show up here as a slope.
    dts = np.array([r["dt_ms"] for r in results])
    if len(dts) > 2 and dts.std() > 1e-6:
        slope = float(np.polyfit(dts, arr, 1)[0])
        corr = float(np.corrcoef(dts, arr)[0, 1])
        print(f"s against dt: slope {slope*1000:+.4f} per 100 ms, r {corr:+.2f} "
              f"(|dt| mean {np.abs(dts).mean():.1f} ms)")
    for tag, frame, why in failures:
        print(f"  failure: {tag} frame {frame}: {why}")

    # The rule, as written down before any of the above existed.
    if not control_ok:
        verdict = "VOID — the synthetic control failed, so nothing here means anything"
    elif iqr > 0.010 or len(measured) < 9:
        verdict = (f"TOO NOISY — no verdict (IQR {iqr:.4f}, {len(measured)} pairs)")
    elif abs(med - s_h0) <= 0.010:
        verdict = (f"ACCEPT H0 — wide arm is {hfov_from_scale(med, 640.0):.1f} deg, "
                   "scaling the calibration by image size is legitimate")
    else:
        verdict = (f"REJECT H0 — wide arm is {hfov_from_scale(med, 640.0):.1f} deg, "
                   f"not {hfov_from_scale(s_h0, 640.0):.1f}; the recorder must log "
                   "the wide format's FOV")
    print(f"\nVERDICT: {verdict}")

    if args.sweep:
        radii = [250.0, 350.0, 450.0, 600.0, 750.0]
        per_pair = radius_sweep(sessions, args.pairs, args.separation, radii)
        # Only the pairs that fitted at *every* radius, so the ladder compares
        # the same scenes at each rung. Pooling whatever fitted at each rung
        # separately mixes a radius effect with a change of subject.
        common = [g for g in per_pair.values() if all(r in g for r in radii)]
        print(f"\nradius ladder — {len(common)} pairs that fitted at every radius")
        xs, ys = [], []
        for r in radii:
            v = [g[r] for g in common]
            if not v:
                print(f"  r < {int(r):4d} px   no pair fitted")
                continue
            m = float(np.median(v))
            xs.append(r)
            ys.append(m)
            print(f"  r < {int(r):4d} px   theta < {math.degrees(math.atan(r/f_uw)):4.1f} deg"
                  f"   s {m:.4f}   n {len(v)}")
        if len(xs) >= 3:
            slope, intercept = np.polyfit(xs, ys, 1)
            print(f"  extrapolated to r=0: s {intercept:.4f} "
                  f"({hfov_from_scale(float(intercept), 640.0):.2f} deg), "
                  f"slope {slope*1000:+.4f} per 1000 px")
            print("  H0 needs 0.3023 at every radius; distortion alone would have "
                  "to bend this line to it")

    if args.parallax:
        run_parallax(sessions, args, s_med=med)

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"f_ultrawide_px": f_uw, "s_h0": s_h0, "control": control,
             "pairs": results, "median": med, "iqr": iqr,
             "radius_bias": bias, "failures": failures, "verdict": verdict},
            indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
