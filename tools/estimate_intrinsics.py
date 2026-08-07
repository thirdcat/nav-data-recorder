#!/usr/bin/env python3
"""Recover an episode's focal lengths from its images and poses.

The episode format carries no intrinsics, so the field of view of an existing
episode set is not written down anywhere. It is still recoverable: the poses fix
the relative geometry between any two frames, which fixes the epipolar geometry,
and only the true focal lengths make observed feature matches consistent with it.

That is enough to answer a question the format otherwise leaves open — whether a
set was produced by cropping to the target aspect or by squeezing a whole frame
into it. A crop keeps `fx == fy`; squeezing a 4:3 frame into 16:9 leaves
`fx/fy ≈ 1.33`.

    python3 tools/estimate_intrinsics.py ./episodes/20260807-140619-477750

Requires numpy and opencv-python.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    print("this tool needs opencv:  pip install opencv-python-headless", file=sys.stderr)
    raise SystemExit(2)


# The episode convention is FLU (X forward, Y left, Z up); the epipolar maths
# below is in OpenCV's (X right, Y down, Z forward). Rows are the OpenCV axes
# expressed in FLU.
FLU_TO_CV = np.array([
    [0.0, -1.0, 0.0],
    [0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0],
])


def quat_to_matrix(x, y, z, w):
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def load_poses(path):
    poses = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            v = [float(x) for x in line.split()]
            R_flu = quat_to_matrix(v[4], v[5], v[6], v[7])
            poses.append((np.array(v[1:4]), R_flu @ FLU_TO_CV.T))
    return poses


def skew(t):
    return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])


def sampson(F, p1, p2):
    """Sampson distance, the first-order approximation to reprojection error.

    Plain algebraic error `x2ᵀ F x1` scales with image coordinates and would let
    a wrong focal length hide behind it; Sampson normalises by the epipolar line
    gradients and is comparable across candidates.
    """
    h1 = np.hstack([p1, np.ones((len(p1), 1))])
    h2 = np.hstack([p2, np.ones((len(p2), 1))])
    Fx1 = h1 @ F.T
    Ftx2 = h2 @ F
    num = np.einsum("ij,ij->i", h2, Fx1) ** 2
    den = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    return num / np.maximum(den, 1e-12)


def collect_matches(images, poses, max_pairs, min_baseline, stride):
    """Feature correspondences with the known relative pose for each pair."""
    sift = cv2.SIFT_create(nfeatures=3000)
    matcher = cv2.BFMatcher()
    cache = {}

    def features(i):
        if i not in cache:
            img = cv2.imread(images[i], cv2.IMREAD_GRAYSCALE)
            cache[i] = sift.detectAndCompute(img, None)
        return cache[i]

    pairs = []
    for i in range(0, len(poses) - stride, stride):
        j = i + stride
        p1, R1 = poses[i]
        p2, R2 = poses[j]
        if np.linalg.norm(p2 - p1) < min_baseline:
            continue

        k1, d1 = features(i)
        k2, d2 = features(j)
        if d1 is None or d2 is None:
            continue
        raw = matcher.knnMatch(d1, d2, k=2)
        # Lowe's ratio test: a match whose best and second-best are close is
        # ambiguous, and ambiguous matches dominate the error if kept.
        good = [m for m, n in raw if m.distance < 0.7 * n.distance]
        if len(good) < 30:
            continue

        x1 = np.array([k1[m.queryIdx].pt for m in good])
        x2 = np.array([k2[m.trainIdx].pt for m in good])

        # X2 = R X1 + t for the pair, from the known world poses.
        R = R2.T @ R1
        t = R2.T @ (p1 - p2)
        pairs.append((x1, x2, skew(t) @ R))
        if len(pairs) >= max_pairs:
            break
    return pairs


def score(pairs, fx, fy, cx, cy):
    K_inv = np.array([[1 / fx, 0, -cx / fx], [0, 1 / fy, -cy / fy], [0, 0, 1]])
    errors = []
    for x1, x2, E in pairs:
        F = K_inv.T @ E @ K_inv
        errors.append(sampson(F, x1, x2))
    allerr = np.concatenate(errors)
    # Median, not mean: a handful of surviving mismatches would otherwise pull
    # the optimum around.
    return float(np.median(allerr))


def estimate(pairs, width, height, isotropic=False):
    """Coarse-to-fine search over (fx, fy).

    The two axes refine independently. Sharing one bracket — as this did
    originally — collapses when the true values differ, which is exactly the
    case the tool exists to detect: a squeezed 4:3 frame has fx/fy near 1.33, so
    a shared window centres between them and converges to neither.
    """
    cx, cy = width / 2.0, height / 2.0
    lo_x = lo_y = 0.25 * width
    hi_x = hi_y = 3.0 * width

    fx = fy = None
    best_score = float("inf")
    for _ in range(8):
        cand_x = np.linspace(lo_x, hi_x, 21)
        cand_y = np.linspace(lo_y, hi_y, 21)
        best = None
        best_score = float("inf")
        for a in cand_x:
            for b in ([a] if isotropic else cand_y):
                s = score(pairs, a, b, cx, cy)
                if s < best_score:
                    best_score, best = s, (a, b)
        fx, fy = best
        span_x = (hi_x - lo_x) / 8
        span_y = (hi_y - lo_y) / 8
        lo_x, hi_x = max(1.0, fx - span_x), fx + span_x
        lo_y, hi_y = max(1.0, fy - span_y), fy + span_y
    return fx, fy, best_score


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episode", help="episode directory (images/ + poses_tum.txt)")
    ap.add_argument("--stride", type=int, default=3, help="frame gap within a pair")
    ap.add_argument("--pairs", type=int, default=25)
    ap.add_argument("--min-baseline", type=float, default=0.05,
                    help="metres; pairs closer than this carry no depth information")
    args = ap.parse_args(argv)

    poses = load_poses(os.path.join(args.episode, "poses_tum.txt"))
    img_dir = os.path.join(args.episode, "images")
    images = [os.path.join(img_dir, n) for n in sorted(os.listdir(img_dir))
              if n.lower().endswith((".jpg", ".png"))]
    if len(images) != len(poses):
        print(f"! {len(images)} images but {len(poses)} poses", file=sys.stderr)
    n = min(len(images), len(poses))
    images, poses = images[:n], poses[:n]

    probe = cv2.imread(images[0])
    height, width = probe.shape[:2]
    print(f"{os.path.basename(args.episode.rstrip('/'))}: {n} frames, {width}x{height}")

    pairs = collect_matches(images, poses, args.pairs, args.min_baseline, args.stride)
    if len(pairs) < 3:
        print("! too few usable pairs — try a larger --stride or a longer episode")
        return 1
    print(f"  {len(pairs)} pairs, {sum(len(p[0]) for p in pairs)} correspondences")

    fx, fy, err = estimate(pairs, width, height)
    hfov = 2 * math.degrees(math.atan((width / 2) / fx))
    vfov = 2 * math.degrees(math.atan((height / 2) / fy))
    print(f"  fx {fx:7.1f}   fy {fy:7.1f}   fx/fy {fx / fy:.3f}")
    print(f"  hfov {hfov:5.1f}   vfov {vfov:5.1f} degrees   (median Sampson {err:.3f} px^2)")

    # How much worse the fit gets when each focal length is moved 10%. A camera
    # that only pans and translates sideways barely constrains fy at all — the
    # answer would then be the search's starting point rather than a measurement,
    # and it is the axis this tool exists to read.
    cx, cy = width / 2.0, height / 2.0
    base = max(err, 1e-9)
    sens_x = min(score(pairs, fx * 1.1, fy, cx, cy),
                 score(pairs, fx * 0.9, fy, cx, cy)) / base
    sens_y = min(score(pairs, fx, fy * 1.1, cx, cy),
                 score(pairs, fx, fy * 0.9, cx, cy)) / base
    print(f"  conditioning: a 10% error costs {sens_x:.1f}x on fx, {sens_y:.1f}x on fy")
    if sens_y < 2.0:
        print("  ! fy is weakly constrained — this clip has little vertical parallax. "
              "Use one with more pitch change or vertical motion before trusting "
              "the fx/fy verdict.")

    ratio = fx / fy
    if abs(ratio - 1.0) < 0.06:
        print("  -> square pixels: the source was cropped to this aspect, or was "
              "natively this shape")
    elif abs(ratio - 4 / 3) < 0.10:
        print("  -> fx/fy near 4:3: a 4:3 frame was squeezed whole into this one, "
              "so the field of view is the full sensor's")
    else:
        print(f"  -> non-square pixels at {ratio:.2f}, matching no common resize")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
