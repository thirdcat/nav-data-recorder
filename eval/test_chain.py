#!/usr/bin/env python3
"""Known-answer test for the window joining, with no model and no GPU.

A learned reconstruction returns each window in its own arbitrary frame and at
its own arbitrary scale. Joining them is where a method that wins per window can
still lose the trajectory, so the joining is tested on input where the answer is
already known: take a trajectory, cut it into overlapping windows, push each
through a random rotation, translation and scale, and require the original back.

This is what separated the two explanations when the seams first went wrong. It
passed, so the joining was not the fault and the windows were — which is what
sent the search to the window length, and from there to the 16 GB card that had
been choosing it.

Needs numpy only, on purpose: the failure it guards against is not a GPU
failure, and a test that can only run on the GPU box would not have been run.
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from umeyama import umeyama  # noqa: E402


def walk(n=240, seed=0):
    """A closed, gently turning walk with a little vertical wander."""
    rng = np.random.default_rng(seed)
    angle = np.linspace(0, 2 * math.pi, n)
    radius = 3.0 + 0.4 * np.sin(3 * angle)
    xs = radius * np.cos(angle)
    zs = radius * np.sin(angle)
    ys = 0.05 * np.sin(5 * angle) + rng.normal(scale=0.002, size=n)
    poses = np.tile(np.eye(4), (n, 1, 1))
    for i in range(n):
        heading = angle[i] + math.pi / 2
        c, s = math.cos(heading), math.sin(heading)
        poses[i, :3, :3] = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        poses[i, :3, 3] = [xs[i], ys[i], zs[i]]
    return poses


def random_similarity(rng):
    a = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(a)
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q, float(rng.uniform(0.5, 2.0)), rng.normal(scale=3.0, size=3)


def chain(poses, window, overlap, rng):
    """Exactly the join `pi3_chain` performs, on windows whose truth is known."""
    step = window - overlap
    chained = {}
    scales = []
    start = 0
    while start + window <= len(poses):
        frames = list(range(start, start + window))
        rot, scale, offset = random_similarity(rng)
        warped = []
        for T in poses[start:start + window]:
            M = np.eye(4)
            M[:3, :3] = rot @ T[:3, :3]
            M[:3, 3] = scale * (rot @ T[:3, 3]) + offset
            warped.append(M)
        warped = np.stack(warped)
        if not chained:
            for f, T in zip(frames, warped):
                chained[f] = T
        else:
            shared = [f for f in frames if f in chained]
            if len(shared) < 3:
                raise SystemExit("windows stopped overlapping — the test is wrong")
            src = np.stack([warped[frames.index(f)][:3, 3] for f in shared])
            dst = np.stack([chained[f][:3, 3] for f in shared])
            rot2, s2, t2 = umeyama(src, dst)
            scales.append(s2)
            for f, T in zip(frames, warped):
                if f in chained:
                    continue
                M = np.eye(4)
                M[:3, :3] = rot2 @ T[:3, :3]
                M[:3, 3] = s2 * (rot2 @ T[:3, 3]) + t2
                chained[f] = M
        start += step
    return chained, scales


def main():
    poses = walk()
    results = []
    for window, overlap in ((24, 12), (48, 24), (96, 48)):
        rng = np.random.default_rng(7)
        chained, scales = chain(poses, window, overlap, rng)
        order = sorted(chained)
        got = np.stack([chained[f][:3, 3] for f in order])
        want = poses[order][:, :3, 3]
        # The chain inherits the first window's arbitrary frame, and that one
        # global similarity is the only freedom correct joining may keep.
        rot, scale, offset = umeyama(got, want)
        aligned = (scale * (rot @ got.T)).T + offset
        err = float(np.linalg.norm(aligned - want, axis=1).max())
        ok = err < 1e-9
        results.append(ok)
        print(f"  window {window:3d} overlap {overlap:3d}  "
              f"{len(chained):3d} frames, {len(scales)+1} windows  "
              f"residual {err * 1000:8.5f} mm  {'PASS' if ok else 'FAIL'}")

    # An overlap too small to determine the fit must be refused, not guessed at.
    try:
        chain(poses, 24, 2, np.random.default_rng(0))
        print("  short overlap is refused                       FAIL")
        results.append(False)
    except SystemExit:
        print("  short overlap is refused                       PASS")
        results.append(True)

    print("all passed" if all(results) else "FAILED")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
