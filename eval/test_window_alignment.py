#!/usr/bin/env python3
"""Deterministic known-geometry tests; SciPy required, no model/GPU."""
import numpy as np
from scipy.spatial.transform import Rotation

from window_alignment import prepare, align_windows, sequential, overlap_pairs, robust_blocks


def fixture(pure_rotation=False):
    rng = np.random.default_rng(42)
    n = 30
    t = np.linspace(0, 2, n)
    truth = np.repeat(np.eye(4)[None], n, axis=0)
    if not pure_rotation:
        truth[:, :3, 3] = np.c_[t, np.sin(t), 0.3*np.cos(2*t)]
    truth[:, :3, :3] = Rotation.from_rotvec(np.c_[0.2*t, -0.1*t, 0.5*t]).as_matrix()
    frames = np.array([np.arange(start, start+14) for start in [0, 5, 10, 16]])
    G = np.repeat(np.eye(4)[None], len(frames), axis=0)
    G[1:, :3, :3] = Rotation.from_rotvec(rng.normal(0, .6, (3, 3))).as_matrix()
    G[1:, :3, 3] = rng.normal(size=(3, 3))
    scales = np.array([1., .7, 1.6, 1.1])
    pred = np.array([np.linalg.inv(g)@truth[f] for g, f in zip(G, frames)])
    pred[..., :3, 3] /= scales[:, None, None]
    return frames, pred, scales, truth, G


def rejected(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("invalid input accepted")


def main():
    for pure in [False, True]:
        frames, pred, scales, truth, G = fixture(pure)
        original = pred.copy()
        f, p = prepare(frames, pred, scales)
        ids, est, transforms, report = align_windows(f, p)
        assert report['success'], report
        assert np.max(abs(est-truth[ids])) < 1e-7
        assert np.max(abs(transforms-G)) < 1e-7
        assert np.array_equal(transforms[0], np.eye(4))
        assert np.array_equal(pred, original)
        # Reordering non-anchor windows must not change exact recovery.
        order = [0, 3, 2, 1]
        _, reordered, _, report = align_windows(f[order], p[order])
        assert report['success'] and np.max(abs(reordered-est)) < 1e-7
    print('exact geometry, fixed metric scales, pure rotation, order, gauge: passed')

    frames, pred, scales, truth, _ = fixture()
    f, p = prepare(frames, pred, scales)
    rejected(lambda: prepare(frames, pred, [1, 0, 1, 1]))
    broken = pred.copy(); broken[0, 0, 0, 0] = np.nan
    rejected(lambda: prepare(frames, broken, scales))
    reflected = pred.copy(); reflected[0, 0, :3, 0] *= -1
    rejected(lambda: prepare(frames, reflected, scales))
    duplicate = frames.copy(); duplicate[0, 0] = duplicate[0, 1]
    rejected(lambda: prepare(duplicate, pred, scales))
    disconnected = frames.copy(); disconnected[-1] += 1000
    rejected(lambda: align_windows(disconnected, p))
    _, weights = overlap_pairs(f)
    assert abs(weights.sum()-sum(np.sum(f == k) > 1 for k in np.unique(f))) < 1e-10
    print('malformed inputs, disconnected graph, frame multiplicity: passed')

    # A single corrupted overlapping observation must not tilt clean windows.
    corrupt = p.copy()
    corrupt[1, 8, :3, 3] += [2, -1, .7]
    corrupt[1, 8, :3, :3] = Rotation.from_euler('z', 60, degrees=True).as_matrix() @ corrupt[1, 8, :3, :3]
    errors = []
    for robust in [False, True]:
        ids, est, _, rep = align_windows(f, corrupt, robust=robust)
        assert rep['success'], rep
        keep = ids != f[1, 8]
        errors.append(float(np.mean(np.linalg.norm(est[keep, :3, 3]-truth[ids[keep], :3, 3], axis=1))))
    assert errors[1] < errors[0]*.5, errors
    # Independent check of robust block cost, including zero and large residual.
    r = np.array([[0., 0., 0.], [3., 4., 0.], [100., -100., 10.]])
    assert np.allclose(np.sum(robust_blocks(r)**2, axis=1), 2*(np.sqrt(1+np.sum(r*r, axis=1))-1))
    print('corrupt overlap robustness:', errors, 'passed')
    print('all passed')


if __name__ == '__main__':
    main()
