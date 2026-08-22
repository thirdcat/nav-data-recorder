#!/usr/bin/env python3
"""Known-answer tests for the Pi3X metric join policies."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pi3_join import fit_join, window_scales  # noqa: E402


def main():
    rng = np.random.default_rng(4)
    source = rng.normal(size=(24, 3))
    rotation, _, _ = np.linalg.svd(rng.normal(size=(3, 3)))[0:3]
    if np.linalg.det(rotation) < 0:
        rotation[:, -1] *= -1
    target = 1.7 * (source @ rotation.T) + np.array([2.0, -1.0, 0.5])

    got = fit_join(source, target, "free")
    assert abs(got[1] - 1.7) < 1e-10
    assert got[4] < 1e-10

    rigid = fit_join(source, target, "depth")
    assert rigid[1] == 1.0
    assert rigid[3] > 1.6
    assert rigid[4] > 0.1

    values = window_scales([0.8, 1.1, 0.9, 20.0, 1.0], "median")
    assert np.all(values == 1.0)
    assert np.all(window_scales([0.8, 1.1], "depth") == [0.8, 1.1])
    print("all passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
