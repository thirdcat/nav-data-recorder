#!/usr/bin/env python3
"""Known image motion through identical endpoints at different sampling rates."""
import cv2
import numpy as np

from high_rate_tracking import follow, pose


def main():
    cv2.setNumThreads(1)
    rng = np.random.default_rng(42)
    source = cv2.GaussianBlur(rng.integers(0, 256, (480, 640), dtype=np.uint8), (5, 5), 0)
    images = [cv2.warpAffine(source, np.float32([[1, 0, i], [0, 1, .5*i]]), (640, 480)) for i in range(13)]
    points = cv2.goodFeaturesToTrack(source, maxCorners=100, qualityLevel=.01, minDistance=20)[:, 0]
    points = points[(points[:, 0] > 100) & (points[:, 0] < 540) & (points[:, 1] > 100) & (points[:, 1] < 380)]
    for stride in [6, 3, 1]:
        indices = list(range(0, 13, stride))
        xy, valid = follow(images, points, indices)
        error = np.linalg.norm(xy-points-[12, 6], axis=1)
        assert valid.mean() > .9 and np.median(error[valid]) < .1
        back, back_valid = follow(images, xy, indices[::-1])
        assert np.median(np.linalg.norm(back[valid & back_valid]-points[valid & back_valid], axis=1)) < .1
    empty, okay = follow(images, np.empty((0, 2)), [0, 12])
    assert empty.shape == (0, 2) and okay.shape == (0,)
    T = pose(dict(qx=0, qy=0, qz=0, qw=1, tx=1, ty=2, tz=3))
    assert np.array_equal(T[:3, :3], np.diag([1, -1, -1]))
    assert np.array_equal(T[:3, 3], [1, 2, 3])
    print('known endpoint translation, reverse path, empty input, ARKit axes: passed')


if __name__ == '__main__':
    main()
