#!/usr/bin/env python3
"""Known rotation conjugacy and non-magnetic world-yaw invariance."""
import numpy as np
from scipy.spatial.transform import Rotation
from imu_rotation_calibration import fit_axes, score


def main():
    rng = np.random.default_rng(11)
    device = Rotation.from_rotvec(rng.normal(0, .2, (50, 3))).as_matrix()
    C = Rotation.from_euler('xyz', [80, -20, 35], degrees=True).as_matrix()
    camera = C@device@C.T
    got, singular = fit_axes(device[:40], camera[:40])
    assert np.max(abs(got-C)) < 1e-12
    assert score(got, device[40:], camera[40:])['p95_deg'] < 1e-10
    assert min(singular) > 0
    assert score(np.eye(3), device, camera)['median_deg'] > 5
    world = Rotation.from_rotvec(rng.normal(size=(6, 3))).as_matrix()
    H = Rotation.from_euler('z', 85, degrees=True).as_matrix()
    before = world[:-1].transpose(0, 2, 1)@world[1:]
    after = (H@world[:-1]).transpose(0, 2, 1)@(H@world[1:])
    assert np.max(abs(before-after)) < 1e-12
    print('rotation axis recovery, separate pair validation, world yaw gauge: passed')


if __name__ == '__main__':
    main()
