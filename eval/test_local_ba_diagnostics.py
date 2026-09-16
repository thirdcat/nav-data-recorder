#!/usr/bin/env python3
import numpy as np

from local_ba_diagnostics import connectivity, sampson
from test_local_rgbd_ba import fixture


def main():
    truth, _, K, track, frame, uv, _, heldout = fixture()
    a = np.flatnonzero(frame == 0)
    b = np.flatnonzero(frame == 5)
    error = sampson(truth, K, frame[a], frame[b], uv[a], uv[b], (960, 720))
    assert error.max() < 1e-8
    shifted = uv[b]+[0, 3]
    error = sampson(truth, K, frame[a], frame[b], uv[a], shifted, (960, 720))
    assert np.median(error) > 1.
    scaled = truth.copy()
    scaled[:, :3, 3] *= 7
    assert np.allclose(error, sampson(scaled, K, frame[a], frame[b], uv[a], shifted, (960, 720)))
    graph = connectivity(6, track, frame, heldout)
    assert graph['connected_all_cameras']
    graph = connectivity(7, track, frame, heldout)
    assert not graph['connected_all_cameras'] and graph['first_camera_component_size'] == 6
    assert graph['observations_per_camera'][-1] == 0
    print('epipolar geometry, scale blindness, disconnected/unobserved camera: passed')


if __name__ == '__main__':
    main()
