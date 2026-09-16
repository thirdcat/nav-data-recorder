#!/usr/bin/env python3
"""Deterministic geometry and leakage controls for local RGB/depth BA."""
import numpy as np
from scipy.spatial.transform import Rotation

from local_rgbd_ba import solve, heldout_score
from build_rgbd_tracks import sample_depth


def fixture():
    rng = np.random.default_rng(7)
    n, m = 6, 35
    truth = np.repeat(np.eye(4)[None], n, axis=0)
    truth[:, :3, 3] = np.c_[np.linspace(0, .4, n), np.zeros(n), np.linspace(0, .1, n)]
    truth[:, :3, :3] = Rotation.from_euler('y', np.linspace(0, 4, n), degrees=True).as_matrix()
    K = np.repeat(np.array([[600., 0, 480], [0, 600, 360], [0, 0, 1.]])[None], n, axis=0)
    points = rng.uniform([-1., -.7, 2.], [1., .7, 4.], size=(m, 3))
    track, frame, uv, depth = [], [], [], []
    for j in range(n):
        cam = (points-truth[j, :3, 3])@truth[j, :3, :3]
        image = cam@K[j].T
        for t in range(m):
            track.append(t)
            frame.append(j)
            uv.append(image[t, :2]/image[t, 2])
            depth.append(cam[t, 2])
    track, frame = np.array(track), np.array(frame)
    initial = truth.copy()
    initial[1:, :3, 3] += rng.normal(0, .04, (n-1, 3))
    initial[1:, :3, :3] = Rotation.from_rotvec(rng.normal(0, .015, (n-1, 3))).as_matrix()@initial[1:, :3, :3]
    return truth, initial, K, track, frame, np.array(uv), np.array(depth), track % 5 == 0


def main():
    truth, initial, K, track, frame, uv, depth, heldout = fixture()
    est, _, rep = solve(initial, K, track, frame, uv, depth, heldout)
    assert rep['success'], rep
    assert np.array_equal(est[0], initial[0])
    assert np.mean(np.linalg.norm(est[:, :3, 3]-truth[:, :3, 3], axis=1)) < .002
    before = heldout_score(initial, K, track, frame, uv, depth, heldout, (960, 720))
    after = heldout_score(est, K, track, frame, uv, depth, heldout, (960, 720))
    assert after['median_px'] < before['median_px']*.05, (before, after)
    # The entire held-out measurement is inaccessible to the optimizer.
    changed_uv, changed_depth = uv.copy(), depth.copy()
    changed_uv[heldout] += [300, -200]
    changed_depth[heldout] *= 10
    again, _, _ = solve(initial, K, track, frame, changed_uv, changed_depth, heldout)
    assert np.array_equal(est, again)
    # Known depth camera K is different from the RGB K; same ray must sample
    # the native pixel, not merely scale the image coordinates.
    rgb_K = np.array([[200., 0, 50], [0, 200, 50], [0, 0, 1.]])
    depth_K = np.array([[50., 0, 30], [0, 50, 30], [0, 0, 1.]])
    z = np.full((60, 60), 2.)
    z[29:32, 34:37] = 3.
    result = sample_depth(np.array([[70., 50.]]), rgb_K, z, depth_K, np.full(z.shape, 2))
    assert result[0] == 3., result
    # Inverse depth is affine on a plane. Subpixel sampling must reproduce it.
    v, u = np.mgrid[:60, :60]
    plane = 1./(.4+.001*u+.002*v)
    pixel = np.array([[73., 53.]])
    interpolated = sample_depth(pixel, rgb_K, plane, depth_K, method='inverse_bilinear')
    expected = 1./(.4+.001*35.75+.002*30.75)
    assert abs(interpolated[0]-expected) < 1e-12
    assert abs(sample_depth(pixel, rgb_K, plane, depth_K)[0]-expected) > .005
    imu = truth[:, :3, :3]
    with_imu, _, imu_report = solve(initial, K, track, frame, uv, depth, heldout, imu_rotation=imu)
    H = Rotation.from_euler('xyz', [20, 30, -50], degrees=True).as_matrix()
    with_rotated_imu, _, _ = solve(initial, K, track, frame, uv, depth, heldout, imu_rotation=H@imu)
    assert imu_report['success'] and imu_report['imu_edges'] == len(initial)-1
    assert np.max(abs(with_imu-with_rotated_imu)) < 1e-6
    fixed, _, fixed_report = solve(initial, K, track, frame, uv, depth, heldout, fixed_rotation=imu)
    assert fixed_report['success'] and np.array_equal(fixed[:, :3, :3], imu)
    assert np.mean(np.linalg.norm(fixed[:, :3, 3]-truth[:, :3, 3], axis=1)) < .002
    regularized, _, prior_report = solve(initial, K, track, frame, uv, depth, heldout,
                                         fixed_rotation=imu, translation_prior_sigma_m=.05)
    assert prior_report['success'] and prior_report['training_graph']['connected_all_cameras']
    assert np.mean(np.linalg.norm(regularized[:, :3, 3]-truth[:, :3, 3], axis=1)) < .002
    assert np.array_equal(regularized[0], initial[0])
    print('metric recovery, fixed gauge, heldout isolation, native depth K: passed')
    print('relative IMU residual and independent IMU world gauge: passed')
    print('fixed IMU rotation with metric translation recovery: passed')
    print('translation increment prior and training graph: passed')
    print('heldout reprojection px', before['median_px'], '->', after['median_px'])


if __name__ == '__main__':
    main()
