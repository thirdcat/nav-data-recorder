#!/usr/bin/env python3
"""Analytic geometry, gauge, scale-bias and held-out-isolation controls."""
import numpy as np

from translation_information import measurement_jacobian, schur_information, summarize_information
from test_local_rgbd_ba import fixture


def main():
    truth, _, K, track, frame, uv, depth, heldout = fixture()
    ids = np.unique(track[~heldout])
    points = []
    for tid in ids:
        j = np.flatnonzero((track == tid) & (frame == 0))[0]
        points.append(np.linalg.inv(K[0])@np.r_[uv[j], 1.]*depth[j])
    points = np.array(points)
    value, J = measurement_jacobian(points[0], truth[3], K[3])
    numerical = np.column_stack([(measurement_jacobian(points[0]+np.eye(3)[k]*1e-6, truth[3], K[3])[0]-value)/1e-6 for k in range(3)])
    assert np.allclose(J, numerical, rtol=1e-5, atol=1e-4)
    H, graph, _ = schur_information(truth, points, K, track, frame, uv, depth, heldout)
    report = summarize_information(H, graph)
    assert report['nullity_before_anchors'] == 3
    assert report['rank_after_anchors'] == 3*(len(truth)-1)
    assert np.max(abs(H@np.tile(np.eye(3), (len(truth), 1)))) < 1e-6
    # Compare elimination to an independently assembled full joint normal matrix.
    n, m = len(truth), len(points)
    joint = np.zeros((3*n+3*m, 3*n+3*m))
    for p, tid in enumerate(ids):
        for j in np.flatnonzero((track == tid) & ~heldout):
            f = frame[j]
            _, jac = measurement_jacobian(points[p], truth[f], K[f])
            jac = jac/np.array([1.5, 1.5, .03+.01*depth[j]])[:, None]
            row = np.zeros((3, len(joint)))
            row[:, 3*f:3*f+3] = -jac
            row[:, 3*n+3*p:3*n+3*p+3] = jac
            joint += row.T@row
    dense = joint[:3*n, :3*n]-joint[:3*n, 3*n:]@np.linalg.inv(joint[3*n:, 3*n:])@joint[3*n:, :3*n]
    assert np.allclose(H, dense, atol=1e-6)
    changed_uv, changed_depth = uv.copy(), depth.copy()
    changed_uv[heldout] += 900
    changed_depth[heldout] *= 8
    again, _, _ = schur_information(truth, points, K, track, frame, changed_uv, changed_depth, heldout)
    assert np.array_equal(H, again)
    rgb_only, g, _ = schur_information(truth, points, K, track, frame, uv, depth, heldout, use_depth=False)
    rgb_report = summarize_information(rgb_only, g)
    assert rgb_report['nullity_before_anchors'] >= 4
    assert rgb_report['max_conditional_camera_std_m'] is None
    extended = np.concatenate([truth, truth[-1:]])
    extended_K = np.concatenate([K, K[-1:]])
    disconnected, g, _ = schur_information(extended, points, extended_K, track, frame, uv, depth, heldout)
    assert summarize_information(disconnected, g)['nullity_before_anchors'] == 6
    assert not g['connected_all_cameras']
    # A common depth scale error still produces a full-rank local model with
    # perfect image/depth fit. Information cannot certify absolute scale.
    scaled = truth.copy()
    scaled[:, :3, 3] *= 1.05
    biased_H, g, _ = schur_information(scaled, points*1.05, K, track, frame, uv, depth*1.05, heldout)
    biased_report = summarize_information(biased_H, g)
    assert biased_report['rank_after_anchors'] == report['rank_after_anchors']
    assert biased_report['max_conditional_camera_std_m'] < .03
    for p, tid in enumerate(ids):
        j = np.flatnonzero((track == tid) & (frame == 5))[0]
        value, _ = measurement_jacobian(points[p]*1.05, scaled[5], K[5])
        assert np.allclose(value, np.r_[uv[j], depth[j]*1.05])
    print('Jacobian, dense Schur, translation gauge, missing-depth scale nullspace: passed')
    print('unobserved camera, held-out isolation, undetectable common scale bias: passed')


if __name__ == '__main__':
    main()
