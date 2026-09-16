#!/usr/bin/env python3
"""Independent pinhole examples for the R2-P1 measurement evaluator."""
import numpy as np
from scipy.spatial.transform import Rotation

from run_window_alignment import projection_errors, reference_score


def main():
    K = np.array([[400., 0, 320], [0, 400, 240], [0, 0, 1]])
    source, target = np.eye(4), np.eye(4)
    target[0, 3] = .2
    points = np.array([[0., 0, 2], [.3, -.1, 4]])
    # Target camera is 0.2 m right of source, so x shifts left by f*0.2/z.
    uv = np.array([[280., 240.], [330., 230.]])
    err = projection_errors(points, uv, K, source, target, (640, 480))
    assert np.max(err) < 1e-10
    wrong = projection_errors(points, uv, K, source, source, (640, 480))
    assert np.allclose(wrong, [40, 20])
    G = np.eye(4)
    G[:3, :3] = Rotation.from_euler('xyz', [15, 20, -35], degrees=True).as_matrix()
    G[:3, 3] = [3, -1, 5]
    assert np.max(projection_errors(points, uv, K, G@source, G@target, (640, 480))) < 1e-10
    behind = target.copy(); behind[2, 3] = 10
    assert np.all(projection_errors(points, uv, K, source, behind, (640, 480)) == 800)
    far = target.copy(); far[0, 3] = 20
    assert np.all(projection_errors(points, uv, K, source, far, (640, 480)) == 800)
    poses = np.repeat(np.eye(4)[None], 4, axis=0)
    poses[:, :3, 3] = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    score = reference_score(G@poses, poses)
    assert score['rmse_position_m'] < 1e-10 and score['median_rotation_deg'] < 1e-10
    scaled = poses.copy(); scaled[:, :3, 3] *= 2
    assert reference_score(scaled, poses)['mean_position_m'] > .5
    print('pinhole sign/units, global gauge, failed projections, SE(3)-only scoring: all passed')


if __name__ == '__main__':
    main()
