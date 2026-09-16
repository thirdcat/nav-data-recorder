#!/usr/bin/env python3
"""Training-photo isolation and fixed evaluation-camera gauge controls."""
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from cache_pi3_training import load_training, validate_split
from evaluate_frame_holdout import fixed_evaluation_cameras
from run_window_alignment import projection_errors


def main():
    entry = {'session': 'unused', 'all_frame_ids': [0, 1, 2, 3], 'train_frame_ids': [0, 2], 'eval_frame_ids': [1, 3]}
    validate_split(entry)
    try:
        validate_split(dict(entry, eval_frame_ids=[0, 3]))
    except ValueError:
        pass
    else:
        raise AssertionError('overlapping split accepted')
    accessed = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root/'input.jpg').write_bytes(b'training-only image fixture')

        class FakeSession:
            def __init__(self, _):
                pass

            def stream(self, name):
                return iter([dict(frame=i, t=float(i), width=8, height=6,
                                  fx=5., fy=5., cx=4., cy=3.) for i in range(4)])

            def frame_path(self, row):
                assert row['frame'] in entry['train_frame_ids']
                accessed.append(('image', row['frame']))
                return str(root/'input.jpg')

            def depth_frame(self, row):
                assert row['frame'] in entry['train_frame_ids']
                accessed.append(('depth', row['frame']))
                return np.ones((3, 4))

            def confidence_frame(self, row):
                assert row['frame'] in entry['train_frame_ids']
                return np.full((3, 4), 2)

        out = root/'out'
        out.mkdir()
        with patch('cache_pi3_training.Session', FakeSession):
            depths, _, times, hashes = load_training(entry, out)
        assert depths.shape == (2, 6, 8) and list(times) == [0, 2]
        assert set(hashes) == {'0', '2'} and len(list(out.glob('*.jpg'))) == 2
        assert set(accessed) == {('image', 0), ('image', 2), ('depth', 0), ('depth', 2)}
    reference = np.repeat(np.eye(4)[None], 3, axis=0)
    reference[:, 0, 3] = [0, .1, .2]
    G = np.eye(4)
    G[:3, :3] = Rotation.from_euler('xyz', [20, -15, 40], degrees=True).as_matrix()
    G[:3, 3] = [1, 2, 3]
    targets, gauge = fixed_evaluation_cameras(G@reference[0], reference[0], reference[1:])
    assert np.allclose(gauge, G) and np.allclose(targets, G@reference[1:])
    point = np.array([[0., 0., 2.]])
    K = np.array([[600., 0, 480], [0, 600, 360], [0, 0, 1.]])
    target_uv = np.array([[450., 360.]])
    assert projection_errors(point, target_uv, K, G@reference[0], targets[0], (960, 720))[0] < 1e-8
    moved = G@reference[0]
    moved[:3, 3] += G[:3, :3]@np.array([.02, 0, 0])
    assert projection_errors(point, target_uv, K, moved, targets[0], (960, 720))[0] > 5
    print('held-out image/depth exclusion, shared evaluation gauge, pose-change sensitivity: passed')


if __name__ == '__main__':
    main()
