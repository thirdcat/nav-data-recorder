#!/usr/bin/env python3
"""Evaluate fixed-rotation information and disjoint training-track fits."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from local_rgbd_ba import solve, heldout_score
from run_window_alignment import reference_score
from translation_information import schur_information, summarize_information


def run(directory, out):
    out.mkdir(exist_ok=False)
    with np.load(directory/'tracks.npz', allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    with np.load(directory/'imu.npz', allow_pickle=False) as z:
        if not np.array_equal(z['frame_ids'], d['frame_ids']):
            raise ValueError('IMU frame mismatch')
        imu = z['camera_rotation']
    initial = d['initial']
    rotations = (initial[0, :3, :3]@imu[0].T)@imu
    args = [d[k] for k in ['K', 'track', 'obs_frame', 'uv', 'depth']]
    with np.load(directory/'fixed_imu/B-RGBD.npz', allow_pickle=False) as z:
        full, points = z['estimate'], z['landmarks']
    H, graph, count = schur_information(full, points, *args, d['heldout'])
    result = {'id': directory.name, 'tracks_sha256': hashlib.sha256((directory/'tracks.npz').read_bytes()).hexdigest(),
              'full': summarize_information(H, graph), 'information_observations': count, 'halves': []}
    train_ids = np.unique(d['track'][~d['heldout']])
    estimates = []
    for half in [0, 1]:
        selected = train_ids[half::2]
        excluded = ~np.isin(d['track'], selected)
        poses, landmarks, report = solve(initial, *args, excluded, fixed_rotation=rotations)
        half_H, half_graph, _ = schur_information(poses, landmarks, *args, excluded)
        result['halves'].append({'training_ids': selected.tolist(), 'solver': report,
                                  'information': summarize_information(half_H, half_graph)})
        estimates.append(poses)
        np.savez_compressed(out/f'half_{half}.npz', frame=d['frame_ids'], t=d['t'], estimate=poses,
                            landmarks=landmarks, convention=d['convention'], excluded=excluded)
    difference = np.linalg.norm(estimates[0][:, :3, 3]-estimates[1][:, :3, 3], axis=1)
    result['split_difference_median_m'] = float(np.median(difference))
    result['split_difference_p95_m'] = float(np.percentile(difference, 95))
    result['split_difference_per_camera_m'] = difference.tolist()
    information = [result['full']]+[h['information'] for h in result['halves']]
    # Gate is decided entirely from training measurements, before looking at
    # original held-out tracks or reference poses below.
    result['training_screen'] = {
        'connected_full_and_halves': all(v['graph']['connected_all_cameras'] for v in information),
        'full_conditional_std_le_2cm': result['full']['max_conditional_camera_std_m'] is not None and result['full']['max_conditional_camera_std_m'] <= .02,
        'split_median_le_1cm': result['split_difference_median_m'] <= .01,
        'split_p95_le_3cm': result['split_difference_p95_m'] <= .03,
        'halves_converged': all(h['solver']['success'] for h in result['halves']),
    }
    result['training_screen_pass'] = all(result['training_screen'].values())
    result['evaluation'] = {}
    for name, poses in [('initial', initial), ('full', full), ('half_0', estimates[0]), ('half_1', estimates[1])]:
        result['evaluation'][name] = {'reference': reference_score(poses, d['reference']) if len(d['reference']) else None,
                                     'heldout': heldout_score(poses, *args, d['heldout'], d['image_size'])}
    (out/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    np.savez_compressed(out/'measurement_information.npz', H=H)
    print(directory.name, 'connected', result['training_screen']['connected_full_and_halves'],
          'std cm', None if result['full']['max_conditional_camera_std_m'] is None else round(result['full']['max_conditional_camera_std_m']*100, 2),
          'split median/p95 cm', round(np.median(difference)*100, 2), round(np.percentile(difference, 95)*100, 2),
          'pass', result['training_screen_pass'], flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--previous', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=False)
    selected = []
    for group in ['arkit_census', 'replication']:
        for folder in sorted((Path(a.previous)/group).iterdir()):
            if folder.is_dir():
                selected.append(folder)
    (out/'selection.json').write_text(json.dumps([str(p) for p in selected], indent=2)+'\n')
    for folder in selected:
        run(folder, out/folder.name)


if __name__ == '__main__':
    main()
