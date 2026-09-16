#!/usr/bin/env python3
"""Post-hoc camera connectivity and scale-free held-out RGB diagnostics."""
import argparse
import json
from pathlib import Path

import numpy as np


def connectivity(n, track, frame, heldout):
    parent = np.arange(n)

    def root(x):
        while parent[x] != x:
            x = parent[x]
        return x

    for tid in np.unique(track[~heldout]):
        frames = np.unique(frame[(track == tid) & ~heldout])
        for f in frames[1:]:
            parent[root(f)] = root(frames[0])
    roots = np.array([root(i) for i in range(n)])
    components = [np.flatnonzero(roots == r).tolist() for r in np.unique(roots)]
    return {'components': components, 'first_camera_component_size': int((roots == roots[0]).sum()),
            'observations_per_camera': np.bincount(frame[~heldout], minlength=n).tolist(),
            'connected_all_cameras': len(components) == 1}


def sampson(poses, K, a, b, uv_a, uv_b, image_size):
    rel = np.linalg.inv(poses[b])@poses[a]
    t = rel[:, :3, 3]
    skew = np.zeros((len(a), 3, 3))
    skew[:, 0, 1], skew[:, 0, 2] = -t[:, 2], t[:, 1]
    skew[:, 1, 0], skew[:, 1, 2] = t[:, 2], -t[:, 0]
    skew[:, 2, 0], skew[:, 2, 1] = -t[:, 1], t[:, 0]
    F = np.linalg.inv(K[b]).transpose(0, 2, 1)@skew@rel[:, :3, :3]@np.linalg.inv(K[a])
    x, y = np.c_[uv_a, np.ones(len(a))], np.c_[uv_b, np.ones(len(a))]
    Fx = np.einsum('nij,nj->ni', F, x)
    Fty = np.einsum('nji,nj->ni', F, y)
    denom = np.sqrt(np.sum(Fx[:, :2]**2, axis=1)+np.sum(Fty[:, :2]**2, axis=1))
    residual = np.abs(np.sum(y*Fx, axis=1))/np.maximum(denom, 1e-30)
    residual[(np.linalg.norm(t, axis=1) < 1e-6) | (denom < 1e-15)] = np.hypot(*image_size)
    return np.minimum(residual, np.hypot(*image_size))


def diagnose(directory):
    with np.load(directory/'tracks.npz', allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    track, frame, heldout = [d[k] for k in ['track', 'obs_frame', 'heldout']]
    pairs, pair_track = [], []
    for tid in np.unique(track[heldout]):
        observations = np.flatnonzero((track == tid) & heldout)
        observations = observations[np.argsort(frame[observations])]
        for a, b in zip(observations[:-1], observations[1:]):
            if np.linalg.norm(d['initial'][frame[a], :3, 3]-d['initial'][frame[b], :3, 3]) >= .01:
                pairs.append((a, b))
                pair_track.append(tid)
    result = {'session': directory.name, 'training_graph': connectivity(len(d['initial']), track, frame, heldout),
              'heldout_pairs': len(pairs), 'heldout_tracks': len(set(pair_track)), 'sampson': {}}
    if not pairs:
        return result
    a, b = np.array(pairs).T
    pair_track = np.array(pair_track)
    candidates = {'initial': d['initial'], 'arkit': d['reference']}
    for name in ['rgbd', 'fixed_imu', 'arkit_control']:
        with np.load(directory/name/'B-RGBD.npz', allow_pickle=False) as z:
            candidates[name] = z['estimate']
    for name, poses in candidates.items():
        error = sampson(poses, d['K'], frame[a], frame[b], d['uv'][a], d['uv'][b], d['image_size'])
        by_track = [np.median(error[pair_track == tid]) for tid in np.unique(pair_track)]
        result['sampson'][name] = {'median_track_px': float(np.median(by_track)), 'median_pair_px': float(np.median(error))}
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('census')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise ValueError('output exists')
    results = [diagnose(p) for p in sorted(Path(a.census).iterdir()) if p.is_dir()]
    out.write_text(json.dumps(results, indent=2, allow_nan=False)+'\n')
    for r in results:
        print(r['session'], 'components', len(r['training_graph']['components']), 'gauge cameras', r['training_graph']['first_camera_component_size'],
              'Sampson px', {k: round(v['median_track_px'], 3) for k, v in r['sampson'].items()})


if __name__ == '__main__':
    main()
