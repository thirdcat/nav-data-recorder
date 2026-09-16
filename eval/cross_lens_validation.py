#!/usr/bin/env python3
"""Validate wide-only pose updates against withheld, asynchronous UW images."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from build_rgbd_tracks import sample_depth
from pi3_poseless import active_hfov, load_calibration, intrinsics_for, depth_calibration
from run_window_alignment import projection_errors

sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'tools'))
from read_session import Session
from export_3dgs import rig_extrinsic, interpolate_camera_pose


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('session')
    ap.add_argument('--tracks', required=True)
    ap.add_argument('--candidate', action='append', default=[], help='NAME=trajectory.npz')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    cv2.setRNGSeed(42)
    with np.load(a.tracks, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    ids, times, initial = data['frame_ids'], data['t'], data['initial']
    candidates = {'B0': initial}
    for arg in a.candidate:
        name, path = arg.split('=', 1)
        if name in candidates:
            raise ValueError('duplicate arm name')
        with np.load(path, allow_pickle=False) as z:
            if not np.array_equal(z['frame'], ids) or not np.allclose(z['t'], times, rtol=0, atol=1e-5):
                raise ValueError('candidate frame/time mismatch')
            candidates[name] = z['estimate']
    session = Session(a.session)
    wide = {int(r['frame']): r for r in session.stream('frames_wide')}
    uw = list(session.stream('frames'))
    uw_t = np.array([r['t'] for r in uw])
    depth_rows = list(session.stream('depth'))
    depth_times = np.array([r['t'] for r in depth_rows])
    dc, _ = depth_calibration(session.manifest)
    repo = Path(__file__).resolve().parent.parent
    calib = load_calibration(repo/'calib/iphone17-1_ultrawide.json')
    E = rig_extrinsic(repo/'calib/iphone17-1_ultrawide.json')
    KU = intrinsics_for(calib, 960, 540, active_hfov(session.manifest, 'ultrawide'))
    sift = cv2.SIFT_create(nfeatures=4000)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    features, hashes = {}, {}

    def features_for(row, size):
        key = row['file']
        if key not in features:
            path = Path(session.frame_path(row))
            im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if im is None:
                raise ValueError(f'cannot decode {path}')
            im = cv2.resize(im, size, interpolation=cv2.INTER_AREA)
            kp, d = sift.detectAndCompute(im, None)
            features[key] = np.array([k.pt for k in kp]).reshape(-1, 2), d
            hashes[key] = hashlib.sha256(path.read_bytes()).hexdigest()
        return features[key]

    pairs, proposed = [], 0
    for fi, (frame, t) in enumerate(zip(ids, times)):
        row = wide[int(frame)]
        xy, desc = features_for(row, tuple(data['image_size'].tolist()))
        if desc is None:
            continue
        train_uv = data['uv'][(data['obs_frame'] == fi) & ~data['heldout']]
        separate = (cKDTree(train_uv).query(xy)[0] >= 12) if len(train_uv) else np.ones(len(xy), bool)
        di = int(np.argmin(abs(depth_times-t)))
        dr = depth_rows[di]
        if abs(dr['t']-t) > .05:
            continue
        kd = np.array([[dc['fx'], 0, dc['cx']], [0, dc['fy'], dc['cy']], [0, 0, 1.]])
        kd[0] *= dr['width']/dc['width']
        kd[1] *= dr['height']/dc['height']
        z = sample_depth(xy, data['K'][fi], np.asarray(session.depth_frame(dr), float), kd)
        for ui in sorted(set(int(np.argmin(abs(uw_t-(t+lag)))) for lag in [0., .4, 1.])):
            tr = uw[ui]
            if tr['t'] < times[0] or tr['t'] > times[-1]:
                continue
            proposed += 1
            target, target_desc = features_for(tr, (960, 540))
            if target_desc is None:
                continue

            def ratios(d1, d2):
                return {m.queryIdx: m.trainIdx for ms in matcher.knnMatch(d1, d2, k=2)
                        if len(ms) == 2 for m, n in [ms] if m.distance < .75*n.distance}

            forward, reverse = ratios(desc, target_desc), ratios(target_desc, desc)
            matches = [(i, j) for i, j in forward.items() if reverse.get(j) == i and separate[i] and np.isfinite(z[i])]
            if len(matches) < 20:
                continue
            i, j = np.array(matches).T
            _, mask = cv2.findFundamentalMat(xy[i], target[j], cv2.FM_RANSAC, 1.5, .999)
            if mask is None:
                continue
            i, j = i[mask.ravel() != 0], j[mask.ravel() != 0]
            if len(i) < 20:
                continue
            points = (np.c_[xy[i], np.ones(len(i))]@np.linalg.inv(data['K'][fi]).T)*z[i, None]
            radius = np.linalg.norm(target[j]-[480, 270], axis=1)/np.hypot(480, 270)
            pairs.append({'wide_frame_index': fi, 'wide_frame': int(frame), 'uw_frame': int(tr['frame']),
                          'uw_time': tr['t'], 'delta_time_s': float(tr['t']-t), 'points': points,
                          'target_uv': target[j], 'outer': radius > .6})
    report = {'session': Path(a.session).name, 'proposed_pairs': proposed, 'accepted_pairs': len(pairs),
              'tracks_sha256': hashlib.sha256(Path(a.tracks).read_bytes()).hexdigest(),
              'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'calibration_sha256': hashlib.sha256((repo/'calib/iphone17-1_ultrawide.json').read_bytes()).hexdigest(),
              'images_sha256': hashes, 'arms': {}}
    for name, poses in candidates.items():
        scores, outer = [], []
        for p in pairs:
            target_pose = interpolate_camera_pose(times, poses, p['uw_time'])@E
            err = projection_errors(p['points'], p['target_uv'], KU, poses[p['wide_frame_index']], target_pose, (960, 540))
            scores.append({'wide_frame': p['wide_frame'], 'uw_frame': p['uw_frame'], 'matches': len(err),
                           'delta_time_s': p['delta_time_s'], 'median_px': float(np.median(err))})
            outer.extend(err[p['outer']].tolist())
        report['arms'][name] = {'pair_median_px': float(np.median([s['median_px'] for s in scores])) if scores else None,
                                 'outer_observations': len(outer), 'outer_median_px': float(np.median(outer)) if outer else None,
                                 'pairs': scores}
        print(name, {k: v for k, v in report['arms'][name].items() if k != 'pairs'}, flush=True)
    (out/'pairs.json').write_text(json.dumps(pairs, default=lambda x: x.tolist(), indent=2)+'\n')
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print('accepted', len(pairs), 'of', proposed)


if __name__ == '__main__':
    main()
