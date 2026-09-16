#!/usr/bin/env python3
"""Score unused RGB/depth frames at one fixed set of ARKit evaluation cameras."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from build_rgbd_tracks import sample_depth
from cache_pi3_training import validate_split
from run_window_alignment import projection_errors
sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'tools'))
from read_session import Session


def camera_pose(row):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([row[k] for k in ['qx', 'qy', 'qz', 'qw']]).as_matrix()@np.diag([1., -1., -1.])
    T[:3, 3] = [row[k] for k in ['tx', 'ty', 'tz']]
    return T


def fixed_evaluation_cameras(initial_first, reference_first, reference_eval):
    """Fix the common gauge once; no candidate-dependent alignment or scale fit."""
    gauge = initial_first@np.linalg.inv(reference_first)
    return gauge@reference_eval, gauge


def evaluate(entry, tracks_path, candidate_path, out, source_exclusion_px=12.):
    validate_split(entry)
    if not np.isfinite(source_exclusion_px) or source_exclusion_px < 0:
        raise ValueError('source exclusion must be finite and nonnegative')
    out.mkdir(exist_ok=False)
    cv2.setNumThreads(1)
    cv2.setRNGSeed(42)
    with np.load(tracks_path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    with np.load(candidate_path, allow_pickle=False) as z:
        candidate = z['estimate']
        if not np.array_equal(data['frame_ids'], z['frame']) or not np.allclose(data['t'], z['t'], atol=1e-6, rtol=0):
            raise ValueError('candidate and training frames differ')
    if list(data['frame_ids']) != entry['train_frame_ids'] or set(entry['eval_frame_ids']) & set(data['frame_ids']):
        raise ValueError('evaluation frames leaked into training cache')
    if not np.allclose(candidate[0], data['initial'][0], atol=1e-6):
        raise ValueError('candidate changed the fixed first-camera gauge')
    session = Session(entry['session'])
    rows = {int(r['frame']): r for r in session.stream('frames')}
    poses = {int(r['frame']): r for r in session.stream('pose')}
    depths = {int(r['frame']): r for r in session.stream('depth')}
    reference_eval = np.array([camera_pose(poses[f]) for f in entry['eval_frame_ids']])
    eval_poses, gauge = fixed_evaluation_cameras(data['initial'][0], camera_pose(poses[entry['train_frame_ids'][0]]), reference_eval)
    eval_by_frame = dict(zip(entry['eval_frame_ids'], eval_poses))
    candidates = {'Pi3X': data['initial'], 'BA': candidate,
                  'ARKit_pose_diagnostic': gauge@np.array([camera_pose(poses[f]) for f in entry['train_frame_ids']])}
    feature_cache, hashes = {}, {}
    sift = cv2.SIFT_create(nfeatures=4000)
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    def features(frame):
        if frame not in feature_cache:
            row, cr = rows[frame], poses[frame]
            path = Path(session.frame_path(row))
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError('cannot decode '+str(path))
            h, w = image.shape
            size = (960, round(h*960/w))
            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            kp, desc = sift.detectAndCompute(image, None)
            xy = np.array([p.pt for p in kp], dtype=float).reshape(-1, 2)
            K = np.array([[cr['fx'], 0, cr['cx']], [0, cr['fy'], cr['cy']], [0, 0, 1.]])
            K[0] *= size[0]/w
            K[1] *= size[1]/h
            feature_cache[frame] = xy, desc, K, size
            hashes[str(frame)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return feature_cache[frame]

    def depth_at(frame, xy, K, size):
        dr = depths[frame]
        kd = K.copy()
        kd[0] *= dr['width']/size[0]
        kd[1] *= dr['height']/size[1]
        confidence = session.confidence_frame(dr)
        if confidence is None:
            raise ValueError('evaluation requires native confidence')
        return sample_depth(xy, K, np.asarray(session.depth_frame(dr), float), kd, np.asarray(confidence))

    def ratios(a, b):
        return {m.queryIdx: m.trainIdx for pair in matcher.knnMatch(a, b, k=2)
                if len(pair) == 2 for m, n in [pair] if m.distance < .75*n.distance}

    train_times = data['t']
    pairs, proposed = [], 0
    for target in entry['eval_frame_ids']:
        t = rows[target]['t']
        before = np.flatnonzero(train_times < t)[-2:]
        after = np.flatnonzero(train_times > t)[:2]
        target_xy, target_desc, target_K, size = features(target)
        for source_index in np.r_[before, after]:
            source = entry['train_frame_ids'][source_index]
            proposed += 1
            xy, desc, K, source_size = features(source)
            if desc is None or target_desc is None:
                continue
            training = data['uv'][(data['obs_frame'] == source_index) & ~data['heldout']]
            separate = cKDTree(training).query(xy)[0] >= source_exclusion_px if len(training) else np.ones(len(xy), bool)
            z = depth_at(source, xy, K, source_size)
            forward, reverse = ratios(desc, target_desc), ratios(target_desc, desc)
            matches = [(i, j) for i, j in forward.items() if reverse.get(j) == i and separate[i] and np.isfinite(z[i])]
            if len(matches) < 20:
                continue
            i, j = np.array(matches).T
            _, mask = cv2.findFundamentalMat(xy[i], target_xy[j], cv2.FM_RANSAC, 1.5, .999)
            if mask is None:
                continue
            i, j = i[mask.ravel() != 0], j[mask.ravel() != 0]
            if len(i) < 20:
                continue
            points = (np.c_[xy[i], np.ones(len(i))]@np.linalg.inv(K).T)*z[i, None]
            target_z = depth_at(target, target_xy[j], target_K, size)
            pairs.append({'source_index': int(source_index), 'source_frame': source, 'target_frame': target,
                          'points': points, 'target_uv': target_xy[j], 'target_depth': target_z,
                          'target_K': target_K, 'image_size': size})
    result = {'id': entry['id'], 'proposed_pairs': proposed, 'accepted_pairs': len(pairs),
              'source_exclusion_px': source_exclusion_px,
              'eval_camera_coverage': len({p['target_frame'] for p in pairs}), 'image_sha256': hashes,
              'tracks_sha256': hashlib.sha256(Path(tracks_path).read_bytes()).hexdigest(),
              'candidate_sha256': hashlib.sha256(Path(candidate_path).read_bytes()).hexdigest(),
              'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'eval_frame_ids': entry['eval_frame_ids'], 'eval_poses': eval_poses.tolist(), 'gauge': gauge.tolist(), 'arms': {}}
    for name, poses in candidates.items():
        scores = []
        for pair in pairs:
            source_pose = poses[pair['source_index']]
            target_pose = eval_by_frame[pair['target_frame']]
            error = projection_errors(pair['points'], pair['target_uv'], pair['target_K'], source_pose, target_pose, pair['image_size'])
            rel = np.linalg.inv(target_pose)@source_pose
            z_pred = (pair['points']@rel[:3, :3].T+rel[:3, 3])[:, 2]
            valid = np.isfinite(pair['target_depth'])
            depth_error = np.abs(z_pred[valid]-pair['target_depth'][valid])/pair['target_depth'][valid]
            scores.append({'source_frame': pair['source_frame'], 'target_frame': pair['target_frame'],
                           'points': len(error), 'median_px': float(np.median(error)),
                           'depth_points': int(valid.sum()), 'depth_abs_rel': float(np.median(depth_error)) if valid.any() else None})
        result['arms'][name] = {'pair_median_px': float(np.median([p['median_px'] for p in scores])) if scores else None,
                                'depth_abs_rel': float(np.median([p['depth_abs_rel'] for p in scores if p['depth_abs_rel'] is not None])) if any(p['depth_abs_rel'] is not None for p in scores) else None,
                                'pairs': scores}
    initial, refined = result['arms']['Pi3X'], result['arms']['BA']
    result['screen_pass'] = (len(pairs) >= 10 and result['eval_camera_coverage'] >= 3
        and refined['pair_median_px'] <= initial['pair_median_px']*.85
        and refined['depth_abs_rel'] is not None and initial['depth_abs_rel'] is not None
        and refined['depth_abs_rel'] <= initial['depth_abs_rel']*1.1)
    (out/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    # NPZ preserves absent target depth as NaN without nonstandard JSON tokens.
    np.savez_compressed(out/'pair_observations.npz', **{f'{i}_{k}': v for i, p in enumerate(pairs) for k, v in p.items()})
    print(entry['id'], 'pairs', len(pairs), 'eval cameras', result['eval_camera_coverage'],
          'px', {k: v['pair_median_px'] for k, v in result['arms'].items()}, 'pass', result['screen_pass'], flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--id', required=True)
    ap.add_argument('--tracks', required=True)
    ap.add_argument('--candidate', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--source-exclusion-px', type=float, default=12.)
    a = ap.parse_args()
    protocol = json.loads(Path(a.protocol).read_text())
    entry = next(e for e in protocol['selection'] if e['id'] == a.id)
    evaluate(entry, a.tracks, a.candidate, Path(a.out), a.source_exclusion_px)


if __name__ == '__main__':
    main()
