#!/usr/bin/env python3
"""Compare identical RGBD endpoints through 5/10/30 Hz LK paths.

This measures correspondence availability, not sharpness or GS appearance.
ARKit/depth agreement is a correlated consistency check, not independent GT.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from build_rgbd_tracks import sample_depth
sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'tools'))
from read_session import Session


def follow(images, points, indices):
    """Keep original point IDs and reject each forward/backward inconsistent step."""
    xy = np.asarray(points, np.float32).copy()
    valid = np.ones(len(xy), bool)
    params = dict(winSize=(31, 31), maxLevel=5,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
    for a, b in zip(indices[:-1], indices[1:]):
        ix = np.flatnonzero(valid)
        if not len(ix):
            break
        source = xy[ix].reshape(-1, 1, 2)
        dest, ok, _ = cv2.calcOpticalFlowPyrLK(images[a], images[b], source, None, **params)
        back, ok_back, _ = cv2.calcOpticalFlowPyrLK(images[b], images[a], dest, None, **params)
        fb = np.linalg.norm(back[:, 0]-source[:, 0], axis=1)
        h, w = images[b].shape
        p = dest[:, 0]
        good = (ok[:, 0] != 0) & (ok_back[:, 0] != 0) & (fb < 1.)
        good &= np.isfinite(p).all(axis=1) & (p[:, 0] > 2) & (p[:, 0] < w-3) & (p[:, 1] > 2) & (p[:, 1] < h-3)
        xy[ix] = p
        valid[ix] = good
    return xy, valid


def pose(row):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([row[k] for k in ['qx', 'qy', 'qz', 'qw']]).as_matrix()@np.diag([1., -1., -1.])
    T[:3, 3] = [row[k] for k in ['tx', 'ty', 'tz']]
    return T


def run(session_path):
    session = Session(str(session_path))
    rows = list(session.stream('frames'))
    poses = {int(r['frame']): r for r in session.stream('pose')}
    depths = {int(r['frame']): r for r in session.stream('depth')}
    times = np.array([r['t'] for r in rows])
    first = int(np.searchsorted(times, times[0]+3.))
    records, hashes = [], {}
    for start in range(first, first+8*12, 12):
        selected = rows[start:start+13]
        if len(selected) != 13:
            records.append({'start_index': start, 'skip': 'recording ended'})
            continue
        if np.max(np.abs(np.diff([r['t'] for r in selected])-1/30)) > .005:
            records.append({'start_index': start, 'skip': 'not contiguous 30 Hz'})
            continue
        ids = [int(r['frame']) for r in selected]
        if any(f not in poses or poses[f].get('tracking') != 'normal' for f in [ids[0], ids[-1]]) or ids[0] not in depths:
            records.append({'start_index': start, 'skip': 'missing normal pose or source depth'})
            continue
        images, Ks = [], []
        for r in selected:
            path = Path(session.frame_path(r))
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise ValueError(f'cannot decode {path}')
            h, w = gray.shape
            images.append(cv2.resize(gray, (960, round(h*960/w)), interpolation=cv2.INTER_AREA))
            hashes[str(r['frame'])] = hashlib.sha256(path.read_bytes()).hexdigest()
            cr = poses[int(r['frame'])]
            K = np.array([[cr['fx'], 0, cr['cx']], [0, cr['fy'], cr['cy']], [0, 0, 1.]])
            K[0] *= 960/w
            K[1] *= images[-1].shape[0]/h
            Ks.append(K)
        corners = cv2.goodFeaturesToTrack(images[0], maxCorners=1200, qualityLevel=.01, minDistance=12, blockSize=7)
        uv = np.empty((0, 2)) if corners is None else corners[:, 0]
        dr = depths[ids[0]]
        kd = Ks[0].copy()
        kd[0] *= dr['width']/images[0].shape[1]
        kd[1] *= dr['height']/images[0].shape[0]
        confidence = session.confidence_frame(dr)
        if confidence is None:
            raise ValueError('missing source depth confidence')
        z = sample_depth(uv, Ks[0], np.asarray(session.depth_frame(dr), float), kd, np.asarray(confidence))
        good = np.isfinite(z)
        uv, z = uv[good], z[good]
        X = (np.c_[uv, np.ones(len(uv))]@np.linalg.inv(Ks[0]).T)*z[:, None]
        relative = np.linalg.inv(pose(poses[ids[-1]]))@pose(poses[ids[0]])
        X = X@relative[:3, :3].T+relative[:3, 3]
        projected = X@Ks[-1].T
        projected = projected[:, :2]/np.maximum(projected[:, 2:], 1e-8)
        record = {'start_index': start, 'source_frame': ids[0], 'target_frame': ids[-1],
                  'duration_s': selected[-1]['t']-selected[0]['t'], 'source_points': len(uv), 'arms': {}}
        for hz, stride in [(5, 6), (10, 3), (30, 1)]:
            indices = list(range(0, 13, stride))
            dest, okay = follow(images, uv, indices)
            back, okay_back = follow(images, dest, indices[::-1])
            okay &= okay_back & (np.linalg.norm(back-uv, axis=1) < 1.)
            error = np.linalg.norm(dest-projected, axis=1)
            consistent = okay & (X[:, 2] > .05) & (error <= 3.)
            record['arms'][str(hz)] = {'retained': int(okay.sum()), 'consistent': int(consistent.sum()),
                'retained_fraction': float(okay.mean()) if len(uv) else None,
                'consistent_fraction': float(consistent.mean()) if len(uv) else None,
                'retained_median_error_px': float(np.median(error[okay])) if okay.any() else None}
        records.append(record)
    pairs = [r for r in records if 'arms' in r and r['source_points'] > 0]
    delta = [r['arms']['30']['consistent_fraction']-r['arms']['5']['consistent_fraction'] for r in pairs]
    return {'session': Path(session_path).name, 'pairs': records, 'images_sha256': hashes,
            'stream_sha256': {n: hashlib.sha256((Path(session_path)/n).read_bytes()).hexdigest() for n in ['pose.jsonl', 'frames.jsonl', 'depth.jsonl']},
            'valid_pairs': len(pairs), 'source_points': sum(r['source_points'] for r in pairs),
            'median_30_minus_5_pp': float(100*np.median(delta)) if delta else None,
            'median_consistent_fraction': {str(hz): float(np.median([r['arms'][str(hz)]['consistent_fraction'] for r in pairs])) if pairs else None for hz in [5, 10, 30]}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('session')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise ValueError('output exists')
    cv2.setNumThreads(1)
    cv2.setRNGSeed(42)
    result = run(a.session)
    result['code_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['pairs', 'images_sha256']}, indent=2))


if __name__ == '__main__':
    main()
