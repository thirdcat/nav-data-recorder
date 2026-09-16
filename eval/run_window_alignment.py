#!/usr/bin/env python3
"""Run the R2-P1 controls and fixed-scale global alignment on one cache.

python3 eval/run_window_alignment.py SESSION --cache pi3win/ID.npz \
    --reference pi3traj/ID.npz --out logs/r2_p1_20260909/ID

Reference poses are evaluated only after all candidate trajectories are built.
Image correspondences never enter window alignment. They are a separate
measurement check, not unseen end-to-end ground truth: Pi3X saw these images.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import scipy
from scipy.spatial.transform import Rotation

from window_alignment import prepare, sequential, align_windows

sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'tools'))
from read_session import Session


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference_score(estimate, reference):
    """One global rigid fit; no similarity scale or per-window refitting."""
    src, dst = estimate[:, :3, 3], reference[:, :3, 3]
    a, b = src-src.mean(0), dst-dst.mean(0)
    U, _, Vt = np.linalg.svd(a.T@b)
    R = (U@np.diag([1, 1, np.sign(np.linalg.det(U@Vt))])@Vt).T
    t = dst.mean(0)-R@src.mean(0)
    error = np.linalg.norm(src@R.T+t-dst, axis=1)
    relative = (R@estimate[:, :3, :3]) @ reference[:, :3, :3].transpose(0, 2, 1)
    angle = np.degrees(Rotation.from_matrix(relative).magnitude())
    return {"mean_position_m": float(error.mean()), "rmse_position_m": float(np.sqrt(np.mean(error**2))),
            "median_rotation_deg": float(np.median(angle)), "p95_rotation_deg": float(np.percentile(angle, 95)),
            "endpoint_distance_m": float(np.linalg.norm(src[-1]-src[0])),
            "path_length_m": float(np.linalg.norm(np.diff(src, axis=0), axis=1).sum())}


def projection_errors(points, target_uv, K, source_pose, target_pose, size):
    """Keep the same points for every arm, including failed projections."""
    relative = np.linalg.inv(target_pose)@source_pose
    cam = points@relative[:3, :3].T+relative[:3, 3]
    positive = np.isfinite(cam).all(1) & (cam[:, 2] > .05)
    diagonal = float(np.linalg.norm(size))
    errors = np.full(len(points), diagonal)
    homogeneous = cam[positive]@K.T
    uv = homogeneous[:, :2]/homogeneous[:, 2:3]
    errors[positive] = np.minimum(np.linalg.norm(uv-target_uv[positive], axis=1), diagonal)
    return errors


def image_pairs(session, ids):
    """Fixed temporal proposals, mutual SIFT ratio matches, F RANSAC.

    Coordinates/K at width 640, with depth read on its native grid. Matching
    and depth validity use no candidate or reference extrinsics. Both endpoint
    depth samples must be valid, confident and locally away from depth edges.
    """
    cv2.setNumThreads(1)
    cv2.setRNGSeed(42)
    images = {int(r['frame']): r for r in session.stream('frames')}
    intrinsic_rows = {int(r['frame']): r for r in session.stream('pose')}
    depth_rows = {int(r['frame']): r for r in session.stream('depth')}
    sif = cv2.SIFT_create(nfeatures=2500)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    loaded = {}

    def load(f):
        if f in loaded:
            return loaded[f]
        row, p = images[f], intrinsic_rows[f]
        path = session.frame_path(row)
        grey = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if grey is None:
            raise ValueError(f'cannot decode frame {f}')
        h, w = grey.shape
        size = (640, round(h*640/w))
        grey = cv2.resize(grey, size, interpolation=cv2.INTER_AREA)
        keys, desc = sif.detectAndCompute(grey, None)
        uv = np.array([k.pt for k in keys], dtype=float).reshape(-1, 2)
        K = np.array([[p['fx']*640/w, 0, p['cx']*640/w],
                      [0, p['fy']*size[1]/h, p['cy']*size[1]/h], [0, 0, 1.]])
        entry = depth_rows[f]
        z = np.asarray(session.depth_frame(entry), dtype=float)
        confidence = session.confidence_frame(entry)
        if confidence is None:
            raise ValueError('this ARKit pilot requires depth confidence')
        confidence = np.asarray(confidence)
        u = np.clip((uv[:, 0]*z.shape[1]/size[0]).astype(int), 1, z.shape[1]-2)
        v = np.clip((uv[:, 1]*z.shape[0]/size[1]).astype(int), 1, z.shape[0]-2)
        depth = z[v, u]
        patch = np.array([z[v+dv, u+du] for dv in [-1, 0, 1] for du in [-1, 0, 1]])
        valid = np.isfinite(patch).all(0) & (patch.min(0) > .2) & (patch.max(0) < 5.)
        valid &= (confidence[v, u] >= 2) & (np.ptp(patch, axis=0) < .1)
        points = (np.c_[uv, np.ones(len(uv))]@np.linalg.inv(K).T)*depth[:, None]
        loaded[f] = {'uv': uv, 'desc': desc, 'valid': valid, 'points': points,
                     'K': K, 'size': size, 'file_sha256': sha256(path)}
        return loaded[f]

    proposals = set()
    for a in np.linspace(0, len(ids)-1, 16, dtype=int):
        for gap in [5, 15, 50]:
            if a+gap < len(ids):
                proposals.add((int(ids[a]), int(ids[a+gap])))
    for a in [0, 2, 4]:
        for b in [len(ids)-5, len(ids)-3, len(ids)-1]:
            if a < b:
                proposals.add((int(ids[a]), int(ids[b])))
    pairs, rejected = [], []
    for fa, fb in sorted(proposals):
        A, B = load(fa), load(fb)
        if A['desc'] is None or B['desc'] is None:
            rejected.append({'a': fa, 'b': fb, 'reason': 'no descriptors'})
            continue

        def ratios(a, b):
            return {m.queryIdx: m.trainIdx for candidates in matcher.knnMatch(a, b, k=2)
                    if len(candidates) == 2 for m, n in [candidates] if m.distance < .75*n.distance}

        forward, reverse = ratios(A['desc'], B['desc']), ratios(B['desc'], A['desc'])
        matches = [(i, j) for i, j in forward.items() if reverse.get(j) == i]
        if len(matches) < 20:
            rejected.append({'a': fa, 'b': fb, 'reason': 'fewer than 20 mutual matches'})
            continue
        ia, ib = np.array(matches).T
        _, mask = cv2.findFundamentalMat(A['uv'][ia], B['uv'][ib], cv2.FM_RANSAC, 1.5, .999)
        if mask is None:
            rejected.append({'a': fa, 'b': fb, 'reason': 'F failed'})
            continue
        keep = mask.ravel().astype(bool) & A['valid'][ia] & B['valid'][ib]
        ia, ib = ia[keep], ib[keep]
        if len(ia) < 20:
            rejected.append({'a': fa, 'b': fb, 'reason': 'fewer than 20 confident depth inliers'})
            continue
        pairs.append({'a': fa, 'b': fb, 'uv_a': A['uv'][ia], 'uv_b': B['uv'][ib],
                      'points_a': A['points'][ia], 'points_b': B['points'][ib],
                      'K_a': A['K'], 'K_b': B['K'], 'size_a': A['size'], 'size_b': B['size']})
    return pairs, {'proposed': len(proposals), 'accepted': len(pairs), 'rejected': rejected,
                   'image_hashes': {str(f): r['file_sha256'] for f, r in loaded.items()}}


def score_pairs(pairs, by_frame):
    scores = []
    for p in pairs:
        a, b = by_frame[p['a']], by_frame[p['b']]
        forward = projection_errors(p['points_a'], p['uv_b'], p['K_b'], a, b, p['size_b'])
        reverse = projection_errors(p['points_b'], p['uv_a'], p['K_a'], b, a, p['size_a'])
        errors = np.concatenate([forward, reverse])
        scores.append({'a': p['a'], 'b': p['b'], 'matches': len(forward),
                       'median_px': float(np.median(errors)), 'p90_px': float(np.percentile(errors, 90))})
    return {'pairs': scores, 'pair_median_px': float(np.median([s['median_px'] for s in scores])) if scores else None}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('session')
    ap.add_argument('--cache', required=True)
    ap.add_argument('--reference', required=True)
    ap.add_argument('--out', required=True, help='new output directory; existing paths refused')
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    with np.load(args.cache, allow_pickle=False) as z:
        frames, poses = prepare(z['frames'], z['pred'], z['scale'])
    report = {'session': Path(args.session).name, 'cache_sha256': sha256(args.cache),
              'reference_sha256': sha256(args.reference), 'arms': {},
              'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                              'scipy': scipy.__version__, 'opencv': cv2.__version__},
              'code_sha256': {p.name: sha256(p) for p in [Path(__file__), Path(__file__).with_name('window_alignment.py')]}}
    candidates = {}
    for name, rotation in [('W0', False), ('WR', True)]:
        ids, estimate, transforms = sequential(frames, poses, rotation=rotation)
        candidates[name] = (ids, estimate, transforms)
        report['arms'][name] = {'solver': {'success': True, 'method': 'sequential'}}
    for name, robust in [('WG-L', False), ('WG-R', True)]:
        ids, estimate, transforms, solver = align_windows(frames, poses, robust=robust)
        candidates[name] = (ids, estimate, transforms)
        report['arms'][name] = {'solver': solver}
        print(name, solver, flush=True)
    # Evaluation references enter only here, after every arm has been computed.
    with np.load(args.reference, allow_pickle=False) as z:
        if str(z['convention']) != 'world_from_camera, +Z forward +Y down (depth frame)':
            raise ValueError('unexpected reference coordinate convention')
        ref = {int(f): T for f, T in zip(z['frame'], z['reference'])}
        timestamps = {int(f): float(t) for f, t in zip(z['frame'], z['t'])}
        convention = str(z['convention'])
    if set(ids) != set(ref):
        raise ValueError('cache and reference frame sets must match exactly')
    for name, (f, estimate, transforms) in candidates.items():
        assert np.array_equal(f, ids)
        report['arms'][name]['reference'] = reference_score(estimate, np.array([ref[int(k)] for k in f]))
        np.savez_compressed(out/(name+'.npz'), frame=f, t=[timestamps[int(k)] for k in f],
                            estimate=estimate, window_transform=transforms, convention=convention,
                            solver_success=report['arms'][name]['solver']['success'])
        print(name, report['arms'][name]['reference'], flush=True)
    session = Session(args.session)
    pairs, pair_report = image_pairs(session, ids)
    # Save the exact correspondence set, including rejected-pair accounting.
    (out/'correspondences.json').write_text(json.dumps(pairs, default=lambda x: x.tolist(), indent=2)+'\n')
    report['image_validation'] = pair_report
    for name, (f, estimate, _) in candidates.items():
        report['arms'][name]['image'] = score_pairs(pairs, dict(zip(f.tolist(), estimate)))
        print(name, 'image pair median px', report['arms'][name]['image']['pair_median_px'], flush=True)
    report['arkit_image_control'] = score_pairs(pairs, ref)
    report['elapsed_s'] = time.monotonic()-started
    (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print('saved', out/'report.json', 'pairs', len(pairs), 'seconds', round(report['elapsed_s'], 2), flush=True)


if __name__ == '__main__':
    main()
