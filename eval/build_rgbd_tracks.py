#!/usr/bin/env python3
"""Build a local RGB/depth track cache with a track-wise train/eval split."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from uw_selfcal import track
from pi3_poseless import depth_calibration, intrinsics_for, active_hfov, load_calibration

sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'tools'))
from read_session import Session


def sample_depth(uv, K, z, K_depth, confidence=None, method='floor'):
    """Sample native depth along RGB rays for co-located camera axes.

    Depth K may differ from RGB K. This function is for ARKit or the wide
    depth camera, not the separate ultra-wide camera with a rig baseline.
    """
    rays = np.c_[uv, np.ones(len(uv))] @ np.linalg.inv(K).T
    pixels = rays @ K_depth.T
    pixels = pixels[:, :2]/pixels[:, 2:3]
    u, v = np.floor(pixels).astype(int).T
    inside = (u >= 1) & (u < z.shape[1]-1) & (v >= 1) & (v < z.shape[0]-1)
    u, v = np.clip(u, 1, z.shape[1]-2), np.clip(v, 1, z.shape[0]-2)
    patch = np.array([z[v+dv, u+du] for dv in [-1, 0, 1] for du in [-1, 0, 1]])
    valid = inside & np.isfinite(patch).all(0) & (patch.min(0) > .2) & (patch.max(0) < 5.)
    valid &= np.ptp(patch, axis=0) < .1
    if confidence is not None:
        valid &= confidence[v, u] >= 2
    depth = z[v, u].astype(float)
    if method == 'inverse_bilinear':
        dx, dy = pixels[:, 0]-u, pixels[:, 1]-v
        corners = np.array([z[v, u], z[v, u+1], z[v+1, u], z[v+1, u+1]], float)
        weights = np.array([(1-dx)*(1-dy), dx*(1-dy), (1-dx)*dy, dx*dy])
        safe = np.where(np.isfinite(corners) & (corners > .05), corners, 1.)
        inverse = np.sum(weights/safe, axis=0)
        depth = 1./np.maximum(inverse, 1e-8)
    elif method != 'floor':
        raise ValueError('unknown depth sampling method')
    return np.where(valid, depth, np.nan)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('session')
    ap.add_argument('--poses', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--lens', choices=['arkit', 'wide'], default='arkit')
    ap.add_argument('--width', type=int, default=960)
    ap.add_argument('--max-tracks', type=int, default=600)
    ap.add_argument('--allow-missing-confidence', action='store_true')
    a = ap.parse_args(argv)
    output = Path(a.out)
    if output.exists():
        raise ValueError('output exists; use a new path')
    cv2.setNumThreads(1)
    cv2.setRNGSeed(42)
    session = Session(a.session)
    image_rows = {int(r['frame']): r for r in session.stream('frames' if a.lens == 'arkit' else 'frames_wide')}
    camera_rows = {int(r['frame']): r for r in session.stream('pose')}
    depth_rows = list(session.stream('depth'))
    by_depth = {int(r['frame']): r for r in depth_rows}
    depth_times = np.array([r['t'] for r in depth_rows])
    with np.load(a.poses, allow_pickle=False) as p:
        ids, initial = p['frame'].copy(), p['estimate'].copy()
        times = p['t'].copy()
        reference = p['reference'].copy() if 'reference' in p else np.empty((0, 4, 4))
        convention = str(p['convention'])
    if len(ids) < 3 or len(set(ids.tolist())) != len(ids):
        raise ValueError('at least three unique frames are required')
    if np.any(np.diff(times) <= 0):
        raise ValueError('pose times must increase')
    if a.lens == 'wide':
        manifest = session.manifest
        calib = load_calibration(Path(__file__).resolve().parent.parent/'calib/iphone17-1_wide.json')
        hfov = active_hfov(manifest, 'wide')
        dc, source = depth_calibration(manifest)
        if 'depth_calibration' not in manifest or hfov is None:
            raise ValueError('wide pilot requires logged RGB FOV and depth intrinsics')
    images, Ks, depths, Kds, confidences, hashes = [], [], [], [], [], {}
    for f, t in zip(ids, times):
        row = image_rows[int(f)]
        if abs(float(row['t'])-t) > 1e-4:
            raise ValueError(f'image and pose times disagree at {f}')
        path = Path(session.frame_path(row))
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f'cannot read {path}')
        h, w = image.shape
        size = (a.width, round(h*a.width/w))
        images.append(cv2.resize(image, size, interpolation=cv2.INTER_AREA))
        hashes[str(f)] = hashlib.sha256(path.read_bytes()).hexdigest()
        if a.lens == 'arkit':
            cr = camera_rows[int(f)]
            K = np.array([[cr['fx'], 0, cr['cx']], [0, cr['fy'], cr['cy']], [0, 0, 1.]])
            K[0] *= size[0]/w
            K[1] *= size[1]/h
            entry = by_depth[int(f)]
            Kd = K.copy()
            Kd[0] *= entry['width']/size[0]
            Kd[1] *= entry['height']/size[1]
        else:
            K = intrinsics_for(calib, *size, hfov)
            ix = int(np.argmin(abs(depth_times-t)))
            entry = depth_rows[ix]
            if abs(entry['t']-t) > .05:
                raise ValueError('depth time gap exceeds 50 ms')
            Kd = np.array([[dc['fx'], 0, dc['cx']], [0, dc['fy'], dc['cy']], [0, 0, 1.]])
            Kd[0] *= entry['width']/dc['width']
            Kd[1] *= entry['height']/dc['height']
        z = np.asarray(session.depth_frame(entry), dtype=float)
        conf = session.confidence_frame(entry)
        if conf is None and not a.allow_missing_confidence:
            raise ValueError('missing depth confidence requires an explicit flag')
        Ks.append(K)
        depths.append(z)
        Kds.append(Kd)
        confidences.append(None if conf is None else np.asarray(conf))
    tid, frame, uv = track(images, max_corners=1200, min_distance=12,
                           fb_tolerance=1., min_length=3)
    z = np.full(len(tid), np.nan)
    for i in range(len(ids)):
        keep = frame == i
        z[keep] = sample_depth(uv[keep], Ks[i], depths[i], Kds[i], confidences[i])
    good_depth = np.bincount(tid[np.isfinite(z)], minlength=int(tid.max())+1) if len(tid) else np.array([])
    eligible = np.flatnonzero(good_depth >= 2)
    rng = np.random.default_rng(42)
    eligible = rng.permutation(eligible)[:a.max_tracks]
    n_eval = int(np.ceil(len(eligible)*.2))
    eval_ids = set(eligible[:n_eval].tolist())
    chosen = np.isin(tid, eligible)
    original_tid = tid[chosen]
    _, tid = np.unique(original_tid, return_inverse=True)
    frame, uv, z = frame[chosen], uv[chosen], z[chosen]
    test = np.array([int(k) in eval_ids for k in original_tid])
    n_tracks = len(eligible)
    if not n_tracks:
        raise ValueError('no track has two reliable depth samples')
    meta = {'session': Path(a.session).name, 'lens': a.lens, 'width': a.width,
            'frames': len(ids), 'tracks': n_tracks, 'eval_tracks': n_eval,
            'observations': len(tid), 'valid_depth_observations': int(np.isfinite(z).sum()),
            'split': 'seed 42 whole-track permutation; first ceil(20%) held out',
            'image_sha256': hashes, 'poses_sha256': hashlib.sha256(Path(a.poses).read_bytes()).hexdigest(),
            'missing_confidence_allowed': a.allow_missing_confidence,
            'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'tracker_sha256': hashlib.sha256(Path(__file__).with_name('uw_selfcal.py').read_bytes()).hexdigest()}
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, frame_ids=ids, t=times, initial=initial, reference=reference,
                        convention=convention, K=np.array(Ks), track=tid, obs_frame=frame,
                        uv=uv, depth=z, heldout=test, image_size=images[0].shape[::-1],
                        metadata=json.dumps(meta))
    print(json.dumps({k: v for k, v in meta.items() if k != 'image_sha256'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
