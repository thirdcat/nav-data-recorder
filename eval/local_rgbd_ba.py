#!/usr/bin/env python3
"""Local pose/landmark BA pilot using native-depth samples and held-out tracks.

No reference poses enter solve(). Metric depth is a measurement, not a fixed
world point. Camera intrinsics and the first camera pose remain fixed.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.sparse import lil_matrix

from window_alignment import robust_blocks
from run_window_alignment import reference_score, projection_errors
from local_ba_diagnostics import connectivity


def solve(initial, K, track, obs_frame, uv, depth, heldout, *, use_depth=True,
          max_nfev=150, imu_rotation=None, imu_sigma_deg=2., fixed_rotation=None,
          translation_prior_sigma_m=None):
    """Optimize only training tracks. IMU rotations, if given, are camera c2w.

    IMU residuals compare adjacent relative rotations, not global yaw origins.
    Missing rotations must be represented by NaN and are not interpolated here.
    """
    initial, K = np.asarray(initial, float).copy(), np.asarray(K, float)
    track, obs_frame = np.asarray(track), np.asarray(obs_frame)
    uv, depth, heldout = np.asarray(uv, float), np.asarray(depth, float), np.asarray(heldout, bool)
    n = len(initial)
    if initial.shape != (n, 4, 4) or K.shape != (n, 3, 3) or n < 2:
        raise ValueError('invalid camera arrays')
    if not np.isfinite(initial).all() or not np.isfinite(K).all():
        raise ValueError('nonfinite cameras')
    if fixed_rotation is not None:
        fixed_rotation = np.asarray(fixed_rotation, float)
        if fixed_rotation.shape != (n, 3, 3) or not np.isfinite(fixed_rotation).all():
            raise ValueError('fixed rotations must cover every camera')
        if not np.allclose(fixed_rotation[0], initial[0, :3, :3], atol=1e-6):
            raise ValueError('fixed rotations must preserve the first camera gauge')
        initial[:, :3, :3] = fixed_rotation
    if uv.shape != (len(track), 2) or any(len(a) != len(track) for a in [obs_frame, depth, heldout]):
        raise ValueError('observation arrays differ in length')
    if not np.isfinite(uv).all() or np.any(obs_frame < 0) or np.any(obs_frame >= n):
        raise ValueError('invalid observation')
    for t in np.unique(track):
        if np.unique(heldout[track == t]).size != 1:
            raise ValueError('train/eval must split entire tracks')
    train = ~heldout
    tracks, tid = np.unique(track[train], return_inverse=True)
    f, measured_uv, measured_z = obs_frame[train], uv[train], depth[train]
    if not len(tracks):
        raise ValueError('no training tracks')
    if translation_prior_sigma_m is not None and translation_prior_sigma_m <= 0:
        raise ValueError('translation prior sigma must be positive')
    translation_prior = np.diff(initial[:, :3, 3], axis=0)
    points = []
    for t in range(len(tracks)):
        ix = np.flatnonzero((tid == t) & np.isfinite(measured_z) & (measured_z > .05))
        if not len(ix):
            raise ValueError('each training track needs a depth initialization')
        rays = np.einsum('nij,nj->ni', np.linalg.inv(K[f[ix]]), np.c_[measured_uv[ix], np.ones(len(ix))])
        cam = rays*measured_z[ix, None]
        world = np.einsum('nij,nj->ni', initial[f[ix], :3, :3], cam)+initial[f[ix], :3, 3]
        points.append(np.median(world, axis=0))
    points = np.array(points)
    pose_size = 6*(n-1)
    valid_z = np.isfinite(measured_z) & (measured_z > .05)
    safe_z = np.where(valid_z, measured_z, 1.)
    sigma_z = .03+.01*safe_z
    imu_edges, imu_relative = np.empty((0, 2), int), np.empty((0, 3, 3))
    if imu_rotation is not None:
        imu_rotation = np.asarray(imu_rotation, float)
        if imu_rotation.shape != (n, 3, 3) or imu_sigma_deg <= 0:
            raise ValueError('invalid IMU input')
        good = np.isfinite(imu_rotation).all(axis=(1, 2))
        imu_edges = np.array([(i, i+1) for i in range(n-1) if good[i] and good[i+1]], int).reshape(-1, 2)
        if len(imu_edges):
            a, b = imu_edges.T
            imu_relative = imu_rotation[a].transpose(0, 2, 1)@imu_rotation[b]

    def unpack(x):
        inc = x[:pose_size].reshape(-1, 6)
        poses = initial.copy()
        if fixed_rotation is None:
            poses[1:, :3, :3] = Rotation.from_rotvec(inc[:, :3]).as_matrix()@initial[1:, :3, :3]
        poses[1:, :3, 3] += inc[:, 3:]
        return poses, points+x[pose_size:].reshape(-1, 3)

    def residual(x):
        poses, pts = unpack(x)
        cam = np.einsum('nji,nj->ni', poses[f, :3, :3], pts[tid]-poses[f, :3, 3])
        rays = cam@np.eye(3)
        rays[:, :2] /= np.maximum(cam[:, 2:3], .05)
        rays[:, 2] = 1
        projected = np.einsum('nij,nj->ni', K[f], rays)[:, :2]
        rgb = robust_blocks((projected-measured_uv)/1.5)
        dz = robust_blocks(((cam[:, 2]-safe_z)/sigma_z)[:, None])[:, 0]
        dz *= valid_z & use_depth
        cheirality = np.minimum(cam[:, 2]-.05, 0)/.01
        measurements = np.c_[rgb, dz, cheirality].ravel()
        prior = x[:pose_size].reshape(-1, 6)/np.array([np.pi/6]*3+[.5]*3)
        extra = []
        if len(imu_edges):
            a, b = imu_edges.T
            estimated = poses[a, :3, :3].transpose(0, 2, 1)@poses[b, :3, :3]
            dr = Rotation.from_matrix(estimated@imu_relative.transpose(0, 2, 1)).as_rotvec()
            extra = robust_blocks(dr/np.radians(imu_sigma_deg)).ravel()
        translation_extra = []
        if translation_prior_sigma_m is not None:
            delta = np.diff(poses[:, :3, 3], axis=0)-translation_prior
            translation_extra = robust_blocks(delta/translation_prior_sigma_m).ravel()
        return np.r_[measurements, prior.ravel(), extra, translation_extra]

    rows = 4*len(f)+pose_size+3*len(imu_edges)
    if translation_prior_sigma_m is not None:
        rows += 3*(n-1)
    sparsity = lil_matrix((rows, pose_size+3*len(tracks)), dtype=np.int8)
    for i, (fr, tr) in enumerate(zip(f, tid)):
        if fr:
            sparsity[4*i:4*i+4, 6*(fr-1):6*fr] = 1
        sparsity[4*i:4*i+4, pose_size+3*tr:pose_size+3*tr+3] = 1
    for j in range(pose_size):
        sparsity[4*len(f)+j, j] = 1
    for i, edge in enumerate(imu_edges):
        for fr in edge:
            if fr:
                start = 4*len(f)+pose_size+3*i
                sparsity[start:start+3, 6*(fr-1):6*fr] = 1
    if translation_prior_sigma_m is not None:
        for i in range(n-1):
            start = 4*len(f)+pose_size+3*len(imu_edges)+3*i
            for fr in [i, i+1]:
                if fr:
                    sparsity[start:start+3, 6*(fr-1)+3:6*fr] = 1
    x0 = np.zeros(sparsity.shape[1])
    before = residual(x0)
    start = time.monotonic()
    opt = least_squares(residual, x0, jac_sparsity=sparsity.tocsr(), max_nfev=max_nfev,
                        x_scale='jac', ftol=1e-7, xtol=1e-7, gtol=1e-6,
                        tr_options={'atol': 1e-7, 'btol': 1e-7})
    poses, pts = unpack(opt.x)
    report = {'success': bool(opt.success), 'status': int(opt.status), 'message': str(opt.message),
              'nfev': int(opt.nfev), 'initial_cost': float(.5*before@before), 'cost': float(opt.cost),
              'optimality': float(opt.optimality), 'elapsed_s': time.monotonic()-start,
              'training_tracks': len(tracks), 'training_observations': len(f),
              'use_depth': use_depth, 'imu_edges': len(imu_edges), 'rotation_fixed': fixed_rotation is not None,
              'translation_prior_sigma_m': translation_prior_sigma_m,
              'training_graph': connectivity(n, track, obs_frame, heldout)}
    return poses, pts, report


def heldout_score(poses, K, track, obs_frame, uv, depth, heldout, image_size):
    """Anchor each unseen track at its first valid depth, then predict others."""
    records, covered = [], set()
    for t in np.unique(track[heldout]):
        obs = np.flatnonzero((track == t) & heldout)
        good = obs[np.isfinite(depth[obs]) & (depth[obs] > .05)]
        if len(good) < 2:
            continue
        anchor = good[np.argmin(obs_frame[good])]
        fa = obs_frame[anchor]
        point = np.linalg.inv(K[fa])@np.r_[uv[anchor], 1.]*depth[anchor]
        errors, relative_depth = [], []
        for ix in obs:
            if ix == anchor:
                continue
            fb = obs_frame[ix]
            errors.append(float(projection_errors(point[None], uv[ix:ix+1], K[fb], poses[fa], poses[fb], image_size)[0]))
            covered.add(int(fb))
            if np.isfinite(depth[ix]):
                rel = np.linalg.inv(poses[fb])@poses[fa]
                pred_z = (rel[:3, :3]@point+rel[:3, 3])[2]
                relative_depth.append(float(abs(pred_z-depth[ix])/depth[ix]))
        covered.add(int(fa))
        records.append({'track': int(t), 'observations': len(obs), 'median_px': float(np.median(errors)),
                        'depth_abs_rel': float(np.median(relative_depth)) if relative_depth else None})
    return {'tracks': len(records), 'frame_coverage': len(covered)/len(poses),
            'median_px': float(np.median([r['median_px'] for r in records])) if records else None,
            'depth_abs_rel': float(np.median([r['depth_abs_rel'] for r in records if r['depth_abs_rel'] is not None])) if records else None,
            'per_track': records}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('tracks')
    ap.add_argument('--out', required=True)
    ap.add_argument('--initial', help='optional alternative initialization on identical frame IDs')
    ap.add_argument('--imu', help='npz with frame_ids and camera_rotation')
    ap.add_argument('--fix-imu-rotation', action='store_true', help='fix camera rotations to IMU, aligned at the first camera')
    ap.add_argument('--translation-prior-sigma-m', type=float, help='optional soft-L1 prior on initial adjacent translation increments')
    ap.add_argument('--depth-only-arm', action='store_true', help='run RGBD only, e.g. for a separate IMU comparison')
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    with np.load(a.tracks, allow_pickle=False) as d:
        data = {k: d[k] for k in d.files}
    initial = data['initial']
    if a.initial:
        with np.load(a.initial, allow_pickle=False) as d:
            if not np.array_equal(d['frame'], data['frame_ids']):
                raise ValueError('initialization frame IDs differ')
            initial = d['estimate']
    imu = None
    if a.imu:
        with np.load(a.imu, allow_pickle=False) as d:
            if not np.array_equal(d['frame_ids'], data['frame_ids']):
                raise ValueError('IMU frame IDs differ')
            imu = d['camera_rotation']
    fixed_rotation = None
    if a.fix_imu_rotation:
        if imu is None or not np.isfinite(imu).all():
            raise ValueError('fixed IMU rotations require complete finite orientation coverage')
        fixed_rotation = (initial[0, :3, :3]@imu[0].T)@imu
        imu = None
    args = [data[k] for k in ['K', 'track', 'obs_frame', 'uv', 'depth', 'heldout']]
    results = {'tracks_sha256': hashlib.sha256(Path(a.tracks).read_bytes()).hexdigest(),
               'metadata': json.loads(str(data['metadata'])), 'arms': {},
               'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'initial_sha256': hashlib.sha256(Path(a.initial).read_bytes()).hexdigest() if a.initial else None,
               'imu_sha256': hashlib.sha256(Path(a.imu).read_bytes()).hexdigest() if a.imu else None,
               'arguments': vars(a)}
    candidates = {'B0': (initial, {'success': True, 'method': 'initial'})}
    for name, use_depth in [('B-RGB', False), ('B-RGBD', True)]:
        if a.depth_only_arm and not use_depth:
            continue
        poses, points, report = solve(initial, *args, use_depth=use_depth, imu_rotation=imu,
                                     fixed_rotation=fixed_rotation,
                                     translation_prior_sigma_m=a.translation_prior_sigma_m)
        candidates[name] = poses, report
        print(name, report, flush=True)
        np.savez_compressed(out/(name+'.npz'), frame=data['frame_ids'], t=data['t'], estimate=poses,
                            convention=data['convention'], solver_success=report['success'], landmarks=points)
    for name, (poses, report) in candidates.items():
        score = heldout_score(poses, *args, data['image_size'])
        move = np.linalg.norm(poses[:, :3, 3]-initial[:, :3, 3], axis=1)
        rotate = np.degrees(Rotation.from_matrix(poses[:, :3, :3]@initial[:, :3, :3].transpose(0, 2, 1)).magnitude())
        entry = {'solver': report, 'heldout': score,
                 'pose_change_median_m': float(np.median(move)), 'pose_change_max_m': float(move.max()),
                 'rotation_change_median_deg': float(np.median(rotate)), 'rotation_change_max_deg': float(rotate.max())}
        if len(data['reference']):
            entry['reference'] = reference_score(poses, data['reference'])
        results['arms'][name] = entry
        print(name, {k: v for k, v in score.items() if k != 'per_track'}, entry.get('reference'), flush=True)
    (out/'report.json').write_text(json.dumps(results, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
