#!/usr/bin/env python3
"""Cross-session camera/device relative-rotation calibration for an IMU pilot.

Fits rotation axes on one ARKit session, with a fixed held-out pair split.
Does not fit time offset, scale, translation, or a per-session extrinsic.
CMDeviceMotion and ARKit are processed, correlated sensor estimates.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def motion_at(session, times, transpose=False):
    rows = read_rows(Path(session)/'motion.jsonl')
    t = np.array([r['t'] for r in rows])
    q = np.array([[r[k] for k in ['qx', 'qy', 'qz', 'qw']] for r in rows])
    unique, index = np.unique(t, return_index=True)
    valid = (times >= unique[0]) & (times <= unique[-1])
    result = np.full((len(times), 3, 3), np.nan)
    result[valid] = Slerp(unique, Rotation.from_quat(q[index]))(times[valid]).as_matrix()
    if transpose:
        result = result.transpose(0, 2, 1)
    return result, valid


def rotation_pairs(session, transpose=False):
    rows = [r for r in read_rows(Path(session)/'pose.jsonl') if r.get('tracking') == 'normal']
    times, ix = np.unique([r['t'] for r in rows], return_index=True)
    rows = [rows[i] for i in ix]
    R = Rotation.from_quat([[r[k] for k in ['qx', 'qy', 'qz', 'qw']] for r in rows]).as_matrix()
    R = R@np.diag([1., -1., -1.])
    # Pair targets are chosen by elapsed time, not raw pose-array index/rate.
    starts = np.arange(0, len(times), 6)
    ends = np.searchsorted(times, times[starts]+.2)
    okay = ends < len(times)
    starts, ends = starts[okay], ends[okay]
    okay = abs((times[ends]-times[starts])-.2) < .02
    starts, ends = starts[okay], ends[okay]
    Q, valid = motion_at(session, times, transpose)
    okay = valid[starts] & valid[ends]
    starts, ends = starts[okay], ends[okay]
    device = Q[starts].transpose(0, 2, 1)@Q[ends]
    camera = R[starts].transpose(0, 2, 1)@R[ends]
    angle = np.degrees(Rotation.from_matrix(device).magnitude())
    okay = (angle >= 1) & (angle <= 45)
    return device[okay], camera[okay]


def fit_axes(device, camera):
    """Camera-frame rotvec = C * device-frame rotvec."""
    a, b = Rotation.from_matrix(device).as_rotvec(), Rotation.from_matrix(camera).as_rotvec()
    U, singular, Vt = np.linalg.svd(a.T@b)
    C = Vt.T@np.diag([1., 1., np.sign(np.linalg.det(Vt.T@U.T))])@U.T
    return C, singular


def score(C, device, camera):
    if not len(device):
        return {'pairs': 0, 'median_deg': None, 'p95_deg': None}
    error = (C@device@C.T)@camera.transpose(0, 2, 1)
    angle = np.degrees(Rotation.from_matrix(error).magnitude())
    return {'pairs': len(angle), 'median_deg': float(np.median(angle)), 'p95_deg': float(np.percentile(angle, 95))}


def calibrate(session):
    candidates = []
    for transpose in [False, True]:
        device, camera = rotation_pairs(session, transpose)
        train = np.arange(len(device)) % 5 != 0
        if train.sum() < 10:
            raise ValueError('too few training rotation pairs')
        C, singular = fit_axes(device[train], camera[train])
        candidates.append({'transpose_attitude': transpose, 'camera_from_device': C.tolist(),
                           'singular_values': singular.tolist(), 'train': score(C, device[train], camera[train]),
                           'heldout': score(C, device[~train], camera[~train])})
    result = min(candidates, key=lambda c: c['train']['median_deg']).copy()
    result['training_session'] = Path(session).name
    result['candidates'] = candidates
    result['time_offset_s'] = 0.
    result['inputs_sha256'] = {n: hashlib.sha256((Path(session)/n).read_bytes()).hexdigest() for n in ['pose.jsonl', 'motion.jsonl']}
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    c = sub.add_parser('calibrate')
    c.add_argument('session')
    c.add_argument('--out', required=True)
    s = sub.add_parser('score')
    s.add_argument('session')
    s.add_argument('--calibration', required=True)
    s.add_argument('--out', required=True)
    d = sub.add_parser('dump')
    d.add_argument('session')
    d.add_argument('--calibration', required=True)
    d.add_argument('--tracks', required=True)
    d.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        raise ValueError('output exists')
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.command == 'calibrate':
        result = calibrate(a.session)
    else:
        calibration = json.loads(Path(a.calibration).read_text())
        C = np.array(calibration['camera_from_device'])
        transpose = calibration['transpose_attitude']
        if a.command == 'score':
            dev, cam = rotation_pairs(a.session, transpose)
            result = {'session': Path(a.session).name, **score(C, dev, cam),
                      'calibration_sha256': hashlib.sha256(Path(a.calibration).read_bytes()).hexdigest()}
        else:
            with np.load(a.tracks, allow_pickle=False) as z:
                ids, times = z['frame_ids'], z['t']
            Q, valid = motion_at(a.session, times, transpose)
            np.savez_compressed(out, frame_ids=ids, t=times, camera_rotation=Q@C.T, valid=valid,
                                calibration_sha256=hashlib.sha256(Path(a.calibration).read_bytes()).hexdigest())
            print('saved', out, 'valid frames', int(valid.sum()), '/', len(valid))
            return
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
