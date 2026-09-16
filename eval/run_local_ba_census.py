#!/usr/bin/env python3
"""Run the preregistered central-block pilot on all cached ARKit/Pi3 trajectories."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--cache-root', default='pi3traj')
    ap.add_argument('--calibration', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--jobs', type=int, default=2)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    scripts = Path(__file__).resolve().parent
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    selected = []
    for cache in sorted(Path(a.cache_root).glob('*.npz')):
        if not re.fullmatch('[a-f0-9]{6}', cache.stem):
            continue
        with np.load(cache, allow_pickle=False) as z:
            if 'reference' not in z.files:
                continue
            sid = cache.stem
            block = out/sid
            block.mkdir()
            n = len(z['frame'])
            if n < 24:
                raise ValueError(f'{sid}: fewer than 24 frames')
            start = (n-24)//2
            data = {k: z[k][start:start+24] for k in ['frame', 't', 'estimate', 'reference']}
            data['convention'] = z['convention']
            np.savez_compressed(block/'pi3.npz', **data)
            np.savez_compressed(block/'arkit.npz', **dict(data, estimate=data['reference']))
        matches = list(Path(a.data_root).glob('*-'+sid))
        if len(matches) != 1:
            raise ValueError(f'{sid}: session not unique')
        selected.append({'session': str(matches[0]), 'id': sid, 'cache': str(cache),
                         'cache_sha256': hashlib.sha256(cache.read_bytes()).hexdigest(),
                         'start_index': start, 'frames': 24})
    (out/'selection.json').write_text(json.dumps(selected, indent=2)+'\n')

    def run(entry):
        block = out/entry['id']
        commands = [
            [str(scripts/'build_rgbd_tracks.py'), entry['session'], '--poses', str(block/'pi3.npz'), '--out', str(block/'tracks.npz')],
            [str(scripts/'imu_rotation_calibration.py'), 'dump', entry['session'], '--calibration', a.calibration, '--tracks', str(block/'tracks.npz'), '--out', str(block/'imu.npz')],
        ]
        base = [str(scripts/'local_rgbd_ba.py'), str(block/'tracks.npz'), '--depth-only-arm']
        imu = ['--imu', str(block/'imu.npz'), '--fix-imu-rotation']
        commands += [base+['--out', str(block/'rgbd')],
                     base+imu+['--out', str(block/'fixed_imu')],
                     base+imu+['--initial', str(block/'arkit.npz'), '--out', str(block/'arkit_control')]]
        history = []
        for i, args in enumerate(commands):
            cmd = [sys.executable, '-u']+args
            with (block/f'{i}.log').open('w') as log:
                result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            history.append({'argv': cmd, 'returncode': result.returncode})
            if result.returncode:
                break
        (block/'commands.json').write_text(json.dumps(history, indent=2)+'\n')
        return {'id': entry['id'], 'completed_stages': len(history), 'success': len(history) == len(commands) and history[-1]['returncode'] == 0}

    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        results = []
        for result in pool.map(run, selected):
            results.append(result)
            print(json.dumps(result), flush=True)
    (out/'execution.json').write_text(json.dumps(results, indent=2)+'\n')
    if not all(r['success'] for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
