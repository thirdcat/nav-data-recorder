#!/usr/bin/env python3
"""Inverse-depth interpolation sensitivity with the original ARKit track mask frozen."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from build_rgbd_tracks import sample_depth
sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'tools'))
from read_session import Session


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('session')
    ap.add_argument('--tracks', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise ValueError('output exists')
    with np.load(a.tracks, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    metadata = json.loads(str(d['metadata']))
    if metadata['lens'] != 'arkit':
        raise ValueError('this sensitivity tool handles ARKit registered depth only')
    s = Session(a.session)
    entries = {int(r['frame']): r for r in s.stream('depth')}
    new_depth = np.full_like(d['depth'], np.nan)
    w, h = d['image_size']
    for i, frame in enumerate(d['frame_ids']):
        entry = entries[int(frame)]
        kd = d['K'][i].copy()
        kd[0] *= entry['width']/w
        kd[1] *= entry['height']/h
        obs = (d['obs_frame'] == i) & np.isfinite(d['depth'])
        new_depth[obs] = sample_depth(d['uv'][obs], d['K'][i], np.asarray(s.depth_frame(entry), float),
                                      kd, np.asarray(s.confidence_frame(entry)), method='inverse_bilinear')
    if not np.array_equal(np.isfinite(new_depth), np.isfinite(d['depth'])):
        raise ValueError('interpolation changed the frozen validity mask')
    delta = np.abs(new_depth-d['depth'])
    metadata['depth_sampling'] = 'inverse_bilinear; original track and depth-validity mask frozen'
    metadata['parent_tracks_sha256'] = hashlib.sha256(Path(a.tracks).read_bytes()).hexdigest()
    metadata['resampling_code_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    metadata['sampling_change_median_m'] = float(np.nanmedian(delta))
    metadata['sampling_change_p95_m'] = float(np.nanpercentile(delta, 95))
    d['depth'], d['metadata'] = new_depth, json.dumps(metadata)
    np.savez_compressed(out, **d)
    print(metadata['session'], metadata['sampling_change_median_m'], metadata['sampling_change_p95_m'])


if __name__ == '__main__':
    main()
