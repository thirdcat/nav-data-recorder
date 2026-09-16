#!/usr/bin/env python3
"""Cache Pi3X from an explicit training-photo allowlist, without held-out RGB/depth.

Run in the GPU evaluation environment. The resulting cache includes dense
depth/confidence and model/input provenance, unlike the legacy pose-only cache.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'tools'))
from read_session import Session


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_split(entry):
    train, test, all_ids = [entry[k] for k in ['train_frame_ids', 'eval_frame_ids', 'all_frame_ids']]
    if len(train) != len(set(train)) or len(test) != len(set(test)):
        raise ValueError('duplicate frame in split')
    if set(train) & set(test) or set(train) | set(test) != set(all_ids):
        raise ValueError('train/eval split must partition the block')
    if not train or not test:
        raise ValueError('both training and evaluation frames are required')


def load_training(entry, directory):
    """Only frames on the allowlist are decoded or have depth loaded."""
    validate_split(entry)
    session = Session(entry['session'])
    images = {int(r['frame']): r for r in session.stream('frames')}
    poses = {int(r['frame']): r for r in session.stream('pose')}
    depths = {int(r['frame']): r for r in session.stream('depth')}
    arrays, Ks, times, hashes = [], [], [], {}
    for order, frame in enumerate(entry['train_frame_ids']):
        row, camera, depth = images[frame], poses[frame], depths[frame]
        if abs(row['t']-camera['t']) > 1e-4 or abs(row['t']-depth['t']) > 1e-4:
            raise ValueError('training RGB/pose/depth time mismatch')
        path = Path(session.frame_path(row))
        shutil.copyfile(path, directory/f'{order:04d}.jpg')
        hashes[str(frame)] = digest(path)
        z = np.asarray(session.depth_frame(depth), np.float32)
        confidence = session.confidence_frame(depth)
        if confidence is None:
            raise ValueError('missing training depth confidence')
        z = np.where(np.isfinite(z) & (z > .2) & (z < 5) & (np.asarray(confidence) >= 2), z, 0.)
        arrays.append(cv2.resize(z, (row['width'], row['height']), interpolation=cv2.INTER_NEAREST))
        Ks.append([[camera['fx'], 0, camera['cx']], [0, camera['fy'], camera['cy']], [0, 0, 1.]])
        times.append(row['t'])
    if np.any(np.diff(times) <= 0):
        raise ValueError('training timestamps must increase')
    return np.stack(arrays), np.array(Ks), np.array(times), hashes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--model-root', required=True)
    ap.add_argument('--weights', required=True)
    a = ap.parse_args()
    protocol = json.loads(Path(a.protocol).read_text())
    for entry in protocol['selection']:
        validate_split(entry)
    out = Path(a.out)
    out.mkdir(exist_ok=False)
    sys.path.insert(0, a.model_root)
    import torch
    from pi3.models.pi3x import Pi3X
    from pi3.utils.basic import load_multimodal_data

    torch.manual_seed(protocol['seed'])
    np.random.seed(protocol['seed'])
    cv2.setNumThreads(1)
    revision = subprocess.check_output(['git', '-C', a.model_root, 'rev-parse', 'HEAD'], text=True).strip()
    if revision != protocol['model_revision']:
        raise ValueError('Pi3 source revision differs from protocol')
    if subprocess.check_output(['git', '-C', a.model_root, 'status', '--porcelain'], text=True).strip():
        raise ValueError('Pi3 source checkout is dirty')
    weights = Path(a.weights)
    if weights.name != protocol['weights_snapshot']:
        raise ValueError('weights snapshot differs from protocol')
    provenance = {'model_revision': revision, 'weight_files_sha256': {p.name: digest(p) for p in weights.iterdir() if p.is_file()},
                  'protocol_sha256': digest(a.protocol), 'code_sha256': digest(__file__),
                  'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
                  'dtype': 'bfloat16', 'conditions': ['intrinsics', 'depths'], 'pose_conditioning': False}
    (out/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    model = Pi3X.from_pretrained(str(weights)).eval().to('cuda')
    for entry in protocol['selection']:
        folder = out/entry['id']
        folder.mkdir()
        start = time.monotonic()
        with tempfile.TemporaryDirectory(prefix='nav_train_only_') as tmp:
            arrays, Ks, times, hashes = load_training(entry, Path(tmp))
            images, conditions = load_multimodal_data(tmp, {'intrinsics': Ks, 'depths': arrays, 'poses': None},
                interval=1, PIXEL_LIMIT=protocol['pixel_limit'], verbose=False, device='cuda')
            assert conditions['poses'] is None and images.shape[1] == len(entry['train_frame_ids'])
            del arrays
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                prediction = model(imgs=images, **conditions)
            poses = prediction['camera_poses'][0].float().cpu().numpy()
            predicted_z = prediction['local_points'][0, ..., 2].float().cpu().numpy()
            conf = torch.sigmoid(prediction['conf'][0, ..., 0]).float().cpu().numpy()
            measured_z = conditions['depths'][0].float().cpu().numpy()
            K_model = conditions['intrinsics'][0].float().cpu().numpy()
            good = np.isfinite(predicted_z) & (predicted_z > .05) & (measured_z > .05) & (conf > .5)
            if good.sum() <= 1000:
                raise ValueError('insufficient training-only metric scale anchors')
            scale = float(np.sum(predicted_z[good]*measured_z[good], dtype=np.float64)/np.sum(predicted_z[good]**2, dtype=np.float64))
            scaled = poses.copy()
            scaled[:, :3, 3] *= scale
            convention = 'world_from_camera, +Z forward +Y down (depth frame)'
            np.savez_compressed(folder/'initial.npz', frame=np.array(entry['train_frame_ids']), t=times,
                                estimate=scaled, convention=convention)
            np.savez_compressed(folder/'dense.npz', predicted_z=predicted_z, confidence=conf,
                                measured_z=measured_z, K_model=K_model, unscaled_poses=poses, metric_scale=scale)
            report = {'id': entry['id'], 'train_frame_ids': entry['train_frame_ids'], 'eval_frame_ids': entry['eval_frame_ids'],
                      'decoded_image_sha256': hashes, 'metric_scale': scale, 'scale_pixels': int(good.sum()),
                      'tensor_shape': list(images.shape), 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
                      'elapsed_s': time.monotonic()-start}
            (folder/'inference.json').write_text(json.dumps(report, indent=2)+'\n')
            print(entry['id'], 'scale', scale, 'seconds', report['elapsed_s'], flush=True)
            del prediction, images, conditions
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
