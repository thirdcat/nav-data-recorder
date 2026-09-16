#!/usr/bin/env python3
"""Save per-view float metrics and renders while the GS model stays in eval mode.

Use with locally generated checkpoints. DN-Splatter's aggregate evaluator
switches back to train mode before saving images; this path keeps metric and
saved-render evaluation conditions identical (including background behavior).
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=False)
    # These are our own checkpoints, including optimizer/config numpy state.
    original_load = torch.load

    def load(*args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return original_load(*args, **kwargs)

    torch.load = load
    from nerfstudio.utils.eval_utils import eval_setup

    config, pipeline, checkpoint, step = eval_setup(Path(a.config), test_mode='test')
    pipeline.eval()
    torch.manual_seed(42)
    datamanager = pipeline.datamanager
    rows = []
    for i, batch in enumerate(datamanager.cached_eval):
        camera = datamanager.eval_dataset.cameras[i:i+1].to(pipeline.device)
        assert not pipeline.model.training
        with torch.inference_mode():
            outputs = pipeline.model.get_outputs_for_camera(camera=camera)
            metrics, _ = pipeline.model.get_image_metrics_and_images(outputs, batch)
        name = Path(datamanager.eval_dataset.image_filenames[i]).stem
        rgb = outputs['rgb'].detach().cpu().numpy()
        gt = batch['image'].detach().cpu().numpy()
        depth = outputs['depth'].detach().cpu().numpy()
        sensor_depth = batch['sensor_depth'].detach().cpu().numpy()
        if rgb.ndim == 4:
            rgb = rgb[0]
        if rgb.shape != gt.shape or not np.isfinite(rgb).all():
            raise ValueError('invalid output RGB')
        for label, array in [('pred', rgb), ('gt', gt)]:
            Image.fromarray(np.round(np.clip(array, 0, 1)*255).astype(np.uint8)).save(out/f'{name}_{label}.png')
        np.savez_compressed(out/f'{name}_float.npz', prediction=rgb, target=gt,
                            depth=depth, sensor_depth=sensor_depth)
        mse = float(np.mean((rgb.astype(float)-gt.astype(float))**2))
        measured_psnr = float(-10*np.log10(max(mse, 1e-30)))
        if abs(measured_psnr-float(metrics['rgb_psnr'])) > 1e-3:
            raise ValueError('reported metric differs from saved float render')
        rows.append({'image': name, 'metrics': {k: float(v) for k, v in metrics.items()},
                     'float_psnr': measured_psnr, 'gt_png_sha256': digest(out/f'{name}_gt.png'),
                     'camera_to_world': camera.camera_to_worlds[0].cpu().numpy().tolist(),
                     'width': int(camera.width[0]), 'height': int(camera.height[0])})
        print(name, measured_psnr, flush=True)
    result = {'config': a.config, 'config_sha256': digest(a.config), 'checkpoint': str(checkpoint),
              'checkpoint_sha256': digest(checkpoint), 'step': step, 'model_eval_mode': True,
              'code_sha256': digest(__file__), 'views': rows,
              'mean': {k: float(np.mean([r['metrics'][k] for r in rows])) for k in rows[0]['metrics']}}
    (out/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result['mean'], indent=2), flush=True)


if __name__ == '__main__':
    main()
