#!/usr/bin/env python3
"""Where the per-window depth scale comes from, on one window of one arm.

`pi3_poseless` fits `s = <pz, meas> / <pz, pz>` per window and that number splits
by arm — ultra-wide 1.05-1.15 against wide 0.98-1.05, the same way in every
session. `docs/POSE.md` carried it for a while as a property of the lens. It is
mostly not.

    eval/.venv/bin/python eval/depth_scale_probe.py SESSION --lens wide
    eval/.venv/bin/python eval/depth_scale_probe.py SESSION --lens wide --mask-depth 0.40

It runs one window and reports `s` twice over: once on the full support, and
once per bearing bin, so a radial effect would show as a trend and a gain would
not. **The full-support number has to match the window-0 scale the real run
printed** — on `d0f44f` that is 0.919 and 1.064 — or the probe is not describing
the run it claims to.

Two things came out of it and both are in `docs/POSE.md`:

**It is not radial.** Across 0-60 degrees of bearing the wide arm sits at
0.9180-0.9207 and the ultra-wide at 1.0631-1.0672. Flat to about 0.4 %, so
whatever separates them is a gain and not a distortion the intrinsics missed.

**Half of it is how much depth the model was given.** `--mask-depth` blanks the
depth outside a centred ellipse of that area fraction, which is the shape of the
LiDAR cone rather than a random thinning. Same lens, same frames, same window,
only the conditioning changes:

```
  d0f44f wide, 87 % of pixels covered     s 0.9187
  d0f44f wide, 40 % covered               s 0.9938
  d0f44f ultra-wide, 34 % covered         s 1.0640
```

The fit's own support moved too, and it is not the cause: restricting the fit
alone to those bearings leaves `s` at 0.918, which is what the bin table above
says. So **the scale the model predicts depends on how much depth it is handed**,
and the ultra-wide arm cannot be given more — the LiDAR cone is a third of its
frame by construction. That confound is in `s` for every ultra-wide window ever
fitted, and it is worth 0.075 of a 0.145 split in this window. The other half is
not explained.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "tools"))
from read_session import Session  # noqa: E402
import pi3_poseless as P  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
BINS = ((0, 10), (10, 20), (20, 30), (30, 35), (35, 60))


def fit(pz, meas, mask) -> float:
    return float((pz[mask] * meas[mask]).sum()
                 / max((pz[mask] * pz[mask]).sum(), 1e-9))


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("session")
    ap.add_argument("--lens", choices=("ultrawide", "wide"), default="ultrawide")
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--mask-depth", type=float, default=1.0,
                    help="blank the depth outside a centred ellipse of this "
                         "area fraction before handing it to the model")
    a = ap.parse_args(argv)

    import torch
    sys.path.insert(0, str(REPO / "eval" / "vendor" / "Pi3"))
    from pi3.models.pi3x import Pi3X
    from pi3.utils.basic import load_multimodal_data

    session = Session(a.session)
    manifest = json.loads((Path(a.session) / "manifest.json").read_text())
    depth_calib, depth_source = P.depth_calibration(manifest)
    rows = P.pair_by_time(session, 0.05,
                          "frames" if a.lens == "ultrawide" else "frames_wide")
    name = f"iphone17-1_{'ultrawide' if a.lens == 'ultrawide' else 'wide'}.json"
    calib = P.load_calibration(REPO / "calib" / name)
    row0 = rows[0]
    K = P.intrinsics_for(calib, int(row0["width"]), int(row0["height"]),
                         P.active_hfov(manifest, a.lens))
    extrinsic = np.eye(4)
    if a.lens == "ultrawide":
        cols = np.asarray(json.loads((REPO / "calib" / name).read_text())
                          ["extrinsic_matrix_columns"], dtype=float).T
        extrinsic[:3, :3] = cols[:, :3]
        extrinsic[:3, 3] = cols[:, 3] / 1000.0

    block = rows[:a.window]
    print(f"  {a.lens} arm, depth grid from {depth_source}")
    model = Pi3X.from_pretrained("yyfz233/Pi3X").eval().to("cuda")
    with tempfile.TemporaryDirectory() as tmp:
        images = Path(tmp) / "images"
        images.mkdir(parents=True)
        depths = []
        for order, row in enumerate(block):
            shutil.copyfile(session.frame_path(row), images / f"{order:04d}.jpg")
            d = (P.depth_in_ultrawide(session, row, K, depth_calib, extrinsic)
                 if a.lens == "ultrawide"
                 else P.depth_in_wide(session, row, K, depth_calib))
            if a.mask_depth < 1.0:
                h, w = d.shape
                v, u = np.mgrid[0:h, 0:w]
                r = np.hypot((u - w / 2) / (w / 2), (v - h / 2) / (h / 2))
                d = np.where(r <= np.sqrt(a.mask_depth * 4 / np.pi),
                             d, 0.0).astype(np.float32)
            depths.append(d)
        stack = np.stack(depths)
        # Check the lever moved before spending the model on it.
        print(f"  depth handed to the model: {(stack > 0.05).mean():.3f} of pixels"
              + (f" (masked to {a.mask_depth:g})" if a.mask_depth < 1.0 else ""))
        cond = {"intrinsics": np.stack([K] * len(block)), "depths": stack,
                "poses": None}
        imgs, cond = load_multimodal_data(str(images), cond, interval=1,
                                          PIXEL_LIMIT=255000, verbose=False,
                                          device="cuda")
        measured = cond["depths"]
        dtype = (torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8
                 else torch.float16)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            res = model(imgs=imgs, **cond)
        pz = res["local_points"][0].float().cpu().numpy()[..., 2]
        conf = torch.sigmoid(res["conf"][0, ..., 0]).float().cpu().numpy()
        meas = measured[0].float().cpu().numpy()

    good = (meas > 0.05) & np.isfinite(meas) & (conf > 0.5) & (pz > 0.05)
    print(f"  s over the full support {fit(pz, meas, good):.4f}  "
          f"({good.sum():,} px of {pz.size:,}) — this must match the window-0 "
          f"scale the run printed")

    # Bearings on the grid the model actually ran at, not the source frame.
    _, h, w = pz.shape
    sx, sy = w / int(row0["width"]), h / int(row0["height"])
    v, u = np.mgrid[0:h, 0:w]
    ang = np.degrees(np.arctan(np.hypot((u - K[0, 2] * sx) / (K[0, 0] * sx),
                                        (v - K[1, 2] * sy) / (K[1, 1] * sy))))
    ang = np.broadcast_to(ang, pz.shape)
    print(f"  grid {w}x{h}, bearings reach {ang.max():.1f} deg")
    for lo, hi in BINS:
        m = good & (ang >= lo) & (ang < hi)
        if m.sum() < 5000:
            print(f"  {lo:2d}-{hi:2d} deg   {m.sum():>10,} px   too few to read")
            continue
        print(f"  {lo:2d}-{hi:2d} deg   {m.sum():>10,} px   s {fit(pz, meas, m):.4f}"
              f"   median measured {np.median(meas[m]):.2f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
