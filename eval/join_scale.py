#!/usr/bin/env python3
"""What the chain did to each window's measured scale, and what undoing it costs.

`pi3_poseless.py` measures every window's metric scale from that window's own
LiDAR depth and applies it. Then `umeyama` re-fits a **free scale** at each
join, because it was written for `pi3_chain`, where windows genuinely carry
arbitrary scales and forcing them equal would bake the first window's into
everything after it. In the poseless mode that premise is gone: every window is
already anchored to metres, so the free scale throws away every depth
measurement except the first window's, and whatever it drifts to compounds.

    python3 eval/join_scale.py pi3traj/*_local.npz
    python3 eval/join_scale.py pi3traj/<id>_wide_local.npz --anchor out.npz

It drifts a long way. The cumulative factor over the three dual-lens sessions:

```
  d0f44f  wide 1.787   ultra-wide 0.794
  d67f0a  wide 0.937   ultra-wide 0.295
  15fbb3  wide 0.846   ultra-wide 1.340
```

Those are not compensating a disagreement between windows: `s_first/s_last`
predicts 0.93, 1.00 and 0.97 for the wide arms. And **the join residual cannot
see any of it**, because it is measured after `umeyama` has absorbed the scale
it fitted — so the "window-to-window agreement" row in `docs/POSE.md` is a
statement about shape only. That is why the ultra-wide can lead three of three
on joins while one of its chains loses seventy per cent of its scale.

**Undoing it does not help, which is why nothing here changes `pi3_poseless`.**
`--anchor` writes the trajectory each window's own depth would have built, and
scored on 40 cm surface thickness that moves -1.12, +0.79, +0.64 and -0.38 cm
across four arms — two directions, three of them under the 0.7 cm this data
resolves. The wide arm still leads both sessions. What does move is the size of
the lead: d0f44f's 1.78 cm falls to 0.61 and 15fbb3's 0.38 cm tie opens to 2.14.
So this is not a defect to fix; it is a degree of freedom nobody registered, and
the per-session margins are sensitive to it while the direction is not.

`--anchor` is a rescale and not a re-chain: the rotations and the join fits stay
as they were. It answers "is the depth-anchored scale better", not "what would
the anchored chain be".
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def window_owners(n: int, window: int, overlap: int) -> tuple[np.ndarray, int]:
    """Which window first placed each pose, in the order the chain built them.

    This mirrors the start list in `pi3_poseless.main` rather than reading it
    from the npz, because it is not stored. The count it produces is checked
    against the recorded `window_scale` length by the caller — a reconstruction
    that silently disagrees with the run it describes would be worse than none.
    """
    step = max(window - overlap, 1)
    starts = list(range(0, max(n - window, 0) + 1, step))
    if starts and starts[-1] + window < n:
        starts.append(n - window)
    owner = np.full(n, -1, dtype=np.int64)
    for k, s0 in enumerate(starts):
        for i in range(s0, min(s0 + window, n)):
            if owner[i] < 0:
                owner[i] = k
    return owner, len(starts)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("trajectory", nargs="+")
    ap.add_argument("--anchor", metavar="OUT.npz",
                    help="write the trajectory each window's own depth scale "
                         "would have built; one input only")
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--overlap", type=int, default=12)
    a = ap.parse_args(argv)
    if a.anchor and len(a.trajectory) != 1:
        raise SystemExit("--anchor takes one trajectory")

    for path in a.trajectory:
        d = dict(np.load(path))
        if "join_scales" not in d:
            print(f"{path}: no join_scales — not a poseless chain")
            continue
        js, ws = d["join_scales"], d["window_scale"]
        cum = np.concatenate([[1.0], np.cumprod(js)])
        print(f"{path}")
        print(f"  {len(ws)} windows, {len(js)} joins, {int((js > 1).sum())} above 1")
        print(f"  cumulative join scale {cum[-1]:.3f}, against {ws[0] / ws[-1]:.3f} "
              f"predicted by the windows' own scales")
        print(f"  join scales: " + " ".join(f"{j:.3f}" for j in js))
        if not a.anchor:
            continue

        poses = d["estimate"]
        owner, n_windows = window_owners(len(poses), a.window, a.overlap)
        if n_windows != len(ws):
            raise SystemExit(
                f"reconstructed {n_windows} windows but the run recorded "
                f"{len(ws)} — --window/--overlap do not match this file")
        if len(cum) < n_windows:  # the chain stopped early at a refused join
            cum = np.concatenate([cum, np.full(n_windows - len(cum), cum[-1])])

        p = poses[:, :3, 3]
        delta = np.diff(p, axis=0) / cum[owner][1:, None]
        out = poses.copy()
        out[1:, :3, 3] = p[0] + np.cumsum(delta, axis=0)
        d["estimate"] = out
        d["convention"] = str(d.get("convention", "")) + \
            "; join scale removed, each window back on its own depth scale"
        np.savez(a.anchor, **d)
        before = np.linalg.norm(np.diff(p, axis=0), axis=1).sum()
        print(f"  anchored: walked {before:.2f} -> "
              f"{np.linalg.norm(delta, axis=1).sum():.2f} m, wrote {a.anchor}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
