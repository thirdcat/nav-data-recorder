# eval

Scoring harness for pose sources that are not ARKit, and for the capture-quality
signals the recorder shows. Research scaffolding, not part of the capture or
export path.

## Why this is not in `tools/`

`tools/` runs on numpy and nothing else — `read_session.py` still reads a session
with numpy absent, and that is a property worth keeping, because it is what lets
the session format be inspected anywhere. This directory needs PyTorch, a CUDA
device and about 7 GB of environment.

**`eval/` may import from `tools/`. `tools/` must never import from `eval/`.**
That one rule is the whole boundary.

## What runs where

The split follows the hardware, and the interface between the halves is an npz.

| | machine | needs |
|---|---|---|
| `tools/depth_odometry.py --dump-poses` | anywhere | numpy |
| `run_matrix.py`, `build_vis.py`, `test_chain.py` | anywhere | numpy |
| `pi3_eval.py`, `pi3_chain.py`, `test_chunked_conv.py` | GPU box | torch, CUDA, weights |

`--dump-poses` writes both trajectories as 4×4 world-from-camera in the depth
convention, with the frames, timestamps, inlier fractions and both conditioning
figures. **A second estimator scored against that file inherits the axis
convention instead of choosing one**, which is the point: this repository has
shipped a transposed rotation once and spent a day on it. Nothing in here
rebuilds `ARKIT_TO_DEPTH`.

So the small file crosses the wire, not the pipeline. Sessions are copied to the
GPU box once; the npz dumps follow; the results come back as json.

## Setup

```bash
./eval/setup.sh              # clone Pi3 at the pinned commit, build eval/.venv
./eval/setup.sh --check      # report what is present, install nothing
```

Neither `eval/.venv` nor `eval/vendor/` is committed: one is per-machine and per
CUDA build, the other belongs to someone else. The Pi3 commit is pinned in
`setup.sh`, and it matters — the numbers in `docs/POSE.md` were produced against
that commit, and a model that drifts under a fixed harness reproduces nothing.

## Running it

```bash
export PI3_ROOT="$PWD/eval/vendor/Pi3"
export NAV_DATA="$HOME/nav_data"

# 1. the reference and the depth-only estimate, from the numpy side
python3 tools/depth_odometry.py "$NAV_DATA/<session>" --imu-rotation \
        --dump-poses traj/<id>.npz

# 2. the learned estimate, scored against the same frames
eval/.venv/bin/python eval/pi3_chain.py "$NAV_DATA/<session>" \
        --dump traj/<id>.npz --window 400 --overlap 48

# 3. what a recording-time indicator would have shown, across every session
python3 eval/build_vis.py
```

`--window 400` means "one pass if it fits". It fits for sessions up to somewhere
between 131 and 260 image frames on a 98 GB card — 131 peaks at 38.8 GiB, 260
does not fit and falls back to 96-frame windows. The exact ceiling is unmeasured.
Below it there are no seams at all, and seams were the cause of every joining
problem recorded here.

## Tests

Standalone programs, nonzero on failure, matching `tools/test_*.py`.

```bash
python3 eval/test_chain.py                  # no GPU: joining recovers a known trajectory
eval/.venv/bin/python eval/test_chunked_conv.py   # GPU: the conv patch is exact
```

`test_chain.py` needs no GPU on purpose. It feeds the joining code windows of a
known trajectory placed in arbitrary frames and scales, and requires the original
back. When the seams first went wrong that test is what separated "the joining is
broken" from "the windows are" — it passed at 0.000 mm, so the joining was not
the fault.

## What is deliberately absent

No fork of Pi3. The one change needed — running the decoder convolutions in
batch chunks so a long sequence takes the same path a short one does — is a
runtime wrapper in `chunked_conv.py`, five functions that touch no upstream
source. Pinning by commit and patching at runtime keeps the seam between our work
and theirs visible.
