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

If `huggingface.co` is TLS-intercepted — one machine here returns
`CERTIFICATE_VERIFY_FAILED` with no corporate root installed, while github.com
clones fine — copy the cache entry from a machine that has it and run with
`HF_HUB_OFFLINE=1` rather than weakening TLS:

```bash
# on the machine that has the weights
tar -c -C ~/.cache/huggingface/hub models--yyfz233--Pi3X | ssh other 'tar -x -C ~/.cache/huggingface/hub'
export HF_HUB_OFFLINE=1
```

That also makes the weights provably identical between the two, which is worth
having when their numbers are going to be compared.

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

`--window 400` means "one pass if it fits", and below the ceiling there are no
seams at all — seams caused every joining problem recorded here. Measured on the
96 GB card:

```
  frames    131     160     190     220     250
  peak     38.8    52.9    70.2    90.2    OOM   GiB
```

**220 frames**, which at the 5 Hz image rate is about 44 seconds of walking.
Twelve of the thirteen sessions are inside it. Note that `nvidia-smi` sampling
reports a few GiB more than `torch.max_memory_allocated` because it includes the
CUDA context and the allocator's reserve, so compare like with like.

## Two rules for a measurement to mean anything

**One measurement per process.** Running the same forward twice inside a single
process does not reproduce — 0.5 to 1.4% on the fitted scale, measured on two
different cards — while a fresh process reproduces to printed precision. The
suspected cause is Pi3 offering its attention backend as a list, which lets the
runtime choose per call, but that is not confirmed, so the rule follows the
measurement rather than the diagnosis. `fov_sweep.py` forks a child per arm for
exactly this reason; `--in-process` is how the parent runs each child and is not
for interactive use.

**Chunking is exact, but bfloat16 is not.** `test_chunked_conv.py` compares the
patched and unpatched paths and allows 1e-2. On the 96 GB card the difference is
exactly zero; on an RTX 6000 Ada it is 3.5e-03, because the batch size decides
which kernel the runtime picks and the kernels round differently. In fp32 the
same comparison gives 5.7e-05 — the difference tracks arithmetic precision,
which a batch-splitting mistake would not do, since that mixes frames and moves
a pose by its own magnitude.

**Never compare arms across hosts.** The same configuration on two cards gives
ATE 9.9 cm and 9.7 cm — each host reproduces itself and they differ by 2%, which
is the size of some effects worth looking for. The ICP and ARKit columns do
match exactly across hosts, because those are arithmetic over a copied npz; that
confirms the copy, not the model.

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
