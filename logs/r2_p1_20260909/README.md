# R2-P1: fixed-scale Pi3X window alignment pilot

Completed 2026-09-09. Plan, pre-run conditions and results:
[METRIC_RECONSTRUCTION_PLAN.md §1.2](../../docs/METRIC_RECONSTRUCTION_PLAN.md#12-r2-p1--캐시-기반-window-정렬-실험).

Decision: **do not adopt WG-R**. The ARKit position screen passed, but only one
image pair per session met the predefined correspondence criteria (10 required).
On the failed session, the rotation-only sequential control also did better than
either global arm. No new Pi3X inference or 3DGS training was performed.

Run from the repository root with Python, NumPy, SciPy and OpenCV. Output paths
must not exist; use fresh directory names to repeat without overwriting results.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -u eval/run_window_alignment.py \
  /home/myeongcheol/nav_data/20260813-162849-2994fa \
  --cache pi3win/2994fa.npz --reference pi3traj/2994fa.npz \
  --out logs/r2_p1_20260909/2994fa
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -u eval/run_window_alignment.py \
  /home/myeongcheol/nav_data/20260813-162946-7d3d52 \
  --cache pi3win/7d3d52.npz --reference pi3traj/7d3d52.npz \
  --out logs/r2_p1_20260909/7d3d52
python3 logs/r2_p1_20260909/summarize.py
```

The summary/plot script additionally uses Matplotlib. No GPU is needed.
Each `report.json` records input/code hashes, environment, solver termination,
reference metrics, image pair proposals/rejections, image hashes and pair scores.
`correspondences.json` preserves exact selected pixels and sampled depth points.
These are local capture-derived artifacts, not curated repository fixtures.

Each arm's `.npz` contains capture `frame` IDs, original monotonic `t` in seconds,
`estimate` (world-from-camera, +X right/+Y down/+Z forward camera axes),
`window_transform` (world-from-metric-window), `solver_success`, and `convention`.
Original window translations are multiplied by cached metric scale exactly once;
the optimizer has no scale variable. The first window transform is identity.
WR/W0 retain the earliest estimate; WG arms average all transformed estimates.
The reference trajectory is read only after all candidate trajectories are built.
Saved trajectories are experimental and have not passed the adoption gate.

`summary.json` records the gate decision. `local_window_diagnostic.json` records
an additional per-window fit to ARKit for diagnosis only. Those fits never enter
the optimizer or the global trajectory scores. `aggregation_diagnostic.json`
records a post-result control: changing only mean versus first-estimate output
aggregation does not explain the global arms' worse result against WR.

| ARKit-aligned mean position difference | 2994fa | 7d3d52 |
| --- | --- | --- |
| W0 | 2.4632 m | 0.24358 m |
| WR | 1.5050 m | 0.24353 m |
| WG-L | 1.7133 m | 0.24301 m |
| WG-R | 1.8901 m | 0.24304 m |

ARKit is a comparison reference, not absolute ground truth. The independent
image check is independent of this optimizer's objective; it reuses photographs
that Pi3X may have consumed. Image score values from one accepted pair are not
session-wide validation. Neither metric scale accuracy nor 3DGS quality was
established by this pilot.

Validation performed:

```bash
python3 eval/test_window_alignment.py
python3 eval/test_window_reprojection.py
python3 eval/test_pi3_join.py
python3 tools/make_test_session.py /tmp/nav-r2-p1-fixture-20260909
python3 tools/read_session.py /tmp/nav-r2-p1-fixture-20260909/20260807-014530-fixture
```

All passed. Synthetic cases cover exact and pure-rotation alignment, fixed metric
scales, gauge, window order, corrupt overlapping poses, disconnected graph,
malformed cache, projection sign/units, invalid projections, and SE(3)-only scoring.
The real-data uncertainty is reported separately from these implementation checks.
