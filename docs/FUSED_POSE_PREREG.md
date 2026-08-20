# Pre-registration: the fused trajectory as a gated 3DGS pose arm

Written before any export was run and before `eval/fuse_rate.py` was touched.
What follows is what the fused arm has to prove, which instrument measures it,
what the instrument reads on an answer already known, and what number counts as
a failure. Nothing below is adjusted after the fact; a gate that fails is
reported as a failure.

## The claim being tested

`docs/3DGS.md` scored two pose arms on `cb4586` — ARKit 24.282 dB against
depth-ICP 21.760 — and left a third out on purpose:

> A third arm — `eval/fuse_rate.py`'s ICP-plus-Pi3X fusion — was not run. It is
> written in the Pi3X world frame and carries no `reference`, so nothing
> establishes that the exporter's conversion describes it.

The claim to be established is narrow and is **not** about splat quality:
**`tools/export_3dgs.py --poses` reads the fused file as the thing it actually
is.** Whether the resulting splat is better or worse than ARKit's needs a
training run and is out of scope here.

## What is already wrong, measured first

Two facts were measured before the design was fixed, because both change what
the gate has to do.

**The file does carry a `reference`, and that is worse than carrying none.**
`_write_output` writes `reference=traj["reference"]` — ARKit's own poses, copied
out of the ICP trajectory. `substitute_poses` finds that key, checks it against
the session, finds it perfect, and prints *"its own ARKit poses agree with the
session to 0.00e+00 m"* — while substituting an `estimate` written in a
different world. The check runs, passes, and says nothing whatever about the
poses being substituted. The `UNCHECKED` path `docs/3DGS.md` describes is never
reached.

**The world frame is not the hazard it was assumed to be.** The exporter builds
the initial point cloud from the same poses it writes into `images.txt`, so a
global rigid transform of the whole trajectory moves cameras and cloud together
and the reconstruction is unchanged. The ICP arm is proof: its `estimate[0]` is
the identity, so it is written in *camera 0's* frame and not ARKit's, and it
exported and trained to 21.760 dB anyway. What the exporter cannot survive is a
**camera-axis** convention difference (a right-multiplier on the 3x3) or a
**scale** difference against the metric depth. Those are the two things to test.

**An instrument that compares two trajectories cannot separate them.** The first
attempt recovered the camera-axis multiplier `C` by fitting relative-motion
translations of Pi3X against ARKit. Run on the ICP trajectory, whose convention
is correct by construction, it returned `C` angles up to **135 degrees** and
scale up to **1.39**. The relative translation is expressed in the estimator's
own rotated frame, so accumulated absolute-rotation drift enters the fit and is
indistinguishable from a constant convention offset. That instrument is
rejected here rather than tuned.

## The instrument

**`tools/export_3dgs.py --poses` followed by `tools/eval_views.py`, `lidar`
column.** No training, no GPU, nothing fitted.

The exporter back-projects every training frame's depth through
`camera_to_world(pose)` into one cloud, and `eval_views` splats that cloud
through each held-out camera's `world_to_camera(pose)` and scores it against the
photograph that camera actually took and the depth it actually measured. Both
ends use the substituted poses. If the npz's 4x4 is not what `camera_to_world`
means, the cloud and the held-out camera disagree and the column collapses.

This is the production path, not a proxy for it, and its answer on this session
is already published.

## The known answer the instrument must reproduce first

`docs/3DGS.md`, `cb4586`, ARKit poses, 167 images / 21 held out, quarter
resolution:

```
  lidar PSNR                17.32 dB
  lidar depth median abs     0.038 m
  lidar depth abs-rel        0.020
  lidar coverage             28 %
  mean image on lidar mask  13.15 dB
```

**C0.** Exporting `cb4586` with `--poses traj/cb4586.npz --pose-key reference`
and scoring it must return all five within `PSNR +/- 0.05 dB`, `depth +/- 0.001
m`, `abs-rel +/- 0.001`, `coverage +/- 0.01`. If it does not, the harness is
measuring itself and every number below is void.

**Amendment, after running C0 and before running any arm.** Three of the five
reproduce to the last digit — `17.32`, `13.05`, `13.15` — and the export is
byte-identical to the default one (`images.txt` and `sparse_pc.ply` both
`cmp`-clean), so the pose path reproduces exactly. The other two do not, and
both are accounted for without touching the data:

- **Coverage.** `eval_views.evaluate` writes `lidar.coverage = mask.mean()` and
  then `entry["lidar"].update(got)`, where `depth_error`'s return carries its
  own `coverage` key. The second silently overwrites the first, so the column
  changes meaning from "how much of the frame the cloud covers" to "how much of
  it has both cloud and valid LiDAR depth" whenever depth is present. The splat
  coverage is **0.2762** mean / 0.2962 median — the published 28 % — and the
  0.20 printed today is the narrower quantity under the same name.
- **Depth.** 0.038 m is not a quarter-resolution reading. This export gives
  0.032 m at `--scale 0.25`, **0.036 m at 0.5 and at 1.0** — which is also the
  `0.036 m` recorded for the same session at line 1299. The published row is
  labelled quarter resolution and carries a full-resolution depth figure.

C0 is therefore treated as **passed on the reproducing triple**, and the depth
and coverage rows are recorded as doc defects rather than as data disagreement.
This relaxation is stated here rather than applied silently. It cannot change a
verdict below: G2's bar is 0.50 m, which is more than thirteen times every one
of these readings.

## Criteria

Each is a number with a threshold, fixed here. "Convention" bars are hard gates;
"quality" figures are reported and gate nothing, because a trajectory can be
honestly converted and still be inaccurate, and conflating those is how a
comparison-setup error gets read as a result.

### 1. Convention gate — the file is in the exporter's frame

An arm passes only if all three hold:

```
  G1  lidar coverage              >= 0.15      (ARKit reads 0.28)
  G2  lidar depth median abs      <= 0.50 m    (ARKit reads 0.038)
  G3  lidar PSNR - mean-on-mask   >= 2.0 dB    (ARKit reads +4.2)
```

The bars are set an order of magnitude away from ARKit's reading and are not
tuned to any arm's result: a camera-axis or scale error does not degrade these
numbers, it destroys them — the cloud lands metres from where the held-out
camera is looking. A *near*-miss on any of the three is to be reported as
undecided, not as a pass, because near a bar the instrument cannot separate a
convention error from an inaccurate trajectory.

### 1b. Two numeric gates, added after criterion 1 was measured

**Added after running the controls of criterion 2 and before gating any arm.**
The image instrument caught N1 and N4 and was *marginal* on N2 and *blind* on
N3, so it is not sufficient on its own and two pure-numpy probes are added.
Each carries its own known-answer control on the same session: the ICP
trajectory in `traj/<id>.npz`, whose camera axes are `diag(1,-1,-1)` applied by
the same code that writes `reference` and whose scale is ICP on metric depth —
so its truth is `C = 0 deg` and `s = 1.000`, and whatever the probe reads there
is the probe's floor on that session, not a defect of the arm.

```
  G4  camera axes   fit the gauge G on POSITIONS ONLY at every frame, then
                    average D_i = R_ark(i)^T G R_est(i) over the trajectory.
                    A constant camera-axis error makes every D_i equal, so the
                    average is that error; estimator noise averages down.
                    Bar: |C| < 5.0 deg AND |C| <= max(2 x ICP's reading, 1.0).

  G5  metric scale  median of |p_j - p_i|_est / |p_j - p_i|_ark over all frame
                    pairs whose ARKit baseline is 0.3-1.0 m. World-frame
                    distances only, so no rotation enters and drift cannot
                    contaminate it. Bar: |s - 1| <= 0.05.
```

The 0.3–1.0 m band and the 0.05 bar are set by the known answer, not by any
arm's result. Across all 18 sessions the ICP trajectory reads a median of
**1.0027** in that band; the 0.02–0.10 m band reads 1.082 because per-step noise
inflates short baselines, and the 3–20 m band scatters because drift dominates.
13 of the 18 known-correct trajectories fall within ±0.05, so ±0.05 is this
probe's reproducibility on an input that is right by construction.

Both probes were checked to recover a known injection before use: 5.042 deg for
an injected 5 deg, 2.104 for an injected 2, 179.999 for `diag(1,-1,-1)`, and
1.0797 for a 1.1 scale on a trajectory reading 0.9815. Each is blind to the
other's fault and both are blind to the gauge (P1).

### 2. Controls that must FAIL

A check that has never failed has not been shown to work. Each of these is built
by corrupting the fused file in a known way and must be refused by criterion 1:

```
  N1  camera axes right-multiplied by diag(1, -1, -1)   the ARKit/COLMAP flip
  N2  all translations scaled by 1.1                    metric scale broken
  N3  camera axes right-multiplied by 5 deg about X     small axis error
  N4  poses of a different session (5bd1ed) at cb4586's frame ids
```

N4 must be refused by `substitute_poses`'s existing `reference` check, before
the instrument is reached. N3 is the sensitivity limit and may pass; if it does,
that is recorded as the smallest camera-axis error this instrument can see, not
as a success.

### 3. Control that must PASS — the instrument keys on the right thing

```
  P1  fused, left-multiplied by a 5 deg world rotation and a 1 m translation
```

This is a gauge change. Cameras and cloud move together, so P1 must return the
*same* numbers as the uncorrected arm to within `PSNR +/- 0.05 dB` and
`coverage +/- 0.01`. If P1 fails, the instrument is measuring the world frame —
which does not matter — and criterion 1 is void.

### 4. Reported, not gated

On `cb4586`, for the fused arm: raw position disagreement with ARKit (median and
max, in metres) and Umeyama-aligned RMS. The ICP arm's figures on that session
are **2.67 m median / 6.46 m max raw, 8.1 cm aligned RMS**, and it lost the
two-arm comparison by 2.5 dB. If the fused arm is not closer to ARKit than that,
it is said so plainly.

### 5. Regression — the fusion's own scores do not move

`eval/fuse_rate.py`'s reported ATEs are computed after a rigid alignment and are
therefore invariant to any global rigid transform of the fused trajectory. So
rebasing the output into ARKit's world must leave every number in the JSON
report **bit-identical**. If any of them moves, the rebase is not rigid.

## What a pass does and does not license

A pass licenses one sentence: the fused trajectory can be handed to
`tools/export_3dgs.py --poses` and the exporter's conversion describes it. It
licenses **no** claim about whether the fused arm builds a better or worse
splat than ARKit or ICP. That question needs a training run.
