# Where the pose comes from, and what replacing it would cost

The ultra-wide probe settled the geometry: the lens reaches 96.3° with margin,
and the rectification works. What it cannot settle is pose, because reaching the
ultra-wide means leaving ARKit — see *Why not the 0.5x ultra-wide?* in
[DATA_FORMAT.md](DATA_FORMAT.md). This page is what would have to be rebuilt.

## ARKit and the ultra-wide, one more time

Confirmed four independent ways, so it can stop being re-litigated:

- Every format in `ARWorldTrackingConfiguration.supportedVideoFormats` reports
  the **wide-angle** capture device. Each session logs them under `ar.formats`,
  so any recording proves this for the device it was made on.
- An `AVCaptureSession` cannot coexist with an `ARSession`; starting one
  interrupts the other.
- Apple's answer on the [developer
  forums](https://developer.apple.com/forums/thread/719837), to exactly this
  question, is *"That is not possible at this time."*
- [SplatKing](https://radiancefields.com/splatking) wanted both and shipped them
  as separate modes: dual-lens photo/video with no poses, and a LiDAR mode that
  is ARKit with the 1× wide alone.

## What ARKit is doing

Apple documents the approach but not the algorithm. From [Understanding World
Tracking](https://developer.apple.com/documentation/arkit/understanding-world-tracking),
it is **visual-inertial odometry**: features are found in the camera image and
tracked across frames, and that is compared against the motion sensors. An
[independent benchmark](https://arxiv.org/pdf/2207.06780) characterises the
implementation as a *tightly-coupled filtering VIO similar to MSCKF, with a
sliding-window filter, bundle adjustment and motion/structure marginalisation* —
and measures its drift at roughly **0.02 m per second**.

The parts that matter for anyone thinking of replacing it:

**The IMU propagates, the camera corrects.** Gyro and accelerometer run about an
order of magnitude faster than the camera and carry the pose between frames,
which is what makes ARKit smooth and low-latency. Integrated alone they drift
badly — position is a double integral of accelerometer bias, so error grows with
the square of time. Each camera frame pulls that back and re-estimates the
biases. *Tightly coupled* means the feature observations and the inertial
residuals go into one estimator rather than two that hand poses to each other.

**Metric scale comes from the accelerometer, not the camera.** A single moving
camera cannot know whether it is a small room seen closely or a large one seen
far away; the accelerometer measures in m/s² and anchors it. This is why ARKit
asks you to move the device before tracking converges, and why
`tracking=limited` frames sit at the origin — the exporter drops them for
exactly this reason.

**Gravity fixes two of the three rotation axes outright.** The accelerometer
sees gravity, so roll and pitch are observable absolutely and do not drift. Only
yaw and position accumulate error. This is why the episode format's Z-up world
is trustworthy while its yaw origin has to be rebased per episode.

**LiDAR is not what produces the pose.** `sceneDepth` improves depth
initialisation, plane fitting and low-texture robustness, but world tracking on
a non-Pro iPhone works without it. Replacing ARKit is a VIO problem, not a depth
problem.

## If the ultra-wide path were built

None of ARKit is reusable — it is closed, and the parts that matter are the
tuning rather than the architecture. But the architecture is standard and there
are mature open implementations: **OpenVINS** (MSCKF, closest to what the
benchmark describes), **ORB-SLAM3** (mono/stereo-inertial), **VINS-Fusion**,
and **COLMAP** for offline structure-from-motion.

Two things make the job smaller than it first looks:

- **Offline beats online.** Nothing here needs a real-time pose. COLMAP with
  bundle adjustment over a whole clip is *more* accurate than ARKit's online
  VIO, which reports its current best guess and cannot revisit it. This is the
  same asymmetry *Known gaps* in [OUTPUT_FORMAT.md](OUTPUT_FORMAT.md) already
  notes against the reference set, and it would flip in our favour.
- **Pose the wide camera, not the ultra-wide.** In a multi-cam session the
  lenses are rigidly coupled with factory extrinsics, which the probe already
  captures — the measured offset between the ultra-wide and the reference camera
  is **19.2 mm**. So SLAM can run on the better-behaved wide stream and the pose
  transfers to the ultra-wide by a fixed rigid transform.

### Depth odometry, and where it breaks

LiDAR depth is the obvious alternative pose source, and its advantage is real:
it is **metric by construction**, which removes the scale problem outright, and
it does not care whether a wall has texture on it.

Its weakness is geometric rather than photometric. Point-to-plane ICP can only
recover the motion the visible geometry constrains, and a view containing a
single plane constrains exactly one axis. `tools/test_depth_odometry.py`
demonstrates this the hard way — the test scene was first built as a large room,
which put only the back wall in frame, and the estimator then recovered forward
motion to a millimetre while missing lateral motion and yaw *entirely*. Shrinking
the room so the side walls, floor and ceiling are all in view fixed it:

```
  forward 20 cm                    t  0.12 cm   r  0.00°   inliers  83%   PASS
  sideways 15 cm                   t  0.01 cm   r  0.00°   inliers  94%   PASS
  yaw 5 deg                        t  0.02 cm   r  0.00°   inliers  90%   PASS
  walk + turn (5 Hz stride)        t  0.03 cm   r  0.00°   inliers  81%   PASS
  flat wall stays put (degenerate) drift  0.00 cm                        PASS
```

That failure is not a bug to fix, it is the method's shape, and indoor homes are
full of it: a corridor barely constrains motion along its own axis, and a blank
wall at arm's length constrains nothing but distance to it. The phone's sensor
narrows the window further — 256×192, useful to roughly 5 m, and a sparse dot
pattern upsampled against the RGB image rather than a dense scan.

So depth odometry is a strong *component* and a fragile *whole*. Which is why
ARKit fuses it rather than relying on it, and why the honest comparison is
RGB-D-inertial against ARKit, not depth alone.

`tools/depth_odometry.py` runs that comparison on any recorded session, scoring
ICP against ARKit's own trajectory for the same frames — every session already
carries both, so this is measurable today, before any capture code is written.
ARKit is a reference rather than ground truth, so agreement means the two made
the same journey; but a depth-only track that cannot match a fused one will not
beat it either.

What is still genuinely hard:

- **Metric scale.** SfM alone is scale-free. Three anchors are available: LiDAR
  depth (best, if captured in the same session), IMU integration (what ARKit
  does, and the fiddly part), or the 19.2 mm stereo baseline — which is real but
  short, about 2% of a one-metre depth, so it is a weak triangulation and better
  used as a scale constraint across many frames than as a depth sensor.
- **Time sync.** ARKit hands over one `ARFrame` with image, depth, pose and
  intrinsics already aligned. A hand-built path has to align streams itself, and
  VIO is sensitive to IMU-to-camera timing offsets at the millisecond level.
- **Rolling shutter.** Fast rotation smears a frame; ARKit compensates and a
  naive pipeline does not.
- **Gravity is free.** `motion.jsonl` already records `CMDeviceMotion` at
  100 Hz, so roll, pitch and the Z-up axis do not have to come out of SLAM at
  all — only yaw and position do.

## Where this stands (2026-08-09)

**Depth-only odometry is within reach of ARKit, and was not until the map
association was fixed.** Measured on two loop-closed walks, where the trajectory
returns to its start and so scores itself without any reference at all:

```
                             path     loop error     of path
  ARKit                    18.89 m      0.269 m        1.4%
  frame to frame           30.23 m      0.828 m        2.7%
  frame to map, projective 45.13 m      8.258 m       18.3%
  frame to map, nearest    23.63 m      0.410 m        1.7%

  ARKit                    19.19 m      0.146 m        0.8%
  frame to frame           31.78 m      1.392 m        4.4%
  frame to map, projective 65.69 m     17.812 m       27.1%
  frame to map, nearest    23.09 m      0.663 m        2.9%
```

Both sessions are well lit (73% and 75% of depth pixels at medium or high
confidence), walked slowly (0.51 and 0.57 m/s), and geometrically healthy — the
conditioning gate flags 0–1% of frames.

The last row is one change, described in *What the working systems do* below:
associating against the map by nearest-neighbour search rather than by
rasterising it into a depth image. It is worth **20× and 27×**, it moves the map
path from far behind the frame-to-frame baseline to comfortably ahead of it, and
it takes the estimator from thirteen times worse than ARKit to within a factor
of two to four. The estimated path length falls from 45 and 66 m to 23 m against
a true 19 m, which is the same story told as accumulated noise.

ARKit still wins. It is no longer winning by an order of magnitude, and the
remaining gap is now a normal engineering distance rather than evidence that the
approach is wrong.

Still true from the earlier work, and worth keeping:

- **5 Hz depth is unusable.** Frames land ~10 cm and several degrees apart, far
  outside ICP's basin. This was a self-inflicted limit — depth was pinned to the
  stills gate — and is fixed: depth now runs at 30 Hz as a superset of the image
  frames, and `tools/export_episodes.py --hz 5` decimates at export instead.
- **Frame spacing is not what limits it.** Striding one of the loop sessions to
  three times its spacing — 1.6 cm to 4.8 cm, wider than the sessions that fail
  — changed the drift by nothing at all, 0.6 cm/s either way. Whatever is wrong
  is not ICP's convergence basin.

### Agreement with ARKit is not accuracy, and this document made that mistake

Every figure above the loop measurement was scored against ARKit, and one
session agreed with it closely enough to look like a win: 0.6 cm/s of relative
drift, against ARKit's own published ~2 cm/s. It was read as depth beating
ARKit. It was not. With a real reference the same estimator loses.

What that 0.6 cm/s measured was **the two making the same journey**, which is
what this page said an ARKit comparison could show and no more. The trap is
easy to walk into precisely because the number looks like an error and is
actually an agreement.

### The pitch hypothesis, proposed and refuted the same day

Across four one-way sessions the fraction of frames the conditioning gate
flagged fell monotonically with how far down the camera was aimed — 46% at
19.6°, 17% at 25.0°, 9% at 27.3°, 0% at 32.1° — which is exactly the mechanism
*Indoors will hit the degenerate case* predicts: floor in frame makes the
vertical axis observable.

The two loop sessions were then walked over one route at two pitches, 21.5° and
37.7°, at matched speed. **Both flagged 0%,** and the steeper one scored
*worse*. So pitch is not the causal variable; visible constraining geometry is,
and pitch was a confounded proxy for it across four different rooms. A
controlled pair beats a four-point correlation, and this is why the sessions
were asked for.

Aiming down is still the right advice — it is what puts a horizontal surface in
frame — but it does not predict whether depth odometry will work.

### Frame-to-model now works, and three things had to be true for it

Keyframes and per-voxel fusion were the diagnosed fix, and they were necessary
without being sufficient. `tools/test_depth_odometry.py` accumulates a rendered
120-frame walk with 0.5% depth noise, which is the test that decides this — the
pairwise cases above all passed throughout, because drift does not live in a
pair of frames. Maximum position error over 2.2 m:

```
  frame to frame (the baseline)       2.63 cm
  frame to model, keyframes fused     0.43 cm
  every frame folded in (the old bug) 1.39 cm
  unsmoothed depth                    6.62 cm
```

Three separate defects had to come out, and each was found by measuring rather
than by reasoning about the code:

**The map was rendering one pixel per voxel.** A 3 cm voxel covers about three
pixels across at two metres, so a one-pixel splat left the rendered map 14%
filled and ICP found a target for one source point in seven. Drawing each map
sample at the size it actually subtends took association from 14% to 95%.

**The z-buffer was picking the near tail of the noise.** Sensor noise spreads a
surface into a slab a couple of centimetres thick; nearest-wins per pixel then
samples its leading edge, measured at **−2.0 cm** against the true depth. Every
keyframe wrote that bias back into the map, which is what made the error grow
monotonically rather than wander. Averaging within a slab's depth of the
nearest sample, re-centred once so the window is symmetric, brings it to −0.8 cm.

**Nothing detected degenerate geometry.** Held level, the camera sees two walls
with the floor and ceiling outside the 49.3° vertical view, and vertical motion
becomes unobservable — the point-to-plane Hessian goes singular while ICP still
reports 95% inliers. `lstsq` answered anyway, and one such frame cost the whole
trajectory: 160 cm over 2.2 m. Solving through the eigendecomposition and
dropping directions below a thousandth of the strongest bounds it to 23 cm, of
which none is recoverable — nothing measured that axis. The tracker now reports
the conditioning per frame, refuses to fold an under-constrained pose into the
map, and says how many frames were blind.

**Smoothing the depth first is most of the result**, which was the surprise.
Normals come from differencing pixels two apart — a 2 cm baseline at 2 m — so a
centimetre of per-pixel noise makes them nearly random, and point-to-plane ICP
is built entirely on normals. It is 6.62 cm against 0.43 cm, so `--raw-depth`
exists to keep the effect visible instead of assumed.

What this does *not* establish: the scene is a synthetic box, fully in view from
the first frame, which is generous to both paths and especially to the baseline
— frame-to-frame never faces the situation a map exists for. The result to trust
is the shape rather than the margin: the baseline grows with the frame count and
the map does not. `--frame-to-frame` keeps the baseline available on real data.

## The honest summary

Rectified ultra-wide frames are ready. Poses for them are a project, not a
patch: an offline VIO or SfM pipeline with a scale anchor, against an ARKit
baseline that drifts about 0.02 m/s and costs nothing. Whether that trade is
worth 96.3° instead of 73.1° is a question about the training mix, not about the
phone.

Depth odometry is the candidate for that pose source, and the honest position
moved a long way in one day. Loop closure first put it 13–34× behind ARKit;
correcting one thing — how the map is searched — put it within 2–4×, ahead of
its own frame-to-frame baseline, on both sessions. The remaining known gaps are
the ones the LiDAR-inertial literature closed years ago and this estimator has
not: no inertial term at all, a fixed degeneracy cutoff, a fixed correspondence
threshold.

So the useful summary is not *depth odometry loses*. It is that a
deliberately-minimal implementation, missing the single most standard component
in its field, lands within a small factor of a mature commercial VIO. That is a
reason to build the next piece, not to stop.

None of which would have been visible a day earlier. Every number before the
loop sessions was scored against ARKit, and against ARKit the broken and the
working configurations were nearly indistinguishable — 26.9 against 27.3 cm/s.
The measurement that separates them is a walk that comes back to where it
started.

## How the scoring got trustworthy, in three steps

The loop measurement above is the first score on this page that needs no
reference. Getting there took retiring two earlier ones, and both failures are
worth keeping because both looked like results at the time.

**An ARKit comparison at an unknown rate.** The first real walk scored 7.2 cm/s
frame-to-frame, against ARKit's ~2 cm/s. But 30.44 m over 32.2 s is 0.95 m/s,
which at 30 Hz puts frames 3.2 cm apart — and the run reported them 19 cm apart,
which is 0.95 m/s at **5 Hz**. The clip was scored through a stride, at a
spacing this page calls unusable. The tool printed the rate depth *arrived* at
and silently scored whatever survived the stride and the pose join; it now
prints both — `depth arrived at 30.0 Hz, scored at 30.0 Hz` — and says how many
frames the join dropped.

**A geometric-consistency score that ranks backwards.** With no loop available,
the obvious reference-free measure is how well a trajectory explains the depth
it was computed from: median point-to-plane residual, computed identically for
the estimate and for ARKit over the same frame pairs.
`tools/depth_odometry.py` reports it. It does not work as a quality score, and
the way it fails is instructive:

```
  session   weak cond   residual est   residual ARKit   ratio    drift
  b4b41a       0%          3.1 mm          3.3 mm       0.94    0.6 cm/s
  b36df8      17%          1.5 mm          1.8 mm       0.83    2.1 cm/s
  532cea       9%          3.0 mm          3.7 mm       0.81   26.9 cm/s
  31c6aa      46%          3.3 mm          5.1 mm       0.65   13.7 cm/s
```

The estimate beats ARKit on residual in **every** session, and beats it hardest
in the worst ones. Ranking by residual picks the diverging trajectories as the
best. The mechanism is the one this page already recorded from the dark session
— along directions the geometry barely constrains, a large wrong translation
buys a small residual gain — and it turns out to have nothing to do with
lighting. The residual tracks conditioning, not accuracy, and conditioning is
measured directly and better.

So the residual is a sanity check, not a score. It would catch a grossly wrong
pose. It cannot tell a working trajectory from a diverging one, and the tool
says so where it prints it.

**Loop closure.** Two walks that return to their starting point, marked on the
floor, with a pause at each end. That is the measurement above.

It also scored the frames ARKit had not converged on, whose pose sits near the
origin by construction and then jumps when it converges. That is a fault in the
reference rather than in the estimator. `tools/export_episodes.py` has always
dropped them; the scorer now does too.

## The lighting question, now closed

For a while every depth-odometry figure on this page came from a session ARKit
rates at **98.4% `ARConfidenceLevel.low`**, with not one high-confidence pixel in
30 metres, recorded at 3:34 in the morning. ARKit's depth is guided by the colour
image, so a dark room returns a full-looking map the sensor does not stand
behind, and those numbers measured the lighting rather than the method.

Well-lit sessions exist now — six of seven clear the bar, several at 61–69%
high-confidence pixels — and the verdict did not move in the direction that
excuse predicted. It got *worse*: the dark session scored 7.2 cm/s
frame-to-frame and the well-lit ones score 26.9 and 13.7. Lighting was a real
defect in the measurement and not the reason the method loses.

The debugging that found it is worth keeping, because each step ruled something
out by measurement:

| suspected | test | verdict |
| --- | --- | --- |
| ICP itself | register a frame against itself | exact — not it |
| basis change | all 24 signed axis permutations | best 40.8 mm vs ICP's 12.7 — not it |
| depth/pose timing | sweep ±67 ms | flat — not it |
| depth intrinsics scale | sweep fx ×0.5 … ×2 | ~4× magnitude error at every scale — not it |
| weak-direction blowup | Tikhonov damping sweep | monotone improvement, but λ=1 is just identity |

The finding that resolves it: ICP's answer fits the depth *better* than ARKit's
own pose does — 12.7 mm of point-to-plane residual against 40.2 mm at the
reference. It was never mis-solving. It was solving faithfully for data with no
signal in it, and along directions the geometry barely constrains, a large wrong
translation buys a small residual gain.

`tools/depth_odometry.py` prints the confidence distribution first and refuses to
quietly score a session that fails it, so this cannot recur silently.

The direction that survives is fusion rather than replacement: depth has to
*improve* a trajectory rather than carry one, which is what the metric-scale
advantage is actually good for.

## The fusion design

The plan is not to replace ARKit's pose but to correct it, which is a much
easier bar and the one the metric-scale advantage is actually good for. The
whole of it is one idea: **damp toward ARKit, not toward zero.**

Tikhonov damping is the obvious response to an ill-conditioned point-to-plane
solve, and applied the obvious way it fails. Shrinking the solution toward the
origin asks the estimator to prefer *"the camera did not move"*, and on a real
session the sweep was monotone all the way to the point where the answer had
simply become the identity — the error only matched the truth's magnitude
because it had stopped answering. Shrinking toward the prior asks it to prefer
*"whatever ARKit said"*, which is a good default rather than an absurd one.

`anchor_to_prior` does it in the eigenbasis of the point-to-plane Hessian. Each
direction keeps the fraction

```
    ev / (ev + lambda * ev_max)
```

of ICP's deviation from the prior. A direction the geometry pins hard passes
through almost untouched; one it barely sees stays where ARKit put it. That is
the same conditioning number the tool already reports per frame, used as a
weight instead of as a warning.

Measured on rendered walks, against a simulated ARKit that drifts by a random
walk of 4 mm a frame:

| path | ARKit alone | depth alone | fused |
| --- | --- | --- | --- |
| good geometry (pitched down) | 8.19 cm | 18.99 cm | **1.52 cm** |
| degenerate (level, floor out of frame) | 8.04 cm | 27.61 cm | **2.53 cm** |

The second row is the one that matters. Depth alone is at its *worst* there —
worse than ARKit by 3× — and fusion still improves on ARKit by more than 3×
rather than being dragged down. A fusion that only helped when depth was already good would be
worth nothing, because that is not when help is needed.

`lambda` defaults to **0.02**. The sweep keeps improving down to 0.005, but real
sessions measure conditioning in the 5e-3 to 1e-2 band, so a smaller lambda
would wave through exactly the directions the real data cannot see. At 0.02 a
direction at cond 0.01 keeps a third of its correction and a well-observed one
keeps 98%. Worth re-tuning once a session exists whose depth is not
lighting-limited.

**The prior does not have to be ARKit, and that was an error here.** This page
previously said fusion applies only on the ARKit path, because the ultra-wide is
reached by leaving ARKit and there is no pose left to anchor to. That is wrong.
`CMDeviceMotion` is independent of ARKit and survives the switch intact — 100 Hz
bias-corrected gyro, gravity, and user acceleration. An IMU prior is what the
LiDAR-inertial literature anchors to in the first place; ARKit was the unusual
choice, not the necessary one. `anchor_to_prior` is unchanged by this; only
where the prior comes from is.

## Frame-to-map: the association was the whole problem

For a while this section said the map path collapsed on real data and that the
verdict was suspended pending a well-lit session. Well-lit sessions arrived and
it collapsed harder: 18.3% and 27.1% of path length against 2.7% and 4.4% for
the frame-to-frame baseline it exists to beat. Seven times worse, in the
direction opposite to the synthetic result, with the conditioning gate flagging
almost nothing. That gap was the thing to explain.

**The association was projective.** `render_map()` rasterised the map into a
depth image *at the predicted pose* and matched per pixel. That closes a loop: a
poor prediction renders the map in the wrong place, which yields poor
correspondences, which yields a worse pose, which predicts worse. Frame-to-frame
has no such loop — the previous frame sits where it sits regardless of the
prediction — which is why it merely drifted where the map diverged. At 73°
horizontal there is little margin for a prediction to be wrong in.

No system in the literature registers this way. FAST-LIO2 searches an
incremental kd-tree of raw points; KISS-ICP searches a voxel-hashed point map.
`LocalMap` was already a voxel hash, so the fix was a change of lookup and not
of data structure: for each source point, the voxel it falls in plus its 26
neighbours, nearest fused point wins, rejected beyond `max_dist`. The rendered
path is still there behind `--projective-association`, because a 20× claim
should stay checkable.

Match rates say the search is not the fragile part: median 0.99, minimum 0.78
across 1104 frames. The `fill` term that drove keyframe insertion moved from
"fraction of rendered pixels the map covered" to "fraction of source points that
found a correspondence", and never once fell below its 0.6 threshold — so that
threshold is now doing nothing and should be revisited on its own evidence.

### The synthetic test was rewarding the wrong thing

The same change makes the *rendered* benchmark worse — the accumulated 120-frame
walk goes from 0.43 cm to 1.37 cm — while making the real sessions 20× better.
That is the clearest statement available of how unrepresentative the synthetic
scene is: a closed box, fully in view from the first frame, so the render is
always well filled and the prediction always nearly right. It is generous to
projective association in exactly the way a real room is not, and it was
scoring a method that fails in reality above one that works.

The keyframe ordering flipped with it. `every frame folded in` was the
configuration that diverged under projective association, and the test asserted
keyframes beat it. Synthetically every-frame is now ahead, 1.15 cm against
1.37 cm, and on the two loop sessions the two split one each — keyframes 1.7%
against every-frame 3.3% on one, 2.9% against 1.9% on the other. Neither
ordering is established any more, so the test prints both and asserts neither.
FAST-LIO2 keeps no keyframes at all, which is the direction this points; there
is not yet evidence to follow it.

## What the working systems do that this does not

Narrow-field depth registration is not an unexplored problem — it is what
solid-state LiDAR odometry has been solving since Livox shipped sensors with a
70° field of view. Four differences separate this implementation from the ones
that work, and none of them is exotic.

**No inertial coupling at all.** The estimator here has never read
`motion.jsonl`. Its prediction is a constant-velocity extrapolation of the last
pose, and in a direction the geometry cannot see it "coasts on the prediction" —
that is, on constant velocity. Every current LiDAR-inertial system is instead
tightly coupled: [FAST-LIO2](https://arxiv.org/abs/2107.06829) and Point-LIO
through an iterated EKF, [LIO-SAM](https://arxiv.org/pdf/2007.00258) through
preintegrated IMU factors in a graph. The reason it matters is structural rather
than incremental: in an IEKF the IMU-propagated state is a *prior with a
covariance*, so a direction the scan match cannot constrain simply keeps the
inertial prediction, with no hand-set eigenvalue cutoff anywhere. This file
instead solves with `lstsq` at a fixed `rcond` and reports a conditioning number
as a warning. The literature's own summary of the case is blunt: IMU
measurements are the standard mitigation for LiDAR degeneration in featureless
environments.

**Rendered association instead of nearest-neighbour search.** Covered above.
FAST-LIO2 uses an incremental kd-tree over raw points; [KISS-ICP](https://arxiv.org/abs/2209.15397)
uses a voxel-hashed point map. `LocalMap` here is already a voxel hash — the
keys, fused points and fused normals are all present — so this is a change of
lookup, not of data structure.

**Degeneracy handled as a hard threshold rather than a distribution.** The gate
here drops eigen-directions below a thousandth of the strongest. Two published
methods do better on exactly this problem:
[X-ICP](https://arxiv.org/abs/2211.16335) (Tuna, Nubert, Nava, Khattak, Hutter,
T-RO 2023) analyses alignment strength against the optimisation's principal
directions and then constrains the update, leaving degenerate directions
untouched rather than damping them arbitrarily; and
[DRPM](https://arxiv.org/abs/2410.10784) (Hatleskog and Alexis, RA-L) is
point-to-plane specific — it models how noise in the points *and the surface
normals* propagates into the Hessian, converts that into a degeneracy
probability, and derives its parameters from the sensor's datasheet instead of a
tuned cutoff. Code at [ntnu-arl/drpm](https://github.com/ntnu-arl/drpm).

DRPM is the closest published work to this sensor's problem, because normal
noise is this sensor's problem: normals come from differencing a 256×192 map two
pixels apart, and *Smoothing the depth first is most of the result* above
measures what that costs — 6.62 cm against 0.43 cm.

**A fixed correspondence threshold.** `max_dist` is pinned at 0.15 m. KISS-ICP
derives its data-association threshold from the recent history of motion
deviation, which is most of why it needs almost no per-sensor tuning.

One system deliberately omits the IMU — KISS-ICP reaches 0.50% relative
translation error on KITTI with a constant-velocity model and nothing else. It
is worth knowing why that does not transfer: KITTI is a vehicle carrying a 360°
spinning LiDAR on smooth trajectories. A handheld 73° depth camera indoors is
the opposite regime on both axes.

### The order to try them in

Each step is independently scorable now that loop closure exists.

1. ~~**Nearest-neighbour association** against the voxel map, replacing the
   render.~~ **Done, and it was worth 20×.** Frame-to-map goes from 18.3% /
   27.1% to 1.7% / 2.9%, past the frame-to-frame baseline and to within two to
   four times ARKit.
2. ~~**IMU as the prediction**, replacing constant velocity~~ **Done, and it is
   a wash — see below.** The plumbing and the frame convention are now
   measured and in place, behind `--imu`. The half that matters, IMU as the
   prior `anchor_to_prior` damps toward in degenerate directions, is still
   untested and still the largest single thing this estimator is missing.
3. **DRPM-style probabilistic degeneracy** in place of the `rcond` cutoff.
4. **A registration-quality gate on map insertion**, and an adaptive
   `max_dist`. Today the only gate is conditioning; a frame that registered
   badly is still folded in, and a keyframe at a wrong pose poisons every later
   registration against it. The keyframe rule itself needs re-deriving — its
   `fill` threshold has stopped firing at all, and whether keyframes beat
   every-frame is no longer established either way.

Current baselines to beat: **frame-to-map 1.7% / 2.9%**, frame-to-frame 2.7% /
4.4%, ARKit 1.4% / 0.8%.

### The IMU is wired in, and predicting with it is a wash

`--imu` replaces the constant-velocity prediction with CoreMotion's relative
attitude and measured acceleration over the depth-frame interval. Loop closure:

```
                    baseline    --imu     ARKit
  5bd1ed              1.7%       2.2%      1.4%
  cb4586              2.9%       1.3%      0.8%
```

One session each way. The mean improves, 2.3% to 1.75%, and two sessions
splitting is not a result — this page refuted a four-point correlation with a
two-session controlled pair earlier the same day and should hold itself to that.
`--imu` stays opt-in.

That it changes little was predictable and worth stating in advance next time.
The acceleration term contributes `½·a·Δt²`, and at 30 Hz that is **half a
millimetre** for a full 1 m/s² — it cannot move a prediction much. What the
literature uses an IMU for is not a better prediction; it is a prior with a
covariance in the directions the geometry cannot see. Neither loop session has
those: the conditioning gate flags 0% and 1% of frames. **The experiment that
would test the useful half needs a loop-closed session that is geometrically
degenerate** — phone held level, floor out of frame, the configuration that made
one earlier walk 46% blind. That session does not exist yet.

### The extrinsic is measured, and getting it wrong cost a day's confidence

Predicting from CoreMotion needs the fixed rotation from the IMU's device frame
to the frame the estimator works in, and this repository has already shipped one
transposed rotation prior, so it was measured rather than assumed — two
independent ways over 11,329 samples from seven sessions. Gravity is the same
physical direction in `pose.gravX/Y/Z` and `motion.gx/gy/gz`, which fixes it by
Kabsch; relative rotation axes fix it again using yaw, which gravity cannot see.
The two agree within 0.24–3.45° per session and the joint fit lands **0.288°
from an exact signed permutation**, so the permutation is what the code uses:

```
  [ 0  -1   0]
  [ 1   0   0]      90° about Z: portrait-natural device frame against the
  [ 0   0   1]      camera's native landscape frame
```

Then it was wired into the wrong frame. That permutation maps device to **ARKit
camera** coordinates, because `pose.gravX/Y/Z` is defined there. The tracker does
not work in ARKit camera coordinates — it works in the depth frame, +Z forward
and +Y down, which `ARKIT_TO_DEPTH` produces and which the reference trajectory
is converted into before scoring. Missing that conjugation flips the pitch and
yaw components of every rotation increment and leaves roll alone. Measured
against what the tracker actually consumes: **1.165° per frame at 30 Hz**, which
is 35° of injected error per second, and the loop error went to 40.6% and 40.8%.
With the conjugation it is 0.094°.

The verification did not catch it, and the reason is the part worth keeping.
The check compared the transformed IMU rotation against **ARKit's** relative
rotation — and ARKit's camera frame is the one place the transform was already
correct. A transform has to be verified in the frame that consumes it, not
against the reference it was derived from. A check that never visits the
consumer cannot fail.

### A note on the name

This page and the tool have called the map path *frame-to-model*, which is
KinectFusion's term for raycasting a TSDF. What is here is a voxel map of fused
points and normals, and the literature calls registering against one
*scan-to-map*. `LocalMap`, `self.map` and `render_map` already say map.

The prose on this page now says map. `tools/test_depth_odometry.py` still prints
`frame to model` and the quoted block above is its output verbatim, so the two
disagree until that label is changed — worth doing on the next pass through that
file rather than as a rename commit of its own.
