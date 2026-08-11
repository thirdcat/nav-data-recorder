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
  frame to map, keyframes fused       0.43 cm
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

A later measurement from the pair-wise side puts a number on it. Over 164 frame
pairs with a known reference pose, the correlation between a registration's pose
error and its own point-to-plane residual is **r = −0.057** — nothing. The worst
pair in that set is off by 17.59 cm while its residual sits at 1.62 mm, the
smallest in the table. What does track the error is the Hessian eigenvalue ratio.
This is the same finding as above, arrived at independently and with the
reference supplied rather than inferred, and it extends the conclusion in a
direction that matters for tuning: **the residual scale is not a source of
weights either.** Normalising a sensor block by its own robust residual scale
sounds principled and is close to uninformative here, because the quantity being
normalised is not measuring what the weight is supposed to express. The lever
that does exist is correspondence gating, not an isotropic σ.

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

### What tuning is worth, measured before building anything

Loop closure makes a parameter sweep possible for the first time, and the point
of running one is not to pick values — two sessions cannot choose values without
being overfitted — but to find out whether the remaining error lives in the
parameters at all. One factor at a time around the defaults, loop error as
percent of path, the two loop sessions:

```
  max_dist        0.05  2.15 / 2.94     0.10  1.74 / 2.87
                  0.15  1.74 / 2.87     0.25  1.74 / 2.87    0.40  1.74 / 2.87

  voxel           0.02  4.01 / 2.85     0.03  1.74 / 2.87    0.04  2.07 / 2.36
                  0.05  0.55 / 1.81     0.06  0.77 / 1.92    0.08  0.78 / 2.19
                  0.12  3.12 / 1.90

  keyframe_dist   0.00  3.33 / 1.91     0.02  2.86 / 3.29
                  0.05  1.74 / 2.87     0.10  1.87 / 2.25

  map_range       3.0   7.58 / 3.54     6.0   1.74 / 2.87    10.0  2.29 / 1.99
```

**`max_dist` is inert, for a structural reason.** Identical to two decimals at
0.10, 0.15, 0.25 and 0.40 — because nearest-neighbour association searches a
voxel and its 26 neighbours, so no correspondence can ever be farther than about
`√3 · 1.5 · voxel`, roughly 8 cm at the default. Above that the threshold cannot
bind. The search radius is set by `voxel`, not by `max_dist`. That retires the
adaptive-threshold idea borrowed from KISS-ICP: it would be tuning a parameter
that does nothing.

**`map_range` below about 6 m manufactures degeneracy.** At 3 m the conditioning
gate goes from 0 and 9 flagged frames to 196 and 186, and the loop error roughly
quadruples. Trimming the map removes the geometry that was constraining the
solve. It is the cleanest demonstration on this page that degeneracy is a
property of what is *in view*, not of the room.

**`voxel` is the one real lever, and the synthetic benchmark disagrees about
it.** On real sessions 0.05–0.08 is a broad basin, better than the 0.03 default
on both walks and by 3× on one, turning over by 0.12. At 0.05 the loop error is
**0.6% and 1.8%** against ARKit's 1.4% and 0.8% — the first time depth-only
odometry has beaten ARKit on anything scored without a reference, on one session
of two.

The default stays at 0.03 anyway, because switching it fails the rendered
benchmark two ways: accumulated drift goes 1.37 cm to 2.41 cm, and `smoothing
earns its place` **inverts** — unsmoothed depth becomes better than smoothed,
1.40 cm against 2.41 cm. That inversion is mechanically sensible rather than
noise: a 5 cm voxel already averages away what the depth smoother was there to
remove, and doing both over-smooths.

So real data and the rendered scene now prefer different voxel sizes. That is
the second independent sign that the synthetic scene is unrepresentative — the
first being that it scored projective association above the nearest-neighbour
search that is 20× better in reality. Which of the two is right about `voxel` is
not decidable from two walks in one building on one afternoon. `--voxel 0.05` is
one flag away for anyone who wants it, and a third loop session settles it.

### Vertical is most of the error, and it is a rotation problem

Splitting the loop-closure error into components says the same thing on every
session:

```
  session                total     vertical    share
  20260809-074415-5bd1ed  0.41 m    0.40 m      98%
  20260809-074458-cb4586  0.66 m    0.33 m      50%
  20260810-114339-1696fa  1.65 m    1.35 m      82%
```

ARKit's own vertical wanders 7–14 cm over the same walks, which is what a
person on a flat floor looks like. Seven explanations were measured and
discarded before the eighth held:

| suspected | test | verdict |
| --- | --- | --- |
| the floor leaves frame | fraction of points with horizontal normals | 30–52% — not it |
| normals are noisy | scatter against incidence angle | 0.4–1.2° across the range — not it |
| grazing incidence ruins them | same, binned 0–85° | worst bin is *head-on* on one session — not it |
| the vertical axis is weakly conditioned | its eigenvalue against the strongest | 20–50× above the gate — not it |
| the conditioning number mixes units | refit with a length-normalised Hessian | ordering unchanged, all scenes ≈2 m — not it |
| the gate is too permissive | sweep `min_conditioning` | 5e-3 helps one session 7×, hurts another 3× — not it |
| a corridor is degenerate | shoot one deliberately | best conditioning of any session, 0/760 flagged — not it |

**The estimated attitude drifts against gravity.** Gravity is known in the
camera frame every frame, so `R · g_cam` must be constant if the trajectory is
right. It is not:

```
  session                 attitude drift, end / max     vertical error
  20260809-074415-5bd1ed        8.34° / 12.45°             0.403 m
  20260809-074458-cb4586        9.07° / 11.22°             0.328 m
  20260810-114339-1696fa       34.64° / 35.44°             1.347 m
```

Perfectly rank-ordered with the vertical error, and the magnitudes agree: tilt
by θ and forward motion d leaks `d·sin θ` into vertical, which over 19 m at 8°
is about 1.3 m.

This is an error that need not exist. *Gravity fixes two of the three rotation
axes outright*, at the top of this page, has been true the whole time — the
estimator solves all six degrees of freedom from depth and ignores the two that
are given free and drift-free.

### Correcting it after the fact makes things worse, and that is informative

`--gravity-lock` rotates each solved pose so `R · g_cam` returns to the
reference. It works, exactly: attitude drift goes to **0.00°** on all three
sessions. The trajectories get much worse anyway.

```
  session      baseline   gravity-lock   gain 0.2   + floor-lock
  5bd1ed        0.41 m      24.64 m       21.02 m     11.71 m
  cb4586        0.66 m      10.26 m       10.92 m      9.85 m
  1696fa        1.65 m       1.43 m        0.75 m     57.81 m
```

Without the map it is harmless — frame-to-frame with the same correction scores
0.735 m against a 0.828 m baseline on one session and is unchanged on another.
So the map is what breaks: it accumulates in the drifted frame, and snapping the
pose to gravity every frame while the map stays where it was sets the two
fighting. The one session it helps, 1696fa, is the one whose drift is 34° —
there the drift costs more than the fight.

`--floor-lock` is the same story from the other side. Camera height above a
gravity-fitted floor plane is a genuine per-frame absolute observable — the
estimates are 1.18–1.23 m ± 0.03, which is a hand holding a phone — and it cuts
vertical error on all three sessions, by 8.5× on the worst. But total error
rises on two of them, because forcing the height while the attitude is still
tilted just moves the error sideways.

The obvious next move was to put the correction *inside* the solve rather than
after it — `anchor_to_prior` damps each direction by how well the geometry sees
it, so ICP would return a pose consistent with both the map and gravity instead
of being overruled. `--gravity-anchor` does that. **Eleven of its twelve
configurations diverged before the walk ended** (λ of 0.005, 0.02 and 0.1, with
and without the floor lock, across three sessions). Only λ=0.1 on the
worst-drifting session finished, at 1.158 m against a 1.653 m baseline.

Two failures with the same shape are worth more than either alone. Snapping
after the solve fights the map; damping inside the solve still lets ICP put the
rotation somewhere wrong and then argues with it. Both are attempts to reconcile
a rotation that has *already* gone 28–66° astray, against a map built while it
went.

### The rotation was never ours to solve

Gravity fixes roll and pitch absolutely — the first page of this document has
said so all along. What that was missing is how much more the IMU knows.
`CMDeviceMotion` reports a fused attitude, and measured against ARKit's across
all thirteen recorded sessions, aligned at the first frame:

```
  median 0.4–3.4°,  p90 0.6–4.0°,  at the end 0.5–3.5°
```

The end value equals the median. Over twenty to fifty seconds it does not
accumulate at all — **including yaw**, which gravity cannot see and which the
gyro carries well enough at this length that it never becomes the problem.

Set against the same sessions' depth-ICP attitude drift of 28–66°, the position
is stark: the IMU knows the rotation to about a degree, and the estimator is
solving it wrong by up to sixty-six. That is not an error to correct. It is a
quantity that should never have been an unknown.

So `--imu-rotation` takes the whole attitude from `CMDeviceMotion` and leaves
ICP a three-degree-of-freedom translation problem — the point-to-plane Jacobian
reduces to the normals, and the normal equations to a 3×3. The map is built at
IMU-rotated poses from the first frame, so there is no drifted frame for a
correction to fight. Loop closure over all eight loop-closed sessions, as
absolute end-to-start distance:

```
  session   ARKit     baseline   --imu-rotation   + --floor-lock
  1696fa    0.126 m    1.653 m       0.123 m         0.182 m
  f0d073    0.286 m    4.724 m       0.694 m         1.054 m
  1868dd    0.378 m    8.451 m       1.381 m         2.668 m
  683ef1    0.490 m    4.143 m       1.137 m         0.621 m
  cb4586    0.146 m    0.663 m       0.196 m         0.030 m
  3c7c6b    0.127 m    3.024 m       1.942 m         2.334 m
  5bd1ed    0.269 m    0.410 m       0.322 m         0.346 m
  5acd1b    0.055 m    1.495 m       3.078 m †       1.055 m
```

† `5acd1b` is the session whose loop score moves 5.5× under a 0.03% change in the
depth intrinsic scale, for the reasons in the next section. Read that row as
"between one and three metres, and worse with the flag than without", not as
three decimal places.

Every cell above was re-measured independently against the frozen tracker, and
fifteen of the sixteen reproduced exactly. The exception is `5acd1b` under
`--imu-rotation`, which had been recorded as identical to its own baseline; the
rotation was in fact applied on all 676 frames and the session gets **worse**.

Seven of eight improve, by up to thirteen times, and **two beat ARKit** — 1696fa
at 0.123 m against 0.126, and cb4586 with the floor lock at 0.030 m against
0.146. The eighth does not merely fail to improve: taking the rotation from the
IMU roughly doubles its loop error. Attitude drift settles at 0.3–3.5°, which is
the IMU's own error against ARKit rather than zero; a zero there would have meant
the rotation was being snapped again rather than simply taken.

Read the metres, not the percentages the tool prints. Those are relative to the
*estimated* path length, which shrinks when the rotation stops wandering: 5bd1ed
improves from 0.410 m to 0.322 m and both read as 1.7%.

### A 0.03% error in the depth intrinsics moved a session by 5.5×

The tool used to recover the depth intrinsic scale from the principal point,
`s = depth_width / (2·cx)`, reasoning in a comment that `cx` sits within a pixel
or two of half the colour width and that this beats joining another stream for
the ratio. It does sit that close — and that is precisely why the derivation
fails. A half-pixel is 0.03% of the width, and it is being used as the
*definition* of the scale rather than as an approximation to it.

What the data says, across all thirteen sessions:

```
  colour frame (frames.jsonl)   1920x1440 in 13 of 13
  depth map                     256x192 — exactly 7.5x down, exactly 4:3
  cx                            always ~959.0, never 960
  cy                            ~721.2 or ~728.7, up to 8.7 px off centre
```

The depth map is an exact integer downscale of the colour frame, so the scale is
uniformly `1/7.5` and the principal point rides along with it. `cy` sitting 8.7 px
below centre is the direct refutation of the assumption the old derivation rested
on: the principal point is not the image centre, in either axis.

[VLIO.md derives this convention](VLIO.md#the-depth-intrinsic-scale-and-why-it-is-uniform-175)
independently, from the pair-wise side, and owns it. What follows here is only
what the convention did to the trajectories.

The two `cy` values above are not two hardware configurations. They are one lens
at different focus positions, and a single session crosses between them — reading
only the first frame of each session is what made them look like two populations:

```
  31c6aa   fx 1278.4 – 1422.8 (11.3%)   cx 956.4 – 960.2   cy 721.2 – 729.1
  b4b41a   fx 1287.4 – 1354.1 ( 5.2%)   cx 957.4 – 961.1   cy 715.2 – 727.3
  1868dd   fx 1289.3 – 1355.1 ( 5.1%)   cx 958.4 – 960.0   cy 721.1 – 728.9
  5acd1b   fx 1336.5 – 1348.6 ( 0.9%)   cx 959.1 – 960.3   cy 728.5 – 729.0
```

Focus breathing moves the focal length by up to 11% within one recording. Reading
intrinsics per frame, which the tool does, handles that correctly. Deriving the
*depth scale* from `cx` does not, and the reason is sharper than a bias: **`cx`
crosses 960 mid-session in eight of the thirteen sessions**, so the old expression
did not merely mis-scale, it changed the sign of its own error partway through a
walk. A fixed downscale ratio was being re-estimated every frame from a quantity
that moves for unrelated reasons, giving a time-varying, scene-correlated error on
a constant of the sensor geometry.

The principal point is not a clean function of focus either. Within-session
correlation between `fx` and `cy` runs from −0.43 to +0.50 across the thirteen
sessions and is −0.002 in 1868dd, so `cy` is not simply riding along with the
focal length; it wanders substantially on its own. That makes deriving a fixed
ratio from it less defensible rather than more.

The general form of the rule: **a downscale ratio is a property of the buffer
sizes, not of the optics, so it must not be derived from an optical parameter.**
This repo reads intrinsics per frame *because* of focus breathing, which is
correct; the error was taking a frame-invariant quantity out of those per-frame
values. `fx` and the principal point have to move every frame, and the 7.5×
sensor downscale must not.

What it did is out of proportion to its size. Same code, same 676 frames, only
the convention changed:

```
  5acd1b, no flags      s = w/(2·cx)     8.254 m
                        uniform 1/7.5    1.495 m      ARKit 0.055 m
                        scale difference     0.03%
```

The same perturbation, measured pair-by-pair rather than accumulated, moves the
translation estimate by 0.005–0.067 cm. **Accumulation is what turns a
rounding-level input error into a metre-level output**, and it does it selectively:
sessions that converge barely move, while 5acd1b — the one session the tool
already flags as *worse than assuming no motion* — moves by 5.5×.

That sensitivity is itself the result for that session. A loop score which
responds to 0.03% like this is not a quantity to quote to three decimals, and the
table marks it rather than pretending otherwise.

### Two harnesses agreeing was not verification

This is also why the eight-session table above and the tool itself disagreed for
a whole day. The table was produced by a batch harness that scaled by
`1920`/`1440` — the correct convention — while `main()` used the principal-point
derivation. Both existed in the repo; neither was reading the other.

Worse, the check that was supposed to catch this is what hid it. An independent
harness written to verify the table reproduced two of its sessions **to three
decimal places**, which read as confirmation. It was not: both harnesses had
independently assumed 1920×1440, so they shared the one choice under test. Two
numbers matching means the numbers match. It says nothing about whether they are
right, and it says least of all when both were produced by someone starting from
the same obvious assumption.

The reference has to be fixed before the comparison, and it has to be fixed from
the data — here, `frames.jsonl` recording 1920×1440 and the depth map being an
exact 7.5× of it. Neither piece of code was evidence about the other.

### The image term was accurate about the wrong reference

Once the rotation comes from the IMU, ICP is solving three unknowns, and the
directions a depth map cannot see are exactly the ones an image gradient can. The
pair-wise measurement in
[VLIO.md](VLIO.md) established that the two terms are geometrically orthogonal
rather than redundant, and predicted that the gain would concentrate where the
point-to-plane conditioning is worst. Both terms were added to the same 3×3, with
fixed block scales chosen not to depend on the residuals, a Huber weight, a
five-level pyramid on a 480×360 grayscale image, and LM damping with step
rejection. All three configurations below take rotation from the IMU, so the only
thing that varies is which residual blocks are in the normal equations.

```
  session   ARKit     icp        + photometric
  1696fa    0.126 m   0.123 m      0.120 m
  f0d073    0.286 m   0.694 m      0.545 m
  5acd1b    0.055 m   3.078 m      2.675 m
  ------------------------------------------
  683ef1    0.490 m   1.137 m      1.383 m
  1868dd    0.378 m   1.381 m      2.152 m
  cb4586    0.146 m   0.196 m      3.386 m     17x worse
  5bd1ed    0.269 m   0.322 m      9.411 m     29x
  3c7c6b    0.127 m   1.942 m     56.026 m     29x
```

Three improve and five get worse, three of them by more than an order of
magnitude.

The tempting reading is that it helps where depth is weak and hurts where depth is
strong. That does not survive the correlation: depth-only loop error against damage
is r = −0.150, and 1696fa has the best depth-only score of the eight and is not
hurt at all. Two of the three sessions destroyed do happen to be among the best
depth-only runs, which is what makes the pattern look real, and it is not enough.

What does hold up is stronger and stranger. **Pair-wise quality does not predict
trajectory damage at all**, measured on the same sessions from the other side:

```
  session   baseline   photo median   photo p90    icp      + photo    damage
  cb4586     13.4 cm      0.69 cm      1.32 cm    0.196 m    3.386 m    17.3x
  5bd1ed     11.3         0.60         1.81       0.322      9.411      29.2x
  5acd1b      9.9         1.59        15.73       3.078      2.675       0.87x
```

The two sessions the term destroys have the **cleanest** pair-wise tails of the
eight. The session with by far the worst pair-wise tail is one of the three the
term improves. Across all eight, damage correlates with the pair-wise median at
r = +0.191 and with the p90 at r = +0.151 — nothing.

So the term is not making bad pair-wise estimates and then accumulating them. It
is doing well pair by pair, and the trajectory falls apart anyway. **The thing to
fix is how the term is applied, not the term.**

The long-baseline explanation is also ruled out rather than merely doubted. The
pair-wise measurements were taken on consecutive 5 Hz stills, which is the same
0.2 s spacing this pipeline has — 9.1–13.4 cm of travel, inside the basin, with
pair-wise medians of 0.46–1.59 cm. Distance between images is not the problem.

That left intermittency: this pipeline writes one image per six depth frames, so
the photometric block enters 16–17% of pairs and the other 84% are depth alone.
The control is to make it *rarer* — `--photometric-stride` halves it to about 8%,
changing nothing else — and if intermittency is the mechanism the damage should
grow. It does the opposite.

```
  session   icp       both 17%    both 8%
  1696fa    0.123 m    0.120 m     0.111 m
  f0d073    0.694      0.545       0.887
  1868dd    1.381      2.152       5.501
  683ef1    1.137      1.383       0.644
  cb4586    0.196      3.386       0.385
  3c7c6b    1.942     56.026       0.779
  5bd1ed    0.322      9.411       0.572
  5acd1b    3.078      2.675       2.250
```

Applying the term half as often is *better* in six of eight, and the three
sessions it had destroyed recover almost entirely — 3c7c6b from 56.026 m to
0.779, 5bd1ed from 9.411 to 0.572, cb4586 from 3.386 to 0.385. At 8% the term is
a coin flip against depth alone; at 17% it is harmful. **Damage scales with how
much the term is used**, which is what a noise source looks like, not an
intermittent-information source. The intermittency hypothesis is dead.

The conditioning split says the same thing from the other direction, and it is
the sharpest disagreement with the pair-wise result. Pooled over all eight
sessions, split by the same translation-only conditioning:

```
  conditioning   frames   icp median/p90    both median/p90
  < 0.01            280   2.64 / 7.29 cm    4.05 / 17.53 cm
  0.01 – 0.05       847   1.27 / 3.74       2.02 / 11.56
  >= 0.05         5,933   0.61 / 1.66       0.80 /  2.73
```

**`both` is worse in every band**, and worst in relative terms exactly where the
prediction said the gain would concentrate. Pair-wise, the image term was 4.9×
*better* than depth in the `cond < 0.01` band. In the accumulated solve it is 1.5×
worse in that same band. Whatever the term contributes pair-wise is not surviving
into the trajectory.

### The two blocks were anchored to different things

The remaining explanation was that the blocks disagree about what they are measured
against. The depth block is frame-to-map: it matches the current frame against
everything accumulated. The photometric block is frame-to-frame, and specifically
against the *previous image frame*, whose pose is itself an estimate carrying its
own error. On that account the image term is accurate about a reference that has
already drifted, and pulls the current pose toward a six-frames-old estimate while
the depth term pulls it toward the map.

That is testable with a flag that already existed. Under `--frame-to-frame` the
depth block anchors to the previous frame — the same *kind* of reference the
photometric block uses — and nothing else changes: same sessions, same frames, same
term, same weights.

```
  session   frame-to-frame   + photometric
  1696fa       0.463 m          0.254 m      1.8x better
  f0d073       1.105            0.765        1.4x
  1868dd       4.816            3.104        1.6x
  683ef1       5.122            1.782        2.9x
  cb4586       0.217            0.231        0.94x
  3c7c6b       4.876            2.872        1.7x
  5bd1ed       0.539            0.349        1.5x
  5acd1b       2.154            1.564        1.4x
```

**Seven of eight improve**, and the eighth loses 6%. The three sessions the term
had destroyed against the map are simply fine here: 3c7c6b improves instead of
going to 56 m, 5bd1ed improves instead of 9.4 m, cb4586 is level instead of 3.4 m.
**The catastrophes were the anchor mismatch, not the term.**

The pooled conditioning split shows the same thing in a form worth keeping:

```
  conditioning   frames   f2f median/p90     + photo median/p90
  < 0.01            360   6.815 / 37.931 cm   7.183 / 43.456 cm
  0.01 – 0.05     1,094   2.299 / 11.086      2.382 / 11.225
  >= 0.05         5,606   1.087 /  3.486      1.106 /  3.535
```

Against the map the image term was 1.53× worse in the degenerate band; anchored the
same way as depth it is 1.05× worse there, and the loop error nonetheless falls by
30–65%. Per-frame accuracy is marginally noisier and the accumulated result is much
better, which is what removing a *bias* looks like as opposed to removing noise.
The term is buying drift resistance and paying a little jitter for it.

There is a reason the damage was worst exactly where the geometry is weakest, and
it is not a coincidence: the degenerate direction is both the one the image sees
best and the one the anchor error leaks along, so the most informative axis is also
the most contaminated one. That is why `cond < 0.01` was the band with the largest
relative harm and is now the band with the largest gain.

A third setting was supposed to close the ladder — `--stride 6`, which decimates
depth onto the image frames so that both blocks anchor to the identical previous
frame — and it does not, because it is not a clean test. Matching the anchors that
way also stretches the depth baseline by six times, pushing both blocks toward the
edge of their convergence basins at once. Two variables move, so the column cannot
adjudicate anything; it came out 2 better and 4 worse of the six sessions where the
image join survived, and that is uninterpretable rather than contrary. The
frame-to-frame comparison is the one that changes a single variable, and it is
decisive.

So the fix is not to the term but to what it is measured against: a reference
carried by the map — a keyframe patch attached to the voxel planes, in the style
FAST-LIVO2 uses — rather than the previous image frame's estimated pose. Rendering
an image from the map would also work and costs a subsystem; the patch does not,
and this pipeline has pixel-aligned depth, which makes attaching one cheaper here
than in the papers.

The `--photometric --no-depth` column is not evidence about the term either way.
With the depth block off, the 84% of frames that have no image have no constraint
at all and coast on the prediction; it lands between 1.8 m and 49.4 m and says
only that photometry alone cannot carry a trajectory at this image rate.

Reweighting is the obvious move and is the wrong one. This page has already
recorded why: the residual scale carries almost no information about pose error,
so a σ ratio is not the lever it appears to be, and an intermittent constraint is
not something a weight can make continuous.

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
3. ~~**Gravity through `anchor_to_prior`**~~ **Done, and superseded.** Damping
   toward gravity inside the solve diverged eleven times out of twelve.
   **Taking the rotation from the IMU outright** is what worked — seven of
   eight sessions improved and two beat ARKit. It is also the piece that
   carries to the ultra-wide path, where CoreMotion is untouched by leaving
   ARKit. What is left there:
   - ~~`5acd1b` scores identically with the flag on and off, to three
     decimals.~~ **Closed, and it was a reporting error rather than a
     behaviour.** The flag was applied on all 676 frames; the cell had been
     filled in with the baseline. Re-measured, it goes 1.495 m → 3.078 m, so
     this is the session the rotation makes **worse**, not the one it leaves
     alone. A number identical to its own control to three decimals should have
     been read as a measurement that did not happen — that is not a result
     shape the estimator produces.
   - `3c7c6b` is the one session it does not help. It is also the one whose
     IMU attitude is worst against ARKit, at 3.4°.
   - The floor lock now has a rotation worth standing on: it is the best
     configuration on three sessions and the worst on three others, which is
     not yet a rule.
4. **Anchor the photometric residual to the map**, as a reference patch on the
   voxel planes rather than the previous image frame. This is now the highest
   value item on the list and it is not speculative: the term already improves
   seven of eight sessions once the two blocks share a reference, and the only
   reason it damages the default frame-to-map path is that its own reference is
   an estimated pose. A rendered image from the map would serve too and costs a
   subsystem; a patch does not, and pixel-aligned depth makes attaching one
   cheaper here than in the papers that do it.
5. **DRPM-style probabilistic degeneracy** in place of the `rcond` cutoff — but
   note the sweep found no threshold that serves all three sessions, and that
   conditioning was ruled out as the cause of the dominant error. This is
   further down than it looked.
6. **A registration-quality gate on map insertion.** Today the only gate is
   conditioning; a frame that registered badly is still folded in, and a
   keyframe at a wrong pose poisons every later registration against it. The
   keyframe rule itself needs re-deriving — its `fill` threshold has stopped
   firing at all, and whether keyframes beat every-frame is not established
   either way, in the sweep above or anywhere else.
   ~~and an adaptive `max_dist`~~ — dropped, the parameter is inert.
7. **A rendered scene that resembles a room.** The current one is a small closed
   box in full view from frame one, and it has now been wrong twice about
   changes that were large improvements on real data. It is still valuable as a
   regression guard, where it catches real breakage; it is not usable for
   choosing between designs.

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

The prose and the test now both say map.
