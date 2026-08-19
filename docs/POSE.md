# Where the pose comes from, and what replacing it would cost

Reaching the ultra-wide means leaving ARKit — see *Why not the 0.5x ultra-wide?*
in [DATA_FORMAT.md](DATA_FORMAT.md) — so this page is what would have to be
rebuilt. It now has been, far enough to measure. `MultiCamRecorder` records both
lenses with LiDAR depth and no tracker, `eval/pi3_poseless.py --lens` recovers
poses from either arm offline, and one walk now goes through both lenses at
once — so the lens can be compared without the path moving with it.

## What the ultra-wide route is worth, measured

Scored **without ARKit anywhere in the loop** — gravity from CoreMotion, which
never saw the estimator, and surface thickness against the depth sensor's own
0.77 cm floor. Sections below carry the workings.

**These numbers are from one walk through both lenses at once**, which the
recorder can now do. Every earlier version of this table scored three ultra-wide
walks against three *different* wide walks, so lens and path moved together. The
rows that changed when that confound came out are marked.

| question | answer | how it was settled |
|---|---|---|
| position | **wide leads 3 sessions of 4, the fourth a tie** — 9.20 / 10.38 / 9.70 / 9.42 cm against 10.98 / 11.73 / 10.08 / 10.85 | 40 cm-cell surface thickness, both arms of one walk. The three leads run 1.35-1.78 cm where injecting 2 cm of jitter moves the reading 1.9; the fourth is 0.38 and below what the cell resolves. **Reverses the old three-pair result**, which had the path moving with the lens. The *direction* survives cutting `d67f0a` to its unflagged span and survives removing the chain's free join scale; the per-session margins do not — they move to 0.81, and to 0.61 and 2.14 |
| rotation | **mixed, leaning wide** — 1.42/1.40, 0.71/1.20, 2.07/1.74, 0.96/1.76° wide against ultra-wide | gravity residual, device-to-camera rotation fitted not assumed. Wide leads two, ultra-wide one, one tie. **Withdrawn from "ultra-wide wins 3 of 3"** |
| window-to-window agreement | **ultra-wide leads 3 of 4, the fourth a dead tie — and in shape only** — 0.30/0.52/0.30/0.30 cm against 0.58/1.17/0.41/0.30 | median join residual. No longer the outright sweep it was: `57f29e`'s two arms land 0.003 cm apart. And it is measured *after* the join absorbs a freely fitted scale, so it cannot see the chain's scale, which drifts to 0.295 on the arm that leads this row |
| against ARKit | **both lenses trail** | ARKit residualises at 2.45-3.92° on gravity across three sessions; the arms above are scored on shorter walks and are not directly comparable |
| where the gain comes from | **the scene taken in, not the geometry** | masking the periphery costs as much as cropping it, so angular spread was not doing the work |
| resolution | **does not matter at this scale** | seen three ways: 640x480 beat 1920x1440 on rotation, resample-only is free, and matching the wide camera's angular resolution to the ultra-wide's costs nothing |
| rectification | **do not** | correcting the lens costs +3.19 cm; resampling through the identity is free, so the cost is the correction, and the ultra-wide is already 3-5x flatter than the wide lens |
| cropping | **do not** | the rectifier's trim from 102.5° to 96.3° throws away scene, which is the thing that was buying the gain |

**The obvious confound was tested and is not the answer.** The ultra-wide arm
reprojects its depth through the 19.272 mm extrinsic and the wide arm does not,
so the position row could have been reporting that transform rather than the
lens. Scaling the baseline to 0 and to 2x moves the coarse thickness by 0.29 cm
and 0.05 cm in the two sessions — below what the cell resolves — and deleting
the baseline outright still leaves the gap to the wide arm at 1.49 and 1.30 cm.
The position difference belongs to the lens and the cone, not to the transform.

That cuts the other way too: a 2x error in the baseline does not register in
this pipeline, so this measurement cannot validate the 19.272 mm magnitude
either. `calib/README.md` keeps it open, now with a second route that failed to
recover it.

**What this does not say.** Four sessions of one apartment, single
observations, and one of the four is a tie. Nothing here beats ARKit. The old
three-pair numbers are not deleted below; they are left in place with their
confound named, because the reason they pointed the other way is the more useful
thing to keep.

One measurement was withdrawn on the way. Revisit-based position scoring looked
like it worked and did not: injecting a known 5 cm offset returned 45 cm, and
the score surface turned out flat. It is documented below rather than deleted,
because the reason it failed is the reason to distrust the next thing that looks
like it works.

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

## Three measurements that redirect this track

Taken together the three below say the same thing from three sides: **the
non-ARKit pose source that works is Pi3X on its own, and depth ICP does not
improve it, is not rescued by fusing with it, and fails the unbiased checks.**
That is a redirect rather than a defeat — the reason to want a non-ARKit pose at
all is the multi-camera path, which is exclusive with ARKit, and Pi3X supplies
one offline where the GPU is available anyway.

All three hold **at the 5 Hz anchor rate we record**, which is the rate that
matters today. A later section finds the one regime where ICP does earn its
place: once the anchors are 3 s apart, as they are when thermal throttling stops
the image writes, conditioning-gated ICP recovers about a sixth of the error.

### The anchor rate is not the lever

Pi3X anchors sit exactly on the image frames — ARFrame spacing 12, which is
5 Hz, because that is the still rate. So "more anchors" is a capture setting
with a thermal bill, not a solver knob. Before paying it, thin the anchors we
have and watch the error grow (`eval/fuse_control.py --anchor-sweep`, scored on
the depth rows that are never anchors in any arm):

```
             5.00Hz  2.50Hz  1.67Hz  1.25Hz  0.83Hz  0.62Hz
  median       6.79    6.77    6.66    6.54    6.80    8.51   cm
  vs 5 Hz     1.00x   1.00x   1.01x   1.04x   1.17x   1.29x
```

Dropping to a quarter of the anchors costs 4 %. One anchor every 1.6 seconds
costs 29 %. **The 6.8 cm is the anchors' own error, not the interpolation
between them** — which arithmetic agrees with: at 0.6 m/s the gap between
anchors is 12 cm, and the sagitta of a straight line across it is millimetres.
Raising the still rate would buy nothing here, and that was the obvious
recommendation before this table existed.

### And the unbiased checks say ICP, not Pi3X, is the problem

Everything above is ATE against ARKit, which cannot referee a comparison where
one candidate resembles ARKit. `compare_tracks.py` already computes the two
checks that need no reference — how flat a walk on one floor comes out, and how
that plane sits against gravity:

```
              plane thickness, cm        plane tilt off gravity, deg
             ICP   Pi3X-i   ARKit        ICP    Pi3X-i   ARKit
   2735cf    7.3      1.4     3.6       86.58     4.60    5.94
   2be6a9    5.4      1.1     1.2       86.41     0.42    1.51
   3c7c6b   40.8      0.8     1.5       28.01     0.40    0.55
   dd2a13   12.5      1.1     1.5       83.05     1.55    1.60
   6b92f3   15.5      1.4     1.7       77.05     1.50    1.20
   5acd1b   22.4      1.2     1.0       10.21     0.50    0.51
   1868dd   25.8      2.9     0.8        0.69     0.48    0.12
   cb4586    4.6      1.5     1.0        1.60     0.65    0.14
```

Interpolated Pi3X lands at 0.8–2.9 cm of thickness against ARKit's 0.8–3.6 —
indistinguishable. ICP runs 2.6–40.8. And in four sessions of thirteen ICP's
fitted plane sits **77–87° off gravity**, which is not a drifting walk on a
floor; it is a trajectory that has lost the floor entirely.

No reference is involved in either column. Whatever ARKit's faults, they are not
what produced these.

## The rate fusion buys nothing, measured against the arm it was missing

`eval/fuse_rate.py` combines Pi3X's 5 Hz global anchors with ICP's 30 Hz local
motion and reported all eleven sessions passing. `HANDOVER.md` §6 flagged that
both of its criteria were guaranteed: the anchor criterion is an identity — the
fusion writes the Pi3X pose into anchor rows and the scoring compares that array
with itself, which is why `delta_cm` was exactly `0.0` — and the between-anchor
criterion compares against ICP alone, which drifts.

The arm that was never run is **interpolating the Pi3X anchors with no ICP at
all**. `eval/fuse_control.py` runs it, on the same anchors and the same scoring:

```
  between-anchor ATE against ARKit, cm
  session  frames   ICP alone   Pi3X only   fusion   ICP buys
   1696fa     630        5.23        3.79     3.85      -0.06
   1868dd    1270       39.63       22.43    22.50      -0.07
   2735cf     565       50.48        5.29     5.46      -0.16
   2be6a9     330       29.72        8.02     8.13      -0.11
   3c7c6b     535       56.93        5.59     5.77      -0.18
   5acd1b     560       99.71        3.73     4.30      -0.58
   5bd1ed     915       10.54        5.88     5.92      -0.04
   683ef1     535       45.03        2.59     3.21      -0.62
   6b92f3     415       70.71       81.73    82.15      -0.41
   cb4586     830        8.12        7.83     7.83      +0.00
   ce02ac     370       16.20        9.82     9.88      -0.06
   dd2a13     375       82.05       33.64    34.16      -0.52
   f0d073     575       20.85        6.79     6.91      -0.12
```

**ICP's relative motion helps in one session of thirteen, and that one is a
tie.** The median contribution is −0.12 cm: adding ICP between the anchors makes
the trajectory very slightly worse. Interpolating two Pi3X poses in a straight
rigid line does the job the 30 Hz estimator was there to do.

The other doubt in the same section is also settled, and negatively. The fusion
spreads its endpoint correction by left multiplication, so the rotation turns
the camera about the world origin; §6 worried that the lever arm would move
intermediate frames by centimetres. Pivoting about the left anchor instead
changes the median session by **0.00 cm**, and the largest change anywhere is
0.36 cm on a session already 34 cm out. It is not the problem.

**What this does and does not license.** The contrast is sound: both arms are
scored identically on the same rows, so the difference isolates ICP. The
absolute column is not — it is ATE against ARKit, and Pi3X and ARKit are known
to nearly agree, so "Pi3X interpolation scores well" is partly the statement
that Pi3X resembles the thing it is being scored against. What survives without
that caveat is the negative: whatever the reference, ICP is not adding to it.

## Depth conditioning ranks ICP's intervals, but only matters when the gaps are long

The section above kills the fusion at the rate we actually record. It leaves a
question, because a separate measurement says ICP's *local* motion is good:
per-frame `frame_conditioning` predicts frame-to-frame relative translation
error at Spearman **−0.446** over 11 054 pairs in 15 sessions, monotone across
all ten deciles, 4.4 mm median error in the top quartile against 12.1 mm in the
bottom. (The ICP *residual* predicts nothing, r = −0.057. Conditioning is the
signal; the thing the solver minimises is not.) If some of ICP's increments are
accurate to 4 mm, why does adding them make the trajectory worse?

**Because at 5 Hz there is nothing to add.** The anchors are 0.2 s and ~12 cm
apart and the sagitta of a straight line across that is millimetres — below
ICP's own best. So the question is not whether the gate works but whether any
regime gives it room. `eval/fuse_control.py --gate-sweep` thins the anchors and
tries five ways of choosing which gaps ICP is allowed to carry:

```
  anchors   interp      all  drop_worst_25  drop_worst_50   best_25  random_25   worst_25
   5.00Hz     6.79   -0.12/-0.23   -0.09/-0.08   -0.00/-0.05  -0.02/-0.02  -0.03/-0.05  -0.04/-0.15
   1.25Hz     6.54   -1.74/-2.04   -0.60/-0.58   -0.15/-0.09  -0.04/-0.02  -0.26/-0.60  -0.64/-1.48
   0.62Hz     8.51   -0.75/-3.44   +0.19/+0.01   +0.35/+0.83  +0.18/+0.47  -0.15/-0.45  -1.01/-3.57
   0.31Hz    21.25   +3.33/+0.25   +3.69/+2.93   +3.66/+3.45  +1.77/+1.76  +2.33/-2.11  +1.13/-2.47
                              cm recovered against pure interpolation, median/mean over 13 sessions
```

Two things are in that table.

**ICP crosses from harmful to useful somewhere near 0.6 Hz.** Below ~1.6 s of
gap the straight line wins; past ~3 s it does not, and at 0.31 Hz gated ICP
recovers 3.7 cm of a 21 cm error. That rate is not hypothetical — thermal
throttling stops image writes while poses keep recording (`HANDOVER.md` §2), so
a hot session has exactly this shape and this is the arm that covers it.

**Conditioning genuinely ranks the intervals, and the evidence is the mean.**
`best_25 > random_25 > worst_25` holds at every one of the four rates on the
mean: +1.76/−2.11/−2.47 at 0.31 Hz, +0.47/−0.45/−3.57 at 0.62, −0.02/−0.60/−1.48
at 1.25. The median does not separate them — at 0.31 Hz random scores +2.33
against best's +1.77 — and reading only the median would have said the gate
selects nothing. It does not select *the biggest wins*; it removes *the
disasters*. Dropping the badly conditioned half takes the worst session from
−32.9 cm to −2.5 cm while leaving the median gain where using everything put it.
That is why the table prints both, and why `drop_worst_50` rather than `best_25`
is the arm worth keeping: capping the tail costs none of the typical gain.

**Two things nearly made this a false negative.** The gate first thresholded
every frame in an interval against a session-wide quantile, which admits 13.5 %
of intervals at 5 Hz and **0 %** at 0.31 Hz — the sparse arm scored exactly
`+0.000` and read as "ICP is neutral" when it meant the gate never fired. Ranking
intervals holds the admitted fraction at 25 % regardless of rate. And the random
control is what separates "conditioning picks the right gaps" from "long gaps
favour ICP"; without it the +3.33 of the ungated arm would have been credited to
the gate.

## The same gate does nothing for metric scale

The natural next move, once conditioning is shown to rank ICP's intervals, is to
spend the well-conditioned frames on the other job depth does here. Pi3X is not
metric on its own; `eval/pi3_chain.py:137` fits each window's scale by least
squares against that window's LiDAR depth, gated on Pi3X's own confidence and
not on anything the depth frame knows about itself. Gating that fit on depth
quality is the obvious improvement.

**It is not supported.** Against `|log(scale ratio)|` over 1 430 anchor gaps:

```
                             per-pixel conf    frame cond   range   median err
  all sessions                     +0.116        -0.130     -0.233     +6.4%
  without the two near ones        -0.065        -0.037     -0.098     +5.7%
  and only scenes beyond 1 m       -0.036                              +5.8%
```

The apparent `+0.116` — the wrong sign to begin with — is entirely two
near-stationary sessions where everything in view sits under a metre. Control
for range and both signals go to zero. Per-session Spearmans flip from +0.283 to
−0.458, which is what noise looks like.

The reason is visible in hindsight and worth writing down, because conflating
these two is what made the idea look obvious: **conditioning measures pose
observability — whether the visible geometry pins six degrees of freedom — and
scale needs depth accuracy.** A flat wall filling the frame is terrible
conditioning and perfectly good ranging. They are different quantities, and only
the first one has anything to do with ICP. The scale fit already averages over
a thousand-odd pixels per window, which is enough to be indifferent to the
per-frame quality variation these sessions contain.

What this leaves open is the criterion, not the gate: every number above is
scale measured against ARKit. There was for a long time no reference-free way to
score whether a metric scale is *right* — path length needs a truth, loop
closure only speaks for loops, and plane thickness is scale-invariant.

**The last of those was wrong, and it was hiding a criterion.** Plane thickness
of *one cloud* is scale-invariant. A trajectory's thickness is not, because it
re-poses many clouds and leaves the LiDAR's own metres alone: stretch the
trajectory and the same wall, seen from two places, lands twice. Sweeping a
uniform scale therefore puts a minimum where trajectory and depth agree, and
`eval/scale_meter.py` reads it off.

It is calibrated against a trajectory already believed right, which is the check
inject-and-recover cannot substitute for. On `1696fa`'s ARKit reference — metric
to about one per cent by the tape measurement below — it reads **1.000**.
Pre-scaling that same trajectory by 1.15 and by 0.90 moves the minimum to 0.870
and 1.110, recovering **1.149** and **0.901**.

Its range is narrow and the tool says so rather than guessing. The well has to
clear the 0.7 cm this data resolves, and it only does on dense trajectories:
0.8-1.2 cm at plus or minus five per cent on 1696fa's 760 poses, against
0.02-0.56 cm on the 93-159-pose poseless dual-lens arms, two of six of which do
not bracket their minimum at all. It refuses those instead of printing an argmin.
So the criterion now exists for ARKit-grade trajectories and still does not exist
for the ones that most need it — the poseless arms, whose scale no reference
checks. Pose density is the likely difference, since 1696fa walks the same ten
metres with seven times the poses, but that has not been tested.

### The ruler settles it: ARKit is metric to about one per cent

Three sessions recorded a steel tape at a known extension — 1.000 m hung
vertically on a wardrobe door, 2.000 m laid along the floor in two others.
Measured from LiDAR depth alone, with no reference to any pose estimator's
opinion of scale:

```
  session   truth      measured    error   blade thickness p90
  505b2c    1.000 m    0.9711 m    -2.89%        2.0 cm
  e11854    2.000 m    2.0085 m    +0.42%        2.0 cm
  e9d8f7    2.000 m    2.0186 m    +0.93%        3.5 cm
```

The thickness column is the check that the thing measured was the blade: a
2 cm-thick collinear cluster is a tape, a floor is not. The two 2 m readings
agree with each other and with the truth to under a per cent; the 1 m reading is
3 % short, which is the length whose endpoints are least well defined (the case
sits on the floor and the hook is at the top, so what exactly the 1.000 m spans
is ambiguous in a way the floor-laid tape is not).

**So ARKit's metric scale is right, and the ~5 % session-scale disagreement
between Pi3X and ARKit recorded above belongs to Pi3X.** Before the tape, either
could have been the one that was off.

Getting here took three attempts and the two that failed are the instructive
part. Segmenting the blade per frame measured 0.19 m against a 2.000 m truth,
because partial visibility meant the "blade" it found was a fragment.
Accumulating every yellow surface point and fitting one line to all of them gave
3.95 m against a 1.000 m truth, because the wood grain is also warm-coloured and
also elongated. What worked was requiring the tape to be what it physically is:
one *connected* cluster, long and thin. Colour alone identified neither.

### And the accelerometer, the one candidate left, fails its positive control

`motion.jsonl` looked like the answer: `userAcceleration` is in physical units,
no pose estimator ever sees it, and scaling a trajectory scales its acceleration
exactly. The sampling is generous — IMU at 100 Hz against a 60 Hz pose stream.
Fit `s = <a_track, a_imu> / <a_imu, a_imu>` after rotating the IMU into the
world frame and low-passing both identically, and a metric trajectory should
return 1.0.

**ARKit's own trajectory returns 0.47.** Across 28 sessions the median is 0.47 at
a 1 Hz cutoff and 0.39 at 2 Hz, with `s_acc` tracking `rho` almost exactly
(0.59/0.59, 0.53/0.54, 0.52/0.53 …) — the signature of a regression attenuated
by noise in its regressor, not of a measured gain. So the two signals are not
describing the same motion, and a cross-spectrum says where they fail to:
**no band reaches coherence 0.5 anywhere**, the whole table topping out at 0.45
in 0.2–0.5 Hz. The physical explanation — the IMU sits in the body and the pose
is the camera's, so rotation about a few centimetres of lever arm loads the
accelerometer only — is refuted by splitting each session on rotation rate:
coherence is *lower* on the quiet halves (0.24 against 0.31, higher in 2 of 8).

The likeliest remaining cause is that ARKit's published pose is smoothed enough
that its second derivative is not the device's acceleration. Whatever it is, the
estimator cannot reproduce 1.0 for the trajectory the entire project is scored
against, so it cannot referee anything else.

Two consequences worth carrying forward. First, **absolute scale in this corpus
was unverifiable until a ruler was recorded**: LiDAR is metric but is the thing
being fitted, the IMU does not work, and planes, gravity and loop closure are
all scale-blind or ARKit-derived. That gap is now closed — see below — and the
way it closed is the cheap one: a known length in frame, once per session.
Second, the calibration that caught this was not the obvious one. Injecting a
known scale factor into the trajectory and checking it comes back would have
**passed trivially**, because scaling the trajectory scales `a_track` linearly
and any linear estimator returns `k · s`. Proportionality is not calibration.
What caught it was running the witness on a trajectory already believed metric.

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

## The map association reduces drift and does not repair the geometry

Frame-to-map beat frame-to-frame on loop error, 1.7 % and 2.9 % against 2.7 %
and 4.4 %, and that is recorded above. Loop error is a drift measure. The
failure this corpus actually has is a different quantity: a single depth frame
is dominated by floor, which constrains its own normal and little else, so the
horizontal is weakly determined however little the estimate drifts. A map
carries geometry from many frames and many viewpoints, so it is the obvious
place to look for a repair.

It is not there. Both paths run on the same three sessions, scored by the checks
that need no reference — how flat a walk on one floor comes out, and how that
plane sits against gravity:

```
              floor thickness    tilt off gravity
  2994fa  map      125.8 cm          23.01 deg
          frame    119.0             24.01
          ARKit      4.7              0.37

  5bd1ed  map       77.2             15.87
          frame     98.0              5.95
          ARKit      3.1              0.33

  2735cf  map      100.2              3.95
          frame     79.9              2.86
          ARKit      5.2              0.46
```

**Both depth paths reconstruct a floor about a metre thick**, against ARKit's
three to five centimetres, and the map is the worse of the two on two sessions
of three. The two estimates differ by 49 to 247 cm in the median pose, so this
is not a lever that failed to move.

The instrument was checked before the numbers were believed: rebasing the points
onto the pose they came from reproduces the original cloud to under a micrometre,
the npz reference matches each row's own ARKit pose exactly, and all 184 image
frames of 5bd1ed are present in the dump.

So changing the association does not fix it, and neither will a wider lens: the
LiDAR depth camera is a virtual device over the wide camera, so pairing it with
the ultra-wide in a multi-cam session leaves the depth frustum at 74.6°. **The
information the horizontal needs is not in the depth**, and the paths that
remain are the ones that use the photographs.

### The ultra-wide extrinsic was never missing

This page and the planning around it treated the ultra-wide to LiDAR transform
as an unknown needing a calibration rig. It is in `calib/`, read off the device:
the ultra-wide sits **19.272 mm** from the wide camera, rotated **0.462°**, and
since ARKit's depth already arrives in the wide camera's frame that transform is
the one a multi-cam path needs. Nothing has to be composed or measured.

What made it look missing is worth recording, because it is the same shape as
the other silent failures here. `MultiCamDepthProbe` read the *depth* frame's
`cameraCalibrationData.extrinsicMatrix`, got a translation of `(0, 0, 0)`, and
that was written down as "extrinsics unavailable". But the extrinsic is
expressed relative to a reference camera and **the reference gets the
identity**, so zeros mean "this is the reference". The wide lens's own file
confirms it: exactly identity, every digit. A zero that means "you are asking
the origin about its own position" reads the same as a zero that means "no
data", and nothing in the report distinguished them.

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

That sensitivity is itself the result for that session, and it is not mysterious.
`5acd1b` has **638 of its 675 frames below the 0.01 conditioning threshold and not
one frame above 0.05** — it is almost entirely degenerate, start to finish. A
problem that the geometry never determined does not have a stable answer to
perturb, so a 0.03% change in the input picks a different one. A loop score that
responds to 0.03% like this is not a quantity to quote to three decimals, and the
table marks it rather than pretending otherwise.

This is worth separating from the sessions that merely score badly. `3c7c6b` is a
poor result; `5acd1b` is an underdetermined one. The two look similar in a loop
column and want different responses — the first is something to improve, the second
is something to detect and refuse.

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

### Anchoring the image term to the map, measured three ways

The section above calls the fix "not speculative". That reading does not survive
being measured, and what replaces it is narrower and better supported.

A reference frame can be wrong in two independent ways. It can be *stale* — far
enough from the current view that the optimiser cannot reach it — and it can be
*outside the map*, so the pose error it carries is not the error the depth block
is already carrying. Three configurations separate them. All take rotation from
the IMU and nothing else varies:

```
  arm   reference                                  distance to it    basin    beats icp
  B     previous image frame                       0.096 – 0.142 m   8/8 in     3/8
  C     latest map keyframe that carried an image  0.204 – 0.308 m   8/8 out    3/8
  D     C, with every image frame forced into      0.097 – 0.137 m   8/8 in     1/8
        the map, so the reference is always fresh
```

The basin column is against the 18 cm the synthetic truth test demonstrates the
five-level pyramid can recover. `C` is outside it in all eight sessions, and the
mechanism is in the counts rather than inferred: a frame becomes the photo
reference only when it is *both* a keyframe and carries an image, and at 5 Hz
images against 30 Hz depth that intersection is small — 35 of 203 keyframes in
1696fa, 29 of 215 in f0d073, 31 of 238 in 5acd1b. The reference ages between
updates and convergence failures rise with it, 4/126 in `B` against 17/119 in `C`.

```
  session   ARKit    icp      B prev    C stale   E dense   D dense + photo
  1696fa    0.126    0.123     0.120     0.126     0.130      0.124
  f0d073    0.286    0.694     0.545     0.481     0.613      0.476
  1868dd    0.378    1.381     2.152     0.889     1.749      1.680
  683ef1    0.490    1.137     1.383     3.025     1.244      2.743
  cb4586    0.146    0.196     3.386     3.669     0.242      2.221
  3c7c6b    0.127    1.942    56.026     3.507     2.406      3.311
  5bd1ed    0.269    0.322     9.411     5.574     0.260      1.502
  5acd1b    0.055    3.078     2.675     2.285     2.680      3.430
```

**All three reference states are a net loss**, and the one closest to what was
asked for is the worst of them: `D` — fresh *and* resident in the map — beats the
depth-only control on one session of eight, and loses to it by 11.3× on cb4586.
Choosing a better frame to anchor to does not carry the frame-to-frame result
across.

`E` is what makes `D` readable: `--keyframe-on-image` with no image term, so the
only change from the baseline is that image frames are forced into the map. It
improves three sessions of eight. **A denser keyframe set is not a lever on its
own**, so `D`'s numbers are about the image term rather than about the map having
moved under it.

One real effect survives, and it is not an improvement. `B` and `D` use nearly the
same reference image at nearly the same distance and differ in whether that frame
is in the map, and the catastrophes are capped: 3c7c6b 56.026 → 3.311, 5bd1ed
9.411 → 1.502, cb4586 3.386 → 2.221. Since `E` shows the denser map does nothing
by itself, this is an interaction — a stiffer depth block limits how far the image
term can drag the pose. That is damage control, not information, and the two are
worth not confusing.

What none of these arms can reach is the asymmetry underneath them. Under
`--frame-to-frame` both blocks reference *the same single frame*, and the term
improves seven of eight. Under frame-to-map the depth block references an
accumulation of dozens of keyframes, and no single reference image can be that.
Every arm above picks a different frame and they all meet the same wall. The
surviving form of the hypothesis is the one where a point's image reference is the
keyframe that put *that point's* geometry into the map, so both blocks carry the
same pose error at the same place and it cancels. That is the FAST-LIVO2
structure, and this pipeline's pixel-aligned depth makes it cheaper here than in
the papers, which need a plane warp to approximate what an exact reprojection
gives us for free.

### The arm that measured nothing, and the rule that caught it

`--keyframe-on-image` also switched the image term on without `--photometric`.
The block that loads the image is entered for either flag, and once inside it went
on to build the photometric pair without checking which flag brought it there. The
report was printed under `if args.photometric`, so a run that had silently
acquired an image term said nothing about it.

It was caught by the rule this page wrote down after 5acd1b: **a number identical
to its own control is a measurement that did not happen.** `D` and `E` agreed to
three decimals on all eight sessions, and their maps agreed to the exact voxel
count — 294710 in both for 1696fa, which poses differing by a micron would not
produce. Raising `--photometric-weight` to 50 moved that session from 0.124 m to
23.332 m, which established that the term was connected, and therefore that the
two arms were one computation rather than the term being negligible.

Two things changed. The pair is built only under `--photometric`, and the report
is printed whenever the term ran rather than whenever it was asked for, so an
unrequested run now says `! photometric term ran without --photometric`. The test
covers this path in both directions, because checking only that the term runs when
asked would not have caught it — the failure was the other direction.

### The conditioning gate on map insertion does not fire

Item 6 below proposes a registration-quality gate on map insertion, noting that
conditioning is the only gate today. Measured, that gate is already inert. Its
input is the translation-only ICP conditioning, against `min_conditioning = 1e-3`:

```
  session   median    p10       below threshold
  1696fa    0.27      0.109     0/759
  5bd1ed    0.227     0.0572    0/1104
  cb4586    0.132     0.0438    0/1006
  f0d073    0.144     0.0579    0/694
  1868dd    0.1       0.0107    12/1525
  683ef1    0.0855    0.0257    3/647
  5acd1b    0.102     0.0196    6/675
  3c7c6b    0.0932    0.0143    19/650
```

Forty frames of 7,060, and the worst session's p10 sits two orders of magnitude
above the threshold. **Map insertion is effectively ungated today**, for ordinary
keyframes as much as for image-forced ones. A conditioning gate on image keyframes
was designed and then dropped without being run: at this threshold it would move
about one keyframe per session, and a column that reproduces its neighbour is the
result shape this page has learned to distrust.

The inlier fraction is the candidate with something left in it — median 96–99%
across the eight but a minimum of 51–84%, so unlike conditioning it has a tail a
threshold could select. Note also that conditioning is a property of *what a frame
was registered against*, not of the frame: the same sessions measured
frame-to-frame against the previous image's points run 3–5× lower, so a threshold
tuned on one path is not a threshold on the other.

### The keyframes are the worst-registered frames, and that is not a defect

Asked without picking a threshold — do the frames folded into the map look like
frames in general? — the answer is no, and in the direction nobody expects:

```
  session   folded into the map   every scored frame   difference
  cb4586          99.0%                 99.4%            -0.4
  5bd1ed          98.9%                 99.4%            -0.5
  1696fa          98.9%                 99.4%            -0.5
  f0d073          98.4%                 99.0%            -0.6
  1868dd          97.7%                 98.7%            -1.0
  3c7c6b          97.5%                 98.5%            -1.0
  683ef1          96.3%                 97.6%            -1.3
  5acd1b          93.7%                 96.2%            -2.5
```

Eight sessions of eight. The first reading is that map insertion is *anti*-selecting
and wants a quality gate urgently. The second reading is that this is what
keyframing means. A keyframe is due once the camera has moved 5 cm or turned 5°
since the last one, so keyframes land at the *end* of each interval — the moment
of least overlap with the map, because the map was last updated 5 cm ago. Frames
with the best overlap are the ones just after a keyframe, and those are exactly
the ones the rule declines. The frames that are hardest to register are the frames
that carry ground the map does not have yet.

`--keyframe-min-inliers` was added to test the first reading, and it refutes it:

```
  session   ARKit    icp      gate 0.92   gate 0.95      keyframes, icp -> 0.92
  1696fa    0.126    0.123      1.714       3.237            203 -> 86
  f0d073    0.286    0.694      3.271       4.854            206 -> 85
  1868dd    0.378    1.381      9.302       6.961            562 -> 165
  683ef1    0.490    1.137      2.318       0.735            212 -> 14
  cb4586    0.146    0.196      2.838       4.951            323 -> 101
  3c7c6b    0.127    1.942      2.192       2.380            227 -> 91
  5bd1ed    0.269    0.322      5.422       4.838            321 -> 39
  5acd1b    0.055    3.078      2.467       2.337            240 -> 14
```

Better than the control on one session of eight, geometric mean 4.6× worse, worst
case 16.8×. The mechanism is in the gate's own input, measured under the gate:

```
  session   inlier median, icp -> gate 0.92     frames below 0.92, icp -> gate
  5bd1ed          99.4%  ->  51.9%                     87  ->   961
  cb4586          99.4%  ->  57.8%                     30  ->   703
  1696fa          99.4%  ->  73.1%                     21  ->   436
  1868dd          98.7%  ->  69.1%                    224  ->  1118
  683ef1          97.6%  ->  50.5%                    232  ->   606
  5acd1b          96.2%  ->  50.2%                    245  ->   647
```

**The gate destroys the quantity it gates on.** It measures a frame against the
map that it is itself deciding whether to update, so a refusal starves the map, a
starved map registers the next frame worse, and the worse fraction refuses more.
5acd1b keeps 14 keyframes of 676 frames. What looked like a threshold choice is a
closed loop with positive feedback, and no threshold escapes it — 0.95 is worse
than 0.92, not better.

So item 6 needs restating. A gate on map insertion cannot key on how well the
frame registered *against the map*, because that is endogenous. It has to key on
something the map cannot influence: the frame-only conditioning this tool already
computes from a frame's own points and normals, the IMU's own consistency, or the
image. The flag stays, defaulted off, because the runaway is the result.

This also puts a precondition on the per-point reference design. If each point's
image reference is the keyframe that contributed its geometry, then every
reference sits on a frame drawn from the worse-registered tail — harmless if that
is a consequence of keyframing, fatal to the cancellation argument if it is a
bias. The two readings produce the same inlier fraction, so they have to be
separated on something else, and pose error separates them directly:

```
  session   keyframe inliers   keyframe error   other frames   verdict
  1696fa         98.9%             4.3 cm          4.3 cm       level
  683ef1         96.3%            34.1            34.2          level
  1868dd         97.7%            35.6            38.7          keyframes better
  cb4586         99.0%             7.1             7.6          keyframes better
  3c7c6b         97.5%            50.5            55.0          keyframes better
  5bd1ed         98.9%            10.2            11.4          keyframes better
  f0d073         98.4%            24.4            23.0          keyframes worse, 6%
  5acd1b         93.7%            94.9            54.1          keyframes worse, 75%
```

**The keyframes register less well and are not positioned less well.** In six of
eight they are level with or better than the frames around them, which is the
benign reading: a keyframe overlaps the map less because it was taken after the
camera moved, not because its pose is wrong. The one session where it is a real
bias is 5acd1b — the session with 638 of 675 frames below the conditioning
threshold, which this page already separates as underdetermined rather than
merely poor.

The tempting generalisation is that conditioning decides which reading applies:
where the geometry pins the pose down, keyframing at the point of least overlap
costs nothing, and where it does not, the frame folded in is a guess and the
deficit is real. It was written here first as the best available explanation
resting on n = 1, and six degenerate sessions captured the next morning take it to
n = 14. **It does not hold up.**

```
  the four most degenerate sessions, by frame-only conditioning
    dd2a13   0.00172   keyframe deficit 0.98×
    683ef1   0.00395   keyframe deficit 1.00×
    5acd1b   0.00395   keyframe deficit 1.75×
    6b92f3   0.00404   keyframe deficit 1.24×
```

The two *most* degenerate sessions of all show no deficit, and across fourteen the
correlation between conditioning and the deficit is r = −0.313. The deficit itself
is real but rare — worse by more than 5% in four of fourteen, median 0.990 — so
the benign reading stands and the account of the exceptions does not. 5acd1b
remains the outlier it looked like before it was promoted to a mechanism, and
promoting it was the error: a session flagged as degenerate on independent grounds
is exactly what makes an outlier read as a mechanism.

That makes conditioning the fifth session-level statistic tried here and the fifth
to fail, after per-pair error, ICP conditioning against damage, image sharpness,
and reference distance.

`--match-radius-probe` asks the remaining mechanistic question directly, against
the same map at the same predicted pose so that nothing feeds back: a keyframe
that matches poorly because the map has no points there keeps whatever it matched
as the radius tightens, while one that matches poorly because its predicted pose
is wrong is matching at a distance and falls away faster. As a ratio of keyframe
to non-keyframe match fraction, from 0.15 m down to 0.02 m:

```
  1696fa   0.994 -> 0.985      cb4586   0.995 -> 0.979
  6b92f3   0.952 -> 0.901      dd2a13   0.963 -> 0.894
  5acd1b   0.968 -> 0.975
```

The wall pair separates cleanly: their keyframes are matching at a distance, which
is the wrong-pose signature, and they are the two sessions where the geometry
cannot supply the answer. The benign pair shows the same sign at a tenth the size.
5acd1b is the one that does not — flat, the absent-points signature — despite
being the session whose keyframes carry the largest *absolute* pose deficit. The
two measurements disagree there and neither is obviously wrong, so it stays an
open thread rather than a conclusion.

The first probe written for this used *widening* radii and returned identical
numbers at 0.15, 0.30 and 0.60 m. That is the tell described elsewhere on this
page — a perturbation that changes nothing tests nothing — and the cause is worth
recording, because it is also why `max_dist` measured as inert. `_voxel_matches`
searches a point's own voxel and the 26 adjacent ones, so at a 0.03 m voxel no
candidate it can return is more than about 0.10 m away and a 0.15 m rejection
threshold never fires. The stencil is the real radius; `max_dist` is a ceiling
above it.

The comparison is also a within-trajectory contrast — a keyframe and its
neighbours share whatever drift has accumulated by then — so it is evidence about
keyframes against their surroundings rather than against an independent baseline.

So the precondition holds on the sessions worth building for, and fails where this
page already says to detect and refuse rather than improve.

### Six degenerate sessions, and what they are for

The eight sessions above flag 0–3% of frames as degenerate, which is why the half
of the IMU that matters — a prior with covariance in the directions the geometry
cannot see — has never been testable here. Six sessions captured on 2026-08-12
fix that. Two configurations, both loop-closed:

```
  session   capture                     conf   frame cond   degenerate   ARKit    icp
  2735cf    level phone, floor out      53%     0.00451       9%        0.146   1.551
  ce02ac    level phone, floor out      56%     0.00521       2%        0.190   0.254
  2be6a9    level phone, floor out      54%     0.0109        2%        0.258   0.859
  02a524    level phone, floor out      65%     0.00791       4%        0.535   0.634
  6b92f3    facing a wall at 0.5 m      80%     0.00404      37%        0.311   1.588
  dd2a13    facing a wall at 0.5 m      80%     0.00172      44%        0.125   0.771
```

Holding the phone level so the floor leaves the frame works — it removes the plane
that was pinning two axes — but it also points the sensor at distant surfaces, and
ARKit's depth confidence falls with range. Those four sessions are degenerate *and*
low-confidence, which confounds the two. The wall pair fixes that: sidestepping
along a wall at 0.5 m keeps confidence at 80%, equal to the best sessions here,
while making the geometry far worse than anything previously recorded — 37% and
44% of frames past the strict threshold against a previous maximum of 3%.
`02a524` is the one to leave out: ARKit's own loop is 0.535 m, so there is no
reference to score against.

**These are a reference and a stress set, not a target.** The wall pair in
particular is close to the worst case a depth-first system can be given — a single
plane fills the view, and sidestepping moves along exactly the direction that
plane cannot constrain. They are here to compare against ARKit and to give the
degenerate regime a population larger than one. A method that also succeeds on
them would be welcome; making them succeed is explicitly not a goal, and work
should not be scoped around them.

They earn their place immediately by correcting two things on this page.

**The conditioning gate is not inert, the old dataset was.** Map insertion was
described above as effectively ungated on the evidence that `min_conditioning`
fires on 40 frames of 7,060. On the wall pair it fires on 71 of 456 and 72 of 507
— 14% and 16%, with the p10 of the translation-only conditioning at 0.000273 and
0.000645, below the threshold rather than two orders above it. The gate was never
dead; it had never been shown data it was written for.

**And the image term does its worst damage exactly where it was predicted to
help.** [VLIO.md](VLIO.md) argues the two terms are geometrically orthogonal and
that the gain should concentrate where point-to-plane conditioning is worst.
Sidestepping along a textured wall is the sharpest available case: depth cannot
see motion along the plane and an image gradient sees nothing else.

```
  session   icp      + photometric
  6b92f3    1.588 m     5.750 m      3.6× worse
  dd2a13    0.771       3.715        4.8×
```

This is the previous-image anchor, already known to damage, so it is not a verdict
on the term in principle. It is a verdict on the prediction: the axis the image
sees best is also the axis a wrong anchor leaks along, and when that axis is the
*only* one carrying information, the leak is the whole signal. The complementarity
argument and the anchor problem are not independent — the second is strongest
precisely where the first is most attractive.

### No weight makes the single-reference image term pay

Reweighting is argued against above rather than measured, and the argument is
about σ ratios and intermittency. A plain weight sweep is a different question and
was still open, so it was closed. The decision rule was written down before the
runs: for the term to count as carrying information, some weight has to beat the
depth-only control on a majority of the eight *and* the benefit must not be
monotone in the direction of switching the term off — because the limit of that
direction is the control itself. What that predicts is an interior optimum.

```
  session   icp      w = 0.1   w = 0.3   w = 1.0     interior optimum
  1696fa    0.123     0.191     0.205     0.120           no
  f0d073    0.694     0.625     0.647     0.545           no
  1868dd    1.381     1.122     3.239     2.152           no
  683ef1    1.137     1.313     0.982     1.383          yes
  cb4586    0.196     0.176     0.456     3.386           no
  3c7c6b    1.942     1.955     2.014    56.026           no
  5bd1ed    0.322     0.258     0.649     9.411           no
  5acd1b    3.078     2.070     1.955     2.675          yes

  beats icp             5/8       3/8       3/8
  geometric mean       0.946     1.329     3.412
  worst session         1.55×     2.35×    29.23×
```

The first clause passes and the second fails: two sessions of eight show an
interior optimum, and both the win count and the geometric mean improve
monotonically as the weight falls, toward a limit of 1.000 that is the control.
`w = 0.1` is not a tuning that was found, it is the term nearly switched off, and
0.946 is not distinguishable from 1.000 with per-session ratios spread 0.67–1.55.

That makes three independent dials turned in the same direction. `--photometric-stride 2`
halves how often the term applies and improves six of eight. Arm `C` weakens the
constraint by leaving the reference stale and removes the catastrophes. Lowering
the weight does it directly. **Frequency, strength, and weight are unrelated
knobs, and using less of the term is better on all three** — which is what a noise
source looks like, not an intermittent information source.

Had the rule been written after the table, `5/8` and `0.946` would have read as a
tuning worth keeping.

### What can be shown while recording, and what must not be

Five of eighteen captures here were unusable, and none of them said so at the
time: too dark, a loop that never closed, geometry that turned out flat. Each was
found at a desk hours later. The recorder already warns about depth confidence
for exactly this reason, and the same argument extends to the geometry.

What may be shown is constrained by what has held up. **Frame-only conditioning
is the one signal that tracks the outcome** — its median against the depth
estimator's loop error runs r = −0.68 over thirteen sessions, where per-pair
photometric error (+0.19), ICP conditioning against damage (−0.15), image
sharpness (−0.34) and reference distance (−0.00 within an arm) each failed. It is
a property of a frame's own points and normals, so nothing the estimator does can
flatter it.

The estimator's own opinion of itself must not be shown. The point-to-plane
residual correlates with actual pose error at r = −0.057, the inlier fraction
saturates at 96–99% and stays there through the frames that go wrong, and a gate
keyed on it destroyed the quantity it gated on. A live indicator built on either
would be confidently wrong at exactly the moments it mattered, which is worse
than no indicator: it would license the bad capture rather than merely fail to
catch it.

The app has computed `cond` and `weakAxis` on every depth frame since 0.1.64, in
`ARRecorder.frameConditioning`, and records them in `depth.jsonl`. Against the
Python implementation over 2,499 frames of five sessions it agrees to 0.1–0.2%
with r = 1.000, so the number on the phone and the number in this document are
the same number. `weakAxis` names the direction the geometry leaves free, which
turns a warning into an instruction — a flat wall filling the view constrains its
normal and nothing else, and the useful thing to say is which way to turn.

The threshold is 0.007 and it is soft. Around it the tiers overlap, so it ranks a
capture rather than sorting it. That is enough for the job, which is to say "this
one is going badly" while the phone is still up and the walk can be repeated.

### A pose source that never sees ARKit, at ARKit's accuracy

The reason this page exists is that the ultra-wide route costs ARKit's pose, so
a replacement had to come first. One now scores level with it. Feed-forward
multi-view reconstruction (Pi3X) supplies the trajectory, the session's own LiDAR
supplies the metre, and CoreMotion supplies nothing at all — the scale is a
closed-form least-squares fit of the predicted point map to the measured depth,
over thousands of pixels for one unknown, and it never looks at the reference.

```
  session   ARKit    Pi3X     icp      ATE Pi3X / icp
  cb4586     1.0%    0.4%     1.1%      7.8 /   8.1 cm
  1696fa     1.2     1.3      1.2       3.9 /   5.2
  1868dd     1.2     1.3      4.4      22.5 /  39.7
  5bd1ed     1.4     0.9      1.7       5.9 /  10.6
  3c7c6b     1.0     1.4     16.5       5.8 /  57.7
  2735cf     1.4     1.1     12.6       5.4 /  50.9
  ce02ac     1.6     1.1      2.1       9.8 /  16.3
  dd2a13     1.5     3.9      8.9      33.6 /  82.0
  f0d073     2.5     2.5      5.9       6.9 /  21.0
  2be6a9     2.7     2.8      8.9       8.2 /  30.0
  6b92f3     3.1     3.6     16.2      81.6 /  70.9
  5acd1b     0.6     0.8     32.5       3.8 / 100.8
  683ef1     5.6     5.1     12.2       2.6 /  45.0
  ---------------------------------------------------
  median     1.4     1.3      8.9
```

Loop error as a percentage of path length, and absolute error after rigid
alignment. **Median 1.3% against ARKit's 1.4%**, ahead on five sessions of
thirteen, and against the depth-only estimator it is 3.4× better by ATE.

Two things this is not. It is **not causal** — the model sees a whole window at
once, which suits an offline export and disqualifies it from the live path, so
"better than icp" here means "batch beats incremental" and not "replace the
odometry". And **metric depth must not be given to the model as a conditioning
input**, which is the opposite of the obvious move. Doing so pins the predicted
depth near metric while leaving the translations where they were, breaking the
single-scale property the whole design rests on; the fitted scale then agrees
with the truth in the easy sessions and not in the hard ones, and the geometric
mean against icp goes from 0.41 to 0.66. Condition on intrinsics, fit the scale
afterwards.

Every joining problem this ran into came from a seam, and the seams came from a
16 GB card. At 96 frames a window peaks at 33.8 GiB and twelve of thirteen
sessions fit in a single pass with no seams at all. Long sequences also need the
decoder's convolutions run in batch chunks — Pi3X folds frames into the batch
and its heads use `padding_mode='replicate'`, which routes through an `F.pad`
kernel that indexes with int32 and refuses past 2^31 elements. Chunking is exact
in exact arithmetic, and on one card exactly zero in bfloat16 too; on another it
differs by 3.5e-03, because the batch size decides which kernel the runtime picks
and the kernels round differently. That reading is measured rather than assumed —
the same comparison in fp32 gives 5.7e-05, so it tracks precision, which a
batch-splitting mistake would not.

With the patch the single pass holds to **220 frames** on a 96 GB card, which at
5 Hz is about 44 seconds of walking:

```
  frames    131     160     190     220     250
  peak     38.8    52.9    70.2    90.2    OOM   GiB
```

Twelve of the thirteen sessions are inside that, and 1868dd at 258 is the one
that is not. So the long-horizon reconstruction models built for thousands of
frames address a problem this capture does not have; what would bring one into
range is a session longer than about forty seconds, and even then the first move
is a global scale across chunks rather than a different model.

### Narrowing the field of view makes it worse, in every session

The pose source was validated on the wide camera at 71.3°, and the ultra-wide
route feeds a 96.3° rectified pinhole. Widening cannot be synthesised from a
narrower capture, so the gradient was measured on the side that can be reached:
a narrower pinhole from a pinhole is exactly a centre crop, holding `fx` and
shifting `cx`, with no resampling anywhere.

```
  ATE in cm            71.8° (native)    62°     52°     42°    42°/native
  f0d073                     6.9         5.9     5.0     8.1      1.18×
  1696fa                     3.9         4.5     4.3     4.9      1.27×
  ce02ac                     9.8        11.1    12.4    16.9      1.72×
  6b92f3                    81.6        95.7   117.2   131.4      1.61×
  5acd1b                     3.8         4.5     6.3     7.3      1.94×
  2be6a9                     8.2        17.4     8.4    15.9      1.94×
  2735cf                     5.4         6.4     7.1    11.0      2.05×
  5bd1ed                     5.9         7.2     7.6    12.4      2.10×
  683ef1                     2.6         3.3     5.0     5.5      2.11×
  cb4586                     7.8         9.6    12.8    18.7      2.40×
  dd2a13                    33.6        64.4   107.2   124.4      3.70×
  3c7c6b                     5.8         8.7    17.5    23.7      4.10×
  1868dd                    14.5        24.3    70.8    96.4      6.65×
```

**Thirteen of thirteen worse, median 2.05×**, and monotone in ten. A control is
built into the manipulation: every arm crops from the full frame and is then
downsampled to the same pixel budget, so the 42° arm carries 1.7× more pixels per
degree than native and is still worse. Coverage is what matters, not sampling
density.

Extrapolating the sign, **96.3° should be neutral to favourable**, and the
ultra-wide lens is therefore not a compromise the pose source has to absorb. The
worst case is 1868dd at 6.65×, and it is the one session too long for a single
pass — a narrow field damages the joins as well, so with seams present the loss
bites twice.

### Correcting the lens makes it worse, and the ultra-wide is flatter anyway

Two measurements that change how the ultra-wide route should be built.

**The wide camera is not the pinhole this page assumed, and the ultra-wide is
closer to one.** Apple's factory distortion tables, read off the device and
stored in `calib/`, as radial magnification against the corner radius:

```
             fov      at 25%   at 50%   at 75%   at 100%   max correction
  wide      72.6°     +1.11%   +2.56%   +2.97%   +3.90%     98.9 px
  ultrawide 102.5°    -0.59%   -0.63%   -0.38%   -5.16%    130.5 px
```

Out to three quarters of the corner radius the ultra-wide is **three to five
times more rectilinear** than the wide lens. Its distortion is concentrated in
the extreme corner, which `tools/rectify_ultrawide.py` discards anyway when it
crops 102.5° down to 96.3°. The rectifier's own docstring says a 106° lens is
not a pinhole; measured, the lens that is not a pinhole is the wide one, and
every pose number on this page was produced from its uncorrected frames.

**Correcting them makes Pi3X worse, in all four sessions tried.**

```
  session   Pi3X ATE native   rectified   change     ICP (control)
  2735cf         18.26          24.21      +5.95      50.9 / 50.9
  5bd1ed         10.06          11.76      +1.70      10.6 / 10.6
  683ef1          4.46           7.93      +3.48      45.0 / 45.0
  cb4586          8.78          11.53      +2.75       8.1 /  8.1
```

The control is built in: depth ICP never looks at the image, and its number is
identical to the digit in every arm. Nothing in the pipeline moved except the
pixels.

Two explanations fit and they have different consequences. Rectification
resamples every pixel with Lanczos, and Pi3X matches on high frequencies that
a resample softens — in which case the cost is the resampling and not the
correction. Or the model was trained on phone imagery with exactly this
distortion in it, and removing it moves the frame out of the training
distribution.

A third arm settles it: remap through the identity, which resamples exactly the
same way and corrects nothing.

```
  session   native   resample-only   rectified     resample-native   rectify-resample
  2735cf     18.26       16.76         24.21           -1.50              +7.45
  5bd1ed     10.06        9.66         11.76           -0.40              +2.10
  683ef1      4.46        4.52          7.93           +0.07              +3.41
  cb4586      8.78        8.56         11.53           -0.22              +2.96
```

**Resampling is free** — median -0.31 cm, better in three sessions of four. The
entire cost is the correction: +3.19 cm median, worse in four of four. So the
resampling explanation is dead and the remaining one stands: **Pi3X has already
learned this distortion.** Phone-wide-angle distortion is inside its training
distribution, and taking it out moves the frame outside.

The consequence for the ultra-wide route is **feed it uncorrected**, and three
measurements now agree on it. The lens is already three to five times flatter
than what the model reads successfully today. The cost of correcting is the
correction itself, not the resample it rides on, so the same logic transfers.
And the only part of the ultra-wide that is badly distorted is the extreme
corner, which a crop removes without touching a pixel of the rest.

That takes `tools/rectify_ultrawide.py` off the pose path entirely — no
resampling, no lost field to the rectifier's crop, no third tool between the
camera and the model.

### The first real ultra-wide walk, and it beats the wide camera on rotation

`MultiCamRecorder` produced its first session: 133 ultra-wide frames at 106.2°,
634 depth frames at 320x240 — 1.56x ARKit's 256x192 — and no poses, because
reaching the lens means leaving the tracker. An ordinary ARKit walk of the same
room was recorded straight afterwards as the thing to compare against.

**The depth had to be put where the image could use it.** The first pose run
came out with join residuals of 2 to 10 cm against the wide session's 0.2 to
0.7, and per-window depth scales piled up between 1.07 and 1.18 instead of
scattering about 1 — the signature of a systematic error. The two do not see the
same cone: depth is a 4:3 map of roughly **72° x 58°** from the LiDAR device,
which is the wide camera plus a scanner, while the ultra-wide frame is 16:9 and
**106.2° x 73.7°**. Resizing one onto the other covers the whole image with
measurements that exist for 41 % of it.

The depth figure was the factory calibration scaled down, because that was the
only number available at the time. The recorder now logs the depth camera's own
intrinsics and the active format measures **70.38°**, not 72.6 — see *One walk,
both lenses* below. It does not change the finding here, which is about a 34°
mismatch, but any tool that needs the depth cone should read
`manifest.depth_calibration` rather than this paragraph.

Unprojecting each depth pixel, moving it by the extrinsic in `calib/` and
projecting into the ultra-wide fixes it:

```
                     join residuals cm                    join scales
  naive resize   2.1 3.9 4.2 1.7 1.6 3.1 9.8 2.7 3.5     1.08 - 1.57
  reprojected    0.2 0.5 0.5 0.1 0.3 0.3 0.8 0.3 0.4     0.99 - 1.09
  wide session   0.2 0.3 0.2 0.2 0.6 0.6 0.3 0.2 0.3     0.85 - 0.98
```

#### Scored against gravity, which never saw the estimator

There is no reference trajectory here and there cannot be, so the score has to
be one that needs none. CoreMotion reports gravity in the device frame, so a
correct set of rotations carries every frame's gravity onto the same world
direction. This is the check this page warned was an identity in an ARKit
session — there the `gravity` field is derived from ARKit's own poses — and here
it is not: `motion.jsonl` comes from an instrument that never saw the estimator.

The device-to-camera rotation is **fitted, not assumed**. A first attempt applied
the pose straight to device-frame gravity and made every arm worse including the
control, because `R C g` is `R C R^T` applied to world gravity and depends on R;
a fixed convention does not cancel. Alternating a Wahba solve for `C` with the
mean direction converges in a few steps, and what is left is the part no
convention can explain.

```
  arm                          poses    raw   after fitted C    p90
  wide, same room (control)     142    8.15°       3.19°       6.96°
  ultra-wide, naive depth       132    5.98°       6.22°       9.57°
  ultra-wide, reprojected       132    5.98°       2.78°       5.62°
```

**The control passes** — a trajectory independently measured at 6.6 cm ATE
against ARKit residualises at 3.19°, which is what makes the other rows
readable. The naive arm *increases* the scatter, 5.98° to 6.22°: rotating by
those poses scatters gravity rather than gathering it, which is what a wrong
rotation looks like.

**The reprojected ultra-wide residualises at 2.78°, below the wide camera's
3.19°.** Every field-of-view number on this page until now was an extrapolation
from narrowing the wide lens. This is the measurement, at the real 106.2°, scored
without ARKit anywhere in the loop.

The conclusion inverts without the depth reprojection: at 6.22° against 3.19°
the naive arm says the ultra-wide is twice as bad. The 19.272 mm extrinsic
recovered two days earlier is what decides which of those two answers this page
reports.

#### Three pairs, and the ultra-wide wins all three

```
  pair     lens        format      poses   walked     raw    residual
  room 0   ultra-wide  1920x1080    132     7.83 m   5.98°     2.78°
           wide        1920x1440    142     7.83 m   8.15°     3.19°
  room 1   ultra-wide  1920x1080    138     9.16 m   3.04°     2.26°
           wide        1920x1440    137     8.54 m   2.51°     2.43°
  room 2   ultra-wide   640x480     133    13.40 m   2.63°     2.76°
           wide        1920x1440    120    15.15 m   1.99°     3.85°
```

**Room 2 is the one that carries weight.** Its ultra-wide session came out at
640x480 — nine times fewer pixels than the wide camera's 1920x1440 — and it
still wins, 2.76° against 3.85°. Field of view covers a nine-fold loss of
resolution and has margin left. That format mismatch was a recorder defect,
since fixed: AVFoundation was left to pick the ultra-wide's format and chose
differently per run. It accidentally produced the strongest evidence here.

The `raw` column is the one to read before believing any of it. In rooms 1 and
2 the *ultra-wide* session's raw gravity scatter is the larger of the pair —
3.04° against 2.51°, 2.63° against 1.99° — so the phone was tilting about more
during those walks. The ultra-wide is not winning from an easier hand.

#### Position from revisits does not work, and the numbers below are withdrawn

`eval/revisit_drift.py` closes the gap rotation left. When a trajectory places
two frames from opposite ends of a walk in the same spot it makes a claim the
photographs can test, and the correction they demand is the position error
accumulated between them. No reference is consulted, which is what makes it
usable on a session that has none.

Drift grows with distance, so the raw centimetres are not comparable between
walks — normalising by the path actually walked between the two frames is what
makes them so.

```
  pair     ultra-wide                        wide                       revisits
  room 0   34.4 cm / 6.85 m  = 5.28 %        37.7 / 6.01  = 6.25 %       8 / 8
  room 1   41.6 cm / 4.57 m  = 9.14 %        33.3 / 4.01  = 9.36 %       8 / 8
  room 2   41.0 cm / 10.80 m = 3.06 %        18.1 / 14.70 = 1.23 %       5 / 2
```

**Rooms 0 and 1 are a tie.** In raw centimetres room 1 looks like a win for the
wide camera, 33.3 against 41.6; normalised it is 9.36 against 9.14, because the
wide session's revisits happened to span a shorter stretch of walk.

**Room 2 is a clear win for the wide camera** and it is the pair whose ultra-wide
session came out at 640x480. Rotation survived that nine-fold loss of resolution
and position did not, which is a coherent story rather than a contradiction:
angular spread is what rotation needs and feature precision is what position
needs. Its wide figure also rests on **two** revisits against the ultra-wide's
five.

**None of that table survives calibration.** Two checks, run after it was
written:

A frame against itself reads **0.03 cm**, so there is no constant floor. But
injecting a known offset into a real revisit pair and asking the descent to walk
it back:

```
  injected    recovered
      0 cm       8.2 cm
      5 cm      45.5 cm
     10 cm      40.6 cm
     20 cm      32.3 cm
```

The recovered value has no relation to the injection and *decreases* as the
injection grows. The score surface says why:

```
  offset cm        X        Y        Z
        -5    0.3523   0.3241   0.3264
         0    0.3174   0.3174   0.3174
        +5    0.3152   0.3514   0.3513
       +10    0.3163   0.3682   0.3791
       +30    0.3286   0.3546   0.3362
```

Along X the score at 0, +5 and +10 cm is 0.3174, 0.3152, 0.3163 — indistinguishable
— and +30 scores lower than +20. There is no bowl to descend, so the descent
follows noise and reports wherever it stopped.

The reason is in the absolute level. A genuine cross-session pairing scores
0.02-0.15 and 0.35 is the band a deliberately wrong pairing occupies; these
revisit pairs sit at **0.32**, inside it, despite being the same place in the
same session. Two visits forty frames apart differ in exposure, in what has
moved, and above all in viewpoint — the pairing rule admits anything within
30 cm and 20°, and at that separation the occlusions differ enough to flatten
the score.

So **position remains unmeasured**. The rotation result is untouched: gravity is
a different instrument entirely, and it passed its control.

One real bug was found on the way and is worth keeping. A sweep where every
offset failed to find enough pixels used to return a minimum at zero, turning
"could not be measured" into "measured perfectly" — room 0's wide arm scored
nothing at all until that was made to return `nan` instead.

#### Position, from how thick the reconstruction comes out

A trajectory with position error smears the surfaces it builds, and the sensor's
own floor is known: two clouds from alternate frames of one session, with zero
alignment error by construction, sit 0.77 cm apart. Anything beyond that is the
pose. Unlike the revisit measure this needs no long-baseline photometric match
to succeed, and it was calibrated by injection before it was believed:

```
  injected per-frame jitter    thickness      rise
        0.0 cm                   3.68 cm     +0.00
        0.5                      3.92        +0.24
        1.0                      4.40        +0.72
        2.0                      5.45        +1.77
        5.0                      6.35        +2.66
       10.0                      6.42        +2.73
```

**Monotone from 0.5 to 2 cm and saturating past 5**, which is the cell size
talking: measured inside a 10 cm voxel, error larger than the cell leaves it. So
this reads position error in the 1-3 cm range and cannot separate anything
coarser. Point count does not move it — 3.68, 3.71, 3.76 cm at 100 %, 75 % and
50 % of pixels — which matters because the ultra-wide clouds carry 34 % more
points than the wide ones.

One cell size cannot cover the range, so `eval/surface_thickness.py` reports
two. A voxel only sees error smaller than itself, and its floor rises with its
size because a large cell contains real curvature and counts that as thickness:

```
  voxel     reads up to    floor on a good trajectory
   10 cm        5 cm              3.7 cm
   20 cm       10 cm              5.8
   40 cm       20 cm              9.2
   80 cm       20 cm             15.9   (1-2 cm injections invisible)
```

All nine trajectories, both scales:

```
  arm            fine (10 cm)   coarse (40 cm)
  room 0 ARKit       4.28            9.76
  room 0 wide        4.79           12.52
  room 0 ultra       4.77           10.74
  room 1 ARKit       4.25           11.03
  room 1 wide        5.25           13.60
  room 1 ultra       5.45           12.96
  room 2 ARKit       4.41           10.10
  room 2 wide        4.63           12.67
  room 2 ultra       5.35           12.72
```

**ARKit is thinnest in all six cells**, and strikingly steady on the fine scale —
4.25 to 4.41 across three rooms — while both Pi3X arms run 4.6 to 5.5. Read
against the injection scale that is roughly **1 cm of position error beyond
ARKit's**.

The two lenses split by scale. On the fine cell the wide camera is level or
slightly ahead; on the coarse cell the ultra-wide leads in all three rooms,
though the third is a tie. That fits the rotation result — better rotations
suppress the large-scale bending that a coarse cell sees, and do nothing for the
fine texture a small cell measures. It is also within a centimetre or two of a
floor that is itself 9 to 11 cm, so it is a direction and not yet a
measurement.

##### Two explanations for the fine-scale gap, both refuted

The ultra-wide has two disadvantages that could account for losing the fine
cell, and neither does. Both were tested by imposing them on the wide camera,
where they can be taken away.

**Angular resolution.** The ultra-wide runs 0.0795° per pixel against the wide
camera's 0.0429, so it resolves 1.85x coarser. Shrinking the wide frames to
match, field of view untouched:

```
  session   full res   at ultra-wide resolution   change
  5bd1ed      10.06            9.66               -0.39
  683ef1       4.46            4.47               +0.02
  cb4586       8.78            8.07               -0.71
```

No cost — median **-0.39 cm**, worse in one of three. This is the third
independent sighting of the same thing: room 2's ultra-wide session won rotation
at 640x480 against 1920x1440, and the resample-only arm was free. Pi3X is
insensitive to resolution at this scale.

**Depth coverage.** A wide session carries depth over its whole frame; the
ultra-wide's LiDAR cone covers 41 % of its. Cutting the wide camera's depth
conditioning to the same fraction, image untouched:

```
  session   full depth   41 % only   change
  5bd1ed       10.06        9.84      -0.22
  683ef1        4.46        4.75      +0.30
  cb4586        8.78        8.67      -0.11
```

Also no cost — median **-0.11 cm**. The model handles two thirds of a frame
arriving unconditioned.

So neither asymmetry explains the fine-scale gap, and the likeliest remaining
explanation is that the gap is noise: three rooms spread over 4.63 to 5.45 cm on
that cell and the lens difference is 0.02 to 0.72, which three pairs cannot
separate. Two refutations point the same way.

So position is decided to the extent this instrument can decide it: **the
ultra-wide is not behind the wide camera, and neither matches ARKit.** Rotation
remains the one place the ultra-wide is ahead.

The unit label on the first calibration run said centimetres and printed metres,
which made a working instrument look like it was reading 0.04 and ignoring a
5 cm injection. It was reading 4 cm and responding correctly. Nearly discarded
for it.

#### Registering the ultra-wide walk to the ARKit one does not work yet

The other route to a position score would be to align the ultra-wide session to
the ARKit walk of the same room and borrow its trajectory as a reference. It
does not register: fitness at 5 cm is **0.03 and 0.04** on the two pairs, below
what a deliberately wrong pairing scores. The ultra-wide cloud is 3.5x larger
than the ARKit one built over the same room, which is what a trajectory bent by
tens of centimetres does to a cloud — no rigid transform can undo it.

The order is the reverse of what was hoped. Registration does not score a
trajectory; a trajectory has to be good enough before registration is possible.
The revisit measure above is the one that works on what exists today.

### What a wider view actually buys is scene, not geometry

Narrowing the wide camera makes every session worse, but a crop removes two
things at once: the angular spread of the bearings, and the peripheral content.
Masking separates them — same frame size, same `fx`, same principal point, black
outside the requested field. The geometry is untouched and only the scene goes.

```
  session   native    crop 52°   mask 52°   crop 42°   mask 42°
  5bd1ed     10.06      16.05      14.47      20.14      23.68
  683ef1      4.46       8.70       9.52      11.89      35.40
  cb4586      8.78      13.26      19.80      19.46      19.73

  crop to 52°: 1.60x native        mask to 52°: 2.14x native
  crop to 42°: 2.22x               mask to 42°: 2.35x
```

**Masking costs as much as cropping and often more.** Holding the geometry fixed
and deleting only the content reproduces the whole penalty, so the angular
spread was not what was doing the work. Pi3X solves pose from feature
correspondence, and a correspondence is decided by what is in a pixel, not by
the bearing that pixel sits at.

That masking is *worse* fits too: a black border is a hard edge the scene never
contained, so the arm loses the content and gains false features on top.
683ef1's 35.40 against a crop's 11.89 is that at its worst.

Two consequences for the ultra-wide. Its gain should scale with **how much
scene it takes in** rather than with the angle — 106.2° against 71.3° is about
2.2x the solid angle, which is the size of effect to expect and is the order of
what the rotation measurements showed. And **its frames should not be cropped**:
the rectifier's trim from 102.5° to 96.3° costs whatever that ring of scene was
worth. The earlier conclusion to feed the lens uncorrected already followed from
distortion; this supports it for an independent reason.

### Loop closure is not a proxy for trajectory quality

Loop closure has been the score on this page throughout, for a good reason: it
needs no reference, which is exactly the property the ultra-wide route will
require. It is also, measured, capable of moving the wrong way.

```
  as the field of view narrows and the trajectory gets worse
    loop follows ATE                        4 sessions
    flat                                    2
    loop improves while ATE degrades        7
```

**Eight of thirteen mislead.** The sharpest is 6b92f3: ATE goes 81.6 → 131.4 cm
while the loop score improves monotonically, 3.63 → 3.46 → 2.89 → 2.79%. Three
independent demonstrations point at one cause:

- 6b92f3 at native resolution already beats the depth estimator on loop and
  loses to it on ATE.
- Narrowing the field degrades the trajectory and improves the loop in eight
  sessions.
- Dropping 20 frames of 260 moved the loop from 1.78% to 2.83% and barely moved
  the ATE.

**The loop rests entirely on one frame and the ATE is an average over all of
them.** So a change that bends the trajectory can leave the endpoints where they
were, or move them closer. The risk is not that the loop carries no information;
it is that a change making the trajectory worse can read as an improvement, and
tuning against it walks in the wrong direction.

While ARKit is available the acceptance criterion is ATE. What replaces it after
that is the next section.

### What can be checked without a reference

Three candidates were measured. One is worth having, one is weak, and one does
not exist.

```
  quantity            signal                        sensitivity   independent?
  rotation            CoreMotion attitude            0.4–3.4°      yes
  translation scale   accelerometer                  ±5%           yes
  translation scale   two metric anchors disagree    unmeasured    no
  position            GPS                            none          —
```

**The accelerometer is the one that fills the gap.** Scaling a trajectory by s
scales its acceleration by s, and CoreMotion reports acceleration directly,
having seen neither camera nor depth. Band-limit the trajectory, difference it
twice, rotate into the device frame, and least-squares fit it against
`userAcceleration`. Nothing is integrated — integrating is what fails at these
speeds. Across ARKit's own trajectories the fit sits between 0.897 and 1.006,
which is a noise floor rather than a demonstration of independence, since ARKit
fuses the IMU. Against the depth-only estimate, which never sees the
accelerometer, it spans −0.031 to 1.234 and tracks the outcome: loop error
against the fit at r = +0.75, and against the per-axis correlation at r = −0.82.

On the learned trajectory, which is what the ultra-wide route would actually
carry, the same check runs at the image rate rather than the depth rate, and
that changes which half of it survives. Sampling ARKit on the same 5 Hz frames
puts the floor at 0.416–0.660 rather than 0.897–1.006, so **the fitted scale
stops being usable as an absolute number** — differencing twice at 0.2 s against
a 2 Hz cutoff is close enough to Nyquist to lose half the signal, and the
control is built in, because the reference sits on the same frames.

The correlation does survive, at r = −0.811 against ATE, and it separates
cleanly: eleven sessions between 0.44 and 0.78, dd2a13 at 0.026 and 6b92f3 at
0.063, which are the two the trajectory gets wrong. So the usable form of this
check after ARKit is gone is **a floor on the correlation, not a band around a
scale of one** — an absolute threshold near 0.4, which needs no reference at all.
What it has not yet shown is that it catches a failure nobody already knew
about: those two sessions were the wall pair, flagged long before this.

The correlation is the better detector, and the reason is worth keeping. A scale
fit answers "how much too big", and this estimator's dominant failure is drift
rather than a uniform scale error — 3c7c6b and 6b92f3 hold a fit near 1 while
their correlation falls to 0.26, a trajectory of the right size and the wrong
shape. 5acd1b, degenerate in 638 of its 675 frames, comes out at −0.031: its
motion bears no relation to what the phone physically felt.

**GPS has nothing to offer and this closes it.** Across twenty sessions the
horizontal accuracy is 18.8 m against a median walk of 11.5 m — 1.6× the whole
trajectory — and in thirteen of them the reported position does not move at all
during the walk, an indoor fix repeated. Sixteen report a vertical accuracy of
exactly 30.0 m, which is a placeholder and not a measurement. The one session
with a large position span covers 50.5 m against 7.5 m walked, which is evidence
that the span is noise rather than motion.

Two anchors disagreeing is worth measuring but is not independent: both read the
same depth, so both can be wrong the same way and pass quietly.

The whole harness has been run on a second card, which is how the bfloat16 note
above was found. It reproduces: the ARKit and depth-only columns match exactly,
as they must, since those are arithmetic over a copied file rather than anything
the model does, and the learned column agrees to printed precision. Two rules came
out of running it twice. **One measurement per process** — the same forward twice
in one process moves the fitted scale by up to 1.4%, while a fresh process
reproduces exactly, so a sweep forks per arm. And **never compare arms across
hosts**, because the same configuration on two cards differs by about 2%, which is
the size of effects worth looking for.

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
4. **What to do about the image term.** ~~Anchor the photometric residual to the
   map; this is not speculative~~ — the *frame-level* form of that is refuted
   three ways. A reference fresh but outside the map, stale but inside it, and
   fresh and inside it are all a net loss, the last worst at one session of
   eight, and no weight rescues any of them. Four options remain, and they are
   listed as options because none is favoured by evidence yet:
   - **Per-point references.** A point's image reference is the keyframe that
     put *that point's* geometry into the map, so both blocks carry the same
     pose error in the same place and it cancels. This is the form the frame-level
     experiments could not reach, and FAST-LIVO2's structure. One precondition
     holds: the keyframes it would draw from register less well than average in
     every session but are *positioned* no worse in ten of fourteen. What is
     missing is the direction — the one arrangement known to work,
     `--frame-to-frame` at seven of eight, gets there by making the *depth*
     block worse, and the evidence that the good block can instead be raised to
     meet the image term is FAST-LIVO2 doing it, not anything measured here.
   - **Render a reference view from the map.** Serves the same purpose and costs
     a subsystem; pixel-aligned depth is why the patch version looked cheaper,
     but the render is the one that matches the depth block's reference exactly
     rather than approximately.
   - **Leave the image term out of the frame-to-map path.** Depth with IMU
     rotation is the current best configuration on this data, and every attempt
     to add the image term to it has cost accuracy. This is the option the
     measurements presently support, and it should be beaten rather than assumed
     away.
   - **Spend the image on something other than a residual** — this is the one
     that paid. Feed-forward reconstruction over a window of images, scaled by
     the LiDAR, reaches ARKit parity; see the section above. The image is worth
     more as geometry than as a residual term.
5. **DRPM-style probabilistic degeneracy** in place of the `rcond` cutoff — but
   note the sweep found no threshold that serves all three sessions, and that
   conditioning was ruled out as the cause of the dominant error. This is
   further down than it looked.
6. **A registration-quality gate on map insertion**, keyed on something the map
   cannot influence. The obvious version is measured and it runs away: gating on
   the inlier fraction against the map starves the map, which lowers the next
   frame's fraction, which refuses more, ending at 14 keyframes of 676 frames
   and 4.6× the loop error. Conditioning as it stands is not the alternative
   either — it fires on 40 frames of 7,060. The candidates left are exogenous
   ones: frame-only conditioning from a frame's own points and normals, IMU
   consistency, or the image.
   ~~and an adaptive `max_dist`~~ — dropped, the parameter is inert.
7. **A rendered scene that resembles a room.** The current one is a small closed
   box in full view from frame one, and it has now been wrong twice about
   changes that were large improvements on real data. It is still valuable as a
   regression guard, where it catches real breakage; it is not usable for
   choosing between designs.

Current baselines, as medians over the thirteen scored sessions rather than the
two that happened to have loop closure when this line was first written:
**learned reconstruction with a LiDAR scale 1.3%**, depth-only frame-to-map
8.9%, ARKit 1.4%. The old figure here — frame-to-map 1.7% / 2.9% — was the two
best sessions, and reading it as the state of the estimator overstated it by
five times.

And the criterion itself has moved: loop closure is reported but is no longer
what a change is judged on, because eight sessions of thirteen improve it while
the trajectory gets worse. While ARKit is available, that judgement is ATE.

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

### One walk, both lenses — and the calibration was wrong for both

Every lens comparison on this page scored **three ultra-wide walks against three
different wide walks**. Path and lens moved together; room 2 was 13.40 m against
15.15 m. `MultiCamRecorder` now records the wide lens off the LiDAR device
alongside the ultra-wide, so one walk goes through both and that confound is
gone by construction. `eval/pi3_poseless.py --lens wide` poses the second arm.

Four of the five dual-lens sessions have been scored, and **the position result
reverses**. Both arms of each, same walk, same 24-frame window:

```
                     gravity residual        surface thickness cm
                     fitted mean deg          10 cm      40 cm
  d0f44f  wide            1.42                 4.75       9.20
  d0f44f  ultra-wide      1.40                 5.09      10.98
  d67f0a  wide            0.71 *               4.55      10.38
  d67f0a  ultra-wide      1.20 *               5.43      11.73
  15fbb3  wide            2.07                 4.07       9.70
  15fbb3  ultra-wide      1.74                 4.03      10.08
  57f29e  wide            0.96                 4.10       9.42
  57f29e  ultra-wide      1.76                 4.84      10.85

  ARKit control      2.45 / 3.85 / 3.92        —          —
  * both restricted to the first 10.1 s, where both arms are still flagged good
```

`57f29e` was the one dual-lens session never scored, with no reason recorded
anywhere, so the rule for reading it was written down before the run. It came
back the cleanest capture of the four: **neither arm flags a break**, the join
residuals are the smallest on the page at 0.30 cm on both arms, and the chain's
free scale stays nearer 1 than anywhere else — 1.083 and 0.962 against the 0.295
to 1.787 the others reach. It is also the session where the two arms agree most
on shape and separate most cleanly on position, which is the combination this
page had been reading as a contradiction.

Injecting known per-frame jitter into this data moves the coarse reading 0.7 cm
for 1 cm injected and 1.9 cm for 2 cm, so **0.7 cm is the smallest difference
worth reading**. The wide arm leads by 1.78, 1.35 and 1.43 cm in three sessions
and by 0.38 in the fourth, which is a tie.

**`d67f0a`'s thickness cells span past both arms' break flags.** The asterisk in
that table sits on the gravity column; the thickness numbers are the whole 31 s,
and the chain flags itself untrustworthy at t+10.1 s on the wide arm and t+19.2 s
on the ultra-wide. Cut both arms to the span where neither is flagged and the
lead survives, smaller:

```
  d67f0a, first 10.1 s      10 cm    40 cm
  wide                       3.72     7.32
  ultra-wide                 4.03     8.13
```

0.81 cm rather than 1.35, still above what the cell reads. The other two sessions
carry no break to cut at.

`15fbb3` being the tie is not obviously luck. It is the session whose first
seven seconds are a desk close-up at 0.57 m, and its two arms also agree on path
length to 0.5 % where the other two differ by 9 %. A wider lens with nothing
further away to put in it is the condition this page already predicts should
buy nothing — the gain comes from scene taken in.

Scored over the whole walk the ultra-wide looks twice as good on rotation — 4.03°
against 7.79° — but that is the two arms flagging themselves broken at different
times, not a rotation difference. Cut to the span where both are still good, the
wide arm leads. The break times themselves turned out not to mean what they
looked like; see below.

The control is what makes any of it readable: ARKit, whose accuracy is
established elsewhere, residualises at 2.45-3.92° from a raw 26-62°.

#### "The ultra-wide stays trustworthy twice as long" — withdrawn

This page reported, for about a day, that the ultra-wide arm survives to t+19.2 s
where the wide arm goes untrustworthy at t+10.1 s on the same walk. **It rests on
one join clearing its bar by one part in 440.**

```
  d67f0a wide       join 3   residual 5.284 cm   bar 5.272 cm   +0.012
  d67f0a ultra-wide join 7   residual 6.267 cm   bar 3.586 cm   +2.681
```

The bar is `6 x` the median of the preceding residuals, so it moves with the arm.
Had that wide join come in 0.3 % lower, nothing later would have fired either —
its largest remaining residual is 3.121 cm against a 5.689 cm bar — and the wide
arm would have finished the walk unflagged, reading as the *more* robust of the
two. A first-crossing flag against a self-referential threshold is a coin flip
near the bar. The tool now prints the margin so the next reader sees it.

The two breaks also have nothing in common. The wide arm's is an exposure step:
frame luminance goes 0.27 to 0.75 in four frames walking out of a dark hallway,
and depth validity dips from 0.98 to 0.79. The ultra-wide's is motion blur at the
session's second-fastest rotation, 76.6 deg/s mean and 108.5 peak, with sharpness
collapsing threefold while luminance stays flat.

And the cross-check refuses the tidy story: **at the wide arm's break the
ultra-wide sees a larger exposure swing — 7.0x against 3.7x — and does not
break.** Whatever separates the arms there, it is not the size of the lighting
change.

What survives is the continuous version of the same question. Median join
residual, every join of all four sessions, favours the ultra-wide in three and
ties the fourth, and pooled: **0.341 cm over 33 joins against 0.517 over 31**.
That is in the table above; the break times are not.

`57f29e` is the tie, and it ties hard — 0.299 cm against 0.302, which is a
thousandth of a centimetre apart on a statistic whose spread across sessions is
0.3 to 1.2. So the sweep this row used to report is gone, and what is left is
three wins and a session that cannot tell the lenses apart.

#### The reprojection was the obvious suspect, and it is not the cause

Only the ultra-wide arm moves its depth through the 19.272 mm extrinsic; the
wide arm's depth is already in its frame. So the position gap could have been
the transform rather than the lens, and `calib/README.md` records that the
*magnitude* of that baseline has never been recovered from a real capture.

The lever was checked before the sweep: scaling the translation changes the
reprojected depth map by a mean of 1.88 cm, which is the baseline itself, so it
is not below the noise floor of the input. The rule was written down before the
results were read — 40 cm cell, both sessions, same direction required, 0.7 cm
minimum. Only the translation is scaled: dropping the 0.462° rotation or the
cone mapping would remove what makes the ultra-wide arm work at all, which is
removing a precondition rather than isolating a term.

```
  ultra-wide, 40 cm cell     x0.0    x1.0    x2.0   spread    wide arm
  d0f44f                    10.69   10.98   10.88     0.29        9.20
  d67f0a                    11.68   11.73   11.72     0.05       10.38
```

**Nothing moves.** Deleting the baseline entirely still leaves the gap to the
wide arm at 1.49 and 1.30 cm. The position difference is the lens and the cone.

The gravity residual disagrees with itself across sessions — `d0f44f` puts x0.0
best at 1.28°, `d67f0a` puts x2.0 best at 3.81°, and its good prefix puts x1.0
best at 1.20° — which is what noise looks like, and is why the rule named one
cell in advance instead of reading whichever instrument answered.

This does not validate the 19.272 mm figure. A 2x error in it does not register
here, so the measurement is insensitive to the thing it would have to be
sensitive to in order to confirm it. Two routes have now failed to recover that
magnitude from a real capture; the direction remains verified from the probe.

The per-window depth scales still split by arm, 1.05-1.15 against 0.98-1.05.
That is not the baseline either. Two sections below take it apart, and about half
of it turns out not to be a property of the lens at all.

#### The join throws that measurement away, and the agreement row cannot see it

Each window measures its own metric scale from its own LiDAR depth, and then
`umeyama` re-fits a **free scale** at every join. That is right for `pi3_chain`,
where windows carry arbitrary scales and forcing them equal would bake the first
window's into everything after it. In the poseless mode the premise is gone —
every window is already anchored to metres — so the free scale discards every
depth measurement except the first window's.

```
  cumulative join scale       wide    ultra-wide   predicted by the windows
  d0f44f                     1.787         0.794      0.931 / 1.003
  d67f0a                     0.937         0.295      0.971 / 0.966
  15fbb3                     0.846         1.340      0.745 / 0.813
  57f29e                     1.083         0.962      0.991 / 1.030
```

The last column is `s_first / s_last`, what the factor would be if it were only
reconciling the windows' own disagreement. It is not that. `d0f44f`'s wide arm
has all eight joins above 1 and rising, 1.018 to 1.125: the chain inflates, the
next window is fitted against the inflated chain and inflates again. After the
first window there is nothing anchoring it. `eval/join_scale.py` reports this for
any poseless run.

**The join residual is measured after that scale is absorbed** —
`pi3_poseless.py:478` residualises against `js * (src @ R.T) + t`, with `js` the
scale just fitted. So the window-to-window agreement row in the table above is a
statement about *shape*, and nothing else. The ultra-wide leading it three of
three is compatible with one of its chains losing seventy per cent of its scale,
which is exactly what `d67f0a` does. Local agreement and cumulative position were
being scored on quantities the join treats differently, and that is the mechanism
behind the split this page could not explain.

**Anchoring it back is not better, which is why `pi3_poseless` is unchanged.**
`eval/join_scale.py --anchor` rebuilds the trajectory each window's own depth
would have made:

```
  40 cm cell            as chained    depth-anchored
  d0f44f  wide                9.20              9.99
  d0f44f  ultra-wide         10.98             10.60
  15fbb3  wide                9.70              8.58
  15fbb3  ultra-wide         10.08             10.72
```

Two directions, and three of the four moves are under the 0.7 cm this data
resolves. The wide arm still leads both sessions, so **the position result does
not depend on this choice.** Its margin does: `d0f44f`'s 1.78 cm lead falls to
0.61, below what the cell reads, and `15fbb3`'s 0.38 cm tie opens to 2.14 cm. A
degree of freedom nobody had registered moves the per-session margins by more
than the lens does.

`--anchor` is a rescale and not a re-chain — rotations and join fits stay as they
were — so it answers "is the depth-anchored scale better", not "what would the
anchored chain be". Nothing above justifies the GPU run that would answer the
second question.

#### Half the depth-scale split is how much depth the model was given

`s = <pz, meas> / <pz, pz>` splits by arm the same way in every session, and this
page called it the signature of a systematic depth error. Four causes were tested
on one window of `d0f44f`, the window the real run scored at 0.919 for the wide
arm and 1.064 for the ultra-wide.

**Not the depth calibration.** `2d9844` is the one session carrying the depth
camera's own logged intrinsics — 70.4 degrees, not the 72.6 the others scale out
of the factory file — and it splits by 0.104, wider than the three that fall
back.

**Not the gap fill**, which was the mechanism-shaped suspect. `depth_in_ultrawide`
scatters depth forward and closes the lattice with `cv2.dilate`, a maximum filter,
so a filled pixel takes the *farthest* surface within 7x7 — the opposite of what
the same function's scatter step argues for two lines above, and **97.7 % of the
ultra-wide arm's covered pixels are filled that way**. The bias it puts on the
mean is 1.0005. A 7x7 window on a 3840-wide frame is narrower than one 320-wide
depth pixel, so the maximum has almost nothing to choose between. The
inconsistency is real; the effect is not there.

**Not radial.** `eval/depth_scale_probe.py` fits `s` per bearing bin. Across 0-60
degrees the wide arm sits at 0.9180-0.9207 and the ultra-wide at 1.0631-1.0672 —
flat to 0.4 %. Whatever separates them is a gain, not a distortion the intrinsics
missed.

**It is, in half, the coverage.** The wide arm hands the model depth on 87 % of
its pixels. The ultra-wide can only reach 34 %, because the LiDAR cone is a third
of its frame. Mask the wide arm's depth to a centred ellipse of 40 % — same lens,
same frames, same window, nothing changed but the conditioning:

```
  d0f44f, one window       coverage        s
  wide                        0.870   0.9187
  wide, depth masked          0.397   0.9938
  ultra-wide                  0.340   1.0640
```

The fit's own support shrank with the mask and that is not the cause: the bin
table says `s` over those same bearings, unmasked, is 0.918. So the scale the
model returns depends on **how much depth it was conditioned on** — 0.075 of a
0.145 split — and the ultra-wide arm cannot be given more. Every ultra-wide `s`
fitted on this device carries that confound by construction. The remaining half
is not explained.

That also re-labels a closed item. Widening the wide arm's depth coverage was
dismissed because cutting it to 41 % moved surface thickness by -0.11 cm. Still
true, and a different question: coverage does not move the *trajectory*, and it
moves the *fitted scale* by 0.075. The measurement that closed the question was
not a function of the lever.

Setting that up turned up a calibration error that applies to everything here.

**The factory calibration describes a format nothing records at.** `calib/` is
measured at 4032x3024. Scaling it to the frame size is only valid if the active
format is a resample of that one, and on this device it is not — different
formats read out different parts of the sensor. The ultra-wide half of this was
already known and fixed. The wide half was not, because the recorder logged the
ultra-wide's active field of view and logged nothing for the wide lens, so there
was no logged value to prefer:

```
                factory, 4032x3024    active format      error
  Ultra-wide       102.5 deg           106.2007 deg      +3.7
  Wide              72.6                69.52744         -3.1
  Depth             72.6 (assumed)      70.37728         -2.2
```

**The error was never a property of the ultra-wide. It belonged to whichever
lens nobody was watching.**

It was caught before the device confirmed it, which is the part worth keeping.
The two lenses expose one scene at the same instant, so a similarity fit between
paired frames recovers `f_wide / f_ultrawide` with no external reference. Nine
pairs, three sessions, 1.0 ms mean capture separation: **s = 0.3176, IQR
0.0038**. The rule was written down before the fit ran. The factory assumption
predicts 0.3023 — 4.0x the IQR away, rejected. The device's logged 69.52744°
predicts 0.3198 — 0.6x away.

That measurement could not say *which* lens was wrong, only that the pair
disagreed with the factory numbers, and it declined to guess. It was right to:
the alternative on the table was that the logged 106.2007° was too low, and
picking it would have been wrong.

One trap inside it. Both lenses are raw, so the fit was repeated at shrinking
radii to extrapolate the distortion away, and the r→0 value of 0.3100 is
**further from the truth than the uncorrected 0.3176**. `videoFieldOfView` is a
whole-frame quantity that already contains the distortion; a paraxial focal
length is a different number. Correcting toward paraxial and then comparing
against a whole-frame figure moves away from the answer. The uncorrected fit was
the matching comparison.

**The depth is not the wide camera's cone either.** 70.38° against the wide
video's 69.53° — under a degree, off the same physical device, but a plain
resize of the depth map onto the wide image is wrong by that much.
`eval/pi3_poseless.py` resamples through the intrinsics on both arms.

None of this needs inferring again. `AVDepthData` carries the depth camera's
intrinsics for the running format and the recorder was discarding them; it now
writes `depth_calibration` into `manifest.json`, along with each lens's active
field of view and the build number that wrote the session. The extrinsic comes
back exactly identity, confirming from a real capture what `calib/README.md`
argued from a probe.

#### Confirmed again on a session that carries both answers

`0d6306` is the first capture where the probe's measurement and the device's own
logged value exist side by side. Ten pairs:

```
  probe, 10 pairs      s 0.3185, IQR 0.0009      69.75 deg
  device logged                                  69.52744
  factory scaled       s 0.3023                  72.58
```

The probe lands **0.32 % from what the device says and 5.0 % from the factory
assumption**. Two instruments sharing no input agree; the earlier nine-pair
result could only reject a hypothesis.

**It had to be rescued first, and the rescue is checked.** The session was shot
in a dark room — mean frame luminance 0.046 — and every pair failed on match
count, 14 to 20 against the pre-registered floor of 25. `--equalize` equalises
local contrast before matching and the same pairs come back with 61-89 inliers.
That is a change to the instrument, so the synthetic control goes through it:
known downscales of 0.3200, 0.3023 and 0.2500 return within 0.02 %. A rescue the
control cannot see would not be worth having.

**The depth camera's field of view is not one number.** This page and
`HANDOVER.md` carried three candidates and asked which to use. The two sessions
that log `depth_calibration` disagree with each other:

```
  2d9844   fx 453.82 at a 640x480 reference   70.38 deg
  0d6306   fx 467.67                          68.76 deg
```

Same device, same 320x240 depth map, both read off `AVDepthData` for the format
actually running. The active depth format changes between sessions, so "which of
the three numbers is right" was ill-posed — **none of them is a constant of the
device**. That is what `depth_calibration` is in the manifest for, and it is why
every session captured before it exists is using a cone that was never measured
on it.

What is still open is the *magnitude* of the 19.272 mm baseline. Recovering it
from the depth-dependent parallax gives per-pair slopes from 0.10x to 1.49x the
prediction, far wider than each fit's own error, and sweeping the assumed depth
field of view from 65 to 106° does not move it. The synthetic control recovers
an injected focal length correctly, so the algebra is right and something in the
real data is not modelled — but that control warps and regresses with the same
depth map, so it is structurally blind to a depth-to-image registration error,
which is the obvious suspect it cannot see. No focal length was reported from
that path.

### A note on the name

This page and the tool have called the map path *frame-to-model*, which is
KinectFusion's term for raycasting a TSDF. What is here is a voxel map of fused
points and normals, and the literature calls registering against one
*scan-to-map*. `LocalMap`, `self.map` and `render_map` already say map.

The prose and the test now both say map.
