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

## Where this stands (2026-08-08)

Measured, on an iPhone 16 Pro, against ARKit as reference:

- **5 Hz depth is unusable.** Frames land ~10 cm and several degrees apart, far
  outside ICP's basin. This was a self-inflicted limit — depth was pinned to the
  stills gate — and is fixed: depth now runs at 30 Hz as a superset of the image
  frames, and `tools/export_episodes.py --hz 5` decimates at export instead.
- **At 30 Hz it converges.** 97% inliers, ~1.1 cm per-frame error, ATE 7.3 cm
  rms against ARKit over a 6.6 s clip.
- **The per-frame error is a floor, not a convergence failure.** It barely moves
  between 30 Hz and a 5 Hz decimation of the same clip (1.1 vs 1.9 cm, ATE 7.3
  vs 7.2), so it is sensor noise or a systematic model error rather than ICP
  failing to track.
- **A real walk exists now, and one pass over it is not a verdict.** The earlier
  clips all covered under a metre, which is too close to stationary for the
  numbers to mean much; a 30 m, 32 s loop finally is not. Frame-to-frame scored
  7.2 cm/s of drift on it against ARKit's published ~2 cm/s — but at a rate the
  output did not record, and against a reference that included frames ARKit had
  not converged on. Both are fixed in the tool; see *Measured* below.
- **Indoors will hit the degenerate case constantly.** The blind configuration
  found below is not exotic — it is a phone held level in a room, and a corridor
  or a wall at arm's length is worse. Holding the phone tilted down, which the
  1.15 m viewpoint guidance already asks for, keeps a horizontal surface in
  frame and is what makes the vertical axis observable at all.

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

## Measured: depth-only ICP against ARKit

A 30 m, 32 s walk around a room and back, depth at 30 Hz, registered
frame-to-frame:

```
  drift 7.2 cm/s against ARKit, over 32.2 s and 30.44 m
```

ARKit's own published drift is ~2 cm/s, so naive frame-to-frame depth ICP is
roughly **3.5x worse than the thing it would replace**. Per-frame translation
error is 23 cm against 19 cm of actual motion, with 68% inlier median and some
frames at 0%.

That settles the original question in the direction the degeneracy argument
predicted: metric scale is genuinely free, and it is not enough. What it does
*not* settle is whether a proper local-map formulation closes the gap — frame to
frame is the weakest possible variant, and the one real systems do not use. The
map is the default path in `tools/depth_odometry.py` and `--frame-to-frame` is
the opt-out, so that comparison is one flag; it has never been run on this clip.

### The rate that run scored at is not known, so it has to be redone

Those two numbers cannot both describe the same pass. 30.44 m over 32.2 s is
0.95 m/s, which at 30 Hz puts consecutive frames 3.2 cm apart — but the run
reported the frames 19 cm apart, which is 0.95 m/s at **5 Hz**. Either the clip
was scored through a stride, or most of its depth frames never reached the
scorer. Five hertz is the spacing this page already calls unusable, and it would
account for the whole result on its own.

The tool could not have told anyone: it printed the rate the depth *arrived* at,
computed over the full index, and then silently scored whatever survived the
stride and the pose join. It now prints both — `depth arrived at 30.0 Hz, scored
at 30.0 Hz` — and says how many frames the join dropped.

It also scored the frames ARKit had not converged on, whose pose sits near the
origin by construction and then jumps when it converges. That is a fault in the
reference rather than in the estimator, and it is concentrated in exactly the
opening seconds the `--limit 200` run was reading — which is the mechanism that
made those first 200 frames score 18.6 cm/s. `tools/export_episodes.py` has
always dropped them; the scorer now does too.

So 7.2 cm/s stands as an upper bound on the weakest variant at an unknown rate,
not as the measurement. The one to trust is the re-run: same clip, frame-to-model
against `--frame-to-frame`, with the scored rate written down beside it.

## The depth question is still open, and why

Every depth-odometry figure in this document was computed on a session ARKit
rates at **98.4% `ARConfidenceLevel.low`**, with not one high-confidence pixel in
30 metres. It was recorded at 3:34 in the morning. ARKit's depth is guided by the
colour image, so a dark room returns a full-looking map the sensor does not stand
behind — and none of those numbers measure depth odometry. They measure the
lighting.

The debugging that led there is worth keeping, because each step ruled something
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

**So the depth path is neither proven nor disproven.** What it needs is one
session in good light where the confidence map shows a real majority of medium
and high pixels. `tools/depth_odometry.py` now prints that distribution first and
refuses to quietly score a session that fails it.

Worth noting for when that session exists: if ARKit's pose is available and the
plan is to fuse rather than replace, the bar drops a long way. Depth then has to
*improve* a trajectory rather than carry one, which is what the metric-scale
advantage is actually good for. That only applies on the ARKit path, though —
the ultra-wide is reached by leaving ARKit, so there is no pose there to fuse
with.
