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

## The honest summary

Rectified ultra-wide frames are ready. Poses for them are a project, not a
patch: an offline VIO or SfM pipeline with a scale anchor, against an ARKit
baseline that drifts about 0.02 m/s and costs nothing. Whether that trade is
worth 96.3° instead of 73.1° is a question about the training mix, not about the
phone.
