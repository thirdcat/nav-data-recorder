# Session data format

Schema version 1. `tools/read_session.py` is the reference reader.

This is what the app writes to the phone. For the shape the VLN dataset
consumes downstream — an `images/` + `poses_tum.txt` episode, and the
coordinate-frame change that gets there — see
[OUTPUT_FORMAT.md](OUTPUT_FORMAT.md).

## Layout

One directory per session under `Documents/sessions/`:

```
20260807-014530-a1b2c3/
├── manifest.json      what everything else means
├── location.jsonl     GPS, ~1 Hz
├── heading.jsonl      compass, ~1 Hz
├── motion.jsonl       IMU, 100 Hz
├── pose.jsonl         camera pose + intrinsics, one row per ARKit frame
├── planes.jsonl       detected floors, walls, ceilings, tables
├── frames.jsonl       index of written images (stills mode)
├── frames/            000000.jpg, 000012.jpg, … (stills mode)
├── depth.jsonl        index into depth.bin
├── depth.bin          raw float16 depth maps
├── confidence.bin     per-pixel depth confidence (only if enabled)
├── events.jsonl       thermal, tracking, backgrounding, errors
├── video.mov          HEVC, PTS in the master clock domain (video mode)
├── .complete          present once every file was closed cleanly
└── .upload.json       local bookkeeping; not part of the dataset
```

The directory name is `yyyyMMdd-HHmmss-<random>` in **UTC**, so lexical order is
chronological order.

## Time

Every `t` is **seconds in the device's monotonic since-boot clock**
(`CACurrentMediaTime()`).

```
unix_time = t + manifest["clockAnchor"]
```

The monotonic clock is the master rather than wall time because three of the
four sources already speak it — `CMDeviceMotion.timestamp`, `ARFrame.timestamp`
and `CMSampleBuffer` PTS are all in that domain — and because it does not jump
when NTP corrects the system clock mid-session. Only `CLLocation` reports wall
time, and it is converted on the way in. Its original value is kept in
`location.wall` as a cross-check.

Video PTS values are written in the same domain, **not** starting from zero: a
frame's presentation timestamp is numerically equal to the matching
`PoseSample.t`. Aligning video to sensors is a lookup, not a computation.

`manifest["startClock"]` is the master clock when recording began, so
`t - startClock` is seconds-into-session.

## Why JSONL

A session can die mid-recording — crash, battery, force quit. With newline-delimited
JSON every completed line before the failure is still readable, and at worst the
final line is torn. A single JSON array would be unparseable in that situation.
The reader drops a trailing malformed line rather than rejecting the stream.

Cross-check a session's integrity with `manifest["counts"]`, written after the
files close, against the actual line counts. A mismatch means the session was
interrupted between the last flush and the close.

## Streams

### location.jsonl

| field | meaning |
| --- | --- |
| `t` | master clock |
| `wall` | CoreLocation's own Unix timestamp |
| `lat`, `lon` | degrees |
| `alt` | mean sea level, metres |
| `altEllipsoidal` | WGS-84 ellipsoidal height — use this one for anything geodetic |
| `hAcc`, `vAcc` | accuracy, metres; **negative means invalid** |
| `speed`, `speedAcc` | m/s; negative means invalid |
| `course`, `courseAcc` | degrees clockwise from true north; negative means invalid |

CoreLocation signals invalidity with negative values rather than nulls. Filter on
`hAcc > 0` before using a fix.

**Indoors this is a context stream, not a pose source.** A fix inside a building
is metres to tens of metres wrong when it arrives at all, and `speed` and
`course` are usually flagged invalid because there is no coherent motion to
derive them from. It answers "which building", nothing finer. ARKit's
visual-inertial odometry in `pose.jsonl` is what actually localises the camera.

### motion.jsonl

`CMDeviceMotion` at 100 Hz. The reference frame is recorded in
`manifest.device.attitudeReferenceFrame` and **must be read from there** — the
quaternions are meaningless without it.

- `xArbitraryZVertical` (the indoor default) — gravity-aligned, arbitrary yaw
  origin, no magnetometer input
- `xArbitraryCorrectedZVertical` — the same, plus yaw pulled towards magnetic
  north

Correction is off by default because a home is full of things that lie about
where magnetic north is: steel studs, wiring, appliances, speaker magnets. An
uncorrected frame drifts smoothly, which is far easier to model than one that
snaps as you walk past a refrigerator. Outdoors the trade-off reverses.

- `ax ay az` — user acceleration, **G**, device frame, gravity already removed
- `gx gy gz` — gravity vector, G
- `rx ry rz` — rotation rate, rad/s, bias-corrected
- `qx qy qz qw` — attitude quaternion
- `mx my mz` — calibrated magnetic field, µT
- `magAcc` — calibration accuracy; `-1` means uncalibrated

Fused device motion is recorded rather than raw accelerometer and gyroscope
because CoreMotion's sensor fusion has access to calibration data a downstream
pipeline does not.

### pose.jsonl

One row per `ARFrame`, including frames that were not encoded into the video (a
dropped or thermally-paused frame still gets a pose).

- `frame` — index into the video track: the Nth frame appended to `video.mov`
- `tx ty tz` — camera position, metres, ARKit world frame
- `qx qy qz qw` — orientation, world-from-camera
- `fx fy cx cy` — pinhole intrinsics in pixels, for the full-resolution frame
- `tracking` — `normal`, `limited:<reason>`, or `notAvailable`
- `exposure` — seconds; useful for rejecting motion-blurred frames
- `gravX gravY gravZ` — unit gravity vector in **camera** coordinates

**The world frame is session-local.** The origin is wherever the session
started, and yaw is arbitrary. Poses are a local odometry track, not a
geographic one — fuse with `location.jsonl` to georeference.

**An `ar.interruptionEnded` event is a hard break.** ARKit resets the world
origin when capture resumes, so poses either side of that event are in different
coordinate frames and must not be concatenated.

### depth.jsonl + depth.bin

`depth.jsonl` indexes into `depth.bin`:

```json
{"t": 12345.6, "frame": 180, "offset": 393216, "length": 98304,
 "width": 256, "height": 192, "format": "float16",
 "confidenceOffset": null, "confidenceLength": null}
```

`depth.bin` is raw row-major float16 metres, frames concatenated with no header.
Read `length` bytes at `offset`.

Float16 rather than ARKit's native Float32: the quantisation error is about a
millimetre at 10 m, far below the sensor's noise floor, and it halves a payload
that is otherwise the largest thing on disk. **Non-finite values mean no
return** — indoors that is mostly glass, mirrors, glossy screens and anything
past roughly 5 m, which is the sensor's useful range.

Depth is captured at its own rate (30 Hz by default) rather than per video
frame, so it can be traded against storage independently. **In stills mode it
runs faster than the images and is a superset of them** — every captured image
still gets a depth map carrying the same `frame`, plus depth-only frames in
between. The export rate and the capture rate are different questions: episodes
want 5 Hz, but frame-to-frame depth registration wants every frame it can get,
because ICP converges on small motion and 5 Hz at walking pace puts frames about
10 cm and several degrees apart. Decimating afterwards is free —
`tools/export_episodes.py --hz 5` — and frames never captured are gone. Indoors it is the
primary signal and a room scan runs for minutes, so 30 Hz is viable; outdoors,
at 30 Hz it alone would be roughly 10 GB per hour.

When confidence capture is enabled, `confidence.bin` holds one byte per pixel
(`ARConfidenceLevel`: 0 low, 1 medium, 2 high) at the offsets given in the index.

### frames.jsonl + frames/

Stills mode writes one JPEG per captured frame and indexes it here:

```json
{"t": 12345.2, "frame": 12, "file": "frames/000012.jpg",
 "width": 1920, "height": 1440, "bytes": 498231}
```

**`frame` is the join key to `pose.jsonl` and `depth.jsonl`.** All three come
out of the same `ARFrame`, so joining on it is exact — no timestamp search, no
interpolation, no sub-frame offset to correct for. `Session.posed_images()` in
the reference reader does that join and is the shape a posed-image training
pipeline wants.

Making that true required one gate to drive both the image and its depth map.
Sampling them independently does not work: `ARFrame.sceneDepth` is nil on some
frames, so a depth-only rate gate does not advance in step with the image gate
and the two land on different frames within seconds. Sessions recorded before
this was fixed have no exact matches at all; the reader falls back to the
nearest depth in time and reports the offset as `depth_dt`, which is `0.0` for
a real pairing.

What that gate guarantees is that every *image* has a depth map from its own
`ARFrame`. It does not cap the depth rate: `depthHz` runs its own gate in both
modes, at 30 Hz by default, so depth is a superset of the images rather than a
copy of their schedule. That is deliberate — pose estimation from depth wants
every frame it can get, and the export rate is a separate decision made later by
`tools/export_episodes.py --hz`.

Images are written in the camera's **native landscape orientation, unrotated**,
which is the orientation the recorded intrinsics describe. Rotating them without
transforming `fx fy cx cy` to match would silently invalidate every pose.

Stills rather than video by default: extracting frames back out of an HEVC file
costs a lossy generation and a seek-and-decode step in the dataloader. Video
mode remains available and is roughly half the bytes.

### How the phone was held

**Nothing in the images says whether a session was shot landscape or portrait.**
ARKit always delivers `capturedImage` in the camera's native landscape
orientation, so a portrait-held recording produces the same 1920×1440 buffers as
a landscape one — with the world rotated 90° inside them. Same dimensions, same
structure, different content.

`gravX gravY gravZ` in `pose.jsonl` is what resolves it:

```python
roll = math.degrees(math.atan2(pose["gravX"], -pose["gravY"]))
```

Rotate the image by `-roll` and gravity points down in it. `0` means landscape,
`±90` portrait, and anything between means the phone was tilted — which is the
normal handheld case, and why this is a continuous value rather than one of four
discrete orientations.

`Session.shot_orientation()` summarises this across a session and warns when the
orientation changed mid-recording, which would otherwise produce a silently
inconsistent set of images.

**Rotating images to upright is a downstream choice, not something the recorder
does.** The stored intrinsics describe the unrotated buffer; a 90° rotation
requires swapping `fx`↔`fy` and `cx`↔`cy` to match, and rotating the pixels
without that invalidates every pose.

### Camera geometry

Field of view is **fixed by the lens, not the capture format**.

An iPhone 16 Pro reports its widest wide-angle format as **74.6° horizontal**
(Settings → Hardware capabilities). At a 4:3 1920×1440 frame that implies
roughly 60° vertical and 87° diagonal, and an `fx` near 1260.

**Whether ARKit delivers that full width is a separate question** — a format may
crop for stabilisation, which shows up as a larger `fx` and a narrower view.
Measure it from a session rather than assuming: `Session.field_of_view()` reads
the recorded per-frame intrinsics, which is the only figure that describes the
images actually on disk.

**A 16:9 format does not widen the horizontal view** — that is set by the lens.
What it changes is sensor height. Measured against a 4:3 1920×1440 frame at
73.1° horizontal, a 3840×2160 frame at the same horizontal view works out near
45° vertical, against 58°.

The recorder prefers 4:3 by default (`CaptureConfig.formatPreference`), on three
grounds:

- **Vertical coverage is the scarce axis indoors.** Floor and ceiling are what
  a navigation model needs; 58° also lands close to the 60° vertical the
  Matterport3D panorama views use, and Habitat's VLN-CE default sensor is 4:3.
- **The extra pixels are discarded anyway.** A 224-pixel encoder is already
  fed a 55× oversampled image at 1920×1440; 4K makes that 165×.
- **It costs.** Roughly 150 MB/min against 450 MB/min at 5 Hz, and a heavier
  thermal load on a device that throttles.

That is a judgement rather than a measurement — whether a given 16:9 format
crops vertically or all round is device-specific and Apple does not document it.
The setting exists so it can be checked: switch to *16:9 — more pixels*, record,
and compare the `sensor fov` line. If horizontal stays put and only vertical
falls, the reasoning above holds; if horizontal also drops, 16:9 is strictly
worse and the default is right for a second reason.

**Which axis is horizontal in the world depends on how the phone was held:**

| held | world H | world V |
| --- | --- | --- |
| landscape | 73.1° | 58.1° |
| portrait | 58.1° | 73.1° |

Portrait costs about 15° of horizontal coverage — the axis that matters most for
navigation, where the question is what lies left and right. Landscape is the
right default; `Session.world_field_of_view()` reports the effective figures for
a given session.

`Session.field_of_view()` computes these from a session's actual intrinsics
rather than quoting a spec figure, since focal length varies slightly between
devices and shifts as autofocus hunts. That is also why `fx fy cx cy` are stored
per frame rather than once per session.

For comparison, Habitat-based VLN work usually renders RGB at **90° horizontal**
FOV, and Matterport3D panoramas cover 360°. The wide camera is narrower than
either, which is a real domain gap if the model is pretrained on those — though
at 74.6° the gap is about 15°, not the ~28° an earlier estimate here suggested.

#### The rig this is collected for

Capture is only as useful as its match to deployment. The target is a **Unitree
G1** with an **OAK-1-W** head camera:

| | horizontal | vertical | resolution | height |
| --- | --- | --- | --- | --- |
| OAK-1-W on the G1 | 96.3° | 64.6° | 848×480 | 1.15 m |
| phone, 4:3 | 73.1° | 58.1° | 1920×1440 | *whatever you hold it at* |
| phone, 16:9 4K | 73.1° | 45.3° | 3840×2160 | |
| phone, ultra-wide | 106.2° | 89.9° | — | *not reachable from ARKit* |

Three things follow.

**4:3 is the right frame shape**, and this is why rather than by taste: it is
6.5° short of the target vertically, where 16:9 is 19.3° short. Resolution is
not the deciding factor — deployment is 0.41 MP, so a 1920×1440 capture is
already 6.8× oversampled and 4K would be 20×.

**The horizontal gap, 23.2°, is the one real mismatch** and no aspect ratio
fixes it. Options, cheapest first: crop the robot's camera to 73° at inference
(free, exact match, costs the robot field of view); train across the gap and
accept it; capture with the ultra-wide instead; or synthesise the wider view
offline from posed RGB-D.

**Viewpoint height is free to match and costs a lot to get wrong.** Hold the
phone at **1.15 m** — roughly hip height, not chest — and match the robot's
camera pitch. A policy trained from one height and run from another pays for the
difference, and no amount of downstream processing recovers it. The reader
reports measured height whenever a floor plane was classified, and flags a
session that drifts more than 15 cm off.

Worth noting for later: the ultra-wide at 106.2° covers the target's 96.3° with
margin, and a 1609×911 centre crop of a 1920×1440 ultra-wide frame reproduces the
OAK-1-W's geometry almost exactly before downsampling to 848×480. That is the
strongest argument for the AVFoundation path — the ultra-wide is not a
nice-to-have there, it is the lens that matches the deployment camera.

#### Where the vertical view lands

58° vertical is not much, and where it points decides what it covers. Held level
at chest height, the bottom of the frame meets the floor about **2.7 m ahead** —
everything nearer is not recorded at all, which for a navigation model is the
part that matters most.

| pitch below horizontal | nearest floor in frame |
| --- | --- |
| 0° | 2.7 m |
| 10° | 1.9 m |
| 15° | 1.6 m |
| 20° | 1.3 m |

Tilting down trades ceiling for near floor: past about 20° the ceiling leaves
the frame entirely.

**Match the reference set, not this geometry.** On its own the trade argues for
about 15°, but the reference episodes were shot at 24.6°, 25.4° and 31.5° below
horizontal, and the VLM prompts downstream were tuned on that floor-heavy view.
Aim for **~25°**: agreeing with the data the labeller already works on beats
winning an argument about ceilings. See [OUTPUT_FORMAT.md](OUTPUT_FORMAT.md).

`Session.camera_aim()` reports the mean pitch of a session, derived from gravity
in camera coordinates, and warns when a recording was held level — the same
figure the reference episodes are quoted in, so the comparison is one command. Consistency
matters as much as the value — a dataset shot at wildly varying pitch is harder
to learn from than one shot consistently at the wrong pitch.

For reference, on the vertical axis: Matterport3D's R2R views use 60°, and
Habitat's VLN-CE default (640×480 at 90° horizontal) works out to 73.7°.

#### Why not the 0.5x ultra-wide?

The ultra-wide lens is genuinely wide — an iPhone 16 Pro reports **106.2°
horizontal**, clearing the 90° reference outright. ARKit does not offer it.
Every format in `ARWorldTrackingConfiguration.supportedVideoFormats` reports the
built-in **wide-angle** camera; world tracking is calibrated against that lens
and there is no configuration that moves it to the ultra-wide.

It is, however, reachable *alongside* the LiDAR depth camera outside ARKit. On
an iPhone 16 Pro, `supportedMultiCamDeviceSets` includes
`LiDARDepthCamera + UltraWideCamera` — so an `AVCaptureMultiCamSession` can run
wide RGB, LiDAR depth and 106° ultra-wide RGB simultaneously off one clock, with
factory extrinsics between the lenses. That is the whole of what the
offline-SLAM path needs; what it costs is ARKit's pose.

Reaching the ultra-wide means an `AVCaptureSession`, which cannot coexist with
an `ARSession` — so it costs the entire reason this app uses ARKit:

- no 6-DoF pose (the ultra-wide stream would need its own SLAM)
- no depth alignment (the LiDAR map is registered to the wide camera's frame)
- pinhole intrinsics no longer describe the image; a ~118° diagonal lens needs
  distortion coefficients, which ARKit does not supply
- worse low light (smaller aperture and sensor), which matters indoors

**Measurement confirms the gap rather than closing it.**
`tools/estimate_intrinsics.py` recovers 96.3° × 64.6° from the simulation
episodes, to the decimal — the official set really is shot at the target rig.
It also recovers **66.1°** from the `IMG_108x` episodes, but those are an
iPhone 16 Pro trial and a minority of the mix: 66.1° is this very phone, cropped
about 12% by the video stabilisation ARKit does not apply. That makes them a
useful check on the estimator and no kind of target. Matching deployment still
means widening, and the ultra-wide is still the only lens on the phone that
reaches.

**A shipping app confirms the split.** [Gaussian
SplatKing](https://radiancefields.com/splatking) captures for 3DGS and wants
both halves of this, so it built both — as *separate modes*. Its Photo and Video
modes fire the 0.5× ultra-wide and 1× wide simultaneously; its LiDAR mode pairs
ARKit depth with the 1× wide **alone**, and that is the only mode that exports
poses (a COLMAP text model). A team optimising for reconstruction coverage still
could not put ARKit and the ultra-wide in one session. Apple's own answer on the
[developer forums](https://developer.apple.com/forums/thread/719837), to exactly
this question, is *"That is not possible at this time."*

What replacing ARKit's pose would actually involve is written up in
[POSE.md](POSE.md).

The first link in that chain is now testable without building any of it.
Settings → **Ultra-wide probe** captures ultra-wide stills with their
`AVCameraCalibrationData`, and `tools/rectify_ultrawide.py` reprojects them onto
a pinhole of any requested field of view — see [ULTRAWIDE.md](ULTRAWIDE.md).
That answers whether a rectified ultra-wide frame is geometrically good enough
to be worth the pose problem; it does not touch the pose problem.

Each session logs its available formats to `events.jsonl` under `ar.formats`,
including each format's capture device, so any recording answers this question
for the device it was made on rather than relying on the claim above.
Settings → Hardware capabilities reports the same thing for the AVFoundation
side, including which camera combinations `supportedMultiCamDeviceSets` allows.

**The cheaper fix is to narrow the simulator, not widen the phone.** Habitat's
camera sensor takes an `hfov` parameter, and Matterport3D panoramas are
equirectangular, so perspective crops can be rendered at any FOV. Matching the
rendered data to the phone's measured ~73° costs nothing and closes the gap
exactly; matching the phone to the renderer is not available at any price.

### planes.jsonl

ARKit plane anchors — the structural skeleton of an indoor scene, derived from
the LiDAR return rather than guessed from imagery.

| field | meaning |
| --- | --- |
| `id` | anchor UUID, stable within a session |
| `event` | `added`, `updated` or `removed` |
| `alignment` | `horizontal` or `vertical` |
| `classification` | `floor`, `wall`, `ceiling`, `table`, `seat`, `door`, `window`, or `none:<reason>` |
| `tx ty tz`, `qx qy qz qw` | anchor pose in the session world frame |
| `cx cy cz` | plane centre, in the anchor's own frame |
| `width`, `height`, `rotationOnYAxis` | extent in metres |

Planes grow and merge as a room is explored, so one plane produces many rows.
`updated` rows are throttled to at most one per second per plane; `added` and
`removed` are never dropped. A plane that ARKit merges into another is reported
`removed` — spurious detections do disappear, so do not assume every `added` is
real.

`Session.final_planes()` in the reference reader collapses the stream to the
last state of each surviving plane, which is usually what a consumer wants.

### events.jsonl

`{"t": …, "kind": …, "detail": …}`. The ones that change how the data should be
read:

| kind | meaning |
| --- | --- |
| `thermal.throttle` / `thermal.resume` | video and depth paused / resumed; GPS and IMU unaffected |
| `app.background` | camera suspended by iOS from here until `app.foreground` |
| `ar.interrupted` / `ar.interruptionEnded` | camera stopped; **world origin resets on resume** |
| `location.foregroundOnly` | Always authorization was missing, so the track dies when the screen locks |
| `location.authorization` | authorization changed mid-session |
| `ar.tracking` | `limited:excessiveMotion` and `limited:insufficientFeatures` are common indoors — blank walls give the tracker nothing to hold onto |
| `session.stop` | `user`, `outOfStorage`, or another termination reason |

`manifest["terminationReason"]` is `null` for a normal user-initiated stop.

## Alignment

`Session.align_to_poses()` in the reference reader produces one row per camera
pose, carrying the nearest GPS fix and IMU sample plus the depth entry for that
frame.

It is nearest-neighbour, not interpolation, and it reports the actual gap in
`location_dt` / `motion_dt`. Interpolating a 1 Hz GPS stream onto 30 Hz video
would manufacture a precision the data does not have; exposing the gap lets a
consumer reject rows that are too stale for its purposes.

## Known gaps

- **No video during backgrounding or thermal throttling.** Pose rows continue,
  video frames do not. Expect `pose.jsonl` to have more rows than the video has
  frames, and use `PoseSample.frame` rather than row index to index into it.
- **Autofocus is on by default, and it moves the field of view.** Two sessions
  on the same lens and format measured 73.1° and 70.9° horizontal — focus hunting
  changes the focal length, and `fx fy cx cy` are recorded per frame for exactly
  this reason. Do not assume they are constant.

  It *can* be switched off: `ARWorldTrackingConfiguration.isAutoFocusEnabled`
  defaults to `true`, and Settings → *Lock focus* sets it `false` for fixed
  focus. (An earlier version of this document claimed ARKit exposed no such
  mode. It was wrong.) The trade is a fixed focal plane — anything within about
  a metre softens — so it is off by default and left as something to measure:
  record both ways and compare the H spread `tools/export_episodes.py` reports.
  `ar.started` logs `autofocus=` so a session says which way it was shot.
- **`video.mov` is unplayable if `.complete` is missing.** The file is only
  finalised on a clean stop; a session killed mid-recording keeps every JSONL row
  but loses the video container. Stills mode has no such failure: every JPEG
  already on disk stays readable.
- **Camera FOV is ~73° horizontal** (measured, 4:3 landscape), against the
  96.3° of the deployment rig the official episodes are shot at and the 90° that
  Habitat-based VLN pipelines typically assume. `tools/estimate_intrinsics.py`
  recovers these from an episode set's own poses. See *Camera geometry* above and
  *Known gaps* in [OUTPUT_FORMAT.md](OUTPUT_FORMAT.md).
- **Poses are online VIO estimates, not a bundle-adjusted trajectory.** ARKit
  reports its current best guess, and indoor drift is on the order of 1–2% of
  distance travelled. For a room-scale loop that is centimetres; over a whole
  flat it is not. Relative pose between nearby frames is much better than
  absolute pose across a long session.
- **Poses are session-local, so sessions do not share coordinates.** Two scans of
  the same room have unrelated origins and yaw. Registering them against each
  other is a downstream problem — the app does not yet persist an `ARWorldMap`
  to relocalise into.
