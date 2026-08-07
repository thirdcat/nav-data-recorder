# Session data format

Schema version 1. `tools/read_session.py` is the reference reader.

## Layout

One directory per session under `Documents/sessions/`:

```
20260807-014530-a1b2c3/
├── manifest.json      what everything else means
├── location.jsonl     GPS, ~1 Hz
├── heading.jsonl      compass, ~1 Hz
├── motion.jsonl       IMU, 100 Hz
├── pose.jsonl         camera pose + intrinsics, one row per ARKit frame
├── depth.jsonl        index into depth.bin
├── depth.bin          raw float16 depth maps
├── confidence.bin     per-pixel depth confidence (only if enabled)
├── events.jsonl       thermal, tracking, backgrounding, errors
├── video.mov          HEVC, PTS in the master clock domain
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
when NTP corrects the system clock mid-drive. Only `CLLocation` reports wall
time, and it is converted on the way in. Its original value is kept in
`location.wall` as a cross-check.

Video PTS values are written in the same domain, **not** starting from zero: a
frame's presentation timestamp is numerically equal to the matching
`PoseSample.t`. Aligning video to sensors is a lookup, not a computation.

`manifest["startClock"]` is the master clock when recording began, so
`t - startClock` is seconds-into-session.

## Why JSONL

A session can die mid-drive — crash, battery, force quit. With newline-delimited
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

### motion.jsonl

`CMDeviceMotion` at 100 Hz, in the `xArbitraryCorrectedZVertical` reference
frame (gravity-aligned, magnetometer-corrected yaw, arbitrary yaw origin).

- `ax ay az` — user acceleration, **G**, device frame, gravity already removed
- `gx gy gz` — gravity vector, G
- `rx ry rz` — rotation rate, rad/s, bias-corrected
- `qx qy qz qw` — attitude quaternion
- `mx my mz` — calibrated magnetic field, µT
- `magAcc` — calibration accuracy; `-1` means uncalibrated

Fused device motion is recorded rather than raw accelerometer and gyroscope
because CoreLocation's sensor fusion has access to calibration data a downstream
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
return** — sky, glass, anything past a few metres.

Depth is captured at its own rate (5 Hz by default) rather than per video frame:
at 30 Hz it would be roughly 10 GB per hour on its own.

When confidence capture is enabled, `confidence.bin` holds one byte per pixel
(`ARConfidenceLevel`: 0 low, 1 medium, 2 high) at the offsets given in the index.

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
- **Autofocus is left on.** ARKit does not expose a fixed-focus mode, so
  intrinsics can shift slightly as focus hunts. `fx fy cx cy` are recorded per
  frame for exactly this reason — do not assume they are constant.
- **`video.mov` is unplayable if `.complete` is missing.** The file is only
  finalised on a clean stop; a session killed mid-drive keeps every JSONL row
  but loses the video container.
