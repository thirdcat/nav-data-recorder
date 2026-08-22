# A capture preset for scanning, and what raising the rate does not buy

The 5 Hz stills rate was never a capture limit. It is the rate **VLN episodes
are consumed at**, and the recorder throttles to it because storing more would
only be decimated away again. Depth already runs at 30 Hz on the same walk and
ARKit pose at 60, measured on disk: 29.99 Hz and 58.9 Hz across the corpus. One
stream was throttled and it was the images.

A Gaussian splat is not consumed at any rate, so that reason does not apply to
it. This page is the preset that separates the two, the numbers behind its
values, and — the longer half — what the change does **not** fix.

**None of this has been compiled.** There is no Mac behind this repo;
`.github/workflows/build-unsigned.yml` is the only compiler the app has. Nothing
below has been run on a device, and no session has been recorded under the scan
preset. The argument for why it builds is in §"Why this should compile"; it is
an argument, not a result.

## What the preset is

```
                       vln            scan
  stills               5 Hz           30 Hz
  depth                30 Hz          30 Hz        (unchanged — see below)
  duration cap         none           60 s
  exposure             auto           locked after 3 s   (multi-cam path only)
  multi-cam stills     5 Hz           10 Hz              (not 30 — see below)
  lenses               whichever recorder you opened; the manifest says which
```

`vln` is bit-for-bit the current defaults, so every session already in
`~/nav_data` was recorded under it and nothing about the existing path moves.

Two properties of the design matter more than the numbers.

**The name is derived, not stored.** `manifest["preset"]` is computed from the
values at session start by comparing them against each preset, so it cannot
disagree with the rates printed beside it. A config hand-edited away from both
reads `"custom"`, which is a legitimate answer and not a failure — `config` still
carries the exact rates. `docs/POSE.md` cost real time to the opposite of this:
sessions that did not say what format they ran under left every later reader
inferring it, and one capture came back on an old build and took a day to
notice.

**Nothing new is stored in `CaptureConfig`.** The preset adds no coding key, so
every settings blob already in `UserDefaults` and every `manifest.json` already
on disk keeps decoding unchanged. That is deliberate: Swift's synthesised
`Decodable` does **not** fall back to a property's default value for a missing
key, so a single new stored field would have thrown on every existing install's
settings and silently reset them through the `try?` in `AppSettings`.

## The measurements

All from `~/nav_data`, 58 sessions, read off the files rather than the manifest.

### Write bandwidth

JPEG sizes, every frame in the corpus:

```
  resolution     n      median     mean      p90      max
  1920x1440    5 065    343 KB    391 KB   649 KB   1280 KB
  3840x2160      886   1386 KB   1394 KB  1577 KB   2189 KB
  1920x1080      272    378 KB    377 KB   428 KB    446 KB
   640x 480    1 078     61 KB     66 KB    99 KB    128 KB
```

A per-stream model built from those, checked against the total bytes on disk
divided by session duration:

```
  ARKit path, 5 Hz                 predicted 6.4 MB/s   measured median 6.33, max 9.98
  multi-cam, 5 Hz per lens        predicted 11.0 MB/s   measured 10.7 – 11.7
```

The model reproduces the disk, so it can be extrapolated:

```
  ARKit path at 30 Hz stills
    images    391 KB x 30                        11.7 MB/s
    depth     256x192 float16 x 30                2.9
    conf      256x192 uint8   x 30                1.5
    pose+IMU                                      0.04
                                                 -------
                                                 16.2 MB/s   0.97 GB/min
    on the brightest session's frames (1173 KB)  39.6 MB/s   2.4  GB/min
```

**About 1 GB for a 60-second scan, and up to 2.4 GB in a bright, detailed
room.** Two consequences worth stating rather than discovering:

- `RecordingCoordinator.minimumFreeBytes` is 2 GiB and the 1 Hz tick stops the
  session below it. A worst-case scan is larger than that floor, so a phone near
  it will end a scan on storage rather than on the cap. The floor was not
  changed here — it is a safety margin for a stream that must not fail
  mid-write, and moving it is a separate decision.
- The Settings footprint estimate assumed 500 KB an image. It now uses the
  measured 391 KB mean. At 5 Hz that made the old estimate 28 % high; at 30 Hz
  the same error would have been three quarters of a gigabyte a minute.

**Depth volume does not change.** The image gate and the depth gate are both
`RateGate`s taking their first sample on the same frame, so at 30 Hz they run in
lockstep: the union is 30 Hz, not 60. Every session in the corpus has exactly
6.0 depth frames per image (172 / 1032 on `cb4586`); a scan converges that to
1.0 and the depth bytes stay where they are. The scan preset buys images and
costs image bytes, nothing else.

### The multi-camera path is a different budget, and it is already over

Every multi-cam session logs the same line:

```
  hardware cost 0.62418437, system pressure cost 1.9852095
```

`hardwareCost` under 1.0 is what makes the configuration legal at all.
`systemPressureCost` is AVFoundation's statement about whether it can be
*sustained*, and 1.985 is roughly twice the threshold — **at 5 Hz, before a
single JPEG is encoded**, because those costs are computed from the active
formats and know nothing about what the app does with the frames.

That path also shoots 3840x2160 at a 1386 KB median, and runs both lenses and
the depth writes through one serial queue that was building a fresh `CIContext`
per frame. So the scan preset gives it 10 Hz, not 30:

```
  multi-cam at 10 Hz per lens   18.2 MB/s   1.1 GB/min
  multi-cam at 30 Hz per lens   47.4 MB/s   2.8 GB/min
```

The `CIContext` is now built once. A context carries a Metal device, a command
queue and a shader cache; at 5 Hz across two lenses the recorder was discarding
ten of them a second and getting away with it, and that is the first thing that
breaks when the rate goes up.

### Thermal: the 35-second ceiling is a starting condition, not a duration

`docs/3DGS.md` cites a ~35 s session ceiling from `HANDOVER.md` §2 and builds the
whole merge argument on it. Read against `events.jsonl` across all 58 sessions,
that is not what the ceiling is.

**`thermal.throttle` has fired exactly once**, across the 46 ARKit sessions that
observe thermal state at all — the 12 multi-cam sessions do not observe it, so
they say nothing. Session `7d3d52`, 2026-08-13:

```
  56359.4   2994fa  session.start   thermal=fair
            2994fa  runs 35.0 s
  56416.4   7d3d52  session.start   thermal=fair
  56434.6   7d3d52  thermal         serious
  56434.6   7d3d52  thermal.throttle
```

It started already warm, as the second of a back-to-back pair, and reached
`serious` 18.2 s in — about 53 s of camera-on inside the preceding 75 s of wall
time. Meanwhile the two longest ARKit sessions in the corpus both **started at
`nominal` and never left it**:

```
  1868dd   51.8 s   started nominal   no thermal event
  87bc2c   52.4 s   started nominal   no thermal event   (battery 1.0)
```

Nine of the 46 ARKit sessions started at `fair` rather than `nominal`. Seven of
those nine began **18 to 64 seconds after the previous session started**, i.e.
inside a minute of another capture. The remaining two follow a device reboot —
the monotonic clock resets across them — so nothing in the corpus explains their
warmth and they are not evidence either way.

So the load that produces the ceiling is **back-to-back captures, not long
ones**, and 35 s was the length of the session that happened to precede the one
that throttled.

The 57.8 s figure is `d06152`, a multi-cam session — and `MultiCamRecorder` does
not observe `ProcessInfo.thermalState` at all, so its silence is not evidence
about anything.

**What this does and does not license.** It removes the claim that a session
cannot exceed 35 s. It does not license 30 Hz: nothing in the corpus ran above
5 Hz, so the added load is unmeasured. Two things bound it rather than settle it:

- The added load is **not** 6x the session. ARKit, the LiDAR and the ISP were
  already running at 60 Hz — the gate discards frames, it does not stop the
  sensor. What is six times larger is the JPEG encoding and the NAND writes, and
  those are a part of the budget, not the whole of it. That is an argument, not
  a measurement.
- `degradeOnThermalPressure` already pauses images and depth at `serious` while
  poses keep flowing, and logs every transition. A thermal overrun degrades
  visibly rather than failing.

The 60 s cap is therefore a **bound on an untested load**, not a thermal result.
It is roughly the longest capture anyone has recorded. If the first scans show
no thermal transition it should be raised on that evidence, and the evidence is
already in `events.jsonl`.

### The encoder is the untested part, and it is now loud

`StillsWriter` encodes on **one serial queue** and its own comment prices a JPEG
at about 20 ms. Against a 33 ms budget at 30 Hz that is 60 % duty with no
headroom; against 16.7 ms at 60 Hz it cannot work at all, which is why the scan
preset is 30 and not 60. When the queue falls behind, `maxPending = 8` makes
`capture` return `false` and the frame is dropped.

That failure used to be entirely silent — the return value was discarded, and a
session that recorded 22 Hz under a manifest saying 30 looked complete. Three
things now say so:

- `counts["framesDropped"]` in `manifest.json`, read after the encoder drains.
- a `stills.backlog` row in `events.jsonl`, at most once a second, so drops
  clustered at the end (a device heating up) can be told from drops spread
  evenly (a rate the encoder never had).
- an orange line on the recording screen while it is happening.

**If the first scan reports a non-zero `framesDropped`, the scan preset's rate
is wrong and should be 15 Hz.** That is the pre-registered response, and it is
written here before the first capture rather than after.

## What raising the rate does not fix

This is the half that matters, and it is measured rather than argued.

### Coverage barely moves

`docs/3DGS.md` measures the binding constraint: the median piece of surface is
seen from **one or two distinct directions**, the best session gets 26 % of its
vertical surface to three views and the median gets 12 %. It also measures what
six times the frames buys, by counting over every depth frame instead of only
the image-paired ones:

```
  session   at 5 Hz          at 30 Hz         gain
   cb4586   167 fr, 19.7%   1007 fr, 20.3%   +0.6 pp
   2735cf   114 fr,  5.0%    688 fr,  6.0%   +1.0 pp
   2994fa   166 fr, 14.6%   1001 fr, 18.6%   +4.0 pp
```

**Under one point on the sessions that walk in a line.** Against that, the same
page measures a second overlapping walk at **+7 to +23 points** on the shared
surface. The field of view caps any surface at 4.7 direction bins of 15° no
matter how long you walk past it, and frames added along a path the camera
already travelled land in bins that are already occupied.

So the scan preset is not the coverage fix and must not be sold as one. What it
buys is photometric supervision — six times as many images constraining the same
Gaussians — which the coverage measurement says nothing about in either
direction, and which nobody here has trained.

### It weakens the evaluation protocol, and by a mechanism with an exact number

At 5 Hz consecutive frames are far enough apart that a held-out view is a real
question. Measured on the four longest sessions, median interframe baseline:

```
  session   5 Hz     10 Hz    15 Hz    30 Hz    60 Hz
  cb4586   12.29 cm   6.21     4.12     2.05     1.03
  1868dd   13.33      6.73     4.48     2.23     1.12
  87bc2c    7.57      3.86     2.54     1.27     0.64
  2994fa   12.29      6.17     4.13     2.06     1.03
```

At 30 Hz **every held-out view has a training view about 2 cm away**. That is
what `--guard-m` exists to stop, and the default split does not survive it.
Simulating `tools/export_3dgs.py`'s own split — `--holdout-every 8`,
`--guard-m 0.15`, the same 10° alignment test — on the real poses:

```
                        holdout/8, guard 0.15 m
  session    rate   frames   held out   training frames left
  cb4586      5 Hz     168        21        101
              30 Hz   1007       126          0
  1868dd      5 Hz     255        32        165
              30 Hz   1525       191          3
  87bc2c      5 Hz     258        33        149
              30 Hz   1547       194          6
  2994fa      5 Hz     167        21        100
              30 Hz   1001       126          0
```

**At 30 Hz the default split leaves no training set at all.** The mechanism is
arithmetic, not subtlety: holding out every eighth frame at 2 cm spacing puts a
held-out camera every 16 cm, and a 15 cm guard radius around each one covers the
whole walk. The 30 % the guard band costs at 5 Hz becomes 87 %.

The fix is exact and it is a flag, not a code change — **the holdout stride has
to scale with the rate**:

```
                        rate-matched holdout stride, guard 0.15 m
  session    rate   stride   held out   training frames left   guard cost
  cb4586      5 Hz      8         21          101                27 %
              30 Hz    48         21          632                35 %
  1868dd      5 Hz      8         32          165                23 %
              30 Hz    48         32         1009                32 %
  87bc2c      5 Hz      8         33          149                30 %
              30 Hz    48         33          892                40 %
  2994fa      5 Hz      8         21          100                28 %
              30 Hz    48         21          650                33 %
```

`--holdout-every 48` at 30 Hz restores the same held-out views, at the same
spacing through the walk, with six times the training frames and a guard cost
within a few points of the 5 Hz figure. So a scan session must be exported as

```
python3 tools/export_3dgs.py <session> <out> --holdout-every 48 --guard-m 0.15
```

and **`--holdout-every 8` on a scan session is a silent error** — it does not
crash, it produces an empty or near-empty training set and a model that has
learned nothing, which reads downstream as a bad scene rather than as a bad
split. This is exactly why `manifest["preset"]` and `config.stillsHz` have to be
in the session: the correct stride is a function of the capture rate, and no
exporter can pick it from an image directory.

None of that has been run through the exporter — the simulation above reads the
same poses and reimplements `apply_guard_band`'s radius-and-angle test, and the
tools themselves were not touched.

### Exposure: fixed on one path, not fixable on the other

`docs/3DGS.md` records that exposure sits at 16.7 ms in 24 of 27 sessions
because `pickFormat` sorts by pixel area and `1920x1440@60` caps the shutter,
and that at 0.6 m/s the camera moves 10 mm during one exposure.

**The rate change does nothing to that, and the ARKit path cannot.**
`ARWorldTrackingConfiguration` exposes no exposure control at all — only
`ARFrame.camera.exposureDuration` to read back, which the recorder already
writes into every pose row. Taking the 30 fps format would let auto-exposure
open to 1/30 and halve noise in the dark sessions at the cost of doubling motion
blur; `docs/3DGS.md` argues that is a setting to measure both arms of, not a
default to flip, and this change does not flip it. `formatPreference` is
untouched and identical in both presets.

What *is* fixable is the other exposure problem: the multi-camera path runs
**continuous auto-exposure on both lenses**, so brightness drifts within a walk
and between walks, and a room captured as several fragments hands that drift to
the merge. The scan preset locks exposure on both devices after a 3 s warm-up.
Three details are load-bearing:

- **Delayed, not set at configuration.** `.locked` freezes whatever the device
  has converged on now, and locking at configuration pins a value measured
  against the inside of a pocket.
- **Both lenses.** Only the ultra-wide had its optics pinned before; the wide
  arm off the LiDAR device kept auto-exposure, which is half of the brightness
  the merge inherits.
- **The values it settled on go into `notes`.** A lock that caught a bad moment
  produces a whole session at the wrong exposure and is indistinguishable on
  disk from one that worked. The duration, the ISO, and whether the device was
  still adjusting when the lock was taken are all written down.

This is untested, like everything else here. It is also the one change on this
page that a per-image appearance embedding downstream would otherwise have to
undo — which is a model learning to unpick something the capture chose to do.

## Why this should compile

Stated as an argument because it cannot be stated as a result.

- **No new stored property anywhere that is decoded.** `CaptureConfig` gains an
  enum, three computed properties and one method; its `CodingKeys` are
  unchanged, so every persisted settings blob decodes exactly as before.
- **`SessionManifest.preset` is `String?`.** Synthesised `Decodable` treats a
  missing key for an `Optional` as `nil` rather than throwing, so
  `SessionStore.readManifest` keeps reading every manifest already on the
  device. It sits between `config` and `video` in the declaration, and is passed
  in that position at the one call site, so the memberwise initialiser's order
  still matches.
- **`applying` and `matches` share one list.** `matches` is
  `applying(preset) == self` against the synthesised `Equatable`, and `applying`
  starts from the preset and copies the *non*-owned fields across. There is no
  second list to drift, and a field added later and forgotten becomes
  preset-owned by default — which reads as `"custom"`, the safe direction.
- **Every `switch` returns explicitly** rather than using switch-expression
  syntax, and the target is `SWIFT_VERSION 5.0` on iOS 17.
- **The AVFoundation calls are the narrow set.**
  `isExposureModeSupported(.locked)`, `exposureMode`, `lockForConfiguration()`,
  `exposureDuration`, `iso`, `isAdjustingExposure` — all long-standing
  `AVCaptureDevice` members, and the lock is guarded on support and on the
  `lockForConfiguration` throw.
- **`queue.async` from inside `queue`**, which is what the duration cap does
  when it calls `stop()`, enqueues rather than deadlocking. `queue.sync` would
  not.
- **The multi-cam manifest's new values are built into locals** before the
  `[String: Any]` literal. That literal was already large, and heterogeneous
  dictionary literals are where the Swift type checker times out.

What CI will catch that this argument cannot: a SwiftUI body that type-checks
too slowly, a `ViewBuilder` child count, and any API I have misremembered.

## What was not verified

- **Nothing was compiled.** No device ran any of it.
- **No session has been recorded at 30 Hz**, so the encoder's real throughput,
  the actual thermal curve and the real drop rate are all unknown. The
  instruments to answer all three are in this change and their pre-registered
  reading is above.
- **No splat has been trained on a scan session**, so whether six times the
  photometric supervision is worth anything is open. The coverage measurement
  says it will not move angular coverage; it says nothing about pixels.
- **The exposure lock has never been taken.** Whether 3 s is long enough for
  both devices to converge is a guess, which is why the values it lands on are
  recorded.
- **The holdout simulation is a reimplementation**, not a run of
  `tools/export_3dgs.py`. It reads the same poses and reproduces
  `apply_guard_band`'s radius-and-angle test; the tools were not touched.
- **`systemPressureCost` is Apple's number about Apple's formats.** It does not
  include the JPEG encoding or the disk writes this app adds on top, so it
  under-reports the real load rather than over-reporting it.
