# Factory calibration read off this device

The two lenses of the back triple camera, as iOS reports them, captured by
**Settings → Ultra-wide probe** on iPhone17,1 and copied here unmodified.

```
             fx @ 4032x3024   FOV      focal    pixel     extrinsic vs reference
  Wide          2745.4       72.6 deg  6.70 mm  2.44 um   identity — this IS the reference
  UltraWide     1616.6      102.5 deg  2.26 mm  1.40 um   below
```

**Ultra-wide relative to wide:**

```
  translation   (-19.234, -0.019, -1.210) mm      magnitude 19.272 mm
  rotation      0.4620 deg about (+0.72, -0.67, +0.17)
```

## These are the factory format's numbers, and nothing records at that format

The table above is measured at 4032x3024. **No capture path in this repo runs at
4032x3024**, and on this device a different format is not a resample of that one
— it is a different readout with a different field of view. Scaling these
numbers down therefore describes a camera that was not used.

How far off, measured on `20260818-160918-2d9844`, where the recorder logs each
lens's active format instead of leaving it to be inferred:

```
                    factory, 4032x3024    active format        error
  Ultra-wide            102.5 deg          106.2007 deg        +3.7
  Wide                   72.6              69.52744           -3.1
  Depth                  72.6 (assumed)    70.37728           -2.2
```

The ultra-wide half of this was found first and `eval/pi3_poseless.py` was fixed
to prefer the logged field of view. The wide half survived another week because
the recorder logged nothing for that lens, so there was no logged value to
prefer and the scaled factory number looked like the only option. **The error
was never specific to the ultra-wide; it was specific to whichever lens had
nobody watching it.**

Two independent confirmations that the wide figure is real and not a logging
artefact:

- The two lenses shoot one scene simultaneously, so a similarity fit between
  them recovers `f_wide / f_ultrawide` directly. Nine pairs across three
  sessions, at 1.0 ms mean capture separation, give **s = 0.3176, IQR 0.0038**
  (`eval/wide_fov_probe.py`). The factory assumption predicts 0.3023, which is
  **4.0x the IQR away**; the logged 69.52744 deg predicts 0.3198, **0.6x away**.
  That measurement was made and pre-registered *before* the recorder logged
  anything, and it picked the right answer.
- `AVDepthData.cameraCalibrationData` reports the depth camera's own intrinsics
  for the running format: `fx = fy = 453.82` at 640x480, i.e. 70.377 deg. The
  recorder now writes this into `manifest.json` as `depth_calibration`, so no
  reader has to scale anything.

**The depth is 70.38 deg and the wide video is 69.53 deg — they are not the same
cone**, though they come off the same device. It is under a degree, but anything
that lays depth onto the wide image by a plain resize is wrong by that much;
`eval/pi3_poseless.py` resamples through the intrinsics instead.

One number here is still unreconciled. Diagnostics on this handset reports the
LiDAR depth camera at `fov (h) 74.6 deg`, the factory calibration says 72.6, and
the active format measures 70.38. Three figures for one camera. The active
format's is the one to use because it is the one that ran, but nobody has
explained the other two, and a parallax measurement that swept assumed depth
fields of view from 65 to 106 deg could not reproduce the expected baseline at
any of them. That is recorded as open, not resolved.

## Why this file exists

`docs/POSE.md` and this project's planning spent time on the premise that the
ultra-wide to LiDAR transform was unknown and would need a calibration rig. It
was never unknown. Apple measures it at the factory, iOS hands it over, and
`UltraWideProbe.swift` has been asking for it since 2026-08-07.

The reason it looked missing: `MultiCamDepthProbe` read
`AVDepthData.cameraCalibrationData.extrinsicMatrix` and got a translation of
`(0, 0, 0)`, which was recorded as "extrinsics unavailable". That is the wrong
reading. The extrinsic is expressed relative to a reference camera and **the
reference camera gets the identity**, so zeros mean "this is the reference", not
"no data". `file 2` here confirms it directly: the wide camera's own extrinsic
is exactly identity, to every digit.

## Reading it

- `extrinsic_matrix_columns` is four 3-vectors: three rotation **columns** then
  the translation. Written as named columns rather than a nested matrix because
  `matrix_float3x3` is column-major and a bare array of arrays gives a reader no
  way to tell which convention was meant.
- **Translation is in millimetres.** Everything else in this repo is metres.
- The pose is the lens relative to the reference camera. Rotation is
  orthonormal to 8e-08 with determinant +1, so it can be used as-is.
- `reference_dimensions` is the frame the intrinsics belong to. Scaling `fx, fy,
  cx, cy` to the actual image size is only valid **while the active format is a
  resample of this one**, and on this device it usually is not — see *These are
  the factory format's numbers* below before using them this way.
- `geometric_distortion_correction: false` means these are the **raw** lenses,
  and the distortion tables apply. With Apple's correction on, iOS withholds
  calibration entirely — there would be nothing here to read.

## What it is good for

The LiDAR depth camera reports `fov (h) 74.6°`, read off Diagnostics on this
handset — close to the wide camera and nothing like the ultra-wide, which is
what confirms it is the wide camera plus a scanner and that its depth is in the
wide camera's frame. The reprojection in `eval/pi3_poseless.py` rests on that,
and it is now measured rather than assumed: a multi-cam session reports the
depth camera's extrinsic as **exactly identity**, every digit, which is the same
claim this file argues for from the probe.

Read that figure as identifying *which* camera, not as its field of view. The
active format measures 70.38° and the three numbers do not agree; see above.

ARKit's `sceneDepth` arrives in the **wide camera's** frame — that is the
assumption `export_3dgs.py` already runs on when it scales the depth intrinsics
by the image size, and it works. Since the wide camera is also the reference
here, the ultra-wide's extrinsic **is** the LiDAR-depth-to-ultra-wide transform.
Nothing needs composing.

At 2 m the 19.3 mm baseline displaces a point by 19.3 mm and the 0.46 degrees of
rotation by 16.1 mm. Neither is negligible against the millimetre-scale
agreement a renderer wants, so both belong in any reprojection.

## The direction is checked: apply `R x + t`

A point in the ultra-wide's frame goes to the wide's by `R x + t`, not by the
inverse. That was the last thing here taken on documentation rather than
measurement, and it did not need a new capture — the probe shoots both lenses at
once, which is a calibrated stereo pair with a 19.272 mm baseline.

Take a textured patch in the ultra-wide, find it in the wide by correlation, and
compare the shift with what each convention predicts. The two predict shifts in
**opposite directions**, so one patch decides it:

```
  pair       measured shift        R x + t at 2 m     inverse at 2 m
  63913195   (-26, +2) ncc 0.955   (-26.4, -0.0)      (+56.0, +31.9)
  9745df4d   (-16, -34) ncc 0.972  (-26.1, -0.2)      (+55.7, +30.7)
  c8377926   ( -8, -14) ncc 0.990  (-26.0, +0.1)      (+58.6, +30.8)
  7a74eb30   (+20, -54) ncc 0.993  (-26.8, -0.7)      (+57.0, +40.7)
```

Shifts are relative to where an infinitely distant point lands, so they are the
parallax alone. `R x + t` is nearer the measurement in all four; the inverse has
the wrong sign in both axes. The first pair agrees to **0.4 px**, which also
says that patch really is near 2 m and that the 0.462° rotation belongs in the
transform — drop it and the agreement goes.

The other three disagree in magnitude because their patches are not at 2 m; only
the sign is being read there, and the sign is unambiguous.

## What the probe could not check, and what the multi-cam capture settled

`AVCaptureSession` cannot coexist with the `ARSession` the recorder is built on,
so a probe capture carries no depth and no poses. Projecting actual LiDAR points
into an ultra-wide frame therefore had to wait for a multi-cam capture.

Those captures now exist (2026-08-18, `20260818-1219xx` and `-160918-2d9844`),
and they settle two of the three things this file could only argue for:

- **The depth camera is the extrinsic reference.** Reported as exactly identity
  on a real session, which is what "zeros mean this is the reference" predicted.
- **The intrinsics do not have to be inferred at all.** `AVDepthData` carries
  them per frame for the format actually running. The recorder was discarding
  that and now writes it to `manifest.json`.

Still open: the parallax between the two lenses does not come out where the
19.272 mm baseline says it should. Fitting the depth-dependent term across eight
pairs gives slopes scattered from 0.10x to 1.49x the prediction, far wider than
each fit's own standard error, and no assumed depth field of view, edge
rejection or radius restriction moves it. The direction check above still
stands — that was four patches on a stereo pair and it is unaffected — but the
*magnitude* of the baseline has not been recovered from a real capture, and
until it is, anything that leans on 19.272 mm quantitatively rather than
directionally is unverified.

## Provenance

Captured 2026-08-15 on iPhone17,1, iOS 26.6, four shots per lens, focus locked,
`constituent delivery supported=true, eligible lenses=2`, content-aware
distortion correction off. These values are this handset's, not the model's: the
19.2 mm figure published for iPhone Pro spatial video is the nominal spacing and
this unit measures 19.272 mm.
