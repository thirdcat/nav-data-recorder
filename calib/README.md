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
- `reference_dimensions` is the frame the intrinsics belong to. Scale `fx, fy,
  cx, cy` by the actual image size before using them, the way
  `tools/export_3dgs.py` already scales ARKit's.
- `geometric_distortion_correction: false` means these are the **raw** lenses,
  and the distortion tables apply. With Apple's correction on, iOS withholds
  calibration entirely — there would be nothing here to read.

## What it is good for

ARKit's `sceneDepth` arrives in the **wide camera's** frame — that is the
assumption `export_3dgs.py` already runs on when it scales the depth intrinsics
by the image size, and it works. Since the wide camera is also the reference
here, the ultra-wide's extrinsic **is** the LiDAR-depth-to-ultra-wide transform.
Nothing needs composing.

At 2 m the 19.3 mm baseline displaces a point by 19.3 mm and the 0.46 degrees of
rotation by 16.1 mm. Neither is negligible against the millimetre-scale
agreement a renderer wants, so both belong in any reprojection.

## What is not yet checked

**The direction of the transform.** Apple documents `extrinsicMatrix` as the
camera's pose with respect to the reference, but whether to apply `R x + t` or
its inverse has not been verified against data — and `UltraWideProbe.swift`
warns in its own comments about exactly this class of silent transposition bug.
The check is cheap once there is a capture with depth: reproject the LiDAR
points into an ultra-wide frame and see whether they land on the image. Getting
it backwards misses by twice the baseline, which is unmistakable.

The probe cannot supply that check itself. `AVCaptureSession` cannot coexist
with the `ARSession` the recorder is built on, so a probe capture carries no
depth and no poses.

## Provenance

Captured 2026-08-15 on iPhone17,1, iOS 26.6, four shots per lens, focus locked,
`constituent delivery supported=true, eligible lenses=2`, content-aware
distortion correction off. These values are this handset's, not the model's: the
19.2 mm figure published for iPhone Pro spatial video is the nominal spacing and
this unit measures 19.272 mm.
