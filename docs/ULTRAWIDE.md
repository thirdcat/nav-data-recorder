# Ultra-wide + rectification — a validation, not a capture path

The ultra-wide is the only lens on the phone that reaches the OAK-1-W's 96.3°.
It is also wide enough that a pinhole model stops describing it, which matters
here more than it would for most uses: a VLN policy learns that a position in
the image means a bearing, and on a 106° lens it does not.

Apple ships the correction — `AVCameraCalibrationData` carries a radial
magnification lookup table — so the question is answerable rather than
theoretical. **This page is how to answer it.** Nothing here is on the recording
path, and running it produces no poses.

Read [DATA_FORMAT.md](DATA_FORMAT.md) *Why not the 0.5x ultra-wide?* first for
why this is a validation rather than a plan: an `AVCaptureSession` cannot coexist
with an `ARSession`, so a working rectification is a necessary step towards
ultra-wide capture but nowhere near a sufficient one.

## Two modes, and why the toggle exists

Apple already rectifies the ultra-wide. `AVCaptureDevice`'s
`geometricDistortionCorrectionEnabled` is **on by default** for that lens, and
when it is on the frames arrive corrected — but the calibration data is
withheld, because there is nothing left to describe. `AVCapturePhotoOutput.h`
spells the conditions out:

> Camera calibration data delivery (intrinsics, extrinsics, lens distortion
> characteristics, etc.) is only supported if
> `virtualDeviceConstituentPhotoDeliveryEnabled` is YES and
> `contentAwareDistortionCorrectionEnabled` is NO and the source device's
> `geometricDistortionCorrectionEnabled` property is set to NO.

**Measured, this device does the opposite.** On an iPhone 16 Pro running
iOS 26.5, `isCameraCalibrationDataDeliverySupported` is true only with GDC
**on** — with GDC off, the mode the header says is required, it reports false.
That is why the probe has a toggle and prints the flag: the header is a claim,
the log is a measurement, and where they disagree the device wins.

| GDC | image | calibration on this device |
| --- | --- | --- |
| **on** (probe default) | corrected in the camera pipeline | **delivered** |
| **off** | raw lens | not delivered |

The mode that works is also the easier one to consume. A GDC-on frame is already
a pinhole, so the distortion tables have nothing to say and Apple ships none;
what arrives is `intrinsicMatrix`, which is the part that actually matters —
without it there is no way to know what bearing a pixel means. Rectification
then collapses to a reprojection from one pinhole onto another, and
`tools/rectify_ultrawide.py` handles both cases from the same code path.

If a future device flips back to matching the documentation, the toggle is
already there and the tool already reads the tables.

## What is being tested

Three things, in order of how likely they are to kill the idea:

1. **Does the calibration arrive at all?** Answered: yes, with GDC on. If a
   device shows `calibration delivery UNSUPPORTED` in both toggle positions,
   the rest is moot on it.
2. **Does the lens cover 96.3° at 16:9?** The horizontal figure understates the
   demand — an 848×480 pinhole at 96.3° reaches **104.8° across the diagonal**,
   and corners are where a source frame runs out. The rectifier computes this
   and reports the margin.
3. **Is the corrected image actually straight?** This is the one arithmetic
   cannot answer. Straight edges in the world are the ground truth.

## Capturing

Settings → **Ultra-wide probe**.

Shoot a scene with **long straight edges** — a doorframe, a skirting board, the
wall/ceiling join. A scene with nothing straight in it cannot answer question 3,
and the tool will say so rather than invent a number. Stand back about two
metres so the extreme edges of the frame have structure in them, and move
between shots. Six to ten shots is plenty.

The probe locks focus before the first shot, so every frame shares one focal
length; a rectification checked against a lens that refocused mid-run is a
rectification checked against two lenses.

`calibration.json` is rewritten after every shot, so the directory is complete
and rectifiable at all times — leaving the screen, or a crash, costs nothing but
the shots you had not taken yet.

Watch the first lines of the log. They report the device, the constituents, and
whether calibration delivery is supported. **`calibration delivery UNSUPPORTED`
means stop** — the run cannot answer anything, and the JSON will say
`"calibration": "unavailable"` rather than pretend otherwise.

Output lands in `Documents/uw_probe/<id>/`, one directory per lens:

```
uw_probe/20260807-141233-4b91cc/
  UltraWideCamera/
    0000.jpg  0001.jpg  …
    calibration.json
  WideAngleCamera/
    …
```

Both lenses come back because the calibration only ships with constituent
delivery, and asking for constituents means asking for more than one. The wide
frames are a free control: same instant, same scene, a lens that *is* nearly a
pinhole.

Pull the directory off with Files.app, the same way as a session.

### `calibration.json`

```json
{
  "reference_dimensions": [4224, 2376],
  "intrinsics": { "fx": 2093.4, "fy": 2093.4, "cx": 2112.0, "cy": 1188.0 },
  "lens_distortion_center": [2112.0, 1188.0],
  "lens_distortion_lookup_table": [0.0, 0.0002, …],
  "inverse_lens_distortion_lookup_table": [0.0, -0.0002, …],
  "pixel_size_mm": 0.0014,
  "extrinsic_matrix_columns": [[…], […], […], […]],
  "images": ["0000.jpg", …],
  "notes": ["device TripleCamera", "constituents UltraWideCamera + WideAngleCamera", …]
}
```

Intrinsics are written as named scalars rather than a matrix on purpose:
`matrix_float3x3` is column-major, and a JSON array of arrays gives a reader no
way to tell which convention was meant.

The two lookup tables are magnification factors at `n` linearly spaced radii,
from 0 to the distance between the distortion centre and the **farthest corner**
— not half the diagonal, because the centre is not the image centre.
`lens_distortion_lookup_table` maps distorted → rectified;
`inverse_lens_distortion_lookup_table` maps rectified → distorted, which is the
direction an image warp needs.

## Rectifying

```bash
python3 tools/rectify_ultrawide.py uw_probe/20260807-141233-4b91cc/UltraWideCamera \
    -o ./rectified --check-lines
```

```
source     4224x2376  fx 2093.4 fy 2093.4  H 106.2 V 69.5  diagonal 115.4
output     848x480  f 374.9  H 96.3 V 64.6  diagonal 104.8
coverage   every output pixel has source behind it (+10.6 degrees of diagonal margin)
distortion largest correction 217.7 source px
round trip 0.005 px (forward o inverse = identity)
```

What to look at:

- **`coverage`.** Any `!` here means the output has black corners and the
  requested field of view is not available. Lower `--hfov` until it clears.
- **`round trip`.** Apple ships both tables and they should compose to the
  identity. A residual above a pixel means one of them is being read wrong, and
  every image out of the tool is then suspect.
- **`distortion`.** If this were small, the ultra-wide would already be a
  pinhole and none of this would be needed. It is not small.
- **`--check-lines`.** Median bow of edge chains, before and after. This is the
  verdict. Bow is measured in bins along each chain rather than pointwise,
  because an edge has width and a contour traces both sides of it — a pointwise
  residual has a floor of half the edge width no matter what the geometry does.

Then **look at the images**. A number that says straight and an image that looks
bent means the measurement is wrong, not the lens.

## What the tooling is checked against

`tools/test_rectify_ultrawide.py` builds a lens with a known distortion
polynomial, expresses it as a pair of Apple-style tables, renders a grid of
world-straight lines through it, and rectifies:

```
  table round trip                     0.0049 px                          PASS
  distortion is non-trivial            217.7 px                           PASS
  coverage at 96.3 degrees             0 pixels outside                   PASS
  bearing 10 deg lands right           0.46 px                            PASS
  bearing 30 deg lands right           0.24 px                            PASS
  bearing 45 deg lands right           0.27 px                            PASS
  straight lines come back straight    1.84 -> 0.42 px                    PASS
  over-wide request reports gaps       70.4% outside                      PASS
```

The bearing checks are the ones that matter for VLN: they assert that a known
angle off the optical axis lands at the pixel the requested pinhole puts it at,
which is exactly the property the raw ultra-wide lacks and the whole point of
rectifying. The last case checks that asking for more than the lens has is
reported rather than silently cropped.

Note what this does *not* test: the numbers above come from a synthetic lens, so
they confirm the geometry and the table conventions, not that Apple's tables
describe the real lens well. Only imagery answers that — hence `--check-lines`.

## If it passes

A clean rectification is one of four things the ultra-wide path needs, and the
cheapest. The others are unchanged, and are the reason this is a probe:

- **Pose.** No ARKit means no 6-DoF pose. Offline RGB-D SLAM over the capture,
  seeded from nothing, against ARKit's IMU-coupled VIO.
- **Depth alignment.** LiDAR depth is registered to the wide camera's frame, so
  it would have to be reprojected through `extrinsic_matrix_columns`.
- **Low light.** Smaller aperture and sensor, indoors.
96.3° itself is not in doubt: the estimator recovers it to the decimal from the
official simulation episodes. The `IMG_108x` episodes measure 66.1°, but they
are an iPhone 16 Pro trial and a minority of the mix — this same phone, cropped
by video stabilisation — so they are a check on the estimator, not a target. See
*Known gaps* in [OUTPUT_FORMAT.md](OUTPUT_FORMAT.md).
