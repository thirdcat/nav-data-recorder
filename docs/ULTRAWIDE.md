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

Measured on an iPhone 16 Pro (iOS 26.5), the device behaves exactly as
documented:

| GDC | image | calibration |
| --- | --- | --- |
| **off** (probe default) | raw lens | **delivered** — tables, intrinsics, extrinsics |
| **on** | corrected in the camera pipeline | withheld entirely |

So GDC off is the working mode, and it is also the one worth having: it puts the
correction under our control and hands over the intrinsics with it. The toggle
stays so the same scene can be shot both ways and compared.

`tools/rectify_ultrawide.py` handles a capture with no tables too — a frame that
arrives already rectified is a pinhole, and the reprojection onto the requested
field of view is unchanged. That path exists for the GDC-on captures, which
carry no intrinsics and so cannot actually be used, and for any future device
that ships one without the other.

## What the lens actually measures

From a real GDC-off capture, run through the rectifier with no images needed:

```
source 4032x3024  fx=fy 1623.4
  rectified pinhole FOV   H 102.3   V 85.9   diagonal 114.4
output 848x480 @ 96.3     f 379.8   needs diagonal 104.1
  coverage: 0 of 407040 px outside   margin +10.3 deg
  distortion: largest correction 29.6 source px (within the exported field)
  round trip (forward o inverse): 0.213 px
```

**The lens covers the target with room to spare.** 96.3° at 16:9 demands 104.1°
diagonal and the ultra-wide's rectified pinhole is 114.4°, so nothing falls off
the edge. Headroom runs out around 100°:

| requested H | V | pixels with no source |
| --- | --- | --- |
| 96.3 (target) | 64.6 | 0.00% |
| 100.0 | 68.0 | 0.00% |
| 102.0 | 69.9 | 1.02% |
| 106.0 | 73.8 | 7.99% |

Note that 102.3° is the *pinhole* field of view, not the 106.2° the device
advertises for the format — the two differ because the raw lens is not a
pinhole, which is the entire reason for the tables.

The round trip is the load-bearing number. Apple ships both tables, and putting
a point through the inverse and back through the forward returns it to **0.213
px** — with only 42 table entries spread over a 2527 px radius, that is 62 px
between samples and the residual is interpolation error, not a misread
convention. The reading convention is therefore confirmed on real data rather
than only on the synthetic lens in the self-test.

Two things worth knowing about the real table that the synthetic one did not
show. It is **not monotonic** — magnification rises to 0.0082 by index 15, dips
back to 0.0021 near index 27, then climbs steeply to 0.0847 at the rim, the
signature of a mustache profile rather than simple barrel. And the correction is
modest where it matters: **29.6 px** across the 96.3° export, against 204 px at
the full frame's corner. Most of this lens's distortion lives in the outer ring
that a 96.3° crop discards anyway.

## What is being tested

Three things, in order of how likely they are to kill the idea:

1. **Does the calibration arrive at all?** Answered: yes, with GDC off. If a
   device shows `calibration delivery UNSUPPORTED` in both toggle positions,
   the rest is moot on it.
2. **Does the lens cover 96.3° at 16:9?** Answered: yes, with 10.3° of diagonal
   margin. The horizontal figure understates the demand — an 848×480 pinhole at
   96.3° reaches **104.1° across the diagonal**, and corners are where a source
   frame runs out. The rectifier computes this and reports the margin.
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
