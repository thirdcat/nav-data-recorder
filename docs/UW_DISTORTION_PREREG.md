# Pre-registration: measuring the ultra-wide's distortion for the active format

Written **before** `tools/fit_uw_distortion.py` existed. Everything below —
model, thresholds, gates, decision rule — is fixed here so that the answer
cannot be tuned into existence afterwards. Results go in the section at the
bottom, which was empty when this file was committed to the working tree.

## Why a new instrument

`docs/3DGS.md` measures that adding the raw ultra-wide arm to a training set
costs the *same wide photographs* 0.7 dB (wide-only 17.65, wide subset of the
mixed run 16.94). Two fixes to the ultra-wide camera model have been tried and
both failed, both for the same recorded reason:

```
  rectify with the factory lookup at 102.5 deg     15.86 dB   vs 16.76 raw
  fit the factory lookup to OPENCV                 12.58 dB   vs 12.79 control
  same, reversed table direction                   12.79 dB   vs 12.79 control
```

The reason is in `calib/README.md` and `docs/POSE.md`: **the factory table is
measured at 4032x3024 / 102.5 deg and the capture ran an active 3840x2160 /
106.2007 deg format.** Different formats read out different parts of this
sensor, so the factory table does not describe the recorded pixels. The same
trap already cost the wide lens 3.1 degrees.

So the factory table is not evidence about the active format, in either
direction, and no part of this measurement may use it. The instrument here is
plumb-line calibration: a straight edge in the world must image as a straight
line under a correct camera model, and the residual curvature is the
distortion. It reads the recorded pixels and nothing else.

### What the factory table would say, for contrast only

Computed from `calib/`, quoted so the measured answer can be compared against
it, **not** used as an input to the fit:

```
  lens        frame        active fov    corner displacement predicted by the
                                         factory table if it applied
  wide        1920x1440     69.52744      +37.5 px  (naive full-frame resample: +45.3)
  wide         640x480      69.52744      +12.5 px  (naive full-frame resample: +15.1)
  ultra-wide  3840x2160    106.2007       -99.6 px  (lens_distortion_lookup_table)
                                         +156.1 px  (inverse table)
```

The +45.3 px is the number the eyeball check refers to: a 1920x1440 ARKit frame
whose floor planks run straight across it with no visible bow cannot be
carrying a 45 px corner correction.

## The model being fitted

Rectification (raw pixel -> straightened pixel), radial only, about the frame
centre:

```
  c    = (w/2, h/2)
  R    = hypot(w, h) / 2                     half-diagonal, so rho = 1 at the corner
  rho  = |p - c| / R
  p_rect = c + (p - c) * (1 + a1*rho^2 + a2*rho^4)
```

Reported quantities:

- `a1`, `a2` — dimensionless, and resolution-independent for a fixed format.
- `D_corner = R * (a1 + a2)` — signed pixel displacement at the frame corner.
  This is the headline number.
- `D_rms` — RMS of `R*rho*(a1*rho^2 + a2*rho^4)` over the frame. PSNR is a
  whole-image average, so this is the honest quantity to put on the dB scale.
- `k1_opencv = -a1 * (R/f)^2` to first order, so the result can be handed to an
  `OPENCV` camera model later. Reported, not fitted.

Decisions fixed in advance:

- **The distortion centre is the frame centre.** The factory
  `lens_distortion_center` sits within 6 px of the factory frame centre at
  4032x3024, i.e. 0.24 % of the half-diagonal. Fitting a centre adds two
  parameters that the plumb-line data constrains far more weakly than `a1`.
- **`a1` alone is the primary fit.** `a1+a2` is reported as a secondary fit. If
  the two disagree on `D_corner` by more than 30 %, the radial coverage is not
  supporting the fourth-order term and only `a1` is quoted.
- **Focal length is not identifiable from straightness** and is not fitted. A
  uniform scale maps straight lines to straight lines. This is why the
  objective below is a *ratio*.

## The objective

Per edge chain `i`: rectify its points, fit a total-least-squares line, take
the RMS perpendicular residual `e_i`, divide by the rectified chord length
`L_i`. The dimensionless bow `s_i = e_i / L_i` is invariant to a uniform
rescale of the image, which removes the degeneracy where a large negative `a1`
shrinks every residual toward zero by shrinking the image.

Global cost is a soft-L1 (Huber, delta = the median `s_i` at `a=0`) sum of
`s_i`, so a handful of genuinely curved objects — a chair back, a cable, a
bag — cannot drag the fit. Minimised by coarse grid then Nelder-Mead.

Hard constraint: the radial map must stay monotone out to `rho = 1.05`, i.e.
`1 + 3*a1*rho^2 + 5*a2*rho^4 > 0`. A fold-over is rejected as infeasible.

## Chain extraction, frozen before the ultra-wide is touched

A line-segment detector is the wrong instrument here: it *assumes* straightness
and would chop a bowed line into straight pieces, deleting the signal. So:
Canny edges, 8-connected chain tracing broken at junction pixels, subpixel
refinement along the gradient normal by 3-point parabola, then splitting only
at **corners** (local turn > 30 deg measured over a +/-5-point window) — a
smooth distortion bow does not trip that.

Acceptance per chain: chord >= 0.12 x image diagonal, >= 40 points, and the
deviation from straight must be explained by a smooth quadratic in arclength
(R^2 >= 0.5) or already be under 0.2 px. Chains bowed by more than sag/chord =
0.15 are dropped as obviously not straight in the world.

**These parameters are frozen by the two wide-arm controls below and then
applied to the ultra-wide unchanged.** If the ultra-wide answer needs a
different detector setting, that is a failure to report, not a knob to turn.

## Acceptance criteria

### 1. Known-answer control, run first

Take wide-arm frames, warp them with a rectification coefficient I choose, and
demand the fitter return it. The warp is defined as the *inverse* of the model
above, so the injected image has an exactly known correct `a1` — no
polynomial-inversion approximation sits between the truth and the answer.

Injected values: `a1_inj` in `{-0.10, -0.05, +0.05, +0.10}`, run on the
640x480 multicam wide arm and on a 1920x1440 ARKit wide session.

**Pass:** for every injection, the recovered corner displacement, after
subtracting the null baseline measured in criterion 2 on the same frames,
matches the injected corner displacement within `max(1.0 px, 10 % of
|D_injected|)`.

A fitter that cannot return an injected coefficient says nothing about a
measured one, so a failure here stops the whole exercise.

### 2. Null control

Run the fitter unmodified on the wide arm, both resolutions. A 1920x1440 ARKit
frame was checked by eye: floor planks run straight across it with no visible
bow, and the factory table would predict about 45 px of corner displacement if
it applied.

**Pass:** `|D_corner|` below 25 % of the factory prediction at that resolution
— under 9.4 px at 1920x1440 and under 3.1 px at 640x480.

**Fail:** a large coefficient means the fitter is measuring its own line
detector, not the lens, and the ultra-wide number is void.

### 3. Data sufficiency, checked before any coefficient is quoted

Blurry, dim indoor frames may simply not contain enough long straight edges. If
so the honest output is "cannot measure", not a coefficient fitted to six short
segments. Required, per lens/session being fitted:

```
  frames contributing at least one chain      >= 30
  accepted chains, total                      >= 300
  accepted chains with mean rho >= 0.6        >= 80
  median accepted chord                       >= 0.12 x image diagonal
```

Any one of these unmet -> report the shortfall and stop for that arm.

### 4. Stability

- Bootstrap over frames, 200 resamples. The 90 % interval half-width on
  `D_corner` must be under 25 % of `|D_corner|`.
- The four multicam sessions (`20260820-211848-d06152`,
  `20260819-173935-0d6306`, `20260818-160005-57f29e`,
  `20260818-121905-d0f44f`) must agree: spread of `D_corner` across sessions
  under 30 % of the mean.

Either one violated -> the number is reported as unstable and no conclusion is
drawn from it.

### 5. Residual straightness, before and after

Report the median `s_i` (bow / chord) and the median absolute sagitta in pixels
over accepted chains, at `a = 0` and at the fitted `a`. A fit that does not
reduce the median bow is not measuring distortion whatever its coefficient
says.

## The decision rule, written before the fit runs

`docs/3DGS.md` anchors the scale: three horizontal centimetres at 2 m through
f = 653 px is ten pixels of texture displacement, and that is worth 1.6 dB.
Read linearly in dB, **0.7 dB is 4.4 px of texture displacement.**

The mixed run trained at `--downscale 4`, so the ultra-wide images were
960x540 and a native-resolution displacement must be divided by 4 before it is
put on that scale.

```
  D_rms at 960x540 >= 4.4 px   -> the un-modelled distortion is large enough to
                                  account for 0.7 dB
  D_rms at 960x540 <= 1.0 px   -> it is not; the mis-posed hypothesis survives
                                  and this one is refuted
  between                      -> inconclusive on magnitude
```

**Stated limitation, in advance.** The 0.7 dB was lost by the *wide* holdouts,
so a large ultra-wide displacement is a sufficient-magnitude argument, not a
demonstration of the mechanism. The mechanism requires the ultra-wide's rays
to corrupt shared Gaussians that the wide views then render, and confirming
that needs a training run, which is out of scope here. This measurement can
refute the camera-model hypothesis on magnitude, or fail to refute it. It
cannot by itself prove it.

## Amendments, all made before the ultra-wide was fitted

Recorded rather than quietly applied. None of these was chosen by looking at
an ultra-wide coefficient.

1. **Canny thresholds come from the gradient distribution, not the median
   intensity.** The median-intensity auto-Canny first written kept 10k edge
   pixels out of 8.3M on an ultra-wide frame and 1.7 usable chains per frame at
   1920x1440. The high threshold is now the 97th percentile of the Sobel
   magnitude, low is 0.4 of that, after CLAHE. The percentile was chosen on
   wide-arm frames. What the ultra-wide contributed to this decision was the
   edge-pixel *count*, not any fitted value.

2. **Chains trace through junctions instead of being cut at them.** Deleting
   junction pixels severs a floor-plank seam wherever a chair leg crosses it.
   The walk now prefers the neighbour that best continues the tangent, and
   `_split_at_corners` cuts afterwards on the 30-degree geometric test.

3. **The already-straight escape in the smoothness test is 1.0 px, not 0.2.**
   At 0.2 px a chain carrying half a pixel of pure localisation noise fails the
   quadratic test and is dropped, which removes *straight* chains
   preferentially and biases the fitted coefficient upward in magnitude. The
   escape has to sit above the noise floor.

4. **The objective is a raw-pixel residual divided by the map's local
   Jacobian, not the rectified residual divided by the chord.** This was found
   by the null control, which is what it is for. Both simpler forms are
   exploitable:

   - `rectified residual / chord` — a chain pointing at the image centre
     carries no distortion information, but a larger `a1` stretches it
     lengthwise, so the ratio falls for free and the fit is dragged positive.
     On the 640x480 null this returned **+9.5 px while making the median
     residual slightly worse**, which is the signature of a cost being paid in
     the wrong currency.
   - a raw sum of rectified residuals — minimised by shrinking the outer
     field, so it is dragged negative.

   Dividing the rectified residual by `|J n|` — the magnification of the
   radial map along the fitted line's normal — puts the residual back in raw
   image pixels, where the edge-localisation noise is isotropic and constant.
   A uniform rescale multiplies residual and Jacobian alike, so the objective
   is exactly gauge-invariant. Verified on synthetic geometry: exact recovery
   at zero noise, and a *flat* cost (no pull in either direction) when every
   chain is radial.

5. **An identifiability band is reported alongside the bootstrap.** A bootstrap
   over frames cannot see an unidentifiable direction — if the cost is flat in
   `a1` for every frame, every resample agrees on the same arbitrary answer and
   the interval looks tight. So each fit also reports the span of `a1` over
   which the cost stays within 2 % of its minimum, and the cost reduction from
   `a1 = 0` to the fit. A wide band or a negligible cost reduction means the
   data does not contain the answer, whatever the point estimate says.

## Results

**Headline: the instrument passes every control, and then fails to measure the
ultra-wide. No coefficient is quoted for that lens.** The wide arm comes back
null, which refutes the factory table on the lens where it can be checked. The
ultra-wide's own frames do not contain long enough straight edges to resolve
the quantity at the size that would matter.

### 1. Known-answer control — PASS, 8 of 8

Recovered corner displacement against the exact composition of each frame's own
measured error with the injection:

```
  frames                    injected a1    got        expect      err      tol
  wide 640x480  d06152         -0.10     -41.91 px   -41.10 px   -0.80    4.11
                               -0.05     -21.98      -21.25      -0.73    2.12
                               +0.05     +16.94      +18.42      -1.48    1.84
                               +0.10     +36.04      +38.24      -2.20    3.82
  wide 1920x1440 cb4586        -0.10    -112.20     -107.92      -4.28   10.79
                               -0.05     -47.59      -46.33      -1.26    4.63
                               +0.05     +75.71      +77.27      -1.56    7.73
                               +0.10    +140.05     +139.30      +0.75   13.93
```

Every one inside the pre-registered `max(1.0 px, 10 %)`. On pure synthetic
geometry the same estimator is exact to 0.00001 in `a1` at zero noise and to
0.3 px of corner displacement at 0.3 px of point noise.

The controls also produced the discriminator that the rest of this rests on.
Where a real distortion is present, correcting it **drops the cost by 3–24 %**
and visibly reduces the median straightness residual. Where none is present,
the cost drops by 0.1–0.3 % and the residual does not move.

### 2. Null control — PASS

```
  frames                          a1        corner      identifiability band   cost drop
  wide 640x480   d06152        -0.00352    -1.41 px      [ -9.6, +6.0]           0.1 %
  wide 1920x1440 pooled (3)    +0.00274    +3.29 px      [-14.4, +21.6]          0.1 %
    cb4586                     +0.01283   +15.40 px      [-24.0, +57.6]          0.3 %
    1868dd                     -0.00019    -0.23 px      [-13.2, +10.8]          0.0 %
    e11854                     +0.00484    +5.80 px      [-18.0, +28.8]          0.2 %
```

Against the pre-registered thresholds — under 3.1 px at 640x480 and under
9.4 px at 1920x1440 — the 640x480 arm and the pooled 1920x1440 estimate both
pass. Of the three sessions taken singly, `1868dd` and `e11854` pass and
`cb4586` returns +15.4 px, above the line. Its identifiability band is the
widest of the three, which is the reading the pre-registration asked for.

**This is a positive result, not just a clean control.** The factory table
predicts +12.5 px at 640x480 and +37.5 px (naively resampled, +45.3) at
1920x1440. The measurement returns a few pixels with bands that exclude those
figures. `docs/POSE.md` and `calib/README.md` argued from a similarity fit and
from the device's own logged field of view that the factory calibration does
not describe the active format; this is a third instrument, sharing no input
with either, agreeing.

Stability under the chord requirement, which is what a real measurement looks
like:

```
  wide 640x480 d06152     chord >= 0.12    0.20    0.30 of the diagonal
                             -1.4 px    -1.5    -1.7      band [-8.8, +4.8]
```

### 3. Ultra-wide — FAILS the pre-registered gates. No coefficient.

```
  session      chains  frames    a1       corner      band              cost   median residual
                                                                       drop   before -> after
  d06152         731     60   -0.05364  -118.2 px  [-282.0,  +48.5]     1.0 %  1.465 -> 1.498
  0d6306          61     17   (gated out: 17 frames, 61 chains, 17 outer)
  57f29e        1031     45   +0.00070    +1.5 px  [ -59.5,  +70.5]     0.0 %  1.336 -> 1.342
  d0f44f         781     57   -0.00326    -7.2 px  [-105.7, +101.3]     0.0 %  1.441 -> 1.434
  pooled        2281    152   -0.01310   -28.9 px  [-141.0,  +83.7]     0.1 %  1.436 -> 1.439
  pooled a1+a2  2281    152             -29.3 px  [-140.9,  +81.6]     0.2 %  1.436 -> 1.442
```

Three separate pre-registered gates refuse this:

- **Criterion 4, agreement.** The spread across sessions had to be under 30 %
  of the mean. It is about four times the mean. These are the *same lens* at
  the *same pinned 3840x2160 / 106.2007 deg format*, so one lens would have to
  have three different distortions. At least two of the three are scene
  artefacts, which is the proof that the instrument is not reading the lens.
- **Criterion 5, residual straightness.** No ultra-wide session's fit reduces
  the median residual; two make it slightly worse. Compare 3–24 % cost drops
  when a real distortion is injected.
- **Identifiability.** Bands run from +/-20 px on the best-behaved session to
  +/-280 px on the worst. The bootstrap intervals are much tighter than the
  bands, which is exactly the failure mode amendment 5 was written for: a
  bootstrap over frames cannot see a direction that is flat in every frame.

The fourth-order fit changes the corner value by 1.5 %, so `a2` is not adding
anything and nothing is hiding in it.

Sensitivity to the chord requirement, declared post hoc and reported as a
diagnostic rather than a result:

```
  chord >=            0.12       0.20       0.30 of the diagonal
  UW d06152        -118.2 px  -134.7 px  -202.6 px   band [-436.2,  +2.2]
  UW 57f29e          +1.5       -0.4       -1.5      band [ -19.8, +17.6]
  UW d0f44f          -7.2       -2.9       +9.1      band [ -26.4, +44.1]
```

Two sessions converge on zero as the chains get longer; `d06152` walks further
away. Requiring longer chains does not reconcile them.

### Why it fails, in numbers

Not "the fitter is bad" — the controls say it is not. The scenes do not carry
the signal. An ultra-wide chain of the median length these rooms yield is 0.16
of the frame diagonal, and at the 960-px working scale a *tangential* chain at
`rho = 0.55` bows by:

```
  chord / diagonal     a1 = -0.02   -0.05   -0.10       (RMS bow, working px)
      0.12               0.03       0.07    0.13
      0.16 (the median)  0.05       0.12    0.23
      0.20               0.07       0.18    0.36
      0.30               0.16       0.41    0.82
      0.40               0.29       0.73    1.46
```

The measured straightness noise floor on ultra-wide chains is **1.44 working
pixels** — against 0.31 px at 640x480 and 0.52 px at 1920x1440 on the wide arm.
So at the length these frames actually deliver, a corner displacement of 110 px
hides twelve times under the per-chain noise, and only ensemble averaging over
~750 chains brings it anywhere near visibility. That is the whole story of the
wide bands.

Confirmed twice more, independently of the image noise:

- **Response slope.** Injecting known coefficients into ultra-wide *images* and
  regressing recovered on injected gives a slope of **0.745**, not 1 — a 26 %
  attenuation. The wide arm's slope is ~1 (that is criterion 1).
- **Chain geometry alone.** Injecting the same coefficients into the extracted
  ultra-wide *chain coordinates*, with no image processing involved at all,
  reproduces the same 0.74 slope. On the 640x480 wide chains the same probe
  returns 1.04 and 0.89. So the attenuation is a property of the chain
  population — their length and their own residual curvature — not of blur,
  resampling or the edge detector.

To resolve this lens with this method you would need chains of about 0.4 of the
frame diagonal, i.e. straight edges roughly 1760 px long at 3840x2160, in
quantity. At `chord >= 0.30` these sessions yield 55–64 chains apiece; 0.40
leaves too few to fit. Handheld, motion-blurred passes through cluttered rooms
do not contain them.

### 4. The comparison that matters — cannot be made

`docs/3DGS.md`: three horizontal centimetres at 2 m through f = 653 px is ten
pixels of texture displacement, worth 1.6 dB. Read linearly, 0.7 dB is 4.4 px.
The ultra-wide trained at `--downscale 4`, i.e. 960x540, where its focal length
is 360 px. So the threshold converts two ways:

```
  reading                                     corner displacement at 3840x2160
                                              that would produce it
  4.4 px compared as plain pixels at 960x540          54 px
  4.4 px at f = 653 rescaled to f = 360 (2.4 px)      30 px
```

**So the question was always "is the ultra-wide's corner displacement above
about 30 to 54 px?" and the instrument's resolution on that lens is +/-20 px at
best and +/-280 px at worst.** On `57f29e` and `d0f44f` the answer would be no;
on `d06152` — the session the 0.7 dB was actually measured on — the band spans
the threshold several times over. The pre-registered decision rule therefore
returns **inconclusive**, and it returns it for the honest reason: the
measurement is not precise enough, not because the answer sits in the middle.

Neither hypothesis is settled. The camera-model explanation is not supported
and not refuted, and the mis-posed explanation is untouched by this work.

### What is safe to carry forward

- The plumb-line instrument is calibrated and works, at 640x480 and 1920x1440,
  to within a few pixels of corner displacement. `tools/fit_uw_distortion.py`
  and its self-test are the record.
- **The wide lens's factory distortion table does not describe its active
  format**, measured directly from pixels for the first time: a few px against
  the table's +12.5 / +37.5–45.3 px. Third independent confirmation of the
  `calib/README.md` finding.
- **The ultra-wide's distortion at 3840x2160 / 106.2007 deg remains unmeasured.**
  Anyone attempting it again needs frames built for it — a scene with long
  straight edges spanning at least 40 % of the frame diagonal, held still
  enough that the straightness noise floor drops below a pixel. A calibration
  target, or a deliberate slow pass along a corridor, would do it. Another
  handheld pass through a cluttered room will not.
- Do not read the `-118 px` on `d06152` as a measurement. It is stable under
  the chord threshold, which makes it a property of that room's geometry rather
  than of noise, but the same lens returns `+1.5` and `-7.2` on two other
  sessions at the identical format, and no lens has three distortions.
