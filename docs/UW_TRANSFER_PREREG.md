# Pre-registration — transferring Apple's radial table to the active format by field angle

Written **before** the transfer was implemented or run against any session. Every
number in *What is fixed before the test* is a function of `calib/` and the
capture manifests only; no session pixel was read to produce it.

## The claim under test

Two attempts to give the raw ultra-wide a camera model failed, and both were
blamed on the factory calibration being measured at 4032x3024 / 102.5 deg while
the capture ran 3840x2160 / 106.2007 deg. That blame is only half right.

**Distortion is a property of the lens as a function of field angle.** Two
readouts of the same sensor are the same physical image plane sampled at
different pitches, so at equal field angle their raw pixel radii differ by
exactly one scalar

```
  k = pitch_calibrated / pitch_active
```

Transferring Apple's table to another format is therefore a **change of the
radius axis by k**, and nothing else. The magnification values themselves are
untouched. `rectify_3dgs_export.py --hfov 102.5` instead matched the *horizontal
extent*, which is the wrong invariant, and stretched the table by the wrong
factor.

## Definitions, fixed here

Convention for the two tables, as `tools/rectify_ultrawide.py` already uses them
and as the 0.005 px table round trip in `docs/ULTRAWIDE.md` confirms:

- `lens_distortion_lookup_table` maps **ideal (paraxial) -> raw**, indexed by the
  **ideal** radius.
- `inverse_lens_distortion_lookup_table` maps **raw -> ideal**, indexed by the
  **raw** radius.
- Both are indexed on `[0, r_max]` where `r_max` is the distance from the
  distortion centre to the farthest corner of the *calibration* frame.

With `fx` the factory intrinsic and `m_lens` the forward table, the raw radius
at field angle theta in calibration pixels is

```
  R_cal(theta) = fx * tan(theta) * (1 + m_lens(fx * tan(theta)))
```

and the transfer is

```
  k                 = (W_active / 2) / R_cal(hfov_active / 2)
  f_paraxial_active = fx * k                       [pixels of the active frame]
  m_active(r)       = m_cal(r / k)                 [either table]
  r_max_active      = k * r_max_cal
  centre_active     = image_centre_active + k * (centre_cal - image_centre_cal)
```

`k` is dimensionless and is exactly the pixel-pitch ratio, so the two routes the
brief names are the same route:
`pitch_active = pitch_cal / k` and `f_mm = fx * pitch_cal = f_paraxial_active * pitch_active`.

## The trap: which focal length is used where

`videoFieldOfView` — the 106.2007 in the manifest — is a **whole-frame quantity
that already contains the distortion** (`docs/POSE.md`). It is therefore a
**field angle at the frame's horizontal edge**, and

```
  f = (W/2) / tan(hfov/2)        is NOT a paraxial focal length
```

That expression evaluates to **1441.56 px** on the active ultra-wide format. It
is used **nowhere** in this work except as a labelled ablation (A1 below).
Correcting toward paraxial and then comparing against a whole-frame figure moved
away from the answer once already; the rule here is that `videoFieldOfView` only
ever enters as the *argument of a tangent inside `R_cal`*, never as a divisor.

The paraxial focal length of the active format is `fx * k`. Everything that
needs a focal length — `fl_x`/`fl_y` in an export, the ray direction in a
rectifier, the predicted lens-to-lens scale — uses that number and only that
number.

## What is fixed before the test

Computed from `calib/` and the manifests alone.

```
                                        ULTRA-WIDE          WIDE
  factory format                        4032x3024           4032x3024
  factory fx                            1616.6437           2745.3555
  factory pixel pitch                   1.4000 um           2.4400 um
  factory r_max                         2526.80 px          2536.14 px
  active format                         3840x2160           640x480
  active videoFieldOfView               106.2007 deg        69.52744 deg

  R_cal(hfov/2)                         2117.29 cal px      1962.73 cal px
  k                                     0.906820            0.163038
  f_paraxial_active                     1466.00 px          447.60 px
  active pixel pitch                    1.5439 um           14.9658 um
  active half-width, physical           2.9642 mm           4.7891 mm
     as a fraction of the calibrated    105.0 %             97.4 %
  active frame corner                   2202.9 active px    400.0 active px
     in calibration pixels              2429.3              2453.4
     as a fraction of r_max             96.1 %              96.7 %
```

Both active formats read essentially the same lens circle as the calibrated one:
the corner of each sits within 4 % of the calibrated maximum radius. That is the
observation this whole exercise rests on, and it holds for **both** lenses.

The ultra-wide table is a **mustache**, not a barrel: raw->ideal magnification
rises to +0.00819 at index 15 (924 cal px), falls back to +0.00209 at index 27
(1664 cal px), then climbs steeply to +0.08486 at the rim. The transfer places
the peak at 838 active px and the trough at 1509 active px, both inside the
radius range the annulus test samples. **A monotone radial polynomial cannot
reproduce that shape**, which is why the OPENCV fit in
`tools/add_ultrawide_distortion.py` is a separate question from this one.

## Ablations that must be reported alongside the result

- **A1 — paraxial reading.** Treat `videoFieldOfView` as paraxial:
  `k = 1441.56 / 1616.6437 = 0.891565`. This is the trap, run deliberately.
- **A2 — horizontal-extent match.** What `rectify_3dgs_export.py --hfov 102.5`
  does today.
- **A3 — same physical width.** `k = 3840/4032 = 0.952381`, i.e. the active
  format is a pure vertical crop of the calibrated one.
- **A4 — identity warp.** The same resampling with the magnification set to
  zero. This is the control for the resampling itself, not for the lens.

## Acceptance criteria

### 1. Round trip

Distort then undistort must return the original pixel to well under a pixel,
across the full radius **including the corner** of the active frame.

- **PASS** if the maximum error over a dense grid covering `0 <= r <= r_corner`
  is **< 0.01 px**, using a numerically inverted forward map.
- Also report the round trip through Apple's *shipped pair* of tables. That is
  expected to be near 0.2 px at calibration scale — 42 samples over a 2527 px
  radius is 62 px between knots — and is interpolation error, not a misread
  convention. It is reported, not used as the criterion.

### 2. Wide-arm null control

The same machinery applied to the WIDE lens must not contradict what is already
measured there. Recorded before the test:

```
  wide arm active format                        69.52744 deg, not the factory 72.58
  fit_uw_distortion.py corner displacement      -1.41 px at 640x480
                                                +3.29 px at 1920x1440
  what the untransferred factory table demands  +15.6 px and +46.8 px
```

- **PASS** if (a) the transferred model's predicted corner displacement for the
  wide arm lies inside the identifiability band that `tools/fit_uw_distortion.py`
  itself reports on that arm, and (b) applying the transferred model to the wide
  arm's edge chains does **not increase** the plumb-line straightness residual.
- **FAIL** if the straightness residual increases, or if the predicted corner
  displacement is further from the measured value than the untransferred factory
  table's own prediction is.
- If the identifiability band is wide enough to contain both the measurement and
  the factory prediction, the control is reported **UNINFORMATIVE** rather than
  passed, and the band is quoted.

### 3. The cheap decisive test — the annulus profile

`eval/uw_radial_profile.py` fits the ultra-wide-to-wide similarity in annuli and
finds a reproducible non-flat profile in ultra-wide pixels:

```
  annulus            d06152     57f29e
     0 -  300        0.3134     0.3134
   300 -  600        0.3171     0.3173
   600 -  900        0.3182     0.3179
   900 - 1250        0.3170     0.3165
  peak-to-trough      1.53 %     1.44 %
```

Its control — one frame against itself downscaled by 0.3200 — returns 0.3201 and
is flat.

If the transferred model describes the lens, undistorting the ultra-wide before
matching must **flatten** this profile.

- **PASS** if peak-to-trough falls to **<= 0.50 %** on **both** sessions (at
  least two thirds of the structure removed), *and* the resampling control A4
  leaves the profile within 0.20 % of the published one, *and* the
  self-downscale control still returns 0.3200 +/- 0.0010 and stays flat.
- **PARTIAL** if peak-to-trough falls by at least one third but stays above
  0.50 %.
- **FAIL** otherwise, including any increase.
- **VOID** if the A4 control moves the profile by more than 0.20 %, because then
  the resampling is doing the work rather than the lens model.
- Any ablation A1-A3 that flattens the profile as well as, or better than, the
  transfer means the test has not separated them, and the result is reported as
  **NOT SEPARATED** whatever the primary number is.

### 4. Corner displacement, against this page's anchor

Report, for the transferred model on the active ultra-wide format:

- the radial displacement at the frame corner, in native 3840x2160 pixels;
- the RMS displacement over the whole frame, which is what a PSNR sees;
- both expressed on `docs/3DGS.md`'s own anchor — 10 px of texture displacement
  at `f = 653 px` is 1.6 dB, and the unattributed penalty is 0.7 dB = 4.4 px.

This is a report, not a threshold. A displacement far larger than 4.4 px on that
anchor means the model is claiming more than the missing 0.7 dB, and that has to
be said out loud rather than treated as confirmation.

## Sanity check on `k`, declared before the test

The transfer implies a paraxial ratio `f_wide / f_ultrawide`. The innermost
annulus measures the secant scale near the axis; divided back through the model
it extrapolates to that ratio. Require agreement within **1 %**. If it
disagrees, report by how much and say which lens carries the disagreement,
rather than adjusting `k`.

`k` is never fitted to session data in this work. It comes from `calib/` and the
manifest, and it stays where it lands.

## What would refute the whole idea

- The profile does not flatten, or flattens no better than an ablation.
- The wide arm's straightness gets worse under the transfer.
- The round trip does not close, which would mean the table is being read in the
  wrong direction after all.

---

# Results

Run 2026-08-22 against the pre-registration above, in the order it is written.
No threshold was moved after a number was seen. CPU only; no GPU, no training.

```
python3 tools/uw_field_angle_transfer.py report
python3 tools/uw_field_angle_transfer.py roundtrip
python3 tools/uw_field_angle_transfer.py wide-null ~/nav_data/20260820-211848-d06152
python3 tools/uw_field_angle_transfer.py wide-null ~/nav_data/20260820-211951-87bc2c
python3 tools/uw_field_angle_transfer.py profile ~/nav_data/20260820-211848-d06152 \
    ~/nav_data/20260818-160005-57f29e --modes none,identity,field-angle,paraxial,width --control
python3 tools/uw_field_angle_transfer.py profile ... --modes field-angle --wide-too
```

## Which focal length went where

- `videoFieldOfView` 106.2007 entered exactly once, as `theta = hfov/2` inside
  `R_cal(theta)`. It was never divided into a half-width.
- `(W/2)/tan(hfov/2) = 1441.56 px` was computed only to run ablation A1 and to
  print it next to the answer.
- **The paraxial focal length of the active ultra-wide format is `fx * k` =
  1466.00 px**, equivalently a 1.5439 um effective pitch on a 2.26330 mm lens.
  That number is what the rectifier's output pinhole, the `fl_x/fl_y` the
  opt-in export flags write, and the predicted lens-to-lens scale all use.
- For the wide arm the same route gives 447.60 px at 640x480. Where a session
  hands over a paraxial focal length directly — an ARKit `pose.jsonl` `fx` —
  the tool uses it and skips the field-of-view reading entirely
  (`mode="focal"`): 87bc2c reports `fx = 1348.01` at 1920x1440.

## The reading of `videoFieldOfView`, checked against a second instrument

`docs/POSE.md` asserts the whole-frame reading; the device's own two 4:3 *wide*
formats now test it, because ARKit publishes a pinhole intrinsic matrix and
`AVCaptureDeviceFormat` publishes a field of view, and both describe the same
lens.

```
                                        paraxial hfov   whole-frame hfov
  ARKit 1920x1440, from fx = 1348.01       70.914             69.322
  multicam 640x480, videoFieldOfView            --            69.52744 (logged)

  read as whole-frame, the two formats differ by   -0.21 deg
  read as paraxial,    they differ by              +1.39 deg
```

The whole-frame reading makes them agree **6.7x better**. That is independent
of anything in this work and it supports the primary `k`.

## 1. Round trip — PASS

Over `0 <= r <= r_corner` and over the whole 3840x2160 frame. `r_corner` is
2209.07 active px, measured from the *distortion centre* rather than the frame
centre, which is why it is 6 px larger than the 2202.9 half-diagonal quoted
above:

```
                                   ultra-wide        wide
  radial round trip, max          1.47e-08 px      2.8e-13 px
  at the corner exactly                0.00 px          0.00 px
  planar round trip, max          6.79e-09 px      3.5e-12 px
  monotone (no fold-over)                 yes             yes
```

Well under the 0.01 px threshold, by seven orders of magnitude, because the
model carries one map and its own numerical inverse.

Apple's **shipped pair** of tables, reported as promised: max **0.574 px**,
**0.181 px** at the corner for the ultra-wide, 0.366 px for the wide. That is
the 62-calibration-pixel knot spacing, as expected, and it is not the criterion.

It has one consequence worth stating, because it is the model's own uncertainty
where the model matters most. Read the corner *forward and inverted*, the
raw->ideal displacement is **120.3 px**; read straight off the shipped inverse
table it is **139.5 px**. The two compose to 0.18 px yet differ by 16 % at the
rim, because that is where the magnification climbs 1.4 points per knot. Any
corner number from this calibration carries that ambiguity.

## 2. Wide-arm null control — the instrument has no power, and what power it has favours the factory table

```
  session   frame       chains   plumb-line a1 fit    identifiability band    cost reduction
  d06152    640x480      1063     corner  -0.58 px    [-10.4, +7.6] px          1.0e-04
  87bc2c    1920x1440    1130     corner  -0.15 px    [-15.6, +13.2] px         2.1e-06
```

**The published -1.41 px and +3.29 px are not measurements of near-zero
distortion.** They are the centre of a basin whose cost is flat to one part in
10,000 and one part in 500,000. The band is 18 and 29 pixels wide. An
instrument that cannot separate -10 px from +8 px has not excluded -12 px.

What the transfer predicts, and what the *untransferred* factory table predicts:

```
                              d06152 640x480    87bc2c 1920x1440
  field-angle transfer          -12.29 px          -36.70 px      (k 0.16304 / 0.49102)
  factory scaled by image size  -13.01 px          -39.02 px      (k 0.15873 / 0.47619)
```

`docs/3DGS.md`'s "+15.6 and +46.8" is the same claim written in the other
direction (ideal->raw at `r_max`, times the half-diagonal). **The field-angle
reframing changes the wide arm by 6 %.** It cannot rescue it, and it was never
going to: both active wide formats are 4:3 and their corners sit at 97.4 % and
97.0 % of the calibrated maximum radius, so they read essentially the same cone
the calibration was measured over. The reframing has room to matter on the
16:9 ultra-wide and no room at all here.

So the criterion falls to (b), the straightness test, evaluated directly at the
model rather than fitted — with the sign test the pre-registration did not ask
for and which is the only thing that makes a 4 % improvement meaningful:

```
  median chain residual, px     d06152 640x480     87bc2c 1920x1440
  uncorrected                       0.32889            0.75556
  the a1 fit's own optimum          0.32884            0.75740
  transferred model                 0.31646  -3.8%     0.71663  -5.2%
  transferred x 0.5                 0.31764  -3.4%     0.69903  -7.5%
  transferred x 2                   0.33976  +3.3%     0.80131  +6.1%
  transferred NEGATED               0.35028  +6.5%     0.84292  +11.6%
```

**The straightness residual does not increase — it falls, and reversing the
model hurts about twice as much as applying it helps.** On d06152 the mean and
p90 fall too (0.7832 -> 0.7755, 1.3288 -> 1.2933); on 87bc2c the mean is flat
and the p90 rises (4.105 -> 4.232) while the median falls, so that session's
evidence is weaker and is reported as such. The minimum sits between 0.5x and
1.0x the factory shape on both.

**Verdict: (a) FAIL on the literal band test, (b) PASS.** Taken together the
control is *not* a refutation of the transfer — it is a refutation of the
belief that the wide arm's distortion had been measured. The wide arm's *field
of view* result (69.53, not 72.58) is untouched: three instruments still agree
on it and none of them is this one. Only the distortion half of that paragraph
in `docs/3DGS.md` is affected.

## 3. The annulus profile — FAIL, and NOT SEPARATED

The harness reproduces the published baseline exactly before it changes
anything, which is what says it is the same instrument:

```
  d06152                   0.3134  0.3171  0.3182  0.3170   peak-to-trough 1.51 %
  57f29e                   0.3134  0.3173  0.3179  0.3165                  1.44 %
```

Undistorting the ultra-wide before matching:

```
  session   model                                          peak-to-trough
  d06152    published baseline    .3134 .3171 .3182 .3170       1.51 %
            A4 identity warp      .3138 .3174 .3184 .3179       1.46 %
            field-angle           .3125 .3158 .3151 .3153       1.06 %
            A1 paraxial           .3130 .3157 .3156 .3157       0.88 %
            A2/A3 width           .3129 .3154 .3157 .3152       0.90 %
  57f29e    published baseline    .3134 .3173 .3179 .3165       1.44 %
            A4 identity warp      .3140 .3178 .3186 .3165       1.45 %
            field-angle           .3134 .3161 .3157 .3144       0.87 %
            A1 paraxial           .3138 .3162 .3158 .3141       0.77 %
            A2/A3 width           .3140 .3157 .3159 .3143       0.60 %
```

Controls, both clean:

- **A4, the identity warp** — the same Lanczos resampling with the
  magnification set to zero — leaves the non-flatness intact (1.46 % and
  1.45 % against 1.51 % and 1.44 %). The resampling is not doing the work. It
  does move the outermost annulus by 0.28 % on d06152, which is worth knowing
  and is smaller than any effect claimed here.
- **The self-downscale control** returns 0.3200-0.3203 against a truth of
  0.3200 and stays flat, before and after every warp.

**The transfer flattens the profile by about a third and stops.** 1.51 -> 1.06 %
and 1.44 -> 0.87 % is PARTIAL on the pre-registered scale, nowhere near the
0.50 % PASS. And **both ablations do at least as well** — the paraxial reading
that this whole page argues is wrong flattens it *better* on both sessions, and
the plain size-scaled table best of all on 57f29e. The differences between the
three, 0.1-0.3 points, are the size of the session-to-session spread. By the
rule written before the run this is **NOT SEPARATED**: the test does not
distinguish the transfer from the errors it was built to replace.

**Why it stops is the useful part.** The measured structure is about twice what
the ultra-wide's table can produce, and the wide arm supplies the rest:

```
  annulus                       0-300   300-600  600-900  900-1250   peak-to-trough
  predicted, ultra-wide only    +0.00%   +0.36%   +0.63%   +0.48%       0.63 %
  predicted, both factory tables +0.00%  +1.65%   +2.68%   +2.83%       2.83 %
  MEASURED d06152               +0.00%   +1.18%   +1.53%   +1.15%       1.53 %
  MEASURED 57f29e               +0.00%   +1.24%   +1.44%   +0.99%       1.44 %
```

The measurement sits **between** the two predictions. The ultra-wide table
alone explains 41 % of it; both tables together over-explain it by 1.8x. The
profile is a *ratio of two lenses*, so correcting one arm can never flatten it
unless the other really is a pinhole, and it is not.

Correcting **both** arms with their transferred tables — the physically
complete version of this test, `--wide-too` — over-corrects and inverts the
profile:

```
  d06152  both corrected   0.3103  0.3089  0.3071  0.3050   peak-to-trough 1.76 %
  57f29e  both corrected   0.3120  0.3098  0.3067  0.3052                  2.26 %
```

That is the same conclusion the wide-arm control reached from the other side:
the ultra-wide's table is roughly right and the wide's is too strong for its
active format. Matching the measured profile needs the wide's magnification cut
to about **41 %**; the straightness sign test above puts its minimum between
50 % and 100 %. The two instruments agree on the direction and disagree on the
size, so the honest statement is a factor of **1 to 2.4**, not a number.

## 4. Corner displacement, on this page's own anchor

Active ultra-wide 3840x2160, transferred model, raw->ideal:

```
                          field-angle    A1 paraxial   A2/A3 width
  paraxial focal, px         1466.00        1441.56       1539.66
  corner displacement, px     120.3          120.3          90.2
     read off the shipped
     inverse table instead    139.5          160.2          90.7
  RMS over the frame, px       16.88          19.32         11.27
  at --downscale 4 (960x540)
     corner / RMS, px        30.1 / 4.22   30.1 / 4.83   22.5 / 2.82
  on the f = 653 px anchor
     corner, px and dB      53.6  8.6 dB   54.5  8.7 dB  38.2  6.1 dB
     RMS, px and dB          7.52 1.20 dB   8.75 1.40 dB  4.78 0.76 dB
```

The unattributed penalty is **0.7 dB = 4.4 px** on that anchor. The transferred
model's RMS displacement is **7.5 px = 1.20 dB, 1.7x the entire gap it was
meant to explain**. That is not confirmation. Either the model overstates the
lens, or the raw ultra-wide costs more than 0.7 dB and something else in the
export is giving some of it back. Both readings are live and neither is settled
here.

## Sanity check on `k` — PASS on the ultra-wide, FAIL on the wide

The pre-registered check, with a 1 % tolerance:

```
  measured s(0-300) = 0.3134, divided back through the model to r -> 0
     with the wide taken as a pinhole                       0.31292
     with the factory wide table as well                    0.31082

  field-angle transfer, ultra-wide only (wide near-pinhole) 0.31449   +0.50 %
  field-angle transfer, both factory tables                 0.30532   -1.77 %
  the trap, both read as paraxial focal lengths             0.31982   +2.20 %
```

**The ultra-wide half of the transfer lands inside 1 %.** The trap lands
outside it, at 4.4x the error, and in the opposite direction — which is the
independent confirmation that the paraxial reading of `videoFieldOfView` is the
wrong one and that 1466 px, not 1441.6 px, is the active format's paraxial
focal length. The disagreement is carried by the wide lens, as the
pre-registration required be stated rather than absorbed into `k`.

## What this leaves

- The transfer is **defined, exact, and testable**, and its synthetic
  round-trip test (`tools/test_uw_field_angle_transfer.py`) recovers a known
  `k` to 1e-7 and a known paraxial focal length to 1e-4 px from a second
  readout of one lens, where the paraxial reading of the same field of view is
  wrong by 1.6 % — the same sign and size as the real 1441.6 against 1466.0.
- The paraxial focal length of the active ultra-wide format is **1466.0 px**,
  and the two-lens scale agrees to 0.5 %. That number is worth keeping whatever
  happens to the distortion table.
- The distortion transfer is **not shown to describe the ultra-wide**. It
  removes a third of the annulus structure and two ablations remove as much or
  more. It stays opt-in, behind `--field-angle-hfov`, exactly as the previous
  two attempts should have.
- The blocker is now named and it is not the ultra-wide: **the wide arm's
  magnification is too strong by somewhere between 1x and 2.4x and nobody has
  measured it**, because the plumb-line fit is flat to 1e-4 on these rooms. The capture
  `docs/3DGS.md` already asks for — long straight edges across 40 % of the
  frame diagonal, or a checkerboard — would settle the wide arm as well, and
  the wide arm has to be settled before the annulus profile can say anything
  about the ultra-wide at all.
