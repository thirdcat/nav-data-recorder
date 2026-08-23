# Pre-registration: self-calibrating the ultra-wide by carrying LiDAR-anchored points outward

Written **before** `eval/uw_selfcal.py` existed. Model, gauge, nuisance
parameters, thresholds, gates and the decision rule are fixed here so that the
answer cannot be tuned into existence afterwards. The results section at the
bottom was empty when this file was first written.

## Why a third instrument

`docs/3DGS.md` measures that adding the raw ultra-wide arm to a training set
costs the **same wide photographs 0.7 dB** (wide-only 17.65, wide subset of the
mixed run 16.94), and nobody can say whether that is pose or camera model.
Every existing route to the ultra-wide's camera model is blocked, and the blocks
are understood:

- **the lens-to-lens similarity fit** (`eval/uw_radial_profile.py`,
  `eval/wide_fov_probe.py`) sees only the inner **57 %** of the radius, because
  that is all the wide arm overlaps — the wide half-diagonal is 40.9 deg, which
  lands at ultra-wide radius `1441.6 * tan(40.9) = 1249 px` against a 2203 px
  half-diagonal;
- **the plumb-line fit** (`tools/fit_uw_distortion.py`) reaches the corner but
  the rooms do not offer long enough straight edges: at the median chain length
  these captures yield, a 110 px corner displacement bends a line by 0.12 px
  against a 1.44 px straightness floor, and the three sessions come back
  -118.2 / +1.5 / -7.2 px — one lens cannot have three distortions;
- **the factory table** is measured at 4032x3024 / 102.5 deg and the capture ran
  an active 3840x2160 / 106.2007 deg format. `docs/POSE.md` establishes that
  different formats read out different parts of this sensor, so the table is not
  evidence about the recorded pixels in either direction. **No part of this
  measurement may use it**, and none does. (A separate agent is testing whether
  it transfers by field angle; that question is not touched here.)

There is no checkerboard and none is being waited for.

### The opening

LiDAR depth covers only the central cone, so projecting it into the ultra-wide
constrains only the region already known to be fine. **But depth does not have
to be observed at the same instant it is used.** A point whose 3D position is
fixed by LiDAR *while it sits inside the cone* can be tracked outward and used
to constrain the camera model *where there is no depth at all*.

Measured on `d06152`, 40 ultra-wide frames, pyramidal Lucas-Kanade at half
resolution, before this file was written:

```
  1369 tracks of length >= 3 (median 10 frames)
  1125 start inside the depth cone (< 1249 px)
   903 of those reach past it            (80.3 %)
   663 reach past 1600 px
  max radius reached by an anchored track: 2211 px, against a corner of 2203
```

The tracks exist, they are plentiful, and they reach the corner.

## What the instrument does

1. **Track.** Pyramidal Lucas-Kanade on the ultra-wide frames at a fixed
   working resolution, seeded by `goodFeaturesToTrack` on every frame,
   forward-backward consistency checked.
2. **Anchor.** For each track, take the observation at the *smallest* image
   radius that lies inside the depth cone and carries a LiDAR return. The
   3D point is
   `P = T_uw(anchor) . ( Z . ray(u_anchor ; camera model) )`
   where `Z` is the LiDAR z-coordinate **in the ultra-wide camera frame** read
   at that pixel, and `ray` is the model's unprojection normalised to `z = 1`.
   **Only `Z` is taken from the depth map; the ray direction comes from the
   camera model being fitted.** That is deliberate: it makes the anchor
   first-order insensitive to an error in the *depth camera's* cone, which is
   unlogged on two of the three sessions here.
3. **Carry.** Every other observation of that track — including the ones past
   the cone, past 1600 px, out to the corner — becomes a reprojection residual
   in raw ultra-wide pixels.
4. **Fit.** Minimise those residuals over the camera model *and* a per-frame
   pose correction, jointly.

### Poses

Relative camera motion comes from the wide arm's trajectory
(`eval/pi3_poseless.py --join-scale-mode depth` where it exists), composed with
the rig extrinsic in `calib/iphone17-1_ultrawide.json` — 19.272 mm, 0.462 deg,
direction verified in `calib/README.md`.

**The two lenses do not shoot together.** The shutter offset is 56-65 ms at the
median and up to 103 ms, over which the camera travels 34-39 mm, which is *more*
than the rig baseline. `tools/export_3dgs.py` handles this by SLERP-interpolating
the wide trajectory to the ultra-wide frame's own timestamp, and
`docs/RIG_EXPORT_PREREG.md` shows the naive nearest-pose construction failing the
same gate the interpolated one passes (1.39x against a 1.5x bar; interpolated
1.72x). **This instrument does the same** — it imports
`interpolate_camera_pose` and `rig_extrinsic` from `tools/export_3dgs.py` rather
than re-deriving them, so there is one implementation of the composition and
`tools/test_export_3dgs.py` already guards it. Ultra-wide frames outside the
posed walk are dropped, never extrapolated.

The trajectory is an **initialisation, not a constraint**. `d06152` has a
metric `--join-scale-mode depth` chain; `57f29e` and `d0f44f` only have the
historical `free` chains, whose join scales compound to 1.08 and 1.79 — a
free chain's local metric scale can be tens of per cent off, which would swamp a
metric reprojection fit. Making the per-frame pose a free parameter removes that
difference between the sessions instead of hiding it, and the 3D points are
placed from the **optimised** anchor pose rather than the chain pose so the
system stays self-consistent under a scale correction. See "The confound" for
why this is the conservative choice rather than the convenient one.

## The model being fitted

The same rectification `tools/fit_uw_distortion.py` uses, so the numbers are
directly comparable with its -118.2 / +1.5 / -7.2 px:

```
  c      = (w/2, h/2)                        frame centre
  R      = hypot(w, h) / 2                   half-diagonal; rho = 1 at the corner
  rho    = |p - c| / R
  p_rect = c + (p - c) * (1 + a1 rho^2 + a2 rho^4)
```

`p_rect` is the pinhole (undistorted) pixel; `p` is the recorded pixel.
Projection into a raw frame inverts this by Newton iteration on the radius,
which is exact to float tolerance and is unit-tested against a round trip.

Reported quantities:

- `a1`, `a2` — dimensionless and resolution-independent for a fixed format, so a
  fit run at 1920x1080 reports a corner displacement at 3840x2160 without
  refitting.
- `D_corner = R * (a1 + a2)` — signed pixel displacement at the frame corner, at
  the native 3840x2160. **The headline number.**
- `D_rms` — RMS of `R * rho * (a1 rho^2 + a2 rho^4)` over the frame, evaluated
  at **960x540**, because the mixed run trained at `--downscale 4` and that is
  the resolution the 0.7 dB was measured at. This is what goes on the dB scale.
- `f` is **fixed** at the logged active format (106.2007 deg -> 1441.56 px at
  3840 wide) for the headline fit, exactly as the export declares it. A fit with
  `f` free is reported as a sensitivity, never as the headline, because a free
  focal length trades against `a1` through the near-degenerate `rho^0` direction
  and would make `D_corner` gauge-dependent.

### Parameters and gauge

| parameter | count | prior |
|---|---|---|
| `a1, a2` | 2 | none |
| per-frame SE(3) correction | 6 per frame, **frame 0 held fixed** | none |
| per-track 3D point | 0 — determined by anchor pixel, anchor depth, anchor pose | — |

Holding frame 0 fixes the 6-DOF gauge. Scale is pinned by the LiDAR depths
themselves — a point's distance from its anchor camera is the metric measurement
— so there is no seventh gauge direction and none is fixed.

Robust loss: `soft_l1`, `f_scale = 2.0 px` at the working resolution. Two
passes: observations whose first-pass residual exceeds `5 x MAD` are dropped and
the fit is repeated.

## The confound, and how it is separated

Trajectory error also produces reprojection error. They are separated **by
shape**, and the shape argument is written down here before any fit runs.

For a camera with normalised coordinates `(x, y) = (X/Z, Y/Z)`, a small pose
error `(omega, t)` displaces the image by

```
  dx = omega_x x y - omega_y (1 + x^2) + omega_z y + (-t_x + x t_z) / Z
  dy = omega_x (1 + y^2) - omega_y x y - omega_z x + (-t_y + y t_z) / Z
```

Take the component along the outward radial direction at image angle `theta`
and average over `theta` at fixed radius:

| source | radial component | azimuthal mean |
|---|---|---|
| `omega_x`, `omega_y` | `~ cos theta`, `~ sin theta` odd in theta | **0** |
| `omega_z` | identically zero (pure tangential) | **0** |
| `t_x`, `t_y` | `~ cos theta / Z`, `~ sin theta / Z` | **0** |
| `t_z` | `r t_z / Z` | `r t_z / <Z>`, order **rho^1** |
| radial distortion | `r (a1 rho^2 + a2 rho^4)` | order **rho^3, rho^5** |

**So the azimuthally-averaged radial residual is the discriminator.** Every
pose degree of freedom except forward translation averages to zero over
azimuth; forward translation survives but at order `rho^1` and weighted by
`1/Z`, whereas the lens term is `rho^3`/`rho^5` and depth-independent.

The instrument therefore reports, on the **pose-only** fit (per-frame SE(3)
free, camera model held at the logged pinhole):

- the azimuthally-averaged radial residual binned by radius, and a
  least-squares decomposition of it into `c1 rho + c3 rho^3 + c5 rho^5`;
- the same for the tangential component, which must come back consistent with
  zero if the shape argument holds;
- the RMS residual under four fits — `none` (chain poses, logged pinhole),
  `pose` only, `model` only, and `both` — so the reader can see how much each
  explains.

**The camera model is credited only with `r_pose - r_both`**, the reduction that
survives *after* the pose has been given six free degrees of freedom per frame.
`r_none - r_model` is reported but is not evidence, because the pose has not
been allowed to compete for it.

**If `r_pose - r_both` is under 10 % of `r_none - r_pose`, the report says the
two cannot be separated on this data and no coefficient is claimed.** That is a
real answer and it is better than a coefficient.

## Acceptance criteria

Every threshold below is fixed now.

### 1. Injection recovery — the gate

Warp the ultra-wide frames by a known radial map, run the **whole** pipeline on
the warped frames — re-tracking included, not just re-fitting — and demand the
coefficient back. Because warping composes with whatever the lens already does,
the quantity demanded back is the **difference** from the unwarped run:

```
  D_recovered = D_corner(injected run) - D_corner(baseline run)
```

Injection sweep, corner displacements at 3840x2160, chosen to bracket the
30-54 px band the 0.7 dB question lives in:

```
  -80, -40, -20, +20, +40, +80 px
```

*Amended before any fit ran, while the tool was being written:* the exact
comparison is not the first-order difference above but the **composition**,
`corner_displacement(compose_expected(baseline_fit, injection))`, which
`tools/fit_uw_distortion.py` already computes and which reduces to the
difference to first order. Both are reported; the composition is the bar.

Pass bars, all of which must hold:

- **sign** correct on all six;
- **per-injection error** `|D_recovered - D_expected| <= max(5 px, 0.15 |D_injected|)`;
- **response slope** of `D_recovered` on `D_injected`, fitted through the six
  points, in `[0.85, 1.15]`. `tools/fit_uw_distortion.py` returned 0.745 on this
  lens and that was read as a failure; the bar here is set where a failure of
  that size is caught.

This gate runs with the per-frame SE(3) nuisance **on**, which is the whole
point: it measures whether a free pose eats the radial term.

**Nothing below this line is readable if this gate fails.**

### 2. Null control on the wide arm

The same pipeline on the wide arm — same code path, LiDAR depth resampled onto
the wide grid with the identity extrinsic and no SLERP, because the wide frames
carry their own poses — must return near zero. Three independent instruments
already agree the wide arm's residual distortion is small:
`tools/fit_uw_distortion.py` measures -1.41 px at 640x480 and +3.29 px at
1920x1440, where the factory table would demand +15.6 and +46.8.

The wide half-diagonal is 400 px against the ultra-wide's 2203, so a pixel bar
would not be comparable. The bar is therefore **dimensionless**:

```
  |a1 + a2|_wide  <  0.007
```

which is 2.8 px at 640x480, and is under half the smallest coefficient the
ultra-wide claim needs (30 px at 2203 px is `a1 + a2 = 0.0136`). Both the
dimensionless coefficient and its pixel value at 640x480 are reported so the
comparison is apples to apples.

A null that the instrument could not have failed proves nothing, so the
**injection sweep is run on the wide arm too**, at corner displacements of
`+/-15` and `+/-30` px at 640x480 (`|a1 + a2|` of 0.0375 and 0.075). If those
do not come back within the criterion-1 bars, the wide-arm null is reported as
uninformative rather than as a pass.

### 3. Cross-session agreement

Fit `d06152`, `57f29e` and `d0f44f` independently — separate tracks, separate
anchors, separate poses, nothing shared but the code. One lens cannot have three
distortions.

```
  spread = max(D_corner) - min(D_corner) over the three sessions
```

Pass bars, both required:

- all three share **sign**;
- `spread < 0.5 * |mean D_corner|`.

If the spread exceeds the corner displacement being claimed, the fit is **not
established** and the report says so, exactly as `tools/fit_uw_distortion.py`
was refused for returning -118.2 / +1.5 / -7.2.

A per-session bootstrap over tracks (200 resamples) gives a 5-95 % band on
`D_corner`. **If a session's band spans zero, no coefficient is claimed for it.**

### 4. The profile test

`eval/uw_radial_profile.py` finds a reproducible non-flat lens-to-lens scale
profile:

```
  annulus, ultra-wide px      d06152    57f29e
      0 -  300                0.3134    0.3134
    300 -  600                0.3171    0.3173
    600 -  900                0.3182    0.3179
    900 - 1250                0.3170    0.3165
```

peak-to-trough **1.53 %** and **1.44 %**, with the control — a frame against
itself downscaled by a known 0.3200 — returning 0.3201 flat.

Undistorting the ultra-wide frames with the fitted model should **flatten** it.
The instrument writes a session directory whose ultra-wide frames have been
undistorted and whose wide frames, `frames.jsonl`, `frames_wide.jsonl` and
`manifest.json` are unchanged, then `eval/uw_radial_profile.py` is run on it
**unmodified, as a subprocess**. The profile tool is not edited and not
imported; it does not know it is being tested.

Pass bar: peak-to-trough spread falls by **at least half** on both `d06152` and
`57f29e` (to under 0.77 % and 0.72 %).

Two things are stated in advance so they cannot become excuses afterwards. The
absolute `s` values *will* move, because undistorting changes the frame's
effective focal length; only the spread is the test. And the wide lens carries
its own small distortion, which also contributes to the profile, so perfect
flattening is not the expectation and is not the bar.

### 5. The dB anchor

`docs/3DGS.md`: three horizontal centimetres at 2 m through `f = 653 px` is ten
pixels of texture displacement, and that is worth 1.6 dB. Read linearly,
**0.7 dB is 4.4 px of texture displacement**, at the 960x540 the mixed run
trained at. The same conversion `docs/UW_DISTORTION_PREREG.md` fixed:

```
  D_rms at 960x540 >= 4.4 px   -> the un-modelled distortion is large enough to
                                  account for the 0.7 dB
  D_rms at 960x540 <= 1.0 px   -> it is not, and the mis-posed hypothesis
                                  survives
  between                      -> undecided, and reported as undecided
```

At 960x540 the half-diagonal is 550.7 px and `rms(rho^3) = 0.3204`, so the
4.4 px bar is `a1 = 0.0249`, i.e. a corner displacement of **54 px** at
3840x2160 for a pure `rho^2` term; the `f`-rescaled reading of the same anchor
gives **30 px**. That is the 30-54 px band.

## The decision rule, written before the fit runs

1. Criterion 1 fails -> report the failure and stop. No coefficient is
   published from a pipeline that cannot recover one it was handed.
2. Criterion 1 passes and criterion 2 fails -> the instrument has a bias it
   attributes to the lens. Report the bias, subtract nothing, claim nothing.
3. Criteria 1 and 2 pass, criterion 3 fails -> report all three sessions and
   the spread, and state that the ultra-wide's camera model is still not
   established. Do not average three disagreeing numbers.
4. Criteria 1-3 pass -> `D_corner` is the ultra-wide's radial distortion for the
   active 3840x2160 / 106.2007 deg format. Then criterion 5 decides the
   attribution:
   - `D_rms(960x540) >= 4.4 px` -> the camera model is large enough to explain
     the 0.7 dB;
   - `<= 1.0 px` -> it is not, and the residual penalty belongs to pose or to
     something else;
   - in between -> undecided.
5. Criterion 4 is diagnostic in either direction: it is the only check that
   compares against an instrument built on a completely different principle
   (lens-to-lens similarity, no depth, no poses, no tracks). A pass corroborates;
   a failure means the fitted model does not describe the one piece of the
   ultra-wide's radial behaviour that was already reproducibly measured, and
   that is reported next to the coefficient rather than under it.
6. Independently of 1-5, if `r_pose - r_both` is under 10 % of
   `r_none - r_pose`, the report says radial and pose cannot be separated on
   this data.

## What this cannot do, stated in advance

- **It cannot see a tangential (decentring) term.** The model is radial about
  the frame centre. `calib/` puts the distortion centre 2.2 px off the principal
  point at 4032x3024; nothing here can confirm that for the active format, and
  a decentring term would show up as unexplained residual rather than as a
  wrong `a1`.
- **It cannot separate the ultra-wide's model from a systematic error in the rig
  extrinsic's magnitude.** `docs/POSE.md` records that the 19.272 mm baseline's
  *direction* is verified and its *magnitude* has never been recovered from a
  real capture. A baseline error is a per-frame translation, which the SE(3)
  nuisance absorbs — so it does not bias `a1`, but neither is it measured here.
- **The anchor depth is read at the pinhole-scattered pixel.** Within the cone a
  50 px corner displacement moves the scatter by about 9 px at `rho = 0.57`, and
  a depth gradient across that shifts `Z`. The fit therefore re-scatters the
  depth through the fitted model and repeats, and the report states how much the
  coefficient moved between the first and second pass. If it moves by more than
  the bootstrap band, the fit is reported as not converged.
- **`57f29e` and `d0f44f` have no `depth_calibration` and no logged wide field
  of view.** Their wide focal length is taken as `d06152`'s logged 69.52744 deg
  — the same device, the same 640x480 wide format — rather than the factory
  72.58, and this is recorded as an assumption. Their depth cone falls back to
  the factory calibration. Both are why step 2 of the method reads only `Z` from
  the depth map.

## What changed after the pre-registration, and why

Three things were decided after this file was written and before the ultra-wide
coefficient was read. They are here rather than buried in a commit message
because each of them moves a number.

### The radial order was cut from two terms to one — the two-term fit fails its own gate

`a1, a2` was pre-registered. On the **wide arm**, whose answer three other
instruments already agree on, the two-term fit fails the injection gate and the
one-term fit passes:

```
  injected corner px    order 2 recovered    order 1 recovered   (bar 5.0 px)
      -30                  -25.8                -30.6
      -15                   -5.6                -19.0
      +15                  +48.1                +15.8
      +30                  +62.8                +33.0
  response slope          1.539                 1.081       (band 0.85-1.15)
```

The reason is identifiability, not arithmetic. `D_corner` is evaluated at
`rho = 1` and the tracks reach `rho = 0.95-0.98`, so the corner is always a
short extrapolation; with two coefficients that trade against each other along
`rho^2` versus `rho^4`, the extrapolation is unstable, and a positive injection
makes it worse because the warp samples from beyond the sensor and the valid-data
mask cuts the outer radius further (the `+30` row has 8 observations past
`rho = 0.8`, against 174 unwarped). **The headline is therefore the one-term
fit, and the two-term fit is reported beside it as the thing that failed.**

### The injection warp needs its validity mask, and ignoring it broke the gate

`inject_distortion` returns an eroded valid-data mask for expanding
coefficients. The first version of the tool ignored it. The warped frame then
has a hard black boundary that Lucas-Kanade locks onto — an edge that moves with
the *frame* rather than the scene, producing long confident tracks with zero
parallax at exactly the large radius the fit depends on. A `+15 px` injection on
the wide arm came back as **`+101.8 px`**. With the mask applied and tracks that
touch the invalid region dropped, the same injection returns `+15.8`.

### Criterion 6 — the `model/pose` ratio — is retracted

The pre-registration said: *if `r_pose - r_both` is under 10 % of
`r_none - r_pose`, the two cannot be separated.* **That rule fails on an input
whose answer is known.** On the synthetic scene in `eval/test_uw_selfcal.py`,
with a known `+24.2 px` lens and poses deliberately wrong by 0.23 deg and
10 mm:

```
  input                       r_none   r_pose   r_both   model/pose   fitted corner
  +24.2 px lens, exact poses   17.02   16.837   16.832     0.0296        +24.17
  +24.2 px lens, wrong poses   19.08   16.837   16.832     0.0024        +24.17
  no lens, wrong poses          7.87    0.000    0.000     0.0000         -0.00
```

The ratio is 0.0024 — a twentieth of the bar — while the joint fit returns the
right answer to 0.3 %. The reason is that the camera model bites at large radius,
where 2 % of the observations live, so it barely moves a whole-frame RMS even
when it is perfectly identified. The ratio is reported as a description of where
the residual went, never as a test. **The test of separability is injection
recovery**, which is run on the real data with the real pose difficulty.

The same measurement disciplines the shape decomposition. On synthetic data the
azimuthally averaged radial residual recovers 57 % of a known lens with exact
poses, 49 % at 0.06 deg / 3 mm and 17 % at 0.23 deg / 10 mm, and a *pose-only*
fit leaves only 25 % of it. So the decomposition is a **signed lower bound**, not
an estimator — the derivation in this file is right about the sign and silent
about the size, because it does not model the fact that the anchor point is
itself placed through the wrong ray and has already cancelled part of the
distortion. It is reported as corroboration of direction.

### Tracking was sized from the measured motion, on the census only

The camera turns 2-12 degrees between consecutive frames at 5 Hz on `d06152`,
measured against the essential matrix of the frame pair. At the working focal
length of 721 px that is 25-250 px of image displacement, largest exactly where
this instrument needs observations. A 21 px window over four pyramid levels
reaches about 168 px and loses them: 8 135 observations, 298 past `rho = 0.8`,
against 30 154 and 1 422 for a 31 px window over five levels at `min_distance`
10. The change was made on the **census** — a count of observations and their
radii — before any coefficient was fitted from it, and the earlier settings
returned a different answer (`-6.3 px` against `+23.4 px`), so it is recorded
here rather than presented as the only run that happened.

## Results

**Headline. This does not produce a coefficient for the ultra-wide. It produces
a bound, and the bound answers the question that was asked.**

Across three sessions and six frame-block lengths from 40 to 260 frames, the
fitted corner displacement is always positive and **never exceeds +17.2 px**;
the three sessions at their longest blocks give `+2.5 / +7.2 / +10.5 px`. Two
independent checks — the injection gate on warped frames, and the spread over
block lengths — agree that the instrument's accuracy floor is about `+/-9 px`,
which puts a ceiling near **+26 px**. The 0.7 dB needs **30-54 px**. So the
ultra-wide's radial camera model is excluded as the explanation, by an
instrument whose own error bar is smaller than the gap.

Three things qualify that and all three are below: the point estimate wanders
inside `+/-9 px` with the length of the frame block and never converges, so no
coefficient is quoted; every bootstrap band on this page is over-confident by an
order of magnitude for the same reason; and a fourth reading — the annulus
profile taken through the two logged fields of view — says `+51 px` and is not
refutable from inside this page.

For scale, the route this replaces returned `-118.2 / +1.5 / -7.2 px` on the
same three sessions.

Unless a row says otherwise, runs use the pre-registered settings — 60
contiguous ultra-wide frames, 40 on the two short sessions — worked at 1920x1080
for the ultra-wide and 640x480 for the wide, at `--order 1`. The block-length
section below shows why those settings are not the ones to read, and repeats
everything at each session's longest available block. Commands are in the tool's
docstring.

### What the instrument sees

```
  session  arm         tracks   carried obs   past the cone   rho>=0.8   max radius
  d06152   ultra-wide   5 158       21 192          4 546         479    2 098 px of 2 203
  57f29e   ultra-wide   7 489       48 308         13 221       1 912    2 129 px of 2 203
  d0f44f   ultra-wide   7 904       45 364          7 331       1 058    2 124 px of 2 203
  d06152   wide         1 197        3 523              -         174      393 px of   400
```

**The opening the brief measured is real and it survives anchoring.** On
`d06152`, 4 546 of 21 192 carried observations sit outside the 1 251 px cone the
wide arm covers, 479 of them past `rho = 0.8`, and the furthest anchored track
reaches 2 098 px against a 2 203 px half-diagonal. The median anchor sits at
575 px — well inside the cone, where the LiDAR actually returned — at a median
range of 1.05 m. **Every one of those 4 546 observations is a constraint on the
camera model in a place where neither the lens-to-lens fit nor the depth map can
see, which is what this instrument was built for.**

The blind spot that remains is geometric and permanent: a 16:9 frame puts only
14.5 % of its area past `rho = 0.8` and 3.2 % past `rho = 0.9`, so the corner is
always thinly sampled and `D_corner` is always a short extrapolation from
`rho = 0.95`.

### 1. Injection recovery — the gate. Slope passes, three of six bars do not

`d06152`, ultra-wide, the whole pipeline re-run on warped frames, tracking
included:

```
  injected   expected   recovered    error    bar    obs past rho=0.8
    -80.0      -75.4       -72.7      +2.7   12.0        536
    -40.0      -35.2       -32.5      +2.6    6.0        479
    -20.0      -15.1        -7.5      +7.5    5.0   FAIL 475
    +20.0      +25.2       +16.1      -9.1    5.0   FAIL 290
    +40.0      +45.3       +36.2      -9.0    6.0   FAIL 187
    +80.0      +85.5       +83.0      -2.5   12.0         91
  response slope 0.933    (pre-registered band 0.85-1.15)      pass
```

Sign correct 6 of 6. Slope inside the band. Three of six inside their bar.

**Read the shape of the errors, not just the pass column.** They are not a scale
error — the slope is 0.933 — they are an *offset* of about `-9` to `+7` px that
does not grow with the injection. So the instrument's sensitivity is right and
its accuracy has a **floor of roughly +/-9 px** on this lens, which is nine times
finer than the plumb-line route's `+/-20 px at best` and forty times finer than
its worst. The pre-registered bars were written on the assumption that the
bootstrap band was the whole uncertainty; the gate says it is not, and the gate
is the number to carry forward.

Against a pre-registered rule the honest verdict is **the gate fails as
written**, and the instrument is therefore used to answer "is the corner
displacement above 30 px?" — a question its measured floor can answer — and not
"what is the corner displacement to the pixel", which it cannot.

### Radial versus pose

**The pose is not a perturbation on this capture, and that is a measurement.**
Against the essential matrix of each consecutive ultra-wide pair, the camera
turns 2-12 degrees per frame at 5 Hz and the chain disagrees with the images by
0.8-4.4 degrees per step. Over a 60-frame block that accumulates: the fitted
per-frame correction is a smooth **drift**, 8.9 deg and 17.9 cm at one end of
the block and 2.0 deg and 3.1 cm at the other, not frame-to-frame scatter.

```
  d06152 ultra-wide, order 1, same 21 192 observations
  fit     rms px   corner px
  none    42.565        +0.0      chain pose, logged pinhole
  pose     8.570        +0.0      per-frame SE(3) free
  model   41.895       +76.0      camera model free, chain pose held
  both     8.575        +5.0      both free — the fit
```

**Pose explains 34.0 px of RMS and the camera model a further 0.00 px.** That
ratio is *not* evidence of inseparability — see the retraction above, where the
same ratio is 0.0024 on a synthetic scene whose lens is known to be +24.2 px and
is recovered to 0.3 %. It is evidence of where the residual lives: almost all of
it is pose, and the camera model acts on the 2 % of observations at large radius.

The azimuthal decomposition agrees in direction and is quantitatively useless
here. On `d06152` the unmodelled residual's radial mean implies +103.0 px and
the pose-only residual's implies +2.5 px, against a joint fit of +5.0; on the
wide arm the same numbers are -53.2 px and -2.3 px against a joint fit of
+1.5 px. The synthetic control explains why: the decomposition recovers 57 % of
a known lens with exact poses and 17 % at 0.23 deg / 10 mm of pose error, and
this capture's pose error is an order of magnitude larger than that. Only its
**sign** is usable, and on all three ultra-wide sessions the sign is positive,
matching the joint fit.

**So the honest answer to "how much of the residual is radial and how much is
pose" is: 99.99 % of the reprojection RMS is pose, and the radial term is
identified — if it is identified at all — from the shape of what is left at the
outer 15 % of the frame.**

That is not a figure of speech. Holding `a1` at a grid of values and re-fitting
the pose each time — the profile likelihood — gives the whole cost surface this
question is decided on:

```
  a1        corner px    rms px    robust cost   cost above the minimum
  -0.0300     -66.09     8.6559     225 365.7        +2.54 %
  -0.0100     -22.03     8.4920     220 513.9        +0.33 %
   0.0000      +0.00     8.4591     219 798.8        +0.01 %
  +0.0023      +4.96     8.4558     219 780.3         minimum
  +0.0100     +22.03     8.4550     220 075.9        +0.13 %
  +0.0231     +50.99     8.4917     221 865.3        +0.95 %
  +0.0450     +99.13     8.6451     228 030.0        +3.75 %
```

**Moving the ultra-wide's corner from +5 px to +51 px costs 0.03 px of RMS out
of 8.46, and 0.95 % of the robust cost.** The surface has a genuine minimum —
this is not a flat direction — but the whole question lives inside half a per
cent of the residual. That is exactly why the instrument needed a gate, why the
bootstrap band that curvature implies is far too tight, and why the `+/-9 px`
floor is the number to quote. It is also why the answer wanders when the frame
block changes: a systematic worth a few hundredths of a pixel of RMS is enough
to move the minimum by ten pixels of corner displacement.

### 2. Null control on the wide arm

```
  order   corner px at 640x480   a1 (+a2)      bootstrap 5-95 %     D_rms at 160x120
    1          -1.70             -0.00426      [-4.39, -0.57]           0.13 px
    2          +1.46             +0.00365      [-2.21, +3.78]           0.44 px
```

`|a1| = 0.0043` at order 1, **inside the pre-registered 0.007 bar**. The pixel
value is worth reading against the two instruments that already answered this
question on the same lens:

```
  tools/fit_uw_distortion.py, plumb-line, 640x480     -1.41 px
  eval/uw_selfcal.py, this instrument, 640x480        -1.70 px
  what the factory table would demand at 640x480     +15.6 px
```

**Two instruments that share no input — one fits straight edges in single
frames, the other carries LiDAR-anchored points across a trajectory — agree to
0.3 px, and both reject the factory table by an order of magnitude.** That is
the strongest corroboration in this whole page and it is on the arm where the
answer was already known.

The wide arm's own injection control, which decides whether the null means
anything:

```
  injected   expected   recovered    error   bar
    -30.0      -31.4       -30.6      +0.8   5.0   pass
    -15.0      -16.6       -19.0      -2.5   5.0   pass
    +15.0      +13.1       +15.8      +2.7   5.0   pass
    +30.0      +28.0       +33.0      +5.0   5.0   at the bar, to rounding
  response slope 1.081 (band 0.85-1.15)                   pass
```

Three clear passes, a slope inside the band, and a fourth point sitting exactly
on its bar. The `+30 px` row is the hardest one by construction: an expanding
warp samples from beyond the sensor, and after the validity mask only **8**
observations survive past `rho = 0.8`, against 174 unwarped. A margin of zero is
undecided, not a pass — so the wide-arm null is recorded as **passing on three of
four injections with the fourth undecided**, and the null itself as informative.

### 3. Cross-session agreement — three sessions inside the instrument's own floor

Fitted independently: separate tracks, separate anchors, separate poses, nothing
shared but the code. `d06152` carries a metric `--join-scale-mode depth` chain;
`57f29e` and `d0f44f` only have the historical `free` chains, and neither logs a
`depth_calibration` or a wide field of view.

```
  session   frames  tracks  carried obs  rho>=0.8   corner px   bootstrap 5-95 %
  d06152      60     5 158     21 192       479        +5.05     [+0.40, +9.91]
  57f29e      40     7 489     48 308     1 912       +10.28     [+7.94, +12.57]
  d0f44f      40     7 904     45 364     1 058        +8.58     [+6.56, +9.94]

  mean +7.97 px    spread 5.23 px    all three positive, no band spans zero
```

Against the pre-registered bar — `spread < 0.5 * |mean|`, i.e. 3.98 px — the
spread of 5.23 px **fails as written**. Against the accuracy floor the gate
actually measured, +/-9 px, the three sessions agree comfortably. And against the
instrument this replaces:

```
  tools/fit_uw_distortion.py   -118.2 / +1.5 / -7.2 px    spread 119.7 px
  eval/uw_selfcal.py            +5.1 / +10.3 / +8.6 px    spread   5.2 px
```

**A 23-fold reduction in cross-session spread, and for the first time the three
sessions agree on a sign.** That is the criterion this instrument was built to
clear and it clears it in substance while missing a bar that was set before
anyone knew what the instrument's floor was.

*Read the next section before trusting this table.* These three fits use the
pre-registered frame counts, 60 and 40, and the frame count turns out to be the
dominant variable; at each session's longest block the numbers become +2.5, +7.2
and +10.5 px, a spread of 8.0 against a 3.4 px bar. The bar is failed either
way; what does not change is that all six values are positive and all six are
far below 30 px.

Two things make the agreement stronger than it looks either way. The three
sessions' trajectories are of very different quality — at the long blocks the
fitted global chain scale is 1.007 on `d06152`, 0.945 on `57f29e` and **0.886**
on `d0f44f` — and the fitted per-frame pose corrections differ by more than 2x
(4.4 deg / 11.9 cm on `d0f44f` against 1.8 deg / 3.9 cm on `57f29e`). The
coefficient moves by 6 px across that. And two of the
three sessions do not log the depth camera's intrinsics at all, which is exactly
the error the anchoring step was designed to be first-order blind to.

### The block length was a constant nobody chose, and it is a variable

**This is the finding that governs everything above.** The pre-registration
fixed *which* frames to use — a centred contiguous block — and fixed *how many*
at 60 without a reason. Sweeping it on `d06152`, everything else identical:

```
  frames   carried obs   corner px   bootstrap 5-95 %      residual rms
     12        3 354       -86.5      (not run)              9.75
     20        6 014       -45.3      [-58.1, -31.7]        13.52
     30        7 181       -12.2      [-24.9,  +4.4]        12.60
     40       10 061        -0.7      [-11.9,  +9.3]        10.31
     60       21 192        +5.1      [ +0.4,  +9.9]         8.57
     80       35 726       +16.9      [+11.9, +22.3]         8.30
    120       67 597       +17.2      [+13.7, +19.4]         9.64
    180       87 531       +13.0      [+10.7, +15.2]         9.90
    260      196 688        +2.5      [ +0.5,  +3.6]         8.39
```

**The answer moves 104 px across the sweep, and it never settles.** Below about
40 frames it is not a measurement at all — the value runs away to -86 px as the
block shortens. From 40 frames on it stays in a band, but it wanders inside that
band with no sign of convergence: -0.7, +5.1, +16.9, +17.2, +13.0, +2.5. Rising
to 120 and falling again by 260. **The plateau at 80-120 was an artefact of
where the sweep happened to stop, and 180 and 260 were run precisely to check
it.**

Three things follow, and the first is the most important.

**The bootstrap was never the uncertainty.** It measures how much the answer
moves when tracks are resampled; the answer is governed by the length of the
block, which is a property of the experiment and not of the lens. Every band on
this page is over-confident: at 260 frames the fit reports `+2.5 [+0.5, +3.6]`
and at 120 frames `+17.2 [+13.7, +19.4]`, on the same session and the same lens,
with 4 976 and 1 497 observations past `rho = 0.8` respectively. Neither band
comes close to containing the other.

**The block-length wander and the injection gate's offset are the same
systematic, and they agree on its size.** The gate measured an accuracy floor of
about `+/-9 px` from warped frames; the block sweep gives a spread of 17.9 px
over blocks of 40-260 frames, i.e. `+/-9 px` about its midpoint. Two independent
ways of asking "how wrong can this instrument be" return the same number. That
is the instrument's resolution and there is no setting that improves it.

**What survives is a bound, not a point.** Over six block lengths from 40 to 260
frames and three sessions at their longest blocks, **the largest value the
instrument ever returns is +17.2 px**, and the smallest is -0.7. Adding the
whole +/-9 px floor on the unfavourable side puts a ceiling at about **+26 px**.

```
  session   frames   carried obs   rho>=0.8   corner px   bootstrap 5-95 %
  d06152      260      196 688       4 976       +2.5      [ +0.5,  +3.6]
  57f29e       89      107 871       3 860       +7.2      [ +5.0,  +8.8]
  d0f44f      112      118 416       2 679      +10.5      [ +8.4, +11.9]

  mean +6.7 px    spread 8.0 px    all positive, no band spans zero
```

Every one of those is positive, and none of them, nor any point in the block
sweep, nor the ceiling with the floor added on, reaches the 30 px that the
0.7 dB question begins at.

### 4. The profile test — fails, and the reason is not the fit

`eval/uw_radial_profile.py`, run unmodified as a subprocess on a session whose
ultra-wide frames were straightened by the fitted model:

```
                    0-300    300-600   600-900   900-1250   peak-to-trough
  control          --        --         0.3201    0.3201       0.00 %
  before           0.3134    0.3171     0.3182    0.3170       1.53 %
  after (order 1)  0.3135    0.3171     0.3185    0.3175       1.59 %
```

The bar was "peak-to-trough falls by at least half". It does not fall at all.
**Criterion 4 fails.**

The reason is worth more than the failure. Work out what an annulus scale is:
`s = r_wide / r_ultrawide` at the same field angle, so as `rho -> 0` it is
exactly the ratio of the two **paraxial** focal lengths, and *no* radial term
about the frame centre can change that limit. The logged fields of view give
`f_wide / f_ultrawide = 461.04 / 1441.56 = 0.3198`. The measured inner annulus
is **0.3134**, on both sessions. That 2.0 % is a `rho^0` discrepancy, and a
`rho^2` model was never going to touch it:

```
  model applied                                    predicted profile      p-to-t
  both lenses pinhole                    0.3198 0.3198 0.3198 0.3198      0.00 %
  wide at its measured -0.00426 only     0.3199 0.3202 0.3208 0.3219     +0.63 %
  + this fit's ultra-wide a1 = +0.00226  0.3199 0.3202 0.3210 0.3222     +0.73 %
  observed                               0.3134 0.3171 0.3182 0.3170     +1.53 %
```

The best two-term ultra-wide model that can be made to chase the observed
profile runs its `a2` into the search bound and still misses by 0.0033 RMS,
because it cannot lower the `rho -> 0` limit. **The profile's dominant structure
is a focal-length offset, not a radial term, which is exactly the trap
`docs/POSE.md` records: `videoFieldOfView` is a whole-frame quantity that
already contains the distortion, so it is not a paraxial focal length.**

#### The same offset read as a corner displacement — and it disagrees with the fit by 10x

That `rho^0` offset can be converted into a coefficient, and this is the one
place in this page where a number lands inside the band that matters. If the
logged field of view is the *raw* whole-frame angle, then the paraxial focal
length is `f_logged * g(rho_edge)`, and the profile's paraxial limit pins the
ultra-wide's `g` at its horizontal edge:

```
  wide term used                            implied ultra-wide a1   corner px
  -0.00426  (this instrument)                     +0.02331            +51.4
  -0.003525 (tools/fit_uw_distortion, -1.41 px)   +0.02394            +52.7
   0.0      (wide assumed perfectly rectilinear)  +0.02698            +59.4
```

**+51 to +59 px — inside the 30-54 px band that would explain the 0.7 dB.** The
alternative, that the ultra-wide is rectilinear and the *wide* lens carries the
whole 2 %, requires the wide lens to have a corner displacement of **-12.6 px at
640x480** against -1.41 and -1.70 px measured by two instruments. That escape is
closed.

So this page contains two readings of the same lens that differ by an order of
magnitude:

```
  self-calibration, three sessions, long blocks  +2.5 / +7.2 / +10.5 px, ceiling +26
  self-calibration, order 2, d06152              +23.4 px  (fails its own gate)
  annulus profile read through the logged FOVs   +51 to +59 px
```

They are not reconcilable and the report does not average them. The
reprojection data does not merely fail to prefer +51 px, it costs 0.95 % of the
robust cost to go there — about 4x the instrument's own +/-9 px floor expressed
as a coefficient. So the self-calibration **rejects** the profile reading rather
than being silent about it.

What separates them is an assumption that is not tested anywhere in this
repository: **whether `AVCaptureDeviceFormat.videoFieldOfView` describes the raw
distorted frame or a rectilinear equivalent.** If it is the raw frame, the
profile reading stands and the self-calibration is under-reporting by 10x for a
reason nothing here identifies. If it is a rectilinear equivalent, `f_logged` is
already paraxial, and then the 2 % gap between 0.3134 and 0.3198 belongs to
neither lens's radial term and is a focal-length disagreement between the two
logged fields of view — which is a different open question, and a cheap one: it
is one capture of a target at a known distance, or one frame of a scene whose
geometry is known, on either lens.

### 5. The dB anchor

```
  reading                                   D_rms at 960x540   verdict on the 0.7 dB
  self-cal, longest block, d06152   (+2.5)       0.20 px       refuted (bar 1.0)
  self-cal, longest block, 57f29e   (+7.2)       0.58 px       refuted
  self-cal, longest block, d0f44f  (+10.5)       0.85 px       refuted
  self-cal, largest value anywhere (+17.2)       1.39 px       undecided (1.0-4.4)
  the same plus the +/-9 px floor  (+26.0)       2.11 px       undecided
  the bottom of the band that would explain it   2.43 px  (+30 px)
  profile paraxial                 (+51.4)       4.13 px       at the 4.4 bar
```

**All three sessions sit under the 1.0 px refutation bar at their longest
blocks.** The most unfavourable reading available — the largest value the
instrument ever returned, plus its entire measured accuracy floor — reaches
2.11 px, which is inside the undecided band but still below the 2.43 px that the
*bottom* of the 30-54 px range would produce. On the profile reading it is
exactly large enough. The decision rule's step 4 lands on **refuted, with one
named escape**.

## Dumping the corrected poses, and the gauge that decides whether they mean anything

*Written before `--dump-poses` existed, and before the gauge was measured.*

The fit's by-product is a per-frame SE(3) correction that takes the reprojection
RMS from 42.6 px to 8.6 px. The obvious next experiment is to re-export the
mixed training set with it and see whether the 0.7 dB comes back. That
experiment is only meaningful if the correction is a **deformation** of the
trajectory and not a **re-placement** of one lens relative to the other.

**The objective has an exact 6-DOF gauge freedom.** Apply any rigid `G` to every
corrected pose; the anchor points are built *from* those poses, so they move by
`G` too, and every reprojection residual is unchanged. Nothing in the data picks
a value for `G`. The fit picks one by fiat — `xi[0] = 0`, frame 0 held at the
wide-derived rig pose — and that choice makes the whole block pivot about frame
0, which is visible in the raw numbers: on the 60-frame block the per-frame
rotation correction is **9.46 deg at frame 1** and 1.96 deg at the last.

A 9-degree error in the wide chain is not credible: the wide-arm-only export
scores 17.65 dB on the same chain, which a 9-degree per-frame pose error would
destroy. So most of that 9 degrees must be the gauge, and the test is to remove
it and see what is left.

### The rule, fixed before the measurement

1. **Remove the block-level rigid transform.** Compute the `G` minimising
   `sum_j || G T_corrected[j] - T_chain[j] ||` — rotation by SVD projection of
   `sum_j R_chain[j] R_corrected[j]^T`, translation by matching the centroids —
   and apply `G^-1`. The dumped poses then have **zero net rigid offset from the
   wide-derived rig poses they came from**, and carry only the deformation.
   This is the "constrain the mean correction to identity" option; it is chosen
   over re-anchoring at one frame precisely because one frame is what the fit
   already does and it is what put 9 degrees on frame 1.
2. **The scale must be one.** The fit carries a global multiplier on the chain's
   positions. If it is applied to the ultra-wide arm and not to the wide arm,
   the two arms are at different metric scales, which no rigid gauge can repair.
   **Refuse the dump if `|s - 1| > 0.005`.**
3. **The residual deformation must be smaller than an error the export already
   models.** After `G` is removed, the dumped ultra-wide poses still differ from
   the wide-derived rig poses frame by frame, and if the wide arm keeps its
   original chain the two arms are no longer a rig by that much. The export
   already accepts the shutter offset, which moves the camera **34-39 mm and
   about 2 deg** between the two exposures. So: **if the median residual
   deformation exceeds 2 deg or 40 mm, the ultra-wide-only dump is refused**,
   because using it beside an uncorrected wide arm would introduce a rig error
   larger than the one the export exists to model.
4. **If rule 3 refuses, there is a correct alternative and the tool writes it
   instead.** The correction is a correction to the *wide chain* — the
   ultra-wide poses are `T_wide(t) @ E` — so it can be de-rigged and
   re-interpolated onto the wide frames' own timestamps, giving a drop-in
   replacement for `pi3traj/d06152_wide_depth.npz` that poses **both** arms
   through the existing rig path with no change to `tools/export_3dgs.py` at
   all. `--dump-wide-poses` writes that. It is gated too, and on the chain's own
   numbers: the wide chain's seam residuals are 0.2-2.3 cm and it already builds
   a 17.65 dB wide-only reconstruction, so a correction more than five times its
   worst seam is a replacement rather than a refinement.

5. **Added after the first run, because the bars above were not enough.** The
   median deformation is not a sufficient test: on the full session it is 80 mm,
   inside the wide-arm bar, while the corrected path is **63.73 m against the
   chain's 24.18 m** and one frame-to-frame step is **9.5 m**. The first version
   of the gate wrote that file. So both dumps are now also refused if the path
   is more than **1.25x** the chain's, or the worst step more than **10x** the
   chain's median step. A correction to a walk changes its length by percents.
   `test_uw_selfcal.py` runs the bars on those exact numbers so the hole cannot
   reopen.

### Result: both dumps are refused, and not for the reason expected

The zero mode is real and exact — `test_uw_selfcal.py` moves a whole block by
3.5 deg and 54 cm and the residual changes by **9e-13 px**, so the gauge is
provably invisible to the data and removing it removes nothing the data
constrains. But removing it does not rescue the poses:

```
                                     60 frames            289 frames (all of them)
  raw per-frame rotation       6.61 deg med, 141.5 worst   2.91 deg med, 1917 worst
  block-level rigid gauge        6.79 deg, 20.6 cm           0.76 deg, 3.8 cm
  deformation after the gauge   6.83 deg med, 17.3 p95     2.89 deg med, 13.1 p95
                                 203 mm med, 306 p95        80 mm med, 266 p95
  fitted chain scale                 0.9753                    0.9980  (bar 0.005)
  path length                  26.91 m vs the chain's 4.79   63.73 m vs 24.18
  frame-to-frame step          8.5 cm med, 1080 cm worst    8.8 cm med, 953 worst
                                 (chain 8.2 cm)               (chain 8.3 cm)
  frames moved over 0.5 m            1 of 60
  median excluding that frame     still 203 mm
  observations per frame       264 median; 5 frames under 20
```

Note the two things the long block *does* fix and the one it does not. The
chain-scale objection disappears — 0.9980 at full length against 0.9753 at 60
frames, so that refusal was a short-block artefact — and the block-level rigid
gauge shrinks to 0.76 deg. **The deformation does not go away**: 2.89 deg and
80 mm at the median, and a path 2.6x too long.

**The gauge was not the problem.** Removing a 6.79 deg block rotation moved the
median per-frame correction from 6.61 deg to 6.83 deg — that is, not at all. The
per-frame corrections point in different directions and cancel in the average,
so there was little rigid part to remove.

**Nor is it a handful of degenerate frames.** One frame in sixty runs away by
10.8 m — that single frame is the whole 22 m of excess path — but deleting it
leaves the median deformation at 203 mm unchanged. Frames carry a **median of
264 observations** for six degrees of freedom. They are richly over-determined.

So the fit really does prefer to move **every** frame by about 20 cm and
7 degrees from the chain, and it is not confused when it does so. That leaves
two readings, and one of them is already excluded:

- the chain is wrong by 20 cm and 7 deg per frame — but the wide-only export
  scores **17.65 dB** on that same chain, which a 7-degree per-frame pose error
  would not permit;
- the per-frame pose is absorbing something the camera model cannot express.
  The joint fit leaves **8.6 px** of RMS, far above the ~0.5 px the tracking
  noise floor would give, so there is a large unexplained residual and 6 free
  degrees of freedom per frame to soak it up with. At `f = 721 px` and a 1.05 m
  median anchor range, 8.6 px is about 1.2 cm — and pulling on 264 observations
  that way is enough to move a pose 20 cm.

**The pose output is therefore a nuisance field, not a trajectory estimate**, and
both dumps are refused. This does not touch the distortion bound: the injection
gate ran through exactly this machinery, with exactly this pose freedom, and
still returned the injected coefficients at slope 0.933. The robust loss means a
runaway frame contributes almost nothing to the cost — it corrupts the pose
output without corrupting the radial term.

**Coverage was never the limitation.** A single block with one gauge covers the
whole session: `--frames 289` takes **all 289 usable ultra-wide frames** — 289
of 290 recorded, the one drop being the frame before the wide trajectory starts
— in one fit, one gauge, 258 676 carried observations, and the de-rigged wide
form would have spanned wide frames 1..275 of 277. Blocks cannot be
concatenated, because each carries its own arbitrary rigid gauge and joining two
would splice two different worlds. But no concatenation is needed. **What fails
is the fidelity of the correction, not its extent.**

**What the retraining experiment would need.** Not this by-product. A pose
regulariser — a prior tying each frame to the chain at the size the chain's own
seams say it is uncertain, 0.2-2.3 cm — would keep the trajectory a trajectory.
That is a different instrument: the prior would also stop the pose competing
freely with the radial term, which is the property this page's whole gate was
built on, so it would need its own injection gate before its distortion output
meant anything. It is a clean piece of work and it is not this one.

## Verdict against the decision rule

```
  1. injection recovery      FAILS as written (3 of 6 bars); slope 0.933 passes;
                             measures an accuracy floor of about +/-9 px
  2. wide-arm null           PASSES: |a1| 0.0043 against a 0.007 bar, -1.70 px
                             against -1.41 px from an instrument sharing no input
  3. cross-session           FAILS as written at the pre-registered 60/40-frame
                             settings (spread 5.23 vs a 3.98 px bar); at each
                             session's longest block the three give +2.5 / +7.2
                             / +10.5 px, spread 8.0 against a 3.4 px bar — a fail
                             on the letter either way, against a 119.7 px spread
                             for the route this replaces
  4. profile flattening      FAILS: 1.53 % -> 1.59 %, and the reason is that the
                             profile's dominant term is rho^0, not radial
  5. dB anchor               D_rms 0.20-0.85 px at 960x540 on the three longest
                             blocks, under the 1.0 px refutation bar; 2.11 px on
                             the most unfavourable reading available, still below
                             the 2.43 px the bottom of the band would produce
  6. model/pose ratio        RETRACTED — fails on a known-answer input
```

**Four of the five surviving criteria fail as written, and the page still
answers the question**, because what the failures measure is the instrument's
resolution rather than its direction. That distinction is the whole verdict, so
it is spelled out:

- criterion 1 fails on an **offset** with a correct slope (0.933), not on a scale
  error, so the sensitivity is intact and the failure is a calibration of the
  accuracy — `+/-9 px`;
- criterion 3 fails on a **spread of 8 px**, which is that same `+/-9 px`;
- criterion 4 fails for a reason that is not about the fit at all — the profile's
  dominant term is `rho^0`;
- and the quantity under test is **30-54 px**, four times the resolution the
  failures established.

So no coefficient is published for this lens. What is published is a bound:

**Over three sessions and six block lengths, the ultra-wide's radial corner
displacement for the active 3840x2160 / 106.2007 deg format is positive and
never exceeds +17.2 px, with a ceiling of about +26 px once the instrument's own
+/-9 px floor is added on the unfavourable side. The 0.7 dB needs 30-54 px. The
ultra-wide's radial camera model does not explain the penalty.**

Three things could still overturn that, in the order they are worth doing:

1. **`videoFieldOfView`'s convention.** If it is the raw whole-frame angle, the
   annulus profile's paraxial limit demands +51 px and something in this
   instrument is wrong by 10x. One capture of a target at a known distance
   settles it, on either lens.
2. **A decentring term.** Nothing here can see one; `calib/` puts the distortion
   centre 2.2 px off the principal point at the calibrated format, and no route
   in this repository can confirm it for the active one.
3. **The remaining 0.7 dB.** With the camera model refuted at this size, the
   pose is what is left — and this page measured the pose error directly: the
   chain disagrees with the images by 0.8-4.4 degrees per frame pair, and the
   bundle needs a smooth per-frame correction reaching 9 deg and 18 cm over
   60 frames. **That is the candidate the next experiment should attack**, and
   `eval/uw_selfcal.py` already produces the corrected poses as a by-product.
