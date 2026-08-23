# Pre-registration: the ultra-wide's distortion from a checkerboard, by straightness

Written **before** `tools/fit_uw_board.py` existed and before any coefficient was
fitted from these captures. Model, detector, gates, thresholds and the decision
rule are fixed here so the answer cannot be tuned into existence afterwards. The
Results section at the bottom was empty when this file was first written.

Everything above the Results heading is frozen. What was measured before it was
written is stated explicitly in "What was measured first" — coverage and the
straightness noise floor, never a coefficient.

## Why a fourth instrument

`docs/3DGS.md` measures that adding the raw ultra-wide arm to a training set
costs the *same wide photographs* **0.7 dB**. On that page's own anchor — ten
pixels of texture displacement is worth 1.6 dB — 0.7 dB is 4.4 px, i.e. a corner
displacement of roughly **30-54 px** at 3840x2160. Three instruments have tried
to reach it:

- **the lens-to-lens similarity fit** (`eval/uw_radial_profile.py`) sees only the
  inner 57 % of the radius, because that is all the wide arm overlaps;
- **the plumb-line fit** (`tools/fit_uw_distortion.py`) reaches the corner but
  was **refused** on the ultra-wide: indoor rooms yield a median chain of 0.16 of
  the frame diagonal, where a 110 px corner displacement bends a line by 0.12 px
  against a **1.44 px** straightness floor. Three sessions came back
  `-118.2 / +1.5 / -7.2 px`, no fit lowered the residual, and one lens cannot
  have three distortions;
- **the LiDAR-anchored self-calibration** (`eval/uw_selfcal.py`) bounds it from
  the other side: across three sessions and six block lengths the fitted corner
  displacement is always positive and **never exceeds +17.2 px**, with a measured
  accuracy floor of +/-9 px, i.e. a **ceiling near +26 px** — and concludes the
  camera model is too small to explain 0.7 dB.

`docs/UW_DISTORTION_PREREG.md` closed by naming exactly what would settle it:

> Anyone attempting it again needs frames built for it — a scene with long
> straight edges spanning at least 40 % of the frame diagonal, held still enough
> that the straightness noise floor drops below a pixel. A calibration target, or
> a deliberate slow pass along a corridor, would do it.

That capture now exists: six takes of a 60 mm 8x6-square checkerboard through the
multi-cam path, `20260823-0032xx` — `1e87b0` (51 frames), `098cbe` (34),
`161205` (22), `e52acc` (22), `0b4627` (22), `899c75` (13). This file is the
attempt the old one asked for.

**The factory table is not an input to any part of this**, in either direction,
for the reason in `calib/README.md` and `docs/POSE.md`: it is measured at
4032x3024 / 102.5 deg and the capture ran an active 3840x2160 / 106.2007 deg
format. It appears below only as a number to compare against.

## The property the data has, and what it forces

The full 7x5 interior grid is almost never in frame — the board runs off the
frame edge — so detections come back as sub-grids of every shape from 3x3 to
8x6. A sub-grid does **not** say which block of the board it is, so the 3D object
points are unknown up to an integer offset.

That does not touch straightness. **Every detected row and every detected column
is a set of collinear world points whatever sub-block it came from**, because the
board is planar and its lines are straight. So the primary instrument here is the
same plumb-line fit `tools/fit_uw_distortion.py` already implements, fed board
rows and columns instead of Canny edge chains.

The secondary focal-length reading in the last section does use correspondences,
and the gauge argument that makes that legal is stated there, together with the
synthetic control that has to pass before the number is quoted.

## The model being fitted

Unchanged from `docs/UW_DISTORTION_PREREG.md` and `docs/UW_SELFCAL_PREREG.md`, so
every number on this page is directly comparable with theirs:

```
  c      = (w/2, h/2)                    frame centre
  R      = hypot(w, h) / 2               half-diagonal; rho = 1 at the corner
  rho    = |p - c| / R
  p_rect = c + (p - c) * (1 + a1 rho^2 + a2 rho^4)
```

At 3840x2160, `R = 2203.06 px`. Reported: `a1` (`a2` secondary),
`D_corner = R (a1 + a2)`, `D_rms` evaluated at **960x540** because the mixed run
trained at `--downscale 4`, and `k1_opencv` for handoff. The distortion centre is
the frame centre, and focal length is not identifiable from straightness and is
not fitted — all three decisions carried over unchanged.

## The instrument

`tools/fit_uw_board.py`. It **imports** `ChainSet`, `fit_coefficients`,
`identifiability`, `bootstrap`, `corner_displacement`, `rms_displacement`,
`inject_distortion` and `compose_expected` from `tools/fit_uw_distortion.py` and
does not modify them. Only the chain *source* is new. `tools/fit_uw_distortion.py`
is not edited, so its existing behaviour reproduces byte-for-byte.

### Detection, frozen before any fit

1. `cv2.findChessboardCornersSBWithMeta(gray, (3,3), EXHAUSTIVE|ACCURACY|LARGER)`
   at working long sides **1920, 1280, 960** for the ultra-wide and **640, 480**
   for the wide arm. `LARGER` is required: without it the detector must be told
   the exact sub-grid, which is unknown per frame.
2. **Decoy rejection.** The monitor displaying the board also shows a checkerboard
   *thumbnail icon* in the desktop dock, and at full resolution the detector
   prefers it — it is a denser grid. Any detection whose median corner spacing is
   below **0.01 x the frame diagonal** in native pixels (44 px at 3840x2160, 8 px
   at 640x480) is rejected. The icon runs at ~10 px at 3840x2160; the board runs
   at 400-750 px.
3. **Grid gate.** Fit a DLT homography from the integer lattice `(j, i)` to the
   detected corners. Require `max |residual| <= 0.12 x median spacing`. If it
   fails, **peel**: drop whichever boundary row or column most reduces the max
   residual, down to a 3x3 minimum, and re-test; reject the detection if it still
   fails. This is what catches the detector extending a row onto the monitor
   bezel, and the outright bogus grids it lays across the ceiling.

   **0.12 cannot reject a genuine detection.** Measured on synthetic boards —
   random pose, random sub-grid size 3x3 to 8x8, random distance, spanning at
   least 0.15 of the diagonal, 300 trials per coefficient — the largest
   `max-residual / spacing` a *true* radial distortion produces is:

   ```
     a1      D_corner    median    p95     max      (over 210-230 valid trials)
     0.00       0.0 px   0.0000   0.0000   0.0000
     0.01     +22.0      0.0015   0.0081   0.0133
     0.02     +44.1      0.0032   0.0149   0.0231
     0.03     +66.1      0.0035   0.0235   0.0712
     0.05    +110.1      0.0065   0.0354   0.0876
     0.08    +176.2      0.0126   0.0593   0.2487
   ```

   So the gate sits 3.4x above the p95 at a coefficient twice the largest answer
   anyone has proposed, and 2x above it at `a1 = 0.08`, which is larger than any
   value this repo has ever entertained.
4. Among the working scales that survive, keep the detection with the most
   corners.
5. **Subpixel refinement at native resolution.** `cv2.cornerSubPix` with a window
   of `clip(round(spacing / 12), 5, 21)` px; a corner that moves more than
   `0.5 x window` is reverted to its seed. The unrefined seeds are also fitted
   and reported as a sensitivity, so the refinement is a check rather than a knob.

### Chains

Every detected **row** of at least 3 corners and every detected **column** of at
least 3 corners becomes one chain, in native pixel coordinates. No chord gate and
no smoothness gate is applied — the grid gate in step 3 already did that job, and
adding a length gate would preferentially delete the short high-radius rows that
carry the corner. Sensitivity at `chord >= 0.20` and `>= 0.30` of the diagonal is
reported, exactly as `docs/UW_DISTORTION_PREREG.md` reported it, as a diagnostic
and not as a result.

The fit runs in **native** coordinates (3840x2160 and 640x480). `a1` is
resolution-independent, so this is the same number the old tool reports from its
960-px working scale; residuals are quoted both natively and rescaled to a
960-px long side so they can be read against the old **1.44 px** floor.

### The objective is not touched

`fit_coefficients(chains, order=1)` — coarse grid on `a1` over `[-0.4, +0.4]`,
then Nelder-Mead, Huber with `delta` = the median residual at `a = 0`, monotonicity
enforced to `rho = 1.05`, residual divided back through the local Jacobian of the
radial map. Amendment 4 of `docs/UW_DISTORTION_PREREG.md` explains why any simpler
objective is exploitable; none of it is re-litigated here.

## What was measured first, before this file was written

Coverage and the noise floor, never a coefficient. Detector settings above were
fixed by these numbers; the fit had not been run.

```
  arm            frames   detections     median span   frames rho_max>0.7  >0.9
  ultra-wide       164    144  (88 %)     0.595 diag          99            23
  wide 640x480     163     85  (52 %)     0.641 diag          69             7

  arm            chains   median chord   chains rho>=0.6   median residual at a=0
  ultra-wide      1439     0.311 diag         172          1.81 px  (0.45 px at 960)
  wide 640x480     689     0.385 diag         124          0.22 px  (0.33 px at 960)
```

Against the capture that was refused: median chain **0.311 of the diagonal**
where the rooms gave 0.16, and a straightness floor of **0.45 px** at the 960
working scale where the rooms gave **1.44 px**. Per chain that is roughly an
order of magnitude more signal against noise. All four sufficiency gates of
`docs/UW_DISTORTION_PREREG.md` — 30 contributing frames, 300 chains, 80 chains
past `rho = 0.6`, median chord 0.12 of the diagonal — are met on both arms and
are re-checked by the tool rather than asserted here.

## Acceptance criteria

Every threshold below is fixed now.

### 1. Injection recovery — the gate

Warp **every** ultra-wide frame by a known radial map with
`inject_distortion`, then re-run the **whole** pipeline on the warped frames —
detection included, not just the fit — and demand the coefficient back. Chains
with any corner outside the warp's validity mask are dropped.

The quantity demanded back is not the injection but the **composition** of the
injection with whatever the lens already does, `compose_expected(baseline_fit,
injection, order)`, which `tools/fit_uw_distortion.py` already computes. Both the
composition and the plain difference from the baseline are reported; the
composition is the bar.

Injections, as corner displacements at 3840x2160, chosen to bracket the 30-54 px
band the 0.7 dB question lives in — the same six `eval/uw_selfcal.py` used, so
the two instruments are graded on the same sweep:

```
  -80, -40, -20, +20, +40, +80 px      (a1 = D / 2203.06)
```

Pass bars, all required:

- **sign** correct on all six;
- **per-injection error** `|D_recovered - D_expected| <= max(5 px, 0.15 |D_injected|)`;
- **response slope** of recovered on injected, through the six points, in
  `[0.85, 1.15]`. `tools/fit_uw_distortion.py` returned **0.745** on ultra-wide
  room frames and that was read as a failure; the bar is set where a failure of
  that size is caught.

For reference, the same tool's existing injection control on *wide* frames
recovered 8 of 8 at 0.75-4.28 px against injections of 17-140 px.

**Nothing below this line is readable if this gate fails.**

The wide arm gets its own sweep at `+/-15` and `+/-30` px at 640x480
(`a1 = D / 400.0`), on the same bars. A null the instrument could not have failed
proves nothing, so if the wide sweep misses its bars the wide null in criterion 2
is recorded as uninformative rather than as a pass.

### 2. Wide-arm null

Run the same fitter on the wide arm's board detections at 640x480. Two
instruments that share no input already agree the wide arm's residual distortion
is small — `tools/fit_uw_distortion.py` gives **-1.41 px** and
`eval/uw_selfcal.py` gives **-1.70 px** — and both reject the factory table's
**+15.6 px**. The wide board coverage here is real: 85 detections, 689 chains,
7 frames past `rho = 0.9`.

Pass bar, carried over verbatim from `docs/UW_SELFCAL_PREREG.md` so the two
instruments are graded identically:

```
  |a1 + a2|_wide  <  0.007          (2.8 px at 640x480)
```

and the point estimate is additionally required to sit within **3.0 px** of the
`-1.55 px` mean of the two prior instruments, i.e. in `[-4.55, +1.45]`. Landing
far from `-1.4 / -1.7` means the fitter is wrong and the ultra-wide number is
void.

### 3. Cross-take agreement

Fit the six takes independently — separate frames, separate detections, nothing
shared but the code. One lens cannot have six distortions; the previous route was
refused for exactly this.

```
  spread = max(D_corner) - min(D_corner) over the six takes
```

Pass bars, both required:

- all six share **sign**;
- `spread < 0.5 x |mean D_corner|`, the bar `docs/UW_SELFCAL_PREREG.md` set.

The older `docs/UW_DISTORTION_PREREG.md` bar — spread under 30 % of the mean — is
reported alongside. A per-take bootstrap over frames (200 resamples) gives a
5-95 % band; **if a take's band spans zero, no coefficient is claimed for that
take.** Takes that fail the four sufficiency gates are fitted and reported but
excluded from the spread, and both the all-six and gated-subset spreads are
printed.

### 4. Straightness residual must fall

No fit was ever accepted on this lens because none lowered the residual. Report,
over accepted chains, the median / mean / p90 per-chain RMS straightness residual
in native pixels and at the 960 working scale, at `a = 0` and at the fit, plus the
Huber cost drop and the identifiability band.

Pass bars, both required:

- the **median** residual falls;
- the Huber cost drops by at least **3 %**. That number is not invented here: it
  is the discriminator `docs/UW_DISTORTION_PREREG.md` established from its
  injection controls — a real distortion drops the cost by **3-24 %**, and where
  none is present the drop is **0.1-0.3 %**. Every ultra-wide room session scored
  0.0-1.0 % and was refused.

### 5. The answer, against the +17.2 px bound

`eval/uw_selfcal.py` bounds the ultra-wide's corner displacement at **never above
+17.2 px, ceiling ~+26 px** once its own +/-9 px floor is added, and concludes the
camera model is too small to explain 0.7 dB, which needs **30-54 px**. This
instrument either confirms that bound or contradicts it, and the report says
which, plainly. A contradiction is a finding, not an error to hide.

The rule, fixed now, on `|D_corner|` at 3840x2160 with its identifiability band:

```
  |D| <= 26 px and the band does not require more    -> CONFIRMS the bound
  |D| >= 30 px and the band excludes +/-26 px        -> CONTRADICTS the bound
  otherwise                                          -> the two instruments
                                                        overlap; report the
                                                        overlap and say so
```

`D_rms` at 960x540 is put on `docs/UW_DISTORTION_PREREG.md`'s own dB scale
unchanged: `>= 4.4 px` accounts for the 0.7 dB, `<= 1.0 px` refutes it, between
is inconclusive on magnitude.

The stated limitation of the old page carries over unchanged: the 0.7 dB was lost
by the *wide* holdouts, so a large ultra-wide displacement is a
sufficient-magnitude argument, not a demonstration of the mechanism.

### 6. Secondary, and only if its own control passes: the paraxial focal length

Straightness cannot see focal length — a uniform scale takes straight lines to
straight lines — so `f` comes from a separate, correspondence-based fit, reported
as a cross-check and never as part of the headline.

**Why sub-grids are legal here.** Zhang's constraint is on the first two columns
of `H = K [r1 r2 t]`. An unknown integer offset `d` inside the board plane sends
object points `X -> X + d`, hence `H -> H T` with `T` a pure in-plane
translation, whose effect is `h1 -> h1`, `h2 -> h2`, `h3 -> h3 + dx h1 + dy h2`.
The intrinsic constraints are untouched: **the offset is a gauge, absorbed by the
extrinsic**, so `cv2.calibrateCamera` may be run on sub-grids with object points
`(j s, i s, 0)` and an arbitrary origin. The same holds for a 90-degree relabel of
rows against columns, which is a rotation about the plane normal. What would
*not* be absorbed is a **reflection**, so the sign of `cross(d_col, d_row)` in
image coordinates is checked across every detection and must be constant.

Two controls, both pre-registered, both of which must pass before any `f` is
quoted:

- **synthetic sub-grid recovery.** Generate views of a known board through a known
  `f` and `k1`, cut a random sub-block out of each with a random in-plane offset
  and a random 90-degree relabel, and run the same `calibrateCamera` call. It must
  return `f` within **1 %** and the implied corner displacement within **5 px**.
- **constant handedness** across the real detections, as above.

Reported: `fx`, `fy`, the principal point, and the corner displacement implied by
the fitted `k1` — obtained by solving `R = r_u (1 + k1 (r_u/f)^2)` for `r_u` and
taking `r_u - R`, which is exactly this page's `D_corner` convention — against the
straightness answer. `tools/uw_field_angle_transfer.py` derives **1466.0 px** for
the active format from the pitch ratio between the two readouts and corroborates
it to 0.5 % against the two-lens scale; `(W/2)/tan(hfov/2)` on the logged
106.2007 deg gives **1441.56 px**, which is the paraxial misreading. The board is
a third route and is either consistent with 1466.0 or it is not.

## Threats this cannot remove, stated in advance

- **The target is a monitor, not a printed board.** An LCD panel is flat to well
  under a pixel at this magnification and its rendered lines are straight, so it
  is a legitimate plumb-line target; but the *square size in millimetres* is a
  display zoom, not the 60 mm the file name claims. Nothing on this page depends
  on the physical square size: straightness does not use it, and Zhang's
  intrinsics are invariant to it (it scales only the extrinsic translation). No
  metric quantity is claimed from these takes.
- **Rolling shutter.** These are handheld passes and a rotating rolling-shutter
  camera does bend straight lines. It is not modelled. The defence is that the six
  takes have independent motion, so it should not survive criterion 3 with a
  consistent sign — which makes criterion 3 the test for it, and a failure there
  must be read as possibly rolling shutter rather than as noise.
- **The 0.7 dB mechanism.** Unchanged from the old page: magnitude only.

## Amendments, all made after the fits ran and all declared

Recorded rather than quietly applied. Each one can only refuse a number or add a
reading; none of them turns a refusal into an answer.

1. **A reprojection-RMS floor on the secondary focal-length fit.** The ultra-wide
   calibration came back at a **41.5 px** reprojection RMS. A planar board
   through a pinhole-plus-`k1` camera reprojects to well under a pixel — the
   wide arm does it at 0.97 px and the synthetic control at 0.26 — so a fit that
   cannot is not describing this camera. `mode_focal` now refuses to quote a
   focal length above 2 px of reprojection RMS. This can only withhold a number.

2. **The subpixel refinement is reported as a variant, not replaced.** On the
   ultra-wide the pre-registered `cornerSubPix` step *raises* the straightness
   floor from 0.547 px to 1.804 px, i.e. it is worse than the detector's own
   corners; on the wide arm the two agree to 0.007 px. Rather than pick the
   better one after the fact — which would select the variant with less
   curvature and bias the answer toward zero — **both are carried through every
   criterion below**, and criterion 1 grades them. Both pass, and they bracket
   the ultra-wide answer at `+1.70` and `+9.62 px`.

3. **Run-to-run repeatability is reported.** `findChessboardCornersSB` with
   `CALIB_CB_LARGER` is not bit-reproducible across runs; three independent runs
   of the same pooled fit gave `+9.17 / +9.39 / +9.62 px` on the ultra-wide and
   `+12.10 / +12.10 / +12.14 px` on the wide. That 0.45 px and 0.04 px are part
   of the error budget below.

## Results

**Headline, in three parts.**

1. **The ultra-wide is a null, and the +17.2 px bound is CONFIRMED.** The
   instrument recovers injected coefficients on these very frames 6 of 6 at
   0.6-3.1 px against injections of 20-80 px, slope **1.039** — where the old
   room-frame plumb-line managed 0.745 — and then finds **`D_corner = +9.62 px`**
   (`+1.70 px` on the seed variant), with the cost dropping **0.31 %** and the
   straightness residual not moving. That is the pre-registered signature of *no
   distortion present*, produced by an instrument that has just demonstrated it
   would have seen 20 px. `D_rms` at 960x540 is **0.78 px**, under the 1.0 px
   refutation bar. The camera model is refuted as the explanation of the 0.7 dB,
   by a fourth instrument, and it agrees with `eval/uw_selfcal.py`'s three
   sessions (`+5.1 / +10.3 / +8.6`, mean `+7.97`) to about a pixel.

2. **The wide arm is not null, and criterion 2 fails — which is the finding.**
   The board says **`+12.10 px` at 640x480** at order 1, **`+16.63 px`** at order
   2, rising monotonically to `+15.86 px` as the fit is restricted to outer
   chains, with the cost dropping **34-75 %** and the median residual collapsing
   from 0.638 px to 0.079 px. An independent correspondence fit on the same
   corners returns **`+11.0 px`**. `tools/fit_uw_distortion.py`'s `-1.41 px` and
   `eval/uw_selfcal.py`'s `-1.70 px` are both wrong, and **the factory table's
   `+12.5 / +15.6 px` for this arm is right**.

3. **The reason the ultra-wide's rooms never worked is measurable here, and it
   is not the scene.** The ultra-wide's board detections carry a **5.23 px**
   median non-projective residual against the wide arm's **0.61 px** on the same
   board at the same instant — except on the one take shot nearly still, where
   the ultra-wide's residual falls to **0.59 px**. It is rolling shutter, it
   scales as focal length times readout height, and it is what makes six takes
   of one lens disagree by 92 px.

### What the instrument saw

```
  arm            frames   detections   chains   median chord   chains rho>=0.6   peeled
  ultra-wide       164      143 (87%)   1431     0.309 diag          171           30
  wide 640x480     163       85 (52%)    689     0.385 diag          124            4
```

All four pre-registered sufficiency gates pass on both arms — 30 contributing
frames, 300 chains, 80 chains past `rho = 0.6`, median chord 0.12 of the
diagonal. Against the capture that was refused, the median chain is **0.31 of
the diagonal** where the rooms gave 0.16, and the straightness floor is
**0.45 px** at the 960 working scale where the rooms gave **1.44 px**.

### 1. Injection recovery — PASS at order 1 on both arms; order 2 fails

Every frame warped, detection re-run from scratch, chains outside the validity
mask dropped. The bar is the composition, `max(5 px, 0.15 |D_inj|)`.

```
  ultra-wide, order 1, subpix corners                ultra-wide, order 1, seeds
    inj    expected  recovered   error   bar           expected  recovered  error
   -80.0    -71.58     -74.41    -2.83  12.00           -78.44    -80.03    -1.59
   -40.0    -31.21     -34.31    -3.10   6.00           -38.37    -40.30    -1.93
   -20.0    -11.02     -13.05    -2.03   5.00           -18.34    -19.43    -1.09
   +20.0    +29.37     +30.06    +0.69   5.00           +21.73    +24.50    +2.77
   +40.0    +49.57     +50.61    +1.04   6.00           +41.77    +44.55    +2.78
   +80.0    +89.98     +90.58    +0.60  12.00           +81.84    +83.95    +2.11
   slope     1.039   (bar 0.85-1.15)   6/6 pass          1.035               6/6 pass

  wide 640x480, order 1, subpix corners              wide, order 1, seeds
   -30.0    -19.90     -20.17    -0.27   5.00           -19.70    -19.57    +0.14
   -15.0     -3.93      -4.33    -0.40   5.00            -3.72     -3.77    -0.05
   +15.0    +28.19     +26.66    -1.52   5.00           +28.44    +27.79    -0.65
   +30.0    +44.35     +41.44    -2.90   5.00           +44.62    +42.96    -1.67
   slope     1.028                       4/4 pass         1.044               4/4 pass
```

**PASS on all four order-1 variants**: every injection inside its bar, every
sign correct, every slope inside `[0.85, 1.15]`.

The single most important row is `-20 / +20 px` on the ultra-wide. The question
this page exists to answer is whether the corner displacement is 30-54 px, and
the instrument returns a 20 px injection to within 2.8 px on the frames the
answer is read from. `tools/fit_uw_distortion.py` on room frames returned a
slope of 0.745 and was refused; the same estimator on board chains returns
1.039. **The capture, not the fitter, was the problem, and the capture fixed it.**

**Order 2 fails the gate on the ultra-wide** — slope 1.189 (subpix) and 1.203
with 2 of 6 injections outside their bars (seeds). It passes on the wide arm
(1.123 and 1.104). So the ultra-wide's `a1+a2` fit is disqualified by criterion
1 before the pre-registered 30 % agreement rule is even consulted, and the
`+51 px` it returns is not readable. On the wide arm order 2 is readable.

### 2. Wide-arm null — FAILS as written, and the failure is the result

```
  order   a1 (+a2)              corner at 640x480   band          bootstrap      cost drop   residual
    1     +0.03024                 +12.10 px       [+10.4,+13.6]  [+11.1,+12.8]    34.1 %   0.220 -> 0.213
    2     -0.02601 / +0.06758      +16.63 px       [+15.4,+17.8]      -            50.8 %   0.220 -> 0.159
```

Against the pre-registered bar `|a1 + a2| < 0.007` the fit returns **0.0302**, a
factor of 4.3 over, and the point estimate misses the `[-4.55, +1.45]` window by
10.6 px. **Criterion 2 fails.**

The pre-registration says that means the fitter is wrong and the ultra-wide
number is void. It does not, and criterion 1 is why: the wide arm's own
injection sweep passes 4 of 4 with slope 1.028, so this fitter demonstrably
returns what is put into these wide frames. What fails is the *expectation*.

Everything about this fit reads as a measurement rather than a basin:

```
  restriction              chains   corner    cost drop   median residual before -> after
  all chains                 698    +12.14      34.3 %        0.218 -> 0.213
  chains with rho >= 0.3     562    +12.19      39.6 %        0.261 -> 0.227
  chains with rho >= 0.4     410    +12.71      51.7 %        0.307 -> 0.193
  chains with rho >= 0.5     252    +14.38      74.3 %        0.485 -> 0.108
  chains with rho >= 0.6     127    +15.86      75.3 %        0.638 -> 0.079
  chord >= 0.30              517    +12.22      38.5 %        0.266 -> 0.238
  seed corners, no subpix    689    +12.33      39.4 %        0.213 -> 0.207
  five moving takes          516    +12.66      44.1 %        0.259 -> 0.207
```

The residual **collapses to 0.079 px** on the outer chains, and the coefficient
rises with radius exactly as a lens needing an `r^5` term does — which is what
the order-2 fit's `+16.63 px` says directly. Compare the same tool on this arm's
room frames: a 0.1 % cost drop and a flat cost to 1e-4.

**So the factory table is right about the wide arm and three pages of this repo
are wrong about it.** The comparison, at 640x480:

```
  tools/fit_uw_distortion.py, room frames, plumb-line       -1.41 px   cost drop 0.1 %
  eval/uw_selfcal.py, LiDAR-anchored tracks                 -1.70 px
  this instrument, board rows and columns, order 1         +12.10 px   cost drop 34.1 %
  this instrument, order 2                                 +16.63 px   cost drop 50.8 %
  this instrument, outer chains only                       +15.86 px   cost drop 75.3 %
  this instrument, correspondence fit (Zhang, k1)          +10.99 px   reprojection 0.97 px
  what the factory table demands   docs/UW_DISTORTION_PREREG.md   +12.5 px
                                   docs/3DGS.md, docs/UW_SELFCAL_PREREG.md   +15.6 px
```

`docs/3DGS.md` already carried the correction that makes this readable: *"the
plumb-line cost is flat to 1e-4 over that arm... The -1.41 and +3.29 px are the
centre of a flat basin, not a measurement of near-zero distortion."* They were.
`eval/uw_selfcal.py`'s `-1.70 px` agreed with a flat basin to 0.3 px, which its
own page called "the strongest corroboration in this whole page" — two
instruments can agree and both be reading a basin, and that is what happened.

Three consequences follow immediately and none of them is about the ultra-wide:

- `docs/3DGS.md`'s *"the wide arm's [table] is too strong, by a factor of 1 to
  2.4"* is contradicted; measured, it is right to within 0.4-1.0 px.
- The escape `docs/3DGS.md` left open — the annulus profile read through the two
  logged fields of view puts the ultra-wide at `+51 to +59 px`, *"but only if
  ... the wide lens carr[ies] -12.6 px against two independent measurements of
  -1.4 and -1.7"* — is closed in the direction that kills it. The wide lens
  carries **+12.1 to +16.6 px**, the opposite sign.
- `eval/uw_radial_profile.py`'s annulus profile is a ratio of two lenses, and the
  wide half of that ratio has been wrong by 13-18 px throughout.

### 3. Cross-take agreement — FAILS on both arms, for two different reasons

```
  ultra-wide                                           wide 640x480
  take     det chains rho_med out60   corner   drop      det chains out60   corner   drop
  1e87b0    42   379   0.443    71    +4.62   0.11%       19   123    33   +13.19  64.9%
  098cbe    29   284   0.380    23    -8.72   0.27%       19   135    35   +12.52  54.1%
  161205    20   201   0.386    14    -9.30   0.28%       12    91    25   +13.00  25.1%
  e52acc    18   198   0.376    25   +41.79   3.42%       12    90    13    +7.80  21.5%
  0b4627    20   184   0.424    38   +34.24   4.23%       10    68    18   +13.08  51.4%
  899c75    13   182   0.179     0   +82.68  38.18%       13   182     0    -1.98   1.1%

  ultra-wide   mean +24.22   spread 91.98   signs not shared     FAIL
  wide         mean  +9.60   spread 15.16   signs not shared     FAIL
```

Both fail both bars. The two failures are not the same failure.

**On the wide arm the spread is one take, and it is the take the pre-registered
sufficiency gate already excludes.** `899c75` was shot from far enough away that
the board sits at a median `rho` of 0.31 with **zero** chains past `rho = 0.6`;
it is fitting a sub-pixel bow in the inner field and extrapolating it to a corner
it never saw, and its cost drop of 1.1 % says so. Drop it and the remaining five
give `+13.19 / +12.52 / +13.00 / +7.80 / +13.08` — mean +11.9, spread 5.4, all
one sign. Drop `e52acc` too, whose 13 outer chains are the next-thinnest, and the
four best-covered takes agree to **0.7 px**. That is a declared post-hoc reading
and it is offered as a mechanism, not as a pass.

**On the ultra-wide the spread is real and it is rolling shutter.** See below.

### 4. Straightness residual — wide PASSES, ultra-wide FAILS, and the failure is the null

```
  arm / variant                 corner    cost drop   median residual   at 960 scale
  ultra-wide, subpix, order 1   + 9.62      0.31 %    1.804 -> 1.817    0.451 -> 0.454
  ultra-wide, seeds,  order 1   + 2.02      0.04 %    0.547 -> 0.550
  ultra-wide, chord >= 0.30     + 9.95      0.40 %    2.485 -> 2.497
  ultra-wide, sharper half      + 9.99      0.23 %    1.632 -> 1.660
  ultra-wide, five moving takes + 9.98      0.35 %    2.093 -> 2.103
  wide,       subpix, order 1   +12.10     34.11 %    0.220 -> 0.213
  wide,       outer chains      +15.86     75.33 %    0.638 -> 0.079
```

The ultra-wide **fails both bars**: the median residual rises rather than falls,
and the cost drop of 0.03-0.40 % is an order of magnitude under the 3 % bar. The
wide arm passes both by a wide margin.

`docs/UW_DISTORTION_PREREG.md` fixed the meaning of those two numbers before any
of this ran: *"Where a real distortion is present, correcting it drops the cost
by 3-24 % and visibly reduces the median straightness residual. Where none is
present, the cost drops by 0.1-0.3 % and the residual does not move."* The
ultra-wide scores 0.03-0.40 % with a residual that does not move. **On an
instrument that has just returned a 20 px injection to 2.8 px, that is not a
failure to measure — it is a measurement of nothing.**

The distinction from the room-frame refusal is the injection gate. There the
same "no cost drop" reading came with a response slope of 0.745 and bands of
`+/-280 px`, so nothing could be concluded. Here it comes with a slope of 1.039
and bands of `+/-24 px`.

### 5. The answer — the +17.2 px bound is CONFIRMED

```
  ultra-wide, 3840x2160, order 1, pooled over all six takes

    variant   a1          D_corner   identifiability   bootstrap 5-95 %   D_rms at 960x540
    subpix    +0.004368   + 9.62 px  [-13.2, +35.2]    [ +3.8, +15.7]         0.78 px
    seeds     +0.000920   + 2.02 px  [-11.0, +15.4]    (not run)              0.16 px

    k1 for an OPENCV model at f = 1466 px:  -0.00986   (subpix variant)
    run-to-run repeatability, three runs:   +9.17 / +9.39 / +9.62 px
    answer-blind subsets:  sharper half +9.99,  five moving takes +9.98
```

Against the pre-registered rule: `|D| = 9.6 px <= 26 px`, the bootstrap band
`[+3.8, +15.7]` sits entirely inside it, and nothing in the data requires more.
**CONFIRMS.** The 2 %-cost identifiability band reaches `+35.2 px`, which grazes
the bottom of the 30-54 px range, so the confirmation rests on the point
estimate and the bootstrap rather than on that band — but the band is a
permissive upper edge, not a requirement, and the injection gate independently
fixes the instrument's accuracy at 0.6-3.1 px over `+/-80 px`.

Read against the instrument it was built to check:

```
  eval/uw_selfcal.py, three sessions, pre-registered blocks   +5.05 / +10.28 / +8.58 px
  eval/uw_selfcal.py, longest blocks                          +2.5 /  +7.2 / +10.5 px
  eval/uw_selfcal.py, largest value anywhere                          +17.2 px
  eval/uw_selfcal.py, plus its own +/-9 px floor, the ceiling         +26.0 px
  this instrument, board rows and columns, pooled            +9.62 px (seeds +2.02)
  what 0.7 dB needs                                                 +30 to +54 px
```

Two instruments that share **no input at all** — one carries LiDAR-anchored
points outward across a trajectory on three room walks, the other fits
checkerboard rows in single frames of six different takes — land at `+7.97` and
`+9.62 px`. The route this replaces returned `-118.2 / +1.5 / -7.2`.

On this page's own dB scale, `D_rms` at 960x540 is **0.78 px** against the
pre-registered `<= 1.0 px` refutation bar and the `>= 4.4 px` bar that would
account for the loss. Even the identifiability band's upper edge, `+35.2 px`,
gives 2.86 px — inside the undecided band, not at the 4.4 px that explains
0.7 dB. **The decision rule returns refuted.**

**The mechanism is now visible in the App source.**
`App/Sources/Capture/UltraWideProbe.swift` records it in a comment: *"Geometric
distortion correction lives on the device, not the output, and it is on by
default for the ultra-wide — which is Apple already rectifying the lens for you.
That is why it withholds the distortion tables."* `MultiCamRecorder.swift`,
which produced every session in this repo including these six takes, never
touches `isGeometricDistortionCorrectionEnabled` — so it runs at the default,
and the recorded ultra-wide frames are already rectified. That is one coherent
account of everything: a measured null on the ultra-wide, a factory table that
never helped because it describes the raw lens rather than the recorded pixels,
and a wide arm that keeps its distortion and matches its table. It is
circumstantial — the flag is not logged in `manifest.json` — and the cheap way to
settle it is to log the flag, or to record one session with it explicitly off.

### 6. Focal length — the wide arm is quoted, the ultra-wide is refused

Both pre-registered controls pass: the synthetic sub-grid recovery returns
`f` to **0.099 %** and the corner to **0.79 px** through random in-plane offsets
and 90-degree relabels, confirming that an unknown sub-block is a gauge; and
handedness is constant over all 144 ultra-wide and 85 wide detections.

```
  arm          views   reproj RMS    fx        fy       cx       cy      k1        D_corner
  wide           86      0.97 px    473.8    474.5    318.6    243.7   -0.0357    +11.03 px
  wide, low-motion half  0.65 px    476.7    477.8    320.7    244.2   -0.0229     +6.76
  wide, low-motion 25%   0.65 px    473.0    474.0    320.8    245.8   -0.0104     +3.03
  ultra-wide    143     41.54 px   1953.3   1922.8   1932.5   1195.2   -0.0083    +24.26
```

**The ultra-wide is refused** by amendment 1: a 41.5 px reprojection RMS means a
pinhole-plus-`k1` camera does not describe these frames, so its `fx = 1953` is
not a measurement and cannot be compared with the 1466.0 px
`tools/uw_field_angle_transfer.py` derives. Restricting to the one still take
brings the RMS to 0.46 px but leaves 13 views of a small central board, which
returns `fx = 1554` with the principal point 5 px off centre — badly
conditioned, and not quotable either.

**The wide arm is quoted and it corroborates the argument that produced 1466.0.**
`f = 474.1 px` at 640x480 with the principal point at `(318.6, 243.7)`, 1.6 px
from the frame centre, and the reprojection at 0.97 px. The logged
`videoFieldOfView` of 69.52744 deg gives `(W/2)/tan(hfov/2) = 461.4 px`, so the
board's paraxial focal length is **2.75 % larger** than the field-of-view
reading. That is the same sign and the same order as the **1.70 %** by which
`tools/uw_field_angle_transfer.py` finds the ultra-wide's paraxial 1466.0 exceeds
its own `(W/2)/tan` value of 1441.56, and it is measured directly rather than
transferred. The premise — `videoFieldOfView` is a whole-frame quantity that
already contains the distortion — now has a direct measurement behind it on the
arm where one is possible.

### Why six takes of one lens disagree by 92 px: rolling shutter, measured

The ultra-wide's per-take spread is not the scene and not the fitter. It is in
the detections themselves, before any distortion model is fitted. For every
detection, fit the homography that a planar board through *any* projective
camera must satisfy, and look at what is left:

```
  arm          median   p90     max     median board motion between frames
  ultra-wide    5.23    28.19   53.42        110-220 px on the five moving takes
  wide 640x480  0.61     1.21    4.99         35- 55 px on the same takes

  the one take shot nearly still, 899c75:
  ultra-wide    0.59 px                            6 px of board motion
  wide          0.47 px                            2 px
```

The ultra-wide carries **nine times** the wide arm's non-projective residual on
the moving takes and **1.25 times** on the still one. Both arms see the same
board at the same instant, so the target is flat — a curved screen would show in
the wide arm scaled by focal length, at 13 px rather than 0.6. What is left is a
camera effect that switches off when the camera stops: rolling shutter. The size
is right to a factor of 1.1 — focal length 1466/474 = 3.1 times, readout height
2160/480 = 4.5 rows, and 5.23/0.61 = 8.6 against the 9-14 those predict.

This is why `899c75` returns `+82.68 px` with a 38 % cost drop and a falling
residual. It is the cleanest take in the set — 0.59 px of homography residual —
and it has **zero** chains past `rho = 0.6`, a median `rho` of 0.179 and a median
chord of 0.16 of the diagonal. A `+82 px` corner at `rho = 0.18` is 0.47 px of
bow: it is fitting a half-pixel of inner-field curvature and extrapolating it 30
times outward. The pre-registered outer-coverage gate exists for exactly this,
and pooling is what defeats it — the pooled fit is stable at `+9.4 to +10.0 px`
across every answer-blind subset tried.

The general lesson is one this repo has met before in `HANDOVER.md` §8: **the
capture rate and the readout time were constants nobody chose.** A calibration
target shot handheld at 5 Hz through a 3840x2160 rolling shutter puts tens of
pixels of unmodelled warp into every frame. The wide arm escaped only because it
runs at 640x480.

### What is safe to carry forward

- **The ultra-wide's radial distortion at the active 3840x2160 / 106.2007 deg
  format is small: `D_corner = +9.6 px`, and under 26 px on every reading.** It
  is not 30-54 px, so it does not account for the 0.7 dB. Two instruments sharing
  no input now agree, and this one's injection gate resolves 20 px on the frames
  the answer comes from.
- **The wide arm's factory distortion table is right for its active format:
  `+12.1 px` at order 1, `+16.6 px` at order 2, `+15.9 px` on outer chains alone,
  against the table's `+12.5 / +15.6 px`.** The `-1.41` and `-1.70 px` that three
  documents rest on are the centre of a flat basin. Anything derived from
  "the wide arm is nearly rectilinear" needs redoing — in particular the annulus
  profile in `eval/uw_radial_profile.py`, which is a ratio of two lenses.
- **The wide arm's paraxial focal length at 640x480 is 474.1 px**, 2.75 % above
  the logged field of view's 461.4, with the principal point 1.6 px from centre.
- **A checkerboard fixes what the rooms could not, and rolling shutter is the
  next constant to pin.** The straightness floor went from 1.44 px to 0.45 px and
  the response slope from 0.745 to 1.039. What is left on the ultra-wide is a
  5.2 px per-frame rolling-shutter warp. Anyone wanting the ultra-wide's corner
  to better than a few pixels should shoot the board **on a tripod, one still
  frame per pose**, with the board pushed into the corners — the still take here
  had 0.59 px of residual and would have answered outright if it had carried any
  outer coverage.
- **`tools/fit_uw_distortion.py` is unmodified** and its behaviour reproduces;
  the board path is `tools/fit_uw_board.py`, which imports its `ChainSet`,
  objective, bootstrap and injection machinery unchanged.
