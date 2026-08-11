# What the image can add to depth odometry, and what it costs

Depth odometry answers "can a pose come from somewhere other than ARKit". It
answers it well enough to be interesting and not well enough to ship: holding
rotation at the IMU's attitude improves most loop-closure sessions and beats ARKit
on some, while the worst are still metres off. See [POSE.md](POSE.md) for that work
— and take the specific loop-closure figures from there rather than from here,
because as of this writing they are being re-run on a stable tree. A **0.027 %**
difference in the depth intrinsic scale moved one session's loop error between
1.495 m and 8.254 m, which says that a *diverging* trajectory's loop number is not
a quantity to quote to three decimal places. This page is about the term that is
missing from that pipeline. **The short version: per frame pair it works well, and
wired into the trajectory it is currently net negative** — of eight sessions, three
improve and five worsen, three of those by 17–29×. Both halves are below, and the gap
between them is the most useful thing here.

The candidate is a **photometric residual** — the intensity difference between a
frame and the previous frame's depth map reprojected into it — which is what the
FAST-LIVO family adds to LiDAR odometry, and what the same family credits for
surviving scenes where the geometry alone degenerates. The question is whether
this capture can support it, and the answer here is measured rather than argued.

## The ultra-wide question is the same question, inverted

The ask that started this was ultra-wide RGB with a photometric term. Two facts
have to be separated before it can be answered.

**The ultra-wide is not on the recording path, and no recorded session contains
one.** Every format in `events.jsonl`'s `ar.formats` on all thirteen sessions
reports `WideAngleCamera`; the ultra-wide needs an `AVCaptureSession`, which
cannot coexist with an `ARSession`. [DATA_FORMAT.md](DATA_FORMAT.md) *Why not the
0.5x ultra-wide?* has the details and Apple's own answer. So the photometric term
cannot be tested on ultra-wide imagery today.

**It can be tested on the wide camera today**, and that is the useful ordering,
because of what the ultra-wide path costs. Reaching the ultra-wide means
`AVCaptureMultiCamSession` — `LiDARDepthCamera + UltraWideCamera` is in
`supportedMultiCamDeviceSets` on this phone — and it costs ARKit's pose.
[ULTRAWIDE.md](ULTRAWIDE.md) lists pose as one of four things that path needs. It
is not one of four. It is the one that has to come first, and a photometric term
is how the ultra-wide's own frames would pay for it.

One thing survives the loss of ARKit, and it happens to be the important one.
Rotation on the depth path comes from **CoreMotion**, not ARKit, and CoreMotion
does not care whether an `ARSession` is running. The measurement that made depth
odometry work — IMU attitude agreeing with ARKit to 0.4–3.4° with no drift over
20–50 s, and translation-only ICP beating the six-DoF solve — therefore carries
over to a session that has no ARKit in it at all. What would be lost is the
reference to score against, not the ingredient.

## What the sensors actually hand a photometric term

Measured over all thirteen sessions in `~/nav_data`:

| | |
| --- | --- |
| RGB | 1920×1440 JPEG, quality 0.85, **5.0 Hz** exactly |
| depth | 256×192 float16, 30 Hz, **already registered to the RGB camera's frame** |
| confident depth (`confidence == 2`) | 22.7 % – 75.1 % of the map |
| translation between consecutive RGB frames | median 0.089 – 0.246 m |
| rotation between consecutive RGB frames | median 1.7° – 4.2°, p90 up to 17.5° |
| RGB-to-depth pairing gap | median 0 ms (the streams share frame numbers) |
| exposure logged | duration only — **no ISO, no gain** |

Three of these matter more than the rest.

**Depth is pixel-registered to the image.** No extrinsic, no rectification, and —
the part that matters for a photometric term — no plane approximation. FAST-LIVO
warps a patch under a constant-depth assumption and FAST-LIVO2 replaces it with a
plane prior it also refines, because a sparse LiDAR gives it nothing better. A
per-pixel depth map registered to the image gives the exact reprojection for
free, so the approximation those papers argue over does not arise here.

**5.0 Hz is an app rate gate, not a hardware limit.** `ar.formats` offers
1920×1440@60 and `stillsHz` is a stepper in Settings. The 0.089–0.246 m baseline
that direct alignment has to cross is a setting, and raising it is the cheapest
improvement available anywhere in this document.

It is also cheaper to *test* than it looks, because the rate cannot reach the
exposure. Across all 18,954 pose rows in the thirteen sessions the longest
exposure is **16.667 ms — exactly 1/60 s**, the ARKit video format's frame
period, and not one row exceeds it. Auto-exposure is bounded by the 60 Hz video
stream rather than by the stills rate, each still is JPEG-encoded independently,
and the same 60 Hz stream drives auto-exposure itself. So a frame captured in a
15 Hz session is photometrically identical to one captured in a 5 Hz session, and
**a 15 Hz capture decimated three-to-one is a valid 5 Hz control** — a better one
than a second walk down the same corridor, which holds neither the trajectory nor
the exposure history fixed. One capture yields both arms of the comparison.

The real cost of a higher rate is heat, and it fails loudly rather than quietly:
`RecordingCoordinator.applyThermalState` pauses video and depth outright at
`.serious`, so thermal pressure leaves a gap rather than a degraded stream. Check
`events.jsonl` for `thermal.throttle` before trusting a fast capture. The one
session that logged a thermal transition (5acd1b, to `.fair`, 74 % through)
degraded nothing: RGB, depth and pose held 5.0 / 30.0 / 60.0 Hz across the split
with identical maximum gaps, because `.fair` is not a throttling state.

**The pairing is free.** Stills mode writes an RGB frame and a depth frame off
the same `ARFrame`, so the joint update needs no interpolation and no time-offset
calibration. That is the one place this capture is *easier* than the rigs the
papers are built for, which resample a continuously-spinning LiDAR onto camera
timestamps to get the same property.

## The depth intrinsic scale, and why it is uniform 1/7.5

This is settled and worth stating, because getting it wrong is invisible and
expensive. The depth map is 256×192 and the colour frame is 1920×1440, which is
**exactly 7.5× in both axes**. That integer ratio is what makes 1440 the
self-consistent reference: intrinsics belonging to some taller buffer could not
reduce to a 4:3 depth map by an integer factor. So `c_y = 728.7` is a genuine
principal point sitting 8.7 px below centre, not evidence of a 1456-row buffer, and
the correct scale is a **uniform 1/7.5** applied to `fx fy cx cy` alike.

The tempting alternative, deriving the scale as `w / 2c_x`, assumes the principal
point is exactly half the width. It is not, and `tools/depth_odometry.py`'s
command-line path currently makes that assumption — being fixed alongside POSE.md's
re-run.

That alternative is worse than a constant bias, because the principal point moves.
Autofocus shifts the intrinsics *within* a session, and by more than the clusters in
any single frame suggest:

| | fx range | cx range | cy range | `c_x` crosses 960 |
| --- | --- | --- | --- | --- |
| 31c6aa | 1278.4 – 1422.8 (11.3 %) | 956.4 – 960.2 | 721.2 – 729.1 | yes |
| 532cea | 1278.3 – 1417.7 (10.9 %) | 949.8 – 959.0 | 721.3 – 728.6 | no |
| b4b41a | 1287.4 – 1354.1 (5.2 %) | 957.4 – 961.1 | 715.2 – 727.3 | yes |
| 1868dd | 1289.3 – 1355.1 (5.1 %) | 958.4 – 960.0 | 721.1 – 728.9 | no |
| 5acd1b | 1336.5 – 1348.6 (0.9 %) | 959.1 – 960.3 | 728.5 – 729.0 | yes |

`c_x` crosses 960 in **eight of the thirteen** sessions, so a scale defined as
`w / 2c_x` does not merely sit slightly off — it drifts with focus and reverses sign
mid-session, giving a fixed sensor geometry a time-varying, scene-correlated error.

The general rule, and it is the same rule as the pyramid ladder above: **a downscale
ratio is a property of buffer sizes, not of optics, so it must never be derived from
an optical parameter.** This repository already reads intrinsics per frame precisely
because focus breathing moves them, and that is correct; what is not correct is
extracting a frame-invariant quantity from those per-frame values. `fx` and the
principal point must vary per frame; the 1/7.5 sensor downscale must not.

The same requirement propagates into the pyramid: one scalar sets both `fx` and `fy`,
so every level must scale x and y identically. A coarsest level of 30×23 does not
(22.5 is not an integer, giving a 2.2 % `fy` error), which is why the levels here are
32×24, 64×48, 120×90, 240×180, 480×360, 960×720 — all exact, and asserted at import
rather than left to a comment, because a contained defect passes an end-to-end test.

**What that convention is worth, in one comparison.** The same class of perturbation,
on two kinds of estimate:

| | perturbation | effect |
| --- | --- | --- |
| per frame pair, no accumulation (this page) | 0.16 % | 0.005 – 0.067 cm |
| accumulated trajectory, diverging session (POSE.md) | 0.027 % | 1.495 m against 8.254 m |

A perturbation **six times smaller** produces an effect four orders of magnitude
larger. That is the difference between a quantity that accumulates and one that does
not, and it is why per-frame-pair numbers can be quoted to three decimal places while
a diverging trajectory's loop closure cannot be quoted at all.

## Why the image is worth adding

Point-to-plane ICP sees translation **along surface normals** and is blind
**across** them: slide a camera along a flat wall and the point-to-plane residual
does not change. Image gradients are the mirror image — texture sliding across a
surface is exactly what they measure, and motion along the viewing ray is what
they measure worst. The two are complementary by construction, not by tuning.

That is the argument. Here is the measurement. For each session, the
translation-only 3×3 normal matrix `Σ nnᵀ` gives the axis ICP is least able to
see; frame *k*'s confident depth is then reprojected into frame *k+1* under
ARKit's relative pose, perturbed along that axis, and the photometric residual
read off:

| session | conf2 | ICP cond | residual min offset | within ±2 cm | rise at 2 cm | rise at 5 cm | basin |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 31c6aa | 53.8 % | 9.6e-03 | +0.5 cm | 71 % | 1.15× | 1.40× | 18.5 cm |
| a7b288 | 4.5 % | 2.7e-03 | +0.2 cm | 100 % | 1.45× | 1.94× | 14.0 cm |
| 532cea | 74.5 % | 4.0e-02 | +0.0 cm | 92 % | 1.14× | 1.39× | 19.5 cm |
| b36df8 | 75.1 % | 2.1e-03 | +0.0 cm | 93 % | 1.23× | 1.80× | 19.8 cm |
| b4b41a | 67.4 % | 2.2e-01 | +0.0 cm | 100 % | 2.56× | 4.42× | 20.0 cm |
| 5bd1ed | 57.1 % | 7.7e-02 | +0.0 cm | 92 % | 2.56× | 4.08× | 20.0 cm |
| cb4586 | 62.9 % | 1.0e-01 | +0.0 cm | 100 % | 2.01× | 3.20× | 20.0 cm |
| 1696fa | 63.5 % | 2.9e-01 | +0.0 cm | 100 % | 2.28× | 3.63× | 20.0 cm |
| 1868dd | 29.8 % | 3.2e-02 | +0.0 cm | 92 % | 1.99× | 3.93× | 19.5 cm |
| f0d073 | 30.7 % | 8.2e-02 | +0.0 cm | 92 % | 2.58× | 3.52× | 19.5 cm |
| 683ef1 | 28.0 % | 2.3e-02 | +0.0 cm | 86 % | 1.66× | 2.20× | 19.5 cm |
| 5acd1b | 22.7 % | 2.7e-02 | +0.2 cm | 92 % | 1.57× | 2.45× | 19.5 cm |
| 3c7c6b | 37.5 % | 3.5e-02 | +0.0 cm | 93 % | 1.29× | 2.14× | 19.8 cm |

Read three things off it.

**The residual's minimum is where ARKit says the camera went.** Eleven of
thirteen sessions put it at 0.0 cm along the axis geometry cannot see, and
71–100 % of individual frame pairs (median 92 %) land within ±2 cm. The image
observes the direction the depth map does not.

**The signal is steep enough to descend.** 2 cm off the minimum the residual is
1.14–2.58× larger, and 5 cm off it is 1.39–4.42× larger. In several sessions the
rise along ICP's *weakest* axis exceeds the rise along its strongest — 1868dd
1.99× against 2.13×, f0d073 2.58× against 1.80×, b4b41a 2.56× against 2.24× —
which is the complementarity claim in numbers rather than in prose.

**The convergence basin is wider than the motion.** The residual rose
monotonically to the ±20 cm edge of the sweep in twelve of thirteen sessions, so
the basin is a *lower* bound of 20 cm at a 480×360 working resolution — against a
0.089–0.246 m inter-frame baseline. A solver started at zero motion should reach
the minimum, which is worth knowing before building the state machine that
predicts one.

An information-matrix comparison was run alongside this and is **not** quoted
here. Summing `Σ ggᵀ` over sixteen thousand image pixels and inverting it says
the photometric term localises translation to eighty microns, which is nonsense:
dense image residuals are nowhere near independent, and the Cramér-Rao bound
built on pretending they are overstates precision by whatever the correlation
length happens to be. The curve above carries every correlation with it, so it is
the one to trust. The information matrices are still useful for *direction* — for
choosing which axis to perturb — and that is all they are used for.

## Per frame pair, the image term is the estimator

**Read this section together with *In a trajectory it is negative, so far* below.**
Everything here is measured on isolated frame pairs, and the trajectory result went
the other way in five of eight sessions. The per-pair numbers are still the right
place to start — they are what says the information exists and is reachable — but
they do not carry over on their own.

The residual sweep says the information is present. A solver says whether it is
reachable. Translation-only, rotation fixed to ARKit's relative rotation,
Levenberg-Marquardt with step rejection, six-level pyramid, scalar gain
re-estimated in closed form, Huber on both blocks, started from **zero motion** —
164 frame pairs over all thirteen sessions, scored against ARKit's relative
translation:

| ICP conditioning | pairs | depth alone | image alone | both | do nothing |
| --- | --- | --- | --- | --- | --- |
| `cond < 0.01` | 60 | 6.21 cm | **1.26 cm** | 1.37 cm | 12.91 cm |
| `0.01 – 0.05` | 50 | 2.09 cm | **0.68 cm** | 0.68 cm | 11.18 cm |
| `cond ≥ 0.05` | 52 | 1.03 cm | **0.58 cm** | 0.59 cm | 11.62 cm |

Photometric-alone median error is under 2 cm in **all thirteen** sessions
(0.37–1.62 cm). Read two things off the table.

**The image term's advantage is largest exactly where the geometry is worst** —
4.9× at `cond < 0.01`, 3.1× in the middle band, 1.8× where conditioning is
comfortable. That monotone ordering is the prediction this whole document rests on,
and it is the reason to believe the term follows the stated mechanism rather than
absorbing error somewhere convenient.

**Coupling the depth block in adds nothing here, and costs a little where it
matters most.** `both` is level with `photo` in the two better-conditioned bands
and *worse* in the degenerate one — 1.37 cm against 1.26 cm. That is not an
argument against fusion in a trajectory, where the depth block also carries scale
and continuity; it is a measured statement that, per frame pair, the image term is
the estimator and the depth term is not improving it.

### A block's residual cannot be used to weight it

The obvious fix for the above is to trust the depth block less, by weighting each
block with `1/σ²` where σ comes from its own residual rather than from a nominal
1 cm of depth noise. Measured: it changes almost nothing — `both` moved by 0.00 cm in the degenerate band
and 0.05 cm in the middle one. The reason is worth stating,
because it is a trap that looks like good practice:

| session | pair | ICP pose error | its own residual | Hessian eigen ratio |
| --- | --- | --- | --- | --- |
| 532cea | 20 | **17.59 cm** | **1.62 mm** | 3.9e-03 |
| 532cea | 40 | 1.16 cm | 4.54 mm | 1.6e-01 |
| 532cea | 60 | 24.33 cm | 3.67 mm | 1.3e-02 |
| cb4586 | 54 | 0.18 cm | 6.55 mm | 2.0e-02 |
| cb4586 | 135 | 0.91 cm | 6.29 mm | 8.3e-01 |

Over fourteen such pairs, ICP pose error against its own residual scale correlates
at **r = −0.057** — nothing. The worst pose error in the table has the *smallest*
residual. Point-to-plane residuals measure how well two surfaces agree, and they
stay millimetric while the pose slides along an axis the normals cannot see; that
is what degeneracy *is*. The quantity that does track the error is the **Hessian's
conditioning**, which is why the gate belongs there — and why `cond`/`weakAxis`,
already logged on device, is the right signal.

So the depth block's problem is not that it is over-trusted. It is that a bad
projective association produces a *confidently wrong* constraint — large
eigenvalue, wrong direction — and no isotropic σ can discount that. The lever is
tighter association gating, not reweighting.

### Six pyramid levels, not the papers' three

The single most consequential implementation detail, and it is one where following
the literature gives the wrong answer. FAST-LIVO and FAST-LIVO2 use three levels,
tuned for 10 Hz rigs whose inter-frame displacement is half of ours. At 5 Hz a
pyramid starting at 240×180 begins **outside its own basin**. On the synthetic
ground-truth test:

```
3 levels (coarsest 240x180)   18 cm displacement -> recovered 14.90 cm, error 14.27 cm
5 levels (coarsest  60x45)    18 cm displacement -> recovered 18.07 cm, error  0.18 cm
6 levels (coarsest  30x23)    all five cases pass, worst error 0.179 cm
```

An earlier draft of this page reported a "convergence radius of about 15 cm" from
this data, with inter-frame baseline correlating against photometric error at
r = +0.930. That correlation was real and the conclusion drawn from it was wrong:
it was measuring the three-level pyramid, not the problem. With six levels the
three sessions that failed — the three fast walks — return 1.49, 1.06 and 1.62 cm.
The lesson generalises past this document: a defect in our own code is caught by
review, and a borrowed convention that does not fit our sensor is not.

## Two measurements that change the design

### The logged exposure is worse than useless

The obvious first move is to divide out `pose.jsonl`'s `exposure` before
differencing two frames. Measured on the 25 frame pairs across all sessions where
auto-exposure moved by at least a quarter stop, it is the wrong move.

For each pair, the scalar gain that actually minimises the residual is
`α_fit = ⟨a,b⟩ / ⟨a,a⟩` over corresponding pixels. Against what the log predicts:

```
exposure predicts a gain of     0.50 .. 2.00      (median |log2| 0.794 stops)
the image actually wants        0.93 .. 1.11      (median |log2| 0.037 stops)
correlation of the two, in log space             +0.43
```

**A full stop of logged exposure change shows up in the image as 0.04 stops.**
The camera pipeline has already compensated — with gain, which is not logged, and
with tone mapping — so the frames arrive very nearly brightness-constant and
dividing by the duration injects the factor of two it was meant to remove. Doing
it inflated the warped residual by up to 20× on the pairs where exposure jumped.

Two consequences. FAST-LIVO2 carries an inverse-exposure scalar τ in its
19-dimensional state and *estimates* it; that is the right shape, but τ must be
initialised at 1 and never seeded from this log. And if this is ever to be
revisited, the app should log ISO alongside duration — on its own, duration
cannot explain the brightness of a frame it was recorded with.

Fitting the scalar gain from the image is still worth doing: it improved the
residual by 1.00–1.98× on those same pairs. It is cheap, it is closed-form, and
it belongs in the loop.

### Do not linearise the JPEG

Stored sRGB values are **closer** to brightness-constant than an sRGB-linearised
copy of them — median |log2 α_fit| of 0.037 stops against 0.060. The pipeline's
transfer curve is not plain sRGB gamma, so applying its inverse moves away from
radiance rather than towards it. Re-running the whole residual sweep in sRGB
without exposure correction confirmed it end to end: every session's numbers held
or improved, and the session with a full stop of exposure movement (cb4586)
scored 100 % within ±2 cm with a 2.01× rise, unaided.

Use the eight-bit values as stored, divided by 255. This is one of those results
that only shows up if it is measured, since both corrections are individually
well-motivated and both are wrong here.

## The design that follows

**Rotation is not estimated.** It comes from CoreMotion attitude, per POSE.md:
handing rotation to the geometry drifts 28–66°, and snapping it to gravity
afterwards makes the trajectory sixty times worse. Only translation is solved.

**One joint solve at 5 Hz, depth-only in between.** The depth stream runs at
30 Hz and the image at 5 Hz. Rather than treat that as a 5:1 asynchrony — which
is the arrangement FAST-LIVO2's own supplement calls less robust than a
synchronised sequential update — use the fact that the two streams share frame
numbers: at each RGB frame, solve depth and photometric residuals together in one
normal-equation system; between RGB frames, run depth alone. The joint update is
the one that gets the image, and it is free of any time-offset calibration.

**Gate the geometric block on conditioning, and do not try to weight it by its
residual.** `1/σ_I²` on the photometric block with σ_I from the residual's MAD is
fine, because a photometric residual does report photometric error. The geometric
block is different: its residual is uncorrelated with its pose error (r = −0.057,
measured above), so no σ derived from it discounts a wrong answer. Reach for
association gating and the conditioning number instead. Huber on both, with the
threshold and every σ **fixed once per pyramid level** — re-deriving them per
iteration makes a line search compare a different objective each time, so a step
looks like an improvement when only the scale it is measured against grew.

**Six-level pyramid, coarsest 30×23.** Not the three the papers use; see above.
This is the difference between recovering an 18 cm displacement to 0.18 cm and
missing it by 14 cm.

**Levenberg-Marquardt, not plain Gauss-Newton.** Both Hessians run at eigenvalue
ratios around 2e-02 on indoor geometry, and an undamped solve amplifies the step
along the weak axis by the inverse of that. Damp proportionally to the diagonal and
reject any step that does not reduce the cost.

**Frame-to-map from the start — this recommendation was inverted, and the
trajectory result is what corrected it.** An earlier version of this page said to
begin frame-to-frame and add the map reference later, "so that the term's value is
measured on its own". That ordering is the failure mode. The depth block anchors to
an accumulated map; a photometric block anchored to the *previous image frame*
anchors to a pose that is itself an estimate carrying its own error, so the residual
is accurate about a reference that is already wrong and pulls the current pose back
toward it. The papers all match against reference patches held in a persistent map,
and that is not the drift-resistance refinement this page took it for — it is what
makes the two blocks measure against the same thing. It is also nearly free here,
because the depth pipeline already maintains a local map whose points can carry the
patches.

**Gate on conditioning, but do not switch the image term off.** Sessions already
carry per-frame `cond` and `weakAxis` in `depth.jsonl` — the translation-only
conditioning computed on device, populated in the four newest sessions. That is the
signal that actually tracks where the geometric block goes wrong, and the table
above says the image term is informative at every conditioning value measured.
None of the four papers surveyed defines a rule for disabling one sensor's term;
all handle it with outlier rejection instead, and so should this.

## In a trajectory it is negative, so far

The per-pair case above is the strongest part of this document and it is not enough.
Wired into the depth-odometry trajectory and scored by loop closure — the one
reference-free measure this project has — the term is **net negative**. Figures from
POSE.md's eight-session re-run, in metres:

| session | ARKit | depth alone | with the image term | |
| --- | --- | --- | --- | --- |
| f0d073 | 0.286 | 0.694 | 0.545 | better |
| 1696fa | 0.126 | 0.123 | 0.120 | better |
| 5acd1b | 0.055 | 3.078 | 2.675 | better |
| 683ef1 | 0.490 | 1.137 | 1.383 | worse |
| 1868dd | 0.378 | 1.381 | 2.152 | worse |
| cb4586 | 0.146 | 0.196 | **3.386** | 17× worse |
| 5bd1ed | 0.269 | 0.322 | **9.411** | 29× worse |
| 3c7c6b | 0.127 | 1.942 | **56.026** | 29× worse |

Three improve, five worsen, three catastrophically. **This is what the frame-pair
experiment could not see**, and it is why loop closure was named the acceptance score
before any of this was built.

What makes it diagnostic rather than merely disappointing is that per-pair quality
does not predict the damage at all — median error against the damage factor is
**r = +0.191**, and the p90 tail is **r = +0.151**:

| session | per-pair photo p90 | trajectory outcome |
| --- | --- | --- |
| cb4586 | **1.32 cm** — tightest of the eight | 17× worse |
| 5bd1ed | **1.81 cm** — second tightest | 29× worse |
| 5acd1b | **15.73 cm** — worst tail | **improved** |

The two sessions the term destroys have the cleanest per-pair behaviour, and the
session with the worst per-pair tail is one it helps. So the damage is not coming
from bad per-frame-pair estimates; it is coming from how the term is applied.

Two hypotheses have been tested and killed. The **0.2 s image interval** is not the
problem: these 164 pairs are exactly 12 ARKit frames apart, the same 0.2 s, with
baselines of 9.1–13.4 cm and per-pair errors of 0.46–1.59 cm on these same eight
sessions. **Intermittency** is not the problem either — the obvious suspicion was that
images arrive at 5 Hz against depth's 30 Hz, so the term participates in only ~17 % of
updates and the objective alternates along the trajectory. Applying it *even more
sparsely* (8 %) should then have been worse. It was better, in six of eight sessions,
and the sessions it had destroyed largely recovered: 3c7c6b 56.0 → 0.8 m, 5bd1ed
9.4 → 0.6 m, cb4586 3.4 → 0.4 m. **Damage proportional to dose is the signature of a
noise source, not of an intermittent information source.**

What remains, and what POSE.md is now testing, is an **anchor mismatch**. The depth
block is frame-to-map: it aligns the current frame to an accumulated map. The
photometric block is frame-to-frame against the previous *image* frame, whose pose is
itself an estimate carrying its own error. The term is then accurate about a reference
that is already wrong, and every update pulls the current pose back toward that error.
That single mechanism explains all three observations — excellent per-pair behaviour,
no correlation between per-pair quality and trajectory damage, and damage scaling with
dose.

It also explains why this page's experiment cannot see it, which is the precise scope
of everything above: **the reference pose here was taken from ARKit and held fixed, so
the anchor was correct by construction.** What was measured is what the term does when
its anchor is right. That result stands. What does not follow from it is "therefore
adding it to the odometry helps" — that step fails the moment the anchor is an
estimate. The conditioning table is the sharpest form of the contradiction: in the
`cond < 0.01` band where the image term is 4.9× *better* than depth here, the
accumulated solve has it 1.5× *worse*. Same term, same band definition, opposite sign.

That reversal is not just consistent with an anchor mismatch, it is what an anchor
mismatch predicts. The degenerate axis is the direction the image sees best, and it is
therefore also the direction along which a wrong reference pose pulls hardest — the
term drags the current pose toward the previous frame's error with exactly the
confidence that made it valuable. **The most informative axis becomes the most
contaminated one.** So the band where the image contributes most under a correct anchor
is the band where it does most damage under a wrong one, which is why the harm
concentrates where the earlier per-pair result said the benefit would.

**The two results are at different layers, not in contradiction.** That the image
observes translation the geometry cannot is a fact about the sensors, and it survived
a ground-truth round-trip; that the trajectory gets worse is a fact about how the term
is currently applied. Reading the second as a refutation of the first would throw away
the measurement model on the strength of an integration bug, and that is not what the
data says.

**What it does change in the recommendation.** Everything in *The design that follows*
stands as the description of a term that works per frame pair. What does not yet stand
is switching it on. Until intermittency is settled, the honest position is that this is
a validated measurement model with an unsolved integration problem — not a component
ready to enable, and not a component to abandon.

## What this does not establish

**Two earlier drafts of this section were wrong, in opposite directions**, and both
errors are instructive enough to keep. The first said the image is weakest where the
geometry is worst, from an r = −0.79 correlation between depth-ICP loop error and
the photometric rise at 2 cm — measured across eight sessions all walked at
0.45–0.67 m/s, so the dominant variable was held nearly constant. The second said
the term has a ~15 cm convergence radius, from an r = +0.930 correlation with
inter-frame baseline — which was measuring a three-level pyramid rather than the
problem. With six levels the fast sessions return 1.06–1.62 cm and that correlation
dissolves.

What survives is narrower. Depth confidence does not predict photometric error
(**r = +0.087** over 164 pairs), and once the pyramid is deep enough to reach the
minimum, neither does anything else about the room. The remaining honest caution is
about *capture* rather than about scene content:

3c7c6b holds the largest inter-frame optical flow of the thirteen (31.8 px), the
lowest image gradient (0.0159), and the worst IMU-to-ARKit attitude agreement
(3.44°, against 0.42–0.61° for the other recent sessions). That third number is the
informative one, because CoreMotion attitude shares no code path and no sensor with
either the depth solve or the image residual — so three independent measures
degrading together describes a capture being shaken, not three estimators failing.

With six pyramid levels its photometric *median* is a respectable 1.42 cm, so this
is no longer a failure case; what it keeps is the worst **tail**, at a p90 of
17.55 cm against that 1.42 cm median. That is the shape to expect from motion blur:
most pairs are fine and the shaken ones are badly wrong. It is the tail, not the
median, that integrates into a trajectory — so **a photometric term does not fix a
blurred frame, it just stops being the thing that fails first.**

**ARKit's relative pose is the reference, and it is not truth.** Over a 0.2 s
baseline it is the best available, but the ±2 cm agreement above is a joint
statement about ARKit and the photometric minimum, not a measurement of either
alone. The reference-free score remains loop closure, which needs the solver
wired into the trajectory rather than run per frame pair.

**Rolling shutter and JPEG artefacts are unmodelled.** The residual noise floor
measured 1.4–5.0× the single-frame sensor noise, and that gap is where they live,
along with occlusion and the brightness residue.

**The per-pair numbers did not carry over, and that was foreseeable.** Every
measurement on this page except the trajectory table comes from frame pairs scored
against ARKit, which supports "the information is there and a solver reaches it" and
nothing about integration. Three times in one day on this project a per-frame metric
moved opposite to the trajectory: a residual score ranked a diverging track higher,
snapping attitude to gravity drove rotation drift to zero while making the trajectory
sixty times worse, and now a term with a 1.32 cm per-pair p90 made a trajectory 17×
worse. So the acceptance score stays loop closure, and **never the photometric
residual** — scoring a term by the quantity it minimises is the same mistake in a new
costume.

## The order to try them in

1. ~~**Per-pair solver.**~~ **Done.** The bars were photometric-alone median under
   2 cm in at least nine of thirteen sessions, and the joint solve beating depth
   alone where `cond < 0.01`. Both met, and by more than asked: **13 of 13**, and
   1.37 cm against 6.21 cm. The third bar, under 20 % non-convergence from a standing
   start, was **missed at 24 %** — and chasing that miss is what produced the pyramid
   result, the damping fix and the residual-cannot-weight-it finding. A bar that fails
   informatively is worth more than one that passes.
2. ~~**Into the trajectory.**~~ **Done, and it failed** — three of eight sessions
   improve, five worsen, three by 17–29×, and per-pair quality does not predict which
   (r = +0.191 on the median). Two explanations have since been killed: the 0.2 s
   image interval, and intermittency. **Give the two blocks the same anchor** and
   re-score; a photometric residual measured against an estimated pose is the live
   suspect, so the reference has to be the map, not the previous frame. Two cautions
   still hold: those sessions are all slow walks, and a diverging baseline's loop
   number moves under 0.027 % perturbations.
3. **Capture the missing controls.** Demoted from where an earlier draft put it: a
   higher RGB rate is not required, because the six-level pyramid handles the
   baselines this data contains. It is still worth shooting, for two things this set
   lacks. A **long straight corridor at a steady pace** is the degenerate case the
   whole document is about and there is no controlled example of one. A **fast walk**
   (~1.2 m/s) at 15 Hz tests the pyramid claim on real data rather than synthetic.
   Exposure is pinned at 1/60 s regardless of the stills rate, so a 15 Hz capture
   supplies its own 5 Hz control by decimation — a better control than walking the
   same corridor twice, which holds neither trajectory nor exposure history fixed.
4. **Then the ultra-wide.** `AVCaptureMultiCamSession`, CoreMotion attitude, no
   ARKit — and a pose source that has already been scored on the wide camera
   before it is asked to work without a reference.

## Where the numbers come from

The load-bearing one is a **synthetic round-trip test**: take a real frame and its
real depth, render the second view under a translation chosen by the test, and require
the number back. It exits nonzero on failure like the repo's other `test_*.py`, and it
is what turned this page's per-pair numbers from plausible into checked — the solver
recovers 6, 10, 12 and 18 cm displacements to within 0.18 cm. Before it existed, an
agreement with ARKit of 0.58–2.00 cm looked like a working solver and was not. It is
also worth knowing what nearly derailed it: the first version of the test negated the
translation, the solver dutifully returned `−t_true`, and a clean sign flip in a test
result is evidence about the harness rather than about the estimator.

The rest were produced by five read-only programs kept outside the repository, because
they are measurement scaffolding rather than tooling: a
per-session input survey, the residual-basin sweep behind the first table, the
exposure test, an exposure-cap check, and the translation solver behind the rest.
They import `tools/read_session.py` and write nothing. Both the photometric space and
the exposure division are switchable in the sweep, so the two radiometric findings can
be contradicted rather than taken on faith.

A second, independent implementation of the solver was commissioned in parallel
and its first run disagreed with everything here — 38.87 cm median error on the
*control* method, against 1.12–6.52 cm. That second number was the diagnosis:
a defect in something all three methods shared rather than in the one under test.
Reading it found a relative-pose composition reversed (the function's docstring
described the opposite of what its constructor built, which is how it survived
review), `gx, gy = np.gradient(I)` — numpy returns `d/drow, d/dcol`, so those names
are swapped — and nearest-neighbour sampling quantising the residual at about 6 mm,
half the order of the acceptance threshold. Worth recording for two reasons: none of
the three is visible by reading for correctness, and a numeric acceptance bar found
all three in one run.

The literature comparison is a survey of FAST-LIVO2 (arXiv 2408.14035),
FAST-LIVO (2203.00893), R3LIVE (2109.07982) and DSO (1607.02565), read for
mechanism rather than for results. Independently confirmed against the paper:
the 10 Hz synchronised sequential update, the scalar inverse-exposure τ in a
19-dimensional state, and the ~800 m single-wall HIT Graffiti Wall sequence as
the headline LiDAR-degeneracy result. The plane-prior-versus-constant-depth
ablation (0.22 m against under 0.01 m of drift) is quoted from the paper's
supplement and was **not** independently confirmed; the qualitative claim, that
constant depth is "a wild assumption significantly reducing the accuracy of
affine warping", is in the paper's contribution list.
