# What the image can add to depth odometry, and what it costs

Depth odometry answers "can a pose come from somewhere other than ARKit". It
answers it well enough to be interesting and not well enough to ship: holding
rotation at the IMU's attitude improves most loop-closure sessions and beats ARKit
on some, while the worst are still metres off. See [POSE.md](POSE.md) for that work
— and take the specific loop-closure figures from there rather than from here,
because as of this writing they are being re-run on a stable tree. A **0.027 %**
difference in the depth intrinsic scale moved one session's loop error between
1.495 m and 8.254 m, which says that a *diverging* trajectory's loop number is not
a quantity to quote to three decimal places.

This page asks what the image can add to that pipeline, and it turned out to be two different
routes with two different answers.

**The photometric residual works per frame pair and not in a trajectory.** It beats depth 4.9×
where the geometry degenerates, and bolted onto the trajectory it was net negative — five of
eight sessions worse, three by 17–29×. The cause was that the depth block anchors to an
accumulated map while the image block anchored to the previous frame's *estimated* pose.
Matching the anchors, by pulling depth down to frame-to-frame, improves seven of eight
sessions by 1.4–2.9×; doing it the other way, moving the image reference into the map, still
fails at one of eight. What is left is that a single reference frame cannot equal an
accumulated map, so the anchor has to be matched **per point**, not per frame.

**Handing the sequence to a feed-forward reconstruction model and fixing its scale with LiDAR
depth reached parity with ARKit** — loop closure 1.4 % against ARKit's 1.5 %, ahead on five of
thirteen sessions, and 12 of 13 against depth-only ICP. That clears the blocker this page was
written against: the ultra-wide path costs ARKit's pose, so it needed a replacement, and there
now is one.

The photometric work is not superseded by that, and is worth reading first: two of its
findings steered the second route, and the mistakes it took to get there are the reason the
second route was measured the way it was.

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
is not one of four. It is the one that has to come first, and the ultra-wide's own frames are
how it would be paid for.

**That item is now closed enough to unblock the path.** A pose source that never reads ARKit
sits level with it — see *Where that line stands* — so the ultra-wide capture no longer trades
a working pose for none.

**And the wider lens should help the pose source, not merely the coverage.** The one untested
difference was the field of view: the pose source was validated on the wide camera's ~72°, and
the ultra-wide path would feed it 96.3°. A wider view cannot be synthesised from a narrower
one, but the gradient can be measured on the side that can — crop the wide frames to narrower
pinholes, which for a pinhole is exact, and score against ARKit as usual. All thirteen
sessions, absolute trajectory error in centimetres:

| session | 71.8° native | 62° | 52° | 42° | 42° / native |
| --- | --- | --- | --- | --- | --- |
| 683ef1 | 2.6 | 3.3 | 5.0 | 5.5 | 2.11× |
| 5acd1b | 3.8 | 4.5 | 6.3 | 7.3 | 1.94× |
| 1696fa | 3.9 | 4.5 | 4.3 | 4.9 | 1.27× |
| 2735cf | 5.4 | 6.4 | 7.1 | 11.0 | 2.05× |
| 3c7c6b | 5.8 | 8.7 | 17.5 | 23.7 | **4.10×** |
| 5bd1ed | 5.9 | 7.2 | 7.6 | 12.4 | 2.10× |
| f0d073 | 6.9 | 5.9 | 5.0 | 8.1 | 1.18× |
| cb4586 | 7.8 | 9.6 | 12.8 | 18.7 | 2.40× |
| 2be6a9 | 8.2 | 17.4 | 8.4 | 15.9 | 1.94× |
| ce02ac | 9.8 | 11.1 | 12.4 | 16.9 | 1.72× |
| 1868dd (4 windows) | 14.5 | 24.3 | 70.8 | 96.4 | **6.65×** |
| dd2a13 | 33.6 | 64.4 | 107.2 | 124.4 | 3.70× |
| 6b92f3 | 81.6 | 95.7 | 117.2 | 131.4 | 1.61× |

**Narrowing the field degrades the trajectory in 13 of 13 sessions** — median 2.05×, range
1.18–6.65×, and strictly monotone in the field of view in 10 of 13. The largest losses are the
one multi-window session, where a narrower field also degrades the joins, and the two where
point-to-plane ICP collapses. Extrapolating the sign rather than the magnitude: 96.3° should be
neutral to favourable, and the deployment lens is not a compromise the pose source has to
absorb.

One control is built into the manipulation. Each arm crops from the full 1920×1440 and every
arm still downsamples to the loader's budget, so the 42° arm carries **1.7× more pixels per
degree** than native and still does worse. It is the coverage that matters, not the sampling
density.

**A caution about how to read that table, which nearly cost me the conclusion.** Pooled over
all 52 (session, field-of-view) points, field of view against log ATE gives a weak
**r = −0.29** — weak enough to report as "field of view barely matters", which is the opposite
of what every single session says. The pooling is what destroys it: ATE levels differ 30×
between these sessions, so between-session variance swamps a within-session manipulation. This
is the mirror of the window-length mistake recorded later on this page — there, a
within-session reproduction hid a confound common to every window; here, pooling across
sessions hid an effect present in every one of them. **Match the unit of analysis to the unit
of manipulation**, in both directions.

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
command-line path used to make that assumption. **Fixed in `a680dfe`, and fixed better
than this page originally advised**: rather than hardcoding 1920, it reads the colour
size from `frames.jsonl`, computes `sx` and `sy` separately, and prints both along with
where they came from — so the invariant is visible in every run instead of asserted in a
comment. The `2c_x` derivation survives only as a documented fallback for video sessions,
which have no `frames.jsonl`, rounded to the capture formats' 16-pixel granularity.

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

**Read this section together with *In a trajectory the anchor decides the sign*
below.** Everything here is measured on isolated frame pairs with the reference pose
held fixed from ARKit, which turned out to be the whole difference: bolted onto a
trajectory whose anchor is an estimate, these same numbers went the other way in five
of eight sessions. They are still the right place to start — they are what says the
information exists and is reachable — but they carry over only once the image term and
the depth term measure against the same reference.

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

Read that as advice about *how to form* the weights, not as a lever to tune. On the
trajectory, no weight on the photometric block makes it a net gain while its reference is a
single frame — see *Reweighting does not rescue it either*. Weighting is where this design
has to be correct and not where its remaining problem lives.

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

Confirmed in part, and narrowed since. Matching the two anchors *by weakening depth to
frame-to-frame* turns five-of-eight-worse into seven-of-eight-better. But moving the image
reference into the map while depth keeps its map anchor — the form this recommendation
actually proposes — beats the control in **one session of eight**. A single reference
frame, however fresh and however much it lives in the map, cannot equal an accumulation of
keyframes. **So "anchor to the map" is only correct if it means per-point: each point's
patch referenced to the keyframe that contributed that point's geometry.** Frame-level
variants are all refuted; see *And it is not enough* below.

**But "anchor to the map" is not the same as "anchor to the latest keyframe", and the
difference is measurable.** A map reference has to stay inside the convergence basin,
which the synthetic test puts at 18 cm. Measured (POSE.md's arm matrix), the distance
from the current frame to whatever the photometric term references:

| reference | sessions | distance median | angle median | inside the 18 cm basin |
| --- | --- | --- | --- | --- |
| previous image frame | 8 | 0.096 – 0.142 m | 1.7 – 4.2° | **8 of 8** |
| latest map keyframe | 8 | 0.204 – 0.308 m | 4.9 – 10.4° | **0 of 8** |

The mechanism is visible in the logs rather than inferred. A reference only refreshes on
a frame that is *both* a keyframe and carries an image, and that intersection is narrow
at 5 Hz — image-carrying keyframes against total keyframes, across the eight: 35/203,
29/215, 116/642, 45/253, 51/352, 44/259, 67/351, 31/238. Non-convergence rises with it,
4/126 → 17/119. So a keyframe-anchored reference is not measuring the term — it is
measuring staleness.

The keyframe-anchored arm also reproduces the dose effect from the other direction. It is
worse than the depth-only control in five of eight sessions, but in the three the
previous-image anchor had *destroyed* it is dramatically better — 3c7c6b 56.0 → 3.5 m,
5bd1ed 9.4 → 5.6, cb4586 3.4 → 3.7 — and better than the control in three others. A
reference outside the basin enters as weak noise rather than as information, so less of it
does less harm. That is the same shape as applying the term at 8 % instead of 17 %.

**Reference freshness is therefore a precondition of the term, not a refinement of it.**
The bound has to be per-point and expressed in image motion rather than in time, which is
what FAST-LIVO2 does: a new patch is added to a map point when 20 frames have passed *or*
its projection has moved more than 40 px (§V-C — note the paper's own section numbering
differs between arXiv versions).

Converting that 40 px is where a constant-pixel bound stops working, and the reason is
worth keeping. At `fx = 1338` for a 1920-wide frame, the working 480×360 level has
`fx = 334.6 px`, so a displacement `d` at scene depth `Z` moves a point by
`334.6 · d / Z` pixels. The depth that matters is the depth of the **confident** points
the term actually samples, which is closer than the scene as a whole — measured over the
eight loop sessions, `confidence == 2` medians run **1.21 – 3.35 m**, pooled median
**1.69 m**, where the all-valid median is up to 0.9 m further out.

| session | confident-point depth | 18 cm basin, in px | 40 px, in cm |
| --- | --- | --- | --- |
| 1868dd | 1.21 m | 49.7 | 14.5 |
| 1696fa | 1.47 m | 40.9 | 17.6 |
| cb4586 | 1.52 m | 39.6 | 18.2 |
| 3c7c6b | 2.11 m | 28.6 | 25.2 |
| 683ef1 | 2.61 m | 23.1 | 31.2 |
| 5acd1b | 3.35 m | 18.0 | 40.1 |

At the pooled 1.69 m the paper's 40 px works out to **20.2 cm** — within about 12 % of our
18 cm basin, so the rule is close to calibrated rather than badly wrong. But the *same*
18 cm is anywhere from 18 to 50 px depending on the session, a 2.8× spread, and at 5acd1b
40 px means 40 cm, which is more than twice the basin. **So the bound belongs in metres,
not pixels** — express it as predicted metric displacement against the measured basin, and
the depth dependence takes care of itself. Picking a smaller constant pixel threshold
instead would over-refresh in the close sessions while still being too loose in the far
ones.

**Gate on conditioning, but do not switch the image term off.** Sessions already
carry per-frame `cond` and `weakAxis` in `depth.jsonl` — the translation-only
conditioning computed on device, populated in the four newest sessions. That is the
signal that actually tracks where the geometric block goes wrong, and the table
above says the image term is informative at every conditioning value measured.
None of the four papers surveyed defines a rule for disabling one sensor's term;
all handle it with outlier rejection instead, and so should this.

## In a trajectory the anchor decides the sign

The per-pair case above is the strongest part of this document and it is not enough.
Wired into the depth-odometry trajectory and scored by loop closure — the one
reference-free measure this project has — the term was **net negative under the anchor
it was first given**, which the rest of this section takes apart. Figures from
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

What remains — and what has since been **confirmed** — is an **anchor mismatch**. The depth
block is frame-to-map: it aligns the current frame to an accumulated map. The
photometric block is frame-to-frame against the previous *image* frame, whose pose is
itself an estimate carrying its own error. The term is then accurate about a reference
that is already wrong, and every update pulls the current pose back toward that error.
That single mechanism explains all three observations — excellent per-pair behaviour,
no correlation between per-pair quality and trajectory damage, and damage scaling with
dose.

### Confirmed: give both blocks the same anchor and the sign flips

POSE.md tested it with one variable. `--frame-to-frame` moves the *depth* block's
anchor from the accumulated map to the previous frame, so both blocks reference the
same kind of thing; same sessions, same frames, same term, same weights. Loop closure
in metres:

| session | depth alone | with the image term | |
| --- | --- | --- | --- |
| 683ef1 | 5.122 | 1.782 | 2.9× better |
| 1696fa | 0.463 | 0.254 | 1.8× |
| 3c7c6b | 4.876 | 2.872 | 1.7× |
| 1868dd | 4.816 | 3.104 | 1.6× |
| 5bd1ed | 0.539 | 0.349 | 1.5× |
| f0d073 | 1.105 | 0.765 | 1.4× |
| 5acd1b | 2.154 | 1.564 | 1.4× |
| cb4586 | 0.217 | 0.231 | 0.94× — slightly worse |

**Seven of eight improve**, and the three sessions the term had destroyed under
frame-to-map are the ones it now helps: 3c7c6b improves instead of reaching 56 m,
5bd1ed instead of 9.4 m, cb4586 holds level instead of 3.4 m. The destruction was the
anchor mismatch, not the term.

The conditioning bands move with it. What was 1.53× worse in the `cond < 0.01` band
under a map anchor becomes **1.05×** — essentially neutral per frame — while loop error
falls 30–65 %. **Slightly noisier per frame, substantially better accumulated, is the
signature of removing a bias rather than a noise source.** The image term buys drift
resistance and pays a little jitter for it, which is what it was supposed to do.

Note the ordering: the *recommendation* to anchor both blocks to the same reference was
retracted and inverted in this document before this measurement existed, on the
strength of the mechanism alone. That is the one prediction this page made in advance,
and it held.

It also explains why this page's per-pair experiment could not see any of it, which is
the precise scope of everything above: **the reference pose here was taken from ARKit and held fixed, so
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

**Where this leaves the recommendation.** The measurement model is validated and the
integration failure is identified and fixed in the diagnostic setting. What is not yet
built is the version worth shipping: `--frame-to-frame` matches the anchors by making
the *depth* block weaker, which is why its baselines are worse than frame-to-map's. The
right target is the opposite — keep depth's map anchor and give the image term a map
reference too, as voxel-plane reference patches in the FAST-LIVO2 style. That is the
next step, and the numbers above are the reason to take it.

### And it is not enough: frame-level map anchoring does not transfer the gain

The `--frame-to-frame` result above says matching the anchors works. The obvious next step
is to keep depth's map anchor and move the *image* anchor into the map, which is what this
document recommended. **Measured, that does not transfer.** All figures below are from
POSE.md after a bug fix — before it, the two right-hand arms were silently the same
computation, so any earlier version of this table is void.

| session | ARKit | depth alone | B: fresh, outside map | C: stale, in map | E: dense map, no image | D: fresh, in map |
| --- | --- | --- | --- | --- | --- | --- |
| 1696fa | 0.126 | 0.123 | 0.120 | 0.126 | 0.130 | 0.124 |
| f0d073 | 0.286 | 0.694 | 0.545 | 0.481 | 0.613 | 0.476 |
| 1868dd | 0.378 | 1.381 | 2.152 | 0.889 | 1.749 | 1.680 |
| 683ef1 | 0.490 | 1.137 | 1.383 | 3.025 | 1.244 | 2.743 |
| cb4586 | 0.146 | 0.196 | 3.386 | 3.669 | 0.242 | **2.221** |
| 3c7c6b | 0.127 | 1.942 | 56.026 | 3.507 | 2.406 | 3.311 |
| 5bd1ed | 0.269 | 0.322 | 9.411 | 5.574 | 0.260 | 1.502 |
| 5acd1b | 0.055 | 3.078 | 2.675 | 2.285 | 2.680 | 3.430 |

Arm D has everything this page asked for: its reference is 8/8 inside the basin
(0.097–0.137 m, 1.7–4.2°), it lives in the map, and the term demonstrably runs. It beats
the depth-only control in **one session of eight**. All three reference states lose:

| reference state | beats control | worst case |
| --- | --- | --- |
| fresh, outside the map (B) | 3 / 7 | 29.2× |
| stale, inside the map (C) | 3 / 8 | 18.7× |
| **fresh, inside the map (D)** | **1 / 8** | 11.3× |

So changing *which frame* the image term references does not move the frame-to-frame gain
onto the frame-to-map path. Every frame-level variant hits the same wall, and the wall has
a shape: **in frame-to-frame both blocks reference one single frame; in frame-to-map the
depth block references an accumulation of dozens of keyframes, and no single reference
image can equal an accumulation.** That is the structural difference the anchor hypothesis
was always about, and frame-level experiments cannot address it.

**That makes per-point references the only surviving form of the hypothesis, rather than a
refinement of it.** If each point's photometric reference is the same keyframe that
contributed that point's geometry to the map, the two blocks' errors are identical *per
point* and cancel there. That is the only construction under which "the same anchor" is
even definable against an accumulated map — and it is what FAST-LIVO2 actually does, with
patches attached to voxels rather than a reference frame chosen per update.

One measured consolation, and it is an interaction rather than an effect. Going from B to D
holds the reference nearly fixed and only densifies the map, and it **suppresses the
catastrophes**: 3c7c6b 56.0 → 3.3 m, 5bd1ed 9.4 → 1.5, cb4586 3.4 → 2.2. Yet arm E — the
same densification with no image term at all — does nothing on its own (3 of 8). A denser
map stiffens the depth block and limits how far the image term can drag the trajectory. It
buys damage suppression, not accuracy, and it is worth knowing which of those you are
getting.

### Reweighting does not rescue it either, and that is the third dose result

The remaining frame-level lever is the weight on the photometric block. POSE.md had argued
against reweighting from the σ-ratio side; this settles it from the other side, with the
decision rule written down **before** the numbers came in: the term counts as carrying
information only if some `w > 0` beats the control in a majority of the eight sessions
*and* the gain is not monotone in the direction of switching the term off — that is, if
there is an interior optimum, which is what an over-weighted real information source looks
like.

| | control | w = 0.1 | w = 0.3 | w = 1.0 |
| --- | --- | --- | --- | --- |
| sessions beating the control | — | **5 / 8** | 3 / 8 | 3 / 8 |
| geometric mean ratio (control = 1) | 1.000 | **0.946** | 1.329 | 3.412 |
| worst session | — | 1.55× | 2.35× | 29.23× |
| interior optimum | — | 2 of 8 sessions | | |

The first condition passes and **the second fails**. Every summary improves monotonically
as the weight falls, and the limit of that trend at `w → 0` is exactly the control. So the
0.946 is not "a setting 5 % better than depth alone" — it is *almost switched off, and
therefore almost the control*, and with per-session ratios spread 0.67–1.55 it is not
distinguishable from 1.000 at n = 8. Only two sessions show an interior optimum.

**No weight makes a single-reference photometric term a net gain on the frame-to-map path.**

That is the third independent way of reducing the term's dose, and all three improve:

| way of using less of the term | result |
| --- | --- |
| apply it on 8 % of updates instead of 17 % | 6 of 8 sessions improve |
| give it a stale reference, so its constraint is weak | no catastrophes, still a net loss |
| turn its weight down from 1.0 to 0.1 | geometric mean 3.412 → 0.946 |

Three unrelated dials, one direction. **That is the signature of a noise source, not of an
over-weighted information source** — and it converges with the per-pair finding above that
measuring σ from the residual changed nothing. Weighting is not the lever at either level.

None of this speaks to per-point references. Every arm here uses one reference frame, so
these experiments vary the *dose* of a constraint already shown to be anchored wrongly.
A negative result narrows the remaining hypothesis rather than weakening it.

### The conditioning gate is not doing the job this page assigned it

*Gate on conditioning* above assumed the gate was live. In POSE.md's setting it is not: over
8 sessions and 7060 frames the map-insertion gate at `min_conditioning = 1e-3` fires on
**40 frames, 0.6 %**, and the worst session's p10 conditioning is 0.0143 — two orders of
magnitude above the threshold. Map insertion is therefore effectively ungated today, for
ordinary keyframes as much as for image ones.

**Do not port the thresholds in this document across to fix that.** Conditioning is not a
property of a frame; it is a property of a frame *registered against something*, and the
two settings differ systematically. Measured on the same eight sessions, the
translation-only conditioning at RGB cadence against a single previous frame's
`confidence == 2` points runs **median 0.013–0.069**, where POSE.md's frame-to-map figure
is **median 0.086–0.27** — three to five times better conditioned, because an accumulated
map spans more normal directions than one depth frame does. A threshold tuned on either
number is wrong for the other. What the two agree on is that 1e-3 is far below where the
variance lives.

The candidate worth pursuing is the one with live variance: ICP inlier fraction, median
96–99 % with minima of 51–84 %. Measured here at frame-pair level, where each pair has an
independent reference, it is worth knowing **how** to judge it — because the obvious test
throws it away:

| | |
| --- | --- |
| correlation with log ICP error, pooled over 125 pairs | **r = −0.196** |
| per-session correlation | −0.454 to +0.161, sign flips |
| median error, pairs below 0.92 inlier fraction | **12.06 cm** |
| median error, pairs at or above 0.92 | **1.87 cm** |
| ratio at cuts 0.98 / 0.95 / 0.92 | 2.50× / 4.70× / 6.46× |

The correlation is nil and the threshold behaviour is decisive, which is not a
contradiction: the signal is **saturated** — median 100.0 %, p10 98.8 % — so a
correlation over the full range is dominated by noise in the flat part. Conditional on
falling below a cut, the error is multiples worse, and the ratio grows monotonically as
the cut tightens.

**So a saturated gate signal has to be judged by conditional error ratio, not by
correlation**, or it gets discarded for looking inert. That is the opposite failure mode
from the two inert parameters this project has already found: those fired usefully on a
quantity that meant nothing, and this one rarely fires on a quantity that means a great
deal. Both are "the gate does not work", and the statistic that distinguishes them is
different in each case.

Two cautions carry over from the conditioning case. The saturation is worse here than in
POSE.md's frame-to-map setting (100.0 % against 96–99 %), because a frame-to-frame
association over `confidence == 2` points is cleaner — so the *cuts* do not transfer even
though the method does. And a gate that excludes 3 % of frames will move a loop score
barely at all, so the count of excluded frames and their conditional error have to be
reported alongside, or a small trajectory difference will be misread as the gate being
pointless.

### The frames the map is built from are the worst-registered ones

Found while checking whether the gate had anything to gate (POSE.md). Comparing the ICP
inlier fraction of frames that got folded into the map against all frames in the session:

| session | folded into the map | all frames | difference |
| --- | --- | --- | --- |
| 5bd1ed | 98.9 % | 99.4 % | −0.5 |
| cb4586 | 99.0 % | 99.4 % | −0.4 |
| 1696fa | 98.9 % | 99.4 % | −0.5 |
| f0d073 | 98.4 % | 99.0 % | −0.6 |
| 1868dd | 97.7 % | 98.7 % | −1.0 |
| 3c7c6b | 97.5 % | 98.5 % | −1.0 |
| 683ef1 | 96.3 % | 97.6 % | −1.3 |
| 5acd1b | 93.7 % | 96.2 % | −2.5 |

**Negative in 8 of 8**, and largest in the least-determined session. An earlier version of
this section blamed a variable that changes meaning between assignment and use; that was
wrong and the correction matters. `fill` is set from the map lookup before the solve and
then reassigned from the value ICP returns — but in the default path the association is
computed once at the predicted pose and reused across iterations, so what ICP returns is
that same constant. The reassignment is a no-op and the docstring's intent is intact.

**The adverse selection is the definition of a keyframe, not a defect in it.** A keyframe is
due after 5 cm of travel or 5° of rotation, so keyframes land at the *end* of each interval
— the moment furthest from the last map update, which is the moment of least overlap with
the map. The best-overlapped frames are the ones just after a keyframe, and the rule
excludes exactly those. Frames that are hard to register are frames carrying ground the map
does not have yet. Those are the same frames.

Which of those two readings holds decides whether patches can be attached to keyframes at
all — one is benign, the other is fatal to the per-point cancellation. **It has been
settled, and by a more direct measurement than the match fraction could give.** The
question is really "are keyframe *poses* worse", so POSE.md measured absolute pose error
after ARKit alignment for keyframes against the rest:

| session | keyframe inlier | keyframe error | other frames | |
| --- | --- | --- | --- | --- |
| 1696fa | 98.9 % | 4.3 cm | 4.3 cm | level |
| 683ef1 | 96.3 % | 34.1 cm | 34.2 cm | level |
| 1868dd | 97.7 % | 35.6 cm | 38.7 cm | keyframes better |
| cb4586 | 99.0 % | 7.1 cm | 7.6 cm | keyframes better |
| 3c7c6b | 97.5 % | 50.5 cm | 55.0 cm | keyframes better |
| 5bd1ed | 98.9 % | 10.2 cm | 11.4 cm | keyframes better |
| f0d073 | 98.4 % | 24.4 cm | 23.0 cm | keyframes worse by 6 % |
| 5acd1b | 93.7 % | 94.9 cm | 54.1 cm | **keyframes worse by 75 %** |

**Keyframes register less well and are not positioned less well** — lower inlier fraction in
8 of 8, and equal or better pose in 6 of 8. The benign reading holds: low overlap, not bad
pose.

The exception carries the rest of the answer. The one session with a real deficit is
5acd1b, which has 638 of 675 frames below the conditioning threshold — the session POSE.md
had already separated out as under-determined, to be detected and rejected rather than
improved. So **conditioning decides which reading applies**: where the geometry determines
the pose, taking keyframes at the moment of least overlap costs nothing, and where it does
not, the frame being folded in is a guess and the deficit is real.

**That sentence rests on one session.** Seven sessions say the benign reading holds and one
says it does not, and the one that does not is the one already flagged on independent
grounds — which is why it reads as a mechanism rather than as an outlier. But n = 1 on the
side that carries the explanation, so treat it as the best available account and not as an
established rule. The `max_dist` sweep on 5acd1b is what would promote it: if that session's
keyframe deficit shrinks as the threshold grows, the bad-pose mechanism is confirmed where
it matters.

That narrows the precondition usefully. It is not "fix keyframe selection" but "exclude the
sessions that were already going to be excluded", which this page wanted anyway.

Two limits on that measurement. Absolute error is dominated by accumulated drift and a
keyframe shares its drift with its neighbours, so this is a contrast between a keyframe and
*its neighbourhood*, not against an independent baseline — sound at that level and not
beyond. And it does not say *why* the inlier fraction is lower. The `max_dist` sweep above
would answer that independently, and the place it discriminates is 5acd1b: if that session's
keyframe deficit shrinks as the threshold grows, the misplaced-prediction mechanism is
confirmed on the one session where the two readings diverge.

Gating map insertion on the match fraction is not the fix, and measurement is emphatic
about why. At cuts of 0.92 and 0.95 the loop error's geometric mean rises to 4.60 and 4.63
against the depth-only control, beating it in 1 and 2 sessions of 8, with 5acd1b reduced
from 240 keyframes to 14. The gate's own input collapses with it — median inlier fraction
falls from 96–99 % to 50–69 % once the gate is on. **A map-insertion gate conditioned on
registration quality against that map is a positive feedback loop**: reject, so the map
starves, so the next frame registers worse, so reject more. That 0.95 is worse than 0.92 is
the signature. Any such gate has to be conditioned on something the map cannot influence —
conditioning computed from the frame's own points and normals, IMU self-consistency, or the
image.

## The image can also arrive as learned geometry, and one lesson transferred

The photometric term is one way to let the image constrain pose. A second is to hand a
short image sequence to a feed-forward reconstruction model and take its relative poses.
POSE.md owns those measurements; two results from that line belong here because they are
about method rather than about that model.

### Conditioning a reconstruction model on metric depth destroys the fit that makes it metric

π³ predicts **scale-invariant** local point maps: the reconstruction is determined up to one
global scale, so depth and translation share a single unknown. That shared unknown is the
entire reason a post-hoc fit works — a scale recovered from depths is the scale the
translations need.

Pi3X accepts metric depth as a conditioning input, and doing so looks obviously right when
metric depth is exactly what you have. It is the wrong move. Conditioning is soft — feeding
*ARKit's own poses* in returns a trajectory scaled 1.097 rather than 1.000 — so metric
information reaches the depths without reaching the translations, and the reconstruction
stops being a similarity of the truth. Then a scale fitted on depths is not the scale the
translations need, and the fit silently measures its own input.

Measured over 13 sessions, the ratio between the depth-fitted scale and the scale the
trajectory actually needed:

| | conditioned on depth | unconditioned |
| --- | --- | --- |
| degenerate sessions | 0.77 – 0.84 | 0.96 – 1.02 |
| median &#124;ratio − 1&#124; | 3.6 % | 2.5 % |
| sessions beating depth-only ICP | 12 / 13 | **13 / 13** |
| geometric mean of the error ratio | 0.661 | **0.411** |

**Feed the depth to the fit, not to the model.** And note that the correlation coefficient
is useless for reading this: it moves from −0.024 to +0.110 while the result improves
sharply, because success collapses the variance the correlation needs. Read the median
deviation and the movement of the degenerate sessions instead — the same trap as a saturated
gate signal, with the sign reversed.

### A hyperparameter held constant by a resource limit is a confound you cannot see

This page reported, from 13 sessions, that a learned model's scale fails under
rotation-dominated motion: rotation per metre against the scale error at **r = +0.705**,
against conditioning's −0.147. The arithmetic was right. The conclusion was wrong.

Every one of those measurements used a 24-frame window, and the window was 24 because a
16 GB GPU could not fit 48. On a larger machine at 96 frames the correlation does not
reproduce, and the session that had looked structurally hopeless — a sidestep along a
textured wall, which we had explained by rotation dominance — goes from 29.7 % of path error
to 3.9 %, beating depth-only ICP's 8.9 %. What the number described was **the window length,
not the scene.** Short windows starved the rotation-heavy stretches of information, and
chunk chaining then propagated it.

Two conclusions on this page were built on that and are retracted: the rotation-dominance
failure mode, and the claim that the two methods' failure axes are independent. Half of the
second survives — depth-ICP still fails as its normals degenerate — but the other axis was
an artefact.

The general form is worth more than either result. **A parameter you never varied because
your hardware fixed it is still a variable, and it is the hardest kind to see**, because it
does not appear in the sweep, the ablation table, or the correlation matrix. It is not the
familiar trap of a number belonging to something other than what you think; it is a constant
you did not choose being read as a property of the data. The check is cheap and we did not
do it: before believing any relationship, ask which settings were never varied, and why.

### Where that line stands: level with ARKit, which is what this page was waiting for

Single-pass reconstruction over a whole session, scale fitted to LiDAR depth, no
conditioning, rotation from the model. Thirteen sessions, from POSE.md:

| | ARKit | learned + LiDAR scale | depth-only ICP |
| --- | --- | --- | --- |
| loop closure, median | 1.5 % | **1.4 %** | 8.9 % |
| ATE, median | — | **6.9 cm** | 38.0 cm |
| sessions where it beats the other | — | 12 / 13 vs ICP, **5 / 13 vs ARKit** | — |

**A pose source that never looks at ARKit is now level with ARKit**, and ahead of it on five
sessions. That is the sentence this document has been working towards since its first
section: the ultra-wide path costs ARKit's pose, so it was blocked on a replacement, and the
replacement now exists at parity. It is non-causal — it sees the whole session at once — so
it belongs to offline episode export and **not** to anything that replaces online odometry;
that restriction is free for us, because episode export is offline anyway.

Two sessions remain worse than ARKit and were not fixed by longer windows, so they are
probably a real limit: the two sidesteps along a textured wall (3.6 % and 3.9 % against
ARKit's 3.1 % and 1.5 %). Both are outside the declared scope.

**One of them is the most useful failure on this page**, because of *how* it fails. 6b92f3
beats ICP on loop closure (3.6 % against 16.2 %) and loses to it on ATE (81.6 cm against
70.9 cm). The trajectory shape is wrong while the start and end happen to coincide — so the
reference-free metric passes while the trajectory is bad. Everything above has treated loop
closure as the honest score because it needs no reference, and this is a measured example of
it being fooled. **On the ultra-wide path, loop closure is the only score we will have.** So
the case for validating the method where a reference exists and carrying the *method* rather
than the score is not a stylistic preference; 6b92f3 is what the alternative looks like.

**And that failure can now be produced on demand, in the majority of sessions.** Narrowing the
field of view degrades the trajectory in all thirteen (above). Loop closure does not follow:

| as the field narrows | sessions |
| --- | --- |
| loop closure tracks ATE | 4 — 1696fa, f0d073, 2be6a9, 1868dd |
| loop closure stays flat | 2 — 3c7c6b, cb4586 |
| **loop closure moves opposite** | **7 — 683ef1, 5acd1b, 2735cf, 5bd1ed, ce02ac, dd2a13, 6b92f3** |

Counted the way that matters for a decision — **ATE got worse while loop closure got better in
8 of 13 sessions.** The clearest is 6b92f3, the session that already disagreed at native: ATE
81.6 → 131.4 cm while loop closure improves monotonically, 3.63 → 2.79 %. And 683ef1's ATE more
than doubles while its loop closure improves.

A start-and-end distance cannot see a trajectory bending and coming back, and degrading the
input is one of the ways to make it bend. So the risk on the reference-free path is not that
loop closure is uninformative — **it is that a change which makes the trajectory worse is read
as an improvement, in most sessions.** Tuning against loop closure alone walks in the wrong
direction more often than the right one.

Three independent demonstrations now say the same thing, and they share one cause — **loop
closure rests entirely on the last frame while ATE averages the whole walk**:

| | |
| --- | --- |
| 6b92f3 at native | beats ICP on loop closure, loses to it on ATE |
| narrowing the field | ATE worse and loop better in 8 of 13 |
| dropping 20 of 260 frames | loop moves 1.78 → 2.83 %, ATE barely moves |

**So loop closure comes off the acceptance criterion.** While ARKit exists, ATE is the score.
For the ultra-wide path, where ARKit does not exist, the replacement has to be built out of
signals that are independent of the estimator — and that inventory is now measured rather than
assumed:

| what it checks | signal | sensitivity | independent of depth and images? |
| --- | --- | --- | --- |
| rotation | CoreMotion attitude | 0.4–3.4° | yes |
| translation scale | **accelerometer, second derivative of the trajectory** | **±5 %** | **yes** |
| translation scale | two metric anchors disagreeing | unmeasured | no — both fit the same LiDAR depth |
| position | GPS | **none** | moot |

The accelerometer entry is the one that closes a real gap, because a trajectory scaled by *s*
has accelerations scaled by *s* and CoreMotion reports acceleration directly — never touching
the camera or the depth map. Band-limit both to 2 Hz, difference the trajectory twice, rotate
into the device frame, and least-squares fit one against the other: over 19 sessions the fit
lands in **0.919–1.020** with median correlation 0.875. A ±5 % band would have caught the 24–29 %
scale failures that conditioning produced, immediately. Two cautions: the fit is the usable
statistic and the RMS ratio is not, because differentiation noise adds in quadrature to a
magnitude but not to a cross term; and this was calibrated against ARKit, which is IMU-fused, so
it establishes the noise floor rather than proving independence — the confirming run is on a
depth-only trajectory, which needs no GPU.

GPS is measured and closed. Across 20 sessions its horizontal accuracy is 18.8 m against an
11.5 m median walk — **1.6× the whole trajectory** — and in 13 of them the reported position
does not move at all during the walk, while 16 report a vertical accuracy of exactly 30.0 m,
which is a placeholder rather than a measurement. It cannot bound anything at this scale.

The long-horizon machinery is not needed at this length. Sessions run 76–260 image frames;
all but one fit a single pass on a 98 GB card, and the one that needs four windows (260
frames) shows no sign of the seam step that 24-frame windows produced. Chunk alignment,
hybrid memory and Sim(3) loop closure become relevant only past that.

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
minimum, neither does anything else about the room.

**Four attempts to predict the trajectory damage from a session-level statistic have
now all failed**, and the pattern is worth stating because each kept looking plausible:

| proposed explanation | correlation with trajectory damage |
| --- | --- |
| per-pair photometric error (median / p90) | r = +0.191 / +0.151 |
| ICP conditioning | r = −0.150 |
| image sharpness (gradient RMS / Laplacian variance) | r = −0.335 / −0.290 |
| reference distance, within an arm | r = −0.001 (keyframe) / +0.588 (previous, n = 7) |

A fifth, for a different estimator, looked like the one that finally worked — rotation per
metre against a learned model's scale error, r = +0.705 — and it was an artefact of a fixed
window length. See *A hyperparameter held constant by a resource limit*. Five attempts, five
failures; the argument below for why there may be nothing to find has held up better than any
of the candidates.

Each has the sign its story predicts and none has the magnitude, on n = 8. The sharpness
attempt has a decisive counterexample: 5bd1ed is the second-sharpest session of the eight
and its damage is 29×. Meanwhile matching the anchors improves seven of eight. **The
damage is a property of the reference, not of the imagery** — three independent "the image
is bad" explanations failed while the one structural explanation held.

The fourth row needs care, because it is tempting to read reference distance as the
predictor that finally works. It is not. Within the keyframe-anchored arm the correlation
is nil, which is what a variable confined to one side of a threshold should give — all
eight sit outside the basin, so there is no contrast, only "how far outside". And across
arms the contrast exists but does not split the outcome: the previous-image anchor is
8/8 *inside* the basin and the keyframe anchor 8/8 *outside*, yet **both beat the
depth-only control in only three sessions**. Crossing the basin boundary changes the size
of the failure, not its sign. So reference distance belongs in this document as a
**precondition for the term to carry information at all**, never as an explanation of how
much harm it does.

There may be no session-level predictor to find, and there is a structural reason to
expect that. Trajectory damage is an **accumulated** quantity and therefore depends on the
order in which errors arrive; every statistic tried above is a summary of a frame
distribution, and a distribution erases order. Four failures is weak evidence on its own,
but combined with that argument it is enough to stop looking.

The remaining honest caution is about *capture* rather than about scene content:

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
2. ~~**Into the trajectory.**~~ **Failed, diagnosed, and fixed in the diagnostic
   setting.** Bolted on, it made five of eight sessions worse. Two explanations were
   killed by measurement — the 0.2 s image interval and intermittency — and the third
   held: the two blocks were anchored to different references. Matching them gives
   seven of eight better, 1.4–2.9×. Two cautions still stand: those sessions are all
   slow walks, and a diverging baseline's loop number moves under 0.027 % perturbations.
3. ~~**Settle why keyframes register worse.**~~ **Done, and it clears the way.** Keyframes
   register worse in 8 of 8 sessions but are positioned no worse in 6 of 8, so the benign
   reading holds and per-point patches may anchor to them. The one real exception is a
   session already marked for rejection on conditioning grounds, which turns this
   precondition into "exclude the under-determined sessions", a thing this page wanted
   regardless. Do **not** reach for an inlier gate on the way — measurement shows it runs
   away.
4. **Voxel-attached, per-point reference patches.** No longer one option among several:
   frame-level map anchoring has been measured and refuted (1 of 8), so this is the only
   surviving form of the hypothesis. Each point's patch references the keyframe that put
   that point in the map, so the two blocks' errors coincide per point and cancel there.
   Patches, not rendering — rendering needs a whole subsystem and patches do not, and
   pixel-registered depth makes attaching them cheaper here than in the papers. Bound patch
   refresh in **metres against the basin**, not in pixels. Note what this step is betting
   on: the only direction of anchor-matching that has been *shown* to work is the one that
   weakens the depth block, and the evidence that it can be inverted is one paper.
5. **Capture the missing controls.** Demoted from where an earlier draft put it: a
   higher RGB rate is not required, because the six-level pyramid handles the
   baselines this data contains. It is still worth shooting, for two things this set
   lacks. A **long straight corridor at a steady pace** is the degenerate case the
   whole document is about and there is no controlled example of one. A **fast walk**
   (~1.2 m/s) at 15 Hz tests the pyramid claim on real data rather than synthetic.
   Exposure is pinned at 1/60 s regardless of the stills rate, so a 15 Hz capture
   supplies its own 5 Hz control by decimation — a better control than walking the
   same corridor twice, which holds neither trajectory nor exposure history fixed.
6. **Then the ultra-wide** — no longer blocked, since the pose source is at ARKit parity.
   `AVCaptureMultiCamSession`, CoreMotion attitude, no ARKit. One check first, and it needs no
   ultra-wide ground truth: re-warp wide frames to 96.3° and confirm the pose accuracy
   survives the field-of-view change, because that is the only untested difference between
   what was validated and what would be deployed.

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
