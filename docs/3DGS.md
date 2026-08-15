# Can this recorder feed a Gaussian splat, and what has to change first

Two things this capture has that a photogrammetry pipeline does not: metric
scale, and a pose per frame that nobody has to solve for. Structure-from-motion
is the expensive, fragile first step of every 3DGS pipeline, and we can skip it.
That is a real advantage and it is why this is worth doing at all.

It is also not the thing that decides whether it works.

**The binding constraint is that a walk sees each surface once.** Measured over
all 23 sessions long enough to score, the median piece of surface is observed
from **one or two distinct directions**. Counting only walls and other vertical
surfaces — floors are covered for free and flatter the number — the best session
in the set gets 24 % of its surface to three views and the median session gets
10 %. A Gaussian is fitted, not measured: with one view its depth and extent are
unconstrained, and the optimiser fills them with whatever reproduces that one
image. That produces a splat which renders beautifully from the path walked and
falls apart one step to the side — which is exactly the product being asked for.

So this page is in the order the work has to happen: what the capture is, the
one measurement that ranks the problems, what is already built, and what to
change. The backend question turns out to be the easy part, and it comes last.

## What the capture is, measured

Twenty-seven sessions in `~/nav_data`, read with the numbers rather than the
manifest. Four are too short to score anything from — two of them never reach
`tracking == "normal"` at all — which is why the coverage tables below have 23
rows. Per-session configuration is uniform; the interesting spread is all in
what the operator did.

```
                    range across 27 sessions      typical
  images                 13 – 260                 ~110
  duration (s)            3 – 52                   24
  path length (m)         0 – 63                   12
  image rate            5 Hz (config)              5 Hz
  image size            1920x1440 JPEG q=0.85
  depth                 256x192 float16 metric, 30 Hz
  confidence            256x192 uint8, ARKit 0/1/2
  pose                  ~60 Hz, per-frame fx/fy/cx/cy
  exposure (ms)         1.3 – 16.7                16.7
  interframe baseline    3.7 – 26.5 cm            12 cm
  depth conf=high         2 % – 84 %              ~45 %
```

Three of those rows are worth pulling out.

**Exposure is pinned at 16.7 ms in 24 of 27 sessions.** That is 1/60 s, and it
is not a choice anyone made — `pickFormat` sorts by pixel area and the winner is
`1920x1440@60`, which caps the shutter. At the measured walking speed of about
0.6 m/s the camera translates **10 mm during a single exposure**. The three
sessions that got below 16.7 ms are the bright ones, where auto-exposure had
room to close down on its own.

**Depth confidence varies by a factor of forty between sessions**, 2 % to 84 %
high. That is not sensor variation; it is what the phone was pointed at. Glass,
distance and grazing angles all return low confidence, and a session at 2 % has
essentially no depth supervision to offer.

**Sharpness spans two orders of magnitude**, Laplacian variance 1 to 2115. The
low end is the night sessions, where the image is both blurred and nearly
textureless. Photometric loss on a frame like that is not a weak signal, it is
an ambiguous one.

## The measurement that ranks everything else

Number of images is the wrong question. Two hundred images of the same wall from
the same place is one observation. The question is **how many distinct
directions each piece of surface is seen from**, and it is directly measurable
from what is already recorded.

`tools/survey_coverage.py` measures it. Back-project every image-paired depth
map with its ARKit pose, drop the points into a 5 cm voxel grid, and for each
voxel count the number of *coarsely distinct* directions it was viewed from —
15° bins, so standing still and taking ninety frames counts once.

```bash
python3 tools/survey_coverage.py ~/nav_data/*/
```

Twenty-three sessions with enough tracked frames to score, best and worst, with
the ones named elsewhere on this page. These are **all surfaces** — see below
for why that number reads high and what to use instead:

```
  session  frames  median   >=2    >=3    >=5    range   linearity
   cb4586     167     2     58%    33%    11%    1.61m      2.0
   b36df8      81     2     58%    30%     3%    1.39m      9.3
   532cea     122     2     52%    28%     3%    2.02m     10.9
   2994fa     166     2     50%    26%     6%    1.84m      4.3
   5bd1ed     184     1     46%    21%     5%    2.09m      1.9
   1868dd     255     1     38%    12%     1%    1.57m      3.4
   2735cf     114     1     29%     5%     0%    2.17m     24.9
   5acd1b     113     1     24%     5%     0%    3.12m      3.3
   ce02ac      75     1     23%     4%     0%    2.00m     24.6
```

### What actually predicts it — and two things that do not

The obvious explanation is the shape of the walk, and it is wrong. Across all
23 sessions, correlated against three-or-more-view coverage **on vertical
surfaces**, which is the number that matters:

```
  predictor                       pearson   spearman
  median range to the surface      -0.61      -0.63
  camera pitch below horizontal    +0.36      +0.38
  path linearity                   -0.23      -0.03
```

**Distance dominates, and path linearity predicts nothing.** That is not what
this page originally said. The two most linear walks in the set do sit near the
bottom — but so do sessions with linearity 3.3, and the rank correlation says
the line shape is a coincidence of this set.

The geometry behind the distance term closes completely, and it is worth writing
down because it also prices every other lever. Walking a length `L` parallel to
a wall at distance `d`, the direction from a point on that wall to the camera
sweeps

```
      swept angle  =  2 · atan( L / 2d ),   capped at the field of view
```

The cap is exact and independent of distance: a surface stays in frame only
while it is within half the field of view of the optical axis, which is the same
angular window wherever the wall is. So **the field of view is a hard ceiling on
how much angular coverage any surface can ever get** — 70.6° here, which is 4.7
bins of 15°, and no amount of walking adds a fifth.

Distance does not change that ceiling. It sets the **price**, because reaching
the ceiling costs `2·d·tan(FOV/2)` metres of walking parallel to that surface:

```
  wall at 0.8 m   ceiling reached after  1.13 m of walking past it
  wall at 1.5 m                          2.12 m
  wall at 3.0 m                          4.25 m
  wall at 6.0 m                          8.50 m
```

That is the whole correlation. These sessions are 9–30 m long but they turn, so
the stretch spent parallel to any *one* wall is short — a couple of metres. At
1.5 m that is enough to finish the sweep; at 3 m it buys less than half of it.
It also explains why the measured medians are 1–2 rather than the 4.7 the
ceiling allows: almost nothing in these captures gets its full parallel run.

Report vertical surfaces separately or the number lies. Floors and ceilings are
covered nearly for free by any walk, and they can carry the aggregate on their
own:

```
                       all surfaces   vertical only
  31c6aa                     23 %            2 %
  af9142                      5 %            0 %
```

31c6aa keeps 70 288 surface voxels in total and 13 216 of them vertical, so
81 % of what it captured was floor or ceiling. It looks mid-table until those
are taken out, and then it is the worst in the set.

`tools/plot_coverage.py` draws the same measurement as a plan view — the room,
the path walked through it, and each column of wall coloured by how many
directions saw it. It is the readable form of the table and it takes its numbers
from the same function, so the two cannot drift apart.

```bash
python3 tools/plot_coverage.py out.png ~/nav_data/20260809-074458-cb4586
```

Put two sessions of the same room side by side and it is the stage-4 experiment.
`--vertical-only` is the flag; use it. The same nine sessions, scored that way:

```
  session  frames   >=2    >=3    >=5    range
   b36df8      81   49%    24%     2%    1.16m
   cb4586     167   45%    21%     5%    1.45m
   2994fa     166   42%    17%     1%    1.78m
   5bd1ed     184   39%    15%     2%    2.09m
   532cea     122   33%    14%     1%    1.83m
   1868dd     253   36%    10%     1%    1.46m
   2735cf     114   28%     5%     0%    1.74m
   ce02ac      75   24%     5%     0%    1.63m
   5acd1b     113   26%     6%     0%    3.05m
```

Note what happens to 532cea, the session with the sharpest images: 26 % on all
surfaces, 14 % on walls. It was walked down the middle of a wide concourse at
1.83 m median range, and half of what it captured was floor.

The bin width is not doing the work. Sweeping it on 5bd1ed:

```
  bin      median  >=2 views  >=3 views  >=5 views
  30.0°       1      32 %        10 %       1 %
  15.0°       1      46 %        21 %       5 %
   7.5°       2      58 %        36 %      14 %
   3.75°      2      67 %        49 %      30 %
```

Even at 3.75° — a definition of "different view" so generous that at 2 m it
means a 13 cm baseline — half the surface fails to reach five views.

**More images do not fix this.** 1868dd has the most frames of any session, 255,
and 12 % three-view coverage — 10 % on walls, at the median of the set: it
walked further, so the extra frames bought new surface rather than new angles on
old surface. The
self-test in `tools/test_survey_coverage.py` states the same thing where it can
be checked against arithmetic — ten times the frames along one line adds at most
one direction cell.

This is the number to move, and nothing in the training code moves it.

## What follows for capture

Ranked by the evidence behind them, and labelled with what that evidence is.
Two entries that were on this list before the correlations were run have been
removed, and they are recorded at the bottom rather than quietly deleted.

1. **Get closer to what you want reconstructed.** *Measured, strongest signal
   (Spearman −0.63).* Stay within about 1.5 m of the surfaces that matter. The
   top of the table is 1.4–1.6 m; the bottom includes a session at 3.1 m. In a
   corridor this means walking near one wall and then the other, not down the
   middle — which also happens to give the two passes below.
2. **Pitch the camera down 25–35°.** *Measured, moderate (+0.38 on vertical
   surfaces).* The reason is not that floors matter — half of the raw +0.61
   correlation was exactly that, and it disappears when horizontal surfaces are
   excluded. What survives is that pitching down puts the *near* part of the
   room in frame instead of the far end of the corridor, which is item 1 by
   another route.
3. **If you walk it twice, change the height, not the line.** *Geometry, not
   observed in this data — but the geometry is exact and it reverses what this
   page said first.* A second pass at a different distance from the same wall
   adds **zero** new viewing directions: the window a surface is visible through
   sits around the same axis wherever the camera is, so the second pass retraces
   the first one's arc. Raising or lowering the phone by 0.8 m tilts the arc out
   of that plane and every direction it visits is new.

   ```
     second pass                        cells it gets   new ones
     the same line again                       6            0
     1.0 m further from the wall               4            0
     0.7 m nearer the wall                     6            0
     same distance, 0.8 m higher               6            6
     same distance, 0.8 m lower                6            6
   ```

   Walking a *different line through a room* is still worth something, but for a
   different reason than the one originally given: it puts you near different
   walls, which is item 1 applied to more of the room. It is not a second look
   at the same wall. `tools/test_survey_coverage.py` pins this.
4. **Take the 30 fps format.** `pickFormat` currently sorts on pixel area, which
   selects `1920x1440@60` and pins the shutter to 1/60 s. The same resolution
   exists at 30 fps and would let auto-exposure open to 1/30, halving noise in
   the dark sessions. It is a trade, not a free win — motion blur doubles at the
   same walking speed — which is the argument for making it a *setting* and
   measuring both arms, not for flipping the default. See §"Not settled".
5. **Raise the still rate for scanning.** *Measured, and it is small.* 5 Hz is a
   VLN dataset rate and the setting already goes to 30 Hz. Depth is already
   recorded at 30 Hz, so what a six-times-denser capture would do to coverage
   can be measured on sessions we have rather than argued about — count over
   every depth frame instead of only the image-paired ones:

   ```
     session   at 5 Hz          at 30 Hz         gain
      cb4586   167 fr, 19.7%   1007 fr, 20.3%   +0.6 pp
      2735cf   114 fr,  5.0%    688 fr,  6.0%   +1.0 pp
      2994fa   166 fr, 14.6%   1001 fr, 18.6%   +4.0 pp
   ```

   Six times the frames buys under one point on the two sessions that walk in a
   line, and four on the one that turns the most. Exactly what the geometry
   says: at 5 Hz consecutive frames are already 12 cm apart, which at 1.5 m is
   under 5°, so the frames in between land in bins that are already occupied.
   It is not nothing — more images also means more photometric supervision,
   which this measurement says nothing about — but it is not the fix, and it
   costs six times the storage and the thermal budget that `HANDOVER.md` §2
   shows is already spent.

Nothing in the top three is a code change to the reconstruction. That ordering
is the point of the section — and note that items 1 and 3 say the same thing
from two directions, which is what a real mechanism looks like: item 1 makes a
surface's angular sweep finish, item 3 adds a second sweep in a plane the first
one could not reach.

**Two recommendations this page used to make, withdrawn by measurement.**

- *"Sweep the phone while walking — rotation adds view directions."* It does
  not. The direction a surface is viewed from depends on where the camera **is**,
  not where it points, so rotating in place contributes nothing to this
  measurement. Rotation while translating does help, but only by keeping a
  surface in frame for longer — which is a weaker version of item 1, and the
  angle between camera aim and direction of travel correlates with coverage at
  −0.03, i.e. not at all.
- *"Stop pointing at the floor."* Backwards. Pitching down is the second
  strongest predictor in the table, and it survives excluding the floor itself.
  The original argument — that floor pixels are wasted because the walk already
  covers the floor — is true about floors and wrong about what pitching down
  does to the rest of the room.

Both were plausible, both were written before the correlation was run, and both
were wrong. That is the argument for `tools/survey_coverage.py` existing at all:
a capture-protocol opinion is cheap to hold and cheap to check, and these two
cost about twenty minutes to falsify.

## What is already built

Four tools, each with a self-test that puts a known geometry in and demands the
known answer back:

```
  tools/export_3dgs.py       a session -> a training set, in both conventions
  tools/survey_coverage.py   the coverage measurement this page is built on
  tools/plot_coverage.py     the same measurement drawn as a plan view
  tools/eval_views.py        held-out scoring, with controls that can win
```

`python3 tools/test_export_3dgs.py`, `test_survey_coverage.py` and
`test_eval_views.py` cover them; the middle one covers `plot_coverage.py` too,
since the two share their counting.

### The export

One session in, a training set out.

```bash
python3 tools/export_3dgs.py ~/nav_data/20260813-162849-2994fa /tmp/gs
```

It writes, in one pass:

```
  images/              the JPEGs, symlinked (or resized, with intrinsics scaled)
  sparse/0/            COLMAP text model — cameras.txt, images.txt, points3D.txt
                       and points3D.ply, which is where the reference 3DGS loader
                       looks before it falls back to parsing the text
  sparse_pc.ply        the same cloud under the name transforms.json points at
  depths/              per-image metric depth, 16-bit millimetres, image-sized
  depth_normals_mask/  the confidence mask, in DN-Splatter's polarity
  transforms.json      the same cameras in Nerfstudio's convention
  holdout.txt          the frames reserved for evaluation
  export.json          what was kept, what was dropped, and why
```

`--depth-format npy` swaps `depths/*.png` for `depth/*.npy` holding float metres
and sets `depth_unit_scale_factor` to 1.0, which is the layout DN-Splatter's
parser reads. It is not the default because a full-resolution frame is 11 MB
that way; at `--downscale 2` a session costs about 440 MB.

**The mask polarity is inverted and it is silent when wrong.** DN-Splatter
computes `1 - mask/255` and keeps what is positive, so **0 means valid and 255
means invalid**. Writing the intuitive polarity would train on exactly the
pixels the LiDAR said not to trust, and nothing renders the mask, so it would
never be seen. The mask is also binary, because the loader thresholds rather
than weights — the ARKit levels are collapsed here, where the collapse is
visible, rather than somewhere it is not. Continuous weighting needs a patch to
DN's regulariser, not a different file.

Both conventions are written because they genuinely differ, and the difference
is a trap. COLMAP wants world-to-camera with +Z forward and +Y down; Nerfstudio
wants camera-to-world in OpenGL axes, which **is ARKit's camera frame verbatim**.
So `transforms.json` gets the raw pose and `images.txt` gets the basis change —
and a converter that emitted the same matrix into both would look right in one
of them.

Four decisions in there are load-bearing:

- **One COLMAP camera per image.** Autofocus is on. `docs/DATA_FORMAT.md:129`
  records `fx` swinging up to 11 % within a session; the milder cb4586 measures
  0.9 % across 157 distinct focal lengths in 167 frames, with the principal
  point never landing on the image centre. A shared camera model absorbs all of
  that into the geometry instead.
- **Depth intrinsics from the dimension ratio, never from `2·cx`.** 256x192 is
  exactly 1/7.5 of 1920x1440. The principal point is an optical quantity that
  moves with focus; using it as a width proxy injects that motion into every
  back-projected point. The app's own conditioning metric makes this mistake
  (`ARRecorder.swift:578`) and `docs/DATA_FORMAT.md:134` already flags it.
- **Split on `ar.interruptionEnded`.** ARKit re-origins after an interruption.
  Frames either side are not in one coordinate system, and concatenating them
  produces a reconstruction that cannot converge for a reason nothing reports.
  None of the 27 recorded sessions contains one, so this path is covered by the
  self-test and has never run on real data — worth knowing before trusting it on
  the first session that does interrupt.
- **The init cloud averages within a voxel rather than picking a survivor.**
  Picking keeps one frame's noise; averaging cancels it across the frames that
  saw the same surface, and the observation count comes back as the only honest
  per-point confidence available.

### How it was checked

The self-tests are synthetic and decisive — a known geometry in, the known
answer demanded back. The one that matters most puts **two cameras a known
baseline apart looking at one wall**: a single camera stays self-consistent
under a flipped translation sign, because the same flip applies going out and
coming back, but two cannot — a sign error puts the second camera's copy of the
wall at twice the baseline instead of on top of the first.

That still cannot show the algebra was applied to the *right* data. So a second
check reads only the exported files, splats the point cloud through each COLMAP
camera, and compares the colour that lands on a pixel with the colour the JPEG
has there — **with a control arm**: the same splat against an unrelated frame
from the same session.

```
                          reprojection   control   ratio
  532cea (station, day)     13.0 / 255   43.5/255   3.3x
  2994fa (room, night)       9.5 / 255   73.0/255   7.7x
```

Without the control the first column means nothing — a dim scene scores well
against anything. The ratio is the result.

Third, the format itself is checked by the reference implementation rather than
by our own reader agreeing with our own writer. `pycolmap` opens the cb4586
export and reports 167 PINHOLE cameras, about 404 000 points, and — projecting
the cloud through a mid-sequence camera — 48 % of points ahead of it and 43 % of
those inside the frame. Two things fell out of that read worth keeping:

- **157 of 167 cameras have a distinct `fx`**, spanning 0.9 % within this
  session. Autofocus, confirmed from the written file rather than the source.
- **`cx` runs 958.1 to 959.4 and never reaches the image centre of 960.0.** The
  principal point is a real optical quantity sitting about a pixel and a half
  off centre and moving, which is the concrete reason depth intrinsics are
  scaled by the dimension ratio and never by `2·cx`.

One expected gap: `num_observations = 0`. There are no 2D tracks because the
points were measured rather than triangulated. Gaussian-splatting trainers read
`points3D` for position and colour only, so this is fine — but a tool that wants
a bundle-adjustment-shaped model will find it empty.

### The bar a trained splat has to clear

`tools/eval_views.py` scores held-out views, and reports the controls next to
whatever a trainer produces rather than instead of it.

```bash
python3 tools/export_3dgs.py ~/nav_data/20260809-074458-cb4586 /tmp/gs
python3 tools/eval_views.py /tmp/gs
```

The export reserves every eighth frame for evaluation, and — this is the part
that is easy to get wrong — **keeps those frames' depth out of the initial point
cloud** while still scoring against it. Otherwise the trainer begins holding the
answer to the question it is about to be asked.

The split is `index % 8 == 0` and not something better centred, because that is
what `gsplat`'s `test_every` and the reference 3DGS loader mean. A split that
disagrees with the trainer's is worse than none: the frames withheld from the
cloud would be training views, and the frames actually scored would be ones
whose geometry was handed over at initialisation. Verified against `gsplat`'s
own parser on cb4586 — same 21 names, both sides.

Measured on cb4586, 146 training views and 21 held out, at quarter resolution:

```
                PSNR    depth error   abs-rel   coverage
  mean image   13.05          -           -         -
  lidar splat  17.32       0.038 m     0.020      28 %
```

The point cloud only lands on 28 % of a held-out frame, so its PSNR is scored
on those pixels alone — and the floor is re-scored on the same subset for the
comparison to mean anything. It gets **13.15 dB** there, so the raw measurements
beat the trivial baseline by 4.2 dB on like-for-like pixels. Quoting 17.32
against the full-frame 13.05 would have been comparing a quarter of the image to
all of it.

Both numbers are pinned before any training run. The first is the floor: a
model that does not clear 13.05 dB has learned the session's exposure, not its
geometry. The second is the one that matters — it is what the raw measurements
alone already predict at a viewpoint they were not given.

One honesty note about that 3.8 cm: the truth it is scored against is the
held-out frame's *own* LiDAR depth, so this is the cross-view self-consistency
of the depth-and-pose system, not absolute accuracy. That is the right
comparator — a splat is asked to predict exactly that held-out depth map — but
it does mean 3.8 cm is roughly a noise floor, and a splat landing near it has
matched the sensor rather than beaten it.

## Backend

### The local build constraint, measured

The system CUDA toolkit is nvcc 12.8; the PyTorch that works here is
2.13+cu130. That split is not a detail, it is the whole build question, and it
was settled by running it rather than reading about it:

```
  gsplat + torch 2.13.0+cu130, nvcc 12.8   FAILS
      RuntimeError: The detected CUDA version (12.8) mismatches the version
      that was used to compile PyTorch (13.0)
  gsplat + torch 2.9.1+cu128,  nvcc 12.8   builds, 50 objects, sm_89
```

**A CUDA-extension backend needs a torch cu128 wheel on this box**, not the
cu130 one `eval/.venv` runs. That is a second environment, not a change to the
existing one — `eval/` is pinned to what produced the numbers in `docs/POSE.md`
and must not be disturbed. Build environments live in `../gs3d/venvs/`, results
in `../gs3d/BACKENDS.md`, and the acceptance criteria were frozen before any run.

Installing `cuda-toolkit==13.0.3.0` into the cu130 environment did **not** fix
it: the PyTorch extension builder still picked `/usr/local/cuda/bin/nvcc`.

### What actually built and trained here

Every row is a trainer that was run, not read about. Smoke is 1000 iterations on
a 12-view synthetic known-pose scene; VRAM is the peak line from
`nvidia-smi -l 1`, so it is whole-device rather than process.

```
  backend               built  smoke  peak VRAM  s/1k iters  poses  metric depth
  DN-Splatter            yes    pass    2225 MB    25.6       yes   yes
  gsplat 1.5.3           yes    pass    1975 MB     4.0       yes   yes, needs an adapter
  Nerfstudio splatfacto  yes    pass   10920 MB    24.0       yes   NO supervision
  Brush (Rust/WGPU)      yes    pass    1798 MB     3.8       yes   no
  INRIA gaussian-splat   yes    pass    2027 MB     5.1       yes   inverse-depth PNG only
  OpenSplat              NO      -         -         -         -    -
  3DGRUT                 not tested
```

Three of those are decisions rather than data points. **Splatfacto renders depth
but has no metric-depth target loss**, so the obvious choice — the one everything
recommends for phone capture — is the one that cannot use the signal this project
exists to exploit. **Brush builds without CUDA at all**, which makes it the
escape hatch if the packaging fight ever restarts, but it has no depth input
either. **OpenSplat did not configure**, stopping at a missing `OpenCVConfig.cmake`.

**Use DN-Splatter.** It is the only tested trainer that took known poses, metre
depth, a confidence-mask layout and depth-derived normal supervision through a
complete run. Runner-up is current `gsplat` with a project-specific adapter and
loss — four times faster in the smoke and its CUDA extension builds cleanly —
and the reason to switch would be that DN's pinned `nerfstudio==1.1.3` /
`gsplat==1.0.0` stack becomes more expensive to keep than the adapter is to
write. The recipe as it actually ran is in `../gs3d/BACKENDS.md`; note that DN's
own metadata cannot be installed whole on this box (`vdbfusion` has no wheel,
`PyMCubes` will not build against NumPy 2) and neither is imported by the
depth-training path.

### What the literature settles, independent of the build

The one result that matters most is a confirmation of the plan below rather
than a surprise. DN-Splatter (Turkulainen et al., WACV 2025,
[arXiv:2403.17822](https://arxiv.org/abs/2403.17822), code at
[maturk/dn-splatter](https://github.com/maturk/dn-splatter)) reports, for adding
**sensor** depth supervision to indoor scenes:

```
  depth abs-rel   0.1481 -> 0.0351      about 4x better
  PSNR             23.41 -> 23.74       about +0.3 dB
  Chamfer-L1      0.0749 -> 0.0239      about 3x better
  mesh F-score     0.584 -> 0.924
```

**Depth supervision barely moves PSNR and transforms geometry.** A stage that
accepted or rejected it on PSNR would have concluded it does nothing. That is
exactly why stage 3 below is scored on rendered-depth error, and it was worth
writing that down before seeing these numbers rather than after.

Three things follow regardless of which trainer wins:

- **Depth supervision is not optional here, it is the fix.** Every remedy for
  the one-view problem works by supplying the geometry that multi-view
  triangulation would otherwise provide. We have that geometry in metres, per
  pixel, with a confidence channel — better than the monocular depth most of
  this literature has to settle for, and DN-Splatter's own sparse-view ablation
  shows monocular priors buying only +0.2 to +1.1 dB where sensor depth buys
  the numbers above.
- **The confidence map should weight the depth loss, not gate it.** A hard
  `conf >= 2` mask throws away the medium-confidence returns that are most of
  the far geometry, and sessions measured at 2 % high confidence would
  supervise on almost nothing. CDGS ([arXiv:2502.14684](https://arxiv.org/abs/2502.14684))
  is the published form of per-pixel confidence weighting; DN-Splatter's own
  weighting is derived from colour gradients instead, so folding the ARKit
  channel in is an implementation step on top, not a flag.
- **SfM is not needed and should not be run.** Poses are recorded, and COLMAP on
  a forward walk with 12 cm baselines is precisely the case it fails on. This
  also rules out the pose-free and few-view families — InstantSplat, CF-3DGS,
  dust3r — which spend their effort re-estimating what ARKit already gives, and
  the whole online-SLAM tier (SplaTAM, MonoGS, Splat-SLAM, Photo-SLAM) for the
  same reason.

Two representation choices are worth testing once the pipeline runs, both
cheap on 16 GB: **MCMC densification** ([arXiv:2404.09591](https://arxiv.org/abs/2404.09591),
shipped in gsplat as a pluggable strategy) in place of the hand-tuned
clone/split heuristics, and **2D surfels** ([2DGS, arXiv:2403.17888](https://arxiv.org/abs/2403.17888))
in place of 3D ellipsoids, which is aimed squarely at the wall-floor-ceiling
geometry a corridor is made of. Neither is a substitute for stage 4.

## Merging sessions, which is not optional

A session cannot exceed about 35 seconds — `HANDOVER.md` §2 measured it, and
that is the ceiling on everything above. A room does not fit in 35 seconds. So
several fragments have to become one map, and two ARKit worlds share nothing:
each picks its own origin and its own yaw.

`tools/align_sessions.py` does it.

```bash
python3 tools/align_sessions.py ~/nav_data/<a> ~/nav_data/<b> \
    --control ~/nav_data/<somewhere-else> --coverage
```

**They do share gravity.** `worldAlignment = .gravity` means both worlds are
+Y up, so the unknown is a yaw and a translation — four degrees of freedom, not
six. That is what makes the global search affordable: sweep the yaw in full,
and for each yaw the translation falls out of a cross-correlation of two floor
plans. Nothing has to guess an initial pose. One pair here needed **270°** of
yaw and the sweep found it.

Only vertical surfaces enter the plan. A floor correlates with any other floor
at every offset, and an early version matched two corridors by their floors and
was confidently four metres wrong.

### It works, on five pairs

```
  pair                yaw    tilt   fitness@5cm   control   ratio
  5bd1ed <- cb4586   357°   0.28°       85%         12%      7.4x
  1868dd <- f0d073   270°   0.59°       51%          6%      7.9x
  2735cf <- ce02ac     3°   0.18°       65%         10%      6.4x
  2994fa <- 7d3d52    18°   2.59°       60%         17%      3.6x
  2be6a9 <- 02a524   174°   1.27°       32%         13%      2.5x
```

`tilt` is how far the recovered rotation tips the vertical, and it is a
measurement rather than an error: gravity is supposed to be shared, so this
should be zero. It is under 0.6° on three pairs. The outlier is the night pair,
which is exactly where `HANDOVER.md` §1 says ARKit was struggling in the dark —
and it is the only pair where constraining the fit to four degrees of freedom
made it *worse* (51 % against 60 %). The gravity really did disagree there.

### The gate was wrong first, and the repo already said why

The first version scored alignment by ICP's point-to-plane residual, and the
control beat the true pair: **1.4 cm against 2.0 cm.** The control was better by
the number being used to judge it.

Two things were wrong and both are recorded in `HANDOVER.md` §6. The residual is
what ICP minimises, so it grades itself. Worse, it is conditioned on matching —
a registration that finds a home for a fifth of its points is scored on that
easiest fifth. The control matched 21 % of its points and looked tight doing it.

The criterion is now **fitness**: the fraction of *every* source point that
lands within a fixed distance of any surface. Failing to place a point costs
exactly what placing it wrongly costs, and there is no subset to hide in. On
that measure the control is 12–17 % against the true pairs' 51–85 %.

### A criterion that could not fail, and had to be caged

The coverage gain below is the right question to ask about a merge. It is also,
on its own, **worthless as a check that the merge is real** — and finding that
out cost the third gate design on this page.

Aligning a subway concourse against an apartment gives a fitness of 0.12, which
the control test rejects immediately. Ask the same pair for its coverage gain
and it reports **+45 percentage points** — higher than any genuine pair here.
The reason is mechanical: forced into one frame, two unrelated clouds still put
some points in the same voxels, and those voxels then collect views from two
arbitrary directions. On 1 318 accidental voxels that reads as spectacular
improvement.

So `--coverage` now refuses to compute unless `--control` is given *and* the
alignment cleared it. The order is enforced rather than documented: fitness
against a control establishes that the merge is real, and only then does
coverage say whether it was worth taking. The numbers in the next section were
all produced through that gate.

### The criterion that actually matters

Merging always adds area, and area proves nothing. The question is whether the
surface *both* sessions touched gains viewing directions — which is the quantity
this whole page is about. `--coverage` restricts the measurement to the shared
columns:

```
  pair               shared voxels   one session   merged    gain
  5bd1ed <- cb4586       11 851        22% at 3+   45%      +23 pp
  2994fa <- 7d3d52        7 071        21% at 3+   42%      +21 pp
  2735cf <- ce02ac       11 820         8% at 3+   15%       +7 pp
```

Scoring the merged map as one space says the same thing from the other side, and
it is worth seeing because the aggregate on its own reads as a disappointment.
`survey_coverage.py --transforms` on 5bd1ed + cb4586:

```
  the merged map, all of it            56 629 voxels   20% at 3+ views
    surface both walks reached         13 755  (24%)   47% at 3+, 13% at 5+
    surface only one walk reached      42 874  (76%)   11% at 3+,  1% at 5+
```

The two sessions alone score 15 % and 21 %, so 20 % looks like merging achieved
nothing. It is a weighted average of two very different populations: **four
times the coverage where the walks overlap, and no change at all where they do
not.** Merging cannot add an angle to a place the second walk never went.

It is not a property of that pair. Across every pair the alignment verified:

```
  pair               both walks reached        one walk only     ratio   overlap
                    voxels   >=3    >=5       voxels    >=3               share
  5bd1ed <- cb4586  11 877   44%    12%       38 235    11%      4.0x      24%
  2994fa <- 7d3d52   7 071   42%    14%       37 903    11%      3.8x      16%
  1868dd <- f0d073   4 309   58%     8%       50 358     9%      6.6x       8%
  2be6a9 <- 02a524   2 382   23%     1%       35 276     4%      5.7x       6%
  2735cf <- ce02ac  11 820   15%     2%       24 445     1%     13.4x      33%
```

Between four and thirteen times, on every pair. And the overlap share is
between 6 % and 33 % — so most of a merged map is still surface one walk saw
once. The lever is strong and its reach is an operator choice.

That is an instruction, not a caveat. When a room is captured as several
fragments — and the 35-second ceiling means it must be — the fragments should be
walked to **overlap on purpose**. Two disjoint walks buy area. Two overlapping
walks buy reconstruction.

**A second session roughly doubles three-view coverage on the shared surface.**
Set that against the other lever measured on this page: six times as many frames
along one path buys 0.6 to 4.0 percentage points. A second 35-second walk buys
7 to 23. The geometry said why in advance — new frames on one path add no
directions, new *positions* do — and a second session is nothing but new
positions.

This is now the strongest coverage lever there is, it costs one more walk, and
it needed no change to the app.

### More than two, and which sessions are even the same place

`tools/align_set.py` takes a set and returns one frame.

```bash
python3 tools/align_set.py ~/nav_data/*/ --out transforms.json
python3 tools/export_merged.py transforms.json /tmp/gs_merged --depth-format npy
```

It aligns every ordered pair, admits the edges that stand out, and grows a
maximum spanning tree from the reference — so a fragment that shares nothing
with the reference can still reach it through a neighbour. Each session is then
scored *directly against the reference cloud* after composition, because an edge
being good says the hop was good and only that says the session landed in the
right place.

**An absolute threshold was the wrong instrument, and it took a real pair to
show it.** The first version admitted an edge above 0.35 fitness and refused a
genuine pair at 0.32 — a pair that beats an unrelated-room control 2.5x and
gains 20 points of coverage when merged. Absolute fitness measures how much of a
session found a home, so it falls with how little area two sessions happen to
share: that pair overlapped on 2 364 voxels where an accepted one overlapped on
11 851. Nothing was wrong with the alignment; the threshold was measuring
overlap area and being read as alignment quality.

Edges are now admitted on a **ratio against the other candidates in their own
row**, which are the controls the matrix already contains. On four sessions
recorded within two minutes:

```
  fitness                              ratio to the rest of the row
          02a524 2735cf 2be6a9 ce02ac          02a524 2735cf 2be6a9 ce02ac
  02a524       -   0.12   0.34   0.13   02a524      -    0.5    2.7    0.6
  2735cf    0.15      -   0.20   0.65   2735cf    0.4      -    0.5    3.6
  2be6a9    0.32   0.14      -   0.15   2be6a9    2.2    0.6      -    0.6
  ce02ac    0.12   0.76   0.23      -   ce02ac    0.2    4.3    0.5      -
```

True edges sit at 2.2–4.3x, everything else at 0.2–0.6x. And the answer is that
**those four recordings are two different rooms**, which the tool reports as two
groups rather than as a failure to align. Point it at a folder and it says which
recordings are of the same place.

### Making the merged arm comparable to the single one

`export_merged.py` writes the reference session's frames first and **holds out
only those**, so a single-session export of the same reference is scored on the
very same photographs. Left to itself the every-eighth rule would sample a
different set of frames from the longer merged list, and the two runs would not
be a comparison. Verified: both arms hold out 23 frames with identical names and
identical image bytes.

### The held-out numbers on this page are flattered, and by how much

Every eighth frame of a 5 Hz walk is 12 cm from the frames on either side of it,
so a held-out view is usually an interpolation between two images the trainer
was given. Measured on 5bd1ed: **6 of 23 held-out views have a training camera
within 10 cm and 10°.** That is a property of the split, not of the merge — the
merged arm has the same 6 of 23 — so the comparison below is fair, but the
absolute PSNR everywhere on this page should be read with it in mind.

`--guard-m` removes those neighbours from training:

```
  guard band   training images   near-duplicate held-out views
     none            184                 6 of 23
     0.15 m          129                 0 of 23
     0.25 m          116                 0 of 23
     0.40 m           89                 0 of 23
```

Fifteen centimetres clears them all and costs 30 % of the training set. A
contiguous block holdout would be the other option and is a different question:
it asks the model about a part of the room nobody walked, rather than asking it
to place surfaces it saw from further away.

### Merging trains worse, and the reason is not the merge

The pre-registered comparison ran (`gs3d/MERGE_PREREG.md`), on the guard-banded
datasets so that merging could not win by donating a near-duplicate of a
held-out view, and with all four budget cells so that data and optimisation were
not confounded the way the first attempt confounded them:

```
  guard_single   7 000 iters ( 66 epochs)   21.99 dB
  guard_single  18 028 iters (170 epochs)   22.64
  guard_merged   7 000 iters ( 26 epochs)   19.86
  guard_merged  18 028 iters ( 66 epochs)   20.39
```

**Merging loses at every budget** — 1.60 dB behind at matched epochs, 2.25 dB at
matched iterations. Single was still improving at 7 000 (up 0.65 dB by 18 028),
so by the pre-registered rule the matched-epoch cell is the one that decides,
and merging loses there. The negative control was not needed: it exists to
attack a positive result.

This is not a contradiction of the coverage result above. Merging really does
multiply angular coverage 3.8–13.4x on the shared surface. Coverage counts
whether a voxel was seen from a direction and tolerates centimetres of
misplacement; a renderer does not.

#### How tight the registration actually is, and what sets the limit

A first attempt at this number was wrong, and the way it was wrong is the useful
part. `NearestVoxel` sorted one key per point and took `searchsorted`'s first
slot, so it examined **a single arbitrary point per cell** — correct only when
the cloud is fused at exactly the query voxel, which its docstring assumed and
nothing enforced. Run the instrument on a cloud against itself and a point's
distance to *itself* came back as 4.3 cm. Fixed, and pinned by a test that puts
20 points in a cell rather than the 0.06 the old test used.

With a correct lookup, and scored against a 5 mm reference cloud:

```
                                   p50    p75    p90   (cm)
  floor: two disjoint halves of
  one session, zero misalignment    0.77   1.11   1.58
  cross-session, 5 cm clouds        0.88   1.59   3.19
  cross-session, 1 cm clouds        0.80   1.26   2.44
```

The floor is the interesting row. Two clouds built from **alternate frames of
the same session** have exactly zero alignment error by construction, and they
still disagree by 0.77 cm — and refining the voxel from 2 cm to 5 mm barely
moves it (1.03 → 0.77). **The floor is the depth sensor's own repeatability,
not the grid.** The same wall observed twice from the same pose lands 8 mm
apart. So the cross-session median of 0.80 cm is *at* the floor: no rigid
solver can do better on this data.

#### The geometry does not constrain the direction that matters

The 1 cm and 5 cm alignments score the same residual while their transforms
differ by 2.55 cm, which is only possible if the difference lies in a direction
the residual cannot see. Displacing the best transform along each axis says
which:

```
  displacement from the optimum      -3 cm   -1 cm   +1 cm   +3 cm   (p50, cm)
  X, horizontal                       0.85    0.81    0.80    0.84
  Z, horizontal                       0.87    0.82    0.79    0.80
  Y, vertical — normal to the floor   1.40    0.92    0.84    1.38
```

**Height is pinned to about a centimetre; horizontal position is free to three
and beyond.** The floor supplies most of the points and constrains only its own
normal, and the walls that would constrain the horizontal are few and largely
parallel. Point-to-plane ICP is doing exactly what it can and nothing more.

That was measured on one pair, and one pair is an anecdote. Refining all four
photometrically — with a bracket wide enough to converge rather than clip, which
the first attempt was not — says the same thing four times:

```
  pair                score before   after    horizontal   vertical   ratio
  5bd1ed <- cb4586       0.0435      0.0224       1.7 cm     0.5 cm    3.6x
  1868dd <- f0d073       0.1234      0.0821      14.2        1.9       7.4x
  2994fa <- 7d3d52       0.1485      0.0230      18.7        3.5       5.3x
  2be6a9 <- 02a524       1.3574      0.7786      28.5        3.6       7.8x
```

**The correction the photographs demand is 3.6 to 7.8 times larger horizontally
than vertically, in every pair.** The anisotropy is a property of the geometry
these rooms present, not of the one pair it was first seen in — and the
horizontal errors are far larger than that pair suggested: 14 to 29 cm on three
of the four, against its 1.7 cm.

The first refinement run reported 8-9 cm for those three, and those numbers were
wrong in a specific way worth recording: coordinate descent started at a 4 cm
bracket and halved it each pass, so a pair whose optimum lay 19 cm away spent
its whole budget travelling and reported where it stopped. `--refine-span` now
sets the first bracket and the run says so when a result is a bound rather than
a minimum.

`2be6a9 <- 02a524` is the other thing that falls out. After a 29 cm correction it
still scores 0.78, against 0.082 for the worst genuine pair — an order of
magnitude apart. No alignment makes those two sets of photographs agree, which
is what a false pair looks like from the photometric side, and the geometric gate
admitted it at fitness 0.32.

Three horizontal centimetres at 2 m through f = 653 px is **ten pixels** of
texture displacement — and unlike the vertical component, no amount of depth
quality removes it, because the information is not in the depth. That is the
1.6 dB.

The conclusion for this track is therefore not "tune the ICP". It is that
**geometric registration is finished at roughly its limit, and a photometric
refinement stage is required rather than optional** — which is what the
reloc-then-refine pipelines do, and what this repo does not yet have.

### Dropping the blurry frames makes it worse, and the reason is coverage

`--sharp-ratio` defaults to 0.0, so the exporter's blur filter has never run:
every export on this page carries `"blurry": 0`. Measured against each session's
own median Laplacian variance, 10-30 % of the frames that survive the tracking
gate are below half of it, and on `guard_single_5bd1ed` that is 24 of 106
training images and **4 of 23 held-out ones** — one of those at 0.06x, a smear
no reconstruction can reproduce.

Turning the filter on looked obviously right. It is not:

```
  arm                              imgs  iters  epochs   psnr
  guard_single (all frames)         106   7000    66.0   21.99
  guard_sharp  (24 blurry dropped)   82   5400    65.9   20.81   -1.19 dB
  guard_sharp  (24 blurry dropped)   82   7000    85.4   21.00   -1.00 dB
```

Both cells lose, and the matched-*compute* cell loses while giving each surviving
image 85 passes against 66 — so this is not the smaller set being undertrained.

The arm was built by filtering `train_filenames` in place rather than
re-exporting, because the exporter applies its blur test *before* choosing the
holdout: re-exporting would have renumbered the frames and scored a different
set of photographs. The held-out 23 are identical by construction, and the
initialisation cloud was left alone, so training data is the only thing that
moved.

**What the dropped frames were carrying is viewpoints.** Of the 24, sixteen have
no other training frame within 30 cm and 20°, and they fall in six runs along
the walk, the longest thirteen frames long — a whole stretch of the path leaves
with them. Angular coverage is this capture's binding constraint, with a median
surface voxel seen from one or two directions, and against that a soft
photograph of an otherwise unseen viewpoint is worth more than no photograph.

So the filter stays off, and the reason is now recorded rather than accidental.
A sharpness rule that could pay for itself would have to be **coverage aware** —
drop a soft frame only where another frame already covers its viewpoint — and
that is a different rule from the one that exists.

### The metric rewards a blurred target, and what to do about it

Per-image scoring of the 23 held-out photographs, rendered from the trained
model, reproduces nerfstudio's aggregate exactly — and then shows something the
aggregate hides:

```
  image     sharpness   psnr        image    sharpness   psnr
  000103       0.06     25.75       000000      2.86     16.34
  000121       0.33     18.38       000032      3.00     19.51
```

**The blurriest held-out photograph scores near the top and the sharpest scores
last**, at a correlation of -0.44 between an image's own sharpness and the score
it gives. Dropping the single blurriest image *lowers* the mean, 21.99 -> 21.82.
A blurred target has no high frequencies to miss, so a render that resolves
nothing still scores well, while a sharp photograph charges the model for every
detail it failed to reproduce.

Changing metric does not fix it: SSIM is **more** blur-biased, not less, at
-0.63 against the same sharpness. LPIPS was not separable here either.

What does fix it is what the comparisons already do — score every arm on the
identical 23 images — plus reporting **paired per-image counts** rather than two
means, since the pairing removes the target's difficulty entirely:

```
  arm                            psnr    ssim   psnr wins   ssim wins
  guard_single  7 000           21.99   0.773       -           -
  guard_single 18 028           22.63   0.778     20/23       16/23
  guard_merged 18 028           20.39   0.734      4/23        1/23
  guard_photo  18 028           20.68   0.745      7/23        6/23
  guard_sharp   5 400           20.81   0.750      5/23        5/23
```

Every ranking on this page survives, and the counts are far stronger evidence
than the means were: the merged arm beats the single one on **one photograph in
twenty-three** by SSIM. The absolute numbers, though, are inflated by the four
soft images in the holdout, and should be read as a ranking rather than a
quality.

### Not done

- **A tree, not a pose graph.** There is no loop closure, so error accumulates
  along a chain. The composed-versus-direct columns are there to make that
  visible, not to fix it.
- **Drift inside a session is not modelled.** The transform is rigid, so a
  session that drifted is fitted with one compromise. The night pair's 2.59° may
  be partly that rather than gravity.
- **Appearance is not touched, and it now has a name.** Two sessions have two
  exposures, and the ARKit recording path has no lock. Merged geometry does not
  make merged photometry, and a splat handed inconsistent brightness will absorb
  some of it into the geometry.

  The published remedy is a per-image appearance embedding — the NeRF-W idea,
  carried into this ecosystem by **Splatfacto-W**
  ([arXiv:2407.12306](https://arxiv.org/abs/2407.12306)), which lives in
  Nerfstudio alongside DN-Splatter. A coarser variant gives one embedding per
  *sequence* rather than per image, which is the shape this problem actually
  has: two walks, two exposures, not 351 independent ones.

  **The hazard to check before adopting it.** An embedding is fitted per
  training image, so a held-out image has none. Whatever is done about that —
  optimise one on part of the held-out frame, or fall back to a default — is a
  choice that can quietly invalidate every held-out number on this page. Find
  out which it does before reading any result it produces.

  The other half of the fix is upstream and cheaper: the multi-camera path
  reports `exposure lock true` on every lens, so a scan-mode capture could
  simply not vary in the first place.

## The plan, staged

Each stage has a criterion that can fail, written before it runs. The lesson
being applied is `HANDOVER.md` §6: an acceptance test that cannot fail has not
measured anything.

```
  1  export                        done, 3.3x and 7.7x against control
  2  one splat, any quality        done, 24.28 dB against a 13.05 dB floor
  2b merging sessions              done as geometry: 4x coverage where walks overlap
  3  does depth supervision pay    running
  3b does a second walk pay        running, four arms
  4  does the walk pattern pay     needs one new recording
  5  quality knobs                 not started
```

Stage 2b was not in the plan. It arrived because the 35-second thermal ceiling
forces it, and it turned out to be the strongest coverage lever measured — which
also makes stage 3b, the training half of the same question, more interesting
than stage 4 was expected to be.

**Stage 1 — export.** Done. Criterion was the reprojection-versus-control ratio
above 1.5x; measured 3.3x and 7.7x.

**Stage 2 — one splat, any quality. Done, and it cleared.** DN-Splatter, 7000
iterations on cb4586, 146 training views and 21 held out, 12 minutes at about
100 ms an iteration and 2.9 GB of the card. Scored two independent ways:

```
                            PSNR    SSIM   depth abs-rel   depth RMSE   delta1
  mean-image floor         13.07      -          -             -          -
  raw LiDAR cloud          17.16      -       0.0151           -          -
  the splat, ns-eval       24.28   0.837     0.0107        0.036 m      0.9987
  the splat, repo harness  25.84   0.881       -             -           -
```

The two harnesses disagree by 1.6 dB because one scores at full resolution and
the other at a quarter; the ranking is what matters and both put the splat about
11 dB over the floor. **Criterion was 13.05 dB and the result is 24.28.**

Two cautions on the depth column, since it is the one stage 3 turns on. It is
scored against the held-out frames' *own* LiDAR, which the model never trained
on — so it is a real held-out measurement — but the model was trained to agree
with LiDAR in general, so this says the depth supervision generalises, not that
the geometry is right in absolute terms. And δ1 of 0.9987 is above the 0.90–0.95
that the literature calls a defensible bar for consumer LiDAR, which is a reason
to be suspicious of it rather than pleased with it.

The renders look like the coverage measurement predicts: flat walls, floor and
large objects come out; chair legs and hanging cloth smear. Those are exactly
the surfaces a single viewing direction cannot place.

The original plan for this stage follows, for the reasoning behind the choice of
session. Train DN-Splatter on **cb4586**:
167 frames of a static apartment interior, and the best angular coverage in the
set on all surfaces (33 %) with 21 % on walls, second only to b36df8's 24 % —
which has half as many frames. It is not the sharpest session — 532cea is —
but 532cea is a station concourse where pedestrians walk through most frames,
and starting on moving subjects would confuse a pipeline bug with a data
problem. Criterion, already measured and fixed: **held-out PSNR above 13.05 dB**,
the mean training image. That is a low bar on purpose; the question at this
stage is whether the pipeline is connected, and a bar that trivial still fails
loudly if a convention is wrong.

**Stage 3 — is depth supervision earning its place.** Train with and without the
depth loss, same seed, same iterations, same session. One thing has to be fixed
before it can run: `ns-render dataset --rendered-output-names depth` writes a
*colourised picture* of depth, three-channel 8-bit, not metres. Read as
millimetres it produced 1.332 m against a true 0.036 m — a plausible-looking
measurement of nothing. `tools/eval_views.py` now refuses that input and says
why rather than scoring it, so stage 3 needs either raw depth out of the
renderer or DN's own internal depth metrics, which is what the stage-2 numbers
above use. Criterion, stated now:
**rendered-depth versus LiDAR-depth median error on held-out frames**, not PSNR.
DN-Splatter's own numbers are the reason — sensor depth supervision moved depth
error about fourfold and PSNR by 0.3 dB, so a comparison read off PSNR would
have concluded it does nothing. The reference to beat is the 0.038 m the raw
cloud already achieves. If depth supervision does not improve held-out geometric
error, it does not go in.

**Stage 4 — does the walk pattern matter as much as the measurement says.**
Record two sessions of the same room back to back: one as currently walked, one
walked closer and at two heights — waist and overhead — per §"What follows for
capture" items 1 and 3. Same room, same light, same phone, minutes apart. Two
criteria, in this order:

1. **Does coverage move at all?** `tools/survey_coverage.py --vertical-only` on
   both, and `tools/plot_coverage.py` for the pair of plan views. This costs one
   walk and no GPU, and it is the only place the item-3 prediction gets tested.
2. **Does the splat follow?** Held-out geometric error on both, same trainer,
   same settings.

The first can fail without the second being run, and that is the point of
ordering them. This is the experiment that says whether the next effort belongs
in the capture app or the trainer.

**Stage 5 — only then, quality.** Format 30 fps versus 60, still rate, image
resolution, densification strategy. All of these are second-order until stage 4
answers.

## Not settled

- **60 fps versus 30 fps is an untaken trade, not a bug.** 30 fps halves the
  noise and doubles the motion blur at a fixed walking speed. Which one 3DGS
  prefers on *this* content is not something to reason out; it is a two-arm
  measurement on one recorded pair, and it needs an app change to run.
- **Moving people.** 532cea is a station concourse and pedestrians walk through
  half the frames. Every one becomes a smear of Gaussians. Nothing in the
  current path masks them, and it is unclear whether that matters for the
  intended use or is the main visible artefact.
- **Rolling shutter is unrecorded.** No per-row readout time is stored, so a
  rolling-shutter-aware solver has nothing to work with. At 0.6 m/s it is
  probably below the noise; at a normal walking pace it may not be.
- **The estimator's world versus ARKit's.** The export uses ARKit poses. The
  ICP trajectories in `traj/` are a second, independent pose source that
  disagrees with ARKit by a median of 20–30 cm over a session. Which one makes a
  better splat is an open and directly testable question, and `eval/fuse_rate.py`
  exists to produce a third candidate.
- **The ultra-wide raises the ceiling by about a third, on paper.** From the
  geometry above, the cap on angular coverage per surface *is* the field of
  view: 70.6° is 4.7 bins of 15°, 96.3° rectified is 6.4, and the raw 106° lens
  is 7.1. It is not a trade in the way 30-versus-60 fps is — per metre walked
  the angular rate is unchanged, so a wider lens is weakly better and never
  worse for coverage; it simply stops clamping sooner. It does cost more walking
  to cash in (3.35 m instead of 2.12 m past a wall at 1.5 m), and it costs
  ARKit, which is the part `docs/POSE.md` prices.

  Nothing here is measured. The number to be sceptical of is the ceiling: the
  measured medians are 1–2 against a ceiling of 4.7, so this capture is nowhere
  near clamping, and raising a ceiling nobody is touching buys nothing. **Get
  the walk to the current ceiling first, then the ultra-wide is worth costing.**

- **An attempt to simulate the wider lens failed, and the failure is the useful
  part.** The attempt: re-render an exported cloud from the recorded camera path
  at hypothetical fields of view, with a dilated z-buffer for occlusion, and
  count directions as the survey does. It reported that 70.6° → 96.3° raises
  three-view coverage 1.14x. **Do not use that number.**

  The instrument fails to reproduce the known answer: on cb4586 it puts
  three-view coverage at 70 % where the measurement says 33 %. That gap is not
  an occlusion bug — dilating the buffer moved it from 81 % to 70 % and no
  further. It is that `survey_coverage.py` counts a direction only where *that
  camera's own depth map returned a confident value*, while this counts anything
  geometrically in frustum. Real LiDAR does not return at grazing incidence,
  past about 5 m, on glass, or in the dark — precisely the peripheral, distant,
  oblique surfaces a wider lens would add. **The bias runs in favour of the
  answer the experiment was looking for.**

  The entry above says what to do instead: the ceiling follows from the field of
  view analytically, and no simulation of the sensor is needed to get it. That
  is the lesson — the question had a closed form, and building an instrument for
  it produced a number that was both wrong and flattering.

- **Global map from `LocalMap`.** `HANDOVER.md` §5 proposes dumping
  `depth_odometry.py`'s fused voxel map as the init cloud. The exporter
  back-projects raw depth instead, which is simpler and lands in ARKit's frame
  where the poses already are. The fused map would be less noisy and carries
  proper normals — worth comparing at stage 5, not before.
