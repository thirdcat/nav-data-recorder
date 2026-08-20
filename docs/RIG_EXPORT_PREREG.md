# Pre-registration: one training set from both lenses of a multi-cam session

Written before the code existed and before any gate was run. What follows is
what the multi-camera export path has to prove, how each claim is measured, and
what number counts as a failure. Nothing below is adjusted after the fact; a
gate that fails is reported as a failure.

## The claim being tested

`MultiCamRecorder` records two lenses on one walk. Posing them independently
does not put them in one coordinate frame: `eval/pi3_poseless.py` run on each
arm gives trajectories whose best-fit-aligned residuals are 2.4 cm (57f29e),
9.1 (d0f44f), 4.6 (15fbb3) and 33.8 (d67f0a) at the median, against a rig
baseline of 1.93 cm, with scale ratios 0.946-1.058. **Those two trajectories
cannot be glued.**

The rig, however, is calibrated. `calib/iphone17-1_ultrawide.json` gives the
ultra-wide's pose relative to the wide camera as `R x + t`, 19.272 mm and
0.462 deg, direction verified against a stereo pair (`calib/README.md`). So the
claim is: **pose one arm, and the other arm's poses follow from a fixed rigid
transform** — no relocalisation, no second solve, one coordinate frame.

## What the data says about the pairing, measured first

Two facts were measured before the design was fixed, because they change what
the transform has to do.

**The two lenses do not shoot at the same instant.** Nearest-in-time gap from
each ultra-wide frame to a wide frame, five sessions:

```
  session   median   p90    max     (ms)
  57f29e     62.5    99.4   100.6
  d0f44f     56.0    94.2   102.8
  15fbb3     58.4    92.0   108.0
  d67f0a     65.2    94.2   101.4
  0d6306     60.7    93.3   110.4
```

**Over that gap the camera moves further than the rig baseline.** From the wide
trajectories, at the measured walking speeds:

```
  session   speed     motion over 60 ms      rotation over 60 ms
  57f29e   0.37 m/s   22.2 mm (p90 36.6)     1.96 deg
  d0f44f   0.45 m/s   27.1 mm (p90 40.3)     1.57 deg
  15fbb3   0.21 m/s   12.8 mm (p90 21.4)     1.81 deg
```

Against a rig baseline of **19.272 mm** and a rig rotation of **0.462 deg**.
The timing error is the same size as the translation being modelled and four
times the rotation. So "give the ultra-wide frame the pose of the nearest wide
frame, moved by the extrinsic" would ship an error larger than the transform it
applies, and it would look correct.

The export therefore **interpolates the wide trajectory to the ultra-wide
frame's own timestamp** (SLERP on rotation, linear on translation, between the
two bracketing wide poses) and applies the extrinsic there. `--rig-pose-time
nearest` keeps the naive construction so that the two can be compared rather
than argued about.

## Criteria

Each is a number with a threshold, fixed here.

### 1. Regression — the single-lens path is untouched

`~/nav_data/20260809-074458-cb4586` exported the existing way must still give
**exactly 167 images and 21 holdout**, and `sparse/0/cameras.txt`,
`sparse/0/images.txt`, `sparse/0/points3D.txt`, `sparse_pc.ply` and
`transforms.json` must be **byte-identical** to a baseline taken before any
edit. `python3 tools/test_export_3dgs.py` must pass every test.

*Fails if:* any count moves, any checksum differs, or any test fails.

### 2. Identity gate — the derivation path is a no-op when the transform is

The same shape of gate as `substitute_poses`, where `--pose-key reference`
reproduces the default export at a maximum pose difference of 0.000e+00.

Route **the wide lens** through the new derivation path with an **identity**
extrinsic — same stream, same timestamps, so the interpolation is evaluated at
the sample points it was built from — and compare against the plain wide export,
whose poses come straight from the trajectory.

*Passes if:* the same frame set, and `sparse/0/images.txt` and
`sparse/0/cameras.txt` identical byte for byte.
*Fails if:* one line differs. There is no tolerance here; both sides compute the
same thing and any difference is a bug in the composition or the interpolation.

### 3. Geometry gate — the baseline comes out at 19.272 mm

In the combined export, the derived ultra-wide camera centre must sit
**19.272 mm ± 0.1 mm** from the wide camera centre it was derived from, at every
paired step. This catches a millimetre/metre unit error, a dropped rotation on
the translation, and a composition applied on the wrong side.

**This criterion is reported twice, because the two constructions read it
differently and only one of them is defensible.**

- *against the interpolated wide pose the derived pose was built from* — the
  construction's own reference. Both `--rig-pose-time` settings must pass here.
- *against the nearest wide frame's own camera centre*, which is the literal
  wording of the criterion as pre-registered. Under `--rig-pose-time nearest`
  the two are the same thing and it must pass. Under `interpolate` it **will
  fail by construction**, by the 12-37 mm the camera actually travels between
  the two lenses' shutters, measured above. That failure is the measurement of
  the timing offset, not of the transform, and it is reported as a failure of
  the criterion as written rather than redefined into a pass.

### 4. Consistency gate — the one that tests the rig

Reading **only the exported files**: back-project a wide frame's depth through
its exported pose and intrinsics, project the points into the paired ultra-wide
frame through *its* exported pose and intrinsics, and compare the colour that
lands there with the ultra-wide image's own colour at that pixel. Mean absolute
difference over the landing pixels, 0-255.

Against a **control**: the same points projected into an unrelated ultra-wide
frame from the same session, chosen far enough away in the walk to be a
different view. Without the control the first column means nothing — a dim or
flat scene scores well against anything.

`docs/3DGS.md` "How it was checked" records the existing bar for this style of
test on the single-lens export: **3.3x and 7.7x** reprojection-versus-control.

*Passes if:* ratio **>= 1.5x**, the same bar stage 1 of the plan was accepted
on.
*Fails if:* below it. **If the ratio is near 1.0 the rig transform is not
working**, and that is what gets reported — the export does not ship on a ratio
that cannot tell the paired frame from an unrelated one.

Secondary, and the reason `--rig-pose-time` exists: the same measurement is run
with `nearest` instead of `interpolate`. The prediction from the timing
measurement above is that `interpolate` scores better. That prediction is
recorded now so that the result can contradict it.

### 5. The trajectory must belong to this session

Four of these sessions were recorded within minutes of each other and have the
same shape, the same frame numbering and the same lens configuration, so handing
the exporter the wrong `pi3traj/*.npz` is a live mistake — and so is handing it
the *ultra-wide* arm's trajectory to pose the *wide* arm, since the two share
frame numbers 0..N.

These files carry no `reference`, so `substitute_poses`'s existing check cannot
run. They do carry `t` per frame. The export must therefore compare the
trajectory's timestamps against the session's own for the same frame numbers and
**refuse anything above 1 ms**.

*Passes if:* the right file is accepted and both wrong files are refused — the
other session's trajectory, and the same session's other lens. Run on a
known-wrong input, not only on the right one.

### 6. A trajectory that flagged itself broken is refused

`eval/pi3_poseless.py` writes `broken_from` when a window join disagrees with
the chain. `d67f0a` carries it on both arms (wide from frame 48, ultra-wide from
96). Exporting past that point silently inherits the error.

*Passes if:* d67f0a is refused by default and the message names the frame, and
`--allow-broken-trajectory` is required to override.

## What this does not test

- **Nothing is trained.** No GPU here. Every number below is a property of the
  exported files, not of a reconstruction.
- **The magnitude of the 19.272 mm baseline is still unverified against a real
  capture.** `calib/README.md` records that its direction is settled by a stereo
  pair and that a parallax fit across eight pairs gave slopes scattered from
  0.10x to 1.49x of the prediction. Criterion 3 checks that the export *applies*
  19.272 mm, not that 19.272 mm is right. Criterion 4 is the only thing here
  that can notice if it is not, and it is not a sensitive test of a 19 mm term.
- **The ultra-wide is a raw, uncorrected lens.** `docs/3DGS.md` records that its
  radial table reaches -0.0516 against the wide's +0.0390, and the export
  declares `PINHOLE` with `k1..p2` zero for both. A trainer that does not model
  distortion will absorb it, and criterion 4's residual carries it too — which
  makes criterion 4 conservative, not flattering.
- **0d6306 has no trajectory in `pi3traj/`**, so it cannot be exported without a
  GPU run. It is listed as a dual-lens session and is not covered here.
