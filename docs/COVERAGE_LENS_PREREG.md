# Pre-registration — can `tools/survey_coverage.py` measure the ultra-wide's coverage gain?

Written 2026-08-19, before running any command that produces a number.
Nothing below may be edited after the first result is read; corrections are
appended with a timestamp.

## The question

`docs/3DGS.md` derives that the field of view is a hard ceiling on angular
coverage per surface (70.6° = 4.7 bins of 15°, raw ultra-wide 106° = 7.1). One
attempt to measure the gain by simulation failed and is documented as failed.
Dual-lens sessions now record both lenses on one walk with per-arm trajectories
in `pi3traj/<id>_{wide,ultrawide}_local.npz`. **Can `survey_coverage.py` measure
the gain directly?**

The deliverable is a verdict on answerability first, a number only second.

## S — the structural criterion, and it decides everything else

`survey_coverage.py` counts a viewing direction for a voxel only where **that
camera's own depth map returned a value**. Therefore:

> **S.** An instrument that can answer this question must produce a *different*
> coverage number when the lens's field of view is the only thing changed, with
> the walk, the depth and the trajectory held fixed. If the coverage number is
> not a function of the lens, the instrument is field-of-view blind and the
> question is **NOT ANSWERABLE** with it.

Test of S, decided before running: hold the trajectory and the depth stream
fixed and change only the declared camera. If the survey returns bit-identical
output, S fails.

S fails ⇒ report "not answerable", report *why*, and **do not publish a
wide-vs-ultrawide coverage comparison as an answer**, however tempting the
numbers look. A second flattering instrument on that page is worse than nothing.

Two further preconditions, either of which alone also fails S:

- **S1 — is there one depth stream or two?** If the session carries a single
  depth stream registered to one of the two lenses, both arms back-project the
  *same* points and the lens cannot enter the measurement. Check the manifest
  and the on-disk streams.
- **S2 — is the depth's field of view the lens's?** `export_3dgs.depth_points`
  scales the pose intrinsics by `depth_width / image_width`, i.e. it assumes the
  depth grid spans the image's field. If depth and image are different cameras,
  feeding the wider lens's intrinsics inflates every back-projected ray by
  `f_declared / f_depth` and manufactures scene. Quantify that factor.

## Gate 1 — reproduce the published row before reading a new one

```
python3 tools/survey_coverage.py --vertical-only ~/nav_data/*cb4586
```

must return **167 frames, 45 % at ≥2 views, 21 % at ≥3, 5 % at ≥5, range
1.45 m** (docs/3DGS.md, "the same nine sessions, scored that way").

- PASS: frames exactly 167; each percentage within ±1 pp of the published
  figure after the same rounding; range within ±0.01 m.
- FAIL: stop. Everything downstream is unreadable and the report says so.

## Gate 2 — the npz pose path, and how it is allowed to be believed

`survey_coverage.py` reads poses from `pose.jsonl`. Add `--poses <npz>` taking
`estimate` / `frame`, 4×4 world_from_camera in the depth convention (+Z forward,
+Y down) — the shape of `traj/<id>.npz` and `pi3traj/<id>_*_local.npz`. Reuse
`export_3dgs.substitute_poses`; a second copy of the basis change is a second
place for it to be wrong.

Self-tests in `tools/test_survey_coverage.py`, all pre-registered:

1. **Round trip.** An npz holding the session's *own* poses must reproduce the
   ARKit survey exactly — every fraction, every voxel count. Agreement that is
   not exact means the conversion is not the identity it claims to be.
2. **Rigid invariance.** Rotating and translating the whole trajectory must
   leave every coverage fraction unchanged and move the camera centres by the
   known amount. Coverage is a property of the geometry, not of the frame.
3. **The lever test (S, in miniature).** Declaring a different lens — the same
   depth grid, the same trajectory, an image width and focal length scaled
   together — must leave the survey **bit-identical**. This test is written to
   *pass*, and its passing is what proves the instrument is field-of-view blind.
4. **Scale is not free.** A trajectory scaled by k must move the numbers, so
   test 2's invariance is not the trivial one.

## Gate 3 — the confound must be reported with every number

Whatever is reported for a dual-lens session must carry, alongside it:

- the fraction of that arm's frame that carried a depth return, measured on the
  session, not quoted;
- the fraction of that arm's *solid angle* the depth camera covers at all.

A coverage number for the ultra-wide arm printed without those two is a number
that looks like a measurement and is not.

## Gate 4 — attribution rule for any difference between arms

If S somehow passes and both arms are scored, a difference between them counts
as a lens effect only if it survives every one of these, each of which can move
the number on its own:

- the two trajectories are separate Pi3 solutions with separate gauge and scale
  (`window_scale` differs between arms), and a k % scale error is a k % voxel
  size change, which moves coverage directly;
- the two arms have different frame counts and different frame rates;
- `conf_min` is inoperative on these sessions (no `confidence.bin`), so their
  numbers are not on the same gate as cb4586's `conf ≥ 2`;
- the wide arm's active-format focal length is not logged on these three
  sessions and the factory fallback is measured wrong by ~3° (`calib/README.md`).

Any difference not surviving all four is reported as unattributable.

## What is NOT permitted

- Reporting a wide-vs-ultrawide ratio if S fails.
- Counting a direction where the depth returned nothing (that is the failed
  simulation's error, and its bias ran towards the wanted answer).
- Using the ultra-wide's intrinsics to back-project the wide-registered depth.
- Training anything, editing `tools/export_3dgs.py` or `eval/fuse_rate.py`,
  or committing.

## Prior belief, stated so it can be contradicted

`docs/POSE.md` already records that the LiDAR depth camera is a virtual device
over the wide camera, so pairing it with the ultra-wide "leaves the depth
frustum at 74.6°". If that holds on disk, S1 fails and the expected verdict is
NOT ANSWERABLE. Writing the expectation down first is what stops the run from
being an argument for it.

---

# Results, appended after the fact

## Gate 1 — PASSED, exactly

```
     session frames  voxels median    >=2    >=3    >=5   range   spread   lin
74458-cb4586    167   24033      1    45%    21%     5%   1.45m    6.21m   2.0
```

167 frames, 45 / 21 / 5, 1.45 m — the published row to the digit, before and
after the changes below.

## S1 — FAILED. There is one depth stream and it belongs to the wide camera

Measured on disk, not quoted. Every wide frame has a depth frame at its **exact**
timestamp; not one ultra-wide frame does:

```
  session  depth  ultra-wide  wide   exact-t match          nearest depth to
                                     uw      wide           an ultra-wide frame
  57f29e     420      90       86    0/90    86/86          8.8 ms median, 82.7 max
  d0f44f     538     114      109    0/114  106/109        10.7 ms median, 27.7 max
  15fbb3     458      98       93    0/98    90/93           8.6 ms median, 108 max
```

One `depth.bin`, one 320x240 grid, locked to the wide camera's clock — which is
what a virtual depth device over the wide camera looks like from the file
system. The manifest says the same in words: *"the depth is already in its
frame"*. **Both arms would back-project the same points.** The lens cannot enter
the measurement.

## S2 — the frustum, measured, and the inflation if it is papered over

Active formats from `calib/README.md`: depth 70.38° x 55.74°, wide video
69.53° x 55.00°, ultra-wide 106.20° x 73.68°.

```
  share of the frame inside the depth frustum at all
    wide         100.0 % of width x 100.0 % of height = 100.0 %
    ultra-wide    52.9 %          x  70.6 %           =  37.4 %

  measured on disk, median over 40 sampled depth frames per session
    session   valid depth px   of the wide frame   of the ultra-wide frame
    57f29e        91.6 %            94.8 %                35.5 %
    d0f44f        86.6 %            89.3 %                33.3 %
    15fbb3        91.7 %            94.1 %                35.4 %
```

The 19.272 mm rig baseline moves the ultra-wide figure by 0.1-0.3 pp, so the
number is geometry, not parallax. **62.6 % of the ultra-wide frame — the entire
extra field of view that lens is bought for — is outside the depth frustum
before a single pixel is examined.**

If the mismatch is papered over by handing the depth grid the ultra-wide's
intrinsics, `depth_points` spreads every ray by 226.91 / 120.13 = **1.89x**.
That does not add coverage; it draws the same room bigger. The self-test
`test_the_survey_is_blind_to_which_lens_took_the_picture` asserts both halves.

## VERDICT: NOT ANSWERABLE with this instrument

S fails, so by the rule written above **no wide-vs-ultra-wide coverage
comparison is reported**, and the arm numbers criterion 3 asked for are
deliberately not produced. Producing them would have required inventing an
ultra-wide depth stream that was never recorded, and the resulting table would
have been quoted.

The closed form stands and needs no instrument: the field of view caps swept
angle at 4.7 bins of 15° for the wide and 7.1 for the raw ultra-wide, and the
measured medians are 1-2. Get the walk to the current ceiling first.

## Correction 1, 2026-08-19 — rigid invariance is not exact, and why

Pre-registered self-test 2 demanded that a rotated, translated trajectory score
*identically*. It does not, and the cause is not the substitution. `fold_votes`
identifies a voxel by an XOR of three linear multiples with no mixing step, so
distinct voxels can share a key and have their direction sets unioned — which
can only push coverage up. Folding on the voxel index itself instead:

```
                   voxels (exact / hashed)      >=3 coverage (exact / hashed)
  fixture           189 898 / 166 212            0.57 % / 3.85 %   (at >=2: 8.1 / 19.0)
  cb4586 vertical    24 108 /  24 033           20.57 % / 20.76 %
  5bd1ed vertical    47 401 /  46 658           13.64 % / 14.55 %
  1868dd vertical    52 243 /  51 896            9.55 % /  9.84 %
```

On real sessions the bias is **0.2 to 0.9 pp, always upward**, and does not move
the published table at its printed precision — cb4586 reads 21 % either way.
On the synthetic fixture it is 2.3x, because a room of exactly planar surfaces
makes voxel indices regular and an unmixed XOR of linear multiples collide
systematically. The test now asserts 1 pp on the shipped fold and a thousandth
on the unhashed one, so the gap is measured rather than tolerated. **The fold
was not changed**: doing so would move published numbers by up to 0.9 pp in the
middle of a task whose reproduction gate pins them, which is not a change to
make silently.

## Correction 2, 2026-08-19 — a threshold that was wrong arithmetic

Pre-registered self-test 3's second half demanded that a 2x-wider declared lens
raise median range by more than 1.4x. It raises it by 1.35x, because a ray at
angle theta has range z/cos(theta) and rays near the optical axis barely move.
The claim was right and the number attached to it was not. The surviving
assertions are the ones the geometry forces: range grows, surface spreads over
1.55x the voxels, mean views per voxel falls 1.24 -> 1.06, and >=3 coverage
falls. Nothing about the direction of the result changed.

## What was built

`--poses <npz>` / `--poses-key` on `tools/survey_coverage.py`, reusing
`export_3dgs.substitute_poses` rather than repeating the basis change, plus a
refusal: `--vertical-only` against a trajectory with no `reference` raises,
because only the reference check establishes that the file's +Y is up and every
`pi3traj/<id>_*_local.npz` says in its own convention string that it has none.

Exercised on real data — the arm this repository actually has a second
trajectory for:

```
  cb4586 vertical-only, ARKit           167 frames  45 %  21 %  5 %  1.45 m
  cb4586 vertical-only, traj/cb4586.npz 167 frames  42 %  20 %  5 %  1.44 m
    (its own ARKit poses match the session to 0.00e+00 m)
```

The depth-ICP trajectory scores slightly *worse* coverage than ARKit on the same
depth, which is the same direction as the held-out photometric result already on
`docs/3DGS.md`.
