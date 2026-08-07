# Export format — VLN episode directories

[docs/DATA_FORMAT.md](DATA_FORMAT.md) describes what the app writes to the
phone. This describes what the VLN dataset consumes, and how one turns into the
other.

They are not the same shape and should not be. The session format is a capture
log — every stream the device produced, at its own rate, with the failure modes
of a recording that can die mid-write. The episode format is a training sample:
a fixed-stride image sequence and one pose per image, with everything else
thrown away.

The convention below was derived from and verified against a reference episode
set at

```
/data/vln_dataset/lge/realworld/humanvideo/iphone-16pro/20260518/exported/
```

which was produced by the offline reconstruction path (iPhone `.MOV` → metric
SLAM → export), not by this app. Matching it is the point: the labelling
pipeline downstream is already written against it.

## The episode directory

```
<episode_id>/
├── images/            000000.jpg, 000001.jpg, … 848×480
├── poses_tum.txt      one pose per image, TUM format
├── trajectory.png     optional; a plot of the track, for eyeballing
├── .vln_cache/        written by the pipeline, not by us
└── labeled_output_v2/ written by the pipeline, not by us
```

**Only the first two are ours.** `.vln_cache/` holds person-proximity arrays
the labeller computes and reuses; `labeled_output_v2/` holds the VLM output —
`segmentation_manifest.json`, plus `accepted/<episode_id>_seg0.json` carrying
the phase skeleton, the L1/L2/L3 instructions and the per-keyframe chains of
thought. An export is complete without them.

The episode ID is opaque to the consumer — the labeller only copies it into
`parent_episode_id`. The reference set's
`results_metric_IMG_1084_s0_eend_st2_samp0.02_20260601_074616` encodes the
reconstruction run's parameters; a session ID from this app
(`20260807-014530-a1b2c3`) serves the same purpose and is already sorted
chronologically.

Notably absent: **there is no intrinsics file.** The format carries no `fx fy
cx cy`, no distortion, no image dimensions outside the JPEGs themselves. The
labeller works from pixels and poses only. Anything geometric that a consumer
might want later — unprojection, depth fusion — has to go back to the session
directory, so keep it.

## images/

Six-digit zero-padded, contiguous from `000000.jpg`, one per pose row, in
capture order. **The filename index is the row index in `poses_tum.txt`.**
That is the whole of the association — no timestamps are matched, no join key
is stored, and a gap in the numbering would silently shift every pose after it.

The reference set is 848×480 at 5 Hz. The recorder already captures stills at
5 Hz, so only the frame size is an export concern.

**848×480 is not an arbitrary downscale — it is the deployment camera's own
frame.** The OAK-1-W on the Unitree G1 outputs exactly that, and the reference
episodes match it because that is what the policy will see at inference. The
resize is therefore a geometric decision, not a cosmetic one; see *Known gaps*.

## poses_tum.txt

```
# timestamp tx ty tz qx qy qz qw
0.000000 0.00000000 0.00000000 -0.00000000 0.04255279 0.20403262 0.00000000 0.97803883
0.200000 0.04172053 0.00848572 -0.00633830 0.04256243 0.20490098 0.00039984 0.97785652
…
```

One comment line, then one row per image, space-separated. Seconds and metres.
The quaternion is `world_from_camera`, scalar-last, unit norm — TUM's own
convention, and the same handedness ARKit uses.

### Coordinate frames

This is the part that is not written down anywhere upstream, and getting it
wrong produces a file that parses cleanly and means something else.

**World: Z-up.** Gravity-aligned, `+Z` up, `X`/`Y` spanning the floor plane.
Verified numerically: across all three reference episodes the Z column of the
translations has a standard deviation of 11–23 mm while X and Y span metres —
that is a person walking a flat floor, and only one axis can be the vertical
one.

**Camera: FLU** — `+X` forward along the optical axis, `+Y` left, `+Z` up.
Right-handed; `det(R) = 1` throughout. Verified by projecting each body axis
onto the direction of travel and onto world up:

| body axis | · velocity | · world up |
| --- | --- | --- |
| `+X` | **+0.90** | −0.42 (σ 0.02) |
| `+Y` | −0.06 | +0.05 (σ 0.05) |
| `+Z` | +0.41 | **+0.91** (σ 0.01) |

`+X` tracks where the walk is going and `+Z` holds world up to within a
degree, which fixes the frame. The off-axis components are one constant tilt,
not noise: the phone was held pitched about 25° down, so forward dips below the
horizon by exactly as much as up leans away from vertical. `arcsin(0.416) =
24.6°`, `arccos(0.906) = 25.0°`.

This is **not** the OpenCV camera frame (`X` right, `Y` down, `Z` forward) that
most SLAM front-ends emit, and it is not ARKit's. Both a gravity alignment and
a basis change have already been applied by the time a file looks like this.

### The origin

Position is rebased so the first frame sits at `(0, 0, 0)`, and time is rebased
so the first row is `t = 0`. Yaw is rebased too — world `+X` is the initial
camera heading projected onto the floor plane, which shows up as `qz ≈ 0` in
the first row (exactly `0.000000` in two of the three reference episodes; the
third is `0.023`, the residue of an extra roll correction).

Roll and pitch are *not* rebased, and must not be — they are measured against
gravity, and the first row's `q = (0.043, 0.204, 0, 0.978)` is a real 24°
downward tilt of a real phone in a real hand.

An episode frame is therefore session-local in exactly the way the session
format's is. Two episodes never share coordinates.

### Converting a recorded session

ARKit hands back a `world_from_camera` transform in a **Y-up** world with an
**`X` right, `Y` up, `Z` back** camera. Two basis changes get to the target:

```python
import math
import numpy as np

# ARKit camera (X right, Y up, Z back) -> FLU (X forward, Y left, Z up),
# as columns: forward = -Z_ar, left = -X_ar, up = +Y_ar.
M = np.array([[0.0, -1.0, 0.0],
              [0.0,  0.0, 1.0],
              [-1.0, 0.0, 0.0]])

R0 = quat_to_R(first.qx, first.qy, first.qz, first.qw)
p0 = np.array([first.tx, first.ty, first.tz])

# World: +Z is ARKit's up; +X is the initial heading flattened onto the floor.
fwd0 = -R0[:, 2]
x_t = np.array([fwd0[0], 0.0, fwd0[2]])
if np.linalg.norm(x_t) < 1e-3:      # started pointed at the floor or ceiling
    x_t = np.array([R0[0, 1], 0.0, R0[2, 1]])   # fall back to camera up
x_t /= np.linalg.norm(x_t)
z_t = np.array([0.0, 1.0, 0.0])
N = np.column_stack([x_t, np.cross(z_t, x_t), z_t])

# Per pose:
R_tum = N.T @ quat_to_R(p.qx, p.qy, p.qz, p.qw) @ M
t_tum = N.T @ (np.array([p.tx, p.ty, p.tz]) - p0)
```

`N.T` is an orthonormal change of basis and `M` is a signed permutation, so the
transform is rigid — distances and the metric scale pass through untouched.
There is no scale factor to estimate, which is the one thing this path gets for
free that the offline SLAM path does not.

The degenerate case in the fallback is real: if the session begins with the
camera pointed straight down (phone flat on a table, someone about to pick it
up), the heading has no horizontal projection and the yaw origin is undefined.
Camera up is the sane substitute, and it is worth logging when it fires.

The rows to export are `Session.posed_images()` — the join of `frames.jsonl`
and `pose.jsonl` on `frame`, which is exact rather than a nearest-timestamp
search.

`tools/export_episodes.py` implements all of this:

```bash
python3 tools/export_episodes.py ~/nav_data/20260807-131829-9e6858 -o ./episodes
python3 tools/export_episodes.py ~/nav_data/* -o ./episodes --fit whole
python3 tools/export_episodes.py ~/nav_data/* -o ./episodes --hfov 66.1
```

It drops frames whose ARKit tracking had not converged — those sit at the world
origin and would otherwise pin a cluster of identical positions to the start of
every trajectory — splits on `ar.interruptionEnded`, refuses a session that was
not shot landscape (a landscape crop of an unrotated portrait frame is silently
wrong), prints the field of view it produced, and writes an
`alignment_summary.txt` over everything it exported.

### Checking the result

`alignment_summary.txt` in the reference set reports three per-episode numbers,
and all three are computable from `poses_tum.txt` alone. Reproducing them is a
cheap end-to-end check that an exporter got the frames right — if the basis is
wrong these go wild, because they read the vertical axis and the roll about the
direction of travel.

```python
roll = np.degrees(np.arcsin(np.clip(R[:, 2, 1], -1, 1)))  # world-Z part of body +Y
roll.std(), abs(roll.mean()), xyz[:, 2].std()
```

| episode | N | `roll_std` ° | mean roll ° | vertical σ, m |
| --- | --- | --- | --- | --- |
| IMG_1082 | 52 | 1.560 | 3.911 | 0.0113 |
| IMG_1083 | 50 | 0.337 | 0.000 | 0.0234 |
| IMG_1084 | 92 | 2.912 | 2.715 | 0.0139 |

The exporter's own output reproduces the frame table above, which is the check
that matters — it is derived from the poses alone, so agreeing with it means the
basis came out right rather than merely self-consistent. Against a fixture
walking a circle pitched 25° down:

| body axis | · velocity | · world up | reference |
| --- | --- | --- | --- |
| `+X` | +0.906 | −0.423 | +0.90, −0.42 |
| `+Y` | +0.031 | −0.001 | −0.06, +0.05 |
| `+Z` | +0.422 | +0.906 | +0.41, +0.91 |

`arcsin(0.423) = 25.0°` and `arccos(0.906) = 25.1°` — the fixture's pitch,
recovered from the exported file.

The thresholds the reference pipeline flags on are `roll_std > 4°`,
`|mean_roll| > 4°` and vertical σ `> 0.1 m`. All three reference episodes pass.

Note that the summary calls the last column `y_spread_std`: it is named for the
pre-transform Y-up frame and computed on the post-transform Z. The numbers
above are the vertical axis under both names.

## Known gaps

- **Nothing the phone can produce fills an 848×480 frame correctly, and the
  export has to pick which way to be wrong.** The target frame carries the
  OAK-1-W's 96.3° × 64.6°; the phone's 4:3 stills carry 73.1° × 58.1°. It is
  narrower on *both* axes, so there is no crop that matches — this is the same
  23.2° horizontal gap *The rig this is collected for* in
  [DATA_FORMAT.md](DATA_FORMAT.md) already names, surfacing again at export
  time. Two ways to land it, neither free:

  - **Fit the aspect.** Crop the 4:3 frame to 16:9 and resize. Pixel aspect
    matches, but vertical falls to 45.3° against the target's 64.6°, throwing
    away the axis the 4:3 default exists to protect.
  - **Fit nothing and resize whole.** Keeps all 58.1° of vertical, but the
    degrees-per-pixel then differs from deployment on both axes, so a fixed
    image position no longer means a fixed bearing.

  Either way it is a deliberate, logged step, not a silent `resize()` — and it
  should record the FOV it produced, since that is what a consumer would
  otherwise have to guess. The clean fix is upstream of the export: the
  ultra-wide reproduces the OAK-1-W's geometry almost exactly, which is the
  argument DATA_FORMAT.md makes for the AVFoundation path.

  **Measured, the gap is smaller than that framing suggests — because the real
  reference episodes are not shot at the target FOV either.** The episode format
  stores no intrinsics, but they are recoverable: the poses fix the epipolar
  geometry between any two frames, and only the true focal lengths make the
  observed feature matches consistent with it. `tools/estimate_intrinsics.py`
  does that, and on three sets it reads:

  | set | fx | fy | fx/fy | H° | V° |
  | --- | --- | --- | --- | --- | --- |
  | simulation, `ep000_oak` | 379.9 | 379.9 | 1.000 | 96.3 | 64.6 |
  | real reference, `IMG_1083` | 652.0 | 649.6 | 1.004 | 66.1 | 40.6 |
  | this recorder, `--fit crop` | 595.4 | 595.4 | 1.000 | 70.9 | 43.9 |

  Three things follow. The simulation set recovers the OAK-1-W spec to the
  decimal, which is independent validation of the tool on real imagery rather
  than synthetic correspondences. `fx/fy ≈ 1` in both reference sets means
  **nothing was squeezed** — the sets were cropped or natively 16:9, so `--fit
  crop` is the mode that matches them and is the exporter's default. And the
  real reference is *narrower than the phone can see*: 66.1° against ARKit's
  73.1° is a 12% linear crop, about what iPhone video stabilisation takes and
  ARKit does not apply. So this recorder's 70.9° sits between the two reference
  sets, closer to the deployment camera than the existing real-world reference
  is. The pipeline already spans 66–96°; our value falls inside that spread
  rather than outside it.

  `--hfov DEGREES` exists for when matching one set exactly matters more than
  keeping the width — `--hfov 66.1` reproduces the real reference's geometry
  (measured back out at 66.1° × 40.4°). Unlike the fixed `--fit` crops, the
  `--hfov` crop is recomputed per frame from that frame's focal length, so
  autofocus breathing lands in the crop rectangle instead of in the output field
  of view; that is the one mode where a consumer assuming fixed intrinsics is
  actually right. Asking for more than the lens has is clamped and warned about,
  not padded.
- **Uniform timestamps are assumed but not guaranteed.** Every reference file
  steps exactly 0.2 s because it was resampled off a video at a fixed stride.
  ARKit stills land near 5 Hz, not on it, and a thermal pause or a dropped
  frame leaves a real gap. Exporting true rebased timestamps is the honest
  choice and matches the labeller's use of them as segment boundaries — but
  anything downstream that infers a frame rate by reading row 1 will be
  slightly wrong. Worth confirming with the pipeline owner before picking.
- **An `ar.interruptionEnded` event splits an episode.** ARKit resets the world
  origin when capture resumes, so poses either side are in different frames.
  An exporter must cut there and emit two episodes, not concatenate across it.
  The session format records the event; the episode format has nowhere to say
  it, which is precisely why the cut has to happen during export.
- **Poses are online VIO, not bundle-adjusted.** The reference set went through
  offline metric SLAM over the whole clip; ARKit reports its current best guess
  and drifts 1–2% of distance travelled. Room-scale that is centimetres and the
  trade is worth it — but the two paths do not produce equally good long
  trajectories, and a whole-flat walk is where the difference shows.
- **The reference episodes were shot steeper than this repo recommends.** Their
  measured pitch is 31.5°, 25.4° and 24.6° below horizontal, against the ~15°
  *Where the vertical view lands* in [DATA_FORMAT.md](DATA_FORMAT.md) argues
  for — past 20°, on a 58° vertical view, the ceiling leaves the frame
  entirely. So the labelled data the VLM prompts were tuned on is floor-heavy
  by a margin, and it is worth knowing which way that cuts before deciding
  whether to match the reference or the recommendation. `Session.camera_aim()`
  reports the same figure for a recorded session, so the comparison is one
  command rather than an argument. Pitch is a domain shift the episode format
  cannot express and the export cannot fix.
