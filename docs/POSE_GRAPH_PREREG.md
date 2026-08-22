# Pre-registration — loop closure over the session graph

Written 2026-08-22, before a line of the optimiser existed. Every threshold and
every constant below was fixed before the first result was read. Nothing here
may be edited after a result is read; corrections are appended with a timestamp
at the bottom.

## The question

`docs/3DGS.md` "Not done" opens with:

> **A tree, not a pose graph.** There is no loop closure, so error accumulates
> along a chain. The composed-versus-direct columns are there to make that
> visible, not to fix it.

`tools/align_set.py` reduces the session graph to a maximum spanning tree rooted
at the reference. Every session reaches the reference by exactly one path, so
every measurement that is *not* on that path is discarded, and any disagreement
between two paths is absorbed silently by whichever path the tree happened to
pick.

**Build the missing step: a pose-graph optimisation over the session graph.**

## What this pre-registration does *not* claim

The measured loss this sits next to is that merging trains **worse** — 22.64 dB
single against 20.39 dB merged, 22 held-out photographs of 23 — while
multiplying angular coverage 3.8–13.4x on the shared surface. The stated reason
is that "coverage tolerates centimetres of misplacement; a renderer does not".

It would be natural to present loop closure as the fix for that. **It is not
claimed here.** This work has no GPU, trains nothing, and renders nothing.
Whatever the optimisation moves, in metres, is the only quantity this page will
report. A PSNR or coverage improvement is **NOT DEMONSTRATED** and the report
must say so in those words unless it is demonstrated.

There is a second reason for that caution, and it is structural rather than
practical. Loop closure removes **inconsistency** between paths. It cannot
remove a **bias** shared by every edge. `docs/3DGS.md` measures that the
geometric estimator's horizontal error is largely the latter: the floor supplies
most of the points and constrains only its own normal, so every edge is loose in
the same direction. A graph of consistently wrong edges optimises to a
consistently wrong answer. This is stated here so it cannot be discovered later
and presented as a caveat.

## The design, and what each choice is answerable to

### Nodes, edges, residual

Node `i` carries `X_i = (R_i, t_i)`, world-from-session, the same convention
`align_set.py` writes. Edge `(i, j)` carries a measurement `Z_ij = (R_z, t_z)`,
the transform taking session `j`'s points into session `i`'s frame — which is
exactly what `align_sessions.align_clouds(target=i, source=j)` returns.

Residual, in the **global gravity-aligned frame** so that "horizontal" and
"vertical" mean what the measurements meant:

```
  predicted   R̂_j = R_i R_z            t̂_j = t_i + R_i t_z
  e_R = Log(R_j R̂_j^T)   ∈ R³ (rad)   e_t = t_j - t̂_j   ∈ R³ (m)
```

`e_R`'s Y component is yaw (`worldAlignment = .gravity`, +Y up); X and Z are
tilt. `e_t`'s Y component is height; X and Z are horizontal. Zero residual iff
`X_j = X_i Z_ij` exactly.

Gauss–Newton with Levenberg damping on `δ = (ω, u)` per node, left-multiplied
on rotation and added to translation in the global frame. The reference node's
six variables are **eliminated**, not softly constrained: the gauge is fixed by
construction and the reference cannot drift.

### Edges are weighted by the photometric score, never by fitness

`docs/3DGS.md` measures, across the corpus tree, that neither geometric
criterion separates real from false:

- fitness: real 0.51–0.85, false 0.21–0.63 — a false edge at 0.63 outscores
  three of four real ones.
- ratio: the default `--min-ratio 2.0` rejects two real pairs of four and admits
  five false ones.
- photometric: real 0.04–0.15, false 0.35–1.03. A 2.3x gap, no overlap.

And in `eval/verify_groups.py`'s verdict, two **cut** edges score fitness 0.73
while a **kept** one scores 0.33; the ordering by fitness is close to reversed.

So:

> **W1.** Edge information is a function of the photometric score alone.
> Geometric fitness enters nowhere in the weight. It is carried through the
> report for comparison and is never read by the solver.

Admission uses the same threshold and the same code path as the existing
verdict — `COHERENT = 0.25`, imported from `eval/verify_groups.py` rather than
copied, so there is one constant and not two.

The weight among admitted edges:

```
  w_ij = (s_ref / s_ij) ** p        s_ref = 0.05,  p = 2
```

`p = 2` means the residual σ is **proportional** to the photometric score. The
justification is the only relation on the page between that score and a measured
misalignment — the four refinement rows, of which three are genuine pairs:

```
  score before refinement   horizontal correction the photographs then demanded
      0.0435                        1.7 cm
      0.1234                       14.2 cm
      0.1485                       18.7 cm
```

That is three points and two parameters. It fixes an **ordering**, not a law,
and `s_ref` is a gauge that cancels — only ratios of weights reach the estimate.
**A7 below measures how much the answer depends on `p` at all**, precisely
because this justification is thin.

### The information matrix is anisotropic, and here is the number

`docs/3DGS.md` measures, on four pairs, that the correction the photographs
demand is **3.6 to 7.8 times larger horizontally than vertically**, and that
geometric ICP is already at its floor: 0.80 cm cross-session against a 0.77 cm
sensor repeatability. An isotropic error model would contradict four
measurements. So:

```
  σ_v    = 0.020 m     median vertical correction over the three genuine pairs
                       {0.5, 1.9, 3.5} cm; agrees with "height is pinned to
                       about a centimetre"
  σ_h    = 0.104 m     = 5.21 × σ_v, the geometric mean of the per-pair ratios
                       on the three genuine pairs {3.6, 7.4, 5.3}x. The false
                       pair's 7.8x is excluded — it is not a pair.
  σ_tilt = 0.5°        measured residual tilt on genuine pairs, 0.18-0.59°
  σ_yaw  = 2.4°        NOT MEASURED. See below.
  Ω_ij   = w_ij · blkdiag( diag(1/σ_tilt², 1/σ_yaw², 1/σ_tilt²),
                           diag(1/σ_h²,    1/σ_v²,   1/σ_h²)    )
```

**`σ_yaw` is a modelling choice and is labelled as one.** No measurement of the
yaw error of a cross-session alignment exists on the page: `photo_align.refine`
sweeps yaw but `refine_transforms.py` records only the translation. It is set to
`σ_h / L` with `L = 2.5 m`, a representative distance from a session's origin to
the surface that constrains its alignment — the camera centres of the twelve
placed sessions span a **median radius of 1.05 m** about their own centroid
(measured 2026-08-22, before this file was written: 0.17, 0.38, 0.54, 0.60,
0.85, 0.88, 1.21, 1.64, 1.97, 2.38, 5.06, 7.48 m), and depth reaches 0.2–5.0 m
beyond them. **A8 measures how much the answer depends on it.**

### A prediction about the anisotropy, made before running it

With `worldAlignment = .gravity` every node rotation and every edge rotation is
close to a pure yaw. Under a pure yaw the translation residual separates: the Y
component of `e_t` depends only on Y components, and X/Z only on X/Z. The only
coupling between the horizontal and vertical blocks is through
`∂e_t/∂ω_i = skew(R_i t_z)`, whose horizontal rows carry the *vertical* lever
arm `p_y` — and these sessions are all handheld at roughly one height.

> **P1 (prediction).** On a gravity-aligned graph, changing σ_h/σ_v from 5.21 to
> 1.0 with σ_h held fixed moves **no node by more than 1 mm**. The anisotropy is
> the right model and is nearly a gauge on the *estimate*; where it is not a
> gauge is the reported covariance and any χ² consistency test.
>
> **P2 (prediction).** On a graph with deliberate 5° tilts on nodes and edges,
> the same change moves at least one node by **more than 1 mm**. If P2 fails the
> anisotropy is inert everywhere and the model choice is untestable here.

Both are falsifiable in either direction and both outcomes are reportable.

### An edge that cannot be judged does not get a weight

`eval/verify_groups.py` reports "too few usable pairs" on three tree edges of
eleven rather than guessing, and `eval/umeyama.py` raises `ScaleNotObservable`
rather than returning a plausible 1.0. The same posture here:

> **U1.** An edge with no finite photometric score is **refused**, by raising,
> not weighted by a default. A node left unreachable from the reference by that
> refusal is reported as unplaced, never placed at whatever the tree said.

## Acceptance criteria

Each has a stated pass condition and can fail. Failures are reported, not
absorbed.

### A1 — exact recovery on a known answer, noiseless, with a loop

Six nodes at chosen true poses in a 6 × 6 × 0.4 m box, yaw uniform, tilt ±0.5°.
Eight edges: the ring `0-1-2-3-4-5-0` plus chords `(0,3)` and `(1,4)` — five
tree edges, three independent loops. Measurements exact. Initialise from tree
composition **plus a deliberate perturbation** of every non-reference node
(5 cm, 2°), so that the optimiser is not handed the answer.

**PASS** iff every node matches truth within **1e-9 m** and **1e-9 rad** in
≤ 50 iterations. An optimiser that has never been run on a known answer has not
been shown to work.

### A2 — a loop that does not close under the tree, and closes after

Same graph, measurements corrupted with the pre-registered noise
(σ_h = 0.104 m, σ_v = 0.020 m, σ_yaw = 2.4°, σ_tilt = 0.5°), 200 seeds. Repeated
with an isotropic misspecified noise (5 cm, 1° on every axis) to check the claim
does not depend on the generative model matching the information matrix.

- **A2a** (prediction) median over seeds of the worst chord disagreement under
  tree composition ≥ **0.03 m**.
- **A2b** worst edge residual after optimisation ≤ worst under the tree, in
  ≥ **95 %** of seeds.
- **A2c** RMS node position error against truth: **median improvement ≥ 20 %**,
  and worse than the tree in **≤ 10 %** of seeds. Both noise settings.
- **A2d** the reference node moves **exactly** zero.

### A3 — must not degrade a chain that is already consistent

A random spanning tree, no chord, noisy measurements, 200 seeds, and every σ
setting used anywhere in A6–A8. A tree admits a zero-residual solution, so the
optimum *is* the tree composition.

**PASS** iff the optimised poses equal the tree composition within **1e-9 m**
and **1e-9 rad** on every node, every seed, every setting.

### A4 — independent check against a closed form

A translation-only graph (all rotations identity, translation measurements
only) is a linear least-squares problem. Solve it independently by assembling
the weighted incidence matrix and calling `np.linalg.lstsq`, code that shares
nothing with the solver.

**PASS** iff the two agree within **1e-9 m** on every node. "Verify
independently" means a second implementation, not a second run.

### A5 — the analytic Jacobian is the derivative it claims to be

Analytic Jacobian against central finite differences on a random graph.

**PASS** iff max absolute error ≤ **1e-6** relative to the column norm.

### A6 — the anisotropy predictions P1 and P2

As stated above. Report the measured maximum node displacement in both cases.

### A7 — how much does the photometric weight exponent matter

Sweep `p ∈ {0, 1, 2, 3}` (`p = 0` is "all admitted edges equal") on the real
corpus graph and on the synthetic graph. Report the maximum node displacement
across the sweep.

**Decision rule, fixed now:** if the real-corpus answer moves by more than
**2 cm** between `p = 0` and `p = 2`, the weight map is load-bearing while being
justified by three points, and every number downstream must be reported with
that stated, not as a single value.

### A8 — how much does σ_yaw matter

Sweep `σ_yaw ∈ {0.6°, 1.2°, 2.4°, 4.8°}`. Same 2 cm decision rule as A7.

### A9 — reproduce the existing corpus verdict before changing anything

`eval/verify_groups.py` on `gs3d/night/corpus_coobs.json` must return the eleven
rows published in `docs/3DGS.md` — 4 keep, 4 cut, 3 too-few-usable — with each
photometric score within **±0.001** of the published value, and 8 components
with exactly the published membership.

Then: the new tool's own admission rule, run over those same eleven edges, must
return the **identical** keep / cut / unjudgeable verdict.

**FAIL ⇒ report as a finding. Do not absorb it, do not adjust a threshold to
recover it.**

Adding a chord that the tree never used *may legitimately merge components* —
`docs/3DGS.md` says so explicitly ("an edge the tree never used could rejoin
two"). Any such merge is a change to the published grouping and must be reported
in those words, with the edge and its score.

### A10 — an unjudgeable edge is refused, not defaulted

Constructing an edge from a `None` score raises. A node reachable only through
such an edge is reported unplaced. Tested.

### A11 — what the optimisation changes on the real corpus, in metres

Candidate chords are fixed **now**, so they cannot be chosen after seeing a
result. They are every non-tree unordered pair among the twelve placed sessions
of `corpus_coobs.json` whose co-observation count (already in that file,
computed from poses alone) is **≥ 50**, which is these nine and no others:

```
  e11854 - 683ef1   206        e9d8f7 - 683ef1   104        683ef1 - 40b157    71
  e11854 - 2994fa   204        505b2c - a09199    97        2994fa - 532cea    70
  e11854 - 40b157   126        683ef1 - 2994fa    87
  e9d8f7 - 2994fa   112
```

Each is aligned geometrically by the existing `align_sessions.align_clouds` at
the existing defaults, then scored by the existing
`verify_groups.score_edge`, then admitted or refused by `COHERENT = 0.25`.

Report: the displacement of every node in metres, split horizontal and vertical;
the loop residual before and after; the χ² before and after; and, in these
words, that **a coverage or PSNR improvement has NOT been demonstrated**.

If no chord is admitted, there is no loop on this corpus, and that is the
result — reported as such, with the synthetic acceptance standing on its own.

## Files

- `tools/pose_graph.py` — the solver. numpy only, no SciPy, no imports from
  `eval/`, per `AGENTS.md`.
- `tools/test_pose_graph.py` — A1–A8, A10. Exits nonzero on failure.
- `eval/close_loops.py` — the pipeline stage. Reads what `align_set.py` writes,
  scores candidate chords with photographs, optimises, writes the same schema
  back. In `eval/` because it reads photographs.
- `tools/align_set.py`, `tools/align_sessions.py` — additive only, behind a
  flag, so every existing result reproduces byte for byte.

Not touched: `tools/export_3dgs.py`, `tools/survey_coverage.py`,
`eval/pi3_poseless.py`, `eval/fuse_rate.py`, `App/**`, `docs/3DGS.md`.

---

## Appended after results

*(timestamped corrections only; nothing above is edited)*

**2026-08-22, after every result was read.**

**1. A6/P1 FAILED, and the failure is the useful part.** Predicted: on a
gravity-aligned graph, dropping σ_h/σ_v from 5.21 to 1.0 moves no node by more
than 1 mm. Measured: **11.55 mm**. The prediction is wrong and the bar is not
moved.

The derivation behind it was right about the plane it named and blind to the one
it did not. Of that 11.55 mm, **0.34 mm is horizontal** — the horizontal/vertical
separation P1 argued for is real and holds. The other 11.5 mm is vertical, and
arrives by a route the derivation never considered: σ_v is not only the weight
on the vertical residual, it is the price at which **tilt** is recruited to
absorb one, and `∂e_t_y/∂ω_{x,z} = ∓p_{x,z}` carries a lever arm of *metres*.
Measured on the same graph: the worst node's tilt is 0.27° under the anisotropic
weighting and 0.013° under the isotropic one. Sweeping the vertical spread of
the node origins from 0 to 1.6 m changes the 11.55 mm by 0.01 mm — it is not the
`p_y` term the derivation blamed — while removing the vertical *measurement*
noise collapses it to 0.05 mm.

So the corrected statement is: **σ_h/σ_v is nearly a gauge on the horizontal
estimate and load-bearing on the vertical one, through tilt.**

P2 passed (33.18 mm on a 5°-tilted graph), so the anisotropy is not inert.

**2. A4 was substituted, and the reason is the same coupling.** The
pre-registered closed-form check assumed a translation-only graph has a
translation-only solution. It does not: `∂e_t/∂ω_i = skew(R_i t_z)` lets a node
move sideways by turning, and with `|t_z|` of metres that is a *cheaper* way to
move than translating. The minimiser of the stated objective is therefore not
the linear least squares answer, and the 1e-9 bar was unreachable by
construction, not by defect.

A4 is now agreement with a **second, independently written minimiser**:
finite-difference Levenberg–Marquardt on a flat parameter vector, sharing the
residual definition and nothing else — no analytic Jacobian, no block assembly,
no gauge elimination. Result 1.31e-10 m, bar 1e-9. A4b keeps the closed form as
a clamped limit (rotation σ at 1e-5 rad): 6.20e-08 m against a 1e-4 m bar, which
is the tolerance the conditioning allows.

**3. A2a's bar was not a discriminating one.** Predicted ≥ 3 cm of chord
disagreement under tree composition; measured **46.3 cm**. It passes by a factor
of fifteen because the disagreement is dominated by σ_yaw = 2.4° acting over a
6 m box, not by the translation noise the 3 cm was reasoned from. It should be
read as a sanity check that the graph has loops at all, not as evidence the
noise model is calibrated.

**4. A8 on the real corpus returned exactly zero, and that is a property of the
graph, not of the sweep.** Sweeping σ_yaw over 0.6–4.8° moved the real-corpus
answer by < 5 µm, while the same sweep moves the synthetic six-node,
three-loop graph by 12.4 cm. With **one** loop and the same σ_yaw on every edge,
its absolute value cancels: the loop error splits between edges by the ratios
`1/w_e` alone. The most uncalibrated constant in this file happens not to bite on
the only real loop available — which is luck, not validation.

**5. The other bars: A1, A2b-d, A3, A4, A5, A6/P2, A7, A9, A10, A11 all
passed.** A9 reproduced the published verdict exactly (4 keep, 4 cut, 3
too-few-usable, every score to four decimals, 8 components with the published
membership) because `eval/close_loops.py` calls
`eval/verify_groups.score_edge` rather than re-implementing it. A7 on the real
corpus moved the answer by 0.73 cm across `p ∈ {0,1,2,3}`, under the 2 cm
decision rule, so the thinly-justified weight map is **not** load-bearing here.

**6. A11: nine chords proposed, one admitted.** `505b2c <- a09199`, photometric
0.0824, 97 co-observing frames. It closes one triangle. The tree left a
**2.95 cm and 0.933°** disagreement between its two paths; the optimisation
moved `a09199` by **2.19 cm** and `40b157` by 0.29 cm, χ² 1.593 → 0.408. The
published grouping did **not** change: no chord joined two components.

**A coverage or PSNR improvement has NOT been demonstrated.**
