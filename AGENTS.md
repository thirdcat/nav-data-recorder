# Repository Guidelines

## Where to start — the work plan is the entry point

The end product is the real2sim half of a real2sim2real pipeline: a scanner app
whose output is a metric-aligned mesh + 3DGS that a robot simulator loads. All
research and app work is ordered by one document:

**`docs/METRIC_RECONSTRUCTION_PLAN.md`** — §1 is the status table, **§1.6 is the
ranked work list**, §9 is the change history. A new session proceeds like this:

1. Read §1 and §1.6. Take the **lowest-ranked row of §1.6 whose §1 status is
   `예정` and whose `선행` column is satisfied**. Do not jump past a `대기` row
   unless its prerequisite is actually done; do not reopen anything listed under
   "내려간 것" in §1.6 or "닫힌 것" in `HANDOVER.md` §6 without new evidence.
2. Before running anything, set that row's §1 status to `진행`, and fix the
   acceptance criterion in numbers using the §8 experiment template — *before*
   looking at results. Record the command, seed and output location so a restart
   does not rerun it.
3. Results go under `logs/<experiment>/` with a README; the document links to
   them and carries only the summary. Then update the §1 row (`완료` /
   `보류` with reason) and add a §9 history row stating the previous judgement,
   what changed and why. Never create a second plan file.
4. Rank 0 of §1.6 is "commit the uncommitted work". If `git status` shows the
   backlog still there, that is the first task. Peers share this checkout: do
   not switch branches, and treat unknown `git status` entries as theirs.

Working rules that cost the most to learn are in `HANDOVER.md` §8 — pre-register
the decision rule, run every instrument on a known answer first, keep the capture
behaviour and the instrument in separate commits, one GPU job at a time under
`flock`. `HANDOVER.md` also has the GPU environment lines.

Context a session needs and cannot derive from the tree: recorded sessions are
in `~/nav_data` (80 sessions, not in the repo); 3DGS training runs live in
`~/uv_workspace/gs3d`; the sibling project with the same end goal is
`~/uv_workspace/LiteReality-Agent` (RoomPlan shell + generated assets + MuJoCo
export — its sim-ready contract is `doc/Sim-Ready-intergration/Mujoco.md`, its
capture reader `src/litereality_agent/evidence_kit/read_scan.py`); the S1
converter in the plan targets that reader's format.

## Project Structure & Module Organization

- `App/Sources/` contains the Swift iOS app, grouped by responsibility: ARKit,
  motion, location, storage, upload, models, and SwiftUI views.
- `tools/` contains Python session readers, packers, exporters, calibration
  utilities, and their executable self-tests. `docs/` defines setup, session
  formats, pose conventions, and output conventions.
- `sample_data/` provides recorded fixtures; generated captures and derived
  exports should remain local unless they are intentionally curated.
- `eval/` holds the GPU scoring harness for non-ARKit pose sources. It requires
  PyTorch and CUDA, so the dependency runs one way only: `eval/` may import from
  `tools/`, and `tools/` must never import from `eval/`. That is what keeps
  `tools/` runnable on numpy alone.
- `project.yml` is the source of truth for the Xcode project. The `.xcodeproj`
  is generated and must not be committed.

## Build, Test, and Development Commands

Run commands from the repository root:

```bash
python3 tools/make_test_session.py /tmp/fixture
python3 tools/read_session.py /tmp/fixture/20260807-014530-fixture
python3 tools/test_estimate_intrinsics.py
python3 tools/test_rectify_ultrawide.py
python3 tools/test_depth_odometry.py
python3 tools/test_export_3dgs.py
python3 tools/test_survey_coverage.py
python3 tools/test_eval_views.py
python3 tools/test_align_sessions.py
```

The first two commands create and inspect a phone-free fixture. The remaining
commands run synthetic geometry self-tests. The calibration and rectification
tests require NumPy; rectification also requires `opencv-python-headless`. The
four 3DGS tests need NumPy, and Pillow for all but the last; they cover
`export_3dgs.py`, `survey_coverage.py` with `plot_coverage.py`, `eval_views.py`,
and `align_sessions.py`.
On macOS with XcodeGen, generate the project with
`xcodegen generate --spec project.yml`; CI then builds it with `xcodebuild`.
The supported build path is the GitHub Actions or Codemagic workflow because
the app requires an iOS/macOS toolchain and signing setup.

## Coding Style & Naming Conventions

Use four-space indentation. Follow Swift lowerCamelCase for members and
PascalCase for types, keeping capture/storage responsibilities in their
existing directories. Python modules, functions, and CLI options use
`snake_case`; keep scripts runnable from the repository root. No formatter or
linter is configured, so preserve nearby style and keep changes focused.

## Testing Guidelines

Tests are standalone `tools/test_*.py` programs and should exit nonzero on
failure. Add deterministic synthetic cases when changing geometry or session
joining, and run the relevant self-test plus the fixture reader. Document any
changed timestamp, coordinate-frame, or file-schema behavior in `docs/`.

## Commits and Pull Requests

Use short, imperative, sentence-case commit subjects, matching history (for
example, `Add a session packer`). PRs should explain the behavioral or data
format impact, link relevant issues, include test commands and CI results, and
attach screenshots or sample output when changing the iOS UI or exported data.

## Security and Configuration

Never commit signing material (`*.p8`, `*.p12`, certificates, or provisioning
profiles), upload tokens, or private session data. Keep credentials in CI or
local environment configuration; review generated files before staging.
