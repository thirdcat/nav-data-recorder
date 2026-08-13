# Repository Guidelines

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
```

The first two commands create and inspect a phone-free fixture. The remaining
commands run synthetic geometry self-tests. The calibration and rectification
tests require NumPy; rectification also requires `opencv-python-headless`. The
three 3DGS tests need NumPy and Pillow, and cover `export_3dgs.py`,
`survey_coverage.py` with `plot_coverage.py`, and `eval_views.py`.
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
