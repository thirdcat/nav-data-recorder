#!/usr/bin/env python3
"""Self-tests for the plumb-line distortion fitter.

The important one is `test_recovers_an_injected_coefficient`: a rendered scene
whose lines are straight by construction, warped by a coefficient chosen here,
must come back.  A fitter that cannot return an injected coefficient says
nothing about a measured one.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_uw_distortion import (  # noqa: E402
    ChainSet, compose_expected, corner_displacement, extract_chains,
    fit_coefficients, half_diagonal, inject_distortion, is_monotone, measure,
    radial_gain, rectify_points, rms_displacement,
)


WIDTH, HEIGHT = 1440, 810


def _straight_scene(width: int = WIDTH, height: int = HEIGHT) -> np.ndarray:
    """A room's worth of straight edges: long spans, few crossings, plus noise.

    Junction breaking is deliberate in the tracer, so the fixture keeps the
    horizontal and vertical families in separate columns rather than laying a
    dense grid whose every intersection would cut the chains short.  Two arcs
    are drawn as distractors: they are curved in the *world* and the robust
    cost has to survive them.
    """
    rng = np.random.default_rng(7)
    image = np.full((height, width), 40, dtype=np.int16)
    image += rng.normal(0.0, 3.0, image.shape).clip(-12, 12).astype(np.int16)
    image = np.clip(image, 0, 255).astype(np.uint8)

    # Thick bands, not thin lines: a wall-ceiling junction or a shelf edge is a
    # step between two large regions, and only that survives the heavy
    # pre-smoothing the real frames need.
    split = int(0.58 * width)
    for index, y in enumerate(range(20, height - 30, 56)):    # shelf edges
        shade = 200 if index % 2 else 120
        cv2.fillPoly(image, [np.array([[4, y], [split - 8, y + 13],
                                       [split - 8, y + 41], [4, y + 28]])],
                     shade, cv2.LINE_AA)
    for index, x in enumerate(range(split + 12, width - 30, 62)):
        shade = 210 if index % 2 else 130                     # door frames
        cv2.fillPoly(image, [np.array([[x, 4], [x + 30, 4],
                                       [x + 12, height - 4],
                                       [x - 18, height - 4]])],
                     shade, cv2.LINE_AA)

    cv2.ellipse(image, (int(0.3 * width), int(0.72 * height)),
                (190, 190), 0, 200, 320, 235, 9, cv2.LINE_AA)
    cv2.ellipse(image, (int(0.82 * width), int(0.18 * height)),
                (160, 160), 0, 20, 150, 60, 9, cv2.LINE_AA)
    return image


def _scene_frames(root: Path, count: int = 34) -> list[Path]:
    """The same straight scene, shifted, so frame bootstrapping has frames."""
    paths = []
    base = _straight_scene()
    for index in range(count):
        shift = np.float32([[1, 0, (index % 7) - 3], [0, 1, (index % 5) - 2]])
        frame = cv2.warpAffine(base, shift, (WIDTH, HEIGHT),
                               borderMode=cv2.BORDER_REFLECT)
        path = root / f"{index:06d}.png"
        assert cv2.imwrite(str(path), frame)
        paths.append(path)
    return paths


def test_model_arithmetic() -> None:
    corner = np.array([[WIDTH, HEIGHT]], dtype=np.float64)
    moved = rectify_points(corner, WIDTH, HEIGHT, [0.1])
    offset = float(np.hypot(*(moved - corner)[0]))
    assert abs(offset - 0.1 * half_diagonal(WIDTH, HEIGHT)) < 1e-9
    assert abs(corner_displacement(WIDTH, HEIGHT, [0.06, 0.04])
               - 0.1 * half_diagonal(WIDTH, HEIGHT)) < 1e-9

    # The centre never moves, whatever the coefficients.
    centre = np.array([[WIDTH / 2, HEIGHT / 2]])
    assert np.allclose(rectify_points(centre, WIDTH, HEIGHT, [0.3, -0.2]), centre)

    # RMS over the frame is smaller than the corner value but the same sign.
    assert 0.0 < rms_displacement(WIDTH, HEIGHT, [0.1]) \
        < abs(corner_displacement(WIDTH, HEIGHT, [0.1]))

    assert is_monotone([0.1])
    assert not is_monotone([-0.9])       # folds over well inside the frame
    assert radial_gain(np.array([0.0]), [0.5])[0] == 1.0


def test_injection_is_exactly_invertible() -> None:
    """rectify(inject(p)) must return p: the injected truth has no slack."""
    grid = np.stack(np.meshgrid(np.linspace(0, WIDTH, 40),
                                np.linspace(0, HEIGHT, 40)), axis=-1)
    for coefficients in ([0.08], [-0.08], [0.05, 0.02]):
        source = rectify_points(grid, WIDTH, HEIGHT, coefficients)
        assert np.isfinite(source).all()
    # With no native error the composition is the injection itself.
    assert abs(compose_expected([0.0], [0.07], 1)[0] - 0.07) < 1e-9
    # With one, the composition is not the naive sum.
    composed = compose_expected([0.02], [0.07], 1)[0]
    assert abs(composed - 0.09) > 1e-4


def test_extracts_long_smooth_chains() -> None:
    chains = extract_chains(_straight_scene())
    assert len(chains) >= 12
    diagonal = float(np.hypot(WIDTH, HEIGHT))
    for chain in chains:
        assert float(np.hypot(*(chain[-1] - chain[0]))) >= 0.12 * diagonal


def test_bow_is_scale_invariant() -> None:
    """A uniform zoom must not change the cost, or the fit shrinks the image."""
    chains = extract_chains(_straight_scene())
    flat = ChainSet(chains, [0] * len(chains), WIDTH, HEIGHT)
    plain = flat.bows([0.0])
    zoomed = ChainSet([c * 1.5 - np.array([WIDTH / 4, HEIGHT / 4])
                       for c in chains], [0] * len(chains), WIDTH, HEIGHT)
    assert np.allclose(plain, zoomed.bows([0.0]), atol=1e-9)


def test_recovers_an_injected_coefficient() -> None:
    with tempfile.TemporaryDirectory() as directory:
        paths = _scene_frames(Path(directory))
        baseline = measure(paths, draws=0)
        assert baseline["sufficiency"]["passed"], baseline["sufficiency"]
        # The scene is straight by construction, so the fit must say so.
        assert abs(baseline["corner_px"]) < 1.0, baseline["corner_px"]

        for injected in (-0.10, -0.05, 0.05, 0.10):
            report = measure(paths, inject=[injected], draws=0)
            assert report["sufficiency"]["passed"], (injected, report)
            # The truth is the composition of the scene's own tiny error with
            # the injection, not their sum.
            truth = compose_expected(baseline["coefficients"], [injected], 1)
            expected = corner_displacement(WIDTH, HEIGHT, truth)
            error = report["corner_px"] - expected
            assert abs(error) <= max(1.0, 0.10 * abs(expected)), \
                (injected, expected, report["corner_px"])
            # The correction must actually straighten what it was fitted to.
            straight = report["straightness"]
            assert straight["median_residual_px_after"] \
                < 0.5 * straight["median_residual_px_before"], straight


def test_injected_image_really_carries_the_bow() -> None:
    """Guard the control itself: the warp must bend the lines it claims to."""
    scene = _straight_scene()
    warped, mask = inject_distortion(scene, [0.10])
    assert mask.dtype == bool and mask.any()
    before = extract_chains(scene)
    after = extract_chains(warped, mask=mask)
    flat_before = ChainSet(before, [0] * len(before), WIDTH, HEIGHT)
    flat_after = ChainSet(after, [0] * len(after), WIDTH, HEIGHT)
    assert np.median(flat_after.sagitta_px([0.0])) > \
        4.0 * np.median(flat_before.sagitta_px([0.0]))


def test_fit_refuses_a_fold_over() -> None:
    chains = extract_chains(_straight_scene())
    flat = ChainSet(chains, [0] * len(chains), WIDTH, HEIGHT)
    fitted = fit_coefficients(flat, 1)
    assert is_monotone(fitted)


def main() -> int:
    test_model_arithmetic()
    test_injection_is_exactly_invertible()
    test_extracts_long_smooth_chains()
    test_bow_is_scale_invariant()
    test_injected_image_really_carries_the_bow()
    test_fit_refuses_a_fold_over()
    test_recovers_an_injected_coefficient()
    print("fit_uw_distortion self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
