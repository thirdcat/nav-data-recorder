#!/usr/bin/env python3
"""Self-tests for the held-out scoring.

A scoring harness is the last thing that should be trusted on the grounds that
it looks reasonable: it is the instrument every later decision is read off, and
a metric that is subtly wrong produces confident, consistent, wrong answers for
as long as nobody checks it. So each metric here is given an input whose score
is known on paper.

    python3 tools/test_eval_views.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_views as ev  # noqa: E402
import export_3dgs as ex  # noqa: E402


FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILURES.append(name)


def test_psnr_against_arithmetic() -> None:
    print("PSNR is the quantity it claims to be")
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    check("identical images saturate", ev.psnr(a, a) == 99.0, f"{ev.psnr(a, a)}")

    # A uniform offset of d has mean squared error d^2, so PSNR is known exactly.
    for offset in (1, 5, 20):
        b = np.clip(a.astype(np.int32) - offset, 0, 255).astype(np.uint8)
        # Clipping breaks the arithmetic where a was already dark, so compare on
        # a flat mid-grey image where no pixel can clip.
        flat = np.full((64, 64, 3), 128, np.uint8)
        shifted = np.full((64, 64, 3), 128 - offset, np.uint8)
        want = 20.0 * np.log10(255.0 / offset)
        got = ev.psnr(flat, shifted)
        check(f"a uniform offset of {offset} scores {want:.2f} dB",
              abs(got - want) < 1e-6, f"got {got:.4f}")
        del b

    check("worse images score lower",
          ev.psnr(np.full((32, 32, 3), 128, np.uint8),
                  np.full((32, 32, 3), 138, np.uint8))
          < ev.psnr(np.full((32, 32, 3), 128, np.uint8),
                    np.full((32, 32, 3), 130, np.uint8)))


def test_ssim_bounds_and_ordering() -> None:
    print("SSIM is bounded and ordered")
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (96, 96), dtype=np.uint8).astype(np.float64)
    img = ev._blur(img, 3, 2.0)  # something with structure rather than pure noise

    check("identical is one", abs(ev.ssim(img, img) - 1.0) < 1e-6,
          f"{ev.ssim(img, img)}")
    noisy = img + rng.normal(scale=20.0, size=img.shape)
    mild = img + rng.normal(scale=3.0, size=img.shape)
    check("more noise scores lower", ev.ssim(mild, img) > ev.ssim(noisy, img),
          f"{ev.ssim(mild, img):.4f} vs {ev.ssim(noisy, img):.4f}")
    check("stays within bounds", -1.0 <= ev.ssim(noisy, img) <= 1.0)
    unrelated = rng.integers(0, 256, img.shape).astype(np.float64)
    check("an unrelated image scores near zero", ev.ssim(unrelated, img) < 0.2,
          f"{ev.ssim(unrelated, img):.4f}")


def test_depth_error() -> None:
    print("depth error measures metres")
    truth = np.full((40, 40), 2.0)                          # 2 m everywhere
    pred = np.full((40, 40), 2.05)                          # 5 cm too far
    mask = np.ones((40, 40), bool)
    got = ev.depth_error(pred, truth, mask)
    check("five centimetres reads as 0.05 m", abs(got["median_abs_m"] - 0.05) < 1e-9,
          f"{got}")
    check("and as 2.5 percent relative", abs(got["abs_rel"] - 0.025) < 1e-9, f"{got}")
    check("coverage is the fraction compared", abs(got["coverage"] - 1.0) < 1e-9)

    # Pixels with no return on either side must not be scored as perfect.
    sparse = np.zeros((40, 40))
    sparse[:10] = 2.0
    got = ev.depth_error(np.full((40, 40), 2.0), sparse, np.ones((40, 40), bool))
    check("empty depth is excluded, not counted as agreement",
          abs(got["coverage"] - 0.25) < 1e-9, f"{got}")
    check("too little overlap returns nothing rather than a number",
          ev.depth_error(np.ones((40, 40)), np.zeros((40, 40)),
                         np.ones((40, 40), bool)) is None)


def test_depth_is_read_in_whichever_unit_the_export_declared() -> None:
    print("depth sidecars are read through transforms.json, not by guessing")
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "depth"))
        np.save(os.path.join(tmp, "depth", "000000.npy"),
                np.full((4, 4), 2.5, np.float32))
        with open(os.path.join(tmp, "transforms.json"), "w") as fh:
            fh.write('{"depth_unit_scale_factor": 1.0, "frames": ['
                     '{"file_path": "images/000000.jpg",'
                     ' "depth_file_path": "depth/000000.npy"}]}')
        index, unit = ev.depth_index(tmp)
        check("the frame is indexed by image name", "000000.jpg" in index, f"{index}")
        check("the declared unit is metres", unit == 1.0, f"{unit}")
        got = ev.load_depth_metres(index["000000.jpg"], (8, 8), unit)
        check("npy comes back in metres at the asked size",
              got.shape == (8, 8) and abs(got[0, 0] - 2.5) < 1e-6, f"{got.shape} {got[0,0]}")

    # And the millimetre PNG layout reaches the same metres.
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "depths"))
        from PIL import Image as PILImage
        PILImage.fromarray(np.full((4, 4), 2500, np.uint16)).save(
            os.path.join(tmp, "depths", "000000.png"))
        with open(os.path.join(tmp, "transforms.json"), "w") as fh:
            fh.write('{"depth_unit_scale_factor": 0.001, "frames": ['
                     '{"file_path": "images/000000.jpg",'
                     ' "depth_file_path": "depths/000000.png"}]}')
        index, unit = ev.depth_index(tmp)
        got = ev.load_depth_metres(index["000000.jpg"], (4, 4), unit)
        check("png16 comes back as the same 2.5 m", abs(got[0, 0] - 2.5) < 1e-6,
              f"{got[0, 0]}")
    check("a model with no transforms.json degrades quietly",
          ev.depth_index(tempfile.gettempdir() + "/definitely-not-here") == ({}, 0.001))


def test_mixed_camera_resolutions_get_separate_mean_controls() -> None:
    print("mixed camera resolutions use matching mean controls")
    from PIL import Image as PILImage

    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "images"))
        PILImage.fromarray(np.full((4, 6, 3), 32, np.uint8)).save(
            os.path.join(tmp, "images", "wide.jpg"))
        PILImage.fromarray(np.full((8, 10, 3), 224, np.uint8)).save(
            os.path.join(tmp, "images", "ultrawide.jpg"))
        cams = {
            1: {"w": 6, "h": 4},
            2: {"w": 10, "h": 8},
        }
        train = [
            {"camera": 1, "name": "wide.jpg"},
            {"camera": 2, "name": "ultrawide.jpg"},
        ]
        got = ev.build_mean_images(tmp, train, cams, 1.0)
        check("one mean exists per output size", set(got) == {(6, 4), (10, 8)}, f"{set(got)}")
        check("wide mean keeps wide shape and value",
              got[(6, 4)].shape == (4, 6, 3) and np.all(got[(6, 4)] == 32),
              f"{got[(6, 4)].shape} {got[(6, 4)][0, 0]}")
        check("ultrawide mean keeps ultrawide shape and value",
              got[(10, 8)].shape == (8, 10, 3) and np.all(got[(10, 8)] == 224),
              f"{got[(10, 8)].shape} {got[(10, 8)][0, 0]}")


def test_scoring_a_render_does_not_collide_with_the_depth_truth() -> None:
    """Regression: the colour truth and the depth truth are different arrays.

    They were briefly the same name, and the failure was a broadcast error at
    the moment a real render was first scored — late, and only because a render
    existed to score. Nothing about the controls-only path would have shown it.
    """
    print("colour and depth ground truth stay separate")
    tools = os.path.dirname(os.path.abspath(__file__))
    with tempfile.TemporaryDirectory() as tmp:
        made = subprocess.run([sys.executable, os.path.join(tools, "make_test_session.py"), tmp],
                              capture_output=True, text=True)
        sessions = [os.path.join(tmp, d) for d in os.listdir(tmp)
                    if os.path.isdir(os.path.join(tmp, d))]
        if made.returncode != 0 or not sessions:
            check("fixture generated", False, made.stderr.strip()[:200])
            return

        model = os.path.join(tmp, "model")
        ex.export(sessions[0], model, pix_stride=6, image_mode="copy",
                  require_exact_depth=False, max_points=0, holdout_every=4)

        # Stand in for a trainer: copy the held-out images back as "renders",
        # which makes the render column a perfect reconstruction.
        renders = os.path.join(tmp, "renders")
        os.makedirs(renders)
        held = [ln.strip() for ln in open(os.path.join(model, "holdout.txt"))
                if ln.strip() and not ln.startswith("#")]
        for name in held:
            shutil.copy(os.path.join(model, "images", name),
                        os.path.join(renders, name))

        result = ev.evaluate(model, scale=0.25, render_dir=renders)
        summary = result["summary"]
        check("scoring a render runs at all", "render_psnr" in summary, f"{summary}")
        check("a perfect render scores near the ceiling",
              (summary["render_psnr"] or 0) > 40, f"{summary['render_psnr']}")
        check("and beats the mean-image floor",
              (summary["render_psnr"] or 0) > (summary["mean_psnr"] or 99),
              f"{summary['render_psnr']} vs {summary['mean_psnr']}")
        check("the lidar control still reports depth",
              summary["lidar_depth_median_abs_m"] is not None, f"{summary}")


def test_a_picture_of_depth_is_not_scored_as_depth() -> None:
    """The failure that produced 1.332 m against a true 0.036 m.

    `ns-render` writes depth through a turbo colormap by default: a
    three-channel 8-bit PNG of hues. Read as millimetres it yields a
    metre-scale error that looks exactly like a measurement of a bad model.
    """
    print("a colourised depth picture is refused, not scored")
    from PIL import Image as PILImage
    rng = np.random.default_rng(4)
    with tempfile.TemporaryDirectory() as tmp:
        PILImage.fromarray(rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)).save(
            os.path.join(tmp, "000000.png"))
        got, why = ev.read_rendered_depth(tmp, "000000.jpg", (8, 8), 0.001)
        check("nothing is returned", got is None)
        check("and the reason names the problem", "colourised" in why, why)

    with tempfile.TemporaryDirectory() as tmp:
        PILImage.fromarray(np.full((8, 8), 2500, np.uint16)).save(
            os.path.join(tmp, "000000.png"))
        got, why = ev.read_rendered_depth(tmp, "000000.jpg", (8, 8), 0.001)
        check("a real 16-bit depth PNG is accepted",
              got is not None and abs(got[0, 0] - 2.5) < 1e-9, f"{why}")

    with tempfile.TemporaryDirectory() as tmp:
        np.save(os.path.join(tmp, "000000.npy"), np.full((8, 8, 1), 2500.0, np.float32))
        got, why = ev.read_rendered_depth(tmp, "000000.jpg", (8, 8), 1.0)
        check("a single-channel float npy is accepted",
              got is not None and abs(got[0, 0] - 2500.0) < 1e-9, f"{why}")

        got = ev.load_depth_metres(os.path.join(tmp, "000000.npy"), (4, 4), 0.001)
        check("HxWx1 depth is reduced to metres",
              got.shape == (4, 4) and abs(got[0, 0] - 2.5) < 1e-9, f"{got.shape}")

    with tempfile.TemporaryDirectory() as tmp:
        got, why = ev.read_rendered_depth(tmp, "000000.jpg", (8, 8), 0.001)
        check("a missing file says so rather than guessing", got is None and why)


def test_splat_puts_a_point_where_projection_says() -> None:
    print("the control splat agrees with the projection it is scored against")
    p = {"tx": 0.3, "ty": -0.2, "tz": 1.1, "qx": 0.1, "qy": 0.3, "qz": -0.05,
         "qw": 0.947, "fx": 800.0, "fy": 800.0, "cx": 320.0, "cy": 240.0}
    (qw, qx, qy, qz), t = ex.world_to_camera(p)
    img = {"qw": qw, "qx": qx, "qy": qy, "qz": qz, "t": t, "camera": 1, "name": "x"}
    cam = {"w": 640, "h": 480, "fx": 800.0, "fy": 800.0, "cx": 320.0, "cy": 240.0}

    # Two metres in front, nudged off the principal point so it is not sitting
    # on a pixel boundary: u = 800*0.001/2 + 320 = 320.4, which floors to 320
    # under any rounding of the last bit. Exactly 320.0 would not.
    c2w = ex.camera_to_world(p)
    offset = np.array([0.001, 0.001, 2.0])
    world = c2w[:3, 3] + c2w[:3, :3] @ offset
    colour, depth, mask = ev.splat(world[None, :], np.array([[200, 100, 50]], np.uint8),
                                   cam, img, 1.0)
    check("exactly one pixel is covered", int(mask.sum()) == 1, f"{int(mask.sum())}")
    v, u = np.argwhere(mask)[0]
    check("it lands in the pixel the projection floors to", (u, v) == (320, 240),
          f"({u}, {v})")
    check("at the right depth", abs(depth[v, u] - 2.0) < 1e-6, f"{depth[v, u]}")
    check("with the right colour", tuple(colour[v, u]) == (200, 100, 50))

    # The convention itself, stated where it can fail: a coordinate short of an
    # integer belongs to the pixel below, not the one it is nearest to.
    just_under = np.array([(319.9 - 320.0) * 2.0 / 800.0, 0.001, 2.0])
    _, _, mask = ev.splat((c2w[:3, 3] + c2w[:3, :3] @ just_under)[None, :],
                          np.array([[1, 1, 1]], np.uint8), cam, img, 1.0)
    check("u = 319.9 belongs to pixel 319", np.argwhere(mask)[0][1] == 319,
          f"{np.argwhere(mask)[0]}")

    # Behind the camera must not be drawn at all.
    behind = c2w[:3, 3] + c2w[:3, :3] @ np.array([0.0, 0.0, -2.0])
    _, _, mask = ev.splat(behind[None, :], np.array([[1, 2, 3]], np.uint8), cam, img, 1.0)
    check("a point behind the camera is not drawn", not mask.any())

    # The nearer of two points on the same ray owns the pixel.
    near = c2w[:3, 3] + c2w[:3, :3] @ (offset * np.array([0.5, 0.5, 0.5]))
    colour, depth, _ = ev.splat(np.stack([world, near]),
                                np.array([[10, 10, 10], [250, 250, 250]], np.uint8),
                                cam, img, 1.0)
    check("the nearer point wins the pixel",
          abs(depth[240, 320] - 1.0) < 1e-6 and colour[240, 320, 0] == 250,
          f"{depth[240, 320]} {colour[240, 320]}")


def test_splat_applies_opencv_distortion() -> None:
    print("the control splat applies the frame's OpenCV distortion")
    p = {"tx": 0.0, "ty": 0.0, "tz": 0.0, "qx": 0.0, "qy": 0.0,
         "qz": 0.0, "qw": 1.0, "fx": 100.0, "fy": 100.0,
         "cx": 100.0, "cy": 100.0}
    (qw, qx, qy, qz), t = ex.world_to_camera(p)
    img = {"qw": qw, "qx": qx, "qy": qy, "qz": qz, "t": t,
           "camera": 1, "name": "distorted", "distortion":
           np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0])}
    cam = {"w": 200, "h": 200, "fx": 100.0, "fy": 100.0,
           "cx": 100.0, "cy": 100.0, "model": "OPENCV",
           "distortion": [0.0] * 6}
    c2w = ex.camera_to_world(p)
    # x=y=0.5 at z=1: r2=.5, radial=1.1, so the pixel is (155,155).
    world = c2w[:3, 3] + c2w[:3, :3] @ np.array([0.5, 0.5, 1.0])
    _, _, mask = ev.splat(world[None, :], np.array([[1, 2, 3]], np.uint8),
                          cam, img, 1.0)
    check("radial distortion moves a point outward",
          np.argwhere(mask).tolist() == [[155, 155]], f"{np.argwhere(mask)}")


def test_holdout_geometry_does_not_reach_the_cloud() -> None:
    """The leak this harness exists to avoid, checked end to end."""
    print("held-out frames do not contribute to the initial cloud")
    tools = os.path.dirname(os.path.abspath(__file__))
    with tempfile.TemporaryDirectory() as tmp:
        made = subprocess.run([sys.executable, os.path.join(tools, "make_test_session.py"), tmp],
                              capture_output=True, text=True)
        if made.returncode != 0:
            check("fixture generated", False, made.stderr.strip()[:200])
            return
        sessions = [os.path.join(tmp, d) for d in os.listdir(tmp)
                    if os.path.isdir(os.path.join(tmp, d))]
        if not sessions:
            check("fixture generated", False, "no session directory")
            return

        out_all = os.path.join(tmp, "all")
        out_half = os.path.join(tmp, "half")
        common = dict(pix_stride=4, image_mode="copy", with_depth=False,
                      require_exact_depth=False, max_points=0)
        full = ex.export(sessions[0], out_all, holdout_every=0, **common)[0]
        half = ex.export(sessions[0], out_half, holdout_every=2, **common)[0]

        check("the split reserved about half", abs(half["holdout"] * 2 - half["images"]) <= 2,
              f"{half['holdout']} of {half['images']}")
        # Every other frame withheld should remove roughly half the raw points.
        ratio = half["points_raw"] / max(full["points_raw"], 1)
        check("withholding half the frames removes about half the points",
              0.35 < ratio < 0.65, f"kept {ratio:.2f}")
        check("and the cloud is genuinely smaller",
              half["points_raw"] < full["points_raw"],
              f"{half['points_raw']} vs {full['points_raw']}")
        check("holdout.txt lists them", os.path.exists(os.path.join(out_half, "holdout.txt")))
        names = [ln.strip() for ln in open(os.path.join(out_half, "holdout.txt"))
                 if ln.strip() and not ln.startswith("#")]
        check("one name per held-out frame", len(names) == half["holdout"],
              f"{len(names)} vs {half['holdout']}")


def main() -> int:
    test_psnr_against_arithmetic()
    test_ssim_bounds_and_ordering()
    test_depth_error()
    test_depth_is_read_in_whichever_unit_the_export_declared()
    test_mixed_camera_resolutions_get_separate_mean_controls()
    test_scoring_a_render_does_not_collide_with_the_depth_truth()
    test_a_picture_of_depth_is_not_scored_as_depth()
    test_splat_puts_a_point_where_projection_says()
    test_splat_applies_opencv_distortion()
    test_holdout_geometry_does_not_reach_the_cloud()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} failed: {', '.join(FAILURES)}")
        return 1
    print("all passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
