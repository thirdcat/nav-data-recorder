#!/usr/bin/env python3
"""Known geometry in, the known answer demanded back.

The instrument in `eval/uw_selfcal.py` is a bundle adjustment with the camera
model as a free parameter, and the failure it is most exposed to is a per-frame
pose that quietly absorbs the very radial term being measured.  So the decisive
test here is not "does it run": it is a synthetic scene with a **known**
distortion and **wrong** poses, where the coefficient has to come back anyway.

    python3 eval/test_uw_selfcal.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "eval"), str(REPO / "tools")]

import cv2

import uw_selfcal as S
from fit_uw_distortion import (compose_expected, corner_displacement,
                               inject_distortion, rectify_points)


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

def test_round_trip():
    """`distort_points` must invert `rectify_points` to float tolerance."""
    rng = np.random.default_rng(0)
    w, h = 1920, 1080
    points = rng.uniform([0, 0], [w, h], size=(4000, 2))
    for a in ([0.0, 0.0], [0.05, -0.01], [-0.08, 0.02], [0.12, 0.0]):
        back = S.distort_points(rectify_points(points, w, h, a), w, h, a)
        err = np.abs(back - points).max()
        assert err < 1e-6, f"round trip off by {err:.2e} at {a}"
    print("  round trip: rectify -> distort returns the same pixel")


def test_warp_map_matches_injector():
    """One map, not two: the depth takes the same warp the image takes."""
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, size=(240, 320), dtype=np.uint8)
    for a in ([0.06], [-0.05]):
        mine = cv2.remap(img, *S.warp_map(320, 240, a), cv2.INTER_LANCZOS4,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        theirs, _ = inject_distortion(img, a)
        assert np.array_equal(mine, theirs), f"warp map disagrees at {a}"
    print("  warp map reproduces fit_uw_distortion.inject_distortion exactly")


def test_rodrigues():
    rng = np.random.default_rng(2)
    for _ in range(20):
        v = rng.normal(size=3) * 0.4
        mine = S.rodrigues(v[None])[0]
        theirs, _ = cv2.Rodrigues(v)
        assert np.abs(mine - theirs).max() < 1e-12
    print("  rodrigues agrees with cv2.Rodrigues")


# ---------------------------------------------------------------------------
# A synthetic scene the answer is known for
# ---------------------------------------------------------------------------

class FakeArm:
    """Everything `Problem` reads off an `Arm`, and nothing else."""

    def __init__(self, w, h, hfov_deg, poses):
        self.w, self.h = w, h
        self.native_w, self.native_h = w, h
        self.f = (w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        self.R_half = 0.5 * math.hypot(w, h)
        self.native_R_half = self.R_half
        self.rows = [{"pose": T} for T in poses]
        self.lens = "ultrawide"

    def cone_radius(self):
        return self.R_half * 0.567


def synthetic(truth=(0.030, -0.008), n_frames=14, n_points=2500,
              pose_noise=(0.0, 0.0), seed=3, cone=0.567):
    """A walk past a cloud, imaged through a lens with a known radial term.

    `cone` is the fraction of the half-diagonal inside which a point may be
    anchored, which is the constraint the real problem is built around: depth
    exists only in the middle and the model has to be fitted from where it does
    not.
    """
    rng = np.random.default_rng(seed)
    w, h, hfov = 1920, 1080, 106.2007
    poses = []
    for i in range(n_frames):
        u = i / max(n_frames - 1, 1)
        R = S.rodrigues(np.array([[0.10 * math.sin(3 * u), 0.25 * u, 0.05 * u]]))[0]
        t = np.array([0.55 * u, 0.06 * math.sin(4 * u), 0.30 * u])
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, t
        poses.append(T)
    poses = np.stack(poses)

    arm = FakeArm(w, h, hfov, poses)
    centre = np.array([w / 2.0, h / 2.0])

    # A cloud in front of the first camera, deep enough to give real parallax.
    P = np.stack([rng.uniform(-8, 8, n_points), rng.uniform(-5, 5, n_points),
                  rng.uniform(1.5, 7.0, n_points)], axis=1)

    obs_t, obs_f, obs_uv = [], [], []
    for j, T in enumerate(poses):
        X = (P - T[:3, 3]) @ T[:3, :3]
        ahead = X[:, 2] > 0.2
        pin = np.full((n_points, 2), np.nan)
        pin[ahead] = np.stack([arm.f * X[ahead, 0] / X[ahead, 2] + centre[0],
                               arm.f * X[ahead, 1] / X[ahead, 2] + centre[1]], axis=1)
        raw = S.distort_points(np.nan_to_num(pin), w, h, truth)
        inside = ahead & (raw[:, 0] > 1) & (raw[:, 0] < w - 2) \
            & (raw[:, 1] > 1) & (raw[:, 1] < h - 2)
        for k in np.flatnonzero(inside):
            obs_t.append(k)
            obs_f.append(j)
            obs_uv.append(raw[k])
    obs_t = np.array(obs_t, np.int64)
    obs_f = np.array(obs_f, np.int64)
    obs_uv = np.array(obs_uv, float)

    # Anchor each track at its smallest radius, but only if that radius is
    # inside the cone.  Depth is the true z in that camera — the LiDAR's job.
    radius = np.hypot(obs_uv[:, 0] - centre[0], obs_uv[:, 1] - centre[1])
    best = np.full(n_points, -1, np.int64)
    best_r = np.full(n_points, np.inf)
    for o in range(len(obs_t)):
        if radius[o] < best_r[obs_t[o]]:
            best_r[obs_t[o]] = radius[o]
            best[obs_t[o]] = o
    ok = (best >= 0) & (best_r < cone * arm.R_half)
    depth = np.zeros(n_points)
    for k in np.flatnonzero(ok):
        j = obs_f[best[k]]
        T = poses[j]
        depth[k] = ((P[k] - T[:3, 3]) @ T[:3, :3])[2]
    anchors = {"anchor_obs": best, "anchored": ok, "anchor_depth": depth,
               "radius": radius, "anchor_radius": best_r}

    # The trajectory handed to the fit is *wrong* by design.
    if any(pose_noise):
        noisy = poses.copy()
        for j in range(1, n_frames):
            dR = S.rodrigues(rng.normal(size=(1, 3)) * pose_noise[0])[0]
            noisy[j, :3, :3] = poses[j, :3, :3] @ dR
            noisy[j, :3, 3] = poses[j, :3, 3] + rng.normal(size=3) * pose_noise[1]
        arm.rows = [{"pose": T} for T in noisy]
    return arm, (obs_t, obs_f, obs_uv), anchors, truth


def _fit(arm, obs, anchors, **kw):
    problem = S.Problem(arm, *obs, anchors, order=2)
    seed = problem.zero()
    seed[-1] = S.align(problem)
    fit = S.solve(problem, fit_model=True, fit_pose=True, init=seed, **kw)
    return problem, fit


def test_recovers_a_known_lens_with_exact_poses():
    arm, obs, anchors, truth = synthetic()
    problem, fit = _fit(arm, obs, anchors)
    got = corner_displacement(arm.native_w, arm.native_h, fit["coefficients"])
    want = corner_displacement(arm.native_w, arm.native_h, truth)
    print(f"  exact poses: corner {got:+.2f} px against a true {want:+.2f} px "
          f"({problem.n_tracks} tracks, "
          f"{int(problem.active.sum())} carried observations)")
    assert abs(got - want) < 2.0, f"{got:+.2f} vs {want:+.2f}"


def test_recovers_a_known_lens_through_a_wrong_trajectory():
    """The one that matters: a free per-frame pose must not eat the lens."""
    arm, obs, anchors, truth = synthetic(pose_noise=(0.004, 0.010))
    problem, fit = _fit(arm, obs, anchors)
    got = corner_displacement(arm.native_w, arm.native_h, fit["coefficients"])
    want = corner_displacement(arm.native_w, arm.native_h, truth)
    print(f"  poses wrong by 0.23 deg and 1.0 cm: corner {got:+.2f} px "
          f"against a true {want:+.2f} px")
    assert abs(got - want) < 6.0, f"{got:+.2f} vs {want:+.2f}"


def test_returns_zero_for_a_rectilinear_lens():
    """The null the wide arm's control is the real-data version of."""
    arm, obs, anchors, _ = synthetic(truth=(0.0, 0.0), pose_noise=(0.004, 0.010))
    problem, fit = _fit(arm, obs, anchors)
    got = corner_displacement(arm.native_w, arm.native_h, fit["coefficients"])
    print(f"  a truly rectilinear lens returns {got:+.2f} px")
    assert abs(got) < 6.0, got


def test_shape_decomposition_is_a_signed_lower_bound():
    """What the azimuthal radial mean can and cannot be asked for.

    The pre-registration derives that a pose error contributes to the
    azimuthally averaged radial residual only at order rho^1, so `b1` takes the
    pose and `b3`/`b5` are the lens.  That derivation is about a residual field
    with *both* unmodelled, and it is right about the sign — but it is measured
    here to be a **lower bound**, for a reason the derivation does not cover:
    the anchor point is itself placed through the wrong (pinhole) ray, so part
    of the distortion has already cancelled before the residual is formed.

    So this reports the recovery fraction rather than asserting the shape is the
    estimator.  The estimator is the joint fit, and the injection gate is what
    validates it.
    """
    want = None
    for label, noise in (("exact poses", (0.0, 0.0)),
                         ("0.06 deg, 3 mm", (0.001, 0.003)),
                         ("0.23 deg, 10 mm", (0.004, 0.010))):
        arm, obs, anchors, truth = synthetic(pose_noise=noise)
        problem = S.Problem(arm, *obs, anchors, order=2)
        base = problem.zero()
        base[-1] = S.align(problem)
        shape = S.shape_decomposition(problem, base)
        want = corner_displacement(arm.native_w, arm.native_h, truth)
        got = shape["implied_corner_px_native"]
        print(f"  unmodelled residual, {label:<16}: implies {got:+6.1f} px of a "
              f"true {want:+.1f} ({100 * got / want:3.0f} %), "
              f"b1 {shape['radial_terms_px']['b1']:+.2f} px")
        assert np.sign(got) == np.sign(want), "wrong sign"
        assert 0.0 < got < abs(want) * 1.2, f"{got:+.1f} is not a lower bound"
        if noise == (0.0, 0.0):
            assert got > 0.4 * want, "the clean case should recover most of it"


def test_a_free_pose_absorbs_most_of_the_lens():
    """Why the joint fit, and not the residual shape, is the estimator.

    This is the number that says how much a per-frame SE(3) can hide.  It is
    measured rather than asserted small, because if it were 100 % the whole
    instrument would be unidentifiable and the injection gate would fail.
    """
    arm, obs, anchors, truth = synthetic()
    problem = S.Problem(arm, *obs, anchors, order=2)
    base = problem.zero()
    base[-1] = S.align(problem)
    pose_only = S.solve(problem, fit_model=False, fit_pose=True, init=base)
    left = S.shape_decomposition(problem, pose_only["x"])["implied_corner_px_native"]
    want = corner_displacement(arm.native_w, arm.native_h, truth)
    print(f"  a free per-frame pose leaves {left:+.1f} px in the residual shape "
          f"of a {want:+.1f} px lens ({100 * (1 - left / want):.0f} % absorbed)")
    assert 0.0 < left < want, f"{left:+.1f} is not a lower bound on {want:+.1f}"


def test_the_objective_cannot_see_a_rigid_gauge():
    """The zero mode that makes `--dump-poses` dangerous, demonstrated.

    Rigidly move every corrected pose and the residual must not change by a
    float — the anchor points are built from the poses, so they move with them.
    If this ever stopped holding, the gauge removal in `dump_poses` would be
    removing something the data actually constrains.
    """
    arm, obs, anchors, _ = synthetic()
    problem = S.Problem(arm, *obs, anchors, order=1)
    x = problem.zero()
    x[0] = 0.01
    rng = np.random.default_rng(11)
    x[problem.order:-1] = rng.normal(size=problem.n_params - problem.order - 1) * 0.01
    before = problem.residual(x)
    Rw, tw = problem.poses(*problem.unpack(x)[1:])

    # `xi[0] = 0` is the gauge fix, so the rigid move cannot be written into the
    # parameters — it has to be applied to the chain the parameters are relative
    # to.  `poses()` is affine in `T_chain`, so this is exactly `G` applied to
    # every corrected pose.
    G = np.eye(4)
    G[:3, :3] = S.rodrigues(np.array([[0.05, -0.03, 0.02]]))[0]
    G[:3, 3] = np.array([0.4, -0.2, 0.3])
    problem.T_chain = np.einsum("ij,njk->nik", G, problem.T_chain)
    after = problem.residual(x)
    Rw2, tw2 = problem.poses(*problem.unpack(x)[1:])

    moved_deg = _angle_between(Rw[0], Rw2[0])
    moved_m = float(np.linalg.norm(tw2[0] - tw[0]))
    assert moved_deg > 1.0 and moved_m > 0.1, "the test did not move the block"
    worst = float(np.abs(after - before).max())
    print(f"  a {moved_deg:.1f} deg / {moved_m * 100:.0f} cm rigid move of the "
          f"whole block changes the residual by {worst:.2e} px")
    assert worst < 1e-6, f"the rigid mode is not a zero mode: {worst:.2e}"


def _angle_between(A, B):
    tr = np.clip((np.trace(A.T @ B) - 1) / 2, -1, 1)
    return math.degrees(math.acos(tr))


def test_gauge_removal_recovers_a_known_deformation():
    """Inject a rigid move plus a known wobble; only the wobble may survive."""
    n = 12
    rng = np.random.default_rng(7)
    T_chain = np.tile(np.eye(4), (n, 1, 1))
    for j in range(n):
        T_chain[j, :3, :3] = S.rodrigues(np.array([[0.02 * j, 0.01 * j, 0.0]]))[0]
        T_chain[j, :3, 3] = [0.1 * j, 0.0, 0.0]

    wobble = np.zeros((n, 6))
    wobble[:, 0] = 0.004 * np.sin(np.arange(n))          # ~0.23 deg
    wobble[:, 4] = 0.010 * np.cos(np.arange(n))          # 1.0 cm
    T_true = T_chain.copy()
    for j in range(n):
        dT = np.eye(4)
        dT[:3, :3] = S.rodrigues(wobble[j:j + 1, :3])[0]
        dT[:3, 3] = wobble[j, 3:]
        T_true[j] = T_chain[j] @ dT

    G = np.eye(4)
    G[:3, :3] = S.rodrigues(np.array([[0.12, -0.07, 0.03]]))[0]   # ~8 deg
    G[:3, 3] = np.array([0.5, -0.3, 0.2])                          # 62 cm
    T_moved = np.einsum("ij,njk->nik", G, T_true)

    recovered = np.einsum("ij,njk->nik",
                          S.rigid_gauge(T_moved, T_chain), T_moved)
    err_R = max(_angle_between(T_true[j][:3, :3], recovered[j][:3, :3])
                for j in range(n))
    err_t = float(np.abs(recovered[:, :3, 3] - T_true[:, :3, 3]).max())
    print(f"  an 8 deg / 62 cm rigid move is removed to {err_R:.4f} deg and "
          f"{err_t * 1000:.3f} mm, leaving the 0.23 deg / 1.0 cm wobble")
    # `rigid_gauge` fixes the rotation from orientation averaging and then the
    # translation from the centroids, which is the exact minimiser of the
    # position term *given* that rotation but not the joint optimum.  The leak
    # is sub-millimetre on a 62 cm move — a twentieth of the 19.272 mm rig
    # baseline and a hundredth of the 34-39 mm the shutter offset already costs.
    assert err_R < 0.02 and err_t < 1e-3, (err_R, err_t)

    # And the wobble itself is still there, not flattened by the gauge fit.
    left = max(_angle_between(T_chain[j][:3, :3], recovered[j][:3, :3])
               for j in range(n))
    assert left > 0.15, f"the gauge removal ate the deformation: {left:.3f} deg"


def test_the_dump_refuses_what_it_actually_produced():
    """The bars, run on the numbers the real fit returned.

    The first version of the gate checked medians only. On the full 289-frame
    `d06152` block the median correction is 80 mm — inside the wide-arm bar —
    while the corrected path is 63.73 m against the chain's 24.18 m and one
    frame-to-frame step is 9.5 m. It wrote the file. These are those exact
    numbers, kept so that hole cannot reopen.
    """
    full = {"trajectory_scale": 0.9980073649730202,
            "deformation_deg_median": 2.8871396783710894,
            "deformation_m_median": 0.07985358178574699,
            "path_m_chain": 24.181886333024014,
            "path_m_corrected": 63.734298869594596,
            "step_m_chain_median": 0.0834437800687575,
            "step_m_corrected_max": 9.53475900911697}
    for wide in (False, True):
        why = S.dump_refusals(full, wide, worst_join_m=0.023)
        assert any("path" in w for w in why), (wide, why)
        assert any("step" in w for w in why), (wide, why)
    print(f"  the full session's own numbers are refused for both arms: "
          f"{len(S.dump_refusals(full, True, 0.023))} reasons on the wide dump")

    # And a trajectory that really is a small refinement must pass, or the bars
    # are just "refuse everything".
    good = dict(full, deformation_deg_median=0.4, deformation_m_median=0.012,
                path_m_corrected=24.9, step_m_corrected_max=0.30,
                trajectory_scale=1.0009)
    assert S.dump_refusals(good, False, 0.023) == []
    assert S.dump_refusals(good, True, 0.023) == []
    print("  a 0.4 deg / 12 mm refinement of the same walk passes both")


def test_injection_composes():
    """Recovery is judged against the exact composition, not the sum."""
    native = [0.03, -0.008]
    injected = [-0.02, 0.0]
    composed = compose_expected(native, injected, 2)
    rho = np.linspace(0, 1, 64)
    direct = rho * (1 + composed[0] * rho ** 2 + composed[1] * rho ** 4)
    inter = rho * (1 + injected[0] * rho ** 2)
    truth = inter * (1 + native[0] * inter ** 2 + native[1] * inter ** 4)
    assert np.abs(direct - truth).max() < 2e-4
    print("  compose_expected reproduces the composed radial map")


def test_profile_spread_parser():
    text = ("CONTROL: an ultra-wide frame against itself downscaled\n"
            "    r    0- 300 px   s 0.3201  (400 inliers)\n"
            "    r  300- 600 px   s 0.3200  (300 inliers)\n"
            "    r  600- 900 px   -- too few matches\n"
            "QUESTION: the two real lenses.\n"
            "  d06152: 40 pairs\n"
            "    r    0- 300 px   s 0.3134   (40 frames, 900 inliers)\n"
            "    r  300- 600 px   s 0.3171   (40 frames, 800 inliers)\n"
            "    r  600- 900 px   s 0.3182   (39 frames, 700 inliers)\n"
            "    r  900-1250 px   s 0.3170   (38 frames, 500 inliers)\n"
            "    spread across annuli: +1.15 %  (0-300 px -> 900-1250 px)\n")
    got = S.profile_spread(text)
    assert got["control"]["values"] == [0.3201, 0.3200]
    assert got["lenses"]["values"] == [0.3134, 0.3171, 0.3182, 0.3170]
    assert abs(got["lenses"]["spread_pct"] - 1.5316) < 0.01, got
    print("  profile parser reads the annulus table and its peak-to-trough")


def main() -> int:
    print("uw_selfcal self-test")
    for fn in (test_round_trip, test_warp_map_matches_injector, test_rodrigues,
               test_gauge_removal_recovers_a_known_deformation,
               test_the_objective_cannot_see_a_rigid_gauge,
               test_the_dump_refuses_what_it_actually_produced,
               test_injection_composes, test_profile_spread_parser,
               test_recovers_a_known_lens_with_exact_poses,
               test_recovers_a_known_lens_through_a_wrong_trajectory,
               test_returns_zero_for_a_rectilinear_lens,
               test_shape_decomposition_is_a_signed_lower_bound,
               test_a_free_pose_absorbs_most_of_the_lens):
        fn()
    print("all passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
