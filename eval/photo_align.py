#!/usr/bin/env python3
"""Measure a cross-session alignment with photographs instead of geometry.

`docs/3DGS.md` records why this exists. Point-to-plane ICP pins the vertical to
about a centimetre and leaves the horizontal free to three and beyond, because
the floor supplies most of the points and constrains only its own normal. Three
horizontal centimetres at 2 m is ten pixels of texture displacement, so the
direction geometry cannot see is the direction a renderer cares about most.

The probe: take a frame from the source session, unproject its depth, move it
through the candidate transform, project into an overlapping frame of the
reference session, and compare what lands with what that camera actually
photographed. Sweep the transform and watch the score. A sharp minimum away
from zero is a measurement of the misalignment ICP could not feel; a flat curve
means the images cannot see it either.

Scoring is zero-mean normalised cross-correlation on greyscale, which is
invariant to the exposure and white-balance differences between two walks —
those are large here and would otherwise dominate any absolute difference.

    python3 eval/photo_align.py A_SESSION B_SESSION --transform pair.json
    python3 eval/photo_align.py A_SESSION B_SESSION --sweep 0.05

No GPU. Reads sessions and, if given, the JSON `align_set.py` writes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "tools"))
import align_sessions as al  # noqa: E402
from export_3dgs import camera_to_world  # noqa: E402
from read_session import Session  # noqa: E402

DEPTH_NEAR_M = 0.2
DEPTH_FAR_M = 5.0


def grey(path: str, width: int, height: int) -> np.ndarray:
    """Luminance at the depth map's resolution, in [0, 1].

    Sampled down to the depth grid rather than the depth up to the image: the
    probe compares what one camera saw against what another saw at the same
    surface point, and the surface points only exist where there is depth.
    """
    img = Image.open(path).convert("L").resize((width, height), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def intrinsics(row: dict[str, Any], w: int, h: int) -> tuple[float, float, float, float]:
    pose = row["pose"]
    sx, sy = w / row["width"], h / row["height"]
    return pose["fx"] * sx, pose["fy"] * sy, pose["cx"] * sx, pose["cy"] * sy


def frame_bundle(session: Session, row: dict[str, Any]) -> dict[str, Any] | None:
    """Everything one frame contributes: depth, luminance, pose, intrinsics."""
    entry = row.get("depth")
    if entry is None or row["pose"].get("tracking") != "normal":
        return None
    z = np.asarray(session.depth_frame(entry), dtype=np.float32)
    conf = session.confidence_frame(entry)
    h, w = z.shape
    path = session.frame_path(row)
    if not os.path.exists(path):
        return None
    ok = np.isfinite(z) & (z > DEPTH_NEAR_M) & (z < DEPTH_FAR_M)
    if conf is not None:
        ok &= np.asarray(conf) >= 2
    if ok.sum() < 500:
        return None
    return {"z": z, "ok": ok, "grey": grey(path, w, h),
            "pose": camera_to_world(row["pose"]),
            "K": intrinsics(row, w, h), "shape": (h, w)}


def sample(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear sample, plus the mask of samples that landed inside."""
    h, w = image.shape
    inside = (u >= 0) & (u <= w - 1.001) & (v >= 0) & (v <= h - 1.001)
    uu = np.clip(u, 0, w - 1.001)
    vv = np.clip(v, 0, h - 1.001)
    u0, v0 = np.floor(uu).astype(np.int64), np.floor(vv).astype(np.int64)
    du, dv = uu - u0, vv - v0
    out = (image[v0, u0] * (1 - du) * (1 - dv) + image[v0, u0 + 1] * du * (1 - dv)
           + image[v0 + 1, u0] * (1 - du) * dv + image[v0 + 1, u0 + 1] * du * dv)
    return out, inside


def reproject_score(ref: dict[str, Any], src: dict[str, Any],
                    transform: np.ndarray) -> tuple[float, int]:
    """Correlate the source frame's pixels against the reference photograph.

    Returns 1 - NCC so that lower is better and the number reads like an error,
    and the count of pixels the comparison rested on.
    """
    h, w = src["shape"]
    fx, fy, cx, cy = src["K"]
    vv, uu = np.mgrid[0:h, 0:w]
    ok = src["ok"]
    z = src["z"][ok]
    cam = np.stack([(uu[ok] - cx) * z / fx, (vv[ok] - cy) * z / fy, z], axis=1)

    world = cam @ src["pose"][:3, :3].T + src["pose"][:3, 3]
    world = world @ transform[:3, :3].T + transform[:3, 3]

    inv = np.linalg.inv(ref["pose"])
    local = world @ inv[:3, :3].T + inv[:3, 3]
    front = local[:, 2] > DEPTH_NEAR_M
    if front.sum() < 200:
        return float("nan"), 0

    rfx, rfy, rcx, rcy = ref["K"]
    u = local[front, 0] * rfx / local[front, 2] + rcx
    v = local[front, 1] * rfy / local[front, 2] + rcy
    got, inside = sample(ref["grey"], u, v)

    # Occlusion: only trust a sample where the reference's own depth agrees that
    # this surface is the one it can see. Without this the score rewards a
    # transform that hides the source behind the reference's furniture.
    zref, _ = sample(ref["z"].astype(np.float32), u, v)
    agree = np.abs(zref - local[front, 2]) < 0.10

    keep = inside & agree & np.isfinite(got)
    if keep.sum() < 200:
        return float("nan"), int(keep.sum())

    a = src["grey"][ok][front][keep]
    b = got[keep]
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denom < 1e-9:
        return float("nan"), int(keep.sum())
    return 1.0 - float((a * b).sum() / denom), int(keep.sum())


def overlapping_pairs(ref_rows, src_rows, transform, *, max_centre_m=1.2,
                      max_angle_deg=35.0, limit=24, min_index_gap=0
                      ) -> list[tuple[int, int]]:
    """Frames that plausibly see the same surface, under the candidate transform.

    Position alone is not enough — two cameras a metre apart facing opposite
    walls share nothing — so the viewing direction has to agree as well.
    """
    ref_c = np.array([r["pose"][:3, 3] for r in ref_rows])
    ref_d = np.array([r["pose"][:3, 2] for r in ref_rows])
    src_c = np.array([r["pose"][:3, 3] for r in src_rows]) @ transform[:3, :3].T + transform[:3, 3]
    src_d = np.array([r["pose"][:3, 2] for r in src_rows]) @ transform[:3, :3].T

    cos_max = np.cos(np.radians(max_angle_deg))
    pairs = []
    for j in range(len(src_rows)):
        d = np.linalg.norm(ref_c - src_c[j], axis=1)
        aim = ref_d @ src_d[j]
        good = np.flatnonzero((d < max_centre_m) & (aim > cos_max))
        if min_index_gap:
            # For the self-check: pairing a frame with itself is an identity and
            # would report a perfect zero however biased the probe is.
            good = good[np.abs(good - j) >= min_index_gap]
        if len(good):
            pairs.append((int(good[np.argmin(d[good])]), j, float(d[good].min())))
    pairs.sort(key=lambda p: p[2])
    # One reference frame may be nearest to many source frames; keep it varied.
    seen, out = set(), []
    for i, j, _ in pairs:
        if i in seen:
            continue
        seen.add(i)
        out.append((i, j))
        if len(out) >= limit:
            break
    return out


def sweep(ref_bundles, src_bundles, pairs, transform, axis, offsets) -> list[dict]:
    rows = []
    for offset in offsets:
        moved = transform.copy()
        moved[:3, 3] = transform[:3, 3] + axis * offset
        scores, pixels = [], 0
        for i, j in pairs:
            s, n = reproject_score(ref_bundles[i], src_bundles[j], moved)
            if np.isfinite(s):
                scores.append(s)
                pixels += n
        rows.append({"offset_m": float(offset),
                     "score": float(np.median(scores)) if scores else float("nan"),
                     "pairs": len(scores), "pixels": pixels})
    return rows


def parabola_min(offsets: np.ndarray, scores: np.ndarray) -> float:
    """Sub-sample minimum from the three points around the best sample."""
    k = int(np.nanargmin(scores))
    if k == 0 or k == len(scores) - 1:
        return float(offsets[k])
    y0, y1, y2 = scores[k - 1], scores[k], scores[k + 1]
    denom = y0 - 2 * y1 + y2
    if abs(denom) < 1e-12:
        return float(offsets[k])
    step = offsets[1] - offsets[0]
    return float(offsets[k] + 0.5 * step * (y0 - y2) / denom)


def refine(ref_bundles, src_bundles, pairs, transform, *, passes=4,
           span=0.04, steps=9, yaw_span_deg=1.5) -> tuple[np.ndarray, list[dict]]:
    """Coordinate descent on the photometric score.

    The three axis sweeps are each run with the other two held at zero, so their
    minima do not compose into a joint optimum — the surface is not separable.
    Descending one axis at a time and repeating is enough here because the
    starting point is already within a few centimetres, and it keeps the whole
    thing a sweep of a scalar rather than a solver with its own failure modes.

    Yaw is included because a rotation about the vertical is the other thing a
    floor-dominated ICP cannot pin, and a degree of yaw is centimetres across a
    room. Pitch and roll are left alone: gravity constrains them and the
    vertical sweep above already lands on zero.
    """
    current = transform.copy()
    history = []

    def score_of(T):
        got = [reproject_score(ref_bundles[i], src_bundles[j], T)[0] for i, j in pairs]
        got = [g for g in got if np.isfinite(g)]
        return float(np.median(got)) if got else float("nan")

    axes = (("X", np.array([1.0, 0, 0])), ("Z", np.array([0, 0, 1.0])),
            ("Y", np.array([0, 1.0, 0])))
    for step in range(passes):
        width = span / (2 ** step)
        for name, axis in axes:
            offsets = np.linspace(-width, width, steps)
            scores = []
            for offset in offsets:
                T = current.copy()
                T[:3, 3] = current[:3, 3] + axis * offset
                scores.append(score_of(T))
            best = parabola_min(offsets, np.array(scores))
            best = float(np.clip(best, -width, width))
            current[:3, 3] = current[:3, 3] + axis * best
            history.append({"pass": step, "axis": name, "moved_m": best,
                            "score": score_of(current)})
        yaw_width = np.radians(yaw_span_deg) / (2 ** step)
        angles = np.linspace(-yaw_width, yaw_width, steps)
        scores = []
        for angle in angles:
            T = current.copy()
            R = al.yaw_matrix(angle)
            pivot = current[:3, 3]
            T[:3, :3] = R @ current[:3, :3]
            T[:3, 3] = pivot
            scores.append(score_of(T))
        best = float(np.clip(parabola_min(angles, np.array(scores)), -yaw_width, yaw_width))
        pivot = current[:3, 3].copy()
        current[:3, :3] = al.yaw_matrix(best) @ current[:3, :3]
        current[:3, 3] = pivot
        history.append({"pass": step, "axis": "yaw", "moved_deg": float(np.degrees(best)),
                        "score": score_of(current)})
    return current, history


def load_transform(path: str | None, ref_id: str, src_id: str) -> np.ndarray | None:
    if not path:
        return None
    data = json.loads(Path(path).read_text())
    for key in ("transforms", "sessions", "placements"):
        table = data.get(key)
        if isinstance(table, dict):
            for name, value in table.items():
                if src_id in name:
                    return np.asarray(value if not isinstance(value, dict)
                                      else value["transform"], dtype=float)
    return None


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("reference")
    ap.add_argument("source")
    ap.add_argument("--transform", default=None,
                    help="JSON from align_set.py; recomputed if absent")
    ap.add_argument("--sweep", type=float, default=0.05,
                    help="half-width of the offset sweep, metres")
    ap.add_argument("--steps", type=int, default=11)
    ap.add_argument("--pairs", type=int, default=24)
    ap.add_argument("--voxel", type=float, default=0.01)
    ap.add_argument("--refine", action="store_true",
                    help="descend the photometric score and write the result")
    ap.add_argument("--refine-span", type=float, default=0.04,
                    help="half-width of the first coordinate-descent bracket, "
                         "metres; the descent halves it each pass, so a run that "
                         "moves by the total available span reported a bound "
                         "rather than a minimum")
    ap.add_argument("--refine-passes", type=int, default=4)
    ap.add_argument("--out", default=None, help="write the refined transform here")
    ap.add_argument("--min-gap", type=int, default=0,
                    help="require paired frames to be at least this far apart "
                         "in the walk; with --self-check this measures drift "
                         "between two visits rather than agreement between "
                         "neighbours")
    ap.add_argument("--self-check", action="store_true",
                    help="score a session against itself under the identity, "
                         "pairing only frames far apart in the walk: the true "
                         "minimum is zero on every axis by construction")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    ref_session, src_session = Session(a.reference), Session(a.source)
    T = load_transform(a.transform, ref_session.id, src_session.id)
    if a.self_check:
        T = np.eye(4)
    elif T is None:
        print("registering geometrically first ...", file=sys.stderr)
        ca = al.session_cloud(a.reference, voxel=a.voxel, pix_stride=2, conf_min=2)
        cb = al.session_cloud(a.source, voxel=a.voxel, pix_stride=2, conf_min=2)
        T = np.asarray(al.icp(ca, cb, np.asarray(al.align_clouds(ca, cb)["transform"]),
                              voxel=max(a.voxel * 1.5, 0.015), iterations=40)["transform"])

    ref_rows = [r for r in ref_session.posed_images() if r.get("depth") is not None]
    src_rows = [r for r in src_session.posed_images() if r.get("depth") is not None]
    lite = lambda rows: [{"pose": camera_to_world(r["pose"])} for r in rows]
    pairs = overlapping_pairs(lite(ref_rows), lite(src_rows), T, limit=a.pairs,
                              min_index_gap=a.min_gap or (8 if a.self_check else 0))
    if not pairs:
        print("no overlapping frames under this transform")
        return 1

    ref_bundles, src_bundles = {}, {}
    for i, j in pairs:
        if i not in ref_bundles:
            ref_bundles[i] = frame_bundle(ref_session, ref_rows[i])
        if j not in src_bundles:
            src_bundles[j] = frame_bundle(src_session, src_rows[j])
    pairs = [(i, j) for i, j in pairs
             if ref_bundles.get(i) is not None and src_bundles.get(j) is not None]
    print(f"{len(pairs)} overlapping frame pairs", file=sys.stderr)

    if a.refine:
        # Filter before taking the median: one pair that finds too few pixels
        # returns nan, and np.median propagates it, which reported the starting
        # score of a perfectly measurable pair as "nan".
        start = [reproject_score(ref_bundles[i], src_bundles[j], T)[0]
                 for i, j in pairs]
        start = [v for v in start if np.isfinite(v)]
        before = float(np.median(start)) if start else float("nan")
        refined, history = refine(ref_bundles, src_bundles, pairs, T,
                                  span=a.refine_span, passes=a.refine_passes)
        reach = a.refine_span * (2 - 2.0 ** (1 - a.refine_passes))
        print(f"\n{'pass':>5} {'axis':>5} {'moved':>10} {'score':>9}")
        for h in history:
            moved = (f"{h['moved_m'] * 100:+.2f} cm" if "moved_m" in h
                     else f"{h['moved_deg']:+.3f} deg")
            print(f"{h['pass']:>5} {h['axis']:>5} {moved:>10} {h['score']:>9.4f}")
        shift = refined[:3, 3] - T[:3, 3]
        print(f"\n  photometric score {before:.4f} -> {history[-1]['score']:.4f}")
        print(f"  translation moved {np.linalg.norm(shift) * 100:.2f} cm "
              f"(X {shift[0] * 100:+.2f}, Y {shift[1] * 100:+.2f}, Z {shift[2] * 100:+.2f})")
        # A descent that spends its whole budget on one axis did not find a
        # minimum, it ran out of room, and the number is a lower bound.
        pinned = [n for n, v in zip("XYZ", shift) if abs(v) > 0.9 * reach]
        if pinned:
            print(f"  ** ran to the edge of the search on {', '.join(pinned)} "
                  f"(reach {reach*100:.1f} cm): this is a bound, not a minimum. "
                  f"Re-run with a wider --refine-span.")
        if a.out:
            Path(a.out).write_text(json.dumps(
                {"reference": ref_session.id, "source": src_session.id,
                 "geometric": T.tolist(), "refined": refined.tolist(),
                 "score_before": float(before), "score_after": history[-1]["score"]},
                indent=2))
            print(f"  wrote {a.out}")
        return 0

    offsets = np.linspace(-a.sweep, a.sweep, a.steps)
    axes = {"X (horizontal)": np.array([1.0, 0, 0]),
            "Z (horizontal)": np.array([0, 0, 1.0]),
            "Y (vertical)": np.array([0, 1.0, 0])}
    result = {"reference": ref_session.id, "source": src_session.id, "axes": {}}
    for name, axis in axes.items():
        rows = sweep(ref_bundles, src_bundles, pairs, T, axis, offsets)
        best = parabola_min(offsets, np.array([r["score"] for r in rows]))
        result["axes"][name] = {"sweep": rows, "argmin_m": best}

    if a.json:
        print(json.dumps(result, indent=2))
        return 0

    print("\n1 - NCC against the reference photographs, median over frame pairs")
    print("lower is better; the offset is applied to the geometric transform\n")
    print(f"{'axis':>18} " + "".join(f"{o * 100:>+7.0f}cm" for o in offsets))
    for name, entry in result["axes"].items():
        cells = "".join(f"{r['score']:>9.4f}" if np.isfinite(r["score"]) else f"{'-':>9}"
                        for r in entry["sweep"])
        print(f"{name:>18} {cells}")
    print()
    for name, entry in result["axes"].items():
        print(f"  {name:>18}: minimum at {entry['argmin_m'] * 100:+.1f} cm")
    print("\n  a minimum away from zero is misalignment the geometry could not feel.")
    print("  a flat row means the photographs cannot see that direction either.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
