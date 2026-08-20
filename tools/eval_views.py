#!/usr/bin/env python3
"""Score held-out views of an exported dataset, against controls that can win.

A splat trained on a walk will render its training views beautifully. The
question is what it does at a viewpoint it was not given, and the only way that
question has an answer is if something else is scored the same way on the same
frames. So this reports three columns, not one:

    mean      the average training image. The floor. A result that does not
              clear this has not learned the scene, it has learned its exposure.
    lidar     the exported point cloud, splatted through the held-out camera.
              This is what the raw measurements alone already predict, before
              any optimisation. A trained model that does not beat it has spent
              a GPU-hour to reproduce its own input.
    render    whatever a trainer wrote out, if `--render-dir` is given.

    python3 tools/eval_views.py /tmp/gs
    python3 tools/eval_views.py /tmp/gs --render-dir runs/splat/renders

Depth is reported separately from colour and matters more here. Depth
supervision is known to move rendered-depth error several-fold while moving
PSNR by a few tenths of a decibel, so a comparison scored on PSNR alone would
report "no difference" for the change that mattered most.

Note that the exporter keeps held-out frames out of the initial point cloud, so
the `lidar` column is a genuine prediction at those views rather than a lookup.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_3dgs import quat_to_matrix  # noqa: E402

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None


def read_cameras(model: str) -> dict[int, dict[str, Any]]:
    cams = {}
    with open(os.path.join(model, "sparse", "0", "cameras.txt")) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            cams[int(p[0])] = {"w": int(p[2]), "h": int(p[3]), "fx": float(p[4]),
                               "fy": float(p[5]), "cx": float(p[6]), "cy": float(p[7])}
    return cams


def read_images(model: str) -> list[dict[str, Any]]:
    out = []
    with open(os.path.join(model, "sparse", "0", "images.txt")) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            if len(p) < 10:
                continue
            out.append({"qw": float(p[1]), "qx": float(p[2]), "qy": float(p[3]),
                        "qz": float(p[4]), "t": np.array([float(v) for v in p[5:8]]),
                        "camera": int(p[8]), "name": p[9]})
    return out


def read_ply(path: str) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as fh:
        blob = fh.read()
    head, body = blob.split(b"end_header\n", 1)
    count = 0
    for line in head.decode("ascii", "replace").splitlines():
        if line.startswith("element vertex"):
            count = int(line.split()[-1])
    rec = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                    ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
                    ("r", "u1"), ("g", "u1"), ("b", "u1")])
    a = np.frombuffer(body[:count * rec.itemsize], dtype=rec)
    xyz = np.stack([a["x"], a["y"], a["z"]], axis=1).astype(np.float64)
    rgb = np.stack([a["r"], a["g"], a["b"]], axis=1)
    return xyz, rgb


def splat(xyz: np.ndarray, rgb: np.ndarray, cam: dict[str, Any],
          img: dict[str, Any], scale: float
          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest-point-per-pixel projection. Returns colour, depth and coverage.

    Pixel index `i` covers the continuous range `[i, i+1)` and has its centre at
    `i + 0.5` — COLMAP's convention, so a projected coordinate belongs to the
    pixel it floors to. A point landing exactly on an integer sits on a boundary
    and which side it falls is decided by the last bit of the arithmetic; that
    is unavoidable and does not matter, because a half-pixel is far below what
    any of these scores resolve.
    """
    w, h = max(1, int(cam["w"] * scale)), max(1, int(cam["h"] * scale))
    R = quat_to_matrix(img["qx"], img["qy"], img["qz"], img["qw"])
    p = xyz @ R.T + img["t"]
    ahead = p[:, 2] > 0.05
    p, c = p[ahead], rgb[ahead]
    if not len(p):
        return (np.zeros((h, w, 3), np.uint8), np.zeros((h, w)), np.zeros((h, w), bool))

    u = (cam["fx"] * scale) * p[:, 0] / p[:, 2] + cam["cx"] * scale
    v = (cam["fy"] * scale) * p[:, 1] / p[:, 2] + cam["cy"] * scale
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, c, z = (u[inside].astype(np.int64), v[inside].astype(np.int64),
                  c[inside], p[inside, 2])

    # Painter's algorithm, far to near, so the nearest surface owns the pixel.
    order = np.argsort(-z)
    colour = np.zeros((h, w, 3), np.uint8)
    depth = np.zeros((h, w))
    mask = np.zeros((h, w), bool)
    colour[v[order], u[order]] = c[order]
    depth[v[order], u[order]] = z[order]
    mask[v[order], u[order]] = True
    return colour, depth, mask


def _blur(a: np.ndarray, radius: int = 5, sigma: float = 1.5) -> np.ndarray:
    k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(a, ((radius, radius), (radius, radius)), mode="reflect")
    out = np.apply_along_axis(lambda m: np.convolve(m, k, "valid"), 1, pad)
    return np.apply_along_axis(lambda m: np.convolve(m, k, "valid"), 0, out)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse <= 1e-12 else float(10.0 * math.log10(255.0 ** 2 / mse))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Gaussian-window SSIM on luminance, the usual 11x11 sigma 1.5."""
    x = a.astype(np.float64).mean(axis=2) if a.ndim == 3 else a.astype(np.float64)
    y = b.astype(np.float64).mean(axis=2) if b.ndim == 3 else b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mx, my = _blur(x), _blur(y)
    sxx, syy, sxy = _blur(x * x) - mx * mx, _blur(y * y) - my * my, _blur(x * y) - mx * my
    num = (2 * mx * my + c1) * (2 * sxy + c2)
    den = (mx * mx + my * my + c1) * (sxx + syy + c2)
    return float(np.mean(num / np.maximum(den, 1e-12)))


def depth_error(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray
                ) -> dict[str, float] | None:
    """Median absolute and relative depth error, both sides already in metres."""
    both = mask & (truth > 0.05) & (pred > 0.05)
    if both.sum() < 100:
        return None
    err = np.abs(pred[both] - truth[both])
    return {
        "median_abs_m": round(float(np.median(err)), 4),
        "abs_rel": round(float(np.median(err / truth[both])), 4),
        "coverage": round(float(both.mean()), 4),
    }


def load(path: str, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB").resize(size, Image.BILINEAR))


def read_rendered_depth(directory: str, image_name: str, size: tuple[int, int],
                        unit_scale: float) -> tuple[np.ndarray | None, str]:
    """A trainer's rendered depth in metres, or nothing and the reason why.

    Renderers write two different things under the name "depth". One is a
    measurement. The other is a picture of a measurement — Nerfstudio's
    `ns-render` colourises through a turbo map by default, and the result is a
    three-channel 8-bit PNG whose values are hues. Read that as millimetres and
    a plausible-looking metre-scale error comes out; the first time this ran it
    reported 1.332 m against a true 0.036 m and nothing about the number said it
    was nonsense.

    So the shape and dtype are checked rather than assumed, and an
    uninterpretable input produces no score at all.
    """
    stem = os.path.splitext(image_name)[0]
    for suffix in (".npy", ".png", ".tiff", ".tif"):
        path = os.path.join(directory, stem + suffix)
        if not os.path.exists(path):
            continue
        if suffix == ".npy":
            raw = np.load(path).astype(np.float64)
        else:
            with Image.open(path) as dm:
                if dm.mode in ("RGB", "RGBA", "P", "L"):
                    return None, (f"{dm.mode} — a colourised picture of depth, "
                                  "not depth; re-render raw")
                raw = np.asarray(dm).astype(np.float64)
        if raw.ndim != 2:
            return None, f"{raw.ndim}-dimensional, not a depth map"
        h, w = raw.shape
        ys = np.clip((np.arange(size[1]) + 0.5) * h / size[1], 0, h - 1).astype(np.int64)
        xs = np.clip((np.arange(size[0]) + 0.5) * w / size[0], 0, w - 1).astype(np.int64)
        return raw[np.ix_(ys, xs)] * unit_scale, ""
    return None, "no depth file found for this frame"


def depth_index(model: str) -> tuple[dict[str, str], float]:
    """Where each image's depth lives, read from the export rather than guessed.

    The exporter writes millimetre PNGs or metre `.npy` depending on which
    trainer the dataset is for, and records both the path and the unit in
    `transforms.json`. Reading that is the difference between this harness
    working on either layout and silently scoring nothing on one of them.
    """
    path = os.path.join(model, "transforms.json")
    if not os.path.exists(path):
        return {}, 0.001
    doc = json.load(open(path))
    index = {}
    for frame in doc.get("frames", []):
        depth = frame.get("depth_file_path")
        if depth:
            index[os.path.basename(frame["file_path"])] = os.path.join(model, depth)
    return index, float(doc.get("depth_unit_scale_factor", 0.001))


def load_depth_metres(path: str, size: tuple[int, int], unit_scale: float) -> np.ndarray:
    if path.endswith(".npy"):
        raw = np.load(path).astype(np.float64)
        h, w = raw.shape
        ys = np.clip((np.arange(size[1]) + 0.5) * h / size[1], 0, h - 1).astype(np.int64)
        xs = np.clip((np.arange(size[0]) + 0.5) * w / size[0], 0, w - 1).astype(np.int64)
        return raw[np.ix_(ys, xs)] * unit_scale
    with Image.open(path) as im:
        return np.asarray(im.resize(size, Image.NEAREST)).astype(np.float64) * unit_scale


def evaluate(model: str, *, scale: float = 0.25, render_dir: str | None = None,
             render_depth_dir: str | None = None,
             render_depth_scale: float = 0.001) -> dict[str, Any]:
    if Image is None:
        raise SystemExit("Pillow is required: pip install pillow")

    held_path = os.path.join(model, "holdout.txt")
    if not os.path.exists(held_path):
        raise SystemExit(f"{held_path} not found — export with --holdout-every")
    held = {ln.strip() for ln in open(held_path)
            if ln.strip() and not ln.startswith("#")}

    cams, images = read_cameras(model), read_images(model)
    xyz, rgb = read_ply(os.path.join(model, "sparse_pc.ply"))
    depths, unit_scale = depth_index(model)

    # The floor: the average training frame. Built from training views only,
    # for the same reason the point cloud is.
    train = [im for im in images if im["name"] not in held]
    if not train or not held:
        raise SystemExit("need both training and held-out frames")
    first = cams[train[0]["camera"]]
    size = (max(1, int(first["w"] * scale)), max(1, int(first["h"] * scale)))
    mean_image = np.zeros((size[1], size[0], 3), np.float64)
    for im in train:
        mean_image += load(os.path.join(model, "images", im["name"]), size)
    mean_image = (mean_image / len(train)).astype(np.uint8)

    rows: list[dict[str, Any]] = []
    rejected: set[str] = set()
    for im in images:
        if im["name"] not in held:
            continue
        cam = cams[im["camera"]]
        size = (max(1, int(cam["w"] * scale)), max(1, int(cam["h"] * scale)))
        truth = load(os.path.join(model, "images", im["name"]), size)

        entry: dict[str, Any] = {"name": im["name"]}
        entry["mean"] = {"psnr": round(psnr(mean_image, truth), 2),
                         "ssim": round(ssim(mean_image, truth), 4)}

        colour, depth, mask = splat(xyz, rgb, cam, im, scale)
        # Score colour only where the cloud actually put something, otherwise
        # this measures how much of the frame is empty, not how right it is.
        # The floor is then re-scored on that same subset, because 16 dB over a
        # quarter of the frame and 13 dB over all of it are not comparable
        # numbers, and the only reason to have a control is comparability.
        entry["lidar"] = {
            "psnr": round(psnr(colour[mask], truth[mask]), 2) if mask.any() else None,
            "coverage": round(float(mask.mean()), 4),
        }
        if mask.any():
            entry["mean"]["psnr_on_lidar_mask"] = round(
                psnr(mean_image[mask], truth[mask]), 2)

        depth_path = depths.get(im["name"])
        if depth_path and os.path.exists(depth_path):
            truth_depth = load_depth_metres(depth_path, size, unit_scale)
            got = depth_error(depth, truth_depth, mask)
            if got:
                # `depth_error` reports its own `coverage` — the pixels where
                # the cloud and the LiDAR truth *both* have a value, which is
                # narrower than the cloud's own coverage set just above. Merging
                # it in unrenamed replaced one with the other under the same
                # key, so `lidar_coverage` printed the depth-comparison figure
                # while the number quoted in `docs/3DGS.md` was the cloud's. Two
                # quantities, one name, and no way to see which one you had.
                got = {("depth_coverage" if k == "coverage" else k): v
                       for k, v in got.items()}
                entry["lidar"].update(got)

            if render_depth_dir:
                pred_depth, why = read_rendered_depth(
                    render_depth_dir, im["name"], size, render_depth_scale)
                if pred_depth is None:
                    rejected.add(why)
                else:
                    got = depth_error(pred_depth, truth_depth,
                                      np.ones(size[::-1], bool))
                    entry.setdefault("render", {}).update(got or {})

        if render_dir:
            cand = os.path.join(render_dir, im["name"])
            if not os.path.exists(cand):
                cand = os.path.join(render_dir, os.path.splitext(im["name"])[0] + ".png")
            if os.path.exists(cand):
                pred = load(cand, size)
                entry.setdefault("render", {}).update(
                    {"psnr": round(psnr(pred, truth), 2),
                     "ssim": round(ssim(pred, truth), 4)})
        rows.append(entry)

    def gather(source: str, key: str) -> float | None:
        vals = [r[source][key] for r in rows
                if source in r and r[source].get(key) is not None]
        return round(float(np.median(vals)), 4) if vals else None

    summary = {
        "model": model,
        "held_out": len(rows),
        "training": len(train),
        "scale": scale,
        "mean_psnr": gather("mean", "psnr"),
        "mean_psnr_on_lidar_mask": gather("mean", "psnr_on_lidar_mask"),
        "lidar_psnr": gather("lidar", "psnr"),
        "lidar_coverage": gather("lidar", "coverage"),
        "lidar_depth_median_abs_m": gather("lidar", "median_abs_m"),
        "lidar_depth_abs_rel": gather("lidar", "abs_rel"),
    }
    if rejected:
        summary["render_depth_rejected"] = sorted(rejected)
    if any("render" in r for r in rows):
        summary.update({
            "render_psnr": gather("render", "psnr"),
            "render_ssim": gather("render", "ssim"),
            "render_depth_median_abs_m": gather("render", "median_abs_m"),
            "render_depth_abs_rel": gather("render", "abs_rel"),
        })
    return {"summary": summary, "frames": rows}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", help="a directory written by export_3dgs.py")
    ap.add_argument("--scale", type=float, default=0.25,
                    help="evaluate at this fraction of full resolution")
    ap.add_argument("--render-dir", default=None,
                    help="a trainer's rendered held-out images, named to match")
    ap.add_argument("--render-depth-dir", default=None,
                    help="a trainer's rendered depth as raw single-channel PNG/TIFF/npy — "
                         "a colourised depth picture is refused, not scored")
    ap.add_argument("--render-depth-scale", type=float, default=0.001,
                    help="metres per stored unit in --render-depth-dir")
    ap.add_argument("--json", action="store_true", help="include the per-frame rows")
    a = ap.parse_args(argv)

    result = evaluate(a.model, scale=a.scale, render_dir=a.render_dir,
                      render_depth_dir=a.render_depth_dir,
                      render_depth_scale=a.render_depth_scale)
    s = result["summary"]
    if a.json:
        print(json.dumps(result, indent=1))
        return 0

    print(f"{s['held_out']} held-out views, {s['training']} training, "
          f"at {int(a.scale * 100)}% resolution\n")
    print(f"{'':10} {'PSNR':>7} {'SSIM':>7} {'depth err':>10} {'abs-rel':>8} {'cover':>7}")
    print(f"{'mean':10} {s['mean_psnr'] or 0:>7.2f} {'-':>7} {'-':>10} {'-':>8} {'-':>7}")
    print(f"{'lidar':10} {s['lidar_psnr'] or 0:>7.2f} {'-':>7} "
          f"{(s['lidar_depth_median_abs_m'] or 0):>9.3f}m "
          f"{s['lidar_depth_abs_rel'] or 0:>8.4f} {s['lidar_coverage'] or 0:>7.2f}")
    if s.get("mean_psnr_on_lidar_mask") is not None:
        print(f"{'':10} scored on the same pixels the cloud covers, "
              f"the mean image gets {s['mean_psnr_on_lidar_mask']:.2f} dB")
    for reason in s.get("render_depth_rejected", []):
        print(f"\nrendered depth not scored: {reason}")
    if "render_psnr" in s:
        # An absent depth score prints as a dash. Printing 0.000 m for "not
        # measured" reads as a perfect result, which is the one wrong answer
        # this whole file exists to avoid.
        err = (f"{s['render_depth_median_abs_m']:>9.3f}m"
               if s.get("render_depth_median_abs_m") is not None else f"{'-':>10}")
        rel = (f"{s['render_depth_abs_rel']:>8.4f}"
               if s.get("render_depth_abs_rel") is not None else f"{'-':>8}")
        print(f"{'render':10} {s['render_psnr'] or 0:>7.2f} {s['render_ssim'] or 0:>7.4f} "
              f"{err} {rel} {'-':>7}")
        if s["render_psnr"] is not None and s["mean_psnr"] is not None:
            verdict = "clears" if s["render_psnr"] > s["mean_psnr"] else "FAILS"
            print(f"\n{verdict} the mean-image floor "
                  f"({s['render_psnr']:.2f} vs {s['mean_psnr']:.2f} dB)")
    else:
        print("\nno --render-dir given: these are the controls a trainer has to beat")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
