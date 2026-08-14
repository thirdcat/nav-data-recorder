#!/usr/bin/env python3
"""Draw the coverage measurement from above, because a table hides the shape.

`survey_coverage.py` answers "how much of the wall has enough views". This
answers "which wall", which is the question a person holding a phone can act on:
the picture shows the room, the path walked through it, and which surfaces came
out unconstrained.

    python3 tools/plot_coverage.py out.png ~/nav_data/20260809-074458-cb4586
    python3 tools/plot_coverage.py before-after.png ~/nav_data/one ~/nav_data/two

Several sessions become panels side by side, which is what makes this worth
having: recording a room twice — once as usual and once walked closer, or twice
on different lines — and putting the two plans next to each other is the whole
of the stage-4 experiment in `docs/3DGS.md`.

The counting is `survey_coverage.voxel_view_counts`, not a second copy of it, so
the percentage printed under each panel is the same number the table reports.
Colour is the *median* over each vertical column of voxels rather than a
recount in the plan: counting flat would add the directions that saw the top of
a wall to the ones that saw its bottom, and make every session look better in
proportion to how tall its geometry is.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from read_session import Session  # noqa: E402
from survey_coverage import merged_view_counts, voxel_view_counts  # noqa: E402

try:
    from PIL import Image, ImageDraw
except ImportError:  # pragma: no cover
    Image = None

# One view (nothing constrains the Gaussian's depth) through five or more.
RAMP = [(196, 48, 48), (214, 118, 40), (198, 176, 46), (120, 176, 64), (54, 152, 96)]
PAPER = (250, 249, 246)
INK = (40, 40, 44)


def column_medians(cells: np.ndarray, counts: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Collapse 3D voxels onto the ground plane, keeping the median count."""
    columns, inverse = np.unique(cells[:, [0, 2]], axis=0, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    bounds = np.searchsorted(inverse[order], np.arange(len(columns) + 1))
    medians = np.array([np.median(counts[order[a:b]])
                        for a, b in zip(bounds[:-1], bounds[1:])])
    return columns, medians


def panel(session_dir, *, size: int, margin: int, voxel: float,
          vertical_only: bool, pix_stride: int, label: str | None,
          transforms: dict | None = None):
    """One plan view. `session_dir` may be a list, which draws them as one space."""
    if isinstance(session_dir, (list, tuple)):
        cells, counts, centres, _ = merged_view_counts(
            list(session_dir), transforms or {}, voxel=voxel,
            vertical_only=vertical_only, pix_stride=pix_stride)
        name_default = f"merged({len(session_dir)})"
    else:
        cells, counts, centres, _ = voxel_view_counts(
            session_dir, voxel=voxel, vertical_only=vertical_only,
            pix_stride=pix_stride)
        name_default = Session(session_dir).id[-6:]
    columns, medians = column_medians(cells, counts)
    ground = columns.astype(np.float64) * voxel

    lo = np.minimum(ground.min(0), centres[:, [0, 2]].min(0))
    hi = np.maximum(ground.max(0), centres[:, [0, 2]].max(0))
    span = max(float((hi - lo).max()), 1e-6)
    scale = (size - 2 * margin) / span

    img = Image.new("RGB", (size, size + 34), PAPER)
    draw = ImageDraw.Draw(img)

    def to_px(x: float, z: float) -> tuple[float, float]:
        return (margin + (x - lo[0]) * scale, size - margin - (z - lo[1]) * scale)

    box = max(1, int(round(voxel * scale)))
    for (x, z), n in zip(ground, medians):
        px, py = to_px(x, z)
        draw.rectangle([px, py, px + box, py + box],
                       fill=RAMP[min(int(n), len(RAMP)) - 1])

    path = [to_px(c[0], c[2]) for c in centres]
    if len(path) > 1:
        draw.line(path, fill=(28, 28, 30), width=2)
    sx, sy = path[0]
    draw.ellipse([sx - 4, sy - 4, sx + 4, sy + 4], fill=(255, 255, 255),
                 outline=(28, 28, 30), width=2)

    name = label or name_default
    surface = "wall" if vertical_only else "surface"
    draw.text((margin, size + 8),
              f"{name}   {100 * float((counts >= 3).mean()):.0f}% of {surface} at 3+ views",
              fill=INK)
    draw.text((margin, 10), f"{span:.1f} m across   {len(centres)} frames",
              fill=(120, 120, 126))
    return img


def compose(panels: list, vertical_only: bool):
    gap, legend_h = 20, 44
    width = sum(p.width for p in panels) + gap * (len(panels) - 1)
    sheet = Image.new("RGB", (width, panels[0].height + legend_h), PAPER)
    x = 0
    for p in panels:
        sheet.paste(p, (x, 0))
        x += p.width + gap

    draw = ImageDraw.Draw(sheet)
    y = panels[0].height + 8
    surface = "wall" if vertical_only else "surface"
    draw.text((34, y),
              f"colour: median distinct viewing directions over each column of {surface}",
              fill=INK)
    x0 = 470
    for i, colour in enumerate(RAMP):
        draw.rectangle([x0 + i * 58, y - 2, x0 + i * 58 + 22, y + 14], fill=colour)
        draw.text((x0 + i * 58, y + 18), ["1", "2", "3", "4", "5+"][i],
                  fill=(120, 120, 126))
    return sheet


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", help="PNG to write")
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--voxel", type=float, default=0.05)
    ap.add_argument("--pix-stride", type=int, default=2)
    ap.add_argument("--size", type=int, default=470, help="panel edge in pixels")
    ap.add_argument("--all-surfaces", action="store_true",
                    help="include floors and ceilings, which any walk covers for free")
    ap.add_argument("--label", action="append", default=None,
                    help="panel caption, repeatable and matched in order")
    ap.add_argument("--transforms", default=None,
                    help="a JSON from align_set.py; every session listed is then "
                         "drawn as ONE space in a single panel")
    a = ap.parse_args(argv)

    if Image is None:
        raise SystemExit("Pillow is required: pip install pillow")

    transforms = None
    if a.transforms:
        with open(a.transforms) as fh:
            doc = json.load(fh)
        transforms = {k: np.asarray(v, dtype=float)
                      for k, v in doc["transforms"].items()}

    labels = (a.label or []) + [None] * (len(a.sessions) + 1)
    targets = [a.sessions] if transforms else list(a.sessions)
    panels = []
    for target, label in zip(targets, labels):
        try:
            panels.append(panel(target, size=a.size, margin=34, voxel=a.voxel,
                                vertical_only=not a.all_surfaces,
                                pix_stride=a.pix_stride, label=label,
                                transforms=transforms))
        except ValueError as exc:
            name = target if isinstance(target, str) else "merged"
            print(f"skipping {os.path.basename(os.path.normpath(str(name)))}: {exc}")
    if not panels:
        raise SystemExit("nothing to draw")

    compose(panels, not a.all_surfaces).save(a.out)
    print(a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
