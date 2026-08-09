#!/usr/bin/env python3
"""Turn recorded sessions into VLN episode directories.

Implements the conversion in docs/OUTPUT_FORMAT.md: a session's posed images
become an `images/` directory at the deployment frame size and a `poses_tum.txt`
in the target convention — Z-up world, FLU camera, rebased to the first frame.

    python3 tools/export_episodes.py ~/nav_data/20260807-131829-9e6858 -o ./episodes
    python3 tools/export_episodes.py ~/nav_data/* -o ./episodes --fit whole
    python3 tools/export_episodes.py ~/nav_data/* -o ./episodes --hfov 66.1

Requires numpy and Pillow.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from read_session import Session, TARGET_RIG  # noqa: E402

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None


# Basis change from the ARKit camera (X right, Y up, Z back) to FLU
# (X forward, Y left, Z up). Columns are the FLU axes in ARKit camera
# coordinates, so `R_arkit @ M` has the FLU axes as its columns.
ARKIT_TO_FLU = np.array([
    [0.0, -1.0, 0.0],
    [0.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0],
])

# A 180-degree roll about the ARKit camera's optical axis. This is applied
# before the ARKit-camera-to-FLU basis change because it is a camera-frame
# correction, not a change to the world or to the camera position.
FLIP_180 = np.diag([-1.0, -1.0, 1.0])

# ARKit's world is gravity-aligned with +Y up.
ARKIT_WORLD_UP = np.array([0.0, 1.0, 0.0])


# ---------------------------------------------------------------- rotations

def quat_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat(R: np.ndarray) -> tuple[float, float, float, float]:
    """Rotation matrix to (x, y, z, w), scalar last.

    Shepperd's method: pick the branch whose denominator is largest, because
    the naive trace formula loses precision as the trace approaches -1.
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    n = math.sqrt(x * x + y * y + z * z + w * w)
    return x / n, y / n, z / n, w / n


def world_basis(first_R: np.ndarray) -> tuple[np.ndarray, bool]:
    """Target world axes expressed in ARKit's world, as columns.

    +Z is ARKit's up. +X is the first camera's heading flattened onto the floor,
    which puts the episode's yaw origin along the initial direction of view.

    Returns the basis and whether the degenerate fallback was used — a session
    that begins pointing straight down or up has no horizontal heading, and the
    camera's own up vector is the sane substitute.
    """
    forward = -first_R[:, 2]
    x_axis = np.array([forward[0], 0.0, forward[2]])
    degenerate = float(np.linalg.norm(x_axis)) < 1e-3
    if degenerate:
        up = first_R[:, 1]
        x_axis = np.array([up[0], 0.0, up[2]])
        if float(np.linalg.norm(x_axis)) < 1e-9:
            # Both the optical axis and camera up are vertical, which is not
            # geometrically possible for a rigid camera. Refuse rather than
            # emit an arbitrary yaw.
            raise ValueError("cannot establish a yaw origin from the first pose")
    x_axis = x_axis / np.linalg.norm(x_axis)
    z_axis = ARKIT_WORLD_UP
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack([x_axis, y_axis, z_axis]), degenerate


# ---------------------------------------------------------------- geometry

class FrameGeometry:
    """The crop and resize taking a captured frame to the episode frame."""

    def __init__(self, src_w: int, src_h: int, dst_w: int, dst_h: int, fit: str,
                 fx: float | None = None, fy: float | None = None,
                 hfov: float | None = None):
        self.src_w, self.src_h = src_w, src_h
        self.dst_w, self.dst_h = dst_w, dst_h
        self.fit = fit
        self.clamped = False

        if hfov is not None:
            # Crop to an exact horizontal field of view. The crop width that
            # subtends `hfov` at this frame's focal length is fixed by the
            # focal length alone; the height then follows from wanting square
            # pixels after the resize — fx*dst_w/crop_w == fy*dst_h/crop_h.
            self.crop_w = int(round(2.0 * fx * math.tan(math.radians(hfov) / 2.0)))
            self.crop_h = int(round(self.crop_w * (fy / fx) * (dst_h / dst_w)))
            if self.crop_w > src_w or self.crop_h > src_h:
                # Asking for a wider view than the lens has. Fall back to the
                # widest crop that fits and record it, rather than padding.
                self.clamped = True
                scale = min(src_w / self.crop_w, src_h / self.crop_h)
                self.crop_w = int(self.crop_w * scale)
                self.crop_h = int(self.crop_h * scale)
        elif fit == "crop":
            # Centre crop to the destination aspect, then scale. Keeps
            # degrees-per-pixel equal on both axes, at the cost of whichever
            # axis is surplus — for a 4:3 source and a 16:9 target, vertical.
            target_aspect = dst_w / dst_h
            if src_w / src_h > target_aspect:
                self.crop_h = src_h
                self.crop_w = int(round(src_h * target_aspect))
            else:
                self.crop_w = src_w
                self.crop_h = int(round(src_w / target_aspect))
        elif fit == "whole":
            # No crop: the whole frame is squeezed into the destination. Keeps
            # the full field of view, at the cost of anisotropic
            # degrees-per-pixel, so an image position no longer maps to a fixed
            # bearing the way it does on the deployment camera.
            self.crop_w, self.crop_h = src_w, src_h
        else:
            raise ValueError(f"unknown fit mode: {fit}")

        self.left = (src_w - self.crop_w) // 2
        self.top = (src_h - self.crop_h) // 2

    @property
    def box(self) -> tuple[int, int, int, int]:
        return (self.left, self.top, self.left + self.crop_w, self.top + self.crop_h)

    def output_fov(self, fx: float, fy: float) -> tuple[float, float]:
        """Horizontal and vertical FOV of the exported frame, degrees.

        Cropping changes the field of view; scaling does not. So this is a
        property of the crop alone, computed from the source intrinsics.
        """
        h = 2 * math.degrees(math.atan((self.crop_w / 2) / fx))
        v = 2 * math.degrees(math.atan((self.crop_h / 2) / fy))
        return h, v

    def describe(self) -> str:
        return (f"{self.src_w}x{self.src_h} -> crop {self.crop_w}x{self.crop_h} "
                f"@({self.left},{self.top}) -> {self.dst_w}x{self.dst_h} [{self.fit}]")


class GeometryPlan:
    """How each frame is cropped — one fixed crop, or one per frame.

    `--fit crop|whole` is a fixed rectangle: every frame is cut the same way, so
    the exported field of view breathes exactly as the lens does. `--hfov` is the
    other choice: the crop tracks each frame's focal length so the *output* field
    of view is the constant, which is what a consumer of the episode format has
    to assume, since the format records no intrinsics at all.
    """

    def __init__(self, src_w: int, src_h: int, dst_w: int, dst_h: int,
                 fit: str, hfov: float | None):
        self.src_w, self.src_h = src_w, src_h
        self.dst_w, self.dst_h = dst_w, dst_h
        self.fit, self.hfov = fit, hfov
        self._fixed = (FrameGeometry(src_w, src_h, dst_w, dst_h, fit)
                       if hfov is None else None)

    def for_frame(self, fx: float, fy: float) -> FrameGeometry:
        if self._fixed is not None:
            return self._fixed
        return FrameGeometry(self.src_w, self.src_h, self.dst_w, self.dst_h,
                             self.fit, fx=fx, fy=fy, hfov=self.hfov)

    def describe(self) -> str:
        if self._fixed is not None:
            return self._fixed.describe()
        return (f"{self.src_w}x{self.src_h} -> crop to H={self.hfov:.1f} degrees "
                f"per frame -> {self.dst_w}x{self.dst_h} [hfov]")


# ---------------------------------------------------------------- splitting

def split_on_interruptions(rows: list[dict[str, Any]],
                           session: Session) -> list[list[dict[str, Any]]]:
    """Cut the frame list wherever ARKit reset its world origin.

    Poses either side of an `ar.interruptionEnded` are in different coordinate
    frames. The episode format has nowhere to record that, so the split has to
    happen here — concatenating across one would silently produce a trajectory
    that teleports.
    """
    breaks = sorted(e["t"] for e in session.events()
                    if e["kind"] == "ar.interruptionEnded")
    if not breaks:
        return [rows]

    segments: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    remaining = list(breaks)
    for row in rows:
        while remaining and row["t"] >= remaining[0]:
            remaining.pop(0)
            if current:
                segments.append(current)
                current = []
        current.append(row)
    if current:
        segments.append(current)
    return segments


# ---------------------------------------------------------------- export

def export_segment(session: Session,
                   rows: list[dict[str, Any]],
                   out_dir: str,
                   geometry: GeometryPlan | None,
                   uniform_dt: float | None,
                   copy_images: bool,
                   rotate_180: bool = False) -> dict[str, Any]:
    """Write one episode directory. Returns its summary statistics."""
    first = rows[0]["pose"]
    R0 = quat_to_matrix(first["qx"], first["qy"], first["qz"], first["qw"])
    if rotate_180:
        R0 = R0 @ FLIP_180
    p0 = np.array([first["tx"], first["ty"], first["tz"]])
    N, degenerate = world_basis(R0)
    t0 = rows[0]["t"]

    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)

    lines = ["# timestamp tx ty tz qx qy qz qw"]
    rolls: list[float] = []
    positions: list[np.ndarray] = []

    for index, row in enumerate(rows):
        pose = row["pose"]
        R_ar = quat_to_matrix(pose["qx"], pose["qy"], pose["qz"], pose["qw"])
        if rotate_180:
            R_ar = R_ar @ FLIP_180
        R = N.T @ R_ar @ ARKIT_TO_FLU
        t = N.T @ (np.array([pose["tx"], pose["ty"], pose["tz"]]) - p0)
        qx, qy, qz, qw = matrix_to_quat(R)

        stamp = index * uniform_dt if uniform_dt is not None else row["t"] - t0
        lines.append(f"{stamp:.6f} {t[0]:.8f} {t[1]:.8f} {t[2]:.8f} "
                     f"{qx:.8f} {qy:.8f} {qz:.8f} {qw:.8f}")

        # Roll about the direction of travel: the world-Z component of body +Y.
        rolls.append(math.degrees(math.atan2(R[2, 1], R[2, 2])))
        positions.append(t)

        if copy_images:
            dst = os.path.join(out_dir, "images", f"{index:06d}.jpg")
            if geometry is None and not rotate_180:
                shutil.copyfile(row["path"], dst)
            else:
                with Image.open(row["path"]) as img:
                    if rotate_180:
                        rotation = getattr(Image, "Transpose", Image).ROTATE_180
                        img = img.transpose(rotation)
                    if geometry is None:
                        img.save(dst, "JPEG", quality=92)
                    else:
                        g = geometry.for_frame(pose["fx"], pose["fy"])
                        img.crop(g.box) \
                           .resize((g.dst_w, g.dst_h), Image.LANCZOS) \
                           .save(dst, "JPEG", quality=92)

    with open(os.path.join(out_dir, "poses_tum.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    xyz = np.array(positions)
    roll = np.array(rolls)
    mean_x = float(np.cos(np.radians(roll)).mean())
    mean_y = float(np.sin(np.radians(roll)).mean())
    return {
        "episode": os.path.basename(out_dir),
        "frames": len(rows),
        "roll_std": float(roll.std()),
        "mean_roll": float(abs(math.degrees(math.atan2(mean_y, mean_x)))),
        "vertical_std": float(xyz[:, 2].std()),
        "duration": float(rows[-1]["t"] - t0),
        "degenerate_yaw": degenerate,
    }


def export_session(path: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    session = Session(path)
    print(f"\n{session.id}")

    if not session.is_complete:
        print("  ! session is incomplete — recorded up to the last flush only")

    orientation = session.shot_orientation()
    rotate_180 = bool(orientation and orientation["upside_down"])
    if orientation and rotate_180:
        print("  ! shot landscape (upside down); rotating images and poses by 180 degrees")
    elif orientation and orientation["dominant"] in ("portrait", "portrait (inverted)"):
        print(f"  ! shot {orientation['dominant']}; images are stored unrotated so "
              f"a landscape crop would be wrong. Skipping (use --force to override).")
        if not args.force:
            return []
    if orientation and orientation["mixed"]:
        print(f"  ! orientation changed mid-session — the crop is only right for part of it")

    rows = session.posed_images()
    if not rows:
        print("  ! no posed images (video mode, or nothing recorded)")
        return []

    total = len(rows)
    if not args.keep_limited:
        rows = [r for r in rows if r["tracking"] == "normal"]
        dropped = total - len(rows)
        if dropped:
            # Poses before ARKit converges sit at the origin; keeping them puts
            # a cluster of identical positions at the start of the trajectory.
            print(f"  dropped {dropped}/{total} frames with unconverged tracking")
    if not rows:
        print("  ! nothing left after filtering")
        return []

    if args.hz:
        # Same fixed-schedule rule as the recorder's own gate, so decimating a
        # 30 Hz capture to 5 Hz picks the frames a 5 Hz capture would have taken
        # rather than drifting against them.
        kept, due, interval = [], None, 1.0 / args.hz
        for row in rows:
            if due is None or row["t"] >= due:
                kept.append(row)
                due = (row["t"] + interval if due is None
                       else (due + interval if row["t"] - due < interval
                             else row["t"] + interval))
        span = rows[-1]["t"] - rows[0]["t"]
        if span > 0:
            print(f"  decimated {len(rows)} -> {len(kept)} frames "
                  f"({len(rows) / span:.1f} Hz captured, "
                  f"{len(kept) / span:.1f} Hz exported)")
        rows = kept

    geometry = None
    if not args.poses_only:
        if Image is None:
            print("  ! Pillow is not installed — writing poses only")
        else:
            src_w, src_h = rows[0]["width"], rows[0]["height"]
            geometry = GeometryPlan(src_w, src_h, args.width, args.height,
                                    args.fit, args.hfov)
            print(f"  {geometry.describe()}")

            # Across every frame, not just the first: ARKit leaves autofocus on
            # and cannot be told not to, so focal length breathes as focus
            # hunts. The episode format carries no intrinsics at all, so any
            # drift is lost at this boundary — reporting the spread is the only
            # place it can be seen.
            fovs = [geometry.for_frame(r["pose"]["fx"], r["pose"]["fy"])
                    .output_fov(r["pose"]["fx"], r["pose"]["fy"]) for r in rows]
            hs = [f[0] for f in fovs]
            vs = [f[1] for f in fovs]
            h, v = sum(hs) / len(hs), sum(vs) / len(vs)
            print(f"  output fov  H={h:.1f} V={v:.1f} degrees (mean) "
                  f"(target {TARGET_RIG['hfov']}, {TARGET_RIG['vfov']}: "
                  f"{h - TARGET_RIG['hfov']:+.1f}, {v - TARGET_RIG['vfov']:+.1f})")
            h_spread = max(hs) - min(hs)
            if h_spread > 0.5:
                print(f"  ! focal length drifted within the session: H spans "
                      f"{min(hs):.1f}-{max(hs):.1f} degrees ({h_spread:.1f} of breathing). "
                      f"The episode format cannot record this; the session can."
                      + ("" if args.hfov is None else
                         " Asked for a fixed --hfov and did not get it — see the "
                         "clamp warning above."))
            if args.hfov is not None:
                clamped = sum(1 for r in rows
                              if geometry.for_frame(r["pose"]["fx"],
                                                    r["pose"]["fy"]).clamped)
                if clamped:
                    print(f"  ! {clamped}/{len(rows)} frames could not reach "
                          f"H={args.hfov:.1f} — the lens is not that wide. Those "
                          f"frames are the widest crop that fits, so the field of "
                          f"view is not constant after all.")
            if args.fit == "whole":
                dpp_x = h / args.width
                dpp_y = v / args.height
                print(f"  ! anisotropic: {dpp_x:.4f} vs {dpp_y:.4f} degrees/pixel — "
                      f"image position does not map to a fixed bearing")

    segments = split_on_interruptions(rows, session)
    if len(segments) > 1:
        print(f"  split into {len(segments)} episodes at ar.interruptionEnded "
              f"(the world origin resets there)")

    summaries = []
    for index, segment in enumerate(segments):
        if len(segment) < args.min_frames:
            print(f"  segment {index}: {len(segment)} frames — below --min-frames, skipped")
            continue
        name = session.id if len(segments) == 1 else f"{session.id}_seg{index}"
        out = os.path.join(args.out, name)
        summary = export_segment(session, segment, out, geometry,
                                 args.uniform_timestamps, geometry is not None,
                                 rotate_180=rotate_180)
        if summary["degenerate_yaw"]:
            print(f"  ! {name}: started pointing vertically — yaw origin taken "
                  f"from camera up")
        flags = []
        if summary["roll_std"] > 4:
            flags.append(f"roll_std {summary['roll_std']:.2f}>4")
        if summary["mean_roll"] > 4:
            flags.append(f"mean_roll {summary['mean_roll']:.2f}>4")
        if summary["vertical_std"] > 0.1:
            flags.append(f"vertical {summary['vertical_std']:.3f}>0.1")
        status = "  ".join(flags) if flags else "ok"
        print(f"  {name}: {summary['frames']} frames, {summary['duration']:.1f}s  "
              f"roll_std {summary['roll_std']:.3f}  mean_roll {summary['mean_roll']:.3f}  "
              f"vertical {summary['vertical_std']:.4f}  [{status}]")
        summaries.append(summary)
    return summaries


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sessions", nargs="+", help="session directories")
    parser.add_argument("-o", "--out", required=True, help="output directory for episodes")
    parser.add_argument("--fit", choices=["crop", "whole"], default="crop",
                        help="crop to the target aspect (default) or squeeze the whole frame")
    parser.add_argument("--hfov", type=float, default=None, metavar="DEGREES",
                        help="crop each frame to this exact horizontal field of "
                             "view instead of to the target aspect; the crop "
                             "tracks focus breathing so the output fov is fixed")
    parser.add_argument("--width", type=int, default=TARGET_RIG["width"])
    parser.add_argument("--height", type=int, default=TARGET_RIG["height"])
    parser.add_argument("--keep-limited", action="store_true",
                        help="keep frames whose ARKit tracking had not converged")
    parser.add_argument("--uniform-timestamps", type=float, default=None,
                        metavar="DT",
                        help="emit a fixed stride instead of true rebased times")
    parser.add_argument("--hz", type=float, default=None,
                        help="decimate to this rate before exporting. Capture "
                             "runs best-effort and can be faster than the "
                             "episode wants; this is where that is resolved.")
    parser.add_argument("--min-frames", type=int, default=10)
    parser.add_argument("--poses-only", action="store_true", help="skip image export")
    parser.add_argument("--force", action="store_true",
                        help="export even when the session was not shot landscape")
    args = parser.parse_args(argv)

    if args.hfov is not None:
        if args.hfov <= 0 or args.hfov >= 180:
            parser.error("--hfov must be between 0 and 180 degrees")
        if args.fit != "crop":
            parser.error("--hfov sets the crop itself, so it cannot be combined "
                         "with --fit whole")

    os.makedirs(args.out, exist_ok=True)
    summaries = []
    for path in args.sessions:
        path = path.rstrip("/")
        try:
            summaries.extend(export_session(path, args))
        except FileNotFoundError as exc:
            print(f"\n{path}\n  ! {exc}")

    if not summaries:
        print("\nnothing exported")
        return 1

    # Mirrors alignment_summary.txt in the reference set, and is computable from
    # poses_tum.txt alone — so it doubles as an end-to-end check that the basis
    # came out right. A wrong frame sends these wild.
    path = os.path.join(args.out, "alignment_summary.txt")
    with open(path, "w") as f:
        f.write("episode\tframes\troll_std\tmean_roll\ty_spread_std\n")
        for s in summaries:
            f.write(f"{s['episode']}\t{s['frames']}\t{s['roll_std']:.3f}\t"
                    f"{s['mean_roll']:.3f}\t{s['vertical_std']:.4f}\n")
    print(f"\n{len(summaries)} episode(s) -> {args.out}")
    print(f"summary -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
