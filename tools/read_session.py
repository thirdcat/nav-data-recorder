#!/usr/bin/env python3
"""Read a session recorded by NavDataRecorder.

Reference implementation of the on-disk format described in
docs/DATA_FORMAT.md. Standard library only — numpy is used for depth arrays if
it is installed, and plain lists are returned if it is not.

    python3 tools/read_session.py /path/to/20260807-014530-a1b2c3
    python3 tools/read_session.py <dir> --depth-frame 0
    python3 tools/read_session.py <dir> --align
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from bisect import bisect_left
from typing import Any, Iterator

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is optional
    np = None


# The rig this data is being collected *for*. Capture is only as useful as its
# match to deployment, so the reader checks each session against it rather than
# leaving the comparison to memory.
TARGET_RIG = {
    "name": "Unitree G1 head camera (OAK-1-W)",
    "hfov": 96.3,
    "vfov": 64.6,          # derived from 96.3 horizontal at 848x480
    "width": 848,
    "height": 480,
    "camera_height_m": 1.15,
}

# Empirical values from five sessions on one iPhone 16 Pro. They are useful
# for detecting the opening focus transient, not a claim of universal camera
# precision.
FOCUS_RATE_THRESHOLD = 10.0
FOCUS_STABLE_STILLS = 5


class Session:
    """A recorded session on disk.

    Every stream is stamped in the device's monotonic clock (`t`). Wall-clock
    time is `t + manifest["clockAnchor"]`; nothing in the format depends on the
    system clock being correct during the drive.
    """

    def __init__(self, path: str):
        self.path = path
        manifest_path = os.path.join(path, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"no manifest.json in {path}")
        with open(manifest_path) as f:
            self.manifest = json.load(f)

    # -- basics ----------------------------------------------------------

    @property
    def id(self) -> str:
        return self.manifest["id"]

    @property
    def clock_anchor(self) -> float:
        return self.manifest["clockAnchor"]

    def to_unix(self, t: float) -> float:
        """Convert a monotonic sample timestamp to Unix seconds."""
        return t + self.clock_anchor

    @property
    def is_complete(self) -> bool:
        """False if the recording was cut short before the files were closed.

        An incomplete session is still readable — that is the point of JSONL —
        but video.mov was never finalised and will not play.
        """
        return os.path.exists(os.path.join(self.path, ".complete"))

    # -- streams ---------------------------------------------------------

    def stream(self, name: str) -> Iterator[dict[str, Any]]:
        """Yield rows from one of the .jsonl files, skipping a torn final line.

        A session killed mid-write (crash, battery, force quit) can leave one
        truncated row at the end of a file. Everything before it is intact, so
        the reader drops that line rather than refusing the whole stream.
        """
        path = os.path.join(self.path, f"{name}.jsonl")
        if not os.path.exists(path):
            return
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # Only ever expected on the last line of a torn file.
                    continue

    def locations(self) -> list[dict[str, Any]]:
        return list(self.stream("location"))

    def motion(self) -> list[dict[str, Any]]:
        return list(self.stream("motion"))

    def poses(self) -> list[dict[str, Any]]:
        return list(self.stream("pose"))

    def headings(self) -> list[dict[str, Any]]:
        return list(self.stream("heading"))

    def frames(self) -> list[dict[str, Any]]:
        """Written RGB images, in capture order. Empty in video mode."""
        return list(self.stream("frames"))

    def frame_path(self, entry: dict[str, Any]) -> str:
        """Absolute path to a frame's JPEG."""
        return os.path.join(self.path, entry["file"])

    def posed_images(self) -> list[dict[str, Any]]:
        """The VLN-shaped view: one row per written image, with its pose.

        Joined on `frame`, which both streams carry, so this is an exact match
        rather than a nearest-timestamp search — the image and the pose came out
        of the same ARFrame.
        """
        poses = {p["frame"]: p for p in self.stream("pose")}
        depth_rows = self.depth_index()
        depth_by_frame = {d["frame"]: d for d in depth_rows}
        depth_times = [d["t"] for d in depth_rows]

        rows = []
        for entry in self.stream("frames"):
            pose = poses.get(entry["frame"])
            if pose is None:
                # Should not happen: poses are written for every frame. If it
                # does, the image is unusable for training and is dropped.
                continue

            # Exact frame match is the intended case — image and depth from the
            # same ARFrame. Sessions recorded before the capture gates were
            # unified have depth on *different* frames, so fall back to the
            # nearest in time rather than reporting no depth at all. `depth_dt`
            # says which happened: 0.0 is an exact pairing, anything else is an
            # approximation the consumer may want to reject.
            depth = depth_by_frame.get(entry["frame"])
            depth_dt = 0.0 if depth is not None else None
            if depth is None and depth_rows:
                near = _nearest(depth_rows, depth_times, entry["t"])
                if near is not None:
                    depth = near
                    depth_dt = entry["t"] - near["t"]

            rows.append({
                "t": entry["t"],
                "frame": entry["frame"],
                "file": entry["file"],
                "path": self.frame_path(entry),
                "width": entry["width"],
                "height": entry["height"],
                "pose": pose,
                "depth": depth,
                "depth_dt": depth_dt,
                "tracking": pose["tracking"],
            })
        return rows

    def planes(self) -> list[dict[str, Any]]:
        return list(self.stream("planes"))

    def final_planes(self) -> dict[str, dict[str, Any]]:
        """The last known state of each plane that was not removed.

        Planes grow and merge as a room is explored, so the raw stream holds
        many rows per plane. This collapses it to the room's final structure —
        usually what a consumer wants, with the full history still available
        from `planes()`.
        """
        latest: dict[str, dict[str, Any]] = {}
        for row in self.stream("planes"):
            if row["event"] == "removed":
                latest.pop(row["id"], None)
            else:
                latest[row["id"]] = row
        return latest

    def events(self) -> list[dict[str, Any]]:
        return list(self.stream("events"))

    def depth_index(self) -> list[dict[str, Any]]:
        return list(self.stream("depth"))

    # -- depth -----------------------------------------------------------

    def depth_frame(self, entry: dict[str, Any]):
        """Load one depth map, in metres.

        Returns a numpy array shaped (height, width) when numpy is available,
        otherwise a list of rows. Non-finite values mean the sensor got no
        return for that pixel — sky, glass, and anything past a few metres.
        """
        path = os.path.join(self.path, "depth.bin")
        width, height = entry["width"], entry["height"]
        count = width * height
        with open(path, "rb") as f:
            f.seek(entry["offset"])
            raw = f.read(entry["length"])
        if len(raw) < count * 2:
            raise ValueError(
                f"depth frame {entry['frame']} truncated: "
                f"{len(raw)} bytes, expected {count * 2}")

        if np is not None:
            return np.frombuffer(raw, dtype=np.float16, count=count).reshape(height, width)

        # '<%de' is little-endian IEEE 754 half precision.
        values = struct.unpack(f"<{count}e", raw[: count * 2])
        return [list(values[y * width:(y + 1) * width]) for y in range(height)]

    def confidence_frame(self, entry: dict[str, Any]):
        """Per-pixel confidence for a depth frame, 0 (low) to 2 (high)."""
        if entry.get("confidenceOffset") is None:
            return None
        path = os.path.join(self.path, "confidence.bin")
        width, height = entry["width"], entry["height"]
        with open(path, "rb") as f:
            f.seek(entry["confidenceOffset"])
            raw = f.read(entry["confidenceLength"])
        if np is not None:
            return np.frombuffer(raw, dtype=np.uint8).reshape(height, width)
        return [list(raw[y * width:(y + 1) * width]) for y in range(height)]

    # -- alignment -------------------------------------------------------

    def align_to_poses(self) -> list[dict[str, Any]]:
        """Join each camera pose with the sensor readings nearest in time.

        This is the shape most training pipelines actually want: one row per
        video frame, carrying the GPS fix and IMU sample that were current when
        that frame was captured. Nearest-neighbour rather than interpolation —
        GPS at 1 Hz against 30 Hz video does not benefit from pretending to a
        precision it does not have, and the `*_dt` fields make the real gap
        explicit so a consumer can reject rows that are too stale.
        """
        poses = self.poses()
        locations = self.locations()
        motion = self.motion()
        depth = {d["frame"]: d for d in self.depth_index()}

        loc_times = [row["t"] for row in locations]
        imu_times = [row["t"] for row in motion]

        aligned = []
        for pose in poses:
            t = pose["t"]
            row: dict[str, Any] = {
                "t": t,
                "unix": self.to_unix(t),
                "frame": pose["frame"],
                "pose": pose,
                "depth": depth.get(pose["frame"]),
            }
            loc = _nearest(locations, loc_times, t)
            if loc is not None:
                row["location"] = loc
                row["location_dt"] = t - loc["t"]
            imu = _nearest(motion, imu_times, t)
            if imu is not None:
                row["motion"] = imu
                row["motion_dt"] = t - imu["t"]
            aligned.append(row)
        return aligned

    @staticmethod
    def upright_roll(pose: dict[str, Any]) -> float | None:
        """In-image roll, degrees, needed to make a frame upright w.r.t. gravity.

        Rotate the image by `-roll` and gravity points down in it. 0 means the
        phone was held landscape; ±90 means portrait; anything else means it was
        tilted, which is the common case handheld.
        """
        import math
        gx, gy = pose.get("gravX"), pose.get("gravY")
        if gx is None or gy is None:
            return None
        # Gravity projected onto the image plane. Near zero means the camera is
        # pointing almost straight up or down, where roll is undefined.
        if math.hypot(gx, gy) < 1e-3:
            return None
        return math.degrees(math.atan2(gx, -gy))

    def shot_orientation(self) -> dict[str, Any] | None:
        """How the phone was held, summarised across the session.

        Needed because nothing in the JPEGs themselves says: ARKit always
        delivers the camera's native landscape buffer, so a portrait recording
        is byte-structurally identical to a landscape one with the world rotated
        inside it.
        """
        import math
        rolls = [r for r in (self.upright_roll(p) for p in self.stream("pose"))
                 if r is not None]
        if not rolls:
            return None

        def bucket(roll: float) -> str:
            if -45 <= roll < 45:
                return "landscape"
            if 45 <= roll < 135:
                return "portrait"
            if -135 <= roll < -45:
                return "portrait (inverted)"
            return "landscape (upside down)"

        counts: dict[str, int] = {}
        for roll in rolls:
            key = bucket(roll)
            counts[key] = counts.get(key, 0) + 1
        dominant = max(counts, key=lambda k: counts[k])
        # Circular mean, so rolls straddling ±180 do not average to zero.
        mean_x = sum(math.cos(math.radians(r)) for r in rolls) / len(rolls)
        mean_y = sum(math.sin(math.radians(r)) for r in rolls) / len(rolls)
        return {
            "dominant": dominant,
            "counts": counts,
            "mean_roll": math.degrees(math.atan2(mean_y, mean_x)),
            "mixed": len(counts) > 1,
            "upside_down": dominant == "landscape (upside down)",
        }

    def focus_settle(self) -> dict[str, Any] | None:
        """When autofocus settled, measured on the written stills.

        The rate is measured between consecutive stills rather than across
        every ARKit pose: stills are the rows the episode exporter can drop.
        A flat lens from the first sample is considered settled at the session
        start; otherwise the settle time is the first still after five
        consecutive below-threshold rate measurements. The settle timestamp is
        the first still in that quiet run.
        """
        import math

        poses = {p["frame"]: p for p in self.stream("pose")}
        samples = []
        for frame in self.stream("frames"):
            pose = poses.get(frame["frame"])
            fx = pose.get("fx") if pose is not None else None
            if pose is None or fx is None:
                continue
            if not math.isfinite(frame["t"]) or not math.isfinite(fx):
                continue
            samples.append((frame["t"], fx))

        result = {
            "settled_after": None,
            "rate_threshold": FOCUS_RATE_THRESHOLD,
            "consecutive_stills": FOCUS_STABLE_STILLS,
        }
        if len(samples) < FOCUS_STABLE_STILLS + 1:
            return None

        session_start = (min(p["t"] for p in poses.values())
                         if poses else samples[0][0])
        rates = []
        for previous, current in zip(samples, samples[1:]):
            dt = current[0] - previous[0]
            if dt <= 0:
                return None
            rates.append(abs(current[1] - previous[1]) / dt)

        for start in range(len(rates) - FOCUS_STABLE_STILLS + 1):
            window = rates[start:start + FOCUS_STABLE_STILLS]
            if all(rate < FOCUS_RATE_THRESHOLD for rate in window):
                # If the first five measurements are already flat, there is
                # no opening transient to discard. Otherwise the first still
                # in the quiet run is the destination of rate[start].
                settled_index = 0 if start == 0 else start + 1
                result["settled_after"] = samples[settled_index][0] - session_start
                return result
        return result

    def rig_delta(self) -> dict[str, Any] | None:
        """How far this session's optics sit from the deployment camera.

        A navigation policy trained on one viewpoint and run on another pays for
        the difference, so the mismatch is worth seeing on every session rather
        than being rediscovered later.
        """
        fov = self.field_of_view()
        if fov is None:
            return None
        return {
            "hfov": fov[0], "vfov": fov[1],
            "d_hfov": fov[0] - TARGET_RIG["hfov"],
            "d_vfov": fov[1] - TARGET_RIG["vfov"],
        }

    def camera_height(self) -> float | None:
        """Camera height above the detected floor, metres.

        ARKit's world origin sits at wherever the session started, not on the
        ground, so height only becomes knowable once a floor plane is found —
        and only if one is classified as such.
        """
        floors = [p for p in self.final_planes().values()
                  if p["classification"] == "floor"]
        if not floors:
            return None
        floor_y = sum(p["ty"] for p in floors) / len(floors)
        heights = [p["ty"] - floor_y for p in self.stream("pose")]
        if not heights:
            return None
        return sum(heights) / len(heights)

    def camera_aim(self) -> dict[str, Any] | None:
        """How the camera was pointed, in degrees below horizontal.

        Matters more indoors than it sounds. The vertical field of view is the
        scarce one, and where it lands is set entirely by pitch: held level at
        chest height, the bottom of the frame meets the floor about 2.7 m ahead,
        so everything nearer than that is simply not recorded. Tilting down
        trades ceiling for near floor.

        Derived from gravity in camera coordinates: the camera looks along -Z,
        so the component of gravity along that axis is the sine of the pitch.
        """
        import math
        pitches = []
        for pose in self.stream("pose"):
            gz = pose.get("gravZ")
            if gz is None:
                continue
            pitches.append(math.degrees(math.asin(max(-1.0, min(1.0, -gz)))))
        if not pitches:
            return None
        mean = sum(pitches) / len(pitches)
        spread = max(pitches) - min(pitches)
        return {"mean_pitch_down": mean, "spread": spread,
                "min": min(pitches), "max": max(pitches)}

    def world_field_of_view(self) -> tuple[float, float] | None:
        """(horizontal, vertical) FOV in **world** terms, degrees.

        The sensor FOV is fixed, but which axis is horizontal in the world
        depends on how the phone was held. Holding it portrait swaps them, which
        costs about 13 degrees of horizontal coverage — the axis VLN cares most
        about.
        """
        fov = self.field_of_view()
        orientation = self.shot_orientation()
        if fov is None or orientation is None:
            return None
        if orientation["dominant"].startswith("portrait"):
            return (fov[1], fov[0])
        return (fov[0], fov[1])

    def field_of_view(self) -> tuple[float, float, float] | None:
        """(horizontal, vertical, diagonal) FOV in degrees, from intrinsics.

        Measured rather than assumed: focal length varies a little between
        devices and shifts as autofocus hunts, so this reads the first pose's
        actual intrinsics instead of quoting a spec figure.
        """
        import math
        for pose in self.stream("pose"):
            fx, fy = pose["fx"], pose["fy"]
            if fx <= 0 or fy <= 0:
                continue
            frames = self.frames()
            if frames:
                w, h = frames[0]["width"], frames[0]["height"]
            elif self.manifest.get("video"):
                w, h = self.manifest["video"]["width"], self.manifest["video"]["height"]
            else:
                # Fall back to inferring the sensor size from the principal point.
                w, h = pose["cx"] * 2, pose["cy"] * 2
            return (
                2 * math.degrees(math.atan(w / (2 * fx))),
                2 * math.degrees(math.atan(h / (2 * fy))),
                2 * math.degrees(math.atan(math.hypot(w, h) / (2 * fx))),
            )
        return None

    # -- reporting -------------------------------------------------------

    def summary(self) -> str:
        m = self.manifest
        lines = [
            f"session      {self.id}",
            f"complete     {self.is_complete}",
            f"device       {m['device']['model']} (iOS {m['device']['systemVersion']}), "
            f"LiDAR={m['device']['hasLiDAR']}",
            f"app          {m['appVersion']} ({m['appBuild']}), schema {m['schemaVersion']}",
        ]
        if m.get("endedAt"):
            lines.append(f"duration     {m['endedAt'] - m['startedAt']:.1f} s")
        if m.get("terminationReason"):
            lines.append(f"ended by     {m['terminationReason']}")
        if m.get("video"):
            v = m["video"]
            lines.append(
                f"video        {v['width']}x{v['height']} {v['codec']} "
                f"@{v['nominalFPS']}fps, {v['bitrate'] // 1_000_000} Mbps")
        counts = m.get("counts", {})
        if counts:
            lines.append("counts       " + ", ".join(
                f"{k}={v}" for k, v in sorted(counts.items())))

        actual = {name: sum(1 for _ in self.stream(name))
                  for name in ("location", "motion", "pose", "planes", "frames",
                               "depth", "events")}
        lines.append("on disk      " + ", ".join(
            f"{k}={v}" for k, v in sorted(actual.items())))

        fov = self.field_of_view()
        if fov:
            lines.append(f"sensor fov   H={fov[0]:.1f} V={fov[1]:.1f} D={fov[2]:.1f} degrees")

        orientation = self.shot_orientation()
        if orientation:
            detail = ", ".join(f"{k} {v}" for k, v in sorted(orientation["counts"].items()))
            lines.append(f"held         {orientation['dominant']} "
                         f"(mean roll {orientation['mean_roll']:+.1f} deg)")
            if orientation["mixed"]:
                lines.append(f"WARNING      orientation changed mid-session: {detail}")
            world = self.world_field_of_view()
            if world:
                lines.append(f"world fov    H={world[0]:.1f} V={world[1]:.1f} degrees")

        delta = self.rig_delta()
        if delta:
            lines.append(f"vs target    {TARGET_RIG['name']}: "
                         f"H {delta['d_hfov']:+.1f}, V {delta['d_vfov']:+.1f} degrees")

        height = self.camera_height()
        if height is not None:
            dh = height - TARGET_RIG["camera_height_m"]
            flag = "" if abs(dh) < 0.15 else "   <-- off target"
            lines.append(f"height       {height:.2f} m above the detected floor "
                         f"({dh:+.2f} vs {TARGET_RIG['camera_height_m']} m){flag}")
        else:
            lines.append(f"height       unknown — no floor plane classified; hold the "
                         f"phone at {TARGET_RIG['camera_height_m']} m to match the rig")

        aim = self.camera_aim()
        if aim:
            lines.append(f"aim          {aim['mean_pitch_down']:+.1f} deg below horizontal "
                         f"(range {aim['min']:+.0f} to {aim['max']:+.0f})")
            # The reference episodes sit at 24.6-31.5 deg below horizontal and the
            # downstream prompts were tuned on that view, so matching them beats
            # the geometric optimum.
            if aim["mean_pitch_down"] < 15:
                lines.append("             NOTE shallower than the reference episodes "
                             "(24.6-31.5 deg down); aim ~25 deg to match them")
            elif aim["mean_pitch_down"] > 40:
                lines.append("             NOTE steeper than any reference episode")

        images = self.frames()
        if images:
            total = sum(row["bytes"] for row in images)
            first = images[0]
            lines.append(f"images       {len(images)} x {first['width']}x{first['height']} jpg, "
                         f"{total / 1e6:.0f} MB")

        planes = self.final_planes()
        if planes:
            kinds: dict[str, int] = {}
            for row in planes.values():
                kinds[row["classification"]] = kinds.get(row["classification"], 0) + 1
            lines.append(f"planes       {len(planes)} final: " + ", ".join(
                f"{k}×{v}" for k, v in sorted(kinds.items())))

        # The manifest is written after the files are closed, so a mismatch
        # means the session was interrupted between the two.
        mismatches = [k for k, v in actual.items()
                      if k in counts and counts[k] != v]
        if mismatches:
            lines.append(f"WARNING      count mismatch in: {', '.join(mismatches)}")

        gaps = self.gaps()
        if gaps:
            lines.append(f"gaps         {len(gaps)} location gaps over 5 s")
            for start, end in gaps[:5]:
                lines.append(f"             {end - start:.1f} s at t={start:.1f}")

        notable = [e for e in self.events()
                   if e["kind"].split(".")[0] in ("thermal", "ar", "app")
                   and e["kind"] not in ("ar.started",)]
        if notable:
            lines.append(f"events       {len(notable)} notable")
            for e in notable[:8]:
                lines.append(f"             t={e['t'] - self.manifest['startClock']:8.1f}  "
                             f"{e['kind']}: {e['detail']}")
        return "\n".join(lines)

    def gaps(self, threshold: float = 5.0) -> list[tuple[float, float]]:
        """Stretches with no GPS fix for longer than `threshold` seconds."""
        times = [row["t"] for row in self.stream("location")]
        return [(times[i - 1], times[i])
                for i in range(1, len(times))
                if times[i] - times[i - 1] > threshold]


def _nearest(rows: list[dict[str, Any]], times: list[float], t: float):
    """Row whose timestamp is closest to `t`, or None if there are no rows."""
    if not rows:
        return None
    i = bisect_left(times, t)
    if i == 0:
        return rows[0]
    if i >= len(rows):
        return rows[-1]
    before, after = rows[i - 1], rows[i]
    return before if (t - times[i - 1]) <= (times[i] - t) else after


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Read a NavDataRecorder session.")
    parser.add_argument("session", help="path to a session directory")
    parser.add_argument("--depth-frame", type=int, default=None,
                        help="print statistics for the Nth recorded depth frame")
    parser.add_argument("--align", action="store_true",
                        help="print the first few pose-aligned rows")
    parser.add_argument("--posed", action="store_true",
                        help="print the posed-image rows a VLN pipeline would consume")
    args = parser.parse_args(argv)

    try:
        session = Session(args.session)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(session.summary())

    if args.depth_frame is not None:
        index = session.depth_index()
        if not index:
            print("\nno depth frames in this session")
            return 0
        if args.depth_frame >= len(index):
            print(f"\nerror: only {len(index)} depth frames", file=sys.stderr)
            return 1
        entry = index[args.depth_frame]
        frame = session.depth_frame(entry)
        print(f"\ndepth frame {args.depth_frame}: "
              f"{entry['width']}x{entry['height']} at t={entry['t']:.3f}")
        if np is not None:
            finite = frame[np.isfinite(frame)]
            if finite.size:
                print(f"  range {finite.min():.2f}–{finite.max():.2f} m, "
                      f"mean {finite.mean():.2f} m, "
                      f"{100 * finite.size / frame.size:.0f}% of pixels valid")
        else:
            flat = [v for row in frame for v in row if v == v and abs(v) != float("inf")]
            if flat:
                print(f"  range {min(flat):.2f}–{max(flat):.2f} m "
                      f"(install numpy for more)")

    if args.posed:
        rows = session.posed_images()
        print(f"\nposed images: {len(rows)}")
        usable = [r for r in rows if r["tracking"] == "normal"]
        print(f"  {len(usable)} with normal tracking, "
              f"{len(rows) - len(usable)} limited or unavailable")

        exact = sum(1 for r in rows if r["depth_dt"] == 0.0)
        approx = [r for r in rows if r["depth_dt"] not in (None, 0.0)]
        none_ = sum(1 for r in rows if r["depth"] is None)
        print(f"  depth: {exact} exact frame match, {len(approx)} nearest-in-time, {none_} missing")
        if approx:
            worst = max(abs(r["depth_dt"]) for r in approx)
            print(f"    WARNING image and depth are on different frames "
                  f"(worst offset {worst * 1000:.0f} ms)")

        for row in rows[:5]:
            p = row["pose"]
            if row["depth"] is None:
                d = "no"
            elif row["depth_dt"] == 0.0:
                d = "exact"
            else:
                d = f"{row['depth_dt'] * 1000:+.0f}ms"
            print(f"  {row['file']}  t={row['t']:.3f}  "
                  f"xyz=({p['tx']:+.2f},{p['ty']:+.2f},{p['tz']:+.2f})  "
                  f"depth={d}  {row['tracking']}")

    if args.align:
        rows = session.align_to_poses()
        print(f"\naligned {len(rows)} frames")
        for row in rows[:5]:
            loc = row.get("location")
            where = (f"{loc['lat']:.6f},{loc['lon']:.6f} "
                     f"(dt {row['location_dt']:+.2f}s)") if loc else "no fix"
            print(f"  frame {row['frame']:5d}  t={row['t']:.3f}  {where}"
                  f"  depth={'yes' if row['depth'] else 'no'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
