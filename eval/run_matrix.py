#!/usr/bin/env python3
"""Run depth_odometry over (session x arm) and collect the numbers that decide.

Every arm sees the same sessions, the same frames and the same thread count, so
the only thing that varies between columns is the flag under test.
"""
import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NAV = Path(os.environ.get("NAV_DATA", Path.home() / "nav_data"))

# The eight sessions docs/POSE.md scores. Order matches the table there.
SESSIONS = [
    "20260810-114339-1696fa",
    "20260810-215938-f0d073",
    "20260810-215834-1868dd",
    "20260810-221854-683ef1",
    "20260809-074458-cb4586",
    "20260810-222004-3c7c6b",
    "20260809-074415-5bd1ed",
    "20260810-221919-5acd1b",
]

PATTERNS = {
    "loop": re.compile(r"end-start distance ([\d.]+) m \(ARKit ([\d.]+) m"),
    "ate": re.compile(r"rms ([\d.]+) cm\s+max ([\d.]+) cm"),
    "travelled": re.compile(r"ARKit travelled ([\d.]+) m over ([\d.]+) s"),
    "rpe": re.compile(r"translation\s+median ([\d.]+) cm\s+p90 ([\d.]+) cm"),
    "rpe_rot": re.compile(r"rotation\s+median ([\d.]+)°\s+p90 ([\d.]+)°"),
    "frames": re.compile(r"^(\S+): (\d+) depth frames"),
    "keyframes": re.compile(r"map: (\d+) keyframes of (\d+) frames, (\d+) with images \(([\d.]+)%\), (\d+) voxels"),
    "photo": re.compile(r"photometric: evaluated (\d+)/(\d+) frames \((\d+)%\), evaluations (\d+), accepted LM steps (\d+)"),
    "photo_pairs": re.compile(r"pairs (\d+) available/(\d+) selected/(\d+) solved"),
    "photo_ref": re.compile(r"photometric reference: (\S+) (\d+) frames, no reference (\d+) skipped"),
    "photo_dist": re.compile(r"photometric reference distance: median ([\d.]+) m\s+angle median ([\d.]+)°"),
    "cond": re.compile(r"frame-only conditioning\s+median ([\d.]+)\s+p10 ([\d.]+)\s+degenerate (\d+)/(\d+)"),
    "residual": re.compile(r"point-to-plane residual\s+estimate ([\d.]+) mm\s+arkit ([\d.]+) mm"),
    "inliers": re.compile(r"ICP inliers: median (\d+)%"),
}

# Conditioning-banded relative pose error, e.g.
#   cond >= 0.05 757 frames  translation median 0.4 cm p90 0.8 cm  rotation median 0.06°
BAND = re.compile(
    r"(cond < 0\.01|0\.01 <= cond < 0\.05|cond >= 0\.05)\s+(\d+) frames"
    r"(?:\s+translation median ([\d.]+) cm p90 ([\d.]+) cm)?")


def parse(text):
    out = {}
    for key, pat in PATTERNS.items():
        m = pat.search(text)
        if m:
            out[key] = [float(g) if re.fullmatch(r"[\d.]+", g) else g
                        for g in m.groups()]
    bands = {}
    for m in BAND.finditer(text):
        name, count, med, p90 = m.groups()
        bands[name] = {
            "frames": int(count),
            "median_cm": float(med) if med else None,
            "p90_cm": float(p90) if p90 else None,
        }
    if bands:
        out["bands"] = bands
    return out


def run_one(session, arm_name, flags, outdir, timeout):
    log = outdir / f"{arm_name}__{session}.log"
    cmd = [sys.executable, str(REPO / "tools" / "depth_odometry.py"),
           str(NAV / session)] + flags
    env = dict(os.environ)
    # Same thread budget for every arm, so the columns stay comparable and the
    # pool does not oversubscribe the machine.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[var] = "2"
    start = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, env=env, cwd=str(REPO))
        text = proc.stdout + proc.stderr
        rc = proc.returncode
    except subprocess.TimeoutExpired as exc:
        text = (exc.stdout or b"").decode(errors="replace") + "\n!! TIMEOUT"
        rc = -9
    elapsed = time.time() - start
    log.write_text(text)
    record = {"session": session, "arm": arm_name, "flags": flags,
              "returncode": rc, "seconds": round(elapsed, 1),
              "log": str(log)}
    record.update(parse(text))
    return record


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True,
                    help="path to a JSON file mapping arm name -> flag list")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sessions", nargs="*", default=SESSIONS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=5400)
    args = ap.parse_args(argv)

    arms = json.loads(Path(args.arms).read_text())
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    jobs = [(s, name, flags) for name, flags in arms.items()
            for s in args.sessions]
    print(f"{len(jobs)} runs, {args.workers} at a time", flush=True)

    results = []
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(run_one, s, n, f, outdir, args.timeout): (s, n)
                   for s, n, f in jobs}
        for fut in concurrent.futures.as_completed(futures):
            rec = fut.result()
            results.append(rec)
            loop = rec.get("loop")
            mark = "ok " if rec["returncode"] == 0 else "FAIL"
            print(f"  {mark} {rec['arm']:<28} {rec['session'][-6:]}  "
                  f"loop={loop[0] if loop else '?'}  {rec['seconds']}s",
                  flush=True)

    results.sort(key=lambda r: (r["arm"], SESSIONS.index(r["session"])
                                if r["session"] in SESSIONS else 99))
    (outdir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {outdir / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
