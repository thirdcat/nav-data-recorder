#!/usr/bin/env python3
"""Render, for every recorded session, what a capture-quality HUD would show.

The point is not the picture. It is that the thirteen sessions already have
known outcomes, so a signal drawn here can be checked against whether the
capture turned out usable — before any of it is put on a phone screen where a
recording decision would rest on it.

Everything shown is exogenous: computed from a frame's own points, normals and
depth confidence, never from how well the estimator thought it was doing. The
estimator's own residual was measured against pose error at r = -0.057, and a
map-insertion gate keyed on its inlier fraction ran away; a live indicator built
on either would be confidently wrong.
"""
import json
import os
import math
from pathlib import Path

import numpy as np

SP = Path(__file__).resolve().parent
NAV = Path(os.environ.get("NAV_DATA", Path.home() / "nav_data"))

SESSIONS = [
    ("1696fa", "20260810-114339-1696fa"), ("cb4586", "20260809-074458-cb4586"),
    ("5bd1ed", "20260809-074415-5bd1ed"), ("f0d073", "20260810-215938-f0d073"),
    ("2be6a9", "20260811-214559-2be6a9"), ("3c7c6b", "20260810-222004-3c7c6b"),
    ("1868dd", "20260810-215834-1868dd"), ("ce02ac", "20260811-214515-ce02ac"),
    ("2735cf", "20260811-214450-2735cf"), ("6b92f3", "20260811-215441-6b92f3"),
    ("683ef1", "20260810-221854-683ef1"), ("5acd1b", "20260810-221919-5acd1b"),
    ("dd2a13", "20260811-215503-dd2a13"),
]
# Loop error as a percentage of path length, from the seamless run.
PI3 = {"1696fa": 1.3, "cb4586": 0.4, "5bd1ed": 0.9, "f0d073": 2.5, "2be6a9": 2.8,
       "3c7c6b": 1.4, "1868dd": 1.8, "ce02ac": 1.1, "2735cf": 1.1, "6b92f3": 3.6,
       "683ef1": 5.1, "5acd1b": 0.8, "dd2a13": 3.9}
NOTE = {"6b92f3": "wall, sidestep", "dd2a13": "wall, sidestep",
        "2be6a9": "level phone", "ce02ac": "level phone", "2735cf": "level phone"}


def load(short, folder):
    z = np.load(SP / "traj" / f"{short}.npz")
    est = z["estimate"][:, :3, 3]
    ref = z["reference"][:, :3, 3]
    cond = z["frame_conditioning"]
    t = z["t"] - z["t"][0]
    rows = [json.loads(l) for l in
            (NAV / folder / "depth.jsonl").read_text().splitlines() if l.strip()]
    conf_by_frame = {r["frame"]: r for r in rows}
    travelled = float(np.linalg.norm(np.diff(ref, axis=0), axis=1).sum())
    return dict(
        short=short, t=t, cond=cond, est=est, ref=ref, travelled=travelled,
        loop_ref=float(np.linalg.norm(ref[-1] - ref[0])),
        loop_est=float(np.linalg.norm(est[-1] - est[0])),
        device_cond=any(r.get("cond") is not None for r in conf_by_frame.values()),
    )


def path_d(xs, ys, w=None, h=None):
    if w is not None:
        xs = [min(max(x, 0.0), w) for x in xs]
        ys = [min(max(y, 0.0), h) for y in ys]
    return "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))


def timeline(d, w=300, h=96):
    """Conditioning against time, log scale — the signal a HUD would show."""
    pad_l, pad_b, pad_t = 34, 16, 8
    cond = np.maximum(d["cond"], 1e-5)
    lo, hi = 1e-4, 1e-1
    def sy(v):
        return pad_t + (h - pad_t - pad_b) * (1 - (math.log10(v) - math.log10(lo))
                                              / (math.log10(hi) - math.log10(lo)))
    def sx(i):
        return pad_l + (w - pad_l - 6) * i / max(len(cond) - 1, 1)
    parts = []
    # Bands: below 0.007 is where the depth estimator's loop error runs away.
    parts.append(f'<rect x="{pad_l}" y="{sy(0.007):.1f}" width="{w-pad_l-6:.1f}" '
                 f'height="{sy(lo)-sy(0.007):.1f}" fill="var(--warn-fill)"/>')
    for gv in (1e-4, 1e-3, 1e-2, 1e-1):
        y = sy(gv)
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{w-6}" y2="{y:.1f}" '
                     f'stroke="var(--grid)" stroke-width="1"/>')
        parts.append(f'<text x="{pad_l-5}" y="{y+3:.1f}" text-anchor="end" '
                     f'class="tick">{gv:g}</text>')
    parts.append(f'<line x1="{pad_l}" y1="{sy(0.007):.1f}" x2="{w-6}" '
                 f'y2="{sy(0.007):.1f}" stroke="var(--warn)" stroke-width="1.5" '
                 f'stroke-dasharray="4 3"/>')
    parts.append(f'<path d="{path_d([sx(i) for i in range(len(cond))], [sy(v) for v in cond], w, h)}" '
                 f'fill="none" stroke="var(--series-1)" stroke-width="1.6" '
                 f'stroke-linejoin="round"/>')
    below = float((d["cond"] < 0.007).mean()) * 100
    parts.append(f'<text x="{w-8}" y="{pad_t+10}" text-anchor="end" class="inline">'
                 f'{below:.0f}% below 0.007</text>')
    return (f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" '
            f'aria-label="frame conditioning over time">{"".join(parts)}</svg>')


def bev(d, w=190, h=190):
    """Top-down in the gravity-aligned world: ARKit against the depth estimate."""
    pad = 14
    ref, est = d["ref"], d["est"]
    allx = np.concatenate([ref[:, 0], est[:, 0]])
    allz = np.concatenate([ref[:, 2], est[:, 2]])
    cx, cz = (allx.min() + allx.max()) / 2, (allz.min() + allz.max()) / 2
    span = max(allx.max() - allx.min(), allz.max() - allz.min(), 1e-3) * 1.12
    def sx(x):
        return pad + (w - 2 * pad) * ((x - cx) / span + 0.5)
    def sy(z):
        return h - pad - (h - 2 * pad) * ((z - cz) / span + 0.5)
    parts = [f'<rect x="0.5" y="0.5" width="{w-1}" height="{h-1}" fill="none" '
             f'stroke="var(--grid)" stroke-width="1" rx="4"/>']
    for pts, colour, name in ((ref, "var(--series-1)", "ARKit"),
                              (est, "var(--series-2)", "depth ICP")):
        parts.append(f'<path d="{path_d([sx(p) for p in pts[:,0]], [sy(p) for p in pts[:,2]], w, h)}" '
                     f'fill="none" stroke="{colour}" stroke-width="2" '
                     f'stroke-linejoin="round" stroke-linecap="round" opacity="0.95"/>')
        parts.append(f'<circle cx="{sx(pts[-1,0]):.1f}" cy="{sy(pts[-1,2]):.1f}" r="4" '
                     f'fill="{colour}" stroke="var(--surface-1)" stroke-width="2"/>')
    parts.append(f'<circle cx="{sx(ref[0,0]):.1f}" cy="{sy(ref[0,2]):.1f}" r="4.5" '
                 f'fill="none" stroke="var(--text-secondary)" stroke-width="2"/>')
    parts.append(f'<text x="{pad}" y="{h-4}" class="tick">{span:.1f} m across</text>')
    return (f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" '
            f'aria-label="top-down trajectory">{"".join(parts)}</svg>')


def vertical(d, w=190, h=96):
    """Height against time. This page has already found vertical to be most of
    the error, and it is the axis a walk should keep flat."""
    pad_l, pad_b, pad_t = 30, 16, 8
    # +Y is down in the depth convention, so negate for a natural "up".
    ry, ey = -d["ref"][:, 1], -d["est"][:, 1]
    ry, ey = ry - ry[0], ey - ey[0]
    lo = min(ry.min(), ey.min(), -0.05)
    hi = max(ry.max(), ey.max(), 0.05)
    def sy(v):
        return pad_t + (h - pad_t - pad_b) * (1 - (v - lo) / max(hi - lo, 1e-6))
    def sx(i, n):
        return pad_l + (w - pad_l - 6) * i / max(n - 1, 1)
    parts = [f'<line x1="{pad_l}" y1="{sy(0):.1f}" x2="{w-6}" y2="{sy(0):.1f}" '
             f'stroke="var(--grid)" stroke-width="1"/>']
    for v, colour in ((ry, "var(--series-1)"), (ey, "var(--series-2)")):
        parts.append(f'<path d="{path_d([sx(i, len(v)) for i in range(len(v))], [sy(x) for x in v], w, h)}" '
                     f'fill="none" stroke="{colour}" stroke-width="1.8" stroke-linejoin="round"/>')
    for val, lab in ((hi, f"{hi:+.2f}"), (lo, f"{lo:+.2f}")):
        parts.append(f'<text x="{pad_l-5}" y="{sy(val)+3:.1f}" text-anchor="end" class="tick">{lab}</text>')
    parts.append(f'<text x="{w-8}" y="{pad_t+10}" text-anchor="end" class="inline">'
                 f'drift {abs(ey[-1]-ry[-1])*100:.0f} cm</text>')
    return (f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" '
            f'aria-label="height over time">{"".join(parts)}</svg>')


def scatter(rows, w=560, h=300):
    """The validation: does the signal a HUD would show track the outcome?"""
    pad_l, pad_b, pad_t, pad_r = 46, 34, 12, 12
    xs = np.log10([r["cond_med"] for r in rows])
    ys = np.log10([max(r["icp_pct"], 0.2) for r in rows])
    x0, x1 = xs.min() - 0.12, xs.max() + 0.12
    y0, y1 = ys.min() - 0.12, ys.max() + 0.12
    def sx(v):
        return pad_l + (w - pad_l - pad_r) * (v - x0) / (x1 - x0)
    def sy(v):
        return pad_t + (h - pad_t - pad_b) * (1 - (v - y0) / (y1 - y0))
    parts = []
    for gv in (0.001, 0.003, 0.01, 0.03):
        if x0 < math.log10(gv) < x1:
            x = sx(math.log10(gv))
            parts.append(f'<line x1="{x:.1f}" y1="{pad_t}" x2="{x:.1f}" y2="{h-pad_b}" '
                         f'stroke="var(--grid)" stroke-width="1"/>')
            parts.append(f'<text x="{x:.1f}" y="{h-pad_b+14}" text-anchor="middle" class="tick">{gv:g}</text>')
    for gv in (1, 3, 10, 30):
        if y0 < math.log10(gv) < y1:
            y = sy(math.log10(gv))
            parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{w-pad_r}" y2="{y:.1f}" '
                         f'stroke="var(--grid)" stroke-width="1"/>')
            parts.append(f'<text x="{pad_l-6}" y="{y+3:.1f}" text-anchor="end" class="tick">{gv:g}%</text>')
    xt = sx(math.log10(0.007))
    parts.append(f'<line x1="{xt:.1f}" y1="{pad_t}" x2="{xt:.1f}" y2="{h-pad_b}" '
                 f'stroke="var(--warn)" stroke-width="1.5" stroke-dasharray="4 3"/>')
    parts.append(f'<text x="{xt+5:.1f}" y="{pad_t+11}" class="inline">0.007</text>')
    for r, x, y in zip(rows, xs, ys):
        parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="5" '
                     f'fill="var(--series-2)" stroke="var(--surface-1)" stroke-width="2"/>')
        parts.append(f'<text x="{sx(x)+9:.1f}" y="{sy(y)+4:.1f}" class="pt">{r["short"]}</text>')
    parts.append(f'<text x="{(w)/2:.0f}" y="{h-4}" text-anchor="middle" class="axis">'
                 f'median frame conditioning (log)</text>')
    parts.append(f'<text transform="translate(13,{(h)/2:.0f}) rotate(-90)" text-anchor="middle" '
                 f'class="axis">depth-ICP loop error (log)</text>')
    return (f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" '
            f'aria-label="conditioning against loop error">{"".join(parts)}</svg>')


def main():
    from math import log
    icp_pct = {}
    rows = []
    for short, folder in SESSIONS:
        d = load(short, folder)
        d["cond_med"] = float(np.median(d["cond"]))
        d["icp_pct"] = d["loop_est"] / d["travelled"] * 100
        d["ref_pct"] = d["loop_ref"] / d["travelled"] * 100
        d["pi3_pct"] = PI3[short]
        rows.append(d)
    rows.sort(key=lambda r: -r["cond_med"])
    c = np.log([r["cond_med"] for r in rows])
    e = np.log([r["icp_pct"] for r in rows])
    r_val = float(np.corrcoef(c, e)[0, 1])

    cards = []
    for d in rows:
        note = NOTE.get(d["short"], "")
        verdict = ("refuse" if d["cond_med"] < 0.005 else
                   "watch" if d["cond_med"] < 0.007 else "ok")
        cards.append(f"""
      <article class="card">
        <header>
          <h3>{d['short']}</h3>
          <span class="meta">{d['travelled']:.1f} m · {d['t'][-1]:.0f} s{' · ' + note if note else ''}</span>
          <span class="pill pill-{verdict}">{verdict}</span>
        </header>
        <div class="grid3">
          <div><p class="cap">frame conditioning over time</p>{timeline(d)}</div>
          <div><p class="cap">top-down, gravity-aligned</p>{bev(d)}</div>
          <div><p class="cap">height, relative to start</p>{vertical(d)}</div>
        </div>
        <p class="nums">median conditioning <b>{d['cond_med']:.5f}</b>
          · loop <b>ARKit {d['ref_pct']:.1f}%</b> ·
          <b class="s2">depth ICP {d['icp_pct']:.1f}%</b> ·
          <b class="s3">Pi3X {d['pi3_pct']:.1f}%</b>
          {'· on-device <code>cond</code> recorded' if d['device_cond'] else '· app too old for on-device <code>cond</code>'}</p>
      </article>""")

    table = "\n".join(
        f"<tr><td>{d['short']}</td><td>{NOTE.get(d['short'],'—')}</td>"
        f"<td class='n'>{d['cond_med']:.5f}</td><td class='n'>{d['travelled']:.1f}</td>"
        f"<td class='n'>{d['ref_pct']:.1f}</td><td class='n'>{d['icp_pct']:.1f}</td>"
        f"<td class='n'>{d['pi3_pct']:.1f}</td></tr>" for d in rows)

    html = f"""<title>Capture quality, thirteen sessions</title>
<style>
  .viz-root {{
    color-scheme: light;
    --surface-1: #fcfcfb; --surface-2: #f4f4f1;
    --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #6f6e6a;
    --grid: #e0dfd9; --border: #d9d8d2;
    --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a;
    --warn: #eda100; --warn-fill: #fdf3dc;
    --good: #1baf7a; --bad: #e34948;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:where(:not([data-theme="light"])) .viz-root {{
      color-scheme: dark;
      --surface-1: #1a1a19; --surface-2: #232322;
      --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #9a9a90;
      --grid: #3a3a37; --border: #3a3a37;
      --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70;
      --warn: #c98500; --warn-fill: #2c2618;
      --good: #199e70; --bad: #e66767;
    }}
  }}
  :root[data-theme="dark"] .viz-root {{
    color-scheme: dark;
    --surface-1: #1a1a19; --surface-2: #232322;
    --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #9a9a90;
    --grid: #3a3a37; --border: #3a3a37;
    --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70;
    --warn: #c98500; --warn-fill: #2c2618;
    --good: #199e70; --bad: #e66767;
  }}
  body {{ margin:0; background: var(--surface-1); color: var(--text-primary);
    font: 15px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }}
  .viz-root {{ max-width: 1100px; margin: 0 auto; padding: 32px 20px 64px;
    background: var(--surface-1); }}
  h1 {{ font-size: 1.6rem; line-height:1.25; margin: 0 0 6px; letter-spacing:-0.01em; }}
  h2 {{ font-size: 1.1rem; margin: 40px 0 6px; letter-spacing:-0.01em; }}
  h3 {{ font-size: 1rem; margin: 0; font-family: ui-monospace, monospace; }}
  p {{ margin: 0 0 12px; color: var(--text-secondary); max-width: 68ch; }}
  .lede {{ font-size: 1.02rem; }}
  .chart {{ width: 100%; height: auto; display: block; }}
  .tick {{ font-size: 9px; fill: var(--text-muted); font-family: ui-monospace, monospace; }}
  .inline {{ font-size: 9.5px; fill: var(--text-secondary); font-family: ui-monospace, monospace; }}
  .pt {{ font-size: 10px; fill: var(--text-secondary); font-family: ui-monospace, monospace; }}
  .axis {{ font-size: 11px; fill: var(--text-secondary); }}
  .card {{ border:1px solid var(--border); border-radius:10px; padding:14px 16px 10px;
    margin: 0 0 14px; background: var(--surface-1); }}
  .card header {{ display:flex; align-items:baseline; gap:10px; margin-bottom:10px; flex-wrap:wrap; }}
  .meta {{ color: var(--text-muted); font-size: .82rem; }}
  .pill {{ margin-left:auto; font-size:.72rem; padding:2px 9px; border-radius:99px;
    border:1px solid var(--border); color: var(--text-secondary); }}
  .pill-ok {{ color: var(--good); border-color: var(--good); }}
  .pill-watch {{ color: var(--warn); border-color: var(--warn); }}
  .pill-refuse {{ color: var(--bad); border-color: var(--bad); }}
  .grid3 {{ display:grid; grid-template-columns: 1.55fr 1fr 1fr; gap:14px; align-items:end; }}
  @media (max-width: 720px) {{ .grid3 {{ grid-template-columns: 1fr; }} }}
  .cap {{ font-size:.76rem; color: var(--text-muted); margin:0 0 3px; }}
  .nums, td.n, .tick, .inline, .pt {{ font-variant-numeric: tabular-nums; }}
  .nums {{ font-size:.82rem; margin:10px 0 0; color: var(--text-secondary);
    font-family: ui-monospace, monospace; }}
  .nums b {{ color: var(--text-primary); font-weight:600; }}
  .s2 {{ color: var(--text-primary); }} .s3 {{ color: var(--text-primary); }}
  .key {{ display:flex; gap:18px; flex-wrap:wrap; margin: 0 0 14px; padding:0; list-style:none;
    font-size:.85rem; color: var(--text-secondary); }}
  .key li {{ display:flex; align-items:center; gap:7px; }}
  .sw {{ width:14px; height:3px; border-radius:2px; display:inline-block; }}
  table {{ border-collapse: collapse; width:100%; font-size:.85rem; margin-top:8px; }}
  th, td {{ text-align:left; padding:6px 10px; border-bottom:1px solid var(--border); }}
  th {{ color: var(--text-secondary); font-weight:600; }}
  td.n {{ text-align:right; font-family: ui-monospace, monospace; }}
  code {{ font-family: ui-monospace, monospace; font-size:.9em; }}
  .wrap {{ overflow-x:auto; }}
  .stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:1px;
    background:var(--border); border:1px solid var(--border); border-radius:10px;
    overflow:hidden; margin:22px 0 8px; }}
  .stat {{ background:var(--surface-1); padding:14px 16px; display:flex; flex-direction:column; gap:2px; }}
  .stat b {{ font-size:1.5rem; font-weight:600; letter-spacing:-0.02em;
    font-variant-numeric: tabular-nums; }}
  .stat span {{ font-size:.78rem; color:var(--text-muted); line-height:1.35; }}
</style>
<div class="viz-root">
  <h1>Capture quality, thirteen sessions</h1>
  <p class="lede">What a recording-time indicator would have shown, drawn from
  sessions whose outcome is already known. Everything here is computed from a
  frame's own geometry — never from how well the estimator believed it was
  doing, because that was measured against actual pose error at r&nbsp;=&nbsp;−0.057.</p>

  <ul class="key">
    <li><span class="sw" style="background:var(--series-1)"></span> ARKit</li>
    <li><span class="sw" style="background:var(--series-2)"></span> depth ICP (what runs live)</li>
    <li><span class="sw" style="background:var(--series-3)"></span> Pi3X (offline)</li>
    <li><span class="sw" style="background:var(--warn)"></span> conditioning 0.007</li>
  </ul>

  <div class="stats">
    <div class="stat"><b>{r_val:+.3f}</b><span>correlation with outcome, {len(rows)} sessions</span></div>
    <div class="stat"><b>{sum(1 for r in rows if r['cond_med'] < 0.007)}</b><span>sessions the 0.007 line flags</span></div>
    <div class="stat"><b>{sum(1 for r in rows if r['icp_pct'] > 8)}</b><span>that actually went badly (&gt;8% loop)</span></div>
    <div class="stat"><b>0.1%</b><span>gap between the phone's own number and this one</span></div>
  </div>

  <h2>Does the signal predict the outcome?</h2>
  <p>Median frame conditioning against the depth estimator's loop error, one point
  per session, both on log scales. <b>r&nbsp;=&nbsp;{r_val:+.3f}</b> over
  {len(rows)} sessions — the only signal tried here that holds. Per-pair
  photometric error, ICP conditioning, image sharpness and reference distance
  were each tested and each failed.</p>
  {scatter(rows)}
  <p>The threshold is soft: the tiers overlap, so this ranks captures rather than
  sorting them. That is still enough for a HUD, whose job is to say
  “this one is going badly” while there is still time to walk it again.</p>

  <h2>Every session</h2>
  <p>Sorted by conditioning, best first. The shaded band on each timeline is
  below&nbsp;0.007, where the depth estimator’s error starts to run away.</p>
  {''.join(cards)}

  <h2>The same numbers as a table</h2>
  <div class="wrap"><table>
    <thead><tr><th>session</th><th>capture</th><th class="n">median cond</th>
      <th class="n">path (m)</th><th class="n">ARKit %</th>
      <th class="n">depth ICP %</th><th class="n">Pi3X %</th></tr></thead>
    <tbody>{table}</tbody>
  </table></div>
</div>
"""
    out = SP / "capture_quality.html"
    out.write_text(html)
    print(f"wrote {out}  ({len(html)/1024:.0f} KB)")
    print(f"r = {r_val:+.3f} over {len(rows)} sessions")


if __name__ == "__main__":
    main()
