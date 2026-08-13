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

import re

import numpy as np

SP = Path(__file__).resolve().parent
TRAJ = Path(os.environ.get("TRAJ", SP.parent / "traj"))
PI3TRAJ = Path(os.environ.get("PI3TRAJ", SP.parent / "pi3traj"))
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
# 세션마다 무엇을 찍었고 왜 찍었는지. 촬영 의도를 모르면 나쁜 점수가
# 추정기 탓인지 장면 탓인지 구분할 수 없다.
ABOUT = {
    "1696fa": ("실내 보행", "표준 촬영. 바닥이 화면에 들어오고 기하가 잘 잡힌다. "
                            "13세션 중 조건수가 가장 좋고, 여기서 못하면 어디서도 못한다."),
    "cb4586": ("실내 보행", "19 m로 두 번째로 긴 걸음. 조건이 좋은 쪽이고 "
                            "ARKit·추정기 모두 1% 근처로 끝난다."),
    "5bd1ed": ("실내 보행", "가장 긴 19 m 걸음. 조건이 좋고 결과도 좋다."),
    "f0d073": ("실내 보행", "조명이 어두운 편이라 depth confidence가 낮다. "
                            "기하보다 조명이 제약인 사례."),
    "1868dd": ("긴 실내 보행", "31 m로 가장 길다. 258 이미지 프레임이라 "
                               "13세션 중 유일하게 단일 패스(220프레임)를 넘어가고, "
                               "여러 윈도우를 이어붙여야 한다."),
    "683ef1": ("실내 보행", "ARKit 자신도 5.6%로 흔들린 세션. "
                            "기준선이 약하니 점수를 읽을 때 감안해야 한다."),
    "5acd1b": ("실내 보행", "675 프레임 중 638개가 조건수 임계 아래. "
                            "나쁜 결과가 아니라 미결정된 결과이고, 개선 대상이 아니라 "
                            "검출해서 거부할 대상이다."),
    "3c7c6b": ("실내 보행", "IMU 자세가 ARKit 대비 3.4°로 가장 나쁜 세션. "
                            "깊이 기반 추정이 16.5%로 무너진다."),
    "2735cf": ("폰 수평, 바닥 제외", "축퇴 기하를 일부러 만들려고 찍었다. "
                                     "먼 벽을 보게 되면서 depth confidence도 같이 낮아졌다."),
    "ce02ac": ("폰 수평, 바닥 제외", "같은 의도. 넷 중 조건수가 가장 나은 쪽."),
    "2be6a9": ("폰 수평, 바닥 제외", "같은 의도. 미터당 회전이 136.7°로 가장 크다."),
    "6b92f3": ("벽 옆걸음, 0.5 m", "**목표가 아닌 세션.** 벽을 마주보고 옆으로 걷는, "
                                   "깊이 기반 SLAM의 최악 조건 — 평면 하나가 화면을 채우고 "
                                   "이동 방향이 그 평면이 구속할 수 없는 축이다. "
                                   "ARKit 대비 비교와 축퇴 데이터셋 용도로만 쓴다."),
    "dd2a13": ("벽 옆걸음, 0.5 m", "**목표가 아닌 세션.** 같은 조건이고 축퇴가 더 심하다 "
                                   "(프레임의 44%가 임계 아래, 13세션 중 최고). "
                                   "여기서 성공하는 기법이 있으면 좋지만 요구사항은 아니다."),
}
NOTE = {k: v[0] for k, v in ABOUT.items()}
OUT_OF_SCOPE = {"6b92f3", "dd2a13"}


def load(short, folder):
    z = np.load(TRAJ / f"{short}.npz")
    est = z["estimate"][:, :3, 3]
    ref = z["reference"][:, :3, 3]
    cond = z["frame_conditioning"]
    t = z["t"] - z["t"][0]
    rows = [json.loads(l) for l in
            (NAV / folder / "depth.jsonl").read_text().splitlines() if l.strip()]
    conf_by_frame = {r["frame"]: r for r in rows}
    travelled = float(np.linalg.norm(np.diff(ref, axis=0), axis=1).sum())
    # The learned trajectory lives on the image frames, a sixth of the depth
    # rate, so it is a sparser path over the same walk rather than a subset of
    # the same samples.
    pi3 = None
    pi3_file = PI3TRAJ / f"{short}.npz"
    if pi3_file.exists():
        pz = np.load(pi3_file)
        pi3 = (pz["estimate"][:, :3, 3], pz["reference"][:, :3, 3])
    return dict(
        pi3=pi3,
        short=short, t=t, cond=cond, est=est, ref=ref, travelled=travelled,
        loop_ref=float(np.linalg.norm(ref[-1] - ref[0])),
        loop_est=float(np.linalg.norm(est[-1] - est[0])),
        device_cond=any(r.get("cond") is not None for r in conf_by_frame.values()),
    )


def align_to(estimate, reference):
    """Put the estimate in the reference's frame, rotation and offset only.

    The estimator starts at the identity while ARKit's first pose carries the
    tilt the phone was actually held at, so the two trajectories live in frames
    that differ by that rotation. Drawn raw they sit 3.7 m apart on a session
    whose real disagreement is 5 cm. No scale is fitted — both are metric, and
    fitting one would hide the failure most worth seeing. This is the same
    alignment the reported ATE uses, so the picture and the number agree.
    """
    ec = estimate - estimate.mean(0)
    rc = reference - reference.mean(0)
    U, _, Vt = np.linalg.svd(ec.T @ rc)
    R = U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    return ec @ R + reference.mean(0)


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
                 f'{below:.0f}% 가 0.007 아래</text>')
    return (f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" '
            f'aria-label="frame conditioning, 시간축">{"".join(parts)}</svg>')


def bev(d, w=190, h=190):
    """Top-down in the gravity-aligned world: ARKit against the depth estimate."""
    pad = 14
    ref, est = d["ref"], align_to(d["est"], d["ref"])
    extra = [align_to(*d["pi3"])] if d.get("pi3") is not None else []
    allx = np.concatenate([ref[:, 0], est[:, 0]] + [e[:, 0] for e in extra])
    allz = np.concatenate([ref[:, 2], est[:, 2]] + [e[:, 2] for e in extra])
    cx, cz = (allx.min() + allx.max()) / 2, (allz.min() + allz.max()) / 2
    span = max(allx.max() - allx.min(), allz.max() - allz.min(), 1e-3) * 1.12
    def sx(x):
        return pad + (w - 2 * pad) * ((x - cx) / span + 0.5)
    def sy(z):
        return h - pad - (h - 2 * pad) * ((z - cz) / span + 0.5)
    parts = [f'<rect x="0.5" y="0.5" width="{w-1}" height="{h-1}" fill="none" '
             f'stroke="var(--grid)" stroke-width="1" rx="4"/>']
    paths = [(ref, "var(--series-1)", "ARKit"), (est, "var(--series-2)", "depth ICP")]
    if d.get("pi3") is not None:
        p_est, p_ref = d["pi3"]
        paths.append((align_to(p_est, p_ref), "var(--series-3)", "Pi3X"))
    for pts, colour, name in paths:
        parts.append(f'<path d="{path_d([sx(p) for p in pts[:,0]], [sy(p) for p in pts[:,2]], w, h)}" '
                     f'fill="none" stroke="{colour}" stroke-width="2" '
                     f'stroke-linejoin="round" stroke-linecap="round" opacity="0.95"/>')
        parts.append(f'<circle cx="{sx(pts[-1,0]):.1f}" cy="{sy(pts[-1,2]):.1f}" r="4" '
                     f'fill="{colour}" stroke="var(--surface-1)" stroke-width="2"/>')
    parts.append(f'<circle cx="{sx(ref[0,0]):.1f}" cy="{sy(ref[0,2]):.1f}" r="4.5" '
                 f'fill="none" stroke="var(--text-secondary)" stroke-width="2"/>')
    parts.append(f'<text x="{pad}" y="{h-4}" class="tick">가로 {span:.1f} m</text>')
    return (f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" '
            f'aria-label="위에서 본 궤적">{"".join(parts)}</svg>')


def vertical(d, w=190, h=96):
    """Height against time. This page has already found vertical to be most of
    the error, and it is the axis a walk should keep flat."""
    pad_l, pad_b, pad_t = 30, 16, 8
    # +Y is down in the depth convention, so negate for a natural "up".
    aligned = align_to(d["est"], d["ref"])
    ry, ey = -d["ref"][:, 1], -aligned[:, 1]
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
                 f'차이 {abs(ey[-1]-ry[-1])*100:.0f} cm</text>')
    return (f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" '
            f'aria-label="높이, 시간축">{"".join(parts)}</svg>')


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
                 f'class="axis">깊이 ICP 루프 오차 (로그)</text>')
    return (f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" '
            f'aria-label="conditioning 과 루프 오차">{"".join(parts)}</svg>')


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
    # 범위 밖 두 세션을 뒤로 뺀다. conditioning 순서에 섞이면 목표를
    # 못 맞춘 세션으로 읽히는데, 애초에 목표가 아니다.
    rows.sort(key=lambda r: (r["short"] in OUT_OF_SCOPE, -r["cond_med"]))
    scoped = [r for r in rows if r["short"] not in OUT_OF_SCOPE]
    unscoped = [r for r in rows if r["short"] in OUT_OF_SCOPE]
    c = np.log([r["cond_med"] for r in rows])
    e = np.log([r["icp_pct"] for r in rows])
    r_val = float(np.corrcoef(c, e)[0, 1])

    cards = []
    for d in rows:
        if d is unscoped[0] if unscoped else False:
            cards.append("""
      <h2>범위 밖 — 이 둘은 만족시킬 필요가 없다</h2>
      <p>벽을 마주보고 옆으로 걷는 촬영이다. 깊이만으로는 이동 축이 <b>원리적으로</b>
      관측되지 않으므로, 여기서 깊이 기반 추정기가 틀리는 것은 결함이 아니라 예정된
      결과다. 위 열한 세션과 같은 기준으로 읽으면 안 된다 — ARKit 과의 비교용, 그리고
      축퇴 데이터셋용으로만 둔다.</p>""")
        kind, about = ABOUT.get(d["short"], ("", ""))
        about = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", about)
        out = d["short"] in OUT_OF_SCOPE
        verdict = ("refuse" if d["cond_med"] < 0.005 else
                   "watch" if d["cond_med"] < 0.007 else "ok")
        cards.append(f"""
      <article class="card">
        <header>
          <h3>{d['short']}</h3>
          <span class="meta">{d['travelled']:.1f} m · {d['t'][-1]:.0f} s · {kind}</span>
          {'<span class="pill pill-out">범위 밖</span>' if out else ''}
          <span class="pill pill-{verdict}">{verdict}</span>
        </header>
        <p class="about">{about}</p>
        <div class="grid3">
          <div><p class="cap">frame conditioning, 시간축</p>{timeline(d)}</div>
          <div><p class="cap">위에서 본 궤적 · 중력 정렬 · 강체 정렬</p>{bev(d)}</div>
          <div><p class="cap">높이, 출발점 기준</p>{vertical(d)}</div>
        </div>
        <p class="nums">conditioning 중앙값 <b>{d['cond_med']:.5f}</b>
          · 루프 <b>ARKit {d['ref_pct']:.1f}%</b> ·
          <b class="s2">depth ICP {d['icp_pct']:.1f}%</b> ·
          <b class="s3">Pi3X {d['pi3_pct']:.1f}%</b>
          {'· 기기에서 <code>cond</code> 기록됨' if d['device_cond'] else '· 앱 버전이 낮아 <code>cond</code> 없음'}</p>
      </article>""")

    sep = ('<tr><td colspan="7" class="sep">범위 밖 — 만족 대상 아님, '
           '깊이만으로는 이동 축이 관측되지 않는 촬영</td></tr>')
    def row(d):
        return (
        f"<tr><td>{d['short']}</td><td>{NOTE.get(d['short'],'—')}</td>"
        f"<td class='n'>{d['cond_med']:.5f}</td><td class='n'>{d['travelled']:.1f}</td>"
        f"<td class='n'>{d['ref_pct']:.1f}</td><td class='n'>{d['icp_pct']:.1f}</td>"
        f"<td class='n'>{d['pi3_pct']:.1f}</td></tr>")
    table = "\n".join([row(d) for d in scoped] + ([sep] if unscoped else [])
                      + [row(d) for d in unscoped])

    html = f"""<title>촬영 품질, 13 세션</title>
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
  .card header {{ display:flex; align-items:baseline; gap:10px; margin-bottom:8px; flex-wrap:wrap; }}
  .card header .pill:first-of-type {{ margin-left:auto; }}
  .meta {{ color: var(--text-muted); font-size: .82rem; }}
  .pill {{ font-size:.72rem; padding:2px 9px; border-radius:99px;
    border:1px solid var(--border); color: var(--text-secondary); }}
  .pill-ok {{ color: var(--good); border-color: var(--good); }}
  .pill-watch {{ color: var(--warn); border-color: var(--warn); }}
  .pill-refuse {{ color: var(--bad); border-color: var(--bad); }}
  .pill-out {{ color: var(--text-primary); border-color: var(--text-secondary);
    background: var(--surface-2); margin-left:auto; }}
  .about {{ font-size:.85rem; margin:0 0 12px; max-width:none; }}
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
  td.sep {{ color: var(--text-primary); background: var(--surface-2);
    font-size:.8rem; border-top:1px solid var(--text-secondary); }}
  code {{ font-family: ui-monospace, monospace; font-size:.9em; }}
  .wrap {{ overflow-x:auto; }}
  ul.plain {{ margin:0 0 12px; padding-left:20px; color:var(--text-secondary);
    font-size:.92rem; max-width:68ch; }}
  ul.plain li {{ margin-bottom:4px; }}
  .stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:1px;
    background:var(--border); border:1px solid var(--border); border-radius:10px;
    overflow:hidden; margin:22px 0 8px; }}
  .stat {{ background:var(--surface-1); padding:14px 16px; display:flex; flex-direction:column; gap:2px; }}
  .stat b {{ font-size:1.5rem; font-weight:600; letter-spacing:-0.02em;
    font-variant-numeric: tabular-nums; }}
  .stat span {{ font-size:.78rem; color:var(--text-muted); line-height:1.35; }}
</style>
<div class="viz-root">
  <h1>촬영 품질, 13 세션</h1>
  <p class="lede">촬영 중에 화면이 보여줬어야 할 것을, 결과를 이미 아는 세션들로
  그린 것. 여기 있는 값은 모두 프레임 자신의 기하에서 나온다 — 추정기가 스스로
  잘하고 있다고 믿는 값은 쓰지 않는다. 그 값은 실제 포즈 오차와
  r&nbsp;=&nbsp;−0.057 로 무관하다고 측정됐다.</p>

  <ul class="key">
    <li><span class="sw" style="background:var(--series-1)"></span> ARKit</li>
    <li><span class="sw" style="background:var(--series-2)"></span> 깊이 ICP (실시간으로 도는 것)</li>
    <li><span class="sw" style="background:var(--series-3)"></span> Pi3X (오프라인, 5&nbsp;Hz)</li>
    <li><span class="sw" style="background:var(--warn)"></span> conditioning 0.007</li>
  </ul>

  <div class="stats">
    <div class="stat"><b>{r_val:+.3f}</b><span>결과와의 상관, {len(rows)} 세션</span></div>
    <div class="stat"><b>{sum(1 for r in rows if r['cond_med'] < 0.007)}</b><span>0.007 선이 표시하는 세션</span></div>
    <div class="stat"><b>{sum(1 for r in rows if r['icp_pct'] > 8)}</b><span>실제로 나빴던 세션 (루프 &gt;8%)</span></div>
    <div class="stat"><b>0.1%</b><span>폰이 계산한 값과 이 값의 차이</span></div>
  </div>

  <h2>frame conditioning 이란</h2>
  <p>깊이 프레임 하나가 카메라의 위치를 <b>얼마나 결정하는지</b>를 나타내는 값이다.
  보이는 표면의 법선을 모아 point-to-plane Hessian 을 만들고, 가장 약한 축과 가장
  강한 축의 고윳값 비를 취한다. 0 에 가까울수록 기하가 어느 방향을 못 본다는 뜻이다.</p>
  <ul class="plain">
    <li><b>평평한 벽 하나가 화면을 채우면</b> 벽에 수직인 방향만 잡히고, 벽을 따라
      미끄러지는 두 방향은 깊이만으로 알 수 없다 — 값이 작아진다.</li>
    <li><b>모서리나 문틀처럼 서로 다른 방향의 면이 보이면</b> 세 축이 모두 잡힌다 —
      값이 커진다.</li>
    <li>바닥이 화면에 들어오는 것만으로도 큰 차이가 난다. 축퇴 세션을 만들려고
      폰을 수평으로 들어 바닥을 뺀 이유가 그것이다.</li>
  </ul>
  <p>중요한 성질은 <b>프레임 자신의 점과 법선만으로 계산된다</b>는 것이다. 추정기가
  잘하고 있다고 믿는 값이 아니라서, 추정기가 스스로를 속일 수 없다. 그리고 앱이
  0.1.64 부터 촬영 중에 이미 계산해 <code>depth.jsonl</code> 에 기록하고 있다 —
  같은 프레임에서 Python 구현과 0.1~0.2% 안에서 일치한다.</p>

  <h2>이 신호가 결과를 예측하는가</h2>
  <p>세션별 conditioning 중앙값과 깊이 기반 추정기의 루프 오차, 양쪽 모두 로그
  축. {len(rows)} 세션에서 <b>r&nbsp;=&nbsp;{r_val:+.3f}</b> 로, 여기서 시도한 신호
  중 유일하게 성립한다. 프레임쌍 photometric 오차, ICP 조건수, 이미지 선명도,
  참조 거리는 각각 시험했고 각각 실패했다.</p>
  {scatter(rows)}
  <p>임계는 무르다. 좋은 쪽과 나쁜 쪽이 0.007 근처에서 겹치므로 이 값은 촬영을
  <b>줄 세우지 정렬하지는 못한다</b>. 그래도 화면 표시에는 충분하다 — 다시 걸을
  시간이 남아 있을 때 “이건 잘 안 되고 있다”를 말하는 것이 그 일이기 때문이다.</p>

  <h2>세션별</h2>
  <p>conditioning 이 좋은 순. 시간축의 음영은 0.007 아래 구간으로, 깊이 기반
  추정기의 오차가 커지기 시작하는 곳이다. 위에서 본 궤적은 세 가지를 겹쳐 그렸고,
  추정 궤적은 ARKit 의 좌표계로 <b>강체 정렬</b>했다 — 추정기는 단위행렬에서
  출발하고 ARKit 의 첫 포즈는 폰을 들고 있던 기울기를 담고 있어서, 정렬하지 않으면
  실제로 5&nbsp;cm 차이인 궤적이 3.7&nbsp;m 떨어져 보인다. ATE 가 쓰는 것과 같은
  정렬이라 그림과 숫자가 같은 것을 말한다. 크기는 맞추지 않는다 — 둘 다 미터
  단위이고, 맞추면 가장 봐야 할 실패가 숨는다.</p>
  <p><b>범위 밖</b>인 두 세션은 이 목록에서 빠져 <b>맨 뒤</b>에 따로 있다. 벽을
  마주보고 옆으로 걷는 촬영이라 깊이만으로는 이동 축이 원리적으로 관측되지 않아,
  아래 열한 세션과 같은 기준으로 읽을 수 있는 촬영이 아니다.</p>
  {''.join(cards)}

  <h2>같은 숫자를 표로</h2>
  <div class="wrap"><table>
    <thead><tr><th>세션</th><th>촬영</th><th class="n">cond 중앙값</th>
      <th class="n">경로 (m)</th><th class="n">ARKit %</th>
      <th class="n">깊이 ICP %</th><th class="n">Pi3X %</th></tr></thead>
    <tbody>{table}</tbody>
  </table></div>
</div>
"""
    out = Path(os.environ.get("VIS_OUT", SP / "capture_quality.html"))
    out.write_text(html)
    print(f"wrote {out}  ({len(html)/1024:.0f} KB)")
    print(f"r = {r_val:+.3f} over {len(rows)} sessions")


if __name__ == "__main__":
    main()
