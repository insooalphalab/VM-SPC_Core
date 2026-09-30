"""교과서 차트 패턴 4종(하이 타이트 플래그·강세 깃발·컵앤핸들·역헤드앤숄더) — 검증이력 9.30 사전 등록 기준 그대로.

체결 규칙은 research/validate_flag.py 의 simulate(손절·5일 시간 손절·최대 20일), 요약은 validate_box.summarize.
책식 성공률(20일 안 고가 +10% 한 번이라도, 손절 무시)을 패턴·무작위 둘 다 함께 낸다.

  python research/validate_patterns.py            → results/pattern_validation.json (센서 + 표본 밖 코스피)
  python research/validate_patterns.py --kosdaq   → results/pattern_validation_kosdaq.json (재현 확인)
  python research/validate_patterns.py --kospi2   → results/pattern_validation_kospi2.json (9.38 새 표본)
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "vm_predict", _ROOT / "scenario", _ROOT / "research"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import json
import sys

import numpy as np
import pandas as pd

import validate_box as vb
import validate_flag as vf
from box_rules import MAX_HOLD
from v2_config import LONG_HISTORY, results_dir, sensor_universe
from v2_datastore import load_bars

START = vf.START
TOUCH = 0.10


def _event(bars, pattern, e, stop, level, sim, **extra) -> dict:
    o = bars["open"].to_numpy(float)
    return {"pattern": pattern, "e": e, "entry_date": bars.index[e], "stop_pct": 1 - stop / o[e],
            "level_pct": level / o[e] - 1, "ret": sim[0], "days": sim[1], "how": sim[2], **extra}


def _first(bars) -> int:
    return int(bars.index.searchsorted(START))


# ① 하이 타이트 플래그 ─────────────────────────────────────────
def find_htf(bars) -> list[dict]:
    o, h, l, c = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    out, busy, seen = [], -1, set()
    for j in range(max(_first(bars), 70), len(c) - 1):
        lo = j - 26
        pk = lo + int(np.argmax(h[lo:j]))
        k = j - 1 - pk                                  # 깃발 일수(pk 다음 날 ~ j 전날)
        if not (15 <= k <= 25) or pk in seen or c[j] <= h[pk]:
            continue
        if h[pk] / l[pk - 40:pk].min() < 1.9:
            continue
        flag = slice(pk + 1, j)
        if (c[flag] < 0.75 * h[pk]).any():
            continue
        seen.add(pk)
        e, stop = j + 1, l[flag].min()
        if e <= busy or not stop < o[e]:
            continue
        sim = vf.simulate(o, h, l, c, e, stop, o[e])
        if sim:
            busy = e + sim[1] - 1
            out.append(_event(bars, "① 하이 타이트 플래그", e, stop, o[e], sim))
    return out


# ② 강세 깃발(9.29) ────────────────────────────────────────────
def find_flag(bars) -> list[dict]:
    out = []
    for ev in vf.find(bars):
        ev = dict(ev)
        ev["pattern"] = f"② 강세 깃발 {ev.pop('entry')}"
        ev.pop("code", None)
        out.append(ev)
    return out


# ③ 컵앤핸들 ─────────────────────────────────────────────────
def find_cup(bars) -> list[dict]:
    o, h, l, c, v = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
    out, busy, seen = [], -1, set()
    for j in range(max(_first(bars), 160), len(c) - 1):
        lo = j - 11
        r = lo + int(np.argmax(h[lo:j]))
        k = j - 1 - r
        if not (4 <= k <= 10) or r in seen or c[j] <= h[r]:
            continue
        handle = slice(r + 1, j)
        if l[handle].min() < 0.88 * h[r]:
            continue
        a0, a1 = r - 120, r - 30
        a = a0 + int(np.argmax(h[a0:a1 + 1]))
        if abs(h[r] / h[a] - 1) > 0.05 or h[a + 1:r].max(initial=0) > max(h[a], h[r]):
            continue
        bot = a + 1 + int(np.argmin(l[a + 1:r]))
        depth = 1 - l[bot] / h[a]
        pos = (bot - a) / (r - a)
        if not (0.12 <= depth <= 0.35 and 0.2 <= pos <= 0.8):
            continue
        seen.add(r)
        e, stop = j + 1, l[handle].min()
        if e <= busy or not stop < o[e]:
            continue
        sim = vf.simulate(o, h, l, c, e, stop, o[e])
        if sim:
            busy = e + sim[1] - 1
            dry = bool(v[handle].mean() < v[r - 20:r].mean())
            out.append(_event(bars, "③ 컵앤핸들", e, stop, o[e], sim, dried=dry))
    return out


# ④ 역헤드앤숄더 ──────────────────────────────────────────────
def find_ihs(bars) -> list[dict]:
    o, h, l, c = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    n = len(c)
    piv = [i for i in range(5, n - 5) if l[i] == l[i - 5:i + 6].min()]
    out, busy, used = [], -1, set()
    first = _first(bars)
    for i3, p3 in enumerate(piv):
        if p3 + 5 < first:
            continue
        found = None
        for i2 in range(i3 - 1, -1, -1):
            p2 = piv[i2]
            if p3 - p2 > 60:
                break
            if p3 - p2 < 10 or not l[p2] < l[p3]:
                continue
            for i1 in range(i2 - 1, -1, -1):
                p1 = piv[i1]
                if p2 - p1 > 60:
                    break
                if p2 - p1 < 10 or not l[p2] < l[p1]:
                    continue
                if l[p2] > 0.95 * min(l[p1], l[p3]) or abs(l[p1] / l[p3] - 1) > 0.10:
                    continue
                found = (p1, p2)
                break
            if found:
                break
        if not found or p3 in used:
            continue
        p1, p2 = found
        neck = h[p1:p3 + 1].max()
        for j in range(p3 + 5, min(p3 + 26, n - 1)):
            if c[j] < l[p3]:
                break
            if c[j] > neck:
                used.add(p3)
                e, stop = j + 1, l[p3]
                if e > busy and stop < o[e]:
                    sim = vf.simulate(o, h, l, c, e, stop, o[e])
                    if sim:
                        busy = e + sim[1] - 1
                        out.append(_event(bars, "④ 역헤드앤숄더", e, stop, o[e], sim))
                break
    return out


FINDERS = (find_htf, find_flag, find_cup, find_ihs)


def controls(bars, ev, rng) -> tuple[float, float]:
    """무작위 날짜 N_CTRL개, 같은 손절·기준선 거리 → (평균 수익, 책식 성공률)."""
    o, h, l, c = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    lo, hi = _first(bars), len(c) - MAX_HOLD
    rets, touch = [], 0
    for d in rng.integers(lo, hi, vb.N_CTRL):
        d = int(d)
        rets.append(vf.simulate(o, h, l, c, d, o[d] * (1 - ev["stop_pct"]), o[d] * (1 + ev["level_pct"]))[0])
        touch += h[d:d + MAX_HOLD].max() >= o[d] * (1 + TOUCH)
    return float(np.mean(rets)), touch / vb.N_CTRL


def run(codes: list[str], label: str, rng) -> pd.DataFrame:
    rows = []
    for code in codes:
        bars = load_bars(LONG_HISTORY, code)
        if bars is None or len(bars) < 300:
            continue
        bars = bars[(bars[["open", "high", "low", "close"]] > 0).all(1)]
        h = bars["high"].to_numpy(float)
        o = bars["open"].to_numpy(float)
        for fn in FINDERS:
            for ev in fn(bars):
                e = ev["e"]
                ev["touch"] = bool(h[e:e + MAX_HOLD].max() >= o[e] * (1 + TOUCH))
                ev["ctrl_ret"], ev["ctrl_touch"] = controls(bars, ev, rng)
                ev["ctrl_tgt"], ev["ctrl_days"], ev["ctrl_to"], ev["tgt_pct"] = 0.0, 0.0, 0.0, np.nan
                ev["code"], ev["universe"] = code, label
                rows.append(ev)
    df = pd.DataFrame(rows)
    cal = load_bars(LONG_HISTORY, "069500").index
    df["block"] = cal.searchsorted(df["entry_date"]) // vb.BLOCK
    return df


def _codes(f: str) -> list[str]:
    return json.loads((_ROOT / "data" / LONG_HISTORY / f).read_text(encoding="utf-8"))["codes"]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    rng = np.random.default_rng(0)
    kosdaq = "--kosdaq" in sys.argv
    kospi2 = "--kospi2" in sys.argv
    if kospi2:
        df = run(_codes("oos_kospi2_codes.json"), "oos_kospi2", rng)
    elif kosdaq:
        df = run(_codes("oos_kosdaq_codes.json"), "oos_kosdaq", rng)
    else:
        df = pd.concat([run(sorted(sensor_universe()), "stocks", rng), run(_codes("oos_codes.json"), "oos_kospi", rng)],
                       ignore_index=True)
    res = {}
    for (u, pat), g in df.groupby(["universe", "pattern"]):
        s = vb.summarize(g, rng, verdict=True)
        s.update({"win_rate": round(float((g["ret"] - vb.COST > 0).mean()), 4),
                  "book_success": round(float(g["touch"].mean()), 4),
                  "ctrl_book_success": round(float(g["ctrl_touch"].mean()), 4),
                  "exit_mix": g["how"].value_counts(normalize=True).round(3).to_dict()})
        if "dried" in g and g["dried"].notna().any():
            s["dried_excess"] = {str(k): round(float((x["ret"] - x["ctrl_ret"]).mean()), 4)
                                 for k, x in g.groupby("dried")}
        res[f"{u}/{pat}"] = s
    verdict = {}
    if not (kosdaq or kospi2):
        for pat in sorted(df["pattern"].unique()):
            a, b = res.get(f"stocks/{pat}", {}), res.get(f"oos_kospi/{pat}", {})
            ok = [x.get("excess_ci", [0])[0] > 0 for x in (a, b)]
            verdict[pat] = "통과" if all(ok) else "단서" if any(ok) else "채택 안 함"
    name = ("pattern_validation_kospi2.json" if kospi2 else "pattern_validation_kosdaq.json" if kosdaq
            else "pattern_validation.json")
    df.to_csv(results_dir() / name.replace(".json", "_events.csv"), index=False)
    (results_dir() / name).write_text(json.dumps({"verdict": verdict, **res}, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, s in res.items():
        print(f"{k:26s} n={s['n']:5d} 독립 {s['n_indep']:3d} | 승률 {s['win_rate']:.0%} | 책식 성공 {s['book_success']:.0%}"
              f" (무작위 {s['ctrl_book_success']:.0%}) | 초과 {s['excess']:+.2%} [{s['excess_ci'][0]:+.2%}, {s['excess_ci'][1]:+.2%}]"
              f" | 손절거리 {s['stop_pct']:.1%} {s.get('dried_excess', '')}")
    print(verdict)
    return 0


if __name__ == "__main__":
    sys.exit(main())
