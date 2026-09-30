"""급등 후 깃발(거래대금 마름) 검증 — 검증이력 9.29 사전 등록 기준 그대로.

깃대 : 고점일 pk 의 20일 수익률 +30% 이상, pk 포함 20일 안에 거래대금이 직전 20일 평균의 3배 이상인 날(급등일 s)
깃발 : pk 다음 날부터 3~7일, 매일 종가가 pk 고가의 -10% 이내이고 10일선 위
마름 : 깃발 구간 평균 거래대금 <= 급등일 거래대금의 1/3  (안 마른 깃발 = 비교 대상 ②)
진입 A(확인) : 깃발 3일 이상 뒤 종가가 pk 고가 위 → 다음 날 시가. 5일째 종가까지 진입가 위로 한 번도 못 가면 정리
진입 B(눌림) : 깃발 조건 처음 충족(3일째) → 다음 날 시가. 5일째 종가까지 pk 고가를 못 넘으면 정리
손절 = 깃발 구간 저점, 최대 20일(그날 종가), 목표 없음. 대조군 ① = 같은 종목 무작위 날짜·같은 손절·기준선 거리.
판정: 마른 깃발이 센서 191종목과 표본 밖(코스피 200 + 코스닥 200) 둘 다 초과수익 신뢰구간 하한 > 0.

  python research/validate_flag.py      → results/flag_validation.json
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
from box_rules import MAX_HOLD
from v2_config import LONG_HISTORY, results_dir, sensor_universe
from v2_datastore import load_bars

POLE_RET, POLE_WIN, SPIKE_X = 0.30, 20, 3.0
FLAG_MIN, FLAG_MAX, FLAG_DD, MA = 3, 7, 0.10, 10
DRY = 1 / 3
TIME_STOP = 5
START = pd.Timestamp("2016-01-01")


def simulate(o, h, l, c, e: int, stop: float, level: float):
    """e일 시가 진입. 손절(갭이면 시가) · TIME_STOP일째 종가까지 종가가 level 을 못 넘으면 정리 · MAX_HOLD일 종가."""
    if e + MAX_HOLD - 1 >= len(c):
        return None
    entry, best = o[e], -np.inf
    for j in range(e, e + MAX_HOLD):
        if j > e and o[j] <= stop:
            return o[j] / entry - 1, j - e + 1, "stop"
        if l[j] <= stop:
            return stop / entry - 1, j - e + 1, "stop"
        best = max(best, c[j])
        if j - e + 1 == TIME_STOP and best <= level:
            return c[j] / entry - 1, TIME_STOP, "time"
    return c[e + MAX_HOLD - 1] / entry - 1, MAX_HOLD, "timeout"


def find(bars: pd.DataFrame) -> list[dict]:
    o, h, l, c, v = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
    val = c * v
    avg = pd.Series(val).rolling(POLE_WIN).mean().shift(1).to_numpy()
    ma = pd.Series(c).rolling(MA).mean().to_numpy()
    first = max(int(bars.index.searchsorted(START)), POLE_WIN * 2)
    out, busy, seen = [], {"A": -1, "B": -1}, set()
    for b in range(first, len(c) - 1):
        lo = b - FLAG_MAX
        pk = lo + int(np.argmax(h[lo:b + 1]))
        k = b - pk
        if k != FLAG_MIN or pk in seen:              # 깃발 3일째에서 한 번만 판정
            continue
        if not c[pk] / c[pk - POLE_WIN] - 1 >= POLE_RET:
            continue
        w = np.arange(pk - POLE_WIN + 1, pk + 1)
        spikes = w[val[w] >= SPIKE_X * avg[w]]
        if not len(spikes):
            continue
        s = spikes[np.argmax(val[spikes])]
        top = h[pk]

        def ok(j):
            return c[j] >= (1 - FLAG_DD) * top and c[j] >= ma[j]
        if not all(ok(j) for j in range(pk + 1, b + 1)):
            continue
        seen.add(pk)
        flag = np.arange(pk + 1, b + 1)
        base = {"dried": bool(val[flag].mean() <= DRY * val[s]), "code": None}
        # B: 눌림 — 3일째 종가 확인 후 다음 날 시가
        e = b + 1
        stop = l[flag].min()
        if e > busy["B"] and stop < o[e]:
            sim = simulate(o, h, l, c, e, stop, top)
            if sim is not None:
                busy["B"] = e + sim[1] - 1
                out.append({**base, "entry": "B", "e": e, "entry_date": bars.index[e], "stop_pct": 1 - stop / o[e],
                            "level_pct": top / o[e] - 1, "ret": sim[0], "days": sim[1], "how": sim[2]})
        # A: 확인 — 깃발 유지 중(pk 로부터 FLAG_MAX+1 일 안) 종가가 pk 고가 위
        for j in range(b + 1, min(pk + FLAG_MAX + 2, len(c) - 1)):
            if c[j] > top:
                fl = np.arange(pk + 1, j)
                stop = l[fl].min()
                e = j + 1
                if e > busy["A"] and stop < o[e]:
                    sim = simulate(o, h, l, c, e, stop, o[e])
                    if sim is not None:
                        busy["A"] = e + sim[1] - 1
                        out.append({**base, "dried": bool(val[fl].mean() <= DRY * val[s]), "entry": "A", "e": e,
                                    "entry_date": bars.index[e], "stop_pct": 1 - stop / o[e], "level_pct": 0.0,
                                    "ret": sim[0], "days": sim[1], "how": sim[2]})
                break
            if not ok(j):
                break
    return out


def controls(bars: pd.DataFrame, ev: dict, rng) -> tuple[float, float, float, float]:
    o, h, l, c = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    lo, hi = int(bars.index.searchsorted(START)), len(c) - MAX_HOLD
    rets, dd = [], 0
    for d in rng.integers(lo, hi, vb.N_CTRL):
        d = int(d)
        ret, days, _ = simulate(o, h, l, c, d, o[d] * (1 - ev["stop_pct"]), o[d] * (1 + ev["level_pct"]))
        rets.append(ret)
        dd += days
    return float(np.mean(rets)), 0.0, dd / vb.N_CTRL, 0.0


def universe(name: str) -> list[str]:
    if name == "stocks":
        return sorted(sensor_universe())
    codes = []
    for f in ("oos_codes.json", "oos_kosdaq_codes.json"):
        codes += json.loads((_ROOT / "data" / LONG_HISTORY / f).read_text(encoding="utf-8"))["codes"]
    return codes


def run(name: str, rng) -> pd.DataFrame:
    rows = []
    for code in universe(name):
        bars = load_bars(LONG_HISTORY, code)
        if bars is None or len(bars) < 300:
            continue
        bars = bars[(bars[["open", "high", "low", "close"]] > 0).all(1)]
        for ev in find(bars):
            ev["ctrl_ret"], ev["ctrl_tgt"], ev["ctrl_days"], ev["ctrl_to"] = controls(bars, ev, rng)
            ev["code"], ev["tgt_pct"] = code, np.nan
            rows.append(ev)
    df = pd.DataFrame(rows)
    cal = load_bars(LONG_HISTORY, "069500").index
    df["block"] = cal.searchsorted(df["entry_date"]) // vb.BLOCK
    df["universe"] = name
    return df


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    rng = np.random.default_rng(0)
    df = pd.concat([run(u, rng) for u in ("stocks", "oos")], ignore_index=True)
    df.to_csv(results_dir() / "flag_events.csv", index=False)
    res = {}
    for (u, ent, dried), g in df.groupby(["universe", "entry", "dried"]):
        s = vb.summarize(g, rng, verdict=True)
        s["exit_mix"] = g["how"].value_counts(normalize=True).round(3).to_dict()
        s["win_rate"] = round(float((g["ret"] > 0).mean()), 4)
        res[f"{u}/{ent}/{'마름' if dried else '안 마름'}"] = s
    verdict = {}
    for ent in ("A", "B"):
        a, b = res.get(f"stocks/{ent}/마름", {}), res.get(f"oos/{ent}/마름", {})
        passed = a.get("excess_ci", [0])[0] > 0 and b.get("excess_ci", [0])[0] > 0
        verdict[ent] = "통과" if passed else "기준 미달"
    out = {"verdict": verdict, "rule": __doc__.split("\n\n")[0], **res}
    (results_dir() / "flag_validation.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, s in res.items():
        print(f"{k:22s} n={s['n']:5d} 독립={s['n_indep']:4d} 초과 {s['excess']:+.2%} [{s['excess_ci'][0]:+.2%}, {s['excess_ci'][1]:+.2%}] "
              f"순기대(0.3%) {s['net_exp']['0.3%']:+.2%} 승률 {s['win_rate']:.0%} 손절 {s['stop_pct']:.1%} {s['exit_mix']}")
    print(verdict)
    return 0


if __name__ == "__main__":
    sys.exit(main())
