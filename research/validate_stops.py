"""검증된 셋업(9.27: T² 동반 가짜 이탈)의 손절 방식 비교 — 검증이력 9.31 사전 등록 기준 그대로.

사건 = 기준 손절로 찾은 가짜 이탈(이탈일 T² > 관리한계) 고정, 손절·청산만 바꿔 같은 사건을 다시 체결한다.
  python research/validate_stops.py      → results/stop_validation.json
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
from box_rules import BOX, FAIL_RECOVER, MAX_HOLD, t2_flags
from v2_config import LONG_HISTORY, results_dir, sensor_universe
from v2_datastore import load_bars

START = pd.Timestamp("2016-01-01")
TIME_DAYS = 3
VARIANTS = ["기준", "① 스윙 저점", "②a ATR 1.0", "②b ATR 1.5", "③ 기준+3일", "①+③"]


def simulate(o, h, l, c, e, stop, tgt, timed: bool):
    """e일 시가 진입. 손절(갭이면 시가) 먼저, 목표, timed 면 TIME_DAYS일째 종가 ≤ 진입가에 정리, MAX_HOLD일 종가."""
    if e + MAX_HOLD - 1 >= len(c):
        return None
    entry = o[e]
    for j in range(e, e + MAX_HOLD):
        if j > e:
            if o[j] <= stop:
                return o[j] / entry - 1, "stop", j - e + 1
            if o[j] >= tgt:
                return o[j] / entry - 1, "target", j - e + 1
        if l[j] <= stop:
            return stop / entry - 1, "stop", j - e + 1
        if h[j] >= tgt:
            return tgt / entry - 1, "target", j - e + 1
        if timed and j - e + 1 == TIME_DAYS and c[j] <= entry:
            return c[j] / entry - 1, "time", j - e + 1
    return c[e + MAX_HOLD - 1] / entry - 1, "timeout", MAX_HOLD


def atr5(h, l, c, i) -> float:
    tr = [max(h[k] - l[k], abs(h[k] - c[k - 1]), abs(l[k] - c[k - 1])) for k in range(i - 4, i + 1)]
    return float(np.mean(tr))


def events(bars) -> list[dict]:
    o, h, l, c = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    H = bars["high"].rolling(BOX).max().shift(1).to_numpy()
    L = bars["low"].rolling(BOX).min().shift(1).to_numpy()
    t2 = t2_flags(bars)
    first = max(int(bars.index.searchsorted(START)), BOX + 5)
    out, busy = [], -1
    for b in range(first, len(c)):
        if b <= busy or np.isnan(L[b]) or not c[b] < L[b]:
            continue
        r = next((k for k in range(b + 1, min(b + 1 + FAIL_RECOVER, len(c))) if c[k] > L[b]), None)
        if r is None or r + 1 >= len(c):
            continue
        e, tgt, base = r + 1, H[b], l[b:r + 1].min()
        if o[e] <= base or o[e] >= tgt:
            continue
        sim = simulate(o, h, l, c, e, base, tgt, False)
        if sim is None:
            continue
        busy = e + sim[2] - 1                                                 # 9.27 과 같은 겹침 규칙(기준 체결 기준)
        if not t2[b]:
            continue
        a = atr5(h, l, c, e - 1)
        stops = {"기준": base, "① 스윙 저점": l[b - 2:b + 1].min(), "②a ATR 1.0": o[e] - a, "②b ATR 1.5": o[e] - 1.5 * a,
                 "③ 기준+3일": base, "①+③": l[b - 2:b + 1].min()}
        out.append({"e": e, "entry_date": bars.index[e], "tgt": tgt, "stops": stops})
    return out


def run(codes: list[str], label: str, rng) -> pd.DataFrame:
    rows = []
    for code in codes:
        bars = load_bars(LONG_HISTORY, code)
        if bars is None or len(bars) < 300:
            continue
        bars = bars[(bars[["open", "high", "low", "close"]] > 0).all(1)]
        o, h, l, c = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        lo, hi = int(bars.index.searchsorted(START)), len(c) - MAX_HOLD
        for ev in events(bars):
            e = ev["e"]
            days = rng.integers(lo, hi, vb.N_CTRL)
            for v in VARIANTS:
                stop, timed = ev["stops"][v], v in ("③ 기준+3일", "①+③")
                if not stop < o[e]:
                    continue
                ret, how, _ = simulate(o, h, l, c, e, stop, ev["tgt"], timed)
                sp, tp = 1 - stop / o[e], ev["tgt"] / o[e] - 1
                ctrl = np.mean([simulate(o, h, l, c, int(d), o[d] * (1 - sp), o[d] * (1 + tp), timed)[0] for d in days])
                rows.append({"universe": label, "code": code, "e": e, "entry_date": ev["entry_date"], "variant": v,
                             "ret": ret, "how": how, "stop_pct": sp, "tgt_pct": tp, "ctrl_ret": float(ctrl)})
    df = pd.DataFrame(rows)
    cal = load_bars(LONG_HISTORY, "069500").index
    df["block"] = cal.searchsorted(df["entry_date"]) // vb.BLOCK
    return df


def _codes(f: str) -> list[str]:
    return json.loads((_ROOT / "data" / LONG_HISTORY / f).read_text(encoding="utf-8"))["codes"]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    rng = np.random.default_rng(0)
    df = pd.concat([run(sorted(sensor_universe()), "센서", rng), run(_codes("oos_codes.json"), "코스피 밖", rng),
                    run(_codes("oos_kosdaq_codes.json"), "코스닥 밖", rng)], ignore_index=True)
    df.to_csv(results_dir() / "stop_validation_events.csv", index=False)
    res = {}
    for (u, v), g in df.groupby(["universe", "variant"], sort=False):
        ex = (g["ret"] - g["ctrl_ret"]).to_numpy()
        lo, hi = vb.boot_mean(ex, g["block"].to_numpy(), rng)
        base = df[(df.universe == u) & (df.variant == "기준")].set_index(["code", "e"])["ret"]
        pair = g.set_index(["code", "e"])["ret"] - base.reindex(g.set_index(["code", "e"]).index)
        pair = pair.dropna()
        blk = g.set_index(["code", "e"]).loc[pair.index, "block"].to_numpy()
        plo, phi = vb.boot_mean(pair.to_numpy(), blk, rng) if v != "기준" else (0.0, 0.0)
        res[f"{u}/{v}"] = {"n": int(len(g)), "net": round(float((g["ret"] - vb.COST).mean()), 4),
                           "win": round(float((g["ret"] - vb.COST > 0).mean()), 4),
                           "stop_rate": round(float((g["how"] == "stop").mean()), 4),
                           "stop_dist": round(float(g["stop_pct"].median()), 4),
                           "excess": round(float(ex.mean()), 4), "excess_ci": [round(lo, 4), round(hi, 4)],
                           "vs_base": round(float(pair.mean()), 4), "vs_base_ci": [round(plo, 4), round(phi, 4)],
                           "exit_mix": g["how"].value_counts(normalize=True).round(3).to_dict()}
    verdict = {}
    for v in VARIANTS[1:]:
        ex_ok = all(res.get(f"{u}/{v}", {}).get("excess_ci", [0])[0] > 0 for u in ("코스피 밖", "코스닥 밖"))
        pair_ok = all(res.get(f"{u}/{v}", {}).get("vs_base_ci", [0])[0] > 0 for u in ("센서", "코스피 밖"))
        verdict[v] = "채택" if ex_ok and pair_ok else "현행 유지"
    (results_dir() / "stop_validation.json").write_text(json.dumps({"verdict": verdict, **res}, ensure_ascii=False, indent=1),
                                                          encoding="utf-8")
    for k, s in res.items():
        print(f"{k:18s} n={s['n']:4d} 손절거리 {s['stop_dist']:5.1%} 손절률 {s['stop_rate']:4.0%} 승률 {s['win']:4.0%} 순수익 {s['net']:+.2%} | "
              f"무작위 대비 {s['excess']:+.2%} [{s['excess_ci'][0]:+.2%}, {s['excess_ci'][1]:+.2%}] | "
              f"기준 대비 {s['vs_base']:+.2%} [{s['vs_base_ci'][0]:+.2%}, {s['vs_base_ci'][1]:+.2%}]")
    print(verdict)
    return 0


if __name__ == "__main__":
    sys.exit(main())
