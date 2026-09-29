"""박스권 시나리오 검증 — Failure(가짜 이탈) / Not Failure(돌파 리테스트). 검증이력 9.18 사전 등록 기준 그대로.

  python research/validate_box.py         → results/box_scenario_validation.json      (센서 종목, 2016~)
  python research/validate_box.py --etf   → results/box_scenario_validation_etf.json  (타겟 ETF 재현, 2006~)
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "vm_predict", _ROOT / "pair_spc", _ROOT / "dart_events",
           _ROOT / "stock_track", _ROOT / "scenario", _ROOT / "research"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import argparse
import json
import sys

import numpy as np
import pandas as pd

from box_rules import BOX, FAIL_RECOVER, MAX_HOLD, RETEST_WITHIN
from v2_compression import compression_frame
from v2_config import LONG_HISTORY as LONG_BASKET, load_baskets, results_dir, sensor_universe
from v2_datastore import load_bars

COST = 0.003
COST_SENS = (0.001, 0.003, 0.005, 0.01)
N_CTRL = 20
COMP_LOOKBACK = 10
START = pd.Timestamp("2016-01-01")
BLOCK = 20
N_BOOT = 2000
MIN_INDEP = 30
FOCUS = {"005930": "삼성전자", "000660": "SK하이닉스", "058470": "리노공업", "039030": "이오테크닉스"}


def simulate(o, h, l, c, e: int, stop: float, tgt: float):
    """e일 시가 진입 → (총수익, 보유일, 결과). 창이 모자라면 None."""
    if e + MAX_HOLD - 1 >= len(c):
        return None
    entry = o[e]
    for j in range(e, e + MAX_HOLD):
        if j > e:
            if o[j] <= stop:
                return o[j] / entry - 1, j - e + 1, "stop"
            if o[j] >= tgt:
                return o[j] / entry - 1, j - e + 1, "target"
        if l[j] <= stop:                      # 같은 날 둘 다 닿으면 손절 먼저
            return stop / entry - 1, j - e + 1, "stop"
        if h[j] >= tgt:
            return tgt / entry - 1, j - e + 1, "target"
    return c[e + MAX_HOLD - 1] / entry - 1, MAX_HOLD, "timeout"


def etf_universe() -> list[str]:
    return sorted({b["target"]["code"] for b in load_baskets()})


def find_events(bars: pd.DataFrame, comp: np.ndarray) -> list[dict]:
    o, h, l, c = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    H = bars["high"].rolling(BOX).max().shift(1).to_numpy()
    L = bars["low"].rolling(BOX).min().shift(1).to_numpy()
    first = int(bars.index.searchsorted(START))
    out = []
    for pattern in ("failure", "retest"):
        busy = -1
        for b in range(max(first, BOX), len(c)):
            if b <= busy or np.isnan(H[b]):
                continue
            hb, lb = H[b], L[b]
            mb = (hb + lb) / 2
            r = None
            if pattern == "failure" and c[b] < lb:
                r = next((k for k in range(b + 1, min(b + 1 + FAIL_RECOVER, len(c))) if c[k] > lb), None)
                stop, tgt = (l[b:r + 1].min(), hb) if r is not None else (None, None)
            elif pattern == "retest" and c[b] > hb:
                for k in range(b + 1, min(b + 1 + RETEST_WITHIN, len(c))):
                    if c[k] < mb:
                        break
                    if l[k] <= hb:
                        r = k
                        break
                stop, tgt = mb, hb + (hb - lb)
            if r is None or r + 1 >= len(c):
                continue
            e = r + 1
            if o[e] <= stop or o[e] >= tgt:
                continue
            sim = simulate(o, h, l, c, e, stop, tgt)
            if sim is None:
                continue
            ret, days, how = sim
            busy = e + days - 1
            out.append({"pattern": pattern, "b": b, "e": e, "entry_date": bars.index[e], "entry": o[e],
                        "stop_pct": 1 - stop / o[e], "tgt_pct": tgt / o[e] - 1, "ret": ret, "days": days, "how": how,
                        "compressed": bool(comp[max(0, b - COMP_LOOKBACK):b].any())})
    return out


def controls(bars: pd.DataFrame, ev: dict, rng) -> tuple[float, float, float, float]:
    """같은 종목 무작위 날짜 N_CTRL개, 같은 손절·목표 거리 → (평균 총수익, 목표 도달률, 평균 보유일, 시간초과율)."""
    o, h, l, c = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    lo = int(bars.index.searchsorted(START))
    hi = len(c) - MAX_HOLD
    rets, tg, dd, to = [], 0, 0, 0
    for d in rng.integers(lo, hi, N_CTRL):
        ret, days, how = simulate(o, h, l, c, int(d), o[d] * (1 - ev["stop_pct"]), o[d] * (1 + ev["tgt_pct"]))
        rets.append(ret)
        tg += how == "target"
        dd += days
        to += how == "timeout"
    return float(np.mean(rets)), tg / N_CTRL, dd / N_CTRL, to / N_CTRL


def boot_mean(x: np.ndarray, block: np.ndarray, rng) -> tuple[float, float]:
    df = pd.DataFrame({"b": block, "x": x}).groupby("b")["x"].agg(["sum", "count"]).to_numpy()
    idx = rng.integers(0, len(df), (N_BOOT, len(df)))
    bs = df[idx, 0].sum(1) / df[idx, 1].sum(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(lo), float(hi)


def summarize(g: pd.DataFrame, rng, verdict: bool) -> dict:
    ex = (g["ret"] - g["ctrl_ret"]).to_numpy()
    lo, hi = boot_mean(ex, g["block"].to_numpy(), rng)
    n_indep = int(g["block"].nunique())
    out = {"n": int(len(g)), "n_indep": n_indep,
           "target_rate": round(float((g["how"] == "target").mean()), 4),
           "ctrl_target_rate": round(float(g["ctrl_tgt"].mean()), 4),
           "stop_pct": round(float(g["stop_pct"].median()), 4), "tgt_pct": round(float(g["tgt_pct"].median()), 4),
           "net_exp": {f"{c:.1%}": round(float((g["ret"] - c).mean()), 4) for c in COST_SENS},
           "ctrl_net_exp": round(float((g["ctrl_ret"] - COST).mean()), 4),
           "excess": round(float(ex.mean()), 4), "excess_ci": [round(lo, 4), round(hi, 4)],
           "days": round(float(g["days"].mean()), 2), "ctrl_days": round(float(g["ctrl_days"].mean()), 2),
           "timeout_rate": round(float((g["how"] == "timeout").mean()), 4),
           "ctrl_timeout_rate": round(float(g["ctrl_to"].mean()), 4)}
    if verdict:
        out["status"] = ("Active" if n_indep >= MIN_INDEP else "잠정 통과") if lo > 0 else "HOLD"
    return out


def main() -> int:
    global START
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--etf", action="store_true")
    args = ap.parse_args()
    if args.etf:
        START = pd.Timestamp("2006-01-01")
    suffix = "_etf" if args.etf else ""
    rng = np.random.default_rng(0)
    cal = load_bars(LONG_BASKET, "069500").index
    rows = []
    for code in (etf_universe() if args.etf else sorted(sensor_universe())):
        bars = load_bars(LONG_BASKET, code)
        if bars is None or len(bars) < 300:
            continue
        bars = bars[(bars[["open", "high", "low", "close"]] > 0).all(1)]
        comp = compression_frame(bars)["compressed"].fillna(False).to_numpy()
        for ev in find_events(bars, comp):
            ev["ctrl_ret"], ev["ctrl_tgt"], ev["ctrl_days"], ev["ctrl_to"] = controls(bars, ev, rng)
            ev["code"] = code
            rows.append(ev)
    df = pd.DataFrame(rows)
    df["block"] = cal.searchsorted(df["entry_date"]) // BLOCK

    out = {"universe": "etf" if args.etf else "stocks", "n_codes": int(df["code"].nunique()),
           "start": str(START.date()), "cost": COST, "patterns": {}}
    for pat, g in df.groupby("pattern"):
        out["patterns"][pat] = {
            "all": summarize(g, rng, verdict=True),
            "compressed": summarize(g[g["compressed"]], rng, verdict=False),
            "not_compressed": summarize(g[~g["compressed"]], rng, verdict=False),
            "focus": {FOCUS[c]: summarize(g[g["code"] == c], rng, verdict=False)
                      for c in FOCUS if (g["code"] == c).sum() >= 3},
        }
    path = results_dir() / f"box_scenario_validation{suffix}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    df.drop(columns=["entry_date"]).assign(entry_date=df["entry_date"].dt.strftime("%Y-%m-%d")).to_csv(
        results_dir() / f"box_scenario_events{suffix}.csv", index=False, encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
