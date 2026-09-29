"""DART 사건 검증 A(실적 서프라이즈)·B(자본 정책 공시) — 검증이력 9.17 사전 등록 기준 그대로.

  python research/validate_dart_events.py      → results/dart_events_validation.json
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "vm_predict", _ROOT / "pair_spc", _ROOT / "dart_events",
           _ROOT / "stock_track", _ROOT / "scenario", _ROOT / "research"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import json
import sys

import numpy as np
import pandas as pd

from client import dart_dir
from collect import BENCH
from fundamentals import quarterly
from v2_config import LONG_HISTORY as LONG_BASKET, results_dir, sensor_universe
from v2_datastore import load_bars

HORIZONS = (20, 60)
MAIN_H = 60
TOP, BOT = 0.8, 0.2
MIN_SEASON_EVENTS = 10
MIN_INDEP = 30
MAX_LAG_DAYS = 100          # 분기 말 → 접수일이 이보다 길면 정정본 날짜로 보고 제외
DEDUP_DAYS = 90             # 같은 종목·유형 재공시(약 60거래일)
N_BOOT = 2000
EXPECT = {"buyback": +1, "rights": -1, "cb": -1}


def _status(passed: bool, n_indep: int, H: int) -> str:
    """판정은 주 기간(H=60)에만 붙인다. 다른 기간은 통과해도 참고(다중비교 방지)."""
    if H != MAIN_H:
        return "참고(기준 충족)" if passed else "참고"
    if not passed:
        return "HOLD"
    return "Active" if n_indep >= MIN_INDEP else "잠정 통과"


# ── 가격 → 사건 뒤 초과수익 ────────────────────────────────
def price_panel(codes) -> tuple[pd.DataFrame, pd.Series]:
    from stock_track.data import kospi_codes, market_of
    kospi = kospi_codes()
    close = pd.DataFrame({c: load_bars(LONG_BASKET, c)["close"] for c in codes if load_bars(LONG_BASKET, c) is not None})
    close = close.sort_index()
    bench = {m: load_bars(LONG_BASKET, code)["close"].reindex(close.index) for m, code in zip(("KOSPI", "KOSDAQ"), BENCH)}
    mkt = pd.Series({c: market_of(c, kospi) for c in close.columns})
    return close, mkt, bench


def excess_forward(close, mkt, bench, H) -> pd.DataFrame:
    """날짜 d 종가 진입 → d+H 종가까지 초과수익(종목 − 소속 시장 기준지수)."""
    r = close.shift(-H) / close - 1
    out = {}
    for c in close.columns:
        b = bench[mkt[c]]
        out[c] = r[c] - (b.shift(-H) / b - 1)
    return pd.DataFrame(out)


def attach(ev: pd.DataFrame, ex: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """접수일 다음 거래일을 진입일로 붙이고 H별 초과수익을 가져온다."""
    idx = next(iter(ex.values())).index
    rd = pd.to_datetime(ev["rcept_dt"], format="%Y%m%d")
    pos = idx.searchsorted(rd, side="right")
    ok = pos < len(idx)
    ev = ev[ok].copy()
    ev["entry"] = idx[pos[ok]]
    ev["entry_i"] = pos[ok]
    for H, df in ex.items():
        ev[f"ex{H}"] = [df.at[d, c] if c in df.columns else np.nan for d, c in zip(ev["entry"], ev["stock_code"])]
    return ev


# ── A. 실적 서프라이즈 ─────────────────────────────────────
def sue(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for code, g in df.groupby("stock_code"):
        g = g.set_index(["year", "q"]).sort_index()
        full = pd.MultiIndex.from_product([range(g.index.get_level_values(0).min(), g.index.get_level_values(0).max() + 1), (1, 2, 3, 4)])
        g = g.reindex(full)
        yoy = g["op"] - g["op"].shift(4)
        sd = yoy.shift(1).rolling(8, min_periods=4).std()
        g["sue"] = yoy / sd
        g["stock_code"] = code
        out.append(g.dropna(subset=["sue", "rcept_dt"]).reset_index(names=["year", "q"]))
    return pd.concat(out, ignore_index=True)


def test_a(ev: pd.DataFrame, H: int, rng) -> dict:
    ev = ev.dropna(subset=[f"ex{H}"])
    base = float((ev[f"ex{H}"] > 0).mean())
    seasons = []
    for (y, q), g in ev.groupby(["year", "q"]):
        if len(g) < MIN_SEASON_EVENTS:
            continue
        hi, lo = g["sue"].quantile(TOP), g["sue"].quantile(BOT)
        top, bot = g[g["sue"] >= hi][f"ex{H}"], g[g["sue"] <= lo][f"ex{H}"]
        seasons.append({"season": f"{y}Q{q}", "n": len(g), "diff": top.mean() - bot.mean(),
                        "top_hit": int((top > 0).sum()), "top_n": len(top)})
    s = pd.DataFrame(seasons)
    idx = rng.integers(0, len(s), (N_BOOT, len(s)))
    diff_b = s["diff"].to_numpy()[idx].mean(1)
    hit_b = s["top_hit"].to_numpy()[idx].sum(1) / s["top_n"].to_numpy()[idx].sum(1)
    d_lo, d_hi = np.percentile(diff_b, [2.5, 97.5])
    h_lo, h_hi = np.percentile(hit_b, [2.5, 97.5])
    top_hit = s["top_hit"].sum() / s["top_n"].sum()
    passed = bool(d_lo > 0 and h_lo > base)
    return {"H": H, "n_events": int(len(ev)), "n_seasons": int(len(s)), "base": round(base, 4),
            "spread": round(float(s["diff"].mean()), 4), "spread_ci": [round(d_lo, 4), round(d_hi, 4)],
            "top_hit": round(float(top_hit), 4), "top_hit_ci": [round(h_lo, 4), round(h_hi, 4)],
            "status": _status(passed, len(s), H),
            "first_season": s["season"].iloc[0], "last_season": s["season"].iloc[-1]}


# ── B. 자본 정책 공시 ──────────────────────────────────────
def dedup(ev: pd.DataFrame) -> pd.DataFrame:
    ev = ev.sort_values("rcept_dt")
    keep, last = [], {}
    for i, x in ev.iterrows():
        d = pd.Timestamp(x["rcept_dt"])
        key = (x["stock_code"], x["type"])
        if key in last and (d - last[key]).days < DEDUP_DAYS:
            continue
        last[key] = d
        keep.append(i)
    return ev.loc[keep]


def test_b(ev: pd.DataFrame, typ: str, H: int, base: float, rng) -> dict:
    g = ev[(ev["type"] == typ)].dropna(subset=[f"ex{H}"])
    y = (g[f"ex{H}"] > 0).astype(float)
    blocks = pd.DataFrame({"b": g["entry_i"] // H, "hit": y}).groupby("b")["hit"].agg(["sum", "count"]).to_numpy()
    idx = rng.integers(0, len(blocks), (N_BOOT, len(blocks)))
    boot = blocks[idx, 0].sum(1) / blocks[idx, 1].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    ex_b = pd.DataFrame({"b": g["entry_i"] // H, "ex": g[f"ex{H}"]}).groupby("b")["ex"].agg(["sum", "count"]).to_numpy()
    exb = ex_b[idx, 0].sum(1) / ex_b[idx, 1].sum(1)
    e_lo, e_hi = np.percentile(exb, [2.5, 97.5])
    sign = EXPECT[typ]
    passed = bool((lo > base) if sign > 0 else (hi < base))
    return {"type": typ, "H": H, "expect": "+" if sign > 0 else "−", "n_events": int(len(g)), "n_indep": int(len(blocks)),
            "hit": round(float(y.mean()), 4), "hit_ci": [round(lo, 4), round(hi, 4)], "base": round(base, 4),
            "mean_ex": round(float(g[f"ex{H}"].mean()), 4), "mean_ex_ci": [round(e_lo, 4), round(e_hi, 4)],
            "status": _status(passed, len(blocks), H)}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    rng = np.random.default_rng(0)
    codes = sorted(sensor_universe())
    close, mkt, bench = price_panel(codes)
    ex = {H: excess_forward(close, mkt, bench, H) for H in HORIZONS}
    # 평소 비율: 2016년 이후 전 종목·전 거래일
    base = {H: float((df.loc["2016":].stack() > 0).mean()) for H, df in ex.items()}

    fin = pd.read_csv(dart_dir() / "financials.csv", dtype={"stock_code": str, "rcept_dt": str})
    qop = quarterly(fin, "op").rename(columns={"value": "op"})
    n_all = len(qop)
    s = sue(qop)                                   # 과거 분기 값은 전부 쓰고, 사건 날짜가 불확실한 분기만 뺀다
    s = s[(s["lag"] >= 0) & (s["lag"] <= MAX_LAG_DAYS)]
    ea = attach(s, ex)
    res_a = [test_a(ea, H, rng) for H in HORIZONS]

    evb = pd.read_csv(dart_dir() / "events.csv", dtype={"stock_code": str, "rcept_dt": str})
    evb = attach(dedup(evb), ex)
    res_b = [test_b(evb, t, H, base[H], rng) for t in EXPECT for H in HORIZONS]

    out = {"main_H": MAIN_H, "universe": len(codes), "price_codes": int(close.shape[1]),
           "base_all_days": {str(k): round(v, 4) for k, v in base.items()},
           "A_earnings_surprise": {"quarters_total": n_all, "quarters_used": int(len(s)), "results": res_a},
           "B_capital_events": res_b}
    path = results_dir() / "dart_events_validation.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
