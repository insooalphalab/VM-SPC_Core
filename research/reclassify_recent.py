"""'최근 경향' 등급 신설과 '2021 이후에만' 결과 재분류 — 검증이력 9.85 사전 등록 그대로.

  python research/reclassify_recent.py   → results/reclassify_recent.json
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "vm_predict", _ROOT / "scenario", _ROOT / "research", _ROOT / "stock_track"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import json
import sys
import warnings

import numpy as np
import pandas as pd

import validate_box as vb
import validate_rebreakout as vr
from validate_compression_regime import regime
from validate_holding_factors import boot, spreads
from v2_config import results_dir

OLD, RECENT, HALF = pd.Timestamp("2016-01-01"), pd.Timestamp("2021-06-01"), pd.Timestamp("2024-01-01")
COST = vb.COST
RD = lambda f: pd.read_csv(results_dir() / f, parse_dates=["entry_date"])


def level_ci(x: pd.DataFrame, rng) -> tuple[float, float, float]:
    blk = (x.entry_date.rank(method="dense").astype(int) // vb.BLOCK).to_numpy()
    agg = np.array([(g.R.sum(), len(g)) for _, g in x.groupby(blk)], float)
    s = agg[rng.integers(0, len(agg), (vb.N_BOOT, len(agg)))].sum(1)
    lo, hi = np.percentile(s[:, 0] / s[:, 1], [2.5, 97.5])
    return float(x.R.mean()), float(lo), float(hi)


def judge(old, rec, h1, h2, T) -> str:
    excl = rec[1] > 0 if rec[0] > 0 else rec[2] < 0
    same = np.sign(h1) == np.sign(rec[0]) == np.sign(h2)
    if abs(rec[0]) >= T and excl and same and min(abs(h1), abs(h2)) >= T / 2:
        return "최근 경향" + (" (앞 기간도 같은 방향)" if old is not None and np.sign(old) == np.sign(rec[0]) else "")
    return "아님"


def event_item(d: pd.DataFrame, a_mask, b_mask, T: float, rng) -> dict:
    a, b = d[a_mask], d[b_mask]
    per = lambda lo, hi: (a[(a.entry_date >= lo) & (a.entry_date < hi)], b[(b.entry_date >= lo) & (b.entry_date < hi)])
    o = per(OLD, RECENT)
    old = float(o[0].R.mean() - o[1].R.mean()) if len(o[0]) and len(o[1]) else None
    r = per(RECENT, pd.Timestamp("2100-01-01"))
    rec = vr.diff_ci(r[0], r[1], rng)
    h1 = per(RECENT, HALF)
    h2 = per(HALF, pd.Timestamp("2100-01-01"))
    d1, d2 = float(h1[0].R.mean() - h1[1].R.mean()), float(h2[0].R.mean() - h2[1].R.mean())
    return {"앞 기간(2016~2021-05)": None if old is None else round(old, 3), "최근(2021-06~) [CI]": [round(x, 3) for x in rec],
            "2021-06~2023": round(d1, 3), "2024~": round(d2, 3), "건수 최근(해당/비교)": [int(len(r[0])), int(len(r[1]))],
            "판정": judge(old, rec, d1, d2, T)}


def spread_item(panel: pd.DataFrame, col: str, T: float, rng) -> dict:
    sp = spreads(panel, col, "x20").spread
    old = sp[sp.index < RECENT]
    rec = sp[sp.index >= RECENT]
    r = boot(rec, rng)
    d1, d2 = float(rec[rec.index < HALF].mean()), float(rec[rec.index >= HALF].mean())
    return {"앞 기간(2016~2021-05)": round(float(old.mean()), 4) if len(old) else None, "최근(2021-06~) [CI]": [round(x, 4) for x in r],
            "2021-06~2023": round(d1, 4), "2024~": round(d2, 4), "판정": judge(float(old.mean()) if len(old) else None, r, d1, d2, T)}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    res = {}
    # 1 횡보·전환 가짜 이탈 평균 R
    w = RD("winrate_events.csv")
    w = w[(w.depth >= 0.035) & (w.reg == "횡보·전환")].assign(R=lambda x: (x.ret - COST) / x.stop_pct.clip(lower=0.01))
    rec = level_ci(w[w.entry_date >= RECENT], rng)
    d1 = float(w[(w.entry_date >= RECENT) & (w.entry_date < HALF)].R.mean())
    d2 = float(w[w.entry_date >= HALF].R.mean())
    old = float(w[w.entry_date < RECENT].R.mean())
    res["1 횡보·전환 가짜 이탈(깊이 3.5%↑) 평균 R"] = {"앞 기간(2016~2021-05)": round(old, 3), "최근(2021-06~) [CI]": [round(x, 3) for x in rec],
                                               "2021-06~2023": round(d1, 3), "2024~": round(d2, 3), "판정": judge(old, rec, d1, d2, 0.2)}
    # 2·3 박스 폭 · 변동성 (상승장 돌파, 9.52 사건)
    e = RD("breakout_edge_events.csv")
    e = e[e.reg == "상승장"]
    for i, (nm, col) in enumerate((("박스 폭", "width"), ("250일 변동성", "vol250")), 2):
        q1, q2 = e[col].quantile([1 / 3, 2 / 3])
        res[f"{i} 상승장 돌파 {nm} 상위 − 하위 1/3"] = event_item(e, e[col] > q2, e[col] <= q1, 0.15, rng)
    # 4 섹터 강세
    ia = RD("industry_action_events.csv")
    ia = ia[(ia.reg == "상승장") & ia.ranked]
    res["4 상승장 돌파 섹터 강세 − 나머지"] = event_item(ia, ia.strong, ~ia.strong, 0.15, rng)
    # 5 RS 70 미만 + ΔRS20 ≥ 15
    ra = RD("rs_accel_events.csv")
    ra = ra[(ra.reg == "상승장") & (ra.rs_now < 70)]
    res["5 상승장 RS<70 돌파 ΔRS20 ≥ 15 − 나머지"] = event_item(ra, ra.d_rs >= 15, ra.d_rs < 15, 0.15, rng)
    # 6 거래량 배수
    bv = RD("breakout_volume_events.csv")
    res["6 상승장 RS≥70 돌파 거래량 < 1.5배 − ≥ 2배"] = event_item(bv, bv.vx < 1.5, bv.vx >= 2, 0.15, rng)
    # 7 돌파 마진
    tp = RD("trend_principles_events.csv")
    tp = tp[tp.reg == "상승장"]
    res["7 상승장 돌파 마진 ≤ 2% − > 4%"] = event_item(tp, tp.strength <= 0.02, tp.strength > 0.04, 0.15, rng)
    # 8 돌파일 지수 상승
    dv = RD("rs_divergence_events.csv")
    dv = dv[dv.reg == "상승장"]
    res["8 상승장 돌파일 지수 상승 − 아님"] = event_item(dv, dv.idx_up.astype(bool), ~dv.idx_up.astype(bool), 0.15, rng)
    # 9 횡보장 천장 나이
    bd = RD("base_duration_events.csv")
    bd = bd[bd.reg == "횡보·전환"]
    res["9 횡보·전환 돌파 천장 나이 21~30 − 1~10일"] = event_item(bd, bd.age >= 21, bd.age <= 10, 0.15, rng)
    # 10 매출 +25%
    eg = RD("earnings_growth_events.csv")
    res["10 상승장 RS≥70 돌파 매출 +25% − 나머지(자료 있음)"] = event_item(eg, eg.c_rev == 1, eg.c_rev == 0, 0.15, rng)
    # 11 확인 매수
    ce = RD("confirm_entry_events_main.csv")
    a = ce[ce["5일 · 상단 +10%|R"].notna()].assign(R=lambda x: x["5일 · 상단 +10%|R"])
    b = ce.assign(R=ce.R0)
    both = pd.concat([a.assign(g=1), b.assign(g=0)], ignore_index=True)
    res["11 5일 · 상단 +10% 확인 매수 − 즉시 매수"] = event_item(both, both.g == 1, both.g == 0, 0.10, rng)
    # 12 · 13 보유 판단 패널
    pnl = pd.read_csv(results_dir() / "holding_factors_panel.csv", parse_dates=["date"]).set_index("date")
    pnl["trend"] = pnl[["ma60", "ma120", "hi52", "rs"]].groupby(level=0).rank(pct=True).mean(axis=1, skipna=False)
    pnl["reg"] = regime().reindex(pnl.index).to_numpy()
    res["12 상승장 추세 강도 상위 − 하위(20일 초과수익)"] = spread_item(pnl[pnl.reg == "상승장"], "trend", 0.01, rng)
    fp = pnl[pnl.for20.notna()]
    fp = fp[fp.groupby(level=0).for20.transform("size") >= 50]
    res["13 외국인 20일 순매수 상위 − 하위(20일 초과수익)"] = spread_item(fp, "for20", 0.01, rng)
    (results_dir() / "reclassify_recent.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in res.items():
        print(k, "|", v["앞 기간(2016~2021-05)"], "|", v["최근(2021-06~) [CI]"], "|", v["2021-06~2023"], "/", v["2024~"], "|", v["판정"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
