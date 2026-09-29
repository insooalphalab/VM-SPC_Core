"""거래량 기초 기술분석 검증 — 검증이력 9.19 사전 등록 기준 그대로.

9.18 박스 사건(results/box_scenario_events[_etf].csv, research/validate_box.py 결과)을 패턴 안에서 거래량 규칙으로 나눠 무작위 대비 초과 기댓값을 비교한다.
  python research/validate_volume.py     → results/volume_rules_validation.json
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

from v2_config import results_dir
from v2_datastore import load_bars
from validate_box import BLOCK, LONG_BASKET, N_BOOT

VOL_WIN = 20
PROFILE_WIN = 120


def features(events: pd.DataFrame) -> pd.DataFrame:
    """사건마다 이탈·돌파일(b)의 거래량 비율과 (박스 고가, 목표가] 매물 비중. b 는 validate_box 와 같은 필터를 거친 봉 기준 인덱스."""
    out = []
    for code, g in events.groupby("code"):
        bars = load_bars(LONG_BASKET, code)
        bars = bars[(bars[["open", "high", "low", "close"]] > 0).all(1)]
        v = bars["volume"].to_numpy(float)
        tp = ((bars["high"] + bars["low"] + bars["close"]) / 3).to_numpy(float)
        avg = pd.Series(v).rolling(VOL_WIN).mean().shift(1).to_numpy()
        hi30 = bars["high"].rolling(30).max().shift(1).to_numpy()
        for i, ev in g.iterrows():
            b = int(ev["b"])
            vr = v[b] / avg[b] if avg[b] > 0 else np.nan
            overhead = np.nan
            if ev["pattern"] == "retest" and b >= PROFILE_WIN:
                hb = hi30[b]
                tgt = ev["entry"] * (1 + ev["tgt_pct"])
                w = slice(b - PROFILE_WIN, b)
                tot = v[w].sum()
                overhead = v[w][(tp[w] > hb) & (tp[w] <= tgt)].sum() / tot if tot > 0 else np.nan
            out.append({"idx": i, "vol_ratio": vr, "overhead": overhead})
    return events.join(pd.DataFrame(out).set_index("idx"))


def diff_ci(ex: np.ndarray, grp: np.ndarray, block: np.ndarray, rng) -> dict:
    """grp=1(예상 쪽) − grp=0 평균 초과의 블록 부트스트랩 CI + 예상 쪽 자체 CI."""
    df = pd.DataFrame({"b": block, "g": grp, "x": ex})
    agg = df.groupby(["b", "g"])["x"].agg(["sum", "count"]).unstack(fill_value=0)
    s1, n1 = agg[("sum", 1)].to_numpy(), agg[("count", 1)].to_numpy()
    s0, n0 = agg[("sum", 0)].to_numpy(), agg[("count", 0)].to_numpy()
    idx = rng.integers(0, len(agg), (N_BOOT, len(agg)))
    m1 = s1[idx].sum(1) / np.maximum(n1[idx].sum(1), 1)
    m0 = s0[idx].sum(1) / np.maximum(n0[idx].sum(1), 1)
    d_lo, d_hi = np.percentile(m1 - m0, [2.5, 97.5])
    g_lo, g_hi = np.percentile(m1, [2.5, 97.5])
    e1, e0 = ex[grp == 1], ex[grp == 0]
    return {"n_yes": int(len(e1)), "n_no": int(len(e0)),
            "ex_yes": round(float(e1.mean()), 4), "ex_no": round(float(e0.mean()), 4),
            "diff": round(float(e1.mean() - e0.mean()), 4), "diff_ci": [round(d_lo, 4), round(d_hi, 4)],
            "ex_yes_ci": [round(g_lo, 4), round(g_hi, 4)]}


def rules(df: pd.DataFrame, rng) -> dict:
    cal = load_bars(LONG_BASKET, "069500").index
    df = df.copy()
    df["ex"] = df["ret"] - df["ctrl_ret"]
    df["block"] = cal.searchsorted(pd.to_datetime(df["entry_date"])) // BLOCK
    out = {}
    r = df[(df["pattern"] == "retest") & df["vol_ratio"].notna()]
    out["V1_breakout_volume"] = {**diff_ci(r["ex"].to_numpy(), (r["vol_ratio"] >= 2).astype(int).to_numpy(), r["block"].to_numpy(), rng),
                                 "yes": "돌파일 거래량 ≥ 20일 평균 2배", "target_rate_yes": _tr(r[r["vol_ratio"] >= 2]),
                                 "target_rate_no": _tr(r[r["vol_ratio"] < 2])}
    f = df[(df["pattern"] == "failure") & df["vol_ratio"].notna()]
    out["V2_quiet_breakdown"] = {**diff_ci(f["ex"].to_numpy(), (f["vol_ratio"] < 1).astype(int).to_numpy(), f["block"].to_numpy(), rng),
                                 "yes": "이탈일 거래량 < 20일 평균", "target_rate_yes": _tr(f[f["vol_ratio"] < 1]),
                                 "target_rate_no": _tr(f[f["vol_ratio"] >= 1])}
    o = df[(df["pattern"] == "retest") & df["overhead"].notna()]
    lo, hi = o["overhead"].quantile([1 / 3, 2 / 3])
    o = o[(o["overhead"] <= lo) | (o["overhead"] >= hi)]
    out["V3_thin_overhead"] = {**diff_ci(o["ex"].to_numpy(), (o["overhead"] <= lo).astype(int).to_numpy(), o["block"].to_numpy(), rng),
                               "yes": f"위쪽 매물 비중 하위 1/3 (≤ {lo:.1%})", "no": f"상위 1/3 (≥ {hi:.1%})",
                               "target_rate_yes": _tr(o[o["overhead"] <= lo]), "target_rate_no": _tr(o[o["overhead"] >= hi])}
    return out


def _tr(g: pd.DataFrame) -> float:
    return round(float((g["how"] == "target").mean()), 4) if len(g) else None


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    rng = np.random.default_rng(0)
    res = {}
    for name, fn in (("stocks", "box_scenario_events.csv"), ("etf", "box_scenario_events_etf.csv")):
        ev = pd.read_csv(results_dir() / fn, dtype={"code": str})
        res[name] = rules(features(ev), rng)
    verdict = {}
    for k in res["stocks"]:
        s, e = res["stocks"][k], res["etf"][k]
        ok = s["diff_ci"][0] > 0 and np.sign(e["diff"]) == np.sign(s["diff"])
        verdict[k] = "통과" if ok else "HOLD"
    out = {"verdict": verdict, **res}
    (results_dir() / "volume_rules_validation.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
