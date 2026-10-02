"""유동성 3종(Amihud · HL 스프레드 · 무변동일)으로 돌파 · 가짜 이탈 성과가 갈리는가 — 검증이력 9.98 사전 등록 그대로.

  python research/validate_liquidity.py   → results/liquidity_validation.json
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
from reclassify_recent import event_item, RECENT
from v2_config import LONG_HISTORY, results_dir
from v2_datastore import load_bars

WIN = 20
T = 0.15


def liq_frame(b: pd.DataFrame) -> pd.DataFrame:
    h, l, c, v = (b[k] for k in ("high", "low", "close", "volume"))
    r = c.pct_change()
    amt = (c * v).replace(0, np.nan)
    amihud = (r.abs() / amt).rolling(WIN, min_periods=15).mean()
    lh = np.log(h / l) ** 2
    beta = lh + lh.shift(1)
    gamma = np.log(pd.concat([h, h.shift(1)], axis=1).max(1) / pd.concat([l, l.shift(1)], axis=1).min(1)) ** 2
    k = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    hl = (2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))).clip(lower=0).rolling(WIN, min_periods=15).mean()
    zero = (r == 0).astype(float).rolling(WIN, min_periods=15).mean()
    return pd.DataFrame({"amihud": amihud, "hl": hl, "zero": zero})


def attach(ev: pd.DataFrame) -> pd.DataFrame:
    out = []
    for code, g in ev.groupby("code"):
        b = load_bars(LONG_HISTORY, code)
        if b is None or not len(b):
            continue
        b = b[(b[["open", "high", "low", "close"]] > 0).all(1)]
        f = liq_frame(b)
        pos = b.index.searchsorted(g.entry_date) - 1                     # 신호일 = 진입 전날
        ok = pos >= 0
        g = g[ok].copy()
        for col in f:
            g[col] = f[col].to_numpy()[pos[ok]]
        out.append(g)
    d = pd.concat(out, ignore_index=True).dropna(subset=["amihud", "hl", "zero"])
    yr = d.entry_date.dt.year
    for col in ("amihud", "hl", "zero"):
        d[col + "_p"] = d.groupby(yr)[col].rank(pct=True)
    d["comp"] = d[["amihud_p", "hl_p", "zero_p"]].mean(1)
    d["comp_p"] = d.groupby(yr).comp.rank(pct=True)
    return d


def verdict(d: pd.DataFrame, col: str, rng) -> dict:
    hi, lo = d[col] > 2 / 3, d[col] <= 1 / 3
    it = event_item(d, hi, lo, T, rng)
    old = d.entry_date < RECENT
    oci = vr.diff_ci(d[hi & old], d[lo & old], rng)
    rci = it["최근(2021-06~) [CI]"]
    excl = lambda ci: ci[1] > 0 or ci[2] < 0
    if abs(oci[0]) >= T and abs(rci[0]) >= T and np.sign(oci[0]) == np.sign(rci[0]) and excl(oci) and excl(rci):
        v = "검증됨"
    else:
        v = it["판정"]
    it["앞 기간 [CI]"] = [round(x, 3) for x in oci]
    it["R 비유동 / 유동(최근)"] = [round(float(d[hi & ~old].R.mean()), 3), round(float(d[lo & ~old].R.mean()), 3)]
    it["판정"] = v
    return it


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    bk = pd.read_csv(results_dir() / "sideways_breakout_events.csv", parse_dates=["entry_date"], dtype={"code": str})
    bk = bk[(bk.reg == "상승장") & (bk.rs >= 0.70)].assign(R=lambda x: x.R2)
    fb = pd.read_csv(results_dir() / "winrate_events.csv", parse_dates=["entry_date"], dtype={"code": str})
    fb = fb.assign(R=lambda x: (x.ret - vb.COST) / x.stop_pct.clip(lower=0.01))
    res = {}
    for name, ev in (("상승장 RS≥70 돌파", bk), ("가짜 이탈", fb)):
        ev = ev.assign(code=ev.code.str.zfill(6))[["code", "entry_date", "R"]]
        d = attach(ev)
        d.to_csv(results_dir() / f"liquidity_events_{'bo' if '돌파' in name else 'fb'}.csv", index=False)
        out = {"건수": len(d), "지표 중앙값": {c: float(f"{d[c].median():.3g}") for c in ("amihud", "hl", "zero")}}
        for lab, col in (("복합(주 판정)", "comp_p"), ("Amihud", "amihud_p"), ("HL 스프레드", "hl_p"), ("무변동일", "zero_p")):
            out[lab] = verdict(d, col, rng)
        res[name] = out
    (results_dir() / "liquidity_validation.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
