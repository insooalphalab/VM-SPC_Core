"""재돌파(돌파 → 박스 복귀 → 다시 돌파) — 검증이력 9.55 사전 등록 기준 그대로.

  python research/validate_rebreakout.py   → results/rebreakout_validation.json
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
import warnings

import numpy as np
import pandas as pd

import validate_box as vb
import validate_breakout_trend as vbt
import validate_stops as vs
from box_rules import BOX
from validate_compression_regime import regime
from v2_config import LONG_HISTORY, results_dir, sensor_universe
from v2_datastore import load_bars

START, SPLIT = pd.Timestamp("2016-01-01"), pd.Timestamp("2021-06-01")
BACK, AGAIN = 10, 10
COST = vb.COST


def events(code: str, reg: pd.Series, rng) -> list[dict]:
    bars = load_bars(LONG_HISTORY, code)
    if bars is None or len(bars) < 400:
        return []
    bars = bars[(bars[["open", "high", "low", "close"]] > 0).all(1)]
    o, h, l, c = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    Hs = bars["high"].rolling(BOX).max().shift(1).to_numpy()
    ma20 = pd.Series(c).rolling(20).mean().to_numpy()
    tr = np.maximum(h - l, np.maximum(abs(h - np.r_[np.nan, c[:-1]]), abs(l - np.r_[np.nan, c[:-1]])))
    atr = pd.Series(tr).rolling(14).mean().to_numpy()
    rg = reg.reindex(bars.index).ffill().to_numpy()
    first = max(int(bars.index.searchsorted(START)), 60)
    lo_c, hi_c = first, len(c) - vbt.HOLD
    out, busy = [], -1
    for t1 in range(first, len(c) - 2):
        if np.isnan(Hs[t1]) or not (c[t1] > Hs[t1] and c[t1 - 1] <= Hs[t1 - 1]):
            continue
        H = Hs[t1]
        back = next((k for k in range(t1 + 1, min(t1 + 1 + BACK, len(c))) if c[k] < H), None)
        if back is None:
            continue
        t2 = next((k for k in range(back + 1, min(back + 1 + AGAIN, len(c) - 1)) if c[k] > H), None)
        if t2 is None or t2 <= busy:
            continue
        e, stop = t2 + 1, H - atr[t2]
        if not stop < o[e]:
            continue
        sim = vbt.simulate(o, h, l, c, ma20, e, stop)
        if sim is None:
            continue
        busy = e + sim[2] - 1
        sp = 1 - stop / o[e]
        ctrl = np.mean([vbt.simulate(o, h, l, c, ma20, int(d), o[d] * (1 - sp))[0] for d in rng.integers(lo_c, hi_c, 10)])
        out.append({"code": code, "entry_date": bars.index[e], "reg": rg[t2], "ret": sim[0], "stop_pct": sp, "ctrl": float(ctrl),
                    "win": float(sim[0] - COST > 0), "R": (sim[0] - COST) / max(sp, 0.01),
                    "pull": 1 - l[back:t2].min() / H, "gap_days": t2 - t1})
    return out


def diff_ci(a: pd.DataFrame, b: pd.DataFrame, rng) -> tuple[float, float, float]:
    x = pd.concat([a.assign(g=1), b.assign(g=0)])
    x["block"] = x.entry_date.rank(method="dense").astype(int) // vb.BLOCK
    agg = np.array([(y.loc[y.g == 1, "R"].sum(), (y.g == 1).sum(), y.loc[y.g == 0, "R"].sum(), (y.g == 0).sum()) for _, y in x.groupby("block")], float)
    idx = rng.integers(0, len(agg), (vb.N_BOOT, len(agg)))
    s = agg[idx].sum(1)
    with np.errstate(all="ignore"):
        bs = s[:, 0] / s[:, 1] - s[:, 2] / s[:, 3]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return float(a.R.mean() - b.R.mean()), float(lo), float(hi)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    reg = regime()
    codes = sorted(set(sensor_universe()) | set(vs._codes("oos_codes.json")) | set(vs._codes("oos_kospi2_codes.json"))
                   | set(vs._codes("oos_kosdaq_codes.json")))
    rb = pd.DataFrame([ev for code in codes for ev in events(code, reg, rng)])
    rb.to_csv(results_dir() / "rebreakout_events.csv", index=False)
    fb = pd.read_csv(results_dir() / "breakout_edge_events.csv", parse_dates=["entry_date"])
    s = lambda g: {"n": int(len(g)), "승률": round(float(g.win.mean()), 3), "R": round(float(g.R.mean()), 3),
                   "초과": round(float((g.ret - g.ctrl).mean()), 4)}
    res = {"장세별": {r: {"재돌파": s(rb[rb.reg == r]), "첫 돌파": s(fb[fb.reg == r])} for r in ("상승장", "횡보·전환", "하락장")}}
    rbu, fbu = rb[rb.reg == "상승장"], fb[fb.reg == "상승장"]
    tr = diff_ci(rbu[rbu.entry_date < SPLIT], fbu[fbu.entry_date < SPLIT], rng)
    te = diff_ci(rbu[rbu.entry_date >= SPLIT], fbu[fbu.entry_date >= SPLIT], rng)
    res["상승장 재돌파 − 첫 돌파 R"] = {"맞춤": [round(x, 3) for x in tr], "예측": [round(x, 3) for x in te]}
    res["verdict"] = ("채택" if te[0] >= 0.15 and te[1] > 0 and tr[0] > 0 else
                      "방향만 일치" if te[0] > 0 and tr[0] > 0 else "채택 안 함")
    q = rbu.pull.quantile([1 / 3, 2 / 3])
    res["상승장 되밀림 깊이별"] = {k: s(g) for k, g in (("얕음", rbu[rbu.pull <= q.iloc[0]]),
                                                ("중간", rbu[(rbu.pull > q.iloc[0]) & (rbu.pull <= q.iloc[1])]), ("깊음", rbu[rbu.pull > q.iloc[1]]))}
    res["되밀림 깊이 경계"] = [round(float(x), 4) for x in q]
    (results_dir() / "rebreakout_validation.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
