"""쌍끌이(외국인 + 기관 동시) CUSUM 검증 — 검증이력 9.23 사전 등록 기준 그대로.

  python research/validate_joint_flow.py     → results/joint_flow_validation.json
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

from collect_investor_detail import load_detail
from stock_track.data import kospi_codes, market_of
from flow_alarm import flow_cusum
from v2_config import results_dir, sensor_universe
from v2_datastore import load_bars

LB = "_long_history"
HORIZONS = (5, 10, 20)
MAIN_H = 20
N_BOOT = 2000
MIN_INDEP = 30
PAIRS = (("외국인", "기관합계"), ("외국인", "연기금"))


def states(x: pd.Series) -> tuple[pd.Series, pd.Series]:
    """그날 상방·하방 경보가 켜져 있는지(9.21과 같은 정의 — stock_track/flow_alarm)."""
    cu = flow_cusum(x)
    return cu.alarm_up, cu.alarm_down


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    kos, rng = kospi_codes(), np.random.default_rng(0)
    bench = {"KOSPI": load_bars(LB, "069500")["close"], "KOSDAQ": load_bars(LB, "229200")["close"]}
    cal = bench["KOSPI"].index
    events, base_parts = [], {H: [] for H in HORIZONS}
    for code in sorted(sensor_universe()):
        bars, d = load_bars(LB, code), load_detail(code)
        if bars is None or d is None or len(d) < 300:
            continue
        c = bars["close"]
        bm = bench[market_of(code, kos)].reindex(c.index)
        fwd = {H: ((c.shift(-1 - H) / c.shift(-1) - 1) - (bm.shift(-1 - H) / bm.shift(-1) - 1)) for H in HORIZONS}
        for H in HORIZONS:
            base_parts[H].append((fwd[H].loc[d.index.min():].dropna() > 0).to_numpy())
        st = {}
        for col in {x for pair in PAIRS for x in pair}:
            s = d[col].reindex(c.index).loc[d.index.min():d.index.max()].dropna()
            st[col] = states(s)
        for a, b in PAIRS:
            for kind, k in (("buy", 0), ("sell", 1)):
                both = (st[a][k] & st[b][k].reindex(st[a][k].index, fill_value=False))
                onset = both & ~both.shift(1, fill_value=False)
                for dt in onset[onset].index:
                    events.append({"pair": f"{a}+{b}", "kind": kind, "code": code, "date": dt,
                                   **{f"ex{H}": fwd[H].get(dt, np.nan) for H in HORIZONS}})
    ev = pd.DataFrame(events)
    base = {H: float(np.concatenate(v).mean()) for H, v in base_parts.items()}
    ev["bi"] = cal.searchsorted(ev["date"])
    res = []
    for (pair, kind), g in ev.groupby(["pair", "kind"]):
        leans = []
        for H in HORIZONS:
            gg = g.dropna(subset=[f"ex{H}"])
            y = (gg[f"ex{H}"] > 0).astype(float)
            blk = pd.DataFrame({"b": gg["bi"] // H, "y": y}).groupby("b")["y"].agg(["sum", "count"]).to_numpy()
            idx = rng.integers(0, len(blk), (N_BOOT, len(blk)))
            bs = blk[idx, 0].sum(1) / blk[idx, 1].sum(1)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            ok = (lo > base[H]) if kind == "buy" else (hi < base[H])
            leans.append((y.mean() > base[H]) if kind == "buy" else (y.mean() < base[H]))
            status = ("참고(기준 충족)" if ok else "참고") if H != MAIN_H else \
                (("Active" if len(blk) >= MIN_INDEP else "잠정 통과") if ok else "HOLD")
            res.append({"pair": pair, "kind": kind, "H": H, "n": int(len(gg)), "n_codes": int(gg["code"].nunique()),
                        "n_indep": int(len(blk)), "hit": round(float(y.mean()), 4), "hit_ci": [round(lo, 4), round(hi, 4)],
                        "base": round(base[H], 4), "mean_ex": round(float(gg[f"ex{H}"].mean()), 4), "status": status})
        for r in res[-len(HORIZONS):]:
            r["consistent_all_H"] = bool(all(leans))
    out = {"base": {str(k): round(v, 4) for k, v in base.items()}, "results": res}
    (results_dir() / "joint_flow_validation.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
