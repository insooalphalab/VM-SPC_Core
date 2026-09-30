"""'약한 20일선 도달'에서 청산하는 규칙 — 검증이력 9.34 사전 등록 기준 그대로.

  python research/validate_weak_touch.py   → results/weak_touch_validation.json
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
import validate_stops as vs
from v2_config import LONG_HISTORY, results_dir, sensor_universe
from v2_datastore import load_bars


def run(codes: list[str], label: str) -> pd.DataFrame:
    rows = []
    for code in codes:
        bars = load_bars(LONG_HISTORY, code)
        if bars is None or len(bars) < 300:
            continue
        bars = bars[(bars[["open", "high", "low", "close"]] > 0).all(1)]
        o, h, l, c = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        ma20 = pd.Series(c).rolling(20).mean().shift(1).to_numpy()
        for ev in vs.events(bars):
            e, stop, tgt = ev["e"], ev["stops"]["기준"], ev["tgt"]
            ret, how, hold = vs.simulate(o, h, l, c, e, stop, tgt, False)
            # 처음 20일선에 닿은 날 — 그 전에(또는 같은 날) 손절·목표로 끝났으면 규칙이 적용되지 않음
            touch = next((j for j in range(e, e + hold) if h[j] >= ma20[j]), None)
            weak = False
            if touch is not None and touch < e + hold - 1 or (touch == e + hold - 1 and how == "timeout"):
                rng_ = max(h[touch] - l[touch], 1e-9)
                weak = (h[touch] - c[touch]) / rng_ > 0.6 or c[touch] < ma20[touch]
            a = c[touch] / o[e] - 1 if weak else ret
            rows.append({"universe": label, "code": code, "entry_date": ev["entry_date"], "weak": weak,
                         "base": ret, "A": a, "B": 0.5 * a + 0.5 * ret if weak else ret})
    df = pd.DataFrame(rows)
    df["block"] = load_bars(LONG_HISTORY, "069500").index.searchsorted(df["entry_date"]) // vb.BLOCK
    return df


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    df = pd.concat([run(sorted(sensor_universe()), "센서"), run(vs._codes("oos_codes.json"), "코스피 밖"),
                    run(vs._codes("oos_kosdaq_codes.json"), "코스닥 밖")], ignore_index=True)
    res = {}
    for u, g in df.groupby("universe", sort=False):
        for rule in ("A", "B"):
            for scope, x in (("전체", g), ("약한 도달만", g[g.weak])):
                d = (x[rule] - x["base"]).to_numpy()
                lo, hi = vb.boot_mean(d, x["block"].to_numpy(), rng)
                res[f"{u}/{rule}/{scope}"] = {"n": int(len(x)), "base_net": round(float((x["base"] - vb.COST).mean()), 4),
                                             "rule_net": round(float((x[rule] - vb.COST).mean()), 4),
                                             "diff": round(float(d.mean()), 4), "ci": [round(lo, 4), round(hi, 4)]}
    verdict = {r: "채택" if all(res[f"{u}/{r}/전체"]["ci"][0] > 0 for u in ("코스피 밖", "코스닥 밖")) else "참고 문구만"
               for r in ("A", "B")}
    (results_dir() / "weak_touch_validation.json").write_text(json.dumps({"verdict": verdict, **res}, ensure_ascii=False, indent=1),
                                                                encoding="utf-8")
    for k, s in res.items():
        print(f"{k:22s} n={s['n']:4d} 현행 {s['base_net']:+.2%} → 규칙 {s['rule_net']:+.2%} | 차이 {s['diff']:+.2%} [{s['ci'][0]:+.2%}, {s['ci'][1]:+.2%}]")
    print(verdict)
    return 0


if __name__ == "__main__":
    sys.exit(main())
