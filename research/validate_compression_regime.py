"""응축 × 코스피 장세 — 검증이력 9.39 사전 등록 기준 그대로.

  python research/validate_compression_regime.py   → results/compression_regime_validation.json
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
from v2_compression import compression_frame
from v2_config import LONG_HISTORY, results_dir, sensor_universe
from v2_datastore import load_bars

H = 20
START = pd.Timestamp("2016-01-01")
EXPECT = {"상승장": 1, "하락장": -1}


def regime() -> pd.Series:
    k = load_bars(LONG_HISTORY, "069500")["close"]
    ma60 = k.rolling(60).mean()
    up = (k > ma60) & (ma60 > ma60.shift(20))
    dn = (k < ma60) & (ma60 < ma60.shift(20))
    return pd.Series(np.where(up, "상승장", np.where(dn, "하락장", "횡보·전환")), index=k.index)


def events(codes: list[str], reg: pd.Series) -> pd.DataFrame:
    rows = []
    for code in codes:
        bars = load_bars(LONG_HISTORY, code)
        if bars is None or len(bars) < 400:
            continue
        bars = bars[(bars[["open", "high", "low", "close"]] > 0).all(1)]
        f = compression_frame(bars)
        c, h, l = (bars[k].to_numpy(float) for k in ("close", "high", "low"))
        hi20 = pd.Series(h).rolling(20).max().to_numpy()
        lo20 = pd.Series(l).rolling(20).min().to_numpy()
        n = len(c)
        fwd = np.full(n, np.nan)
        fwd[:n - H] = c[H:] / c[:n - H] - 1
        brk = np.full(n, np.nan)                                      # 1 상방, 0 하방, nan 없음
        for t in range(20, n - H):
            for j in range(t + 1, t + H + 1):
                if c[j] > hi20[t]:
                    brk[t] = 1
                    break
                if c[j] < lo20[t]:
                    brk[t] = 0
                    break
        comp = f["compressed"].fillna(False).to_numpy()
        start = comp & ~np.r_[False, comp[:-1]]
        rg = reg.reindex(bars.index).ffill().to_numpy()
        d = pd.DataFrame({"date": bars.index, "fwd": fwd, "brk": brk, "exp": f["expanded"].to_numpy(), "start": start, "reg": rg})
        d = d[(d.date >= START) & d.fwd.notna()]
        base = d.groupby("reg").agg(b_fwd=("fwd", "mean"), b_up=("brk", "mean"), b_exp=("exp", "mean"))
        ev = d[d.start].join(base, on="reg")
        ev["code"] = code
        rows.append(ev)
    out = pd.concat(rows, ignore_index=True)
    out["ex"] = out["fwd"] - out["b_fwd"]
    out["ex_up"] = out["brk"] - out["b_up"]
    out["block"] = reg.index.searchsorted(out["date"]) // vb.BLOCK
    return out


def summarize(d: pd.DataFrame, rng) -> dict:
    res = {}
    for r, g in d.groupby("reg"):
        lo, hi = vb.boot_mean(g["ex"].to_numpy(), g["block"].to_numpy(), rng)
        gu = g.dropna(subset=["brk"])
        ulo, uhi = vb.boot_mean(gu["ex_up"].to_numpy(), gu["block"].to_numpy(), rng)
        x = {"n": int(len(g)), "ex": round(float(g["ex"].mean()), 4), "ci": [round(lo, 4), round(hi, 4)],
             "up_share": round(float(gu["brk"].mean()), 3), "base_up": round(float(gu["b_up"].mean()), 3),
             "ex_up_ci": [round(ulo, 3), round(uhi, 3)],
             "exp": round(float(g["exp"].mean()), 3), "base_exp": round(float(g["b_exp"].mean()), 3)}
        if r in EXPECT:
            s = EXPECT[r]
            x["verdict"] = ("재현" if s * x["ex"] > 0 and (lo > 0 if s > 0 else hi < 0) else
                            "방향만 일치" if s * x["ex"] > 0 else "반대")
        res[r] = x
    return res


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    reg = regime()
    sets = {"코스피 1~200": vs._codes("oos_codes.json"), "코스피 다음 200": vs._codes("oos_kospi2_codes.json"),
            "센서(참고)": sorted(sensor_universe()), "코스닥 밖(참고)": vs._codes("oos_kosdaq_codes.json")}
    res = {u: summarize(events(c, reg), rng) for u, c in sets.items()}
    final = {}
    for r in EXPECT:
        v = [res[u].get(r, {}).get("verdict", "반대") for u in ("코스피 1~200", "코스피 다음 200")]
        final[r] = ("유망" if all(x == "재현" for x in v) else "약함" if all(x in ("재현", "방향만 일치") for x in v) else "채택 안 함")
    res["최종"] = final
    (results_dir() / "compression_regime_validation.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for u, rr in res.items():
        if u == "최종":
            continue
        print(f"\n== {u}")
        for r in ("상승장", "횡보·전환", "하락장"):
            x = rr.get(r)
            if x:
                print(f"  {r:5s} n={x['n']:5d} 20일 초과 {x['ex']:+.2%} [{x['ci'][0]:+.2%}, {x['ci'][1]:+.2%}] | 상방 이탈 {x['up_share']:.0%} "
                      f"(같은 장세 평소 {x['base_up']:.0%}) | 변동성 확대 {x['exp']:.0%} (평소 {x['base_exp']:.0%}) {x.get('verdict', '')}")
    print("\n최종:", final)
    return 0


if __name__ == "__main__":
    sys.exit(main())
