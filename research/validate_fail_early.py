"""가짜 이탈 매수 후 1~3일 움직임과 손절 확률 — 검증이력 9.92 사전 등록 그대로.

  python research/validate_fail_early.py   → results/fail_early.json  (사건 = results/box_scenario_events.csv failure)
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

from reclassify_recent import level_ci, RECENT
from validate_compression_regime import regime
from v2_config import LONG_HISTORY, results_dir
from v2_datastore import load_bars

BINS, LAB = [-np.inf, -0.5, 0, 0.5, np.inf], ["< −0.5R", "−0.5~0", "0~+0.5", "≥ +0.5R"]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    d = pd.read_csv(results_dir() / "box_scenario_events.csv", parse_dates=["entry_date"], dtype={"code": str})
    d = d[d.pattern == "failure"].copy()
    d["code"] = d["code"].str.zfill(6)
    rows = []
    for code, g in d.groupby("code"):
        b = load_bars(LONG_HISTORY, code)
        b = b[(b[["open", "high", "low", "close"]] > 0).all(1)]
        o, c = b["open"].to_numpy(float), b["close"].to_numpy(float)
        for _, x in g.iterrows():
            e = int(x.e)
            if e >= len(b) or b.index[e] != x.entry_date:
                continue
            r1 = x.stop_pct
            row = {"code": code, "entry_date": x.entry_date, "how": x.how, "ret": x.ret, "days": x.days, "bull1": bool(c[e] > o[e])}
            for k in (1, 2, 3):
                ck = c[e + k - 1] / x.entry - 1
                row[f"m{k}"] = ck / r1 if x.days > k else np.nan
                row[f"rem{k}"] = (x.ret - ck) / r1 if x.days > k else np.nan
            rows.append(row)
    ev = pd.DataFrame(rows)
    ev["reg"] = regime().reindex(ev.entry_date).to_numpy()
    ev.to_csv(results_dir() / "fail_early_events.csv", index=False)
    res = {}
    per = {"앞(2016~2021-05)": ev.entry_date < RECENT, "최근(2021-06~)": ev.entry_date >= RECENT}
    for k in (1, 2, 3):
        x = ev[ev[f"m{k}"].notna()].copy()
        x["bin"] = pd.cut(x[f"m{k}"], BINS, labels=LAB)
        out, mono = {}, []
        for pn, msk in per.items():
            y = x[msk.reindex(x.index)]
            t = y.groupby("bin").agg(n=("how", "size"), 손절=("how", lambda s: (s == "stop").mean()), 목표=("how", lambda s: (s == "target").mean()),
                                    남은R=(f"rem{k}", "mean"))
            sr = t["손절"].to_numpy()
            mono.append(bool(np.all(np.diff(sr) < 0)))
            out[pn] = {str(i): {"n": int(r.n), "손절": round(r.손절, 3), "목표": round(r.목표, 3), "남은R": round(r.남은R, 3)} for i, r in t.iterrows()}
        w = x[(x.bin == LAB[0]) & (x.entry_date >= RECENT)].assign(R=lambda z: z[f"rem{k}"])
        ci = level_ci(w, rng) if len(w) > 30 else (np.nan,) * 3
        wo = x[(x.bin == LAB[0]) & (x.entry_date < RECENT)][f"rem{k}"].mean()
        use = ci[0] < 0 and wo < 0 and ci[2] < 0
        out["가장 약한 구간 남은R 최근 [CI]"] = [round(v, 3) for v in ci]
        out["판정"] = ("추세 있음" if all(mono) else "추세 없음") + " · " + (f"{k}일째 종가 조기 정리 후보" if use else "조기 정리 규칙 없음(들고 가는 게 맞음)")
        out["장세별 최근 손절(구간 순)"] = {rg: [round(v, 3) for v in x[(x.entry_date >= RECENT) & (x.reg == rg)].groupby("bin").how.apply(lambda s: (s == "stop").mean())]
                                     for rg in ("상승장", "횡보·전환", "하락장")}
        res[f"{k}일째 종가"] = out
    b1 = ev[ev.days > 1]
    res["1일째 양봉 vs 음봉(서술, 전체)"] = {("양봉" if k else "음봉"): {"n": len(g), "손절": round(float((g.how == "stop").mean()), 3), "남은R": round(float(g.rem1.mean()), 3)}
                                     for k, g in b1.groupby("bull1")}
    (results_dir() / "fail_early.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
