"""수급 CUSUM 견고성 격자 (검증이력 9.21 격자 사전 등록) — 창 W × 문턱 h 전 칸을 한 표로.

  python research/flow_cusum_grid.py     → results/flow_cusum_grid.json
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

from stock_track.data import kospi_codes, market_of
from stock_track.flow import load_flow
from flow_alarm import flow_cusum, onsets as alarm_onsets
from v2_config import results_dir, sensor_universe
from v2_datastore import load_bars

LB = "_long_history"
WINDOWS = (20, 60, 120)
THRESH = (3.0, 4.5, 6.0)
K = 0.5
HORIZONS = (20, 60)
SOURCES = ("외국인", "기관합계")


def onsets(x: pd.Series, w: int, h: float) -> tuple[pd.Series, pd.Series]:
    m = max(10, w * 2 // 3)                      # 격자는 창 길이에 비례한 최소 표본(사전 등록 당시 그대로)
    return alarm_onsets(flow_cusum(x, w, h, K, norm_min=m, innov_min=m))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    kos = kospi_codes()
    bench = {"KOSPI": load_bars(LB, "069500")["close"], "KOSDAQ": load_bars(LB, "229200")["close"]}
    hits = {(s, w, h, kind, H): [] for s in SOURCES for w in WINDOWS for h in THRESH for kind in ("up", "down") for H in HORIZONS}
    exs = {k: [] for k in hits}
    base = {H: [] for H in HORIZONS}
    for code in sorted(sensor_universe()):
        bars, f = load_bars(LB, code), load_flow(code)
        if bars is None or f is None:
            continue
        c = bars["close"]
        bm = bench[market_of(code, kos)].reindex(c.index)
        fwd = {H: ((c.shift(-1 - H) / c.shift(-1) - 1) - (bm.shift(-1 - H) / bm.shift(-1) - 1)) for H in HORIZONS}
        for H in HORIZONS:
            base[H].append(fwd[H].loc[f.index.min():].dropna().to_numpy())
        for s in SOURCES:
            x = f[s].reindex(c.index).loc[f.index.min():f.index.max()].dropna()
            for w in WINDOWS:
                for h in THRESH:
                    up, dn = onsets(x, w, h)
                    for kind, flag in (("up", up), ("down", dn)):
                        d = flag[flag].index
                        for H in HORIZONS:
                            v = fwd[H].reindex(d).dropna().to_numpy()
                            exs[(s, w, h, kind, H)].append(v)
    base_hit = {H: float((np.concatenate(v) > 0).mean()) for H, v in base.items()}
    rows = []
    for key, parts in exs.items():
        v = np.concatenate(parts) if parts else np.array([])
        s, w, h, kind, H = key
        rows.append({"src": s, "W": w, "h": h, "kind": kind, "H": H, "n": int(len(v)),
                     "hit": round(float((v > 0).mean()), 4) if len(v) else None,
                     "gap": round(float((v > 0).mean() - base_hit[H]), 4) if len(v) else None,
                     "mean_ex": round(float(v.mean()), 4) if len(v) else None})
    df = pd.DataFrame(rows)
    verdict = {}
    for s in SOURCES:
        d20 = df[(df["src"] == s) & (df["H"] == 20)]
        down_neg = int((d20[d20["kind"] == "down"]["gap"] < 0).sum())
        up_pos = int((d20[d20["kind"] == "up"]["gap"] > 0).sum())
        verdict[s] = {"down_below_base": f"{down_neg}/9", "up_above_base": f"{up_pos}/9",
                      "consistent": bool(down_neg >= 8 and up_pos >= 5)}
    out = {"base": {str(k): round(v, 4) for k, v in base_hit.items()}, "verdict": verdict, "cells": rows}
    (results_dir() / "flow_cusum_grid.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"base": out["base"], "verdict": verdict}, ensure_ascii=False, indent=1))
    for s in SOURCES:
        for H in HORIZONS:
            print(f"\n[{s}] {H}일 — 매도 경보 / 매수 경보: 지수 이김 − 평소 (%p), 평균 초과(%), 사건 수")
            print("   W \\ h " + "".join(f"{h:>30}" for h in THRESH))
            for w in WINDOWS:
                cells = []
                for h in THRESH:
                    dn = df[(df.src == s) & (df.W == w) & (df.h == h) & (df.kind == "down") & (df.H == H)].iloc[0]
                    up = df[(df.src == s) & (df.W == w) & (df.h == h) & (df.kind == "up") & (df.H == H)].iloc[0]
                    cells.append(f"{dn.gap * 100:+5.1f}/{up.gap * 100:+5.1f} ({dn.mean_ex * 100:+.1f}/{up.mean_ex * 100:+.1f}) n{dn.n}")
                print(f"  {w:5d}  " + "".join(f"{c:>30}" for c in cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
