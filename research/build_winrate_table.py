"""오늘의 후보 순위용 승률표 — 비슷한 과거 사건을 묶어 낸 승률(종목 하나로 쪼개지 않음).

가짜 이탈(9.18 규칙: 손절 = 이탈~복귀 최저가, 목표 = 박스 상단, 최대 20일, 비용 0.3%) 사건을
T² 동반 여부 × 코스피 장세(9.37 정의) × 목표 거리 3등분으로 묶어 칸마다
승률(비용 뺀 수익 > 0), 평균 R, 무작위 진입 승률(같은 손절·목표 거리)을 낸다.
돌파(9.40 규칙: 손절 상단 − 1ATR, 20일선 이탈 청산)는 장세별 한 칸.
표본: 센서 191 + 코스피 1~200 + 코스피 다음 200, 2016~.

  python research/build_winrate_table.py   → results/scenario/winrate_table.json
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
from box_rules import MAX_HOLD, t2_flags
from validate_compression_regime import regime
from v2_config import LONG_HISTORY, results_dir, sensor_universe
from v2_datastore import load_bars

COST = vb.COST
N_CTRL = 10


def failure_events(codes: list[str], reg: pd.Series, rng) -> pd.DataFrame:
    vb.START = pd.Timestamp("2016-01-01")
    rows = []
    for code in codes:
        bars = load_bars(LONG_HISTORY, code)
        if bars is None or len(bars) < 300:
            continue
        bars = bars[(bars[["open", "high", "low", "close"]] > 0).all(1)]
        o, h, l, c = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        t2 = t2_flags(bars)
        lo, hi = int(bars.index.searchsorted(vb.START)), len(c) - MAX_HOLD
        for ev in vb.find_events(bars, np.zeros(len(bars), bool)):
            if ev["pattern"] != "failure":
                continue
            wins = 0
            for d in rng.integers(lo, hi, N_CTRL):
                d = int(d)
                r = vb.simulate(o, h, l, c, d, o[d] * (1 - ev["stop_pct"]), o[d] * (1 + ev["tgt_pct"]))
                wins += r is not None and r[0] - COST > 0
            rows.append({"code": code, "entry_date": ev["entry_date"], "t2": bool(t2[ev["b"]]), "ret": ev["ret"],
                         "stop_pct": ev["stop_pct"], "tgt_pct": ev["tgt_pct"], "ctrl_win": wins / N_CTRL})
    d = pd.DataFrame(rows)
    d["reg"] = reg.iloc[reg.index.searchsorted(d["entry_date"]) - 1].to_numpy()
    return d


def cell(g: pd.DataFrame) -> dict:
    net = g["ret"] - COST
    return {"n": int(len(g)), "win": round(float((net > 0).mean()), 3),
            "R": round(float((net / g["stop_pct"].clip(lower=0.01)).mean()), 2),
            "ctrl_win": round(float(g["ctrl_win"].mean()), 3)}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    reg = regime()
    codes = sorted(set(sensor_universe()) | set(vs._codes("oos_codes.json")) | set(vs._codes("oos_kospi2_codes.json")))
    d = failure_events(codes, reg, rng)
    q1, q2 = (float(x) for x in d["tgt_pct"].quantile([1 / 3, 2 / 3]))
    d["tgt_bin"] = np.where(d["tgt_pct"] <= q1, "가까움", np.where(d["tgt_pct"] <= q2, "중간", "멂"))
    fail = {f"{'T2' if t else '-'}|{r}|{b}": cell(g) for (t, r, b), g in d.groupby(["t2", "reg", "tgt_bin"])}
    fail_coarse = {f"{'T2' if t else '-'}|{r}": cell(g) for (t, r), g in d.groupby(["t2", "reg"])}
    bo = vbt.run(codes, reg, rng)
    bo["stop_pct"] = bo["stop_pct"]
    breakout = {r: {"n": int(len(g)), "win": round(float((g["net"] > 0).mean()), 3),
                    "R": round(float((g["net"] / g["stop_pct"].clip(lower=0.01)).mean()), 2)} for r, g in bo.groupby("reg")}
    out = {"made": pd.Timestamp.now().strftime("%Y-%m-%d"), "tgt_cut": [round(q1, 4), round(q2, 4)],
           "failure": fail, "failure_coarse": fail_coarse, "breakout": breakout,
           "note": "센서 191 + 코스피 1~400, 2016~. 승률 = 비용 0.3% 뺀 수익 > 0. ctrl_win = 같은 손절·목표 거리 무작위 진입 승률"}
    (results_dir() / "scenario").mkdir(parents=True, exist_ok=True)
    (results_dir() / "scenario" / "winrate_table.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("목표 거리 경계:", q1, q2)
    for k, v in sorted(fail.items()):
        print(f"  {k:20s} n={v['n']:5d} 승률 {v['win']:.0%} (평소 {v['ctrl_win']:.0%}) {v['R']:+.2f}R")
    for k, v in breakout.items():
        print(f"  돌파|{k:6s} n={v['n']:5d} 승률 {v['win']:.0%} {v['R']:+.2f}R")
    return 0


if __name__ == "__main__":
    sys.exit(main())
