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
for _p in (_ROOT, _ROOT / "core", _ROOT / "vm_predict", _ROOT / "scenario", _ROOT / "research", _ROOT / "stock_track"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import json
import sys
import warnings

import numpy as np
import pandas as pd

import validate_box as vb
import validate_breakout_trend as vbt
from collect_program import load_program
from validate_program_shock import share_z
import validate_stops as vs
from box_rules import BOX, MAX_HOLD, t2_flags
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
        prog = load_program(code)                                   # 9.79: 이탈일 프로그램 비중 z(자료 없으면 nan)
        pz = share_z(bars, prog) if prog is not None and not prog.empty else np.full(len(bars), np.nan)
        Lbox = bars["low"].rolling(BOX).min().shift(1).to_numpy()
        lo, hi = int(bars.index.searchsorted(vb.START)), len(c) - MAX_HOLD
        for ev in vb.find_events(bars, np.zeros(len(bars), bool)):
            if ev["pattern"] != "failure":
                continue
            wins = 0
            for d in rng.integers(lo, hi, N_CTRL):
                d = int(d)
                r = vb.simulate(o, h, l, c, d, o[d] * (1 - ev["stop_pct"]), o[d] * (1 + ev["tgt_pct"]))
                wins += r is not None and r[0] - COST > 0
            stop_px = ev["entry"] * (1 - ev["stop_pct"])
            r = ev["e"] - 1                                         # 복귀일
            cpos = (c[r] - l[r]) / (h[r] - l[r]) if h[r] > l[r] else np.nan
            z = pz[ev["b"]]
            rows.append({"cpos": cpos, "prog_z": float(z) if np.isfinite(z) else np.nan, "code": code, "entry_date": ev["entry_date"], "t2": bool(t2[ev["b"]]), "ret": ev["ret"],
                         "depth": 1 - stop_px / Lbox[ev["b"]],
                         "stop_pct": ev["stop_pct"], "tgt_pct": ev["tgt_pct"], "ctrl_win": wins / N_CTRL})
    d = pd.DataFrame(rows)
    d["reg"] = reg.iloc[reg.index.searchsorted(d["entry_date"]) - 1].to_numpy()
    return d


DEPTH_CAP = 0.12
REGS = ("횡보·전환", "하락장")                       # 기준 = 상승장


def design(d: pd.DataFrame) -> np.ndarray:
    return np.c_[np.ones(len(d)), d["t2"].astype(float), *[(d["reg"] == r).astype(float) for r in REGS],
                 np.clip(d["depth"].to_numpy(float), 0, DEPTH_CAP), d["cpos"].to_numpy(float), d["prog_z"].to_numpy(float)]


def fit_logit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    b = np.zeros(X.shape[1])
    for _ in range(100):
        p = 1 / (1 + np.exp(-X @ b))
        step = np.linalg.solve(X.T @ (X * (p * (1 - p))[:, None]) + 1e-6 * np.eye(len(b)), X.T @ (y - p))
        b += step
        if np.abs(step).max() < 1e-8:
            break
    return b


def calibration(d: pd.DataFrame, b: np.ndarray) -> dict:
    p = 1 / (1 + np.exp(-design(d) @ b))
    y = ((d["ret"] - COST) > 0).to_numpy(float)
    q = pd.qcut(p, 5, duplicates="drop")
    rows = [{"예측": round(float(p[m].mean()), 3), "실제": round(float(y[m].mean()), 3), "n": int(m.sum())}
            for m in (q == c for c in q.categories)]
    order = np.argsort(p)
    ranks = np.empty(len(p)); ranks[order] = np.arange(len(p))
    pos = y == 1
    auc = (ranks[pos].mean() - (pos.sum() - 1) / 2) / (~pos).sum() if pos.any() and (~pos).any() else np.nan
    return {"구간별": rows, "AUC": round(float(auc), 3), "Brier": round(float(np.mean((p - y) ** 2)), 4)}


def cell(g: pd.DataFrame) -> dict:
    net = g["ret"] - COST
    return {"n": int(len(g)), "win": round(float((net > 0).mean()), 3),
            "R": round(float((net / g["stop_pct"].clip(lower=0.01)).mean()), 2),
            "ctrl_win": round(float(g["ctrl_win"].mean()), 3)}


def breakout_rs() -> dict:
    """RS 등급 70 이상 돌파(9.62 채택)의 장세별 승률·R — validate_trend_principles 사건표, 상승장은 9.72 50일선 청산 기준."""
    f = results_dir() / "trend_principles_events.csv"
    if not f.exists():
        return {}
    d = pd.read_csv(f)
    d = d[d["rs"] >= 0.70]
    out = {r: {"n": int(len(g)), "win": round(float(g["win"].mean()), 3), "R": round(float(g["R"].mean()), 2)} for r, g in d.groupby("reg")}
    ex = results_dir() / "breakout_exits_events.csv"                  # 9.72: 상승장 돌파는 50일선 청산 규칙의 승률·R
    if ex.exists():
        e = pd.read_csv(ex)
        out["상승장"] = {"n": int(len(e)), "win": round(float(e["E2 50일선|win"].mean()), 3), "R": round(float(e["E2 50일선|R"].mean()), 2)}
    return out


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
    d.to_csv(results_dir() / "winrate_events.csv", index=False)
    d = d.dropna(subset=["depth", "cpos", "prog_z"])            # 9.48 복귀 캔들 종가 위치 · 9.79 이탈일 프로그램 비중 z
    y = ((d["ret"] - COST) > 0).to_numpy(float)
    half = d["entry_date"] < pd.Timestamp("2021-06-01")
    b_train = fit_logit(design(d[half]), y[half.to_numpy()])
    cal = calibration(d[~half], b_train)                        # 앞 절반으로 맞추고 뒤 절반으로 확인
    b_all = fit_logit(design(d), y)                             # 화면에 쓰는 계수는 전체 기간
    model = {"features": ["상수", "T2", *REGS, f"깊이(상한 {DEPTH_CAP:.0%})", "복귀 캔들 종가 위치", "이탈일 프로그램 비중 z"], "coef": [round(float(x), 4) for x in b_all],
             "cpos_mean": round(float(d["cpos"].mean()), 3), "prog_z_mean": round(float(d["prog_z"].mean()), 3),
             "coef_train_2016_2021": [round(float(x), 4) for x in b_train], "depth_cap": DEPTH_CAP,
             "검증_2021_06_이후": cal, "n": int(len(d))}
    print("모형 계수:", dict(zip(model["features"], model["coef"])))
    print("뒤 절반 보정:", cal)
    bo = vbt.run(codes, reg, rng)
    bo["stop_pct"] = bo["stop_pct"]
    breakout = {r: {"n": int(len(g)), "win": round(float((g["net"] > 0).mean()), 3),
                    "R": round(float((g["net"] / g["stop_pct"].clip(lower=0.01)).mean()), 2)} for r, g in bo.groupby("reg")}
    out = {"made": pd.Timestamp.now().strftime("%Y-%m-%d"), "tgt_cut": [round(q1, 4), round(q2, 4)], "model": model,
           "failure": fail, "failure_coarse": fail_coarse, "breakout": breakout, "breakout_rs": breakout_rs(),
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
