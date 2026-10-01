"""신호 조합 — 승률 모형에 후보 변수를 넣어 기간 밖 예측력으로 판정. 검증이력 9.48 사전 등록 기준 그대로.

  python research/validate_signal_combo.py   → results/signal_combo_validation.json
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
HIST = "--hist" in _sys.argv         # 9.79: 2016~ 이력 확장 재판정(규칙·기준 그대로, 기간만) — 결과 파일에 _hist
SUF = "_hist" if HIST else ""
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
import validate_stops as vs
from box_rules import BOX, FAIL_RECOVER, t2_flags
from collect_credit import load_credit
from collect_program import load_program
from validate_compression_regime import regime
from validate_program_shock import share_z
from v2_config import LONG_HISTORY, results_dir, sensor_universe
from v2_datastore import load_bars

COST = vb.COST
START, SPLIT = (pd.Timestamp("2016-03-01"), pd.Timestamp("2021-06-01")) if HIST else (pd.Timestamp("2024-12-01"), pd.Timestamp("2025-10-01"))
DEPTH_CAP = 0.12
CANDIDATES = {"복귀 캔들 종가 위치": "cpos", "신용잔고 변화": "credit", "프로그램 비중 z": "prog_z", "복귀 속도(일)": "rec_days"}


def events(code: str, reg: pd.Series) -> list[dict]:
    bars, cr, prog = load_bars(LONG_HISTORY, code), load_credit(code), load_program(code)
    if bars is None or cr is None or prog is None or len(bars) < 300:
        return []
    bars = bars[(bars[["open", "high", "low", "close"]] > 0).all(1)]
    o, h, l, c = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    H = bars["high"].rolling(BOX).max().shift(1).to_numpy()
    L = bars["low"].rolling(BOX).min().shift(1).to_numpy()
    t2 = t2_flags(bars)
    loan = cr["loan_shares"].reindex(bars.index).to_numpy(float)
    pz = share_z(bars, prog)
    rg = reg.reindex(bars.index).ffill().to_numpy()
    out, busy = [], -1
    for b in range(max(int(bars.index.searchsorted(START)), BOX + 5), len(c)):
        if b <= busy or np.isnan(L[b]) or not c[b] < L[b]:
            continue
        r = next((k for k in range(b + 1, min(b + 1 + FAIL_RECOVER, len(c))) if c[k] > L[b]), None)
        if r is None or r + 1 >= len(c):
            continue
        e, tgt, stop = r + 1, H[b], l[b:r + 1].min()
        if o[e] <= stop or o[e] >= tgt:
            continue
        sim = vs.simulate(o, h, l, c, e, stop, tgt, False)
        if sim is None:
            continue
        busy = e + sim[2] - 1
        base = np.nanmean(loan[b - 5:b])
        out.append({"code": code, "entry_date": bars.index[e], "win": float(sim[0] - COST > 0), "t2": float(t2[b]),
                    "reg_side": float(rg[b - 1] == "횡보·전환"), "reg_down": float(rg[b - 1] == "하락장"),
                    "depth": min(1 - stop / L[b], DEPTH_CAP),
                    "cpos": (c[r] - l[r]) / (h[r] - l[r]) if h[r] > l[r] else np.nan,
                    "credit": loan[b] / base - 1 if base > 0 and not np.isnan(loan[b]) else np.nan,
                    "prog_z": pz[b], "rec_days": float(r - b)})
    return out


def fit(X, y):
    b = np.zeros(X.shape[1])
    for _ in range(100):
        p = 1 / (1 + np.exp(-X @ b))
        step = np.linalg.solve(X.T @ (X * (p * (1 - p))[:, None]) + 1e-6 * np.eye(len(b)), X.T @ (y - p))
        b += step
        if np.abs(step).max() < 1e-8:
            break
    return b


def auc(p, y):
    order = np.argsort(p)
    ranks = np.empty(len(p)); ranks[order] = np.arange(len(p))
    pos = y == 1
    return (ranks[pos].mean() - (pos.sum() - 1) / 2) / (~pos).sum()


def logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


BASE = ["t2", "reg_side", "reg_down", "depth"]


def design(d, cols):
    X = d[cols].to_numpy(float)
    mu, sd = X.mean(0), X.std(0)
    return mu, sd


def predict(train, test, cols):
    tr = train[cols].to_numpy(float); te = test[cols].to_numpy(float)
    mu, sd = tr.mean(0), np.where(tr.std(0) > 0, tr.std(0), 1)
    Xtr = np.c_[np.ones(len(tr)), (tr - mu) / sd]; Xte = np.c_[np.ones(len(te)), (te - mu) / sd]
    b = fit(Xtr, train["win"].to_numpy())
    return 1 / (1 + np.exp(-Xte @ b)), b


def compare(train, test, cols_a, cols_b, rng) -> dict:
    pa, _ = predict(train, test, cols_a)
    pb, coef = predict(train, test, cols_b)
    y = test["win"].to_numpy()
    blocks = (test["entry_date"].rank(method="dense").astype(int) // vb.BLOCK).to_numpy()
    ub = np.unique(blocks)
    diffs = []
    for _ in range(1000):
        pick = rng.choice(ub, len(ub))
        m = np.concatenate([np.where(blocks == k)[0] for k in pick])
        if y[m].min() == y[m].max():
            continue
        diffs.append(auc(pb[m], y[m]) - auc(pa[m], y[m]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"auc_base": round(float(auc(pa, y)), 4), "auc_new": round(float(auc(pb, y)), 4),
            "auc_diff": round(float(auc(pb, y) - auc(pa, y)), 4), "ci": [round(float(lo), 4), round(float(hi), 4)],
            "logloss_base": round(logloss(pa, y), 4), "logloss_new": round(logloss(pb, y), 4),
            "coef_new_std": round(float(coef[-1]), 3)}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    reg = regime()
    codes = sorted(set(sensor_universe()) | set(vs._codes("oos_codes.json")) | set(vs._codes("oos_kospi2_codes.json"))
                   | set(vs._codes("oos_kosdaq_codes.json")))
    d = pd.DataFrame([ev for code in codes for ev in events(code, reg)])
    d.to_csv(results_dir() / f"signal_combo_events{SUF}.csv", index=False)
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=BASE + list(CANDIDATES.values()))   # z 분모 0(거래 없는 날)이면 inf
    train, test = d[d.entry_date < SPLIT], d[d.entry_date >= SPLIT]
    res = {"n_train": int(len(train)), "n_test": int(len(test)), "win_test": round(float(test.win.mean()), 3)}
    pb, _ = predict(train, test, BASE)
    res["기본 모형"] = {"auc_test": round(float(auc(pb, test.win.to_numpy())), 4), "logloss_test": round(logloss(pb, test.win.to_numpy()), 4)}
    keep = []
    for name, col in CANDIDATES.items():
        r = compare(train, test, BASE, BASE + [col], rng)
        r["verdict"] = "남김" if r["ci"][0] > 0 and r["logloss_new"] < r["logloss_base"] else "뺌"
        res[name] = r
        if r["verdict"] == "남김":
            keep.append(col)
    if keep:
        res["남긴 변수 전부"] = compare(train, test, BASE, BASE + keep, rng) | {"vars": keep}
    (results_dir() / f"signal_combo_validation{SUF}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
