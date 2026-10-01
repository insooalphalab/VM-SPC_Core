"""돌파 전 베이스 기간 — 검증이력 9.66 사전 등록 그대로.

  python research/validate_base_duration.py   → results/base_duration_validation.json
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

import validate_rebreakout as vr
from box_rules import BOX
from v2_config import LONG_HISTORY, results_dir
from v2_datastore import load_bars

SPLIT = pd.Timestamp("2021-06-01")
CAP, STOP_R = 500, -0.95
BASE_BINS, BASE_LAB = [0, 46, 91, 10 ** 6], ["30~45", "46~90", "91+"]
AGE_BINS, AGE_LAB = [0, 11, 21, 31], ["1~10", "11~20", "21~30"]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    ev = pd.read_csv(results_dir() / "trend_principles_events.csv", parse_dates=["entry_date"], dtype={"code": str})
    ev["code"] = ev["code"].str.zfill(6)
    base, age = {}, {}
    for code, g in ev.groupby("code"):
        b = load_bars(LONG_HISTORY, code)
        b = b[(b[["open", "high", "low", "close"]] > 0).all(1)]
        h, c = b["high"].to_numpy(float), b["close"].to_numpy(float)
        for i, e in g.iterrows():
            t = int(b.index.searchsorted(e.entry_date)) - 1           # 돌파일
            win = h[t - BOX:t]
            H = win.max()
            age[i] = t - (t - BOX + int(np.flatnonzero(win == H)[-1]))
            k = t - 1
            while k >= 0 and t - k <= CAP and c[k] <= H:
                k -= 1
            base[i] = t - 1 - k
    ev["base"] = pd.Series(base)
    ev["age"] = pd.Series(age)
    ev["base_b"] = pd.cut(ev.base, BASE_BINS, labels=BASE_LAB, right=False)
    ev["age_b"] = pd.cut(ev.age, AGE_BINS, labels=AGE_LAB, right=False)
    ev["stopped"] = ev.R <= STOP_R
    ev.to_csv(results_dir() / "base_duration_events.csv", index=False)

    s = lambda g: {"n": int(len(g)), "승률": round(float(g.win.mean()), 3), "R": round(float(g.R.mean()), 3),
                   "손절 종료": round(float(g.stopped.mean()), 3)}
    res = {"장세별": {}}
    for r, g in ev.groupby("reg"):
        res["장세별"][r] = {"① 베이스 기간": {k: s(x) for k, x in g.groupby("base_b")},
                           "② 천장 나이": {k: s(x) for k, x in g.groupby("age_b")}}
    u = ev[ev.reg == "상승장"]
    for name, col, longv, shortv in (("① 베이스 46+ − 30~45", "base", lambda x: x >= 46, lambda x: x < 46),
                                     ("② 천장 나이 21~30 − 1~10", "age", lambda x: x >= 21, lambda x: x <= 10)):
        for scope, x in (("전체", u), ("RS ≥ 70 안", u[u.rs >= 0.70])):
            a, b = x[longv(x[col])], x[shortv(x[col])]
            tr = vr.diff_ci(a[a.entry_date < SPLIT], b[b.entry_date < SPLIT], rng)
            te = vr.diff_ci(a[a.entry_date >= SPLIT], b[b.entry_date >= SPLIT], rng)
            res[f"상승장 {scope} {name} R"] = {"n": [int(len(a)), int(len(b))], "맞춤": [round(v, 3) for v in tr], "예측": [round(v, 3) for v in te],
                                              "verdict": "채택" if te[0] >= 0.15 and te[1] > 0 and tr[0] > 0 else
                                                         "방향만 일치" if te[0] > 0 and tr[0] > 0 else "채택 안 함"}
    (results_dir() / "base_duration_validation.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
