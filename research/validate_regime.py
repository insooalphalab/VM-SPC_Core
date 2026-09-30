"""반려된 전략의 코스피 추세별 재검토 — 검증이력 9.38 사전 등록 기준 그대로.

사건(무작위 대비 초과 포함)을 코스피 상승 추세 / 그 외로 나눈다. 판정은 코스피 1~200 + 다음 200 두 표본.
  python research/validate_patterns.py --kospi2   (먼저)
  python research/validate_regime.py              → results/regime_validation.json
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
from validate_market_trend import kospi_trend
from v2_config import LONG_HISTORY, results_dir
from v2_datastore import load_bars

TREND = {"① 하이 타이트 플래그", "② 강세 깃발 A", "② 강세 깃발 B", "③ 컵앤핸들", "돌파 리테스트", "칼만 상한 돌파 되밀림"}  # 상승 추세에서 나을 것
REVERT = {"④ 역헤드앤숄더", "가짜 이탈 전체", "칼만 하한 이탈 복귀"}                                               # 상승 추세가 아닐 때 나을 것
JUDGE = ("코스피 1~200", "코스피 다음 200")


def box_and_kalman(codes: list[str], rng) -> pd.DataFrame:
    """9.18 박스(가짜 이탈 전체·돌파 리테스트) + 9.27 칼만 관리선(하한 이탈 복귀·상한 돌파 되밀림) 사건과 대조군."""
    import validate_levels as vl
    vb.START = pd.Timestamp("2016-01-01")
    rows = []
    for code in codes:
        bars = load_bars(LONG_HISTORY, code)
        if bars is None or len(bars) < 300:
            continue
        bars = bars[(bars[["open", "high", "low", "close"]] > 0).all(1)]
        evs = [dict(ev, pattern={"failure": "가짜 이탈 전체", "retest": "돌파 리테스트"}[ev["pattern"]])
               for ev in vb.find_events(bars, np.zeros(len(bars), bool))]
        evs += [dict(ev, pattern={"failure": "칼만 하한 이탈 복귀", "retest": "칼만 상한 돌파 되밀림"}[ev["pattern"]])
                for ev in vl.find(bars, vl.levels_kalman(bars), np.zeros(len(bars), bool), vb.START)]
        for ev in evs:
            ev["ctrl_ret"] = vb.controls(bars, ev, rng)[0]
            rows.append(ev)
    return pd.DataFrame(rows)


def split(d: pd.DataFrame, up: pd.Series, rng) -> dict:
    d = d.copy()
    d["entry_date"] = pd.to_datetime(d["entry_date"])
    d["up"] = up.iloc[up.index.searchsorted(d["entry_date"]) - 1].to_numpy()
    d["ex"] = d["ret"] - d["ctrl_ret"]
    d["block"] = up.index.searchsorted(d["entry_date"]) // vb.BLOCK
    out = {}
    for pat, g in d.groupby("pattern"):
        fav_up = pat in TREND
        r = {}
        for lab, x in (("상승 추세", g[g.up]), ("그 외", g[~g.up])):
            if len(x) < 10:
                r[lab] = {"n": int(len(x))}
                continue
            lo, hi = vb.boot_mean(x["ex"].to_numpy(), x["block"].to_numpy(), rng)
            r[lab] = {"n": int(len(x)), "excess": round(float(x["ex"].mean()), 4), "ci": [round(lo, 4), round(hi, 4)]}
        fav, opp = (r["상승 추세"], r["그 외"]) if fav_up else (r["그 외"], r["상승 추세"])
        if "excess" in fav and "excess" in opp:
            gap = fav["excess"] - opp["excess"]
            r["유리한 쪽"] = "상승 추세" if fav_up else "그 외"
            r["차이"] = round(gap, 4)
            r["verdict"] = ("재현" if fav["ci"][0] > 0 and gap >= 0.005 else
                            "방향만 일치" if fav["excess"] > 0 and gap >= 0.005 else "채택 안 함")
        out[pat] = r
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    up = kospi_trend()
    res = {}
    seen = pd.read_csv(results_dir() / "pattern_validation_events.csv")
    k1 = pd.concat([seen[seen.universe == "oos_kospi"], box_and_kalman(vs._codes("oos_codes.json"), rng)])
    k2 = pd.concat([pd.read_csv(results_dir() / "pattern_validation_kospi2_events.csv"),
                    box_and_kalman(vs._codes("oos_kospi2_codes.json"), rng)])
    res["코스피 1~200"] = split(k1, up, rng)
    res["코스피 다음 200"] = split(k2, up, rng)
    res["코스닥 밖(참고)"] = split(pd.read_csv(results_dir() / "pattern_validation_kosdaq_events.csv"), up, rng)
    res["센서(참고)"] = split(seen[seen.universe == "stocks"], up, rng)
    final = {}
    for pat in set(res["코스피 1~200"]) | set(res["코스피 다음 200"]):
        vs_ = [res[u].get(pat, {}).get("verdict", "채택 안 함") for u in JUDGE]
        final[pat] = ("재현" if all(v == "재현" for v in vs_) else
                      "방향만 일치" if all(v in ("재현", "방향만 일치") for v in vs_) else "채택 안 함")
    res["최종"] = final
    (results_dir() / "regime_validation.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    pats = sorted(res["최종"])
    for pat in pats:
        side = "상승 추세 유리 예측" if pat in TREND else "그 외 유리 예측"
        print(f"\n{pat} ({side})")
        print(f"  최종: {res['최종'].get(pat, '-')}")
        for u, r in res.items():
            if u == "최종":
                continue
            x = r.get(pat)
            if not x:
                continue
            cell = lambda k: (f"n={x[k]['n']} {x[k]['excess']:+.2%} [{x[k]['ci'][0]:+.2%}, {x[k]['ci'][1]:+.2%}]"
                              if "excess" in x.get(k, {}) else f"n={x.get(k, {}).get('n', 0)}")
            v = f" → {x['verdict']}" if u in JUDGE and "verdict" in x else ""
            print(f"  {u:6s} 상승 추세 {cell('상승 추세')} | 그 외 {cell('그 외')} | 차이(유리−반대) {x.get('차이', float('nan')):+.2%}{v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
