"""코스피 추세 상태 가설 — 검증이력 9.37 사전 등록 기준 그대로(새 표본: 코스피 시총 다음 200).

가설: 복귀일 코스피(069500)가 상승 추세(종가 > 60일선 그리고 60일선이 20일 전보다 높음)인 가짜 이탈 사건은
나머지보다 평균 R(순수익 ÷ 손절 거리)이 0.2R 이상 낮다.
  python research/validate_market_trend.py            → results/market_trend_validation.json
  python research/validate_market_trend.py --seen     # 이미 본 3표본에서 같은 계산(참고)
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

import explore_nolift as en
import validate_box as vb
import validate_stops as vs
from v2_config import LONG_HISTORY, results_dir
from v2_datastore import load_bars


def kospi_trend() -> pd.Series:
    k = load_bars(LONG_HISTORY, "069500")["close"]
    ma60 = k.rolling(60).mean()
    return (k > ma60) & (ma60 > ma60.shift(20))


def label(d: pd.DataFrame, up: pd.Series) -> pd.DataFrame:
    i = up.index.searchsorted(d["entry_date"]) - 1                  # 진입 전날 = 복귀일
    d = d.assign(up=up.iloc[i].to_numpy(), R=(d["ret"] - vb.COST) / d["stop_pct"].clip(lower=0.01))
    d["block"] = up.index.searchsorted(d["entry_date"]) // vb.BLOCK
    return d


def diff_ci(d: pd.DataFrame, rng) -> tuple[float, float, float]:
    """(나머지 − 상승 추세) 평균 R 차이와 20일 블록 부트스트랩 95% CI."""
    g = d.groupby("block")
    blocks = [(x.loc[x.up, "R"].sum(), x.up.sum(), x.loc[~x.up, "R"].sum(), (~x.up).sum()) for _, x in g]
    a = np.array(blocks, float)
    idx = rng.integers(0, len(a), (vb.N_BOOT, len(a)))
    s = a[idx].sum(1)
    boot = s[:, 2] / s[:, 3] - s[:, 0] / s[:, 1]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    point = d.loc[~d.up, "R"].mean() - d.loc[d.up, "R"].mean()
    return float(point), float(lo), float(hi)


def report(d: pd.DataFrame, name: str, rng) -> dict:
    u, o = d[d.up], d[~d.up]
    ulo, uhi = vb.boot_mean(u["R"].to_numpy(), u["block"].to_numpy(), rng)
    diff, lo, hi = diff_ci(d, rng)
    out = {"n": int(len(d)), "up_n": int(len(u)), "up_R": round(float(u.R.mean()), 3), "up_R_ci": [round(ulo, 3), round(uhi, 3)],
           "up_nolift": round(float(u.nolift.mean()), 3), "other_n": int(len(o)), "other_R": round(float(o.R.mean()), 3),
           "other_nolift": round(float(o.nolift.mean()), 3), "diff": round(diff, 3), "diff_ci": [round(lo, 3), round(hi, 3)]}
    print(f"{name}: 상승 추세 n={len(u)} 안 뜸 {u.nolift.mean():.0%} R {u.R.mean():+.2f} [{ulo:+.2f}, {uhi:+.2f}] | "
          f"나머지 n={len(o)} 안 뜸 {o.nolift.mean():.0%} R {o.R.mean():+.2f} | 차이 {diff:+.2f} [{lo:+.2f}, {hi:+.2f}]")
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    up = kospi_trend()
    if "--seen" in sys.argv:
        for f in ("nolift_sensor", "nolift_kospi", "nolift_kosdaq"):
            report(label(pd.read_csv(results_dir() / f"{f}.csv", parse_dates=["entry_date"]), up), f, rng)
        return 0
    d = label(en.features(vs._codes("oos_kospi2_codes.json"), "코스피 다음 200"), up)
    d.to_csv(results_dir() / "market_trend_kospi2.csv", index=False)
    r = report(d, "새 표본", rng)
    if r["diff"] >= 0.2 and r["diff_ci"][0] > 0:
        verdict = "재현"
    elif r["diff"] >= 0.2:
        verdict = "방향만 일치"
    else:
        verdict = "채택 안 함"
    r["verdict"], r["skip_candidate"] = verdict, bool(r["up_R_ci"][1] < 0.1)
    print("가설 1:", verdict, "· 건너뛰기 후보" if r["skip_candidate"] else "")
    r2 = retest_by_trend(up, rng)
    (results_dir() / "market_trend_validation.json").write_text(json.dumps({"가설1": r, "가설2": r2}, ensure_ascii=False, indent=1),
                                                               encoding="utf-8")
    return 0


def retest_by_trend(up: pd.Series, rng) -> dict:
    """가설 2: 돌파 리테스트(9.18 규칙)는 코스피 상승 추세에서만 무작위보다 낫다."""
    rows = []
    for code in vs._codes("oos_kospi2_codes.json"):
        bars = load_bars(LONG_HISTORY, code)
        if bars is None or len(bars) < 300:
            continue
        bars = bars[(bars[["open", "high", "low", "close"]] > 0).all(1)]
        for ev in vb.find_events(bars, np.zeros(len(bars), bool)):
            if ev["pattern"] != "retest":
                continue
            ev["ctrl_ret"] = vb.controls(bars, ev, rng)[0]
            rows.append(ev)
    d = pd.DataFrame(rows)
    d["up"] = up.iloc[up.index.searchsorted(d["entry_date"]) - 1].to_numpy()
    d["ex"] = d["ret"] - d["ctrl_ret"]
    d["block"] = up.index.searchsorted(d["entry_date"]) // vb.BLOCK
    out = {}
    for lab, x in (("상승 추세", d[d.up]), ("그 외", d[~d.up])):
        lo, hi = vb.boot_mean(x["ex"].to_numpy(), x["block"].to_numpy(), rng)
        out[lab] = {"n": int(len(x)), "excess": round(float(x["ex"].mean()), 4), "ci": [round(lo, 4), round(hi, 4)]}
        print(f"가설 2 리테스트 {lab}: n={len(x)} 초과 {x['ex'].mean():+.2%} [{lo:+.2%}, {hi:+.2%}]")
    u, o = out["상승 추세"], out["그 외"]
    gap = u["excess"] - o["excess"]
    out["차이"] = round(gap, 4)
    out["verdict"] = ("재현" if u["ci"][0] > 0 and gap >= 0.005 else
                      "방향만 일치" if u["excess"] > 0 and gap >= 0.005 else "채택 안 함")
    print("가설 2:", out["verdict"], f"(차이 {gap:+.2%})")
    return out


if __name__ == "__main__":
    sys.exit(main())
