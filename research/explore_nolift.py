"""검증된 셋업(9.27)에서 '처음부터 안 뜨는' 거래를 진입 시점 정보로 거를 수 있는지 — 탐색(검증이력 9.35).

  python research/explore_nolift.py            # 센서(찾는 곳)
  python research/explore_nolift.py --oos      # 표본 밖 코스피·코스닥(고정한 후보 확인)
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "vm_predict", _ROOT / "scenario", _ROOT / "research"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import sys
import warnings

import numpy as np
import pandas as pd

import validate_stops as vs
from box_rules import BOX, FAIL_RECOVER
from v2_config import LONG_HISTORY, sensor_universe
from v2_datastore import load_bars

NOLIFT = 0.02


def features(codes: list[str], label: str) -> pd.DataFrame:
    k = load_bars(LONG_HISTORY, "069500")["close"]
    rows = []
    for code in codes:
        bars = load_bars(LONG_HISTORY, code)
        if bars is None or len(bars) < 300:
            continue
        bars = bars[(bars[["open", "high", "low", "close"]] > 0).all(1)]
        o, h, l, c, v = (bars[x].to_numpy(float) for x in ("open", "high", "low", "close", "volume"))
        L = bars["low"].rolling(BOX).min().shift(1).to_numpy()
        v20 = pd.Series(v).rolling(20).mean().shift(1).to_numpy()
        ma20 = pd.Series(c).rolling(20).mean().to_numpy()
        mk = k.reindex(bars.index).ffill()
        mk20 = (mk / mk.shift(20) - 1).to_numpy()
        mk_ma = (mk / mk.rolling(20).mean() - 1).to_numpy()
        for ev in vs.events(bars):
            e, stop, tgt = ev["e"], ev["stops"]["기준"], ev["tgt"]
            r = e - 1
            b = next(j for j in range(r - FAIL_RECOVER, r) if c[j] < L[j] and all(c[x] <= L[j] for x in range(j, r)))
            ret, how, hold = vs.simulate(o, h, l, c, e, stop, tgt, False)
            rows.append({"universe": label, "code": code, "entry_date": bars.index[e], "ret": ret, "how": how,
                         "nolift": h[e:e + hold].max() / o[e] - 1 < NOLIFT,
                         "rec_days": r - b,                                   # 이탈 → 복귀 걸린 날
                         "depth": 1 - l[b:r + 1].min() / L[b],                  # 박스 하단 아래로 얼마나 깊이
                         "rec_close": c[r] / L[b] - 1,                          # 복귀일 종가가 하단 위로 얼마나
                         "rec_vol": v[r] / v20[r],                              # 복귀일 거래량 배수
                         "rec_body": (c[r] - o[r]) / max(h[r] - l[r], 1e-9),    # 복귀일 몸통(+1 = 장대양봉)
                         "gap": o[e] / c[r] - 1,                                # 진입 시가 갭
                         "stop_pct": 1 - stop / o[e], "tgt_pct": tgt / o[e] - 1,
                         "below_ma20": o[e] / ma20[r] - 1,                      # 진입가가 20일선 대비
                         "mk20": mk20[r], "mk_ma": mk_ma[r]})                   # 코스피 20일 수익률, 20일선 대비
    return pd.DataFrame(rows)


def show(d: pd.DataFrame) -> None:
    base = d.nolift.mean()
    print(f"사건 {len(d)} · 안 뜸(+{NOLIFT:.0%} 못 감) {base:.0%} · 평균 {d.ret.mean():+.2%}")
    for col in ("rec_days", "depth", "rec_close", "rec_vol", "rec_body", "gap", "stop_pct", "tgt_pct", "below_ma20", "mk20", "mk_ma"):
        x = d[col]
        bins = [x.min() - 1, 1.5, 2.5, x.max() + 1] if col == "rec_days" else np.unique(x.quantile([0, 1 / 3, 2 / 3, 1]).to_numpy())
        cut = pd.cut(x, bins, include_lowest=True, duplicates="drop")
        parts = []
        for iv, g in d.groupby(cut, observed=True):
            parts.append(f"{iv.left:+.3g}~{iv.right:+.3g}: 안 뜸 {g.nolift.mean():3.0%} 평균 {g.ret.mean():+5.1%} (n={len(g)})")
        print(f" {col:10s} " + " | ".join(parts))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    if "--oos" in sys.argv:
        for f, u in (("oos_codes.json", "코스피 밖"), ("oos_kosdaq_codes.json", "코스닥 밖")):
            d = features(vs._codes(f), u)
            d.to_csv(_ROOT / "results" / f"nolift_{'kospi' if 'kosdaq' not in f else 'kosdaq'}.csv", index=False)
            print(f"\n===== {u} =====")
            show(d)
    else:
        d = features(sorted(sensor_universe()), "센서")
        d.to_csv(_ROOT / "results" / "nolift_sensor.csv", index=False)
        print("===== 센서 =====")
        show(d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
