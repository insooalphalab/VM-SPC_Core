"""검증된 셋업(9.27)에서 목표(박스 상단) 전에 꺾이는 거래의 모양 — 탐색(검증이력 9.33).

센서 종목에서 찾고, 같은 구간 나누기를 표본 밖(코스피·코스닥)에 그대로 적용해 재현되는지만 본다.
  python research/explore_reversal.py
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "vm_predict", _ROOT / "scenario", _ROOT / "research"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import sys

import numpy as np
import pandas as pd

import validate_stops as vs
from box_rules import MAX_HOLD, t2_flags
from v2_config import LONG_HISTORY, sensor_universe
from v2_datastore import load_bars


def paths(codes: list[str], label: str) -> pd.DataFrame:
    kospi = load_bars(LONG_HISTORY, "069500")["close"]
    rows = []
    for code in codes:
        bars = load_bars(LONG_HISTORY, code)
        if bars is None or len(bars) < 300:
            continue
        bars = bars[(bars[["open", "high", "low", "close"]] > 0).all(1)]
        o, h, l, c, v = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
        ma20 = pd.Series(c).rolling(20).mean().shift(1).to_numpy()
        v20 = pd.Series(v).rolling(20).mean().shift(1).to_numpy()
        t2 = t2_flags(bars)
        mk = kospi.reindex(bars.index).ffill().to_numpy()
        for ev in vs.events(bars):
            e, stop, tgt = ev["e"], ev["stops"]["기준"], ev["tgt"]
            ret, how, hold = vs.simulate(o, h, l, c, e, stop, tgt, False)
            end = e + hold - 1
            touch = next((j for j in range(e, end + 1) if h[j] >= ma20[j]), None)
            r = {"universe": label, "code": code, "how": how, "ret": ret, "hold": hold,
                 "mfe": h[e:end + 1].max() / o[e] - 1, "tgt_pct": tgt / o[e] - 1, "stop_pct": 1 - stop / o[e],
                 "touch": touch is not None}
            if touch is not None:
                k = touch - e + 1
                r.update({"touch_day": k,
                          "vol_x": v[touch] / v20[touch],                          # 20일선 닿은 날 거래량 배수
                          "wick": (h[touch] - c[touch]) / max(h[touch] - l[touch], 1e-9),   # 윗꼬리 비율(1=고가에서 다 밀림)
                          "close_above": c[touch] >= ma20[touch],                # 그날 종가가 20일선 위에서 마감
                          "mkt": mk[touch] / mk[e - 1] - 1,                       # 진입 전날 → 닿은 날 코스피
                          "t2_touch": bool(t2[touch]),
                          "left": tgt / c[touch] - 1})                            # 닿은 날 종가에서 목표까지 남은 거리
            rows.append(r)
    return pd.DataFrame(rows)


def rate(g: pd.DataFrame) -> str:
    if len(g) < 20:
        return f"n={len(g):3d} (적음)"
    return f"n={len(g):4d} 목표 {(g.how == 'target').mean():4.0%} 손절 {(g.how == 'stop').mean():4.0%} 평균 {g.ret.mean():+5.1%}"


def report(d: pd.DataFrame, title: str) -> None:
    print(f"\n===== {title} (사건 {len(d)}) =====")
    st = d[d.how == "stop"]
    print(f"결과: 목표 {(d.how == 'target').mean():.0%} · 손절 {(d.how == 'stop').mean():.0%} · 20일 만료 {(d.how == 'timeout').mean():.0%}")
    print(f"손절 난 거래 중 먼저 20일선을 찍은 비율 {st.touch.mean():.0%}, 목표까지 절반 이상 갔다 온 비율 "
          f"{(st.mfe >= st.tgt_pct / 2).mean():.0%}, 한 번도 +2% 못 간 비율 {(st.mfe < 0.02).mean():.0%}")
    t = d[d.touch.astype(bool)].copy()
    t[["close_above", "t2_touch"]] = t[["close_above", "t2_touch"]].astype(bool)
    print(f"20일선 닿은 거래 {len(t)} → 그 뒤 {rate(t)}")
    print(" 닿은 날짜:", " | ".join(f"{a}: {rate(t[m])}" for a, m in
                                   [("1~2일", t.touch_day <= 2), ("3~5일", t.touch_day.between(3, 5)), ("6일+", t.touch_day >= 6)]))
    print(" 그날 종가:", " | ".join(f"{a}: {rate(t[m])}" for a, m in
                                   [("20일선 위 마감", t.close_above), ("다시 아래 마감", ~t.close_above)]))
    print(" 윗꼬리:  ", " | ".join(f"{a}: {rate(t[m])}" for a, m in
                                   [("짧음(<0.3)", t.wick < 0.3), ("중간", t.wick.between(0.3, 0.6)), ("김(>0.6)", t.wick > 0.6)]))
    print(" 거래량:  ", " | ".join(f"{a}: {rate(t[m])}" for a, m in
                                   [("평소 이하(<1)", t.vol_x < 1), ("1~2배", t.vol_x.between(1, 2)), ("2배+", t.vol_x > 2)]))
    print(" 코스피:  ", " | ".join(f"{a}: {rate(t[m])}" for a, m in
                                   [("하락", t.mkt < 0), ("0~+3%", t.mkt.between(0, 0.03)), ("+3%+", t.mkt > 0.03)]))
    print(" T²:      ", " | ".join(f"{a}: {rate(t[m])}" for a, m in [("여전히 밖", t.t2_touch), ("안으로", ~t.t2_touch)]))
    print(" 남은 거리:", " | ".join(f"{a}: {rate(t[m])}" for a, m in
                                   [("<5%", t.left < 0.05), ("5~10%", t.left.between(0.05, 0.10)), ("10%+", t.left > 0.10)]))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    import warnings
    warnings.filterwarnings("ignore")
    s = paths(sorted(sensor_universe()), "센서")
    report(s, "센서 — 찾는 곳")
    oos = pd.concat([paths(vs._codes("oos_codes.json"), "코스피 밖"), paths(vs._codes("oos_kosdaq_codes.json"), "코스닥 밖")])
    for u, g in oos.groupby("universe"):
        report(g, f"{u} — 확인")
    pd.concat([s, oos]).to_csv(_ROOT / "results" / "reversal_explore.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
