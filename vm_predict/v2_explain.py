"""Stage 2.5(선택): 특정 날짜의 신호(breadth 또는 T²)가 왜 그렇게 나왔는지 종목별/신호별로 역분해.

breadth·T² 둘 다 '바스켓 평균'이라, 그날의 값을 종목별/신호별로 정확히 분해할 수 있다(근사가 아님):
  - breadth: 그날 각 센서의 VM Score 자체가 기여분이다(균등가중 평균이므로) — 높은/낮은 순으로
    정렬하면 그날 breadth를 밀어올린/끌어내린 종목이 바로 보인다.
  - T²: 신호별(band_z/vol_z/profile_ratio) 무조건부 기여도로 정확히 분해된다(Σ기여도=T², v2_hotelling
    참고). 가장 큰 기여 신호를 고른 뒤, 그 신호의 그날 종목별 raw 값을 다시 정렬해 어떤 종목이
    바스켓 평균을 그렇게 끌고 갔는지 본다 — 반도체 FDC의 'T² 기여도 플롯 → 챔버/센서 드릴다운'과
    같은 2단계 root-cause 절차.

한계: 이건 이미 확정된(유의미하다고 검증된) breadth/T² 신호 하나를 사후에 설명하는 진단 도구다.
"어떤 종목이 제일 잘 맞히는가"를 새로 검색하는 게 아니므로, 이전에 찾은 다중비교 함정과는
성격이 다르다 — 다만 상관된 신호(band_z·vol_z 등) 간 기여를 완전히 분리하진 못하는 건 T² 분해의
알려진 한계이니 "1차 진단"으로만 쓸 것.

  python v2_explain.py --basket <이름> --date YYYY-MM-DD [--signal breadth|t2]
  --date 생략 시 가장 최근 유효 날짜. --signal 생략 시 둘 다 출력.
  python v2_explain.py --basket <이름> --report            # FDC 보고서 톤 요약문 (방향예측 + 이상탐지)
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "vm_predict", _ROOT / "pair_spc"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import argparse
import sys

import pandas as pd

from v2_compute_engine import (build_breadth, build_sensor_features, build_t2_features, confusion_matrix,
                                full_report)
from v2_config import Params, get_basket
from v2_cusum import cusum, rolling_innovation
from v2_datastore import load_bars
from v2_hotelling import hotelling_t2
from v2_signals import T2_FEATURE_KEYS, kalman_features, vm_scores


def explain_breadth(basket: dict, vm: dict, breadth_df: pd.DataFrame, p: Params, d: pd.Timestamp) -> None:
    breadth = breadth_df["breadth"].get(d)
    print(f"\n=== Breadth 분해 — {basket['name']} {d.date()} ===")
    if pd.isna(breadth):
        print("  (그날 breadth 결측)")
        return
    print(f"  breadth={breadth:.3f}  threshold={p.breadth_threshold}  "
          f"{'>= 임계값(상승예측)' if breadth >= p.breadth_threshold else '< 임계값(하락예측)'}")
    rows = []
    for s in basket["sensors"]:
        code = s["code"]
        if code not in vm or d not in vm[code].index:
            continue
        v = vm[code].loc[d, "vm_score"]
        rows.append((s["name"], v))
    rows.sort(key=lambda r: (r[1] is None or pd.isna(r[1]), -(r[1] if pd.notna(r[1]) else 0)))
    print(f"  {'종목':16s} {'VM Score':>10s}  (균등가중 평균이 breadth이므로, 이 값 자체가 기여분)")
    for name, v in rows:
        mark = "  <- 결측(집계 제외)" if pd.isna(v) else (" ↑" if v >= breadth else " ↓")
        vs = "NaN" if pd.isna(v) else f"{v:.3f}"
        print(f"  {name:16s} {vs:>10s}{mark}")


def explain_t2(basket: dict, vm: dict, t2_feats: pd.DataFrame, t2, p: Params, d: pd.Timestamp) -> None:
    print(f"\n=== T² 분해 — {basket['name']} {d.date()} ===")
    if d not in t2.t2.index or pd.isna(t2.t2.get(d)):
        print("  (그날 T² 결측)")
        return
    val, ucl = t2.t2.loc[d], t2.ucl
    print(f"  T²={val:.2f}  UCL={ucl:.2f}  {'-> 관리이탈(breach)' if val > ucl else '-> 정상범위'}")
    contrib = t2.contributions.loc[d].sort_values(ascending=False)
    print(f"  {'신호':16s} {'기여도':>10s} {'비중':>8s}")
    for key, c in contrib.items():
        print(f"  {key:16s} {c:10.2f} {c/val:7.1%}")
    top_key = contrib.index[0]

    print(f"\n  -> 가장 큰 기여 신호 '{top_key}'를 그날 종목별 raw 값으로 드릴다운:")
    rows = []
    for s in basket["sensors"]:
        code = s["code"]
        if code not in vm or d not in vm[code].index:
            continue
        raw = vm[code].loc[d, top_key]
        rows.append((s["name"], raw))
    rows.sort(key=lambda r: (r[1] is None or pd.isna(r[1]), -(r[1] if pd.notna(r[1]) else 0)))
    mean_val = t2_feats.loc[d, top_key]
    print(f"     바스켓 평균({top_key})={mean_val:.3f}")
    for name, raw in rows:
        rs = "NaN" if pd.isna(raw) else f"{raw:.3f}"
        print(f"     {name:16s} {rs:>10s}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--basket")
    ap.add_argument("--date")
    ap.add_argument("--signal", choices=["breadth", "t2"])
    ap.add_argument("--report", action="store_true", help="FDC 보고서 톤 요약문 출력")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    p = Params()
    basket = get_basket(args.basket)
    sensor_feats, rs = build_sensor_features(basket, p)
    vm = vm_scores(sensor_feats, rs, p)
    breadth_df = build_breadth(vm, p)
    t2_feats = build_t2_features(vm, p)
    t2 = hotelling_t2(t2_feats, p.t2_window, p.t2_alpha)

    if args.date:
        d = pd.Timestamp(args.date)
    else:
        valid = breadth_df["breadth"].dropna().index
        d = valid.max() if len(valid) else breadth_df.index.max()

    if args.report:
        target_bars = load_bars(basket["name"], basket["target"]["code"])
        target_kf = kalman_features(target_bars, p)
        target_cusum = cusum(target_kf["innov"], target_kf["sigma"], p.cusum_k, p.cusum_h)
        breadth_innov, breadth_sigma = rolling_innovation(breadth_df["breadth"], p.cusum_sigma_window)
        breadth_cusum = cusum(breadth_innov, breadth_sigma, p.cusum_k, p.cusum_h)
        cm = confusion_matrix(breadth_df["breadth"], target_bars["close"], p)
        print(full_report(basket, vm, breadth_df, t2_feats, t2, cm, target_cusum, breadth_cusum, p, d))
        return 0

    if args.signal in (None, "breadth"):
        explain_breadth(basket, vm, breadth_df, p, d)
    if args.signal in (None, "t2"):
        explain_t2(basket, vm, t2_feats, t2, p, d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
