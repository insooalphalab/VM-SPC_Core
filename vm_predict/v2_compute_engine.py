"""Stage 2: VM Score·Index Breadth·컨퓨전매트릭스·CUSUM(양방향)·Hotelling's T² 전부 계산
→ results/{basket}/v2_summary_metrics.json

  python v2_compute_engine.py                       # basket_watchlist.json 의 첫 바스켓
  python v2_compute_engine.py --basket kospi_top10_to_etf
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "vm_predict", _ROOT / "pair_spc"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import argparse
import dataclasses
import json
import logging
import math
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from v2_config import KST, Params, get_basket, results_dir
from v2_cusum import cusum, rolling_innovation
from v2_datastore import load_bars
from v2_hotelling import hotelling_t2
from v2_signals import T2_FEATURE_KEYS, kalman_features, profile_ratio_series, relative_strength_matrix, vm_scores

log = logging.getLogger("v2_compute_engine")


def _breakouts(alarm: pd.Series) -> list[str]:
    """알람이 유지되는 매일이 아니라, False→True 로 처음 돌파하는 날짜만 (차트 마커용)."""
    rising = alarm & ~alarm.shift(1, fill_value=False)
    return [d.strftime("%Y-%m-%d") for d in alarm.index[rising]]


def _series_json(s: pd.Series) -> list:
    return [None if pd.isna(v) else round(float(v), 6) for v in s]


def build_sensor_features(basket: dict, p: Params) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    sensors = [s["code"] for s in basket["sensors"]]
    bars_map, closes = {}, {}
    for code in sensors:
        bars = load_bars(basket["name"], code)
        if bars is None or bars.empty:
            log.warning("%s: 일봉 없음 (v2_data_collector.py 먼저 실행했는지 확인)", code)
            continue
        bars_map[code] = bars
        closes[code] = bars["close"]

    rs = relative_strength_matrix(closes, sensors, p)
    feats = {}
    for code, bars in bars_map.items():
        kf = kalman_features(bars, p)
        kf["profile_ratio"] = profile_ratio_series(bars, p)
        feats[code] = kf
    return feats, rs


def build_breadth(vm: dict[str, pd.DataFrame], p: Params) -> pd.DataFrame:
    scores = pd.DataFrame({c: df["vm_score"] for c, df in vm.items()})
    n = scores.shape[1]
    valid_count = scores.notna().sum(axis=1)
    breadth = scores.mean(axis=1, skipna=True)
    breadth[valid_count / n < p.breadth_min_coverage] = np.nan
    return pd.DataFrame({"breadth": breadth, "valid_count": valid_count, "n_sensors": n})


def build_t2_features(vm: dict[str, pd.DataFrame], p: Params) -> pd.DataFrame:
    """바스켓 평균 3개 신호(수급·상대강도 제외) 시계열 — 섹션 6, T2_FEATURE_KEYS 참고."""
    out = {}
    for key in T2_FEATURE_KEYS:
        cols = pd.DataFrame({c: df[key] for c, df in vm.items()})
        coverage = cols.notna().sum(axis=1) / cols.shape[1]
        mean = cols.mean(axis=1, skipna=True)
        mean[coverage < p.t2_min_coverage] = np.nan
        out[key] = mean
    return pd.DataFrame(out)


def confusion_matrix(breadth: pd.Series, target_close: pd.Series, p: Params) -> dict:
    """익일 등락 예측 vs 실제. 익일 등락폭이 p.dead_zone 미만인 '무승부'인 날은 집계에서 뺀다
    (v2_config.Params.dead_zone 주석에 근거 정리됨)."""
    idx = breadth.index.intersection(target_close.index)
    breadth, close = breadth.reindex(idx), target_close.reindex(idx)
    next_close = close.shift(-1)
    next_ret = next_close / close - 1
    actual_up = next_ret > 0
    predicted_up = breadth >= p.breadth_threshold
    mask = breadth.notna() & next_ret.notna() & (next_ret.abs() >= p.dead_zone)
    dead_zone_excluded = int((breadth.notna() & next_ret.notna() & (next_ret.abs() < p.dead_zone)).sum())
    pred, act = predicted_up[mask], actual_up[mask]
    tp, fp = int((pred & act).sum()), int((pred & ~act).sum())
    fn, tn = int((~pred & act).sum()), int((~pred & ~act).sum())
    n = tp + fp + fn + tn
    n_pos, precision = tp + fp, (tp / (tp + fp) if (tp + fp) else None)
    base_rate = (tp + fn) / n if n else None
    # precision이 '예측 안 했을 때 기대되는 상승 확률(base_rate)'보다 유의하게 높은지 이항비율 z검정.
    # 단일 임계값에서 나온 값이며, 다른 임계값들을 같이 훑어서 제일 높은 걸 고르면(다중비교) 이
    # 검정의 전제가 깨진다 — z는 '사전에 고정한 이 설정 하나'에 대해서만 해석 가능하다.
    z = None
    if n_pos and precision is not None and base_rate not in (None, 0, 1):
        se = math.sqrt(base_rate * (1 - base_rate) / n_pos)
        z = (precision - base_rate) / se if se > 0 else None
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n_days": n,
        "accuracy": (tp + tn) / n if n else None,
        "precision": precision,
        "recall": tp / (tp + fn) if (tp + fn) else None,
        "base_rate": base_rate,
        "z_score": z,
        "significant_95": bool(z is not None and z >= 1.96),
        "dead_zone": p.dead_zone,
        "dead_zone_excluded": dead_zone_excluded,
        "predicted_up": predicted_up, "actual_up": actual_up,
    }


def _top_contrib_stocks(basket: dict, vm: dict, key: str, d: pd.Timestamp, n: int = 2) -> list[str]:
    rows = []
    for s in basket["sensors"]:
        code = s["code"]
        if code not in vm or d not in vm[code].index:
            continue
        raw = vm[code].loc[d, key]
        if pd.notna(raw):
            rows.append((s["name"], raw))
    rows.sort(key=lambda r: -abs(r[1]))
    return [name for name, _ in rows[:n]]


def full_report(basket: dict, vm: dict, breadth_df: pd.DataFrame, t2_feats: pd.DataFrame, t2,
                 cm: dict, target_cusum, breadth_cusum, p: Params, d: pd.Timestamp) -> str:
    """FDC 일일 보고서 톤의 요약문. '방향 예측'과 '개별 이상탐지'를 명확히 분리해서 서술한다 —
    예측은 검증된 breadth 신호의 근거(z, 적중률)를 대고, 탐지는 원인 특정 없이 '어디부터 봐야
    하는지'만 좁혀서 사람이 확인하도록 안내한다(원인 자동판정은 하지 않음). v2_explain.py의
    CLI(--report)와 run_basket()의 대시보드 JSON 페이로드(daily_report) 양쪽에서 재사용한다."""
    lines = [f"[{basket['name']} 일일 리포트 — {d.date()}]", ""]

    breadth = breadth_df["breadth"].get(d)
    if pd.isna(breadth):
        lines.append("■ 방향 예측: breadth 결측으로 산출 불가(센서 커버리지 부족).")
    else:
        up = breadth >= p.breadth_threshold
        z, prec, base = cm.get("z_score"), cm.get("precision"), cm.get("base_rate")
        sig = "통계적으로 유의(95%)" if cm.get("significant_95") else "통계적으로 유의하지 않음 — 참고용"
        lines.append(
            f"■ 방향 예측: {basket['target']['name']} 익일 {'상승' if up else '하락'} 우세 "
            f"(breadth={breadth:.3f}, 임계값 {p.breadth_threshold})"
        )
        lines.append(f"   근거: 과거 동일 임계값 기준 적중률 {prec:.1%} vs 기저율 {base:.1%} (z={z:.2f}, {sig})")
        lines.append("   ※ 바스켓 전체 평균 기준 예측이며, 개별 종목 단위 예측력은 검증되지 않았습니다.")

    lines.append("")
    detections = []
    if d in target_cusum.alarm_up.index and bool(target_cusum.alarm_up.get(d)):
        detections.append(f"타겟({basket['target']['name']}) 가격 CUSUM 상방 이탈")
    if d in target_cusum.alarm_down.index and bool(target_cusum.alarm_down.get(d)):
        detections.append(f"타겟({basket['target']['name']}) 가격 CUSUM 하방 이탈")
    if d in breadth_cusum.alarm_up.index and bool(breadth_cusum.alarm_up.get(d)):
        detections.append("바스켓 breadth CUSUM 상방 이탈")
    if d in breadth_cusum.alarm_down.index and bool(breadth_cusum.alarm_down.get(d)):
        detections.append("바스켓 breadth CUSUM 하방 이탈")

    t2_breach = d in t2.t2.index and pd.notna(t2.t2.get(d)) and t2.t2.loc[d] > t2.ucl
    if t2_breach:
        contrib = t2.contributions.loc[d].sort_values(ascending=False)
        top_key = contrib.index[0]
        top_stocks = _top_contrib_stocks(basket, vm, top_key, d, n=2)
        detections.append(
            f"Hotelling's T² 관리이탈 (T²={t2.t2.loc[d]:.1f} > UCL={t2.ucl:.1f}, "
            f"주요 기여신호='{top_key}'({contrib.iloc[0]/t2.t2.loc[d]:.0%}), "
            f"관련 상위 종목: {', '.join(top_stocks) if top_stocks else '특정 불가'})"
        )

    if detections:
        lines.append("■ 이상탐지 (개별 원인은 자동 특정되지 않음 — 아래 종목부터 우선 검토 바랍니다):")
        for det in detections:
            lines.append(f"   - {det}")
        lines.append("   ※ 위 방향 예측과 별개로 감지된 개별 변동입니다. 예측 방향과 반대되는 개별 종목의")
        lines.append("     수치 이상이 있을 수 있으므로, 해당 종목은 사전에 개별 검토 후 보고 바랍니다.")
    else:
        lines.append("■ 이상탐지: 현재 감지된 개별 이상 없음 (정상 범위).")

    return "\n".join(lines)


def run_basket(basket: dict, p: Params | None = None) -> int:
    """바스켓 하나에 대해 Stage2 전체를 계산해 results/{basket}/v2_summary_metrics.json 을 쓴다.

    v2_run.py(여러 바스켓 일괄 실행)와 이 파일의 CLI(main) 양쪽에서 재사용한다.
    """
    p = p or Params()
    target_code, target_name = basket["target"]["code"], basket["target"]["name"]
    target_bars = load_bars(basket["name"], target_code)
    if target_bars is None or target_bars.empty:
        log.error("타겟 %s(%s) 일봉 없음 — v2_data_collector.py 먼저 실행", target_name, target_code)
        return 1

    log.info("바스켓 '%s': 센서 신호 계산 중...", basket["name"])
    sensor_feats, rs = build_sensor_features(basket, p)
    if not sensor_feats:
        log.error("유효한 센서 일봉이 하나도 없음")
        return 1
    vm = vm_scores(sensor_feats, rs, p)
    breadth_df = build_breadth(vm, p)

    log.info("컨퓨전 매트릭스 계산 중...")
    cm = confusion_matrix(breadth_df["breadth"], target_bars["close"], p)

    log.info("CUSUM(타겟 ETF 칼만 잔차 + breadth) 계산 중...")
    target_kf = kalman_features(target_bars, p)
    target_cusum = cusum(target_kf["innov"], target_kf["sigma"], p.cusum_k, p.cusum_h)
    breadth_innov, breadth_sigma = rolling_innovation(breadth_df["breadth"], p.cusum_sigma_window)
    breadth_cusum = cusum(breadth_innov, breadth_sigma, p.cusum_k, p.cusum_h)

    log.info("Hotelling's T² 계산 중...")
    t2_feats = build_t2_features(vm, p)
    t2 = hotelling_t2(t2_feats, p.t2_window, p.t2_alpha)

    valid_breadth_dates = breadth_df["breadth"].dropna().index
    report_date = valid_breadth_dates.max() if len(valid_breadth_dates) else breadth_df.index.max()
    report_text = full_report(basket, vm, breadth_df, t2_feats, t2, cm, target_cusum, breadth_cusum, p, report_date)

    sensor_summary = []
    for s in basket["sensors"]:
        code = s["code"]
        if code not in vm:
            sensor_summary.append({"code": code, "name": s["name"], "available": False})
            continue
        df = vm[code]
        latest = df.dropna(subset=["vm_score"])
        sensor_summary.append({
            "code": code, "name": s["name"], "available": True,
            "valid_ratio": round(float(df["vm_score"].notna().mean()), 4),
            "latest_date": None if latest.empty else latest.index[-1].strftime("%Y-%m-%d"),
            "latest_vm_score": None if latest.empty else round(float(latest["vm_score"].iloc[-1]), 4),
            "latest_buy_candidate": None if latest.empty else bool(latest["buy_candidate"].iloc[-1]),
        })

    dates = target_bars.index
    predicted_up = cm["predicted_up"].reindex(breadth_df.index)
    actual_up = cm["actual_up"].reindex(breadth_df.index)
    payload = {
        "basket": {"name": basket["name"], "target": basket["target"], "sensors": basket["sensors"]},
        "period": {"start": dates[0].strftime("%Y-%m-%d"), "end": dates[-1].strftime("%Y-%m-%d")},
        "generated_at": datetime.now(KST).isoformat(),
        "params": dataclasses.asdict(p),
        "target_series": {
            "dates": [d.strftime("%Y-%m-%d") for d in dates],
            "open": _series_json(target_bars["open"]), "high": _series_json(target_bars["high"]),
            "low": _series_json(target_bars["low"]), "close": _series_json(target_bars["close"]),
            "kalman_level": _series_json(np.exp(target_kf["level"])),
            "cusum_pos": _series_json(target_cusum.c_pos), "cusum_neg": _series_json(target_cusum.c_neg),
            "cusum_alarm_up_dates": _breakouts(target_cusum.alarm_up),
            "cusum_alarm_down_dates": _breakouts(target_cusum.alarm_down),
        },
        "breadth_series": {
            "dates": [d.strftime("%Y-%m-%d") for d in breadth_df.index],
            "breadth": _series_json(breadth_df["breadth"]),
            "valid_count": [int(v) for v in breadth_df["valid_count"]],
            "n_sensors": int(breadth_df["n_sensors"].iloc[0]) if len(breadth_df) else 0,
            "cusum_pos": _series_json(breadth_cusum.c_pos), "cusum_neg": _series_json(breadth_cusum.c_neg),
            "cusum_alarm_up_dates": _breakouts(breadth_cusum.alarm_up),
            "cusum_alarm_down_dates": _breakouts(breadth_cusum.alarm_down),
            "predicted_up": [None if pd.isna(v) else bool(v) for v in predicted_up],
            "actual_up": [None if pd.isna(v) else bool(v) for v in actual_up],
        },
        "t2_series": {
            "dates": [d.strftime("%Y-%m-%d") for d in t2.t2.index],
            "t2": _series_json(t2.t2), "ucl": round(t2.ucl, 4),
            "breach_dates": [d.strftime("%Y-%m-%d") for d in t2.t2.index[t2.t2 > t2.ucl]],
        },
        "confusion_matrix": {k: cm[k] for k in ("tp", "fp", "fn", "tn", "n_days", "accuracy", "precision", "recall",
                                                 "base_rate", "z_score", "significant_95",
                                                 "dead_zone", "dead_zone_excluded")},
        "sensor_summary": sensor_summary,
        "daily_report": {"date": report_date.strftime("%Y-%m-%d"), "text": report_text},
    }

    out_dir = results_dir() / basket["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "v2_summary_metrics.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    acc = f"{cm['accuracy']:.1%}" if cm["accuracy"] is not None else "N/A"
    log.info("저장 완료: %s (TP=%d FP=%d FN=%d TN=%d, Accuracy=%s)",
             out_path, cm["tp"], cm["fp"], cm["fn"], cm["tn"], acc)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--basket")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    return run_basket(get_basket(args.basket))


if __name__ == "__main__":
    sys.exit(main())
