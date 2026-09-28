"""종목 추적(역방향) 진입점. 전 바스켓을 돌려 기간별 합산 게이트까지 판정하고 대시보드를 다시 그린다.

  python stock_track/run.py              # 기준 지수 일봉·ETF 비중 갱신(API) → 계산 → 검증 → 렌더링
  python stock_track/run.py --no-fetch   # 저장된 데이터만으로 (API 호출 없음)

vm_predict/v2_run.py 가 먼저 돌아 data/ 가 최신이어야 한다.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "vm_predict", _ROOT / "pair_spc"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import argparse
import json
import logging
import sys
import time
from datetime import datetime

import pandas as pd

from kis_client import RateLimitedCaller
from v2_config import KST, load_baskets, results_dir
from v2_data_collector import DEFAULT_HISTORY_DAYS

from stock_track.data import BENCHMARKS, kospi_codes, load_weights, update_benchmarks, update_weights
from stock_track.model import FEATURE_SETS, HORIZONS, TRAIN_DAYS, build_panel, run_basket
from stock_track.validation import pooled_gate

log = logging.getLogger("stock_track")
VALIDATION_FILE = "stock_track_validation.json"


def track_path(basket_name: str):
    return results_dir() / basket_name / "stock_track" / "stock_track.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--feature-set", choices=list(FEATURE_SETS), default="price",
                    help="price=가격 10개, price_flow=가격+수급 4개, flow=수급 4개(검증이력 9.8)")
    ap.add_argument("--dry", action="store_true", help="게이트 결과만 출력(결과 파일·대시보드 안 건드림)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    t0 = time.time()

    caller = RateLimitedCaller()
    features = FEATURE_SETS[args.feature_set]
    if args.dry:
        args.no_fetch = args.no_render = True
    if not args.no_fetch:
        log.info("기준 지수(KOSPI200·KOSDAQ150 ETF) 일봉 갱신")
        update_benchmarks(caller, DEFAULT_HISTORY_DAYS)
    kospi = kospi_codes()

    pooled: dict[int, list[pd.DataFrame]] = {k: [] for k in HORIZONS}
    done = []
    for basket in load_baskets():
        panel = build_panel(basket, kospi)
        if panel is not None:
            panel = panel.dropna(subset=features)
        if panel is None or panel["date"].nunique() < TRAIN_DAYS + 60:
            log.info("[%s] 유효 거래일 부족 — 검증 대기", basket["name"])
            if not args.dry:
                track_path(basket["name"]).unlink(missing_ok=True)
            continue
        weights = load_weights(basket["name"])
        if not args.no_fetch:
            try:
                weights = update_weights(caller, basket)
            except Exception:
                log.exception("[%s] ETF 비중 조회 실패 — 이전 값 사용", basket["name"])
        r = run_basket(basket, panel, features)
        for k in HORIZONS:
            if r["oos"][k] is not None:
                pooled[k].append(r["oos"][k])
        done.append((basket, r, weights))
        log.info("[%s] 완료 (%d종목, 기준일 %s)", basket["name"], len(r["market"]), r["as_of"])

    status = {}
    for k in HORIZONS:
        if pooled[k]:
            status[str(k)] = pooled_gate(pd.concat(pooled[k], ignore_index=True), k)
        else:
            status[str(k)] = {"k": k, "status": "HOLD", "reason": "데이터 없음", "hcp": None}
        s = status[str(k)]
        log.info("[%s] T+%d: %s [%s] 고신뢰 적중 %s, CI %s, 평소 %s, 독립표본 %s", args.feature_set, k, s["status"],
                 s["reason"], s.get("hcp"), s.get("ci"), s.get("base"), s.get("n_indep"))
    if args.dry:
        return 0
    (results_dir() / VALIDATION_FILE).write_text(json.dumps(
        {"generated_at": datetime.now(KST).isoformat(), "feature_set": args.feature_set, "horizons": status},
        ensure_ascii=False, indent=1), encoding="utf-8")

    names = {s["code"]: s["name"] for b, _, _ in done for s in b["sensors"]}
    for basket, r, weights in done:
        w = weights.get("weights", {})
        stocks = [{"code": c, "name": names.get(c, c), "market": r["market"][c], "weight": w.get(c),
                   "p": {str(k): (round(float(r["today"][k][c]), 4) if k in r["today"] and c in r["today"][k] else None)
                         for k in HORIZONS},
                   "usual": {str(k): (round(float(r["usual"][k][c]), 4) if c in r["usual"].get(k, {}) else None)
                             for k in HORIZONS}}
                  for c in r["market"]]
        stocks.sort(key=lambda s: (s["weight"] is None, -(s["weight"] or 0)))
        payload = {"basket": basket["name"], "as_of": r["as_of"], "weights_as_of": weights.get("as_of"),
                   "benchmarks": {m: f"{b['name']}({b['code']})" for m, b in BENCHMARKS.items()},
                   "feature_set": args.feature_set, "horizons": HORIZONS, "status": status, "stocks": stocks}
        path = track_path(basket["name"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    if not args.no_render:
        from v2_render_dashboard import render_basket
        for basket, _, _ in done:
            render_basket(basket)
    log.info("종목 추적 완료 — %d개 바스켓, %.1f분", len(done), (time.time() - t0) / 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
