"""ETF 순위 점수판 진입점 (검증이력 9.12).

  python etf_rank/run.py          # 계산 → 검증 → results/etf_rank/etf_rank.json → 대시보드·인덱스 재생성
  python etf_rank/run.py --dry    # 검증 결과만 출력

vm_predict/v2_run.py 가 먼저 돌아 v2_summary_metrics.json·data/ 가 최신이어야 한다.
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

from v2_config import KST, load_baskets, results_dir

from etf_rank.model import etf_frame, evaluate, today_scores
from stock_track.data import BENCHMARKS, load_benchmark
from stock_track.model import HORIZONS

log = logging.getLogger("etf_rank")
BENCH_CODE = BENCHMARKS["KOSPI"]["code"]


def out_path():
    return results_dir() / "etf_rank" / "etf_rank.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    t0 = time.time()

    bm = load_benchmark("KOSPI")["close"]
    frames, primary, seen = [], set(), set()
    for b in load_baskets():
        code = b["target"]["code"]
        if code == BENCH_CODE or code in seen:
            continue
        seen.add(code)
        f = etf_frame(b, bm)
        if f is None or f["date"].nunique() < 400:
            log.info("[%s] %s 데이터 부족 — 제외", b["name"], b["target"]["name"])
            continue
        frames.append(f)
        if b["name"].endswith("_to_etf"):
            primary.add(code)
    panel_all = pd.concat(frames, ignore_index=True)
    panel_pri = panel_all[panel_all["code"].isin(primary)]
    log.info("ETF %d개(판정용 테마 대표 %d개)", panel_all["code"].nunique(), len(primary))

    status, info_all, scores = {}, {}, {}
    for k in HORIZONS:
        status[str(k)], _ = evaluate(panel_pri, k)
        info_all[str(k)], _ = evaluate(panel_all, k)
        scores[k] = today_scores(panel_pri, panel_all, k)
        s, a = status[str(k)], info_all[str(k)]
        log.info("T+%d: %s — 순위상관 %s CI %s (%d일, 독립 %s) | 5분위 차 %s | 확신 적중 %s vs 평소 %s || 전체 ETF 순위상관 %s CI %s",
                 k, s["status"], s.get("ic_mean"), s.get("ic_ci"), s.get("n_days", 0), s.get("n_indep"),
                 s.get("quintile_spread"), s.get("hit"), s.get("hit_base"), a.get("ic_mean"), a.get("ic_ci"))
    if args.dry:
        return 0

    latest = panel_all["date"].max()
    etfs = []
    for code, g in panel_all.groupby("code"):
        etfs.append({
            "code": code, "name": g["name"].iloc[0], "basket": g["basket"].iloc[0], "primary": code in primary,
            "p": {str(k): (round(float(scores[k][code]), 4) if code in scores[k] else None) for k in HORIZONS},
            "usual": {str(k): (round(float(g[f"y{k}"].mean()), 4) if g[f"y{k}"].notna().any() else None) for k in HORIZONS},
        })
    payload = {"generated_at": datetime.now(KST).isoformat(), "as_of": latest.strftime("%Y-%m-%d"),
               "benchmark": f"{BENCHMARKS['KOSPI']['name']}({BENCH_CODE})", "horizons": HORIZONS,
               "status": status, "info_all": info_all, "etfs": etfs}
    out_path().parent.mkdir(parents=True, exist_ok=True)
    out_path().write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    from v2_render_dashboard import render_basket
    from vm_spc.build_index import build_index
    for b in load_baskets():
        if (results_dir() / b["name"] / "v2_summary_metrics.json").exists():
            render_basket(b)
    build_index()
    log.info("ETF 순위 완료 — %.1f분", (time.time() - t0) / 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
