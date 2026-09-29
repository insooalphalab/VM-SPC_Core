"""정방향 두 도구에 4개 기간(T+1/T+5/T+10/T+20)을 적용한 검증 (검증이력 9.9 사전 등록 그대로).

  A. ETF 자체 예측  — breadth ≥ 0.345 고정 규칙, 라벨 = k일 뒤 ETF 방향(|수익| < 0.2%·√k 보합 제외)
  B. 챔피언-챌린저 — Ridge(피처 5 + 종목 더미), 라벨 = k일 수익률의 바스켓 내 상위/하위 40%

게이트는 stock_track.validation.pooled_gate(기간 단위 합산, 1/바스켓 수 가중, 블록 부트스트랩)를 그대로 쓴다.
운영(T+1) 결과는 건드리지 않고 results/forward_horizons_validation.json 에 검증 결과만 쓴다.

  python research/forward_horizons.py      # vm_predict/v2_run.py 가 5년 데이터로 먼저 돌아 있어야 한다(A)
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "vm_predict", _ROOT / "pair_spc", _ROOT / "dart_events",
           _ROOT / "stock_track", _ROOT / "scenario", _ROOT / "research"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import json
import logging
import math
import sys
import time
from datetime import datetime

import pandas as pd

from v2_config import KST, Params, load_baskets, results_dir
from v2_datastore import load_bars
from vm_spc import labeling
from vm_spc.features import FEATURE_COLS, build_feature_frame
from vm_spc.models.ridge_baseline import fit_ridge, predict_proba_ridge
from vm_spc.pipeline import _design_matrix
from stock_track.model import HORIZONS, horizon_splits
from stock_track.validation import pooled_gate

log = logging.getLogger("vm_spc.forward_horizons")
DEAD_ZONE = 0.002


def etf_rows(basket: dict, summary: dict, k: int) -> pd.DataFrame:
    """A: 고정 규칙 예측을 pooled_gate 입력 형식으로 (p=1 상승 / p=0 하락 — 모든 날이 신호)."""
    ts, br = summary["target_series"], summary["breadth_series"]
    close = pd.Series(ts["close"], index=pd.to_datetime(ts["dates"]), dtype=float)
    breadth = pd.Series(br["breadth"], index=pd.to_datetime(br["dates"]), dtype=float).reindex(close.index)
    ret_k = close.shift(-k) / close - 1
    ok = breadth.notna() & ret_k.notna() & (ret_k.abs() >= DEAD_ZONE * math.sqrt(k))
    thr = summary["params"]["breadth_threshold"]
    return pd.DataFrame({"date": close.index[ok], "ticker_id": basket["target"]["code"],
                         "y": (ret_k[ok] > 0).astype(float).to_numpy(),
                         "p": (breadth[ok] >= thr).astype(float).to_numpy(), "basket": basket["name"]})


def rank_labels(frame: pd.DataFrame, k: int, min_active: int) -> pd.DataFrame:
    """B: labeling.attach_relative_labels 와 같은 규칙을 k일 수익률에 적용."""
    df = frame.sort_values(["ticker_id", "date"]).copy()
    df["fwd"] = df.groupby("ticker_id")["close"].shift(-k) / df["close"] - 1
    lab = labeling.rank_labels(df, "fwd", min_active=min_active)
    return lab.sort_values(["date", "ticker_id"]).reset_index(drop=True)


def rank_oos(basket: dict, frame: pd.DataFrame, tickers: list[str], k: int) -> pd.DataFrame | None:
    lab = rank_labels(frame, k, min(labeling.MIN_ACTIVE, max(2, len(tickers) - 1)))
    if lab.empty:
        return None
    X = _design_matrix(lab, tickers)
    parts = []
    for tr, te in horizon_splits(lab["date"], k):
        if lab.loc[tr, "y"].nunique() < 2:
            continue
        m = fit_ridge(X.iloc[tr], lab.loc[tr, "y"])
        parts.append(pd.DataFrame({"date": lab.loc[te, "date"].to_numpy(), "ticker_id": lab.loc[te, "ticker_id"].to_numpy(),
                                   "y": lab.loc[te, "y"].to_numpy(), "p": predict_proba_ridge(m, X.iloc[te])}))
    return pd.concat(parts, ignore_index=True).assign(basket=basket["name"]) if parts else None


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    t0 = time.time()
    a_rows = {k: [] for k in HORIZONS}
    b_rows = {k: [] for k in HORIZONS}
    p = Params()
    for basket in load_baskets():
        spath = results_dir() / basket["name"] / "v2_summary_metrics.json"
        if spath.exists():
            summary = json.loads(spath.read_text(encoding="utf-8"))
            for k in HORIZONS:
                a_rows[k].append(etf_rows(basket, summary, k))

        target = load_bars(basket["name"], basket["target"]["code"])
        bars = {s["code"]: b for s in basket["sensors"]
                if (b := load_bars(basket["name"], s["code"])) is not None and not b.empty}
        if target is None or len(bars) < 2:
            continue
        frame = build_feature_frame(bars, target, p).dropna(subset=FEATURE_COLS)
        tickers = sorted(bars)
        for k in HORIZONS:
            o = rank_oos(basket, frame, tickers, k)
            if o is not None:
                b_rows[k].append(o)
        log.info("[%s] 완료", basket["name"])

    out = {"generated_at": datetime.now(KST).isoformat(), "A_etf_breadth": {}, "B_champion_rank": {}}
    for key, rows in (("A_etf_breadth", a_rows), ("B_champion_rank", b_rows)):
        for k in HORIZONS:
            if not rows[k]:
                out[key][str(k)] = {"status": "HOLD", "reason": "데이터 없음"}
                continue
            s = pooled_gate(pd.concat(rows[k], ignore_index=True), k)
            out[key][str(k)] = s
            log.info("%s T+%d: %s [%s] 적중 %s, CI %s, 평소 %s, 독립표본 %s", key, k, s["status"], s["reason"],
                     s.get("hcp"), s.get("ci"), s.get("base"), s.get("n_indep"))
    (results_dir() / "forward_horizons_validation.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("완료 — %.1f분", (time.time() - t0) / 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
