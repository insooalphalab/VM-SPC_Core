"""피처·라벨·기간별 Walk-Forward. 1단계 프로토타입은 Ridge 하나만 쓴다(설계 9.6: 표본이 적은 긴 기간은
Ridge만, 챔피언-챌린저에서도 LightGBM 이 Ridge 를 거의 못 이겼음).

피처(모두 t 시점까지의 값만 사용):
  종목 통계 5개  — vm_spc.features 그대로(칼만 잔차·가격 Z·CUSUM·T²·타겟 ETF 대비 공적분 잔차 Z)
  ETF·지수 흐름 5개 — 타겟 ETF 5일/20일 로그수익, 소속 시장 지수 20일 로그수익, 종목의 ETF 대비 5일/20일 상대수익
라벨(k 기간): 1(종목 k일 수익률 - 소속 시장 지수 k일 수익률 > 0)
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "vm_predict", _ROOT / "pair_spc"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import numpy as np
import pandas as pd

from v2_config import Params
from v2_datastore import load_bars
from vm_spc.features import FEATURE_COLS as STAT_COLS, build_feature_frame
from vm_spc.models.ridge_baseline import fit_ridge, predict_proba_ridge

from stock_track.data import load_benchmark, market_of
from stock_track.flow import FLOW_FEATURES, flow_features

HORIZONS = [1, 5, 10, 20]
FLOW_COLS = ["etf_mom5", "etf_mom20", "bm_mom20", "rel_etf5", "rel_etf20"]
FEATURES = [*STAT_COLS, *FLOW_COLS]
# 검증이력 9.8: 1차 판정 = 가격+수급, 수급 단독은 참고용
FEATURE_SETS = {"price": FEATURES, "price_flow": [*FEATURES, *FLOW_FEATURES], "flow": FLOW_FEATURES}
TRAIN_DAYS = 250
DEAD_ZONE = 0.002
BASE_EMBARGO = 10         # 기존 splitter 와 같은 롤링 피처 기억 차단 — 기간 k 가 더 길면 k


def test_days(k: int) -> int:
    return 60 if k <= 5 else 120


def build_panel(basket: dict, kospi: set[str], p: Params | None = None) -> pd.DataFrame | None:
    target = load_bars(basket["name"], basket["target"]["code"])
    if target is None or target.empty:
        return None
    bars = {s["code"]: b for s in basket["sensors"]
            if (b := load_bars(basket["name"], s["code"])) is not None and not b.empty}
    if len(bars) < 2:
        return None
    panel = build_feature_frame(bars, target, p or Params())
    etf_log = np.log(target["close"])
    etf = pd.DataFrame({"etf_mom5": etf_log.diff(5), "etf_mom20": etf_log.diff(20)})
    bms = {m: load_benchmark(m) for m in ("KOSPI", "KOSDAQ")}

    frames = []
    for code, g in panel.groupby("ticker_id"):
        g = g.set_index("date").sort_index()
        mkt = market_of(code, kospi)
        bm = bms[mkt]
        if bm is None or bm.empty:
            continue
        slog, blog = np.log(g["close"]), np.log(bm["close"]).reindex(g.index)
        g = g.join(etf, how="left")
        g["bm_mom20"] = blog.diff(20)
        g["rel_etf5"] = slog.diff(5) - g["etf_mom5"]
        g["rel_etf20"] = slog.diff(20) - g["etf_mom20"]
        g["market"] = mkt
        ff = flow_features(code, bars[code])
        g = g.join(ff if ff is not None else pd.DataFrame(index=g.index, columns=FLOW_FEATURES, dtype=float), how="left")
        for k in HORIZONS:
            stock_fwd = g["close"].shift(-k) / g["close"] - 1
            bm_fwd = bm["close"].shift(-k).reindex(g.index) / bm["close"].reindex(g.index) - 1
            excess = stock_fwd - bm_fwd
            # 다른 도구와 같게 애매한 차이는 제외: |초과수익| < 0.2%·√k 인 날은 라벨 없음
            decided = excess.notna() & (excess.abs() >= DEAD_ZONE * np.sqrt(k))
            g[f"y{k}"] = (excess > 0).astype(float).where(decided)
        frames.append(g.reset_index())
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True)
    all_feats = [*FEATURES, *FLOW_FEATURES]
    out[all_feats] = out[all_feats].replace([np.inf, -np.inf], np.nan)
    return out.dropna(subset=FEATURES).sort_values(["date", "ticker_id"], ignore_index=True)


def horizon_splits(dates: pd.Series, k: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """[train 250일 - purge k] --embargo max(10,k)-- [test T일 중 앞 T-k일만 평가] → T일씩 슬라이딩.
    purge: train 끝 k일의 라벨은 test 쪽 미래를 본다. test 끝 k일: 라벨이 다음 구간으로 넘어간다."""
    dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    uniq = np.array(sorted(dates.unique()))
    pos = pd.Series(np.arange(len(uniq)), index=uniq).reindex(dates.to_numpy()).to_numpy()
    T, emb = test_days(k), max(BASE_EMBARGO, k)
    splits, start = [], 0
    while True:
        tr_lo, tr_hi = start, start + TRAIN_DAYS - k
        te_lo = start + TRAIN_DAYS + emb
        te_hi = min(te_lo + T, len(uniq)) - k
        if te_hi - te_lo < 20:
            break
        splits.append((np.where((pos >= tr_lo) & (pos < tr_hi))[0], np.where((pos >= te_lo) & (pos < te_hi))[0]))
        if te_lo + T >= len(uniq):
            break
        start += T
    return splits


def run_basket(basket: dict, panel: pd.DataFrame, features: list[str] = FEATURES) -> dict:
    """기간별 OOS 예측(검증용)과 오늘 예측·종목별 평소 비율을 만든다."""
    panel = panel.dropna(subset=features)
    oos, today, usual = {}, {}, {}
    latest = panel["date"].max()
    now_rows = panel[panel["date"] == latest]
    for k in HORIZONS:
        lab = panel.dropna(subset=[f"y{k}"]).reset_index(drop=True)
        parts = []
        for tr, te in horizon_splits(lab["date"], k):
            y_tr = lab.loc[tr, f"y{k}"]
            if y_tr.nunique() < 2:
                continue
            m = fit_ridge(lab.loc[tr, features], y_tr)
            part = lab.loc[te, ["date", "ticker_id", f"y{k}"]].rename(columns={f"y{k}": "y"})
            part["p"] = predict_proba_ridge(m, lab.loc[te, features])
            parts.append(part)
        oos[k] = pd.concat(parts, ignore_index=True).assign(basket=basket["name"]) if parts else None

        # 오늘 예측: 라벨이 확정된 가장 최근 250거래일로 학습
        recent = lab[lab["date"] >= np.sort(lab["date"].unique())[-TRAIN_DAYS]] if lab["date"].nunique() >= TRAIN_DAYS else None
        if recent is not None and recent[f"y{k}"].nunique() == 2 and not now_rows.empty:
            m = fit_ridge(recent[features], recent[f"y{k}"])
            today[k] = dict(zip(now_rows["ticker_id"], predict_proba_ridge(m, now_rows[features])))
        usual[k] = lab.groupby("ticker_id")[f"y{k}"].mean().to_dict()   # 초과수익 양성 비율(평소)
    return {"as_of": latest.strftime("%Y-%m-%d"), "oos": oos, "today": today, "usual": usual,
            "market": dict(zip(now_rows["ticker_id"], now_rows["market"]))}
