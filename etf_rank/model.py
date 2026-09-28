"""ETF 단위 피처·라벨·Walk-Forward·순위 상관 검증 (검증이력 9.12)."""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "vm_predict", _ROOT / "pair_spc"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import json

import numpy as np
import pandas as pd

import v2_premium as prem
from v2_config import Params, results_dir
from v2_datastore import load_bars
from v2_signals import kalman_features
from vm_spc.models.ridge_baseline import fit_ridge, predict_proba_ridge

from stock_track.data import load_benchmark
from stock_track.model import HORIZONS, TRAIN_DAYS, horizon_splits
from stock_track.validation import pooled_gate

FEATURES = ["breadth", "kalman_z", "cusum_state", "log_t2", "premium_dz", "rel20"]
DEAD_ZONE = 0.002
MIN_ETFS_PER_DAY = 5
N_BOOT = 1000


def _series(block: dict, key: str) -> pd.Series:
    return pd.Series(block[key], index=pd.to_datetime(block["dates"]), dtype=float)


def etf_frame(basket: dict, bm_close: pd.Series) -> pd.DataFrame | None:
    """ETF 하나의 날짜별 피처 + 기간별 지수 대비 초과수익/라벨."""
    code = basket["target"]["code"]
    bars = load_bars(basket["name"], code)
    spath = results_dir() / basket["name"] / "v2_summary_metrics.json"
    if bars is None or bars.empty or not spath.exists():
        return None
    s = json.loads(spath.read_text(encoding="utf-8"))
    idx = bars.index
    ts = s["target_series"]
    df = pd.DataFrame(index=idx)
    df["close"] = bars["close"]
    df["breadth"] = _series(s["breadth_series"], "breadth").reindex(idx)
    df["kalman_z"] = kalman_features(bars, Params())["band_z"]
    df["cusum_state"] = (_series(ts, "cusum_pos") + _series(ts, "cusum_neg")).reindex(idx)
    df["log_t2"] = np.log1p(_series(s["t2_series"], "t2").reindex(idx))
    ps = prem.premium_series(basket["name"], code)
    df["premium_dz"] = ps["dz"].reindex(idx) if ps is not None else np.nan
    bm = bm_close.reindex(idx)
    df["rel20"] = np.log(df["close"]).diff(20) - np.log(bm).diff(20)
    for k in HORIZONS:
        ex = (df["close"].shift(-k) / df["close"] - 1) - (bm.shift(-k) / bm - 1)
        df[f"ex{k}"] = ex
        df[f"y{k}"] = (ex > 0).astype(float).where(ex.notna() & (ex.abs() >= DEAD_ZONE * np.sqrt(k)))
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)
    df["code"], df["name"], df["basket"] = code, basket["target"]["name"], basket["name"]
    return df.rename_axis("date").reset_index()


def daily_ic(oos: pd.DataFrame, k: int) -> pd.Series:
    """날짜별 ETF 간 Spearman 순위 상관(점수 p vs 실제 k일 초과수익)."""
    g = oos.dropna(subset=[f"ex{k}"]).groupby("date")
    return g.apply(lambda d: d["p"].rank().corr(d[f"ex{k}"].rank()) if len(d) >= MIN_ETFS_PER_DAY else np.nan).dropna()


def block_ci(values: pd.Series, k: int, seed: int = 0) -> tuple[float, float]:
    v = values.sort_index().to_numpy()
    blocks = [v[i:i + k].mean() for i in range(0, len(v), k)]
    b = np.array(blocks)
    rng = np.random.default_rng(seed)
    boot = b[rng.integers(0, len(b), size=(N_BOOT, len(b)))].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def quintile_spread(oos: pd.DataFrame, k: int) -> float | None:
    """날짜별 점수 상위 20% − 하위 20% 평균 k일 초과수익의 평균."""
    def one(d):
        if len(d) < MIN_ETFS_PER_DAY:
            return np.nan
        n = max(1, len(d) // 5)
        s = d.sort_values("p")
        return s[f"ex{k}"].tail(n).mean() - s[f"ex{k}"].head(n).mean()
    v = oos.dropna(subset=[f"ex{k}"]).groupby("date").apply(one).dropna()
    return float(v.mean()) if len(v) else None


def evaluate(panel: pd.DataFrame, k: int) -> tuple[dict, pd.DataFrame | None]:
    """패널 전체(여러 ETF)로 기간 k Walk-Forward → OOS 점수 → 순위 상관 판정."""
    lab = panel.dropna(subset=[f"y{k}"]).sort_values(["date", "code"]).reset_index(drop=True)
    parts = []
    for tr, te in horizon_splits(lab["date"], k):
        if lab.loc[tr, f"y{k}"].nunique() < 2:
            continue
        m = fit_ridge(lab.loc[tr, FEATURES], lab.loc[tr, f"y{k}"])
        part = lab.loc[te, ["date", "code", f"y{k}", f"ex{k}"]].copy()
        part["p"] = predict_proba_ridge(m, lab.loc[te, FEATURES])
        parts.append(part)
    if not parts:
        return {"k": k, "status": "HOLD", "reason": "데이터 부족"}, None
    oos = pd.concat(parts, ignore_index=True)
    ic = daily_ic(oos, k)
    if ic.empty:
        return {"k": k, "status": "HOLD", "reason": "데이터 부족"}, oos
    lo, hi = block_ci(ic, k)
    gate = pooled_gate(oos.rename(columns={f"y{k}": "y", "code": "ticker_id"}).assign(basket=oos["code"]), k)
    out = {"k": k, "ic_mean": round(float(ic.mean()), 4), "ic_ci": [round(lo, 4), round(hi, 4)],
           "n_days": int(len(ic)), "n_indep": int(len(ic) // k), "n_etfs": int(oos["code"].nunique()),
           "quintile_spread": None if (q := quintile_spread(oos, k)) is None else round(q, 5),
           "hit": gate.get("hcp"), "hit_base": gate.get("base"), "hit_ci": gate.get("ci")}
    out["status"], out["reason"] = ("Active", "통과") if lo > 0 else ("HOLD", "유의성 미달")
    return out, oos


def today_scores(train_panel: pd.DataFrame, score_panel: pd.DataFrame, k: int) -> dict[str, float]:
    """검증한 패널(테마 대표)의 라벨 확정된 최근 250거래일로 학습해, 최신일 각 ETF 점수(지수를 이길 확률)."""
    lab = train_panel.dropna(subset=[f"y{k}"])
    dates = np.sort(lab["date"].unique())
    latest = score_panel["date"].max()
    now = score_panel[score_panel["date"] == latest]
    if len(dates) < TRAIN_DAYS or now.empty:
        return {}
    recent = lab[lab["date"] >= dates[-TRAIN_DAYS]]
    if recent[f"y{k}"].nunique() < 2:
        return {}
    m = fit_ridge(recent[FEATURES], recent[f"y{k}"])
    return dict(zip(now["code"], predict_proba_ridge(m, now[FEATURES])))
