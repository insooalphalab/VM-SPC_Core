"""평가지표 + 날짜 클러스터 부트스트랩 CI (명세 1.6 / 1.8 / 2.7).

공통 평가모집합: 데드존 밖인 날만(레이블링 단계에서 이미 걸러짐).
  S_base(Tier 1): P > 0.50       S_core(Tier 2): P >= 0.60
Legacy 는 0/1 이진 판정이라 S_base = S_core 가 된다.
채택 판정은 High-Conf Precision 하나만 쓰고, 나머지는 참고 지표다(다중비교 방지).
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

TIER1_THRESHOLD = 0.50
TIER2_THRESHOLD = 0.60
N_BOOT = 1000
CI_LEVEL = 0.95
BOOT_SEED = 42


def _prec(y: np.ndarray, mask: np.ndarray) -> float | None:
    n = int(mask.sum())
    return float(y[mask].mean()) if n else None


def date_clustered_bootstrap_ci(y_true, y_pred_proba, dates, n_boot: int = N_BOOT,
                                metric: str = "high_conf_precision", level: float = CI_LEVEL,
                                seed: int = BOOT_SEED) -> tuple[float | None, float | None]:
    """날짜 단위 리샘플링(같은 날 관측치는 통째로 뽑거나 뺌) → 횡단면 상관 보정된 CI."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred_proba, dtype=float)
    thr_mask = p >= TIER2_THRESHOLD if metric == "high_conf_precision" else p > TIER1_THRESHOLD
    g = pd.DataFrame({"d": pd.to_datetime(np.asarray(dates)), "hit": y * thr_mask, "n": thr_mask.astype(float)})
    agg = g.groupby("d")[["hit", "n"]].sum()
    hits, ns = agg["hit"].to_numpy(), agg["n"].to_numpy()
    if ns.sum() == 0:
        return None, None
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(agg), size=(n_boot, len(agg)))
    h, n = hits[draws].sum(axis=1), ns[draws].sum(axis=1)
    stats = h[n > 0] / n[n > 0]
    if len(stats) == 0:
        return None, None
    a = (1 - level) / 2
    return float(np.quantile(stats, a)), float(np.quantile(stats, 1 - a))


def signal_frequency_per_year(n_signals: int, n_eval_days: int) -> float | None:
    return n_signals / n_eval_days * 250 if n_eval_days else None


def compute_metrics(y_true, y_pred_proba, dates=None, with_ci: bool = True) -> dict:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred_proba, dtype=float)
    s_base, s_core = p > TIER1_THRESHOLD, p >= TIER2_THRESHOLD
    precision = _prec(y, s_base)
    n_pos = int(y.sum())
    recall = float((y * s_base).sum() / n_pos) if n_pos else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall and (precision + recall) > 0 else None)
    n_days = int(pd.Series(pd.to_datetime(np.asarray(dates))).nunique()) if dates is not None else 0
    out = {
        "n_eval": int(len(y)),
        "n_eval_days": n_days,
        "base_rate": float(y.mean()) if len(y) else None,
        "n_s_base": int(s_base.sum()),
        "n_s_core": int(s_core.sum()),
        "precision": precision,
        "high_conf_precision": _prec(y, s_core),
        "brier": float(np.mean((p - y) ** 2)) if len(y) else None,
        "recall": recall,
        "f1": f1,
        "signals_per_year": signal_frequency_per_year(int(s_core.sum()), n_days),
        "ci_lower": None, "ci_upper": None,
    }
    if with_ci and dates is not None:
        out["ci_lower"], out["ci_upper"] = date_clustered_bootstrap_ci(y, p, dates)
    return out


def aggregate_fold_metrics(oos: pd.DataFrame, models: list[str]) -> dict[str, dict]:
    """fold 별 지표를 평균내지 않고 전체 OOS 예측을 풀링한 뒤 한 번에 지표·CI 를 계산한다
    (fold 마다 S_core 표본이 몇 개 안 돼 평균이 불안정해지는 것을 피함)."""
    return {m: compute_metrics(oos["y"], oos[f"p_{m}"], oos["date"]) for m in models}
