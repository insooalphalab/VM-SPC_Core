"""Baseline — Ridge Logistic (L2, 표준화 필수, 표준화 계수로 해석) (명세 1.4 / 2.6)."""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "vm_predict", _ROOT / "pair_spc"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RIDGE_C = 1.0


def fit_ridge(X_train: pd.DataFrame, y_train) -> Pipeline:
    model = Pipeline([("scaler", StandardScaler()),
                      ("logit", LogisticRegression(C=RIDGE_C, max_iter=2000))])  # 기본 penalty = L2
    model.fit(X_train, np.asarray(y_train))
    return model


def predict_proba_ridge(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def get_standardized_coefs(model: Pipeline, feature_names: list[str]) -> dict[str, float]:
    """표준화 후 계수 = 피처 1σ 변화당 로그오즈 변화. 입력 순서가 X 열 순서와 같아야 한다."""
    coefs = model.named_steps["logit"].coef_[0]
    cols = list(model.named_steps["scaler"].feature_names_in_)
    by_col = dict(zip(cols, coefs))
    return {f: float(by_col[f]) for f in feature_names if f in by_col}
