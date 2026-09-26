"""Challenger — 얕은 LightGBM (max_depth 2~3, num_leaves 4~7, min_data_in_leaf 20~30) (명세 1.4 / 2.6).

조기종료(2026-09-26 추가): 원래 고정 n_estimators=200으로 매 fold 무조건 200트리를 다 썼는데,
실측 결과 train AUC 0.72~0.76 vs test(OOS) AUC 0.46~0.52 — 반도체·코스피 top10 두 바스켓
10개 fold 전부에서 갭이 +0.21~+0.27로 일관되게 나타났다(Ridge는 train/test 갭이 +0.03~+0.08로
정상 범위). 즉 Baseline을 못 이기는 이유가 "피처에 잡을 신호가 없어서"가 아니라 "이 노이즈
수준·표본 크기에 200트리는 과하게 커서 학습구간 잡음까지 외운다"는 과적합 문제였다 — 트리 수를
30개로만 줄여도 test AUC가 비슷하거나 오히려 더 좋아지는 것으로 확인됐다(갭 +0.08~+0.17로 축소).

그래서 학습구간(X_train) 뒤쪽 VALID_FRAC(20%)을 시간순으로 떼어 조기종료 검증셋으로 쓰고,
그렇게 찾은 최적 트리 수로 전체 학습구간에 다시 적합한다(검증에 쓴 데이터도 최종모델에 반영).
X_train 이 이미 날짜순으로 정렬돼 들어온다는 전제(vm_spc/pipeline.py._design_matrix, 항상
build_feature_frame 의 날짜정렬 출력을 기반으로 함)를 이용한 것이라 look-ahead 는 없다.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "vm_predict", _ROOT / "pair_spc"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

LGBM_PARAMS = dict(max_depth=3, num_leaves=6, min_data_in_leaf=25, objective="binary")
N_ESTIMATORS = 200        # 조기종료 상한 — 실제 사용 트리수는 검증셋 성능이 멈추는 시점까지만
LEARNING_RATE = 0.05
SEED = 42

VALID_FRAC = 0.2          # 학습구간 뒤쪽 20%를 시간순으로 조기종료 검증셋으로 사용
EARLY_STOPPING_ROUNDS = 20
MIN_FIT_ROWS = 50         # 검증셋을 뗄 만큼 학습구간이 크지 않으면(예: 운영 재학습) 조기종료 생략
MIN_VALID_ROWS = 20


def _make_model(n_estimators: int) -> LGBMClassifier:
    return LGBMClassifier(
        max_depth=LGBM_PARAMS["max_depth"], num_leaves=LGBM_PARAMS["num_leaves"],
        min_child_samples=LGBM_PARAMS["min_data_in_leaf"],  # sklearn API 이름 (= min_data_in_leaf)
        objective=LGBM_PARAMS["objective"], n_estimators=n_estimators, learning_rate=LEARNING_RATE,
        random_state=SEED, verbose=-1)


def fit_lgbm(X_train: pd.DataFrame, y_train) -> LGBMClassifier:
    y = np.asarray(y_train)
    n = len(X_train)
    split = int(n * (1 - VALID_FRAC))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if split < MIN_FIT_ROWS or (n - split) < MIN_VALID_ROWS:
            # 표본이 너무 작아 검증셋을 뗄 수 없는 경우(예: 운영 재학습 창이 짧을 때) 조기종료 없이
            # 원래 방식대로 고정 트리수로 적합한다.
            model = _make_model(N_ESTIMATORS)
            model.fit(X_train, y)
            return model

        X_fit, y_fit = X_train.iloc[:split], y[:split]
        X_val, y_val = X_train.iloc[split:], y[split:]
        probe = _make_model(N_ESTIMATORS)
        probe.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], eval_metric="auc",
                  callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
        best_n = max(1, probe.best_iteration_ or N_ESTIMATORS)

        # 조기종료로 찾은 트리수로, 검증에 썼던 데이터도 포함한 전체 학습구간에 다시 적합한다.
        model = _make_model(best_n)
        model.fit(X_train, y)
        return model


def predict_proba_lgbm(model: LGBMClassifier, X: pd.DataFrame) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.predict_proba(X)[:, 1]


def get_gain_importance(model: LGBMClassifier, feature_names: list[str]) -> dict[str, float]:
    """split gain 기준 기여도 비율(합=1, 종목 더미 합계는 '_ticker' 키로 따로)."""
    gains = model.booster_.feature_importance(importance_type="gain")
    by_col = dict(zip(model.booster_.feature_name(), gains))
    total = float(sum(gains)) or 1.0
    out = {f: float(by_col.get(f, 0.0)) / total for f in feature_names}
    out["_ticker"] = float(sum(v for k, v in by_col.items() if k not in feature_names)) / total
    return out
