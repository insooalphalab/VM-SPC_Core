"""Walk-Forward + Purge/Embargo 분할 (명세 1.4 / 2.5).

  [train 250일] --embargo 10일-- [test 60일] → 60일(약 3개월)씩 슬라이딩

날짜(거래일) 단위로 자른다 — 같은 날 여러 종목 행은 항상 같은 쪽에 들어간다.
- Purge: train 마지막 날의 레이블은 t+1 수익률을 참조한다. embargo(10일) 가 그 t+1 을 이미
  포함하므로 train 레이블이 test 구간에 걸치는 일은 없다(추가로 PURGE_DAYS 만큼 train 끝을 깎는다).
- Embargo: CUSUM·T²·공적분 Z 같은 롤링 피처가 train 마지막 구간의 기억을 test 초반으로 끌고 가는
  것을 막기 위해 train 과 test 사이 10거래일을 어느 쪽에도 쓰지 않는다.
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

TRAIN_DAYS = 250
TEST_DAYS = 60
EMBARGO_DAYS = 10
PURGE_DAYS = 1   # 레이블 horizon(t+1)


def walk_forward_splits(dates: pd.Series, train_days: int = TRAIN_DAYS, test_days: int = TEST_DAYS,
                        embargo_days: int = EMBARGO_DAYS, step_days: int = TEST_DAYS
                        ) -> list[tuple[np.ndarray, np.ndarray, dict]]:
    """dates(행별 날짜) → [(train 행 인덱스, test 행 인덱스, fold 메타)]. 마지막 test 구간이
    test_days 보다 짧으면 버리지 않고 남은 날짜만큼 쓴다(최근 구간 검증을 빠뜨리지 않기 위해),
    단 20거래일 미만이면 통계가 너무 불안정해 생략한다."""
    dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    uniq = np.array(sorted(dates.unique()))
    d_pos = pd.Series(np.arange(len(uniq)), index=uniq)
    row_pos = d_pos.reindex(dates.to_numpy()).to_numpy()

    splits = []
    start = 0
    while True:
        tr_lo, tr_hi = start, start + train_days - PURGE_DAYS      # [tr_lo, tr_hi)
        te_lo = start + train_days + embargo_days
        te_hi = min(te_lo + test_days, len(uniq))
        if te_hi - te_lo < 20:
            break
        tr_idx = np.where((row_pos >= tr_lo) & (row_pos < tr_hi))[0]
        te_idx = np.where((row_pos >= te_lo) & (row_pos < te_hi))[0]
        splits.append((tr_idx, te_idx, {
            "train_start": str(pd.Timestamp(uniq[tr_lo]).date()), "train_end": str(pd.Timestamp(uniq[tr_hi - 1]).date()),
            "test_start": str(pd.Timestamp(uniq[te_lo]).date()), "test_end": str(pd.Timestamp(uniq[te_hi - 1]).date()),
        }))
        if te_hi >= len(uniq):
            break
        start += step_days
    return splits
