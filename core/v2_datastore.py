"""data/{basket}/{code}.csv 일봉 저장소. Stage1(v2_data_collector)이 쓰고 Stage2(v2_compute_engine)가 읽는다."""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "vm_predict", _ROOT / "pair_spc"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

from pathlib import Path

import pandas as pd

from v2_config import data_dir

BAR_COLS = ["open", "high", "low", "close", "volume"]


def bars_path(basket_name: str, code: str) -> Path:
    return data_dir() / basket_name / f"{code}.csv"


def save_bars(basket_name: str, code: str, rows: list[dict], merge: bool = False) -> None:
    """rows 를 저장한다. merge=True 면 기존 CSV 와 합쳐서 겹치는 날짜는 새 값으로 덮어쓴다
    (증분 수집 시 이미 저장된 과거분을 보존하면서 KIS 수정주가 소급 정정도 반영하기 위함)."""
    path = bars_path(basket_name, code)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(rows, columns=["date", *BAR_COLS])
    if merge and path.exists():
        old_df = pd.read_csv(path, dtype={"date": str})
        new_df = pd.concat([old_df, new_df], ignore_index=True)
        new_df = new_df.drop_duplicates("date", keep="last").sort_values("date")
    new_df.to_csv(path, index=False, encoding="utf-8-sig")


def load_bars(basket_name: str, code: str) -> pd.DataFrame | None:
    path = bars_path(basket_name, code)
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype={"date": str})
    if df.empty:
        return None
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.drop_duplicates("date", keep="last").set_index("date").sort_index()[BAR_COLS].astype(float)
    return df[(df["close"] > 0) & (df["high"] >= df["low"])]
