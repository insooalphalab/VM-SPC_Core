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


def save_bars(basket_name: str, code: str, rows: list[dict]) -> None:
    path = bars_path(basket_name, code)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["date", *BAR_COLS]).to_csv(path, index=False, encoding="utf-8-sig")


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
