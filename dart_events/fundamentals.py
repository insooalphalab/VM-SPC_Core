"""DART 주요계정 → 종목별 분기 값(접수일 포함). A 검증과 C 밸류에이션이 같이 쓴다."""
from __future__ import annotations

import numpy as np
import pandas as pd

Q_END = {1: "0331", 2: "0630", 3: "0930", 4: "1231"}
FLOW = {"rev", "op", "ni"}        # 손익 항목은 4분기 = 연간 − 3분기 누적, 자본은 시점 값 그대로


def quarterly(fin: pd.DataFrame, account: str) -> pd.DataFrame:
    """stock_code, year, q, value, rcept_dt, lag(분기 말 → 접수일 일수). 연결(CFS)이 더 많으면 연결만, 아니면 별도만."""
    fin = fin[fin["account"] == account]
    rows = []
    for code, g in fin.groupby("stock_code"):
        fs = "CFS" if (g["fs_div"] == "CFS").sum() >= (g["fs_div"] == "OFS").sum() else "OFS"
        g = g[g["fs_div"] == fs].set_index(["year", "q"])
        for (y, q), x in g.iterrows():
            v = x["amount"]
            if q == 4 and account in FLOW:
                cum = g["add_amount"].get((y, 3))
                if cum is None or pd.isna(cum):
                    parts = [g["amount"].get((y, k)) for k in (1, 2, 3)]
                    cum = sum(parts) if all(p is not None and pd.notna(p) for p in parts) else np.nan
                v = v - cum if pd.notna(v) else np.nan
            rows.append({"stock_code": code, "year": y, "q": q, "value": v, "rcept_dt": x["rcept_dt"], "fs": fs})
    df = pd.DataFrame(rows).sort_values(["stock_code", "year", "q"])
    qend = pd.to_datetime(df["year"].astype(str) + df["q"].map(Q_END), format="%Y%m%d")
    df["lag"] = (pd.to_datetime(df["rcept_dt"], format="%Y%m%d") - qend).dt.days
    return df.reset_index(drop=True)
