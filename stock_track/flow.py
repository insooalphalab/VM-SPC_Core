"""누적 수급 피처(검증이력 9.8 사전 등록 그대로).

종목별 투자자 매매동향은 data/_investor_detail/{코드}.csv(stock_track/collect_investor_detail.py, 기관 세부 포함)를 쓴다.
(2026-09-29 이전 data/_investor/ 4개 주체 저장소와 v2_investor_flow.py 는 이 저장소로 합치며 폐기.)
  frgn_flow5/20 : 외국인 5·20일 누적 순매수 ÷ 같은 기간 거래대금
  inst_flow5/20 : 기타기관(기관합계 − 금융투자) 동일
하루 늦춰(t-1 까지 누적) 쓴다 — 16:30 실행 시점엔 당일 수급이 확정되지 않았을 수 있어서.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "vm_predict", _ROOT / "pair_spc", _ROOT / "stock_track"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import pandas as pd

from collect_investor_detail import load_detail

RAW_COLS = ["개인", "외국인", "기관합계", "금융투자"]
FLOW_FEATURES = ["frgn_flow5", "frgn_flow20", "inst_flow5", "inst_flow20"]


def load_flow(code: str) -> pd.DataFrame | None:
    d = load_detail(code)
    return None if d is None or d.empty else d[RAW_COLS].astype(float)


def flow_features(code: str, bars: pd.DataFrame) -> pd.DataFrame | None:
    flow = load_flow(code)
    if flow is None:
        return None
    flow = flow.reindex(bars.index)
    value = bars["close"] * bars["volume"] / 1e6       # 거래대금(백만원) — 수급 단위와 맞춤
    other_inst = flow["기관합계"] - flow["금융투자"]
    out = pd.DataFrame(index=bars.index)
    for n in (5, 20):
        v = value.rolling(n).sum()
        out[f"frgn_flow{n}"] = flow["외국인"].rolling(n).sum() / v
        out[f"inst_flow{n}"] = other_inst.rolling(n).sum() / v
    return out.shift(1)[FLOW_FEATURES]
