"""기간 k 별 전 바스켓 합산 게이트 (설계 9.6, 구현 전에 고정한 기준 그대로).

  고신뢰 신호     : P ≥ 0.60(초과 예상) 또는 P ≤ 0.40(미달 예상)
  가중치          : 같은 종목·같은 날을 여러 바스켓이 예측하면 각 예측에 1/바스켓 수 (종목·날짜당 합계 1)
  고신뢰 적중률   : Σw·적중 / Σw (신호가 난 행만)
  평소 비율       : 상승 신호 비중 × 초과수익 양성 비율 + 하락 신호 비중 × 음성 비율 (양성 비율은 같은 기간 전체 행)
  독립 표본       : floor(신호가 난 고유 날짜 수 / k) ≥ 30
  블록 부트스트랩 : 연속 k일 날짜 블록을 뽑고 그 날짜의 모든 바스켓·종목을 함께 (B=1000)
  Active          : 독립 표본 ≥ 30 and 고신뢰 적중률 95% CI 하한 > 평소 비율. 아니면 HOLD.
Active/HOLD 는 기간 단위로만 정하고, 바스켓·종목별 수치는 참고로만 쓴다(다중비교 재발 방지).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HIGH, LOW = 0.60, 0.40
MIN_INDEP = 30
N_BOOT = 1000


def pooled_gate(oos: pd.DataFrame, k: int, seed: int = 0) -> dict:
    df = oos.copy()
    df["w"] = 1.0 / df.groupby(["ticker_id", "date"])["basket"].transform("count")
    pos_rate = float((df["w"] * df["y"]).sum() / df["w"].sum())

    sig = df[(df["p"] >= HIGH) | (df["p"] <= LOW)].copy()
    out = {"k": k, "n_rows": int(len(df)), "n_dates": int(df["date"].nunique()), "pos_rate": round(pos_rate, 4)}
    if sig.empty:
        return {**out, "status": "HOLD", "reason": "고신뢰 신호 없음", "hcp": None, "base": None,
                "ci": [None, None], "n_indep": 0}
    sig["up"] = sig["p"] >= HIGH
    sig["hit"] = np.where(sig["up"], sig["y"] == 1, sig["y"] == 0).astype(float)
    up_share = float(sig.loc[sig["up"], "w"].sum() / sig["w"].sum())
    base = up_share * pos_rate + (1 - up_share) * (1 - pos_rate)
    hcp = float((sig["w"] * sig["hit"]).sum() / sig["w"].sum())
    n_indep = int(sig["date"].nunique() // k)

    # 날짜별 합 → 연속 k일 블록별 합 → 블록 복원추출
    per_date = sig.assign(wh=sig["w"] * sig["hit"]).groupby("date")[["wh", "w"]].sum().sort_index()
    block_id = np.arange(len(per_date)) // k
    blocks = per_date.groupby(block_id).sum().to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(blocks), size=(N_BOOT, len(blocks)))
    boot = blocks[idx, 0].sum(1) / blocks[idx, 1].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    if n_indep < MIN_INDEP:
        status, reason = "HOLD", "표본 부족"
    elif lo > base:
        status, reason = "Active", "통과"
    else:
        status, reason = "HOLD", "유의성 미달"
    return {**out, "status": status, "reason": reason, "hcp": round(hcp, 4), "base": round(base, 4),
            "up_share": round(up_share, 4), "ci": [round(float(lo), 4), round(float(hi), 4)],
            "n_signal_rows": int(len(sig)), "n_indep": n_indep}
