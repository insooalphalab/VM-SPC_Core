"""논문 룰(변동성 정규화 다중 MACD + 섹터 피어 필터 + 데드존) — SK하이닉스 · 반도체TOP10 바스켓 — 검증이력 9.96 사전 등록 그대로.

  python research/validate_macd_sector_rule.py   → results/macd_sector_rule.json
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "vm_predict", _ROOT / "scenario", _ROOT / "research", _ROOT / "stock_track"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import json
import sys
import warnings

import numpy as np
import pandas as pd

from v2_config import LONG_HISTORY, load_baskets, results_dir
from v2_datastore import load_bars

START, SPLIT = pd.Timestamp("2016-01-01"), pd.Timestamp("2021-06-01")
COST = 0.003
PERIODS = {"앞(2016~2021-05)": (START, SPLIT), "최근(2021-06~)": (SPLIT, pd.Timestamp("2100-01-01"))}


def score(close: pd.Series, kind: str = "tanh") -> pd.Series:
    sd = close.rolling(63).std()
    ys = [(close.ewm(span=s, adjust=False).mean() - close.ewm(span=3 * s, adjust=False).mean()) / sd for s in (8, 16, 32)]
    if kind == "tanh":
        return np.tanh(sum(ys) / 3)
    return sum(y * np.exp(-y ** 2 / 4) / 0.89 for y in ys) / 3


def run(b: pd.DataFrame, sa: pd.Series, ss: pd.Series | None, hard: bool = True) -> pd.DataFrame:
    """일간 전략 수익률(비용 반영) · 보유 여부 · 거래 목록. 신호는 종가 → 다음날 시가 체결, −5%는 장중(갭이면 시가)."""
    o, l, c = (b[k].to_numpy(float) for k in ("open", "low", "close"))
    A = sa.to_numpy()
    S = ss.to_numpy() if ss is not None else np.ones(len(c))
    n = len(c)
    ret, held = np.zeros(n), np.zeros(n)
    pos, entry, ed, pend, trades = False, 0.0, None, None, []

    def close_trade(px):
        trades.append({"entry_date": ed, "ret": px / entry - 1 - COST})

    for j in range(1, n):
        if pend == "buy":
            pos, entry, ed = True, o[j], b.index[j]
            held[j] = 1
            stop = entry * 0.95
            if hard and l[j] <= stop:
                ret[j] = stop / o[j] - 1 - COST
                close_trade(stop)
                pos = False
            else:
                ret[j] = c[j] / o[j] - 1 - COST / 2
        elif pos:
            held[j] = 1
            stop = entry * 0.95
            if pend == "sell":
                ret[j] = o[j] / c[j - 1] - 1 - COST / 2
                close_trade(o[j])
                pos = False
            elif hard and (o[j] <= stop or l[j] <= stop):
                px = o[j] if o[j] <= stop else stop
                ret[j] = px / c[j - 1] - 1 - COST / 2
                close_trade(px)
                pos = False
            else:
                ret[j] = c[j] / c[j - 1] - 1
        pend = None
        if np.isnan(A[j]) or np.isnan(S[j]):
            continue
        if not pos and A[j] >= 0.2 and S[j] > 0:
            pend = "buy"
        elif pos and (A[j] < -0.1 or S[j] < -0.3):
            pend = "sell"
    out = pd.DataFrame({"ret": ret, "held": held}, index=b.index)
    out.attrs["trades"] = pd.DataFrame(trades, columns=["entry_date", "ret"])
    return out


def stats(r: pd.Series, trades: pd.DataFrame | None, lo, hi, held: pd.Series | None = None) -> dict:
    x = r[(r.index >= lo) & (r.index < hi)]
    if len(x) < 60:
        return {}
    eq = (1 + x).cumprod()
    yrs = len(x) / 252
    out = {"연 수익률": round(float(eq.iloc[-1] ** (1 / yrs) - 1), 3), "샤프": round(float(x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else 0.0, 2),
           "최대 낙폭": round(float((eq / eq.cummax() - 1).min()), 3), "보유 비중": round(float(held[(held.index >= lo) & (held.index < hi)].mean()), 2) if held is not None else 1.0}
    if trades is not None and len(trades):
        t = trades[(trades.entry_date >= lo) & (trades.entry_date < hi)]
        out.update({"거래 수": len(t), "승률": round(float((t.ret > 0).mean()), 2) if len(t) else None,
                    "거래당 평균": round(float(t.ret.mean()), 3) if len(t) else None})
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    targets = [a for a in sys.argv[1:] if not a.startswith("--")] or ["396500"]
    out_all, pooled = {}, {p: {"sh": [], "dd": [], "sec": []} for p in PERIODS}
    for tcode in targets:
        res = one_basket(tcode)
        out_all[res["이름"]] = res
        for p in PERIODS:
            for k in ("sh", "dd", "sec"):
                pooled[p][k] += res["_raw"][p][k]
        res.pop("_raw")
    if targets == ["396500"]:
        res = out_all[next(iter(out_all))]
        hx = res["종목"]["000660 SK하이닉스"]
        hx_ok = all(hx[p]["룰"]["샤프"] >= hx[p]["계속 보유"]["샤프"] and hx[p]["룰"]["최대 낙폭"] > hx[p]["계속 보유"]["최대 낙폭"] for p in PERIODS)
        bk_ok = all(np.median(pooled[p]["sh"]) > 0 for p in PERIODS)
        res["판정"] = {"하이닉스": hx_ok, "바스켓 일반성": bk_ok, "쓸 만함": hx_ok and bk_ok,
                     "섹터 필터 핵심": all(np.mean(pooled[p]["sec"]) >= 0.7 for p in PERIODS)}
        fn = "macd_sector_rule.json"
    else:
        sh_ok = all(np.median(pooled[p]["sh"]) > 0 for p in PERIODS)
        dd_ok = all(np.median(pooled[p]["dd"]) > 0 for p in PERIODS)
        out_all["합계"] = {p: {"종목 수": len(pooled[p]["sh"]), "샤프 차이 중앙값": round(float(np.median(pooled[p]["sh"])), 2),
                             "샤프 나은 종목": int(sum(x > 0 for x in pooled[p]["sh"])), "낙폭 개선 중앙값": round(float(np.median(pooled[p]["dd"])), 3),
                             "섹터 필터 나은 비율": round(float(np.mean(pooled[p]["sec"])), 2)} for p in PERIODS}
        out_all["판정"] = ("쓸 만함" if sh_ok and dd_ok else "위험 조정 수익만" if sh_ok else "아님") +             (" · 섹터 필터 핵심" if all(np.mean(pooled[p]["sec"]) >= 0.7 for p in PERIODS) else " · 섹터 필터 핵심 아님")
        fn = "macd_sector_rule_" + "_".join(targets) + ".json"
    (results_dir() / fn).write_text(json.dumps(out_all, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out_all, ensure_ascii=False, indent=1) if targets != ["396500"] else
          json.dumps(out_all[next(iter(out_all))]["판정"], ensure_ascii=False))
    return 0


def one_basket(tcode: str) -> dict:
    basket = next(x for x in load_baskets() if x["target"]["code"] == tcode)
    codes = [m["code"] for m in basket["sensors"]]
    names = {m["code"]: m["name"] for m in basket["sensors"]}
    bars = {}
    for cd in codes:
        b = load_bars(LONG_HISTORY, cd)
        if b is not None and len(b):
            bars[cd] = b[(b[["open", "high", "low", "close"]] > 0).all(1)]
    rets = pd.DataFrame({cd: b["close"].pct_change() for cd, b in bars.items()})
    res = {"이름": basket["target"]["name"], "바스켓": {cd: names[cd] for cd in bars}, "종목": {}}
    raw = {p: {"sh": [], "dd": [], "sec": []} for p in PERIODS}
    for cd, b in bars.items():
        peer = rets.drop(columns=cd).reindex(b.index).mean(axis=1, skipna=True).fillna(0)
        S_close = (1 + peer).cumprod()
        sa, ss = score(b["close"]), score(S_close)
        variants = {"룰": run(b, sa, ss), "섹터 필터 뺌": run(b, sa, None), "−5% 손절 뺌": run(b, sa, ss, hard=False),
                    "논문 반응 함수": run(b, score(b["close"], "phi"), score(S_close, "phi"))}
        bh = b["close"].pct_change().fillna(0)
        one = {}
        for pn, (lo, hi) in PERIODS.items():
            row = {"계속 보유": stats(bh, None, lo, hi)}
            for vn, v in variants.items():
                row[vn] = stats(v.ret, v.attrs["trades"], lo, hi, v.held)
            one[pn] = row
            if row["룰"] and row["계속 보유"]:
                raw[pn]["sh"].append(row["룰"]["샤프"] - row["계속 보유"]["샤프"])
                raw[pn]["dd"].append(row["룰"]["최대 낙폭"] - row["계속 보유"]["최대 낙폭"])
                raw[pn]["sec"].append(float(row["룰"]["샤프"] > row["섹터 필터 뺌"]["샤프"]))
        res["종목"][f"{cd} {names[cd]}"] = one
    res["바스켓 요약"] = {p: {"종목 수": len(raw[p]["sh"]), "샤프 차이 중앙값": round(float(np.median(raw[p]["sh"])), 2) if raw[p]["sh"] else None,
                          "샤프 나은 종목": int(sum(x > 0 for x in raw[p]["sh"])),
                          "낙폭 개선 중앙값": round(float(np.median(raw[p]["dd"])), 3) if raw[p]["dd"] else None,
                          "섹터 필터 나은 종목": int(sum(raw[p]["sec"]))} for p in PERIODS}
    res["_raw"] = raw
    return res


if __name__ == "__main__":
    sys.exit(main())
