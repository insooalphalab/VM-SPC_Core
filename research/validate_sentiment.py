"""텔레그램 여론 점수(easobi) 검증 — 검증이력 9.41 사전 등록 기준 그대로.

  python reports/sentiment.py              (먼저: 점수 표)
  python research/validate_sentiment.py    → results/sentiment_validation.json
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "vm_predict", _ROOT / "scenario", _ROOT / "research", _ROOT / "reports"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import json
import sys
import warnings

import numpy as np
import pandas as pd

from v2_config import LONG_HISTORY, data_dir, results_dir
from v2_datastore import load_bars

BLOCK, N_BOOT, MIN_DAYS, CUT = 20, 2000, 120, 50
CLOSE = "15:30"


def closes(code: str) -> pd.Series | None:
    """저장된 일봉 중 가장 최근까지 있는 것(검증 이력 → 페이지용 → 바스켓 순)."""
    best = None
    cands = [load_bars(LONG_HISTORY, code), load_bars("_scenario", code)]
    try:
        import render_risk as rr
        cands.append(rr.load_stock(code)[1])
    except Exception:
        pass
    for b in cands:
        if b is not None and len(b) and (best is None or b.index[-1] > best.index[-1]):
            best = b
    return best["close"] if best is not None else None


def entry_index(idx: pd.DatetimeIndex, when: pd.Timestamp) -> int | None:
    """Report 시각이 장 마감 전이면 그날, 이후면 다음 거래일의 위치."""
    day = when.normalize()
    after = when.strftime("%H:%M") > CLOSE
    i = idx.searchsorted(day + pd.Timedelta(days=1) if after else day)
    return int(i) if i < len(idx) else None


def boot(values_by_block: list[np.ndarray], stat, rng) -> tuple[float, float]:
    k = len(values_by_block)
    out = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, k, k)
        out.append(stat(np.concatenate([values_by_block[i] for i in pick])))
    return tuple(float(x) for x in np.nanpercentile(out, [2.5, 97.5]))


def spearman(a: np.ndarray) -> float:
    x, y = a[:, 0], a[:, 1]
    if len(x) < 10:
        return np.nan
    return float(pd.Series(x).rank().corr(pd.Series(y).rank()))


def h1(mk: pd.DataFrame, kospi: pd.Series, rng) -> dict:
    mk = mk.assign(t=pd.to_datetime(mk["time"]))
    mk = mk[mk["t"].dt.strftime("%H:%M") <= CLOSE]
    daily = mk.groupby(mk["t"].dt.normalize())["score"].last()
    idx = kospi.index
    rows = []
    for d, s in daily.items():
        i = idx.searchsorted(d)
        if i < len(idx) and idx[i] == d:
            rows.append({"date": d, "score": s, **{f"f{h}": kospi.iloc[i + h] / kospi.iloc[i] - 1 if i + h < len(idx) else np.nan
                                                   for h in (5, 20)}})
    d = pd.DataFrame(rows)
    out = {"days": int(len(d)), "from": str(d["date"].min().date()) if len(d) else None, "to": str(d["date"].max().date()) if len(d) else None}
    for h in (5, 20):
        x = d.dropna(subset=[f"f{h}"])
        if len(x) < 20:
            out[f"h{h}"] = {"n": int(len(x))}
            continue
        blocks = [g[["score", f"f{h}"]].to_numpy() for _, g in x.groupby(np.arange(len(x)) // BLOCK)]
        lo, hi = boot(blocks, spearman, rng)
        q1, q2 = x["score"].quantile([1 / 3, 2 / 3])
        out[f"h{h}"] = {"n": int(len(x)), "ic": round(spearman(x[["score", f"f{h}"]].to_numpy()), 3), "ic_ci": [round(lo, 3), round(hi, 3)],
                        "top_minus_bottom": round(float(x.loc[x.score > q2, f"f{h}"].mean() - x.loc[x.score < q1, f"f{h}"].mean()), 4)}
    out["verdict"] = ("판단 보류(표본 부족)" if len(d) < MIN_DAYS else
                      "순방향" if out["h5"].get("ic_ci", [0])[0] > 0 else "역방향" if out["h5"].get("ic_ci", [0, 0])[1] < 0 else "정보 없음")
    return out


def h2(st: pd.DataFrame, kospi: pd.Series, rng) -> dict:
    st = st.dropna(subset=["code"]).assign(t=lambda x: pd.to_datetime(x["time"]))
    cache, rows, missing = {}, [], set()
    for _, r in st[(st.score >= CUT) | (st.score <= -CUT)].iterrows():
        if r.code not in cache:
            cache[r.code] = closes(r.code)
        c = cache[r.code]
        if c is None:
            missing.add(r.code)
            continue
        i = entry_index(c.index, r.t)
        if i is None or i + 5 >= len(c):
            continue
        d0, d5 = c.index[i], c.index[i + 5]
        k0, k5 = kospi.asof(d0), kospi.asof(d5)
        rows.append({"date": d0, "side": "긍정" if r.score >= CUT else "부정", "ex": c.iloc[i + 5] / c.iloc[i] - 1 - (k5 / k0 - 1)})
    d = pd.DataFrame(rows)
    out = {"events_with_price": int(len(d)), "codes_without_price": len(missing)}
    for side, g in d.groupby("side") if len(d) else []:
        g = g.sort_values("date")
        blocks = [x["ex"].to_numpy() for _, x in g.groupby(g["date"].rank(method="dense").astype(int) // BLOCK)]
        lo, hi = boot(blocks, np.mean, rng) if len(blocks) > 1 else (np.nan, np.nan)
        out[side] = {"n": int(len(g)), "days": int(g["date"].nunique()), "ex5": round(float(g["ex"].mean()), 4),
                     "ci": [round(lo, 4), round(hi, 4)],
                     "verdict": "판단 보류(표본 부족)" if g["date"].nunique() < 60 else
                                "순방향" if lo > 0 else "역방향" if hi < 0 else "정보 없음"}
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    base = data_dir() / "_reports"
    mk = pd.read_csv(base / "sentiment_market.csv")
    st = pd.read_csv(base / "sentiment_stock.csv", dtype={"code": str})
    kospi = closes("069500")
    res = {"H1 시장 점수": h1(mk, kospi, rng), "H2 종목 점수": h2(st, kospi, rng)}
    (results_dir() / "sentiment_validation.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
