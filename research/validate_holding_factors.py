"""보유 판단 요소 비교 — 아무 종목 · 아무 날의 "오늘 새로 산다면?" — 검증이력 9.80 사전 등록 그대로.

  python research/validate_holding_factors.py   → results/holding_factors_validation.json
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

from box_rules import BOX
from collect_credit import load_credit
from collect_investor_detail import load_detail
from collect_program import load_program
from universe_all import validation_codes
from v2_config import LONG_HISTORY, Params, results_dir
from v2_datastore import load_bars
from v2_signals import _mass_in_range, kalman_features

START, SPLIT = pd.Timestamp("2016-03-01"), pd.Timestamp("2021-06-01")
H_MAIN, H_AUX, BLOCK, N_BOOT, MIN_SPREAD = 20, 60, 12, 2000, 0.01
FACTORS = {
    "5일선 위치": "ma5", "20일선 위치": "ma20", "60일선 위치": "ma60", "120일선 위치": "ma120",
    "20일선 기울기": "slope20", "60일선 기울기": "slope60", "볼린저 %b": "bbp", "볼린저 폭": "bbw",
    "위쪽 매물 비중": "vp", "프로그램 20일 순매수": "prog20", "신용 잔고 20일 변화": "credit20",
    "RS": "rs", "52주 고가 대비": "hi52", "칼만 관리선 위치": "kz", "30일 박스 위치": "boxpos", "5일 수익률": "r5",
}


def sample_days() -> pd.DatetimeIndex:
    idx = load_bars(LONG_HISTORY, "069500").index
    idx = idx[idx >= START]
    wk = pd.Series(idx, index=idx).groupby(idx.to_period("W")).first()
    return pd.DatetimeIndex(wk.values)


def vp_at(bars: pd.DataFrame, pos: np.ndarray, p: Params) -> np.ndarray:
    """v2_signals.profile_ratio_series 와 같은 계산을 표본 날에만."""
    low_a, high_a, vol_a, close_a = (bars[c].to_numpy(dtype=float) for c in ("low", "high", "volume", "close"))
    out = np.full(len(pos), np.nan)
    for j, i in enumerate(pos):
        if i + 1 < p.profile_min_bars:
            continue
        s = max(0, i + 1 - p.profile_lookback)
        low, high, vol, close = low_a[s:i + 1], high_a[s:i + 1], vol_a[s:i + 1], close_a[i]
        age = np.arange(len(low) - 1, -1, -1, dtype=float)
        w = 0.5 ** (age / p.profile_decay_halflife) if p.profile_decay_halflife > 0 else np.ones(len(low))
        up = _mass_in_range(low, high, vol, w, close, close * (1 + p.profile_band_pct))
        dn = _mass_in_range(low, high, vol, w, close * (1 - p.profile_band_pct), close)
        if up + dn > 0:
            out[j] = up / (up + dn)
    return out


def stock_rows(code: str, days: pd.DatetimeIndex, p: Params) -> pd.DataFrame | None:
    b = load_bars(LONG_HISTORY, code)
    if b is None or len(b) < 300:
        return None
    b = b[(b[["open", "high", "low", "close"]] > 0).all(1)]
    c, h, l = b["close"], b["high"], b["low"]
    f = pd.DataFrame(index=b.index)
    for n in (5, 20, 60, 120):
        f[f"ma{n}"] = c / c.rolling(n).mean() - 1
    m20, m60 = c.rolling(20).mean(), c.rolling(60).mean()
    f["slope20"], f["slope60"] = m20 / m20.shift(5) - 1, m60 / m60.shift(20) - 1
    sd = c.rolling(20).std()
    f["bbp"] = (c - (m20 - 2 * sd)) / (4 * sd)
    f["bbw"] = 4 * sd / m20
    f["rs"] = 0.4 * (c / c.shift(63) - 1) + 0.2 * (c / c.shift(126) - 1) + 0.2 * (c / c.shift(189) - 1) + 0.2 * (c / c.shift(252) - 1)
    f["hi52"] = c / h.rolling(252).max() - 1
    f["kz"] = kalman_features(b, p)["band_z"]
    H, L = h.rolling(BOX).max(), l.rolling(BOX).min()
    f["boxpos"] = (c - L) / (H - L)
    f["r5"] = c / c.shift(5) - 1
    prog = load_program(code)
    if prog is not None and not prog.empty:
        pr = prog.reindex(b.index)
        f["prog20"] = pr["prog_net"].rolling(20, min_periods=15).sum() / pr["value"].rolling(20, min_periods=15).sum()
    cr = load_credit(code)
    if cr is not None and not cr.empty:
        ln = cr["loan_shares"].reindex(b.index).ffill(limit=3)
        f["credit20"] = ln / ln.shift(20) - 1
    f[f"fwd{H_MAIN}"] = c.shift(-H_MAIN) / c - 1
    f[f"fwd{H_AUX}"] = c.shift(-H_AUX) / c - 1
    inv = load_detail(code)
    if inv is not None and not inv.empty and prog is not None:
        iv = inv.reindex(b.index)
        val = prog["value"].reindex(b.index).rolling(20, min_periods=15).sum()
        f["for20"] = iv["외국인"].rolling(20, min_periods=15).sum() * 1e6 / val     # 순매수 대금(백만원) ÷ 거래대금(원)
        f["inst20"] = iv["기관합계"].rolling(20, min_periods=15).sum() * 1e6 / val
    keep = b.index.intersection(days)
    if keep.empty:
        return None
    out = f.loc[keep].copy()
    out["vp"] = vp_at(b, b.index.get_indexer(keep), p)
    out["code"] = code
    return out.replace([np.inf, -np.inf], np.nan)


def spreads(d: pd.DataFrame, col: str, fwd: str) -> pd.DataFrame:
    """날짜별 상위 20% − 하위 20% 평균 초과수익과 순위 상관."""
    rows = []
    for day, g in d[[col, fwd]].dropna().groupby(level=0):
        if len(g) < 100:
            continue
        q = g[col].rank(pct=True)
        rows.append({"date": day, "spread": g.loc[q > 0.8, fwd].mean() - g.loc[q <= 0.2, fwd].mean(),
                     "ic": g[col].rank().corr(g[fwd].rank())})
    return pd.DataFrame(rows).set_index("date")


def boot(x: pd.Series, rng) -> tuple[float, float, float]:
    v = x.to_numpy()
    nb = int(np.ceil(len(v) / BLOCK))
    starts = rng.integers(0, max(1, len(v) - BLOCK + 1), (N_BOOT, nb))
    bs = np.array([np.concatenate([v[s:s + BLOCK] for s in row])[:len(v)].mean() for row in starts])
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    p, days = Params(), sample_days()
    parts = []
    for n, code in enumerate(validation_codes(), 1):
        r = stock_rows(code, days, p)
        if r is not None:
            parts.append(r)
        if n % 100 == 0:
            print(f"{n} 종목", flush=True)
    d = pd.concat(parts)
    d.index.name = "date"
    for h in (H_MAIN, H_AUX):                                           # 같은 날 전 종목 평균을 뺀 초과수익
        d[f"x{h}"] = d[f"fwd{h}"] - d.groupby(level=0)[f"fwd{h}"].transform("mean")
    d.to_csv(results_dir() / "holding_factors_panel.csv")
    res, survivors = {"표본": {"종목": int(d.code.nunique()), "날짜": int(d.index.nunique()), "행": int(len(d))}, "요소": {}}, {}
    for name, col in FACTORS.items():
        r = {}
        for h in (H_MAIN, H_AUX):
            sp = spreads(d, col, f"x{h}")
            fit, test = sp[sp.index < SPLIT], sp[sp.index >= SPLIT]
            bf, bt = boot(fit.spread, rng), boot(test.spread, rng)
            r[f"{h}일"] = {"맞춤 상위−하위": [round(v, 4) for v in bf], "예측 상위−하위": [round(v, 4) for v in bt],
                          "순위 상관 맞춤/예측": [round(float(fit.ic.mean()), 4), round(float(test.ic.mean()), 4)]}
        f, t = r[f"{H_MAIN}일"]["맞춤 상위−하위"], r[f"{H_MAIN}일"]["예측 상위−하위"]
        same = np.sign(f[0]) == np.sign(t[0])
        ci_ok = t[1] > 0 if t[0] > 0 else t[2] < 0
        r["verdict"] = ("판단에 씀" if same and ci_ok and abs(t[0]) >= MIN_SPREAD else "참고" if same else "쓰지 않음")
        r["부호"] = "높을수록 좋음" if t[0] > 0 else "낮을수록 좋음"
        res["요소"][name] = r
        if r["verdict"] == "판단에 씀":
            survivors[col] = 1 if t[0] > 0 else -1
    cols = list(FACTORS.values())
    rk = d[cols].groupby(level=0).rank(pct=True)
    res["요소 간 순위 상관(전체 평균)"] = {a: {b_: round(float(rk[a].corr(rk[b_])), 2) for b_ in cols} for a in cols}
    if survivors:
        score = sum(rk[c] * s for c, s in survivors.items()) / len(survivors)
        d["score"] = score
        sp = spreads(d, "score", f"x{H_MAIN}")
        res["통과 요소 합산 점수"] = {"요소": list(survivors), "맞춤": [round(v, 4) for v in boot(sp[sp.index < SPLIT].spread, rng)],
                               "예측": [round(v, 4) for v in boot(sp[sp.index >= SPLIT].spread, rng)]}
        d["rs_t"] = pd.qcut(rk["rs"], 3, labels=["RS 하", "RS 중", "RS 상"])
        res["RS 3등분 안 상위−하위(20일, 전체 기간)"] = {
            c: {str(t_): round(float(spreads(g, c, f"x{H_MAIN}").spread.mean()), 4) for t_, g in d.groupby("rs_t")} for c in survivors if c != "rs"}
    flows = {}
    for name, col in (("외국인 20일 순매수", "for20"), ("기관 20일 순매수", "inst20")):
        if col in d:
            x = d[d[col].notna()]
            out = {}
            for h in (H_MAIN, H_AUX):
                rows = []
                for day, g in x[[col, f"x{h}"]].dropna().groupby(level=0):
                    if len(g) < 50:
                        continue
                    q = g[col].rank(pct=True)
                    rows.append(g.loc[q > 0.8, f"x{h}"].mean() - g.loc[q <= 0.2, f"x{h}"].mean())
                out[f"{h}일 상위−하위"] = [round(v, 4) for v in boot(pd.Series(rows), rng)] if rows else None
            flows[name] = {"종목": int(x.code.nunique()), "날짜": int(x.index.nunique()), **out}
    res["수급(서술, 2021-09~ 198종목)"] = flows
    (results_dir() / "holding_factors_validation.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "요소 간 순위 상관(전체 평균)"}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
