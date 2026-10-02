"""장세를 비지도 군집(GMM)으로 — 60일선 3분할보다 전략 고르기에 나은가 — 검증이력 9.84 사전 등록 그대로.

  python research/validate_regime_gmm.py   → results/regime_gmm_validation.json, results/regime_gmm_daily.csv
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
from sklearn.mixture import GaussianMixture

import validate_box as vb
import validate_rebreakout as vr
from collect_credit import load_credit
from collect_market_investor import load as load_investor
from collect_market_program import load as load_mprog
from universe_all import validation_codes
from validate_compression_regime import regime
from validate_holding_factors import spreads
from v2_config import LONG_HISTORY, results_dir
from v2_datastore import load_bars

FIT0, SPLIT = pd.Timestamp("2016-03-01"), pd.Timestamp("2021-06-01")
K, SMOOTH, MIN_N, COST = 4, 5, 100, vb.COST
FEATS = {"코스피 20일 수익률": "k_ret20", "코스피 20일 변동성": "k_vol20", "종목 변동성 중앙값": "x_vol20", "50일선 위 비율": "above50",
         "거래대금 상위10% 집중도": "conc", "신용 잔고 20일 변화": "credit20", "비차익 프로그램 z": "nabt_z", "외국인 z": "for_z", "연기금 z": "pen_z"}


def flow_z(s: pd.Series) -> pd.Series:
    m = s.rolling(20, min_periods=15).sum()
    return (m - m.rolling(250, min_periods=200).mean()) / m.rolling(250, min_periods=200).std()


def market_features() -> pd.DataFrame:
    inv, mp = load_investor(), load_mprog()
    k = inv["kospi_close"]
    f = pd.DataFrame(index=k.index)
    lr = np.log(k).diff()
    f["k_ret20"], f["k_vol20"] = k / k.shift(20) - 1, lr.rolling(20).std()
    close, value, loan = {}, {}, {}
    for code in validation_codes():
        b = load_bars(LONG_HISTORY, code)
        if b is None or len(b) < 300:
            continue
        b = b[(b[["open", "high", "low", "close"]] > 0).all(1)]
        close[code], value[code] = b["close"], b["close"] * b["volume"]
        cr = load_credit(code)
        if cr is not None and not cr.empty:
            loan[code] = cr["loan_amt"]
    C = pd.DataFrame(close).reindex(f.index)
    V = pd.DataFrame(value).reindex(f.index).rolling(20, min_periods=15).sum()
    f["x_vol20"] = np.log(C).diff().rolling(20).std().median(axis=1)
    m50 = C.rolling(50).mean()
    f["above50"] = (C > m50).where(m50.notna()).mean(axis=1)
    arr = np.sort(np.nan_to_num(V.to_numpy(), nan=0.0), axis=1)[:, ::-1]
    n = V.notna().sum(axis=1).to_numpy()
    top = np.array([arr[i, :max(1, int(n[i] * 0.1))].sum() for i in range(len(arr))])
    f["conc"] = np.where(n > 100, top / np.maximum(arr.sum(axis=1), 1), np.nan)
    L = pd.DataFrame(loan).reindex(f.index).ffill(limit=5)
    both = L.notna() & L.shift(20).notna()
    f["credit20"] = L.where(both).sum(axis=1) / L.shift(20).where(both).sum(axis=1).replace(0, np.nan) - 1
    f["nabt_z"] = flow_z(mp["nabt_net"].reindex(f.index))
    f["for_z"], f["pen_z"] = flow_z(inv["foreign_amt"]), flow_z(inv["pension_amt"])
    return f


def mode5(s: pd.Series) -> pd.Series:
    v = s.to_numpy()
    out = np.full(len(v), -1)
    for i in range(len(v)):
        w = v[max(0, i - SMOOTH + 1):i + 1]
        w = w[w >= 0]
        if len(w):
            out[i] = np.bincount(w).argmax()
    return pd.Series(out, index=s.index)


def events() -> dict[str, pd.DataFrame]:
    bo = pd.read_csv(results_dir() / "trend_principles_events.csv", parse_dates=["entry_date"])
    bo = bo[bo.rs >= 0.70][["entry_date", "R", "win"]]
    fe = pd.read_csv(results_dir() / "winrate_events.csv", parse_dates=["entry_date"])
    fe = fe[fe.depth >= 0.035].assign(R=lambda x: (x.ret - COST) / x.stop_pct.clip(lower=0.01), win=lambda x: (x.ret - COST > 0).astype(float))
    return {"RS≥70 돌파": bo, "가짜 이탈(깊이 3.5%↑)": fe[["entry_date", "R", "win"]]}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    f = market_features()
    cols = list(FEATS.values())
    fit_mask = (f.index >= FIT0) & (f.index < SPLIT) & f[cols].notna().all(axis=1)
    mu, sd = f.loc[fit_mask, cols].mean(), f.loc[fit_mask, cols].std()
    Z = (f[cols] - mu) / sd
    gmm = GaussianMixture(n_components=K, covariance_type="full", n_init=5, random_state=0).fit(Z[fit_mask].to_numpy())
    ok = Z.notna().all(axis=1)
    raw = pd.Series(-1, index=f.index)
    raw[ok] = gmm.predict(Z[ok].to_numpy())
    st = mode5(raw).where(lambda s: s >= 0)
    ma = regime().reindex(f.index)
    daily = pd.DataFrame({"gmm": st, "ma60": ma}).join(f)
    k = load_investor()["kospi_close"]
    daily["k_fwd20"] = k.shift(-20) / k - 1
    daily.to_csv(results_dir() / "regime_gmm_daily.csv")

    fitd = daily[(daily.index >= FIT0) & (daily.index < SPLIT)]
    res = {"상태 설명(맞춤 기간 입력 평균)": {}, "전략": {}}
    for s_, g in fitd.groupby("gmm"):
        res["상태 설명(맞춤 기간 입력 평균)"][f"상태 {int(s_)}"] = {
            "날 비율(맞춤/예측)": [round(len(g) / len(fitd), 3), round(float((daily.loc[daily.index >= SPLIT, "gmm"] == s_).mean()), 3)],
            **{n: round(float(g[c].mean()), 4) for n, c in FEATS.items()},
            "60일선 장세 구성": g.ma60.value_counts(normalize=True).round(2).to_dict(),
            "이후 20일 코스피(맞춤/예측)": [round(float(g.k_fwd20.mean()), 4),
                                    round(float(daily.loc[(daily.index >= SPLIT) & (daily.gmm == s_), "k_fwd20"].mean()), 4)]}
    cal = daily.index
    for name, ev in events().items():
        pos = cal.searchsorted(ev.entry_date) - 1
        ev = ev[pos >= 0].copy()
        pos = pos[pos >= 0]
        ev["gmm"], ev["ma60"] = daily["gmm"].to_numpy()[pos], daily["ma60"].to_numpy()[pos]
        ev = ev.dropna(subset=["gmm", "ma60"])
        tr, te = ev[ev.entry_date < SPLIT], ev[ev.entry_date >= SPLIT]
        out = {}
        for col in ("gmm", "ma60"):
            cell = tr.groupby(col).R.agg(["mean", "size"])
            allow = [s_ for s_, r in cell.iterrows() if r["size"] >= MIN_N and r["mean"] > 0]
            taken = te[te[col].isin(allow)]
            out[col] = {"쓰는 상태": [str(int(a)) if col == "gmm" else a for a in allow],
                        "맞춤 상태별 R": {str(int(a)) if col == "gmm" else a: [round(float(r["mean"]), 3), int(r["size"])] for a, r in cell.iterrows()},
                        "예측 거래": int(len(taken)), "예측 R": round(float(taken.R.mean()), 3) if len(taken) else None,
                        "예측 승률": round(float(taken.win.mean()), 3) if len(taken) else None, "예측 R 합": round(float(taken.R.sum()), 1)}
            out[col]["_taken"] = taken
        a, b = out["gmm"].pop("_taken"), out["ma60"].pop("_taken")
        if len(a) and len(b):
            d_, lo, hi = vr.diff_ci(a, b, rng)
            out["GMM − 60일선 R"] = [round(d_, 3), round(lo, 3), round(hi, 3)]
            out["verdict"] = "GMM이 낫다" if d_ >= 0.10 and lo > 0 else "60일선 유지"
        else:
            out["verdict"] = "60일선 유지(GMM 전략표가 거래 없음)" if not len(a) else "비교 불가"
        res["전략"][name] = out
    # 선행성: 60일선 하락장 진입 직전에 GMM 이 '하락 쪽' 상태로 먼저 바뀌었나
    overlap = fitd.groupby("gmm").ma60.apply(lambda s: (s == "하락장").mean())
    down = int(overlap.idxmax())
    m, g = daily["ma60"].to_numpy(), daily["gmm"].to_numpy()
    leads = []
    for i in range(10, len(m)):
        if m[i] == "하락장" and not (m[i - 10:i] == "하락장").any():
            if g[i] == down:
                j = i
                while j > 0 and g[j - 1] == down and i - (j - 1) <= 20:
                    j -= 1
                leads.append({"date": str(daily.index[i].date()), "선행 일수": i - j})
            else:
                later = next((t for t in range(1, 21) if i + t < len(g) and g[i + t] == down), None)
                leads.append({"date": str(daily.index[i].date()), "선행 일수": -later if later else None})
    L = pd.DataFrame(leads)
    res["선행성"] = {"하락 쪽 상태": down, "하락장 진입 횟수": int(len(L)),
                   "먼저 바뀜(≥1일)": int((L["선행 일수"] >= 1).sum()), "같은 날": int((L["선행 일수"] == 0).sum()),
                   "늦게 바뀜": int((L["선행 일수"] < 0).sum()), "20일 안 안 바뀜": int(L["선행 일수"].isna().sum()),
                   "선행 일수 중앙(먼저 바뀐 경우)": float(L.loc[L["선행 일수"] >= 1, "선행 일수"].median()) if (L["선행 일수"] >= 1).any() else None,
                   "목록": leads}
    # 9.81 추세 강도 점수의 상태별 상위−하위(서술)
    pnl = pd.read_csv(results_dir() / "holding_factors_panel.csv", parse_dates=["date"]).set_index("date")
    pnl["trend"] = pnl[["ma60", "ma120", "hi52", "rs"]].groupby(level=0).rank(pct=True).mean(axis=1, skipna=False)
    pnl["gmm"] = daily["gmm"].reindex(pnl.index).to_numpy()
    res["추세 강도 상위−하위 20일(상태별, 맞춤/예측)"] = {
        f"상태 {int(s_)}": [round(float(spreads(x[x.index < SPLIT], "trend", "x20").spread.mean()), 4) if (x.index < SPLIT).sum() else None,
                          round(float(spreads(x[x.index >= SPLIT], "trend", "x20").spread.mean()), 4) if (x.index >= SPLIT).sum() else None]
        for s_, x in pnl.dropna(subset=["gmm"]).groupby("gmm")}
    (results_dir() / "regime_gmm_validation.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k_: v for k_, v in res.items() if k_ != "선행성"} | {"선행성": {k_: v for k_, v in res["선행성"].items() if k_ != "목록"}},
                     ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
