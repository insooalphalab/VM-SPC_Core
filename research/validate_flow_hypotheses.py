"""수급 3가설 — ① 매집 뒤 돌파 ② 분산(거래량 폭발 돌파 + 메이저 최대 매도) ③ 수급 가격 충격 비대칭 — 검증이력 9.99 사전 등록 그대로.

  python research/validate_flow_hypotheses.py   → results/flow_hypotheses.json
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

import validate_box as vb
import validate_rebreakout as vr
from collect_investor_detail import detail_path, load_detail
from collect_program import load_program, program_path
from validate_breakout_exits import EXITS, simulate
from validate_holding_factors import boot
from v2_config import LONG_HISTORY, results_dir
from v2_datastore import load_bars

SPLIT, HALF = pd.Timestamp("2021-06-01"), pd.Timestamp("2024-01-01")
INV_START = pd.Timestamp("2021-09-01")
BOX = 30


def trade_R(o, h, l, c, atr, ma, t):
    e = t + 1
    if e >= len(c) or np.isnan(atr[t]) or np.isnan(ma[t]):
        return np.nan
    stop0 = c[t] - atr[t]
    if o[e] <= stop0:
        return np.nan
    ret, _ = simulate(o, h, l, c, ma, atr, e, stop0, EXITS["E2 50일선"])
    return (ret - vb.COST) / max(1 - stop0 / o[e], 0.01)


def per_code(code: str, mode: str):
    b = load_bars(LONG_HISTORY, code)
    if b is None or len(b) < 400:
        return [], [], []
    b = b[(b[["open", "high", "low", "close"]] > 0).all(1)]
    prog = load_program(code)
    if prog is None or prog.empty:
        return [], [], []
    val = prog["value"].reindex(b.index).replace(0, np.nan)
    if mode == "inv":
        inv = load_detail(code)
        if inv is None or inv.empty:
            return [], [], []
        iv = inv.reindex(b.index) * 1e6                                   # 백만원 → 원
        F = (iv["외국인"] + iv["기관합계"]) / val                          # ② 메이저
        ACC = {"외국인": iv["외국인"] / val, "연기금": iv["연기금"] / val, "사모": iv["사모"] / val}
        ASYM = iv["외국인"] / val
        indiv = iv["개인"].to_numpy()
        ok = b.index >= INV_START
    else:
        F = prog["prog_net"].reindex(b.index) / val
        ACC = {"프로그램": F}
        ASYM = F
        indiv = None
        ok = b.index >= pd.Timestamp("2016-01-01")
    o, h, l, c, v = (b[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
    n = len(c)
    tr = np.maximum(h - l, np.maximum(abs(h - np.r_[np.nan, c[:-1]]), abs(l - np.r_[np.nan, c[:-1]])))
    atr = pd.Series(tr).rolling(14).mean().to_numpy()
    ma = pd.Series(c).rolling(50).mean().to_numpy()
    H = pd.Series(h).rolling(BOX).max().shift(1).to_numpy()
    va20 = pd.Series(v).rolling(20).mean().shift(1).to_numpy()
    cs = pd.Series(c)
    bbw = (4 * cs.rolling(20).std() / cs.rolling(20).mean()).to_numpy()
    bbw_lo = pd.Series(bbw).rolling(250, min_periods=200).quantile(0.10).shift(1).to_numpy()
    dry = (pd.Series(v).rolling(20).mean() <= 0.7 * pd.Series(v).rolling(60).mean()).to_numpy()
    sq = (bbw <= bbw_lo) & dry
    Fv = F.to_numpy()
    Fmin20 = F.rolling(20, min_periods=15).min().to_numpy()
    accs = {k: (((x > 0).astype(float).where(x.notna()).rolling(20, min_periods=15).sum() >= 12)
                & (x.rolling(20, min_periods=15).sum() > 0)).to_numpy() for k, x in ACC.items()}
    okv = np.asarray(ok)
    brk = np.zeros(n, bool)
    brk[BOX + 1:] = (c[BOX + 1:] > H[BOX + 1:]) & (c[BOX:-1] <= H[BOX:-1])
    dist, acc_rows, asym_rows = [], [], []
    # ② 분산: 거래량 3배 · 양봉 · 30일 고가 돌파
    for t in range(BOX + 1, n - 6):
        if not okv[t] or np.isnan(va20[t]) or np.isnan(Fv[t]):
            continue
        if v[t] >= 3 * va20[t] and c[t] > o[t] and c[t] > H[t]:
            if Fv[t] < 0 and Fv[t] <= Fmin20[t] and (indiv is None or indiv[t] > 0):
                g = "분산"
            elif Fv[t] > 0:
                g = "메이저 매수"
            else:
                g = "그 외"
            dist.append({"code": code, "entry_date": b.index[t + 1], "g": g, "F": Fv[t], "R": trade_R(o, h, l, c, atr, ma, t),
                         "rev5": float((c[t + 1:t + 6] < H[t]).any())})
    # ① 응축 · 마름 뒤 20일 안 첫 돌파
    for t in np.flatnonzero(brk):
        if not okv[t] or t < 21 or t + 1 >= n:
            continue
        w = slice(t - 20, t)
        if not sq[w].any():
            continue
        row = {"code": code, "entry_date": b.index[t + 1], "R": trade_R(o, h, l, c, atr, ma, t)}
        for k, a in accs.items():
            row[k] = bool((sq[w] & a[w]).any())
        acc_rows.append(row)
    # ③ 비대칭 — 주 첫 거래일
    r = cs.pct_change()
    z = (r / r.rolling(60).std()).to_numpy()
    fa = ASYM.to_numpy()
    wk = pd.Series(b.index, index=b.index).groupby(b.index.to_period("W")).first()
    for t in b.index.get_indexer(pd.DatetimeIndex(wk.values)):
        if t < 120 or t + 20 >= n or not okv[t - 59]:
            continue
        zz, ff = z[t - 59:t + 1], fa[t - 59:t + 1]
        m = np.isfinite(zz) & np.isfinite(ff) & (ff != 0)
        up, dn = m & (ff > 0), m & (ff < 0)
        if up.sum() < 10 or dn.sum() < 10:
            continue
        bp = (zz[up] * ff[up]).sum() / (ff[up] ** 2).sum()
        bm = (zz[dn] * ff[dn]).sum() / (ff[dn] ** 2).sum()
        asym_rows.append({"date": b.index[t], "code": code, "asym": bp - bm, "ret60": c[t] / c[t - 60] - 1, "fwd20": c[t + 20] / c[t] - 1})
    return dist, acc_rows, asym_rows


def judge_events(a: pd.DataFrame, b: pd.DataFrame, mode: str, T: float, rng) -> dict:
    out = {"n": [len(a), len(b)], "R": [round(float(a.R.mean()), 3), round(float(b.R.mean()), 3)]}
    if len(a) < 20 or len(b) < 20:
        out["판정"] = "표본 부족"
        return out
    if mode == "inv":
        d = vr.diff_ci(a, b, rng)
        h1 = float(a[a.entry_date < HALF].R.mean() - b[b.entry_date < HALF].R.mean())
        h2 = float(a[a.entry_date >= HALF].R.mean() - b[b.entry_date >= HALF].R.mean())
        out.update({"차이 [CI]": [round(x, 3) for x in d], "반쪽": [round(h1, 3), round(h2, 3)]})
        ok = abs(d[0]) >= T and (d[1] > 0 or d[2] < 0) and np.sign(h1) == np.sign(h2) == np.sign(d[0]) and min(abs(h1), abs(h2)) >= T / 2
        out["판정"] = "최근경향" if ok else "아님"
    else:
        res = {}
        for pn, m in (("앞", lambda x: x[x.entry_date < SPLIT]), ("최근", lambda x: x[x.entry_date >= SPLIT])):
            res[pn] = vr.diff_ci(m(a), m(b), rng) if len(m(a)) >= 20 and len(m(b)) >= 20 else (np.nan,) * 3
        out.update({f"{k} 차이 [CI]": [round(x, 3) for x in v] for k, v in res.items()})
        pas = lambda d: abs(d[0]) >= T and (d[1] > 0 or d[2] < 0)
        same = np.sign(res["앞"][0]) == np.sign(res["최근"][0])
        out["판정"] = "검증됨" if pas(res["앞"]) and pas(res["최근"]) and same else ("최근 통과(반쪽 미확인)" if pas(res["최근"]) else "아님")
    return out


def asym_eval(d: pd.DataFrame, mode: str, rng) -> dict:
    rows = []
    for day, g in d.groupby("date"):
        if len(g) < 50:
            continue
        x = g.fwd20 - g.fwd20.mean()
        q = g.asym.rank(pct=True)
        sp = x[q > 0.8].mean() - x[q <= 0.2].mean()
        t3 = pd.qcut(g.ret60.rank(method="first"), 3, labels=False)
        cs = []
        for k in range(3):
            gg, xx = g[t3 == k], x[t3 == k]
            qq = gg.asym.rank(pct=True)
            cs.append(xx[qq > 0.8].mean() - xx[qq <= 0.2].mean())
        rows.append({"date": day, "sp": sp, "ctl": np.nanmean(cs)})
    s = pd.DataFrame(rows).set_index("date")
    out = {}
    pers = {"전체": s} if mode == "inv" else {"앞": s[s.index < SPLIT], "최근": s[s.index >= SPLIT]}
    for pn, x in pers.items():
        out[pn] = {"주 수": len(x), "상−하 %p [CI]": [round(v * 100, 2) for v in boot(x.sp, rng)],
                   "통제판 %p [CI]": [round(v * 100, 2) for v in boot(x.ctl, rng)],
                   "반쪽": [round(float(x.sp[x.index < HALF].mean()) * 100, 2), round(float(x.sp[x.index >= HALF].mean()) * 100, 2)]}
    r = out["전체" if mode == "inv" else "최근"]
    sp, ctl, hv = r["상−하 %p [CI]"], r["통제판 %p [CI]"], r["반쪽"]
    ok = abs(sp[0]) >= 1.0 and (sp[1] > 0 or sp[2] < 0) and np.sign(ctl[0]) == np.sign(sp[0]) and abs(ctl[0]) >= 0.5 \
        and np.sign(hv[0]) == np.sign(hv[1]) == np.sign(sp[0]) and min(abs(hv[0]), abs(hv[1])) >= 0.5
    if mode == "prog" and ok:
        o = out["앞"]["상−하 %p [CI]"]
        out["판정"] = "검증됨" if abs(o[0]) >= 1.0 and (o[1] > 0 or o[2] < 0) and np.sign(o[0]) == np.sign(sp[0]) else "최근경향"
    else:
        out["판정"] = "최근경향" if ok else "아님"
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    res = {}
    for mode in ("inv", "prog"):
        folder = (detail_path("000660") if mode == "inv" else program_path("000660")).parent
        codes = sorted(p.stem for p in folder.glob("*.csv"))
        if mode == "prog":                                        # 검증용 791종목만
            from universe_all import validation_codes
            codes = sorted(set(codes) & set(validation_codes()))
        D, A, S = [], [], []
        for cd in codes:
            d1, a1, s1 = per_code(cd, mode)
            D += d1
            A += a1
            S += s1
        D, A, S = pd.DataFrame(D), pd.DataFrame(A), pd.DataFrame(S)
        for nm, df in (("dist", D), ("acc", A), ("asym", S)):
            df.to_csv(results_dir() / f"flow_{nm}_{mode}.csv", index=False)
        D, A = D.dropna(subset=["R"]), A.dropna(subset=["R"])
        lab = "투자자 판(197종목, 2021-09~)" if mode == "inv" else "프로그램 판(약 790종목, 2016~)"
        out = {"종목 수": len(codes)}
        out["② 분산"] = judge_events(D[D.g == "분산"], D[D.g == "메이저 매수"], mode, 0.15, rng)
        out["② 5일 안 박스 상단 아래 되돌림"] = {g: round(float(x.rev5.mean()), 3) for g, x in D.groupby("g")}
        out["② 그 외 R"] = round(float(D[D.g == "그 외"].R.mean()), 3)
        out["② 사후 서술: 돌파일 메이저 순매도(크기 무관) − 순매수"] = judge_events(D[D.F < 0], D[D.F > 0], mode, 0.15, rng)
        main_k = "외국인" if mode == "inv" else "프로그램"
        out[f"① 매집({main_k})"] = judge_events(A[A[main_k]], A[~A[main_k]], mode, 0.15, rng)
        if mode == "inv":
            for k in ("연기금", "사모"):
                out[f"① 서술 {k}"] = judge_events(A[A[k]], A[~A[k]], mode, 0.15, rng)
        out["③ 비대칭"] = asym_eval(S, mode, rng)
        res[lab] = out
        print(lab, json.dumps(out, ensure_ascii=False, indent=1), flush=True)
    (results_dir() / "flow_hypotheses.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
