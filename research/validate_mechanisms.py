"""메커니즘별 검증 — 검증이력 9.28 사전 등록 기준 그대로.

  python research/validate_mechanisms.py control   # C 양성 대조군(단기 되돌림)
  python research/validate_mechanisms.py pressure  # A 강제 매매 압력 → 되돌림 (+ 표본 밖)
  python research/validate_mechanisms.py leadlag   # B 섹터 안 선행·후행
결과: results/mechanisms_validation.json (파트별 키)
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "vm_predict", _ROOT / "pair_spc", _ROOT / "dart_events",
           _ROOT / "stock_track", _ROOT / "scenario", _ROOT / "research"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import json
import sys

import numpy as np
import pandas as pd

from stock_track.data import kospi_codes, market_of
from v2_config import LONG_HISTORY, load_baskets, results_dir, sensor_universe
from v2_datastore import load_bars

N_BOOT = 2000
START = pd.Timestamp("2016-01-01")
OUT = results_dir() / "mechanisms_validation.json"


def _save(key: str, val) -> None:
    d = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    d[key] = val
    OUT.write_text(json.dumps(d, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


def _bench():
    return {"KOSPI": load_bars(LONG_HISTORY, "069500"), "KOSDAQ": load_bars(LONG_HISTORY, "229200")}


def _oos_codes() -> list[str]:
    return json.loads((_ROOT / "data" / LONG_HISTORY / "oos_codes.json").read_text(encoding="utf-8"))["codes"]


def block_ci(values: pd.Series, k: int, rng) -> tuple[float, float]:
    """날짜 순서대로 연속 k개 날짜씩 묶어 블록 평균을 복원추출."""
    v = values.dropna().sort_index()
    blocks = np.array([v.iloc[i:i + k].mean() for i in range(0, len(v), k)])
    idx = rng.integers(0, len(blocks), (N_BOOT, len(blocks)))
    lo, hi = np.percentile(blocks[idx].mean(1), [2.5, 97.5])
    return float(lo), float(hi)


# ── C. 양성 대조군 ─────────────────────────────────────
def control(rng) -> dict:
    kos, bench = kospi_codes(), _bench()
    past, fwd = {}, {}
    for code in sorted(sensor_universe()):
        b = load_bars(LONG_HISTORY, code)
        if b is None:
            continue
        c = b["close"]
        bm = bench[market_of(code, kos)]["close"].reindex(c.index)
        past[code] = (c / c.shift(5)) - (bm / bm.shift(5))
        fwd[code] = (c.shift(-5) / c) - (bm.shift(-5) / bm)
    P, F = pd.DataFrame(past).loc[START:], pd.DataFrame(fwd).loc[START:]
    ics = {}
    for d in P.index:
        ok = P.loc[d].notna() & F.loc[d].notna()
        if ok.sum() >= 50:
            ics[d] = P.loc[d][ok].rank().corr(F.loc[d][ok].rank())
    ic = pd.Series(ics)
    lo, hi = block_ci(ic, 5, rng)
    out = {"n_days": int(len(ic)), "ic": round(float(ic.mean()), 4), "ic_ci": [round(lo, 4), round(hi, 4)],
           "detected": bool(hi < 0),
           "by_period": {p: round(float(ic.loc[a:z].mean()), 4) for p, (a, z) in
                         {"2016-2020": ("2016", "2020"), "2021-2026": ("2021", "2026")}.items()}}
    _save("C_positive_control", out)
    return out


# ── A. 강제 매매 압력 ───────────────────────────────────
HOR = (1, 3, 5)
MAIN_K = 3


def pressure_frame(codes: list[str], with_flows: bool) -> pd.DataFrame:
    from box_rules import t2_flags
    kos, bench = kospi_codes(), _bench()
    rows = []
    for code in codes:
        b = load_bars(LONG_HISTORY, code)
        if b is None or len(b) < 300:
            continue
        b = b[(b[["open", "high", "low", "close"]] > 0).all(1)]
        bm = bench[market_of(code, kos)].reindex(b.index)
        o, c, v = b["open"], b["close"], b["volume"]
        ex_today = c.pct_change(fill_method=None) - bm["close"].pct_change(fill_method=None)
        vr = v / v.rolling(20).mean().shift(1)
        t2 = pd.Series(t2_flags(b), index=b.index)
        df = pd.DataFrame({"ex": ex_today, "vr": vr, "t2": t2})
        for k in HOR:
            df[f"f{k}"] = (c.shift(-k) / o.shift(-1) - 1) - (bm["close"].shift(-k) / bm["open"].shift(-1) - 1)
        if with_flows:
            from collect_investor_detail import load_detail
            f = load_detail(code)
            if f is not None:
                df["prsn"] = f["개인"].reindex(b.index)
                df["inst_frgn"] = (f["외국인"] + f["기관합계"]).reindex(b.index)
        df["code"] = code
        rows.append(df.loc[START:])
    return pd.concat(rows)


def event_test(d: pd.DataFrame, mask: pd.Series, rng) -> dict:
    out = {"n": int(mask.sum())}
    cal = pd.DatetimeIndex(sorted(d.index.unique()))
    for k in HOR:
        base = d[f"f{k}"].mean()
        e = d.loc[mask, f"f{k}"].dropna() - base
        # 사건 가중(점 추정과 같은 가중): 연속 k 거래일 블록마다 (합, 개수)를 복원추출 — 9.18 boot_mean 과 같은 방식
        blk = pd.DataFrame({"b": cal.get_indexer(e.index) // k, "x": e.to_numpy()}).groupby("b")["x"].agg(["sum", "count"]).to_numpy()
        daily = e.groupby(level=0).mean()
        if len(blk) > 10:
            idx = rng.integers(0, len(blk), (N_BOOT, len(blk)))
            lo, hi = np.percentile(blk[idx, 0].sum(1) / blk[idx, 1].sum(1), [2.5, 97.5])
        else:
            lo, hi = np.nan, np.nan
        raw = d.loc[mask, f"f{k}"].dropna()
        out[f"k{k}"] = {"excess": round(float(e.mean()), 4), "ci": [round(float(lo), 4), round(float(hi), 4)],
                        "hit": round(float((raw > 0).mean()), 4), "base_hit": round(float((d[f"f{k}"] > 0).mean()), 4),
                        "n_days": int(len(daily))}
    m = out[f"k{MAIN_K}"]
    out["status"] = "통과" if m["ci"][0] > 0 else "HOLD"
    return out


def pressure(rng) -> dict:
    res = {}
    d = pressure_frame(sorted(sensor_universe()), with_flows=True)
    a1 = (d["ex"] <= -0.05) & (d["vr"] >= 3)
    a3 = d["t2"].astype(bool) & (d["ex"] <= -0.03)
    res["A1_panic"] = event_test(d, a1, rng)
    if "prsn" in d:
        a2 = a1 & (d["prsn"] > 0) & (d["inst_frgn"] < 0)
        res["A2_retail_absorbs"] = event_test(d, a2, rng)
    res["A3_t2_drop"] = event_test(d, a3, rng)
    o = pressure_frame(_oos_codes(), with_flows=False)
    res["oos_A1_panic"] = event_test(o, (o["ex"] <= -0.05) & (o["vr"] >= 3), rng)
    res["oos_A3_t2_drop"] = event_test(o, o["t2"].astype(bool) & (o["ex"] <= -0.03), rng)
    for key in ("A1_panic", "A3_t2_drop"):
        res[key]["replicated_oos"] = res[key]["status"] == "통과" and res["oos_" + key]["status"] == "통과"
    _save("A_pressure", res)
    return res


# ── B. 섹터 안 선행·후행 ────────────────────────────────
def leadlag(rng) -> dict:
    from client import dart_dir
    shares = pd.read_csv(dart_dir() / "shares.csv", dtype={"stock_code": str}).set_index("stock_code")["shares"]
    kos, bench = kospi_codes(), _bench()
    events = []
    for bk in load_baskets():
        if not bk["name"].endswith("_to_etf"):
            continue
        codes = [s["code"] for s in bk["sensors"]]
        px = {c: load_bars(bk["name"], c) for c in codes}
        px = {c: v for c, v in px.items() if v is not None and c in shares.index}
        if len(px) < 3:
            continue
        leader = max(px, key=lambda c: px[c]["close"].iloc[-1] * shares[c])
        ex = {}
        for c, b in px.items():
            bm = bench[market_of(c, kos)]["close"].reindex(b.index)
            ex[c] = b["close"].pct_change(fill_method=None) - bm.pct_change(fill_method=None)
        E = pd.DataFrame(ex)
        lead = E[leader]
        fol = E.drop(columns=leader)
        nxt = fol.shift(-1)
        for d in lead.index[(lead.abs() >= 0.04)]:
            sgn = np.sign(lead[d])
            f = nxt.loc[d].dropna()
            if len(f):
                events.append({"date": d, "basket": bk["name"], "leader": leader, "sign": sgn,
                               "next": float((f * sgn).mean()), "same": float((fol.loc[d].dropna() * sgn).mean())})
    ev = pd.DataFrame(events).set_index("date").sort_index()
    daily = ev["next"].groupby(level=0).mean()
    cal = pd.DatetimeIndex(sorted(ev.index.unique()))
    blk = pd.DataFrame({"b": cal.get_indexer(ev.index) // 5, "x": ev["next"].to_numpy()}).groupby("b")["x"].agg(["sum", "count"]).to_numpy()
    idx = rng.integers(0, len(blk), (N_BOOT, len(blk)))
    lo, hi = np.percentile(blk[idx, 0].sum(1) / blk[idx, 1].sum(1), [2.5, 97.5])   # 사건 가중(점 추정과 같은 가중)
    out = {"n_events": int(len(ev)), "n_days": int(len(daily)), "next_day": round(float(ev["next"].mean()), 4),
           "ci": [round(lo, 4), round(hi, 4)], "same_day": round(float(ev["same"].mean()), 4),
           "up_events_next": round(float(ev.loc[ev["sign"] > 0, "next"].mean()), 4),
           "down_events_next": round(float(ev.loc[ev["sign"] < 0, "next"].mean()), 4),
           "status": "통과" if lo > 0 else "HOLD"}
    _save("B_leadlag", out)
    return out


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    rng = np.random.default_rng(0)
    part = sys.argv[1] if len(sys.argv) > 1 else "control"
    fn = {"control": control, "pressure": pressure, "leadlag": leadlag}[part]
    print(json.dumps(fn(rng), ensure_ascii=False, indent=1, default=str))
