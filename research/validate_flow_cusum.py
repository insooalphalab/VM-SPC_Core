"""수급·프로그램매매 CUSUM 경보 검증 — 검증이력 9.21 사전 등록 기준 그대로.

  python research/validate_flow_cusum.py            → results/flow_cusum_validation.json   (9.21: 외국인·기관합계·프로그램)
  python research/validate_flow_cusum.py --detail   → results/inst_detail_validation.json  (9.22: 기관 세부 주체)
프로그램매매(data/_program/)가 아직 없으면 그 원천만 건너뛴다. --no-program 이면 수급(외국인·기관)만.
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

from collect_investor_detail import load_detail
from collect_program import load_program
from stock_track.data import kospi_codes, market_of
from stock_track.flow import load_flow
from flow_alarm import flow_cusum, onsets
from v2_config import results_dir, sensor_universe
from v2_datastore import load_bars

LB = "_long_history"
HORIZONS = (5, 20)
MAIN_H = 20
N_BOOT = 2000
MIN_INDEP = 30
DETAIL_JUDGED = ["연기금", "투신", "사모", "금융투자", "기타법인"]      # 9.22 판정 대상
DETAIL_REF = ["은행", "보험", "기타금융"]                             # 거래가 드물어 참고만


def net_series(code: str, program: bool = True, detail: bool = False) -> dict[str, pd.Series]:
    if detail:
        d = load_detail(code)
        return {} if d is None else {k: d[k] for k in DETAIL_JUDGED + DETAIL_REF if k in d.columns}
    out = {}
    f = load_flow(code)
    if f is not None:
        out["외국인"], out["기관합계"] = f["외국인"], f["기관합계"]
    p = load_program(code) if program else None
    if p is not None and not p.empty:
        out["프로그램"] = p["prog_net"]
    return out


def alarms(x: pd.Series) -> tuple[pd.Series, pd.Series]:
    return onsets(flow_cusum(x))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    kos, rng = kospi_codes(), np.random.default_rng(0)
    bench = {"KOSPI": load_bars(LB, "069500")["close"], "KOSDAQ": load_bars(LB, "229200")["close"]}
    cal = bench["KOSPI"].index
    events, base_hits = [], {H: [] for H in HORIZONS}
    corr = {}
    for code in sorted(sensor_universe()):
        bars = load_bars(LB, code)
        src = net_series(code, "--no-program" not in sys.argv, detail="--detail" in sys.argv)
        if bars is None or not src:
            continue
        c = bars["close"]
        bm = bench[market_of(code, kos)].reindex(c.index)
        # 다음 거래일 종가 진입 → H일 뒤 종가
        fwd = {H: ((c.shift(-1 - H) / c.shift(-1) - 1) - (bm.shift(-1 - H) / bm.shift(-1) - 1)) for H in HORIZONS}
        start = min(s.index.min() for s in src.values())
        for H in HORIZONS:
            base_hits[H].append((fwd[H].loc[start:].dropna() > 0).to_numpy())
        ex0 = c.pct_change(fill_method=None) - bm.pct_change(fill_method=None)
        for name, s in src.items():
            s = s.reindex(c.index).loc[s.index.min():s.index.max()].dropna()
            if len(s) < 300:
                continue
            if True:                                   # 모든 원천: 같은 날 / 다음 날 / 5·20일 순위 상관(참고)
                d = pd.DataFrame({"x": s, "ex0": ex0.reindex(s.index), "f1": ((c.shift(-1) / c - 1) - (bm.shift(-1) / bm - 1)).reindex(s.index),
                                  "f5": fwd[5].reindex(s.index), "f20": fwd[20].reindex(s.index)})
                corr.setdefault(name, []).append({k: d["x"].corr(d[k], method="spearman") for k in ("ex0", "f1", "f5", "f20")})
            up, dn = alarms(s)
            for kind, flag in (("up", up), ("down", dn)):
                for d in flag[flag].index:
                    row = {"src": name, "kind": kind, "code": code, "date": d}
                    for H in HORIZONS:
                        row[f"ex{H}"] = fwd[H].get(d, np.nan)
                    events.append(row)
    ev = pd.DataFrame(events)
    base = {H: float(np.concatenate(v).mean()) for H, v in base_hits.items()}
    ev["block_i"] = cal.searchsorted(ev["date"])
    res = []
    for (src, kind), g in ev.groupby(["src", "kind"]):
        for H in HORIZONS:
            gg = g.dropna(subset=[f"ex{H}"])
            y = (gg[f"ex{H}"] > 0).astype(float)
            blk = pd.DataFrame({"b": gg["block_i"] // H, "y": y}).groupby("b")["y"].agg(["sum", "count"]).to_numpy()
            idx = rng.integers(0, len(blk), (N_BOOT, len(blk)))
            bs = blk[idx, 0].sum(1) / blk[idx, 1].sum(1)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            ok = (lo > base[H]) if kind == "up" else (hi < base[H])
            status = ("참고(기준 충족)" if ok else "참고") if H != MAIN_H else \
                (("Active" if len(blk) >= MIN_INDEP else "잠정 통과") if ok else "HOLD")
            res.append({"src": src, "kind": kind, "H": H, "n": int(len(gg)), "n_indep": int(len(blk)),
                        "hit": round(float(y.mean()), 4), "hit_ci": [round(lo, 4), round(hi, 4)], "base": round(base[H], 4),
                        "mean_ex": round(float(gg[f"ex{H}"].mean()), 4), "status": status})
    out = {"base": {str(k): round(v, 4) for k, v in base.items()}, "results": res,
           "corr": {k: {c: round(float(np.nanmean([r[c] for r in v])), 4) for c in ("ex0", "f1", "f5", "f20")}
                    for k, v in corr.items()}}
    if "--detail" in sys.argv:
        main_res = [r for r in res if r["H"] == MAIN_H and r["src"] in DETAIL_JUDGED]
        passed = [r for r in main_res if r["status"] in ("Active", "잠정 통과")]
        lean = []
        for r in passed:                              # 통과한 주체의 반대 방향도 예상 쪽으로 기우나
            opp = next((x for x in main_res if x["src"] == r["src"] and x["kind"] != r["kind"]), None)
            if opp and ((opp["kind"] == "up" and opp["hit"] > opp["base"]) or (opp["kind"] == "down" and opp["hit"] < opp["base"])):
                lean.append(r["src"])
        out["verdict"] = {"passed": [f'{r["src"]} {r["kind"]}' for r in passed], "opposite_leans": lean,
                          "clue": bool(len(passed) >= 2 or lean)}
    name = "inst_detail_validation.json" if "--detail" in sys.argv else "flow_cusum_validation.json"
    (results_dir() / name).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
