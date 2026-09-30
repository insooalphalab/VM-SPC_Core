"""검증된 셋업(9.27: T² 동반 가짜 이탈)의 익절 방식 비교 — 검증이력 9.32 사전 등록 기준 그대로.

사건·손절은 9.31(validate_stops.events)의 기준 그대로, 익절만 바꿔 같은 사건을 다시 체결한다.
  python research/validate_targets.py      → results/target_validation.json
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "vm_predict", _ROOT / "scenario", _ROOT / "research"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import json
import sys

import numpy as np
import pandas as pd

import validate_box as vb
import validate_stops as vs
from box_rules import MAX_HOLD
from v2_config import LONG_HISTORY, results_dir, sensor_universe
from v2_datastore import load_bars

VARIANTS = ["기준", "① 20일선", "②a 1.5R", "②b 2R", "③ 볼린저 상단", "④ 분할"]


def lines(c: np.ndarray) -> dict:
    s = pd.Series(c)
    ma20, sd = s.rolling(20).mean(), s.rolling(20).std()
    return {"ma20": ma20.shift(1).to_numpy(), "bb": (ma20 + 2 * sd).shift(1).to_numpy(), "ma5": s.rolling(5).mean().to_numpy()}


def simulate(o, h, l, c, ln, e, stop, variant, fixed_tgt):
    """→ (수익, 결과, 보유일). fixed_tgt: 기준·②의 고정 목표가."""
    if e + MAX_HOLD - 1 >= len(c):
        return None
    entry = o[e]
    if variant == "④ 분할":
        return _split(o, h, l, c, ln, e, stop)
    dyn = {"① 20일선": ln["ma20"], "③ 볼린저 상단": ln["bb"]}.get(variant)
    if dyn is not None and dyn[e] <= entry:                      # 이미 선 위에서 진입 — 진입일 종가에 정리
        return c[e] / entry - 1, "target", 1
    for j in range(e, e + MAX_HOLD):
        tgt = dyn[j] if dyn is not None else fixed_tgt
        if j > e:
            if o[j] <= stop:
                return o[j] / entry - 1, "stop", j - e + 1
            if o[j] >= tgt:
                return o[j] / entry - 1, "target", j - e + 1
        if l[j] <= stop:
            return stop / entry - 1, "stop", j - e + 1
        if h[j] >= tgt:
            return tgt / entry - 1, "target", j - e + 1
    return c[e + MAX_HOLD - 1] / entry - 1, "timeout", MAX_HOLD


def _split(o, h, l, c, ln, e, stop):
    entry, ma20, ma5 = o[e], ln["ma20"], ln["ma5"]
    half, j0 = None, None
    if ma20[e] <= entry:
        half, j0 = c[e] / entry - 1, e
    else:
        for j in range(e, e + MAX_HOLD):
            if j > e and o[j] <= stop:
                return o[j] / entry - 1, "stop", j - e + 1
            if l[j] <= stop:
                return stop / entry - 1, "stop", j - e + 1
            if (j > e and o[j] >= ma20[j]) or h[j] >= ma20[j]:
                half, j0 = (o[j] if j > e and o[j] >= ma20[j] else ma20[j]) / entry - 1, j
                break
        if half is None:
            return c[e + MAX_HOLD - 1] / entry - 1, "timeout", MAX_HOLD
    for j in range(j0 + 1, e + MAX_HOLD):                        # 나머지 절반: 본전 손절 + 5일선 종가 이탈
        if o[j] <= entry:
            return 0.5 * half + 0.5 * (o[j] / entry - 1), "split", j - e + 1
        if l[j] <= entry:
            return 0.5 * half, "split", j - e + 1
        if c[j] < ma5[j]:
            return 0.5 * half + 0.5 * (c[j] / entry - 1), "split", j - e + 1
    return 0.5 * half + 0.5 * (c[e + MAX_HOLD - 1] / entry - 1), "split", MAX_HOLD


def run(codes: list[str], label: str, rng) -> pd.DataFrame:
    rows = []
    for code in codes:
        bars = load_bars(LONG_HISTORY, code)
        if bars is None or len(bars) < 300:
            continue
        bars = bars[(bars[["open", "high", "low", "close"]] > 0).all(1)]
        o, h, l, c = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        ln = lines(c)
        lo, hi = max(int(bars.index.searchsorted(vs.START)), 25), len(c) - MAX_HOLD
        for ev in vs.events(bars):
            e, stop = ev["e"], ev["stops"]["기준"]
            risk = o[e] - stop
            fixed = {"기준": ev["tgt"], "②a 1.5R": o[e] + 1.5 * risk, "②b 2R": o[e] + 2 * risk}
            days = rng.integers(lo, hi, vb.N_CTRL)
            sp = 1 - stop / o[e]
            for v in VARIANTS:
                ret, how, hold = simulate(o, h, l, c, ln, e, stop, v, fixed.get(v))
                tp = fixed[v] / o[e] - 1 if v in fixed else None
                ctrl = []
                for d in days:
                    d = int(d)
                    ctrl.append(simulate(o, h, l, c, ln, d, o[d] * (1 - sp), v, o[d] * (1 + tp) if tp is not None else None)[0])
                rows.append({"universe": label, "code": code, "e": e, "entry_date": ev["entry_date"], "variant": v,
                             "ret": ret, "how": how, "hold": hold, "ctrl_ret": float(np.mean(ctrl))})
    df = pd.DataFrame(rows)
    cal = load_bars(LONG_HISTORY, "069500").index
    df["block"] = cal.searchsorted(df["entry_date"]) // vb.BLOCK
    return df


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    rng = np.random.default_rng(0)
    df = pd.concat([run(sorted(sensor_universe()), "센서", rng), run(vs._codes("oos_codes.json"), "코스피 밖", rng),
                    run(vs._codes("oos_kosdaq_codes.json"), "코스닥 밖", rng)], ignore_index=True)
    df.to_csv(results_dir() / "target_validation_events.csv", index=False)
    res = {}
    for (u, v), g in df.groupby(["universe", "variant"], sort=False):
        ex = (g["ret"] - g["ctrl_ret"]).to_numpy()
        lo, hi = vb.boot_mean(ex, g["block"].to_numpy(), rng)
        base = df[(df.universe == u) & (df.variant == "기준")].set_index(["code", "e"])["ret"]
        gi = g.set_index(["code", "e"])
        pair = (gi["ret"] - base.reindex(gi.index)).dropna()
        plo, phi = vb.boot_mean(pair.to_numpy(), gi.loc[pair.index, "block"].to_numpy(), rng) if v != "기준" else (0.0, 0.0)
        res[f"{u}/{v}"] = {"n": int(len(g)), "net": round(float((g["ret"] - vb.COST).mean()), 4),
                           "win": round(float((g["ret"] - vb.COST > 0).mean()), 4),
                           "target_rate": round(float(g["how"].isin(["target", "split"]).mean()), 4),
                           "hold": round(float(g["hold"].mean()), 1),
                           "excess": round(float(ex.mean()), 4), "excess_ci": [round(lo, 4), round(hi, 4)],
                           "vs_base": round(float(pair.mean()), 4), "vs_base_ci": [round(plo, 4), round(phi, 4)],
                           "exit_mix": g["how"].value_counts(normalize=True).round(3).to_dict()}
    verdict = {}
    for v in VARIANTS[1:]:
        ex_ok = all(res.get(f"{u}/{v}", {}).get("excess_ci", [0])[0] > 0 for u in ("코스피 밖", "코스닥 밖"))
        pair_ok = all(res.get(f"{u}/{v}", {}).get("vs_base_ci", [0])[0] > 0 for u in ("센서", "코스피 밖"))
        verdict[v] = "채택" if ex_ok and pair_ok else "현행 유지"
    (results_dir() / "target_validation.json").write_text(json.dumps({"verdict": verdict, **res}, ensure_ascii=False, indent=1),
                                                            encoding="utf-8")
    for k, s in res.items():
        print(f"{k:16s} n={s['n']:4d} 도달 {s['target_rate']:4.0%} 승률 {s['win']:4.0%} 보유 {s['hold']:4.1f}일 순수익 {s['net']:+.2%} | "
              f"무작위 대비 {s['excess']:+.2%} [{s['excess_ci'][0]:+.2%}, {s['excess_ci'][1]:+.2%}] | "
              f"기준 대비 {s['vs_base']:+.2%} [{s['vs_base_ci'][0]:+.2%}, {s['vs_base_ci'][1]:+.2%}]")
    print(verdict)
    return 0


if __name__ == "__main__":
    sys.exit(main())
