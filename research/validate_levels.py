"""레벨 정의 비교 — 30일 박스 vs 매물대 vs 칼만 관리선 (+ 자체 T² 조건). 검증이력 9.27 사전 등록 기준 그대로.

체결 규칙·비용·무작위 대조군·요약은 research/validate_box.py(9.18)를 그대로 쓴다.
  python research/validate_levels.py         → results/levels_validation.json
  python research/validate_levels.py --oos   → results/levels_validation_oos.json (표본 밖 200종목, 먼저 일봉 수집)
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

import validate_box as vb
from box_rules import BOX, FAIL_RECOVER, RETEST_WITHIN, t2_flags
from v2_config import LONG_HISTORY, Params, results_dir, sensor_universe
from v2_datastore import load_bars
from v2_signals import kalman_features

VP_WIN, VP_BINS, VP_TOP = 120, 40, 0.8
BAND = 2.0


# ── 레벨 정의: 날짜 b 에서 쓰는 선(전날까지 데이터) ─────────────
def levels_box(bars: pd.DataFrame) -> dict:
    H = bars["high"].rolling(BOX).max().shift(1).to_numpy()
    L = bars["low"].rolling(BOX).min().shift(1).to_numpy()
    M = (H + L) / 2
    return {"f_line": L, "f_tgt": H, "r_line": H, "r_stop": M, "r_tgt": H + (H - L)}


def levels_vp(bars: pd.DataFrame) -> dict:
    h, l, c, v = (bars[k].to_numpy(float) for k in ("high", "low", "close", "volume"))
    tp = (h + l + c) / 3
    n = len(c)
    S, R = np.full(n, np.nan), np.full(n, np.nan)
    for b in range(VP_WIN + 1, n):
        w = slice(b - VP_WIN, b)
        lo, hi = l[w].min(), h[w].max()
        if not hi > lo:
            continue
        vol, edges = np.histogram(tp[w], bins=VP_BINS, range=(lo, hi), weights=v[w])
        centers = (edges[:-1] + edges[1:]) / 2
        nodes = centers[vol >= np.quantile(vol, VP_TOP)]
        prev = c[b - 1]
        below, above = nodes[nodes < prev], nodes[nodes > prev]
        S[b] = below.max() if len(below) else lo
        R[b] = above.min() if len(above) else hi
    return {"f_line": S, "f_tgt": R, "r_line": R, "r_stop": S, "r_tgt": R + (R - S)}


def levels_kalman(bars: pd.DataFrame) -> dict:
    f = kalman_features(bars, Params())
    pred = (np.log(f["close"]) - f["innov"]).to_numpy()
    sig = f["sigma"].to_numpy()
    C, LCL, UCL = np.exp(pred), np.exp(pred - BAND * sig), np.exp(pred + BAND * sig)
    return {"f_line": LCL, "f_tgt": C, "r_line": UCL, "r_stop": C, "r_tgt": UCL + (UCL - LCL)}


LEVELS = {"box": levels_box, "vp": levels_vp, "kalman": levels_kalman}


# ── 사건 찾기(9.18 find_events 와 같은 규칙, 선만 바꿈) ───────────
def find(bars: pd.DataFrame, lv: dict, t2: np.ndarray, start) -> list[dict]:
    o, h, l, c = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    first = int(bars.index.searchsorted(start))
    out = []
    for pattern in ("failure", "retest"):
        busy = -1
        for b in range(max(first, VP_WIN + 2), len(c)):
            if b <= busy:
                continue
            r = None
            if pattern == "failure":
                line, tgt = lv["f_line"][b], lv["f_tgt"][b]
                if not (np.isfinite(line) and np.isfinite(tgt)) or c[b] >= line:
                    continue
                r = next((k for k in range(b + 1, min(b + 1 + FAIL_RECOVER, len(c))) if c[k] > line), None)
                stop = l[b:r + 1].min() if r is not None else None
            else:
                line, stop, tgt = lv["r_line"][b], lv["r_stop"][b], lv["r_tgt"][b]
                if not (np.isfinite(line) and np.isfinite(stop) and np.isfinite(tgt)) or c[b] <= line or stop >= line:
                    continue
                for k in range(b + 1, min(b + 1 + RETEST_WITHIN, len(c))):
                    if c[k] < stop:
                        break
                    if l[k] <= line:
                        r = k
                        break
            if r is None or r + 1 >= len(c):
                continue
            e = r + 1
            if o[e] <= stop or o[e] >= tgt:
                continue
            sim = vb.simulate(o, h, l, c, e, stop, tgt)
            if sim is None:
                continue
            ret, days, how = sim
            busy = e + days - 1
            out.append({"pattern": pattern, "e": e, "entry_date": bars.index[e], "entry": o[e],
                        "stop_pct": 1 - stop / o[e], "tgt_pct": tgt / o[e] - 1, "ret": ret, "days": days, "how": how,
                        "t2": bool(t2[b])})
    return out


def run(universe: str, rng) -> dict:
    if universe == "etf":
        vb.START = pd.Timestamp("2006-01-01")
        codes = vb.etf_universe()
    elif universe == "oos":                          # 9.27 표본 밖: 코스피 시총 상위, 센서 종목 제외 200
        vb.START = pd.Timestamp("2016-01-01")
        codes = json.loads((_ROOT / "data" / LONG_HISTORY / "oos_codes.json").read_text(encoding="utf-8"))["codes"]
    else:
        vb.START = pd.Timestamp("2016-01-01")
        codes = sorted(sensor_universe())
    cal = load_bars(LONG_HISTORY, "069500").index
    rows = []
    for code in codes:
        bars = load_bars(LONG_HISTORY, code)
        if bars is None or len(bars) < 300:
            continue
        bars = bars[(bars[["open", "high", "low", "close"]] > 0).all(1)]
        t2 = t2_flags(bars)
        for name, fn in LEVELS.items():
            for ev in find(bars, fn(bars), t2, vb.START):
                ev["ctrl_ret"], ev["ctrl_tgt"], ev["ctrl_days"], ev["ctrl_to"] = vb.controls(bars, ev, rng)
                ev["level"], ev["code"] = name, code
                rows.append(ev)
    df = pd.DataFrame(rows)
    df["block"] = cal.searchsorted(df["entry_date"]) // vb.BLOCK
    out = {}
    for (lvl, pat), g in df.groupby(["level", "pattern"]):
        out[f"{lvl}/{pat}"] = {"all": vb.summarize(g, rng, verdict=True),
                               "t2": vb.summarize(g[g["t2"]], rng, verdict=True) if g["t2"].sum() >= 30 else {"n": int(g["t2"].sum())}}
    return out


def main_oos() -> int:
    """표본 밖 재현(9.27): 박스·매물대 가짜 이탈의 T² 조건만."""
    rng = np.random.default_rng(0)
    res = run("oos", rng)
    box_t2, box_all = res["box/failure"]["t2"], res["box/failure"]["all"]
    ok = box_t2.get("excess_ci", [0])[0] > 0 and box_t2.get("excess", -1) > box_all["excess"]
    out = {"replicated": bool(ok), "universe": "코스피 시총 상위, 센서 제외 200, 2016~", **res}
    (results_dir() / "levels_validation_oos.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("재현" if ok else "재현 안 됨")
    return 0


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    if "--oos" in sys.argv:
        return main_oos()
    rng = np.random.default_rng(0)
    res = {u: run(u, rng) for u in ("stocks", "etf")}
    verdict = {}
    for pat in ("failure", "retest"):
        for lvl in ("vp", "kalman"):
            s, e = res["stocks"].get(f"{lvl}/{pat}"), res["etf"].get(f"{lvl}/{pat}")
            sb, eb = res["stocks"][f"box/{pat}"], res["etf"][f"box/{pat}"]
            if not (s and e):
                continue
            better = s["all"]["excess"] > sb["all"]["excess"] and e["all"]["excess"] > eb["all"]["excess"]
            passed = s["all"].get("status") == "Active" and e["all"].get("status") == "Active"
            verdict[f"{lvl}/{pat}"] = "박스보다 나음" if better and passed else "채택 안 함"
    out = {"verdict": verdict, **res}
    (results_dir() / "levels_validation.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(verdict, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
