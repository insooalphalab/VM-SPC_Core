"""겹치는 요소가 많이 동시에 켜질수록 더 강한가(개수 = 강도) — 검증이력 9.88 사전 등록 그대로.

  python research/holding_overlap_count.py   → results/holding_overlap_count.json
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

from validate_compression_regime import regime
from validate_holding_factors import spreads
from v2_config import results_dir

RECENT, HALF = pd.Timestamp("2021-06-01"), pd.Timestamp("2024-01-01")
FAMILY = {"상승장": ("trend", ["ma20", "ma60", "ma120", "slope20", "slope60", "bbp", "rs", "hi52", "boxpos", "-vp"]),
          "하락장": ("trend", ["ma20", "ma60", "ma120", "slope20", "slope60", "bbp", "rs", "hi52", "boxpos", "-vp"]),
          "횡보·전환": ("ma60", ["ma60", "ma120", "ma20"])}


def bins(k: int) -> list[tuple[str, int, int]]:
    a, b = int(np.ceil(k / 3)), int(np.ceil(2 * k / 3))
    return [("0개", 0, 0), (f"1~{a}", 1, a), (f"{a + 1}~{b - 1}", a + 1, b - 1), (f"{b}개 이상", b, k)]


def stats(x: pd.DataFrame, k: int) -> dict:
    out = {}
    for name, lo, hi in bins(k):
        g = x[(x.cnt >= lo) & (x.cnt <= hi)]
        out[name] = {"평균 초과수익": round(float(g.groupby(level=0).x20.mean().mean()), 4) if len(g) else None, "비중": round(len(g) / len(x), 3)}
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    pnl = pd.read_csv(results_dir() / "holding_factors_panel.csv", parse_dates=["date"]).set_index("date")
    pnl["trend"] = pnl[["ma60", "ma120", "hi52", "rs"]].groupby(level=0).rank(pct=True).mean(axis=1, skipna=False)
    pnl["-vp"] = -pnl["vp"]
    pnl["reg"] = regime().reindex(pnl.index).to_numpy()
    res = {}
    for reg, (anchor, fam) in FAMILY.items():
        x = pnl[pnl.reg == reg].dropna(subset=fam + [anchor, "x20"]).copy()
        rk = x[fam].groupby(level=0).rank(pct=True)
        x["cnt"] = (rk > 0.8).sum(axis=1)
        k = len(fam)
        top, bot = bins(k)[-1], bins(k)[0]
        out = {"묶음 크기": k}
        for per, m in (("앞 기간", x.index < RECENT), ("최근", x.index >= RECENT), ("최근 앞 반쪽", (x.index >= RECENT) & (x.index < HALF)),
                       ("최근 뒤 반쪽", x.index >= HALF)):
            y = x[m]
            if y.index.nunique() < 10:
                continue
            st = stats(y, k)
            cnt_sp = (st[top[0]]["평균 초과수익"] or np.nan) - (st[bot[0]]["평균 초과수익"] or np.nan)
            anc_sp = float(spreads(y, anchor, "x20").spread.mean())
            vals = [v["평균 초과수익"] for v in st.values() if v["평균 초과수익"] is not None]
            mono = all(np.diff(vals) > 0) or all(np.diff(vals) < 0)
            out[per] = {"구간별": st, "단조": bool(mono), "개수 ⅔↑ − 0개": round(cnt_sp, 4), "기준 요소 상위−하위": round(anc_sp, 4),
                        "개수가 더 큼": bool(np.sign(cnt_sp) == np.sign(anc_sp) and abs(cnt_sp) - abs(anc_sp) >= 0.005)}
        ok = all(out.get(p_, {}).get("개수가 더 큼") for p_ in ("최근", "최근 앞 반쪽", "최근 뒤 반쪽")) and out.get("최근", {}).get("단조")
        out["판정"] = "개수로 강도 표시" if ok else "기준 요소 하나로 충분"
        res[reg] = out
    (results_dir() / "holding_overlap_count.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for reg, v in res.items():
        print("==", reg, "묶음", v["묶음 크기"], "→", v["판정"])
        for per in ("앞 기간", "최근", "최근 앞 반쪽", "최근 뒤 반쪽"):
            if per in v:
                r = v[per]
                print(f"   {per:8s} 구간 {[(n, None if s['평균 초과수익'] is None else round(s['평균 초과수익']*100, 2), s['비중']) for n, s in r['구간별'].items()]}"
                      f" 단조 {r['단조']} | 개수 ⅔↑−0 {r['개수 ⅔↑ − 0개']*100:+.2f} vs 기준 {r['기준 요소 상위−하위']*100:+.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
