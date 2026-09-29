"""섹터 리포트 흐름 페이지 → results/reports/sector_reports.html (파일 하나, 외부 스크립트 없음).

섹터(테마 대표 바스켓)마다 최근 20거래일 증권사 리포트: 건수, 목표가 상향·하향·신규·유지, 순상향 비율과 직전 20일 대비 변화,
서로 다른 증권사 3곳 이상이 같은 방향으로 목표가를 바꿨으면 "확산"(데일리 리포트 트래커의 산업 롤링 규칙).
방향 예측이 아니라 "애널리스트들이 이 섹터를 지금 어떻게 보고 있나"를 모은 상태 정보다. 검증은 검증이력 9.25.

  python reports/render.py
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "reports"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import html
import json
import sys
from datetime import datetime

import pandas as pd

from sector import BENCH, MIN_REPORTS, WINDOW, load_reports, sectors
from v2_config import KST, results_dir
from v2_datastore import load_bars

SPREAD_BROKERS = 3
DIR_KO = {"up": ("상향", "c-up"), "down": ("하향", "c-dn"), "new": ("신규", "c-new"), "flat": ("유지", "sub"), "none": ("–", "sub")}


def window_stats(rep: pd.DataFrame, cal: pd.DatetimeIndex, end_i: int) -> pd.DataFrame:
    lo = cal[max(0, end_i - WINDOW + 1)]
    hi = cal[end_i]
    return rep[(rep["date"] >= lo) & (rep["date"] <= hi + pd.Timedelta(days=3))]      # 마지막 거래일 뒤 휴장일 작성분 포함


def sector_rows(rep: pd.DataFrame, cal: pd.DatetimeIndex) -> list[dict]:
    end = len(cal) - 1
    cur, prev = window_stats(rep, cal, end), window_stats(rep, cal, end - WINDOW)
    prev = prev[prev["date"] < cur["date"].min()] if not cur.empty else prev
    rows = []
    for name, s in sectors().items():
        g, gp = cur[cur["code"].isin(s["codes"])], prev[prev["code"].isin(s["codes"])]
        g = g[g["dir"] != "none"]
        gp = gp[gp["dir"] != "none"]
        cnt = g["dir"].value_counts()
        n = len(g)
        net = (cnt.get("up", 0) - cnt.get("down", 0)) / n if n >= MIN_REPORTS else None
        pn = len(gp)
        pcnt = gp["dir"].value_counts()
        pnet = (pcnt.get("up", 0) - pcnt.get("down", 0)) / pn if pn >= MIN_REPORTS else None
        up_b = g[g["dir"] == "up"]["broker"].nunique()
        dn_b = g[g["dir"] == "down"]["broker"].nunique()
        changes = g[g["dir"].isin(["up", "down", "new"])].sort_values("date", ascending=False)
        rows.append({"name": name, "etf": s["etf_name"], "etf_code": s["etf_code"], "n": n, "up": int(cnt.get("up", 0)), "down": int(cnt.get("down", 0)),
                     "new": int(cnt.get("new", 0)), "flat": int(cnt.get("flat", 0)), "net": net, "pnet": pnet,
                     "spread": "up" if up_b >= SPREAD_BROKERS and up_b > dn_b else "down" if dn_b >= SPREAD_BROKERS and dn_b > up_b else None,
                     "up_brokers": int(up_b), "down_brokers": int(dn_b), "changes": changes.head(8).to_dict("records")})
    rows.sort(key=lambda r: (r["net"] is None, -(r["net"] or 0), -r["n"]))
    return rows


def _card(r: dict) -> str:
    if r["net"] is None:
        net_html = f'<span class="sub">리포트 {r["n"]}건 — {MIN_REPORTS}건 미만이라 비율 안 냄</span>'
    else:
        cls = "c-up" if r["net"] > 0 else "c-dn" if r["net"] < 0 else "sub"
        delta = ""
        if r["pnet"] is not None:
            d = r["net"] - r["pnet"]
            delta = f' <span class="sub">(직전 20일 {r["pnet"]:+.0%} → {"▲" if d > 0 else "▼" if d < 0 else "–"})</span>'
        net_html = f'순상향 <span class="num {cls}">{r["net"]:+.0%}</span>{delta}'
    badge = ""
    if r["spread"] == "up":
        badge = f'<span class="badge b-up">상향 확산 · 증권사 {r["up_brokers"]}곳</span>'
    elif r["spread"] == "down":
        badge = f'<span class="badge b-dn">하향 확산 · 증권사 {r["down_brokers"]}곳</span>'
    bar = ""
    if r["n"]:
        w = lambda k: r[k] / r["n"] * 100
        bar = (f'<div class="bar"><i class="bu" style="width:{w("up"):.0f}%"></i><i class="bn" style="width:{w("new"):.0f}%"></i>'
               f'<i class="bf" style="width:{w("flat"):.0f}%"></i><i class="bd" style="width:{w("down"):.0f}%"></i></div>')
    items = "".join(
        f'<li><span class="sub">{c["date"]:%m/%d}</span> {html.escape(str(c["name"]))} · {html.escape(str(c["broker"]))} '
        f'<span class="{DIR_KO[c["dir"]][1]}">{DIR_KO[c["dir"]][0]}</span>'
        f'{f" → {c["tp"]:,.0f}원" if pd.notna(c["tp"]) else ""}</li>' for c in r["changes"])
    detail = f'<details><summary>목표가 변경·신규 {len(r["changes"])}건 보기</summary><ul>{items}</ul></details>' if items else ""
    return f"""<div class="card">
      <div class="top"><div class="nm">{html.escape(r['etf'])}</div>{badge}</div>
      <div class="netline">{net_html}</div>
      {bar}
      <div class="cnt sub">리포트 {r['n']}건 · <span class="c-up">상향 {r['up']}</span> · <span class="c-new">신규 {r['new']}</span> ·
        유지 {r['flat']} · <span class="c-dn">하향 {r['down']}</span></div>
      {detail}
    </div>"""


PAGE = """<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>섹터 리포트 흐름</title>
<style>
  :root { --bg:#0f172a; --card:#1e293b; --line:#334155; --tx:#e2e8f0; --mut:#94a3b8; --dim:#64748b; --up:#34d399; --dn:#f87171;
          --new:#67e8f9; --warn:#fbbf24; }
  * { box-sizing:border-box; }
  body { background:var(--bg); color:var(--tx); margin:0 auto; padding:16px; max-width:980px;
         font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Malgun Gothic",sans-serif; }
  h1 { font-size:1.1rem; color:#fff; margin:0 0 4px; }
  .intro { font-size:0.8rem; color:var(--mut); line-height:1.55; margin:0 0 12px; }
  .grid { display:grid; grid-template-columns:1fr; gap:12px; }
  @media (min-width:640px) { .grid { grid-template-columns:1fr 1fr; } }
  @media (min-width:960px) { .grid { grid-template-columns:1fr 1fr 1fr; } }
  .card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px; }
  .top { display:flex; justify-content:space-between; align-items:flex-start; gap:8px; }
  .nm { font-weight:600; color:var(--tx); }
  .badge { font-size:0.68rem; border-radius:999px; padding:1px 8px; white-space:nowrap; }
  .b-up, .b-dn { color:var(--mut); border:1px solid var(--line); }
  .netline { margin:8px 0 6px; font-size:0.85rem; color:var(--mut); } .num { font-weight:600; }
  .bar { display:flex; height:6px; border-radius:3px; overflow:hidden; background:var(--line); margin-bottom:6px; }
  .bar { opacity:.6; } .bar i { display:block; height:100%; } .bu { background:var(--up); } .bn { background:var(--new); }
  .bf { background:#475569; } .bd { background:var(--dn); }
  .cnt { font-size:0.75rem; }
  .sub { color:var(--dim); } .c-up { color:#6ee7b7; } .c-dn { color:#fca5a5; } .c-new { color:#a5f3fc; }
  .ref { font-size:0.75rem; color:var(--warn); border:1px solid rgba(251,191,36,.35); border-radius:8px; padding:6px 10px; margin:0 0 12px; }
  details { margin-top:8px; font-size:0.78rem; } summary { cursor:pointer; color:var(--mut); }
  ul { margin:6px 0 0; padding-left:16px; line-height:1.6; }
  .note { font-size:0.72rem; color:var(--dim); margin-top:16px; line-height:1.6; }
</style></head><body>
  <h1>섹터 리포트 흐름 <span class="sub" style="font-size:0.8rem;font-weight:400">(배경 자료 · 20거래일 누적)</span></h1>
  <p class="ref">20거래일 누적이라 하루하루 크게 바뀌지 않는 배경 정보입니다. 과거 13개월에서 순상향 상위 1/3 섹터가 하위 1/3보다 이후 20일
    KODEX 200 대비 평균 +2.6%p 나았습니다(검증이력 9.25).</p>
  <p class="intro">최근 20거래일(__FROM__ ~ __TO__) 증권사 리포트를 섹터별로 모았습니다. <b>순상향</b> = (목표가 상향 − 하향) ÷ 리포트 수.
    서로 다른 증권사 3곳 이상이 같은 방향으로 목표가를 바꾸면 <b>확산</b>으로 표시합니다. 애널리스트 시각의 상태 정보이며 방향 예측이 아닙니다.</p>
  <div class="grid">__CARDS__</div>
  <p class="note">__VALIDATION__<br>출처: 텔레그램 "버틀러 — 증권사 리포트 요약"(__PERIOD__, 리포트 __NREP__건). 같은 종목·증권사·작성일은 1건.
    바스켓 센서 종목만 섹터에 셉니다. 생성 __GEN__</p>
</body></html>
"""


def build() -> str:
    rep = load_reports()
    cal = load_bars("kospi_top10_to_etf", BENCH).index
    rows = sector_rows(rep, cal)
    vp = results_dir() / "report_sector_validation.json"
    vtxt = "검증(검증이력 9.25): 아직 실행 전."
    if vp.exists():
        v = json.loads(vp.read_text(encoding="utf-8"))
        m = next((r for r in v["results"] if r["H"] == 20), None)
        if m and m.get("ic") is not None:
            vtxt = (f"검증(검증이력 9.25, 20일): 섹터 순상향과 이후 20일 섹터 ETF의 KODEX 200 대비 성과 순위 상관 {m['ic']:+.3f} "
                    f"[{m['ic_ci'][0]:+.3f}, {m['ic_ci'][1]:+.3f}] — {m['status']}. 데이터 {m['first']}~{m['last']}, 독립 구간 {m['n_blocks']}개.")
    page = (PAGE.replace("__CARDS__", "".join(_card(r) for r in rows))
            .replace("__FROM__", f"{cal[-WINDOW]:%m/%d}").replace("__TO__", f"{cal[-1]:%m/%d}")
            .replace("__VALIDATION__", html.escape(vtxt))
            .replace("__PERIOD__", f"{rep['date'].min():%Y-%m-%d}~{rep['date'].max():%Y-%m-%d}")
            .replace("__NREP__", f"{len(rep):,}").replace("__GEN__", datetime.now(KST).strftime("%Y-%m-%d %H:%M")))
    out = results_dir() / "reports" / "sector_reports.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    summary = {"from": f"{cal[-WINDOW]:%Y-%m-%d}", "to": f"{cal[-1]:%Y-%m-%d}", "validation": vtxt,
               "updated": f"{rep['date'].max():%Y-%m-%d}",          # 반영된 마지막 리포트 작성일
               "sectors": {r["name"]: {k: r[k] for k in ("etf_code", "n", "up", "down", "new", "flat", "net", "spread")} for r in rows},
               "up_spread": [r["etf"] for r in rows if r["spread"] == "up"],
               "down_spread": [r["etf"] for r in rows if r["spread"] == "down"]}
    (out.parent / "sector_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    return str(out)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(build())
