"""results/ 아래 폴더별(ETF·바스켓별)로 흩어진 대시보드를 한 화면에서 보는 통합 인덱스.

각 바스켓 폴더(results/{basket}/)를 스캔해 완전히 다른 두 예측을 나란히, 절대 섞이지 않게 보여준다
(2026-09-27, 실제 사용자 혼동 사례로 분리 확정됨 — "한국콜마 69% 상승우세"가 ETF 얘기인지 콜마
얘기인지 구분이 안 된다는 지적):

  ① ETF 자체 예측 — 예측모델(Legacy/Breadth)이 그 타겟 ETF 자신의 익일 방향을 추정한 값
     (v2_summary_metrics.json 의 breadth_series 최신값 vs breadth_threshold, 대시보드 헤더의
     "오늘 기준 실시간 예측 카드"와 정확히 같은 로직).
  ② 센서 종목 중 최고 신호 — 챔피언-챌린저 모델이 그 바스켓 안의 개별 구성종목(센서)들을 서로
     비교해서 가장 확신이 강한 종목 하나를 뽑은 것. ETF 자체의 등락과는 무관하다 — "이 종목이
     바스켓 내 다른 종목들 대비 상대적으로 강할 확률"일 뿐, "이 종목 때문에 ETF가 오른다"는 뜻이
     아니다(대시보드 챔피언-챌린저 탭의 캡션과 동일한 제약).

상세는 각 바스켓 자기 dashboard_v2.html로 링크한다.

  python vm_spc/build_index.py                 # results/index.html 생성
  python vm_spc/pipeline.py                    # 전체 바스켓(기본, --basket 없이) 처리 후 자동 재생성
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "vm_predict", _ROOT / "pair_spc"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import html
import json
import logging
import sys

from v2_config import KST, load_baskets, results_dir
import pandas as pd

from v2_compression import compression_frame, current_state, pooled_expansion
from v2_datastore import load_bars
from v2_render_dashboard import validated_etf_extras
from etf_rank.render import INDEX_CSS as RANK_CSS, index_card_html as rank_index_card_html, load as load_etf_rank

log = logging.getLogger("vm_spc.build_index")


def _load_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _etf_prediction(summary: dict, extras: dict | None = None) -> dict | None:
    """Panel 1(예측모델/Legacy)의 '내일(T+1) 예측'을 그대로 재현 — ETF 자기 자신의 방향 예측.
    대시보드 헤더 JS(오늘 기준 실시간 예측 카드, v2_render_dashboard.py TEMPLATE)와 로직을 맞췄다.
    괴리율 조건이 검증을 통과해 있으면(extras["premium"]) 과거 같은 조건에서의 적중률을 쓴다."""
    br, cm, p = summary["breadth_series"], summary["confusion_matrix"], summary["params"]
    vals = br["breadth"]
    i = len(vals) - 1
    while i >= 0 and vals[i] is None:
        i -= 1
    if i < 0:
        return None
    val = vals[i]
    pred_up = val >= p["breadth_threshold"]
    npv = (cm["tn"] / (cm["tn"] + cm["fn"])) if (cm["tn"] + cm["fn"]) else None
    hit_rate = cm["precision"] if pred_up else npv
    base = cm.get("base_rate")
    usual = None if base is None else (base if pred_up else 1 - base)
    premium = None
    if extras and extras.get("date") == br["dates"][i] and extras.get("pred_up") == pred_up:
        premium = extras.get("premium")
        if premium and premium.get("hit_rate") is not None:
            hit_rate = premium["hit_rate"]
    return {"date": br["dates"][i], "pred_up": pred_up, "hit_rate": hit_rate, "usual": usual, "premium": premium}


def _top_sensor_signal(vm: dict) -> tuple[dict, float, bool] | None:
    """그 바스켓의 오늘 시나리오(구성 센서종목들) 중 |P-0.5| 가 가장 큰(가장 확신이 강한) 종목
    하나 — 어디까지나 '바스켓 내 다른 종목 대비 상대적 우열'이지 ETF 자체 예측이 아니다.

    챔피언이 있으면(p_champion) 검증된 확률을, 게이트 미달(REJECTED)이라 챔피언이 없으면 Baseline
    의 원값을 대신 쓴다 — 계산 자체가 안 된 것(vm_spc 데이터 없음)과, 계산은 됐는데 기준을 못
    넘은 것을 구분해서 빈칸 대신 "미검증"이라고 명시하고 보여주는 쪽이 아예 안 보여주는 것보다
    낫다는 판단(2026-09-27 사용자 피드백).

    반환: (종목, 확률, validated) — validated=False 면 게이트 미달 참고치."""
    best, best_gap, validated = None, -1.0, False
    for s in vm.get("scenario", []):
        p = s.get("p_champion")
        if p is not None:
            gap = abs(p - 0.5)
            if gap > best_gap:
                best, best_gap, validated = s, gap, True
    if best is not None:
        return best, best["p_champion"], True
    # 챔피언 없음(REJECTED) — Baseline 원값으로 대체(참고용, 미검증 표시)
    for s in vm.get("scenario", []):
        p = s.get("p_baseline")
        if p is None:
            continue
        gap = abs(p - 0.5)
        if gap > best_gap:
            best, best_gap = s, gap
    return (best, best["p_baseline"], False) if best is not None else None


_COMP_CACHE: dict[str, pd.DataFrame] = {}


def _compression(basket: dict) -> pd.DataFrame | None:
    code = basket["target"]["code"]
    if code not in _COMP_CACHE:
        bars = load_bars(basket["name"], code)
        _COMP_CACHE[code] = compression_frame(bars) if bars is not None and len(bars) > 150 else None
    return _COMP_CACHE[code]


def collect_rows() -> list[dict]:
    rows = []
    for basket in load_baskets():
        name = basket["name"]
        bdir = results_dir() / name
        summary = _load_json(bdir / "v2_summary_metrics.json")
        if summary is None:
            continue  # Stage1/2 도 아직 안 돈 바스켓 — 표시할 게 없음
        target = summary["basket"]["target"]
        dash_path = bdir / "dashboard_v2.html"
        row = {
            "basket": name, "etf_name": target["name"], "etf_code": target["code"],
            "n_sensors": len(summary["basket"]["sensors"]),
            "has_dashboard": dash_path.exists(),
            "etf_pred": _etf_prediction(summary, validated_etf_extras(name)),
            "champion": None, "top": None, "top_p": None, "top_validated": False, "gap": -1.0,
        }
        cf = _compression(basket)
        row["compression"] = current_state(cf) if cf is not None else None
        vm = _load_json(bdir / "vm_spc" / "vm_spc_dashboard_data.json")
        if vm is not None:
            row["champion"] = vm["decision"]["champion"]
            res = _top_sensor_signal(vm)
            if res is not None:
                row["top"], row["top_p"], row["top_validated"] = res
                row["gap"] = abs(row["top_p"] - 0.5)
        rows.append(row)
    # 정렬: 1차 목표는 'ETF 자체 예측'이므로 그 과거 적중률이 높은 순으로 먼저 정렬하고
    # (적중률 없는 바스켓은 맨 뒤로), 같은 적중률이면 센서 종목 신호 확신이 강한 순으로 보조 정렬한다.
    def _sort_key(r):
        ep = r["etf_pred"]
        hit = ep["hit_rate"] if ep and ep["hit_rate"] is not None else -1.0
        return (-hit, -r["gap"], r["basket"])
    rows.sort(key=_sort_key)
    return rows


def _etf_pred_cell(r: dict) -> str:
    ep = r["etf_pred"]
    if ep is None:
        return '<span class="c-empty">—</span>'
    up, hr, pr = ep["pred_up"], ep["hit_rate"], ep.get("premium")
    arrow, label, cls = ("&#9650;", "상승", "c-up") if up else ("&#9660;", "하락", "c-flat")
    usual = ep.get("usual")
    hr_str = (f"(적중 {hr:.0%}" + (f" · 평소 {usual:.0%})" if usual is not None else ")")) if hr is not None else ""
    note = ""
    if pr and pr["state"] != "중립":
        note_cls = "c-up" if pr["state"] == "동의" else "c-warn"
        note = (f'<span class="note-block">괴리율 {pr["dprt"]:+.2f}% · '
                f'<span class="{note_cls}">예측과 {"같은" if pr["state"] == "동의" else "반대"} 방향</span></span>')
    return f'<span class="val {cls}">{arrow} {label}</span> <span class="sub">{hr_str}</span>{note}'


def _sensor_signal_cell(r: dict) -> str:
    top = r["top"]
    if top is None:
        if r["champion"] is None:
            return '<span class="c-empty">검증 대기<span class="note-block">(데이터 부족)</span></span>'
        return '<span class="c-muted">신호 없음</span>'
    p = r["top_p"]
    up = p >= 0.5
    conf = p if up else (1 - p)
    if r["top_validated"]:
        arrow, label, cls = ("&#9650;", "상승 우세", "c-up") if up else ("&#9660;", "하락 우세", "c-down")
        suffix = ""
    else:
        # 게이트 미달(REJECTED) — Baseline 원값을 흐리게 보여준다. 빈칸보다 낫다는 판단이지만
        # 검증된 신호와는 확실히 구분되게(연한 색 + 라벨) 표시한다.
        arrow, label, cls = ("&#9650;", "상승", "c-muted") if up else ("&#9660;", "하락", "c-muted")
        suffix = ' <span class="note-inline">(게이트 미달·참고)</span>'
    return (f'<span class="val {cls}">{arrow} {html.escape(top["name"])}</span> '
            f'<span class="sub">({html.escape(top["code"])})</span> '
            f'<span class="val {cls} strong">{conf:.0%}</span>{suffix}')


def _compression_badge(r: dict) -> str:
    cs = r.get("compression")
    if not cs or not cs["compressed"]:
        return ""
    return (f'<div class="comp-badge" title="20일 변동성·고저폭이 모두 자기 과거 250일 중 하위 20% — 곧 크게 움직일 가능성이 '
            f'평소보다 높지만 방향은 알려주지 않습니다">&#9889; 응축 중 {cs["streak"]}일째 · 큰 움직임 가능성↑ (방향 모름)</div>')


def _row_html(r: dict) -> str:
    """표(가로 스크롤)는 모바일에서 셀마다 세로로 뭉개지는 문제가 있어서, 폭에 상관없이
    항상 읽히는 카드 하나로 바꿨다(2026-09-27, 모바일 실사용 피드백) — 좁은 화면에선 1열로
    쌓이고 넓은 화면에선 그리드로 나열된다(TEMPLATE 의 .grid 미디어쿼리).

    스타일은 Tailwind CDN 스크립트(cdn.tailwindcss.com)가 아니라 TEMPLATE 안에 고정 CSS로
    직접 넣는다 — 이 파일은 텔레그램으로 받아 로컬로 여는 경우가 많은데, 그때 외부 CDN 스크립트가
    막히거나 늦게 불러와지면 스타일이 하나도 안 먹혀서 브라우저 기본 파란 링크 목록처럼 보이는
    문제가 실제로 있었다(2026-09-27, 모바일 실사용 스크린샷으로 확인)."""
    link = f"{r['basket']}/dashboard_v2.html" if r["has_dashboard"] else None
    ep = r["etf_pred"]
    strong = " strong" if (ep and ep["hit_rate"] is not None and ep["hit_rate"] >= 0.6) else ""
    tag = "a" if link else "div"
    href_attr = f' href="{link}"' if link else ""
    return f"""<{tag}{href_attr} class="card{strong}">
      <div class="card-top">
        <div class="etf-name">{html.escape(r['etf_name'])}</div>
        <div class="etf-code">{html.escape(r['etf_code'])}</div>
      </div>
      <div class="basket-sub">{html.escape(r['basket'])} &middot; 센서 {r['n_sensors']}종목</div>{_compression_badge(r)}
      <div class="label" title="이 ETF 자신의 내일 방향 예측(예측모델/Legacy 기준) — 구성종목 신호와는 다른 예측입니다.">ETF 자체 예측</div>
      <div class="row-val">{_etf_pred_cell(r)}</div>
      <div class="label" title="이 ETF를 구성하는 센서 종목들 중 서로 비교했을 때 가장 확신이 강한 종목 — 상대적 우열이지, 이 신호 때문에 ETF가 오른다는 뜻이 아닙니다.">구성종목 중 최고 신호</div>
      <div class="row-val last">{_sensor_signal_cell(r)}</div>
    </{tag}>"""


TEMPLATE = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>VM-SPC Core — 전체 ETF 현황</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
  * { box-sizing:border-box; }
  body { background:#0f172a; color:#e2e8f0; margin:0; padding:16px;
         font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Malgun Gothic",sans-serif; }
  a { text-decoration:none; color:inherit; }
  header.card { padding:16px; margin-bottom:16px; }
  header h1 { font-size:1.15rem; font-weight:700; color:#fff; margin:0 0 6px; }
  header p { font-size:0.8rem; color:#94a3b8; margin:4px 0; line-height:1.55; }
  header p b { color:#e2e8f0; }
  header .meta { font-size:0.7rem; color:#64748b; }
  .grid { display:grid; grid-template-columns:1fr; gap:12px; }
  @media (min-width:640px) { .grid { grid-template-columns:1fr 1fr; } }
  @media (min-width:1024px) { .grid { grid-template-columns:1fr 1fr 1fr; } }
  .card { display:block; background:#1e293b; border:1px solid #334155; border-radius:12px; padding:14px; }
  a.card:active, a.card:hover { border-color:#0891b2; }
  .card.strong { border-color:#0e7490; }
  .card-top { display:flex; align-items:flex-start; justify-content:space-between; gap:8px; }
  .etf-name { font-weight:700; color:#67e8f9; font-size:0.98rem; }
  .etf-code { font-size:0.72rem; color:#64748b; white-space:nowrap; padding-top:2px; }
  .basket-sub { font-size:0.72rem; color:#64748b; margin:4px 0 10px; }
  .label { font-size:0.68rem; color:#64748b; border-bottom:1px dotted #475569; display:inline-block;
           margin-bottom:3px; cursor:help; }
  .row-val { margin-bottom:10px; font-size:0.9rem; }
  .row-val.last { margin-bottom:0; }
  .val { font-weight:600; }
  .val.strong { font-weight:700; margin-left:4px; }
  .c-up { color:#34d399; }
  .c-down { color:#f87171; }
  .c-flat { color:#cbd5e1; }
  .c-muted { color:#94a3b8; }
  .c-empty { color:#475569; }
  .c-warn { color:#fbbf24; }
  .comp-badge { display:table; font-size:0.72rem; color:#fbbf24; background:rgba(251,191,36,0.1);
                border:1px solid rgba(251,191,36,0.35); border-radius:6px; padding:2px 6px; margin:-4px 0 8px; }
  .sub { color:#64748b; font-size:0.75rem; }
  .note-inline { color:#475569; font-size:0.68rem; }
  .note-block { display:block; color:#475569; font-size:0.68rem; }
  footer.note { font-size:0.75rem; color:#475569; margin-top:16px; }
__RANK_CSS__
</style>
</head>
<body>
  <header class="card">
    <h1>VM-SPC Core — 전체 ETF 현황</h1>
    <p>이 파이프라인의 1차 목표인 <b>ETF 자체 예측</b>의 적중률이 높은 순으로 정렬했습니다(__N_TOTAL__개 중
      60% 이상 __N_SIGNAL__개). 적중률은 그 ETF를 같은 방향으로 예측했던 날의 적중률에 오늘 괴리율 효과(전체 ETF
      합산 추정)를 반영한 값이고, <b>평소</b>는 예측과 상관없이 그 방향으로 움직인 날의 비율입니다 — 적중률이 평소보다
      얼마나 높은지가 실제 예측 실력입니다. 구성종목 신호는 참고용이고, 자세한 근거는 각 ETF를 눌러 확인하세요.</p>
    <p>__COMP_NOTE__</p>
    <p class="meta">생성 시각: __GENERATED_AT__</p>
  </header>

  <div class="grid">__ROWS__</div>
__RANK_CARD__

  <footer class="note">자동 주문은 하지 않습니다 — 장 마감 후 계산한 참고용 방향 신호이며, 진입 여부는 사람이 판단합니다.</footer>
</body>
</html>
"""


def _compression_note(rows: list[dict]) -> str:
    n = sum(1 for r in rows if r.get("compression") and r["compression"]["compressed"])
    st = pooled_expansion([f for f in _COMP_CACHE.values() if f is not None])
    if st is None:
        return ""
    return (f'<b class="c-warn">&#9889; 응축 중 {n}개</b> — 과거 ETF에서 응축 뒤 20일 안에 변동성이 1.5배 이상 커진 비율은 '
            f'<b>{st["comp_rate"]:.0%}</b>(평소 {st["base_rate"]:.0%})입니다. 큰 움직임이 올 가능성만 알려주고, 방향은 알려주지 않습니다.')


def build_index() -> int:
    rows = collect_rows()
    if not rows:
        log.error("basket_watchlist.json 에 바스켓이 없거나 Stage1/2 가 하나도 안 돌았습니다 — "
                  "vm_predict/v2_run.py 먼저 실행하세요.")
        return 1
    n_signal = sum(1 for r in rows if r["etf_pred"] and r["etf_pred"]["hit_rate"] is not None
                  and r["etf_pred"]["hit_rate"] >= 0.6)
    from datetime import datetime
    html_out = (TEMPLATE
                .replace("__N_TOTAL__", str(len(rows)))
                .replace("__N_SIGNAL__", str(n_signal))
                .replace("__GENERATED_AT__", datetime.now(KST).strftime("%Y-%m-%d %H:%M"))
                .replace("__COMP_NOTE__", _compression_note(rows))
                .replace("__RANK_CARD__", rank_index_card_html(load_etf_rank()))
                .replace("__RANK_CSS__", RANK_CSS)
                .replace("__ROWS__", "".join(_row_html(r) for r in rows)))
    out_path = results_dir() / "index.html"
    out_path.write_text(html_out, encoding="utf-8")
    log.info("생성 완료: %s (%d개 ETF, 신호 %d개)", out_path, len(rows), n_signal)
    return 0


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return build_index()


if __name__ == "__main__":
    sys.exit(main())
