"""대시보드 "종목 추적" 탭 — 결론 먼저: 종목별 네 기간의 방향·확률·평소 비율, HOLD 기간은 흐리게.
방법·검증 수치는 접힌 영역. v2_render_dashboard.py 가 이 함수를 불러 탭에 끼운다."""
from __future__ import annotations

import html

LABEL = {1: "T+1", 5: "T+5", 10: "T+10", 20: "T+20"}


def _cell(k: int, p: float | None, usual: float | None, active: bool) -> str:
    fade = "" if active else " opacity-40"
    lab = f'<div class="text-[10px] text-slate-500">{LABEL[k]}</div>'
    if p is None:
        return f'<div class="rounded bg-slate-800/60 p-1.5 text-center text-xs text-slate-600{fade}">{lab}—</div>'
    up = p >= 0.5
    conf = p if up else 1 - p
    arrow, cls = ("&#9650;", "text-emerald-400") if up else ("&#9660;", "text-red-400")
    if not active:
        cls = "text-slate-300"
    base = "" if usual is None else f'<div class="text-[10px] text-slate-500">평소 {(usual if up else 1 - usual):.0%}</div>'
    return (f'<div class="rounded bg-slate-800/60 p-1.5 text-center{fade}">{lab}'
            f'<div class="text-sm font-semibold {cls}">{arrow} {conf:.0%}</div>{base}</div>')


def track_tab_html(track: dict | None) -> str:
    if not track:
        return ""
    hz = [int(k) for k in track["horizons"]]
    status = track["status"]
    active = [LABEL[k] for k in hz if status.get(str(k), {}).get("status") == "Active"]
    head = ("검증을 통과한 기간: <b class='text-emerald-400'>" + ", ".join(active) + "</b>") if active else \
        "<b class='text-amber-400'>아직 검증을 통과한 기간이 없습니다</b> — 아래 값은 전부 미검증 참고치입니다(흐리게 표시)."

    chips = []
    for k in hz:
        s = status.get(str(k), {})
        ok = s.get("status") == "Active"
        detail = (f"고신뢰 적중 {s['hcp']:.1%} (95% 하한 {s['ci'][0]:.1%}) vs 평소 {s['base']:.1%} · 독립 표본 {s['n_indep']}"
                  if s.get("hcp") is not None else s.get("reason", ""))
        chips.append(
            f'<div class="rounded-lg border {"border-emerald-600/60" if ok else "border-slate-700"} p-2">'
            f'<div class="text-sm font-semibold {"text-emerald-400" if ok else "text-slate-400"}">{LABEL[k]} · '
            f'{"Active" if ok else "HOLD"}{"" if ok else " [" + html.escape(s.get("reason", "미검증")) + "]"}</div>'
            f'<div class="text-[11px] text-slate-500 mt-0.5">{detail}</div></div>')

    cards = []
    for st in track["stocks"]:
        w = st.get("weight")
        wtxt = f"ETF 내 비중 {w:.1f}%" if w is not None else "ETF 미편입"
        cells = "".join(_cell(k, st["p"].get(str(k)), st["usual"].get(str(k)),
                              status.get(str(k), {}).get("status") == "Active") for k in hz)
        cards.append(
            f'<div class="rounded-lg border border-slate-700 bg-slate-800/30 p-3">'
            f'<div class="flex justify-between gap-2"><div class="font-semibold text-slate-200">{html.escape(st["name"])} '
            f'<span class="text-xs text-slate-500">({st["code"]})</span></div>'
            f'<div class="text-xs {"text-cyan-400" if w is not None else "text-slate-500"} shrink-0">{wtxt}</div></div>'
            f'<div class="text-[11px] text-slate-500 mb-2">{st["market"]} 지수 대비</div>'
            f'<div class="grid grid-cols-4 gap-1.5">{cells}</div></div>')

    bm = track.get("benchmarks", {})
    return f"""
    <section class="card p-4 mb-4">
      <h2 class="text-sm font-semibold text-slate-300 mb-1">종목 추적 — 지수보다 나을까? ({track['as_of']} 기준)</h2>
      <p class="text-xs text-slate-400 mb-3">각 종목이 앞으로 1·5·10·20거래일 동안 <b>소속 시장 지수보다 더 오를지</b>의 확률입니다
        (▲ 지수보다 나음 / ▼ 지수보다 못함). {head}</p>
      <div class="grid grid-cols-2 md:grid-cols-4 gap-2 mb-3">{''.join(chips)}</div>
      <p class="text-[11px] text-slate-500 mb-3">종목은 ETF 내 비중 순입니다. 같은 종목이 다른 ETF 탭에서 다르게 나올 수 있는데, 이 탭은 이 ETF의 흐름을
        기준으로 본 값입니다 — 비중이 낮은 종목일수록 이 ETF 관점의 신호로서 중요도가 낮습니다(비중 기준일 {track.get('weights_as_of') or '미확인'}).</p>
      <div class="grid grid-cols-1 md:grid-cols-2 gap-3">{''.join(cards)}</div>
      <details class="mt-3"><summary class="text-xs text-slate-500 cursor-pointer">방법·검증 기준 보기</summary>
        <div class="text-[11px] text-slate-500 mt-2 space-y-1">
          <p>라벨: k일 뒤 (종목 수익률 − 소속 시장 지수 수익률) &gt; 0. 지수는 코스피 종목 {html.escape(bm.get('KOSPI', ''))}, 코스닥 종목 {html.escape(bm.get('KOSDAQ', ''))}.</p>
          <p>입력: 종목 통계 5개(칼만 잔차·가격 Z·CUSUM·T²·ETF 대비 공적분 잔차 Z) + ETF·지수 흐름 5개(ETF 5·20일 수익, 지수 20일 수익,
            종목의 ETF 대비 5·20일 상대수익). 모델은 Ridge 로지스틱 하나(프로토타입).</p>
          <p>검증: 기간별 Walk-Forward(학습 250일, 학습 끝 k일 제거, 공백 max(10,k)일, 검증 60일[T+1·T+5]/120일[T+10·T+20] 중 끝 k일 제외).
            전 바스켓을 합쳐 기간 단위로만 판정 — 확률 60% 이상/40% 이하 신호의 적중률 95% 하한(연속 k일 블록 부트스트랩)이 평소 비율보다
            높고 독립 표본(신호 날짜 수 ÷ k)이 30 이상이면 Active. 같은 종목·날짜를 여러 ETF가 예측하면 1/ETF 수로 가중.</p>
          <p>'평소'는 이 종목이 과거에 그 방향(지수 대비)으로 끝난 비율입니다. 확률이 평소보다 얼마나 높은지가 실제 신호입니다.</p>
          <p>아직 넣지 않은 것: 수급(누적 순매수), LightGBM, 기간별 맞춤 피처 — 통과한 기간에 한해 다음 단계에서 검토합니다.</p>
        </div>
      </details>
    </section>"""
