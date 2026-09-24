"""Stage 3: results/{basket}/v2_summary_metrics.json → results/{basket}/dashboard_v2.html

CDN 기반 라이브러리(Plotly.js·Chart.js·Tailwind)만으로 구성된 단일 독립형 HTML — 서버 없이
더블클릭으로 브라우저에서 바로 열람 가능. API 재호출 없이 이 단계만 다시 돌려도 된다.

  python v2_render_dashboard.py                       # basket_watchlist.json 의 첫 바스켓
  python v2_render_dashboard.py --basket kospi_top10_to_etf
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "vm_predict", _ROOT / "pair_spc"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import argparse
import html
import json
import logging
import sys
from datetime import datetime, timedelta

from v2_config import get_basket, results_dir

CHART_DISPLAY_DAYS = 365  # 그래프는 수집기간과 무관하게 항상 최근 1년치만 보여준다(표·통계는 전체기간)


def _clip_to_trailing_days(date_str: str, end_date_str: str, days: int) -> str:
    """end_date_str 기준 최근 days일 이전이면 그 컷오프로, 아니면 date_str 그대로(둘 중 더 늦은 날짜)."""
    cutoff = (datetime.strptime(end_date_str, "%Y-%m-%d") - timedelta(days=days)).strftime("%Y-%m-%d")
    return max(date_str, cutoff)

log = logging.getLogger("v2_render_dashboard")

TEMPLATE = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>__TITLE__</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min/plotly.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<style>
  body { background:#0f172a; color:#e2e8f0; }
  .card { background:#1e293b; border:1px solid #334155; border-radius:0.75rem; }
  details > summary { cursor: pointer; list-style: none; }
  details > summary::-webkit-details-marker { display: none; }
  details > summary::before { content: '▸ '; }
  details[open] > summary::before { content: '▾ '; }
  .hint { border-bottom: 1px dotted #64748b; cursor: help; }
</style>
</head>
<body class="p-4 md:p-6">
  <header class="card p-4 mb-4">
    <h1 class="text-xl font-bold text-white mb-1">VM-SPC V2 — 반도체 FDC 관리도의 주식 데이터 이식</h1>
    <p class="text-xs text-slate-400 mb-2">SPC 관리도(CUSUM + Hotelling's T&sup2;)를 주식 데이터에 적용해
      추세전환·상관관계붕괴를 인과적으로 탐지합니다.</p>
    __HEADER_HTML__
    __HIGHLIGHTS_HTML__
    __REPORT_HTML__
    <details class="mt-3 text-xs text-slate-400">
      <summary class="text-cyan-400 font-medium">읽는 법 — Panel 1(예측)과 Panel 2&ndash;4(탐지)는 다른 신호입니다</summary>
      <p class="mt-1 pl-4">Panel 1만 <span class="text-slate-200">사전 예측</span>입니다 — 오늘 breadth로 아직
        안 온 내일(T+1) 방향을 추정합니다. Panel 2&ndash;4는 <span class="text-slate-200">사후 탐지</span>입니다 —
        "내일 어떻게 될지"가 아니라 "이미 진행되던 이탈이 이 시점에 확인됐다"는 뒤늦은 진단이라, 알람 난 날이
        실제 변곡점이 아니라 며칠~몇 주 전 이탈이 누적돼 드러난 시점입니다.</p>
    </details>
  </header>

  __TAB_BUTTONS_HTML__

  <div id="tab-content-vm" class="grid grid-cols-1 lg:grid-cols-2 gap-4">
    <section class="card p-4">
      <h2 class="text-sm font-semibold text-slate-300 mb-1">Panel 1 · VM 컨퓨전 매트릭스
        <span class="text-cyan-400 font-normal">(예측)</span></h2>
      <div id="today-prediction" class="rounded-lg border border-cyan-500/40 bg-cyan-500/10 p-3 mb-3"></div>
      __CM_HTML__
      <p class="text-xs text-slate-400 mt-2 italic">&rarr; __INTERP_CM__</p>
    </section>

    <section class="card p-4">
      <h2 class="text-sm font-semibold text-slate-300 mb-3">Panel 2 · 타겟 ETF 캔들스틱 + 칼만 평활선 + CUSUM 마커
        <span class="text-slate-500 font-normal">(탐지)</span></h2>
      <div id="chart-target" style="height:380px;"></div>
      <p class="text-xs text-slate-400 mt-2 italic">&rarr; __INTERP_TARGET__</p>
    </section>

    <section class="card p-4">
      <h2 class="text-sm font-semibold text-slate-300 mb-3">Panel 3 · Top N Breadth 듀얼 CUSUM 관리도
        <span class="text-slate-500 font-normal">(탐지)</span></h2>
      <div style="height:380px;"><canvas id="chart-breadth-cusum"></canvas></div>
      <p class="text-xs text-slate-400 mt-2 italic">&rarr; __INTERP_BREADTH__</p>
    </section>

    <section class="card p-4">
      <h2 class="text-sm font-semibold text-slate-300 mb-3">Panel 4 · Hotelling's T&sup2; 이상치 스코어
        <span class="text-slate-500 font-normal">(탐지)</span></h2>
      <div style="height:380px;"><canvas id="chart-t2"></canvas></div>
      <p class="text-xs text-slate-400 mt-2 italic">&rarr; __INTERP_T2__</p>
    </section>
  </div>

  <div id="tab-content-pair" class="hidden">
    __PAIR_TAB_CONTENT__
  </div>

  <p class="text-xs text-slate-600 mt-4">V1(5대시그널 스코어카드)의 칼만필터·신호 계산을 그대로 이어받아 실제로 쓰는 도구입니다.
    반도체 FAB 공정관리(FDC)의 SPC 관리도(CUSUM·Hotelling's T&sup2;) 기법에서 착안해 설계했습니다.</p>

  <script id="v2-data" type="application/json">__V2_DATA_JSON__</script>
  <script>
    const DATA = JSON.parse(document.getElementById('v2-data').textContent);

    // ── 탭 전환 (예측모델 ↔ 대표종목 상관관계) ────────────────────────
    let pairRendered = false;
    function showTab(name) {
      document.getElementById('tab-content-vm').classList.toggle('hidden', name !== 'vm');
      document.getElementById('tab-content-pair').classList.toggle('hidden', name !== 'pair');
      const btnVm = document.getElementById('tab-btn-vm'), btnPair = document.getElementById('tab-btn-pair');
      if (btnVm) btnVm.classList.toggle('bg-cyan-600', name === 'vm');
      if (btnVm) btnVm.classList.toggle('bg-slate-700', name !== 'vm');
      if (btnPair) btnPair.classList.toggle('bg-cyan-600', name === 'pair');
      if (btnPair) btnPair.classList.toggle('bg-slate-700', name !== 'pair');
      if (name === 'pair' && !pairRendered) { renderPairTab(); pairRendered = true; }
    }
    document.getElementById('tab-btn-vm')?.addEventListener('click', () => showTab('vm'));
    document.getElementById('tab-btn-pair')?.addEventListener('click', () => showTab('pair'));

    // ── Panel 1: 오늘 기준 실시간 예측 카드 ───────────────────────────
    (function () {
      const br = DATA.breadth_series, cm = DATA.confusion_matrix, p = DATA.params;
      let i = br.dates.length - 1;
      while (i >= 0 && br.breadth[i] === null) i--;
      const el = document.getElementById('today-prediction');
      if (i < 0 || !el) { if (el) el.innerHTML = '<div class="text-xs text-slate-400">breadth 미확정 — 예측 불가</div>'; return; }
      const val = br.breadth[i], date = br.dates[i];
      const predUp = val >= p.breadth_threshold;
      const npv = (cm.tn + cm.fn) ? cm.tn / (cm.tn + cm.fn) : null;
      const hitRate = predUp ? cm.precision : npv;
      const hitStr = hitRate === null ? 'N/A' : `${(hitRate * 100).toFixed(1)}%`;
      el.innerHTML = `
        <div class="text-xs text-slate-400">${date} 기준 breadth ${val.toFixed(3)} (임계 ${p.breadth_threshold})</div>
        <div class="text-lg font-bold mt-1 ${predUp ? 'text-emerald-400' : 'text-slate-300'} hint"
          title="과거 같은 방향(${predUp ? '상승' : '하락'})으로 예측했던 사례들의 실제 적중 비율 — 보정된 모형 확률 아님">
          내일(T+1) 예측: ${predUp ? '상승' : '하락'} &middot; 확률 ${hitStr}
        </div>`;
    })();

    // ── 3개 시계열 패널(Panel 2~4) 날짜축 통일 — 분기(1/4/7/10월) 시작점에만 눈금 ──
    const QUARTER_MONTHS = new Set([1, 4, 7, 10]);
    const MONTH_NAME = { 1: 'Jan', 4: 'Apr', 7: 'Jul', 10: 'Oct' };
    function isQuarterStart(dates, i) {
      const month = +dates[i].slice(5, 7);
      if (!QUARTER_MONTHS.has(month)) return false;
      const prevMonth = i > 0 ? +dates[i - 1].slice(5, 7) : null;
      return prevMonth !== month;
    }
    function quarterLabel(dates, i) {
      return `${MONTH_NAME[+dates[i].slice(5, 7)]} ${dates[i].slice(0, 4)}`;
    }
    function quarterScaleX(dates) {
      return {
        ticks: {
          color: '#94a3b8', autoSkip: false, maxRotation: 0,
          callback: (value, index, ticks) => quarterLabel(dates, ticks[index].value),
        },
        afterBuildTicks: axis => {
          axis.ticks = dates.map((_, i) => i).filter(i => isQuarterStart(dates, i)).map(i => ({ value: i }));
        },
        grid: { color: '#334155' },
      };
    }

    // ── Panel 2: Plotly 캔들스틱 + 칼만 평활선 + CUSUM 마커 ──────────────
    (function () {
      const t = DATA.target_series;
      const idx = new Map(t.dates.map((d, i) => [d, i]));
      const markerY = (dates, arr, mul) => dates.map(d => {
        const i = idx.get(d);
        return i === undefined ? null : arr[i] * mul;
      });
      const candle = {
        x: t.dates, open: t.open, high: t.high, low: t.low, close: t.close,
        type: 'candlestick', name: DATA.basket.target.name, xhoverformat: '%Y/%m/%d',
        increasing: { line: { color: '#10b981' } }, decreasing: { line: { color: '#64748b' } },
      };
      const level = {
        x: t.dates, y: t.kalman_level, type: 'scatter', mode: 'lines', xhoverformat: '%Y/%m/%d',
        name: '칼만 평활선', line: { color: '#06b6d4', width: 1.5 },
      };
      const up = {
        x: t.cusum_alarm_up_dates, y: markerY(t.cusum_alarm_up_dates, t.high, 1.02), xhoverformat: '%Y/%m/%d',
        type: 'scatter', mode: 'markers', name: 'Regime Shift Up',
        marker: { symbol: 'triangle-up', size: 11, color: '#ef4444' },
      };
      const down = {
        x: t.cusum_alarm_down_dates, y: markerY(t.cusum_alarm_down_dates, t.low, 0.98), xhoverformat: '%Y/%m/%d',
        type: 'scatter', mode: 'markers', name: 'Trend Break Down',
        marker: { symbol: 'triangle-down', size: 11, color: '#3b82f6' },
      };
      Plotly.newPlot('chart-target', [candle, level, up, down], {
        paper_bgcolor: '#1e293b', plot_bgcolor: '#1e293b',
        font: { color: '#cbd5e1', size: 11 },
        margin: { t: 10, l: 50, r: 10, b: 30 },
        legend: { orientation: 'h', y: 1.15 },
        xaxis: { gridcolor: '#334155', rangeslider: { visible: false }, dtick: 'M3', tickformat: '%b %Y',
                hoverformat: '%Y/%m/%d' },
        yaxis: { gridcolor: '#334155' },
      }, { responsive: true, displayModeBar: false });
    })();

    // ── Panel 3: Chart.js breadth 듀얼 CUSUM ─────────────────────────
    (function () {
      const b = DATA.breadth_series;
      const h = DATA.params.cusum_h;
      new Chart(document.getElementById('chart-breadth-cusum'), {
        type: 'line',
        data: {
          labels: b.dates,
          datasets: [
            {
              label: 'C+ (상향 누적합)', data: b.cusum_pos, borderColor: '#a855f7',
              pointRadius: 0, borderWidth: 1.5,
              segment: { borderColor: ctx => ctx.p1.parsed.y >= h ? '#ef4444' : '#a855f7' },
            },
            {
              label: 'C− (하향 누적합)', data: b.cusum_neg, borderColor: '#06b6d4',
              pointRadius: 0, borderWidth: 1.5,
              segment: { borderColor: ctx => ctx.p1.parsed.y <= -h ? '#3b82f6' : '#06b6d4' },
            },
            {
              label: `+h (${h}σ)`, data: b.dates.map(() => h), borderColor: '#ef4444',
              borderDash: [6, 4], pointRadius: 0, borderWidth: 1,
            },
            {
              label: `−h (${h}σ)`, data: b.dates.map(() => -h), borderColor: '#3b82f6',
              borderDash: [6, 4], pointRadius: 0, borderWidth: 1,
            },
          ],
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          interaction: { mode: 'index', intersect: false },
          scales: {
            x: quarterScaleX(b.dates),
            y: { ticks: { color: '#94a3b8' }, grid: { color: '#334155' } },
          },
          plugins: { legend: { labels: { color: '#cbd5e1', boxWidth: 12, font: { size: 10 } } } },
        },
      });
    })();

    // ── Panel 4: Chart.js T² 바 차트 + UCL ───────────────────────────
    (function () {
      const s = DATA.t2_series;
      const ucl = s.ucl;
      new Chart(document.getElementById('chart-t2'), {
        data: {
          labels: s.dates,
          datasets: [
            {
              type: 'bar', label: 'T²', data: s.t2,
              backgroundColor: s.t2.map(v => v !== null && v > ucl ? '#ef4444' : '#06b6d4'),
              borderWidth: 0,
            },
            {
              type: 'line', label: `UCL (α=${DATA.params.t2_alpha})`, data: s.dates.map(() => ucl),
              borderColor: '#f59e0b', borderDash: [6, 4], pointRadius: 0, borderWidth: 1.5,
            },
          ],
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          interaction: { mode: 'index', intersect: false },
          scales: {
            x: quarterScaleX(s.dates),
            y: { ticks: { color: '#94a3b8' }, grid: { color: '#334155' } },
          },
          plugins: { legend: { labels: { color: '#cbd5e1', boxWidth: 12, font: { size: 10 } } } },
        },
      });
    })();

    // ── PAIR 탭: 대표종목 A vs ETF-ex-A 가격궤적 + SPC Z-Score 관리도 ──────
    function renderPairTab() {
      const pr = DATA.pair && DATA.pair.result;
      if (!pr) return;
      const s = pr.series;

      // 가격궤적: 단위가 다른 두 시계열(종목가 vs ex-self 지수)을 첫 유효일=100 기준으로 리베이스해 비교
      const firstA = s.price_a.find(v => v !== null);
      const firstB = s.index_exa.find(v => v !== null);
      const rebA = s.price_a.map(v => v === null || !firstA ? null : (v / firstA) * 100);
      const rebB = s.index_exa.map(v => v === null || !firstB ? null : (v / firstB) * 100);
      Plotly.newPlot('pair-chart-price', [
        { x: s.dates, y: rebA, type: 'scatter', mode: 'lines', name: `${pr.representative.name}(리베이스)`,
          line: { color: '#06b6d4', width: 1.5 }, xhoverformat: '%Y/%m/%d' },
        { x: s.dates, y: rebB, type: 'scatter', mode: 'lines', name: 'ETF-ex-A(리베이스)',
          line: { color: '#f59e0b', width: 1.5 }, xhoverformat: '%Y/%m/%d' },
      ], {
        paper_bgcolor: '#1e293b', plot_bgcolor: '#1e293b', font: { color: '#cbd5e1', size: 11 },
        margin: { t: 10, l: 50, r: 10, b: 30 }, legend: { orientation: 'h', y: 1.15 },
        xaxis: { gridcolor: '#334155', dtick: 'M3', tickformat: '%b %Y', hoverformat: '%Y/%m/%d' },
        yaxis: { gridcolor: '#334155', title: '시작일=100' },
      }, { responsive: true, displayModeBar: false });

      // SPC Z-Score 관리도: 히스테리시스로 확정된 상태(state)를 배경 음영으로 표시
      const th = pr.params.spc_z_threshold;
      new Chart(document.getElementById('pair-chart-z'), {
        data: {
          labels: s.dates,
          datasets: [
            {
              type: 'line', label: 'Z-Score', data: s.z, borderWidth: 1.5, pointRadius: 0,
              borderColor: s.state.map(st => st === 'alert_up' ? '#ef4444' : st === 'alert_down' ? '#3b82f6' : '#06b6d4'),
              segment: {
                borderColor: ctx => {
                  const st = s.state[ctx.p1.parsedX];
                  return st === 'alert_up' ? '#ef4444' : st === 'alert_down' ? '#3b82f6' : '#06b6d4';
                },
              },
            },
            { type: 'line', label: `+${th}σ`, data: s.dates.map(() => th), borderColor: '#ef4444',
              borderDash: [6, 4], pointRadius: 0, borderWidth: 1 },
            { type: 'line', label: `-${th}σ`, data: s.dates.map(() => -th), borderColor: '#3b82f6',
              borderDash: [6, 4], pointRadius: 0, borderWidth: 1 },
          ],
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          interaction: { mode: 'index', intersect: false },
          scales: { x: quarterScaleX(s.dates), y: { ticks: { color: '#94a3b8' }, grid: { color: '#334155' } } },
          plugins: { legend: { labels: { color: '#cbd5e1', boxWidth: 12, font: { size: 10 } } } },
        },
      });
    }
  </script>
</body>
</html>
"""


def _cm_html(cm: dict) -> str:
    def pct(x):
        return "N/A" if x is None else f"{x:.1%}"
    return f"""
    <div class="grid grid-cols-2 gap-3">
      <div class="rounded-lg border border-emerald-500/40 bg-emerald-500/10 p-4">
        <div class="text-xs text-slate-400 hint" title="T+1 상승 예측 → 실제 상승">TP &middot; 적중</div>
        <div class="text-3xl font-bold text-emerald-400">{cm['tp']}</div>
      </div>
      <div class="rounded-lg border border-red-500/40 bg-red-500/10 p-4">
        <div class="text-xs text-slate-400 hint" title="T+1 상승 예측 → 실제 하락 (오경보)">FP &middot; 과검</div>
        <div class="text-3xl font-bold text-red-400">{cm['fp']}</div>
      </div>
      <div class="rounded-lg border border-amber-500/40 bg-amber-500/10 p-4">
        <div class="text-xs text-slate-400 hint" title="T+1 하락 예측 → 실제 상승 (신호놓침)">FN &middot; 미검</div>
        <div class="text-3xl font-bold text-amber-400">{cm['fn']}</div>
      </div>
      <div class="rounded-lg border border-slate-500/40 bg-slate-500/10 p-4">
        <div class="text-xs text-slate-400 hint" title="T+1 하락 예측 → 실제 하락">TN &middot; 정상</div>
        <div class="text-3xl font-bold text-slate-300">{cm['tn']}</div>
      </div>
    </div>
    <div class="mt-4 grid grid-cols-3 gap-3 text-center">
      <div><div class="text-xs text-slate-400">Accuracy</div><div class="text-xl font-semibold text-cyan-400">{pct(cm['accuracy'])}</div></div>
      <div><div class="text-xs text-slate-400">Precision</div><div class="text-xl font-semibold text-cyan-400">{pct(cm['precision'])}</div></div>
      <div><div class="text-xs text-slate-400">Recall</div><div class="text-xl font-semibold text-cyan-400">{pct(cm['recall'])}</div></div>
    </div>
    <div class="mt-2 text-xs text-slate-500 hint"
      title="익일 등락폭이 이 값 미만이면 방향이 사실상 동전던지기라 판정 자체를 안 하고 제외합니다.">
      표본 {cm['n_days']}일 &middot; 기저율 {pct(cm.get('base_rate'))} &middot; 데드존 &plusmn;{cm.get('dead_zone', 0):.1%}
      ({cm.get('dead_zone_excluded', 0)}일 제외)
    </div>
    {_significance_html(cm)}
    """


def _significance_html(cm: dict) -> str:
    z = cm.get("z_score")
    if z is None:
        return ""
    sig = cm.get("significant_95")
    color = "emerald" if sig else "slate"
    verdict = "통계적으로 유의함" if sig else "관습적 유의수준(1.96) 미달"
    return f"""
    <div class="mt-3 rounded-lg border border-{color}-500/30 bg-{color}-500/5 p-2 text-xs text-slate-400">
      <span class="font-semibold text-{color}-400">z = {z:.2f}</span> &mdash; {verdict}
      <details class="mt-1"><summary class="text-slate-500">근거 보기</summary>
        <p class="mt-1">정밀도 vs 기저율 이항비율 검정. 이 임계값은 사전에 고정한 값 기준입니다 —
        여러 임계값 중 z가 제일 높은 걸 사후에 골랐다면 다중비교 문제로 이 해석이 성립하지 않습니다.</p>
      </details>
    </div>"""


def _highlights_html(payload: dict) -> str:
    t = payload["target_series"]
    br = payload["breadth_series"]
    t2 = payload["t2_series"]
    target_alarms = len(t["cusum_alarm_up_dates"]) + len(t["cusum_alarm_down_dates"])
    breadth_alarms = len(br["cusum_alarm_up_dates"]) + len(br["cusum_alarm_down_dates"])
    t2_valid = sum(1 for v in t2["t2"] if v is not None)
    t2_breach = len(t2["breach_dates"])
    t2_rate = (t2_breach / t2_valid) if t2_valid else None
    rate_str = "N/A" if t2_rate is None else f"{t2_rate:.1%}"

    def stat(label, value, title):
        return f"""<div class="rounded-lg border border-slate-600/40 bg-slate-700/20 p-3 hint" title="{title}">
          <div class="text-2xl font-bold text-cyan-400">{value}</div>
          <div class="text-xs text-slate-300">{label}</div>
        </div>"""

    return f"""
    <div class="grid grid-cols-1 sm:grid-cols-3 gap-3 mt-3">
      {stat("타겟 추세전환 포착 (Panel 2)", f"{target_alarms}건",
            "타겟 ETF 가격 흐름이 평소 패턴에서 벗어난 시점의 수 (CUSUM k=0.5σ h=4.5σ)")}
      {stat("Breadth 이상탐지 포착 (Panel 3)", f"{breadth_alarms}건",
            "10개 센서 종목 신호 평균(breadth)이 평소 패턴에서 벗어난 시점의 수, 타겟과는 별개로 감지")}
      {stat("T² UCL 돌파 (Panel 4)", f"{t2_breach}건 ({rate_str})",
            f"가격밴드·거래량·매물대 신호들의 상관관계가 무너진 시점의 수, 유효 {t2_valid}일 중 (α=0.01)")}
    </div>"""


def _report_html(payload: dict) -> str:
    """v2_compute_engine.full_report() 텍스트를 그대로 렌더링 — 방향예측/이상탐지 구분·통계근거는
    이미 그 함수가 정직하게 구성해뒀으므로 여기선 개행만 <br>로 바꿔서 보여준다(문구 재가공 안 함)."""
    report = payload.get("daily_report")
    if not report or not report.get("text"):
        return ""
    body = html.escape(report["text"]).replace("\n", "<br>")
    return f"""
    <div class="rounded-lg border border-amber-500/40 bg-amber-500/5 p-3 mt-3 text-xs font-mono
                leading-relaxed text-slate-200">{body}</div>"""


def _tab_buttons_html(payload: dict) -> str:
    if not payload.get("pair"):
        return ""
    return """
    <div class="flex gap-2 mb-4">
      <button id="tab-btn-vm" class="px-3 py-1.5 rounded-lg text-sm font-medium bg-cyan-600 text-white">예측모델 (VM-SPC)</button>
      <button id="tab-btn-pair" class="px-3 py-1.5 rounded-lg text-sm font-medium bg-slate-700 text-slate-300">대표종목 상관관계 (PAIR-SPC)</button>
    </div>"""


def _candidates_table_html(candidates: list, selected_code: str | None = None) -> str:
    """대표종목은 2단계로 정해진다: ① 동시상관 통과(후보군 거르기) → ② 그중 공적분(ADF)이 가장
    강한 종목이 최종 선정(2026-09-24 개정, pair_representativeness.py 모듈 docstring 참고)."""
    rows = []
    for c in candidates:
        exself = f"{c['exself_corr']:.3f}" if c.get("exself_corr") is not None else "N/A"
        passed_exself = "예" if c.get("is_representative") else "—"
        if "adf_p" in c:
            adf_str = f"{c['adf_p']:.4f}"
            coint_str = "예" if c.get("is_cointegrated") else "아니오"
        else:
            adf_str, coint_str = "—", "—"
        final = "&#9733; 최종선정" if c["code"] == selected_code else ""
        rows.append(f"""<tr class="border-b border-slate-700/50">
          <td class="py-1.5 pr-3">{c['name']}({c['code']})</td>
          <td class="py-1.5 pr-3">{c['weight'] * 100:.1f}%</td>
          <td class="py-1.5 pr-3">{exself}</td>
          <td class="py-1.5 pr-3">{passed_exself}</td>
          <td class="py-1.5 pr-3">{adf_str}</td>
          <td class="py-1.5 pr-3">{coint_str}</td>
          <td class="py-1.5 text-amber-400">{final}</td>
        </tr>""")
    return f"""
    <table class="w-full text-sm text-slate-300 mt-2">
      <thead><tr class="text-xs text-slate-500 border-b border-slate-600">
        <th class="text-left py-1.5">종목</th><th class="text-left py-1.5">비중</th>
        <th class="text-left py-1.5">Ex-self 동시상관</th><th class="text-left py-1.5">①동시상관 통과</th>
        <th class="text-left py-1.5">ADF p</th><th class="text-left py-1.5">②공적분</th><th class="text-left py-1.5"></th>
      </tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table>"""


def _trim_pair_series(series: dict) -> dict:
    """가격궤적(price_a/index_exa)과 Z-Score(spread/z)가 워밍업 길이가 달라(z는 spc_window만큼
    롤링계산이 필요) 그래프 시작점이 어긋난다 — z가 처음 유효해지는 날짜부터 모든 필드를 동일하게
    잘라내 두 차트의 시계열을 통일한다(_trim_series/_unified_start_date와 같은 원칙). 추가로
    수집기간이 길어도 그래프는 항상 최근 CHART_DISPLAY_DAYS(1년)만 보여준다."""
    z, dates = series["z"], series["dates"]
    start_idx = next((i for i, v in enumerate(z) if v is not None), 0)
    if dates:
        cutoff = _clip_to_trailing_days(dates[start_idx], dates[-1], CHART_DISPLAY_DAYS)
        cutoff_idx = next((i for i, d in enumerate(dates) if d >= cutoff), len(dates))
        start_idx = max(start_idx, cutoff_idx)
    return {k: v[start_idx:] for k, v in series.items()}


def _pair_tab_content_html(payload: dict) -> str:
    pair = payload.get("pair")
    if not pair:
        return ""
    stage0 = pair.get("stage0")
    result = pair.get("result")

    if result is None:
        corr_min = stage0["params"]["exself_corr_min"] if stage0 else 0.7
        cands_html = _candidates_table_html(stage0["candidates"]) if stage0 else ""
        any_passed_exself = bool(stage0 and any(c.get("is_representative") for c in stage0["candidates"]))
        if any_passed_exself:
            reason = (f"동시상관 기준(r&ge;{corr_min})을 통과한 종목은 있지만, 그중 실제로 "
                      f"공적분(ADF)이 확인된 종목이 없습니다.")
        else:
            reason = f"ex-self 동시상관 기준(r&ge;{corr_min})을 통과한 종목이 없습니다."
        return f"""
        <div class="card p-4">
          <h2 class="text-sm font-semibold text-amber-400 mb-2">Stage 0 &mdash; 섹터 대표종목 실증 선정: 대표종목 없음</h2>
          <p class="text-xs text-slate-400">이 섹터는 {reason}
            <span class="text-slate-200 font-medium">단일 종목으로 이 섹터 ETF를 대변할 수 없다는 것
            자체가 유효한 결론</span>이며, 억지로 다음 단계(공적분&middot;SPC)로 넘기지 않습니다.</p>
          {cands_html}
        </div>"""

    rep, coint = result["representative"], result["cointegration"]
    ll = rep.get("leadlag_corr") or {}
    ll_str = " / ".join(f"{k}={v:.2f}" if v is not None else f"{k}=N/A" for k, v in ll.items())
    hl = coint.get("half_life_days")
    hl_str = f"{hl:.1f}일" if hl is not None else "N/A (평균회귀 없음)"
    state = result.get("current_state", "normal")
    state_label = {"normal": "정상", "alert_up": "상방 이탈", "alert_down": "하방 이탈"}.get(state, state)
    state_color = {"normal": "text-emerald-400", "alert_up": "text-red-400",
                   "alert_down": "text-blue-400"}.get(state, "text-slate-300")

    # 항목별 축약 해석 한 줄 — 원자료 숫자만 봐선 판단하기 어려운 부분을 클릭 없이 바로 보여준다.
    exself_short = "매우 강한 동행" if rep["exself_corr"] >= 0.85 else "강한 동행(대표성 기준 통과)"
    ll_vals = [v for v in ll.values() if v is not None]
    leadlag_short = ("선행성 뚜렷치 않음" if (not ll_vals or max(abs(v) for v in ll_vals) < 0.15)
                     else "약한 선행성 감지")
    coint_short = "장기 안정관계 확인됨" if coint["is_cointegrated"] else "장기 안정관계 미확인"
    if hl is None:
        hl_short = "평균회귀 성질 없음"
    else:
        tier = "빠른 회귀" if hl < 20 else ("보통 속도" if hl < 60 else "느린 회귀")
        hl_short = tier if coint["is_cointegrated"] else f"{tier} (공적분 미확인 — 참고용)"
    state_short = {"normal": "이상 없음", "alert_up": "상방 관리이탈 중",
                   "alert_down": "하방 관리이탈 중"}.get(state, "")

    def stat(label, value, short="", desc=""):
        s = f'<div class="text-[11px] text-cyan-300/70 mt-0.5">{short}</div>' if short else ""
        d = f'''<details class="mt-1"><summary class="text-[10px] text-slate-500 cursor-pointer">설명</summary>
          <p class="text-[10px] text-slate-500 mt-1 leading-tight">{desc}</p></details>''' if desc else ""
        return f"""<div class="rounded-lg border border-slate-600/40 bg-slate-700/20 p-3">
          <div class="text-lg font-bold text-cyan-400">{value}</div>
          <div class="text-xs text-slate-300">{label}</div>
          {s}
          {d}
        </div>"""

    kpis = f"""
    <div class="grid grid-cols-2 md:grid-cols-4 gap-3 mb-3">
      {stat("대표종목", f"{rep['name']} &middot; 비중 {rep['weight'] * 100:.1f}%", "공적분 가장 강한 종목으로 선정",
            "①동시상관&ge;0.7 통과 종목 중 ②실제 공적분(ADF)이 가장 강한(p값이 가장 작은) 종목을 최종 대표로 선정합니다. "
            "'비중'은 이 종목이 타겟 ETF 안에서 차지하는 실제 편입비율입니다. 아래 '후보 비교' 참고.")}
      {stat("Ex-self 동시상관", f"{rep['exself_corr']:.3f}", exself_short,
            "A를 뺀 나머지 섹터(ETF-ex-A)와 A의 동행 정도")}
      {stat("리드-래그(k1/k2/k3)", ll_str, leadlag_short, "A가 나머지 섹터를 며칠 선행하는지(참고용, 대표성 판정엔 미반영)")}
      {stat("공적분(ADF)", f"p={coint['adf_p']:.4f} ({'예' if coint['is_cointegrated'] else '아니오'})", coint_short,
            "A와 ETF-ex-A 스프레드가 평균회귀하는 정상시계열인지(p&lt;0.05면 공적분)")}
      {stat("Half-life", hl_str, hl_short, "스프레드가 이탈폭의 절반만큼 되돌아오는 데 걸리는 거래일 수")}
      {stat("현재 상태", f'<span class="{state_color}">{state_label}</span>', state_short,
            f"연속 {result['params']['hysteresis_days']}거래일 확정 기준(플리커링 방지)")}
    </div>"""

    cands_html = _candidates_table_html(stage0["candidates"], rep["code"]) if stage0 else ""
    return f"""
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
      <section class="card p-4 lg:col-span-2">
        <h2 class="text-sm font-semibold text-slate-300 mb-1">Stage 0 &mdash; 대표종목: {rep['name']}({rep['code']})</h2>
        {kpis}
        <details class="mt-2 text-xs text-slate-400">
          <summary class="text-cyan-400 cursor-pointer">후보 비교 보기 (왜 이 종목이 선정됐는지)</summary>
          {cands_html}
        </details>
      </section>
      <section class="card p-4">
        <h2 class="text-sm font-semibold text-slate-300 mb-3">가격 궤적 &mdash; {rep['name']} vs ETF-ex-A (리베이스 100)</h2>
        <div id="pair-chart-price" style="height:380px;"></div>
      </section>
      <section class="card p-4">
        <h2 class="text-sm font-semibold text-slate-300 mb-3">공적분 스프레드 Z-Score SPC 관리도</h2>
        <div style="height:380px;"><canvas id="pair-chart-z"></canvas></div>
      </section>
    </div>"""


def _first_valid_date(dates: list, values: list) -> str | None:
    for d, v in zip(dates, values):
        if v is not None:
            return d
    return None


def _trim_series(series: dict, start_date: str) -> dict:
    """dates 기준 start_date 이전을 잘라낸다 — 3개 패널(타겟·breadth·T²)의 워밍업 길이가
    서로 달라 생기는 '앞부분만 텅 빈' 구간을 없애고 보이는 구간을 통일하기 위함.

    dates와 길이가 같은 배열(종가·CUSUM값 등)은 인덱스로 자르고, 길이가 다른 날짜 문자열
    리스트(알람 날짜·T² 돌파일)는 값으로 걸러낸다 — 안 그러면 Plotly가 그 마커 때문에
    x축을 다시 앞으로 늘려버린다. 스칼라(n_sensors, ucl 등)는 그대로 둔다.
    """
    dates = series["dates"]
    n = len(dates)
    start_idx = next((i for i, d in enumerate(dates) if d >= start_date), n)
    out = {}
    for k, v in series.items():
        if k == "dates":
            out[k] = v[start_idx:]
        elif isinstance(v, list) and len(v) == n:
            out[k] = v[start_idx:]
        elif isinstance(v, list) and v and isinstance(v[0], str):
            out[k] = [d for d in v if d >= start_date]
        else:
            out[k] = v
    return out


def _unified_start_date(payload: dict) -> str:
    """3개 시계열 패널 각각의 '첫 유효값 날짜' 중 가장 늦은 날짜 — 이 날짜부터 전부 보여준다.
    단, 수집기간이 길어도 그래프는 항상 최근 CHART_DISPLAY_DAYS(1년)만 보여준다(표·통계는 전체기간)."""
    t, br, t2 = payload["target_series"], payload["breadth_series"], payload["t2_series"]
    candidates = [
        _first_valid_date(t["dates"], t.get("kalman_level") or t["close"]),
        _first_valid_date(br["dates"], br["breadth"]),
        _first_valid_date(t2["dates"], t2["t2"]),
    ]
    candidates = [c for c in candidates if c is not None]
    start = max(candidates) if candidates else payload["period"]["start"]
    return _clip_to_trailing_days(start, payload["period"]["end"], CHART_DISPLAY_DAYS)


def _latest_alarm(up_dates: list, down_dates: list) -> tuple[str, str] | None:
    """가장 최근 CUSUM 알람의 (날짜, 방향)."""
    candidates = [(d, "상향") for d in up_dates] + [(d, "하향") for d in down_dates]
    return max(candidates, key=lambda x: x[0]) if candidates else None


def _interp_target(payload: dict) -> str:
    t, name = payload["target_series"], payload["basket"]["target"]["name"]
    closes = [c for c in t["close"] if c is not None]
    pct = closes[-1] / closes[0] - 1 if len(closes) >= 2 else None
    pct_str = f"{pct:+.1%}" if pct is not None else "N/A"
    latest = _latest_alarm(t["cusum_alarm_up_dates"], t["cusum_alarm_down_dates"])
    if latest:
        return f"{name}은 기간 중 {pct_str} 움직였고, 가장 최근 감지된 추세전환은 {latest[0]}({latest[1]})입니다."
    return f"{name}은 기간 중 {pct_str} 움직였고, 이 기간엔 CUSUM 추세전환이 감지되지 않았습니다."


def _interp_breadth(payload: dict) -> str:
    br = payload["breadth_series"]
    n_up, n_down = len(br["cusum_alarm_up_dates"]), len(br["cusum_alarm_down_dates"])
    latest = _latest_alarm(br["cusum_alarm_up_dates"], br["cusum_alarm_down_dates"])
    if latest:
        return (f"센서 바스켓 전체 breadth가 정상 궤도를 벗어난 시점이 상향 {n_up}회&middot;하향 {n_down}회 "
                f"감지됐고, 가장 최근은 {latest[0]}({latest[1]})입니다.")
    return "이 기간엔 센서 바스켓 breadth 전체가 정상 궤도를 벗어난 시점이 감지되지 않았습니다."


def _interp_t2(payload: dict) -> str:
    t2 = payload["t2_series"]
    breach = t2["breach_dates"]
    valid = sum(1 for v in t2["t2"] if v is not None)
    if not breach:
        return "이 기간엔 신호 간 상관관계가 무너진(UCL 돌파) 시점이 감지되지 않았습니다."
    rate = len(breach) / valid if valid else None
    rate_str = f"{rate:.1%}" if rate is not None else "N/A"
    return (f"가격밴드&middot;거래량&middot;매물대 신호 간 정상적인 상관관계가 깨진 시점이 {len(breach)}번"
            f"({rate_str}) 감지됐고, 가장 최근은 {breach[-1]}입니다.")


def _interp_confusion(cm: dict) -> str:
    n_pos = cm["tp"] + cm["fp"]
    if n_pos == 0 or cm["precision"] is None or cm.get("base_rate") is None:
        return "이 기간엔 예측(Breadth&ge;임계)이 한 번도 발동하지 않았습니다."
    diff = cm["precision"] - cm["base_rate"]
    tilt = "높지만" if diff > 0 else "낮아"
    tail = "통계적으로도 유의합니다." if cm.get("significant_95") else "통계적으로 확정된 우위는 아닙니다."
    return (f"예측이 발동한 {n_pos}일 중 {cm['tp']}일 적중(정밀도 {cm['precision']:.1%}) &mdash; "
            f"기저율({cm['base_rate']:.1%})보다 {diff:+.1%}p {tilt} {tail}")


def render(payload: dict) -> str:
    b = payload["basket"]

    start_date = _unified_start_date(payload)
    view = dict(payload)  # 차트·하이라이트·해석문은 통일된 구간(view)만 본다. confusion_matrix는 원본 전체.
    view["target_series"] = _trim_series(payload["target_series"], start_date)
    view["breadth_series"] = _trim_series(payload["breadth_series"], start_date)
    view["t2_series"] = _trim_series(payload["t2_series"], start_date)

    if view.get("pair") and view["pair"].get("result"):
        view["pair"] = dict(view["pair"])
        view["pair"]["result"] = dict(view["pair"]["result"])
        view["pair"]["result"]["series"] = _trim_pair_series(view["pair"]["result"]["series"])

    sensors_html = "".join(
        f'<span class="inline-block bg-slate-700/60 rounded px-2 py-0.5 mr-1 mb-1 text-xs">'
        f'{s["name"]}({s["code"]})</span>' for s in b["sensors"])
    header = f"""
      <div class="text-sm text-slate-400">타겟 ETF</div>
      <div class="text-lg font-semibold text-cyan-400 mb-2">{b['target']['name']} ({b['target']['code']})</div>
      <div class="text-sm text-slate-400 mb-1">센서 바스켓 &middot; {b['name']} ({len(b['sensors'])}종목)</div>
      <div class="mb-2">{sensors_html}</div>
      <div class="text-xs text-slate-500">
        그래프 구간 {start_date}~{payload['period']['end']}
        <span class="hint" title="수집 기간은 {payload['period']['start']}부터지만, 그래프는 항상 최근 1년만 표시합니다
          (3개 패널 중 워밍업이 가장 긴 T²가 유효해지는 시점보다 늦으면 그쪽이 우선). 컨퓨전 매트릭스는 전체 수집
          기간 기준입니다.">(수집 {payload['period']['start']}~)</span>
        &middot; W={payload['params']['t2_window']}일 &middot;
        CUSUM k={payload['params']['cusum_k']}&sigma;/h={payload['params']['cusum_h']}&sigma; &middot;
        Breadth&ge;{payload['params']['breadth_threshold']}
      </div>
    """
    html = TEMPLATE
    html = html.replace("__HEADER_HTML__", header)
    html = html.replace("__HIGHLIGHTS_HTML__", _highlights_html(view))
    html = html.replace("__REPORT_HTML__", _report_html(payload))
    html = html.replace("__TAB_BUTTONS_HTML__", _tab_buttons_html(payload))
    html = html.replace("__PAIR_TAB_CONTENT__", _pair_tab_content_html(payload))
    html = html.replace("__CM_HTML__", _cm_html(payload["confusion_matrix"]))
    html = html.replace("__INTERP_TARGET__", _interp_target(view))
    html = html.replace("__INTERP_BREADTH__", _interp_breadth(view))
    html = html.replace("__INTERP_T2__", _interp_t2(view))
    html = html.replace("__INTERP_CM__", _interp_confusion(payload["confusion_matrix"]))
    html = html.replace("__TITLE__", f"VM/SPC V2 — {b['name']}")
    html = html.replace("__V2_DATA_JSON__", json.dumps(view, ensure_ascii=False))
    return html


def render_basket(basket: dict) -> int:
    """v2_summary_metrics.json → dashboard_v2.html. v2_run.py와 이 파일의 CLI 양쪽에서 재사용."""
    src = results_dir() / basket["name"] / "v2_summary_metrics.json"
    if not src.exists():
        log.error("%s 없음 — v2_compute_engine.py 먼저 실행", src)
        return 1
    payload = json.loads(src.read_text(encoding="utf-8"))

    pair_dir = results_dir() / basket["name"] / "pair"
    stage0_path = pair_dir / "stage0_representativeness.json"
    result_path = pair_dir / "pair_analysis_result.json"
    if stage0_path.exists():
        payload["pair"] = {
            "stage0": json.loads(stage0_path.read_text(encoding="utf-8")),
            "result": json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else None,
        }

    out = results_dir() / basket["name"] / "dashboard_v2.html"
    out.write_text(render(payload), encoding="utf-8")
    log.info("생성 완료: %s", out)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--basket")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    return render_basket(get_basket(args.basket))


if __name__ == "__main__":
    sys.exit(main())
