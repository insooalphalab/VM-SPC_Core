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
# 종목 추적 탭·ETF 순위 줄은 2026-09-29 화면에서 뺐다 — 매일 실행에 없어 결과가 낡고, 둘 다 HOLD/약한 신호(검증이력 9.7·9.12).
# 코드(stock_track/, etf_rank/)와 검증 기록은 그대로 두고, 필요하면 python stock_track/run.py · etf_rank/run.py 로 따로 본다.

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
    <h1 class="text-xl font-bold text-white mb-1">VM-SPC Core</h1>
    __HEADER_HTML__
    __CHAMP_SUMMARY_HTML__
    <details class="mt-3 text-xs text-slate-400">
      <summary class="text-cyan-400 font-medium cursor-pointer">예측모델(Legacy) 세부 지표 · 일일 리포트 · 읽는 법</summary>
      <div class="mt-2">
        __SENSORS_HTML__
        __HIGHLIGHTS_HTML__
        __REPORT_HTML__
        <p class="mt-3 pl-1">Panel 1만 <span class="text-slate-200">사전 예측</span>입니다 — 오늘 breadth로 아직
          안 온 내일(T+1) 방향을 추정합니다. Panel 2&ndash;4는 <span class="text-slate-200">사후 탐지</span>입니다 —
          "내일 어떻게 될지"가 아니라 "이미 진행되던 이탈이 이 시점에 확인됐다"는 뒤늦은 진단이라, 알람 난 날이
          실제 변곡점이 아니라 며칠~몇 주 전 이탈이 누적돼 드러난 시점입니다.</p>
      </div>
    </details>
  </header>

  __TAB_BUTTONS_HTML__

  <div id="tab-content-vm" class="grid grid-cols-1 lg:grid-cols-2 gap-4">
    <section class="card p-4">
      <h2 class="text-sm font-semibold text-slate-300 mb-1">내일(T+1) 예측</h2>
      <div id="today-prediction" class="rounded-lg border border-cyan-500/40 bg-cyan-500/10 p-3 mb-3"></div>
      <p class="text-xs text-slate-400 italic">&rarr; __INTERP_CM__</p>
      <details class="mt-2 text-xs text-slate-400">
        <summary class="text-cyan-400 cursor-pointer">과거 예측이 얼마나 맞았는지 숫자로 보기</summary>
        <div class="mt-2">__CM_HTML__</div>
      </details>
      __RANK_LINE_HTML__
    </section>

    <section class="card p-4">
      <h2 class="text-sm font-semibold text-slate-300 mb-3">가격 흐름 이상 탐지
        <span class="text-slate-500 font-normal hint" title="칼만필터 평활선 + CUSUM 관리도">(탐지)</span></h2>
      <div id="chart-target" style="height:380px;"></div>
      <p class="text-xs text-slate-400 mt-2 italic">&rarr; __INTERP_TARGET__</p>
    </section>

    <section class="card p-4">
      <h2 class="text-sm font-semibold text-slate-300 mb-3">센서 종목 전체 이상 탐지
        <span class="text-slate-500 font-normal hint" title="Top N Breadth 듀얼 CUSUM 관리도">(탐지)</span></h2>
      <div style="height:380px;"><canvas id="chart-breadth-cusum"></canvas></div>
      <p class="text-xs text-slate-400 mt-2 italic">&rarr; __INTERP_BREADTH__</p>
    </section>

    <section class="card p-4">
      <h2 class="text-sm font-semibold text-slate-300 mb-3">신호 간 상관관계 이상 탐지
        <span class="text-slate-500 font-normal hint" title="Hotelling's T&sup2; 이상치 스코어">(탐지)</span></h2>
      <div style="height:380px;"><canvas id="chart-t2"></canvas></div>
      <p class="text-xs text-slate-400 mt-2 italic">&rarr; __INTERP_T2__</p>
    </section>
  </div>

  <div id="tab-content-pair" class="hidden">
    __PAIR_TAB_CONTENT__
  </div>

  <div id="tab-content-champ" class="hidden">
    __CHAMP_TAB_CONTENT__
  </div>


  <p class="text-xs text-slate-600 mt-4">V1(5대시그널 스코어카드)의 칼만필터·신호 계산을 그대로 이어받아 실제로 쓰는 도구입니다.
    반도체 FAB 공정관리(FDC)의 SPC 관리도(CUSUM·Hotelling's T&sup2;) 기법에서 착안해 설계했습니다.</p>

  <script id="v2-data" type="application/json">__V2_DATA_JSON__</script>
  <script>
    const DATA = JSON.parse(document.getElementById('v2-data').textContent);

    // ── 탭 전환 (예측모델 / 대표종목 상관관계 / 챔피언-챌린저) ───────────────
    // 숨겨진 div 안에서는 차트 크기를 못 재므로, 각 탭 차트는 처음 열 때 한 번만 그린다.
    const TABS = ['vm', 'pair', 'champ'];
    const tabRendered = { vm: true };
    const tabRenderers = { pair: () => renderPairTab(), champ: () => renderChampTab() };
    function showTab(name) {
      for (const t of TABS) {
        document.getElementById(`tab-content-${t}`)?.classList.toggle('hidden', name !== t);
        const btn = document.getElementById(`tab-btn-${t}`);
        if (btn) { btn.classList.toggle('bg-cyan-600', name === t); btn.classList.toggle('bg-slate-700', name !== t); }
      }
      if (!tabRendered[name]) { tabRenderers[name]?.(); tabRendered[name] = true; }
    }
    for (const t of TABS) document.getElementById(`tab-btn-${t}`)?.addEventListener('click', () => showTab(t));

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
      const baseRate = predUp ? cm.precision : npv;
      const usual = cm.base_rate === null ? null : (predUp ? cm.base_rate : 1 - cm.base_rate);
      const pct = x => x === null || x === undefined ? 'N/A' : `${(x * 100).toFixed(1)}%`;
      const pp = x => `${x >= 0 ? '+' : ''}${(x * 100).toFixed(1)}%p`;
      // 괴리율 보정(검증 통과 시에만 존재): 이 ETF의 같은 방향 적중률 + 전 ETF 합산 괴리율 효과
      const ex = DATA.etf_extras && DATA.etf_extras.date === date ? DATA.etf_extras : null;
      const pr = ex && ex.pred_up === predUp ? ex.premium : null;
      const hitRate = pr ? pr.hit_rate : baseRate;
      const prLine = pr ? `
        <div class="text-xs text-slate-400 mt-1">괴리율 ${pr.dprt > 0 ? '+' : ''}${pr.dprt}%
          (${pr.dz < 0 ? '평소보다 할인' : '평소보다 프리미엄'}) &middot;
          <span class="${pr.state === '동의' ? 'text-emerald-400' : pr.state === '반대' ? 'text-amber-400' : 'text-slate-400'}">예측과 ${pr.state === '중립' ? '무관(중립)' : pr.state === '동의' ? '같은 방향' : '반대 방향'}</span>
        </div>` : '';
      const basis = `
        <details class="mt-1"><summary class="text-[11px] text-slate-500 cursor-pointer">근거 보기</summary>
          <div class="text-[11px] text-slate-500 mt-1">
            이 ETF를 과거에 ${predUp ? '상승' : '하락'}으로 예측했던 날의 적중률은 ${pct(baseRate)}입니다.
            ${pr ? `여기에 39개 ETF 전체에서 확인된 괴리율 효과(${pr.state} ${pp(pr.adjust)})를 더했습니다.
            ETF 하나만 조건별로 쪼개면 표본이 약 40일이라 우연 변동이 효과보다 커서, 효과 크기는 전체 합산으로만 추정합니다.
            괴리율(시장가-NAV)은 다음 날 0 쪽으로 되돌아가는 경향이 있어 할인이면 상승 예측에, 프리미엄이면 하락 예측에 유리합니다.` : ''}
            '평소 ${pct(usual)}'는 예측과 상관없이 다음 날 ${predUp ? '오른' : '내린'} 날의 비율입니다 — 확률이 이 값보다 얼마나 높은지가 실제 예측 실력입니다.
          </div>
        </details>`;
      const rl = ex && ex.pred_up === predUp && ex.reliability ? ex.reliability : null;
      const rlLine = rl ? `<div class="text-xs mt-1 ${rl.grade === '높음' ? 'text-emerald-400' : rl.grade === '낮음' ? 'text-amber-400' : 'text-slate-400'}">
          예측 신뢰도 ${rl.grade}${rl.reasons.length ? ' — ' + rl.reasons.join(', ') : ''}</div>` : '';
      el.innerHTML = `
        <div class="text-xs text-slate-400">${date} 기준 breadth ${val.toFixed(3)} (임계 ${p.breadth_threshold})</div>
        <div class="text-lg font-bold mt-1 ${predUp ? 'text-emerald-400' : 'text-slate-300'} hint"
          title="과거 같은 조건으로 예측했던 사례들의 실제 적중 비율 — 보정된 모형 확률 아님">
          내일(T+1) 예측: ${predUp ? '상승' : '하락'} &middot; 확률 ${pct(hitRate)}
          <span class="text-sm font-normal text-slate-400">(평소 ${pct(usual)})</span>
        </div>${prLine}${rlLine}${basis}`;
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

    // ── 챔피언-챌린저 탭: fold별 High-Conf Precision + 변수기여도 ───────────
    function renderChampTab() {
      const c = DATA.vm_spc;
      if (!c) return;
      const gate1 = c.config.gate1;
      const pctOrNull = v => v === null || v === undefined ? null : v * 100;
      new Chart(document.getElementById('champ-chart-folds'), {
        data: {
          labels: c.folds.map(f => `F${f.fold} ${f.test_start.slice(2, 7)}~${f.test_end.slice(2, 7)}`),
          datasets: [
            { type: 'bar', label: 'Baseline (Ridge)', data: c.folds.map(f => pctOrNull(f.baseline.high_conf_precision)),
              backgroundColor: '#06b6d4' },
            { type: 'bar', label: 'Challenger (LGBM)', data: c.folds.map(f => pctOrNull(f.challenger.high_conf_precision)),
              backgroundColor: '#a855f7' },
            { type: 'line', label: `Gate 1 (${(gate1 * 100).toFixed(0)}%)`, data: c.folds.map(() => gate1 * 100),
              borderColor: '#f59e0b', borderDash: [6, 4], pointRadius: 0, borderWidth: 1.5 },
          ],
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          interaction: { mode: 'index', intersect: false },
          scales: {
            x: { ticks: { color: '#94a3b8', font: { size: 10 } }, grid: { color: '#334155' } },
            // fold 표본이 작아 0%·100%가 실제로 나온다(실측 확인됨) — 축을 좁게 고정하면 그 값이
            // 잘려서 다른 값과 구분이 안 간다. 0~100% 전체를 항상 보여준다.
            y: { min: 0, max: 100, ticks: { color: '#94a3b8', callback: v => v + '%' }, grid: { color: '#334155' } },
          },
          plugins: {
            legend: { labels: { color: '#cbd5e1', boxWidth: 12, font: { size: 10 } } },
            tooltip: { callbacks: { afterLabel: ctx => {
              const f = c.folds[ctx.dataIndex];
              const m = ctx.datasetIndex === 0 ? f.baseline : ctx.datasetIndex === 1 ? f.challenger : null;
              return m ? `S_core ${m.n_s_core}건` : '';
            } } },
          },
        },
      });

      const labels = c.feature_labels;
      const feats = Object.keys(labels);
      const barOpts = (xTitle) => ({
        indexAxis: 'y', responsive: true, maintainAspectRatio: false,
        scales: {
          x: { ticks: { color: '#94a3b8' }, grid: { color: '#334155' },
               title: { display: true, text: xTitle, color: '#94a3b8', font: { size: 10 } } },
          y: { ticks: { color: '#cbd5e1' }, grid: { display: false } },
        },
        plugins: { legend: { display: false } },
      });
      const coef = feats.map(f => c.contributions.ridge_std_coef[f]);
      new Chart(document.getElementById('champ-chart-ridge'), {
        type: 'bar',
        data: { labels: feats.map(f => labels[f]),
                datasets: [{ data: coef, backgroundColor: coef.map(v => v >= 0 ? '#10b981' : '#ef4444') }] },
        options: barOpts('표준화 계수 (1σ당 로그오즈, +는 상승 쪽)'),
      });
      const g = c.contributions.lgbm_gain_share;
      new Chart(document.getElementById('champ-chart-lgbm'), {
        type: 'bar',
        data: { labels: [...feats.map(f => labels[f]), '종목 ID(통제)'],
                datasets: [{ data: [...feats.map(f => g[f] * 100), g._ticker * 100],
                             backgroundColor: [...feats.map(() => '#a855f7'), '#475569'] }] },
        options: barOpts('split gain 비중 (%)'),
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
    """결과가 있는 탭만 버튼을 만든다. 추가 탭이 하나도 없으면 탭 바 자체를 숨긴다(하위 호환)."""
    tabs = [("vm", "예측모델 (VM-SPC)")]
    if payload.get("pair"):
        tabs.append(("pair", "대표종목 상관관계 (PAIR-SPC)"))
    if payload.get("vm_spc"):
        tabs.append(("champ", "챔피언-챌린저 (참고)"))
    if len(tabs) == 1:
        return ""
    buttons = "".join(
        f'<button id="tab-btn-{key}" class="px-3 py-1.5 rounded-lg text-sm font-medium text-white '
        f'{"bg-cyan-600" if i == 0 else "bg-slate-700"}">{label}</button>'
        for i, (key, label) in enumerate(tabs))
    return f'\n    <div class="flex flex-wrap gap-2 mb-4">{buttons}</div>'


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
          <h2 class="text-sm font-semibold text-amber-400 mb-2">대표종목 없음</h2>
          <p class="text-sm text-slate-300"><span class="text-slate-100 font-medium">이 섹터는 종목 하나로 대변할 수 없습니다</span>
            — 억지로 끼워맞추지 않고 그대로 알립니다.</p>
          <details class="mt-2 text-xs text-slate-400">
            <summary class="text-cyan-400 cursor-pointer">전문용어로 자세히 보기</summary>
            <p class="mt-1">{reason}</p>
            {cands_html}
          </details>
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
        # 쉬운 말(short)을 큰 글씨로 먼저 보여주고, 기술 수치(value)는 작게 아래에 둔다 —
        # 전문용어·원자료는 "설명" 토글 안에서만 필요한 사람이 본다.
        primary = short or value
        sub = f'<div class="text-[11px] text-slate-500 mt-0.5">{value}</div>' if short else ""
        d = f'''<details class="mt-1"><summary class="text-[10px] text-slate-500 cursor-pointer">전문용어로 보기</summary>
          <p class="text-[10px] text-slate-500 mt-1 leading-tight">{desc}</p></details>''' if desc else ""
        return f"""<div class="rounded-lg border border-slate-600/40 bg-slate-700/20 p-3">
          <div class="text-base font-bold text-cyan-300">{primary}</div>
          <div class="text-xs text-slate-400">{label}</div>
          {sub}
          {d}
        </div>"""

    kpis = f"""
    <div class="grid grid-cols-2 md:grid-cols-4 gap-3 mb-3">
      {stat("대표종목", f"{rep['name']} &middot; 비중 {rep['weight'] * 100:.1f}%", "",
            "①동시상관&ge;0.7 통과 종목 중 ②실제 공적분(ADF)이 가장 강한(p값이 가장 작은) 종목을 최종 대표로 선정합니다. "
            "'비중'은 이 종목이 타겟 ETF 안에서 차지하는 실제 편입비율입니다. 아래 '후보 비교' 참고.")}
      {stat("동행 정도", f"Ex-self 동시상관 {rep['exself_corr']:.3f}", exself_short,
            "A를 뺀 나머지 섹터(ETF-ex-A)와 A의 동행 정도")}
      {stat("선행성", f"리드-래그(k1/k2/k3) {ll_str}", leadlag_short, "A가 나머지 섹터를 며칠 선행하는지(참고용, 대표성 판정엔 미반영)")}
      {stat("장기 관계", f"공적분(ADF) p={coint['adf_p']:.4f} ({'예' if coint['is_cointegrated'] else '아니오'})", coint_short,
            "A와 ETF-ex-A 스프레드가 평균회귀하는 정상시계열인지(p&lt;0.05면 공적분)")}
      {stat("되돌림 속도", f"Half-life {hl_str}", hl_short, "스프레드가 이탈폭의 절반만큼 되돌아오는 데 걸리는 거래일 수")}
    </div>"""

    cands_html = _candidates_table_html(stage0["candidates"], rep["code"]) if stage0 else ""
    return f"""
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
      <section class="card p-4 lg:col-span-2">
        <h2 class="text-sm font-semibold text-slate-300 mb-1">지금 상태</h2>
        <div class="rounded-lg border {'border-emerald-500/50 bg-emerald-500/10' if state == 'normal' else 'border-red-500/50 bg-red-500/10' if state == 'alert_up' else 'border-blue-500/50 bg-blue-500/10'} p-3 mb-3">
          <div class="text-lg font-bold {state_color}">{state_label}{f' — {state_short}' if state_short else ''}</div>
          <div class="text-xs text-slate-400 mt-0.5">대표종목 {rep['name']}({rep['code']})이 나머지 섹터와 맺는 관계를 매일 점검합니다
            &middot; 연속 {result['params']['hysteresis_days']}거래일 확정 기준</div>
        </div>
        <h3 class="text-xs font-semibold text-slate-400 mb-2">참고 &mdash; 왜 이 종목을 대표로 골랐나</h3>
        {kpis}
        <details class="mt-2 text-xs text-slate-400">
          <summary class="text-cyan-400 cursor-pointer">후보 비교를 전문용어로 자세히 보기</summary>
          {cands_html}
        </details>
      </section>
      <section class="card p-4">
        <h2 class="text-sm font-semibold text-slate-300 mb-3">가격 궤적 비교
          <span class="text-slate-500 font-normal">({rep['name']} vs 나머지 섹터, 시작일=100)</span></h2>
        <div id="pair-chart-price" style="height:380px;"></div>
      </section>
      <section class="card p-4">
        <h2 class="text-sm font-semibold text-slate-300 mb-3">이탈 추이
          <span class="text-slate-500 font-normal hint" title="공적분 스프레드 Z-Score SPC 관리도">(관리도)</span></h2>
        <div style="height:380px;"><canvas id="pair-chart-z"></canvas></div>
      </section>
    </div>"""


_CHAMP_LABEL = {"BASELINE": "Baseline (Ridge Logistic)", "CHALLENGER": "Challenger (LightGBM)",
                "REJECTED": "최종 기각 — Legacy 룰만 운영"}
_PLAIN_CHAMP_LABEL = {"BASELINE": "지금 기준: 단순 모델", "CHALLENGER": "지금 기준: 복잡한 모델"}
_CHAMP_BANNER_CLS = {"REJECTED": "border-amber-500/50 bg-amber-500/10 text-amber-300",
                     "BASELINE": "border-cyan-500/50 bg-cyan-500/10 text-cyan-300",
                     "CHALLENGER": "border-purple-500/50 bg-purple-500/10 text-purple-300"}


def _pct(v, digits: int = 1) -> str:
    return "—" if v is None else f"{v * 100:.{digits}f}%"


def _label_mode_desc(cfg: dict) -> str:
    """라벨링 모드 설명 한 줄 — vm_spc/labeling.py 의 absolute/relative 옵션에 대응."""
    if cfg.get("label_mode") == "relative" and cfg.get("relative_rank"):
        r = cfg["relative_rank"]
        return (f"상대순위: 그날 바스켓 내 상위 {r['top_frac']:.0%}=아웃퍼폼(1) / 하위 {r['bot_frac']:.0%}=언더퍼폼(0), "
                f"중간은 학습&middot;검증에서 제외(유효종목 {r['min_active']}개 미만인 날도 제외)")
    dz = cfg.get("deadzone") or [0, 0]
    return f"데드존: {dz[0]:+.1%} &lt; R(t+1) &lt; {dz[1]:+.1%} 은 학습&middot;검증에서 제외"


def _gate_chip(ok: bool | None, text: str) -> str:
    if ok is None:
        cls, mark = "border-slate-600 bg-slate-700/30 text-slate-500", "·"
    elif ok:
        cls, mark = "border-emerald-500/50 bg-emerald-500/10 text-emerald-300", "&#10003;"
    else:
        cls, mark = "border-red-500/50 bg-red-500/10 text-red-300", "&#10007;"
    return f'<div class="rounded-lg border {cls} px-3 py-2 text-xs"><span class="font-bold mr-1">{mark}</span>{text}</div>'


def _champ_plain_compare(c: dict) -> str:
    """전문용어 없이 한 문장으로 — "누가 몇 번 중 몇 번 맞았는지"만 말한다. 표 전체(지표 이름들)는
    이 문장 아래 접힌 상세에 그대로 남긴다."""
    m = c["metrics"]
    parts = []
    for key, label in (("baseline", "단순 모델"), ("challenger", "복잡한 모델")):
        hcp, n = m[key]["high_conf_precision"], m[key]["n_s_core"]
        parts.append(f"{label}은 강한 신호를 낸 적이 없습니다" if (hcp is None or not n)
                     else f"{label}은 강한 신호 {n}번 중 {hcp:.0%} 맞았습니다")
    return " · ".join(parts) + "."



def _champ_top_signal(c: dict) -> tuple[dict, float, bool] | None:
    """오늘 산출된 예측 중 가장 확신이 강한 것(|P-0.5| 최대) 하나 — 헤더 미리보기용. 챔피언이
    있으면(p_champion) 검증된 값을, 없으면(REJECTED) Baseline 원값을 미검증 참고치로 대신
    쓴다(빈칸보다 낫다는 2026-09-27 사용자 피드백) — 반환: (종목, 확률, validated)."""
    best, best_gap = None, -1.0
    for s in c["scenario"]:
        p = s.get("p_champion")
        if p is None:
            continue
        gap = abs(p - 0.5)
        if gap > best_gap:
            best, best_gap = s, gap
    if best is not None:
        return best, best["p_champion"], True
    for s in c["scenario"]:
        p = s.get("p_baseline")
        if p is None:
            continue
        gap = abs(p - 0.5)
        if gap > best_gap:
            best, best_gap = s, gap
    return (best, best["p_baseline"], False) if best is not None else None


def _champ_direction_table(c: dict) -> str:
    """메인 콘텐츠 — 종목별 '내일 방향 + 확률'. 이 파이프라인이 애초에 예측하는 건 이것 하나뿐이고
    (변동폭은 모델링 대상이 아님), 어느 모델이 더 정확한지는 참고 정보로 아래에 따로 둔다.

    챔피언이 없어도(REJECTED) 빈칸으로 감추지 않는다 — Baseline·Challenger 는 게이트를 못
    넘었을 뿐 확률 자체는 계산돼 있으므로, Baseline 원값을 미검증 참고치로 흐리게 보여준다
    (빈칸보다 낫다는 2026-09-27 사용자 피드백). 검증된 값과는 색·문구로 명확히 구분한다."""
    validated = c["decision"]["champion"] != "REJECTED"
    prob_key = "p_champion" if validated else "p_baseline"
    is_relative = c["config"].get("label_mode") == "relative"
    caption = ("그날 바스켓 평균 대비 상대적으로 강할지&middot;약할지를 확률로 나타냅니다 "
              "(개별 종목의 절대적인 등락을 보장하는 건 아닙니다)." if is_relative else
              "익일 종가가 전일 대비 오를지&middot;내릴지를 확률로 나타냅니다.")
    if not validated:
        caption += (" <b class='text-amber-300'>지금은 게이트를 통과한 모델이 없어 확정된 예측이 아닙니다</b>"
                    " — 아래는 Baseline(단순 모델)의 미검증 참고치입니다.")
    scored = sorted(((abs(s[prob_key] - 0.5), s) for s in c["scenario"] if s.get(prob_key) is not None),
                    key=lambda t: -t[0])
    if not scored:
        return """<p class="text-sm text-slate-300">오늘 산출된 예측이 없습니다.</p>"""
    rows = []
    for _, s in scored:
        p = s[prob_key]
        up = p >= 0.5
        conf = p if up else (1 - p)
        if validated:
            arrow, label, color = ("&#9650;", "상승 우세", "text-emerald-400") if up else ("&#9660;", "하락 우세", "text-red-400")
        else:
            arrow, label, color = ("&#9650;", "상승", "text-slate-400") if up else ("&#9660;", "하락", "text-slate-400")
        rows.append(f"""<tr class="border-b border-slate-700/50">
          <td class="py-1.5 pr-2">{s['name']}<span class="text-slate-500 text-xs">({s['code']})</span></td>
          <td class="py-1.5 px-2 text-right">{s['close']:,.0f}</td>
          <td class="py-1.5 px-2 text-center {color} font-semibold">{arrow} {label}</td>
          <td class="py-1.5 px-2 text-right {color} font-bold">{conf:.0%}</td>
        </tr>""")
    return f"""
      <p class="text-xs text-slate-400 mb-2">{caption}</p>
      <div class="overflow-x-auto">
      <table class="w-full text-sm text-slate-300">
        <thead><tr class="text-xs text-slate-500 border-b border-slate-600">
          <th class="text-left py-1.5">종목</th><th class="text-right px-2">종가</th>
          <th class="text-center px-2">방향</th><th class="text-right px-2">확률</th>
        </tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
      </div>"""


def _champ_benchmark_table(c: dict) -> str:
    m, cfg = c["metrics"], c["config"]
    champ = c["decision"]["champion"]
    cols = [("legacy", f"Legacy Rule<br><span class='text-slate-500 font-normal'>VM Score&ge;{cfg['legacy_threshold']}</span>"),
            ("baseline", "Baseline<br><span class='text-slate-500 font-normal'>Ridge Logistic</span>"),
            ("challenger", "Challenger<br><span class='text-slate-500 font-normal'>LightGBM</span>")]

    def ci(x):
        lo, hi = x.get("ci_lower"), x.get("ci_upper")
        return "—" if lo is None else f"{_pct(lo)} ~ {_pct(hi)}"

    def num(v, fmt="{:.3f}"):
        return "—" if v is None else fmt.format(v)

    def spy(v):
        return "—" if v is None else f"{v:.0f}회"

    outcome_desc = (f"+{cfg['deadzone'][1]:.1%} 이상 오른 비율" if cfg.get("deadzone")
                    else f"그날 바스켓 내 상위 {cfg['relative_rank']['top_frac']:.0%}(아웃퍼폼)에 실제로 들었던 비율")
    rows = [
        ("High-Conf Precision (S_core)", lambda x: f"<span class='text-base font-bold'>{_pct(x['high_conf_precision'])}</span>", True,
         f"채택 기준 지표(주 지표). P&ge;{cfg['tier2']:.2f} 인 날 중 실제 {outcome_desc}"),
        ("95% CI (날짜 클러스터 부트스트랩)", ci, False, f"B={cfg['n_boot']}, 같은 날 관측치는 통째로 리샘플 — Gate 2 판정용"),
        ("|S_core| (Tier 2 신호 수)", lambda x: f"{x['n_s_core']:,}", False, ""),
        ("S_core 신호 빈도 (연환산, 전 종목 합)", lambda x: spy(x["signals_per_year"]), False, ""),
        ("Precision (S_base)", lambda x: _pct(x["precision"]), False, f"P&gt;{cfg['tier1']:.2f} (Tier 1) 기준, 참고"),
        ("|S_base| (Tier 1 신호 수)", lambda x: f"{x['n_s_base']:,}", False, ""),
        ("F1-Score", lambda x: num(x["f1"]), False, "참고"),
        ("Brier Score", lambda x: num(x["brier"]), False, "0.25 이하 권장, 참고 (Legacy 는 이진판정이라 해당 없음)"),
    ]
    head = "".join(
        f"<th class='text-right py-2 px-2 {'text-amber-300' if (k.upper() == champ) else ''}'>{label}"
        f"{' &#9733;' if k.upper() == champ else ''}</th>" for k, label in cols)
    body = []
    for label, fn, primary, tip in rows:
        cls = "bg-cyan-500/5" if primary else ""
        tip_html = f" <span class='hint text-slate-500' title=\"{tip}\">&#9432;</span>" if tip else ""
        cells = "".join(f"<td class='text-right py-1.5 px-2'>{fn(m[k])}</td>" for k, _ in cols)
        body.append(f"<tr class='border-b border-slate-700/50 {cls}'><td class='py-1.5 pr-2 text-slate-400'>{label}{tip_html}</td>{cells}</tr>")
    return f"""
      <div class="overflow-x-auto">
      <table class="w-full text-sm text-slate-300 min-w-[480px]">
        <thead><tr class="text-xs text-slate-300 border-b border-slate-600"><th class="text-left py-2">지표</th>{head}</tr></thead>
        <tbody>{''.join(body)}</tbody>
      </table>
      </div>
      <p class="text-[11px] text-slate-500 mt-2">다중비교 방지: 채택 판정은 High-Conf Precision 하나로만 하고, 나머지는 참고 지표입니다.
        기저율(상승 비율) {_pct(m['baseline']['base_rate'])}.</p>"""


def _champ_scenario_table(c: dict) -> str:
    champ = c["decision"]["champion"]
    active = champ != "REJECTED"
    tier_badge = {
        "S_core": "<span class='rounded px-1.5 py-0.5 bg-emerald-500/20 text-emerald-300 font-semibold'>Tier 2 핵심</span>",
        "S_base": "<span class='rounded px-1.5 py-0.5 bg-cyan-500/15 text-cyan-300'>Tier 1 방향</span>",
        "none": "<span class='text-slate-500'>신호 없음</span>",
        None: "<span class='text-slate-600'>비활성</span>",
    }
    is_relative = c["config"].get("label_mode") == "relative"
    rows = []
    for s in c["scenario"]:
        legacy = ("<span class='text-emerald-300'>발동</span>" if s["legacy_signal"] else "<span class='text-slate-500'>—</span>")
        vm = "—" if s["vm_score"] is None else f"{s['vm_score']:.3f}"
        pb = "—" if s["p_baseline"] is None else f"{s['p_baseline']:.3f}"
        pc = "—" if s["p_challenger"] is None else f"{s['p_challenger']:.3f}"
        line_up = "—" if s["line_up"] is None else f"{s['line_up']:,.0f}"
        line_down = "—" if s["line_down"] is None else f"{s['line_down']:,.0f}"
        strong = "font-semibold text-slate-100"
        pb_cls = strong if champ == "BASELINE" else "text-slate-500"
        pc_cls = strong if champ == "CHALLENGER" else "text-slate-500"
        rows.append(f"""<tr class="border-b border-slate-700/50">
          <td class="py-1.5 pr-2">{s['name']}<span class="text-slate-500 text-xs">({s['code']})</span></td>
          <td class="py-1.5 px-2 text-right">{s['close']:,.0f}</td>
          <td class="py-1.5 px-2 text-right text-emerald-300/80">{line_up}</td>
          <td class="py-1.5 px-2 text-right text-red-300/80">{line_down}</td>
          <td class="py-1.5 px-2 text-right">{vm}</td>
          <td class="py-1.5 px-2 text-center">{legacy}</td>
          <td class="py-1.5 px-2 text-right {pb_cls}">{pb}</td>
          <td class="py-1.5 px-2 text-right {pc_cls}">{pc}</td>
          <td class="py-1.5 pl-2 text-xs">{tier_badge.get(s['tier'], '')}</td>
        </tr>""")
    note = ("챔피언 모델의 P(상승)로 Tier 를 매깁니다 — 굵은 열이 챔피언입니다(어디까지나 약한 방향성 참고 신호입니다)." if active else
            "게이트 판정이 <span class='text-amber-300'>최종 기각</span>이라 ML 방향성 참고 신호(Tier)는 비활성화했습니다. "
            "Baseline·Challenger 확률은 모니터링용으로만 흐리게 표시하고, 운영 판단은 Legacy 룰 열만 봅니다.")
    line_note = ("상대순위 모드라 고정된 가격 기준선이 없습니다(그날 바스켓 내 상대적 우열로 판정)." if is_relative else
                f"상단/하단 기준선은 종가 &times; (1 &plusmn; 데드존 {c['config']['deadzone'][1]:.1%}) — "
                "익일 종가가 이 선 밖으로 나가야 방향이 맞았다/틀렸다를 판정합니다(사이 구간은 노이즈로 보고 판정하지 않음).")
    return f"""
      <p class="text-xs text-slate-400 mb-2">{note} {line_note}</p>
      <div class="overflow-x-auto">
      <table class="w-full text-sm text-slate-300 min-w-[720px]">
        <thead><tr class="text-xs text-slate-500 border-b border-slate-600">
          <th class="text-left py-1.5">종목</th><th class="text-right px-2">종가</th>
          <th class="text-right px-2">상승 기준선</th><th class="text-right px-2">하락 기준선</th>
          <th class="text-right px-2">VM Score</th><th class="text-center px-2">Legacy</th>
          <th class="text-right px-2">P (Baseline)</th><th class="text-right px-2">P (Challenger)</th>
          <th class="text-left pl-2">시그널 계층</th>
        </tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
      </div>"""


def _champ_summary_html(payload: dict) -> str:
    """헤더에 고정으로 뜨는 요약 — 탭을 안 옮겨도 오늘 가장 강한 방향 예측이 뭔지 바로 보인다.
    "어느 모델이 더 정확한가"는 이 파이프라인이 예측하는 대상이 아니라 참고 정보라서, 작게
    아래에 둔다. 근거 상세(Gate 판정·변수기여도·시나리오)는 '챔피언-챌린저' 탭에 그대로 있다."""
    c = payload.get("vm_spc")
    if not c:
        return ""
    d, m = c["decision"], c["metrics"]
    champ = d["champion"]
    active_key = {"BASELINE": "baseline", "CHALLENGER": "challenger"}.get(champ)  # REJECTED 는 셋 다 미강조

    res = _champ_top_signal(c)
    if res is not None:
        top, p, validated = res
        up = p >= 0.5
        conf = p if up else (1 - p)
        if validated:
            arrow, label, color = ("&#9650;", "상승 우세", "text-emerald-400") if up else ("&#9660;", "하락 우세", "text-red-400")
            headline_label = "오늘의 가장 강한 신호"
            note = ""
        else:
            arrow, label, color = ("&#9650;", "상승", "text-slate-400") if up else ("&#9660;", "하락", "text-slate-400")
            headline_label = "오늘의 신호"
            note = " <span class='text-amber-300 font-normal text-sm'>(게이트 미달·참고용)</span>"
        headline = (f"<span class='{color}'>{arrow} {top['name']} {label}</span> "
                    f"<span class='text-slate-400 font-normal text-base'>&middot; 확률 {conf:.0%}</span>{note}")
    else:
        headline_label, headline = "오늘의 신호", "아직 믿을 만한 방향 예측 없음"

    def tile(key: str, label: str, sub: str) -> str:
        hcp = m[key]["high_conf_precision"]
        val = _pct(hcp) if hcp is not None else "—"
        n = m[key]["n_s_core"]
        active = key == active_key
        ring = " ring-1 ring-offset-1 ring-offset-slate-800 ring-cyan-400" if active else ""
        star = " &#9733;" if active else ""
        return f"""<div class="rounded border border-slate-600/50 bg-slate-700/20 px-2 py-1 text-center{ring}">
          <div class="text-sm font-bold text-slate-100">{val}{star}</div>
          <div class="text-[10px] text-slate-400">{label}<span class="text-slate-500"> &middot; {sub}</span></div>
        </div>"""

    tiles = (tile("legacy", "기존 규칙", "Legacy") + tile("baseline", "단순 모델", "Ridge")
            + tile("challenger", "복잡한 모델", "LightGBM"))

    block = f"""
    <div class="rounded-lg border {_CHAMP_BANNER_CLS[champ]} p-3 mt-3">
      <div class="text-xs text-slate-400">{headline_label}</div>
      <div class="text-xl font-bold mt-0.5">{headline}</div>

      <div class="mt-3 pt-3 border-t border-slate-700/60 flex flex-col md:flex-row md:items-center gap-2 justify-between">
        <div class="text-xs text-slate-400">
          참고 &middot; 예측 신뢰도: <span class="text-slate-200">{html.escape(d.get('plain_reason', ''))}</span>
        </div>
        <div class="grid grid-cols-3 gap-1.5">{tiles}</div>
      </div>
      <details class="mt-1 text-[11px] text-slate-500">
        <summary class="cursor-pointer hover:text-slate-400">전문용어로 자세히 보기</summary>
        <p class="mt-1 pl-2">{html.escape(d['reason'])}</p>
      </details>
      <button onclick="document.getElementById('tab-btn-champ')?.click()"
              class="text-[11px] text-cyan-400 hover:text-cyan-300 mt-1 underline underline-offset-2 bg-transparent border-0 p-0 cursor-pointer">
        전체 종목 방향 예측 보기 &rarr;
      </button>
    </div>"""
    if champ != "REJECTED":
        return block
    # 검증을 통과한 모델이 없으면 첫 화면을 차지하지 않게 한 줄로 접는다(2026-09-28 화면 정리 — 대칭 라벨 수정 후
    # 31개 중 29개가 REJECTED 라, 큰 "오늘의 신호" 배너가 대부분 의미 없는 값을 가장 먼저 보여주고 있었다).
    return f"""
    <details class="mt-3 text-xs text-slate-500">
      <summary class="cursor-pointer hover:text-slate-400">구성종목 방향 신호 — 검증을 통과한 모델 없음 (참고치 펼치기)</summary>
      {block}
    </details>"""


def _champ_tab_content_html(payload: dict) -> str:
    c = payload.get("vm_spc")
    if not c:
        return ""
    d, cfg, smp, m = c["decision"], c["config"], c["sample"], c["metrics"]
    champ = d["champion"]
    banner_cls = _CHAMP_BANNER_CLS[champ]

    g1b, g1c = d["gate1"]["baseline"], d["gate1"]["challenger"]
    g2 = d.get("gate2")
    thr = f"{cfg['gate1'] * 100:.0f}%"
    gate2_text = (f"Gate 2 — Challenger&minus;Baseline {g2['margin'] * 100:+.1f}%p (기준 +{cfg['gate2_margin'] * 100:.0f}%p), "
                  f"CI 하한 {_pct(g2['ci_lower'])} {'&gt;' if g2['ci_ok'] else '&le;'} Baseline {_pct(m['baseline']['high_conf_precision'])}"
                  if g2 else "Gate 2 — 둘 다 Gate 1 을 통과해야 진행 (이번엔 해당 없음)")

    min_n = cfg.get('gate1_min_n_core', 30)
    def gate1_text(label: str, g1: dict) -> str:
        hcp = _pct(g1["high_conf_precision"])
        ci = _pct(g1["ci_lower"])
        n_cls = "" if g1.get("n_ok", True) else " text-red-300"
        ref = _pct(g1.get("ci_ref", cfg.get("gate1_ci_lower_min", 0.5)))
        return (f"Gate 1 · {label} {hcp}(기준 &ge;{thr}) · CI 하한 {ci}(기준 &gt; 평소 {ref}) "
                f"· <span class='{n_cls}'>n={g1['n_s_core']}(기준 &ge;{min_n})</span>")

    branch_desc = f"""
      <details class="mt-3 text-xs text-slate-400">
        <summary class="text-cyan-400 cursor-pointer">판정 근거를 전문용어로 자세히 보기</summary>
        <div class="mt-2 grid grid-cols-1 md:grid-cols-3 gap-2">
          {_gate_chip(g1b["passed"], gate1_text("Baseline", g1b))}
          {_gate_chip(g1c["passed"], gate1_text("Challenger", g1c))}
          {_gate_chip(g2['passed'] if g2 else None, gate2_text)}
        </div>
        <p class="mt-3 pl-1 text-slate-500">기준(Gate 1) 은 점추정만 보지 않습니다 — 표본이 작으면 우연히 높게 나온 값도
          점추정 기준은 넘을 수 있어서, 통계적 신뢰구간(CI) 하한이 평소 비율(이 기간 라벨이 1이었던 비율, 최소 50%)보다
          확실히 높고 표본도 충분해야만 통과로 칩니다.</p>
        <ol class="list-decimal pl-8 mt-1 space-y-0.5 text-slate-500">
          <li>단순·복잡한 모델 둘 다 기준 미달 &rarr; <b>최종 기각</b>, 참고 신호도 비활성화, 기존 규칙만 운영</li>
          <li>단순 모델만 기준 통과 &rarr; <b>단순 모델 채택</b>(참고용 신호) — 복잡한 모델만 통과한 예외 시 복잡한 모델 단독 채택</li>
          <li>둘 다 통과 &rarr; 복잡한 모델이 +3.0%p 이상 뚜렷하게 나을 때만 복잡한 모델, 아니면 더 단순한 쪽(오컴의 면도날)
            — 어느 쪽이든 확실한 신호가 아니라 참고용</li>
        </ol>
      </details>"""

    limits = "".join(f"<li>{html.escape(x)}</li>" for x in c.get("limitations", []))
    tickers = ", ".join(t["name"] for t in c["universe"]["tickers"])
    settings = f"""
      <details class="text-xs text-slate-400">
        <summary class="text-cyan-400 cursor-pointer">검증 방식을 전문용어로 자세히 보기</summary>
        <div class="mt-2 pl-4 space-y-1">
          <div>Walk-Forward: 학습 {cfg['train_days']}일 &rarr; Purge {cfg['purge_days']}일 + Embargo {cfg['embargo_days']}일 &rarr;
            검증 {cfg['test_days']}일, {cfg['test_days']}일씩 슬라이딩 &middot; {len(c['folds'])} folds</div>
          <div>{_label_mode_desc(cfg)} ({smp['n_excluded']:,}건 제외, 레이블 {smp['n_labeled']:,}건)</div>
          <div>LightGBM: max_depth={cfg['lgbm']['max_depth']}, num_leaves={cfg['lgbm']['num_leaves']},
            min_data_in_leaf={cfg['lgbm']['min_data_in_leaf']} &middot; Ridge: L2 로지스틱, 표준화 &middot; 두 모델 모두 종목 ID 더미를 통제변수로 포함</div>
          <div>풀링 패널 유니버스 ({len(c['universe']['tickers'])}종목): {html.escape(tickers)}</div>
          <ul class="list-disc pl-5 text-amber-300/80 mt-1">{limits}</ul>
        </div>
      </details>"""

    tw = c["contributions"]["train_window"]
    return f"""
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
      <section class="card p-4 lg:col-span-2">
        <h2 class="text-sm font-semibold text-slate-300 mb-1">내일 방향 예측 &mdash; {c['as_of_feature_date']} 장마감 기준</h2>
        {_champ_direction_table(c)}
        <details class="mt-3 text-xs text-slate-400">
          <summary class="text-cyan-400 cursor-pointer">확률·기준값을 전문용어로 자세히 보기</summary>
          <div class="mt-2">{_champ_scenario_table(c)}</div>
        </details>
      </section>

      <section class="card p-4 lg:col-span-2">
        <h2 class="text-sm font-semibold text-slate-300 mb-2">참고 &mdash; 이 예측, 얼마나 믿을 만한가
          <span class="text-slate-500 font-normal">({smp['oos_start']}~{smp['oos_end']} 검증 기준)</span></h2>
        <div class="rounded-lg border {banner_cls} p-3">
          <div class="text-lg font-bold">{_PLAIN_CHAMP_LABEL.get(champ, '아직 믿을 만한 신호 없음')}</div>
          <div class="text-xs text-slate-300 mt-0.5">{html.escape(d.get('plain_reason', ''))}</div>
        </div>
        <p class="text-sm text-slate-300 mt-3">{_champ_plain_compare(c)}</p>
        <details class="mt-2 text-xs text-slate-400">
          <summary class="text-cyan-400 cursor-pointer">숫자로 자세히 보기</summary>
          <div class="mt-2">{_champ_benchmark_table(c)}</div>
        </details>
        {branch_desc}
      </section>

      <section class="card p-4">
        <h2 class="text-sm font-semibold text-slate-300 mb-3">검증 구간별 적중률</h2>
        <div style="height:300px;"><canvas id="champ-chart-folds"></canvas></div>
        <p class="text-xs text-slate-400 mt-2 italic">&rarr; 빈 칸은 그 구간에서 강한 신호가 한 번도 안 나온 경우입니다.
          구간마다 표본이 작아 흔들림이 크므로, 최종 판정은 전체 기간을 합친 값으로 합니다.</p>
      </section>

      <section class="card p-4">
        <h2 class="text-sm font-semibold text-slate-300 mb-3">무엇을 근거로 판단했나
          <span class="text-slate-500 font-normal">(최근 학습창 {tw[0]}~{tw[1]})</span></h2>
        <div class="text-xs text-slate-400 mb-1">단순 모델이 중요하게 본 요인</div>
        <div style="height:140px;"><canvas id="champ-chart-ridge"></canvas></div>
        <div class="text-xs text-slate-400 mt-3 mb-1">복잡한 모델이 중요하게 본 요인</div>
        <div style="height:150px;"><canvas id="champ-chart-lgbm"></canvas></div>
      </section>

      <section class="card p-4 lg:col-span-2">
        {settings}
        <p class="text-[11px] text-slate-500 mt-2">자동 주문은 하지 않습니다 — 장 마감 후 한 번 계산해 익일 시나리오를 만들고, 진입 여부는 사람이 판단합니다.</p>
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
      <div class="text-sm text-slate-300">
        <span class="text-cyan-400 font-semibold">{b['target']['name']}</span> ({b['target']['code']})
        <span class="text-slate-500">&middot; 센서 {len(b['sensors'])}종목 &middot; {b['name']}</span>
      </div>"""
    sensors_detail = f"""
      <div class="text-xs text-slate-400 mb-1">센서 바스켓 구성 종목</div>
      <div class="mb-2">{sensors_html}</div>
      <div class="text-xs text-slate-500 mb-2">
        그래프 구간 {start_date}~{payload['period']['end']}
        <span class="hint" title="수집 기간은 {payload['period']['start']}부터지만, 그래프는 항상 최근 1년만 표시합니다
          (3개 패널 중 워밍업이 가장 긴 T²가 유효해지는 시점보다 늦으면 그쪽이 우선). 컨퓨전 매트릭스는 전체 수집
          기간 기준입니다.">(수집 {payload['period']['start']}~)</span>
        &middot; W={payload['params']['t2_window']}일 &middot;
        CUSUM k={payload['params']['cusum_k']}&sigma;/h={payload['params']['cusum_h']}&sigma; &middot;
        Breadth&ge;{payload['params']['breadth_threshold']}
      </div>"""
    html = TEMPLATE
    html = html.replace("__HEADER_HTML__", header)
    html = html.replace("__CHAMP_SUMMARY_HTML__", _champ_summary_html(payload))
    html = html.replace("__SENSORS_HTML__", sensors_detail)
    html = html.replace("__HIGHLIGHTS_HTML__", _highlights_html(view))
    html = html.replace("__REPORT_HTML__", _report_html(payload))
    html = html.replace("__TAB_BUTTONS_HTML__", _tab_buttons_html(payload))
    html = html.replace("__PAIR_TAB_CONTENT__", _pair_tab_content_html(payload))
    html = html.replace("__CHAMP_TAB_CONTENT__", _champ_tab_content_html(payload))
    html = html.replace("__RANK_LINE_HTML__", "")
    html = html.replace("__CM_HTML__", _cm_html(payload["confusion_matrix"]))
    html = html.replace("__INTERP_TARGET__", _interp_target(view))
    html = html.replace("__INTERP_BREADTH__", _interp_breadth(view))
    html = html.replace("__INTERP_T2__", _interp_t2(view))
    html = html.replace("__INTERP_CM__", _interp_confusion(payload["confusion_matrix"]))
    html = html.replace("__TITLE__", f"VM/SPC V2 — {b['name']}")
    html = html.replace("__V2_DATA_JSON__", json.dumps(view, ensure_ascii=False))
    return html


def validated_etf_extras(basket_name: str) -> dict | None:
    """etf_extras.json 중 전 바스켓 풀링 검증을 통과한 항목만 남긴다(v2_etf_extras.py) — 통과 못 한
    항목은 계산은 돼 있어도 화면에 쓰지 않는다. build_index.py 도 이 함수를 쓴다."""
    path = results_dir() / basket_name / "etf_extras.json"
    vpath = results_dir() / "etf_extras_validation.json"
    if not path.exists() or not vpath.exists():
        return None
    extras = json.loads(path.read_text(encoding="utf-8"))
    if not extras:
        return None
    validation = json.loads(vpath.read_text(encoding="utf-8"))
    for key in ("premium", "reliability"):
        if not validation.get(key, {}).get("validated"):
            extras[key] = None
    return extras


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

    champ_path = results_dir() / basket["name"] / "vm_spc" / "vm_spc_dashboard_data.json"
    if champ_path.exists():
        payload["vm_spc"] = json.loads(champ_path.read_text(encoding="utf-8"))

    payload["etf_extras"] = validated_etf_extras(basket["name"])

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
