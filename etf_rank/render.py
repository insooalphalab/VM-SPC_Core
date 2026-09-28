"""ETF 순위 점수판 HTML — 인덱스 상단 랭킹 카드(인라인 CSS, CDN 없음)와 ETF 상세의 기간별 한 줄."""
from __future__ import annotations

import html
import json

from v2_config import results_dir

LABEL = {1: "T+1", 5: "T+5", 10: "T+10", 20: "T+20"}
NOTE = ("T+1 신호는 거의 전부 괴리율(시장가가 NAV보다 비싸거나 싼 정도)의 되돌림에서 옵니다 — 다른 SPC 지표의 "
        "ETF 간 순위 상관은 0 근처였습니다. 5·10·20일은 신호가 확인되지 않았습니다.")


def load() -> dict | None:
    path = results_dir() / "etf_rank" / "etf_rank.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _ranked(data: dict, k: int) -> list[dict]:
    """랭킹은 판정에 쓴 테마 대표 ETF 안에서만 — 전체로 세우면 반도체 계열 scan ETF 묶음이 상위를 채운다."""
    return sorted((e for e in data["etfs"] if e["primary"] and e["p"].get(str(k)) is not None),
                  key=lambda e: -e["p"][str(k)])


def _active(data: dict, k: int) -> bool:
    return data["status"].get(str(k), {}).get("status") == "Active"


def index_card_html(data: dict | None) -> str:
    """build_index.py TEMPLATE 용 — 인라인 CSS 클래스(.card 등)만 사용."""
    if not data:
        return ""
    cols = []
    for k in data["horizons"]:
        s = data["status"][str(k)]
        ok = _active(data, k)
        ranked = _ranked(data, k)[:5]
        items = "".join(
            f'<div class="rk-row{"" if ok else " rk-hold"}"><span>{i}. {html.escape(e["name"])}</span>'
            f'<span class="{"c-up" if ok else "c-muted"}">{e["p"][str(k)]:.0%}'
            f'<span class="sub"> 평소 {e["usual"][str(k)]:.0%}</span></span></div>'
            for i, e in enumerate(ranked, 1) if e["usual"].get(str(k)) is not None)
        ic = f'순위상관 {s["ic_mean"]:+.3f}' if s.get("ic_mean") is not None else s.get("reason", "")
        cols.append(f'<div class="rk-col"><div class="rk-head {"c-up" if ok else "c-muted"}">{LABEL[k]} · '
                    f'{"Active" if ok else "HOLD"}</div><div class="sub">{ic}</div>{items}</div>')
    # 화면 정리(2026-09-28): 약한 신호라 인덱스 맨 아래에 접어 둔다.
    return f"""
  <details class="card rk">
    <summary class="rk-title">(참고) ETF 순위 — {html.escape(data['benchmark'])}보다 더 오를 확률 · 약한 신호 ({data['as_of']} 기준)</summary>
    <p class="sub">{NOTE} 순위는 테마 대표 ETF(섹터당 하나)끼리 매겼고, 흐린 칸은 검증을 통과하지 못한 참고치입니다.</p>
    <div class="rk-grid">{''.join(cols)}</div>
  </details>"""


INDEX_CSS = """
  .rk { padding:14px; margin-top:16px; }
  .rk-title { font-weight:600; color:#94a3b8; font-size:0.85rem; cursor:pointer; }
  .rk-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:10px; }
  @media (min-width:1024px) { .rk-grid { grid-template-columns:repeat(4,1fr); } }
  .rk-col { background:#0f172a; border:1px solid #334155; border-radius:10px; padding:10px; }
  .rk-head { font-weight:700; font-size:0.85rem; }
  .rk-row { display:flex; justify-content:space-between; gap:6px; font-size:0.78rem; margin-top:5px; color:#e2e8f0; }
  .rk-hold { opacity:0.45; }
"""


def detail_line_html(data: dict | None, code: str) -> str:
    """ETF 상세 대시보드(예측모델 탭) 용 — Tailwind 클래스."""
    if not data:
        return ""
    etf = next((e for e in data["etfs"] if e["code"] == code), None)
    if etf is None:
        return ""
    rows = []
    for k in data["horizons"]:
        p, u = etf["p"].get(str(k)), etf["usual"].get(str(k))
        ranked = _ranked(data, k)
        rank = next((i for i, e in enumerate(ranked, 1) if e["code"] == code), None)
        ok = _active(data, k)
        pos = f"테마 대표 {len(ranked)}개 중 {rank}위" if rank else "테마 대표 ETF 아님 — 순위 미산정"
        txt = "—" if p is None else (f'{p:.0%} <span class="text-slate-500 text-xs">(평소 {u:.0%}) · {pos}</span>')
        rows.append(f'<div class="flex justify-between text-sm{"" if ok else " opacity-40"}">'
                    f'<span class="text-slate-400">{LABEL[k]} {"<span class=text-emerald-400>Active</span>" if ok else "HOLD"}</span>'
                    f'<span class="{"text-emerald-400" if ok else "text-slate-300"}">{txt}</span></div>')
    # 화면 정리(2026-09-28): 신호가 약해(T+1만, 대부분 괴리율 효과) 예측 카드 아래가 아니라 뒤쪽에 접어 둔다.
    return f"""
      <details class="mt-2 text-xs text-slate-400">
        <summary class="cursor-pointer text-slate-500">(참고) 지수보다 더 오를 확률 · ETF 순위 모델 — 약한 신호</summary>
        <div class="rounded-lg border border-slate-700 p-3 mt-2">
          {''.join(rows)}
          <p class="text-[11px] text-slate-500 mt-2">{NOTE} 판정은 날짜별 ETF 간 순위 상관(점수 vs 실제 초과수익)의 평균이
            블록 부트스트랩 95% 하한에서 0보다 큰지로 합니다(테마 대표 ETF 기준, 검증이력 9.12).</p>
        </div>
      </details>"""
