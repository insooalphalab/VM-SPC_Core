"""오늘의 후보 — 코스피 시총 상위 200종목에서 **현재 장세의 전략표(PLAYBOOK)** 에 맞는 규칙이 오늘 종가로 충족된 종목.

장세(render_risk.market_state: 상승장 / 횡보·전환 / 하락장, 검증이력 9.37~9.40)가 바뀌면 전략표의 해당 줄이 자동으로 쓰인다.
전략을 추가·교체할 때는 finder 함수 하나와 PLAYBOOK 한 줄만 고치면 된다. 목록마다 상위 PICK개를 관심 종목 페이지와
같은 양식으로 results/scenario/screen.html 에 만든다. 추천이 아니라 규칙 충족 목록이다.

  python scenario/screen.py            # 일봉 갱신 포함(첫 실행은 비센서 종목 5년치 수집으로 수 분)
  python scenario/screen.py --no-fetch
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "vm_predict", _ROOT / "scenario", _ROOT / "dart_events", _ROOT / "stock_track"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import html
import io
import json
import sys
import time
import zipfile

import numpy as np
import pandas as pd
import requests

import render_risk as rr
from box_rules import BOX
from v2_compression import compression_frame, current_state
from v2_config import results_dir

MASTER_URL = "https://new.real.download.dws.co.kr/common/master/kospi_code.mst.zip"
MASTER = _ROOT / "state" / "kospi_code_master.txt"
TOP_N, PICK = 200, 5


# ── 규칙(finder): (bars, 오늘 시나리오 상태) → {"t2", "score", "why"} 또는 None. score 가 클수록 위 ─────────
def failure_entry(bars, st):
    """가짜 이탈: 오늘 하단 위로 복귀 → 내일 시가 진입. 순위 = T² 동반 먼저, 이탈 깊이(9.35 방향 일관)."""
    if st.get("kind") == "failure" and st["status"] == "진입 신호":
        d = 1 - st["stop"] / st["L"]
        c = float(bars["close"].iloc[-1])
        return {"t2": st["t2"], "score": d, "tgt_pct": st["H"] / c - 1,
                "why": f"{'T² 동반 · ' if st['t2'] else ''}이탈 깊이 {d:.1%} · 손절 {st['stop']:,.0f} · 목표 {st['H']:,.0f}"}


def failure_wait(bars, st):
    """가짜 이탈: 하단 이탈 중, 복귀하면 진입."""
    if st.get("kind") == "failure" and st["status"] == "복귀 대기":
        d = 1 - st["stop"] / st["L"]
        return {"t2": st["t2"], "score": d, "tgt_pct": st["H"] / st["L"] - 1,
                "why": f"{'T² 동반 · ' if st['t2'] else ''}{st['L']:,.0f} 위로 마감하면 다음 날 매수 · 이탈 깊이 {d:.1%}"}


def breakout_first(bars, st):
    """추세 추종 돌파(9.40): 오늘 박스 상단 첫 돌파 마감 → 내일 시가, 손절 상단 − 1ATR, 20일선 이탈 마감까지 보유."""
    c, h, l = (bars[k].to_numpy(float) for k in ("close", "high", "low"))
    H, prevH = h[-BOX - 1:-1].max(), h[-BOX - 2:-2].max()
    if not (c[-1] > H and c[-2] <= prevH):
        return None
    tr = np.maximum(h[-14:] - l[-14:], np.maximum(abs(h[-14:] - c[-15:-1]), abs(l[-14:] - c[-15:-1])))
    stop = H - tr.mean()
    return {"t2": False, "score": c[-1] / H - 1, "kind": "breakout", "why": f"상단 {H:,.0f} 첫 돌파 · 손절 {stop:,.0f} · 20일선 이탈까지 보유"}


def retest_entry(bars, st):
    """돌파 리테스트(9.18 규칙): 되밀림 후 중간선 위 마감 → 내일 시가."""
    if st.get("kind") == "retest" and st["status"] == "진입 신호":
        return {"t2": False, "score": float(bars["close"].iloc[-1]) / st["H"] - 1, "kind": "retest", "why": f"리테스트 확인 · 손절 {st['stop']:,.0f}"}


RECENT = 5      # 최근 며칠 안에 신호가 났고 아직 살아 있는 거래까지 본다


def failure_recent(bars, st):
    """가짜 이탈: 최근 RECENT일 안에 복귀(진입 신호)했고 아직 손절·목표 전. 순위 = T² 동반, 진입가에서 덜 올라간 순."""
    if st.get("kind") != "failure" or st["status"] != "진행 중":
        return None
    c = bars["close"].to_numpy(float)
    b = int(bars.index.get_loc(st["b_date"]))
    rec = next(k for k in range(b + 1, len(c)) if c[k] > st["L"])
    if len(c) - 1 - rec > RECENT:
        return None
    gain = c[-1] / st["entry"] - 1
    return {"t2": st["t2"], "score": -gain, "tgt_pct": st["H"] / c[-1] - 1,
            "why": f"{bars.index[rec]:%m/%d} 복귀 · 규칙상 매수가 {st['entry']:,.0f} → 지금 {gain:+.1%} · 손절 {st['stop']:,.0f} · 목표 {st['H']:,.0f}"}


def breakout_recent(bars, st):
    """돌파: 최근 RECENT일 안(오늘 제외) 첫 돌파, 그 뒤 손절(상단 − 1ATR) 안 닿고 20일선 위 유지. 순위 = 돌파선에서 덜 올라간 순."""
    c, h, l = (bars[k].to_numpy(float) for k in ("close", "high", "low"))
    n = len(c)
    ma20 = pd.Series(c).rolling(20).mean().to_numpy()
    for t in range(n - 2, n - 2 - RECENT, -1):
        H, prevH = h[t - BOX:t].max(), h[t - BOX - 1:t - 1].max()
        if c[t] > H and c[t - 1] <= prevH:
            tr = np.maximum(h[t - 13:t + 1] - l[t - 13:t + 1],
                            np.maximum(abs(h[t - 13:t + 1] - c[t - 14:t]), abs(l[t - 13:t + 1] - c[t - 14:t])))
            stop = H - tr.mean()
            if l[t + 1:].min() <= stop or (c[t + 1:] < ma20[t + 1:]).any():
                return None
            gain = c[-1] / H - 1
            return {"t2": False, "score": -gain, "kind": "breakout",
                    "why": f"{bars.index[t]:%m/%d} 첫 돌파 · 상단 {H:,.0f} 대비 지금 {gain:+.1%} · 손절 {stop:,.0f} · 20일선 이탈까지 보유"}
    return None


def not_compressed(bars) -> bool:
    cs = current_state(compression_frame(bars))
    return not (cs and cs["compressed"])


# ── 장세별 전략표 — 검증이력 근거와 함께. 새 전략이 검증되면 여기 한 줄 ─────────────────────────────
PLAYBOOK = {
    "상승장": {"strategy": "돌파", "why": "상승장 돌파: 성장주 승률 30% · 이긴 +18~23% / 진 −5~7% (9.40), 리테스트 방향 일치(9.38)",
              "lists": [("돌파 완료(오늘 종가) → 내일 시가 매수", breakout_first), ("최근 5일 안 돌파 완료 · 아직 유효 → 지금 가격 매수 가능", breakout_recent),
                        ("리테스트 확인 완료(오늘 종가) → 내일 시가 매수", retest_entry)], "filter": None},
    "횡보·전환": {"strategy": "가짜 이탈", "why": "횡보·전환 가짜 이탈 +0.5R — 장세 중 최고(9.37·9.38)",
                "lists": [("복귀 완료(오늘 종가) → 내일 시가 매수", failure_entry), ("최근 5일 안 복귀 완료 · 아직 유효 → 지금 가격 매수 가능", failure_recent),
                          ("아직 이탈 중 → 하단 위로 마감하면 다음 날 매수", failure_wait)], "filter": None},
    "하락장": {"strategy": "가짜 이탈 (응축 종목 제외)", "why": "하락장 가짜 이탈 +0.1~0.4R, 응축 종목은 20일 뒤 −2~3%p라 제외(9.39)",
              "lists": [("복귀 완료(오늘 종가) → 내일 시가 매수", failure_entry), ("최근 5일 안 복귀 완료 · 아직 유효 → 지금 가격 매수 가능", failure_recent),
                        ("아직 이탈 중 → 하단 위로 마감하면 다음 날 매수", failure_wait)], "filter": not_compressed},
}


def universe() -> list[tuple[str, str]]:
    """KIS 코스피 마스터(7일마다 새로 받음)에서 보통주 시총 상위 TOP_N (코드, 이름)."""
    if not MASTER.exists() or time.time() - MASTER.stat().st_mtime > 7 * 86400:
        r = requests.get(MASTER_URL, timeout=60)
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        MASTER.write_text(z.read(z.namelist()[0]).decode("cp949"), encoding="utf-8")
    rows = []
    for ln in MASTER.read_text(encoding="utf-8").splitlines():
        p2, code, name = ln[-228:], ln[:9].strip(), ln[21:len(ln) - 228].strip()
        if "ST" not in p2[:3] or not code.isdigit() or code[-1] != "0" or "스팩" in name:
            continue
        try:
            rows.append((int(p2[-15:-6]), code, name))              # 전일 기준 시가총액(억)
        except ValueError:
            continue
    rows.sort(reverse=True)
    return [(c, n) for _, c, n in rows[:TOP_N]]


def seed_from_long_history(codes: list[str]) -> int:
    """검증용으로 받아 둔 data/_long_history 일봉(2015~)이 있으면 data/_scenario 로 복사 — 첫 실행에 5년치를 다시 받지 않게.
    이후엔 load_stock 의 증분 갱신이 최근분만 받는다."""
    import shutil
    from v2_config import LONG_HISTORY, data_dir
    src, dst, n = data_dir() / LONG_HISTORY, data_dir() / rr.OWN_BASKET, 0
    dst.mkdir(parents=True, exist_ok=True)
    for code in codes:
        if not (dst / f"{code}.csv").exists() and (src / f"{code}.csv").exists() and not rr.in_baskets(code):
            shutil.copyfile(src / f"{code}.csv", dst / f"{code}.csv")
            n += 1
    return n


WINRATE = results_dir() / "scenario" / "winrate_table.json"


def winrate_table() -> dict | None:
    """research/build_winrate_table.py 가 만든 묶음 승률표(없으면 None → 예전 순위)."""
    return json.loads(WINRATE.read_text(encoding="utf-8")) if WINRATE.exists() else None


def rate(x: dict, state: str, table: dict | None) -> dict:
    """비슷한 과거 사건 묶음의 승률·R·평소 승률과 별점(★ 정상 수량 = T² 동반이면서 상승장 아님, ☆ 절반)."""
    star = "★" if x["t2"] and state != "상승장" else "☆"
    out = {"star": star, "win": None, "R": None, "ctrl_win": None, "n_hist": 0}
    if not table:
        return out
    if x.get("kind") in ("breakout", "retest"):
        c = table["breakout"].get(state)
    else:
        q1, q2 = table["tgt_cut"]
        b = "가까움" if x["tgt_pct"] <= q1 else "중간" if x["tgt_pct"] <= q2 else "멂"
        key = f"{'T2' if x['t2'] else '-'}|{state}"
        c = table["failure"].get(f"{key}|{b}")
        if not c or c["n"] < 100:                                  # 표본이 작으면 목표 거리 구분 없이
            c = table["failure_coarse"].get(key)
    if c:
        out.update(win=c["win"], R=c["R"], ctrl_win=c.get("ctrl_win"), n_hist=c["n"])
    return out


def scan(fetch: bool) -> dict:
    uni = universe()
    seed_from_long_history([c for c, _ in uni])
    caller = None
    if fetch:
        from kis_client import RateLimitedCaller
        caller = RateLimitedCaller()
    _, kb = rr.load_stock("069500", caller)
    m = rr.market_state(kb)
    state = m["state"] if m else "횡보·전환"
    play = PLAYBOOK[state]
    found = [[] for _ in play["lists"]]
    for code, name in uni:
        try:
            _, bars = rr.load_stock(code, caller if not rr.in_baskets(code) else None)
        except Exception as e:  # 한 종목 실패로 멈추지 않는다
            print(f"건너뜀 {code}: {e}", file=sys.stderr)
            continue
        if len(bars) < 300 or (play["filter"] and not play["filter"](bars)):
            continue
        st = rr.active_state(bars)
        for i, (_, fn) in enumerate(play["lists"]):
            x = fn(bars, st)
            if x:
                found[i].append({"code": code, "name": name, **x})
    table = winrate_table()
    for xs in found:
        for x in xs:
            x.update(rate(x, state, table))
    key = ((lambda x: (-(x["win"] if x["win"] is not None else -1), -(x["R"] or 0), -x["score"])) if table   # 같은 칸이면 기존 순위(이탈 깊이 등)
           else (lambda x: (not x["t2"], -x["score"])))
    lists = [{"label": lab, "n": len(xs), "items": sorted(xs, key=key)[:PICK]} for (lab, _), xs in zip(play["lists"], found)]
    return {"state": state, "strategy": play["strategy"], "why": play["why"], "lists": lists,
            "date": f"{kb.index[-1]:%Y-%m-%d}", "market": m}


def lead_html(r: dict) -> str:
    def block(ls):
        def wr(x):
            if x.get("win") is None:
                return ""
            base = f" (평소 {x['ctrl_win']:.0%})" if x.get("ctrl_win") is not None else ""
            return f'<span class="wr">승률 {x["win"]:.0%}{base} · {x["R"]:+.2f}R</span> '
        items = "".join(f'<li><span class="star">{x.get("star", "☆")}</span> <b>{html.escape(x["name"])}</b> {wr(x)}'
                        f'<span class="sub">{x["code"]} · {html.escape(x["why"])}</span></li>'
                        for x in ls["items"]) or '<li class="sub">해당 없음</li>'
        return (f'<div class="scr"><div class="scr-h">{html.escape(ls["label"])} <span class="sub">{ls["n"]}개 중 상위 '
                f'{len(ls["items"])}</span></div><ul>{items}</ul></div>')
    return (f'<div class="scr-wrap">{"".join(block(ls) for ls in r["lists"])}'
            f'<p class="scr-n">전략: {html.escape(r["strategy"])} — {html.escape(r["why"])} · 코스피 시총 상위 {TOP_N} · '
            f'규칙 충족 목록(추천 아님) · 목록 안 순서 = 승률 높은 순 · 승률 = T² 동반 여부·장세·목표 거리가 같은 과거 사건 묶음 '
            f'(비용 뺀 수익 > 0, 괄호 = 같은 손절·목표 무작위 진입) · ★ 정상 수량(T² 동반, 상승장 아님) · ☆ 절반(운영 규칙, 검증값 아님)</p></div>'
            '<style>.scr-wrap{display:grid;grid-template-columns:1fr;gap:8px;margin-bottom:12px}'
            '@media(min-width:900px){.scr-wrap{grid-template-columns:1fr 1fr 1fr}}'
            '.scr{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:10px 12px;font-size:0.85rem}'
            '.scr-h{font-weight:600;margin-bottom:4px}.scr ul{margin:0;padding-left:4px;list-style:none;line-height:1.7}.scr .star{color:var(--warn)}.scr .wr{color:var(--acc);font-size:0.8rem}'
            '.scr-n{grid-column:1/-1;margin:0;font-size:0.7rem;color:var(--faint);line-height:1.5}</style>')


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    fetch = "--no-fetch" not in sys.argv
    r = scan(fetch)
    out = {k: v for k, v in r.items() if k != "market"}
    (results_dir() / "scenario").mkdir(parents=True, exist_ok=True)
    (results_dir() / "scenario" / "screen.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    codes = list(dict.fromkeys(x["code"] for ls in r["lists"] for x in ls["items"]))
    for ls in r["lists"]:
        print(ls["label"], ls["n"], [x["name"] for x in ls["items"]])
    if not codes:
        print("오늘 규칙 충족 종목 없음")
        return 0
    print(rr.build(codes, fetch=fetch, holdings={}, out_name="screen.html", title=f"오늘의 후보 — {r['strategy']}",
                   lead=lead_html(r), summary_name="screen_summary.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
