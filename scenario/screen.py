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
for _p in (_ROOT, _ROOT / "core", _ROOT / "vm_predict", _ROOT / "scenario", _ROOT / "dart_events", _ROOT / "stock_track", _ROOT / "research"):
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
HOT_MARGIN = 0.04    # 돌파일 종가가 상단 +4% 넘게 마감 = 과열 표시(9.67, 방향만 일치 → 약함 근거로 표시만)
WIDE_BOX = 0.204     # 30일 박스 폭(H ÷ L − 1) 상승장 돌파 상위 1/3 경계 — 최근경향(9.85)
RS_CUT = 0.70        # 돌파 목록은 RS 등급 70 이상만(9.62: 상승장 +0.27R vs 나머지 +0.06R, 2021 이후 차이 +0.32R)


# ── 규칙(finder): (bars, 오늘 시나리오 상태) → {"t2", "score", "why", "info"…} 또는 None. score 가 클수록 위 ─────────
WEAK_CLOSE = 0.70    # 복귀 캔들 종가 위치 < 0.70 = "턱걸이 복귀"(9.46, 진입 직후 휩쏘 5~12%p 많음)


def close_pos(bars, i: int) -> float | None:
    h, l, c = (float(bars[k].iloc[i]) for k in ("high", "low", "close"))
    return (c - l) / (h - l) if h > l else None


def failure_entry(bars, st):
    """가짜 이탈: 오늘 하단 위로 복귀 → 내일 시가 진입. 순위 = T² 동반 먼저, 이탈 깊이(9.35 방향 일관)."""
    if st.get("kind") == "failure" and st["status"] == "진입 신호":
        d = 1 - st["stop"] / st["L"]
        c = float(bars["close"].iloc[-1])
        return {"t2": st["t2"], "score": d, "depth": d, "tgt_pct": st["H"] / c - 1, "cpos": close_pos(bars, -1),
                "why": f"{'T² 동반 · ' if st['t2'] else ''}이탈 깊이 {d:.1%} · 손절 {st['stop']:,.0f} · 목표 {st['H']:,.0f}",
                "info": f"종가 {c:,.0f} · 손절 {st['stop']:,.0f} · 목표 {st['H']:,.0f}"}


def failure_wait(bars, st):
    """가짜 이탈: 하단 이탈 중, 복귀하면 진입."""
    if st.get("kind") == "failure" and st["status"] == "복귀 대기":
        d = 1 - st["stop"] / st["L"]
        return {"t2": st["t2"], "score": d, "depth": d, "tgt_pct": st["H"] / st["L"] - 1,
                "why": f"{'T² 동반 · ' if st['t2'] else ''}{st['L']:,.0f} 위로 마감하면 다음 날 매수 · 이탈 깊이 {d:.1%}",
                "info": f"{st['L']:,.0f} 위 마감 시 · 손절 {st['stop']:,.0f} · 목표 {st['H']:,.0f}"}


def breakout_first(bars, st):
    """추세 추종 돌파: 오늘 박스 상단 첫 돌파 마감 → 내일 시가, 손절 오늘 종가 − 1ATR, 50일선 이탈 마감까지(최대 120일) 보유(9.68·9.72)."""
    c, h, l = (bars[k].to_numpy(float) for k in ("close", "high", "low"))
    H, prevH = h[-BOX - 1:-1].max(), h[-BOX - 2:-2].max()
    if not (c[-1] > H and c[-2] <= prevH):
        return None
    tr = np.maximum(h[-14:] - l[-14:], np.maximum(abs(h[-14:] - c[-15:-1]), abs(l[-14:] - c[-15:-1])))
    stop = c[-1] - tr.mean()
    width = H / l[-BOX - 1:-1].min() - 1
    return {"t2": False, "score": float(width > WIDE_BOX) - (c[-1] / H - 1), "margin": c[-1] / H - 1, "width": width, "kind": "breakout", "info": f"종가 {c[-1]:,.0f} · 손절 {stop:,.0f} · 50일선 이탈 시 정리",
            "why": f"상단 {H:,.0f} 첫 돌파 · 손절 {stop:,.0f} · 50일선 이탈까지 보유"}


def retest_entry(bars, st):
    """돌파 리테스트(9.18 규칙): 되밀림 후 중간선 위 마감 → 내일 시가."""
    if st.get("kind") == "retest" and st["status"] == "진입 신호":
        return {"t2": False, "score": float(bars["close"].iloc[-1]) / st["H"] - 1, "kind": "retest",
                "info": f"손절 {st['stop']:,.0f} · 목표 {st['H'] + (st['H'] - st['L']):,.0f}", "why": f"리테스트 확인 · 손절 {st['stop']:,.0f}"}


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
    return {"t2": st["t2"], "score": -gain, "depth": 1 - st["stop"] / st["L"], "tgt_pct": st["H"] / c[-1] - 1, "cpos": close_pos(bars, rec),
            "info": f"{bars.index[rec]:%m/%d} 복귀 · 매수가 {st['entry']:,.0f} → {gain:+.1%} · 손절 {st['stop']:,.0f} · 목표 {st['H']:,.0f}",
            "why": f"{bars.index[rec]:%m/%d} 복귀 · 규칙상 매수가 {st['entry']:,.0f} → 지금 {gain:+.1%} · 손절 {st['stop']:,.0f} · 목표 {st['H']:,.0f}"}


def breakout_recent(bars, st):
    """돌파: 최근 RECENT일 안(오늘 제외) 첫 돌파, 그 뒤 손절(돌파일 종가 − 1ATR) 안 닿고 50일선 위 유지. 순위 = 돌파선에서 덜 올라간 순."""
    c, h, l = (bars[k].to_numpy(float) for k in ("close", "high", "low"))
    n = len(c)
    ma50 = pd.Series(c).rolling(50).mean().to_numpy()
    for t in range(n - 2, n - 2 - RECENT, -1):
        H, prevH = h[t - BOX:t].max(), h[t - BOX - 1:t - 1].max()
        if c[t] > H and c[t - 1] <= prevH:
            tr = np.maximum(h[t - 13:t + 1] - l[t - 13:t + 1],
                            np.maximum(abs(h[t - 13:t + 1] - c[t - 14:t]), abs(l[t - 13:t + 1] - c[t - 14:t])))
            stop = c[t] - tr.mean()
            if l[t + 1:].min() <= stop or (c[t + 1:] < ma50[t + 1:]).any():
                return None
            gain = c[-1] / H - 1
            width = H / l[t - BOX:t].min() - 1
            return {"t2": False, "score": float(width > WIDE_BOX) - gain, "margin": c[t] / H - 1, "width": width, "kind": "breakout",
                    "info": f"{bars.index[t]:%m/%d} 돌파 · 상단 대비 {gain:+.1%} · 손절 {stop:,.0f} · 50일선 이탈 시 정리",
                    "why": f"{bars.index[t]:%m/%d} 첫 돌파 · 상단 {H:,.0f} 대비 지금 {gain:+.1%} · 손절 {stop:,.0f} · 50일선 이탈까지 보유"}
    return None


def rs_raw(bars) -> float | None:
    """오닐식 가중 상대강도 0.4·r63 + 0.2·r126 + 0.2·r189 + 0.2·r252 (9.62). 순위는 scan 에서 후보 종목 전체 안의 백분위."""
    c = bars["close"].to_numpy(float)
    if len(c) < 253:
        return None
    r = lambda k: c[-1] / c[-1 - k] - 1
    return 0.4 * r(63) + 0.2 * r(126) + 0.2 * r(189) + 0.2 * r(252)


def program_z(code: str, bars, b_date) -> float | None:
    """이탈일 프로그램 순매수 비중(÷ 거래대금)의 직전 60일 대비 z(9.79 승률 모형 변수). 자료 없으면 None → 모형 평균값."""
    try:
        from collect_program import load_program
        from validate_program_shock import share_z
        prog = load_program(code)
        if prog is None or prog.empty:
            return None
        z = share_z(bars, prog)[bars.index.get_loc(b_date)]
        return float(z) if np.isfinite(z) else None
    except Exception:
        return None


def not_compressed(bars) -> bool:
    cs = current_state(compression_frame(bars))
    return not (cs and cs["compressed"])


# ── 장세별 전략표 — 검증이력 근거와 함께. 새 전략이 검증되면 여기 한 줄 ─────────────────────────────
PLAYBOOK = {
    "상승장": {"strategy": "돌파", "why": "상승장 RS 70↑ 돌파: 승률 17% · +0.45R · ×1.5, 50일선까지 보유 — 자주 작게 지고 가끔 크게 번다(9.62·9.72), 리테스트 방향 일치(9.38)",
              "lists": [("돌파 완료(오늘 종가) → 내일 시가 매수", breakout_first), ("최근 5일 안 돌파 완료 · 아직 유효 → 지금 가격 매수 가능", breakout_recent),
                        ("리테스트 확인 완료(오늘 종가) → 내일 시가 매수", retest_entry)], "filter": None},
    "횡보·전환": {"strategy": "가짜 이탈", "why": "횡보·전환 가짜 이탈 — 최근경향 +0.58R(9.85)",
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
FULL_STAR, HALF_STAR = 0.06, 0.035   # 별점 = 이탈 깊이: 6% 이상 ★ · 3.5~6% ☆(반 개) · 3.5% 미만 별 없이 "얕은 이탈" 경고 — 9.43·9.45


def winrate_table() -> dict | None:
    """research/build_winrate_table.py 가 만든 묶음 승률표(없으면 None → 예전 순위)."""
    return json.loads(WINRATE.read_text(encoding="utf-8")) if WINRATE.exists() else None


def rate(x: dict, state: str, table: dict | None) -> dict:
    """승률 = 로지스틱 모형(T² 동반·장세·이탈 깊이, 검증이력 9.45), R·평소 승률 = 같은 T²·장세 사건 묶음.
    별점 = 이탈 깊이: ★ 6% 이상 · ☆(반 개) 3.5~6% · 3.5% 미만은 별 없이 경고. 돌파 목록은 깊이가 없어 별 없음."""
    dep = x.get("depth")
    star = "" if dep is None else "★" if dep >= FULL_STAR else "☆" if dep >= HALF_STAR else ""
    out = {"star": star, "win": None, "R": None, "pf": None, "ctrl_win": None, "n_hist": 0}
    if not table:
        return out
    if x.get("kind") in ("breakout", "retest"):
        c = (table.get("breakout_rs") or {}).get(state) if x.get("kind") == "breakout" else None
        c = c or table["breakout"].get(state)
    else:
        q1, q2 = table["tgt_cut"]
        b = "가까움" if x["tgt_pct"] <= q1 else "중간" if x["tgt_pct"] <= q2 else "멂"
        key = f"{'T2' if x['t2'] else '-'}|{state}"
        c = table["failure"].get(f"{key}|{b}")
        if not c or c["n"] < 100:                                  # 표본이 작으면 목표 거리 구분 없이
            c = table["failure_coarse"].get(key)
    if c:
        out.update(win=c["win"], R=c["R"], pf=c.get("pf"), ctrl_win=c.get("ctrl_win"), n_hist=c["n"])
    m = table.get("model")
    if m and dep is not None and x.get("kind") not in ("breakout", "retest"):
        z = (m["coef"][0] + m["coef"][1] * float(x["t2"]) + m["coef"][2] * (state == "횡보·전환") + m["coef"][3] * (state == "하락장")
             + m["coef"][4] * min(max(dep, 0.0), m["depth_cap"])
             + (m["coef"][5] * (x["cpos"] if x.get("cpos") is not None else m.get("cpos_mean", 0.75)) if len(m["coef"]) > 5 else 0.0)
             + (m["coef"][6] * (x["prog_z"] if x.get("prog_z") is not None else m.get("prog_z_mean", 0.0)) if len(m["coef"]) > 6 else 0.0)
             + (m["coef"][7] * (state == "횡보·전환") if len(m["coef"]) > 7 else 0.0))       # 최근경향(9.85): 오늘은 최근 구간
        out["win"] = round(1 / (1 + np.exp(-z)), 3)
        coarse = table["failure_coarse"].get(f"{'T2' if x['t2'] else '-'}|{state}")
        if coarse:
            out.update(R=coarse["R"], pf=coarse.get("pf"), ctrl_win=coarse.get("ctrl_win"), n_hist=coarse["n"])
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
    import market_actions
    acts = market_actions.active()
    skipped, rs, trend_raw = [], {}, {}
    for code, name in uni:
        if acts.get(code, {}).get("level") == "제외":        # 거래소 조치 중(투자경고·단기과열·실질심사 등) — 손절이 규칙대로 안 될 수 있어 제외
            skipped.append(name)
            continue
        try:
            _, bars = rr.load_stock(code, caller if not rr.in_baskets(code) else None)
        except Exception as e:  # 한 종목 실패로 멈추지 않는다
            print(f"건너뜀 {code}: {e}", file=sys.stderr)
            continue
        if len(bars) < 300 or (play["filter"] and not play["filter"](bars)):
            continue
        rs[code] = rs_raw(bars)
        trend_raw[code] = rr.trend_components(bars, code)
        st = rr.active_state(bars)
        for i, (_, fn) in enumerate(play["lists"]):
            x = fn(bars, st)
            if x and x.get("kind") not in ("breakout", "retest") and st.get("b_date") is not None:
                x["prog_z"] = program_z(code, bars, st["b_date"])
            if x:
                found[i].append({"code": code, "name": name, **x})
    rs_pct = pd.Series({k: v for k, v in rs.items() if v is not None}).rank(pct=True)
    pd.DataFrame(trend_raw).T.to_csv(results_dir() / "scenario" / "trend_xs.csv")   # 리스크 시나리오의 추세 강도 순위 기준(9.81·9.85)
    try:                                                     # 실적 성장 표시(9.78, 방향만 일치 → 약함) — DART 재무 없으면 생략
        from growth import features as growth_features, load_tables
        ni_t, rev_t = load_tables()
    except Exception as e:
        print(f"실적 표시 생략: {e}", file=sys.stderr)
        ni_t = rev_t = None
    today = kb.index[-1] + pd.Timedelta(days=1)
    table = winrate_table()
    for xs in found:
        for x in xs:
            x["rs"] = rs_pct.get(x["code"])
            if ni_t is not None and x.get("kind") == "breakout":
                x["growth"] = growth_features(x["code"], today, ni_t, rev_t)
            x.update(rate(x, state, table))
    # 돌파 목록은 RS 70 이상만(9.62) — 리테스트는 검증 대상이 아니라 그대로.
    weak = [sum(1 for x in xs if x.get("kind") == "breakout" and not (x["rs"] or 0) >= RS_CUT) for xs in found]
    found = [[x for x in xs if x.get("kind") != "breakout" or (x["rs"] or 0) >= RS_CUT] for xs in found]
    # 가짜 이탈 목록은 별(이탈 깊이 3.5% 이상)이 있는 종목만 — 얕은 이탈은 승률이 평소와 거의 같아 아예 뺀다(사용자 결정).
    # 돌파 목록은 깊이 개념이 없어 그대로 둔다.
    shallow = [sum(1 for x in xs if x.get("depth") is not None and not x.get("star")) for xs in found]
    found = [[x for x in xs if x.get("depth") is None or x.get("star")] for xs in found]
    key = ((lambda x: (-(x["win"] if x["win"] is not None else -1), -(x["R"] or 0), -x["score"])) if table   # 같은 칸이면 기존 순위(이탈 깊이 등)
           else (lambda x: (not x["t2"], -x["score"])))
    lists = [{"label": lab, "n": len(xs), "shallow_out": k, "weak_rs_out": w, "items": sorted(xs, key=key)[:PICK]}
             for (lab, _), xs, k, w in zip(play["lists"], found, shallow, weak)]
    return {"state": state, "strategy": play["strategy"], "why": play["why"], "lists": lists, "skipped": skipped,
            "date": f"{kb.index[-1]:%Y-%m-%d}", "market": m}


def lead_html(r: dict) -> str:
    """시점별 목록 — 한 줄 = 한 종목: 별 | 이름 + 칩(깊이·T²·경고) | 승률(큰 숫자, 아래 평소). 둘째 줄은 숫자만."""
    def chips(x):
        out = []
        d = x.get("depth")
        if d is not None:
            cls = "good" if x.get("star") == "★" else "mid" if x.get("star") else "bad"
            out.append(f'<i class="c {cls}">{"얕은 이탈 " if cls == "bad" else "이탈 "}{d:.1%}</i>')
        if x.get("t2"):
            out.append('<i class="c t2">T²</i>')
        if x.get("kind") == "breakout" and x.get("rs") is not None:
            out.append(f'<i class="c good">RS {x["rs"] * 100:.0f}</i>')
        if x.get("kind") == "breakout" and x.get("width", 0) > WIDE_BOX:
            out.append('<i class="c good">넓은 박스 · 최근경향</i>')
        g = x.get("growth") or {}
        if g.get("c_rev") == 1:
            out.append(f'<i class="c good">매출 +{g["rev_g"]:.0%}</i>')
        if g.get("a_ni") == 1:
            out.append(f'<i class="c good">이익 +{g["a_g"]:.0%}</i>')
        if x.get("kind") == "breakout" and x.get("margin", 0) > HOT_MARGIN:   # 9.67 약함: 과열 돌파는 손절 멀어 R 낮음
            out.append(f'<i class="c mid">과열 +{x["margin"]:.1%}</i>')
        return "".join(out)

    def row(x):
        if x.get("win") is None:
            win = '<div class="w"></div>'
        else:
            ev = f'<span>{x["R"]:+.2f}R · ×{x["pf"]:.1f}</span>' if x.get("R") is not None and x.get("pf") else ""   # 기대값 · 수익 배수
            base = f'<span>평소 {x["ctrl_win"]:.0%}</span>' if x.get("ctrl_win") is not None else ""
            win = f'<div class="w"><b>{x["win"]:.0%}</b>{ev}{base}</div>'
        return (f'<li><span class="st">{x.get("star") or ""}</span><div class="m"><div class="n"><b>{html.escape(x["name"])}</b>'
                f'{chips(x)}</div><div class="i">{html.escape(x.get("info", ""))}</div></div>{win}</li>')

    def block(ls):
        items = "".join(row(x) for x in ls["items"]) or '<li class="none">해당 없음</li>'
        out = f' · 얕은 이탈 {ls["shallow_out"]}개 제외' if ls.get("shallow_out") else ""
        out += f' · RS 70 미만 {ls["weak_rs_out"]}개 제외' if ls.get("weak_rs_out") else ""
        return (f'<div class="scr"><div class="scr-h">{html.escape(ls["label"])} <span>{ls["n"]}개 중 {len(ls["items"])}{out}</span></div>'
                f'<ul>{items}</ul></div>')

    legend = (f'<details class="scr-n"><summary>표시 읽는 법</summary>'
              f'<p>전략: {html.escape(r["strategy"])} — {html.escape(r["why"])}. 코스피 시총 상위 {TOP_N}, 거래소 조치 중 '
              f'{len(r.get("skipped", []))}종목 제외. 규칙 충족 목록이며 추천이 아닙니다.</p>'
              f'<p>오른쪽 = 승률(큰 숫자) · 기대값 R · ×수익 배수 · 평소 승률. 승률은 T² 동반·장세·이탈 깊이·복귀 캔들·이탈일 프로그램 매도 로지스틱 모형'
              f'(비용 뺀 수익 > 0, 9.45·9.48·9.79), R과 수익 배수(이긴 거래 R 합 ÷ 진 거래 R 합, 1보다 크면 남는 규칙)는 같은 T²·장세 사건 묶음, '
              f'평소 = 같은 T²·장세에서 무작위로 샀을 때. 목록 안 순서 = 승률 높은 순.</p>'
              f'<p>★ 이탈 6% 이상 · ☆ 3.5~6% · 3.5% 미만 얕은 이탈은 승률이 평소와 거의 같아 목록에서 뺌(9.43) · T² = 이탈일 다변량 관리한계 초과'
              f'(검증된 셋업). 복귀 캔들 종가 위치는 승률 모형 안에 반영(9.48), 아직 복귀 전이면 평균값으로 계산.</p>'
              f'<p>RS = 3·6·9·12개월 가중 수익률의 후보 종목 안 백분위. 돌파는 RS 70 이상만 표시(9.62), 승률·R도 그 묶음 기준. 과열 = 돌파일 종가가 상단 +4% 초과 — R이 낮은 쪽(9.67, 약함). '
              f'매출 = 최근 분기 매출 전년 대비 +25% 이상, 이익 = 최근 4분기 순이익 전년 대비 +20% 이상(DART, 9.78 약함 — 두 기간 모두 R이 높은 쪽). 최근경향 뜻은 맨 아래 "근거와 한계".</p></details>')
    return (f'<div class="scr-wrap">{"".join(block(ls) for ls in r["lists"])}{legend}</div>'
            '<style>.scr-wrap{display:grid;grid-template-columns:1fr;gap:8px;margin-bottom:12px}'
            '@media(min-width:900px){.scr-wrap{grid-template-columns:1fr 1fr 1fr}}'
            '.scr{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:8px 10px}'
            '.scr-h{font-weight:600;font-size:0.85rem;margin-bottom:4px}.scr-h span{color:var(--dim);font-weight:400;font-size:0.72rem}'
            '.scr ul{margin:0;padding:0;list-style:none}'
            '.scr li{display:grid;grid-template-columns:1.1em 1fr auto;gap:6px;align-items:center;padding:5px 0;border-top:1px solid var(--line)}'
            '.scr li:first-child{border-top:none}.scr li.none{display:block;color:var(--dim);font-size:0.8rem}'
            '.scr .st{color:var(--warn);font-size:0.95rem;text-align:center}'
            '.scr .n{display:flex;flex-wrap:wrap;align-items:center;gap:4px;font-size:0.88rem}'
            '.scr .i{font-size:0.7rem;color:var(--faint);margin-top:1px}'
            '.scr .w{text-align:right;line-height:1.1}.scr .w b{display:block;font-size:1.05rem;color:var(--acc)}'
            '.scr .w span{display:block;font-size:0.65rem;color:var(--dim)}'
            '.scr .c{font-style:normal;font-size:0.66rem;padding:0 5px;border-radius:999px;border:1px solid var(--line);color:var(--mut)}'
            '.scr .c.good{color:var(--up);border-color:rgba(52,211,153,.5)}.scr .c.mid{color:var(--warn);border-color:rgba(251,191,36,.5)}'
            '.scr .c.bad{color:var(--dn);border-color:rgba(248,113,113,.5)}.scr .c.t2{color:var(--acc);border-color:rgba(103,232,249,.5)}'
            '.scr-n{grid-column:1/-1;font-size:0.72rem;color:var(--faint)}.scr-n summary{cursor:pointer;color:var(--dim)}'
            '.scr-n p{margin:4px 0;line-height:1.5}</style>')


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
