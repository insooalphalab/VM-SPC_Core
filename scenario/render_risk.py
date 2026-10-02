"""박스권 손절·수량 계산 페이지 — 방향 예측이 아니라 "들어간다면 어디서 끊고 얼마나 살지"를 정하는 도구.

과거 우위는 확인되지 않았다(검증이력 9.18: 돌파 리테스트 종목 +0.73%p·ETF 재현 실패, 가짜 이탈 HOLD, 종목별 차이는 잡음).
그래서 승률·기댓값은 보여 주지 않고, 전 종목 합산 "이 구조로 들어가면 보통 어떻게 끝났나"만 참고로 둔다.

  python scenario/render_risk.py                      → results/scenario/risk_scenarios.html
                                                        (대상 = 루트의 scenario_targets.json, 없으면 예시 파일)
  python scenario/render_risk.py 005930 000660        # 이번만 종목 지정 (--no-fetch: API 호출 없이)
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "vm_predict", _ROOT / "pair_spc", _ROOT / "scenario", _ROOT / "dart_events",
           _ROOT / "stock_track", _ROOT / "reports"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import html
import json
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from v2_compression import compression_frame, current_state
from v2_config import KST, Params, data_dir, load_baskets, results_dir
from v2_cusum import cusum
from v2_signals import kalman_features
from v2_datastore import load_bars
from box_rules import BOX, FAIL_RECOVER, MAX_HOLD, RETEST_WITHIN, t2_flags

TARGETS = _ROOT / "scenario_targets.json"
TARGETS_EXAMPLE = _ROOT / "scenario_targets.example.json"
OWN_BASKET = "_scenario"          # 바스켓에 없는 종목의 일봉: data/_scenario/{code}.csv
OWN_HISTORY_DAYS = 1825
ACTIVE = ("진입 신호", "진행 중")   # 규칙 충족 상태(화면 배지는 "규칙 충족")
CHART_DAYS = 90
DEFAULT_BUDGET = 500_000
BAND = 2.0                 # 관리선 = 칼만 1스텝 예측 ± 2σ
EVENT_NAMES = {"buyback": "자기주식 취득", "rights": "유상증자", "cb": "전환사채 발행"}
Q_DUE = {1: ("0515", 0), 2: ("0814", 0), 3: ("1114", 0), 4: ("0331", 1)}   # 분기 → 정기보고서 법정 기한(월일, 연도+)


# ── 데이터 ─────────────────────────────────────────────
def target_codes() -> list[str]:
    """scenario_targets.json 의 stocks — 문자열("005930") 또는 {"code": ..., "memo": ...} 둘 다 받는다."""
    path = TARGETS if TARGETS.exists() else TARGETS_EXAMPLE
    items = json.loads(path.read_text(encoding="utf-8")).get("stocks", [])
    codes = [str(x["code"] if isinstance(x, dict) else x).strip().zfill(6) for x in items]
    return list(dict.fromkeys(c for c in codes if c.isdigit()))


def target_holdings() -> dict[str, dict]:
    """보유 종목 {코드: {"buy_date": 매수일 또는 None}} — scenario_targets.json 에 "hold": true(예전 avg_price 형식도 보유로 인식).
    평단은 판단에 쓰지 않는다(사용자 결정, 9.86): 일반 보유 = 장세 × 추세 강도, 매수일이 있으면 그날 셋업 규칙을 추적."""
    path = TARGETS if TARGETS.exists() else TARGETS_EXAMPLE
    out = {}
    for x in json.loads(path.read_text(encoding="utf-8")).get("stocks", []):
        if isinstance(x, dict) and (x.get("hold") or x.get("avg_price")):
            bd = x.get("buy_date")
            out[str(x["code"]).strip().zfill(6)] = {"buy_date": pd.Timestamp(bd) if bd else None}
    return out


def apply_basis(pl: list[dict], close: float, hold: dict | None) -> None:
    """계산 기준 가격을 붙인다 — 규칙 충족(진입 신호·진행 중) 시나리오는 보유 중이면 평균 단가, 미보유면 현재가(지금 들어간다면).
    대기 시나리오는 보유 여부와 상관없이 조건 충족 시 예상 진입가(아직 오지 않은 가상의 진입이라 보유 가격과 무관).
    규칙상 진입가는 rule_entry 로 따로 남긴다."""
    for p in pl:
        p["rule_entry"] = p["entry"]
        if hold and hold.get("avg_price") and p["status"] in ACTIVE:
            p["ref"], p["mode"] = hold["avg_price"], "hold"
        elif p["status"] in ACTIVE:
            p["ref"], p["mode"] = close, "now"
        else:
            p["ref"], p["mode"] = p["entry"], "plan"


def load_stock(code: str, caller=None) -> tuple[str, pd.DataFrame]:
    """바스켓 센서면 그 일봉을 쓰고, 아니면 data/_scenario/ 에 받아 둔 일봉(증분 갱신)을 쓴다."""
    for b in load_baskets():
        for s in b["sensors"]:
            if s["code"] == code:
                bars = load_bars(b["name"], code)
                if bars is not None:
                    return s["name"], bars
    names = _own_names()
    if caller is not None:
        from v2_data_collector import update_bars
        from kis_client import fetch_stock_name
        update_bars(caller, OWN_BASKET, code, OWN_HISTORY_DAYS)
        if code not in names:
            names[code] = fetch_stock_name(caller, code)
            (_own_dir() / "names.json").write_text(json.dumps(names, ensure_ascii=False, indent=1), encoding="utf-8")
    bars = load_bars(OWN_BASKET, code)
    if bars is None:
        raise KeyError(f"{code}: 일봉 없음 — API 호출을 허용해 다시 실행하세요")
    now = datetime.now(KST)
    if bars.index[-1].date() == now.date() and (now.hour, now.minute) < (15, 40):
        bars = bars.iloc[:-1]            # 장중에 받은 오늘 봉은 미완성 — 어제 종가 기준으로 맞춘다
    return names.get(code, code), bars


def _own_dir():
    from v2_config import data_dir
    d = data_dir() / OWN_BASKET
    d.mkdir(parents=True, exist_ok=True)
    return d


def _own_names() -> dict:
    p = _own_dir() / "names.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def in_baskets(code: str) -> bool:
    return any(s["code"] == code for b in load_baskets() for s in b["sensors"])


def box_levels(bars: pd.DataFrame, end: int) -> tuple[float, float, float]:
    """end 포함 직전 BOX 거래일의 고가·저가·중간선."""
    w = bars.iloc[end - BOX + 1:end + 1]
    h, l = float(w["high"].max()), float(w["low"].min())
    return h, l, (h + l) / 2


def active_state(bars: pd.DataFrame) -> dict:
    """오늘 기준으로 진행 중인 시나리오(검증과 같은 규칙). 없으면 kind=None."""
    o, h = bars["open"].to_numpy(float), bars["high"].to_numpy(float)
    c, l = bars["close"].to_numpy(float), bars["low"].to_numpy(float)
    n = len(c)
    H = bars["high"].rolling(BOX).max().shift(1).to_numpy()
    L = bars["low"].rolling(BOX).min().shift(1).to_numpy()
    t2 = t2_flags(bars)
    # 가짜 이탈: 최근 FAIL_RECOVER일 안에 L 아래 마감
    for b in range(n - 1, n - 1 - FAIL_RECOVER - 1, -1):
        if c[b] < L[b]:
            rec = next((k for k in range(b + 1, n) if c[k] > L[b]), None)
            stop = float(l[b:(rec if rec is not None else n - 1) + 1].min())
            base = {"kind": "failure", "b_date": bars.index[b], "H": H[b], "L": L[b], "M": (H[b] + L[b]) / 2, "stop": stop,
                    "t2": bool(t2[b])}            # 이탈일 T² 초과 = 검증된 셋업 조건(9.27)
            if rec == n - 1:
                return {**base, "entry": c[-1], "status": "진입 신호",
                        "detail": "오늘 종가로 하단 위 복귀 완료 → 내일 시가 매수"}
            if rec is not None and rec + 1 < n:
                live = _open_trade(o, h, l, c, rec + 1, stop, H[b])
                if live:
                    return {**base, "entry": o[rec + 1], "fixed_entry": True, "status": "진행 중",
                            "detail": f"{bars.index[rec]:%m/%d} 복귀 완료 · {bars.index[rec + 1]:%m/%d} 시가가 규칙상 매수가 · {live}"}
            if rec is None and n - 1 - b < FAIL_RECOVER:
                return {**base, "status": "복귀 대기",
                        "detail": f"{bars.index[b]:%m/%d} 하단 이탈 · {FAIL_RECOVER - (n - 1 - b)}일 안에 {L[b]:,.0f} 위로 마감하면 다음 날 시가 매수"}
            break
    # 돌파 리테스트: 최근 RETEST_WITHIN일 안에 H 위 마감
    for b in range(n - 1, n - 1 - RETEST_WITHIN - 1, -1):
        if b < 0 or np.isnan(H[b]) or c[b] <= H[b]:
            continue
        hb, lb = H[b], L[b]
        mb = (hb + lb) / 2
        base = {"kind": "retest", "b_date": bars.index[b], "H": hb, "L": lb, "M": mb, "stop": mb}
        for k in range(b + 1, n):
            if c[k] < mb:
                return {"kind": None}
            if l[k] <= hb:
                if k == n - 1:
                    return {**base, "entry": c[-1], "status": "진입 신호",
                            "detail": "되밀림 후 중간선 위 마감 완료 → 내일 시가 매수"}
                live = _open_trade(o, h, l, c, k + 1, mb, hb + (hb - lb))
                if not live:
                    return {"kind": None}
                return {**base, "entry": o[k + 1], "fixed_entry": True, "status": "진행 중",
                        "detail": f"{bars.index[k]:%m/%d} 리테스트 확인 · {bars.index[k + 1]:%m/%d} 시가가 규칙상 매수가 · {live}"}
        if n - 1 - b < RETEST_WITHIN:
            return {**base, "status": "리테스트 대기",
                    "detail": f"{bars.index[b]:%m/%d} 상단 돌파 완료 · {RETEST_WITHIN - (n - 1 - b)}일 안에 {hb:,.0f}까지 되밀리고 종가가 {mb:,.0f} 위면 다음 날 시가 매수"}
        break
    return {"kind": None}


def _open_trade(o, h, l, c, e: int, stop: float, tgt: float) -> str | None:
    """e일 시가 진입 거래가 오늘까지 손절·목표·기간 만료 없이 살아 있으면 설명 문자열, 끝났으면 None."""
    n = len(c)
    if e + MAX_HOLD - 1 < n - 1:
        return None
    for j in range(e, n):
        if (j > e and (o[j] <= stop or o[j] >= tgt)) or l[j] <= stop or h[j] >= tgt:
            return None
    return f"{n - e}/{MAX_HOLD}일째"


def atr_pct(bars: pd.DataFrame, n: int = 20) -> float:
    h, l, c = bars["high"], bars["low"], bars["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return float(tr.rolling(n).mean().iloc[-1] / c.iloc[-1])


def control(bars: pd.DataFrame) -> dict:
    """칼만 중심선·관리선(1스텝 예측 ± 2σ, 로그가격) + CUSUM 상태. 모두 그날까지의 데이터만 사용."""
    p = Params()
    f = kalman_features(bars, p)
    pred = np.log(f["close"]) - f["innov"]
    cu = cusum(f["innov"], f["sigma"], p.cusum_k, p.cusum_h)
    lvl, slope, sig = f["level"].iloc[-1], f["slope"].iloc[-1], f["sigma"].iloc[-1]
    nxt = lvl + slope
    if cu.alarm_up.iloc[-1]:
        cstate = ("up", "상방 추세 변화 감지 중")
    elif cu.alarm_down.iloc[-1]:
        cstate = ("down", "하방 추세 변화 감지 중")
    else:
        cstate = ("none", "추세 변화 감지 없음")
    return {"center": np.exp(f["level"]), "ucl": np.exp(pred + BAND * f["sigma"]), "lcl": np.exp(pred - BAND * f["sigma"]),
            "z": f["band_z"], "z_today": float(f["band_z"].iloc[-1]), "sigma": float(sig),
            "next_lo": float(np.exp(nxt - BAND * sig)), "next_hi": float(np.exp(nxt + BAND * sig)),
            "next_center": float(np.exp(nxt)), "cusum": cstate,
            "c_pos": float(cu.c_pos.iloc[-1]), "c_neg": float(cu.c_neg.iloc[-1]), "h": p.cusum_h}


def company(code: str, bars: pd.DataFrame) -> dict:
    """DART 기준 종목 PER·PBR(5년 중 위치), 최근 180일 자본 정책 공시, 다음 정기보고서 기한 — 사실 정보만."""
    from client import dart_dir
    from valuation import _asof, pct_rank, stock_fundamentals
    out = {"per": None, "pbr": None, "per_pct": None, "pbr_pct": None, "events": [], "next_report": None}
    d = dart_dir()
    try:
        fin = pd.read_csv(d / "financials.csv", dtype={"stock_code": str, "rcept_dt": str})
        shares = pd.read_csv(d / "shares.csv", dtype={"stock_code": str}).set_index("stock_code")["shares"]
        mine = fin[fin["stock_code"] == code]
        f = stock_fundamentals(mine).get(code, {}) if not mine.empty else {}
        if code in shares.index and "ni" in f and "eq" in f:
            mcap = bars["close"] * shares[code]
            ni, eq = _asof(f["ni"], "ttm", bars.index), _asof(f["eq"], "equity", bars.index)
            per = (mcap / ni).where(ni > 0)
            pbr = mcap / eq
            if pd.notna(ni.iloc[-1]):
                out["per"] = None if ni.iloc[-1] <= 0 else float(per.iloc[-1])
                out["per_pct"] = pct_rank(per) if out["per"] else None
                out["loss"] = bool(ni.iloc[-1] <= 0)
            if pd.notna(pbr.iloc[-1]):
                out["pbr"], out["pbr_pct"] = float(pbr.iloc[-1]), pct_rank(pbr)
            p5, b5 = per.iloc[-1250:].dropna(), pbr.iloc[-1250:].dropna()       # 참고 자료용 5년 범위
            out["per_rng"] = (float(p5.min()), float(p5.max())) if len(p5) else None
            out["pbr_rng"] = (float(b5.min()), float(b5.max())) if len(b5) else None
        last = fin[(fin["stock_code"] == code) & (fin["account"] == "op")].sort_values(["year", "q"]).tail(1)
        if not last.empty:
            y, q = int(last["year"].iloc[0]), int(last["q"].iloc[0])
            ny, nq = (y + 1, 1) if q == 4 else (y, q + 1)
            md, plus = Q_DUE[nq]
            out["next_report"] = (f"{ny}년 {nq}분기" if nq < 4 else f"{ny}년 사업보고서",
                                  pd.Timestamp(f"{ny + plus}{md}"))
        ev = pd.read_csv(d / "events.csv", dtype={"stock_code": str, "rcept_dt": str})
        ev = ev[(ev["stock_code"] == code) & (pd.to_datetime(ev["rcept_dt"]) >= bars.index[-1] - pd.Timedelta(days=180))]
        out["events"] = [(pd.Timestamp(x), EVENT_NAMES.get(t, t)) for x, t in
                         sorted(zip(ev["rcept_dt"], ev["type"]), reverse=True)[:3]]
    except FileNotFoundError:
        pass
    return out


REF_REPORT_ROWS, REF_EVENT_DAYS = 10, 90


def _eok(v: float) -> str:
    """원 → 억 원(부호 포함)."""
    return "–" if v is None or not np.isfinite(v) else f"{v / 1e8:+,.0f}억"


def reference_html(code: str, bars: pd.DataFrame, close: float) -> str:
    """종목 탭 맨 아래 접힌 "참고 자료(트래커)" — 판단에 쓰지 않는 사실 정보만(9.17 · 9.82 · 9.106: 이후 방향 정보 없음)."""
    parts = []
    # 1 리포트 60일 목록
    try:
        from sector import load_reports
        rep = load_reports()
        g = rep[(rep["code"] == code) & (rep["date"] >= rep["date"].max() - pd.Timedelta(days=STOCK_REPORT_DAYS))]
        g = g.sort_values("date", ascending=False).head(REF_REPORT_ROWS)
        arrow = {"up": "↑", "down": "↓", "flat": "→", "new": "신규"}
        rows = "".join(f'<tr><td>{d:%m/%d}</td><td>{html.escape(str(b))}</td><td>{arrow.get(x, "")}</td>'
                       f'<td>{"" if pd.isna(t) else f"{t:,.0f}"}</td><td>{"" if pd.isna(t) else f"{t / close - 1:+.0%}"}</td></tr>'
                       for d, b, x, t in zip(g["date"], g["broker"], g["dir"], g["tp"]))
        parts.append(f'<h4>리포트 (최근 {STOCK_REPORT_DAYS}일, 최신 {REF_REPORT_ROWS}건)</h4>' +
                     (f'<table><tr><th>날짜</th><th>증권사</th><th></th><th>목표가</th><th>현재가 대비</th></tr>{rows}</table>' if rows else '<p>없음</p>'))
    except Exception:
        pass
    # 2 실적 — 최근 5분기 매출 · 영업이익 YoY · QoQ · 이익률
    try:
        from client import dart_dir
        from fundamentals import quarterly
        fin = pd.read_csv(dart_dir() / "financials.csv", dtype={"stock_code": str, "rcept_dt": str})
        fin = fin[fin["stock_code"] == code]
        if not fin.empty:
            rv = quarterly(fin, "rev").set_index(["year", "q"])["value"]
            op = quarterly(fin, "op").set_index(["year", "q"])["value"]
            keys = list(rv.index)[-5:]

            def yoy(s_, y, q):
                b_ = s_.get((y - 1, q))
                a_ = s_.get((y, q))
                return a_ / b_ - 1 if a_ is not None and b_ is not None and b_ > 0 else None

            prev = lambda y, q: (y - 1, 4) if q == 1 else (y, q - 1)
            fmt = lambda v: "–" if v is None or not np.isfinite(v) else f"{v:+.0%}"
            rows = ""
            for y, q in reversed(keys):
                r_, o_ = rv.get((y, q)), op.get((y, q))
                qq = op.get(prev(y, q))
                qoq = o_ / qq - 1 if o_ is not None and qq is not None and qq > 0 else None
                opm = o_ / r_ if r_ and o_ is not None else None
                rows += (f'<tr><td>{str(y)[2:]}Q{q}</td><td>{(r_ or 0) / 1e8:,.0f}</td><td>{fmt(yoy(rv, y, q))}</td>'
                         f'<td>{(o_ or 0) / 1e8:,.0f}</td><td>{fmt(yoy(op, y, q))}</td><td>{fmt(qoq)}</td>'
                         f'<td>{"–" if opm is None else f"{opm:.1%}"}</td></tr>')
            parts.append('<h4>실적 (억 원, DART)</h4><table><tr><th>분기</th><th>매출</th><th>YoY</th><th>영업이익</th><th>YoY</th>'
                         f'<th>QoQ</th><th>이익률</th></tr>{rows}</table>')
    except Exception:
        pass
    # 3 공시 90일(자본 정책 — 자기주식 · 유상증자 · 전환사채)
    try:
        from client import dart_dir
        ev = pd.read_csv(dart_dir() / "events.csv", dtype={"stock_code": str, "rcept_dt": str})
        ev = ev[(ev["stock_code"] == code) & (pd.to_datetime(ev["rcept_dt"]) >= bars.index[-1] - pd.Timedelta(days=REF_EVENT_DAYS))]
        li = "".join(f'<li>{pd.Timestamp(d):%m/%d} {html.escape(str(n))}</li>'
                     for d, n in sorted(zip(ev["rcept_dt"], ev["report_nm"]), reverse=True))
        parts.append(f'<h4>공시 (최근 {REF_EVENT_DAYS}일, 자본 정책)</h4>' + (f'<ul>{li}</ul>' if li else '<p>없음</p>'))
    except Exception:
        pass
    # 4 수급 — 20 · 60일 누적(억 원)과 거래대금 대비 비중
    try:
        import collect_investor_detail as cid
        import collect_program as cp
        inv, prog = cid.load_detail(code), cp.load_program(code)
        val = (bars["close"] * bars["volume"]).astype(float)
        src = []
        if inv is not None and not inv.empty:
            iv = inv.reindex(bars.index) * 1e6
            src += [("외국인", iv["외국인"]), ("기관(금투 제외)", iv["기관합계"] - iv["금융투자"]), ("연기금", iv["연기금"])]
        if prog is not None and not prog.empty:
            src.append(("프로그램", prog["prog_net"].reindex(bars.index)))
        rows = ""
        for nm, x in src:
            cells = ""
            for n in (20, 60):
                s_, v_ = x.iloc[-n:].sum(min_count=1), val.iloc[-n:].sum()
                share = "" if not v_ or not np.isfinite(s_) else f"{s_ / v_:+.1%}"
                cells += f'<td>{_eok(s_)}</td><td>{share}</td>'
            rows += f'<tr><td>{nm}</td>{cells}</tr>'
        if rows:
            parts.append('<h4>수급 누적 (거래대금 대비)</h4><table><tr><th></th><th>20일</th><th></th><th>60일</th><th></th></tr>'
                         + rows + '</table>')
    except Exception:
        pass
    # 5 밸류 — 지금 · 5년 범위
    try:
        co = company(code, bars)
        vl = []
        if co.get("per") is not None and co.get("per_rng"):
            vl.append(f'PER {co["per"]:.1f}배 · 5년 {co["per_rng"][0]:.1f}~{co["per_rng"][1]:.1f} (위치 {co["per_pct"]:.0%})')
        if co.get("pbr") is not None and co.get("pbr_rng"):
            vl.append(f'PBR {co["pbr"]:.2f}배 · 5년 {co["pbr_rng"][0]:.2f}~{co["pbr_rng"][1]:.2f} (위치 {co["pbr_pct"]:.0%})')
        if vl:
            parts.append('<h4>밸류</h4><p>' + "<br>".join(vl) + '</p>')
    except Exception:
        pass
    if not parts:
        return ""
    return ('<details class="ref"><summary>참고 자료 (트래커) — 판단에 쓰지 않음</summary>' + "".join(parts)
            + '<p class="sub">리포트 · 공시 · 실적 · 수급은 검증에서 이후 방향을 알려 주지 않았다(9.15 · 9.17 · 9.82 · 9.106). 기업 상태 점검용.</p></details>')


def flow_states(code: str, index: pd.DatetimeIndex) -> list[tuple[str, str, str]]:
    """외국인·프로그램 순매수의 CUSUM 경보 상태(9.21과 같은 설정: 60일 정규화, k 0.5, h 4.5).
    방향이 약하게 기울 뿐 판정 미달이라 참고 표시만 한다. (이름, 상태 문구, 색 클래스) 목록."""
    from collect_investor_detail import load_detail
    from collect_program import load_program
    from flow_alarm import flow_cusum
    src = {}
    d = load_detail(code)
    if d is not None and "외국인" in d.columns:
        src["외국인"] = d["외국인"]
    pr = load_program(code)
    if pr is not None and not pr.empty:
        src["프로그램"] = pr["prog_net"]
    out = []
    for name, x in src.items():
        x = x.loc[:index[-1]].dropna()
        if len(x) < 120:
            continue
        cu = flow_cusum(x)
        for flag, word, cls in ((cu.alarm_up, "꾸준한 순매수 경보 중", "c-up"), (cu.alarm_down, "꾸준한 순매도 경보 중", "c-dn")):
            if flag.iloc[-1]:
                off = flag[~flag]
                since = flag.index[flag.index > off.index[-1]][0] if not off.empty else flag.index[0]
                out.append((name, f"{word} ({since:%m/%d}~)", cls))
                break
        else:
            out.append((name, "평소", "sub"))
    return out


def refresh_flows(caller, code: str) -> None:
    """목록 종목의 수급·프로그램·신용잔고 일별 데이터를 증분 갱신(몇 콜)."""
    import collect_credit as cc
    import collect_investor_detail as cid
    import collect_program as cp
    for mod in (cid, cp, cc):
        try:
            mod.update(caller, code)
        except Exception as e:  # 수급 갱신 실패로 페이지 생성을 막지 않는다
            print(f"{code} 수급 갱신 실패: {e}", file=sys.stderr)


SUMMARY: list[dict] = []    # build() 동안 종목별 가이드 요약을 모아 results/scenario/summary.json 으로 (텔레그램이 읽음)
STOCK_REPORT_DAYS = 60     # 종목 하나는 리포트가 드물어 섹터(20거래일)보다 길게


def _verified_text() -> dict:
    """가짜 이탈 + T² 검증 수치(9.27) — 결과 파일에서 읽고, 없으면 기록된 값."""
    try:
        a = json.loads((results_dir() / "levels_validation.json").read_text(encoding="utf-8"))["stocks"]["box/failure"]["t2"]["excess"]
        b = json.loads((results_dir() / "levels_validation_oos.json").read_text(encoding="utf-8"))["box/failure"]["t2"]["excess"]
        lo, hi = sorted((a, b))
        return {"text": f"+{lo * 100:.1f}~{hi * 100:.1f}%p"}
    except (FileNotFoundError, KeyError):
        return {"text": "+1.4~2.0%p"}


VERIFIED = _verified_text()


def stock_reports(code: str, close: float) -> str | None:
    """증권사 리포트 한 줄(텔레그램 리포트 요약, 검증이력 9.25 데이터) — 천천히 바뀌는 배경 정보라 기업 정보 카드에."""
    from sector import load_reports
    rep = load_reports()
    if rep.empty:
        return None
    updated = rep["date"].max()
    g = rep[(rep["code"] == code) & (rep["date"] >= updated - pd.Timedelta(days=STOCK_REPORT_DAYS))].sort_values("date")
    if g.empty:
        return f"{STOCK_REPORT_DAYS}일 리포트 없음 · {updated:%m/%d}"
    cnt = g["dir"].value_counts()
    parts = [f"{STOCK_REPORT_DAYS}일 {len(g)}건(↑{cnt.get('up', 0)} ↓{cnt.get('down', 0)})"]
    tp = g.dropna(subset=["tp"]).groupby("broker").tail(1)["tp"]          # 증권사별 가장 최근 목표가
    if len(tp):
        avg = float(tp.mean())
        parts.append(f"목표가 평균 {avg:,.0f}({avg / close - 1:+.0%}, {len(tp)}곳)")
    last = g.iloc[-1]
    word = {"up": "상향", "down": "하향", "flat": "유지", "new": "신규"}.get(last["dir"], "")
    parts.append(f"최근 {last['broker']} {word}".strip())
    return " · ".join(parts) + f" · {updated:%m/%d}"


NEWS_DAYS = 30


def stock_news(code: str, name: str = "") -> str | None:
    """뉴스·리포트 요약 채널에서 이 종목이 연결된 글(reports/news.py) — 개수 + 가장 최근 제목(원문 링크). 방향 판정 없음."""
    p = data_dir() / "_reports" / "news.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p, dtype={"code": str}, parse_dates=["date"])
    if d.empty:
        return None
    updated = d["date"].max()
    g = d[(d["code"] == code) & (d["date"] >= updated - pd.Timedelta(days=NEWS_DAYS))].sort_values(["date", "msg_id"])
    if g.empty:
        return f'{NEWS_DAYS}일 없음 <span class="sub">· {updated:%m/%d}</span>'
    own = g[g["title"].str.contains(name[:2], regex=False)] if name else g     # 제목에 종목명이 나온 글 우선
    last = (own if len(own) else g).iloc[-1]
    t = last["title"] if len(last["title"]) <= 42 else last["title"][:41] + "…"
    ch = last["channel"] if "channel" in last and isinstance(last["channel"], str) else "aicorporateanalysisdeepdive"
    link = f'https://t.me/{ch}/{int(last["msg_id"])}'
    return (f'{NEWS_DAYS}일 {len(g)}건 · {last["date"]:%m/%d} <a href="{link}" target="_blank">{html.escape(t)}</a>'
            f' <span class="sub">· {updated:%m/%d}</span>')


def pooled_outcomes() -> dict:
    """전 종목 합산 결과 분포(참고) — results/box_scenario_events.csv."""
    p = results_dir() / "box_scenario_events.csv"
    if not p.exists():
        return {}
    d = pd.read_csv(p)
    out = {}
    for pat, g in d.groupby("pattern"):
        v = g["how"].value_counts(normalize=True)
        out[pat] = {"target": float(v.get("target", 0)), "stop": float(v.get("stop", 0)),
                    "timeout": float(v.get("timeout", 0)), "days": float(g["days"].median()), "n": int(len(g))}
    return out


def plans(bars: pd.DataFrame, st: dict) -> list[dict]:
    """시나리오 두 개의 진입·손절·목표. 진행 중 시나리오는 그때 고정된 박스, 아니면 오늘 박스 기준 '예상'."""
    n = len(bars)
    H, L, M = box_levels(bars, n - 1)
    atr = atr_pct(bars) * bars["close"].iloc[-1]
    out = []
    if st.get("kind") == "failure":
        out.append({"kind": "failure", "entry": st.get("entry", st["L"]), "stop": st["stop"], "target": st["H"],
                    "fixed": st.get("fixed_entry", False), "t2": st.get("t2"), **_st(st)})
    else:
        out.append({"kind": "failure", "entry": L, "stop": L - atr, "target": H, "fixed": False,
                    "status": "대기", "detail": f"① {L:,.0f} 아래로 마감 ② {FAIL_RECOVER}일 안에 {L:,.0f} 위로 다시 마감 ③ 다음 날 시가 매수 (손절 = 이탈 중 최저가)"})
    if st.get("kind") == "retest":
        out.append({"kind": "retest", "entry": st.get("entry", st["H"]), "stop": st["M"], "target": st["H"] + (st["H"] - st["L"]),
                    "fixed": st.get("fixed_entry", False), **_st(st)})
    else:
        out.append({"kind": "retest", "entry": H, "stop": M, "target": H + (H - L), "fixed": False,
                    "status": "대기", "detail": f"① {H:,.0f} 위로 마감 ② {RETEST_WITHIN}일 안에 {H:,.0f}까지 되밀려도 종가가 {M:,.0f} 위 ③ 다음 날 시가 매수"})
    return out


def _st(st: dict) -> dict:
    return {"status": st["status"], "detail": st["detail"]}


# ── 렌더링 ─────────────────────────────────────────────
def _poly(vals, x, y) -> str:
    return " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(vals) if pd.notna(v))


def svg_chart(bars: pd.DataFrame, pl: list[dict], ctl: dict, cid: str) -> str:
    d = bars.iloc[-CHART_DAYS:]
    n = len(d)
    W, Hh, padl, padr, padt, padb = 640, 340, 8, 64, 10, 78      # 아래 60px = 거래량
    levels = [p[k] for p in pl for k in ("stop", "target")] or [float(d["close"].iloc[-1])]   # 보유 탭은 시나리오 선 없음
    lo = min(d["low"].min(), min(levels)) * 0.99
    hi = max(d["high"].max(), max(levels)) * 1.01
    y = lambda v: padt + (hi - v) / (hi - lo) * (Hh - padt - padb)
    cw = (W - padl - padr) / n
    x = lambda i: padl + (i + 0.5) * cw
    H, L, M = box_levels(bars, len(bars) - 1)
    parts = [f'<clipPath id="{cid}"><rect x="{padl}" y="{padt}" width="{W - padl - padr}" height="{Hh - padt - padb}"/></clipPath>',
             f'<rect x="{x(n - BOX) - cw / 2:.1f}" y="{y(H):.1f}" width="{BOX * cw:.1f}" height="{y(L) - y(H):.1f}" class="box"/>',
             f'<g clip-path="url(#{cid})">'
             f'<polyline points="{_poly(ctl["ucl"].iloc[-n:], x, y)}" class="cl-b"/>'
             f'<polyline points="{_poly(ctl["lcl"].iloc[-n:], x, y)}" class="cl-b"/>'
             f'<polyline points="{_poly(ctl["center"].iloc[-n:], x, y)}" class="cl-c"/></g>']
    for i, (_, r) in enumerate(d.iterrows()):
        up = r["close"] >= r["open"]
        cls = "up" if up else "dn"
        top, bot = y(max(r["open"], r["close"])), y(min(r["open"], r["close"]))
        parts.append(f'<line x1="{x(i):.1f}" x2="{x(i):.1f}" y1="{y(r["high"]):.1f}" y2="{y(r["low"]):.1f}" class="{cls}"/>'
                     f'<rect x="{x(i) - cw * 0.35:.1f}" y="{top:.1f}" width="{cw * 0.7:.1f}" height="{max(bot - top, 0.8):.1f}" class="{cls}"/>')
    z = ctl["z"].iloc[-n:].to_numpy()
    for i, (_, r) in enumerate(d.iterrows()):          # 관리선 밖으로 마감한 날
        if pd.notna(z[i]) and abs(z[i]) > BAND:
            yy = y(r["high"]) - 5 if z[i] > 0 else y(r["low"]) + 5
            parts.append(f'<circle cx="{x(i):.1f}" cy="{yy:.1f}" r="2.6" class="brk"/>')
    labels = []
    for v, lab, cls in ((H, "박스 고가", "lv-h"), (M, "중간선", "lv-m"), (L, "박스 저가", "lv-l")):
        parts.append(f'<line x1="{padl}" x2="{W - padr}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="{cls}"/>')
        labels.append([y(v), f"{v:,.0f}", "lv-t"])
    for p in pl:                                   # 진행 중·진입 신호 시나리오는 그때 고정된 손절선도 표시
        if p["status"] != "대기":
            parts.append(f'<line x1="{padl}" x2="{W - padr}" y1="{y(p["stop"]):.1f}" y2="{y(p["stop"]):.1f}" class="lv-st"/>')
            labels.append([y(p["stop"]), f'{p["stop"]:,.0f}', "lv-t c-st"])
    if len(pl) > 1:
        tgt = pl[1]["target"]
        parts.append(f'<line x1="{x(n - BOX):.1f}" x2="{W - padr}" y1="{y(tgt):.1f}" y2="{y(tgt):.1f}" class="lv-tg"/>')
        labels.append([y(tgt), f"{tgt:,.0f}", "lv-t"])
    else:                                          # 보유 탭: 시나리오 대신 50일선
        m50 = bars["close"].rolling(50).mean().iloc[-n:].to_numpy()
        pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(m50) if v == v)
        parts.append(f'<polyline points="{pts}" class="lv-ma50" fill="none"/>')
        labels.append([y(m50[-1]), f"50일선 {m50[-1]:,.0f}", "lv-t"])
    labels.sort(key=lambda t: t[0])                 # 가격 라벨이 겹치지 않게 위에서부터 최소 12px 간격
    for i in range(1, len(labels)):
        labels[i][0] = max(labels[i][0], labels[i - 1][0] + 12)
    parts += [f'<text x="{W - padr + 4}" y="{ly + 4:.1f}" class="{c}">{t}</text>' for ly, t, c in labels]
    vol = d["volume"].to_numpy(float)
    vtop, vbot = Hh - padb + 12, Hh - 18
    vmax = np.nanmax(vol) if np.nanmax(vol) > 0 else 1
    avg20 = bars["volume"].rolling(20).mean().shift(1).iloc[-n:].to_numpy()
    for i, (_, r) in enumerate(d.iterrows()):
        vh = (vbot - vtop) * vol[i] / vmax
        hot = pd.notna(avg20[i]) and avg20[i] > 0 and vol[i] >= 2 * avg20[i]
        parts.append(f'<rect x="{x(i) - cw * 0.35:.1f}" y="{vbot - vh:.1f}" width="{cw * 0.7:.1f}" height="{vh:.1f}" '
                     f'class="vb {"up" if r["close"] >= r["open"] else "dn"}{" hot" if hot else ""}"/>')
    parts.append(f'<polyline points="{" ".join(f"{x(i):.1f},{vbot - (vbot - vtop) * v / vmax:.1f}" for i, v in enumerate(avg20) if pd.notna(v))}" class="vavg"/>'
                 f'<text x="{W - padr + 4}" y="{vtop + 8:.0f}" class="ax">거래량</text>')
    parts.append(f'<text x="{padl}" y="{Hh - 4}" class="ax">{d.index[0]:%Y-%m-%d}</text>'
                 f'<text x="{W - padr}" y="{Hh - 4}" class="ax" text-anchor="end">{d.index[-1]:%Y-%m-%d}</text>')
    return f'<svg viewBox="0 0 {W} {Hh}" class="chart" role="img" aria-label="최근 {CHART_DAYS}거래일 캔들과 박스">{"".join(parts)}</svg>'


NAMES = {"failure": ("가짜 이탈 후 복귀", "Failure"), "retest": ("돌파 후 리테스트", "Not Failure")}


BASIS_LABEL = {"hold": "보유 평균가", "now": "지금 가격에 산다면", "plan": "예상 매수가 (조건 충족 시)"}
BADGE = {"진입 신호": "매수 신호 · 내일 시가", "진행 중": "매수 구간 유지 중", "복귀 대기": "이탈 중 · 복귀 기다림",
         "리테스트 대기": "돌파 완료 · 되밀림 기다림", "대기": "조건 전"}


def plan_card(code: str, p: dict, pooled: dict, sigma: float, close: float, hold: dict | None) -> str:
    """계산 기준(p["ref"]): 보유 = 평균 단가, 미보유 규칙 충족 = 현재가, 대기 = 예상 진입가. 손절 거리(σ)는 지금 흔들림 기준이라
    대기면 예상 진입가에서, 그 외엔 현재가에서 잰다."""
    title, eng = NAMES[p["kind"]]
    ref, mode = p["ref"], p["mode"]
    active = p["status"] in ACTIVE
    z_from = ref if mode == "plan" else close
    zs = np.log(z_from / p["stop"]) / sigma if sigma > 0 and z_from > 0 and p["stop"] > 0 else np.nan
    risk, reward = (ref - p["stop"]) / ref, (p["target"] - ref) / ref
    rows = [f'<tr><th>{BASIS_LABEL[mode]}</th><td>{ref:,.0f}</td><td class="sub">'
            f'{("현재 " + f"{close / ref - 1:+.1%}") if mode == "hold" else ""}</td></tr>']
    if p["status"] == "진행 중":
        rows.append(f'<tr class="rule"><th>규칙상 매수가</th><td>{p["rule_entry"]:,.0f}</td><td></td></tr>')
    if z_from <= p["stop"]:
        rows.append(f'<tr><th>손절</th><td class="c-dn">{p["stop"]:,.0f}</td><td class="sub">이미 손절선 아래</td></tr>')
    else:
        rows.append(f'<tr><th>손절</th><td class="c-dn">{p["stop"]:,.0f}</td><td class="sub">{-risk:+.1%}</td></tr>')
    rows.append(f'<tr><th>목표</th><td class="c-up">{p["target"]:,.0f}</td><td class="sub">{reward:+.1%}</td></tr>')
    if risk > 0 and reward > 0:
        rows.append(f'<tr><th>손익비</th><td colspan="2">1 : {reward / risk:.1f}</td></tr>')
    if zs == zs and zs > 0:
        rows.append(f'<tr><th>손절 거리</th><td>{zs:.1f}σ</td><td class="sub">{"가까움" if zs < 1 else ""}</td></tr>')
    size = ""                                       # 금액(주수·원) 표시는 하지 않는다 — 비율(%·σ·손익비)만
    po_html = ""
    t2_html = ""
    if p["kind"] == "failure" and p.get("t2") is not None:
        t2_html = (f'<p class="t2 on">&#9733; T² 동반 — 거래당 {VERIFIED["text"]} · 적중 16~18%(가끔 크게)</p>'
                   if p["t2"] else '<p class="t2">T² 없음 — 우위 미확인</p>')
    return f"""<div class="plan{' active' if active else ''}{' verified' if p.get('t2') else ''}">
      <div class="plan-top"><span class="plan-title">{title}</span><span class="plan-eng">{eng}</span>
        {'<span class="badge vf">검증된 셋업</span>' if p.get('t2') else ''}
        <span class="badge{' on' if active else ''}">{html.escape(BADGE.get(p['status'], p['status']))}</span></div>
      <p class="detail">{html.escape(p['detail'])}</p>{t2_html}
      <table class="lv">{''.join(rows)}</table>
      {size}
      {po_html}
    </div>"""


EVIDENCE_W = {"검증됨": 3.0, "유망": 2.0, "최근경향": 1.5, "약함": 1.0, "참고": 0.5}   # 검증 등급별 무게(통계 확률이 아니라 등급에 따른 판단값)
SECTOR_NET_CUT = 0.2        # 섹터 애널리스트 순상향 ±20% 이상이면 방향 근거로 봄
PANIC_DROP, PANIC_VOL = -0.05, 3.0   # 9.28 투매 정의
MARKET_TREND_GRADE = "유망"   # 9.37 가설 1 새 표본 재현(차이 +0.44R [+0.06, +0.81])
MARKET_ACTIONS: dict = {}   # build() 가 채움: reports/market_actions.active()
MARKET: dict = {"up": None, "date": None, "state": None}   # build() 가 채움: 코스피(069500) 장세


def market_state(bars: pd.DataFrame) -> dict | None:
    """코스피(069500) 장세 — 9.37 정의: 상승장 = 종가 > 60일선 그리고 60일선 상승, 하락장 = 둘 다 반대, 그 외 횡보·전환.
    며칠째인지와 9.37·9.38 결과에 따른 유리한 시나리오를 함께 돌려준다."""
    c = bars["close"]
    if len(c) < 81:
        return None
    ma60 = c.rolling(60).mean()
    up = (c > ma60) & (ma60 > ma60.shift(20))
    dn = (c < ma60) & (ma60 < ma60.shift(20))
    state = np.where(up, "상승장", np.where(dn, "하락장", "횡보·전환"))
    cur = state[-1]
    days = 1
    while days < len(state) and state[-1 - days] == cur:
        days += 1
    return {"state": str(cur), "days": days, "date": c.index[-1],
            "vs_ma60": float(c.iloc[-1] / ma60.iloc[-1] - 1), "ma60_slope": float(ma60.iloc[-1] / ma60.iloc[-21] - 1),
            **REGIME_GUIDE[str(cur)]}


# 장세별 한 줄 — 9.37·9.38·9.39·9.40 결과(센서·코스피·코스닥 표본 범위). act = 할 것, nums = 근거 숫자
REGIME_GUIDE = {
    "상승장": {"favor": "돌파 매매 ↑ (RS 70↑, 50일선 이탈까지 보유) · 가짜 이탈 쉼",
              "nums": "돌파 승률 17% · +0.45R · ×1.5 | 가짜 이탈 기대 ≈ 0"},
    "횡보·전환": {"favor": "가짜 이탈 ↑ (장세 중 최고) · 돌파 추격 ✕",
                "nums": "가짜 이탈 +0.5R | 돌파 무작위 수준"},
    "하락장": {"favor": "가짜 이탈 보통 · 돌파 ✕ · 응축 종목 ✕",
              "nums": "가짜 이탈 +0.1~0.4R | 돌파 −1~2% | 응축 20일 뒤 −2~3%p"},
}


def market_html(m: dict | None) -> str:
    if not m:
        return ""
    cls = {"상승장": "up", "하락장": "dn"}.get(m["state"], "side")
    bp = results_dir() / "reports" / "market_brief.json"
    br = json.loads(bp.read_text(encoding="utf-8")) if bp.exists() else {}
    brief = (f'<span class="mb">시황 · {br["date"][5:10].replace("-", "/")} {br["kind"]}: '
             f'<a href="{br["link"]}" target="_blank">{html.escape(br["title"])}</a></span>') if br.get("title") else ""
    return (f'<div class="market {cls}"><b>코스피 {m["state"]}</b> <span class="md">{m["days"]}일째</span>'
            f'<span class="mf">{m["favor"]}</span><span class="mx">{m["nums"]}</span>'
            f'{brief}<span class="mn">60일선 대비 {m["vs_ma60"]:+.1%} · 60일선 20일 {m["ma60_slope"]:+.1%} · {m["date"]:%m/%d} 종가</span></div>')


def kospi_uptrend(bars: pd.DataFrame) -> bool | None:
    """9.37 정의: 종가 > 60일선 그리고 60일선이 20일 전보다 높음."""
    c = bars["close"]
    if len(c) < 81:
        return None
    ma60 = c.rolling(60).mean()
    return bool(c.iloc[-1] > ma60.iloc[-1] and ma60.iloc[-1] > ma60.iloc[-21])


def sector_net(code: str) -> tuple[str, float] | None:
    """종목이 속한 테마 대표 섹터의 20거래일 애널리스트 순상향(9.25). 여러 섹터면 리포트가 많은 쪽."""
    p = results_dir() / "reports" / "sector_summary.json"
    if not p.exists():
        return None
    secs = json.loads(p.read_text(encoding="utf-8")).get("sectors", {})
    best = None
    for b in load_baskets():
        if b["name"] in secs and any(x["code"] == code for x in b["sensors"]):
            v = secs[b["name"]]
            if v.get("net") is not None and (best is None or v["n"] > best[2]):
                best = (b["target"]["name"], v["net"], v["n"])
    return (best[0], best[1]) if best else None


def stock_report_tilt(code: str) -> int:
    """종목 리포트 60일 상향 − 하향 (±2건 이상일 때만 방향으로 봄)."""
    try:
        from sector import load_reports
        rep = load_reports()
        g = rep[(rep["code"] == code) & (rep["date"] >= rep["date"].max() - pd.Timedelta(days=STOCK_REPORT_DAYS))]
        d = int((g["dir"] == "up").sum() - (g["dir"] == "down").sum())
        return d if abs(d) >= 2 else 0
    except Exception:
        return 0


def trend_components(bars: pd.DataFrame, code: str | None = None) -> dict:
    """보유 판단 재료(9.81 · 9.86 · 9.87): 추세 강도 4요소 + 5일선 위치 · 볼린저 %b · 폭 · 5일 수익률 · 프로그램 60일(코드 주면)."""
    c, h = bars["close"].to_numpy(float), bars["high"].to_numpy(float)
    if len(c) < 253:
        return {}
    r = lambda k: c[-1] / c[-1 - k] - 1
    m20, sd = c[-20:].mean(), c[-20:].std(ddof=1)
    out = {"ma60": c[-1] / c[-60:].mean() - 1, "ma120": c[-1] / c[-120:].mean() - 1, "hi52": c[-1] / h[-252:].max() - 1,
           "rs": 0.4 * r(63) + 0.2 * r(126) + 0.2 * r(189) + 0.2 * r(252), "ma5": c[-1] / c[-5:].mean() - 1,
           "bbp": (c[-1] - (m20 - 2 * sd)) / (4 * sd) if sd > 0 else np.nan, "bbw": 4 * sd / m20, "r5": r(5)}
    if code:
        try:
            from collect_program import load_program
            pg = load_program(code)
            if pg is not None and len(pg) >= 45:
                t = pg.iloc[-60:]
                out["prog60"] = float(t["prog_net"].sum() / t["value"].sum()) if t["value"].sum() else np.nan
        except Exception:
            pass
    return out


def trend_pct(bars: pd.DataFrame) -> float | None:
    """오늘의 후보 200종목(전날 저장 분포) 안에서 네 요소 백분위의 평균. 분포 파일이 없으면 None."""
    f = results_dir() / "scenario" / "trend_xs.csv"
    me = trend_components(bars)
    if not f.exists() or not me:
        return None
    xs = pd.read_csv(f, index_col=0)
    return float(np.mean([(xs[k].dropna() < v).mean() for k, v in me.items() if k in xs]))


TREND_TOP = 0.80            # 추세 강도 상위 20%
TREND_STRONG = 0.95         # 상위 5% = (강함) — 상승장만(9.88 · 9.93: 상승장은 계단, 횡보 · 하락장은 아님)

# 장세별 보유 판단(9.81 · 9.85 · 9.86 · 9.88): (기준 요소, 상위 문구, 하위 문구 또는 None, 등급)
HOLD_RULE = {"상승장": ("trend", "보유 유리", "약세", "최근경향"),
             "하락장": ("trend", "이익 실현 고려", None, "검증됨"),
             "횡보·전환": ("ma60", "되돌림 주의", None, "최근경향")}
# 겹치지 않는 참고(약함) 요소 — 상위 20%일 때만 태그(9.87). (요소, 태그)
HOLD_WEAK = {"상승장": [("bbw", "변동성 큼 +"), ("r5", "단기 강세 +")],
             "하락장": [("bbp", "밴드 상단 −"), ("bbw", "변동성 큼 +"), ("prog60", "프로그램 누적 매수 −")],
             "횡보·전환": [("ma5", "단기 과열 −")]}


def xs_pct(comp: dict) -> dict:
    """오늘의 후보 200종목(전날 저장) 분포 안 백분위. 추세 강도 = 네 요소 백분위 평균."""
    f = results_dir() / "scenario" / "trend_xs.csv"
    if not f.exists() or not comp:
        return {}
    xs = pd.read_csv(f, index_col=0)
    out = {k: float((xs[k].dropna() < v).mean()) for k, v in comp.items() if k in xs and v == v}
    if all(k in out for k in ("ma60", "ma120", "hi52", "rs")):
        out["trend"] = float(np.mean([out[k] for k in ("ma60", "ma120", "hi52", "rs")]))
    return out


# 가짜 이탈 매수 후 k일째 종가 위치(진입가 대비 R) → 이후 손절로 끝날 비율(9.92, 2016~ 전체 7,675건, 평소 59%). 구간 < −0.5 · −0.5~0 · 0~+0.5 · ≥ +0.5R
FAIL_EARLY = {1: (0.80, 0.59, 0.44, 0.45), 2: (0.81, 0.58, 0.39, 0.36), 3: (0.82, 0.56, 0.39, 0.29)}


def fail_early_line(o, c, e: int, d: int, stop: float) -> str | None:
    """매수 후 1~3일째(3일 넘으면 3일째 값) 종가 위치로 손절 확률 — 판단이 아니라 기대치 안내(9.92: 일찍 정리해도 나아지지 않음)."""
    if d < 1 or o[e] <= stop:
        return None
    k = min(d, 3)
    m = (c[e + k - 1] - o[e]) / (o[e] - stop)
    p = FAIL_EARLY[k][int(np.searchsorted([-0.5, 0, 0.5], m, side="right"))]
    cmp = "평소의 절반" if p <= 0.30 else "평소보다 낮음" if p < 0.54 else "평소보다 높음" if p > 0.64 else "평소 수준"   # 평소 59%
    return f"{k}일째 {m:+.1f}R · 손절 확률 {p:.0%} ({cmp}) · 손절선 지키며 보유"


def setup_track(bars: pd.DataFrame, buy_date) -> dict | None:
    """매수일 차트에서 셋업을 찾아 규칙대로 추적. 돌파(9.68 · 9.72): 신호일 종가 − 1ATR 손절, 종가 < 50일선, 최대 120일.
    가짜 이탈(9.18): 이탈~복귀 최저가 손절, 박스 상단 목표, 20일. 못 찾으면 None(일반 보유)."""
    if buy_date is None:
        return None
    idx = bars.index
    e = int(idx.searchsorted(buy_date))
    if e > len(idx) or e < BOX + 60:                        # e == len: 오늘 매수(아직 매수일 봉 없음)
        return None
    o, h, l, c = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    Hs = bars["high"].rolling(BOX).max().shift(1).to_numpy()
    Ls = bars["low"].rolling(BOX).min().shift(1).to_numpy()
    tr = np.maximum(h - l, np.maximum(abs(h - np.r_[np.nan, c[:-1]]), abs(l - np.r_[np.nan, c[:-1]])))
    atr = pd.Series(tr).rolling(14).mean().to_numpy()
    ma50 = pd.Series(c).rolling(50).mean().to_numpy()
    n, d = len(c) - 1, len(c) - e
    lo_ = lambda: l[e:].min() if d > 0 else np.inf
    hi_ = lambda: h[e:].max() if d > 0 else -np.inf
    for t in range(e - 1, e - 4, -1):                       # 매수일 1~3일 전 첫 돌파
        if c[t] > Hs[t] and c[t - 1] <= Hs[t - 1]:
            stop = c[t] - atr[t]
            below = c[e:] < ma50[e:]
            if lo_() <= stop:
                act = "손절선 닿음 → 정리"
            elif below.any():
                act = f"50일선 이탈({idx[e + int(np.argmax(below))]:%m/%d}) → 정리"
            elif d > 120:
                act = "120일 기한 → 정리"
            else:
                act = "보유"
            return {"kind": "돌파", "sig": idx[t], "d": d, "act": act,
                    "line": f"손절 {stop:,.0f} ({stop / c[n] - 1:+.1%}) · 50일선 {ma50[n]:,.0f} ({ma50[n] / c[n] - 1:+.1%}) · D+{d}"}
    for r in range(e - 1, e - 4, -1):                       # 매수일 1~3일 전 복귀
        for b in range(r - 1, r - FAIL_RECOVER - 1, -1):
            if c[b] < Ls[b] and all(c[k] <= Ls[b] for k in range(b, r)) and c[r] > Ls[b]:
                stop, tgt = l[b:r + 1].min(), Hs[b]
                if lo_() <= stop:
                    act = "손절선 닿음 → 정리"
                elif hi_() >= tgt:
                    act = "목표 도달 → 정리"
                elif d > MAX_HOLD:
                    act = f"{MAX_HOLD}일 기한 → 정리"
                else:
                    act = "보유"
                return {"kind": "가짜 이탈", "sig": idx[r], "d": d, "act": act,
                        "line": f"손절 {stop:,.0f} ({stop / c[n] - 1:+.1%}) · 목표 {tgt:,.0f} ({tgt / c[n] - 1:+.1%}) · D+{d}/{MAX_HOLD}",
                        "early": fail_early_line(o, c, e, d, stop) if act == "보유" else None}
    return None


def holding_view(code: str, bars: pd.DataFrame, hold: dict) -> tuple[str, list, str]:
    """보유 종목: (머리 문구, 줄 목록, 등급). 오늘 새로 산다면 기준 — 평단 없음."""
    state = (MARKET.get("state") or {}).get("state")
    pc = xs_pct(trend_components(bars, code))
    lines, grade, verdict = [], "", "판단 없음"
    rule = HOLD_RULE.get(state)
    if rule and rule[0] in pc:
        v = pc[rule[0]]
        if v >= TREND_TOP:
            verdict, grade = rule[1] + (" (강함)" if v >= TREND_STRONG and state == "상승장" else ""), rule[3]   # (강함)은 상승장만(9.93: 횡보 상위 5% 엇갈림)
        elif rule[2] and v <= 1 - TREND_TOP:
            verdict, grade = rule[2], rule[3]
    label = "60일선 위치" if rule and rule[0] == "ma60" else "추세 강도"
    rank = f"{label} 상위 {max(1, round((1 - pc[rule[0]]) * 100))}%" if rule and rule[0] in pc else "추세 강도 계산 전"
    weak = [tag for k, tag in HOLD_WEAK.get(state, []) if pc.get(k, 0) >= TREND_TOP]
    trk = setup_track(bars, hold.get("buy_date"))
    tail = f" ({grade})" if grade else ""
    if trk:
        head = f"보유 · {trk['kind']}({trk['sig']:%m/%d}) → {trk['act']}"
        lines.append(("warn" if trk["act"] != "보유" else "ok", trk["line"]))
        if trk.get("early"):
            lines.append(("info", trk["early"]))
        lines.append(("info", f"{state or '장세'} · {rank} → {verdict}{tail}"))
    else:
        head = f"보유 · {rank} → {verdict}{tail}"
        if hold.get("buy_date") is not None:
            lines.append(("info", f"매수일 {hold['buy_date']:%m/%d}에 셋업 없음 → 일반 보유로 판단"))
    if weak:
        lines.append(("info", "참고(약함): " + " · ".join(weak)))
    return head, lines, grade


def evidence(code: str, bars: pd.DataFrame, st: dict, fl: list, active: list) -> tuple[list, list]:
    """찬성·반대 방향 근거 [(문구, 등급)]. 위험(응축·관리선·거래량)은 방향이 아니라 여기 넣지 않는다."""
    pro, con = [], []
    a = active[0] if active else None
    if a and a["kind"] == "failure" and a.get("t2"):
        pro.append(("T² 동반 가짜 이탈", "검증됨"))
    if a and a["kind"] == "retest":                       # 9.38: 코스피 상승 추세에서만 방향 일치(두 코스피 표본 +0.84·+0.86%p)
        if MARKET["up"]:
            pro.append(("돌파 리테스트 · 코스피 상승 추세", "약함"))
        elif MARKET["up"] is False:
            con.append(("코스피 상승 추세 아닌 돌파", "약함"))
    # 복귀 캔들 종가 위치는 근거로 붙이지 않고 오늘의 후보 승률 모형에 넣었다(9.48)
    if a and a["kind"] == "failure" and MARKET_TREND_GRADE and MARKET["up"]:
        con.append(("코스피 상승 추세 중 혼자 이탈", MARKET_TREND_GRADE))
    sn = sector_net(code)
    if sn and abs(sn[1]) >= SECTOR_NET_CUT:
        (pro if sn[1] > 0 else con).append((f"섹터 애널리스트 {sn[1]:+.0%}", "유망"))
    c, v = bars["close"], bars["volume"]
    ex_today = float(c.iloc[-1] / c.iloc[-2] - 1) if len(c) > 1 else 0.0
    vr = float(v.iloc[-1] / v.iloc[-21:-1].mean()) if len(v) > 21 else 0.0
    if ex_today <= PANIC_DROP and vr >= PANIC_VOL:
        con.append(("투매 당일", "유망"))
    sells = [n for n, t, _ in fl if "순매도" in t]
    buys = [n for n, t, _ in fl if "순매수" in t]
    if sells:
        con.append(("·".join(sells) + " 순매도", "약함"))
    if buys:
        pro.append(("·".join(buys) + " 순매수", "약함"))
    st_m = MARKET.get("state") or {}
    cs_ = current_state(compression_frame(bars)) if st_m.get("state") == "하락장" else None
    if cs_ and cs_["compressed"]:                          # 9.39: 하락장 응축은 20일 뒤 평소보다 −1.9~−2.9%p (4표본 CI < 0)
        con.append(("하락장 응축", "유망"))
    state = st_m.get("state")
    if a and a["kind"] == "failure" and state == "횡보·전환":
        pro.append(("횡보장 가짜 이탈", "최근경향"))                 # 9.85: 2021-06~ +0.58R, 두 반쪽 모두 플러스
    tp = trend_pct(bars) if state in ("상승장", "하락장") else None
    if tp is not None and tp >= TREND_TOP:
        if state == "하락장":
            con.append(("하락장 추세 강도 상위", "검증됨"))          # 9.81: 이후 20일 시장 대비 −1.3~−3.1%p(두 기간)
        else:
            pro.append(("상승장 추세 강도 상위", "최근경향"))        # 9.85: +2.4%p, 두 반쪽 모두 플러스
    tilt = stock_report_tilt(code)
    if tilt:
        (pro if tilt > 0 else con).append((f"종목 리포트 {'상향' if tilt > 0 else '하향'} 우세", "참고"))
    return pro, con


def evidence_verdict(pro: list, con: list) -> str:
    """"찬성 우세 · 신뢰 낮음" 같은 한 마디(텔레그램 요약용). 근거가 없으면 "방향 근거 없음"."""
    wp = sum(EVIDENCE_W[g] for _, g in pro)
    wc = sum(EVIDENCE_W[g] for _, g in con)
    if wp + wc == 0:
        return "방향 근거 없음"
    top = max(EVIDENCE_W[g] for _, g in pro + con)
    trust = "신뢰 보통" if top >= 3 else "신뢰 낮음~보통" if top >= 2 else "신뢰 낮음"
    side = "찬성 우세" if wp > wc * 1.5 else "반대 우세" if wc > wp * 1.5 else "팽팽"
    return f"{side} · {trust}"


def evidence_html(pro: list, con: list) -> str:
    wp = sum(EVIDENCE_W[g] for _, g in pro)
    wc = sum(EVIDENCE_W[g] for _, g in con)
    if wp + wc == 0:
        return '<li class="g-info"><span class="gi">&#8226;</span>방향 근거 없음</li>'
    top = max(EVIDENCE_W[g] for _, g in pro + con)
    trust = "신뢰 보통" if top >= 3 else "신뢰 낮음~보통" if top >= 2 else "신뢰 낮음"
    side = "찬성 우세" if wp > wc * 1.5 else "반대 우세" if wc > wp * 1.5 else "팽팽"
    fmt = lambda xs: " · ".join(f"{html.escape(t)}<span class='gr'>({g})</span>" for t, g in xs) or "–"
    share = wp / (wp + wc)
    return (f'<li class="ev"><span class="gi c-up">+</span><span>찬성 {wp:g}: {fmt(pro)}</span></li>'
            f'<li class="ev"><span class="gi c-dn">−</span><span>반대 {wc:g}: {fmt(con)}</span></li>'
            f'<li class="ev-bar"><span class="gi"></span><span class="evb"><i class="evp" style="width:{share * 100:.0f}%"></i>'
            f'<i class="evc" style="width:{100 - share * 100:.0f}%"></i></span><span class="evt">{side} · {trust}</span></li>')


GUIDE_REPORT_DAYS = 14      # 정기보고서 법정 기한이 이 안이면 알림
REDUCE_FACTOR = 0.5         # 응축·거래량 급증·관리선 밖 중 하나라도 있으면 수량 절반(겹쳐도 절반). 검증값이 아닌 운영 규칙


def guide(bars: pd.DataFrame, st: dict, pl: list[dict], ctl: dict, cs: dict | None, co: dict,
          fl: list[tuple[str, str, str]], vr: float, hold: dict | None = None, code: str = "") -> str:
    """탭 맨 위 결론 가이드. 모든 문구는 검증 결과에서 정한 고정 규칙으로만 만든다(방향 예측 문구 없음).
    등급: warn(행동을 바꿀 것) / ok(확인됨) / info(참고)."""
    H, L, M = box_levels(bars, len(bars) - 1)
    close = float(bars["close"].iloc[-1])
    active = [x for x in pl if x["status"] in ACTIVE]
    lines = []
    sig = ctl["sigma"]
    zs_now = lambda stop: np.log(close / stop) / sig if sig > 0 and stop > 0 else np.nan
    hold_grade = ""
    if hold:
        head, hl, hold_grade = holding_view(code, bars, hold)
        lines += hl
    elif active:
        a = active[0]
        title = NAMES[a["kind"]][0]
        head = (f"{title} · {'복귀 완료 → 내일 시가 매수' if a['status'] == '진입 신호' else '이미 매수 구간 → 지금 가격 매수 가능'}"
                if a["kind"] == "failure" else
                f"{title} · {'확인 완료 → 내일 시가 매수' if a['status'] == '진입 신호' else '이미 매수 구간 → 지금 가격 매수 가능'}"
                ) + (" ★ 검증된 셋업" if a.get("t2") else "")
        if a["kind"] == "failure" and not a.get("t2"):
            lines.append(("info", "T² 없음 — 우위 미확인"))
        z = zs_now(a["stop"])
        rr = (a["target"] - close) / (close - a["stop"]) if close > a["stop"] else None
        if close <= a["stop"]:
            lines.append(("warn", f"이미 손절선 {a['stop']:,.0f} 아래 — 새로 사지 않음"))
        else:
            lines.append(("warn" if z < 1 else "ok",
                          f"지금 {close:,.0f} → 손절 {a['stop']:,.0f} ({a['stop'] / close - 1:+.1%}, {z:.1f}σ) · 목표 {a['target']:,.0f} "
                          f"({a['target'] / close - 1:+.1%}) · 1:{rr:.1f}"
                          + ((" · 손절 가까움 → 수량으로 조절" if a.get("t2") else " · 손절 가까움 → 넓히고 수량 줄이기") if z < 1 else "")))
    elif (MARKET.get("state") or {}).get("state") == "상승장" and (bk := breakout_signal(bars, code)) and (bk["rs"] or 0) >= 0.70:
        head = "돌파 · " + ("확인 완료 → 내일 시가 매수" if bk["first"] else "최근 돌파 · 아직 유효 → 지금 가격 매수 가능")
        lines.append(("ok", bk["info"] + f" · RS {bk['rs'] * 100:.0f}"))
        active = [bk]
    else:
        pos = (close - L) / (H - L) if H > L else 0.5
        up_ = (MARKET.get("state") or {}).get("state") == "상승장"
        head = f"아직 매수 신호 없음 · 박스 {L:,.0f}~{H:,.0f} (위치 {pos:.0%})"
        lines.append(("info", f"매수 조건: {H:,.0f} 위 첫 마감 + RS 70 이상(돌파, 상승장)" if up_ else
                              f"매수 조건: {L:,.0f} 아래 마감 후 {FAIL_RECOVER}일 안 복귀(가짜 이탈)"))
        if st.get("kind") == "failure" and st.get("status") == "복귀 대기":
            lines.append(("warn" if st.get("t2") else "info",
                          f"{st['b_date']:%m/%d} 하단 이탈 중 · {st['L']:,.0f} 위로 마감하면 다음 날 시가 매수"
                          f"{' · T² 동반 → ★ 검증된 셋업' if st.get('t2') else ' · T² 없음'}"))
    ma = MARKET_ACTIONS.get(code)                          # 거래소 시장조치(darthacking) — 손절·가격 제한이 규칙대로 안 될 수 있음
    if ma:
        lines.append(("warn", f"거래소 조치 · {ma['date'][5:].replace('-', '/')} {ma['action'][:40]}"
                              + (" — 손절이 규칙대로 안 될 수 있음" if ma["level"] == "제외" else "")))
    pro, con = ([], []) if hold else evidence(code, bars, st, fl, active)   # 보유는 장세 × 추세 판단 하나로
    if not hold:
        lines.append(("evidence", (pro, con)))
    if any(t == "투매 당일" for t, _ in con):
        lines.append(("warn", "투매 당일 — 복귀 확인 전 매수 보류"))
    # "수량 절반"(응축·관리선 밖·거래량 2배)은 검증 안 된 운영 규칙이라 뺌(9.77: 거래량은 돌파 성과를 못 가름). 응축은 아래 상태 칸에.
    if abs(ctl["z_today"]) > BAND:
        lines.append(("warn", "관리선 밖 마감 — 공시·뉴스 확인"))
    nr = co.get("next_report")
    if nr:
        dd = (nr[1] - bars.index[-1]).days
        if 0 <= dd <= GUIDE_REPORT_DAYS:
            lines.append(("warn", f"실적 기한 D-{dd} ({nr[1]:%m/%d})"))
    icon = {"warn": "&#9888;", "ok": "&#10003;", "info": "&#8226;"}
    items = []
    for kind, body in lines:
        if kind == "evidence":
            items.append(evidence_html(*body))
            continue
        if kind == "size":
            continue
        else:
            items.append(f'<li class="g-{kind}"><span class="gi">{icon[kind]}</span>{html.escape(body)}</li>')
    meta = {"head": head, "verdict": hold_grade if hold else evidence_verdict(pro, con), "verified": bool(active and active[0].get("t2")),
            "panic": any(t == "투매 당일" for t, _ in con), "active": bool(active), "hold": bool(hold)}
    return (f'<div class="guide{" on" if active or hold else ""}"><div class="g-head">{html.escape(head)}</div>'
            f'<ul>{"".join(items)}</ul></div>'), meta


def breakout_signal(bars: pd.DataFrame, code: str) -> dict | None:
    """상승장 관심 종목 — 오늘의 후보와 같은 돌파 규칙(9.62 · 9.68 · 9.72): RS 70 이상 첫 돌파 → 다음날 시가, 손절 돌파일 종가 − 1ATR, 50일선 청산."""
    import screen                                           # 함수 안에서 불러 순환 import 를 피한다(screen 이 이 모듈을 불러 씀)
    x = screen.breakout_first(bars, {}) or screen.breakout_recent(bars, {})
    if not x:
        return None
    rs = xs_pct(trend_components(bars, code)).get("rs")
    return {**x, "rs": rs, "first": bool(screen.breakout_first(bars, {}))}


def stock_section(code: str, name: str, bars: pd.DataFrame, pooled: dict, first: bool, hold: dict | None = None) -> str:
    st = active_state(bars)
    if st.get("kind") == "retest":                          # 리테스트는 지금 전략표에 없음 — 관심 종목도 오늘의 후보와 같은 규칙만
        st = {}
    up = (MARKET.get("state") or {}).get("state") == "상승장"
    pl = [x for x in plans(bars, st) if x["kind"] == "failure" and not up]
    H, L, M = box_levels(bars, len(bars) - 1)
    close = float(bars["close"].iloc[-1])
    apply_basis(pl, close, hold)
    pos = (close - L) / (H - L) if H > L else 0.5
    cs = current_state(compression_frame(bars))
    ctl = control(bars)
    co = company(code, bars)
    zt = ctl["z_today"]
    z_txt = (f'<span class="c-warn">범위 밖 ({zt:+.1f}σ)</span>' if abs(zt) > BAND else f'범위 안 ({zt:+.1f}σ)')
    v20 = bars["volume"].rolling(20).mean().shift(1).iloc[-1]
    vr = float(bars["volume"].iloc[-1] / v20) if v20 and v20 > 0 else float("nan")
    fl = flow_states(code, bars.index)
    fl_html = " · ".join(f'<span class="{c}">{html.escape(n)} {html.escape(t.replace("꾸준한 ", "").replace(" 경보 중", ""))}</span>' for n, t, c in fl) or '<span class="sub">–</span>'
    t2_today = bool(t2_flags(bars)[-1])
    t2_txt = '<span class="c-warn">초과</span>' if t2_today else '<span class="sub">안</span>'
    ck, ctxt = ctl["cusum"]
    c_txt = f'<span class="{"c-up" if ck == "up" else "c-dn" if ck == "down" else "sub"}">{ctxt.replace(" 감지 중", "").replace("추세 변화 감지 없음", "없음").replace("추세 변화", "변화")}</span>'
    val = []
    if co.get("loss"):
        val.append("PER 적자")
    elif co["per"] is not None:
        val.append(f"PER {co['per']:.1f}" + (f" ({co['per_pct']:.0%})" if co["per_pct"] is not None else ""))
    if co["pbr"] is not None:
        val.append(f"PBR {co['pbr']:.2f}" + (f" ({co['pbr_pct']:.0%})" if co["pbr_pct"] is not None else ""))
    ev_txt = " · ".join(f"{d:%m/%d} {n}" for d, n in co["events"]) or "없음"
    sr = stock_reports(code, close)
    nw = stock_news(code, name)
    nr = co["next_report"]
    nr_txt = f"{nr[0]} ~{nr[1]:%m/%d}" if nr else "–"
    bear = (MARKET.get("state") or {}).get("state") == "하락장"
    comp = (f'<span class="c-warn">&#9889; 응축 {cs["streak"]}일째{" · 하락장 → 아래로 풀리기 쉬움" if bear else ""}</span>'
            if cs and cs["compressed"] else '<span class="sub">응축 아님</span>')
    gd, meta = guide(bars, st, pl, ctl, cs, co, fl, vr, hold, code)
    SUMMARY.append({"code": code, "name": name, "date": f"{bars.index[-1]:%Y-%m-%d}", **meta})
    if hold:                                               # 보유 = 지금 상태만: 시나리오 카드·선 대신 현재 위치 숫자
        cv = bars["close"].to_numpy(float)
        comp_ = trend_components(bars, code)
        pc = xs_pct(comp_)
        m50 = cv[-50:].mean() if len(cv) >= 50 else np.nan
        rank_ = lambda k: f"상위 {max(1, round((1 - pc[k]) * 100))}%" if k in pc else "–"
        top_facts = (f'<div><span class="k">50일선</span><span class="v">{m50:,.0f}</span><span class="sub">{close / m50 - 1:+.1%}</span></div>'
                     f'<div><span class="k">60일선 대비</span><span class="v">{comp_.get("ma60", np.nan):+.1%}</span><span class="sub">{rank_("ma60")}</span></div>'
                     f'<div><span class="k">추세 강도</span><span class="v">{rank_("trend")}</span><span class="sub">52주 고가 {comp_.get("hi52", np.nan):+.1%}</span></div>'
                     f'<div><span class="k">5일선 대비</span><span class="v">{comp_.get("ma5", np.nan):+.1%}</span><span class="sub">{rank_("ma5")}</span></div>'
                     f'<div><span class="k">하루 흔들림(ATR)</span><span class="v">{atr_pct(bars):.1%}</span></div>'
                     f'<div><span class="k">변동성</span><span class="v2">{comp}</span></div>')
        pl_chart, plans_html = [], ""
    else:
        top_facts = (f'<div><span class="k">박스 {BOX}일</span><span class="v">{L:,.0f} ~ {H:,.0f}</span><span class="sub">중간 {M:,.0f}</span></div>'
                     f'<div><span class="k">박스 안 위치</span><span class="v">{pos:.0%}</span><div class="bar"><i style="left:{min(max(pos, 0), 1) * 100:.0f}%"></i></div></div>'
                     f'<div><span class="k">하루 흔들림(ATR)</span><span class="v">{atr_pct(bars):.1%}</span></div>'
                     f'<div><span class="k">변동성</span><span class="v2">{comp}</span></div>')
        pl_chart = pl
        plans_html = f'<div class="plans">{"".join(plan_card(code, p, pooled, ctl["sigma"], close, hold) for p in pl)}</div>'

    return f"""<section id="s-{code}" class="stock{' show' if first else ''}">
    {gd}
    <div class="card">
      <div class="head"><div><div class="nm">{html.escape(name)}</div><div class="sub">{code} · {bars.index[-1]:%Y-%m-%d} 종가 기준</div></div>
        <div class="px">{close:,.0f}</div></div>
      <div class="facts">{top_facts}</div>
      {svg_chart(bars, pl_chart, ctl, f"clip{code}")}
      <div class="legend"><span><i class="lg-c"></i>칼만 추세</span><span><i class="lg-b"></i>관리선 ±{BAND:.0f}σ</span>
        <span><i class="lg-d"></i>관리선 밖</span><span><i class="lg-box"></i>박스</span><span><i class="lg-v"></i>거래량 2배↑</span></div>
      <div class="facts ctl">
        <div><span class="k">관리선</span><span class="v2">{z_txt}</span></div>
        <div><span class="k">내일 평소 범위</span><span class="v">{ctl['next_lo']:,.0f} ~ {ctl['next_hi']:,.0f}</span></div>
        <div><span class="k">CUSUM</span><span class="v2">{c_txt}</span></div>
        <div><span class="k">T²</span><span class="v2">{t2_txt}</span></div>
        <div class="flow"><span class="k">수급(참고)</span><span class="v2 muted">{fl_html}</span></div>
        <div><span class="k">거래량</span><span class="v">{vr:.1f}배</span></div>
      </div>
    </div>
    {plans_html}
    <div class="card co">
      <div class="co-row"><span class="k">밸류</span><span>{' · '.join(val) or '–'}</span></div>
      {f'<div class="co-row"><span class="k">리포트</span><span>{html.escape(sr)}</span></div>' if sr else ''}
      {f'<div class="co-row"><span class="k">뉴스</span><span>{nw}</span></div>' if nw else ''}
      <div class="co-row"><span class="k">자본 정책 공시(180일)</span><span>{html.escape(ev_txt)}</span></div>
      <div class="co-row"><span class="k">다음 정기보고서</span><span>{html.escape(nr_txt)}</span></div>
    </div>
    {reference_html(code, bars, close)}
    </section>"""


PAGE = r"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>박스권 손절·수량</title>
<style>
  :root { --bg:#0f172a; --card:#1e293b; --line:#334155; --tx:#e2e8f0; --mut:#94a3b8; --dim:#64748b; --faint:#475569;
          --up:#34d399; --dn:#f87171; --warn:#fbbf24; --acc:#67e8f9; }
  * { box-sizing:border-box; }
  body { background:var(--bg); color:var(--tx); margin:0; padding:16px; max-width:760px; margin:0 auto;
         font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Malgun Gothic",sans-serif; }
  h1 { font-size:1.1rem; margin:0 0 4px; color:#fff; }
  .intro { font-size:0.8rem; color:var(--mut); line-height:1.55; margin:0 0 12px; }
  .budget { display:flex; align-items:center; gap:8px; flex-wrap:wrap; background:var(--card); border:1px solid var(--line);
            border-radius:12px; padding:10px 12px; margin-bottom:12px; font-size:0.85rem; }
  .budget input { width:130px; background:var(--bg); color:var(--tx); border:1px solid var(--line); border-radius:6px;
                  padding:6px 8px; font-size:0.95rem; text-align:right; }
  .tabs { display:flex; gap:6px; overflow-x:auto; margin-bottom:12px; }
  .lv-ma50 { stroke:var(--acc); stroke-width:1.4; stroke-dasharray:4 3; }
  .holdsum { border:1px solid var(--line); background:var(--card); border-radius:12px; padding:10px 12px; margin-bottom:8px; font-size:0.85rem; }
  .holdsum .hs-h { color:var(--dim); font-size:0.75rem; margin-bottom:4px; }
  .holdsum ul { margin:0; padding-left:16px; } .holdsum li { margin:2px 0; } .holdsum b { margin-right:6px; }
  .market { display:flex; flex-wrap:wrap; align-items:baseline; gap:4px 10px; border-radius:12px; padding:10px 12px;
            margin-bottom:8px; font-size:0.85rem; border:1px solid var(--line); background:var(--card); }
  .market b { font-size:0.95rem; } .market.up b { color:var(--up); } .market.dn b { color:var(--dn); } .market.side b { color:var(--warn); }
  .market.up { border-color:rgba(52,211,153,.45); } .market.dn { border-color:rgba(248,113,113,.45); } .market.side { border-color:rgba(251,191,36,.45); }
  .market .md { color:var(--dim); font-size:0.75rem; } .market .mf { color:var(--tx); }
  .market .mx { flex-basis:100%; color:var(--mut); font-size:0.75rem; }
  .market .mb { flex-basis:100%; color:var(--mut); font-size:0.75rem; } .market .mb a { color:inherit; border-bottom:1px dotted var(--dim); text-decoration:none; }
  .market .mn { flex-basis:100%; color:var(--faint); font-size:0.7rem; }
  .tabr { position:absolute; opacity:0; pointer-events:none; }
  .tabs label { flex:0 0 auto; background:var(--card); color:var(--mut); border:1px solid var(--line); border-radius:999px;
                 padding:6px 12px; font-size:0.85rem; cursor:pointer; }
  .tabs label .dot { display:inline-block; width:6px; height:6px; border-radius:50%; background:var(--warn); margin-left:5px; vertical-align:middle; }
  .stock { display:none; }
  .card, .plan { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px; margin-bottom:12px; }
  .head { display:flex; justify-content:space-between; align-items:flex-start; }
  .nm { font-weight:700; color:var(--acc); font-size:1.05rem; } .px { font-size:1.2rem; font-weight:700; }
  .sub { color:var(--dim); font-size:0.75rem; }
  .facts { display:grid; grid-template-columns:1fr 1fr; gap:10px 14px; margin:12px 0; }
  .facts > div { display:flex; flex-direction:column; gap:2px; }
  .k { font-size:0.68rem; color:var(--dim); } .v { font-weight:600; } .v2 { font-size:0.8rem; }
  .bar { position:relative; height:6px; background:linear-gradient(90deg,rgba(248,113,113,.35),rgba(148,163,184,.25),rgba(52,211,153,.35));
         border-radius:3px; margin-top:4px; }
  .bar i { position:absolute; top:-3px; width:3px; height:12px; background:#fff; border-radius:2px; transform:translateX(-50%); }
  .chart { width:100%; height:auto; display:block; margin-top:4px; }
  .chart .box { fill:rgba(251,191,36,0.07); stroke:rgba(251,191,36,0.3); }
  .chart .up { stroke:var(--up); fill:var(--up); } .chart .dn { stroke:var(--dn); fill:var(--dn); }
  .chart .lv-h, .chart .lv-l { stroke:#cbd5e1; stroke-width:1; opacity:.6; }
  .chart .lv-m { stroke:#cbd5e1; stroke-dasharray:4 4; opacity:.5; }
  .chart .lv-tg { stroke:var(--up); stroke-dasharray:2 4; opacity:.6; }
  .chart .lv-st { stroke:var(--dn); stroke-dasharray:2 4; opacity:.7; } .chart .c-st { fill:var(--dn); }
  .chart .cl-c { fill:none; stroke:var(--acc); stroke-width:1.3; opacity:.8; }
  .chart .cl-b { fill:none; stroke:var(--acc); stroke-width:1; stroke-dasharray:3 3; opacity:.45; }
  .chart .brk { fill:var(--warn); }
  .legend { display:flex; flex-wrap:wrap; gap:4px 12px; font-size:0.68rem; color:var(--dim); margin-top:6px; }
  .legend i { display:inline-block; width:14px; height:0; vertical-align:middle; margin-right:4px; border-top:2px solid var(--acc); }
  .legend .lg-b { border-top:1px dashed var(--acc); }
  .legend .lg-d { width:6px; height:6px; border:0; border-radius:50%; background:var(--warn); }
  .chart .vb { opacity:.35; } .chart .vb.hot { opacity:.9; }
  .chart .vavg { fill:none; stroke:var(--mut); stroke-width:1; opacity:.6; }
  .legend .lg-v { height:8px; width:6px; border:0; background:var(--mut); opacity:.9; }
  .legend .lg-box { height:8px; border:1px solid rgba(251,191,36,.4); background:rgba(251,191,36,.08); }
  .facts.ctl { grid-template-columns:1fr 1fr; margin-bottom:0; }
  .guide { background:var(--card); border:1px solid var(--line); border-left:4px solid var(--dim); border-radius:12px;
           padding:12px 14px; margin-bottom:12px; }
  .guide.on { border-left-color:var(--warn); }
  .g-head { font-weight:700; font-size:1rem; color:#fff; margin-bottom:6px; }
  .guide ul { list-style:none; margin:0; padding:0; }
  .guide li { font-size:0.85rem; line-height:1.5; margin:4px 0; display:flex; gap:6px; }
  .guide .gi { flex:0 0 14px; text-align:center; }
  .guide .g-warn { color:var(--warn); } .guide .g-ok { color:var(--up); }
  .guide li.ev { color:var(--tx); } .guide .gr { color:var(--dim); font-size:0.72rem; margin-left:2px; }
  .guide li.ev-bar { align-items:center; } .evb { display:flex; width:120px; height:6px; border-radius:3px; overflow:hidden; background:var(--line); }
  .evb .evp { background:var(--up); } .evb .evc { background:var(--dn); } .evt { color:var(--mut); font-size:0.78rem; }
  .guide .g-info { color:var(--mut); } .guide .g-size { color:var(--tx); } .guide .g-why { color:var(--dim); font-size:0.78rem; }
  .flow .muted { opacity:.75; font-size:0.78rem; line-height:1.5; }
  .co .co-row { display:flex; flex-direction:column; gap:2px; margin-top:8px; font-size:0.85rem; }
  .co .co-row a { color:inherit; border-bottom:1px dotted #64748b; text-decoration:none; }
  details.ref { margin-top:8px; border:1px solid var(--line); border-radius:12px; background:var(--card); padding:8px 12px; font-size:0.8rem; }
  details.ref summary { color:var(--mut); cursor:pointer; }
  details.ref h4 { margin:12px 0 4px; font-size:0.8rem; color:var(--mut); font-weight:600; }
  details.ref table { width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }
  details.ref th, details.ref td { padding:2px 4px; text-align:right; border-bottom:1px solid var(--line); white-space:nowrap; }
  details.ref th:first-child, details.ref td:first-child { text-align:left; }
  details.ref ul { margin:0; padding-left:16px; } details.ref p { margin:2px 0; }
  .chart .lv-t { fill:var(--mut); font-size:11px; } .chart .ax { fill:var(--faint); font-size:10px; }
  .plans { display:grid; grid-template-columns:1fr; gap:0 12px; }
  @media (min-width:640px) { .plans { grid-template-columns:1fr 1fr; } .facts { grid-template-columns:1fr 1fr 1fr 1fr; } }
  .plan.active { border-color:var(--warn); }
  .plan-top { display:flex; align-items:center; gap:6px; flex-wrap:wrap; }
  .plan-title { font-weight:700; } .plan-eng { font-size:0.72rem; color:var(--dim); }
  .badge { margin-left:auto; font-size:0.7rem; color:var(--dim); border:1px solid var(--line); border-radius:999px; padding:1px 8px; }
  .badge.on { color:var(--warn); border-color:rgba(251,191,36,.5); background:rgba(251,191,36,.1); }
  .detail { font-size:0.78rem; color:var(--mut); line-height:1.5; margin:8px 0; }
  table.lv { width:100%; border-collapse:collapse; font-size:0.88rem; }
  table.lv th { text-align:left; font-weight:400; color:var(--dim); font-size:0.75rem; padding:3px 0; width:30%; }
  table.lv td { padding:3px 0; font-weight:600; } table.lv td.sub { font-weight:400; text-align:right; }
  .c-up { color:var(--up); } .c-dn { color:var(--dn); } .c-warn { color:var(--warn); }
  .size { margin-top:10px; padding-top:10px; border-top:1px dashed var(--line); font-size:0.85rem; }
  .size-row { margin:2px 0; } .size b { color:#fff; }
  table.lv tr.rule th, table.lv tr.rule td { color:var(--faint); font-weight:400; font-size:0.75rem; }
  .size-hold { margin-top:10px; padding-top:10px; border-top:1px dashed var(--line); font-size:0.85rem; }
  .plan.verified { border-color:rgba(52,211,153,.6); }
  .badge.vf { color:var(--up); border-color:rgba(52,211,153,.5); background:rgba(52,211,153,.08); }
  .t2 { font-size:0.76rem; color:var(--dim); margin:0 0 8px; line-height:1.5; } .t2.on { color:var(--up); }
  .pooled { margin-top:10px; font-size:0.7rem; color:var(--faint); line-height:1.5; }
  details { font-size:0.75rem; color:var(--faint); margin-top:8px; } details summary { cursor:pointer; color:var(--dim); }
  details p { line-height:1.6; margin:6px 0; }
</style></head>
<body>
  <h1>박스권 손절·수량</h1>
  <p class="intro">보유 = 오늘 새로 산다면(평단 무관) · 관심 = 매수 규칙 기준(지금 가격 / 조건 충족 시 예상 매수가).</p>
  __MARKET__
  __HOLDSUM__
  __RADIOS__<div class="tabs">__TABS__</div>
  __SECTIONS__
  <details><summary>근거와 한계</summary>
    <p>박스 = 최근 __BOX__거래일 고가·저가, 중간선 = 그 가운데. 진입은 조건이 나온 다음 날 시가, 같은 날 손절·목표를 둘 다 닿으면
      손절로 보고, 20거래일 안에 둘 다 안 닿으면 그날 종가로 끝낸다고 가정했습니다.</p>
    <p><b>가짜 이탈 후 복귀는 이탈일에 T²가 관리한계를 넘었을 때만 검증된 셋업입니다</b>(검증이력 9.27): 센서 종목 2016~2026
      거래당 +1.97%p, 사전 등록한 표본 밖(바스켓 밖 코스피 시총 상위 200종목) +1.40%p [+0.05, +2.82]로 재현, 11년 중 8년 플러스.
      T² 없는 가짜 이탈은 우위가 없고, ETF에서는 재현되지 않았습니다. 돌파 리테스트는 센서 종목에서만 기준을 넘었고 ETF·표본 밖
      200종목에서 재현되지 않아 추세장 의존 + 선택 편향으로 봅니다(9.18·9.27). 종목마다 잘 맞는다는 차이는 잡음 수준이었습니다. 손익비는 기댓값을 만들지 못합니다 — 목표를 멀리 둘수록
      목표 도달 확률이 그만큼 낮아집니다.</p>
    <p>관리선 = 칼만 필터 추세의 하루 앞 예측 ± 2σ(그날까지의 데이터만 사용). 관리선 밖으로 마감한 날은 "평소와 다른 상태"라는
      뜻이지 되돌아온다는 신호가 아닙니다 — 과거에 하방 이탈 뒤 반등은 시장 전체 반등이었고, 상방 이탈은 오히려 이어졌습니다
      (검증이력 9.14). CUSUM은 추세 변화를 감지할 뿐 이후 방향을 맞히지 못했습니다. 손절 거리(σ)는 손절선이 하루 평소 흔들림
      안쪽인지 보는 용도입니다. 기업 정보의 공시(자사주·유상증자·CB)도 이후 초과수익과의 관계가 확인되지 않은 사실 정보입니다(9.17).
      거래량도 참고용입니다 — 거래량 실린 돌파·조용한 이탈·위쪽 매물대 모두 기준을 넘지 못했습니다(9.19).</p>
    <p>찬성·반대 무게는 통계로 계산한 확률이 아니라 검증 등급에 따라 정한 값입니다: 검증됨(표본 밖 재현) 3 · 유망(두 표본 일관이나
      간발 미달) 2 · 최근경향 1.5 · 약함(판정 미달, 방향만 일관) 1 · 참고(검증 안 됨·재현 실패) 0.5. 신뢰 문구는 가장 강한 근거의 등급으로 정합니다.
      응축·관리선·거래량은 방향이 아니라 수량 조절 항목이라 여기 넣지 않습니다 — 예외: 코스피 하락장의 응축은 20일 뒤 같은 장세 평소보다 −1.9~−2.9%p로 4개 표본 모두 확실해(9.39) 반대 근거(유망)로 넣습니다. 상승장 응축은 방향이 없었습니다.</p>
    <p>관심 종목의 매수 규칙은 오늘의 후보와 같습니다: 상승장 = RS 70 이상 첫 돌파(다음날 시가, 손절 돌파일 종가 − 1ATR, 50일선 청산), 횡보·하락장 = 가짜 이탈.
      탭의 노란 점 = 오늘 매수 신호 있음.</p>
    <p>수급 흐름은 외국인·프로그램 순매수가 평소와 다르게 꾸준히 쌓이는지(CUSUM 경보)만 보여 줍니다. 하루치 수급은 가격과 같은 날에만
      붙어 있고 다음 날부터는 앞서지 않았습니다(9.15). 꾸준한 순매수·순매도 경보는 외국인에서만 설정과 상관없이 맞는 방향으로 약하게
      기울었지만(20일 평균 초과 차이 약 1%p) 판정 기준에는 못 미쳤고, 기관은 연기금·투신·사모 등으로 나눠도 정보가 없었습니다(9.21·9.22).</p>
    <p>보유 판단(평단 무관): 상승장 = 추세 강도 상위 20% 보유 유리 · 하위 20% 약세(최근경향) / 횡보·전환 = 60일선 위치 상위 20% 되돌림 주의(최근경향) /
      하락장 = 추세 강도 상위 20% 이후 상대 약세 → 이익 실현 고려(검증됨). 상승장 상위 5%는 (강함)(횡보 · 하락장은 상위 5%가 더 강하지 않아 표시 안 함, 9.93). 참고(약함)는 겹치지 않는 요소만(9.81 · 9.85~9.88).
      추세 강도 = 60·120일선 위치 · 52주 고가 대비 · RS의 백분위 평균, 기준은 오늘의 후보 200종목. 매수일을 적은 보유는 그날 셋업 규칙(돌파: 50일선 · 120일 /
      가짜 이탈: 목표 · 20일)을 함께 추적. 가짜 이탈 보유는 매수 후 1~3일째 종가가 진입가 대비 몇 R인지로 손절로 끝날 확률을 표시
      (평소 = 가짜 이탈 전체의 손절 비율 59%. 3일째 −0.5R 아래 82% · +0.5R 이상 29%, 두 기간 같은 계단 — 9.92). 확률이 높아도 남은 기대값은 0 이상이라 일찍 정리하지 않고 손절선만 지킴.</p>
    <p>최근경향 = 2016~2021엔 없거나 반대였지만 최근 5년을 반으로 나눠도 두 쪽 모두 같은 방향인 결과(9.85). 시장이 바뀌면 사라질 수 있어 매년 다시 판정.</p>
    <p>생성 __GENERATED__</p>
  </details>
</body></html>
"""


def hold_summary_html() -> str:
    """맨 위 보유 요약 — 종목당 한 줄(오늘 할 일). 보유 등록이 없으면 비움."""
    rows = [r for r in SUMMARY if r.get("hold")]
    if not rows:
        return ""
    li = "".join(f'<li><b>{html.escape(r["name"])}</b> {html.escape(r["head"].removeprefix("보유 · "))}</li>' for r in rows)
    return f'<div class="holdsum"><div class="hs-h">보유 {len(rows)}</div><ul>{li}</ul></div>'


def build(codes: list[str], fetch: bool = True, holdings: dict | None = None, out_name: str = "risk_scenarios.html",
          title: str = "박스권 매수·손절", lead: str = "", summary_name: str = "summary.json") -> str:
    SUMMARY.clear()
    pooled = pooled_outcomes()
    holdings = target_holdings() if holdings is None else holdings
    tabs, secs, radios, tab_css = [], [], [], []
    caller = None
    if fetch and any(not in_baskets(c) for c in codes):
        from kis_client import RateLimitedCaller
        caller = RateLimitedCaller()
    if fetch and caller is None:
        from kis_client import RateLimitedCaller
        caller = RateLimitedCaller()
    try:
        import market_actions
        MARKET_ACTIONS.clear()
        MARKET_ACTIONS.update(market_actions.active())
    except Exception as e:
        print(f"시장조치 읽기 실패: {e}", file=sys.stderr)
    try:                                   # 시장 추세(9.37) — 069500 일봉을 data/_scenario 에 증분 저장
        _, kb = load_stock("069500", caller)
        MARKET.update(up=kospi_uptrend(kb), date=kb.index[-1], state=market_state(kb))
    except Exception as e:
        print(f"코스피 추세 계산 실패: {e}", file=sys.stderr)
    codes = [c for c in codes if c in holdings] + [c for c in codes if c not in holdings]   # 보유 먼저
    for i, code in enumerate(codes):
        if fetch:
            refresh_flows(caller, code)
        try:
            name, bars = load_stock(code, caller if not in_baskets(code) else None)
        except Exception as e:  # 잘못된 코드 하나로 페이지 전체를 멈추지 않는다
            print(f"건너뜀 {code}: {e}", file=sys.stderr)
            continue
        hold = holdings.get(code)
        sec = stock_section(code, name, bars, pooled, i == 0, hold)
        active = bool(SUMMARY[-1]["active"]) if SUMMARY else False
        # 탭은 자바스크립트 없이 동작(숨긴 라디오 + CSS) — 텔레그램·휴대폰 HTML 뷰어는 스크립트를 막는 경우가 있다
        radios.append(f'<input type="radio" name="stab" class="tabr" id="t-{code}"{" checked" if not radios else ""}>')
        tabs.append(f'<label for="t-{code}">{html.escape(name)}'
                    f'{" · 보유" if hold else ""}{"<span class=dot></span>" if active and not hold else ""}</label>')
        tab_css.append(f'#t-{code}:checked ~ #s-{code} {{ display:block; }} '
                       f'#t-{code}:checked ~ .tabs label[for="t-{code}"] {{ color:var(--bg); background:var(--acc); border-color:var(--acc); font-weight:700; }}')
        secs.append(sec)
    m = MARKET.get("state")
    (results_dir() / "scenario").mkdir(parents=True, exist_ok=True)
    (results_dir() / "scenario" / "market.json").write_text(json.dumps(
        {**m, "date": f"{m['date']:%Y-%m-%d}"} if m else {}, ensure_ascii=False), encoding="utf-8")
    page = (PAGE.replace("<title>박스권 손절·수량</title>", f"<title>{html.escape(title)}</title>")
            .replace("<h1>박스권 손절·수량</h1>", f"<h1>{html.escape(title)}</h1>")
            .replace("__MARKET__", market_html(m) + lead).replace("__HOLDSUM__", hold_summary_html())
            .replace("__RADIOS__", "".join(radios) + f'<style>{"".join(tab_css)}</style>').replace("__TABS__", "".join(tabs)).replace("__SECTIONS__", "".join(secs))
            .replace("__BUDGET__", f"{DEFAULT_BUDGET:,}").replace("__BOX__", str(BOX))
            .replace("__GENERATED__", datetime.now(KST).strftime("%Y-%m-%d %H:%M")))
    out = results_dir() / "scenario" / out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    (out.parent / summary_name).write_text(json.dumps(SUMMARY, ensure_ascii=False, indent=1), encoding="utf-8")
    return str(out)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    print(build(args or target_codes(), fetch="--no-fetch" not in sys.argv))
