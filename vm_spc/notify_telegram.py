"""일일 파이프라인 완료 후 텔레그램으로 요약을 보낸다.

`.streamlit/secrets.toml` 의 `TELEGRAM_BOT_TOKEN`·`TELEGRAM_CHAT_ID` 를 쓴다
(core.secrets_loader — 이미 프로젝트에 설정돼 있었음, 표준 Telegram Bot API 사용).
`vm_spc/build_index.py` 와 같은 데이터(`collect_rows()`)를 재사용해 "오늘 결론 먼저" 원칙 그대로
검증된 방향성 신호를 상위 몇 개만 요약해서 보내고, `results/index.html` 을 파일로 첨부한다.

  python vm_spc/notify_telegram.py                    # 다이제스트 전송
  python vm_spc/notify_telegram.py --dry-run           # 전송 없이 내용만 출력(테스트용)
  python vm_spc/notify_telegram.py --run-status warn   # 파이프라인 단계 중 실패가 있었음을 표시
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "vm_predict", _ROOT / "pair_spc"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import argparse
import json
import logging
import sys
from datetime import datetime

import time

import requests

from secrets_loader import get_secret
from v2_config import KST, load_baskets, results_dir
from vm_spc.build_index import collect_rows

log = logging.getLogger("vm_spc.notify_telegram")

TOP_N = 8
STATE_PATH = _ROOT / "logs" / "last_notified_date.txt"


def _latest_data_date() -> str | None:
    dates = [r["etf_pred"]["date"] for r in collect_rows() if r["etf_pred"]]
    return max(dates) if dates else None


def _read_state() -> str | None:
    return STATE_PATH.read_text(encoding="utf-8").strip() if STATE_PATH.exists() else None


def build_digest(run_status: str = "ok") -> str:
    rows = collect_rows()
    n_configured = len(load_baskets())
    n_ready = sum(1 for r in rows if r["champion"] is not None)

    validated = sorted(
        (r for r in rows if r["top"] is not None and r["top_validated"]),
        key=lambda r: -r["gap"],
    )[:TOP_N]
    etf_hits = [r for r in rows if r["etf_pred"] and r["etf_pred"]["hit_rate"] is not None
               and r["etf_pred"]["hit_rate"] >= 0.6]

    lines = []
    if run_status != "ok":
        lines.append("⚠️ 파이프라인 일부 단계에서 오류가 있었습니다 — logs/ 폴더 확인 필요\n")

    today = datetime.now(KST).strftime("%Y-%m-%d %H:%M")
    lines.append(f"📊 VM-SPC Core 일일 업데이트 — {today}")
    lines.append(f"구성종목 검증 완료 {n_ready}/{n_configured}개 ETF (신규상장 등 데이터 부족으로 검증 대기 {n_configured - n_ready}개)")
    lines.append("")

    if validated:
        lines.append(f"🔹 구성종목 방향성 신호 (검증됨, 상위 {len(validated)}개)")
        for r in validated:
            top = r["top"]
            up = r["top_p"] >= 0.5
            conf = r["top_p"] if up else (1 - r["top_p"])
            arrow = "▲" if up else "▼"
            lines.append(f"{arrow} {top['name']} {conf:.0%} — {r['etf_name']}")
        lines.append("")
    else:
        lines.append("🔹 오늘은 게이트를 통과한 구성종목 신호가 없습니다 (약한 참고치만 존재).")
        lines.append("")

    lines.append(f"🔸 ETF 자체 예측 적중률 60% 이상: {len(etf_hits)}개")
    if etf_hits:
        for r in etf_hits[:TOP_N]:
            ep = r["etf_pred"]
            arrow = "▲" if ep["pred_up"] else "▼"
            usual = f", 평소 {ep['usual']:.0%}" if ep.get("usual") is not None else ""
            lines.append(f"{arrow} {r['etf_name']} (적중 {ep['hit_rate']:.0%}{usual})")

    lines.append("")
    lines.append("자동 주문 없음 · 참고용 방향 신호입니다. 전체 목록은 첨부한 index.html 참고.")
    return "\n".join(lines)


def build_stock_digest() -> str | None:
    """관심 종목 전략(손절·수량 페이지) 요약 — 종목당 한 줄, 검증된 셋업·규칙 충족·보유 종목을 위로. 대기 종목은 한 줄로 묶음."""
    p = results_dir() / "scenario" / "summary.json"
    if not p.exists():
        return None
    rows = json.loads(p.read_text(encoding="utf-8"))
    if not rows:
        return None
    rank = lambda r: (not r["hold"], not r["verified"], not r["panic"], not r["active"])
    lines = [f"🎯 보유·관심 종목 — {rows[0]['date'][5:].replace('-', '/')} 종가"]
    mp = results_dir() / "scenario" / "market.json"
    m = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
    if m.get("state"):
        lines.append(f"📊 코스피 {m['state']} {m['days']}일째 — {m['favor']}")
    bp = results_dir() / "reports" / "market_brief.json"
    br = json.loads(bp.read_text(encoding="utf-8")) if bp.exists() else {}
    if br.get("title"):
        lines.append(f"📰 시황({br['date'][5:10].replace('-', '/')} {br['kind']}): {br['title']}")
    waiting = []
    for r in sorted(rows, key=rank):
        if r["hold"]:
            lines.append(f"🔹 {r['name']}: {r['head'].removeprefix('보유 · ')}")
        elif r["verified"] or r["panic"] or r["active"]:
            mark = "★" if r["verified"] else "⚠" if r["panic"] else "•"
            lines.append(f"{mark} {r['name']}: {r['head']} ({r['verdict']})")
        else:
            waiting.append(r["name"] + ("" if r["verdict"] == "방향 근거 없음" else f"({r['verdict']})"))
    if waiting:
        lines.append("대기: " + ", ".join(waiting))
    lines.append("매수·손절 가격은 첨부한 risk_scenarios.html")
    return "\n".join(lines)


def build_screen_digest() -> str | None:
    """오늘의 후보(scenario/screen.py) — 코스피 시총 상위 200에서 장세에 맞는 규칙 충족 종목 이름만 한 줄씩."""
    p = results_dir() / "scenario" / "screen.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text(encoding="utf-8"))
    def one(x):
        dep = x.get("depth")
        tag = "" if dep is None else f"({dep:.1%} 이탈)" if x.get("star") else f"(⚠얕은 이탈 {dep:.1%})"
        return (x.get("star") or "") + x["name"] + tag + (f" {x['win']:.0%}" if x.get("win") is not None else "")
    names = lambda xs: ", ".join(one(x) for x in xs) or "없음"
    body = "\n".join(f"{ls['label']}: {names(ls['items'])}" for ls in r.get("lists", []))
    return (f"🔎 오늘의 후보 — 코스피 {r.get('state')} · {r.get('strategy')} (코스피 시총 상위 200)\n{body}\n"
            "승률 높은 순 · ★ 이탈 6%↑ ☆ 3.5~6% (얕은 이탈 제외) · 규칙 충족 목록, 추천 아님")


def _retry(fn, tries: int = 3):
    """연결이 잠깐 끊겨도(2026-10-02 ConnectionResetError로 세 번째 메시지 누락) 5 · 15초 쉬고 다시 시도."""
    for k in range(tries):
        try:
            return fn()
        except requests.exceptions.RequestException:
            if k == tries - 1:
                raise
            log.warning("텔레그램 전송 실패 — %d초 뒤 다시 시도", (5, 15)[k])
            time.sleep((5, 15)[k])


def send_message(text: str) -> None:
    token = get_secret("TELEGRAM_BOT_TOKEN")
    chat_id = get_secret("TELEGRAM_CHAT_ID")

    def go():
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          data={"chat_id": chat_id, "text": text}, timeout=15)
        r.raise_for_status()
    _retry(go)


def send_document(path: _Path, caption: str = "") -> None:
    token = get_secret("TELEGRAM_BOT_TOKEN")
    chat_id = get_secret("TELEGRAM_CHAT_ID")

    def go():
        with open(path, "rb") as f:
            r = requests.post(f"https://api.telegram.org/bot{token}/sendDocument",
                              data={"chat_id": chat_id, "caption": caption},
                              files={"document": (path.name, f, "text/html")}, timeout=30)
        r.raise_for_status()
    _retry(go)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="전송 없이 다이제스트 내용만 출력")
    ap.add_argument("--run-status", choices=["ok", "warn"], default="ok",
                    help="warn 이면 파이프라인 단계 실패가 있었다는 경고를 메시지 맨 위에 붙인다")
    ap.add_argument("--force", action="store_true", help="새 거래일이 없어도 전송")
    ap.add_argument("--only", choices=["etf", "stock", "screen"], help="한 묶음만 다시 보낼 때(전송 실패 복구용)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    digest = build_digest(args.run_status)
    if args.dry_run:
        print(digest)
        print("\n" + (build_stock_digest() or "(관심 종목 요약 없음)"))
        print("\n" + (build_screen_digest() or "(오늘의 후보 없음)"))
        return 0

    # 공휴일(평일 휴장)에도 스케줄은 돌기 때문에, 마지막으로 알린 거래일과 최신 데이터 거래일이
    # 같으면 같은 내용을 또 보내지 않는다. 실패 경고(warn)는 항상 보낸다.
    latest = _latest_data_date()
    if args.run_status == "ok" and not args.force and latest and _read_state() == latest:
        log.info("새 거래일 없음(최신 %s, 이미 알림) — 전송 생략", latest)
        return 0

    if args.only:                                           # 빠진 묶음만 다시 — 마지막 알림 기록은 건드리지 않음
        if args.only == "etf":
            send_message(digest)
            send_document(results_dir() / "index.html", caption="전체 ETF 현황")
        elif args.only == "stock":
            send_message(build_stock_digest())
            send_document(results_dir() / "scenario" / "risk_scenarios.html", caption="관심 종목 매수·손절 가이드")
        else:
            send_message(build_screen_digest())
            send_document(results_dir() / "scenario" / "screen.html", caption="오늘의 후보 — 관심 종목과 같은 양식")
        log.info("텔레그램 %s 다시 보냄", args.only)
        return 0

    try:
        send_message(digest)
        index_path = results_dir() / "index.html"
        if index_path.exists():
            send_document(index_path, caption="전체 ETF 현황 — 상세 대시보드는 구글 드라이브 '내 드라이브/VM-SPC_Core/results'에서 확인")
        # 두 번째 대시보드: 관심 종목 손절·수량 가이드(요약 + 파일)
        stock = build_stock_digest()
        risk_path = results_dir() / "scenario" / "risk_scenarios.html"
        if stock and risk_path.exists():
            send_message(stock)
            send_document(risk_path, caption="관심 종목 매수·손절 가이드")
        # 세 번째: 오늘의 후보(코스피 200 스크리닝)
        scr = build_screen_digest()
        scr_path = results_dir() / "scenario" / "screen.html"
        if scr:
            send_message(scr)
            if scr_path.exists():
                send_document(scr_path, caption="오늘의 후보 — 관심 종목과 같은 양식")
        log.info("텔레그램 전송 완료")
    except Exception:
        log.exception("텔레그램 전송 실패")
        return 1
    if latest:
        STATE_PATH.write_text(latest, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
