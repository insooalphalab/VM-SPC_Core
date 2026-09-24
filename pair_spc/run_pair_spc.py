"""PAIR_SPC v2 1-Click 실행: Stage0(대표종목 실증) → Stage1/2(공적분·SPC) → 대시보드 반영.

기존 v2_run.py(예측모델)와 별도 파이프라인이다. 이미 v2_run.py로 만들어진 dashboard_v2.html에
"대표종목 상관관계(PAIR-SPC)" 탭을 추가로 렌더링한다(v2_render_dashboard.render 가 pair 결과가
있으면 자동으로 탭을 얹는다) — 예측모델 탭 내용은 그대로 유지된다.

  python run_pair_spc.py                       # basket_watchlist.json 전체 바스켓
  python run_pair_spc.py --basket <바스켓명>
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "vm_predict", _ROOT / "pair_spc"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import argparse
import logging
import sys

from pair_config import PairParams
from pair_cointegration import run as run_cointegration
from pair_representativeness import run as run_representativeness
from v2_config import get_basket, load_baskets
from v2_render_dashboard import render_basket

log = logging.getLogger("run_pair_spc")


def run_one(basket: dict, p: PairParams) -> None:
    log.info("=== PAIR_SPC '%s' (타겟 %s) ===", basket["name"], basket["target"]["name"])
    try:
        run_representativeness(basket, p)
    except Exception:
        log.exception("바스켓 '%s': Stage0 실패 — 건너뜀", basket["name"])
        return
    try:
        run_cointegration(basket, p)
    except Exception:
        log.exception("바스켓 '%s': Stage1/2 실패 — 건너뜀", basket["name"])
        return
    render_basket(basket)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--basket")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    p = PairParams()
    baskets = [get_basket(args.basket)] if args.basket else load_baskets()
    for basket in baskets:
        run_one(basket, p)
    log.info("PAIR_SPC 전체 완료")
    return 0


if __name__ == "__main__":
    sys.exit(main())
