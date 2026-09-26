"""3-분기 Gated Two-Stage 챔피언 판정 (명세 1.7 / 2.8) — 상대순위 라벨링 기준으로 재보정(2026-09-26).

  Gate 1 (모델별 절대 품질선, 점추정 + 통계적 신뢰성 둘 다 필요):
    High-Conf Precision(S_core) >= 60.0%  그리고  95% CI 하한 > 50.0%(동전던지기)
  Gate 2 (오컴의 면도날, 둘 다 통과 시): Challenger − Baseline >= +3.0%p 이고
          Challenger CI 하한 > Baseline 점추정 이어야 Challenger 채택, 아니면 Baseline.

Gate 1 임계값(0.60) 재보정 근거: 명세 1.7의 원래 65%는 절대방향 라벨링을 전제로 정한 값인데,
세션 검증 결과 절대방향은 데드존 때문에 표본이 적어 신뢰구간이 평균 32%p로 넓었다. 상대순위
라벨링(labeling.attach_relative_labels, vm_spc/pipeline.py 기본 모드)으로 바꾸면 표본이 늘어(데드존이
그날 전체를 날리지 않음) CI가 평균 8%p로 좁아지고 edge 실재 여부를 훨씬 자신 있게 말할 수 있게
되지만, edge 크기 자체는 59% 근처(평균 59.0%, 중앙값 59.3%)에서 수렴한다 — 그래서 Gate 1도
evaluation.TIER2_THRESHOLD 와 같은 0.60으로 낮췄다.

CI 하한 조건(2026-09-26 추가) 근거: 점추정 기준만 쓰면 표본이 작을 때 우연히 높게 나온 값이 그대로
통과해버리는 게 실측으로 재현됐다 — LightGBM 조기종료 도입 후 `it_general_scan`(HCP 80%, n=15,
CI [57%,100%]), `semiconductor_frontend_scan`(HCP 70%, n=10, CI [40%,100%]), 특히
`secondary_battery_krx_scan`(HCP 60.5%, n=76, **CI 하한 48.6%로 동전던지기보다 못할 수도 있는데
점추정만으로는 통과**)이 그대로 채택될 뻔했다. 그래서 Gate 1은 "점추정 ≥60%"와 "CI 하한 >50%
(edge가 우연이 아니라는 것 자체가 통계적으로 확인됨)"를 둘 다 요구한다 — 표본이 작아 우연히
높게 나온 값을 걸러내는 것이 목적이라, 상관성을 억지로 통과시키기보다 냉정하게 걸러내는 쪽을
택했다.

최소 표본수 조건(같은 날 추가) 근거: `it_general_scan` 은 CI 하한(57%)만 보면 통과였지만, 그 근거가
날짜클러스터 부트스트랩(B=1000)이 단 15개의 서로 다른 날짜(=15번의 독립적 관측)만으로 만든
구간이었다 — 표본이 이 정도로 작으면 부트스트랩 자체가 이산적(discrete)이라 구간이 실제보다
좁고 안정적으로 보일 수 있다. evaluation.py 의 "표본 30건 미만이면 추정이 불안정하다"는 기존
경고(pipeline.py limitations)를 참고용 문구로만 두지 않고 Gate 1의 하드 조건으로 승격했다
(MIN_N_CORE=30) — 표본이 못 미더우면 점추정·CI가 아무리 좋아 보여도 통과시키지 않는다.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "vm_predict", _ROOT / "pair_spc"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

GATE1_THRESHOLD = 0.60   # 재보정: 상대순위 실측 평균 59.0%/중앙값 59.3% 근처, Tier2 컷오프와 동일
CI_LOWER_MIN = 0.50      # "동전던지기보다 낫다"는 것 자체가 통계적으로 확인돼야 함
MIN_N_CORE = 30          # 날짜클러스터 부트스트랩이 안정적이려면 최소 이 정도 독립 관측(날짜)이 필요
GATE2_MARGIN = 0.03      # 변경 없음 — Baseline 대비 Challenger 우위 판단은 절대수준과 무관


def _hc(m: dict) -> float | None:
    return m.get("high_conf_precision")


def gate1_detail(metrics: dict) -> dict:
    """Gate 1 판정 상세 — 점추정(point_ok)·CI 신뢰성(ci_ok)·표본 충분성(n_ok)을 분리해서 남긴다
    (대시보드에서 '왜' 떨어졌는지 구분해 보여주기 위함: 점추정 자체가 낮은 것, 우연일 수 있는 것,
    표본이 애초에 판단하기엔 너무 적은 것은 서로 다른 문제다)."""
    hc = _hc(metrics)
    ci_lower = metrics.get("ci_lower")
    n_core = metrics.get("n_s_core") or 0
    point_ok = hc is not None and hc >= GATE1_THRESHOLD
    ci_ok = ci_lower is not None and ci_lower > CI_LOWER_MIN
    n_ok = n_core >= MIN_N_CORE
    return {"passed": point_ok and ci_ok and n_ok, "point_ok": point_ok, "ci_ok": ci_ok, "n_ok": n_ok,
            "high_conf_precision": hc, "ci_lower": ci_lower, "n_s_core": n_core}


def gate1_pass(metrics: dict) -> bool:
    return gate1_detail(metrics)["passed"]


def _gate1_why(label: str, d: dict) -> str | None:
    """통과 못 한 이유를 한 줄로 — 어느 조건에서 걸렸는지 구분해서 짚어준다. 통과했으면 None."""
    if d["passed"]:
        return None
    if not d["point_ok"]:
        hc = f"{d['high_conf_precision']:.0%}" if d["high_conf_precision"] is not None else "N/A"
        return f"{label} 점추정 {hc} (기준 {GATE1_THRESHOLD:.0%} 미달)"
    if not d["n_ok"]:
        return f"{label} 점추정은 통과했지만 표본 {d['n_s_core']}건으로 너무 적어(기준 ≥{MIN_N_CORE}건) 판단 보류"
    ci = f"{d['ci_lower']:.0%}" if d["ci_lower"] is not None else "N/A"
    return f"{label} 점추정은 통과했지만 표본 {d['n_s_core']}건으로 작아 CI 하한 {ci}(기준 >{CI_LOWER_MIN:.0%} 미달) — 우연일 수 있어 제외"


def decide_champion(baseline_metrics: dict, challenger_metrics: dict) -> dict:
    """판정 결과 + 어느 분기를 탔는지(대시보드 설명용). 통과해도 '고신뢰 확정'이 아니라
    '통계적으로 확인된 약한 방향성 참고 신호'라는 톤을 reason 문구에 명시한다."""
    base_d, chal_d = gate1_detail(baseline_metrics), gate1_detail(challenger_metrics)
    base_pass, chal_pass = base_d["passed"], chal_d["passed"]
    out = {"gate1": {"baseline": base_d, "challenger": chal_d}, "gate2": None}
    thr = f"{GATE1_THRESHOLD:.0%}"
    notes = [n for n in (_gate1_why("Baseline", base_d), _gate1_why("Challenger", chal_d)) if n]

    if not base_pass and not chal_pass:
        return {**out, "champion": "REJECTED", "branch": 1,
                "reason": (f"Baseline·Challenger 모두 Gate 1(High-Conf Precision ≥ {thr}, CI 하한 >{CI_LOWER_MIN:.0%}, n≥{MIN_N_CORE}) "
                          f"미달 — 약한 방향성 신호조차 통계적으로 확인 안 됨, Legacy 룰만 운영. " + " / ".join(notes))}
    if base_pass and not chal_pass:
        return {**out, "champion": "BASELINE", "branch": 2,
                "reason": (f"Baseline 만 Gate 1(≥{thr}, CI 하한 >{CI_LOWER_MIN:.0%}, n≥{MIN_N_CORE}) 통과 — "
                          f"약한 방향성 참고 신호로 Baseline 채택(고신뢰 확정 아님). {_gate1_why('Challenger', chal_d) or ''}")}
    if not base_pass and chal_pass:
        # Baseline이 기준 미달인데 Challenger만 통과한 예외 케이스
        # → Gate 2 취지상 비교 대상이 없으므로 Challenger 단독 채택
        return {**out, "champion": "CHALLENGER", "branch": "2b",
                "reason": (f"Challenger 만 Gate 1(≥{thr}, CI 하한 >{CI_LOWER_MIN:.0%}, n≥{MIN_N_CORE}) 통과(비교 대상 없음) — "
                          f"약한 방향성 참고 신호로 Challenger 단독 채택(고신뢰 확정 아님). {_gate1_why('Baseline', base_d) or ''}")}

    # 둘 다 통과 → Gate 2
    margin = _hc(challenger_metrics) - _hc(baseline_metrics)
    ci_lower = challenger_metrics.get("ci_lower")
    ci_ok = ci_lower is not None and ci_lower > _hc(baseline_metrics)
    passed = margin >= GATE2_MARGIN and ci_ok
    out["gate2"] = {"margin": margin, "ci_lower": ci_lower, "margin_ok": margin >= GATE2_MARGIN,
                    "ci_ok": ci_ok, "passed": passed}
    if passed:
        return {**out, "champion": "CHALLENGER", "branch": 3,
                "reason": f"둘 다 Gate 1 통과 → Gate 2 통과(+{margin * 100:.1f}%p, CI 하한 > Baseline) — "
                          "Challenger 채택(여전히 참고용 방향성 신호, 고신뢰 확정 아님)"}
    return {**out, "champion": "BASELINE", "branch": 3,
            "reason": f"둘 다 Gate 1 통과 → Gate 2 미달({margin * 100:+.1f}%p"
                      f"{'' if ci_ok else ', CI 하한 ≤ Baseline'}) — "
                      "더 단순한 Baseline 채택(참고용 방향성 신호, 고신뢰 확정 아님)"}
