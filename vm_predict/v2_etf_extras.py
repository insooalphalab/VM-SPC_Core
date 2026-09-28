"""ETF 자체 예측 보강 두 가지(괴리율 조건 + 신뢰도 지수)를 전 바스켓에 계산하고, 전 바스켓 풀링으로
실제 효과가 있는지 검증한다.

  results/{basket}/etf_extras.json      — 오늘 예측의 괴리율 조건·조건부 적중률·신뢰도 등급
  results/etf_extras_validation.json    — 풀링 검증 결과. validated=false 인 항목은 화면에 쓰지 않는다

검증 기준(기존 Gate 와 같은 원칙): 괴리율은 "동의" 날 적중률 - "반대" 날 적중률, 신뢰도는 "높음" -
"낮음" 의 95% 신뢰구간(날짜 클러스터 부트스트랩) 하한이 0 보다 커야 통과.

  python vm_predict/v2_etf_extras.py        # v2_run.py 가 끝에서 자동 호출한다
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "vm_predict", _ROOT / "pair_spc"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import json
import logging
import sys
from datetime import datetime

import numpy as np
import pandas as pd

import v2_premium as prem
import v2_reliability as rel
from v2_config import KST, load_baskets, results_dir

log = logging.getLogger("v2_etf_extras")

N_BOOT = 1000
VALIDATION_FILE = "etf_extras_validation.json"


def _series(block: dict, key: str) -> pd.Series:
    return pd.Series(block[key], index=pd.to_datetime(block["dates"]))


def basket_frame(basket: dict, summary: dict) -> pd.DataFrame:
    close = _series(summary["target_series"], "close").astype(float)
    breadth = _series(summary["breadth_series"], "breadth").astype(float).reindex(close.index)
    dead_zone = summary["params"]["dead_zone"]
    threshold = summary["params"]["breadth_threshold"]

    df = pd.DataFrame({"close": close, "breadth": breadth})
    df["pred_up"] = (df["breadth"] >= threshold).where(df["breadth"].notna())
    df["next_ret"] = df["close"].shift(-1) / df["close"] - 1
    resolved = df["pred_up"].notna() & df["next_ret"].notna() & (df["next_ret"].abs() >= dead_zone)
    df["hit"] = ((df["pred_up"] == (df["next_ret"] > 0)).astype(float)).where(resolved)

    p = prem.premium_series(basket["name"], basket["target"]["code"])
    df["dprt"] = p["dprt"].reindex(df.index) if p is not None else np.nan
    df["dz"] = p["dz"].reindex(df.index) if p is not None else np.nan
    df["agreement"] = prem.agreement(df["pred_up"], df["dz"])

    t2 = _series(summary["t2_series"], "t2").astype(float)
    df = df.join(rel.reliability_frame(df["close"], t2, df["hit"]))
    return df


def premium_adjustment(pool: pd.DataFrame) -> dict[str, float]:
    """괴리율 상태별 적중률 - 전체 적중률 (전 ETF 합산). ETF별로 쪼개면 조건당 표본이 약 40일이라
    우연 변동이 효과보다 커서, 효과 크기는 전체 풀에서만 추정하고 ETF에는 이 차이만 더한다."""
    sub = pool.dropna(subset=["agreement"])
    overall = sub["hit"].mean()
    return {s: round(float(g["hit"].mean() - overall), 4) for s, g in sub.groupby("agreement")}


def today_extras(df: pd.DataFrame, adjust: dict[str, float] | None) -> dict | None:
    valid = df[df["pred_up"].notna()]
    if valid.empty:
        return None
    d = valid.index[-1]
    row = valid.loc[d]
    pred_up = bool(row["pred_up"])
    resolved = df[df["hit"].notna()]
    same_dir = resolved[resolved["pred_up"] == pred_up]
    base_hit = float(same_dir["hit"].mean()) if len(same_dir) else None
    # "평소" = 예측과 무관하게 다음 날 그 방향으로 움직인 날의 비율(데드존 제외) — 강세장에서 "상승 적중"이
    # 높게 나오는 걸 실력으로 착각하지 않도록 같이 보여준다.
    usual = float((resolved["next_ret"] > 0).mean() if pred_up else (resolved["next_ret"] < 0).mean()) if len(resolved) else None
    out = {"date": d.strftime("%Y-%m-%d"), "pred_up": pred_up, "base_hit_rate": base_hit, "n": int(len(same_dir)),
           "usual_rate": usual, "premium": None, "reliability": None}
    if pd.notna(row["dz"]) and adjust and base_hit is not None and row["agreement"] in adjust:
        state = row["agreement"]
        out["premium"] = {"dprt": round(float(row["dprt"]), 3), "dz": round(float(row["dz"]), 2), "state": state,
                          "adjust": adjust[state], "hit_rate": min(max(base_hit + adjust[state], 0.0), 1.0)}
    if row["grade"] is not None and pd.notna(row["grade"]):
        out["reliability"] = {"grade": row["grade"], "t2_pct": round(float(row["t2_pct"]), 3),
                              "vol_pct": round(float(row["vol_pct"]), 3),
                              "cusum_alarm": bool(row["cusum_alarm"]), "reasons": rel.reasons(row)}
    return out


def _boot_diff(pool: pd.DataFrame, col: str, a: str, b: str, seed: int = 0) -> dict:
    """a 그룹 적중률 - b 그룹 적중률과 날짜 클러스터 부트스트랩 95% CI."""
    sub = pool[pool[col].isin([a, b])]
    if sub.empty:
        return {"diff": None, "ci": [None, None], "validated": False}
    g = sub.groupby(["date", col])["hit"].agg(["sum", "count"]).unstack(col, fill_value=0)
    sa, ca = g[("sum", a)].to_numpy(), g[("count", a)].to_numpy()
    sb, cb = g[("sum", b)].to_numpy(), g[("count", b)].to_numpy()
    if ca.sum() == 0 or cb.sum() == 0:
        return {"diff": None, "ci": [None, None], "validated": False}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(N_BOOT, len(g)))
    boot = sa[idx].sum(1) / np.maximum(ca[idx].sum(1), 1) - sb[idx].sum(1) / np.maximum(cb[idx].sum(1), 1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    diff = sa.sum() / ca.sum() - sb.sum() / cb.sum()
    return {"diff": round(float(diff), 4), "ci": [round(float(lo), 4), round(float(hi), 4)], "validated": bool(lo > 0)}


def _group_rates(pool: pd.DataFrame, col: str) -> dict:
    return {k: {"hit_rate": round(float(v["hit"].mean()), 4), "n": int(len(v))}
            for k, v in pool.groupby(col) if len(v)}


def run_all() -> int:
    frames, built = [], []
    for basket in load_baskets():
        path = results_dir() / basket["name"] / "v2_summary_metrics.json"
        if not path.exists():
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        df = basket_frame(basket, summary)
        built.append((basket, df))
        f = df[df["hit"].notna()][["hit", "agreement", "grade"]].copy()
        f["basket"] = basket["name"]
        frames.append(f.rename_axis("date").reset_index())

    if not frames:
        log.error("v2_summary_metrics.json 이 있는 바스켓이 없습니다 — v2_run.py 먼저 실행")
        return 1
    pool = pd.concat(frames, ignore_index=True)
    validation = {
        "generated_at": datetime.now(KST).isoformat(),
        "n_resolved": int(len(pool)),
        "baseline_hit_rate": round(float(pool["hit"].mean()), 4),
        "premium": {**_boot_diff(pool.dropna(subset=["agreement"]), "agreement", prem.AGREE, prem.OPPOSE),
                    "by_state": _group_rates(pool.dropna(subset=["agreement"]), "agreement")},
        "reliability": {**_boot_diff(pool.dropna(subset=["grade"]), "grade", rel.HIGH, rel.LOW, seed=1),
                        "by_grade": _group_rates(pool.dropna(subset=["grade"]), "grade")},
    }
    adjust = premium_adjustment(pool) if validation["premium"]["validated"] else None
    validation["premium"]["adjust"] = adjust
    (results_dir() / VALIDATION_FILE).write_text(json.dumps(validation, ensure_ascii=False, indent=1), encoding="utf-8")
    for basket, df in built:
        (results_dir() / basket["name"] / "etf_extras.json").write_text(
            json.dumps(today_extras(df, adjust), ensure_ascii=False, indent=1), encoding="utf-8")
    for key in ("premium", "reliability"):
        v = validation[key]
        log.info("%s 검증: 차이 %s, 95%% CI %s → %s", key, v["diff"], v["ci"], "통과" if v["validated"] else "미통과(화면 미표시)")
    return 0


def load_validation() -> dict:
    path = results_dir() / VALIDATION_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return run_all()


if __name__ == "__main__":
    sys.exit(main())
