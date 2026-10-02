"""메타 라벨링 — 2차 모델을 로지스틱 대신 나무 모형(LightGBM · 랜덤 포레스트)으로 — 검증이력 9.83 사전 등록 그대로.

  python research/validate_meta_label.py   → results/meta_label_validation.json
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "vm_predict", _ROOT / "scenario", _ROOT / "research", _ROOT / "stock_track"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import json
import sys
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

import validate_box as vb
from v2_config import results_dir

SPLIT, COST, PRUNE, N_BOOT = pd.Timestamp("2021-06-01"), vb.COST, 0.40, 1000
DEPTH_CAP = 0.12


def models():
    return {
        "로지스틱": lambda: LogisticRegression(C=1e6, max_iter=2000),
        "LightGBM": lambda: lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=8, min_child_samples=200,
                                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0),
        "랜덤 포레스트": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=200, max_depth=6, n_jobs=-1, random_state=0),
    }


def fit_predict(name, Xtr, ytr, Xte):
    m = models()[name]()
    if name == "로지스틱":
        sc = StandardScaler().fit(Xtr)
        m.fit(sc.transform(Xtr), ytr)
        return m.predict_proba(sc.transform(Xte))[:, 1], None
    m.fit(Xtr, ytr)
    imp = getattr(m, "feature_importances_", None)
    return m.predict_proba(Xte)[:, 1], imp


def blocks(dates: pd.Series) -> np.ndarray:
    return (dates.rank(method="dense").astype(int) // vb.BLOCK).to_numpy()


def auc_diff_ci(y, pa, pb, blk, rng):
    ub = np.unique(blk)
    idx = {k: np.where(blk == k)[0] for k in ub}
    out = []
    for _ in range(N_BOOT):
        m = np.concatenate([idx[k] for k in rng.choice(ub, len(ub))])
        if y[m].min() == y[m].max():
            continue
        out.append(roc_auc_score(y[m], pb[m]) - roc_auc_score(y[m], pa[m]))
    return float(roc_auc_score(y, pb) - roc_auc_score(y, pa)), *np.percentile(out, [2.5, 97.5])


def prune_effect(p, R, win, blk, rng) -> dict:
    keep = p > np.quantile(p, PRUNE)
    ub = np.unique(blk)
    idx = {k: np.where(blk == k)[0] for k in ub}
    bs = []
    for _ in range(N_BOOT):
        m = np.concatenate([idx[k] for k in rng.choice(ub, len(ub))])
        bs.append(R[m][keep[m]].mean() - R[m].mean())
    return {"남김 승률": round(float(win[keep].mean()), 3), "전체 승률": round(float(win.mean()), 3),
            "남김 R": round(float(R[keep].mean()), 3), "전체 R": round(float(R.mean()), 3), "쳐낸 R": round(float(R[~keep].mean()), 3),
            "R 차이(남김−전체)": [round(float(R[keep].mean() - R.mean()), 3), *[round(float(x), 3) for x in np.percentile(bs, [2.5, 97.5])]]}


def run(d: pd.DataFrame, feats: list[str], label: str, rng) -> dict:
    tr, te = d[d.entry_date < SPLIT], d[d.entry_date >= SPLIT]
    Xtr, Xte = tr[feats].to_numpy(float), te[feats].to_numpy(float)
    ytr, yte = tr[label].to_numpy(int), te[label].to_numpy(int)
    blk = blocks(te.entry_date)
    out, preds = {"맞춤 n": int(len(tr)), "예측 n": int(len(te)), "예측 승률": round(float(yte.mean()), 3)}, {}
    for name in models():
        p, imp = fit_predict(name, Xtr, ytr, Xte)
        preds[name] = p
        out[name] = {"AUC": round(float(roc_auc_score(yte, p)), 4), "로그 손실": round(float(log_loss(yte, np.clip(p, 1e-6, 1 - 1e-6))), 4),
                     "하위 40% 쳐내기": prune_effect(p, te["R"].to_numpy(float), yte.astype(float), blk, rng)}
        if imp is not None:
            out[name]["중요도"] = {f: round(float(v), 3) for f, v in sorted(zip(feats, imp / imp.sum()), key=lambda x: -x[1])}
    for name in ("LightGBM", "랜덤 포레스트"):
        dlt, lo, hi = auc_diff_ci(yte, preds["로지스틱"], preds[name], blk, rng)
        out[name]["AUC − 로지스틱"] = [round(dlt, 4), round(float(lo), 4), round(float(hi), 4)]
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    res = {}
    # A 가짜 이탈 — 9.79 승률 모형과 같은 사건·피처
    a = pd.read_csv(results_dir() / "winrate_events.csv", parse_dates=["entry_date"]).dropna(subset=["depth", "cpos", "prog_z"])
    a["win"] = ((a.ret - COST) > 0).astype(int)
    a["R"] = (a.ret - COST) / a.stop_pct.clip(lower=0.01)
    a["t2"] = a.t2.astype(float)
    a["reg_side"], a["reg_down"] = (a.reg == "횡보·전환").astype(float), (a.reg == "하락장").astype(float)
    a["depth"] = a.depth.clip(0, DEPTH_CAP)
    fa = ["t2", "reg_side", "reg_down", "depth", "cpos", "prog_z"]
    ra = run(a, fa, "win", rng)
    for name in ("LightGBM", "랜덤 포레스트"):
        lo = ra[name]["AUC − 로지스틱"][1]
        ra[name]["verdict"] = "교체" if lo > 0 and ra[name]["로그 손실"] < ra["로지스틱"]["로그 손실"] else "로지스틱 유지"
    res["A 가짜 이탈"] = ra
    # B 상승장 RS ≥ 70 돌파 — 9.72 사건 + 9.52·9.62·9.77·9.78 피처
    z = lambda f, cols: pd.read_csv(results_dir() / f, parse_dates=["entry_date"], dtype={"code": str}).assign(
        code=lambda x: x.code.str.zfill(6))[["code", "entry_date", *cols]]
    b = z("breakout_exits_events.csv", ["E2 50일선|R"]).rename(columns={"E2 50일선|R": "R"})
    b = (b.merge(z("trend_principles_events.csv", ["rs"]), on=["code", "entry_date"], how="left")
          .merge(z("breakout_edge_events.csv", ["strength", "bcpos", "t2", "comp", "vol250", "width"]), on=["code", "entry_date"], how="left")
          .merge(z("breakout_volume_events.csv", ["vx"]), on=["code", "entry_date"], how="left")
          .merge(z("earnings_growth_events.csv", ["c_rev", "a_ni"]), on=["code", "entry_date"], how="left"))
    b["has_fin"] = b.c_rev.notna().astype(float)
    b[["c_rev", "a_ni"]] = b[["c_rev", "a_ni"]].fillna(0.0)
    fb = ["rs", "strength", "bcpos", "t2", "comp", "vol250", "width", "vx", "c_rev", "a_ni", "has_fin"]
    b = b.dropna(subset=fb + ["R"])
    b["vx"] = np.log1p(b.vx)
    b["win"] = (b.R > 0).astype(int)
    rb = run(b, fb, "win", rng)
    for name in models():
        c = rb[name]["하위 40% 쳐내기"]["R 차이(남김−전체)"]
        rb[name]["verdict"] = "쳐내기 후보" if c[0] >= 0.10 and c[1] > 0 else "효과 없음"
    res["B 상승장 RS≥70 돌파"] = rb
    (results_dir() / "meta_label_validation.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
