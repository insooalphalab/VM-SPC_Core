"""코스피·코스닥 전 종목(보통주) 목록 — KIS 종목 마스터(7일마다 새로 받음). 스팩·우선주·ETF/ETN 제외.

수급 전 종목 수집(collect_program.py --all, collect_credit.py --all)이 쓴다.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import io
import time
import zipfile

import requests

URL = "https://new.real.download.dws.co.kr/common/master/{m}_code.mst.zip"
TAIL = {"kospi": 228, "kosdaq": 222}             # 종목 마스터 줄 끝 고정 길이 부분


def _master(m: str) -> list[str]:
    p = _ROOT / "state" / f"{m}_code_master.txt"
    if not p.exists() or time.time() - p.stat().st_mtime > 7 * 86400:
        r = requests.get(URL.format(m=m), timeout=60)
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        p.write_text(z.read(z.namelist()[0]).decode("cp949"), encoding="utf-8")
    return p.read_text(encoding="utf-8").splitlines()


def all_common_stocks() -> list[tuple[str, str, str]]:
    """(코드, 이름, 시장) — 보통주(코드 끝자리 0), 증권그룹 ST, 스팩 제외. 시가총액 큰 순."""
    rows = []
    for m, tail in TAIL.items():
        for ln in _master(m):
            p2, code, name = ln[-tail:], ln[:9].strip(), ln[21:len(ln) - tail].strip()
            if "ST" not in p2[:3] or len(code) != 6 or not code.endswith("0") or "스팩" in name:
                continue
            try:
                cap = int(p2[-15:-6])
            except ValueError:
                cap = 0
            rows.append((cap, code, name, m))
    rows.sort(reverse=True)
    return [(c, n, m) for _, c, n, m in rows]


def validation_codes() -> list[str]:
    """검증용 종목(약 790) — 센서 + 코스피 1~200·다음 200 + 코스닥 1~200(표본 밖 목록) + 관심 종목.
    새 수집·이력 확장은 전 종목이 아니라 이 목록만 한다(사용자 결정, 2026-10-01)."""
    import json
    from v2_config import LONG_HISTORY, data_dir, sensor_universe
    codes = set(sensor_universe())
    for f in ("oos_codes.json", "oos_kospi2_codes.json", "oos_kosdaq_codes.json"):
        p = data_dir() / LONG_HISTORY / f
        if p.exists():
            codes |= set(json.loads(p.read_text(encoding="utf-8"))["codes"])
    t = _ROOT / "scenario_targets.json"
    if t.exists():
        codes |= {s["code"] for s in json.loads(t.read_text(encoding="utf-8")).get("stocks", [])}
    return sorted(codes)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    u = all_common_stocks()
    print(len(u), "종목 ·", sum(1 for x in u if x[2] == "kospi"), "코스피 ·", sum(1 for x in u if x[2] == "kosdaq"), "코스닥")
