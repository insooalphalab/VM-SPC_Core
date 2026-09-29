"""OpenDART 호출·종목코드↔고유번호 매핑. 키는 core/secrets_loader 의 DART_API_KEY."""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "core", _ROOT / "vm_predict", _ROOT / "pair_spc"):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))

import io
import time
import zipfile
import xml.etree.ElementTree as ET

import pandas as pd
import requests

from secrets_loader import get_secret
from v2_config import data_dir

BASE = "https://opendart.fss.or.kr/api"
MIN_INTERVAL = 0.12          # 분당 1,000건 제한보다 느리게
_last = [0.0]


def dart_dir():
    d = data_dir() / "_dart"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get(endpoint: str, **params) -> dict:
    """status 000(정상)·013(데이터 없음)은 그대로 돌려주고, 나머지는 예외."""
    wait = MIN_INTERVAL - (time.time() - _last[0])
    if wait > 0:
        time.sleep(wait)
    for attempt in range(3):
        try:
            r = requests.get(f"{BASE}/{endpoint}.json", params={"crtfc_key": get_secret("DART_API_KEY"), **params}, timeout=30)
            _last[0] = time.time()
            body = r.json()
            break
        except (requests.RequestException, ValueError):
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
    if body.get("status") not in ("000", "013"):
        raise RuntimeError(f"DART {endpoint} {body.get('status')}: {body.get('message')}")
    return body


def corp_codes(refresh: bool = False) -> pd.DataFrame:
    """상장사 stock_code → corp_code 표(data/_dart/corp_codes.csv 캐시)."""
    path = dart_dir() / "corp_codes.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, dtype=str)
    r = requests.get(f"{BASE}/corpCode.xml", params={"crtfc_key": get_secret("DART_API_KEY")}, timeout=60)
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        root = ET.fromstring(z.read(z.namelist()[0]))
    rows = [{"corp_code": e.findtext("corp_code"), "stock_code": (e.findtext("stock_code") or "").strip(),
             "corp_name": e.findtext("corp_name")} for e in root.iter("list")]
    df = pd.DataFrame([x for x in rows if x["stock_code"]])
    df.to_csv(path, index=False, encoding="utf-8")
    return df
